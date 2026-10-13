"""Isolated alias VJPs and one ordinary-gradient proxy on optimizer parameters."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import torch
from torch import Tensor,nn
from peft_common import resolve_qwen2_5_vl_model
from transformers.models.qwen2_5_vl.stage1_v3_trace import capture_v3_trace,bind_v3_execution
from transformers.models.qwen2_5_vl.stage1_v3_replay import V3ReplayBlock,run_v3_packed_replay
from .v3_config import Stage1V3Config
from .v3_metadata import V3BatchMetadata
from .v3_lifecycle import V3Lifecycle,packed_dropout_mask
from .v3_selection import select_text_keys
from .v3_objective import block_score_sum,text_kl_loss,token_entropy,entropy_weighted_gain_and_rv_sums


@dataclass(frozen=True)
class V3Gradients:
    r_sum_grads: tuple[Tensor,...] | None
    rv_sum_grads: tuple[Tensor,...]
    block_count: int
    kl_proxy: Tensor
    kl_value: Tensor
    # A weighted Gain already includes the RV subtraction; never label it R.
    gain_sum_grads: tuple[Tensor,...] | None = None


class _GradientProxy(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,gradients,*parameters):
        ctx.gradients=gradients
        return value.detach().clone()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,scale):
        return (None,None,*(gradient*scale.to(gradient) for gradient in ctx.gradients))


def optimizer_gradient_proxy(value: Tensor,gradients: tuple[Tensor|None,...],
                              parameters: tuple[nn.Parameter,...]) -> Tensor:
    if value.numel()!=1 or len(gradients)!=len(parameters) or len({id(p) for p in parameters})!=len(parameters):
        raise ValueError('invalid v3 optimizer gradient identity')
    if any(g is not None and (g.shape!=p.shape or g.device!=p.device or not torch.isfinite(g).all())
           for g,p in zip(gradients,parameters)):
        raise ValueError('invalid or nonfinite v3 optimizer gradient')
    active=[(g.detach().to(p),p) for g,p in zip(gradients,parameters) if g is not None]
    if not active: return value.detach().clone()
    grads,params=zip(*active)
    return _GradientProxy.apply(value.detach(),grads,*params)


class _Replay(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model

    def forward(self,trace,blocks,lifecycle):
        qwen=resolve_qwen2_5_vl_model(self.model)
        return run_v3_packed_replay(qwen.model,qwen.lm_head,trace,blocks,
            checkpoint_contexts=lambda i:lifecycle.checkpoint_contexts('packed_replay',('decoder',i)),
            dropout_mask=packed_dropout_mask,parameter_bindings=dict(qwen.model.named_parameters()),
            buffer_bindings={n:b.detach().clone() for n,b in qwen.model.named_buffers()})


def compute_v3_gradients(model: nn.Module,reference: nn.Module,model_inputs: Mapping[str,object],
                         metadata: V3BatchMetadata,clean_logits: Tensor,parameters: tuple[nn.Parameter,...],
                         config: Stage1V3Config,lifecycle: V3Lifecycle) -> V3Gradients:
    if not metadata.blocks:
        zeros=tuple(torch.zeros_like(p,dtype=torch.float32) for p in parameters)
        value=clean_logits.new_zeros((),dtype=torch.float32)
        return V3Gradients(zeros if config.gain_weighting=='none' else None,
            tuple(g.clone() for g in zeros),0,value,value,
            gain_sum_grads=zeros if config.gain_weighting=='token_entropy' else None)
    model=getattr(model,'module',model)
    named=tuple((name,p) for name,p in model.named_parameters() if p.requires_grad)
    if tuple(id(p) for _,p in named)!=tuple(id(p) for p in parameters):
        raise ValueError('v3 alias ordering differs from optimizer-owned trainable parameters')
    if any(p.requires_grad for p in reference.parameters()):
        raise ValueError('v3 reference must be frozen before auxiliary execution')
    aliases={name:p.detach().clone().requires_grad_() for name,p in named}
    alias_parameters=tuple(aliases.values())
    buffers={name:b.detach().clone() for name,b in model.named_buffers()}
    inputs={k:v for k,v in model_inputs.items() if k!='labels'}
    if 'latent_hidden_states' in inputs: inputs['latent_hidden_states']=inputs['latent_hidden_states'].detach()
    inputs.update(use_cache=False,return_dict=True)
    blocks=metadata.blocks
    keys=tuple((b.batch_index,q) for b in blocks for q in b.query_positions)
    candidates=tuple(b.candidate_positions for b in blocks for _ in b.query_positions)
    decoder=resolve_qwen2_5_vl_model(model).model
    with lifecycle.graph('gain_clean'),capture_v3_trace(decoder,keys,candidates) as trace:
        with bind_v3_execution(decoder,lambda i:lifecycle.checkpoint_contexts('gain_clean',('decoder',i)),
                visual=resolve_qwen2_5_vl_model(model).visual,
                vision_contexts=lambda i:lifecycle.checkpoint_contexts('gain_clean',('vision',i))):
            gain_clean=torch.func.functional_call(model,(aliases,buffers),(),inputs).logits
    reference_logits=None
    if config.text_selector == 'kl':
        training_flags=tuple((module,module.training) for module in reference.modules())
        try:
            reference.eval()
            with torch.no_grad():
                reference_logits=torch.func.functional_call(reference,
                    {n:b.detach().clone() for n,b in reference.named_buffers()},(),inputs).logits
        finally:
            for module,training in training_flags: module.training=training
    selected=select_text_keys(blocks,trace.attention_rows,gain_clean,reference_logits,config)
    replay_blocks=tuple(V3ReplayBlock(b.batch_index,b.start,b.end,b.target_positions,b.target_ids,
                       tuple(selected[b.batch_index,q] for q in b.query_positions)) for b in blocks)
    replay=_Replay(model)
    with lifecycle.graph('packed_replay'):
        output=torch.func.functional_call(replay,
            ({'model.'+n:p for n,p in aliases.items()},{'model.'+n:b for n,b in buffers.items()}),
            (trace,replay_blocks,lifecycle))
    row,query=zip(*keys)
    if config.gain_weighting=='token_entropy':
        weights=token_entropy(clean_logits,keys,mode=config.gain_entropy_mode)
        primary,rv=entropy_weighted_gain_and_rv_sums(output.r_logits,output.rv_logits,blocks,weights,
                                                   entropy_mode=config.gain_entropy_mode)
    else:
        primary=block_score_sum(output.r_logits,blocks)
        rv=block_score_sum(output.rv_logits,blocks)
    kl=text_kl_loss(clean_logits[row,query],output.r_logits,keys,temperature=config.temperature)
    def joint(seed):
        gradients=torch.autograd.grad((primary,kl),alias_parameters,grad_outputs=tuple(seed.unbind()),
                                      allow_unused=True,retain_graph=True)
        return {i:g for i,g in enumerate(gradients) if g is not None}
    with lifecycle.backward('joint_r_kl'):
        joint_grads=torch.vmap(joint,randomness='same')(torch.eye(2,device=primary.device,dtype=primary.dtype))
    with lifecycle.backward('rv'):
        rv_grads=torch.autograd.grad(rv,alias_parameters,allow_unused=True)
    primary_grads=tuple(joint_grads[i][0].detach().float() if i in joint_grads else torch.zeros_like(p,dtype=torch.float32)
                  for i,p in enumerate(parameters))
    rv_grads=tuple(g.detach().float() if g is not None else torch.zeros_like(p,dtype=torch.float32)
                   for g,p in zip(rv_grads,parameters))
    kl_grads=tuple(joint_grads[i][1].detach().clone() if i in joint_grads else None for i in range(len(parameters)))
    value=kl.detach()
    return V3Gradients(primary_grads if config.gain_weighting=='none' else None,
        rv_grads,len(blocks),optimizer_gradient_proxy(value,kl_grads,parameters),value,
        gain_sum_grads=primary_grads if config.gain_weighting=='token_entropy' else None)
