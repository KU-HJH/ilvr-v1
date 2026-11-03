"""Independent teacher-input score graphs, isolated from optimizer reducers."""
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vision_gain_packed_execution import PrefixConditioningContext
import torch
from torch import Tensor, nn
from transformers.models.qwen2_5_vl.stage1_vision_gain import Stage1VisionGainRequest
from .vision_gain import Stage1GainPlan, score_stage1_gain_block


@dataclass(frozen=True)
class GainRngState:
    cpu: Tensor
    cuda: tuple[tuple[int, Tensor], ...]


def capture_gain_rng(model: nn.Module) -> GainRngState:
    devices=sorted({p.device.index for p in model.parameters() if p.device.type=='cuda'})
    return GainRngState(torch.get_rng_state(),tuple((d,torch.cuda.get_rng_state(d)) for d in devices))


def _restore_rng(state):
    torch.set_rng_state(state.cpu)
    for device,rng in state.cuda:
        torch.cuda.set_rng_state(rng,device)


@dataclass(frozen=True)
class Stage1GainScores:
    clean_sum: Tensor
    masked_sum: Tensor
    block_count: int
    score_parameters: tuple[Tensor, ...]


def _checkpoint_alias_layer(function, *args, **kwargs):
    """Bind the current score aliases again inside each layer recomputation."""
    from torch.utils.checkpoint import checkpoint
    layer = getattr(function, '__self__', None)
    if not isinstance(layer, nn.Module):
        raise ValueError('gain checkpoint requires a bound layer forward')
    # Called during the outer functional_call, so trainable values here are the
    # score aliases, not the optimizer-owned parameters restored afterward.
    parameters = dict(layer.named_parameters())
    buffers = dict(layer.named_buffers())
    def replay(*inputs):
        return torch.func.functional_call(
            layer, (parameters, buffers), inputs, kwargs, tie_weights=True,
        )
    return checkpoint(replay, *args, use_reentrant=False, preserve_rng_state=True)


@contextmanager
def _score_model_state(model, *, checkpoint_scores=False):
    absent = object()
    checkpointed=[(m,m.gradient_checkpointing,getattr(m,'_gradient_checkpointing_func',absent))
                  for m in model.modules()
                  if isinstance(getattr(m,'gradient_checkpointing',None),bool)]
    ropes=[(m,m.rope_deltas) for m in model.modules() if 'rope_deltas' in vars(m)]
    for m,_,_ in checkpointed:
        m.gradient_checkpointing=checkpoint_scores
        if checkpoint_scores:
            m._gradient_checkpointing_func=_checkpoint_alias_layer
    try:
        yield
    finally:
        for m,value,function in checkpointed:
            m.gradient_checkpointing=value
            if function is absent:
                if hasattr(m,'_gradient_checkpointing_func'):
                    delattr(m,'_gradient_checkpointing_func')
            else:
                m._gradient_checkpointing_func=function
        for m,value in ropes:
            m.rope_deltas=value


def execute_stage1_gain(model: nn.Module, model_inputs: Mapping[str,object],
                        plan: Stage1GainPlan, clean_outputs: object,
                        rng_before_clean: GainRngState,
                        rng_after_clean: GainRngState, *, accumulator=None,
                        execution='independent', branch_mode='independent',
                        conditioning_context: "PrefixConditioningContext | None" = None) -> Stage1GainScores:
    """Replay F/M against fresh leaf aliases sharing pre-update weight values.

    The base graph and .grad belong exclusively to the existing Stage 1 loss.
    No parameter update occurs until all alias gradients have been consumed.
    """
    if conditioning_context is not None and execution != 'packed':
        raise ValueError('prefix conditioning requires packed gain execution')
    if execution == 'packed':
        from .vision_gain_packed_execution import execute_packed_stage1_gain
        return execute_packed_stage1_gain(model, model_inputs, plan, clean_outputs,
                                         rng_before_clean, rng_after_clean, accumulator=accumulator,
                                         branch_mode=branch_mode, conditioning_context=conditioning_context)
    if branch_mode != 'independent':
        raise ValueError('cumulative vision gain requires packed execution')
    if execution != 'independent':
        raise ValueError('unknown Stage 1 vision gain execution')
    if not plan.blocks:
        zero=model_inputs['input_ids'].new_zeros((),dtype=torch.float32)
        return Stage1GainScores(zero,zero,0,())
    while isinstance(model,(torch.nn.parallel.DistributedDataParallel,torch.nn.DataParallel)):
        model=model.module
    if type(model).__name__=='DeepSpeedEngine':
        model=model.module
    aliases={name:p.detach().requires_grad_(True) for name,p in model.named_parameters() if p.requires_grad}
    if not aliases:
        raise ValueError('vision gain requires trainable student parameters')
    # Buffers can be updated during forwards; score branches must not alter base state.
    original_buffers={name:b.detach().clone() for name,b in model.named_buffers()}
    inputs=dict(model_inputs)
    inputs.pop('labels',None)
    inputs.pop('past_key_values',None)
    inputs['use_cache']=False
    inputs['output_hidden_states']=False
    inputs['latent_hidden_states']=inputs['latent_hidden_states'].detach()
    def run(request=None):
        _restore_rng(rng_before_clean)
        kwargs=dict(inputs)
        if request is not None:
            kwargs['stage1_vision_gain_request']=request
        buffers={name:b.clone() for name,b in original_buffers.items()}
        return torch.func.functional_call(model,(aliases,buffers),(),kwargs,tie_weights=True)
    try:
        with _score_model_state(model, checkpoint_scores=accumulator is not None):
            clean=run()
            clean_scores=[score_stage1_gain_block(clean.logits,b) for b in plan.blocks]
            # Reject a replay that changed model mode, input preparation, or stochastic work.
            if clean_outputs is not None:
                for block in plan.blocks:
                    positions=[i-1 for i in block.target_positions]
                    expected=clean_outputs.logits[block.batch_index,positions].detach().float()
                    observed=clean.logits[block.batch_index,positions].detach().float()
                    if not torch.allclose(expected,observed,atol=2e-5,rtol=2e-4,equal_nan=False):
                        raise RuntimeError('vision gain clean replay differs from Stage 1 forward')
            clean_sum = torch.stack(clean_scores).sum()
            if accumulator is not None:
                accumulator.accumulate_branch(
                    clean_sum, tuple(aliases.values()), masked=False,
                    block_count=len(plan.blocks),
                )
                clean_sum = clean_sum.detach()
                del clean, clean_scores
            masked_scores=[]
            for block in plan.blocks:
                request=Stage1VisionGainRequest(block.batch_index,block.start_marker,block.end_marker)
                masked=run(request)
                score = score_stage1_gain_block(masked.logits,block)
                if accumulator is not None:
                    accumulator.accumulate_branch(score, tuple(aliases.values()), masked=True)
                    score = score.detach()
                    del masked
                masked_scores.append(score)
            return Stage1GainScores(clean_sum,torch.stack(masked_scores).sum(),
                                   len(plan.blocks),tuple(aliases.values()))
    finally:
        _restore_rng(rng_after_clean)
