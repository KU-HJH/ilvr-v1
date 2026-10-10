"""Opt-in tracing and checkpoint bindings; no dependency on the training package."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

QueryKey = tuple[int,int]
_EXECUTION = ContextVar('qwen_stage1_v3_execution',default=None)
_PACKED = ContextVar('qwen_stage1_v3_packed_dropout',default=None)


@dataclass
class V3Trace:
    inputs_embeds: Tensor | None = None
    position_ids: Tensor | None = None
    padding: Tensor | None = None
    key_values: tuple[tuple[Tensor,Tensor],...] = ()
    attention_rows: dict[QueryKey,Tensor] = field(default_factory=dict)


def validate_v3_decoder(decoder):
    if decoder.config._attn_implementation not in ('eager','sdpa'):
        raise ValueError('v3 requires eager/sdpa language backend')
    if getattr(decoder.config,'use_sliding_window',False):
        raise ValueError('v3 does not support effective sliding window')
    if any(layer.self_attn.attention_dropout != 0 for layer in decoder.layers):
        raise ValueError('v3 requires zero native attention dropout')


@contextmanager
def bind_v3_execution(decoder: nn.Module, checkpoint_contexts: Callable):
    validate_v3_decoder(decoder)
    if _EXECUTION.get() is not None:
        raise RuntimeError('nested v3 decoder execution')
    token = _EXECUTION.set((decoder,checkpoint_contexts))
    try:
        yield
    finally:
        _EXECUTION.reset(token)


class _LayerCall(nn.Module):
    def __init__(self, layer, original):
        super().__init__()
        self.layer = layer
        self.original = original

    def forward(self,*args,**kwargs):
        return self.original(*args,**kwargs)


def v3_decoder_forward(forward):
    """Scope live/alias/buffer bindings into each retained checkpoint closure."""
    @wraps(forward)
    def wrapped(self,*args,**kwargs):
        execution = _EXECUTION.get()
        if execution is None or execution[0] is not self:
            return forward(self,*args,**kwargs)
        validate_v3_decoder(self)
        enabled = self.gradient_checkpointing and self.training
        old_flag = self.gradient_checkpointing
        restorations = []
        try:
            # Own this opt-in checkpoint only; the original decoder loop runs once.
            self.gradient_checkpointing = False
            for index,layer in enumerate(self.layers):
                original = layer.forward
                restorations.append((layer,'forward' in layer.__dict__,original))
                runner = _LayerCall(layer,original)
                bindings = {'layer.'+n:p for n,p in layer.named_parameters()}
                buffers = {'layer.'+n:b.detach().clone() for n,b in layer.named_buffers()}
                managers = execution[1](index)
                packed = _PACKED.get()

                def call(*a,_runner=runner,_bindings=bindings,_buffers=buffers,_packed=packed,**kw):
                    token = _PACKED.set(_packed)
                    try:
                        return torch.func.functional_call(_runner,(_bindings,_buffers),a,kw)
                    finally:
                        _PACKED.reset(token)

                def layer_forward(*a,_call=call,_managers=managers,**kw):
                    if enabled:
                        return checkpoint(_call,*a,use_reentrant=False,context_fn=lambda:_managers,**kw)
                    with _managers[0]:
                        return _call(*a,**kw)
                layer.forward = layer_forward
            return forward(self,*args,**kwargs)
        finally:
            for layer,had_instance,original in reversed(restorations):
                if had_instance: layer.forward = original
                else: del layer.forward
            self.gradient_checkpointing = old_flag
    return wrapped


def packed_dropout_output(site: str, p: float, x: Tensor) -> Tensor | None:
    packed = _PACKED.get()
    if packed is None:
        return None
    callback,samples,positions,valid = packed
    if x.ndim != 3 or x.shape[:2] != valid.shape:
        raise ValueError('v3 packed dropout shape differs from original token coordinates')
    raw = callback(site,p,sample_indices=samples[valid],token_positions=positions[valid],
                   hidden_size=x.shape[-1],device=x.device)
    mask = torch.ones_like(x)
    mask[valid] = raw.to(x.dtype)*(1/(1-p))
    return x*mask


@contextmanager
def capture_v3_trace(model: nn.Module, query_keys: tuple[QueryKey,...],
                     candidate_positions: tuple[tuple[int,...],...]):
    decoder = model if hasattr(model,'layers') else model.model
    validate_v3_decoder(decoder)
    if len(query_keys)!=len(candidate_positions) or len(set(query_keys))!=len(query_keys):
        raise ValueError('trace query/candidate identity mismatch')
    trace = V3Trace()
    handles,seen,projection_outputs = [],set(),{}

    def before(module,args,kwargs):
        if trace.inputs_embeds is not None:
            raise RuntimeError('duplicate v3 clean decoder forward')
        embeds = kwargs.get('inputs_embeds')
        if embeds is None:
            raise ValueError('v3 capture requires existing prepared inputs_embeds')
        trace.inputs_embeds = embeds
        b,t,_ = embeds.shape
        padding = kwargs.get('attention_mask')
        trace.padding = torch.ones((b,t),device=embeds.device,dtype=torch.long) if padding is None else padding
        if trace.padding.shape != (b,t):
            raise ValueError('v3 clean trace requires source [B,T] padding')
        ids = kwargs.get('position_ids')
        if ids is None: ids = torch.arange(t,device=embeds.device).view(1,1,-1).expand(3,b,-1)
        if ids.ndim==2: ids = ids[None].expand(3,-1,-1)
        trace.position_ids = ids
        kwargs = dict(kwargs,stage2_capture_kv=True,use_cache=False,return_dict=True)
        return args,kwargs

    def after(module,args,kwargs,output):
        trace.key_values = tuple(output.stage2_key_values)

    def observe(attn,args,kwargs,output):
        from .modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb,repeat_kv
        index = attn.layer_idx
        if index in seen: raise RuntimeError('duplicate v3 attention observation')
        seen.add(index)
        q,k = projection_outputs.pop((index,'q')),projection_outputs.pop((index,'k'))
        b,t,_ = q.shape
        # Observation is detached; it adds no saved tensors to checkpoint graphs.
        with torch.no_grad(),torch.autocast(device_type=q.device.type,enabled=False):
            q = q.view(b,t,-1,attn.head_dim).transpose(1,2)
            k = k.view(b,t,-1,attn.head_dim).transpose(1,2)
            cos,sin = kwargs['position_embeddings']
            q,k = apply_multimodal_rotary_pos_emb(q,k,cos,sin,attn.rope_scaling['mrope_section'])
            k = repeat_kv(k,attn.num_key_value_groups)
            for key,candidates in zip(query_keys,candidate_positions):
                row,position = key
                scores = q[row,:,position:position+1].float() @ k[row].float().transpose(-2,-1)
                scores = scores.squeeze(1)*attn.scaling
                visible = (torch.arange(t,device=q.device)<=position) & trace.padding[row].bool()
                weights = scores.masked_fill(~visible,float('-inf')).softmax(-1).mean(0)
                values = weights[list(candidates)]/len(decoder.layers)
                trace.attention_rows[key] = trace.attention_rows.get(key,0)+values

    try:
        handles.append(decoder.register_forward_pre_hook(before,with_kwargs=True))
        handles.append(decoder.register_forward_hook(after,with_kwargs=True))
        for layer in decoder.layers:
            attn = layer.self_attn
            for name in ('q','k'):
                def save(module,args,output,_key=(attn.layer_idx,name)):
                    projection_outputs[_key] = output.detach()
                handles.append(getattr(attn,name+'_proj').register_forward_hook(save))
            handles.append(attn.register_forward_hook(observe,with_kwargs=True))
        yield trace
        if len(seen)!=len(decoder.layers) or len(trace.key_values)!=len(decoder.layers):
            raise RuntimeError('incomplete v3 trace')
    finally:
        for handle in reversed(handles): handle.remove()
