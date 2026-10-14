"""One decoder invocation for independent V-off and Text-off clean-prefix rows."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor,nn
from .stage1_v5_trace import V5Trace,QueryKey,bind_v5_execution,validate_v5_decoder,_PACKED


@dataclass(frozen=True)
class V5ReplayBlock:
    batch_index: int
    start: int
    end: int
    target_positions: tuple[int,...]
    target_ids: tuple[int,...]
    blocked_text_keys: tuple[tuple[int,...],...]


@dataclass(frozen=True)
class V5ReplayOutput:
    voff_logits: Tensor | None
    textoff_logits: Tensor | None
    query_keys: tuple[QueryKey,...]


def run_v5_packed_replay(decoder: nn.Module,lm_head: nn.Module,trace: V5Trace,
                         blocks: tuple[V5ReplayBlock,...],*,execution,need_voff: bool,
                         need_textoff: bool,chunk_positions: int=128) -> V5ReplayOutput:
    validate_v5_decoder(decoder)
    if type(need_voff) is not bool or type(need_textoff) is not bool or type(chunk_positions) is not int or chunk_positions<=0:
        raise ValueError('v5 replay requires boolean branches and positive chunk_positions')
    keys = tuple((b.batch_index,t-1) for b in blocks for t in b.target_positions)
    if len(set(keys))!=len(keys): raise ValueError('duplicate v5 replay queries')
    active = tuple(b for b in blocks if b.target_positions)
    if not active or not (need_voff or need_textoff):
        return V5ReplayOutput(None,None,keys)
    x = trace.inputs_embeds
    if (x is None or trace.padding is None or trace.position_ids is None
            or len(trace.key_values)!=len(decoder.layers)):
        raise ValueError('v5 replay requires a complete attached clean trace')
    for b in active:
        if (not 0<=b.batch_index<x.shape[0] or not 0<=b.start<b.end<x.shape[1]
                or len(b.target_ids)!=len(b.target_positions)
                or len(b.blocked_text_keys)!=len(b.target_positions)
                or tuple(sorted(set(b.target_positions)))!=b.target_positions
                or any(t<=b.end or t>=x.shape[1] for t in b.target_positions)
                or any(k<0 or k>=b.start for ks in b.blocked_text_keys for k in ks)
                or not bool(trace.padding[b.batch_index,b.end:max(b.target_positions)+1].bool().all())):
            raise ValueError('invalid v5 replay block coordinates')
    rows = tuple((kind,b) for kind,needed in (('v',need_voff),('text',need_textoff)) if needed for b in active)
    prefix_length = max(b.end for b in active)
    length = max(max(b.target_positions)-b.end for b in active)
    samples = torch.tensor([b.batch_index for _,b in rows],device=x.device)[:,None].expand(-1,length)
    starts = torch.tensor([b.end for _,b in rows],device=x.device)[:,None]
    positions = starts+torch.arange(length,device=x.device)[None]
    valid = torch.stack([positions[i]<max(b.target_positions) for i,(_,b) in enumerate(rows)])
    positions = positions.clamp_max(x.shape[1]-1)
    embeds = x[samples,positions]*valid[:,:,None]
    ids = trace.position_ids[:,samples,positions]
    bias = x.new_full((len(rows),1,length,prefix_length+length),float('-inf'))
    for i,(kind,b) in enumerate(rows):
        actual = max(b.target_positions)-b.end
        for q in range(actual):
            bias[i,0,q,:b.end] = torch.where(trace.padding[b.batch_index,:b.end].bool(),0.,float('-inf')).to(x.dtype)
            bias[i,0,q,prefix_length:prefix_length+q+1] = 0
        # Padded queries cannot influence valid suffix queries or selected scores.
        bias[i,0,actual:,0] = 0
        if kind=='v':
            bias[i,0,:actual,b.start+1:b.end] = float('-inf')
        else:
            for target,blocked in zip(b.target_positions,b.blocked_text_keys):
                bias[i,0,target-1-b.end,list(blocked)] = float('-inf')
    prefixes = tuple(tuple(torch.cat([
        torch.nn.functional.pad(value[b.batch_index:b.batch_index+1,:,:b.end],(0,0,0,prefix_length-b.end))
        for _,b in rows],dim=0) for value in pair) for pair in trace.key_values)
    token = _PACKED.set((execution.dropout_mask,samples,positions,valid,{}))
    try:
        with bind_v5_execution(decoder,execution,role='packed'):
            output = decoder(inputs_embeds=embeds,attention_mask=bias,position_ids=ids,
                             stage2_prefix_key_values=prefixes,use_cache=False,return_dict=True)
    finally:
        _PACKED.reset(token)
    row_indices,offsets = zip(*[(i,t-1-b.end) for i,(_,b) in enumerate(rows) for t in b.target_positions])
    selected = output.last_hidden_state[row_indices,offsets]
    logits = torch.cat([lm_head(chunk) for chunk in selected.split(chunk_positions)],dim=0)
    n = len(keys)
    voff = logits[:n] if need_voff else None
    textoff = logits[n:] if need_textoff and need_voff else logits if need_textoff else None
    return V5ReplayOutput(voff,textoff,keys)
