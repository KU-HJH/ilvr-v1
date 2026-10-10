"""One batched decoder replay with independent attached prefixes for R and RV."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import torch
from torch import Tensor,nn
from .stage1_v3_trace import V3Trace,QueryKey,bind_v3_execution,validate_v3_decoder,_PACKED


class DropoutMask(Protocol):
    def __call__(self,site: str,p: float,*,sample_indices: Tensor,token_positions: Tensor,
                 hidden_size: int,device: torch.device) -> Tensor: ...


@dataclass(frozen=True)
class V3ReplayBlock:
    batch_index: int
    start: int
    end: int
    target_positions: tuple[int,...]
    target_ids: tuple[int,...]
    blocked_text_keys: tuple[tuple[int,...],...]


@dataclass(frozen=True)
class V3ReplayOutput:
    r_logits: Tensor
    rv_logits: Tensor
    query_keys: tuple[QueryKey,...]


def run_v3_packed_replay(decoder: nn.Module,lm_head: nn.Module,trace: V3Trace,
                          blocks: tuple[V3ReplayBlock,...],*,checkpoint_contexts,dropout_mask: DropoutMask,
                          parameter_bindings,buffer_bindings) -> V3ReplayOutput:
    validate_v3_decoder(decoder)
    if not blocks or trace.inputs_embeds is None or len(trace.key_values)!=len(decoder.layers):
        raise ValueError('packed v3 requires valid blocks and a complete attached trace')
    x = trace.inputs_embeds
    for block in blocks:
        if (not 0<=block.batch_index<x.shape[0] or not 0<=block.start<block.end<x.shape[1]
                or not block.target_positions or len(block.target_ids)!=len(block.target_positions)
                or len(block.blocked_text_keys)!=len(block.target_positions)
                or tuple(sorted(set(block.target_positions)))!=block.target_positions
                or any(t<=block.end or t>x.shape[1] for t in block.target_positions)
                or any(k<0 or k>=block.start for keys in block.blocked_text_keys for k in keys)):
            raise ValueError('invalid v3 replay block coordinates')
    rows = blocks+blocks
    prefix_length = max(b.end for b in blocks)
    length = max(max(b.target_positions)-b.end for b in blocks)
    samples = torch.tensor([b.batch_index for b in rows],device=x.device)[:,None].expand(-1,length)
    starts = torch.tensor([b.end for b in rows],device=x.device)[:,None]
    positions = starts+torch.arange(length,device=x.device)[None]
    valid = torch.stack([positions[i]<max(b.target_positions) for i,b in enumerate(rows)])
    positions = positions.clamp_max(x.shape[1]-1)
    embeds = x[samples,positions]*valid[:,:,None]
    ids = trace.position_ids[:,samples,positions]
    bias = x.new_full((len(rows),1,length,prefix_length+length),float('-inf'))
    for i,b in enumerate(rows):
        end = max(b.target_positions)-b.end
        for q in range(end):
            bias[i,0,q,:b.end] = torch.where(trace.padding[b.batch_index,:b.end].bool(),0.,float('-inf')).to(x.dtype)
            bias[i,0,q,prefix_length:prefix_length+q+1] = 0
        # Padded query outputs are ignored; one visible key avoids all-masked rows.
        bias[i,0,end:,0] = 0
        for target,keys in zip(b.target_positions,b.blocked_text_keys):
            bias[i,0,target-1-b.end,list(keys)] = float('-inf')
        if i>=len(blocks): bias[i,0,:end,b.start+1:b.end] = float('-inf')
    prefixes = tuple(tuple(torch.cat([
        torch.nn.functional.pad(value[b.batch_index:b.batch_index+1,:,:b.end],(0,0,0,prefix_length-b.end))
        for b in rows],dim=0) for value in pair) for pair in trace.key_values)
    token = _PACKED.set((dropout_mask,samples,positions,valid))
    try:
        with bind_v3_execution(decoder,checkpoint_contexts):
            output = torch.func.functional_call(decoder,(dict(parameter_bindings),dict(buffer_bindings)),(),dict(
                inputs_embeds=embeds,attention_mask=bias,position_ids=ids,
                stage2_prefix_key_values=prefixes,use_cache=False,return_dict=True))
    finally:
        _PACKED.reset(token)
    query_rows,query_offsets = zip(*[(i,t-1-b.end) for i,b in enumerate(rows) for t in b.target_positions])
    # A single head invocation for both branches and only the scored positions.
    logits = lm_head(output.last_hidden_state[query_rows,query_offsets])
    n = logits.shape[0]//2
    keys = tuple((b.batch_index,t-1) for b in blocks for t in b.target_positions)
    return V3ReplayOutput(logits[:n],logits[n:],keys)
