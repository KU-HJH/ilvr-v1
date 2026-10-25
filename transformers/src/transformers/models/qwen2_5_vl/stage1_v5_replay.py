"""Clean-prefix mask replay and optional independent/cumulative latent replacement."""
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
    replacement_values: Tensor | None = None


@dataclass(frozen=True)
class V5ReplayOutput:
    voff_logits: Tensor | None
    textoff_logits: Tensor | None
    query_keys: tuple[QueryKey,...]


def _run_attention_mask_replay(decoder: nn.Module,lm_head: nn.Module,trace: V5Trace,
                         blocks: tuple[V5ReplayBlock,...],*,execution,need_voff: bool,
                         need_textoff: bool,chunk_positions: int=128) -> V5ReplayOutput:
    if execution.debug:
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
    for b in active if execution.debug else ():
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


@dataclass(frozen=True)
class _ReplacementRow:
    batch_index: int
    start: int
    stop: int
    targets: tuple[int, ...]
    blocked_text: tuple[tuple[int, ...], ...]
    replacements: tuple[V5ReplayBlock, ...]


def _replacement_rows(decoder, lm_head, trace, rows, execution, chunk_positions):
    """One decoder call with row-specific clean-prefix cutoffs and suffixes."""
    x = trace.inputs_embeds
    prefix_length = max(r.start for r in rows)
    length = max(r.stop-r.start for r in rows)
    samples = torch.tensor([r.batch_index for r in rows], device=x.device)[:, None].expand(-1, length)
    starts = torch.tensor([r.start for r in rows], device=x.device)[:, None]
    positions = starts + torch.arange(length, device=x.device)[None]
    stops = torch.tensor([r.stop for r in rows], device=x.device)[:, None]
    valid = positions < stops
    positions = positions.clamp_max(x.shape[1]-1)
    embeds = x[samples, positions] * valid[:, :, None]
    ids = trace.position_ids[:, samples, positions]
    bias = x.new_full((len(rows), 1, length, prefix_length+length), float('-inf'))
    for i, row in enumerate(rows):
        actual = row.stop-row.start
        for block in row.replacements:
            offset = block.start+1-row.start
            # CopySlices blocks the overwritten clean embedding path while
            # keeping all other suffix embeddings and attached prefix K/V live.
            embeds[i, offset:offset+block.end-block.start-1] = block.replacement_values.detach().to(x)
        prefix_bias = torch.where(trace.padding[row.batch_index, :row.start].bool(), 0., float('-inf')).to(x.dtype)
        bias[i, 0, :actual, :row.start] = prefix_bias
        causal = torch.ones(actual, actual, dtype=torch.bool, device=x.device).tril()
        bias[i, 0, :actual, prefix_length:prefix_length+actual] = x.new_zeros((actual, actual)).masked_fill(~causal, float('-inf'))
        bias[i, 0, actual:, 0] = 0
        for target, blocked in zip(row.targets, row.blocked_text):
            bias[i, 0, target-1-row.start, list(blocked)] = float('-inf')
    prefixes = tuple(tuple(torch.cat([
        torch.nn.functional.pad(value[r.batch_index:r.batch_index+1, :, :r.start],
                                (0, 0, 0, prefix_length-r.start)) for r in rows], dim=0)
        for value in pair) for pair in trace.key_values)
    token = _PACKED.set((execution.dropout_mask, samples, positions, valid, {}))
    try:
        with bind_v5_execution(decoder, execution, role='packed'):
            output = decoder(inputs_embeds=embeds, attention_mask=bias, position_ids=ids,
                stage2_prefix_key_values=prefixes, use_cache=False, return_dict=True)
    finally:
        _PACKED.reset(token)
    row_indices, offsets = zip(*[(i, t-1-r.start) for i, r in enumerate(rows) for t in r.targets])
    selected = output.last_hidden_state[row_indices, offsets]
    return torch.cat([lm_head(chunk) for chunk in selected.split(chunk_positions)], dim=0)


def run_v5_packed_replay(decoder: nn.Module, lm_head: nn.Module, trace: V5Trace,
                        blocks: tuple[V5ReplayBlock, ...], *, execution, need_voff: bool,
                        need_textoff: bool, chunk_positions: int = 128,
                        voff_mode: str = 'attention_mask') -> V5ReplayOutput:
    modes = ('attention_mask', 'cross_helper_reverse_topk_independent',
             'cross_helper_reverse_topk_cumulative')
    if voff_mode not in modes:
        raise ValueError('unsupported v5 replay voff_mode')
    if (type(need_voff) is not bool or type(need_textoff) is not bool
            or type(chunk_positions) is not int or chunk_positions <= 0):
        raise ValueError('v5 replay requires boolean branches and positive chunk_positions')
    keys = tuple((b.batch_index, t-1) for b in blocks for t in b.target_positions)
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate v5 replay queries')
    # Preserve the historical execution path exactly, including Text-off-only
    # and no-query behavior, without requiring or constructing donor features.
    if voff_mode == 'attention_mask' or not need_voff or not keys:
        return _run_attention_mask_replay(decoder, lm_head, trace, blocks,
            execution=execution, need_voff=need_voff, need_textoff=need_textoff,
            chunk_positions=chunk_positions)
    x = trace.inputs_embeds
    if (x is None or trace.padding is None or trace.position_ids is None
            or len(trace.key_values) != len(decoder.layers)):
        raise ValueError('replacement replay requires a complete attached clean trace')
    previous = {}
    for b in blocks:
        v = b.replacement_values
        if (type(b.batch_index) is not int or not 0 <= b.batch_index < x.shape[0]
                or not 0 <= b.start < b.end-1 < x.shape[1]-1
                or b.start <= previous.get(b.batch_index, -1)
                or not isinstance(v, Tensor) or v.shape != (b.end-b.start-1, x.shape[-1])
                or not v.is_floating_point() or v.device != x.device or not bool(torch.isfinite(v).all())
                or len(b.target_ids) != len(b.target_positions)
                or len(b.blocked_text_keys) != len(b.target_positions)
                or tuple(sorted(set(b.target_positions))) != b.target_positions
                or any(t <= b.end or t >= x.shape[1] for t in b.target_positions)
                or any(k < 0 or k >= b.start for ks in b.blocked_text_keys for k in ks)
                or not bool(trace.padding[b.batch_index, b.start:max(b.target_positions, default=b.end)+1].bool().all())):
            raise ValueError('invalid replacement block values or coordinates')
        previous[b.batch_index] = b.end
    active = tuple(b for b in blocks if b.target_positions)
    if voff_mode == 'cross_helper_reverse_topk_independent':
        rows = tuple(_ReplacementRow(b.batch_index, b.start+1, max(b.target_positions),
            b.target_positions, ((),)*len(b.target_positions), (b,)) for b in active)
        if need_textoff:
            rows += tuple(_ReplacementRow(b.batch_index, b.end, max(b.target_positions),
                b.target_positions, b.blocked_text_keys, ()) for b in active)
        logits = _replacement_rows(decoder, lm_head, trace, rows, execution, chunk_positions)
        return V5ReplayOutput(logits[:len(keys)], logits[len(keys):] if need_textoff else None, keys)
    # One causal suffix retains all prior replacements, including blocks whose
    # immediate text has no eligible auxiliary targets.
    rows = []
    for sample in dict.fromkeys(b.batch_index for b in blocks):
        owned = tuple(b for b in blocks if b.batch_index == sample)
        targets = tuple(t for b in owned for t in b.target_positions)
        if not targets:
            continue
        stop = max(targets)
        replacements = tuple(b for b in owned if b.start+1 < stop)
        rows.append(_ReplacementRow(sample, owned[0].start+1, stop, targets,
                                    ((),)*len(targets), replacements))
    voff = _replacement_rows(decoder, lm_head, trace, tuple(rows), execution, chunk_positions)
    row_keys = tuple((r.batch_index, t-1) for r in rows for t in r.targets)
    lookup = {key:i for i,key in enumerate(row_keys)}
    voff = voff.index_select(0, torch.tensor([lookup[k] for k in keys], device=x.device))
    textoff = (_run_attention_mask_replay(decoder, lm_head, trace, blocks,
        execution=execution, need_voff=False, need_textoff=True,
        chunk_positions=chunk_positions).textoff_logits if need_textoff else None)
    return V5ReplayOutput(voff, textoff, keys)
