"""Dependency-free records in original padded decoder coordinates."""
from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from typing import ContextManager
    from torch import nn
    from .stage1_prefix_trace import TraceRequest, AttentionRows
    from .stage1_vision_gain_packed import Stage1PackedGainRequest

QueryKey = tuple[int, int]
BlockKey = tuple[int, int]


@dataclass(frozen=True)
class QueryMask:
    query: QueryKey
    text_keys: tuple[int, ...]
    image_keys: tuple[int, ...]


@dataclass(frozen=True)
class PrefixMaskPlan:
    rows: tuple[QueryMask, ...]
    snapshot_id: str
    parameter_version: int
    teacher_version: int
    schema_version: int = 3


@dataclass(frozen=True)
class Stage1ReferenceRequest:
    query_keys: tuple[QueryKey, ...]
    chunk_rows: int
    consume_logits: Callable[[tuple[QueryKey, ...], Tensor], None]

    def __post_init__(self):
        if type(self.chunk_rows) is not int or self.chunk_rows <= 0:
            raise ValueError("reference chunk_rows must be a positive integer")
        if not callable(self.consume_logits):
            raise ValueError("reference requires a logits consumer")
        if (not isinstance(self.query_keys, tuple)
                or any(not isinstance(key, tuple) or len(key) != 2
                       or any(type(x) is not int for x in key) for key in self.query_keys)
                or len(set(self.query_keys)) != len(self.query_keys)):
            raise ValueError("reference queries must be unique integer coordinate pairs")

    def validate_positions(self, shape):
        if any(not (0 <= b < shape[0] and 0 <= p < shape[1]) for b, p in self.query_keys):
            raise ValueError("reference query outside original decoder positions")


@dataclass(frozen=True)
class ConditionedLayout:
    """Original-coordinate replay geometry, shared by embedding and KV gathers.

    ``mask_bytes`` accounts only for the final additive mask. The retained
    boolean mask is reported separately; compilation uses O(B*T+R) row scratch
    and allocates no attention trace. Trace storage belongs to the observer.
    """
    indices: Tensor
    sample_ids: Tensor
    branch_ids: Tensor
    condition_ids: Tensor
    original_positions: Tensor
    prefix_cutoffs: Tensor
    allowed: Tensor
    score_rows: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    token_count: int
    mask_bytes: int

    @property
    def boolean_mask_bytes(self) -> int:
        return self.allowed.numel() * self.allowed.element_size()


def compile_conditioned_layout(request: 'Stage1PackedGainRequest', plan: PrefixMaskPlan, padding: Tensor, *,
                               element_size: int, max_packed_tokens: int | None,
                               max_mask_bytes: int | None) -> ConditionedLayout:
    """Compile isolated R/RV trajectories without touching a model or its cache."""
    import torch
    from .stage1_vision_gain_packed import Stage1PackedGainRequest

    if not isinstance(request, Stage1PackedGainRequest):
        raise ValueError('conditioned packing requires Stage1PackedGainRequest')
    if request.branch_mode not in {'independent', 'cumulative'}:
        raise ValueError('invalid conditioned branch mode')
    if padding.ndim != 2 or not all(padding.shape):
        raise ValueError('padding must be nonempty [B,T]')
    if type(element_size) is not int or element_size <= 0:
        raise ValueError('element_size must be a positive integer')
    for limit in (max_packed_tokens, max_mask_bytes):
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError('packing budget must be a nonnegative integer')
    if not isinstance(plan, PrefixMaskPlan) or plan.schema_version != 3:
        raise ValueError('unsupported prefix mask plan')
    batch, length = padding.shape
    valid, query_starts = [], {}
    for i, block in enumerate(request.blocks):
        b, start, end = block.batch_index, block.start_marker, block.end_marker
        targets = block.target_positions
        if (any(type(p) is not int for p in (b, start, end, *targets))
                or not 0 <= b < batch or not 0 <= start < start + 1 < end < length
                or len(targets) != len(block.target_ids)
                or tuple(sorted(set(targets))) != targets
                or any(not end < p < length for p in targets)):
            raise ValueError('invalid conditioned block or query coordinates')
        if targets:
            valid.append(i)
            for p in targets:
                query_starts.setdefault((b, p - 1), []).append(start)
    masks = {}
    for row in plan.rows:
        if row.query in masks or row.query not in query_starts:
            raise ValueError('duplicate or unowned mask query')
        b, query = row.query
        for keys in (row.text_keys, row.image_keys):
            if (len(set(keys)) != len(keys)
                    or any(type(p) is not int or not 0 <= p < query for p in keys)):
                raise ValueError('invalid mask source coordinates')
        if any(p >= min(query_starts[row.query]) for p in row.image_keys):
            raise ValueError('image source must precede the block start marker')
        masks[row.query] = row.text_keys + row.image_keys
    if request.branch_mode == 'independent':
        groups = [[i] for i in valid]
    else:
        groups = [[i for i in valid if request.blocks[i].batch_index == row]
                  for row in sorted({request.blocks[i].batch_index for i in valid})]
    spans = [(min(request.blocks[i].end_marker for i in group),
              max(request.blocks[i].target_positions[-1] for i in group)) for group in groups]
    count = 2 * sum(stop - start for start, stop in spans)
    bank_size = batch * length
    mask_bytes = count * (bank_size + count) * element_size
    # Integer-only budget arithmetic precedes replay lists and dense tensors.
    if ((max_packed_tokens is not None and count > max_packed_tokens)
            or (max_mask_bytes is not None and mask_bytes > max_mask_bytes)):
        raise ValueError(f'conditioned packing budget exceeded: tokens={count}, '
                         f'mask_bytes={mask_bytes}, boolean_mask_bytes={count * (bank_size + count)}')
    try:
        indices, samples, branches, conditions, originals, cutoffs = ([] for _ in range(6))
        score_rows = [[(), ()] for _ in request.blocks]
        for branch, (group, (start, stop)) in enumerate(zip(groups, spans)):
            sample = request.blocks[group[0]].batch_index
            for condition in (0, 1):
                cursor = len(indices)
                for position in range(start, stop):
                    indices.append(sample * length + position)
                    samples.append(sample)
                    branches.append(branch)
                    conditions.append(condition)
                    originals.append(position)
                    cutoffs.append(start)
                for i in group:
                    score_rows[i][condition] = tuple(
                        cursor + p - 1 - start for p in request.blocks[i].target_positions)
        device = padding.device
        indices, samples, branches, conditions, originals, cutoffs = (
            torch.tensor(values, dtype=torch.long, device=device)
            for values in (indices, samples, branches, conditions, originals, cutoffs))
        bank = torch.arange(bank_size, device=device)
        key_samples = torch.cat((bank // length, samples))
        key_positions = torch.cat((bank % length, originals))
        unpadded = padding.reshape(-1).bool()
        allowed = torch.empty((count, bank_size + count), dtype=torch.bool, device=device)
        cursor = 0
        for branch, (group, (start, stop)) in enumerate(zip(groups, spans)):
            sample = request.blocks[group[0]].batch_index
            for condition in (0, 1):
                for position in range(start, stop):
                    allowed[cursor, :bank_size] = (
                        (bank // length == sample) & (bank % length < start) & unpadded)
                    allowed[cursor, bank_size:] = (
                        (branches == branch) & (conditions == condition)
                        & (originals <= position) & unpadded[indices])
                    if condition == 1:
                        for i in group:
                            block = request.blocks[i]
                            if position >= block.end_marker:
                                blocked = ((key_samples == sample)
                                           & (key_positions > block.start_marker)
                                           & (key_positions < block.end_marker))
                                allowed[cursor, blocked] = False
                    keys = masks.get((sample, position), ())
                    if keys:
                        # Apply text and image keys together, including every replay
                        # copy of a source position, with O(B*T+R) row scratch.
                        blocked_positions = torch.zeros(length, dtype=torch.bool, device=device)
                        blocked_positions[torch.tensor(keys, dtype=torch.long, device=device)] = True
                        blocked = (key_samples == sample) & blocked_positions[key_positions]
                        allowed[cursor].masked_fill_(blocked, False)
                    cursor += 1
        for pair in score_rows:
            for rows in pair:
                for row in rows:
                    if not allowed[row].any().item():
                        raise ValueError('fully blocked scored row in conditioned layout')
    except (torch.OutOfMemoryError, MemoryError) as error:
        raise ValueError(f'conditioned packing OOM: tokens={count}, mask_bytes={mask_bytes}; '
                         'no blocks were dropped') from error
    return ConditionedLayout(indices, samples, branches, conditions, originals, cutoffs, allowed,
                             tuple(tuple(pair) for pair in score_rows), count, mask_bytes)


def materialize_conditioned_inputs(layout: ConditionedLayout, inputs_embeds: Tensor,
                                   position_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Gather both conditions with autograd intact and preserve original mRoPE."""
    import torch

    if inputs_embeds.ndim != 3 or not inputs_embeds.is_floating_point():
        raise ValueError('conditioned inputs must be floating [B,T,H] embeddings')
    batch, length, hidden = inputs_embeds.shape
    if (layout.allowed.shape != (layout.token_count, batch * length + layout.token_count)
            or layout.indices.device != inputs_embeds.device
            or position_ids.device != inputs_embeds.device):
        raise ValueError('conditioned layout and inputs have incompatible shapes or devices')
    if position_ids.shape == (batch, length):
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    if position_ids.shape != (3, batch, length):
        raise ValueError('conditioned position_ids must be [B,T] or [3,B,T]')
    if layout.allowed.numel() * inputs_embeds.element_size() != layout.mask_bytes:
        raise ValueError('embedding element size does not match preflight mask bytes')
    try:
        embeds = inputs_embeds.reshape(batch * length, hidden).index_select(0, layout.indices).unsqueeze(0)
        positions = position_ids.reshape(3, batch * length).index_select(1, layout.indices).unsqueeze(1)
        mask = inputs_embeds.new_zeros((1, 1, *layout.allowed.shape))
        mask.masked_fill_(~layout.allowed, -torch.inf)
    except (torch.OutOfMemoryError, MemoryError) as error:
        raise ValueError(f'conditioned materialization OOM: mask_bytes={layout.mask_bytes}; '
                         'no blocks were dropped') from error
    return embeds, positions, mask


@dataclass(frozen=True)
class ConditioningRequest:
    trace_request: 'TraceRequest'
    build_plan: Callable[[Tensor, 'nn.Module', Tensor, 'AttentionRows'], PrefixMaskPlan]
    max_packed_tokens: int | None
    max_mask_bytes: int | None
    role_scope: Callable[[str], 'ContextManager[None]']
    trace_chunk_rows: int = 32
    logit_chunk_rows: int = 32
    consume_positive_logits: Callable[[tuple[QueryKey, ...], Tensor], None] | None = None


@dataclass(frozen=True)
class ConditionedReplayScores:
    positive: Tensor
    negative: Tensor
