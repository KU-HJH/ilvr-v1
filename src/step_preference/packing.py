"""Pure layout and visibility planning for packed Stage 2B preference trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch
from torch.nn.attention.flex_attention import create_block_mask

from step_preference.negatives import StepNegative
from step_preference.spans import ReasoningStepSpan


PackedBranchKind = Literal[
    "visual_off", "textual_off", "visual_rejected", "textual_rejected"
]


@dataclass(frozen=True)
class PackedBranch:
    name: str
    kind: PackedBranchKind
    step_id: int
    packed_start: int
    packed_end: int
    clean_ancestor_end: int
    blocked_start: int
    blocked_end: int
    logical_query_positions: tuple[int, ...]
    selected_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class PackedTreePlan:
    root_length: int
    unpadded_length: int
    padded_length: int
    token_ids: torch.Tensor
    clean_embedding_sources: torch.Tensor
    logical_position_sources: torch.Tensor
    branch_ids: torch.Tensor
    branch_local_positions: torch.Tensor
    clean_ancestor_ends: torch.Tensor
    blocked_starts: torch.Tensor
    blocked_ends: torch.Tensor
    valid_slots: torch.Tensor
    root_selected_positions: tuple[int, ...]
    branches: tuple[PackedBranch, ...]

    def branch(self, name: str) -> PackedBranch:
        matches = [branch for branch in self.branches if branch.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def selected_positions(self) -> torch.Tensor:
        positions = list(self.root_selected_positions)
        for branch in self.branches:
            positions.extend(branch.selected_positions)
        return torch.tensor(positions, dtype=torch.long)

    def to_model_inputs(self, device: torch.device) -> dict[str, object]:
        return {
            "block_mask": build_flex_block_mask(self, device),
            "token_ids": self.token_ids.to(device),
            "clean_embedding_sources": self.clean_embedding_sources.to(device),
            "logical_position_sources": self.logical_position_sources.to(device),
            "valid_slots": self.valid_slots.to(device),
            "root_length": self.root_length,
        }


def _one_contiguous_length(value: torch.Tensor, name: str) -> int:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1, sequence_length]")
    mask = value[0].to(dtype=torch.bool, device="cpu")
    length = int(mask.sum().item())
    if length <= 0 or not bool(mask[:length].all()) or bool(mask[length:].any()):
        raise ValueError(f"{name} must contain one nonempty contiguous prefix")
    return length


def _validate_clean_inputs(
    clean_input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, int]:
    if (
        not isinstance(clean_input_ids, torch.Tensor)
        or clean_input_ids.ndim != 2
        or clean_input_ids.shape[0] != 1
    ):
        raise ValueError("clean_input_ids must have shape [1, sequence_length]")
    if tuple(attention_mask.shape) != tuple(clean_input_ids.shape):
        raise ValueError("attention_mask must match clean_input_ids")
    length = _one_contiguous_length(attention_mask, "attention_mask")
    return clean_input_ids[0, :length].to(device="cpu", dtype=torch.long), length


def _validate_spans(
    spans: Sequence[ReasoningStepSpan], root_length: int, *, allow_target_suffix: bool = False
) -> tuple[ReasoningStepSpan, ...]:
    if not spans:
        raise ValueError("packing requires at least one reasoning step")
    validated = tuple(spans)
    previous_target_start = 0
    previous_target_end = 0
    for expected_step_id, span in enumerate(validated):
        if not isinstance(span, ReasoningStepSpan) or span.step_id != expected_step_id:
            raise ValueError("reasoning step IDs must be contiguous and start at zero")
        bounds = (
            span.prev_text_start,
            span.prev_text_end,
            span.latent_start,
            span.latent_end,
            span.target_text_start,
            span.target_text_end,
        )
        if any(type(value) is not int for value in bounds) or not (
            0 <= bounds[0] < bounds[1] <= bounds[2] < bounds[3] <= bounds[4] < bounds[5] <= root_length
        ):
            raise ValueError("reasoning spans must be ordered nonempty intervals in root")
        if span.target_text_start < 1:
            raise ValueError("reasoning target must have a causal prediction row")
        if expected_step_id:
            shares_previous_target = (
                span.prev_text_start == previous_target_start
                and span.prev_text_end == previous_target_end
            )
            follows_previous_target = span.prev_text_start >= previous_target_end
            is_target_suffix = (
                allow_target_suffix
                and previous_target_start <= span.prev_text_start
                and span.prev_text_end == previous_target_end
            )
            if not (shares_previous_target or follows_previous_target or is_target_suffix):
                raise ValueError("reasoning spans must follow the clean trajectory")
        previous_target_start = span.target_text_start
        previous_target_end = span.target_text_end
    return validated


def _append_branch(
    *,
    branches: list[PackedBranch],
    token_chunks: list[torch.Tensor],
    embedding_chunks: list[torch.Tensor],
    logical_chunks: list[torch.Tensor],
    branch_chunks: list[torch.Tensor],
    local_chunks: list[torch.Tensor],
    ancestor_chunks: list[torch.Tensor],
    blocked_start_chunks: list[torch.Tensor],
    blocked_end_chunks: list[torch.Tensor],
    name: str,
    kind: PackedBranchKind,
    step_id: int,
    token_ids: torch.Tensor,
    clean_embedding_sources: torch.Tensor,
    logical_sources: torch.Tensor,
    clean_ancestor_end: int,
    blocked_start: int,
    blocked_end: int,
    logical_query_positions: tuple[int, ...],
    target_token_ids: tuple[int, ...],
) -> None:
    packed_start = sum(chunk.numel() for chunk in token_chunks)
    length = token_ids.numel()
    if length <= 0:
        raise ValueError("packed branches must contain at least one input row")
    branch_id = len(branches) + 1
    token_chunks.append(token_ids)
    embedding_chunks.append(clean_embedding_sources)
    logical_chunks.append(logical_sources)
    branch_chunks.append(torch.full((length,), branch_id, dtype=torch.long))
    local_chunks.append(torch.arange(length, dtype=torch.long))
    ancestor_chunks.append(torch.full((length,), clean_ancestor_end, dtype=torch.long))
    blocked_start_chunks.append(torch.full((length,), blocked_start, dtype=torch.long))
    blocked_end_chunks.append(torch.full((length,), blocked_end, dtype=torch.long))
    selected_positions = tuple(
        packed_start + logical_position - int(logical_sources[0].item())
        for logical_position in logical_query_positions
    )
    branches.append(
        PackedBranch(
            name=name,
            kind=kind,
            step_id=step_id,
            packed_start=packed_start,
            packed_end=packed_start + length,
            clean_ancestor_end=clean_ancestor_end,
            blocked_start=blocked_start,
            blocked_end=blocked_end,
            logical_query_positions=logical_query_positions,
            selected_positions=selected_positions,
            target_token_ids=target_token_ids,
        )
    )


def _rejected_branch(
    rejected: Mapping[str, object], clean_ids: torch.Tensor, span: ReasoningStepSpan
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], int]:
    if not isinstance(rejected, Mapping):
        raise ValueError("rejected branch must be a mapping")
    model_inputs = rejected.get("model_inputs")
    positions = rejected.get("completion_token_positions")
    target_ids = rejected.get("completion_token_ids")
    if not isinstance(model_inputs, Mapping) or not isinstance(positions, tuple):
        raise ValueError("rejected branch has invalid model inputs or completion positions")
    input_ids = model_inputs.get("input_ids")
    attention_mask = model_inputs.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise ValueError("rejected branch must contain input_ids and attention_mask")
    if tuple(input_ids.shape) != tuple(attention_mask.shape):
        raise ValueError("rejected attention_mask must match input_ids")
    length = _one_contiguous_length(attention_mask, "rejected attention_mask")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("rejected input_ids must have shape [1, sequence_length]")
    rejected_ids = input_ids[0, :length].to(device="cpu", dtype=torch.long)
    if (
        not positions
        or any(type(position) is not int for position in positions)
        or positions[0] < 1
        or positions[-1] >= length
        or any(right <= left for left, right in zip(positions, positions[1:]))
    ):
        raise ValueError("completion_token_positions must be strictly increasing causal positions")
    expected_ids = tuple(int(rejected_ids[position].item()) for position in positions)
    if not isinstance(target_ids, tuple) or target_ids != expected_ids:
        raise ValueError("completion_token_ids must match rejected input_ids")
    lcp = 0
    for clean_id, rejected_id in zip(clean_ids.tolist(), rejected_ids.tolist()):
        if clean_id != rejected_id:
            break
        lcp += 1
    start = min(lcp, positions[0] - 1)
    end = positions[-1]
    if start < span.latent_end:
        raise ValueError("rejected branch diverges before the completed latent prefix")
    if end > clean_ids.numel():
        raise ValueError("rejected logical positions must exist in the clean root")
    query_positions = tuple(position - 1 for position in positions)
    return rejected_ids[start:end], query_positions, target_ids, start


def build_packed_tree_plan(
    *,
    clean_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    reasoning_spans: Sequence[ReasoningStepSpan],
    joined_steps: Sequence[Mapping[str, object]],
    pad_to_multiple: int = 128,
) -> PackedTreePlan:
    """Build immutable metadata for one clean root and its causal sidecars."""
    if pad_to_multiple != 128:
        raise ValueError("pad_to_multiple must be 128 for FlexAttention blocks")
    clean_ids, root_length = _validate_clean_inputs(clean_input_ids, attention_mask)
    spans = _validate_spans(reasoning_spans, root_length)
    if len(joined_steps) != len(spans):
        raise ValueError("joined_steps must match reasoning_spans")

    token_chunks = [clean_ids]
    embedding_chunks = [torch.arange(root_length, dtype=torch.long)]
    logical_chunks = [torch.arange(root_length, dtype=torch.long)]
    branch_chunks = [torch.zeros(root_length, dtype=torch.long)]
    local_chunks = [torch.arange(root_length, dtype=torch.long)]
    ancestor_chunks = [torch.zeros(root_length, dtype=torch.long)]
    blocked_start_chunks = [torch.full((root_length,), -1, dtype=torch.long)]
    blocked_end_chunks = [torch.full((root_length,), -1, dtype=torch.long)]
    root_selected_positions: list[int] = []
    branches: list[PackedBranch] = []

    for span, joined in zip(spans, joined_steps):
        if not isinstance(joined, Mapping) or joined.get("span") != span:
            raise ValueError("joined step span must match its reasoning span")
        negative = joined.get("negative")
        if not isinstance(negative, StepNegative) or negative.step_id != span.step_id:
            raise ValueError("joined step must contain its matching StepNegative")
        query_positions = tuple(range(span.target_text_start - 1, span.target_text_end - 1))
        target_ids = tuple(int(clean_ids[position].item()) for position in range(span.target_text_start, span.target_text_end))
        root_selected_positions.extend(query_positions)
        for modality, kind, source_start, source_end in (
            ("visual", "visual_off", span.latent_start, span.latent_end),
            ("textual", "textual_off", span.prev_text_start, span.prev_text_end),
        ):
            source = torch.arange(query_positions[0], query_positions[-1] + 1, dtype=torch.long)
            _append_branch(
                branches=branches,
                token_chunks=token_chunks,
                embedding_chunks=embedding_chunks,
                logical_chunks=logical_chunks,
                branch_chunks=branch_chunks,
                local_chunks=local_chunks,
                ancestor_chunks=ancestor_chunks,
                blocked_start_chunks=blocked_start_chunks,
                blocked_end_chunks=blocked_end_chunks,
                name=f"step-{span.step_id}-{modality}-off",
                kind=kind,
                step_id=span.step_id,
                token_ids=clean_ids[source],
                clean_embedding_sources=source,
                logical_sources=source,
                clean_ancestor_end=span.target_text_start - 1,
                blocked_start=source_start,
                blocked_end=source_end,
                logical_query_positions=query_positions,
                target_token_ids=target_ids,
            )
        for modality, kind in (
            ("visual", "visual_rejected"),
            ("textual", "textual_rejected"),
        ):
            rejected = joined.get(f"{modality}_rejected")
            is_valid = getattr(negative, f"valid_{modality}_negative")
            if is_valid != (rejected is not None):
                raise ValueError(f"{modality} rejected branch must match StepNegative validity")
            if rejected is None:
                continue
            token_ids, rejected_queries, rejected_target_ids, start = _rejected_branch(
                rejected, clean_ids, span
            )
            logical_sources = torch.arange(start, start + token_ids.numel(), dtype=torch.long)
            _append_branch(
                branches=branches,
                token_chunks=token_chunks,
                embedding_chunks=embedding_chunks,
                logical_chunks=logical_chunks,
                branch_chunks=branch_chunks,
                local_chunks=local_chunks,
                ancestor_chunks=ancestor_chunks,
                blocked_start_chunks=blocked_start_chunks,
                blocked_end_chunks=blocked_end_chunks,
                name=f"step-{span.step_id}-{modality}-rejected",
                kind=kind,
                step_id=span.step_id,
                token_ids=token_ids,
                clean_embedding_sources=torch.full((token_ids.numel(),), -1, dtype=torch.long),
                logical_sources=logical_sources,
                clean_ancestor_end=start,
                blocked_start=0,
                blocked_end=0,
                logical_query_positions=rejected_queries,
                target_token_ids=rejected_target_ids,
            )

    unpadded_length = sum(chunk.numel() for chunk in token_chunks)
    padded_length = ((unpadded_length + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    padding = padded_length - unpadded_length

    def padded(chunks: list[torch.Tensor], value: int) -> torch.Tensor:
        result = torch.cat(chunks)
        if padding:
            result = torch.cat((result, torch.full((padding,), value, dtype=result.dtype)))
        return result

    return PackedTreePlan(
        root_length=root_length,
        unpadded_length=unpadded_length,
        padded_length=padded_length,
        token_ids=padded(token_chunks, 0),
        clean_embedding_sources=padded(embedding_chunks, -1),
        logical_position_sources=padded(logical_chunks, 0),
        branch_ids=padded(branch_chunks, -1),
        branch_local_positions=padded(local_chunks, 0),
        clean_ancestor_ends=padded(ancestor_chunks, 0),
        blocked_starts=padded(blocked_start_chunks, 0),
        blocked_ends=padded(blocked_end_chunks, 0),
        valid_slots=padded([torch.ones(unpadded_length, dtype=torch.bool)], False),
        root_selected_positions=tuple(root_selected_positions),
        branches=tuple(branches),
    )


def dense_visibility(plan: PackedTreePlan) -> torch.Tensor:
    """Return the shared tree predicate as a CPU bool ``[P, P]`` matrix."""
    p = plan.padded_length
    metadata = (
        plan.logical_position_sources,
        plan.branch_ids,
        plan.branch_local_positions,
        plan.clean_ancestor_ends,
        plan.blocked_starts,
        plan.blocked_ends,
        plan.valid_slots,
    )
    if any(value.ndim != 1 or value.numel() != p for value in metadata):
        raise ValueError("plan metadata must be one-dimensional padded tensors")
    logical = plan.logical_position_sources.cpu()
    branch = plan.branch_ids.cpu()
    local = plan.branch_local_positions.cpu()
    ancestor = plan.clean_ancestor_ends.cpu()
    blocked_start = plan.blocked_starts.cpu()
    blocked_end = plan.blocked_ends.cpu()
    valid = plan.valid_slots.cpu()
    q_logical, kv_logical = logical[:, None], logical[None, :]
    q_branch, kv_branch = branch[:, None], branch[None, :]
    q_local, kv_local = local[:, None], local[None, :]
    root_visible = (q_branch == 0) & (kv_branch == 0) & (kv_logical <= q_logical)
    blocked_source = (kv_logical >= blocked_start[:, None]) & (kv_logical < blocked_end[:, None])
    sidecar_visible = (q_branch > 0) & (
        ((kv_branch == 0) & (kv_logical < ancestor[:, None]) & ~blocked_source)
        | ((kv_branch == q_branch) & (kv_local <= q_local))
    )
    padding_visible = (~valid[:, None]) & torch.eye(p, dtype=torch.bool)
    return (valid[:, None] & valid[None, :] & (root_visible | sidecar_visible)) | padding_visible


def build_flex_block_mask(plan: PackedTreePlan, device: torch.device) -> object:
    """Create the CUDA FlexAttention mask using the same tree predicate."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("FlexAttention packed masks require a CUDA device")
    logical = plan.logical_position_sources.to(device)
    branch = plan.branch_ids.to(device)
    local = plan.branch_local_positions.to(device)
    ancestor = plan.clean_ancestor_ends.to(device)
    blocked_start = plan.blocked_starts.to(device)
    blocked_end = plan.blocked_ends.to(device)
    valid = plan.valid_slots.to(device)

    def mask_mod(_batch, _head, q_idx, kv_idx):
        q_logical, kv_logical = logical[q_idx], logical[kv_idx]
        q_branch, kv_branch = branch[q_idx], branch[kv_idx]
        q_local, kv_local = local[q_idx], local[kv_idx]
        q_valid, kv_valid = valid[q_idx], valid[kv_idx]
        root_visible = (q_branch == 0) & (kv_branch == 0) & (kv_logical <= q_logical)
        blocked_source = (kv_logical >= blocked_start[q_idx]) & (kv_logical < blocked_end[q_idx])
        sidecar_visible = (q_branch > 0) & (
            ((kv_branch == 0) & (kv_logical < ancestor[q_idx]) & ~blocked_source)
            | ((kv_branch == q_branch) & (kv_local <= q_local))
        )
        padding_visible = (~q_valid) & (q_idx == kv_idx)
        return (q_valid & kv_valid & (root_visible | sidecar_visible)) | padding_visible

    return create_block_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=plan.padded_length,
        KV_LEN=plan.padded_length,
        device=device,
        BLOCK_SIZE=128,
    )
