"""Pure planning and dense visibility for shared-prefix Stage 2B sidecars."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from step_preference.negatives import StepNegative
from step_preference.packing import _one_contiguous_length, _validate_clean_inputs, _validate_spans
from step_preference.spans import ReasoningStepSpan


SidecarBranchKind = Literal[
    "visual_off", "textual_off", "visual_rejected", "textual_rejected"
]


@dataclass(frozen=True)
class SidecarBranch:
    """One branch-local suffix replayed against an immutable clean prefix."""

    name: str
    kind: SidecarBranchKind
    step_id: int
    token_ids: tuple[int, ...]
    clean_embedding_sources: tuple[int, ...]
    logical_positions: tuple[int, ...]
    clean_ancestor_end: int
    blocked_start: int
    blocked_end: int
    selected_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    rejected_model_inputs: Mapping[str, object] | None
    zero_first_embedding: bool = False


@dataclass(frozen=True)
class SidecarPlan:
    """The clean root score rows and visual/textual suffix groups."""

    root_length: int
    root_selected_positions: tuple[int, ...]
    groups: tuple[tuple[SidecarBranch, ...], tuple[SidecarBranch, ...]]
    context_preference: bool = False


def _chosen_branch(
    *,
    clean_ids: torch.Tensor,
    span: ReasoningStepSpan,
    modality: Literal["visual", "textual"],
    suppress_latent_boundaries: bool = False,
) -> SidecarBranch:
    start = span.target_text_start - 1
    if start < span.latent_end:
        raise ValueError("chosen branch begins before the completed latent prefix")
    stop = span.target_text_end - 1
    source_positions = tuple(range(start, stop))
    if modality == "visual":
        blocked_start, blocked_end = (
            (span.latent_start - 1, span.latent_end + 1)
            if suppress_latent_boundaries
            else (span.latent_start, span.latent_end)
        )
    else:
        blocked_start, blocked_end = span.prev_text_start, span.prev_text_end
    return SidecarBranch(
        name=f"step-{span.step_id}-{modality}-off",
        kind=f"{modality}_off",
        step_id=span.step_id,
        token_ids=tuple(int(clean_ids[position].item()) for position in source_positions),
        clean_embedding_sources=source_positions,
        logical_positions=source_positions,
        clean_ancestor_end=start,
        blocked_start=blocked_start,
        blocked_end=blocked_end,
        selected_positions=tuple(range(stop - start)),
        target_token_ids=tuple(
            int(clean_ids[position].item())
            for position in range(span.target_text_start, span.target_text_end)
        ),
        rejected_model_inputs=None,
        zero_first_embedding=suppress_latent_boundaries,
    )


def _rejected_branch(
    *,
    rejected: Mapping[str, object],
    clean_ids: torch.Tensor,
    span: ReasoningStepSpan,
    modality: Literal["visual", "textual"],
) -> SidecarBranch:
    model_inputs = rejected.get("model_inputs")
    completion_positions = rejected.get("completion_token_positions")
    target_token_ids = rejected.get("completion_token_ids")
    if not isinstance(model_inputs, Mapping) or not isinstance(completion_positions, (tuple, list)):
        raise ValueError("rejected branch has invalid model inputs or completion_token_positions")

    input_ids = model_inputs.get("input_ids")
    attention_mask = model_inputs.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise ValueError("rejected branch must contain input_ids and attention_mask")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("rejected input_ids must have shape [1, sequence_length]")
    if tuple(attention_mask.shape) != tuple(input_ids.shape):
        raise ValueError("rejected attention_mask must match input_ids")
    rejected_length = _one_contiguous_length(attention_mask, "rejected attention_mask")
    rejected_ids = input_ids[0, :rejected_length].to(device="cpu", dtype=torch.long)

    if (
        not completion_positions
        or any(type(position) is not int for position in completion_positions)
        or completion_positions[0] < 1
        or completion_positions[-1] >= rejected_length
        or any(
            right <= left
            for left, right in zip(completion_positions, completion_positions[1:])
        )
    ):
        raise ValueError(
            "completion_token_positions must be strictly increasing causal positions"
        )
    expected_target_ids = tuple(
        int(rejected_ids[position].item()) for position in completion_positions
    )
    if not isinstance(target_token_ids, (tuple, list)) or tuple(target_token_ids) != expected_target_ids:
        raise ValueError("completion_token_ids must match rejected input_ids")

    common_prefix_length = 0
    for clean_id, rejected_id in zip(clean_ids.tolist(), rejected_ids.tolist()):
        if clean_id != rejected_id:
            break
        common_prefix_length += 1
    start = min(common_prefix_length, completion_positions[0] - 1)
    stop = completion_positions[-1]
    if start < span.latent_end:
        raise ValueError("rejected branch diverges before the completed latent prefix")

    token_ids = tuple(int(value) for value in rejected_ids[start:stop].tolist())
    selected_positions = tuple(
        position - 1 - start for position in completion_positions
    )
    return SidecarBranch(
        name=f"step-{span.step_id}-{modality}-rejected",
        kind=f"{modality}_rejected",
        step_id=span.step_id,
        token_ids=token_ids,
        clean_embedding_sources=(-1,) * len(token_ids),
        logical_positions=tuple(range(start, stop)),
        clean_ancestor_end=start,
        blocked_start=0,
        blocked_end=0,
        selected_positions=selected_positions,
        target_token_ids=expected_target_ids,
        rejected_model_inputs=model_inputs,
    )


def build_sidecar_plan(
    *,
    clean_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    spans: Sequence[ReasoningStepSpan],
    steps: Sequence[Mapping[str, object]],
) -> SidecarPlan:
    """Build branch-local suffix metadata for one right-padded clean sample."""
    clean_ids, root_length = _validate_clean_inputs(clean_input_ids, attention_mask)
    if len(steps) != len(spans):
        raise ValueError("steps must match spans")
    if not spans:
        return SidecarPlan(
            root_length=root_length,
            root_selected_positions=(),
            groups=((), ()),
        )
    validated_spans = _validate_spans(spans, root_length)

    visual_group: list[SidecarBranch] = []
    textual_group: list[SidecarBranch] = []
    root_selected_positions: list[int] = []
    for span, step in zip(validated_spans, steps):
        if not isinstance(step, Mapping) or step.get("span") != span:
            raise ValueError("joined step span must match its reasoning span")
        negative = step.get("negative")
        if not isinstance(negative, StepNegative) or negative.step_id != span.step_id:
            raise ValueError("joined step must contain its matching StepNegative")

        root_selected_positions.extend(
            range(span.target_text_start - 1, span.target_text_end - 1)
        )
        for modality, group in (
            ("visual", visual_group),
            ("textual", textual_group),
        ):
            group.append(_chosen_branch(clean_ids=clean_ids, span=span, modality=modality))
            rejected = step.get(f"{modality}_rejected")
            is_valid = getattr(negative, f"valid_{modality}_negative")
            if is_valid != (rejected is not None):
                raise ValueError(
                    f"{modality} rejected branch must match StepNegative validity"
                )
            if rejected is not None:
                if not isinstance(rejected, Mapping):
                    raise ValueError("rejected branch must be a mapping")
                group.append(
                    _rejected_branch(
                        rejected=rejected,
                        clean_ids=clean_ids,
                        span=span,
                        modality=modality,
                    )
                )

    return SidecarPlan(
        root_length=root_length,
        root_selected_positions=tuple(root_selected_positions),
        groups=(tuple(visual_group), tuple(textual_group)),
    )


def build_context_sidecar_plan(
    *,
    clean_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    spans: Sequence[ReasoningStepSpan],
    visual_only: bool = False,
    suppress_latent_boundaries: bool = False,
) -> SidecarPlan:
    """Build local visual/textual ablations for every clean target step."""
    if type(visual_only) is not bool:
        raise ValueError("visual_only must be a boolean")
    if type(suppress_latent_boundaries) is not bool:
        raise ValueError("suppress_latent_boundaries must be a boolean")
    if suppress_latent_boundaries and not visual_only:
        raise ValueError("suppress_latent_boundaries requires visual_only")
    clean_ids, root_length = _validate_clean_inputs(clean_input_ids, attention_mask)
    # A full-text target may contain several entries; the next previous-text
    # ablation still covers only the last entry, an exact suffix of that target.
    validated_spans = _validate_spans(spans, root_length, allow_target_suffix=True) if spans else ()
    if suppress_latent_boundaries and any(
        span.target_text_start != span.latent_end + 1
        for span in validated_spans
    ):
        raise ValueError(
            "boundary-suppressed target must immediately follow latent_end"
        )
    root_selected_positions = tuple(
        position
        for span in validated_spans
        for position in range(span.target_text_start - 1, span.target_text_end - 1)
    )
    visual_group = tuple(
        _chosen_branch(
            clean_ids=clean_ids,
            span=span,
            modality="visual",
            suppress_latent_boundaries=suppress_latent_boundaries,
        )
        for span in validated_spans
    )
    textual_group = tuple(
        _chosen_branch(clean_ids=clean_ids, span=span, modality="textual")
        for span in validated_spans
    ) if not visual_only else ()
    groups = (
        visual_group,
        textual_group,
    )
    return SidecarPlan(
        root_length=root_length,
        root_selected_positions=root_selected_positions,
        groups=groups,
        context_preference=True,
    )


def chunk_sidecar_branches(
    branches: Sequence[SidecarBranch], query_budget: int
) -> tuple[tuple[SidecarBranch, ...], ...]:
    """Greedily chunk whole branches under a suffix-row budget."""
    if type(query_budget) is not int or query_budget <= 0:
        raise ValueError("query_budget must be a positive integer")

    chunks: list[tuple[SidecarBranch, ...]] = []
    current: list[SidecarBranch] = []
    current_rows = 0
    for branch in branches:
        rows = len(branch.token_ids)
        if rows <= 0:
            raise ValueError("sidecar branches must contain at least one query row")
        if current and current_rows + rows > query_budget:
            chunks.append(tuple(current))
            current = []
            current_rows = 0
        if rows > query_budget:
            chunks.append((branch,))
        else:
            current.append(branch)
            current_rows += rows
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


def build_sidecar_flex_block_mask(root_length, branches, device):
    """Build rectangular suffix visibility; padded kernel probes have no edges."""
    from torch.nn.attention.flex_attention import create_block_mask

    if type(root_length) is not int or root_length < 0:
        raise ValueError("root_length must be a nonnegative integer")
    rows = []
    offset = 0
    for branch in branches:
        length = len(branch.token_ids)
        rows.extend([(offset, branch.clean_ancestor_end, branch.blocked_start,
                      branch.blocked_end)] * length)
        offset += length
    if not rows:
        raise ValueError("Flex sidecars require at least one query row")
    metadata = torch.tensor(rows, device=device, dtype=torch.long)
    query_length = offset
    kv_length = root_length + query_length

    def mask_mod(batch, head, query, key):
        # create_block_mask probes whole blocks beyond unpadded Q/KV lengths.
        start, ancestor, blocked_start, blocked_end = metadata[query.clamp(max=query_length-1)].unbind(-1)
        root_edge = (key < ancestor) & ~((key >= blocked_start) & (key < blocked_end))
        suffix_edge = (key >= root_length + start) & (key <= root_length + query)
        return (query < query_length) & (key < kv_length) & (root_edge | suffix_edge)

    return create_block_mask(mask_mod, B=1, H=None, Q_LEN=query_length,
                             KV_LEN=kv_length, device=device)


def sidecar_dense_visibility(
    root_length: int,
    branches: Sequence[SidecarBranch],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return rectangular root/branch visibility as a bool ``[Q, root+Q]`` tensor."""
    if type(root_length) is not int or root_length < 0:
        raise ValueError("root_length must be a nonnegative integer")
    branch_tuple = tuple(branches)
    query_rows = sum(len(branch.token_ids) for branch in branch_tuple)
    visible = torch.zeros(
        (query_rows, root_length + query_rows), dtype=torch.bool, device=device
    )
    root_positions = torch.arange(root_length, device=device)

    query_offset = 0
    for branch in branch_tuple:
        branch_rows = len(branch.token_ids)
        if branch_rows <= 0:
            raise ValueError("sidecar branches must contain at least one query row")
        if not (
            0 <= branch.clean_ancestor_end <= root_length
            and 0 <= branch.blocked_start <= branch.blocked_end <= root_length
        ):
            raise ValueError("sidecar root visibility bounds are out of range")
        root_visible = (root_positions < branch.clean_ancestor_end) & ~(
            (root_positions >= branch.blocked_start)
            & (root_positions < branch.blocked_end)
        )
        visible[
            query_offset : query_offset + branch_rows, :root_length
        ] = root_visible
        local_causal = torch.ones(
            (branch_rows, branch_rows), dtype=torch.bool, device=device
        ).tril()
        local_start = root_length + query_offset
        visible[
            query_offset : query_offset + branch_rows,
            local_start : local_start + branch_rows,
        ] = local_causal
        query_offset += branch_rows
    return visible
