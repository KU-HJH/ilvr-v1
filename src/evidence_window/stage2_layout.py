"""Marker-inclusive Stage-2 evidence and reasoning-window extraction."""

from dataclasses import dataclass
from typing import Sequence

import torch

from .layout import TokenIds


@dataclass(frozen=True, order=True)
class EvidenceWindowEdge:
    """One valid evidence-to-later-window intervention candidate."""

    batch_index: int
    evidence_index: int
    window_index: int


@dataclass(frozen=True)
class Stage2InterleavedLayout:
    """Padded marker-inclusive evidence and supervised Stage-2 text windows."""

    evidence_token_mask: torch.Tensor
    window_token_mask: torch.Tensor
    evidence_valid: torch.Tensor
    window_valid: torch.Tensor
    causal_pair_mask: torch.Tensor


def extract_stage2_layout(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    token_ids: TokenIds,
) -> Stage2InterleavedLayout:
    """Extract marker-inclusive evidence spans and causally eligible text windows.

    A delimiter pair is retained structurally even when it has no latent-pad
    interior.  Such a span is not valid evidence and therefore cannot produce
    an edge.  Windows contain only supervised assistant-text positions.
    """
    _validate_inputs(input_ids, attention_mask, labels)

    batch_size, sequence_length = input_ids.shape
    rows = [
        _extract_row(input_ids[row], attention_mask[row], labels[row], token_ids)
        for row in range(batch_size)
    ]
    max_evidence = max((len(evidence) for evidence, _ in rows), default=0)
    max_windows = max((len(windows) for _, windows in rows), default=0)
    device = input_ids.device

    evidence_token_mask = torch.zeros(
        (batch_size, max_evidence, sequence_length), dtype=torch.bool, device=device
    )
    window_token_mask = torch.zeros(
        (batch_size, max_windows, sequence_length), dtype=torch.bool, device=device
    )
    evidence_valid = torch.zeros((batch_size, max_evidence), dtype=torch.bool, device=device)
    window_valid = torch.zeros((batch_size, max_windows), dtype=torch.bool, device=device)
    evidence_end = torch.full((batch_size, max_evidence), -1, dtype=torch.long, device=device)
    window_start = torch.full((batch_size, max_windows), -1, dtype=torch.long, device=device)

    for row_index, (evidence, windows) in enumerate(rows):
        for evidence_index, (end, mask, valid) in enumerate(evidence):
            evidence_token_mask[row_index, evidence_index] = mask
            evidence_valid[row_index, evidence_index] = valid
            evidence_end[row_index, evidence_index] = end
        for window_index, (start, mask) in enumerate(windows):
            window_token_mask[row_index, window_index] = mask
            window_valid[row_index, window_index] = True
            window_start[row_index, window_index] = start

    causal_pair_mask = (
        evidence_valid.unsqueeze(-1)
        & window_valid.unsqueeze(1)
        & (evidence_end.unsqueeze(-1) < window_start.unsqueeze(1))
    )
    return Stage2InterleavedLayout(
        evidence_token_mask=evidence_token_mask,
        window_token_mask=window_token_mask,
        evidence_valid=evidence_valid,
        window_valid=window_valid,
        causal_pair_mask=causal_pair_mask,
    )


def validate_plan_types(
    layout: Stage2InterleavedLayout,
    plan_types: Sequence[Sequence[str]],
    sample_indices: Sequence[int] | torch.Tensor,
) -> None:
    """Reject sequence-plan metadata that disagrees with parsed evidence spans."""
    batch_size = layout.evidence_token_mask.shape[0]
    if len(plan_types) != batch_size or len(sample_indices) != batch_size:
        raise ValueError("plan metadata must have one entry per batch row")

    if isinstance(sample_indices, torch.Tensor):
        row_ids = sample_indices.tolist()
    else:
        row_ids = list(sample_indices)

    for batch_index, (row_plan, row_id) in enumerate(zip(plan_types, row_ids)):
        evidence_count = len(
            layout.evidence_token_mask[batch_index].any(dim=-1).nonzero(as_tuple=False)
        )
        actual = tuple(row_plan)
        valid_order = all(
            step in {"text", "latent"} for step in actual
        ) and all(left != right for left, right in zip(actual, actual[1:]))
        if actual.count("latent") != evidence_count or not valid_order:
            raise ValueError(
                f"sequence plan mismatch for row {row_id}: expected {evidence_count} latent "
                f"steps in alternating order, got {actual}"
            )


def select_stage2_edges(
    layout: Stage2InterleavedLayout,
    max_evidence_edges_per_window: int = 0,
) -> list[EvidenceWindowEdge]:
    """Return all valid causal edges, optionally capped nearest-first per window."""
    if max_evidence_edges_per_window < 0:
        raise ValueError("max_evidence_edges_per_window must be nonnegative")

    edges: list[EvidenceWindowEdge] = []
    batch_size, _, window_count = layout.causal_pair_mask.shape
    for batch_index in range(batch_size):
        for window_index in range(window_count):
            predecessor_indices = layout.causal_pair_mask[batch_index, :, window_index]
            evidence_indices = predecessor_indices.nonzero(as_tuple=False).flatten().tolist()
            if max_evidence_edges_per_window:
                evidence_indices = evidence_indices[-max_evidence_edges_per_window:]
            edges.extend(
                EvidenceWindowEdge(batch_index, evidence_index, window_index)
                for evidence_index in evidence_indices
            )
    return edges


def _validate_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")
    expected_shape = input_ids.shape
    for name, value in {"attention_mask": attention_mask, "labels": labels}.items():
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}")
        if value.device != input_ids.device:
            raise ValueError(f"{name} must be on the same device as input_ids")


def _extract_row(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    token_ids: TokenIds,
) -> tuple[list[tuple[int, torch.Tensor, torch.Tensor]], list[tuple[int, torch.Tensor]]]:
    """Parse one active sequence without scalar tensor reads."""
    length = input_ids.shape[0]
    active_attention = attention_mask.to(dtype=torch.bool)
    positions = torch.arange(length, device=input_ids.device)
    excluded_ids = set(token_ids.special)
    excluded_ids.update(
        {token_ids.latent_start, token_ids.latent_end, token_ids.latent_pad, token_ids.padding}
    )
    excluded_ids.update(token_ids.assistant_termination)
    excluded = torch.isin(input_ids, input_ids.new_tensor(tuple(excluded_ids)))
    reasoning = active_attention & labels.ne(-100) & ~excluded
    delimiter_codes = torch.where(
        active_attention & input_ids.eq(token_ids.latent_start),
        1,
        torch.where(active_attention & input_ids.eq(token_ids.latent_end), -1, 0),
    )
    delimiter_positions = delimiter_codes.ne(0).nonzero(as_tuple=False).flatten()
    event_positions = delimiter_positions.tolist()
    event_codes = delimiter_codes[delimiter_positions].tolist()

    evidence: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    windows: list[tuple[int, torch.Tensor]] = []
    open_start: int | None = None
    text_start = 0
    for position, code in zip(event_positions, event_codes):
        if code == 1:
            if open_start is not None:
                raise ValueError("nested latent delimiter")
            _append_window_if_nonempty(windows, reasoning, positions, text_start, position)
            open_start = position
        else:
            if open_start is None:
                raise ValueError("unmatched latent delimiter")
            evidence_mask = active_attention & positions.ge(open_start) & positions.le(position)
            latent_interior = (
                active_attention
                & input_ids.eq(token_ids.latent_pad)
                & positions.gt(open_start)
                & positions.lt(position)
            )
            evidence.append((position, evidence_mask, latent_interior.any()))
            open_start = None
            text_start = position + 1

    if open_start is not None:
        raise ValueError("unmatched latent delimiter")
    final_stop = length
    if token_ids.assistant_termination:
        termination_ids = input_ids.new_tensor(tuple(token_ids.assistant_termination))
        termination_positions = (
            active_attention
            & torch.isin(input_ids, termination_ids)
            & positions.ge(text_start)
        ).nonzero(as_tuple=False).flatten().tolist()
        if termination_positions:
            final_stop = termination_positions[0]
    _append_window_if_nonempty(windows, reasoning, positions, text_start, final_stop)
    return evidence, windows


def _append_window_if_nonempty(
    windows: list[tuple[int, torch.Tensor]],
    reasoning: torch.Tensor,
    positions: torch.Tensor,
    start: int,
    stop: int,
) -> None:
    mask = reasoning & positions.ge(start) & positions.lt(stop)
    reasoning_positions = mask.nonzero(as_tuple=False).flatten().tolist()
    if reasoning_positions:
        windows.append((reasoning_positions[0], mask))
