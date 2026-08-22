"""Native ILVR evidence-span and reasoning-window layout extraction."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenIds:
    """Tokenizer-resolved IDs needed to parse ILVR latent spans."""

    latent_start: int
    latent_end: int
    latent_pad: int
    padding: int
    special: frozenset[int]
    assistant_termination: frozenset[int] = frozenset()


@dataclass(frozen=True)
class InterleavedLayout:
    """Padded evidence/window masks and their native predecessor relations."""

    evidence_token_mask: torch.Tensor
    window_token_mask: torch.Tensor
    evidence_valid: torch.Tensor
    window_valid: torch.Tensor
    causal_pair_mask: torch.Tensor
    training_pair_mask: torch.Tensor
    native_predecessor: torch.Tensor


def extract_interleaved_layout(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    active_latent_mask: torch.Tensor,
    token_ids: TokenIds,
) -> InterleavedLayout:
    """Extract selected latent evidence and supervised text reasoning windows.

    Every matched delimiter span remains structurally indexed.  Only the
    selected portion of its interior is valid evidence, so an inactive span
    cannot create causal or native-training pairs even though a later window
    can still name it as its structural predecessor.
    """
    _validate_inputs(input_ids, attention_mask, labels, active_latent_mask)

    batch_size, sequence_length = input_ids.shape
    rows = [
        _extract_row(
            input_ids[row], attention_mask[row], labels[row], active_latent_mask[row], token_ids
        )
        for row in range(batch_size)
    ]
    max_evidence = max((len(row[0]) for row in rows), default=0)
    max_windows = max((len(row[1]) for row in rows), default=0)
    device = input_ids.device

    evidence_token_mask = torch.zeros(
        (batch_size, max_evidence, sequence_length), dtype=torch.bool, device=device
    )
    window_token_mask = torch.zeros(
        (batch_size, max_windows, sequence_length), dtype=torch.bool, device=device
    )
    evidence_valid = torch.zeros((batch_size, max_evidence), dtype=torch.bool, device=device)
    window_valid = torch.zeros((batch_size, max_windows), dtype=torch.bool, device=device)
    native_predecessor = torch.full(
        (batch_size, max_windows), -1, dtype=torch.long, device=device
    )
    evidence_end = torch.full((batch_size, max_evidence), -1, dtype=torch.long, device=device)
    window_start = torch.full((batch_size, max_windows), -1, dtype=torch.long, device=device)

    for row_index, (evidence, windows) in enumerate(rows):
        for evidence_index, (_, end, mask) in enumerate(evidence):
            evidence_token_mask[row_index, evidence_index] = mask
            evidence_valid[row_index, evidence_index] = mask.any()
            evidence_end[row_index, evidence_index] = end
        for window_index, (start, predecessor, mask) in enumerate(windows):
            window_token_mask[row_index, window_index] = mask
            window_valid[row_index, window_index] = True
            native_predecessor[row_index, window_index] = predecessor
            window_start[row_index, window_index] = start

    causal_pair_mask = (
        evidence_valid.unsqueeze(-1)
        & window_valid.unsqueeze(1)
        & (evidence_end.unsqueeze(-1) < window_start.unsqueeze(1))
    )

    training_pair_mask = torch.zeros_like(causal_pair_mask)
    if max_evidence and max_windows:
        predecessor = native_predecessor.clamp_min(0)
        predecessor_is_causal = causal_pair_mask.gather(1, predecessor.unsqueeze(1)).squeeze(1)
        training_pair_mask.scatter_(
            1,
            predecessor.unsqueeze(1),
            (native_predecessor.ge(0) & predecessor_is_causal).unsqueeze(1),
        )

    return InterleavedLayout(
        evidence_token_mask=evidence_token_mask,
        window_token_mask=window_token_mask,
        evidence_valid=evidence_valid,
        window_valid=window_valid,
        causal_pair_mask=causal_pair_mask,
        training_pair_mask=training_pair_mask,
        native_predecessor=native_predecessor,
    )


def _validate_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    active_latent_mask: torch.Tensor,
) -> None:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")
    expected_shape = input_ids.shape
    for name, value in {
        "attention_mask": attention_mask,
        "labels": labels,
        "active_latent_mask": active_latent_mask,
    }.items():
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}")
        if value.device != input_ids.device:
            raise ValueError(f"{name} must be on the same device as input_ids")


def _extract_row(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    active_latent_mask: torch.Tensor,
    token_ids: TokenIds,
) -> tuple[list[tuple[int, int, torch.Tensor]], list[tuple[int, int, torch.Tensor]]]:
    """Parse delimiter coordinates and vectorized interval masks for one sequence."""
    length = input_ids.shape[0]
    active_attention = attention_mask.to(dtype=torch.bool)
    excluded_ids = set(token_ids.special)
    excluded_ids.update(
        {token_ids.latent_start, token_ids.latent_end, token_ids.latent_pad, token_ids.padding}
    )
    excluded_ids.update(token_ids.assistant_termination)
    positions = torch.arange(length, device=input_ids.device)
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
    evidence: list[tuple[int, int, torch.Tensor]] = []
    windows: list[tuple[int, int, torch.Tensor]] = []
    open_start: int | None = None
    text_start = 0

    for position, code in zip(event_positions, event_codes):
        if code == 1:
            if open_start is not None:
                raise ValueError("nested latent delimiter")
            _append_window_if_nonempty(
                windows, reasoning, positions,
                text_start,
                position,
                len(evidence) - 1,
            )
            open_start = position
        else:
            if open_start is None:
                raise ValueError("unmatched latent delimiter")
            evidence_mask = active_latent_mask.to(dtype=torch.bool) & (
                (positions > open_start) & (positions < position)
            )
            evidence.append((open_start, position, evidence_mask))
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
    _append_window_if_nonempty(
        windows, reasoning, positions,
        text_start,
        final_stop,
        len(evidence) - 1,
    )
    return evidence, windows


def _append_window_if_nonempty(
    windows: list[tuple[int, int, torch.Tensor]],
    reasoning: torch.Tensor,
    positions: torch.Tensor,
    start: int,
    stop: int,
    predecessor: int,
) -> None:
    """Append a nonempty vectorized reasoning interval."""
    if start >= stop:
        return
    mask = reasoning & (positions >= start) & (positions < stop)
    reasoning_positions = mask.nonzero(as_tuple=False).flatten().tolist()
    if reasoning_positions:
        windows.append((reasoning_positions[0], predecessor, mask))
