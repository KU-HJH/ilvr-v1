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

    for row_index, (evidence, windows) in enumerate(rows):
        for evidence_index, (_, _, mask) in enumerate(evidence):
            evidence_token_mask[row_index, evidence_index] = mask
            evidence_valid[row_index, evidence_index] = mask.any()
        for window_index, (start, predecessor, mask) in enumerate(windows):
            del start  # Temporal positions were needed below to construct pair masks.
            window_token_mask[row_index, window_index] = mask
            window_valid[row_index, window_index] = True
            native_predecessor[row_index, window_index] = predecessor

    causal_pair_mask = torch.zeros(
        (batch_size, max_evidence, max_windows), dtype=torch.bool, device=device
    )
    for row_index, (evidence, windows) in enumerate(rows):
        for evidence_index, (_, evidence_end, _) in enumerate(evidence):
            if not evidence_valid[row_index, evidence_index]:
                continue
            for window_index, (window_start, _, _) in enumerate(windows):
                if evidence_end < window_start:
                    causal_pair_mask[row_index, evidence_index, window_index] = True

    training_pair_mask = torch.zeros_like(causal_pair_mask)
    for row_index in range(batch_size):
        for window_index in range(max_windows):
            predecessor = native_predecessor[row_index, window_index].item()
            if predecessor >= 0 and causal_pair_mask[row_index, predecessor, window_index]:
                training_pair_mask[row_index, predecessor, window_index] = True

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
    """Parse one sequence once, returning structural spans and nonempty windows."""
    length = input_ids.shape[0]
    active_attention = attention_mask.to(dtype=torch.bool)
    excluded_ids = set(token_ids.special)
    excluded_ids.update(
        {token_ids.latent_start, token_ids.latent_end, token_ids.latent_pad, token_ids.padding}
    )
    evidence: list[tuple[int, int, torch.Tensor]] = []
    windows: list[tuple[int, int, torch.Tensor]] = []
    open_start: int | None = None
    text_start = 0

    for position in range(length):
        if not active_attention[position].item():
            continue
        token = input_ids[position].item()
        if token == token_ids.latent_start:
            if open_start is not None:
                raise ValueError("nested latent delimiter")
            _append_window_if_nonempty(
                windows,
                input_ids,
                attention_mask,
                labels,
                text_start,
                position,
                len(evidence) - 1,
                excluded_ids,
            )
            open_start = position
        elif token == token_ids.latent_end:
            if open_start is None:
                raise ValueError("unmatched latent delimiter")
            evidence_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            if position > open_start + 1:
                interior = active_latent_mask[open_start + 1 : position].to(dtype=torch.bool)
                evidence_mask[open_start + 1 : position] = interior
            evidence.append((open_start, position, evidence_mask))
            open_start = None
            text_start = position + 1

    if open_start is not None:
        raise ValueError("unmatched latent delimiter")
    _append_window_if_nonempty(
        windows,
        input_ids,
        attention_mask,
        labels,
        text_start,
        length,
        len(evidence) - 1,
        excluded_ids,
    )
    return evidence, windows


def _append_window_if_nonempty(
    windows: list[tuple[int, int, torch.Tensor]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    start: int,
    stop: int,
    predecessor: int,
    excluded_ids: set[int],
) -> None:
    """Append a text interval only when it contains eligible supervised tokens."""
    if start >= stop:
        return
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for position in range(start, stop):
        if (
            attention_mask[position].item()
            and labels[position].item() != -100
            and input_ids[position].item() not in excluded_ids
        ):
            mask[position] = True
    if mask.any():
        window_start = mask.nonzero(as_tuple=False)[0, 0].item()
        windows.append((window_start, predecessor, mask))
