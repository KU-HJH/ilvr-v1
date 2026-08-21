"""Latent-value interventions used by evidence-window counterfactuals."""

from __future__ import annotations

import operator

import torch


def apply_zero_latent_loo(
    values: torch.Tensor,
    active_mask: torch.Tensor,
    evidence_masks: torch.Tensor,
    evidence_index: int,
) -> torch.Tensor:
    """Return ``values`` with one evidence span's active latents zeroed.

    ``values`` is compact: its rows are in the row-major order of the true
    entries in ``active_mask``.  The same ordering is used by the model's
    ``masked_scatter`` injection, so the token mask can be mapped back to
    compact rows by selecting it at those true positions.
    """
    _validate_loo_inputs(values, active_mask, evidence_masks, evidence_index)

    source = values.reshape(-1, values.shape[-1])
    active_positions = active_mask.to(dtype=torch.bool).reshape(-1).nonzero(as_tuple=False).flatten()
    if source.shape[0] != active_positions.numel():
        raise ValueError("compact latent order does not match image_out_mask")

    selected_token_rows = evidence_masks[:, evidence_index].to(dtype=torch.bool).reshape(-1)
    zero_rows = selected_token_rows.index_select(0, active_positions)
    result = source.clone()
    result[zero_rows] = 0
    return result.reshape_as(values)


def apply_all_null(values: torch.Tensor) -> torch.Tensor:
    """Return a zero-valued copy of every compact latent vector."""
    return torch.zeros_like(values)


def select_loo_spans(valid: torch.Tensor, cap: int) -> list[list[int]]:
    """Select deterministic, temporally ordered evidence ordinals per row.

    A zero cap means full coverage.  Otherwise, valid ordinals are sampled at
    evenly spaced integer positions; duplicate rounded positions are removed,
    then the earliest remaining positions fill any collisions.  This keeps
    the result deterministic and guarantees the requested count.
    """
    if not isinstance(valid, torch.Tensor) or valid.ndim != 2:
        raise ValueError("valid must have shape [B, J]")
    try:
        cap = operator.index(cap)
    except TypeError as exc:
        raise TypeError("cap must be an integer") from exc
    if cap < 0:
        raise ValueError("cap must be non-negative")

    selected: list[list[int]] = []
    for row in valid.to(dtype=torch.bool):
        valid_indices = row.nonzero(as_tuple=False).flatten().tolist()
        target = len(valid_indices) if cap == 0 else min(cap, len(valid_indices))
        if target == 0:
            selected.append([])
            continue
        if target == len(valid_indices):
            selected.append(valid_indices)
            continue
        if target == 1:
            selected.append([valid_indices[0]])
            continue

        # Round positions rather than values, retaining temporal spacing even
        # when valid ordinals are sparse in the original sequence.
        positions = {
            round(index * (len(valid_indices) - 1) / (target - 1))
            for index in range(target)
        }
        for position in range(len(valid_indices)):
            if len(positions) >= target:
                break
            positions.add(position)
        selected.append([valid_indices[position] for position in sorted(positions)])

    return selected


def _validate_loo_inputs(
    values: torch.Tensor,
    active_mask: torch.Tensor,
    evidence_masks: torch.Tensor,
    evidence_index: int,
) -> None:
    if not isinstance(values, torch.Tensor) or values.ndim != 3:
        raise ValueError("values must have shape [B, N, D]")
    if not isinstance(active_mask, torch.Tensor) or active_mask.ndim != 2:
        raise ValueError("active_mask must have shape [B, T]")
    if not isinstance(evidence_masks, torch.Tensor) or evidence_masks.ndim != 3:
        raise ValueError("evidence_masks must have shape [B, J, T]")
    if active_mask.shape[0] != values.shape[0]:
        raise ValueError("active_mask batch dimension must match values")
    if evidence_masks.shape[0] != active_mask.shape[0]:
        raise ValueError("evidence_masks batch dimension must match active_mask")
    if evidence_masks.shape[2] != active_mask.shape[1]:
        raise ValueError("evidence_masks token dimension must match active_mask")
    if values.device != active_mask.device or values.device != evidence_masks.device:
        raise ValueError("intervention tensors must be on the same device")
    try:
        evidence_index = operator.index(evidence_index)
    except TypeError as exc:
        raise TypeError("evidence_index must be an integer") from exc
    if evidence_index < 0 or evidence_index >= evidence_masks.shape[1]:
        raise IndexError("evidence_index is out of range")
