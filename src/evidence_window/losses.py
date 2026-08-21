"""Stable preference losses over global, local, and evidence-window margins."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def global_preference_loss(
    margins: torch.Tensor, valid: torch.Tensor, beta: float
) -> torch.Tensor:
    """Apply one preference comparison to the sum of valid margins per example."""
    _validate_margin_mask(margins, valid)
    beta_value = _validate_beta(beta)
    valid_float = valid.to(dtype=margins.dtype)
    flat_margins = margins.flatten(start_dim=1)
    flat_valid = valid_float.flatten(start_dim=1)
    row_valid = flat_valid.any(dim=1)
    if not row_valid.any():
        return margins.sum() * 0.0
    row_values = (flat_margins * flat_valid).sum(dim=1)
    row_losses = -F.logsigmoid(beta_value * row_values)
    return row_losses[row_valid].mean()


def window_preference_loss(
    margins: torch.Tensor, valid: torch.Tensor, beta: float
) -> torch.Tensor:
    """Average independent preference losses over valid windows."""
    _validate_margin_mask(margins, valid)
    beta_value = _validate_beta(beta)
    valid_float = valid.to(dtype=margins.dtype)
    count = valid_float.sum()
    if not count.bool().item():
        return margins.sum() * 0.0
    return (-F.logsigmoid(beta_value * margins) * valid_float).sum() / count


def evidence_window_preference_loss(
    margins: torch.Tensor,
    pair_mask: torch.Tensor,
    beta: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average weighted independent losses over valid evidence-window pairs."""
    _validate_margin_mask(margins, pair_mask)
    beta_value = _validate_beta(beta)
    if weights is None:
        effective_weights = pair_mask.to(dtype=margins.dtype)
    else:
        if not torch.is_tensor(weights) or weights.shape != margins.shape:
            raise ValueError("weights must have the same shape as margins")
        if weights.device != margins.device:
            raise ValueError("weights and margins must be on the same device")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("weights must be finite and nonnegative")
        effective_weights = weights.to(dtype=margins.dtype) * pair_mask.to(dtype=margins.dtype)
    count = effective_weights.sum()
    if not count.bool().item():
        return margins.sum() * 0.0
    return (-F.logsigmoid(beta_value * margins) * effective_weights).sum() / count


def _validate_margin_mask(margins: torch.Tensor, valid: torch.Tensor) -> None:
    if not torch.is_tensor(margins) or margins.ndim < 2:
        raise ValueError("margins must have shape [B, ...]")
    if not torch.is_tensor(valid) or valid.shape != margins.shape:
        raise ValueError("valid mask must have the same shape as margins")
    if valid.device != margins.device:
        raise ValueError("valid mask and margins must be on the same device")


def _validate_beta(beta: float) -> float:
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise ValueError("beta must be a finite positive number")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a finite positive number")
    return float(beta)

