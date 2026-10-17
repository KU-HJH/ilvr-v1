"""Independent BCE terms for clean and V-off mean GT scores."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class DilBceResult:
    """Mean step loss and FP32 per-step diagnostics."""

    loss: Tensor
    pair_losses: Tensor
    positive_losses: Tensor
    negative_losses: Tensor
    policy_margins: Tensor


def _validate_beta(beta: float) -> None:
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and strictly positive")


def dil_bce_positive(clean_scores: Tensor, beta: float) -> Tensor:
    """Return softplus(-beta * clean score), independently of V-off scores."""
    _validate_beta(beta)
    if not clean_scores.is_floating_point():
        raise ValueError("clean scores must be floating-point")
    clean = clean_scores.float()
    if not torch.isfinite(clean).all():
        raise ValueError("clean scores must be finite after float32 conversion")
    return F.softplus(-beta * clean)


def dil_bce_loss(
    clean_scores: Tensor,
    off_scores: Tensor,
    beta: float,
    *,
    stage2_debugging: bool = True,
) -> DilBceResult:
    """Average the sum of independent clean-positive and V-off-negative BCE."""
    _validate_beta(beta)
    if clean_scores.shape != off_scores.shape:
        raise ValueError("clean and V-off score shapes are inconsistent")
    if clean_scores.device != off_scores.device:
        raise ValueError("clean and V-off score devices are inconsistent")
    if not clean_scores.is_floating_point() or not off_scores.is_floating_point():
        raise ValueError("clean and V-off scores must be floating-point")

    clean, off = clean_scores.float(), off_scores.float()
    if stage2_debugging and (not torch.isfinite(clean).all() or not torch.isfinite(off).all()):
        raise ValueError("clean and V-off scores must be finite after float32 conversion")

    positive = F.softplus(-beta * clean)
    negative = F.softplus(beta * off)
    pairs = positive + negative
    loss = pairs.mean() if pairs.numel() else clean.sum() * 0.0 + off.sum() * 0.0
    return DilBceResult(
        loss=loss,
        pair_losses=pairs,
        positive_losses=positive,
        negative_losses=negative,
        policy_margins=clean - off,
    )
