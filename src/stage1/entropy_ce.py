"""Detached clean-logit selection for the opt-in standard Stage1 CE loss."""
from __future__ import annotations

from fractions import Fraction
import math

import torch


def validate_ce_entropy_percent(percent: float) -> float:
    value = float(percent)
    if not math.isfinite(value) or not 0 < value <= 100:
        raise ValueError('CE entropy percent must be finite and in (0, 100]')
    return value


@torch.no_grad()
def select_entropy_ce_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    percent: float,
    special_token_ids: tuple[int, ...],
    chunk_positions: int = 128,
) -> torch.Tensor:
    """Retain sample-local high-entropy text targets and all supervised specials.

    The original labels and student inputs remain intact. Ranking a target at
    t uses its causal predictor at t-1; nonzero() supplies original-position
    order, which stable argsort retains when entropy values tie.
    """
    percent = validate_ce_entropy_percent(percent)
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError('CE entropy logits and labels must share [B, T]')
    if chunk_positions <= 0:
        raise ValueError('entropy chunk_positions must be positive')
    labels = labels.to(logits.device)
    selected = labels.clone()
    shifted = labels[:, 1:]
    candidates = shifted.ne(-100)
    if special_token_ids:
        specials = torch.tensor(special_token_ids, device=labels.device, dtype=labels.dtype)
        candidates &= ~torch.isin(shifted, specials)
    fraction = Fraction(str(percent)) / 100
    for sample in range(labels.shape[0]):
        predictors = candidates[sample].nonzero(as_tuple=True)[0]
        count = predictors.numel()
        keep = math.ceil(count * fraction)
        if keep == count:
            continue
        chunks = []
        for positions in predictors.split(chunk_positions):
            values = logits[sample].index_select(0, positions).float()
            log_probs = values.log_softmax(-1)
            chunks.append(-(log_probs.exp() * log_probs).sum(-1))
        entropy = torch.cat(chunks)
        if not bool(torch.isfinite(entropy).all().item()):
            raise FloatingPointError('CE selection entropy is nonfinite')
        order = torch.argsort(entropy, descending=True, stable=True)
        selected[sample, predictors[order[keep:]] + 1] = -100
    return selected
