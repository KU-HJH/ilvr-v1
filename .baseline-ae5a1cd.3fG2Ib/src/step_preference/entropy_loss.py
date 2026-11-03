"""Relative entropy-margin objective for stepwise modality ablations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EntropyMarginResult:
    """Token-weighted entropy-margin loss plus modality diagnostics."""

    loss: torch.Tensor
    visual_loss: torch.Tensor
    textual_loss: torch.Tensor
    clean_entropy: torch.Tensor
    visual_ablated_entropy: torch.Tensor
    textual_ablated_entropy: torch.Tensor
    visual_token_count: int
    textual_token_count: int


def _require_nonnegative_finite_margin(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("entropy_margin must be a nonnegative finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("entropy_margin must be a nonnegative finite number")
    return value


def _validate_logits_and_mask(logits: torch.Tensor, target_mask: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError("logits must be a floating [batch, sequence, vocabulary] tensor")
    if logits.shape[1] < 2 or logits.shape[2] == 0:
        raise ValueError("logits must contain at least two positions and one vocabulary item")
    if (
        not isinstance(target_mask, torch.Tensor)
        or target_mask.dtype != torch.bool
        or target_mask.shape != logits.shape[:2]
    ):
        raise ValueError("target_mask must be a bool [batch, sequence] tensor matching logits")
    if target_mask[:, 0].any():
        raise ValueError("target_mask cannot select position 0 without a prior logit")
    if not bool(target_mask[:, 1:].any()):
        raise ValueError("target_mask must select at least one target token")


def token_entropies(logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Return float32 Shannon entropies at selected logical target positions."""
    _validate_logits_and_mask(logits, target_mask)
    selected_logits = logits[:, :-1][target_mask[:, 1:]].float()
    log_probs = F.log_softmax(selected_logits, dim=-1)
    entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
    if not torch.isfinite(entropies).all():
        raise ValueError("selected token entropies must be finite")
    return entropies


def _zero(clean_logits: torch.Tensor) -> torch.Tensor:
    return clean_logits.float().sum() * 0.0


def _modality_values(
    *,
    clean_logits: torch.Tensor,
    ablated_logits: Sequence[torch.Tensor],
    target_masks: Sequence[torch.Tensor],
    entropy_margin: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
    if len(ablated_logits) != len(target_masks):
        raise ValueError("ablated logits and target masks must have the same number of steps")
    losses: list[torch.Tensor] = []
    clean_values: list[torch.Tensor] = []
    ablated_values: list[torch.Tensor] = []
    for ablated, target_mask in zip(ablated_logits, target_masks):
        if not isinstance(ablated, torch.Tensor) or ablated.shape != clean_logits.shape:
            raise ValueError("each ablated logits tensor must match clean_logits shape")
        clean_entropy = token_entropies(clean_logits, target_mask)
        ablated_entropy = token_entropies(ablated, target_mask)
        losses.append(F.relu(clean_entropy + entropy_margin - ablated_entropy))
        clean_values.append(clean_entropy)
        ablated_values.append(ablated_entropy)
    if not losses:
        return None, None, None, 0
    return (
        torch.cat(losses),
        torch.cat(clean_values),
        torch.cat(ablated_values),
        sum(value.numel() for value in losses),
    )


def entropy_margin_loss(
    *,
    clean_logits: torch.Tensor,
    visual_ablated_logits: Sequence[torch.Tensor],
    visual_target_masks: Sequence[torch.Tensor],
    textual_ablated_logits: Sequence[torch.Tensor],
    textual_target_masks: Sequence[torch.Tensor],
    entropy_margin: float,
) -> EntropyMarginResult:
    """Aggregate per-token ``relu(H_clean + margin - H_ablated)`` losses.

    One ablated logits tensor is supplied for every structurally valid reasoning
    step, retaining the deliberately independent forward branches in the trainer.
    """
    entropy_margin = _require_nonnegative_finite_margin(entropy_margin)
    if (
        not isinstance(clean_logits, torch.Tensor)
        or clean_logits.ndim != 3
        or not clean_logits.is_floating_point()
        or clean_logits.shape[1] < 2
        or clean_logits.shape[2] == 0
    ):
        raise ValueError("clean_logits must be a floating [batch, sequence, vocabulary] tensor")
    visual_values = _modality_values(
        clean_logits=clean_logits,
        ablated_logits=visual_ablated_logits,
        target_masks=visual_target_masks,
        entropy_margin=entropy_margin,
    )
    textual_values = _modality_values(
        clean_logits=clean_logits,
        ablated_logits=textual_ablated_logits,
        target_masks=textual_target_masks,
        entropy_margin=entropy_margin,
    )
    visual_losses, visual_clean, visual_ablated, visual_count = visual_values
    textual_losses, textual_clean, textual_ablated, textual_count = textual_values
    zero = _zero(clean_logits)

    visual_loss = visual_losses.mean() if visual_losses is not None else zero
    textual_loss = textual_losses.mean() if textual_losses is not None else zero
    visual_ablated_entropy = visual_ablated.mean() if visual_ablated is not None else zero
    textual_ablated_entropy = textual_ablated.mean() if textual_ablated is not None else zero
    all_clean = [value for value in (visual_clean, textual_clean) if value is not None]
    clean_entropy = torch.cat(all_clean).mean() if all_clean else zero
    all_losses = [value for value in (visual_losses, textual_losses) if value is not None]
    loss = torch.cat(all_losses).mean() if all_losses else zero

    return EntropyMarginResult(
        loss=loss,
        visual_loss=visual_loss,
        textual_loss=textual_loss,
        clean_entropy=clean_entropy,
        visual_ablated_entropy=visual_ablated_entropy,
        textual_ablated_entropy=textual_ablated_entropy,
        visual_token_count=visual_count,
        textual_token_count=textual_count,
    )
