"""Length-normalized Step-DPO scoring and aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StepDPOResult:
    """Step-DPO loss plus modality diagnostics for one trainer batch."""

    loss: torch.Tensor
    visual_loss: torch.Tensor
    textual_loss: torch.Tensor
    visual_margin: torch.Tensor
    textual_margin: torch.Tensor
    visual_pair_count: int
    textual_pair_count: int


def _require_positive_finite(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _validate_target_inputs(
    logits: torch.Tensor, input_ids: torch.Tensor, target_mask: torch.Tensor
) -> None:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError("logits must be a floating [batch, sequence, vocabulary] tensor")
    if logits.shape[-1] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must contain at least two positions and one vocabulary item")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or input_ids.shape != logits.shape[:2]
        or input_ids.is_floating_point()
        or input_ids.dtype == torch.bool
    ):
        raise ValueError("input_ids must be an integer [batch, sequence] tensor matching logits")
    if (
        not isinstance(target_mask, torch.Tensor)
        or target_mask.dtype != torch.bool
        or target_mask.shape != input_ids.shape
    ):
        raise ValueError("target_mask must be a bool [batch, sequence] tensor matching input_ids")
    if target_mask[:, 0].any():
        raise ValueError("target_mask cannot select position 0 without a prior logit")
    if not bool(target_mask[:, 1:].any(dim=1).all()):
        raise ValueError("each completion must select at least one target token")
    selected_ids = input_ids[:, 1:][target_mask[:, 1:]]
    if selected_ids.numel() and (
        (selected_ids < 0).any() or (selected_ids >= logits.shape[-1]).any()
    ):
        raise ValueError("selected input_ids must be valid vocabulary indices")


def mean_completion_logps(
    logits: torch.Tensor, input_ids: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    """Return each completion's float32 mean next-token log probability.

    ``target_mask`` uses logical token coordinates: a selected token at index
    ``i`` is scored from the causal logits at index ``i - 1``.
    """
    _validate_target_inputs(logits, input_ids, target_mask)
    next_log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
    next_ids = input_ids[:, 1:]
    selected = target_mask[:, 1:]
    token_logps = next_log_probs.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(token_logps[selected]).all():
        raise ValueError("selected next-token log probabilities must be finite")
    weights = selected.to(token_logps.dtype)
    return (token_logps * weights).sum(dim=1) / weights.sum(dim=1)


def _validate_score_tensors(*scores: torch.Tensor) -> None:
    if not scores or not all(isinstance(score, torch.Tensor) for score in scores):
        raise ValueError("scores must be tensors")
    shape = scores[0].shape
    if not shape or any(score.shape != shape for score in scores):
        raise ValueError("all score tensors must have the same non-empty shape")
    if any(not score.is_floating_point() for score in scores):
        raise ValueError("scores must be floating-point tensors")
    if any(not torch.isfinite(score).all() for score in scores):
        raise ValueError("scores must be finite")


def dpo_pair_losses(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Return one detached-reference DPO hinge-free loss per valid pair."""
    beta = _require_positive_finite("beta", beta)
    _validate_score_tensors(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected
    )
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen.detach() - reference_rejected.detach()
    return -F.logsigmoid(beta * (policy_margin - reference_margin))


def _validate_aggregate_inputs(
    clean_ce: torch.Tensor,
    score_tensors: tuple[torch.Tensor, ...],
    valid_visual: torch.Tensor,
    valid_textual: torch.Tensor,
) -> None:
    if not isinstance(clean_ce, torch.Tensor) or clean_ce.numel() != 1:
        raise ValueError("clean_ce must be a scalar tensor")
    if not score_tensors or any(not isinstance(value, torch.Tensor) for value in score_tensors):
        raise ValueError("scores must be tensors")
    shape = score_tensors[0].shape
    if not shape or any(value.shape != shape for value in score_tensors):
        raise ValueError("all score tensors must have the same non-empty shape")
    for name, valid in (("valid_visual", valid_visual), ("valid_textual", valid_textual)):
        if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.shape != shape:
            raise ValueError(f"{name} must be a bool tensor matching score shapes")


def _zero_diagnostic(clean_ce: torch.Tensor) -> torch.Tensor:
    return clean_ce * 0.0


def aggregate_step_dpo(
    *,
    clean_ce: torch.Tensor,
    policy_chosen: torch.Tensor,
    policy_visual_rejected: torch.Tensor,
    policy_textual_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_visual_rejected: torch.Tensor,
    reference_textual_rejected: torch.Tensor,
    valid_visual: torch.Tensor,
    valid_textual: torch.Tensor,
    beta: float,
) -> StepDPOResult:
    """Aggregate valid visual/textual pairs with one global pair denominator."""
    beta = _require_positive_finite("beta", beta)
    score_tensors = (
        policy_chosen,
        policy_visual_rejected,
        policy_textual_rejected,
        reference_chosen,
        reference_visual_rejected,
        reference_textual_rejected,
    )
    _validate_aggregate_inputs(clean_ce, score_tensors, valid_visual, valid_textual)

    visual_count = int(valid_visual.sum().item())
    textual_count = int(valid_textual.sum().item())
    zero = _zero_diagnostic(clean_ce)

    def selected_pair_values(valid, policy_rejected, reference_rejected):
        return (
            policy_chosen[valid],
            policy_rejected[valid],
            reference_chosen[valid],
            reference_rejected[valid],
        )

    if visual_count:
        visual_values = selected_pair_values(
            valid_visual, policy_visual_rejected, reference_visual_rejected
        )
        visual_losses = dpo_pair_losses(*visual_values, beta=beta)
        visual_loss = visual_losses.mean()
        visual_margin = (
            (visual_values[0] - visual_values[1])
            - (visual_values[2].detach() - visual_values[3].detach())
        ).mean()
    else:
        visual_losses = None
        visual_loss = zero
        visual_margin = zero

    if textual_count:
        textual_values = selected_pair_values(
            valid_textual, policy_textual_rejected, reference_textual_rejected
        )
        textual_losses = dpo_pair_losses(*textual_values, beta=beta)
        textual_loss = textual_losses.mean()
        textual_margin = (
            (textual_values[0] - textual_values[1])
            - (textual_values[2].detach() - textual_values[3].detach())
        ).mean()
    else:
        textual_losses = None
        textual_loss = zero
        textual_margin = zero

    total_count = visual_count + textual_count
    if total_count == 0:
        loss = clean_ce * 0.0
    elif visual_losses is None:
        loss = textual_losses.sum() / total_count
    elif textual_losses is None:
        loss = visual_losses.sum() / total_count
    else:
        loss = (visual_losses.sum() + textual_losses.sum()) / total_count

    return StepDPOResult(
        loss=loss,
        visual_loss=visual_loss,
        textual_loss=textual_loss,
        visual_margin=visual_margin,
        textual_margin=textual_margin,
        visual_pair_count=visual_count,
        textual_pair_count=textual_count,
    )
