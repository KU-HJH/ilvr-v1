"""Numerically stable primitives for step-wise direct preference optimization."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class StepDpoResult:
    """The batch loss and its per-pair diagnostics."""

    loss: Tensor
    pair_losses: Tensor
    policy_margins: Tensor


@dataclass(frozen=True)
class ContextPreferenceResult:
    loss: Tensor
    pair_losses: Tensor
    policy_margins: Tensor


def context_preference_loss(
    clean_logps: Tensor, ablated_logps: Tensor, beta: float, margin: float = 0.0,
    *, stage2_debugging: bool = True,
) -> ContextPreferenceResult:
    """Compare the same target under clean and locally ablated contexts."""
    if clean_logps.shape != ablated_logps.shape:
        raise ValueError("context preference score shapes are inconsistent")
    if not clean_logps.is_floating_point() or not ablated_logps.is_floating_point():
        raise ValueError("context preference scores must be floating-point")
    for name, value in (("beta", beta), ("margin", margin)):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if beta <= 0 or margin < 0:
        raise ValueError("beta must be positive and margin nonnegative")
    clean, ablated = clean_logps.float(), ablated_logps.float()
    if stage2_debugging and (not torch.isfinite(clean).all() or not torch.isfinite(ablated).all()):
        raise ValueError("context preference scores must be finite")
    delta = clean - ablated
    pairs = -F.logsigmoid(beta * (delta - margin))
    loss = pairs.mean() if pairs.numel() else differentiable_zero(clean) + differentiable_zero(ablated)
    return ContextPreferenceResult(loss, pairs, delta)


def mean_token_logps(logits: Tensor, target_ids: Tensor, token_mask: Tensor,
                    *, stage2_debugging: bool = True) -> Tensor:
    """Return the mean log-probability of selected target tokens per block."""
    if logits.shape[:-1] != target_ids.shape or target_ids.shape != token_mask.shape:
        raise ValueError("logits, target_ids, and token_mask shapes are inconsistent")
    if not logits.is_floating_point():
        raise ValueError("logits must be finite floating-point values")
    if token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be a boolean tensor")

    logits_f = logits.float()
    if stage2_debugging and not torch.isfinite(logits_f).all():
        raise ValueError("logits must be finite after float32 conversion")
    log_probs = F.log_softmax(logits_f, dim=-1)
    selected = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    counts = token_mask.sum(dim=-1, dtype=torch.float32)
    if stage2_debugging and torch.any(counts == 0):
        raise ValueError("every block score requires at least one selected token")
    return (selected * token_mask).sum(dim=-1, dtype=torch.float32) / counts


def differentiable_zero(reference: Tensor) -> Tensor:
    """Return a finite scalar zero connected to ``reference`` for autograd."""
    return reference.float().sum() * 0.0


def step_dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    beta: float,
) -> StepDpoResult:
    """Compute the stable mean-token DPO objective for preference pairs."""
    tensors = (
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
    )
    if not all(t.shape == policy_chosen.shape for t in tensors):
        raise ValueError("DPO score tensor shapes are inconsistent")
    if not all(t.is_floating_point() for t in tensors):
        raise ValueError("DPO score tensors must be floating-point")
    if not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and strictly positive")

    policy_chosen_f = policy_chosen.float()
    policy_rejected_f = policy_rejected.float()
    reference_chosen_f = reference_chosen.detach().float()
    reference_rejected_f = reference_rejected.detach().float()
    converted_tensors = (
        policy_chosen_f,
        policy_rejected_f,
        reference_chosen_f,
        reference_rejected_f,
    )
    if not all(torch.isfinite(t).all() for t in converted_tensors):
        raise ValueError("DPO score tensors must be finite after float32 conversion")

    policy_margins = policy_chosen_f - policy_rejected_f
    reference_margins = reference_chosen_f - reference_rejected_f
    pair_losses = -F.logsigmoid(beta * (policy_margins - reference_margins))

    if pair_losses.numel() == 0:
        loss = differentiable_zero(policy_chosen_f) + differentiable_zero(policy_rejected_f)
    else:
        loss = pair_losses.mean(dtype=torch.float32)
    return StepDpoResult(loss=loss, pair_losses=pair_losses, policy_margins=policy_margins)
