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


def relation_matrix(
    full_window_sums: torch.Tensor,
    loo_window_sums: torch.Tensor,
    lengths: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Return signed, length-normalized full-minus-LOO relations."""
    if not torch.is_tensor(full_window_sums) or full_window_sums.ndim != 2:
        raise ValueError("full_window_sums must have shape [B, K]")
    if not torch.is_tensor(loo_window_sums) or loo_window_sums.ndim != 3:
        raise ValueError("loo_window_sums must have shape [B, J, K]")
    if (
        loo_window_sums.shape[0] != full_window_sums.shape[0]
        or loo_window_sums.shape[2] != full_window_sums.shape[1]
    ):
        raise ValueError("full and LOO window sums must match in B and K dimensions")
    if not torch.is_tensor(lengths) or lengths.shape != full_window_sums.shape:
        raise ValueError("lengths must have shape [B, K]")
    if not torch.is_tensor(pair_mask) or pair_mask.shape != loo_window_sums.shape:
        raise ValueError("pair_mask must have shape [B, J, K]")
    tensors = (loo_window_sums, lengths, pair_mask)
    if any(value.device != full_window_sums.device for value in tensors):
        raise ValueError("relation tensors must be on the same device")
    if (
        not full_window_sums.is_floating_point()
        or not loo_window_sums.is_floating_point()
    ):
        raise ValueError("window sums must be floating point")

    values = (full_window_sums.unsqueeze(1) - loo_window_sums) / lengths.to(
        dtype=full_window_sums.dtype
    ).unsqueeze(1).clamp_min(1)
    return values.masked_fill(~pair_mask.to(dtype=torch.bool), 0)


def masked_relation_jsd(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average JSD over windows with at least two causal evidence candidates."""
    _validate_relation_loss_inputs(student, teacher, mask)
    temperature_value = _validate_positive_finite(temperature, "temperature")
    teacher = teacher.detach()
    window_losses: list[torch.Tensor] = []
    for batch_index in range(student.shape[0]):
        for window_index in range(student.shape[2]):
            valid = mask[batch_index, :, window_index].to(dtype=torch.bool)
            if valid.sum().item() < 2:
                continue
            student_log_probability = F.log_softmax(
                student[batch_index, valid, window_index] / temperature_value,
                dim=0,
            )
            teacher_log_probability = F.log_softmax(
                teacher[batch_index, valid, window_index] / temperature_value,
                dim=0,
            )
            midpoint_log_probability = torch.logaddexp(
                student_log_probability,
                teacher_log_probability,
            ) - math.log(2.0)
            student_probability = student_log_probability.exp()
            teacher_probability = teacher_log_probability.exp()
            student_kl = (
                student_probability
                * (student_log_probability - midpoint_log_probability)
            ).sum()
            teacher_kl = (
                teacher_probability
                * (teacher_log_probability - midpoint_log_probability)
            ).sum()
            window_losses.append(0.5 * (student_kl + teacher_kl))

    count = student.new_tensor(len(window_losses), dtype=torch.long)
    if not window_losses:
        return student.sum() * 0.0, count
    return torch.stack(window_losses).mean(), count


def masked_relation_huber(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average signed-relation Huber loss over every valid causal pair."""
    _validate_relation_loss_inputs(student, teacher, mask)
    delta_value = _validate_positive_finite(delta, "delta")
    values = F.huber_loss(
        student,
        teacher.detach(),
        delta=delta_value,
        reduction="none",
    )
    valid = mask.to(dtype=torch.bool)
    count = valid.sum()
    if not count.bool().item():
        return student.sum() * 0.0, count
    return values.masked_select(valid).mean(), count


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


def _validate_relation_loss_inputs(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if (
        not torch.is_tensor(student)
        or student.ndim != 3
        or not student.is_floating_point()
    ):
        raise ValueError("student must be a floating-point tensor with shape [B, J, K]")
    if not torch.is_tensor(teacher) or teacher.shape != student.shape:
        raise ValueError("teacher must have the same shape as student")
    if not teacher.is_floating_point():
        raise ValueError("teacher must be floating point")
    if not torch.is_tensor(mask) or mask.shape != student.shape:
        raise ValueError("mask must have the same shape as student")
    if teacher.device != student.device or mask.device != student.device:
        raise ValueError("relation loss tensors must be on the same device")


def _validate_positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)
