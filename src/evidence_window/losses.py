"""Stable preference losses over global, local, and evidence-window margins."""

from __future__ import annotations

import math
import operator

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


def stage2_ew_preference_loss(
    contributions: torch.Tensor,
    pair_mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Reference-free Stage-2 loss with one independent sigmoid per edge.

    This deliberately does not aggregate contributions into a sequence margin;
    global aggregation is reserved for saturation diagnostics.
    """
    return evidence_window_preference_loss(contributions, pair_mask, beta)


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
    accumulation_dtype = _relation_accumulation_dtype(student, teacher)
    teacher = teacher.detach()
    window_losses: list[torch.Tensor] = []
    for batch_index in range(student.shape[0]):
        for window_index in range(student.shape[2]):
            valid = mask[batch_index, :, window_index].to(dtype=torch.bool)
            if valid.sum().item() < 2:
                continue
            student_values = student[batch_index, valid, window_index].to(
                dtype=accumulation_dtype
            )
            teacher_values = teacher[batch_index, valid, window_index].to(
                dtype=accumulation_dtype
            )
            student_log_probability = F.log_softmax(
                student_values / temperature_value,
                dim=0,
            )
            teacher_log_probability = F.log_softmax(
                teacher_values / temperature_value,
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
            jsd = 0.5 * (student_kl + teacher_kl)
            if torch.equal(student_values, teacher_values):
                jsd = jsd * 0.0
            window_losses.append(_guard_jsd_roundoff(jsd))

    count = student.new_tensor(len(window_losses), dtype=torch.long)
    if not window_losses:
        return student.to(dtype=accumulation_dtype).sum() * 0.0, count
    return _guard_jsd_roundoff(torch.stack(window_losses).mean()), count


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


def select_residual_tokens(mask: torch.Tensor, cap: int) -> torch.Tensor:
    """Select deterministic target positions, including temporal endpoints."""
    if not torch.is_tensor(mask) or mask.ndim != 1:
        raise ValueError("mask must have shape [T]")
    if isinstance(cap, bool):
        raise TypeError("cap must be an integer")
    try:
        cap_value = operator.index(cap)
    except TypeError as exc:
        raise TypeError("cap must be an integer") from exc
    if cap_value < 0:
        raise ValueError("cap must be non-negative")

    eligible = mask.to(dtype=torch.bool).nonzero(as_tuple=False).flatten()
    target = eligible.numel() if cap_value == 0 else min(cap_value, eligible.numel())
    if target == eligible.numel():
        return eligible
    if target == 0:
        return eligible[:0]
    if target == 1:
        return eligible[:1]

    positions = {
        round(index * (eligible.numel() - 1) / (target - 1))
        for index in range(target)
    }
    for position in range(eligible.numel()):
        if len(positions) >= target:
            break
        positions.add(position)
    selected_positions = torch.tensor(
        sorted(positions),
        dtype=torch.long,
        device=eligible.device,
    )
    return eligible.index_select(0, selected_positions)


def chunked_logit_residual_loss(
    student_dh: torch.Tensor,
    teacher_dh: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    metric: str,
    chunk_size: int,
    delta: float,
) -> torch.Tensor:
    """Compare exact LM-head residuals without allocating ``[tokens, vocab]``."""
    chunk_size_value, delta_value = _validate_logit_residual_inputs(
        student_dh,
        teacher_dh,
        student_weight,
        teacher_weight,
        metric,
        chunk_size,
        delta,
    )
    accumulation_dtype = _residual_accumulation_dtype(
        student_dh,
        teacher_dh,
        student_weight,
        teacher_weight,
    )
    student_values = student_dh.to(dtype=accumulation_dtype)
    teacher_values = teacher_dh.detach().to(dtype=accumulation_dtype)
    zero = (
        student_values.reshape(-1)[:1].sum() * 0.0
        + student_weight.reshape(-1)[0].to(dtype=accumulation_dtype) * 0.0
    )
    if student_dh.shape[0] == 0:
        return zero

    return _BoundedLogitResidual.apply(
        student_values,
        teacher_values,
        student_weight,
        teacher_weight.detach(),
        0 if metric == "huber" else 1,
        chunk_size_value,
        delta_value,
    )


class _BoundedLogitResidual(torch.autograd.Function):
    """Recompute one vocabulary projection chunk at a time during backward."""

    @staticmethod
    def forward(
        ctx,
        student_values: torch.Tensor,
        teacher_values: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_weight: torch.Tensor,
        metric_code: int,
        chunk_size: int,
        delta: float,
    ) -> torch.Tensor:
        ctx.metric_code = metric_code
        ctx.chunk_size = chunk_size
        ctx.delta = delta

        with _autocast_disabled_for(student_values):
            if metric_code == 0:
                total = student_values.new_zeros(())
                for start in range(0, student_weight.shape[0], chunk_size):
                    stop = min(start + chunk_size, student_weight.shape[0])
                    student_chunk = student_weight[start:stop].to(
                        dtype=student_values.dtype
                    )
                    teacher_chunk = teacher_weight[start:stop].to(
                        dtype=teacher_values.dtype
                    )
                    student_logits = torch.matmul(
                        student_values,
                        student_chunk.transpose(0, 1),
                    )
                    teacher_logits = torch.matmul(
                        teacher_values,
                        teacher_chunk.transpose(0, 1),
                    )
                    total = total + F.huber_loss(
                        student_logits,
                        teacher_logits,
                        delta=delta,
                        reduction="sum",
                    )
                ctx.save_for_backward(
                    student_values,
                    teacher_values,
                    student_weight,
                    teacher_weight,
                )
                return total / (student_values.shape[0] * student_weight.shape[0])

            dots = student_values.new_zeros(student_values.shape[0])
            student_norm_sq = torch.zeros_like(dots)
            teacher_norm_sq = torch.zeros_like(dots)
            for start in range(0, student_weight.shape[0], chunk_size):
                stop = min(start + chunk_size, student_weight.shape[0])
                student_chunk = student_weight[start:stop].to(
                    dtype=student_values.dtype
                )
                teacher_chunk = teacher_weight[start:stop].to(
                    dtype=teacher_values.dtype
                )
                student_logits = torch.matmul(
                    student_values,
                    student_chunk.transpose(0, 1),
                )
                teacher_logits = torch.matmul(
                    teacher_values,
                    teacher_chunk.transpose(0, 1),
                )
                dots = dots + (student_logits * teacher_logits).sum(dim=-1)
                student_norm_sq = student_norm_sq + student_logits.square().sum(dim=-1)
                teacher_norm_sq = teacher_norm_sq + teacher_logits.square().sum(dim=-1)
            ctx.save_for_backward(
                student_values,
                teacher_values,
                student_weight,
                teacher_weight,
                dots,
                student_norm_sq,
                teacher_norm_sq,
            )
            return _cosine_residual_from_statistics(
                dots,
                student_norm_sq,
                teacher_norm_sq,
            )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        student_values, teacher_values, student_weight, teacher_weight = saved[:4]
        need_student_values = ctx.needs_input_grad[0]
        need_student_weight = ctx.needs_input_grad[2]
        if not need_student_values and not need_student_weight:
            return (None,) * 7

        student_values_grad = (
            torch.zeros_like(student_values) if need_student_values else None
        )
        student_weight_grad = (
            torch.zeros_like(student_weight) if need_student_weight else None
        )
        if ctx.metric_code == 1:
            dots, student_norm_sq, teacher_norm_sq = saved[4:]

        for start in range(0, student_weight.shape[0], ctx.chunk_size):
            stop = min(start + ctx.chunk_size, student_weight.shape[0])
            with torch.enable_grad(), _autocast_disabled_for(student_values):
                local_student_values = student_values.detach().requires_grad_(
                    need_student_values
                )
                local_student_weight = (
                    student_weight[start:stop]
                    .detach()
                    .to(dtype=student_values.dtype)
                    .requires_grad_(need_student_weight)
                )
                teacher_chunk = teacher_weight[start:stop].to(
                    dtype=teacher_values.dtype
                )
                student_logits = torch.matmul(
                    local_student_values,
                    local_student_weight.transpose(0, 1),
                )
                teacher_logits = torch.matmul(
                    teacher_values,
                    teacher_chunk.transpose(0, 1),
                )
                if ctx.metric_code == 0:
                    chunk_loss = F.huber_loss(
                        student_logits,
                        teacher_logits,
                        delta=ctx.delta,
                        reduction="sum",
                    ) / (student_values.shape[0] * student_weight.shape[0])
                else:
                    local_dots = (student_logits * teacher_logits).sum(dim=-1)
                    local_student_norm_sq = student_logits.square().sum(dim=-1)
                    differentiable_dots = dots + local_dots - local_dots.detach()
                    differentiable_student_norm_sq = (
                        student_norm_sq
                        + local_student_norm_sq
                        - local_student_norm_sq.detach()
                    )
                    chunk_loss = _cosine_residual_from_statistics(
                        differentiable_dots,
                        differentiable_student_norm_sq,
                        teacher_norm_sq,
                    )

                differentiable_inputs = []
                if need_student_values:
                    differentiable_inputs.append(local_student_values)
                if need_student_weight:
                    differentiable_inputs.append(local_student_weight)
                gradients = torch.autograd.grad(
                    chunk_loss,
                    differentiable_inputs,
                    grad_outputs=grad_output,
                )

            gradient_index = 0
            if student_values_grad is not None:
                student_values_grad.add_(gradients[gradient_index])
                gradient_index += 1
            if student_weight_grad is not None:
                student_weight_grad[start:stop].copy_(gradients[gradient_index])

        return (
            student_values_grad,
            None,
            student_weight_grad,
            None,
            None,
            None,
            None,
        )


def _autocast_disabled_for(value: torch.Tensor):
    """Disable autocast for the actual CPU/CUDA projection device."""
    return torch.autocast(device_type=value.device.type, enabled=False)


def _cosine_residual_from_statistics(
    dots: torch.Tensor,
    student_norm_sq: torch.Tensor,
    teacher_norm_sq: torch.Tensor,
) -> torch.Tensor:
    epsilon = 1e-8
    denominator = (
        student_norm_sq.sqrt().clamp_min(epsilon)
        * teacher_norm_sq.sqrt().clamp_min(epsilon)
    )
    similarity = (dots / denominator).clamp(min=-1.0, max=1.0)
    return (1.0 - similarity).mean()


def _validate_logit_residual_inputs(
    student_dh: torch.Tensor,
    teacher_dh: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    metric: str,
    chunk_size: int,
    delta: float,
) -> tuple[int, float]:
    tensors = {
        "student_dh": student_dh,
        "teacher_dh": teacher_dh,
        "student_weight": student_weight,
        "teacher_weight": teacher_weight,
    }
    for name, value in tensors.items():
        if not torch.is_tensor(value) or value.ndim != 2 or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor with two dimensions")
    if teacher_dh.shape[0] != student_dh.shape[0]:
        raise ValueError("student and teacher residuals must have the same token count")
    if student_weight.shape[1] != student_dh.shape[1]:
        raise ValueError("student LM-head hidden size must match student residuals")
    if teacher_weight.shape[1] != teacher_dh.shape[1]:
        raise ValueError("teacher LM-head hidden size must match teacher residuals")
    if student_weight.shape[0] != teacher_weight.shape[0]:
        raise ValueError("student and teacher vocabulary sizes must match")
    if student_weight.shape[0] == 0:
        raise ValueError("LM-head vocabulary must be nonempty")
    devices = {value.device for value in tensors.values()}
    if len(devices) != 1:
        raise ValueError("residual and LM-head tensors must be on the same device")
    if metric not in {"huber", "cosine"}:
        raise ValueError("metric must be 'huber' or 'cosine'")
    if isinstance(chunk_size, bool):
        raise ValueError("chunk_size must be a positive integer")
    try:
        chunk_size_value = operator.index(chunk_size)
    except TypeError as exc:
        raise ValueError("chunk_size must be a positive integer") from exc
    if chunk_size_value <= 0:
        raise ValueError("chunk_size must be a positive integer")
    delta_value = _validate_positive_finite(delta, "delta")
    return chunk_size_value, delta_value


def _residual_accumulation_dtype(*values: torch.Tensor) -> torch.dtype:
    dtype = values[0].dtype
    for value in values[1:]:
        dtype = torch.promote_types(dtype, value.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


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


def _relation_accumulation_dtype(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.dtype:
    dtype = torch.promote_types(student.dtype, teacher.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _guard_jsd_roundoff(value: torch.Tensor) -> torch.Tensor:
    tolerance = 8.0 * torch.finfo(value.dtype).eps
    if torch.isfinite(value).item():
        detached = value.detach().item()
        if detached < -tolerance or detached > math.log(2.0) + tolerance:
            raise FloatingPointError(
                f"JSD outside theoretical bounds before roundoff guard: {detached}"
            )
    upper_bound = value.new_tensor(math.log(2.0))
    if upper_bound.item() > math.log(2.0):
        upper_bound = torch.nextafter(
            upper_bound,
            value.new_tensor(-math.inf),
        )
    return value.clamp(min=0.0, max=upper_bound)
