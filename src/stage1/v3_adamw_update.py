"""Same-state AdamW candidates and current-step FP32 displacement protection.

These functions do not mutate optimizer state. Collective sequencing and native
step ownership belong to v3_zero1_update, not this tensor calculation module.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.optim.adamw import adamw
from torch.optim.optimizer import _get_scalar_dtype


def _fp32(value: Tensor) -> None:
    if value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError('Gain update requires finite FP32 tensors')


def clip_base_gradient(gradient: Tensor, *, global_norm: Tensor, max_norm: float) -> Tensor:
    _fp32(gradient)
    if global_norm.numel() != 1 or not torch.isfinite(global_norm).all() or global_norm < 0:
        raise ValueError('invalid global gradient norm')
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError('invalid clipping threshold')
    result = gradient.clone()
    if max_norm > 0:
        divisor = ((global_norm + 1e-6) / max_norm).clamp(min=1.)
        result.mul_(1. / divisor)
    return result


@torch.no_grad()
def adamw_base_candidate(master: Tensor, gradient: Tensor, state: Mapping[str, Tensor],
                         group: Mapping[str, object]) -> Tensor:
    _fp32(master)
    _fp32(gradient)
    if master.shape != gradient.shape or master.device != gradient.device:
        raise ValueError('AdamW candidate gradient shape/device mismatch')
    candidate = master.detach().clone()
    if state:
        copied = {key: value.detach().clone() for key, value in state.items()}
    else:
        fused = group['fused']
        step = torch.zeros((), dtype=_get_scalar_dtype(is_fused=fused),
                           device=master.device if fused or group['capturable'] else 'cpu')
        copied = dict(step=step, exp_avg=torch.zeros_like(master), exp_avg_sq=torch.zeros_like(master))
    if set(copied) != {'step', 'exp_avg', 'exp_avg_sq'}:
        raise ValueError('unsupported AdamW candidate state')
    if any(not torch.isfinite(t).all() for t in copied.values()) or (copied['exp_avg_sq'] < 0).any():
        raise ValueError('invalid AdamW moment state')
    beta1, beta2 = group['betas']
    adamw([candidate], [gradient], [copied['exp_avg']], [copied['exp_avg_sq']], [], [copied['step']],
          foreach=group['foreach'], capturable=group['capturable'], differentiable=group['differentiable'],
          fused=group['fused'], amsgrad=group['amsgrad'], beta1=beta1, beta2=beta2,
          lr=group['lr'], weight_decay=group['weight_decay'], eps=group['eps'], maximize=group['maximize'])
    _fp32(candidate)
    return candidate


@dataclass(frozen=True)
class ProjectionResult:
    parameters: tuple[Tensor, ...]
    dot_before: float
    dot_after: float
    gain_norm_before: float
    gain_norm_after: float
    projected: bool
    tiny_h: bool
    rounding_fallback: bool


@torch.no_grad()
def project_adamw_increment(base: tuple[Tensor, ...], joint: tuple[Tensor, ...],
                            protection: tuple[Tensor, ...], *, valid_masks: tuple[Tensor, ...],
                            lrs: tuple[float, ...], sum_fp64: Callable[[Tensor], Tensor],
                            min_h_norm_sq: float = 1e-30) -> ProjectionResult:
    if not base or not len(base) == len(joint) == len(protection) == len(valid_masks) == len(lrs):
        raise ValueError('projection group count mismatch')
    if not math.isfinite(min_h_norm_sq) or min_h_norm_sq <= 0:
        raise ValueError('invalid protection norm threshold')
    device = base[0].device
    for b, j, h, mask, lr in zip(base, joint, protection, valid_masks, lrs):
        for t in (b, j, h):
            _fp32(t)
            if t.shape != b.shape or t.device != device:
                raise ValueError('projection shape/device mismatch')
        if mask.dtype != torch.bool or mask.shape != b.shape or mask.device != device:
            raise ValueError('invalid projection mask')
        if type(lr) not in (int, float) or not math.isfinite(lr) or lr < 0:
            raise ValueError('invalid projection LR')
        if lr == 0 and not torch.equal(b[mask], j[mask]):
            raise ValueError('nonzero increment in a zero-LR group')

    def statistics(parameters):
        local = torch.zeros(3, dtype=torch.float64, device=device)
        for b, p, h, mask, lr in zip(base, parameters, protection, valid_masks, lrs):
            delta = p[mask].double() - b[mask].double()
            active_h = h[mask].double() if lr > 0 else torch.zeros_like(delta)
            local[0] += active_h.square().sum()
            local[1] += (active_h * delta).sum()
            local[2] += delta.square().sum()
        global_values = sum_fp64(local)
        if not torch.isfinite(global_values).all() or (global_values[[0, 2]] < 0).any():
            raise ValueError('invalid global projection statistics')
        return global_values

    n, dot, norm_sq = statistics(joint).tolist()
    tiny = 0 < n < min_h_norm_sq
    projected = n >= min_h_norm_sq and dot < 0
    fallback = False
    parameters = joint
    if tiny or projected:
        candidates = []
        for b, j, h, mask, lr in zip(base, joint, protection, valid_masks, lrs):
            candidate = j.clone()
            if tiny:
                candidate[mask] = b[mask]
            elif lr > 0:
                candidate[mask] = (j[mask].double() - (dot / n) * h[mask].double()).float()
            candidates.append(candidate)
        parameters = tuple(candidates)
        _, after, after_norm = statistics(parameters).tolist()
        if projected and after < 0:
            fallback = True
            parameters = tuple(torch.where(mask, b, j) for b, j, mask in zip(base, joint, valid_masks))
            after, after_norm = 0., 0.
    else:
        after, after_norm = dot, norm_sq
    return ProjectionResult(parameters, dot, after, math.sqrt(norm_sq), math.sqrt(after_norm),
                            projected, tiny, fallback)
