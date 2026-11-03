"""Numerically exact primitive losses for the Stage 2 objective."""

from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class DynTerms:
    losses: Tensor
    r_pre: Tensor
    r_post: Tensor
    delta: Tensor


@dataclass(frozen=True)
class VisTerms:
    losses: Tensor
    gain: Tensor


def _require_score_vector(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty score vector")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")


def _require_finite_result(value: Tensor, operation: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{operation} produced nonfinite values")


def _fp32_scalar(value: Real, name: str, *, positive: bool = False) -> Tensor:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-boolean scalar")
    converted = torch.tensor(value, dtype=torch.float32)
    if not torch.isfinite(converted) or (positive and converted.item() <= 0):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{name} must be {qualifier}finite and representable in FP32")
    return converted


def text_entropies(logits: Tensor, text_ids: Tensor) -> Tensor:
    if not isinstance(logits, Tensor) or logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("logits must have shape [tokens, vocab] with non-empty tokens")
    if not logits.dtype.is_floating_point:
        raise TypeError("logits must be a floating-point tensor")
    if not isinstance(text_ids, Tensor) or text_ids.ndim != 1 or text_ids.numel() == 0:
        raise ValueError("text_ids must be a non-empty vector")
    if text_ids.dtype == torch.bool or text_ids.dtype.is_floating_point:
        raise TypeError("text_ids must contain integers")
    ids = text_ids.to(device=logits.device, dtype=torch.long)
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("text_ids must be unique")
    if torch.any(ids < 0) or torch.any(ids >= logits.shape[-1]):
        raise ValueError("text_ids are out of range")
    selected = logits.index_select(-1, ids).float()
    if not torch.isfinite(selected).all():
        raise ValueError("text_entropies selected inputs contain nonfinite values")
    log_p = selected.log_softmax(-1)
    result = -(log_p.exp() * log_p).sum(-1)
    _require_finite_result(result, "text_entropies")
    return result


def target_logps(logits: Tensor, target_ids: Tensor) -> Tensor:
    if not isinstance(logits, Tensor) or logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("logits must have shape [tokens, vocab] with non-empty tokens")
    if not logits.dtype.is_floating_point:
        raise TypeError("logits must be a floating-point tensor")
    if not isinstance(target_ids, Tensor) or target_ids.ndim != 1:
        raise ValueError("target_ids must be a vector")
    if target_ids.numel() != logits.shape[0]:
        raise ValueError("target_ids and logits must have matching token counts")
    if target_ids.dtype == torch.bool or target_ids.dtype.is_floating_point:
        raise TypeError("target_ids must contain integers")
    ids = target_ids.to(device=logits.device, dtype=torch.long)
    if torch.any(ids == -100):
        raise ValueError("target_ids must not contain -100")
    if torch.any(ids < 0) or torch.any(ids >= logits.shape[-1]):
        raise ValueError("target_ids are out of range")
    if not torch.isfinite(logits).all():
        raise ValueError("target_logps inputs contain nonfinite values")
    log_p = logits.float().log_softmax(-1)
    result = log_p.gather(-1, ids[:, None]).squeeze(-1)
    _require_finite_result(result, "target_logps")
    return result


def entropy_transition(
    pre: Tensor,
    post: Tensor,
    ref_pre: Tensor,
    ref_post: Tensor,
    *,
    eps: float,
    gamma_h: float,
) -> DynTerms:
    for value, name in ((pre, "pre"), (post, "post"), (ref_pre, "ref_pre"), (ref_post, "ref_post")):
        _require_score_vector(value, name)
    if not (pre.shape == post.shape == ref_pre.shape == ref_post.shape):
        raise ValueError("entropy transition score vectors must have matching shapes")
    eps32 = _fp32_scalar(eps, "eps", positive=True)
    gamma32 = _fp32_scalar(gamma_h, "gamma_h")
    r_pre = (pre.float() + eps32) / (ref_pre.detach().float() + eps32)
    r_post = (post.float() + eps32) / (ref_post.detach().float() + eps32)
    delta = r_pre - r_post
    losses = -torch.sigmoid(delta - gamma32)
    for value, name in ((r_pre, "entropy_transition r_pre"), (r_post, "entropy_transition r_post"), (delta, "entropy_transition delta"), (losses, "entropy_transition losses")):
        _require_finite_result(value, name)
    return DynTerms(losses, r_pre, r_post, delta)


def vision_gain(clean: Tensor, masked: Tensor, *, margin: float) -> VisTerms:
    _require_score_vector(clean, "clean")
    _require_score_vector(masked, "masked")
    if clean.shape != masked.shape:
        raise ValueError("vision gain score vectors must have matching shapes")
    margin32 = _fp32_scalar(margin, "margin")
    gain = clean.float() - masked.detach().float()
    losses = F.softplus(margin32 - gain)
    _require_finite_result(gain, "vision_gain gain")
    _require_finite_result(losses, "vision_gain losses")
    return VisTerms(losses, gain)
