"""Exact full-reference-vocabulary token BCE for the orig objective."""

from dataclasses import dataclass, fields
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as torch_checkpoint


@dataclass(frozen=True)
class OrigTokenComponents:
    chosen: Tensor
    reference: Tensor
    policy_gt_logp: Tensor
    reference_gt_logp: Tensor
    gt_logratio: Tensor
    reference_prob_sum_error: Tensor | None


def _validate_logits(logits: Tensor, name: str) -> None:
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError(f"{name} must have [P,V] shape with nonempty vocabulary")
    if not logits.is_floating_point():
        raise ValueError(f"{name} must be floating-point")
    if not bool(torch.isfinite(logits.detach().float()).all()):
        raise ValueError(f"{name} must be finite after float32 conversion")


def _validate_pair(clean: Tensor, other: Tensor, targets: Tensor) -> None:
    # Check devices before any value inspection (including meta-device inputs).
    if clean.device != other.device or clean.device != targets.device:
        raise ValueError("logit and target devices must match")
    if clean.ndim != 2 or other.shape != clean.shape:
        raise ValueError("logits must have matching [P,V] shapes")
    if targets.shape != (clean.shape[0],):
        raise ValueError("target IDs must have shape [P]")
    if targets.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("target IDs must be integral")
    _validate_logits(clean, "clean logits")
    _validate_logits(other, "reference/OFF logits")
    # Python vocabulary-size scalars can wrap when compared in a narrow dtype.
    wide_targets = targets.long()
    if bool(((wide_targets < 0) | (wide_targets >= clean.shape[1])).any()):
        raise ValueError("target IDs must be in vocabulary range")


def _components(clean: Tensor, reference: Tensor, targets: Tensor, *,
                validate: bool = True, collect_diagnostics: bool = True) -> tuple[Tensor, ...]:
    logp = clean.float().log_softmax(-1)
    with torch.no_grad():
        logmu = reference.detach().float().log_softmax(-1)
        if validate and (not bool(torch.isfinite(logp).all()) or not bool(torch.isfinite(logmu).all())):
            raise ValueError("FP32 log probabilities are not finite; reduce logit magnitude")
        mu = logmu.exp()
        reference_gt = logmu.gather(1, targets.long()[:, None]).squeeze(1)
        sum_error = (mu.sum(-1) - 1).abs() if collect_diagnostics else None
    logratio = logp - logmu
    policy_gt = logp.gather(1, targets.long()[:, None]).squeeze(1)
    gt_ratio = policy_gt - reference_gt
    return (
        F.softplus(-gt_ratio),
        (mu * F.softplus(logratio)).sum(-1),
        policy_gt, reference_gt, gt_ratio, sum_error,
    )


def dil_bce_orig_components(clean_logits: Tensor, reference_logits: Tensor, target_ids: Tensor, *,
                            validate: bool = True, collect_diagnostics: bool = True) -> OrigTokenComponents:
    """Return FP32 per-position terms, keeping only the clean policy live."""
    if validate:
        _validate_pair(clean_logits, reference_logits, target_ids)
    return OrigTokenComponents(*_components(clean_logits, reference_logits, target_ids,
        validate=validate, collect_diagnostics=collect_diagnostics))


def orig_components_chunked(
    clean_logits: Tensor,
    reference_logits: Tensor,
    target_ids: Tensor,
    *,
    chunk_positions: int = 128,
    checkpoint: bool = True,
    validate: bool = True,
    collect_diagnostics: bool = True,
) -> OrigTokenComponents:
    """Concatenate token terms; checkpoint only the FP32 loss intermediates.

    Input selected logits may still occupy O(PV) storage. No averaging occurs
    here, so a short final chunk retains its correct token weight.
    """
    if type(chunk_positions) is not int or chunk_positions <= 0:
        raise ValueError("chunk_positions must be a positive integer")
    if validate:
        _validate_pair(clean_logits, reference_logits, target_ids)
    components = partial(_components, validate=validate, collect_diagnostics=collect_diagnostics)
    if clean_logits.shape[0] == 0:
        return OrigTokenComponents(*components(clean_logits, reference_logits, target_ids))
    parts = [[] for _ in fields(OrigTokenComponents)]
    for start in range(0, clean_logits.shape[0], chunk_positions):
        end = start + chunk_positions
        args = (clean_logits[start:end], reference_logits[start:end].detach(), target_ids[start:end])
        if checkpoint and torch.is_grad_enabled() and clean_logits.requires_grad:
            values = torch_checkpoint(components, *args, use_reentrant=False)
        else:
            values = components(*args)
        for part, value in zip(parts, values):
            if value is not None:
                part.append(value)
    return OrigTokenComponents(*(torch.cat(part) if part else None for part in parts))
