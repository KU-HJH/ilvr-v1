"""OFF top-k negative and clean-to-OFF group mass primitives."""

from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class TopkStepResult:
    """FP32 token means for one valid preference step."""

    negative_loss: Tensor
    mass_loss: Tensor
    entropy: Tensor
    clean_mass: Tensor
    off_mass: Tensor
    mass_abs_gap: Tensor
    gt_coverage: Tensor


class TopkTokenResult(NamedTuple):
    """FP32 statistics for each target token in one nonempty step."""

    negative_loss: Tensor
    mass_loss: Tensor
    entropy: Tensor
    clean_mass: Tensor
    off_mass: Tensor
    mass_abs_gap: Tensor
    gt_coverage: Tensor


def _group_gap(logits: Tensor, support: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Keep group maxima and log-sum corrections separate in FP32."""
    selected = logits.masked_fill(~support, -torch.inf)
    complement = logits.masked_fill(support, -torch.inf)
    a_max = selected.amax(-1)
    b_max = complement.amax(-1)
    a_correction = (selected - a_max[:, None]).logsumexp(-1)
    b_correction = (complement - b_max[:, None]).logsumexp(-1)
    h = (a_max - b_max) + (a_correction - b_correction)
    return a_max, b_max, a_correction, b_correction, h


def _weighted_off_cross_entropy(
    a_max: Tensor,
    b_max: Tensor,
    a_correction: Tensor,
    b_correction: Tensor,
    h: Tensor,
    target_m: Tensor,
    target_c: Tensor,
) -> Tensor:
    """Binary cross entropy with target weights applied before huge gaps."""
    positive_gap = h >= 0
    weight = torch.where(positive_gap, target_c, target_m)
    hi_max = torch.where(positive_gap, a_max, b_max)
    lo_max = torch.where(positive_gap, b_max, a_max)
    hi_correction = torch.where(positive_gap, a_correction, b_correction)
    lo_correction = torch.where(positive_gap, b_correction, a_correction)
    finite_gap = torch.isfinite(h)
    finite_h = torch.where(finite_gap, h, 0.0)
    finite_linear = torch.where(positive_gap, target_c * finite_h, -target_m * finite_h)
    overflow_linear = (
        weight * hi_max - weight * lo_max
        + weight * (hi_correction - lo_correction)
    )
    linear = torch.where(finite_gap, finite_linear, overflow_linear)
    # Pick one smooth branch at h=0; abs(h) has zero autograd derivative there.
    correction = torch.where(positive_gap, F.softplus(-h), F.softplus(h))
    return linear + (target_m + target_c) * correction


def topk_token_terms(
    clean_logits: Tensor,
    off_logits: Tensor,
    target_ids: Tensor,
    *,
    beta: float,
    k: int,
    compute_mass: bool = True,
    chunk_size: int = 128,
    stage2_debugging: bool = True,
) -> TopkTokenResult:
    """Compute per-token q-weighted OFF BCE and clean-target binary group KL.

    Inputs describe one nonempty step. OFF chooses each token's support;
    selected values and the OFF group mass stay live for gradients.
    """
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and strictly positive")
    if clean_logits.ndim != 2 or off_logits.shape != clean_logits.shape:
        raise ValueError("clean and OFF logits must have matching [T,V] shape")
    tokens, vocab = clean_logits.shape
    if tokens == 0:
        raise ValueError("step must be nonempty")
    if type(k) is not int or not 2 <= k < vocab:
        raise ValueError("k must be an integer with 2 <= k < vocabulary size")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if target_ids.shape != (tokens,):
        raise ValueError("target IDs must have shape [T]")
    if clean_logits.device != off_logits.device or clean_logits.device != target_ids.device:
        raise ValueError("clean, OFF, and target devices must match")
    if not clean_logits.is_floating_point() or not off_logits.is_floating_point():
        raise ValueError("clean and OFF logits must be floating-point")
    if target_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("target IDs must have integer dtype")
    if bool((target_ids < 0).any()) or bool((target_ids >= vocab).any()):
        raise ValueError("target IDs must be in vocabulary range")

    negative_parts = []
    mass_parts = []
    entropy_parts = []
    clean_mass_parts = []
    off_mass_parts = []
    gap_parts = []
    coverage_parts = []
    for start in range(0, tokens, chunk_size):
        stop = min(start + chunk_size, tokens)
        # Check after conversion too: finite FP64 inputs can overflow in FP32.
        off_chunk = off_logits[start:stop].float()
        with torch.no_grad():
            clean_chunk = clean_logits[start:stop].float()
        if not bool(torch.isfinite(clean_chunk).all()) or not bool(torch.isfinite(off_chunk).all()):
            raise ValueError("clean and OFF logits must be finite after float32 conversion")
        ids = off_chunk.detach().topk(k, dim=-1).indices
        # These checks touch k IDs per token and never sort the vocabulary.
        if stage2_debugging:
            if bool((ids < 0).any()) or bool((ids >= vocab).any()):
                raise ValueError("selected top-k IDs are out of range")
            sorted_ids = ids.sort(dim=-1).values
            if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
                raise ValueError("selected top-k IDs are not unique")
        selected = off_chunk.gather(-1, ids)
        logq = selected.log_softmax(-1)
        q = logq.exp()
        negative_parts.append((q * F.softplus(beta * logq)).sum(-1))
        entropy_parts.append(-(q * torch.where(q > 0, logq, 0.0)).sum(-1))
        coverage_parts.append((ids == target_ids[start:stop, None]).any(-1).float())

        if compute_mass:
            support = torch.zeros_like(off_chunk, dtype=torch.bool).scatter_(-1, ids, True)
            # The clean branch only supplies a fixed target for this call.
            with torch.no_grad():
                _, _, _, _, clean_h = _group_gap(clean_chunk, support)
                target_logm = -F.softplus(-clean_h)
                target_logc = -F.softplus(clean_h)
                target_m = target_logm.exp()
                target_c = target_logc.exp()
            off_a_max, off_b_max, off_a_corr, off_b_corr, off_h = _group_gap(off_chunk, support)
            cross_entropy = _weighted_off_cross_entropy(
                off_a_max, off_b_max, off_a_corr, off_b_corr, off_h,
                target_m, target_c,
            )
            # Zero-weight clean groups contribute exactly zero to entropy.
            target_entropy = -(
                target_m * torch.where(target_m > 0, target_logm, 0.0)
                + target_c * torch.where(target_c > 0, target_logc, 0.0)
            )
            mass_parts.append(cross_entropy - target_entropy)
            off_m = (-F.softplus(-off_h)).exp()
            clean_mass_parts.append(target_m)
            off_mass_parts.append(off_m)
            gap_parts.append((target_m - off_m).abs())
        else:
            zeros = off_chunk.new_zeros(stop - start)
            mass_parts.append(zeros)
            clean_mass_parts.append(zeros)
            off_mass_parts.append(zeros)
            gap_parts.append(zeros)

    return TopkTokenResult(
        negative_loss=torch.cat(negative_parts),
        mass_loss=torch.cat(mass_parts),
        entropy=torch.cat(entropy_parts),
        clean_mass=torch.cat(clean_mass_parts),
        off_mass=torch.cat(off_mass_parts),
        mass_abs_gap=torch.cat(gap_parts),
        gt_coverage=torch.cat(coverage_parts),
    )


def topk_negative_and_mass(
    clean_logits: Tensor,
    off_logits: Tensor,
    target_ids: Tensor,
    *,
    beta: float,
    k: int,
    compute_mass: bool = True,
    chunk_size: int = 128,
    stage2_debugging: bool = True,
) -> TopkStepResult:
    """Preserve the original step-level interface by averaging token terms."""
    terms = topk_token_terms(
        clean_logits, off_logits, target_ids,
        beta=beta, k=k, compute_mass=compute_mass,
        chunk_size=chunk_size, stage2_debugging=stage2_debugging,
    )
    return TopkStepResult(*(value.mean() for value in terms))
