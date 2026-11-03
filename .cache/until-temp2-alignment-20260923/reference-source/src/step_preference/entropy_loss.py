"""Entropy diagnostics and margin loss for step-wise preference training."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class EntropyMarginResult:
    """Batch entropy-margin loss and its per-block diagnostics."""

    loss: Tensor
    branch_losses: Tensor
    clean_entropy: Tensor
    ablated_entropy: Tensor


def selected_token_entropies(
    logits: Tensor, token_mask: Tensor, *, stage2_debugging: bool = True,
) -> tuple[Tensor, ...]:
    """Return each selected token entropy, grouped by its original block."""
    if logits.ndim < 2:
        raise ValueError("logits must include block and token dimensions")
    if logits.shape[:-1] != token_mask.shape:
        raise ValueError("logits and token_mask shapes are inconsistent")
    if not logits.is_floating_point():
        raise ValueError("logits must be finite floating-point values")
    if token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be a boolean tensor")

    logits_f = logits.float()
    if stage2_debugging and not torch.isfinite(logits_f).all():
        raise ValueError("logits must be finite after float32 conversion")

    token_count = token_mask.shape[-1]
    block_shape = token_mask.shape[:-1]
    block_count = math.prod(block_shape) if block_shape else 1
    logits_blocks = logits_f.reshape(block_count, token_count, logits_f.shape[-1])
    mask_blocks = token_mask.reshape(block_count, token_count)
    counts = mask_blocks.sum(dim=-1)
    if stage2_debugging and torch.any(counts == 0):
        raise ValueError("every entropy block requires at least one selected token")

    selected_logits = logits_blocks[mask_blocks]
    log_probs = torch.log_softmax(selected_logits, dim=-1)
    probs = log_probs.exp()
    selected_entropy = -(probs * log_probs).sum(dim=-1)
    return tuple(torch.split(selected_entropy, counts.detach().cpu().tolist()))


def selected_token_entropy(logits: Tensor, token_mask: Tensor) -> Tensor:
    """Return token-averaged entropy for diagnostics, one mean per block."""
    block_shape = token_mask.shape[:-1]
    per_block = selected_token_entropies(logits, token_mask)
    if not per_block:
        return logits.float().new_empty(block_shape)
    return torch.stack(
        tuple(values.mean(dtype=torch.float32) for values in per_block)
    ).reshape(block_shape)


def entropy_margin_loss(
    clean_entropy: Tensor,
    ablated_entropy: Tensor,
    margin: float,
    *, stage2_debugging: bool = True,
) -> EntropyMarginResult:
    """Penalize blocks whose ablated entropy does not exceed clean entropy."""
    if clean_entropy.shape != ablated_entropy.shape:
        raise ValueError("clean and ablated entropy shapes are inconsistent")
    if not clean_entropy.is_floating_point() or not ablated_entropy.is_floating_point():
        raise ValueError("entropy tensors must be floating-point")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(margin)
        or margin < 0.0
    ):
        raise ValueError("margin must be finite and nonnegative")

    clean_f = clean_entropy.float()
    ablated_f = ablated_entropy.float()
    if stage2_debugging and (not torch.isfinite(clean_f).all() or not torch.isfinite(ablated_f).all()):
        raise ValueError("entropy tensors must be finite after float32 conversion")
    margin_f = torch.as_tensor(
        margin, dtype=torch.float32, device=clean_f.device
    )
    if stage2_debugging and not bool(torch.isfinite(margin_f)):
        raise ValueError("margin must be finite after float32 conversion and nonnegative")

    branch_losses = torch.relu(margin_f - (ablated_f - clean_f))
    if branch_losses.numel():
        loss = branch_losses.mean(dtype=torch.float32)
    else:
        loss = clean_f.sum() * 0.0 + ablated_f.sum() * 0.0
    return EntropyMarginResult(
        loss=loss,
        branch_losses=branch_losses,
        clean_entropy=clean_f,
        ablated_entropy=ablated_f,
    )
