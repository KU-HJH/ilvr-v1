"""Causal token and evidence-window score primitives."""

from __future__ import annotations

import operator

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# Caps each temporary FP32 vocabulary activation at [B, 32, V].  Auxiliary
# evidence scoring deliberately uses a fixed implementation detail rather than
# adding another experiment knob.
DEFAULT_CAUSAL_TOKEN_CHUNK_SIZE = 32


def causal_token_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the log-probability assigned to each target token.

    Position ``t`` is scored from the logits at ``t - 1``.  Position zero and
    ignored labels (``-100``) have exactly zero score.
    """
    if not torch.is_tensor(logits) or logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if not torch.is_tensor(labels) or labels.ndim != 2:
        raise ValueError("labels must have shape [B, T]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [B, T] matching logits")
    if labels.device != logits.device:
        raise ValueError("labels and logits must be on the same device")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("labels must have an integer dtype")
    if logits.shape[1] == 0:
        return logits.new_zeros((logits.shape[0], 0))

    targets = labels[:, 1:]
    valid = targets.ne(-100)
    if valid.any():
        valid_targets = targets[valid]
        if valid_targets.min() < 0 or valid_targets.max() >= logits.shape[-1]:
            raise ValueError("labels contain a target outside the logits vocabulary")
    safe = targets.clamp_min(0)
    selected = (
        logits[:, :-1].log_softmax(-1)
        .gather(-1, safe.to(dtype=torch.long).unsqueeze(-1))
        .squeeze(-1)
    )
    selected = selected * valid.to(dtype=selected.dtype)
    return torch.cat((logits.new_zeros((logits.shape[0], 1)), selected), dim=1)


def chunked_causal_target_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_chunk_size: int = DEFAULT_CAUSAL_TOKEN_CHUNK_SIZE,
) -> torch.Tensor:
    """Return causal target scores in bounded FP32 token chunks.

    The result has shape ``[B, max(T - 1, 0)]``: result position ``t - 1``
    scores label position ``t``.  Ignored labels (``-100``) have exactly zero
    score.  Grad-enabled chunks are checkpointed so backward recomputes the
    bounded FP32 log-softmax instead of retaining every vocabulary activation.
    """
    if not torch.is_tensor(logits) or logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if not torch.is_tensor(labels) or labels.ndim != 2:
        raise ValueError("labels must have shape [B, T]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [B, T] matching logits")
    if labels.device != logits.device:
        raise ValueError("labels and logits must be on the same device")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("labels must have an integer dtype")
    if isinstance(token_chunk_size, bool):
        raise ValueError("token_chunk_size must be a positive integer")
    try:
        token_chunk_size = operator.index(token_chunk_size)
    except TypeError as exc:
        raise ValueError("token_chunk_size must be a positive integer") from exc
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be a positive integer")

    targets = labels[:, 1:]
    valid = targets.ne(-100)
    if valid.any():
        valid_targets = targets[valid]
        if valid_targets.min() < 0 or valid_targets.max() >= logits.shape[-1]:
            raise ValueError("labels contain a target outside the logits vocabulary")
    if targets.shape[1] == 0:
        return torch.empty(
            (logits.shape[0], 0),
            dtype=torch.float32,
            device=logits.device,
        )

    safe_targets = targets.clamp_min(0).to(dtype=torch.long)
    chunks = []
    for start in range(0, targets.shape[1], token_chunk_size):
        stop = min(start + token_chunk_size, targets.shape[1])
        chunk_logits = logits[:, start:stop]
        chunk_targets = safe_targets[:, start:stop]
        chunk_valid = valid[:, start:stop]
        if torch.is_grad_enabled() and chunk_logits.requires_grad:
            selected = checkpoint(
                _fp32_target_logps_chunk,
                chunk_logits,
                chunk_targets,
                chunk_valid,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            selected = _fp32_target_logps_chunk(
                chunk_logits,
                chunk_targets,
                chunk_valid,
            )
        chunks.append(selected)
    return torch.cat(chunks, dim=1)


def _fp32_target_logps_chunk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    fp32_logits = logits.float()
    selected = (
        F.log_softmax(fp32_logits, dim=-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
    )
    return selected * valid.to(dtype=torch.float32)


def masked_token_sums(scores: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Sum token scores over each ``[B, N, T]`` mask."""
    if not torch.is_tensor(scores) or scores.ndim != 2:
        raise ValueError("scores must have shape [B, T]")
    if not torch.is_tensor(masks) or masks.ndim != 3:
        raise ValueError("masks must have shape [B, N, T]")
    if masks.shape[0] != scores.shape[0] or masks.shape[-1] != scores.shape[-1]:
        raise ValueError("masks must match scores in batch and token dimensions")
    if masks.device != scores.device:
        raise ValueError("masks and scores must be on the same device")
    return torch.einsum("bt,bnt->bn", scores, masks.to(dtype=scores.dtype))


def reference_adjusted_effect(
    policy_full: torch.Tensor,
    policy_cf: torch.Tensor,
    ref_full: torch.Tensor,
    ref_cf: torch.Tensor,
) -> torch.Tensor:
    """Subtract the reference full-vs-counterfactual effect from policy's."""
    _validate_same_shapes(
        ("policy_full", policy_full),
        ("policy_cf", policy_cf),
        ("ref_full", ref_full),
        ("ref_cf", ref_cf),
    )
    return (policy_full - policy_cf) - (ref_full - ref_cf)


def evidence_window_margins(
    delta_by_evidence: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    """Sum reference-adjusted token effects directly inside each window."""
    if not torch.is_tensor(delta_by_evidence) or delta_by_evidence.ndim != 3:
        raise ValueError("delta_by_evidence must have shape [B, J, T]")
    if not torch.is_tensor(masks) or masks.ndim != 4:
        raise ValueError("masks must have shape [B, J, K, T]")
    expected = (
        delta_by_evidence.shape[0],
        delta_by_evidence.shape[1],
        delta_by_evidence.shape[2],
    )
    if (masks.shape[0], masks.shape[1], masks.shape[3]) != expected:
        raise ValueError("masks must match delta_by_evidence in B, J, and T dimensions")
    if masks.device != delta_by_evidence.device:
        raise ValueError("masks and delta_by_evidence must be on the same device")
    return torch.einsum(
        "bjt,bjkt->bjk", delta_by_evidence, masks.to(dtype=delta_by_evidence.dtype)
    )


def _validate_same_shapes(*named_tensors: tuple[str, torch.Tensor]) -> None:
    first_name, first = named_tensors[0]
    if not torch.is_tensor(first):
        raise ValueError(f"{first_name} must be a tensor")
    for name, value in named_tensors[1:]:
        if not torch.is_tensor(value):
            raise ValueError(f"{name} must be a tensor")
        if value.shape != first.shape:
            raise ValueError("score tensors must have matching shapes")
        if value.device != first.device:
            raise ValueError("score tensors must be on the same device")
