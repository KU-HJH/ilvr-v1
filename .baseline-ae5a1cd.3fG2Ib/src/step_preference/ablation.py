"""Additive attention masks for one naive modality-ablation forward."""

from dataclasses import dataclass
from typing import Literal

import torch


AblationModality = Literal["visual", "textual"]


def _validate_position(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer token position")


@dataclass(frozen=True)
class StepAblationSpec:
    """One logical target-token span and the source span hidden from it."""

    modality: AblationModality
    target_start: int
    target_end: int
    source_start: int
    source_end: int

    def __post_init__(self) -> None:
        if self.modality not in ("visual", "textual"):
            raise ValueError("modality must be 'visual' or 'textual'")
        for name in ("target_start", "target_end", "source_start", "source_end"):
            _validate_position(name, getattr(self, name))
        if self.target_start < 1:
            raise ValueError("target_start must be at least 1 for causal next-token prediction")
        if self.target_end <= self.target_start:
            raise ValueError("target interval must be nonempty")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("source interval must be nonempty and nonnegative")


def build_additive_step_attention_mask(
    attention_mask: torch.Tensor,
    spec: StepAblationSpec,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    attn_implementation: str,
) -> torch.Tensor:
    """Build a causal/padding mask with one shifted target-to-source block."""
    if attn_implementation == "flash_attention_2":
        raise ValueError("flash_attention_2 does not support Stage 2 additive edge masks")
    if attn_implementation not in ("eager", "sdpa"):
        raise ValueError(f"unsupported attention implementation: {attn_implementation!r}")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [1, sequence_length]")
    if attention_mask.shape[0] != 1 or attention_mask.shape[1] == 0:
        raise ValueError("attention_mask must have shape [1, sequence_length]")
    if not dtype.is_floating_point:
        raise ValueError("dtype must be a floating compute dtype")

    sequence_length = attention_mask.shape[1]
    if spec.target_end > sequence_length or spec.source_end > sequence_length:
        raise ValueError(
            f"ablation intervals must fit sequence length {sequence_length}"
        )

    resolved_device = torch.device(device)
    minimum = torch.finfo(dtype).min
    positions = torch.arange(sequence_length, device=resolved_device)
    blocked = positions.unsqueeze(0) > positions.unsqueeze(1)
    padded_keys = ~attention_mask[0].to(device=resolved_device, dtype=torch.bool)
    blocked = blocked | padded_keys.unsqueeze(0)
    additive_mask = torch.zeros(
        (1, 1, sequence_length, sequence_length),
        dtype=dtype,
        device=resolved_device,
    ).masked_fill(blocked.view(1, 1, sequence_length, sequence_length), minimum)

    # A logical target token at p is predicted by physical decoder row p - 1.
    query_start = spec.target_start - 1
    query_end = spec.target_end - 1
    additive_mask[
        :, :, query_start:query_end, spec.source_start : spec.source_end
    ] = minimum
    return additive_mask
