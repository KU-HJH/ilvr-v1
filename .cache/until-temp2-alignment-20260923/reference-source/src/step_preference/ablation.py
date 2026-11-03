"""Isolated final-decoder attention ablations for one reasoning step."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .spans import ReasoningStepSpan


@dataclass(frozen=True)
class StepAblationSpec:
    sample_idx: int
    step_id: int
    mode: Literal["visual", "text"]
    query_start: int
    query_end: int
    blocked_kv_start: int
    blocked_kv_end: int


def build_step_ablation_spec(
    span: ReasoningStepSpan,
    mode: Literal["visual", "text"],
    sample_idx: int = 0,
) -> StepAblationSpec:
    """Map one reasoning span to its target-query and modality-KV rectangle."""
    if mode == "visual":
        blocked_kv_start, blocked_kv_end = span.latent_start, span.latent_end
    elif mode == "text":
        blocked_kv_start, blocked_kv_end = span.prev_text_start, span.prev_text_end
    else:
        raise ValueError("ablation mode must be visual or text")
    return StepAblationSpec(
        sample_idx=sample_idx,
        step_id=span.step_id,
        mode=mode,
        query_start=span.target_text_start,
        query_end=span.target_text_end,
        blocked_kv_start=blocked_kv_start,
        blocked_kv_end=blocked_kv_end,
    )


def build_inverted_causal_mask(
    attention_mask: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """Build an additive full-sequence causal mask with key padding."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be two-dimensional")
    if not dtype.is_floating_point:
        raise ValueError("causal mask dtype must be floating point")
    batch_size, sequence_length = attention_mask.shape
    allowed = torch.ones(
        (sequence_length, sequence_length),
        dtype=torch.bool,
        device=attention_mask.device,
    ).tril()
    allowed = allowed[None, None] & attention_mask[:, None, None, :].bool()
    result = torch.zeros(
        (batch_size, 1, sequence_length, sequence_length),
        dtype=dtype,
        device=attention_mask.device,
    )
    return result.masked_fill(~allowed, torch.finfo(dtype).min)


def apply_step_ablation(
    base_mask: Tensor,
    spec: StepAblationSpec,
) -> Tensor:
    """Clone a 4D additive mask and block exactly one query/KV rectangle."""
    if base_mask.ndim != 4:
        raise ValueError("base attention mask must be four-dimensional")
    if not base_mask.is_floating_point():
        raise ValueError("base attention mask must use a floating dtype")
    batch_size, _, query_length, kv_length = base_mask.shape
    if not 0 <= spec.sample_idx < batch_size:
        raise ValueError("sample_idx is out of bounds")
    if not 0 <= spec.query_start < spec.query_end <= query_length:
        raise ValueError("query range is out of bounds")
    if not 0 <= spec.blocked_kv_start < spec.blocked_kv_end <= kv_length:
        raise ValueError("blocked KV range is out of bounds")
    if spec.blocked_kv_end > spec.query_start:
        raise ValueError("blocked KV range must be strictly earlier than every query")

    result = base_mask.clone()
    result[
        spec.sample_idx,
        :,
        spec.query_start : spec.query_end,
        spec.blocked_kv_start : spec.blocked_kv_end,
    ] = torch.finfo(result.dtype).min
    return result
