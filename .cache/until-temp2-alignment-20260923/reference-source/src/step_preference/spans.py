"""Stable Stage 2 sample identities and tokenized reasoning-step spans."""

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentSpan


@dataclass(frozen=True)
class ReasoningStepSpan:
    """One complete latent-mediated reasoning step in token coordinates."""

    step_id: int
    prev_text_start: int
    prev_text_end: int
    latent_start: int
    latent_end: int
    target_text_start: int
    target_text_end: int
    helper_image_idx: int | None


def stable_sample_id(sample: Mapping[str, object]) -> str:
    """Return an explicit ID or a canonical-record digest without reading images."""
    explicit = sample.get("sample_id")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError("sample_id must be a non-empty string")
        return explicit
    canonical = json.dumps(sample, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _latent_plan_entries(
    sequence_plan: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    entries = []
    for step in sequence_plan:
        if not isinstance(step, Mapping):
            raise ValueError("sequence_plan steps must be objects")
        step_type = step.get("type")
        if step_type == "text":
            if not isinstance(step.get("content"), str):
                raise ValueError("text step content must be a string")
        elif step_type == "latent":
            helper_image = step.get("helper_image")
            if not isinstance(helper_image, str) or not helper_image:
                raise ValueError("latent step is missing helper_image")
            entries.append(step)
        else:
            raise ValueError(f"unsupported sequence_plan step type: {step_type}")
    return tuple(entries)


def derive_reasoning_step_spans(
    *,
    sequence_plan: Sequence[Mapping[str, object]],
    assistant_content_start: int,
    assistant_content_end: int,
    latent_spans: Sequence[LatentSpan],
) -> tuple[ReasoningStepSpan, ...]:
    """Derive complete text/latent/text intervals from parsed latent blocks."""
    if assistant_content_start < 0 or assistant_content_end < assistant_content_start:
        raise ValueError("assistant content bounds must be ordered non-negative positions")

    latent_entries = _latent_plan_entries(sequence_plan)
    if len(latent_entries) != len(latent_spans):
        raise ValueError("sequence_plan latent count must match tokenized latent blocks")

    steps = []
    previous_end = assistant_content_start
    for step_id, latent_span in enumerate(latent_spans):
        if not latent_span.latent_positions:
            raise ValueError("latent span must contain latent-pad positions")
        latent_start = latent_span.latent_positions[0]
        latent_end = latent_span.end_position
        next_start = (
            latent_spans[step_id + 1].start_position
            if step_id + 1 < len(latent_spans)
            else assistant_content_end
        )
        prev_text_start = previous_end
        prev_text_end = latent_span.start_position
        target_text_start = latent_span.end_position + 1
        target_text_end = next_start
        if (
            prev_text_start >= prev_text_end
            or latent_start >= latent_end
            or target_text_start >= target_text_end
        ):
            raise ValueError("complete reasoning step requires non-empty previous and target text")
        if (
            prev_text_start < assistant_content_start
            or target_text_end > assistant_content_end
            or latent_span.start_position < prev_text_start
        ):
            raise ValueError("latent spans must be ordered within assistant content")
        steps.append(
            ReasoningStepSpan(
                step_id=step_id,
                prev_text_start=prev_text_start,
                prev_text_end=prev_text_end,
                latent_start=latent_start,
                latent_end=latent_end,
                target_text_start=target_text_start,
                target_text_end=target_text_end,
                helper_image_idx=step_id,
            )
        )
        previous_end = target_text_start
    return tuple(steps)
