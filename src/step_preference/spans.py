"""Stable sample identities and reasoning-plan character spans."""

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable


@dataclass(frozen=True)
class PlanCharSpan:
    """One complete plan entry's half-open interval in assistant content."""

    plan_index: int
    step_type: str
    start: int
    end: int


@dataclass(frozen=True)
class ReasoningStepPlan:
    """The immediate text-latent-text plan entries for one reasoning step."""

    step_id: int
    prev_text: PlanCharSpan
    latent: PlanCharSpan
    target_text: PlanCharSpan
    helper_image_idx: int | None


@dataclass(frozen=True)
class ReasoningStepSpan:
    """One reasoning step's half-open token intervals."""

    step_id: int
    prev_text_start: int
    prev_text_end: int
    latent_start: int
    latent_end: int
    target_text_start: int
    target_text_end: int
    helper_image_idx: int | None


def canonical_sample_id(row: dict) -> str:
    """Return an explicit sample ID or the SHA-256 of canonical source JSON."""
    sample_id = row.get("sample_id")
    if isinstance(sample_id, str) and sample_id:
        return sample_id
    canonical_json = json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def reject_duplicate_sample_ids(sample_ids: Iterable[str]) -> None:
    """Reject a source collection that would produce ambiguous sample joins."""
    seen: set[str] = set()
    for sample_id in sample_ids:
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)


def build_assistant_plan(
    sequence_plan: list,
    latent_size: int,
    *,
    require_reasoning_units: bool = False,
) -> tuple[str, tuple[PlanCharSpan, ...], tuple[ReasoningStepPlan, ...]]:
    """Render a plan and retain its complete-entry and strict-unit character spans."""
    if latent_size <= 0:
        raise ValueError("latent_size must be positive")

    latent_block = "<|latent_start|>" + "<|latent_pad|>" * latent_size + "<|latent_end|>"
    assistant_parts: list[str] = []
    plan_char_spans: list[PlanCharSpan] = []
    cursor = 0

    for plan_index, step in enumerate(sequence_plan):
        if not isinstance(step, dict):
            raise ValueError("sequence_plan steps must be objects")
        step_type = step.get("type")
        if step_type == "text":
            content = step.get("content")
            if not isinstance(content, str):
                raise ValueError("text step content must be a string")
            rendered = content
        elif step_type == "latent":
            helper_image = step.get("helper_image")
            if not isinstance(helper_image, str) or not helper_image:
                raise ValueError("latent step is missing helper_image")
            rendered = latent_block
        else:
            raise ValueError(f"unsupported sequence_plan step type: {step_type}")

        assistant_parts.append(rendered)
        plan_char_spans.append(
            PlanCharSpan(plan_index, step_type, cursor, cursor + len(rendered))
        )
        cursor += len(rendered)

    reasoning_steps: list[ReasoningStepPlan] = []
    helper_ordinal = 0
    for plan_index, step in enumerate(sequence_plan):
        if step["type"] != "latent":
            continue

        has_reasoning_unit = (
            plan_index > 0
            and plan_index + 1 < len(sequence_plan)
            and sequence_plan[plan_index - 1].get("type") == "text"
            and isinstance(sequence_plan[plan_index - 1].get("content"), str)
            and bool(sequence_plan[plan_index - 1]["content"])
            and sequence_plan[plan_index + 1].get("type") == "text"
            and isinstance(sequence_plan[plan_index + 1].get("content"), str)
            and bool(sequence_plan[plan_index + 1]["content"])
        )
        if not has_reasoning_unit:
            if require_reasoning_units:
                raise ValueError(
                    "each latent reasoning unit requires immediate non-empty text neighbors"
                )
            helper_ordinal += 1
            continue

        reasoning_steps.append(
            ReasoningStepPlan(
                step_id=helper_ordinal,
                prev_text=plan_char_spans[plan_index - 1],
                latent=plan_char_spans[plan_index],
                target_text=plan_char_spans[plan_index + 1],
                helper_image_idx=helper_ordinal,
            )
        )
        helper_ordinal += 1

    return "".join(assistant_parts), tuple(plan_char_spans), tuple(reasoning_steps)
