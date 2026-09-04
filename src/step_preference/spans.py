"""Stable sample identities and reasoning-plan character spans."""

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable


LATENT_START_TOKEN = "<|latent_start|>"
LATENT_PAD_TOKEN = "<|latent_pad|>"
LATENT_END_TOKEN = "<|latent_end|>"
ASSISTANT_PLAN_RENDERING_CONCATENATED = "concatenated-v1"
ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF = "adjacent-text-space-lf-v1"


def normalize_assistant_plan_rendering(value: str) -> str:
    if value not in {
        ASSISTANT_PLAN_RENDERING_CONCATENATED,
        ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    }:
        raise ValueError(f"unsupported assistant plan rendering: {value}")
    return value


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


def expand_qwen_image_placeholders(
    rendered_text: str,
    *,
    image_grid_thw,
    image_token: str,
    merge_size: int,
) -> str:
    """Repeat rendered image placeholders exactly as the Qwen processor does."""
    if merge_size <= 0:
        raise ValueError("merge_size must be positive")

    placeholder = "<|placeholder|>"
    expanded = rendered_text
    grid_index = 0
    while image_token in expanded:
        if grid_index >= len(image_grid_thw):
            raise ValueError("fewer image grids than rendered image placeholders")
        grid = image_grid_thw[grid_index]
        product = int(grid.prod().item()) if hasattr(grid, "prod") else _product(grid)
        merge_length = merge_size**2
        if product % merge_length:
            raise ValueError("image grid size is not divisible by merge_size squared")
        expanded = expanded.replace(
            image_token,
            placeholder * (product // merge_length),
            1,
        )
        grid_index += 1
    if grid_index != len(image_grid_thw):
        raise ValueError("more image grids than rendered image placeholders")
    return expanded.replace(placeholder, image_token)


def _product(values) -> int:
    product = 1
    for value in values:
        product *= int(value)
    return product


def map_reasoning_step_spans(
    *,
    rendered_text: str,
    assistant_text: str,
    plan_char_spans: tuple[PlanCharSpan, ...],
    reasoning_steps: tuple[ReasoningStepPlan, ...],
    offset_mapping,
    input_ids,
    latent_pad_token_id: int,
) -> tuple[ReasoningStepSpan, ...]:
    """Map assistant character plans onto half-open full-sequence token spans."""
    offsets = (
        offset_mapping.tolist() if hasattr(offset_mapping, "tolist") else offset_mapping
    )
    ids = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    if len(offsets) != 1 or len(ids) != 1:
        raise NotImplementedError("reasoning span mapping currently supports batch_size=1 only")
    offsets = [tuple(pair) for pair in offsets[0]]
    ids = ids[0]
    if len(offsets) != len(ids):
        raise ValueError("offset_mapping and input_ids must have the same sequence length")

    assistant_start = rendered_text.rfind(assistant_text)
    if assistant_start < 0:
        raise ValueError("assistant text is absent from the rendered chat")

    boundaries = {
        assistant_start + boundary
        for span in plan_char_spans
        for boundary in (span.start, span.end)
    }
    for span in plan_char_spans:
        if span.step_type != "latent":
            continue
        latent_text = assistant_text[span.start : span.end]
        if not (
            latent_text.startswith(LATENT_START_TOKEN)
            and latent_text.endswith(LATENT_END_TOKEN)
        ):
            raise ValueError(
                "latent plan span does not contain a complete latent control block"
            )
        boundaries.add(assistant_start + span.start + len(LATENT_START_TOKEN))
        boundaries.add(assistant_start + span.end - len(LATENT_END_TOKEN))

    for token_index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if any(start < boundary < end for boundary in boundaries):
            raise ValueError(f"token {token_index} crosses a plan boundary")

    if not reasoning_steps:
        return ()

    def token_interval(char_start: int, char_end: int) -> tuple[int, int]:
        selected = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and start >= char_start and end <= char_end
        ]
        if (
            not selected
            or selected != list(range(selected[0], selected[-1] + 1))
            or offsets[selected[0]][0] != char_start
            or offsets[selected[-1]][1] != char_end
        ):
            raise ValueError(
                "character span does not map to an exact contiguous token interval"
            )
        return selected[0], selected[-1] + 1

    mapped: list[ReasoningStepSpan] = []
    for step in reasoning_steps:
        prev_start, prev_end = token_interval(
            assistant_start + step.prev_text.start,
            assistant_start + step.prev_text.end,
        )
        latent_start, latent_end = token_interval(
            assistant_start + step.latent.start + len(LATENT_START_TOKEN),
            assistant_start + step.latent.end - len(LATENT_END_TOKEN),
        )
        target_start, target_end = token_interval(
            assistant_start + step.target_text.start,
            assistant_start + step.target_text.end,
        )
        if any(
            token_id != latent_pad_token_id
            for token_id in ids[latent_start:latent_end]
        ):
            raise ValueError("latent token interval contains a non-latent-pad token")
        mapped.append(
            ReasoningStepSpan(
                step_id=step.step_id,
                prev_text_start=prev_start,
                prev_text_end=prev_end,
                latent_start=latent_start,
                latent_end=latent_end,
                target_text_start=target_start,
                target_text_end=target_end,
                helper_image_idx=step.helper_image_idx,
            )
        )
    return tuple(mapped)


def build_assistant_plan(
    sequence_plan: list,
    latent_size: int,
    *,
    require_reasoning_units: bool = False,
    assistant_plan_rendering: str = ASSISTANT_PLAN_RENDERING_CONCATENATED,
) -> tuple[str, tuple[PlanCharSpan, ...], tuple[ReasoningStepPlan, ...]]:
    """Render a plan and retain its complete-entry and strict-unit character spans."""
    if latent_size <= 0:
        raise ValueError("latent_size must be positive")
    assistant_plan_rendering = normalize_assistant_plan_rendering(
        assistant_plan_rendering
    )

    latent_block = (
        LATENT_START_TOKEN + LATENT_PAD_TOKEN * latent_size + LATENT_END_TOKEN
    )
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

        if (
            assistant_plan_rendering
            == ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF
            and plan_index > 0
            and sequence_plan[plan_index - 1].get("type") == "text"
            and step_type == "text"
        ):
            assistant_parts.append(" \n")
            cursor += 2
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
