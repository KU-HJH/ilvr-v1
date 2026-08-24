"""CoMT trajectory conversion and Stage 2 loss-label helpers."""

from dataclasses import dataclass
import json
from pathlib import Path

from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentTokenIds,
    parse_latent_spans,
)


@dataclass(frozen=True)
class Stage2PreparedExample:
    """A validated CoMT sample ready for processor tokenization."""

    conversation: list[dict]
    original_image_paths: tuple[str, ...]
    latent_block_count: int


def load_jsonl_examples(path: Path) -> list[dict]:
    """Load non-empty JSON Lines records without altering their contents."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_original_images(sample: dict, image_root: Path) -> tuple[str, ...]:
    """Resolve and verify only the original image inputs in a CoMT sample."""
    raw_images = sample.get("image_input")
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("image_input must contain at least one original image")
    if not all(isinstance(raw_image, str) and raw_image for raw_image in raw_images):
        raise ValueError("image_input entries must be non-empty strings")

    paths = tuple(str(Path(image_root) / raw_image) for raw_image in raw_images)
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing original image: {missing[0]}")
    return paths


def _assistant_text(sequence_plan: list, latent_size: int) -> tuple[str, int]:
    if latent_size <= 0:
        raise ValueError("latent_size must be positive")

    latent_block = "<|latent_start|>" + "<|latent_pad|>" * latent_size + "<|latent_end|>"
    assistant_parts: list[str] = []
    latent_block_count = 0
    for step in sequence_plan:
        if not isinstance(step, dict):
            raise ValueError("sequence_plan steps must be objects")
        step_type = step.get("type")
        if step_type == "text":
            content = step.get("content")
            if not isinstance(content, str):
                raise ValueError("text step content must be a string")
            assistant_parts.append(content)
        elif step_type == "latent":
            helper_image = step.get("helper_image")
            if not isinstance(helper_image, str) or not helper_image:
                raise ValueError("latent step is missing helper_image")
            assistant_parts.append(latent_block)
            latent_block_count += 1
        else:
            raise ValueError(f"unsupported sequence_plan step type: {step_type}")
    return "".join(assistant_parts), latent_block_count


def build_stage2_conversation(
    sample: dict,
    *,
    image_root: Path,
    latent_size: int,
) -> list[dict]:
    """Convert a raw CoMT sample into a multimodal user/assistant conversation."""
    sequence_plan = sample.get("sequence_plan")
    if not isinstance(sequence_plan, list):
        raise ValueError("sequence_plan must be a list")

    original_images = resolve_original_images(sample, image_root)
    assistant_text, _ = _assistant_text(sequence_plan, latent_size)
    text_input = sample.get("text_input")
    if not isinstance(text_input, str):
        raise ValueError("text_input must be a string")

    user_content = [{"type": "image", "image": path} for path in original_images]
    user_content.append({"type": "text", "text": text_input})
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]


def prepare_stage2_example(
    sample: dict,
    *,
    image_root: Path,
    latent_size: int,
) -> Stage2PreparedExample:
    """Build a conversation together with validated original-image metadata."""
    conversation = build_stage2_conversation(sample, image_root=image_root, latent_size=latent_size)
    _, latent_block_count = _assistant_text(sample["sequence_plan"], latent_size)
    return Stage2PreparedExample(
        conversation=conversation,
        original_image_paths=resolve_original_images(sample, image_root),
        latent_block_count=latent_block_count,
    )


def build_stage2_labels(
    *,
    input_ids,
    attention_mask,
    assistant_content_start,
    token_ids: LatentTokenIds,
    latent_size,
    supervise_latent_boundaries,
):
    """Mask prompt, padding, and latent slots in token-level Stage 2 labels."""
    if input_ids.shape[0] != 1:
        raise NotImplementedError("Stage 2 labels currently support batch_size=1 only")

    labels = input_ids.clone()
    labels[:, :assistant_content_start] = -100
    labels.masked_fill_(attention_mask == 0, -100)
    spans = parse_latent_spans(
        input_ids[0], token_ids=token_ids, latent_size=latent_size
    )
    for span in spans:
        labels[0, list(span.latent_positions)] = -100
        if not supervise_latent_boundaries:
            labels[0, span.start_position] = -100
            labels[0, span.end_position] = -100
    return labels
