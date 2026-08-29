"""CoMT data preparation, non-truncating collation, and length auditing."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentTokenIds,
    parse_latent_spans,
)


logger = logging.getLogger(__name__)


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

    resolved_root = Path(image_root).resolve(strict=True)
    paths: list[str] = []
    for raw_image in raw_images:
        candidate = Path(raw_image)
        if (
            candidate.is_absolute()
            or not candidate.parts
            or any(part in ("", ".", "..") for part in candidate.parts)
            or any(ord(character) < 32 or ord(character) == 127 for character in raw_image)
        ):
            raise ValueError("image_input must use a safe relative path")
        if candidate.parts[0] == resolved_root.name:
            candidate = Path(*candidate.parts[1:])
        if not candidate.parts:
            raise ValueError("image_input must use a safe relative path")
        resolved = (resolved_root / candidate).resolve(strict=False)
        if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
            raise ValueError("resolved image_input must remain inside image_root")
        if not resolved.is_file():
            raise FileNotFoundError(f"missing original image: {resolved}")
        paths.append(str(resolved))
    return tuple(paths)


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


def find_unique_subsequence_end(sequence: torch.Tensor, pattern: torch.Tensor) -> int:
    """Return the position immediately after one required assistant marker."""
    matches = [
        start + pattern.numel()
        for start in range(sequence.numel() - pattern.numel() + 1)
        if torch.equal(sequence[start : start + pattern.numel()], pattern)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one assistant marker, found {len(matches)}")
    return matches[0]


def _load_original_images(paths: tuple[str, ...]) -> list[Image.Image]:
    """Materialize original image files as detached RGB PIL images."""
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
        images.append(image)
    return images


class Stage2DataCollator:
    """Tokenize exactly one complete CoMT sample without truncation."""

    def __init__(self, processor, token_ids, latent_size, supervise_latent_boundaries):
        self.processor = processor
        self.token_ids = token_ids
        self.latent_size = latent_size
        self.supervise_latent_boundaries = supervise_latent_boundaries
        self.assistant_pattern = processor.tokenizer(
            "<|im_start|>assistant", add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]

    def __call__(self, examples: list[Stage2PreparedExample]) -> dict[str, torch.Tensor]:
        if len(examples) != 1:
            raise NotImplementedError("Stage 2 collator currently supports batch_size=1 only")

        example = examples[0]
        text = self.processor.apply_chat_template(
            example.conversation, tokenize=False, add_generation_prompt=False
        )
        batch = self.processor(
            text=[text],
            images=[_load_original_images(example.original_image_paths)],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        forbidden = {"pixel_values_latent", "image_grid_thw_latent"} & batch.keys()
        if forbidden:
            raise ValueError(f"helper-image tensors leaked into Stage 2: {sorted(forbidden)}")
        if batch["input_ids"].shape[0] != 1:
            raise NotImplementedError("Stage 2 collator currently supports batch_size=1 only")

        parse_latent_spans(
            batch["input_ids"][0], token_ids=self.token_ids, latent_size=self.latent_size
        )
        assistant_content_start = find_unique_subsequence_end(
            batch["input_ids"][0], self.assistant_pattern
        )
        batch["labels"] = build_stage2_labels(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            assistant_content_start=assistant_content_start,
            token_ids=self.token_ids,
            latent_size=self.latent_size,
            supervise_latent_boundaries=self.supervise_latent_boundaries,
        )
        return batch


def build_prepared_example_length_measure(processor):
    """Build a length measurement closure that never truncates a sample."""

    def measure(example: Stage2PreparedExample) -> int:
        rendered = processor.apply_chat_template(
            example.conversation, tokenize=False, add_generation_prompt=False
        )
        encoded = processor(
            text=[rendered],
            images=[_load_original_images(example.original_image_paths)],
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        return int(encoded["input_ids"].shape[1])

    return measure


@dataclass(frozen=True)
class Stage2LengthStats:
    """Distribution of complete Stage 2 samples before length filtering."""

    total: int
    retained: int
    filtered: int
    filtered_percentage: float
    minimum: int
    median: float
    p90: float
    p95: float
    p99: float
    maximum: int
    latent_block_distribution: dict[int, int]


def filter_stage2_examples(*, examples, measure_length, max_length):
    """Keep complete samples at or under ``max_length`` and summarize all samples."""
    kept = []
    lengths: list[int] = []
    block_counts: Counter[int] = Counter()
    for example in examples:
        length = int(measure_length(example))
        lengths.append(length)
        block_count = (
            example.latent_block_count
            if isinstance(example, Stage2PreparedExample)
            else int(
                example.get(
                    "latent_block_count",
                    sum(
                        step.get("type") == "latent"
                        for step in example.get("sequence_plan", [])
                    ),
                )
            )
        )
        block_counts[block_count] += 1
        if length <= max_length:
            kept.append(example)

    if not lengths:
        raise ValueError("cannot compute Stage 2 length statistics for an empty dataset")

    values = torch.tensor(lengths, dtype=torch.float64)
    filtered = len(lengths) - len(kept)
    stats = Stage2LengthStats(
        total=len(lengths),
        retained=len(kept),
        filtered=filtered,
        filtered_percentage=100.0 * filtered / len(lengths),
        minimum=min(lengths),
        median=float(values.quantile(0.50)),
        p90=float(values.quantile(0.90)),
        p95=float(values.quantile(0.95)),
        p99=float(values.quantile(0.99)),
        maximum=max(lengths),
        latent_block_distribution=dict(sorted(block_counts.items())),
    )
    logger.info("Stage 2 length statistics: %s", asdict(stats))
    return kept, stats


def build_audit_parser() -> argparse.ArgumentParser:
    """Build the standalone CoMT Stage 2 length-audit interface."""
    parser = argparse.ArgumentParser(description="Audit CoMT Stage 2 sequence lengths")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--latent-size", type=int, required=True)
    parser.add_argument("--max-sequence-length", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Audit complete rendered samples once and emit a stable JSON summary."""
    args = build_audit_parser().parse_args(argv)
    if args.latent_size <= 0:
        raise ValueError("latent_size must be positive")
    if args.max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")

    processor = AutoProcessor.from_pretrained(args.processor_path)
    examples = [
        prepare_stage2_example(
            sample,
            image_root=args.image_root,
            latent_size=args.latent_size,
        )
        for sample in load_jsonl_examples(args.data_path)
    ]
    _, stats = filter_stage2_examples(
        examples=examples,
        measure_length=build_prepared_example_length_measure(processor),
        max_length=args.max_sequence_length,
    )
    print(json.dumps(asdict(stats), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
