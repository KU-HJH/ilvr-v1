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


def resolve_image_root_path(
    raw_path: str,
    image_root: Path,
    *,
    image_role: str,
) -> str:
    """Resolve one safe image-root-relative path, stripping the root name once."""
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{image_role} image path must be a non-empty string")

    resolved_root = Path(image_root).resolve(strict=True)
    candidate = Path(raw_path)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in ("", ".", "..") for part in candidate.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in raw_path)
    ):
        raise ValueError(f"{image_role} image must use a safe relative path")
    if candidate.parts[0] == resolved_root.name:
        candidate = Path(*candidate.parts[1:])
    if not candidate.parts:
        raise ValueError(f"{image_role} image must use a safe relative path")

    resolved = (resolved_root / candidate).resolve(strict=False)
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise ValueError(f"resolved {image_role} image must remain inside image_root")
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {image_role} image: {resolved}")
    return str(resolved)


def resolve_original_images(sample: dict, image_root: Path) -> tuple[str, ...]:
    """Resolve and verify only the original image inputs in a CoMT sample."""
    raw_images = sample.get("image_input")
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("image_input must contain at least one original image")
    if not all(isinstance(raw_image, str) and raw_image for raw_image in raw_images):
        raise ValueError("image_input entries must be non-empty strings")

    return tuple(
        resolve_image_root_path(
            raw_image,
            image_root,
            image_role="original",
        )
        for raw_image in raw_images
    )


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

    def __init__(
        self,
        processor,
        token_ids,
        latent_size,
        supervise_latent_boundaries,
        enable_stage2_objective=False,
    ):
        self.processor = processor
        self.token_ids = token_ids
        self.latent_size = latent_size
        self.supervise_latent_boundaries = supervise_latent_boundaries
        self.enable_stage2_objective = enable_stage2_objective
        self.assistant_pattern = processor.tokenizer(
            "<|im_start|>assistant", add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]

    def __call__(
        self, examples: list[Stage2PreparedExample]
    ) -> dict[str, torch.Tensor | list[list[tuple[int, int]]]]:
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
        if self.enable_stage2_objective:
            batch["stage2_text_segments"] = [
                _derive_stage2_text_segments(
                    tokenizer=self.processor.tokenizer,
                    rendered_text=text,
                    final_input_ids=batch["input_ids"][0],
                    example=example,
                    token_ids=self.token_ids,
                    latent_size=self.latent_size,
                )
            ]
        return batch


def _one_dimensional_list(value, *, name: str, entries_are_pairs: bool = False) -> list:
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    is_batched = bool(
        value
        and isinstance(value[0], (list, tuple))
        and (
            not entries_are_pairs
            or (value[0] and isinstance(value[0][0], (list, tuple)))
        )
    )
    if is_batched:
        if len(value) != 1:
            raise ValueError(f"{name} must contain exactly one tokenized row")
        value = value[0]
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a one-dimensional token sequence")
    return value


def _token_boundary(offsets: list[tuple[int, int]], character_position: int) -> int:
    for index, (start, end) in enumerate(offsets):
        if start < character_position < end:
            raise ValueError(
                "assistant content has BPE-boundary ambiguity in full rendered tokenization"
            )
        if start >= character_position and end > start:
            return index
    return len(offsets)


def _assistant_content(example: Stage2PreparedExample) -> str:
    if not example.conversation or example.conversation[-1].get("role") != "assistant":
        raise ValueError("Stage 2 conversation must end with one assistant turn")
    content = example.conversation[-1].get("content")
    if (
        not isinstance(content, list)
        or len(content) != 1
        or not isinstance(content[0], dict)
        or content[0].get("type") != "text"
        or not isinstance(content[0].get("text"), str)
    ):
        raise ValueError("Stage 2 assistant turn must contain exactly one text item")
    return content[0]["text"]


def _derive_stage2_text_segments(
    *,
    tokenizer,
    rendered_text: str,
    final_input_ids: torch.Tensor,
    example: Stage2PreparedExample,
    token_ids: LatentTokenIds,
    latent_size: int,
) -> list[tuple[int, int]]:
    """Locate assistant content via full-render offsets and verify its final suffix."""
    content = _assistant_content(example)
    header = "<|im_start|>assistant\n"
    needle = header + content
    matches = []
    cursor = 0
    while True:
        position = rendered_text.find(needle, cursor)
        if position < 0:
            break
        matches.append(position)
        cursor = position + 1
    if len(matches) != 1:
        raise ValueError(
            f"expected one rendered assistant content bound, found {len(matches)}"
        )
    content_character_start = matches[0] + len(header)
    content_character_end = content_character_start + len(content)
    if rendered_text[content_character_start:content_character_end] != content:
        raise ValueError("rendered assistant content bytes do not match the conversation")
    if not rendered_text[content_character_end:].startswith("<|im_end|>"):
        raise ValueError("rendered assistant content is not followed by chat termination")

    try:
        encoded = tokenizer(
            rendered_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        raw_ids = _one_dimensional_list(encoded["input_ids"], name="input_ids")
        raw_offsets = _one_dimensional_list(
            encoded["offset_mapping"], name="offset_mapping", entries_are_pairs=True
        )
    except (KeyError, TypeError, ValueError, NotImplementedError) as error:
        raise ValueError(
            "Stage 2 objective requires tokenizer offsets for the complete rendered text"
        ) from error
    offsets = [tuple(int(value) for value in pair) for pair in raw_offsets]
    if len(offsets) != len(raw_ids) or any(len(pair) != 2 for pair in offsets):
        raise ValueError("tokenizer returned inconsistent full-render offsets")
    raw_content_start = _token_boundary(offsets, content_character_start)
    raw_content_end = _token_boundary(offsets, content_character_end)
    if raw_content_end < raw_content_start:
        raise ValueError("assistant content token bounds are reversed")

    final_ids = [int(value) for value in final_input_ids.tolist()]
    assistant_suffix = [int(value) for value in raw_ids[raw_content_start:]]
    if len(assistant_suffix) > len(final_ids):
        raise ValueError("processor output is shorter than the assistant token suffix")
    final_suffix_start = len(final_ids) - len(assistant_suffix)
    if final_ids[final_suffix_start:] != assistant_suffix:
        raise ValueError(
            "processor output does not preserve the verified assistant token suffix"
        )
    shift = final_suffix_start - raw_content_start
    content_start = raw_content_start + shift
    content_end = raw_content_end + shift

    try:
        spans = parse_latent_spans(
            final_input_ids, token_ids=token_ids, latent_size=latent_size
        )
    except LatentStructureError as error:
        raise ValueError(f"malformed final Stage 2 latent markers: {error}") from error
    if len(spans) != example.latent_block_count:
        raise ValueError(
            "final latent block count does not match prepared example metadata"
        )
    if any(
        span.start_position < content_start or span.end_position >= content_end
        for span in spans
    ):
        raise ValueError("latent marker lies outside verified assistant content bounds")
    segments: list[tuple[int, int]] = []
    start = content_start
    for span in spans:
        segments.append((start, span.start_position))
        start = span.end_position + 1
    segments.append((start, content_end))
    return segments


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
