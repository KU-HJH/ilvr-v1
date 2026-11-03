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
from step_preference.spans import (
    ASSISTANT_PLAN_RENDERING_CONCATENATED,
    PlanCharSpan,
    ReasoningStepPlan,
    build_assistant_plan,
    canonical_sample_id,
    expand_qwen_image_placeholders,
    map_reasoning_step_spans,
    normalize_preference_target_scope,
)
from step_preference.negatives import StepNegative, tokenize_step_completion


logger = logging.getLogger(__name__)


class _ImmutableDict(dict):
    """A JSON-object-compatible dictionary that rejects ordinary mutation."""

    def _reject_mutation(self, *args, **kwargs):
        raise TypeError("immutable dictionary cannot be modified")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __ior__ = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    setdefault = _reject_mutation
    update = _reject_mutation

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (dict(self),)


@dataclass(frozen=True)
class Stage2PreparedExample:
    """A validated CoMT sample ready for processor tokenization."""

    conversation: list[dict]
    original_image_paths: tuple[str, ...]
    latent_block_count: int
    sample_id: str = ""
    sequence_plan: tuple[_ImmutableDict, ...] = ()
    plan_char_spans: tuple[PlanCharSpan, ...] = ()
    reasoning_steps: tuple[ReasoningStepPlan, ...] = ()


def parse_jsonl_examples(contents: str) -> list[dict]:
    """Parse non-empty JSON Lines records from one text snapshot."""
    return [json.loads(line) for line in contents.splitlines() if line.strip()]


def load_jsonl_examples(path: Path) -> list[dict]:
    """Load non-empty JSON Lines records without altering their contents."""
    return parse_jsonl_examples(path.read_text(encoding="utf-8"))


def select_packed_capacity_example(prepared, authenticated_admission, sample_id):
    """Select privately only after the full retained cohort authenticates."""
    from step_preference.capacity_evidence import select_capacity_sample
    return select_capacity_sample(prepared, authenticated_admission, sample_id)


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


def _assistant_text(
    sequence_plan: list,
    latent_size: int,
    *,
    assistant_plan_rendering: str = ASSISTANT_PLAN_RENDERING_CONCATENATED,
) -> tuple[str, int]:
    assistant_text, plan_char_spans, _ = build_assistant_plan(
        sequence_plan,
        latent_size,
        require_reasoning_units=False,
        assistant_plan_rendering=assistant_plan_rendering,
    )
    return assistant_text, sum(span.step_type == "latent" for span in plan_char_spans)


def build_stage2_conversation(
    sample: dict,
    *,
    image_root: Path,
    latent_size: int,
    assistant_plan_rendering: str = ASSISTANT_PLAN_RENDERING_CONCATENATED,
) -> list[dict]:
    """Convert a raw CoMT sample into a multimodal user/assistant conversation."""
    sequence_plan = sample.get("sequence_plan")
    if not isinstance(sequence_plan, list):
        raise ValueError("sequence_plan must be a list")

    original_images = resolve_original_images(sample, image_root)
    assistant_text, _ = _assistant_text(
        sequence_plan,
        latent_size,
        assistant_plan_rendering=assistant_plan_rendering,
    )
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
    require_reasoning_units: bool = False,
    assistant_plan_rendering: str = ASSISTANT_PLAN_RENDERING_CONCATENATED,
) -> Stage2PreparedExample:
    """Build a conversation together with validated original-image metadata."""
    conversation = build_stage2_conversation(
        sample,
        image_root=image_root,
        latent_size=latent_size,
        assistant_plan_rendering=assistant_plan_rendering,
    )
    _, plan_char_spans, reasoning_steps = build_assistant_plan(
        sample["sequence_plan"],
        latent_size,
        require_reasoning_units=require_reasoning_units,
        assistant_plan_rendering=assistant_plan_rendering,
    )
    return Stage2PreparedExample(
        conversation=conversation,
        original_image_paths=resolve_original_images(sample, image_root),
        latent_block_count=sum(span.step_type == "latent" for span in plan_char_spans),
        sample_id=canonical_sample_id(sample),
        sequence_plan=tuple(_ImmutableDict(step) for step in sample["sequence_plan"]),
        plan_char_spans=plan_char_spans,
        reasoning_steps=reasoning_steps,
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
        step_preference_enabled=False,
        negative_store=None,
        packed_admission_records_by_sample_id=None,
        max_packed_sequence_length=4608,
        preference_target_scope="immediate_text",
    ):
        self.preference_target_scope = normalize_preference_target_scope(preference_target_scope)
        if self.preference_target_scope != "immediate_text" and (
            not step_preference_enabled or negative_store is not None
        ):
            raise ValueError("full target scope requires context-preference without negative artifacts")
        self.processor = processor
        self.token_ids = token_ids
        self.latent_size = latent_size
        self.supervise_latent_boundaries = supervise_latent_boundaries
        self.step_preference_enabled = step_preference_enabled
        if negative_store is not None and not step_preference_enabled:
            raise ValueError("a negative store requires step preference metadata")
        self.negative_store = negative_store
        self.packed_admission_records_by_sample_id = (
            _ImmutableDict(packed_admission_records_by_sample_id)
            if packed_admission_records_by_sample_id is not None else None
        )
        self.max_packed_sequence_length = max_packed_sequence_length
        if packed_admission_records_by_sample_id is not None:
            if not step_preference_enabled or negative_store is None:
                raise ValueError("packed admission requires joined step preference negatives")
            if type(max_packed_sequence_length) is not int or max_packed_sequence_length != 4608:
                raise ValueError("packed admission cap must match 4608")
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
        original_images = _load_original_images(example.original_image_paths)
        batch = self.processor(
            text=[text],
            images=[original_images],
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
        if self.step_preference_enabled:
            tokenizer = self.processor.tokenizer
            if not getattr(tokenizer, "is_fast", False):
                raise ValueError("step preference span mapping requires a fast tokenizer")
            expanded_text = expand_qwen_image_placeholders(
                text,
                image_grid_thw=batch["image_grid_thw"],
                image_token=self.processor.image_token,
                merge_size=self.processor.image_processor.merge_size,
            )
            offset_batch = tokenizer(
                [expanded_text],
                padding=True,
                truncation=False,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            offset_mapping = offset_batch.pop("offset_mapping")
            for key, value in offset_batch.items():
                if key not in batch or not torch.equal(value, batch[key]):
                    raise ValueError(
                        f"offset tokenization does not match processor tensor: {key}"
                    )

            assistant_text = example.conversation[1]["content"][0]["text"]
            reasoning_step_spans = map_reasoning_step_spans(
                rendered_text=expanded_text,
                assistant_text=assistant_text,
                plan_char_spans=example.plan_char_spans,
                reasoning_steps=example.reasoning_steps,
                offset_mapping=offset_mapping,
                input_ids=batch["input_ids"],
                latent_pad_token_id=self.token_ids.pad_token_id,
                preference_target_scope=self.preference_target_scope,
            )
            batch["sample_ids"] = (example.sample_id,)
            batch["reasoning_step_spans"] = reasoning_step_spans
            batch["step_preference"] = {
                "expanded_rendered_text": expanded_text,
                "offset_mapping": offset_mapping,
                "steps": tuple({"span": span} for span in reasoning_step_spans),
            }
            if self.negative_store is not None:
                assistant_text = example.conversation[1]["content"][0]["text"]
                joined_steps = []
                for plan_step, token_span in zip(
                    example.reasoning_steps, reasoning_step_spans
                ):
                    key = (example.sample_id, plan_step.step_id)
                    try:
                        negative = self.negative_store[key]
                    except KeyError as error:
                        raise ValueError(f"missing negative join key: {key}") from error
                    if not isinstance(negative, StepNegative):
                        raise ValueError(f"invalid negative row for join key: {key}")
                    chosen_text = assistant_text[
                        plan_step.target_text.start : plan_step.target_text.end
                    ]
                    if negative.chosen_text != chosen_text:
                        raise ValueError(
                            f"negative chosen text does not match source step: {key}"
                        )
                    assistant_prefix = assistant_text[: plan_step.target_text.start]

                    def encode_rejected(text_value):
                        encoded = tokenize_step_completion(
                            self.processor,
                            conversation=example.conversation,
                            images=original_images,
                            assistant_prefix=assistant_prefix,
                            completion_text=text_value,
                        )
                        return {
                            "model_inputs": dict(encoded.model_inputs),
                            "completion_token_positions": encoded.completion_token_positions,
                            "completion_token_ids": encoded.completion_token_ids,
                            "rendered_text": encoded.rendered_text,
                        }

                    joined_steps.append(
                        {
                            "span": token_span,
                            "negative": negative,
                            "visual_rejected": (
                                encode_rejected(negative.rejected_visual_text)
                                if negative.valid_visual_negative
                                else None
                            ),
                            "textual_rejected": (
                                encode_rejected(negative.rejected_textual_text)
                                if negative.valid_textual_negative
                                else None
                            ),
                        }
                    )
                batch["step_preference"]["steps"] = tuple(joined_steps)
        if self.packed_admission_records_by_sample_id is not None:
            from step_preference.packed_admission import PackedAdmissionRecord
            from step_preference.packing import build_packed_tree_plan

            record = self.packed_admission_records_by_sample_id.get(example.sample_id)
            if (not isinstance(record, PackedAdmissionRecord)
                    or record.sample_id != example.sample_id or record.admission_reason != "retained"):
                raise ValueError("packed admission has no retained record for sample ID")
            plan = build_packed_tree_plan(
                clean_input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                reasoning_spans=batch["reasoning_step_spans"],
                joined_steps=batch["step_preference"]["steps"],
            )
            if (plan.unpadded_length != record.packed_unpadded_length
                    or plan.padded_length != record.packed_padded_length
                    or self.max_packed_sequence_length != 4608
                    or plan.padded_length > self.max_packed_sequence_length):
                raise ValueError("packed admission natural plan lengths or cap do not match")
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "latent_block_distribution",
            _ImmutableDict(self.latent_block_distribution),
        )


def summarize_stage2_lengths(*, lengths, latent_block_counts, max_length):
    """Summarize one already-measured sequence length per prepared example."""
    lengths = tuple(lengths)
    latent_block_counts = tuple(latent_block_counts)
    if not lengths:
        raise ValueError("cannot compute Stage 2 length statistics for an empty dataset")
    if len(lengths) != len(latent_block_counts):
        raise ValueError("lengths and latent block counts must have the same cardinality")

    values = torch.tensor(lengths, dtype=torch.float64)
    retained = sum(length <= max_length for length in lengths)
    filtered = len(lengths) - retained
    return Stage2LengthStats(
        total=len(lengths),
        retained=retained,
        filtered=filtered,
        filtered_percentage=100.0 * filtered / len(lengths),
        minimum=min(lengths),
        median=float(values.quantile(0.50)),
        p90=float(values.quantile(0.90)),
        p95=float(values.quantile(0.95)),
        p99=float(values.quantile(0.99)),
        maximum=max(lengths),
        latent_block_distribution=dict(sorted(Counter(latent_block_counts).items())),
    )


def filter_stage2_examples(*, examples, measure_length, max_length):
    """Keep complete samples at or under ``max_length`` and summarize all samples."""
    kept = []
    lengths: list[int] = []
    latent_block_counts: list[int] = []
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
        latent_block_counts.append(block_count)
        if length <= max_length:
            kept.append(example)

    stats = summarize_stage2_lengths(
        lengths=lengths,
        latent_block_counts=latent_block_counts,
        max_length=max_length,
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
