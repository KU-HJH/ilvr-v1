"""CoMT data preparation, non-truncating collation, and length auditing."""

import argparse
from collections.abc import Mapping
from collections import Counter
from dataclasses import asdict, dataclass, replace
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
from stage2_objective.config import ObjectiveConfig, validate_objective_config
from stage2_objective.vocabulary import TextVocabulary
from stage2_objective.windows import align_segments, build_windows
from step_preference.negatives import (
    StepNegativeIndex,
    StepNegativeRecord,
)
from step_preference.spans import (
    ReasoningStepSpan,
    derive_reasoning_step_spans,
    stable_sample_id,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage2PreparedExample:
    """A validated CoMT sample ready for processor tokenization."""

    sample_id: str
    sequence_plan: tuple[dict, ...]
    conversation: list[dict]
    original_image_paths: tuple[str, ...]
    latent_block_count: int
    latent_size: int = 8
    original_final_answer: str | None = None
    sequence_length: int | None = None


@dataclass(frozen=True)
class StepPreferenceBatchItem:
    """One collated reasoning step and its independent preference branches."""

    sample_id: str
    span: ReasoningStepSpan
    chosen_predictor_positions: torch.LongTensor
    chosen_target_ids: torch.LongTensor
    # Legacy data helpers remain available to independent experiments only.
    negative: StepNegativeRecord | None = None
    rejected_visual_inputs: dict[str, torch.Tensor] | None = None
    rejected_visual_target_ids: torch.LongTensor | None = None
    rejected_text_inputs: dict[str, torch.Tensor] | None = None
    rejected_text_target_ids: torch.LongTensor | None = None


def immediate_text_target_indices(example: Stage2PreparedExample) -> dict[int, int]:
    """Map latent block IDs to eligible first items, using CoMT's final-item convention."""
    selected = {}
    block_id = 0
    plan = example.sequence_plan
    for index, step in enumerate(plan):
        if step["type"] != "latent":
            continue
        target_index = index + 1
        if target_index < len(plan):
            target = plan[target_index]
            is_final = (target_index == len(plan) - 1
                        and example.original_final_answer is not None
                        and target.get("content") == f"The final answer is: {example.original_final_answer}")
            if target["type"] == "text" and target["content"] and not is_final:
                selected[block_id] = target_index
        block_id += 1
    return selected


def build_rejected_branch_inputs(
    *,
    clean_model_inputs: Mapping[str, torch.Tensor],
    prefix_end: int,
    rejected_target_ids: torch.Tensor,
    assistant_end_ids: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Replace the clean suffix with one rejected target and assistant end."""
    clean_ids = clean_model_inputs["input_ids"]
    if (
        clean_ids.ndim != 2
        or clean_ids.shape[0] != 1
        or rejected_target_ids.ndim != 1
        or assistant_end_ids.ndim != 1
    ):
        raise ValueError("Stage 2 preference branches require one unbatched target")
    if not 1 <= prefix_end <= clean_ids.shape[1]:
        raise ValueError("Stage 2 preference prefix boundary is out of bounds")
    if rejected_target_ids.numel() == 0:
        raise ValueError("step preference target must contain at least one token")
    branch_ids = torch.cat(
        (
            clean_ids[:, :prefix_end],
            rejected_target_ids.to(device=clean_ids.device, dtype=clean_ids.dtype)[None],
            assistant_end_ids.to(device=clean_ids.device, dtype=clean_ids.dtype)[None],
        ),
        dim=1,
    )
    branch = {
        key: value
        for key, value in clean_model_inputs.items()
        if key not in {"input_ids", "attention_mask", "labels"}
    }
    branch["input_ids"] = branch_ids
    branch["attention_mask"] = torch.ones_like(branch_ids)
    predictor_positions = torch.arange(
        prefix_end - 1,
        prefix_end - 1 + rejected_target_ids.numel(),
        device=branch_ids.device,
    )
    return branch, predictor_positions


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
        sample_id=stable_sample_id(sample),
        sequence_plan=tuple(dict(step) for step in sample["sequence_plan"]),
        conversation=conversation,
        original_image_paths=resolve_original_images(sample, image_root),
        latent_block_count=latent_block_count,
        latent_size=latent_size,
        original_final_answer=sample.get("original_final_answer"),
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
        step_negative_index: StepNegativeIndex | None = None,
        context_preference_enabled: bool = False,
        target_scope: str = "until_next_latent",
        objective_config: ObjectiveConfig | None = None,
        objective_vocabulary: TextVocabulary | None = None,
    ):
        self.processor = processor
        self.token_ids = token_ids
        self.latent_size = latent_size
        self.supervise_latent_boundaries = supervise_latent_boundaries
        self.step_negative_index = step_negative_index
        self.context_preference_enabled = context_preference_enabled
        self.objective_config = objective_config
        self.objective_vocabulary = objective_vocabulary
        if objective_config is not None:
            validate_objective_config(objective_config)
        self.objective_enabled = bool(
            objective_config is not None and objective_config.enabled
        )
        if self.objective_enabled and objective_vocabulary is None:
            raise ValueError("enabled objective requires objective_vocabulary")
        if self.objective_enabled and (
            context_preference_enabled or step_negative_index is not None
        ):
            raise ValueError("new objective cannot be combined with legacy preference data")
        if target_scope not in {"until_next_latent", "immediate_text"}:
            raise ValueError("target_scope must be until_next_latent or immediate_text")
        if target_scope == "immediate_text" and not context_preference_enabled:
            raise ValueError("immediate_text requires same-GT context preference")
        self.target_scope = target_scope
        if context_preference_enabled and step_negative_index is not None:
            raise ValueError("context preference does not accept a negative index")
        self.assistant_pattern = processor.tokenizer(
            "<|im_start|>assistant", add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        self.assistant_end_pattern = processor.tokenizer(
            "<|im_end|>", add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]

    def __call__(self, examples: list[Stage2PreparedExample]) -> dict[str, object]:
        if len(examples) != 1:
            raise NotImplementedError("Stage 2 collator currently supports batch_size=1 only")

        example = examples[0]
        text = self.processor.apply_chat_template(
            example.conversation, tokenize=False, add_generation_prompt=False
        )
        processor_kwargs = dict(
            text=[text],
            images=[_load_original_images(example.original_image_paths)],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if self.objective_enabled:
            if getattr(self.processor.tokenizer, "is_fast", False) is not True:
                raise ValueError("enabled objective requires a fast tokenizer")
            processor_kwargs["return_offsets_mapping"] = True
        batch = self.processor(**processor_kwargs)
        offsets = batch.pop("offset_mapping", None) if self.objective_enabled else None
        if self.objective_enabled and offsets is None:
            raise ValueError("enabled objective requires processor offset_mapping")
        forbidden = {"pixel_values_latent", "image_grid_thw_latent"} & batch.keys()
        if forbidden:
            raise ValueError(f"helper-image tensors leaked into Stage 2: {sorted(forbidden)}")
        if batch["input_ids"].shape[0] != 1:
            raise NotImplementedError("Stage 2 collator currently supports batch_size=1 only")

        assistant_content_start = find_unique_subsequence_end(
            batch["input_ids"][0], self.assistant_pattern
        )
        assistant_content_end = assistant_content_start + find_unique_subsequence_end(
            batch["input_ids"][0, assistant_content_start:], self.assistant_end_pattern
        ) - self.assistant_end_pattern.numel()
        item_bounds = None
        if not self.objective_enabled and self.target_scope == "immediate_text":
            assistant_content_end, item_bounds = self._retokenize_items(
                batch, example, assistant_content_start, assistant_content_end)
        batch["labels"] = build_stage2_labels(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            assistant_content_start=assistant_content_start,
            token_ids=self.token_ids,
            latent_size=self.latent_size,
            supervise_latent_boundaries=self.supervise_latent_boundaries,
        )
        if self.objective_enabled:
            segments = align_segments(
                input_ids=batch["input_ids"],
                offsets=offsets,
                sequence_plan=example.sequence_plan,
                assistant_bounds=(assistant_content_start, assistant_content_end),
                token_ids=self.token_ids,
                sample_id=example.sample_id,
            )
            batch["stage2_objective_plan"] = build_windows(
                batch["input_ids"],
                batch["labels"],
                batch["attention_mask"],
                segments,
                self.objective_vocabulary,
                self.objective_config,
            )
            batch["sample_ids"] = [example.sample_id]
            return batch
        latent_spans = parse_latent_spans(
            batch["input_ids"][0], token_ids=self.token_ids, latent_size=self.latent_size
        )
        reasoning_step_spans = derive_reasoning_step_spans(
            sequence_plan=example.sequence_plan,
            assistant_content_start=assistant_content_start,
            assistant_content_end=assistant_content_end,
            latent_spans=latent_spans,
        )
        batch["reasoning_step_spans"] = [list(reasoning_step_spans)]
        batch["sample_ids"] = [example.sample_id]
        if self.context_preference_enabled:
            ids = batch["input_ids"]
            preference_spans = reasoning_step_spans
            if item_bounds is not None:
                preference_spans = []
                target_indices = immediate_text_target_indices(example)
                for span in reasoning_step_spans:
                    target_index = target_indices.get(span.step_id)
                    if target_index is None:
                        continue  # Remains in clean CE; never select a later item.
                    start, end = item_bounds[target_index]
                    if start != span.target_text_start or start >= end:
                        raise ValueError("immediate_text requires a text item immediately after the latent")
                    preference_spans.append(replace(span, target_text_end=end))
            batch["step_preference_items"] = [
                StepPreferenceBatchItem(
                    sample_id=example.sample_id, span=span,
                    chosen_predictor_positions=torch.arange(
                        span.target_text_start - 1, span.target_text_end - 1, device=ids.device),
                    chosen_target_ids=ids[0, span.target_text_start:span.target_text_end].clone(),
                ) for span in preference_spans
            ]
        elif self.step_negative_index is not None:
            clean_model_inputs = {
                key: value
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            batch["step_preference_items"] = [
                self._build_step_preference_item(
                    sample_id=example.sample_id,
                    span=span,
                    clean_model_inputs=clean_model_inputs,
                )
                for span in reasoning_step_spans
            ]
        return batch

    def _retokenize_items(self, batch, example, start, end):
        """Keep processor image/prompt tensors; replace only assistant body IDs.

        Each data item is encoded independently, so a token cannot straddle
        the reasoning/final-answer boundary. No separators or tokens are added.
        """
        tokenizer = self.processor.tokenizer
        body, _ = _assistant_text(example.sequence_plan, self.latent_size)
        old = tokenizer.decode(batch["input_ids"][0, start:end],
                               skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not body or not old.endswith(body):
            raise ValueError("immediate_text cannot align the rendered assistant with sequence_plan")

        def encode(text):
            return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(
                device=batch["input_ids"].device, dtype=batch["input_ids"].dtype)

        parts = [encode(old[:len(old) - len(body)])]  # Preserve the template's leading newline.
        cursor = start + parts[0].numel()
        bounds = []
        for step in example.sequence_plan:
            text, _ = _assistant_text([step], self.latent_size)
            ids = encode(text)
            bounds.append((cursor, cursor + ids.numel()))
            cursor += ids.numel()
            parts.append(ids)
        replacement = torch.cat(parts)[None]
        original_ids = batch["input_ids"]
        batch["input_ids"] = torch.cat((original_ids[:, :start], replacement, original_ids[:, end:]), dim=1)
        mask = batch["attention_mask"]
        batch["attention_mask"] = torch.cat((mask[:, :start], mask.new_ones(replacement.shape), mask[:, end:]), dim=1)
        if "token_type_ids" in batch:
            types = batch["token_type_ids"]
            if not bool((types[:, start:end] == types[:, start:start + 1]).all()):
                raise ValueError("immediate_text requires a uniform assistant token type")
            batch["token_type_ids"] = torch.cat((types[:, :start], types[:, start:start + 1].expand_as(replacement), types[:, end:]), dim=1)
        return cursor, bounds

    @staticmethod
    def _stored_preference_target(
        token_ids: tuple[int, ...], *, expected_count: int, device: torch.device
    ) -> torch.LongTensor:
        target_ids = torch.tensor(token_ids, device=device, dtype=torch.long)
        if target_ids.numel() == 0:
            raise ValueError("step preference target must contain at least one token")
        if target_ids.numel() != expected_count:
            raise ValueError(
                "step preference target token count disagrees with negative artifact"
            )
        return target_ids

    def _build_step_preference_item(
        self,
        *,
        sample_id: str,
        span: ReasoningStepSpan,
        clean_model_inputs: Mapping[str, torch.Tensor],
    ) -> StepPreferenceBatchItem:
        negative = self.step_negative_index.get(sample_id, span.step_id)
        clean_ids = clean_model_inputs["input_ids"]
        chosen_target_ids = clean_ids[0, span.target_text_start : span.target_text_end]
        if chosen_target_ids.numel() == 0:
            raise ValueError("step preference chosen target must not be empty")
        if chosen_target_ids.numel() != negative.chosen_token_count:
            raise ValueError(
                "chosen target token count disagrees with negative artifact"
            )
        chosen_text = self.processor.tokenizer.decode(
            chosen_target_ids.detach().cpu(),
            skip_special_tokens=True,
        ).strip()
        if chosen_text != negative.chosen_text:
            raise ValueError(
                "clean chosen text disagrees with negative artifact chosen_text"
            )
        chosen_predictor_positions = torch.arange(
            span.target_text_start - 1,
            span.target_text_end - 1,
            device=clean_ids.device,
        )

        def rejected_branch(
            *, token_ids: tuple[int, ...], expected_count: int, valid: bool
        ) -> tuple[dict[str, torch.Tensor] | None, torch.LongTensor | None]:
            if not valid:
                return None, None
            target_ids = self._stored_preference_target(
                token_ids,
                expected_count=expected_count,
                device=clean_ids.device,
            )
            branch, predictor_positions = build_rejected_branch_inputs(
                clean_model_inputs=clean_model_inputs,
                prefix_end=span.target_text_start,
                rejected_target_ids=target_ids,
                assistant_end_ids=self.assistant_end_pattern,
            )
            if predictor_positions.numel() != target_ids.numel():
                raise ValueError("rejected preference predictor count is inconsistent")
            return branch, target_ids

        rejected_visual_inputs, rejected_visual_target_ids = rejected_branch(
            token_ids=negative.rejected_visual_token_ids,
            expected_count=negative.rejected_visual_token_count,
            valid=negative.valid_visual_negative,
        )
        rejected_text_inputs, rejected_text_target_ids = rejected_branch(
            token_ids=negative.rejected_textual_token_ids,
            expected_count=negative.rejected_textual_token_count,
            valid=negative.valid_textual_negative,
        )
        return StepPreferenceBatchItem(
            sample_id=sample_id,
            span=span,
            negative=negative,
            chosen_predictor_positions=chosen_predictor_positions,
            chosen_target_ids=chosen_target_ids,
            rejected_visual_inputs=rejected_visual_inputs,
            rejected_visual_target_ids=rejected_visual_target_ids,
            rejected_text_inputs=rejected_text_inputs,
            rejected_text_target_ids=rejected_text_target_ids,
        )


def build_prepared_example_length_measure(processor, *, target_scope="until_next_latent"):
    """Build a length measurement closure that never truncates a sample."""
    if target_scope not in {"until_next_latent", "immediate_text"}:
        raise ValueError("target_scope must be until_next_latent or immediate_text")

    def measure(example: Stage2PreparedExample) -> int:
        if target_scope == "immediate_text":
            tokenizer = processor.tokenizer
            token_ids = LatentTokenIds(
                pad_token_id=tokenizer.convert_tokens_to_ids("<|latent_pad|>"),
                start_token_id=tokenizer.convert_tokens_to_ids("<|latent_start|>"),
                end_token_id=tokenizer.convert_tokens_to_ids("<|latent_end|>"))
            collator = Stage2DataCollator(processor, token_ids, example.latent_size, True,
                                         context_preference_enabled=True, target_scope=target_scope)
            return int(collator([example])["input_ids"].shape[1])
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
            kept.append(
                replace(example, sequence_length=length)
                if isinstance(example, Stage2PreparedExample)
                else example
            )

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
