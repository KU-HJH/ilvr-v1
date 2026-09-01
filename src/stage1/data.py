"""Batch-safe CoMT preparation and collation for Stage 1 training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from PIL import Image
import torch

from stage1.types import (
    Stage1BatchMetadata,
    Stage1LatentSpan,
    Stage1SampleOccurrence,
    Stage1SampleMetadata,
)
from task_deepseed import interleaved_latent_cot_preprocess_function
from utils_deepseed import (
    normalize_stage1_latent_blocks_1d,
    place_input_image,
    place_output_image,
    remove_assistant_images,
    remove_user_images,
    replace_visual_spectial_tokens,
)


@dataclass(frozen=True)
class PreparedStage1Example:
    """A CoMT row with unique source identity and explicit image ownership."""

    source_index: int
    source_id: str | None
    conversation: list[dict]
    original_image_paths: tuple[str | Image.Image, ...]
    helper_image_paths: tuple[str | Image.Image, ...]
    occurrence: Stage1SampleOccurrence | None = None


def prepare_stage1_comt_examples(
    data_path: Path, image_root: Path
) -> list[PreparedStage1Example]:
    """Load CoMT JSONL rows using an explicit image root and stable row indices."""
    with Path(data_path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    prepared = []
    for source_index, sample in enumerate(rows):
        try:
            conversation = interleaved_latent_cot_preprocess_function(
                sample, image_root=Path(image_root)
            )
        except (FileNotFoundError, ValueError) as error:
            message = (
                f"source_index={source_index}: observed CoMT preparation failure "
                f"({error}); expected a valid sequence_plan with resolvable "
                "original and helper image paths"
            )
            if isinstance(error, FileNotFoundError):
                raise FileNotFoundError(message) from error
            raise ValueError(message) from error
        original_paths = tuple(
            item["image"]
            for turn in conversation
            if turn.get("role") == "user"
            for item in turn.get("content", [])
            if item.get("type") == "image"
        )
        helper_paths = tuple(
            item["image"]
            for turn in conversation
            if turn.get("role") == "assistant"
            for item in turn.get("content", [])
            if item.get("type") == "image"
        )
        raw_source_id = sample.get("id")
        prepared.append(
            PreparedStage1Example(
                source_index=source_index,
                source_id=None if raw_source_id is None else str(raw_source_id),
                conversation=conversation,
                original_image_paths=original_paths,
                helper_image_paths=helper_paths,
            )
        )
    return prepared


def _materialize_image(
    image_reference: object,
    *,
    source_index: int,
    role: str,
    image_index: int,
) -> Image.Image:
    try:
        if isinstance(image_reference, Image.Image):
            return image_reference.convert("RGB")
        with Image.open(Path(image_reference)) as source:
            image = source.convert("RGB")
            image.load()
        return image
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"source_index={source_index}: observed unreadable {role} image "
            f"at index {image_index} ({image_reference!s}: {error}); expected a "
            "readable RGB-compatible image"
        ) from error


def _find_subsequence_starts(sequence: torch.Tensor, pattern: torch.Tensor) -> list[int]:
    if pattern.numel() == 0:
        return []
    return [
        start
        for start in range(sequence.numel() - pattern.numel() + 1)
        if torch.equal(sequence[start : start + pattern.numel()], pattern)
    ]


class Stage1DataCollator:
    """Collate Stage 1 samples without inferring ownership across batch rows."""

    def __init__(self, processor, token_ids, latent_size):
        if (
            not isinstance(latent_size, int)
            or isinstance(latent_size, bool)
            or latent_size <= 0
        ):
            raise ValueError("latent_size must be a positive integer")
        self.processor = processor
        self.token_ids = token_ids
        self.latent_size = latent_size
        self.assistant_pattern = processor.tokenizer(
            "<|im_start|>assistant",
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]

    def __call__(
        self, examples: Sequence[PreparedStage1Example]
    ) -> dict[str, object]:
        examples = tuple(examples)
        if not examples:
            raise ValueError("Stage 1 collator requires at least one example")
        self._validate_sample_identity(examples)

        batch = self._process_full_conversations(examples)
        original = self._process_original_images(examples)
        helper = self._process_helper_images(examples)
        batch["pixel_values"] = original["pixel_values"]
        batch["image_grid_thw"] = original["image_grid_thw"]
        input_ids, attention_mask, latent_spans, unpadded_lengths = (
            self._replace_helper_tokens_with_latent_blocks(examples, batch)
        )
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        labels = self._build_labels(examples, input_ids, attention_mask)
        latent_mask = torch.zeros_like(input_ids)
        for row_index, row_spans in enumerate(latent_spans):
            for span in row_spans:
                latent_mask[row_index, list(span.latent_positions)] = 1
        batch["labels"] = labels
        batch["image_out_mask"] = latent_mask

        metadata = self._build_metadata(
            examples,
            batch,
            helper,
            latent_spans,
            unpadded_lengths,
        )
        self._validate_tensor_shapes(
            examples, metadata, input_ids, attention_mask, labels, latent_mask
        )
        return {
            **batch,
            "pixel_values_latent": helper["pixel_values"],
            "image_grid_thw_latent": helper["image_grid_thw"],
            "stage1_batch_metadata": metadata,
        }

    @staticmethod
    def _validate_sample_identity(examples: Sequence[PreparedStage1Example]) -> None:
        occurrences = tuple(example.occurrence for example in examples)
        if all(occurrence is None for occurrence in occurrences):
            Stage1DataCollator._validate_source_indices(examples)
            return
        if any(occurrence is None for occurrence in occurrences):
            raise ValueError(
                "Stage 1 collator observed a mixed ordinary/equivalence batch; "
                "expected every row to have complete occurrence identity"
            )
        seen_slots = set()
        for example, occurrence in zip(examples, occurrences):
            assert occurrence is not None
            if example.source_index != occurrence.source_index:
                raise ValueError(
                    f"source_index={example.source_index}: observed occurrence source_index "
                    f"{occurrence.source_index}; expected matching source provenance"
                )
            if occurrence.canonical_slot_id in seen_slots:
                raise ValueError(
                    "Stage 1 collator observed duplicate canonical_slot_id; "
                    "expected unique equivalence occurrences within a batch"
                )
            seen_slots.add(occurrence.canonical_slot_id)

    @staticmethod
    def _validate_source_indices(examples: Sequence[PreparedStage1Example]) -> None:
        seen = set()
        for example in examples:
            if example.source_index in seen:
                raise ValueError(
                    f"source_index={example.source_index}: observed duplicate source_index; "
                    "expected unique source_index values within a batch"
                )
            seen.add(example.source_index)

    def _render_conversations(self, conversations: Sequence[list[dict]]) -> list[str]:
        texts = []
        for conversation in conversations:
            rendered = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            rendered = place_input_image(rendered)
            rendered = place_output_image(rendered)
            texts.append(replace_visual_spectial_tokens([rendered])[0])
        return texts

    def _materialize_role_images(self, examples, role):
        attribute = f"{role}_image_paths"
        return [
            [
                _materialize_image(
                    image_reference,
                    source_index=example.source_index,
                    role=role,
                    image_index=image_index,
                )
                for image_index, image_reference in enumerate(
                    getattr(example, attribute)
                )
            ]
            for example in examples
        ]

    def _process_full_conversations(
        self, examples: Sequence[PreparedStage1Example]
    ) -> dict[str, torch.Tensor]:
        texts = self._render_conversations(
            [example.conversation for example in examples]
        )
        original_images = self._materialize_role_images(examples, "original")
        helper_images = self._materialize_role_images(examples, "helper")
        images = [
            originals + helpers
            for originals, helpers in zip(original_images, helper_images)
        ]
        batch = dict(
            self.processor(
                text=texts,
                images=images,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
        )
        self._require_processor_keys(examples[0], batch, "full")
        counts = tuple(
            len(example.original_image_paths) + len(example.helper_image_paths)
            for example in examples
        )
        self._validate_grid_rows(examples, batch["image_grid_thw"], counts, "full")
        return batch

    def _process_original_images(
        self, examples: Sequence[PreparedStage1Example]
    ) -> dict[str, torch.Tensor]:
        conversations = remove_assistant_images(
            [example.conversation for example in examples]
        )
        texts = self._render_conversations(conversations)
        images = self._materialize_role_images(examples, "original")
        batch = dict(
            self.processor(
                text=texts,
                images=images,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
        )
        self._require_processor_keys(examples[0], batch, "original")
        return batch

    def _process_helper_images(
        self, examples: Sequence[PreparedStage1Example]
    ) -> dict[str, torch.Tensor]:
        helper_count = sum(len(example.helper_image_paths) for example in examples)
        if helper_count:
            conversations = remove_user_images(
                [example.conversation for example in examples]
            )
            texts = self._render_conversations(conversations)
            images = self._materialize_role_images(examples, "helper")
            batch = dict(
                self.processor(
                    text=texts,
                    images=images,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
            )
        else:
            batch = {
                "pixel_values": torch.empty((0, 0)),
                "image_grid_thw": torch.empty((0, 3), dtype=torch.long),
            }
        self._require_processor_keys(examples[0], batch, "helper")
        return batch

    @staticmethod
    def _require_processor_keys(example, batch, kind):
        required = {"pixel_values", "image_grid_thw"}
        if kind in {"full", "original"}:
            required.update(("input_ids", "attention_mask"))
        missing = sorted(required - batch.keys())
        if missing:
            raise ValueError(
                f"source_index={example.source_index}: observed missing {kind} "
                f"processor keys {missing}; expected {sorted(required)}"
            )

    def _replace_helper_tokens_with_latent_blocks(self, examples, original):
        input_ids = original["input_ids"]
        attention_mask = original["attention_mask"]
        expected_shape = (
            (len(examples), input_ids.shape[1]) if input_ids.ndim == 2 else None
        )
        if (
            input_ids.ndim != 2
            or attention_mask.ndim != 2
            or tuple(attention_mask.shape) != tuple(input_ids.shape)
            or input_ids.shape[0] != len(examples)
        ):
            raise ValueError(
                f"source_index={examples[0].source_index}: observed input_ids shape "
                f"{tuple(input_ids.shape)} and attention_mask shape "
                f"{tuple(attention_mask.shape)}; expected shared B,T shape "
                f"{expected_shape}"
            )

        rows = []
        spans_by_row = []
        unpadded_lengths = []
        for row_index, example in enumerate(examples):
            row = input_ids[row_index][attention_mask[row_index].bool()]
            row = normalize_stage1_latent_blocks_1d(
                row,
                start_token=self.token_ids.start_token_id,
                end_token=self.token_ids.end_token_id,
                latent_pad_token=self.token_ids.pad_token_id,
                latent_size=self.latent_size,
                source_index=example.source_index,
            )
            observed_blocks = int((row == self.token_ids.start_token_id).sum().item())
            expected_blocks = len(example.helper_image_paths)
            if observed_blocks != expected_blocks:
                raise ValueError(
                    f"source_index={example.source_index}: observed {observed_blocks} "
                    f"latent blocks; expected {expected_blocks} to match helper images"
                )
            spans = self._parse_latent_spans(row, example.source_index)
            rows.append(row)
            spans_by_row.append(spans)
            unpadded_lengths.append(row.numel())

        width = max(unpadded_lengths)
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        pad_token_id = 0 if pad_token_id is None else int(pad_token_id)
        padded = input_ids.new_full((len(rows), width), pad_token_id)
        mask = attention_mask.new_zeros((len(rows), width))
        for row_index, row in enumerate(rows):
            padded[row_index, : row.numel()] = row
            mask[row_index, : row.numel()] = 1
        return padded, mask, tuple(spans_by_row), tuple(unpadded_lengths)

    def _parse_latent_spans(
        self, row: torch.Tensor, source_index: int
    ) -> tuple[Stage1LatentSpan, ...]:
        spans = []
        open_start = None
        positions = []
        for position, raw_token in enumerate(row.tolist()):
            token = int(raw_token)
            if open_start is None:
                if token == self.token_ids.start_token_id:
                    open_start = position
                    positions = []
                elif token == self.token_ids.end_token_id:
                    raise ValueError(
                        f"source_index={source_index}: observed latent_end at position "
                        f"{position} without latent_start; expected ordered complete boundaries"
                    )
                elif token == self.token_ids.pad_token_id:
                    raise ValueError(
                        f"source_index={source_index}: observed latent_pad at position "
                        f"{position} outside a latent block; expected pads only inside a block"
                    )
                continue

            if token == self.token_ids.start_token_id:
                raise ValueError(
                    f"source_index={source_index}: observed nested latent_start at "
                    f"position {position}; expected ordered, nonnested boundaries"
                )
            if token == self.token_ids.end_token_id:
                if len(positions) != self.latent_size:
                    raise ValueError(
                        f"source_index={source_index}: observed {len(positions)} latent "
                        f"slots; expected exactly {self.latent_size} latent slots in "
                        f"block {len(spans)}"
                    )
                spans.append(
                    Stage1LatentSpan(
                        start=open_start,
                        end=position,
                        latent_positions=tuple(positions),
                    )
                )
                open_start = None
                positions = []
                continue
            if token != self.token_ids.pad_token_id:
                raise ValueError(
                    f"source_index={source_index}: observed token {token} inside latent "
                    f"block; expected exactly {self.latent_size} latent slots"
                )
            positions.append(position)

        if open_start is not None:
            raise ValueError(
                f"source_index={source_index}: observed incomplete latent block starting "
                f"at position {open_start}; expected a complete latent_start/latent_end pair"
            )
        return tuple(spans)

    def _build_labels(self, examples, input_ids, attention_mask):
        labels = input_ids.clone()
        pattern = self.assistant_pattern.to(device=input_ids.device, dtype=input_ids.dtype)
        for row_index, example in enumerate(examples):
            unpadded = input_ids[row_index][attention_mask[row_index].bool()]
            matches = _find_subsequence_starts(unpadded, pattern)
            if len(matches) != 1:
                raise ValueError(
                    f"source_index={example.source_index}: observed {len(matches)} "
                    "assistant markers; expected exactly 1 assistant marker"
                )
            labels[row_index, : matches[0] + pattern.numel()] = -100
        labels.masked_fill_(attention_mask == 0, -100)
        labels.masked_fill_(input_ids == self.token_ids.pad_token_id, -100)
        return labels

    def _build_metadata(
        self,
        examples,
        original,
        helper,
        latent_spans,
        unpadded_lengths,
    ):
        original_counts = tuple(len(example.original_image_paths) for example in examples)
        helper_counts = tuple(len(example.helper_image_paths) for example in examples)
        self._validate_grid_rows(
            examples, original["image_grid_thw"], original_counts, "original"
        )
        self._validate_grid_rows(
            examples, helper["image_grid_thw"], helper_counts, "helper"
        )

        original_offset = 0
        helper_offset = 0
        samples = []
        for row_index, example in enumerate(examples):
            original_end = original_offset + original_counts[row_index]
            helper_end = helper_offset + helper_counts[row_index]
            samples.append(
                Stage1SampleMetadata(
                    source_index=example.source_index,
                    source_id=example.source_id,
                    original_grid_range=(original_offset, original_end),
                    helper_grid_range=(helper_offset, helper_end),
                    latent_spans=latent_spans[row_index],
                    helper_to_block=tuple(range(helper_counts[row_index])),
                    unpadded_length=unpadded_lengths[row_index],
                    epoch=None if example.occurrence is None else example.occurrence.epoch,
                    view_slot=None
                    if example.occurrence is None
                    else example.occurrence.view_slot,
                    permutation_offset=None
                    if example.occurrence is None
                    else example.occurrence.permutation_offset,
                    canonical_slot_id=None
                    if example.occurrence is None
                    else example.occurrence.canonical_slot_id,
                    rank=None if example.occurrence is None else example.occurrence.rank,
                    rank_local_sample_ordinal=None
                    if example.occurrence is None
                    else example.occurrence.rank_local_sample_ordinal,
                    rank_local_microbatch_ordinal=None
                    if example.occurrence is None
                    else example.occurrence.rank_local_microbatch_ordinal,
                    row_ordinal_within_microbatch=None
                    if example.occurrence is None
                    else example.occurrence.row_ordinal_within_microbatch,
                )
            )
            original_offset = original_end
            helper_offset = helper_end
        return Stage1BatchMetadata(
            samples=tuple(samples),
            batch_size=len(samples),
            sequence_length=int(original["input_ids"].shape[1]),
        )

    @staticmethod
    def _validate_grid_rows(examples, image_grid_thw, counts, kind):
        expected = sum(counts)
        observed = image_grid_thw.shape[0] if image_grid_thw.ndim >= 1 else 0
        if observed == expected:
            return
        if observed < expected:
            running = 0
            affected = examples[-1]
            for example, count in zip(examples, counts):
                running += count
                if running > observed:
                    affected = example
                    break
        else:
            affected = examples[-1]
        raise ValueError(
            f"source_index={affected.source_index}: observed {observed} flattened "
            f"{kind} image grids; expected {expected} from per-sample counts {counts}"
        )

    @staticmethod
    def _validate_tensor_shapes(
        examples, metadata, input_ids, attention_mask, labels, latent_mask
    ):
        expected = (metadata.batch_size, metadata.sequence_length)
        observed = {
            "input_ids": tuple(input_ids.shape),
            "attention_mask": tuple(attention_mask.shape),
            "labels": tuple(labels.shape),
            "latent_mask": tuple(latent_mask.shape),
        }
        if any(shape != expected for shape in observed.values()):
            raise ValueError(
                f"source_index={examples[0].source_index}: observed B,T shapes "
                f"{observed}; expected every tensor to have B,T={expected}"
            )
