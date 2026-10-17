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


def _collate_processed_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Right-pad row tokens and concatenate row-owned vision tensors in order."""
    if not rows:
        raise ValueError("Stage 1 packing requires at least one processed row")

    token_keys = ("input_ids", "attention_mask", "labels", "image_out_mask")
    token_rows = {
        key: [row[key] for row in rows]
        for key in token_keys
    }
    if any(
        not all(isinstance(value, torch.Tensor) and value.ndim == 1 for value in values)
        for values in token_rows.values()
    ):
        raise ValueError("Stage 1 packing observed non-vector row tokens; expected one unpadded vector per row")
    pad_token_ids = [row["pad_token_id"] for row in rows]
    if not all(isinstance(value, int) for value in pad_token_ids) or len(set(pad_token_ids)) != 1:
        raise ValueError("Stage 1 packing observed inconsistent pad token IDs across rows")

    width = max(value.numel() for value in token_rows["input_ids"])
    input_rows = token_rows["input_ids"]
    input_ids = input_rows[0].new_full((len(rows), width), pad_token_ids[0])
    attention_mask = token_rows["attention_mask"][0].new_zeros((len(rows), width))
    labels = token_rows["labels"][0].new_full((len(rows), width), -100)
    image_out_mask = token_rows["image_out_mask"][0].new_zeros((len(rows), width))
    for row_index in range(len(rows)):
        row_width = input_rows[row_index].numel()
        input_ids[row_index, :row_width] = input_rows[row_index]
        attention_mask[row_index, :row_width] = token_rows["attention_mask"][row_index]
        labels[row_index, :row_width] = token_rows["labels"][row_index]
        image_out_mask[row_index, :row_width] = token_rows["image_out_mask"][row_index]

    def concatenate(key: str) -> torch.Tensor:
        values = [row[key] for row in rows]
        if not all(isinstance(value, torch.Tensor) and value.ndim >= 1 for value in values):
            raise ValueError(f"Stage 1 packing observed invalid {key}; expected tensors with a row dimension")
        nonempty = [value for value in values if value.shape[0]]
        return torch.cat(nonempty, dim=0) if nonempty else values[0]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_out_mask": image_out_mask,
        "pixel_values": concatenate("pixel_values"),
        "image_grid_thw": concatenate("image_grid_thw"),
        "pixel_values_latent": concatenate("pixel_values_latent"),
        "image_grid_thw_latent": concatenate("image_grid_thw_latent"),
    }


class Stage1DataCollator:
    """Collate Stage 1 samples without inferring ownership across batch rows."""

    def __init__(self, processor, token_ids, latent_size, stage1_variant: str = 'standard'):
        if (
            not isinstance(latent_size, int)
            or isinstance(latent_size, bool)
            or latent_size <= 0
        ):
            raise ValueError("latent_size must be a positive integer")
        if stage1_variant not in ('standard', 'v5'):
            raise ValueError('unsupported Stage 1 variant')
        self.stage1_variant = stage1_variant
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

        processed_rows = []
        latent_spans = []
        unpadded_lengths = []
        original_pixel_counts = []
        helper_pixel_counts = []
        original_grid_counts = []
        helper_grid_counts = []
        for example in examples:
            singleton = (example,)
            full = self._process_full_conversations(singleton)
            if self.stage1_variant == 'v5':
                original, helper = self._split_full_vision(singleton, full)
            else:
                original = self._process_original_images(singleton)
                helper = self._process_helper_images(singleton)
            self._validate_grid_rows(
                singleton,
                original["image_grid_thw"],
                (len(example.original_image_paths),),
                "original",
            )
            self._validate_grid_rows(
                singleton,
                helper["image_grid_thw"],
                (len(example.helper_image_paths),),
                "helper",
            )
            self._validate_pixel_grid_cardinality(singleton, original, "original")
            self._validate_pixel_grid_cardinality(singleton, helper, "helper")
            input_ids, attention_mask, row_spans, row_lengths = (
                self._replace_helper_tokens_with_latent_blocks(singleton, full)
            )
            labels = self._build_labels(singleton, input_ids, attention_mask)
            latent_mask = torch.zeros_like(input_ids)
            for span in row_spans[0]:
                latent_mask[0, list(span.latent_positions)] = 1
            processed_rows.append(
                {
                    "input_ids": input_ids[0],
                    "attention_mask": attention_mask[0],
                    "labels": labels[0],
                    "image_out_mask": latent_mask[0],
                    "pad_token_id": self._processor_pad_token_id(),
                    "pixel_values": original["pixel_values"],
                    "image_grid_thw": original["image_grid_thw"],
                    "pixel_values_latent": helper["pixel_values"],
                    "image_grid_thw_latent": helper["image_grid_thw"],
                }
            )
            latent_spans.append(row_spans[0])
            unpadded_lengths.append(row_lengths[0])
            original_pixel_counts.append(self._tensor_row_count(example, original, "original"))
            helper_pixel_counts.append(self._tensor_row_count(example, helper, "helper"))
            original_grid_counts.append(self._grid_row_count(example, original, "original"))
            helper_grid_counts.append(self._grid_row_count(example, helper, "helper"))

        batch = _collate_processed_rows(processed_rows)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        latent_mask = batch["image_out_mask"]

        metadata = self._build_metadata(
            examples,
            batch,
            tuple(latent_spans),
            tuple(unpadded_lengths),
            tuple(original_pixel_counts),
            tuple(helper_pixel_counts),
            tuple(original_grid_counts),
            tuple(helper_grid_counts),
        )
        self._validate_tensor_shapes(
            examples, metadata, input_ids, attention_mask, labels, latent_mask
        )
        self._validate_samplewise_ownership(examples, metadata, attention_mask)
        result = {**batch, "stage1_batch_metadata": metadata}
        if self.stage1_variant == 'v5':
            from stage1.v5_metadata import build_v5_metadata
            result['stage1_v5_metadata'] = build_v5_metadata(
                self.processor, examples, batch, metadata
            )
        return result

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

    def _split_full_vision(self, examples, full):
        """Reuse singleton Qwen raw patches in original-then-helper image order."""
        self._validate_pixel_grid_cardinality(examples, full, "full")
        [example] = examples
        original_count = len(example.original_image_paths)
        grids = full["image_grid_thw"]
        pixels = full["pixel_values"]
        original_grids = grids[:original_count]
        # Pixel rows precede spatial merging: each image owns grid_thw.prod rows.
        original_rows = int(original_grids.prod(dim=1).sum().item())
        original = {
            "image_grid_thw": original_grids,
            "pixel_values": pixels[:original_rows],
        }
        helper = {
            "image_grid_thw": grids[original_count:],
            "pixel_values": pixels[original_rows:]
            if example.helper_image_paths else torch.empty((0, 0)),
        }
        return original, helper

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
        conversations = remove_user_images([example.conversation for example in examples])
        texts = self._render_conversations(conversations)
        images = self._materialize_role_images(examples, "helper")
        processor_images = images if any(images) else None
        batch = dict(
            self.processor(
                text=texts,
                images=processor_images,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
        )
        if processor_images is None:
            if batch.get("pixel_values") is None:
                batch["pixel_values"] = torch.empty((0, 0))
            if batch.get("image_grid_thw") is None:
                batch["image_grid_thw"] = torch.empty((0, 3), dtype=torch.long)
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
        pad_token_id = self._processor_pad_token_id()
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
        latent_spans,
        unpadded_lengths,
        original_pixel_counts,
        helper_pixel_counts,
        original_grid_counts,
        helper_grid_counts,
    ):
        original_offset = 0
        helper_offset = 0
        original_pixel_offset = 0
        helper_pixel_offset = 0
        samples = []
        for row_index, example in enumerate(examples):
            original_end = original_offset + original_grid_counts[row_index]
            helper_end = helper_offset + helper_grid_counts[row_index]
            original_pixel_end = original_pixel_offset + original_pixel_counts[row_index]
            helper_pixel_end = helper_pixel_offset + helper_pixel_counts[row_index]
            samples.append(
                Stage1SampleMetadata(
                    source_index=example.source_index,
                    source_id=example.source_id,
                    original_grid_range=(original_offset, original_end),
                    helper_grid_range=(helper_offset, helper_end),
                    latent_spans=latent_spans[row_index],
                    helper_to_block=tuple(range(helper_grid_counts[row_index])),
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
                    original_pixel_range=(original_pixel_offset, original_pixel_end),
                    helper_pixel_range=(helper_pixel_offset, helper_pixel_end),
                )
            )
            original_offset = original_end
            helper_offset = helper_end
            original_pixel_offset = original_pixel_end
            helper_pixel_offset = helper_pixel_end
        return Stage1BatchMetadata(
            samples=tuple(samples),
            batch_size=len(samples),
            sequence_length=int(original["input_ids"].shape[1]),
        )

    @staticmethod
    def _tensor_row_count(example, batch, kind):
        values = batch["pixel_values"]
        if not isinstance(values, torch.Tensor) or values.ndim < 1:
            raise ValueError(
                f"source_index={example.source_index}: observed {kind} pixel_values "
                f"with shape {getattr(values, 'shape', None)}; expected a tensor with a row dimension"
            )
        return int(values.shape[0])

    @staticmethod
    def _grid_row_count(example, batch, kind):
        values = batch["image_grid_thw"]
        if not isinstance(values, torch.Tensor) or values.ndim < 1:
            raise ValueError(
                f"source_index={example.source_index}: observed {kind} image_grid_thw "
                f"with shape {getattr(values, 'shape', None)}; expected a tensor with a row dimension"
            )
        return int(values.shape[0])

    def _processor_pad_token_id(self) -> int:
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        return 0 if pad_token_id is None else int(pad_token_id)

    @staticmethod
    def _validate_pixel_grid_cardinality(examples, batch, kind):
        [example] = examples
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim < 1:
            raise ValueError(
                f"source_index={example.source_index}: observed {kind} pixel_values "
                f"with shape {getattr(pixel_values, 'shape', None)}; expected a tensor "
                "with a row dimension"
            )
        if (
            not isinstance(image_grid_thw, torch.Tensor)
            or image_grid_thw.ndim != 2
            or image_grid_thw.shape[1] != 3
        ):
            raise ValueError(
                f"source_index={example.source_index}: observed {kind} image_grid_thw "
                f"with shape {getattr(image_grid_thw, 'shape', None)}; expected [N, 3]"
            )
        expected = int(image_grid_thw.prod(dim=1).sum().item())
        observed = int(pixel_values.shape[0])
        if observed != expected:
            raise ValueError(
                f"source_index={example.source_index}: observed {observed} {kind} "
                f"pixel_values rows; expected {expected} from singleton image_grid_thw"
            )

    @staticmethod
    def _validate_samplewise_ownership(examples, metadata, attention_mask):
        for row_index, (example, sample) in enumerate(zip(examples, metadata.samples)):
            observed_length = int(attention_mask[row_index].sum().item())
            if observed_length != sample.unpadded_length:
                raise ValueError(
                    f"source_index={example.source_index}: observed attention length "
                    f"{observed_length}; expected unpadded length {sample.unpadded_length}"
                )
            for block_index, span in enumerate(sample.latent_spans):
                if not (
                    0 <= span.start < span.end < sample.unpadded_length
                    and all(span.start < position < span.end for position in span.latent_positions)
                ):
                    raise ValueError(
                        f"source_index={example.source_index}: observed latent span "
                        f"{block_index} outside the unpadded prefix; expected every "
                        "latent boundary and position to be row-local"
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
