"""Describe auxiliary text in existing collator coordinates without changing input."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import TYPE_CHECKING, Mapping, Sequence

import torch

if TYPE_CHECKING:
    from .data import PreparedStage1Example
    from .v5_config import Stage1V5Config
    from .types import Stage1BatchMetadata


@dataclass(frozen=True)
class V5Block:
    batch_index: int
    source_item_index: int
    start: int
    end: int
    target_positions: tuple[int, ...]
    target_ids: tuple[int, ...]
    candidate_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in (
                self.batch_index, self.source_item_index, self.start, self.end)):
            raise ValueError("v5 block identity must be nonnegative integers")
        if self.start >= self.end:
            raise ValueError("v5 block requires ordered boundaries")
        if not isinstance(self.target_ids, tuple):
            raise ValueError("v5 target IDs must be an immutable tuple")
        if len(self.target_ids) != len(self.target_positions):
            raise ValueError("v5 target IDs and positions must align")
        if any(type(t) is not int or t < 0 for t in self.target_ids):
            raise ValueError("v5 target IDs must be nonnegative integers")
        for positions in (self.target_positions, self.candidate_positions):
            if (not isinstance(positions, tuple)
                    or any(type(p) is not int or p < 0 for p in positions)
                    or tuple(sorted(set(positions))) != positions):
                raise ValueError("v5 positions must be sorted unique integer tuples")
        if any(t <= self.end for t in self.target_positions):
            raise ValueError("v5 targets must follow END")
        if any(p >= self.start for p in self.candidate_positions):
            raise ValueError("v5 candidates must precede START")

    @property
    def query_positions(self) -> tuple[int, ...]:
        return tuple(t - 1 for t in self.target_positions)


@dataclass(frozen=True)
class V5BatchMetadata:
    blocks: tuple[V5Block, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, tuple) or any(not isinstance(b, V5Block) for b in self.blocks):
            raise ValueError("v5 blocks must be an immutable tuple of V5Block")
        keys = [(b.batch_index, q) for b in self.blocks for q in b.query_positions]
        if len(set(keys)) != len(keys):
            raise ValueError("v5 target queries must have unique block ownership")

    @property
    def query_keys(self) -> tuple[tuple[int, int], ...]:
        return tuple((b.batch_index, q) for b in self.blocks for q in b.query_positions)

    def selected_token_count(self, config: Stage1V5Config) -> int:
        """Count the deterministic cardinality of the sample-local training mask.

        Clean ranking changes membership, never cardinality. Match the selector's
        decimal rational exactly, rounding each sample before any window sum.
        """
        from .v5_config import validate_v5_training_config
        validate_v5_training_config(config)
        if config.selection == 'all':
            return len(self.query_keys)
        counts = {}
        for block in self.blocks:
            counts[block.batch_index] = counts.get(block.batch_index, 0) + len(block.target_ids)
        fraction = Fraction(str(config.preference_entropy_top_percent)) / 100
        return sum(math.ceil(count * fraction) for count in counts.values())

    @property
    def original_step_count(self) -> int:
        return len(self.blocks)

    @property
    def target_ids(self) -> tuple[int, ...]:
        return tuple(token for b in self.blocks for token in b.target_ids)


def _source_blocks(processor, example, sample, batch, batch_index):
    if (sample.source_index, sample.source_id) != (example.source_index, example.source_id):
        raise ValueError("metadata/source identity mismatch")
    tokenizer = processor.tokenizer
    ids = batch["input_ids"][batch_index]
    labels = batch["labels"][batch_index]
    mask = batch["attention_mask"][batch_index]
    length = sample.unpadded_length
    if (type(length) is not int or not 0 < length <= ids.numel()
            or not bool(mask[:length].eq(1).all()) or bool(mask[length:].ne(0).any())):
        raise ValueError("unpadded source length disagrees with collator attention mask")
    row = ids[:length]
    assistants = [turn for turn in example.conversation if turn.get("role") == "assistant"]
    if len(assistants) != 1:
        raise ValueError("v5 source mapping requires exactly one assistant trajectory")
    content = assistants[0]["content"]
    spans = sample.latent_spans
    pieces, bounds, image_items = [], {}, []
    cursor = 0
    for item_index, item in enumerate(content):
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            piece = item["text"]
            bounds[item_index] = (cursor, cursor + len(piece))
        elif item.get("type") == "image":
            if len(image_items) >= len(spans):
                raise ValueError("source helper image/latent block count mismatch")
            span = spans[len(image_items)]
            piece = "<|latent_start|>" + "<|latent_pad|>" * len(span.latent_positions) + "<|latent_end|>"
            image_items.append((item_index, cursor, span))
        else:
            raise ValueError("unsupported assistant source item")
        pieces.append(piece)
        cursor += len(piece)
    if len(image_items) != len(spans):
        raise ValueError("source helper image/latent block count mismatch")
    if not spans:
        return ()

    text = tokenizer.decode(row.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    if not torch.equal(encoded["input_ids"][0].to(row.device), row):
        raise ValueError("text offset roundtrip changed original token IDs")
    offsets_tensor = encoded["offset_mapping"][0]
    if offsets_tensor.shape != (length, 2):
        raise ValueError("tokenizer offsets do not cover original tokens")
    offsets = offsets_tensor.tolist()
    if any(not 0 <= a <= b <= len(text) for a, b in offsets):
        raise ValueError("tokenizer offsets outside decoded source")
    # The first latent's known token position anchors the entire source sequence,
    # including repeated text. Every item's bytes are then checked together.
    translation = offsets[spans[0].start][0] - image_items[0][1]
    marker = "<|im_start|>assistant\n"
    if translation < len(marker) or not text[:translation].endswith(marker):
        raise ValueError("source text is not anchored to the assistant boundary")
    body = "".join(pieces)
    if text[translation:translation + len(body)] != body:
        raise ValueError("source item mapping mismatch or truncated assistant text")
    for _, offset, span in image_items:
        expected = translation + offset
        if (not 0 <= span.start < span.end < length
                or offsets[span.start] != [expected, expected + len("<|latent_start|>")]
                or tokenizer.decode([int(row[span.end])]) != "<|latent_end|>"):
            raise ValueError("source latent boundary does not match original token positions")

    special = set(tokenizer.all_special_ids)
    eligible = [p for p, token in enumerate(row.tolist())
                if token not in special and int(labels[p]) != -100]
    item_positions = {
        item_index: tuple(p for p in eligible
            if translation + a <= offsets[p][0] < offsets[p][1] <= translation + b)
        for item_index, (a, b) in bounds.items()
    }
    reasoning = sorted(p for positions in item_positions.values() for p in positions)
    blocks = []
    for item_index, _, span in image_items:
        # Every original block counts, including an empty/non-text immediate item.
        targets = item_positions.get(item_index + 1, ())
        blocks.append(V5Block(batch_index, item_index + 1, span.start, span.end,
            targets, tuple(int(row[p]) for p in targets),
            tuple(p for p in reasoning if p < span.start)))
    return tuple(blocks)


def build_v5_metadata(
    processor: object, examples: Sequence[PreparedStage1Example],
    batch: Mapping[str, object], metadata: Stage1BatchMetadata,
) -> V5BatchMetadata:
    if len(examples) != metadata.batch_size or len(metadata.samples) != len(examples):
        raise ValueError("v5 source/metadata batch size mismatch")
    shape = (metadata.batch_size, metadata.sequence_length)
    if any(not isinstance(batch.get(key), torch.Tensor) or tuple(batch[key].shape) != shape
           for key in ("input_ids", "labels", "attention_mask")):
        raise ValueError("v5 original tensors must match metadata B,T shape")
    blocks = []
    for row, (example, sample) in enumerate(zip(examples, metadata.samples, strict=True)):
        try:
            blocks.extend(_source_blocks(processor, example, sample, batch, row))
        except (ValueError, KeyError, TypeError, IndexError) as error:
            raise ValueError(f"source_index={example.source_index} source_id={example.source_id}: {error}") from error
    return V5BatchMetadata(tuple(blocks))
