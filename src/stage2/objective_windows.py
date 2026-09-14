"""Tokenizer-resolved text vocabulary and Stage 2 block window selection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentStructureError,
    LatentTokenIds,
    parse_latent_spans,
)


_STRUCTURAL_TOKENS = frozenset(
    {
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<tool_call>",
        "</tool_call>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
        "<|latent_pad|>",
        "<|latent_start|>",
        "<|latent_end|>",
    }
)

_CONFIG_CONTROL_FIELDS = (
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "vision_token_id",
    "latent_token_id",
    "latent_pad_token_id",
    "latent_start_token_id",
    "latent_end_token_id",
)


@dataclass(frozen=True)
class TextVocabulary:
    token_ids: tuple[int, ...]
    excluded: dict[int, str]
    output_size: int

    @property
    def identity(self) -> dict[str, Any]:
        """Return a deterministic, JSON-safe description for metadata checks."""
        return {
            "output_size": self.output_size,
            "token_ids": list(self.token_ids),
            "excluded": [
                {"token_id": token_id, "reason": reason}
                for token_id, reason in sorted(self.excluded.items())
            ],
        }


def _validated_id(value: Any, field: str, output_size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must resolve to an integer token ID")
    if not 0 <= value < output_size:
        raise ValueError(f"{field}={value} does not resolve inside output_size={output_size}")
    return value


def build_text_vocabulary(tokenizer, model_config, output_size: int) -> TextVocabulary:
    """Partition actual output rows into text IDs and enumerated exclusions."""
    if isinstance(output_size, bool) or not isinstance(output_size, int) or output_size <= 0:
        raise ValueError("output_size must be a positive integer")
    try:
        vocabulary = tokenizer.get_vocab()
    except (AttributeError, TypeError) as error:
        raise ValueError("tokenizer must expose get_vocab()") from error
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("tokenizer vocabulary must be a non-empty mapping")

    inverse: dict[int, str] = {}
    for token, raw_token_id in vocabulary.items():
        if not isinstance(token, str):
            raise ValueError("tokenizer vocabulary tokens must be strings")
        token_id = _validated_id(raw_token_id, f"tokenizer token {token!r}", output_size)
        if token_id in inverse and inverse[token_id] != token:
            raise ValueError(f"tokenizer maps multiple tokens to token ID {token_id}")
        inverse[token_id] = token

    excluded: dict[int, str] = {}
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        pad_token_id = _validated_id(pad_token_id, "pad_token_id", output_size)
        if pad_token_id not in inverse:
            raise ValueError("pad_token_id cannot resolve to a tokenizer token")
        excluded[pad_token_id] = "padding_token"

    try:
        special_ids = tuple(tokenizer.all_special_ids)
    except (AttributeError, TypeError) as error:
        raise ValueError("tokenizer must expose all_special_ids") from error
    for raw_token_id in special_ids:
        token_id = _validated_id(raw_token_id, "all_special_ids", output_size)
        if token_id not in inverse:
            raise ValueError(f"special token ID {token_id} cannot resolve to a tokenizer token")
        excluded.setdefault(token_id, "special_token")

    for token, token_id in vocabulary.items():
        if token in _STRUCTURAL_TOKENS:
            excluded.setdefault(token_id, f"structural_token:{token}")

    for field in _CONFIG_CONTROL_FIELDS:
        raw_token_id = getattr(model_config, field, None)
        if raw_token_id is None:
            continue
        token_id = _validated_id(raw_token_id, field, output_size)
        if token_id not in inverse:
            raise ValueError(f"{field}={token_id} cannot resolve to a tokenizer token")
        excluded.setdefault(token_id, f"model_config:{field}")

    for token_id in range(output_size):
        if token_id not in inverse:
            excluded.setdefault(token_id, "unmapped_output_row")

    token_ids = tuple(token_id for token_id in range(output_size) if token_id not in excluded)
    if not token_ids:
        raise ValueError("text vocabulary is empty after structural exclusions")
    return TextVocabulary(token_ids=token_ids, excluded=dict(sorted(excluded.items())), output_size=output_size)


@dataclass(frozen=True)
class WindowBlock:
    batch_index: int
    block_index: int
    start_position: int
    end_position: int
    pre_positions: tuple[int, ...]
    post_positions: tuple[int, ...]
    target_ids: tuple[int, ...]


@dataclass(frozen=True)
class WindowSelection:
    blocks: tuple[WindowBlock, ...]
    excluded: dict[str, int]


def _validate_inputs(input_ids, labels, attention_mask, text_segments) -> None:
    for name, value in (
        ("input_ids", input_ids),
        ("labels", labels),
        ("attention_mask", attention_mask),
    ):
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must be a rank-2 tensor")
    if input_ids.shape != labels.shape or input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids, labels, and attention_mask must have identical shapes")
    if len(text_segments) != input_ids.shape[0]:
        raise ValueError("text_segments must contain one entry per batch row")


def _validated_segments(raw_segments, *, length: int, batch_index: int):
    segments = []
    for segment_index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, (tuple, list)) or len(raw_segment) != 2:
            raise ValueError(
                f"batch_index={batch_index}: text_segments[{segment_index}] must be a pair"
            )
        start, end = raw_segment
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)):
            raise ValueError(
                f"batch_index={batch_index}: text_segments[{segment_index}] must use integers"
            )
        if not 0 <= start <= end <= length:
            raise ValueError(
                f"batch_index={batch_index}: text_segments[{segment_index}] is out of bounds"
            )
        if segments and start < segments[-1][1]:
            raise ValueError(f"batch_index={batch_index}: text_segments must be ordered")
        segments.append((start, end))
    return tuple(segments)


def select_objective_windows(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    text_segments,
    *,
    token_ids: LatentTokenIds,
    latent_size: int,
    text_vocabulary: TextVocabulary,
    pre_window: int,
    post_window: int,
) -> WindowSelection:
    """Select exact per-block pre/post states and teacher-forced post targets."""
    _validate_inputs(input_ids, labels, attention_mask, text_segments)
    for name, value in (("pre_window", pre_window), ("post_window", post_window)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(latent_size, bool) or not isinstance(latent_size, int) or latent_size <= 0:
        raise ValueError("latent_size must be a positive integer")
    text_ids = frozenset(text_vocabulary.token_ids)
    if not text_ids:
        raise ValueError("text_vocabulary must contain at least one token ID")

    candidates: list[WindowBlock] = []
    excluded: Counter[str] = Counter()
    sequence_length = input_ids.shape[1]
    for batch_index in range(input_ids.shape[0]):
        try:
            spans = parse_latent_spans(
                input_ids[batch_index], token_ids=token_ids, latent_size=latent_size
            )
        except LatentStructureError as error:
            raise ValueError(f"batch_index={batch_index}: malformed latent markers: {error}") from error
        segments = _validated_segments(
            text_segments[batch_index], length=sequence_length, batch_index=batch_index
        )
        if len(segments) != len(spans) + 1:
            raise ValueError(
                f"batch_index={batch_index}: text_segments count must equal latent blocks plus one"
            )
        for block_index, span in enumerate(spans):
            pre_segment = segments[block_index]
            post_segment = segments[block_index + 1]
            if pre_segment[1] != span.start_position or post_segment[0] != span.end_position + 1:
                raise ValueError(
                    f"batch_index={batch_index} block_index={block_index}: text_segments do not align with latent markers"
                )
            pre_candidates = [
                position
                for position in range(*pre_segment)
                if bool(attention_mask[batch_index, position])
                and int(input_ids[batch_index, position]) in text_ids
            ]
            post_targets = [
                position
                for position in range(*post_segment)
                if bool(attention_mask[batch_index, position])
                and int(labels[batch_index, position]) != -100
                and int(input_ids[batch_index, position]) in text_ids
            ]
            pre_positions = tuple(pre_candidates[-pre_window:])
            target_positions = post_targets[:post_window]
            post_positions = tuple(position - 1 for position in target_positions)
            target_values = tuple(int(labels[batch_index, position]) for position in target_positions)
            if any(
                target != int(input_ids[batch_index, position])
                for target, position in zip(target_values, target_positions)
            ):
                raise ValueError(
                    f"batch_index={batch_index} block_index={block_index}: labels disagree with post target IDs"
                )
            if not pre_positions:
                excluded["empty_pre"] += 1
            if not post_positions:
                excluded["empty_post"] += 1
            candidates.append(
                WindowBlock(
                    batch_index=batch_index,
                    block_index=block_index,
                    start_position=span.start_position,
                    end_position=span.end_position,
                    pre_positions=pre_positions,
                    post_positions=post_positions,
                    target_ids=target_values,
                )
            )

    pre_all = {
        (block.batch_index, position)
        for block in candidates
        for position in block.pre_positions
    }
    post_all = {
        (block.batch_index, position)
        for block in candidates
        for position in block.post_positions
    }
    overlaps = pre_all & post_all
    blocks: list[WindowBlock] = []
    for block in candidates:
        kept_pre = tuple(
            position
            for position in block.pre_positions
            if (block.batch_index, position) not in overlaps
        )
        kept_post_and_targets = tuple(
            (position, target)
            for position, target in zip(block.post_positions, block.target_ids)
            if (block.batch_index, position) not in overlaps
        )
        kept_post = tuple(position for position, _ in kept_post_and_targets)
        kept_targets = tuple(target for _, target in kept_post_and_targets)
        if block.pre_positions and not kept_pre:
            excluded["overlap_empty_pre"] += 1
        if block.post_positions and not kept_post:
            excluded["overlap_empty_post"] += 1
        if not kept_pre or not kept_post:
            continue
        blocks.append(
            WindowBlock(
                batch_index=block.batch_index,
                block_index=block.block_index,
                start_position=block.start_position,
                end_position=block.end_position,
                pre_positions=kept_pre,
                post_positions=kept_post,
                target_ids=kept_targets,
            )
        )
    return WindowSelection(blocks=tuple(blocks), excluded=dict(sorted(excluded.items())))
