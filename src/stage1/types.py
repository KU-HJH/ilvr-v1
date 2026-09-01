"""Immutable Stage 1 tuning and runtime contracts."""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import torch

from peft_common import normalize_decoder_target_suffixes


Stage1TuningMethod = Literal["full", "lora", "dora"]
DEFAULT_STAGE1_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
STAGE1_CANONICAL_VIEW_LENGTH = 3456
STAGE1_CANONICAL_SOURCE_ROWS = 3450


@dataclass(frozen=True)
class Stage1SampleOccurrence:
    """Immutable identity of one source-row occurrence in the canonical view."""

    epoch: int
    view_slot: int
    permutation_offset: int
    canonical_slot_id: int
    source_index: int
    rank: int
    rank_local_sample_ordinal: int
    rank_local_microbatch_ordinal: int
    row_ordinal_within_microbatch: int

    def __post_init__(self) -> None:
        fields = (
            self.epoch,
            self.view_slot,
            self.permutation_offset,
            self.canonical_slot_id,
            self.source_index,
            self.rank,
            self.rank_local_sample_ordinal,
            self.rank_local_microbatch_ordinal,
            self.row_ordinal_within_microbatch,
        )
        if any(type(value) is not int for value in fields):
            raise ValueError("Stage 1 occurrence fields must be integers")
        if self.epoch < 0 or not 0 <= self.view_slot < STAGE1_CANONICAL_VIEW_LENGTH:
            raise ValueError("Stage 1 occurrence has an invalid epoch or view_slot")
        if not 0 <= self.permutation_offset < STAGE1_CANONICAL_VIEW_LENGTH:
            raise ValueError("Stage 1 occurrence has an invalid permutation_offset")
        if self.canonical_slot_id != self.epoch * STAGE1_CANONICAL_VIEW_LENGTH + self.view_slot:
            raise ValueError("Stage 1 occurrence canonical_slot_id does not match epoch/view_slot")
        if not 0 <= self.source_index < STAGE1_CANONICAL_SOURCE_ROWS:
            raise ValueError("Stage 1 occurrence source_index is not canonical for view_slot")
        if self.source_index != self.view_slot % STAGE1_CANONICAL_SOURCE_ROWS:
            raise ValueError("Stage 1 occurrence source_index is not canonical for view_slot")
        if not 0 <= self.rank < 4:
            raise ValueError("Stage 1 occurrence has an invalid source_index or rank")
        if self.permutation_offset % 4 != self.rank:
            raise ValueError("Stage 1 occurrence permutation_offset does not match rank")
        if not 0 <= self.rank_local_sample_ordinal < 4:
            raise ValueError("Stage 1 occurrence rank-local sample ordinal is not canonical")
        if self.permutation_offset % 16 != (
            4 * self.rank_local_sample_ordinal + self.rank
        ):
            raise ValueError(
                "Stage 1 occurrence permutation_offset/rank-local ordinal is not canonical"
            )
        if (
            self.rank_local_sample_ordinal < 0
            or self.rank_local_microbatch_ordinal < 0
            or self.row_ordinal_within_microbatch < 0
        ):
            raise ValueError("Stage 1 occurrence has a negative rank-local ordinal")
        b1_grouping = (
            self.rank_local_sample_ordinal == self.rank_local_microbatch_ordinal
            and self.row_ordinal_within_microbatch == 0
        )
        b2_grouping = (
            self.rank_local_sample_ordinal
            == 2 * self.rank_local_microbatch_ordinal
            + self.row_ordinal_within_microbatch
            and self.row_ordinal_within_microbatch < 2
        )
        if not (b1_grouping or b2_grouping):
            raise ValueError("Stage 1 occurrence has an invalid rank-local grouping")


@dataclass(frozen=True)
class Stage1LatentSpan:
    """One complete latent block in a sample's padded batch row."""

    start: int
    end: int
    latent_positions: tuple[int, ...]


@dataclass(frozen=True)
class Stage1SampleMetadata:
    """Sample-local ownership after processor outputs have been flattened."""

    source_index: int
    source_id: str | None
    original_grid_range: tuple[int, int]
    helper_grid_range: tuple[int, int]
    latent_spans: tuple[Stage1LatentSpan, ...]
    helper_to_block: tuple[int, ...]
    unpadded_length: int
    epoch: int | None = None
    view_slot: int | None = None
    permutation_offset: int | None = None
    canonical_slot_id: int | None = None
    rank: int | None = None
    rank_local_sample_ordinal: int | None = None
    rank_local_microbatch_ordinal: int | None = None
    row_ordinal_within_microbatch: int | None = None


@dataclass(frozen=True)
class Stage1BatchMetadata:
    """Trainer-only ownership metadata for one collated Stage 1 batch."""

    samples: tuple[Stage1SampleMetadata, ...]
    batch_size: int
    sequence_length: int


@dataclass(frozen=True)
class Stage1TeacherTargets:
    """Packed, row-major EMA-teacher targets for one Stage 1 batch."""

    latent_values: torch.Tensor
    latent_mask: torch.Tensor
    selected_counts: tuple[tuple[int, ...], ...]
    diagnostics: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class Stage1TuningConfig:
    method: Stage1TuningMethod
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_suffixes: tuple[str, ...] = DEFAULT_STAGE1_TARGET_SUFFIXES

    def __post_init__(self) -> None:
        if self.method not in ("full", "lora", "dora"):
            raise ValueError(f"unsupported Stage 1 tuning method: {self.method}")
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be finite and in [0, 1)")
        normalize_decoder_target_suffixes(
            self.target_suffixes, stage_label="Stage 1 PEFT target modules"
        )


@dataclass(frozen=True)
class Stage1RuntimeConfig:
    data_path: Path
    image_root: Path
    output_dir: Path
    model_name_or_path: str
    epochs: float
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    seed: int
    attention_implementation: str
    deepspeed_config: Path | None


def resolve_stage1_tuning_config(args: argparse.Namespace) -> Stage1TuningConfig:
    """Resolve validated parser inputs, preserving legacy LoRA as LoRA."""
    if args.stage != "stage1":
        raise ValueError("Stage 1 tuning config requires --stage stage1")

    method: Stage1TuningMethod
    if args.use_lora:
        method = "lora"
    else:
        method = args.tuning_method or "full"

    target_suffixes = (
        normalize_decoder_target_suffixes(
            args.lora_target_modules, stage_label="Stage 1 PEFT target modules"
        )
        if args.lora_target_modules_explicit
        else DEFAULT_STAGE1_TARGET_SUFFIXES
    )
    return Stage1TuningConfig(
        method=method,
        rank=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_suffixes=target_suffixes,
    )
