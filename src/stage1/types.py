"""Immutable Stage 1 tuning and runtime contracts."""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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


@dataclass(frozen=True)
class Stage1BatchMetadata:
    """Trainer-only ownership metadata for one collated Stage 1 batch."""

    samples: tuple[Stage1SampleMetadata, ...]
    batch_size: int
    sequence_length: int


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
