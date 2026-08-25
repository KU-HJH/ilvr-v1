"""Explicit production composition and execution for Stage 1 training."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import time
from typing import Callable

import peft
import torch
from transformers import AutoProcessor, Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
from trl import SFTConfig

from peft_common import capture_qwen2_5_vl_architecture
from stage1.checkpointing import (
    STAGE1_PEFT_SCHEMA_VERSION,
    STAGE1_REPOSITORY_FORMAT_VERSION,
    Stage1ArtifactKind,
    Stage1PeftMetadata,
    Stage1ResumeDecision,
    decode_stage1_peft_metadata_snapshot,
    load_stage1_peft_artifact,
    preflight_stage1_peft_artifact,
    read_stage1_peft_metadata_snapshot,
    resolve_stage1_resume,
)
from stage1.data import Stage1DataCollator, prepare_stage1_comt_examples
from stage1.peft import (
    Stage1ParameterReport,
    _resolve_linear_decoder_targets,
    build_stage1_tuned_model,
    validate_stage1_trainable_boundary,
)
from stage1.trainer import CustomTrainerStage1
from stage1.types import Stage1TuningConfig, resolve_stage1_tuning_config


LATENT_TOKEN_STRINGS = (
    "<|latent_pad|>",
    "<|latent_start|>",
    "<|latent_end|>",
)
_GRADIENT_CHECKPOINTING_KWARGS = {"use_reentrant": False}


@dataclass(frozen=True)
class Stage1TrainingComponents:
    """The one fully composed Stage 1 runtime bundle."""

    model: object
    processor: object
    dataset: object
    collator: object
    trainer: object
    resume_decision: Stage1ResumeDecision
    parameter_report: Stage1ParameterReport


@dataclass(frozen=True)
class Stage1CompositionDependencies:
    """Heavy/external construction boundaries replaceable by unit tests."""

    resolve_resume: Callable
    load_processor: Callable
    load_config: Callable
    load_model: Callable
    load_peft_metadata: Callable
    decode_peft_metadata: Callable
    preflight_peft: Callable
    load_peft_model: Callable
    build_tuned_model: Callable
    validate_boundary: Callable
    prepare_dataset: Callable
    collator_factory: Callable
    trainer_factory: Callable
    build_peft_metadata: Callable
    capture_architecture: Callable
    resolve_target_names: Callable


def _build_stage1_peft_metadata(
    *,
    base_model_reference: str,
    base_architecture: dict[str, object],
    base_vocab_size: int,
    resized_vocab_size: int,
    latent_token_ids: tuple[int, int, int],
    latent_size: int,
    args,
    tuning: Stage1TuningConfig,
    resolved_target_names: tuple[str, ...],
) -> Stage1PeftMetadata:
    return Stage1PeftMetadata(
        schema_version=STAGE1_PEFT_SCHEMA_VERSION,
        stage="stage1",
        base_model_reference=base_model_reference,
        base_model_architecture=base_architecture,
        base_vocab_size=base_vocab_size,
        resized_vocab_size=resized_vocab_size,
        latent_token_ids=latent_token_ids,
        latent_token_strings=LATENT_TOKEN_STRINGS,
        latent_size=latent_size,
        ce_weight=float(args.ce_weight),
        sim_weight=float(args.sim_weight),
        coverage_p=float(args.coverage_p),
        image_pool_k=int(args.image_pool_k),
        tuning_method=tuning.method,
        rank=tuning.rank,
        alpha=tuning.alpha,
        dropout=tuning.dropout,
        requested_target_suffixes=tuning.target_suffixes,
        resolved_target_names=resolved_target_names,
        input_row_format="peft_trainable_tokens_v1",
        output_row_format="ilvr_sparse_output_rows_v1",
        output_row_shape=(3, int(base_architecture["decoder"]["hidden_size"])),
        peft_version=peft.__version__,
        repository_format_version=STAGE1_REPOSITORY_FORMAT_VERSION,
    )


DEFAULT_STAGE1_DEPENDENCIES = Stage1CompositionDependencies(
    resolve_resume=resolve_stage1_resume,
    load_processor=AutoProcessor.from_pretrained,
    load_config=Qwen2_5_VLConfig.from_pretrained,
    load_model=Qwen2_5_VLForConditionalGeneration.from_pretrained,
    load_peft_metadata=read_stage1_peft_metadata_snapshot,
    decode_peft_metadata=decode_stage1_peft_metadata_snapshot,
    preflight_peft=preflight_stage1_peft_artifact,
    load_peft_model=load_stage1_peft_artifact,
    build_tuned_model=build_stage1_tuned_model,
    validate_boundary=validate_stage1_trainable_boundary,
    prepare_dataset=prepare_stage1_comt_examples,
    collator_factory=Stage1DataCollator,
    trainer_factory=CustomTrainerStage1,
    build_peft_metadata=_build_stage1_peft_metadata,
    capture_architecture=capture_qwen2_5_vl_architecture,
    resolve_target_names=_resolve_linear_decoder_targets,
)


def _require_stage1_inputs(args) -> tuple[Path, Path, Path]:
    data_path = Path(args.data_path)
    image_root = Path(args.image_root) if args.image_root else None
    output_dir = Path(args.save_model_path)
    if not data_path.is_file():
        raise ValueError(f"Stage 1 data_path must be a file: {data_path}")
    if image_root is None or not image_root.is_dir():
        raise ValueError(f"Stage 1 image_root must be a directory: {image_root}")
    if args.deepspeed_config is not None:
        deepspeed = Path(args.deepspeed_config)
        if not deepspeed.is_file():
            raise ValueError(
                f"Stage 1 deepspeed_config must be a file: {deepspeed}"
            )
    return data_path, image_root, output_dir


def _tokenizer_ids(tokenizer) -> tuple[int, int, int]:
    try:
        values = tuple(
            int(tokenizer.convert_tokens_to_ids(token))
            for token in LATENT_TOKEN_STRINGS
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Stage 1 tokenizer cannot resolve latent token IDs") from error
    if len(set(values)) != 3 or any(value < 0 for value in values):
        raise ValueError(f"Stage 1 latent token IDs must be distinct: {values}")
    return values


def _add_fresh_latent_tokens(tokenizer) -> tuple[int, tuple[int, int, int]]:
    try:
        old_vocab_size = len(tokenizer)
        vocabulary = tokenizer.get_vocab()
    except (AttributeError, TypeError) as error:
        raise ValueError("Stage 1 tokenizer must expose a stable vocabulary") from error
    present = tuple(token for token in LATENT_TOKEN_STRINGS if token in vocabulary)
    if present:
        raise ValueError(
            "fresh Stage 1 base already contains latent tokens; refusing to remap: "
            f"{present}"
        )
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(LATENT_TOKEN_STRINGS)}
    )
    if added != len(LATENT_TOKEN_STRINGS):
        raise ValueError(
            "fresh Stage 1 tokenizer must add exactly three latent tokens: "
            f"added={added}"
        )
    token_ids = _tokenizer_ids(tokenizer)
    expected_ids = tuple(range(old_vocab_size, old_vocab_size + 3))
    if len(tokenizer) != old_vocab_size + 3 or token_ids != expected_ids:
        raise ValueError(
            "fresh Stage 1 latent tokens must append in pad/start/end order: "
            f"expected={expected_ids}, actual={token_ids}"
        )
    return old_vocab_size, token_ids


def _configure_latent_model(config, token_ids, *, latent_size: int) -> None:
    pad_id, start_id, end_id = token_ids
    config.stage = "stage1"
    config.latent_size = int(latent_size)
    config.latent_token_id = config.latent_pad_token_id = pad_id
    config.latent_start_id = config.latent_start_token_id = start_id
    config.latent_end_id = config.latent_end_token_id = end_id


def _validate_resume_tokens(
    processor, config, *, expected_ids: tuple[int, int, int], expected_size: int
) -> None:
    tokenizer = processor.tokenizer
    actual_ids = _tokenizer_ids(tokenizer)
    if len(tokenizer) != expected_size or actual_ids != expected_ids:
        raise ValueError(
            "Stage 1 resume tokenizer mismatch: "
            f"expected_size={expected_size}, actual_size={len(tokenizer)}, "
            f"expected_ids={expected_ids}, actual_ids={actual_ids}"
        )
    fields = (
        ("latent_token_id", "latent_pad_token_id", expected_ids[0]),
        ("latent_start_id", "latent_start_token_id", expected_ids[1]),
        ("latent_end_id", "latent_end_token_id", expected_ids[2]),
    )
    for legacy, canonical, expected in fields:
        observed = [
            getattr(config, name)
            for name in (legacy, canonical)
            if hasattr(config, name)
        ]
        if not observed or any(value != expected for value in observed):
            raise ValueError(
                f"Stage 1 resume config token mismatch for {legacy}/{canonical}: "
                f"expected={expected}, actual={observed}"
            )


def _validate_full_resume_tokens(processor, config) -> tuple[int, int, int]:
    tokenizer_size = len(processor.tokenizer)
    expected_ids = tuple(range(tokenizer_size - 3, tokenizer_size))
    actual_ids = _tokenizer_ids(processor.tokenizer)
    if tokenizer_size < 3 or actual_ids != expected_ids:
        raise ValueError(
            "Stage 1 full resume latent IDs must preserve the final three "
            f"appended rows: expected={expected_ids}, actual={actual_ids}"
        )
    if getattr(config, "vocab_size", None) != tokenizer_size:
        raise ValueError(
            "Stage 1 full resume config/tokenizer vocabulary mismatch: "
            f"config={getattr(config, 'vocab_size', None)}, "
            f"tokenizer={tokenizer_size}"
        )
    _validate_resume_tokens(
        processor,
        config,
        expected_ids=expected_ids,
        expected_size=tokenizer_size,
    )
    return expected_ids


def _validate_resume_tuning(
    decision: Stage1ResumeDecision,
    tuning: Stage1TuningConfig,
    metadata: Stage1PeftMetadata | None,
) -> None:
    if decision.artifact_kind is Stage1ArtifactKind.FULL:
        if tuning.method != "full":
            raise ValueError(
                "Stage 1 full checkpoint resume requires --tuning_method full"
            )
        return
    if decision.artifact_kind is not Stage1ArtifactKind.PEFT or metadata is None:
        raise ValueError("Stage 1 resume decision has no supported artifact kind")
    expected = (
        metadata.tuning_method,
        metadata.rank,
        metadata.alpha,
        metadata.dropout,
        metadata.requested_target_suffixes,
    )
    actual = (
        tuning.method,
        tuning.rank,
        tuning.alpha,
        tuning.dropout,
        tuning.target_suffixes,
    )
    if actual != expected:
        raise ValueError(
            "Stage 1 PEFT resume tuning mismatch: "
            f"checkpoint={expected}, requested={actual}"
        )


def _validate_resume_semantics(args, metadata: Stage1PeftMetadata) -> None:
    expected = (
        metadata.latent_size,
        metadata.ce_weight,
        metadata.sim_weight,
        metadata.coverage_p,
        metadata.image_pool_k,
    )
    actual = (
        args.latent_size,
        args.ce_weight,
        args.sim_weight,
        args.coverage_p,
        args.image_pool_k,
    )
    if actual != expected:
        raise ValueError(
            "Stage 1 PEFT resume semantic mismatch: "
            f"checkpoint={expected}, requested={actual}"
        )


def build_stage1_training_arguments(args) -> SFTConfig:
    """Build Stage 1 Trainer arguments only from validated runtime inputs."""

    return SFTConfig(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warm_up_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=1,
        optim="adamw_torch_fused",
        bf16=True,
        push_to_hub=False,
        report_to=[],
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=args.max_seq_length_train,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            dict(_GRADIENT_CHECKPOINTING_KWARGS)
            if args.gradient_checkpointing
            else None
        ),
        deepspeed=args.deepspeed_config,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )


def build_stage1_training_components(
    args,
    *,
    dependencies: Stage1CompositionDependencies = DEFAULT_STAGE1_DEPENDENCIES,
) -> Stage1TrainingComponents:
    """Validate sources, construct final topology, then create one Trainer."""

    if args.stage != "stage1":
        raise ValueError("Stage 1 composition requires --stage stage1")
    data_path, image_root, output_dir = _require_stage1_inputs(args)
    tuning = resolve_stage1_tuning_config(args)

    # This is deliberately first: Task 9 classifies the complete local artifact
    # and Trainer state without loading model weights.
    decision = dependencies.resolve_resume(
        output_dir, args.resume_from_checkpoint_path
    )
    checkpoint = decision.checkpoint_path
    metadata = None
    metadata_snapshot = None
    if decision.artifact_kind is Stage1ArtifactKind.PEFT:
        metadata_snapshot = dependencies.load_peft_metadata(checkpoint)
        metadata = dependencies.decode_peft_metadata(metadata_snapshot)
    _validate_resume_tuning(decision, tuning, metadata) if checkpoint else None
    if metadata is not None:
        _validate_resume_semantics(args, metadata)

    processor_source = checkpoint if checkpoint is not None else args.model
    processor = dependencies.load_processor(
        processor_source,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    base_architecture = None
    resolved_target_names: tuple[str, ...] = ()
    if checkpoint is None:
        base_vocab_size, latent_ids = _add_fresh_latent_tokens(processor.tokenizer)
        config = dependencies.load_config(
            args.model, cache_dir=args.cache_dir, trust_remote_code=True
        )
        base_architecture = dependencies.capture_architecture(config)
        if base_architecture["decoder"]["vocab_size"] != base_vocab_size:
            raise ValueError(
                "fresh Stage 1 base config/tokenizer vocabulary mismatch: "
                f"config={base_architecture['decoder']['vocab_size']}, "
                f"tokenizer={base_vocab_size}"
            )
        _configure_latent_model(config, latent_ids, latent_size=args.latent_size)
        config.compress_strategy = args.compress_strategy
        config.use_cache = False
        base_model = dependencies.load_model(
            args.model,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attention_implementation,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
        base_model.resize_token_embeddings(len(processor.tokenizer))
        if tuning.method != "full":
            resolved_target_names = tuple(
                dependencies.resolve_target_names(base_model, tuning.target_suffixes)
            )
        model = dependencies.build_tuned_model(base_model, tuning, latent_ids)
        if tuning.method != "full":
            metadata = dependencies.build_peft_metadata(
                base_model_reference=args.model,
                base_architecture=base_architecture,
                base_vocab_size=base_vocab_size,
                resized_vocab_size=len(processor.tokenizer),
                latent_token_ids=latent_ids,
                latent_size=args.latent_size,
                args=args,
                tuning=tuning,
                resolved_target_names=resolved_target_names,
            )
    elif decision.artifact_kind is Stage1ArtifactKind.FULL:
        config = dependencies.load_config(
            checkpoint, cache_dir=args.cache_dir, trust_remote_code=True
        )
        latent_ids = _validate_full_resume_tokens(processor, config)
        if getattr(config, "latent_size", None) != args.latent_size:
            raise ValueError(
                "Stage 1 full resume latent_size mismatch: "
                f"checkpoint={getattr(config, 'latent_size', None)}, "
                f"requested={args.latent_size}"
            )
        model = dependencies.load_model(
            checkpoint,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attention_implementation,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
        model = dependencies.build_tuned_model(model, tuning, latent_ids)
    else:
        latent_ids = tuple(metadata.latent_token_ids)
        config = dependencies.load_config(
            metadata.base_model_reference,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
        _configure_latent_model(config, latent_ids, latent_size=metadata.latent_size)
        _validate_resume_tokens(
            processor,
            config,
            expected_ids=latent_ids,
            expected_size=metadata.resized_vocab_size,
        )
        preflight = dependencies.preflight_peft(
            processor,
            config,
            checkpoint,
            metadata_snapshot=metadata_snapshot,
        )
        base_model = dependencies.load_model(
            metadata.base_model_reference,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attention_implementation,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
        model = dependencies.load_peft_model(
            base_model,
            processor,
            checkpoint,
            is_trainable=True,
            preflight=preflight,
        )

    model.visual.requires_grad_(False)
    parameter_report = dependencies.validate_boundary(model, tuning, latent_ids)
    dataset = dependencies.prepare_dataset(data_path, image_root)
    if args.max_train_samples is not None:
        dataset = dataset[: args.max_train_samples]
    token_ids = LatentTokenIds(*latent_ids)
    collator = dependencies.collator_factory(processor, token_ids, args.latent_size)
    trainer = dependencies.trainer_factory(
        model=model,
        args=build_stage1_training_arguments(args),
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        sim_weight=args.sim_weight,
        ema_tau=args.ema_tau,
        coverage_p=args.coverage_p,
        image_pool_k=args.image_pool_k,
        helper_group_L=args.helper_group_L,
        ce_weight=args.ce_weight,
        ema_method=tuning.method,
        stage1_peft_metadata=metadata,
    )
    return Stage1TrainingComponents(
        model=model,
        processor=processor,
        dataset=dataset,
        collator=collator,
        trainer=trainer,
        resume_decision=decision,
        parameter_report=parameter_report,
    )


def run_stage1(
    args,
    *,
    component_builder: Callable = build_stage1_training_components,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Train once and, when requested, perform one separately timed final save."""

    components = component_builder(args)
    resume = components.resume_decision.checkpoint_path
    trainer_start = monotonic()
    components.trainer.train(
        resume_from_checkpoint=None if resume is None else str(resume)
    )
    trainer_seconds = monotonic() - trainer_start
    if not math.isfinite(trainer_seconds) or trainer_seconds < 0:
        raise RuntimeError("Stage 1 Trainer timing must be finite and non-negative")
    logging.info("Stage 1 Trainer time: %.3fs", trainer_seconds)

    if not args.final_save:
        return
    save_start = monotonic()
    components.trainer.save_model(args.save_model_path)
    if components.trainer.is_world_process_zero():
        components.processor.save_pretrained(args.save_model_path)
    save_seconds = monotonic() - save_start
    if not math.isfinite(save_seconds) or save_seconds < 0:
        raise RuntimeError("Stage 1 final save timing must be finite and non-negative")
    logging.info("Stage 1 final save time: %.3fs", save_seconds)


__all__ = [
    "DEFAULT_STAGE1_DEPENDENCIES",
    "LATENT_TOKEN_STRINGS",
    "Stage1CompositionDependencies",
    "Stage1TrainingComponents",
    "build_stage1_training_arguments",
    "build_stage1_training_components",
    "run_stage1",
]
