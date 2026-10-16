"""Explicit production composition and execution for Stage 1 training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Callable

import peft
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from transformers import AutoProcessor, Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
from trl import SFTConfig

from deepspeed_runtime import (
    DeepSpeedRuntimeProfile,
    bind_deepspeed_runtime_profile,
)
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
    validate_stage1_vocab_alignment,
)
from stage1.data import Stage1DataCollator, prepare_stage1_comt_examples
from stage1.peft import (
    Stage1ParameterReport,
    _resolve_linear_decoder_targets,
    build_stage1_tuned_model,
    install_stage1_sample_stable_dropout,
    validate_stage1_stochastic_inventory,
    validate_stage1_trainable_boundary,
)
from stage1.equivalence import stage1_checkpoint_context_fn
from stage1.trainer import CustomTrainerStage1
from stage1.types import Stage1TuningConfig, resolve_stage1_tuning_config


LATENT_TOKEN_STRINGS = (
    "<|latent_pad|>",
    "<|latent_start|>",
    "<|latent_end|>",
)


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
    install_sample_stable_dropout: Callable
    validate_stochastic_inventory: Callable
    validate_boundary: Callable
    prepare_dataset: Callable
    collator_factory: Callable
    trainer_factory: Callable
    build_peft_metadata: Callable
    capture_architecture: Callable
    resolve_target_names: Callable
    validate_vocab_alignment: Callable


def _build_stage1_peft_metadata(
    *,
    base_model_reference: str,
    base_architecture: dict[str, object],
    base_vocab_size: int,
    pretrained_model_vocab_size: int,
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
        pretrained_model_vocab_size=pretrained_model_vocab_size,
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
    install_sample_stable_dropout=install_stage1_sample_stable_dropout,
    validate_stochastic_inventory=validate_stage1_stochastic_inventory,
    validate_boundary=validate_stage1_trainable_boundary,
    prepare_dataset=prepare_stage1_comt_examples,
    collator_factory=Stage1DataCollator,
    trainer_factory=CustomTrainerStage1,
    build_peft_metadata=_build_stage1_peft_metadata,
    capture_architecture=capture_qwen2_5_vl_architecture,
    resolve_target_names=_resolve_linear_decoder_targets,
    validate_vocab_alignment=validate_stage1_vocab_alignment,
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


def _validate_resume_semantics(args, metadata: Stage1PeftMetadata, *, v5_resume_artifact: Path | None = None) -> None:
    if metadata.schema_version == 3:
        raise ValueError("schema 3 Stage 1 training resume is unsupported")
    if v5_resume_artifact is not None:
        from stage1.checkpointing import _validate_base_source
        _validate_base_source(args.model, metadata.base_model_reference,
                              v5_resume_artifact=v5_resume_artifact)
    elif args.model != metadata.base_model_reference:
        raise ValueError(
            "Stage 1 PEFT base source mismatch: "
            f"metadata={metadata.base_model_reference!r}, "
            f"requested={args.model!r}"
        )
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

    gradient_checkpointing_kwargs = None
    if args.gradient_checkpointing:
        tuning = resolve_stage1_tuning_config(args)
        gradient_checkpointing_kwargs = {"use_reentrant": False}
        if tuning.method != "full":
            gradient_checkpointing_kwargs["context_fn"] = (
                stage1_checkpoint_context_fn
            )

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
        optim="adamw_torch" if getattr(args, "use_cpu", False) else "adamw_torch_fused",
        bf16=not getattr(args, "use_cpu", False),
        use_cpu=bool(getattr(args, "use_cpu", False)),
        push_to_hub=False,
        report_to=[],
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=32768,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        dataloader_num_workers=getattr(args, "dataloader_num_workers", 0),
        deepspeed=args.deepspeed_config,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )


def validate_stage1_checkpoint_context(model: torch.nn.Module) -> None:
    """Require every resolved checkpoint hook to carry the exact Stage 1 pair."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Stage 1 checkpoint context validation requires an nn.Module")
    enabled = []
    expected_keywords = {
        "use_reentrant": False,
        "context_fn": stage1_checkpoint_context_fn,
    }
    for name, module in model.named_modules():
        if not bool(getattr(module, "gradient_checkpointing", False)):
            continue
        enabled.append(name or "<root>")
        resolved = getattr(module, "_gradient_checkpointing_func", None)
        if (
            getattr(resolved, "func", None) is not activation_checkpoint
            or getattr(resolved, "keywords", None) != expected_keywords
        ):
            raise RuntimeError(
                "Stage 1 PEFT requires the exact non-reentrant checkpoint context "
                f"pair at {name or '<root>'}"
            )
    if not enabled:
        raise RuntimeError(
            "Stage 1 PEFT requires the exact checkpoint context on an enabled module"
        )


def _copy_v5_resume_reference(checkpoint, output_dir):
    import shutil
    from artifact_paths import require_canonical_artifact_path
    from stage1.v5_reference import V5_REFERENCE_FILENAME
    output_dir = require_canonical_artifact_path(output_dir, kind='checkpoint')
    source, destination = checkpoint / V5_REFERENCE_FILENAME, output_dir / V5_REFERENCE_FILENAME
    if source.is_symlink() or destination.is_symlink():
        raise ValueError('immutable v5 reference must be a regular file')
    if source.resolve() != destination.resolve():
        output_dir.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError('cannot overwrite immutable v5 run reference')
        shutil.copyfile(source, destination)


def _load_stage1_model(source, config, args, dependencies):
    """Reuse the authenticated mixed-backend construction for opt-in v5."""
    model_kwargs = dict(torch_dtype=torch.bfloat16, cache_dir=args.cache_dir,
                        trust_remote_code=True)
    if getattr(args, 'stage1_variant', 'standard') == 'v5':
        from stage2.attention_backend import load_qwen2_5_vl_with_attention
        return load_qwen2_5_vl_with_attention(source,
            attention_implementation=args.attention_implementation,
            vision_attention_implementation=args.vision_attention_implementation,
            load_config=lambda source, **kwargs: config,
            load_model=dependencies.load_model, model_load_kwargs=model_kwargs)
    return dependencies.load_model(source, config=config,
        attn_implementation=args.attention_implementation, **model_kwargs)


def build_stage1_training_components(
    args,
    *,
    dependencies: Stage1CompositionDependencies = DEFAULT_STAGE1_DEPENDENCIES,
    deepspeed_runtime_profile: DeepSpeedRuntimeProfile | None = None,
) -> Stage1TrainingComponents:
    """Validate sources, construct final topology, then create one Trainer."""

    if deepspeed_runtime_profile is None:
        profile = bind_deepspeed_runtime_profile(
            args, world_size=int(os.environ.get("WORLD_SIZE", "1"))
        )
    else:
        if not isinstance(deepspeed_runtime_profile, DeepSpeedRuntimeProfile):
            raise TypeError("invalid Stage 1 DeepSpeed runtime profile")
        if getattr(args, "deepspeed_runtime_profile", None) is None:
            args.deepspeed_runtime_profile = deepspeed_runtime_profile
        profile = bind_deepspeed_runtime_profile(
            args, world_size=deepspeed_runtime_profile.world_size
        )
        if profile is not deepspeed_runtime_profile:
            raise RuntimeError("Stage 1 DeepSpeed runtime profile identity changed")

    if args.stage != "stage1":
        raise ValueError("Stage 1 composition requires --stage stage1")
    data_path, image_root, output_dir = _require_stage1_inputs(args)
    tuning = resolve_stage1_tuning_config(args)
    is_v5 = getattr(args, 'stage1_variant', 'standard') == 'v5'
    from stage1.query import Stage1QueryConfig
    query_config = getattr(args, 'stage1_query_config', Stage1QueryConfig())
    if query_config.mode == 'adaptive' and not is_v5:
        raise ValueError('adaptive query requires Stage1-v5')
    if is_v5:
        if (tuning.method != 'dora' or args.ema_tau != 1.
                or not profile.enabled or profile.zero_stage != 1 or profile.world_size != 4
                or args.per_device_train_batch_size != 1 or args.gradient_accumulation_steps != 4
                or getattr(args, 'use_cpu', False) or args.attention_implementation != 'sdpa'
                or args.vision_attention_implementation != 'flash_attention_2'
                or not args.gradient_checkpointing):
            raise ValueError('unsupported Stage1-v5 execution profile')
        from transformers import set_seed
        set_seed(args.seed)

    # This is deliberately first: Task 9 classifies the complete local artifact
    # and Trainer state without loading model weights.
    decision = dependencies.resolve_resume(
        output_dir, args.resume_from_checkpoint_path
    )
    checkpoint = decision.checkpoint_path
    metadata = None
    metadata_snapshot = decision.peft_metadata_snapshot
    if decision.artifact_kind is Stage1ArtifactKind.PEFT:
        if metadata_snapshot is None:
            raise ValueError(
                "Stage 1 PEFT resume decision is missing its classified "
                "metadata snapshot"
            )
        metadata = dependencies.decode_peft_metadata(metadata_snapshot)
    _validate_resume_tuning(decision, tuning, metadata) if checkpoint else None
    if metadata is not None:
        _validate_resume_semantics(args, metadata, v5_resume_artifact=checkpoint if is_v5 else None)

    v5_saved = None
    if is_v5 and checkpoint is not None:
        from stage1.v5_resume import load_v5_contract, validate_v5_resume, resolve_v5_query_resume
        v5_saved = load_v5_contract(checkpoint)
        restored_query = resolve_v5_query_resume(v5_saved, getattr(args, 'stage1_query_explicit_values', {}))
        if restored_query != query_config:
            raise ValueError('resolved query conflicts with saved v5 contract')
        validate_v5_resume(v5_saved, {**getattr(args, 'stage1_v5_explicit_values', {}),
                                     'ce_weight':args.ce_weight,'sim_weight':args.sim_weight})

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
        resized_vocab_size = len(processor.tokenizer)
        pretrained_model_vocab_size = base_architecture["decoder"]["vocab_size"]
        if pretrained_model_vocab_size < base_vocab_size:
            raise ValueError(
                "fresh Stage 1 pretrained model vocabulary is too small for "
                "the base tokenizer: "
                f"model={pretrained_model_vocab_size}, "
                f"tokenizer_base={base_vocab_size}, "
                f"tokenizer_resized={resized_vocab_size}"
            )
        _configure_latent_model(config, latent_ids, latent_size=args.latent_size)
        config.compress_strategy = args.compress_strategy
        config.use_cache = False
        base_model = _load_stage1_model(args.model, config, args, dependencies)
        base_model.resize_token_embeddings(resized_vocab_size)
        dependencies.validate_vocab_alignment(base_model, resized_vocab_size)
        if tuning.method != "full":
            resolved_target_names = tuple(
                dependencies.resolve_target_names(base_model, tuning.target_suffixes)
            )
        model = dependencies.build_tuned_model(
            base_model, tuning, latent_ids, zero_stage=profile.zero_stage
        )
        if tuning.method != "full":
            metadata = dependencies.build_peft_metadata(
                base_model_reference=args.model,
                base_architecture=base_architecture,
                base_vocab_size=base_vocab_size,
                pretrained_model_vocab_size=pretrained_model_vocab_size,
                resized_vocab_size=resized_vocab_size,
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
            attn_implementation=({"": args.attention_implementation, "vision_config": args.vision_attention_implementation}
                                 if is_v5 else args.attention_implementation),
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
        model = dependencies.build_tuned_model(
            model, tuning, latent_ids, zero_stage=profile.zero_stage
        )
    else:
        latent_ids = tuple(metadata.latent_token_ids)
        config = dependencies.load_config(
            args.model,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
        _configure_latent_model(config, latent_ids, latent_size=metadata.latent_size)
        config.compress_strategy = args.compress_strategy
        config.use_cache = False
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
            requested_base_model_reference=args.model,
            metadata_snapshot=metadata_snapshot,
            **({"v5_resume": True} if is_v5 else {}),
        )
        base_model = _load_stage1_model(args.model, config, args, dependencies)
        model = dependencies.load_peft_model(
            base_model,
            processor,
            checkpoint,
            requested_base_model_reference=args.model,
            is_trainable=True,
            preflight=preflight,
            **({"v5_resume": True} if is_v5 else {}),
        )

    stochastic_sites: tuple[str, ...] = ()
    if tuning.method != "full":
        stochastic_sites = tuple(
            dependencies.install_sample_stable_dropout(model)
        )
        dependencies.validate_stochastic_inventory(model, stochastic_sites)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False,
                    "context_fn": stage1_checkpoint_context_fn,
                }
            )

    model.visual.requires_grad_(False)
    parameter_report = dependencies.validate_boundary(
        model, tuning, latent_ids, zero_stage=profile.zero_stage
    )
    dataset = dependencies.prepare_dataset(data_path, image_root)
    if args.max_train_samples is not None:
        dataset = dataset[: args.max_train_samples]
    token_ids = LatentTokenIds(*latent_ids)
    collator = dependencies.collator_factory(processor, token_ids, args.latent_size,
        **({'stage1_variant': 'v5'} if is_v5 else {}))
    training_args = build_stage1_training_arguments(args)
    factory = dependencies.trainer_factory
    v5_kwargs = {}
    if is_v5:
        from copy import deepcopy
        from stage1.ema import Stage1EmaTeacher
        from stage1.v5_reference import (V5ReferenceIdentity, capture_v5_reference,
                                        restore_v5_reference, inspect_v5_reference)
        from stage1.v5_trainer import CustomTrainerStage1V5
        # SFTConfig initializes the existing distributed device context; use its
        # normal process group for identity/error agreement, not manual ranks.
        training_args.device
        reference_model = None
        if checkpoint is not None:
            reference_config = deepcopy(config)
            original = _load_stage1_model(args.model, reference_config, args, dependencies)
            original.resize_token_embeddings(len(processor.tokenizer))
            reference_model = dependencies.build_tuned_model(original,tuning,latent_ids,
                                                               zero_stage=profile.zero_stage)
        teacher = Stage1EmaTeacher(model,decay=1.,method='dora',
                                  initial_reference_model=reference_model)
        error, identity = None, None
        try:
            if checkpoint is not None:
                identity = V5ReferenceIdentity(**v5_saved['reference'])
                restore_v5_reference(teacher,model,processor,base_path=Path(args.model),
                                     directory=checkpoint,expected=identity)
            elif training_args.should_save:
                identity = capture_v5_reference(teacher,model,processor,
                    base_path=Path(args.model),directory=output_dir)
            else:
                identity = inspect_v5_reference(teacher,model,processor,base_path=Path(args.model))
        except Exception as exc:
            error = exc
        if not torch.distributed.is_initialized():
            raise RuntimeError('v5 requires the initialized four-rank process group')
        failures = [None]*profile.world_size
        torch.distributed.all_gather_object(failures, None if error is None else str(error))
        if any(failure is not None for failure in failures):
            if error is not None: raise error
            raise RuntimeError(f'v5 initial reference failed on another rank: {failures}')
        identities = [None]*profile.world_size
        torch.distributed.all_gather_object(identities,asdict(identity))
        if any(value != identities[0] for value in identities):
            raise ValueError('v5 full base/processor/reference identity agreement failed')
        error = None
        try:
            if checkpoint is not None and training_args.should_save:
                _copy_v5_resume_reference(checkpoint, output_dir)
        except Exception as exc:
            error = exc
        torch.distributed.all_gather_object(failures, None if error is None else str(error))
        if any(failure is not None for failure in failures):
            if error is not None: raise error
            raise RuntimeError(f'v5 resume reference publication failed: {failures}')
        factory = CustomTrainerStage1V5
        v5_kwargs = dict(v5_config=args.stage1_v5_config, v5_reference_identity=identity,
                         v5_saved_contract=v5_saved, ema_teacher=teacher,
                         stage1_v5_debug=bool(getattr(args, "stage1_v5_debug", False)))
    trainer = factory(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        sim_weight=args.sim_weight,
        ema_tau=args.ema_tau,
        coverage_p=args.coverage_p,
        image_pool_k=args.image_pool_k,
        helper_group_L=args.helper_group_L,
        textual_states=args.textual_states,
        query_config=query_config,
        ce_weight=args.ce_weight,
        deepspeed_runtime_profile=profile,
        ema_method=tuning.method,
        stage1_peft_metadata=metadata,
        collect_stage1_diagnostics=(bool(getattr(args, "stage1_v5_debug", False)) if is_v5
                                   else bool(getattr(args, "benchmark_evidence_path", None))),
        **v5_kwargs,
    )
    if tuning.method != "full" and args.gradient_checkpointing:
        validate_stage1_checkpoint_context(getattr(trainer, "model", model))
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
    deepspeed_runtime_profile: DeepSpeedRuntimeProfile | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    observability_provider: Callable | None = None,
    rank_config_digest_provider: Callable | None = None,
) -> None:
    """Train once and, when requested, perform one separately timed final save."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if deepspeed_runtime_profile is None:
        profile = bind_deepspeed_runtime_profile(args, world_size=world_size)
    else:
        if getattr(args, "deepspeed_runtime_profile", None) is None:
            args.deepspeed_runtime_profile = deepspeed_runtime_profile
        profile = bind_deepspeed_runtime_profile(args, world_size=world_size)
        if profile is not deepspeed_runtime_profile:
            raise RuntimeError("Stage 1 DeepSpeed runtime profile identity changed")

    producing = bool(
        getattr(args, "benchmark_run_config", None)
        or getattr(args, "benchmark_evidence_path", None)
    )
    run_config_witness = None
    smoke_resume_snapshot = None
    config = None
    if producing:
        from stage1.benchmark import (
            config_digest,
            load_stage1_smoke_resume_witness,
            load_stage1_run_config_witness,
            validate_stage1_runtime_args,
        )

        try:
            if not (
                getattr(args, "benchmark_run_config", None)
                and getattr(args, "benchmark_evidence_path", None)
            ):
                raise ValueError(
                    "benchmark run config and evidence path are required together"
                )
            run_config_witness = load_stage1_run_config_witness(
                Path(args.benchmark_run_config)
            )
            config = validate_stage1_runtime_args(
                run_config_witness,
                args,
                profile,
            )
            smoke_path = getattr(args, "benchmark_smoke_resume_witness", None)
            if config["kind"] == "stage1-smoke-run-manifest":
                if not smoke_path or not getattr(args, "resume_from_checkpoint_path", None):
                    raise ValueError(
                        "smoke benchmark requires a resume checkpoint and witness"
                    )
                smoke_resume_snapshot = load_stage1_smoke_resume_witness(
                    Path(smoke_path),
                    method=config["method"],
                    checkpoint_path=Path(args.resume_from_checkpoint_path),
                    micro_batch=config["micro_batch"],
                    initial_max_steps=3,
                    resumed_max_steps=config["max_steps"],
                )
            elif smoke_path:
                raise ValueError(
                    "smoke resume witness is accepted only by a smoke run manifest"
                )
            local_preflight = {
                "status": "success",
                "config_digest": run_config_witness.config_digest,
                "runtime_digest": config_digest(
                    {
                        "config_digest": run_config_witness.config_digest,
                        "validated_runtime": config,
                        "smoke_resume_marker_digest": (
                            None
                            if smoke_resume_snapshot is None
                            else smoke_resume_snapshot.marker_digest
                        ),
                    }
                ),
                "error_type": None,
                "error_message": None,
            }
        except Exception as error:
            local_preflight = {
                "status": "error",
                "config_digest": (
                    None
                    if run_config_witness is None
                    else run_config_witness.config_digest
                ),
                "runtime_digest": None,
                "error_type": type(error).__name__,
                "error_message": str(error).replace("\0", "\\0")[:4096],
            }
        digest_provider = (
            rank_config_digest_provider or _default_rank_preflight_results
        )
        rank_preflights = tuple(
            digest_provider(local_preflight, 2)
        )
        _require_distributed_preflight_agreement(rank_preflights, world_size=2)
        assert run_config_witness is not None and config is not None
        if config["kind"] == "stage1-smoke-run-manifest":
            assert smoke_resume_snapshot is not None
    total_start = monotonic() if producing else None
    components = component_builder(args)
    model_load_seconds = monotonic() - total_start if producing else 0.0
    resume = components.resume_decision.checkpoint_path
    trainer_start = monotonic()
    train_result = components.trainer.train(
        resume_from_checkpoint=None if resume is None else str(resume)
    )
    trainer_seconds = monotonic() - trainer_start
    if not math.isfinite(trainer_seconds) or trainer_seconds < 0:
        raise RuntimeError("Stage 1 Trainer timing must be finite and non-negative")
    logging.info("Stage 1 Trainer time: %.3fs", trainer_seconds)

    save_seconds = 0.0
    if args.final_save:
        save_start = monotonic()
        components.trainer.save_model(args.save_model_path)
        if components.trainer.is_world_process_zero():
            components.processor.save_pretrained(args.save_model_path)
        save_seconds = monotonic() - save_start
        if not math.isfinite(save_seconds) or save_seconds < 0:
            raise RuntimeError("Stage 1 final save timing must be finite and non-negative")
        logging.info("Stage 1 final save time: %.3fs", save_seconds)

    training_evidence_path = getattr(args, "training_evidence_path", None)
    if training_evidence_path is not None:
        metrics = getattr(train_result, "metrics", None)
        metrics = metrics if isinstance(metrics, dict) else {}
        loss = metrics.get("train_loss")
        state = getattr(components.trainer, "state", None)
        global_step = getattr(state, "global_step", None)
        if profile.zero_stage is None:
            raise ValueError("ordinary Stage 1 evidence requires DeepSpeed")
        evidence = {
            "run_id": Path(args.save_model_path).name,
            "method": args.tuning_method,
            "zero_stage": profile.zero_stage,
            "world_size": profile.world_size,
            "micro_batch": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ema_tau": args.ema_tau,
            "seed": args.seed,
            "global_step": global_step,
            "checkpoint_path": Path(args.save_model_path) / f"checkpoint-{global_step}",
            "loss": loss,
        }
        from stage1.training_evidence import (
            build_stage1_training_evidence,
            publish_stage1_training_evidence,
        )

        build_stage1_training_evidence(**evidence)
        if components.trainer.is_world_process_zero():
            publish_stage1_training_evidence(training_evidence_path, **evidence)

    if not producing:
        return

    from stage1.benchmark import (
        Stage1BenchmarkEvidence,
        _build_artifact_manifest,
        write_stage1_producer_evidence,
    )

    assert run_config_witness is not None
    config = run_config_witness.payload()
    if str(Path(args.save_model_path).resolve()) != config["output_dir"]:
        raise ValueError("producer output path does not match sealed run config")
    if bool(args.final_save) is not config["final_save"]:
        raise ValueError("producer final-save mode does not match sealed run config")
    observer = observability_provider or _default_stage1_observability
    observed = observer(args)
    rank_allocated = tuple(observed["rank_peak_allocated_bytes"])
    rank_reserved = tuple(observed["rank_peak_reserved_bytes"])
    gpu_names = tuple(observed["gpu_names"])
    world_size = config["world_size"]
    if len(rank_allocated) != world_size or len(rank_reserved) != world_size:
        raise ValueError("producer rank memory witness length must equal world size")
    metrics = getattr(train_result, "metrics", None)
    metrics = metrics if isinstance(metrics, dict) else {}
    state = getattr(components.trainer, "state", None)
    optimizer_steps = getattr(state, "global_step", metrics.get("global_step", 0))
    if type(optimizer_steps) is not int:
        optimizer_steps = int(optimizer_steps)
    effective = config["effective_global_batch"]
    if getattr(args, "max_steps", -1) > 0:
        samples_seen = optimizer_steps * effective
    else:
        samples_seen = len(components.dataset) * int(config.get("epochs", args.epochs))
    if not components.trainer.is_world_process_zero():
        return
    artifact = _build_artifact_manifest(
        Path(args.save_model_path), method=config["method"], final_save=args.final_save
    )
    total_seconds = monotonic() - total_start
    if profile.zero_stage is None or profile.config_digest is None:
        raise ValueError("Stage 1 matrix producers require DeepSpeed")
    evidence = Stage1BenchmarkEvidence(
        status="success",
        method=config["method"],
        zero_stage=profile.zero_stage,
        deepspeed_config_digest=profile.config_digest,
        micro_batch=args.per_device_train_batch_size,
        world_size=profile.world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_global_batch=(
            args.per_device_train_batch_size
            * profile.world_size
            * args.gradient_accumulation_steps
        ),
        optimizer_steps=optimizer_steps,
        samples_seen=samples_seen,
        sample_stream_digest=config["audit_digest"],
        model_id=config["model"],
        seed=config["seed"],
        model_load_seconds=float(model_load_seconds),
        trainer_seconds=float(trainer_seconds),
        total_seconds=float(total_seconds),
        final_save_seconds=float(save_seconds),
        samples_per_second=float(metrics.get("train_samples_per_second", 0.0)),
        steps_per_second=float(metrics.get("train_steps_per_second", 0.0)),
        rank_peak_allocated_bytes=rank_allocated,
        rank_peak_reserved_bytes=rank_reserved,
        trainable_parameters=components.parameter_report.trainable_parameters,
        total_parameters=components.parameter_report.total_parameters,
        checkpoint_bytes=artifact["checkpoint_bytes"],
        final_loss=float(metrics.get("train_loss", 0.0)),
        representative_gradient_norm=float(
            getattr(components.trainer, "_representative_gradient_norm", 0.0) or 0.0
        ),
        git_commit=observed["git_commit"],
        dependency_versions=dict(observed["dependency_versions"]),
        gpu_names=gpu_names,
        config_digest=config["config_digest"],
        probe_row_indices=tuple(config["probe_row_indices"]),
    )
    write_stage1_producer_evidence(
        Path(args.benchmark_evidence_path),
        run_config_witness=run_config_witness,
        evidence=evidence,
        smoke_resume_witness=(
            None
            if smoke_resume_snapshot is None
            else smoke_resume_snapshot.summary()
        ),
        artifact_manifest=artifact,
    )


def _require_distributed_preflight_agreement(
    results: tuple[object, ...], *, world_size: int
) -> None:
    expected_fields = {
        "status", "config_digest", "runtime_digest", "error_type", "error_message"
    }
    if len(results) != world_size:
        raise ValueError("Stage 1 distributed preflight rank count mismatch")
    for index, result in enumerate(results):
        if type(result) is not dict or set(result) != expected_fields:
            raise ValueError(
                f"Stage 1 distributed preflight payload mismatch at rank {index}"
            )
        if result["status"] not in {"success", "error"}:
            raise ValueError(
                f"Stage 1 distributed preflight status mismatch at rank {index}"
            )
        if result["status"] == "success":
            for name in ("config_digest", "runtime_digest"):
                value = result[name]
                if (
                    type(value) is not str
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(
                        f"Stage 1 distributed preflight {name} mismatch at rank {index}"
                    )
            if result["error_type"] is not None or result["error_message"] is not None:
                raise ValueError(
                    f"Stage 1 distributed preflight success payload mismatch at rank {index}"
                )
        elif (
            result["runtime_digest"] is not None
            or type(result["error_type"]) is not str
            or type(result["error_message"]) is not str
        ):
            raise ValueError(
                f"Stage 1 distributed preflight error payload mismatch at rank {index}"
            )
    if any(result["status"] == "error" for result in results):
        raise ValueError(
            "Stage 1 distributed preflight failed: "
            + json.dumps(results, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        )
    if len({result["config_digest"] for result in results}) != 1:
        raise ValueError("Stage 1 distributed preflight config digest mismatch")
    if len({result["runtime_digest"] for result in results}) != 1:
        raise ValueError("Stage 1 distributed preflight runtime digest mismatch")


def _default_rank_preflight_results(
    local_result: dict[str, object], world_size: int
) -> tuple[object, ...]:
    if not torch.distributed.is_initialized():
        from accelerate import PartialState

        state = PartialState()
        if state.num_processes != world_size:
            raise ValueError("sealed world size does not match Accelerate rank count")
    gathered: list[object] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local_result)
    return tuple(gathered)


def _default_stage1_observability(args) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark evidence requires CUDA rank observability")
    local = (
        int(torch.cuda.max_memory_allocated()),
        int(torch.cuda.max_memory_reserved()),
        torch.cuda.get_device_name(torch.cuda.current_device()),
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered: list[object] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local)
        ranks = gathered
    else:
        ranks = [local]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    )
    versions = {}
    for package in ("torch", "transformers", "accelerate", "deepspeed", "peft", "trl"):
        versions[package] = importlib_metadata.version(package)
    return {
        "rank_peak_allocated_bytes": tuple(item[0] for item in ranks),
        "rank_peak_reserved_bytes": tuple(item[1] for item in ranks),
        "gpu_names": tuple(item[2] for item in ranks),
        "git_commit": completed.stdout.strip(),
        "dependency_versions": versions,
    }


__all__ = [
    "DEFAULT_STAGE1_DEPENDENCIES",
    "LATENT_TOKEN_STRINGS",
    "Stage1CompositionDependencies",
    "Stage1TrainingComponents",
    "build_stage1_training_arguments",
    "build_stage1_training_components",
    "run_stage1",
    "validate_stage1_checkpoint_context",
]
