import argparse
import logging
import os
import sys
from pathlib import Path

from artifact_paths import resolve_run_artifact_paths, validate_generated_artifact_paths
from deepspeed_runtime import (
    bind_deepspeed_runtime_profile,
    install_zero3_frozen_checkpoint_compatibility,
    install_zero3_stage2_output_compatibility,
)


def _preflight_direct_main(argv: list[str]) -> object:
    """Validate generated ownership before importing the training runtime."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--save_model_path", default="./checkpoints/model_stage1")
    parser.add_argument("--log_file")
    parser.add_argument("--benchmark_run_config")
    parser.add_argument("--benchmark_evidence_path")
    parser.add_argument("--benchmark_smoke_resume_witness")
    parser.add_argument("--deepspeed_config")
    args, _unknown = parser.parse_known_args(argv)
    if args.log_file is None:
        try:
            paths = resolve_run_artifact_paths(Path(args.save_model_path).name)
        except ValueError:
            pass
        else:
            args.log_file = str(paths.log_dir / "train.log")
    validate_generated_artifact_paths(args)
    return bind_deepspeed_runtime_profile(args)


_DIRECT_MAIN_PROFILE = (
    _preflight_direct_main(sys.argv[1:]) if __name__ == "__main__" else None
)

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stage2_objective.execution import ObjectiveRuntime
    from stage2_objective.metadata import ObjectiveMetadata

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoConfig,
    AutoProcessor,
)
from transformers.utils.peft_utils import find_adapter_config_file
from trl import SFTConfig
from peft import PeftConfig

from qwen_vl_utils import process_vision_info

from utils_deepseed import *
from task_deepseed import *
from stage1.checkpointing import (
    decode_stage1_peft_metadata_snapshot,
    preflight_stage1_peft_artifact,
)
from stage2.checkpointing import (
    STAGE2_PEFT_METADATA_FILENAME,
    STEP_PREFERENCE_SCHEMA_VERSION,
    Stage2PeftMetadata,
    StepPreferenceMetadata,
    build_stage2_peft_metadata,
    configure_stage2_metadata,
    digest_small_metadata,
    load_stage2a_lifecycle_identity,
    load_step_preference_metadata,
    prepare_stage2_sparse_token_base,
    resolve_latent_token_ids,
    resolve_stage2_load_decision,
    validate_stage2_peft_compatibility,
    validate_stage2_sparse_processor,
    validate_step_preference_resume_identity,
)
from stage2.data import (
    Stage2DataCollator,
    build_prepared_example_length_measure,
    filter_stage2_examples,
    load_jsonl_examples,
    prepare_stage2_example,
)
from stage2.trainer import Stage2Trainer
from step_preference.config import (
    StepPreferenceConfig,
    preference_prefix_source,
    resolve_step_preference_config,
    selected_phase_steps,
    validate_insertion_runtime,
    validate_preference_rollout,
)
from stage2.peft import (
    build_stage2_tuned_model,
    load_stage1_dora_for_fresh_stage2,
    resolve_stage2_target_module_names,
    validate_gated_execution_peft_identity,
    validate_stage2_parameter_boundary,
)
from step_preference.gated import GatedExecutionIdentity, gated_execution_identity
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    stage2_prefix_dropout_semantics,
)


@dataclass(frozen=True)
class TrainingPath:
    trainer_class: type
    collator_factory: object


def select_training_path(stage: str) -> TrainingPath:
    if stage == "stage1":
        from stage1.trainer import CustomTrainerStage1

        return TrainingPath(CustomTrainerStage1, collate_fn_stage1)
    if stage == "stage2":
        return TrainingPath(Stage2Trainer, Stage2DataCollator)
    raise ValueError(f"Unsupported training stage: {stage}")


@dataclass(frozen=True)
class Stage2TrainingComponents:
    model: object
    processor: object
    train_dataset: object
    data_collator: Stage2DataCollator
    trainer_class: type[Stage2Trainer]
    trainer_resume_path: str | None
    peft_metadata: Stage2PeftMetadata | None
    step_preference_config: StepPreferenceConfig | None = None
    step_preference_metadata: StepPreferenceMetadata | None = None
    objective_runtime: "ObjectiveRuntime | None" = None
    objective_metadata: "ObjectiveMetadata | None" = None


@dataclass(frozen=True)
class Stage2SourceDependencies:
    """All source-sensitive Stage 2 loaders, injected as one safe bundle."""

    find_adapter_config: object
    load_config: object
    load_processor: object
    load_model: object


def _default_stage2_source_dependencies() -> Stage2SourceDependencies:
    """Bind production loaders at call time so tests can replace boundaries."""

    return Stage2SourceDependencies(
        find_adapter_config=find_adapter_config_file,
        load_config=AutoConfig.from_pretrained,
        load_processor=AutoProcessor.from_pretrained,
        load_model=Qwen2_5_VLForConditionalGeneration.from_pretrained,
    )


def build_stage2_source_provenance_kwargs(args) -> Mapping[str, object]:
    """Return one immutable set of Hub selectors for sparse source loading."""

    return MappingProxyType(
        {
            name: getattr(args, name)
            for name in ("cache_dir", "revision", "token", "local_files_only")
            if hasattr(args, name) and getattr(args, name) is not None
        }
    )


def _stage2_gradient_checkpointing_kwargs(
    *, gradient_checkpointing: bool
) -> dict[str, bool] | None:
    if not gradient_checkpointing:
        return None
    return {"use_reentrant": False}


def configure_stage2_model_backend(
    model,
    *,
    gradient_checkpointing: bool,
    latent_activation_checkpointing: bool = False,
) -> None:
    """Apply Stage 2 cache/checkpoint settings outside the latent recurrence."""
    checkpointing_kwargs = _stage2_gradient_checkpointing_kwargs(
        gradient_checkpointing=gradient_checkpointing,
    )
    model.config.use_cache = False
    if latent_activation_checkpointing and gradient_checkpointing:
        raise ValueError("latent activation checkpointing requires native gradient checkpointing disabled")
    model.config.stage2_latent_activation_checkpointing = latent_activation_checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=checkpointing_kwargs
        )
    else:
        model.gradient_checkpointing_disable()


def validate_stage2_distributed_settings(
    per_device_train_batch_size: int,
) -> None:
    """Enforce the rank-local batch contract used by latent rollout."""
    if per_device_train_batch_size != 1:
        raise ValueError(
            "Stage 2 train_micro_batch_size_per_gpu must resolve to 1"
        )


def validate_stage2_backend_settings(
    rollout_backend: str | None,
    gradient_checkpointing: bool,
    stage2_cache_activation_checkpointing: bool,
    stage2_latent_activation_checkpointing: bool = False,
) -> None:
    """Reject incompatible Stage 2 cache and checkpointing combinations."""
    if type(stage2_latent_activation_checkpointing) is not bool:
        raise ValueError("latent activation checkpointing must be boolean")
    if stage2_latent_activation_checkpointing and (
        gradient_checkpointing or rollout_backend not in (None, "no_cache")
    ):
        raise ValueError("latent activation checkpointing requires no_cache and native gradient checkpointing disabled")
    if (
        stage2_cache_activation_checkpointing
        and rollout_backend != "differentiable_cache"
    ):
        raise ValueError(
            "stage2_cache_activation_checkpointing requires differentiable_cache"
        )
    if rollout_backend == "differentiable_cache" and gradient_checkpointing:
        raise ValueError(
            "differentiable_cache is incompatible with global gradient checkpointing"
        )


def _objective_rank0_prepare(producer):
    """Transport one producer's credential-free result or error to all ranks."""
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed:
        return producer()
    envelope = [None]
    if torch.distributed.get_rank() == 0:
        try:
            envelope[0] = (True, producer())
        except Exception as error:
            envelope[0] = (False, f"{type(error).__name__}: {error}")
    torch.distributed.broadcast_object_list(envelope, src=0)
    success, payload = envelope[0]
    if not success:
        raise ValueError(f"objective source preparation failed: {payload}")
    return payload


def build_stage2_training_components(
    args, *, source_dependencies: Stage2SourceDependencies | None = None
) -> Stage2TrainingComponents:
    from stage2_objective.config import (config_from_args, validate_objective_runtime,
                                         validate_objective_ddp_matrix)
    objective_config = config_from_args(args)
    split_attention = getattr(args, "stage2_split_attention", False)
    attention_implementation = "sdpa" if split_attention else args.attention_implementation
    validate_stage2_backend_settings(
        getattr(args, "stage2_rollout_backend", None),
        args.gradient_checkpointing,
        getattr(args, "stage2_cache_activation_checkpointing", False),
        getattr(args, "stage2_latent_activation_checkpointing", False),
    )
    runtime_profile = bind_deepspeed_runtime_profile(args)
    phase = getattr(args, "stage2_phase", "standard")
    preference = resolve_step_preference_config(
        phase=phase,
        attention_implementation=attention_implementation,
        stage2a_checkpoint_path=getattr(args, "stage2a_checkpoint_path", None),
        step_negatives_path=getattr(args, "step_negatives_path", None),
        beta=getattr(args, "step_preference_beta", 0.1),
        lambda_preference=getattr(args, "step_preference_lambda_preference", 1.0),
        lambda_entropy=getattr(args, "step_preference_lambda_entropy", 0.1),
        entropy_margin=getattr(args, "step_preference_entropy_margin", 0.1),
        preference_margin=getattr(args, "step_preference_margin", 0.0),
        ablation=getattr(args, "step_preference_ablation", "attention_mask"),
        modalities=getattr(args, "step_preference_modalities", "visual_text"),
        target_scope=getattr(args, "step_preference_target_scope", "until_next_latent"),
        prefix_mode=getattr(args, "step_preference_prefix_mode", "clean_prefix"),
        branch_mode=getattr(args, "step_preference_branch_mode", "independent"),
        on_layer_checkpoint_stride=getattr(args, "step_on_layer_checkpoint_stride", 1),
    )
    if objective_config.enabled:
        validate_objective_runtime(objective_config,
            stage=getattr(args, "stage", "stage2"), phase=phase,
            backend=args.attention_implementation,
            rollout_backend=getattr(args, "stage2_rollout_backend", None) or "no_cache",
            split_attention=split_attention, microbatch_size=args.per_device_train_batch_size,
            zero_stage=runtime_profile.zero_stage, offload=False,
            old_preference_enabled=preference.enabled,
            legacy_prefix=(preference.prefix_mode if preference.prefix_mode != "clean_prefix" else "none"),
            legacy_ablation=(preference.ablation if getattr(args, "stage2_objective_explicit_legacy", False) else "none"),
            native_checkpointing=args.gradient_checkpointing,
            latent_checkpointing=getattr(args, "stage2_latent_activation_checkpointing", False),
            cache_checkpointing=getattr(args, "stage2_cache_activation_checkpointing", False))
        # Reject unvalidated DDP before initializing source/model dependencies.
        world_size = (torch.distributed.get_world_size()
                      if torch.distributed.is_initialized() else int(os.environ.get("WORLD_SIZE", "1")))
        if runtime_profile.zero_stage is None and world_size > 1:
            cpu = (getattr(args, "use_cpu", False)
                   or os.environ.get("ACCELERATE_USE_CPU", "false").lower() in {"1", "true", "yes"})
            validate_objective_ddp_matrix(objective_config,
                kind="MULTI_CPU" if cpu else "MULTI_GPU", backend=attention_implementation,
                native_checkpointing=args.gradient_checkpointing,
                latent_checkpointing=getattr(args, "stage2_latent_activation_checkpointing", False))
        # Profile binding has already rejected every offload policy.
        from accelerate.state import PartialState
        objective_state = PartialState(cpu=getattr(args, "use_cpu", False))
    else:
        objective_state = None
    validate_insertion_runtime(
        preference=preference,
        microbatch_size=args.per_device_train_batch_size,
        tuning_method=args.tuning_method or "full",
        zero_stage=runtime_profile.zero_stage,
    )
    source_dependencies = (
        source_dependencies or _default_stage2_source_dependencies()
    )
    source_provenance_kwargs = build_stage2_source_provenance_kwargs(args)
    resume_lifecycle = None
    if args.resume_from_checkpoint_path is not None and phase != "standard":
        resume_lifecycle = load_step_preference_metadata(
            Path(args.resume_from_checkpoint_path)
        )
        if resume_lifecycle.phase != phase:
            raise ValueError("cannot resume with changed phase")
    phase_steps = selected_phase_steps(
        phase,
        getattr(args, "stage2_total_optimizer_steps", None),
        getattr(args, "stage2a_ratio", None),
    )
    if phase_steps is not None and args.max_steps != phase_steps:
        raise ValueError("max_steps disagrees with the selected Stage 2 phase budget")
    if phase_steps is not None:
        args.max_steps = phase_steps

    selected_examples = None

    decision = resolve_stage2_load_decision(
        load_model_path=args.load_model_path,
        stage1_adapter_path=getattr(args, "stage1_adapter_path", None),
        stage2a_checkpoint_path=(
            getattr(args, "stage2a_checkpoint_path", None)
            if args.resume_from_checkpoint_path is None
            else None
        ),
        resume_from_checkpoint_path=args.resume_from_checkpoint_path,
        load_config=source_dependencies.load_config,
        discover_adapter_config=source_dependencies.find_adapter_config,
        source_provenance_kwargs=source_provenance_kwargs,
        latent_size=args.latent_size,
        supervise_latent_boundaries=args.supervise_latent_boundaries,
        tuning_method=args.tuning_method,
        tuning_method_explicit=args.tuning_method_explicit,
        rollout_backend=getattr(args, "stage2_rollout_backend", None),
        rollout_backend_explicit=getattr(
            args, "stage2_rollout_backend_explicit", False
        ),
        stage2_cache_activation_checkpointing=getattr(
            args, "stage2_cache_activation_checkpointing", None
        ),
        stage2_cache_activation_checkpointing_explicit=getattr(
            args, "stage2_cache_activation_checkpointing_explicit", False
        ),
        execution_backend=(
            preference.execution_backend if preference.enabled else None
        ),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )
    if objective_config.enabled and (decision.rollout_backend != "no_cache"
            or decision.stage2_cache_activation_checkpointing):
        raise ValueError("objective requires resolved no_cache backend without cache checkpointing")
    # An omitted CLI tuning method may inherit PEFT from the resolved artifact.
    validate_insertion_runtime(
        preference=preference,
        microbatch_size=args.per_device_train_batch_size,
        tuning_method=decision.tuning.method,
        zero_stage=runtime_profile.zero_stage,
    )
    validate_preference_rollout(
        preference=preference,
        rollout_backend=decision.rollout_backend,
        native_checkpointing=args.gradient_checkpointing,
        latent_checkpointing=getattr(
            args, "stage2_latent_activation_checkpointing", False
        ),
        cache_checkpointing=decision.stage2_cache_activation_checkpointing,
        microbatch_size=args.per_device_train_batch_size,
        tuning_method=decision.tuning.method,
        zero_stage=runtime_profile.zero_stage,
        on_layer_checkpoint_stride=preference.on_layer_checkpoint_stride,
    )
    validate_stage2_backend_settings(
        decision.rollout_backend,
        args.gradient_checkpointing,
        decision.stage2_cache_activation_checkpointing,
        getattr(args, "stage2_latent_activation_checkpointing", False),
    )
    gated_execution = (
        gated_execution_identity(
            rollout_backend=decision.rollout_backend,
            native_checkpointing=args.gradient_checkpointing,
            latent_checkpointing=getattr(
                args, "stage2_latent_activation_checkpointing", False
            ),
            cache_checkpointing=decision.stage2_cache_activation_checkpointing,
            ablation=preference.ablation,
        )
        if preference.prefix_mode == "single_image_latent_gate"
        else None
    )
    step_preference_metadata = _build_step_preference_metadata(
        args=args,
        phase=phase,
        phase_steps=phase_steps,
        preference=preference,
        gated_execution=gated_execution,
    )
    if resume_lifecycle is not None:
        validate_step_preference_resume_identity(
            resume_lifecycle, step_preference_metadata
        )
    if decision.trainer_resume_path is not None:
        validate_gated_execution_peft_identity(
            step_preference_metadata, decision.peft_metadata
        )
    args.zero3_stage2_output_compatibility = (
        install_zero3_stage2_output_compatibility(runtime_profile)
    )
    args.zero3_frozen_checkpoint_compatibility = (
        install_zero3_frozen_checkpoint_compatibility(
            runtime_profile,
            gradient_checkpointing=(
                args.gradient_checkpointing
                or getattr(args, "stage2_latent_activation_checkpointing", False)
                or preference.prefix_mode == "v_off_insertion"
            ),
        )
    )
    logging.info(
        "Stage 2 backend settings: stage2_rollout_backend=%s "
        "clean_on_backend=%s "
        "prefix_dropout_semantics=%s "
        "stage2_cache_activation_checkpointing=%s "
        "step_preference_execution_backend=%s "
        "gradient_checkpointing=%s "
        "stage2_latent_activation_checkpointing=%s "
        "step_on_layer_checkpoint_stride=%s",
        decision.rollout_backend,
        decision.rollout_backend,
        stage2_prefix_dropout_semantics(decision.rollout_backend),
        decision.stage2_cache_activation_checkpointing,
        decision.execution_backend,
        args.gradient_checkpointing,
        getattr(args, "stage2_latent_activation_checkpointing", False),
        (preference.on_layer_checkpoint_stride
         if preference.prefix_mode == "v_off_insertion" else "not_applicable"),
    )
    if split_attention:
        logging.info("Stage 2 split attention: language=sdpa vision=flash_attention_2 sidecar=flex_attention")
    sparse_resume = (
        decision.peft_metadata is not None
        and decision.peft_metadata.token_row_source == "sparse_adapter"
    )
    sparse_source_kwargs = source_provenance_kwargs if (sparse_resume or objective_config.enabled) else {}
    processor_source = (
        step_preference_metadata.stage2a_artifact
        if (preference.enabled and step_preference_metadata is not None
            and step_preference_metadata.stage2a_artifact is not None)
        else decision.processor_source
    )
    processor = source_dependencies.load_processor(
        processor_source, **sparse_source_kwargs
    )
    if sparse_resume:
        validate_stage2_sparse_processor(
            processor.tokenizer, decision.peft_metadata
        )
    stage1_preflight = None
    stage1_metadata = None
    if decision.stage1_adapter_source is not None:
        def prepare_student_source():
            base_config = source_dependencies.load_config(
                decision.base_model_source, **(sparse_source_kwargs if objective_config.enabled else {}))
            return preflight_stage1_peft_artifact(
                processor, base_config, Path(decision.stage1_adapter_source),
                requested_base_model_reference=decision.base_model_source,
                metadata_snapshot=decision.stage1_metadata_snapshot)
        stage1_preflight = (_objective_rank0_prepare(prepare_student_source)
                            if objective_config.enabled else prepare_student_source())
        stage1_metadata = decode_stage1_peft_metadata_snapshot(
            decision.stage1_metadata_snapshot
        )
    model_only_kwargs = {
        "attn_implementation": attention_implementation,
    }
    if split_attention:
        model_only_kwargs["stage2_split_attention"] = True
    base_model = source_dependencies.load_model(
        decision.base_model_source,
        **sparse_source_kwargs,
        **model_only_kwargs,
    )
    if decision.stage1_adapter_source is not None:
        model = load_stage1_dora_for_fresh_stage2(
            base_model,
            processor,
            decision.stage1_adapter_source,
            decision.tuning,
            requested_base_model_reference=decision.base_model_source,
            preflight=stage1_preflight,
        )
        token_ids = resolve_latent_token_ids(processor.tokenizer, base_model.config)
    elif (
        decision.peft_metadata is not None
        and decision.peft_metadata.token_row_source == "sparse_adapter"
    ):
        token_ids = prepare_stage2_sparse_token_base(
            base_model,
            processor.tokenizer,
            decision.peft_metadata,
        )
    else:
        token_ids = resolve_latent_token_ids(processor.tokenizer, base_model.config)
    if decision.peft_metadata is not None:
        adapter_config = PeftConfig.from_pretrained(decision.adapter_source)
        # Stage 1 predates this Stage 2 CE policy. Inherit only an absent
        # value from the validated load decision; preserve explicit mismatches
        # and validate the Stage 1 architecture before configuring Stage 2.
        if (getattr(base_model.config, "stage", None) == "stage1"
                and getattr(base_model.config, "supervise_latent_boundaries", None) is None):
            base_model.config.supervise_latent_boundaries = decision.supervise_latent_boundaries
        validate_stage2_peft_compatibility(
            base_model,
            processor.tokenizer,
            decision.peft_metadata,
            adapter_config,
            decision.adapter_source,
        )
    configure_stage2_metadata(
        base_model.config,
        token_ids,
        latent_size=decision.latent_size,
        supervise_latent_boundaries=decision.supervise_latent_boundaries,
        rollout_backend=decision.rollout_backend,
        stage2_cache_activation_checkpointing=(
            decision.stage2_cache_activation_checkpointing
        ),
        execution_backend=decision.execution_backend,
    )
    configure_stage2_model_backend(
        base_model,
        gradient_checkpointing=args.gradient_checkpointing,
        **({"latent_activation_checkpointing": True}
           if getattr(args, "stage2_latent_activation_checkpointing", False) else {}),
    )
    base_model.visual.requires_grad_(False)
    fresh_resolved_target_names = (
        stage1_metadata.resolved_target_names
        if stage1_metadata is not None
        else (
            resolve_stage2_target_module_names(
                base_model, decision.tuning.target_modules
            )
            if decision.peft_metadata is None and decision.tuning.is_peft
            else None
        )
    )
    if decision.stage1_adapter_source is None:
        if (
            decision.peft_metadata is not None
            and decision.peft_metadata.token_row_source == "sparse_adapter"
        ):
            model = build_stage2_tuned_model(
                base_model,
                decision.tuning,
                adapter_path=decision.adapter_source,
                is_trainable=True,
                token_row_source="sparse_adapter",
                latent_token_ids=(
                    token_ids.pad_token_id,
                    token_ids.start_token_id,
                    token_ids.end_token_id,
                ),
            )
        else:
            model = build_stage2_tuned_model(
                base_model,
                decision.tuning,
                adapter_path=decision.adapter_source,
                is_trainable=True,
            )
    sparse_continuation = (
        decision.stage1_adapter_source is not None
        or decision.peft_metadata is not None
        and decision.peft_metadata.token_row_source == "sparse_adapter"
    )
    parameter_report = validate_stage2_parameter_boundary(
        model,
        decision.tuning,
        allow_sparse_token_rows=sparse_continuation,
    )
    logging.info(
        "Stage 2 parameters: total=%d trainable=%d resolved_targets=%s",
        parameter_report.total_parameters,
        parameter_report.trainable_parameters,
        parameter_report.resolved_target_names,
    )
    peft_metadata = decision.peft_metadata
    if (
        peft_metadata is not None
        and preference.enabled
        and decision.trainer_resume_path is None
    ):
        # Stage 2A warm-starts inherit authenticated adapter provenance, while
        # these run-owned fields describe the newly composed Stage 2B run.
        peft_metadata = Stage2PeftMetadata.from_dict(
            {
                **peft_metadata.to_dict(),
                "rollout_backend": decision.rollout_backend,
                "prefix_dropout_semantics": stage2_prefix_dropout_semantics(
                    decision.rollout_backend
                ),
                "stage2_cache_activation_checkpointing": (
                    decision.stage2_cache_activation_checkpointing
                ),
                "execution_backend": decision.execution_backend,
            }
        )
    if peft_metadata is None and decision.tuning.is_peft:
        peft_metadata = build_stage2_peft_metadata(
            tuning=decision.tuning,
            base_model_reference=decision.base_model_source,
            model_config=base_model.config,
            token_ids=token_ids,
            resolved_target_names=fresh_resolved_target_names,
            token_row_source=(
                "sparse_adapter" if stage1_metadata is not None else "base"
            ),
            stage1_adapter_reference=(
                str(stage1_preflight.artifact_dir)
                if stage1_preflight is not None
                else None
            ),
            tokenizer_base_vocab_size=(
                stage1_metadata.base_vocab_size
                if stage1_metadata is not None
                else None
            ),
            pretrained_model_vocab_size=(
                stage1_metadata.pretrained_model_vocab_size
                if stage1_metadata is not None
                else None
            ),
            resized_vocab_size=(
                stage1_metadata.resized_vocab_size
                if stage1_metadata is not None
                else None
            ),
            source_base_model_architecture=(
                stage1_metadata.base_model_architecture
                if stage1_metadata is not None
                else None
            ),
            rollout_backend=decision.rollout_backend,
            stage2_cache_activation_checkpointing=(
                decision.stage2_cache_activation_checkpointing
            ),
            execution_backend=decision.execution_backend,
        )
    validate_gated_execution_peft_identity(
        step_preference_metadata, peft_metadata
    )

    objective_runtime = objective_metadata = None
    trainer_class = Stage2Trainer
    if objective_config.enabled:
        from stage2_objective import reference as objective_reference
        from stage2_objective.execution import ObjectiveRuntime, apply_objective_dropout_policy
        from stage2_objective.metadata import make_objective_metadata, validate_objective_resume
        from stage2_objective.trainer import Stage2ObjectiveTrainer
        from stage2_objective.vocabulary import build_text_vocabulary
        vocabulary = build_text_vocabulary(processor.tokenizer, model.config,
            embedding_rows=model.get_input_embeddings().weight.shape[0],
            output_rows=model.get_output_embeddings().weight.shape[0])
        reference = None
        if objective_config.lambda_dyn > 0:
            reference_loaders = objective_reference.ReferenceLoaders(
                source_dependencies.load_config, source_dependencies.load_processor,
                source_dependencies.load_model)
            def prepare_fixed_source():
                matching_preflight = (stage1_preflight if stage1_preflight is not None
                    and stage1_preflight.artifact_dir == Path(objective_config.reference_model_path).resolve()
                    else None)
                source = objective_reference.prepare_reference_source(
                    Path(objective_config.reference_model_path), processor=processor,
                    vocabulary=vocabulary, loaders=reference_loaders,
                    source_kwargs=dict(source_provenance_kwargs), witnessed_sources={},
                    stage1_preflight=matching_preflight)
                # Keep shared preflight identity within one pickle object graph.
                # Never transport source.load_kwargs (it can contain a Hub token).
                return source.identity, source.stage1_preflight, source._authentication
            identity, reference_preflight, authentication = _objective_rank0_prepare(prepare_fixed_source)
            source = objective_reference.ReferenceSource(identity, reference_preflight,
                dict(source_provenance_kwargs), authentication)
            reference = objective_reference.load_frozen_reference(source, student=model,
                processor=processor, vocabulary=vocabulary, loaders=reference_loaders,
                device=objective_state.device, dtype=next(model.parameters()).dtype,
                backend=attention_implementation, latent_size=decision.latent_size,
                supervise_latent_boundaries=decision.supervise_latent_boundaries,
                split_attention=split_attention)
        objective_runtime = ObjectiveRuntime(objective_config, vocabulary, reference,
            apply_objective_dropout_policy(model),
            visual_ablation=getattr(args, 'stage2_visual_ablation', 'latent_body'))
        objective_metadata = make_objective_metadata(objective_runtime,
            backend=("sdpa_vision_flash_attention_2" if split_attention else attention_implementation),
            latent_size=decision.latent_size,
            supervise_latent_boundaries=decision.supervise_latent_boundaries)
        if decision.trainer_resume_path is not None:
            validate_objective_resume(Path(decision.trainer_resume_path), objective_metadata)
        trainer_class = Stage2ObjectiveTrainer

    if selected_examples is None:
        raw_examples = load_jsonl_examples(Path(args.data_path))
        selected_examples = (
            raw_examples
            if args.max_train_samples is None
            else raw_examples[: args.max_train_samples]
        )
    prepared = [
        prepare_stage2_example(
            item,
            image_root=Path(args.image_root),
            latent_size=decision.latent_size,
        )
        for item in selected_examples
    ]
    kept, _length_stats = filter_stage2_examples(
        examples=prepared,
        measure_length=build_prepared_example_length_measure(
            processor, **({"target_scope": preference.target_scope}
                          if preference.enabled and preference.target_scope == "immediate_text" else {})),
        max_length=args.max_seq_length_train,
    )
    collator = Stage2DataCollator(
        processor,
        token_ids,
        decision.latent_size,
        decision.supervise_latent_boundaries,
        **({"context_preference_enabled": True, "target_scope": preference.target_scope}
           if preference.enabled else {"step_negative_index": None}),
        **({"objective_config": objective_config, "objective_vocabulary": vocabulary}
           if objective_config.enabled else {}),
    )
    return Stage2TrainingComponents(
        model=model,
        processor=processor,
        train_dataset=kept,
        data_collator=collator,
        trainer_class=trainer_class,
        trainer_resume_path=decision.trainer_resume_path,
        peft_metadata=peft_metadata,
        step_preference_config=preference,
        step_preference_metadata=step_preference_metadata,
        objective_runtime=objective_runtime,
        objective_metadata=objective_metadata,
    )


def _build_step_preference_metadata(
    *,
    args,
    phase: str,
    phase_steps: int | None,
    preference: StepPreferenceConfig,
    gated_execution: GatedExecutionIdentity | None = None,
) -> StepPreferenceMetadata | None:
    if phase == "standard":
        return None
    if phase_steps is None:
        raise ValueError("Stage 2 lifecycle phase requires an optimizer budget")
    stage2a_artifact = None
    stage2a_metadata_digest = None
    stage2a_lifecycle_metadata_digest = None
    if preference.enabled and getattr(args, "stage2a_checkpoint_path", None) is not None:
        requested_stage2a = _resolve_stage2a_artifact_path(
            args.stage2a_checkpoint_path
        )
        lifecycle_identity = load_stage2a_lifecycle_identity(
            requested_stage2a,
            total_optimizer_steps=args.stage2_total_optimizer_steps,
            stage2a_ratio=args.stage2a_ratio,
        )
        stage2a_artifact = str(requested_stage2a)
        live_stage2a_digest = digest_small_metadata(
            _stage2a_small_metadata_path(requested_stage2a)
        )
        stage2a_metadata_digest = live_stage2a_digest
        stage2a_lifecycle_metadata_digest = lifecycle_identity.digest
    elif preference.enabled and args.stage2a_ratio != 0:
        raise ValueError("direct preference without Stage 2A requires stage2a_ratio=0")
    return StepPreferenceMetadata(
        schema_version=STEP_PREFERENCE_SCHEMA_VERSION,
        phase=phase,
        total_optimizer_steps=args.stage2_total_optimizer_steps,
        phase_optimizer_steps=phase_steps,
        stage2a_ratio=args.stage2a_ratio,
        stage2a_artifact=stage2a_artifact,
        stage2a_metadata_digest=stage2a_metadata_digest,
        stage2a_lifecycle_metadata_digest=stage2a_lifecycle_metadata_digest,
        beta=preference.beta,
        lambda_preference=preference.lambda_preference,
        lambda_entropy=preference.lambda_entropy,
        entropy_margin=preference.entropy_margin,
        ablation=preference.ablation,
        modalities=preference.modalities,
        target_scope=preference.target_scope,
        preference_margin=preference.preference_margin,
        execution_backend=(
            preference.execution_backend if preference.enabled else None
        ),
        sidecar_prefix_source=(
            preference_prefix_source(preference.prefix_mode)
            if preference.enabled else None
        ),
        prefix_mode=preference.prefix_mode,
        branch_mode=preference.branch_mode,
        on_layer_checkpoint_stride=preference.on_layer_checkpoint_stride,
        gated_execution=gated_execution,
    )


def _resolve_stage2a_artifact_path(path: str | Path) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"Stage 2A checkpoint does not exist: {path}") from error
    if not resolved.is_dir():
        raise ValueError(f"Stage 2A checkpoint is not a directory: {path}")
    return resolved


def _stage2a_small_metadata_path(stage2a_artifact: Path) -> Path:
    peft_metadata = stage2a_artifact / STAGE2_PEFT_METADATA_FILENAME
    if peft_metadata.exists() or peft_metadata.is_symlink():
        return peft_metadata
    return stage2a_artifact / "config.json"


def build_stage2_training_arguments(args) -> SFTConfig:
    validate_stage2_distributed_settings(args.per_device_train_batch_size)
    validate_stage2_backend_settings(
        getattr(args, "stage2_rollout_backend", None),
        args.gradient_checkpointing,
        getattr(args, "stage2_cache_activation_checkpointing", False),
        getattr(args, "stage2_latent_activation_checkpointing", False),
    )
    checkpointing_kwargs = _stage2_gradient_checkpointing_kwargs(
        gradient_checkpointing=args.gradient_checkpointing,
    )
    training_args = SFTConfig(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=(2 if args.dataloader_num_workers > 0 else None),
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        warmup_steps=args.warm_up_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=20,
        logging_first_step=(getattr(args, "step_preference_prefix_mode", None) == "single_image_latent_gate"),
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        optim="adamw_torch",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=args.max_seq_length_train,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=checkpointing_kwargs,
        deepspeed=args.deepspeed_config,
        ddp_find_unused_parameters=False,
    )
    training_args.stage2_debugging = getattr(args, "stage2_debugging", False)
    return training_args

# ==============================================================
# Collate Functions
# ==============================================================

def collate_fn_stage1(examples, processor, args):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    texts = [place_input_image(t) for t in texts]
    texts = [place_output_image(t) for t in texts]
    texts = replace_visual_spectial_tokens(texts)

    image_inputs, _ = process_vision_info(examples)

    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    assistant_examples = remove_user_images(examples)
    assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    assistant_text = replace_visual_spectial_tokens(assistant_text)
    assistant_image_inputs, _ = process_vision_info(assistant_examples)
    assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    if 'pixel_values' in user_batch:
        batch['pixel_values'] = user_batch['pixel_values']
        batch['image_grid_thw'] = user_batch['image_grid_thw']
    
    if 'pixel_values' in assistant_batch:
        batch['pixel_values_latent'] = assistant_batch['pixel_values']
        batch['image_grid_thw_latent'] = assistant_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_end_idx   = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0,0].item()
    pad_token_idx    = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0,0].item()

    new_input_ids, new_attention_mask = process_batch(
        batch["input_ids"], batch["attention_mask"],
        start_token=latent_start_idx, end_token=latent_end_idx,
        replacement_token=latent_token_idx, replacement_length=args.latent_size,
        pad_token=pad_token_idx
    )
    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask
    
    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_with_latent_template(
        batch["input_ids"],
        answer_start_token_pattern,
        pad_token_idx,
        int(latent_start_idx),
        int(latent_end_idx),
        int(latent_token_idx),
        latent_ce_ratio=0,
    )
    batch["labels"] = labels

    image_out_mask = mask_latent_output_tokens_all_segments(
        batch["input_ids"], latent_start_idx, latent_end_idx, latent_token_idx
    )
    batch["image_out_mask"] = image_out_mask
    return batch

    

def run_stage1(args):
    """Load the Stage 1 runtime only after Stage 1 was selected."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    profile = bind_deepspeed_runtime_profile(args, world_size=world_size)
    from stage1.composition import run_stage1 as run_composed_stage1

    return run_composed_stage1(
        args, deepspeed_runtime_profile=profile
    )


def run_stage2(args):
    """Run the existing Stage 2 path without importing Stage 1 Trainer code."""

    profile = bind_deepspeed_runtime_profile(
        args, world_size=int(os.environ.get("WORLD_SIZE", "1"))
    )
    components = build_stage2_training_components(args)
    trainer = components.trainer_class(
        model=components.model,
        args=build_stage2_training_arguments(args),
        train_dataset=components.train_dataset,
        data_collator=components.data_collator,
        processing_class=components.processor,
        stage2_peft_metadata=components.peft_metadata,
        step_preference_config=components.step_preference_config,
        step_preference_metadata=components.step_preference_metadata,
        deepspeed_runtime_profile=profile,
        **({"objective_runtime": components.objective_runtime,
            "objective_metadata": components.objective_metadata}
           if components.objective_runtime is not None else {}),
    )
    trainer.train(resume_from_checkpoint=components.trainer_resume_path)
    if args.save_final_model:
        trainer.save_model(args.save_model_path)
        if trainer.is_world_process_zero():
            components.processor.save_pretrained(args.save_model_path)


def route_training(args, *, stage1_handler=None, stage2_handler=None):
    """Select exactly one public stage handler and return its observable result."""

    if getattr(args, "enable_stage2_objective", False) and args.stage != "stage2":
        raise ValueError("objective requires Stage 2 standard phase")
    if args.stage == "stage1":
        return (stage1_handler or run_stage1)(args)
    if args.stage == "stage2":
        return (stage2_handler or run_stage2)(args)
    raise ValueError(f"Unsupported stage: {args.stage}")


# ==============================================================
# Main Training Function
# ==============================================================
def main_train():
    args = get_args()
    validate_generated_artifact_paths(args)
    if _DIRECT_MAIN_PROFILE is not None:
        args.deepspeed_runtime_profile = _DIRECT_MAIN_PROFILE
    bind_deepspeed_runtime_profile(args)
    seed_everything(seed=42 if args.stage == "stage2" else args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(args.log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ],
    )
    logging.info('==' * 20)
    logging.info(args)
    logging.info('==' * 20)

    return route_training(args)


def __getattr__(name):
    """Retain the historical public trainer import without eager Stage 1 load."""

    if name == "CustomTrainerStage1":
        from stage1.trainer import CustomTrainerStage1

        return CustomTrainerStage1
    raise AttributeError(name)

if __name__ == "__main__":
    main_train()
