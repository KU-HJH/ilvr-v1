import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
from pathlib import Path

_platform_python_version = platform.python_version
if _platform_python_version().endswith("+"):
    platform.python_version = lambda: _platform_python_version().rstrip("+")

from artifact_paths import resolve_run_artifact_paths, validate_generated_artifact_paths
from deepspeed_runtime import bind_deepspeed_runtime_profile


def _preflight_direct_main(argv: list[str]) -> object:
    """Validate generated ownership before importing the training runtime."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--save_model_path", default="./checkpoints/model_stage1")
    parser.add_argument("--log_file")
    parser.add_argument("--benchmark_run_config")
    parser.add_argument("--benchmark_evidence_path")
    parser.add_argument("--benchmark_smoke_resume_witness")
    parser.add_argument("--training_evidence_path")
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
    Stage2PeftMetadata,
    build_stage2_peft_metadata,
    configure_stage2_metadata,
    prepare_stage2_sparse_token_base,
    resolve_latent_token_ids,
    resolve_stage2_load_decision,
    resolve_stage2b_initialization_decision,
    validate_stage2b_peft_base_identity,
    validate_stage2_peft_compatibility,
    validate_stage2_sparse_processor,
)
from stage2.data import (
    Stage2DataCollator,
    build_prepared_example_length_measure,
    filter_stage2_examples,
    load_jsonl_examples,
    prepare_stage2_example,
)
from stage2.length_manifest import validate_stage2_length_manifest
from stage2.peft import (
    build_stage2_tuned_model,
    load_stage1_dora_for_fresh_stage2,
    resolve_stage2_target_module_names,
    validate_stage2_parameter_boundary,
)
from step_preference.config import StepPreferenceConfig
from step_preference.negatives import (
    inspect_stage2a_checkpoint_identity,
    load_negative_store,
    read_negative_manifest,
)
from step_preference.spans import (
    ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    ASSISTANT_PLAN_RENDERING_CONCATENATED,
    reject_duplicate_sample_ids,
)

platform.python_version = _platform_python_version


_PHASE_A_SUMMARY_PREFIX = "STEP_PREFERENCE_PHASE_A_SUMMARY "
_PHASE_A_METRIC_NAMES = (
    "loss/total",
    "loss/ce",
    "loss/dpo_visual",
    "loss/dpo_text",
    "loss/entropy_visual",
    "loss/entropy_text",
    "stats/valid_visual_pairs",
    "stats/valid_textual_pairs",
)


def observe_stage2_checkpoint_artifact(*args, **kwargs):
    """Load optional producer-evidence dependencies only when evidence is requested."""
    from stage2.evidence import observe_stage2_checkpoint_artifact as observe

    return observe(*args, **kwargs)


@dataclass(frozen=True)
class TrainingPath:
    trainer_class: type
    collator_factory: object


def select_training_path(stage: str) -> TrainingPath:
    if stage == "stage1":
        from stage1.trainer import CustomTrainerStage1

        return TrainingPath(CustomTrainerStage1, collate_fn_stage1)
    if stage == "stage2":
        from stage2.trainer import Stage2Trainer

        return TrainingPath(Stage2Trainer, Stage2DataCollator)
    raise ValueError(f"Unsupported training stage: {stage}")


@dataclass(frozen=True)
class Stage2TrainingComponents:
    model: object
    processor: object
    train_dataset: object
    data_collator: Stage2DataCollator
    trainer_class: type
    trainer_resume_path: str | None
    peft_metadata: Stage2PeftMetadata | None
    step_preference_config: StepPreferenceConfig | None = None


@dataclass(frozen=True)
class Stage2BPreflight:
    """Read-only validated inputs required before Stage 2B may construct runtime."""

    prepared_examples: tuple[object, ...]
    negative_store: Mapping[tuple[str, int], object]


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


def preflight_stage2b_artifacts(args) -> Stage2BPreflight:
    """Validate Stage 2B source, checkpoint, and negatives without runtime writes."""
    config = getattr(args, "step_preference_config", None)
    if config is None:
        config = StepPreferenceConfig.from_args(args)
    if not isinstance(config, StepPreferenceConfig) or not config.is_stage2b:
        raise ValueError("Stage 2B artifact preflight requires Stage 2B configuration")

    source_path = Path(args.data_path)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = read_negative_manifest(config.negative_manifest)
    if manifest.source_sha256 != source_sha256:
        raise ValueError("source JSONL SHA-256 does not match negative manifest")

    raw_examples = load_jsonl_examples(source_path)
    selected_examples = (
        raw_examples
        if args.max_train_samples is None
        else raw_examples[: args.max_train_samples]
    )
    prepared = tuple(
        prepare_stage2_example(
            item,
            image_root=Path(args.image_root),
            latent_size=args.latent_size,
            require_reasoning_units=True,
            assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
        )
        for item in selected_examples
    )
    reject_duplicate_sample_ids(example.sample_id for example in prepared)

    validate_stage2b_peft_base_identity(
        stage2a_checkpoint_path=config.stage2a_checkpoint_path,
        requested_base_model_path=args.load_model_path,
    )
    checkpoint_files, tokenizer_identity = inspect_stage2a_checkpoint_identity(
        config.stage2a_checkpoint_path
    )
    generation_settings = manifest.generation_settings_dict
    if generation_settings.get("min_tokens") != config.min_tokens:
        raise ValueError("negative manifest min_tokens does not match Stage 2B")
    if generation_settings.get("max_new_tokens") != config.max_new_tokens:
        raise ValueError("negative manifest max_new_tokens does not match Stage 2B")
    expected_keys = tuple(
        (example.sample_id, step.step_id)
        for example in prepared
        for step in example.reasoning_steps
    )
    negative_store = load_negative_store(
        negative_path=config.negative_path,
        manifest_path=config.negative_manifest,
        source_path=source_path,
        expected_source_sha256=source_sha256,
        expected_selected_row_indices=tuple(range(len(prepared))),
        expected_stage2a_checkpoint_path=config.stage2a_checkpoint_path,
        expected_checkpoint_files=checkpoint_files,
        expected_tokenizer_config_identity=tokenizer_identity,
        expected_latent_size=args.latent_size,
        expected_generation_settings=generation_settings,
        expected_keys=expected_keys,
    )
    for example in prepared:
        assistant_text = example.conversation[1]["content"][0]["text"]
        for step in example.reasoning_steps:
            key = (example.sample_id, step.step_id)
            chosen = assistant_text[step.target_text.start : step.target_text.end]
            if negative_store[key].chosen_text != chosen:
                raise ValueError(
                    f"negative chosen text does not match source step: {key}"
                )
    return Stage2BPreflight(
        prepared_examples=prepared,
        negative_store=MappingProxyType(dict(negative_store)),
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
    *, gradient_checkpointing: bool, tuning_method: str, deepspeed_runtime_profile
) -> dict[str, bool] | None:
    if not gradient_checkpointing:
        return None
    if (
        tuning_method in ("lora", "dora")
        and deepspeed_runtime_profile.enabled
        and deepspeed_runtime_profile.zero_stage == 3
    ):
        raise ValueError(
            "Stage 2 PEFT ZeRO-3 requires gradient checkpointing disabled"
        )
    return {"use_reentrant": False}


def configure_stage2_model_backend(
    model,
    *,
    gradient_checkpointing: bool,
    tuning_method: str = "full",
    deepspeed_runtime_profile=None,
) -> None:
    """Apply Stage 2 cache/checkpoint settings outside the latent recurrence."""
    model.config.use_cache = False
    if gradient_checkpointing:
        if deepspeed_runtime_profile is None:
            from deepspeed_runtime import DeepSpeedRuntimeProfile

            deepspeed_runtime_profile = DeepSpeedRuntimeProfile(
                False, None, None, None, 1
            )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=_stage2_gradient_checkpointing_kwargs(
                gradient_checkpointing=True,
                tuning_method=tuning_method,
                deepspeed_runtime_profile=deepspeed_runtime_profile,
            )
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


def build_stage2_training_components(
    args, *, source_dependencies: Stage2SourceDependencies | None = None
) -> Stage2TrainingComponents:
    from stage2.trainer import Stage2Trainer

    step_preference_config = getattr(args, "step_preference_config", None)
    if step_preference_config is None:
        step_preference_config = StepPreferenceConfig.from_args(args)
    if not isinstance(step_preference_config, StepPreferenceConfig):
        raise TypeError("step_preference_config must be a StepPreferenceConfig")
    assistant_plan_rendering = (
        ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF
        if step_preference_config.enabled
        else ASSISTANT_PLAN_RENDERING_CONCATENATED
    )
    stage2b_preflight = getattr(args, "_stage2b_preflight", None)
    if step_preference_config.is_stage2b:
        if stage2b_preflight is None:
            stage2b_preflight = preflight_stage2b_artifacts(args)
        elif not isinstance(stage2b_preflight, Stage2BPreflight):
            raise TypeError("_stage2b_preflight must be a Stage2BPreflight")
    source_dependencies = (
        source_dependencies or _default_stage2_source_dependencies()
    )
    source_provenance_kwargs = build_stage2_source_provenance_kwargs(args)
    decision_kwargs = dict(
        load_model_path=args.load_model_path,
        stage1_adapter_path=getattr(args, "stage1_adapter_path", None),
        load_config=source_dependencies.load_config,
        discover_adapter_config=source_dependencies.find_adapter_config,
        source_provenance_kwargs=source_provenance_kwargs,
        latent_size=args.latent_size,
        supervise_latent_boundaries=args.supervise_latent_boundaries,
        tuning_method=args.tuning_method,
        tuning_method_explicit=args.tuning_method_explicit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )
    if step_preference_config.is_stage2b:
        decision = resolve_stage2b_initialization_decision(
            stage2a_checkpoint_path=step_preference_config.stage2a_checkpoint_path,
            **decision_kwargs,
        )
    else:
        decision = resolve_stage2_load_decision(
            resume_from_checkpoint_path=args.resume_from_checkpoint_path,
            **decision_kwargs,
        )
    sparse_resume = (
        decision.peft_metadata is not None
        and decision.peft_metadata.token_row_source == "sparse_adapter"
    )
    sparse_source_kwargs = source_provenance_kwargs if sparse_resume else {}
    processor = source_dependencies.load_processor(
        decision.processor_source, **sparse_source_kwargs
    )
    if sparse_resume:
        validate_stage2_sparse_processor(
            processor.tokenizer, decision.peft_metadata
        )
    manifest_path = getattr(args, "stage2_length_manifest", None)
    manifest_sha256 = getattr(args, "stage2_length_manifest_sha256", None)
    if bool(manifest_path) != bool(manifest_sha256):
        raise ValueError(
            "--stage2_length_manifest and --stage2_length_manifest_sha256 "
            "must be provided together"
        )
    manifest_snapshot = (
        validate_stage2_length_manifest(
            path=manifest_path,
            expected_sha256=manifest_sha256,
            data_path=args.data_path,
            image_root=args.image_root,
            processor=processor,
            latent_size=decision.latent_size,
            supervise_latent_boundaries=decision.supervise_latent_boundaries,
            max_sequence_length=args.max_seq_length_train,
            assistant_plan_rendering=assistant_plan_rendering,
        )
        if manifest_path is not None
        else None
    )
    stage1_preflight = None
    stage1_metadata = None
    if decision.stage1_adapter_source is not None:
        base_config = source_dependencies.load_config(decision.base_model_source)
        stage1_preflight = preflight_stage1_peft_artifact(
            processor,
            base_config,
            Path(decision.stage1_adapter_source),
            requested_base_model_reference=decision.base_model_source,
            metadata_snapshot=decision.stage1_metadata_snapshot,
        )
        stage1_metadata = decode_stage1_peft_metadata_snapshot(
            decision.stage1_metadata_snapshot
        )
    model_only_kwargs = {
        "attn_implementation": args.attention_implementation,
    }
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
    )
    configure_stage2_model_backend(
        base_model,
        gradient_checkpointing=args.gradient_checkpointing,
        tuning_method=decision.tuning.method,
        deepspeed_runtime_profile=bind_deepspeed_runtime_profile(args),
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
        )

    if manifest_snapshot is None:
        if step_preference_config.is_stage2b:
            prepared = stage2b_preflight.prepared_examples
        else:
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
                    assistant_plan_rendering=assistant_plan_rendering,
                )
                for item in selected_examples
            ]
        kept, _length_stats = filter_stage2_examples(
            examples=prepared,
            measure_length=build_prepared_example_length_measure(processor),
            max_length=args.max_seq_length_train,
        )
    else:
        if step_preference_config.is_stage2b:
            prepared = stage2b_preflight.prepared_examples
        else:
            prepared = (
                manifest_snapshot.prepared_examples
                if args.max_train_samples is None
                else manifest_snapshot.prepared_examples[: args.max_train_samples]
            )
        kept = [
            prepared[index]
            for index in manifest_snapshot.manifest.retained_indices
            if index < len(prepared)
        ]

    negative_store = (
        stage2b_preflight.negative_store
        if step_preference_config.is_stage2b
        else None
    )
    collator = Stage2DataCollator(
        processor,
        token_ids,
        decision.latent_size,
        decision.supervise_latent_boundaries,
        step_preference_enabled=step_preference_config.is_stage2b,
        negative_store=negative_store,
    )
    return Stage2TrainingComponents(
        model=model,
        processor=processor,
        train_dataset=kept,
        data_collator=collator,
        trainer_class=Stage2Trainer,
        trainer_resume_path=decision.trainer_resume_path,
        peft_metadata=peft_metadata,
        step_preference_config=step_preference_config,
    )


def build_stage2_training_arguments(args) -> SFTConfig:
    validate_stage2_distributed_settings(args.per_device_train_batch_size)
    dataloader_num_workers = getattr(args, "dataloader_num_workers", 0)
    if dataloader_num_workers < 0:
        raise ValueError("Stage 2 dataloader_num_workers must be non-negative")
    dataloader_kwargs = {"dataloader_num_workers": dataloader_num_workers}
    if dataloader_num_workers > 0:
        dataloader_kwargs["dataloader_prefetch_factor"] = 2
    checkpointing_kwargs = _stage2_gradient_checkpointing_kwargs(
        gradient_checkpointing=args.gradient_checkpointing,
        tuning_method=args.tuning_method,
        deepspeed_runtime_profile=bind_deepspeed_runtime_profile(args),
    )
    step_preference_config = getattr(args, "step_preference_config", None)
    if step_preference_config is None:
        step_preference_config = StepPreferenceConfig.from_args(args)
    phase_steps = step_preference_config.phase_steps
    return SFTConfig(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warm_up_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=20,
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
        max_steps=args.max_steps if phase_steps is None else phase_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=checkpointing_kwargs,
        deepspeed=args.deepspeed_config,
        ddp_find_unused_parameters=False,
        **dataloader_kwargs,
    )

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


def publish_stage2_training_evidence(args, *, trainer, train_result, profile):
    """Publish only rank-local values observed by a completed Stage 2 trainer."""
    evidence_path = getattr(args, "benchmark_evidence_path", None)
    if evidence_path is None:
        return None
    global_step = trainer.state.global_step
    optimizer_step = trainer._observed_optimizer_step
    gradient_norm = trainer._representative_gradient_norm
    loss = train_result.training_loss
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("Stage 2 producer requires an observed positive global step")
    if type(optimizer_step) is not int or optimizer_step != global_step:
        raise ValueError("Stage 2 producer optimizer/global step disagreement")
    if type(loss) not in (int, float) or not math.isfinite(loss):
        raise ValueError("Stage 2 producer requires an observed finite loss")
    if type(gradient_norm) not in (int, float) or not math.isfinite(gradient_norm):
        raise ValueError("Stage 2 producer requires an observed finite gradient")
    output = Path(args.save_model_path)
    checkpoint = output / f"checkpoint-{global_step}"
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Stage 2 producer requires its numbered checkpoint")
    method = args.tuning_method
    artifact_kind = observe_stage2_checkpoint_artifact(
        checkpoint, method=method
    )
    try:
        rank = int(os.environ.get("RANK", "0"))
    except ValueError as error:
        raise ValueError("Stage 2 producer RANK is invalid") from error
    from stage2.evidence import publish_stage2_producer_evidence

    return publish_stage2_producer_evidence(
        evidence_path,
        run_id=output.name,
        method=method,
        zero_stage=profile.zero_stage,
        checkpoint_path=checkpoint,
        artifact_kind=artifact_kind,
        local_rank_record={
            "rank": rank,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "loss": float(loss),
            "gradient_norm": float(gradient_norm),
            "worker_error": None,
        },
    )


def publish_step_preference_phase_a_summary(args, *, trainer, train_result):
    """Log one rank-zero scalar summary for the focused Stage 2B verifier."""
    config = getattr(args, "step_preference_config", None)
    if config is None:
        config = StepPreferenceConfig.from_args(args)
    if not config.is_stage2b or not trainer.is_world_process_zero():
        return None
    latest = next(
        (
            entry
            for entry in reversed(trainer.state.log_history)
            if isinstance(entry, Mapping)
            and all(name in entry for name in _PHASE_A_METRIC_NAMES)
        ),
        None,
    )
    if latest is None:
        raise ValueError("Stage 2B trainer log history has no complete metric record")
    summary = {
        "schema_version": 1,
        "global_step": trainer.state.global_step,
        "optimizer_step": trainer._observed_optimizer_step,
        "training_loss": train_result.training_loss,
        "total_loss": latest["loss/total"],
        "gradient_norm": trainer._representative_gradient_norm,
        "metrics": {
            name: latest[name]
            for name in _PHASE_A_METRIC_NAMES
            if name != "loss/total"
        },
    }
    encoded = json.dumps(
        summary, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    logging.info(_PHASE_A_SUMMARY_PREFIX + encoded)
    return summary


def run_stage2(args):
    """Run the existing Stage 2 path without importing Stage 1 Trainer code."""

    profile = bind_deepspeed_runtime_profile(
        args, world_size=int(os.environ.get("WORLD_SIZE", "1"))
    )
    components = build_stage2_training_components(args)
    trainer_kwargs = dict(
        model=components.model,
        args=build_stage2_training_arguments(args),
        train_dataset=components.train_dataset,
        data_collator=components.data_collator,
        processing_class=components.processor,
        stage2_peft_metadata=components.peft_metadata,
        deepspeed_runtime_profile=profile,
        stage2_diagnostics_interval=args.stage2_diagnostics_interval,
    )
    if (
        components.step_preference_config is not None
        and components.step_preference_config.is_stage2b
    ):
        trainer_kwargs["step_preference_config"] = components.step_preference_config
    trainer = components.trainer_class(**trainer_kwargs)
    train_result = trainer.train(
        resume_from_checkpoint=components.trainer_resume_path
    )
    publish_step_preference_phase_a_summary(
        args, trainer=trainer, train_result=train_result
    )
    publish_stage2_training_evidence(
        args, trainer=trainer, train_result=train_result, profile=profile
    )
    if args.save_final_model:
        trainer.save_model(args.save_model_path)
        if trainer.is_world_process_zero():
            components.processor.save_pretrained(args.save_model_path)


def route_training(args, *, stage1_handler=None, stage2_handler=None):
    """Select exactly one public stage handler and return its observable result."""

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
    step_preference_config = getattr(args, "step_preference_config", None)
    if step_preference_config is None:
        step_preference_config = StepPreferenceConfig.from_args(args)
    stage2b_preflight = None
    if step_preference_config.is_stage2b:
        stage2b_preflight = preflight_stage2b_artifacts(args)
    if getattr(args, "stage2b_preflight_only", False):
        return 0
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

    if stage2b_preflight is not None:
        args._stage2b_preflight = stage2b_preflight
    return route_training(args)


def __getattr__(name):
    """Retain historical public trainer imports without eager cross-stage load."""

    if name == "CustomTrainerStage1":
        from stage1.trainer import CustomTrainerStage1

        return CustomTrainerStage1
    if name == "Stage2Trainer":
        from stage2.trainer import Stage2Trainer

        return Stage2Trainer
    raise AttributeError(name)

if __name__ == "__main__":
    main_train()
