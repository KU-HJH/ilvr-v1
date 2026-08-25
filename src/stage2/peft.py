from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

from peft_common import (
    normalize_decoder_target_suffixes,
    resolve_qwen2_5_vl_model,
    resolve_qwen_decoder_target_names,
)

TuningMethod = Literal["full", "lora", "dora"]
ALLOWED_STAGE2_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def normalize_stage2_target_modules(
    value: str | Sequence[str],
) -> tuple[str, ...]:
    return normalize_decoder_target_suffixes(
        value, stage_label="Stage 2 PEFT target modules"
    )


@dataclass(frozen=True)
class Stage2TuningConfig:
    method: TuningMethod
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.method not in ("full", "lora", "dora"):
            raise ValueError(f"unsupported Stage 2 tuning method: {self.method}")
        if self.r <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        normalize_stage2_target_modules(self.target_modules)

    @property
    def is_peft(self) -> bool:
        return self.method in ("lora", "dora")

    @property
    def use_dora(self) -> bool:
        return self.method == "dora"


@dataclass(frozen=True)
class Stage2ParameterReport:
    total_parameters: int
    trainable_parameters: int
    trainable_names: tuple[str, ...]
    resolved_target_names: tuple[str, ...]


def resolve_stage2_target_module_names(
    model: nn.Module, requested: Sequence[str]
) -> tuple[str, ...]:
    """Return full decoder projection paths, excluding matching visual layers."""
    requested = normalize_stage2_target_modules(requested)
    try:
        resolved = resolve_qwen_decoder_target_names(model, requested)
    except ValueError as error:
        if str(error).startswith(
            ("Requested decoder targets did not match", "Requested targets must be decoder-owned")
        ):
            missing = str(error).split(": ", 1)[1]
            raise ValueError(
                f"Stage 2 decoder targets matched no modules: {missing}"
            ) from error
        raise
    modules = dict(model.named_modules())
    matches = {
        suffix: tuple(
            name
            for name in resolved
            if name.endswith(f".{suffix}")
            and isinstance(modules[name], torch.nn.Linear)
        )
        for suffix in requested
    }
    missing = [suffix for suffix, names in matches.items() if not names]
    if missing:
        raise ValueError(f"Stage 2 decoder targets matched no modules: {missing}")
    return tuple(name for suffix in requested for name in matches[suffix])


def resolve_stage2_qwen_model(model: nn.Module) -> nn.Module:
    """Unwrap one engine wrapper and a PEFT model to the Stage 2 Qwen base."""
    try:
        candidate = resolve_qwen2_5_vl_model(model)
    except TypeError as error:
        raise TypeError("model does not satisfy the Stage 2 Qwen interface") from error
    required = (
        "prepare_stage2_base_inputs_embeds",
        "get_rope_index",
        "stage2_rollout_engine",
        "model",
        "lm_head",
    )
    if (
        any(not hasattr(candidate, attribute) for attribute in required)
        or getattr(getattr(candidate, "config", None), "stage", None) != "stage2"
    ):
        raise TypeError("model does not satisfy the Stage 2 Qwen interface")
    return candidate


def _stage2_adapter_target_names(
    model: nn.Module, qwen: nn.Module, tuning: Stage2TuningConfig
) -> tuple[str, ...]:
    if not isinstance(model, PeftModel):
        raise ValueError("Stage 2 PEFT tuning requires a PeftModel")

    config = model.peft_config["default"]
    target_names = tuple(config.target_modules)
    modules = dict(qwen.named_modules())
    unapproved = sorted(
        name
        for name in target_names
        if name.rsplit(".", 1)[-1] not in ALLOWED_STAGE2_TARGET_SUFFIXES
    )
    if unapproved:
        raise ValueError(f"unapproved Stage 2 decoder targets: {unapproved}")

    requested = normalize_stage2_target_modules(tuning.target_modules)
    expected_targets = {
        name
        for name, module in modules.items()
        if name.startswith("model.layers.")
        and name.rsplit(".", 1)[-1] in requested
        and isinstance(getattr(module, "base_layer", module), torch.nn.Linear)
    }
    if (
        not target_names
        or config.modules_to_save is not None
        or any(not name.startswith("model.layers.") for name in target_names)
        or any(name not in modules or not hasattr(modules[name], "lora_A") for name in target_names)
    ):
        raise ValueError("Stage 2 PEFT adapter targets must be decoder LoRA modules")
    if set(target_names) != expected_targets:
        raise ValueError("loaded adapter targets do not match Stage 2 tuning")
    return target_names


def validate_stage2_parameter_boundary(
    model: nn.Module,
    tuning: Stage2TuningConfig,
    *,
    expect_trainable: bool = True,
    allow_sparse_token_rows: bool = False,
) -> Stage2ParameterReport:
    """Validate that Stage 2 training owns only the permitted parameters."""
    qwen = resolve_stage2_qwen_model(model)
    if allow_sparse_token_rows:
        if tuning.method != "dora" or not expect_trainable:
            raise ValueError(
                "sparse Stage 2 token rows require trainable DoRA continuation"
            )
        from stage1.peft import validate_stage1_trainable_boundary
        from stage1.token_rows import find_sparse_output_token_rows
        from stage1.types import Stage1TuningConfig

        rows = find_sparse_output_token_rows(model)
        validate_stage1_trainable_boundary(
            model,
            Stage1TuningConfig(
                method="dora",
                rank=tuning.r,
                alpha=tuning.alpha,
                dropout=tuning.dropout,
                target_suffixes=tuning.target_modules,
            ),
            tuple(rows.token_ids.detach().cpu().tolist()),
        )
    for label, module in (
        ("visual", qwen.visual),
        ("embedding", qwen.model.embed_tokens),
        ("lm_head", qwen.lm_head),
    ):
        for name, parameter in module.named_parameters():
            if parameter.requires_grad and (
                label == "visual"
                or (tuning.is_peft and not allow_sparse_token_rows)
            ):
                raise ValueError(f"{label} parameter is trainable: {name}")

    if tuning.is_peft:
        resolved_target_names = _stage2_adapter_target_names(model, qwen, tuning)
        config = model.peft_config["default"]
        if config.use_dora is not tuning.use_dora:
            raise ValueError("Stage 2 PEFT adapter method does not match tuning")
    else:
        resolved_target_names = resolve_stage2_target_module_names(
            qwen, tuning.target_modules
        )

    parameters = tuple(model.named_parameters())
    trainable = tuple(name for name, parameter in parameters if parameter.requires_grad)
    if not expect_trainable:
        if trainable:
            raise ValueError("Stage 2 inference model has trainable parameters")
    elif tuning.is_peft:
        if not trainable:
            raise ValueError("Stage 2 PEFT adapter has no trainable parameters")
        unexpected = [
            name
            for name in trainable
            if "lora_" not in name
            and not (
                allow_sparse_token_rows
                and (
                    "trainable_tokens_delta" in name
                    or name.endswith("output_delta")
                )
            )
        ]
        if unexpected:
            raise ValueError(
                f"non-adapter parameter is trainable: {unexpected[0]}"
            )
    else:
        if not trainable:
            raise ValueError("Stage 2 full tuning has no trainable language parameters")

    return Stage2ParameterReport(
        total_parameters=sum(parameter.numel() for _, parameter in parameters),
        trainable_parameters=sum(
            parameter.numel() for _, parameter in parameters if parameter.requires_grad
        ),
        trainable_names=trainable,
        resolved_target_names=resolved_target_names,
    )


def load_stage1_dora_for_fresh_stage2(
    base_model: nn.Module,
    processor,
    stage1_adapter_path: str | Path,
    stage2_tuning: Stage2TuningConfig,
    *,
    preflight=None,
) -> nn.Module:
    """Continue one validated Stage 1 DoRA adapter with fresh Stage 2 state."""

    from stage1.checkpointing import (
        Stage1PeftMetadataSnapshot,
        decode_stage1_peft_metadata_snapshot,
        load_stage1_peft_artifact,
        preflight_stage1_peft_artifact,
    )

    qwen = resolve_qwen2_5_vl_model(base_model)
    if preflight is None:
        preflight = preflight_stage1_peft_artifact(
            processor,
            qwen.config,
            Path(stage1_adapter_path),
            model_factory=lambda unused_config: qwen,
        )
    metadata = decode_stage1_peft_metadata_snapshot(
        Stage1PeftMetadataSnapshot(
            artifact_dir=preflight.artifact_dir,
            metadata_json=preflight.metadata_json,
        )
    )
    if metadata.tuning_method != "dora":
        raise ValueError("fresh Stage 2 continuation requires a Stage 1 DoRA adapter")
    if stage2_tuning.method == "full":
        raise ValueError("Stage 1 DoRA cannot initialize Stage 2 full tuning")
    if stage2_tuning.method == "lora":
        raise ValueError("Stage 1 DoRA cannot initialize a new Stage 2 LoRA adapter")
    for label, stage1_value, stage2_value in (
        ("rank", metadata.rank, stage2_tuning.r),
        ("alpha", metadata.alpha, stage2_tuning.alpha),
        ("dropout", metadata.dropout, stage2_tuning.dropout),
        (
            "requested target modules",
            metadata.requested_target_suffixes,
            stage2_tuning.target_modules,
        ),
    ):
        if stage1_value != stage2_value:
            raise ValueError(
                f"Stage 1/Stage 2 {label} mismatch: "
                f"stage1={stage1_value}, stage2={stage2_value}"
            )
    current_resolved = resolve_stage2_target_module_names(
        qwen, stage2_tuning.target_modules
    )
    if current_resolved != metadata.resolved_target_names:
        raise ValueError(
            "Stage 1/Stage 2 resolved target mismatch: "
            f"stage1={metadata.resolved_target_names}, stage2={current_resolved}"
        )

    restored = load_stage1_peft_artifact(
        base_model,
        processor,
        Path(stage1_adapter_path),
        is_trainable=True,
        preflight=preflight,
    )
    resolve_qwen2_5_vl_model(restored).config.stage = "stage2"
    validate_stage2_parameter_boundary(
        restored,
        stage2_tuning,
        allow_sparse_token_rows=True,
    )
    return restored


def build_stage2_tuned_model(
    model: nn.Module,
    tuning: Stage2TuningConfig,
    adapter_path: str | Path | None = None,
    is_trainable: bool = True,
    *,
    token_row_source: str = "base",
    latent_token_ids: Sequence[int] | None = None,
) -> nn.Module:
    """Freeze vision and construct or load a decoder-only Stage 2 adapter."""
    if token_row_source not in ("base", "sparse_adapter"):
        raise ValueError("token_row_source must be base or sparse_adapter")
    if token_row_source == "sparse_adapter":
        if adapter_path is None or latent_token_ids is None:
            raise ValueError(
                "sparse Stage 2 token rows require an adapter and latent token IDs"
            )
        if not is_trainable:
            raise ValueError(
                "sparse Stage 2 inference loading is not supported by this trainer API"
            )
        from stage1.token_rows import initialize_frozen_output_token_rows

        initialize_frozen_output_token_rows(model, latent_token_ids)
    elif latent_token_ids is not None:
        raise ValueError("base token rows do not accept sparse latent token IDs")
    model.visual.requires_grad_(False)
    if tuning.method == "full":
        if adapter_path is not None:
            raise ValueError("full tuning cannot load a PEFT adapter")
        return model

    resolved = resolve_stage2_target_module_names(model, tuning.target_modules)
    if adapter_path is None:
        config = LoraConfig(
            r=tuning.r,
            lora_alpha=tuning.alpha,
            lora_dropout=tuning.dropout,
            target_modules=list(resolved),
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=tuning.use_dora,
            modules_to_save=None,
        )
        wrapped = get_peft_model(model, config)
    else:
        wrapped = PeftModel.from_pretrained(
            model, str(adapter_path), is_trainable=is_trainable
        )
    if token_row_source == "sparse_adapter":
        from stage1.token_rows import (
            attach_sparse_output_token_rows,
            load_sparse_token_rows,
        )

        attach_sparse_output_token_rows(wrapped, latent_token_ids)
        load_sparse_token_rows(wrapped, Path(adapter_path), latent_token_ids)
    if not is_trainable:
        wrapped.requires_grad_(False)
    validate_stage2_parameter_boundary(
        wrapped,
        tuning,
        expect_trainable=is_trainable,
        allow_sparse_token_rows=token_row_source == "sparse_adapter",
    )
    return wrapped


def load_stage2_model_for_inference(
    base_model: nn.Module,
    adapter_path: str | Path,
    metadata,
    tokenizer,
) -> PeftModel:
    """Validate a Stage 1 base, then attach one frozen Stage 2 adapter."""
    from peft import PeftConfig

    from .checkpointing import (
        configure_stage2_metadata,
        resolve_latent_token_ids,
        validate_stage2_peft_compatibility,
    )

    adapter_config = PeftConfig.from_pretrained(str(adapter_path))
    validate_stage2_peft_compatibility(
        base_model,
        tokenizer,
        metadata,
        adapter_config,
        adapter_path,
    )
    token_ids = resolve_latent_token_ids(tokenizer, base_model.config)
    configure_stage2_metadata(
        base_model.config,
        token_ids,
        latent_size=metadata.latent_size,
        supervise_latent_boundaries=metadata.supervise_latent_boundaries,
    )
    tuning = Stage2TuningConfig(
        method=metadata.tuning_method,
        r=metadata.lora_r,
        alpha=metadata.lora_alpha,
        dropout=metadata.lora_dropout,
        target_modules=metadata.requested_target_modules,
    )
    wrapped = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )
    wrapped.eval()
    validate_stage2_parameter_boundary(
        wrapped,
        tuning,
        expect_trainable=False,
    )
    return wrapped
