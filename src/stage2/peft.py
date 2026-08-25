from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

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
    raw = value.split(",") if isinstance(value, str) else list(value)
    targets = tuple(item.strip() for item in raw if item.strip())
    if not targets:
        raise ValueError("Stage 2 PEFT target modules must not be empty")
    if len(set(targets)) != len(targets):
        raise ValueError("Stage 2 PEFT target modules must be unique")
    invalid = sorted(set(targets) - set(ALLOWED_STAGE2_TARGET_SUFFIXES))
    if invalid:
        raise ValueError(f"unsupported Stage 2 PEFT target modules: {invalid}")
    return targets


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
    matches = {
        suffix: tuple(
            name
            for name, module in model.named_modules()
            if name.startswith("model.layers.")
            and name.rsplit(".", 1)[-1] == suffix
            and isinstance(module, torch.nn.Linear)
        )
        for suffix in requested
    }
    missing = [suffix for suffix, names in matches.items() if not names]
    if missing:
        raise ValueError(f"Stage 2 decoder targets matched no modules: {missing}")
    return tuple(name for suffix in requested for name in matches[suffix])


def resolve_stage2_qwen_model(model: nn.Module) -> nn.Module:
    """Unwrap one engine wrapper and a PEFT model to the Stage 2 Qwen base."""
    candidate = getattr(model, "module", model)
    if isinstance(candidate, PeftModel):
        candidate = candidate.get_base_model()

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
) -> Stage2ParameterReport:
    """Validate that Stage 2 training owns only the permitted parameters."""
    qwen = resolve_stage2_qwen_model(model)
    for label, module in (
        ("visual", qwen.visual),
        ("embedding", qwen.model.embed_tokens),
        ("lm_head", qwen.lm_head),
    ):
        for name, parameter in module.named_parameters():
            if parameter.requires_grad and (label == "visual" or tuning.is_peft):
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
        unexpected = [name for name in trainable if "lora_" not in name]
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


def build_stage2_tuned_model(
    model: nn.Module,
    tuning: Stage2TuningConfig,
    adapter_path: str | Path | None = None,
    is_trainable: bool = True,
) -> nn.Module:
    """Freeze vision and construct or load a decoder-only Stage 2 adapter."""
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
    if not is_trainable:
        wrapped.requires_grad_(False)
    validate_stage2_parameter_boundary(
        wrapped, tuning, expect_trainable=is_trainable
    )
    return wrapped
