"""Configuration and admission checks for the Stage 2 objective."""

from dataclasses import dataclass, fields
import math
from typing import Any

import torch


@dataclass(frozen=True)
class ObjectiveConfig:
    enabled: bool = False
    lambda_dyn: float = 0.0
    lambda_vis: float = 0.0
    reference_model_path: str | None = None
    gamma_h: float = 0.0
    vision_margin: float = 0.0
    entropy_eps: float = 1e-6
    pre_window: int = 1
    post_window: int = 1


_ARG_NAMES = {
    "enabled": "enable_stage2_objective",
    "lambda_dyn": "stage2_lambda_dyn",
    "lambda_vis": "stage2_lambda_vis",
    "reference_model_path": "stage2_reference_model_path",
    "gamma_h": "stage2_gamma_h",
    "vision_margin": "stage2_vision_margin",
    "entropy_eps": "stage2_entropy_eps",
    "pre_window": "stage2_pre_window",
    "post_window": "stage2_post_window",
}


def config_from_args(args: Any) -> ObjectiveConfig:
    defaults = ObjectiveConfig()
    values = {
        field.name: getattr(
            args,
            _ARG_NAMES[field.name],
            getattr(args, field.name, getattr(defaults, field.name)),
        )
        for field in fields(ObjectiveConfig)
    }
    config = ObjectiveConfig(**values)
    validate_objective_config(config)
    return config


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or not torch.isfinite(torch.tensor(converted, dtype=torch.float32)):
        raise ValueError(f"{name} must be finite and representable in FP32")
    return converted


def validate_objective_config(config: ObjectiveConfig) -> None:
    if not isinstance(config, ObjectiveConfig):
        raise TypeError("config must be an ObjectiveConfig")
    if type(config.enabled) is not bool:
        raise TypeError("enabled must be a boolean")
    lambda_dyn = _finite_number(config.lambda_dyn, "lambda_dyn")
    lambda_vis = _finite_number(config.lambda_vis, "lambda_vis")
    _finite_number(config.gamma_h, "gamma_h")
    _finite_number(config.vision_margin, "vision_margin")
    eps = _finite_number(config.entropy_eps, "entropy_eps")
    if lambda_dyn < 0 or lambda_vis < 0:
        raise ValueError("objective lambdas must be non-negative")
    if eps <= 0 or torch.tensor(eps, dtype=torch.float32).item() <= 0:
        raise ValueError("entropy_eps must be positive and representable in FP32")
    for value, name in ((config.pre_window, "pre_window"), (config.post_window, "post_window")):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if config.reference_model_path is not None and (
        not isinstance(config.reference_model_path, str) or not config.reference_model_path.strip()
    ):
        raise ValueError("reference_model_path must be a non-empty string or None")
    if config.enabled and lambda_dyn > 0 and config.reference_model_path is None:
        raise ValueError("positive lambda_dyn requires reference_model_path")
    if not config.enabled and config != ObjectiveConfig():
        raise ValueError("objective parameters must remain at defaults while disabled")


def validate_objective_ddp_matrix(config, *, kind, backend, native_checkpointing,
                                  latent_checkpointing):
    if kind not in {'MULTI_CPU', 'MULTI_GPU'}:
        return
    if (kind != 'MULTI_CPU' or config.lambda_dyn > 0 or backend != 'eager'
            or native_checkpointing or latent_checkpointing):
        raise ValueError('objective DDP currently requires CPU eager without checkpointing '
                         'and lambda_dyn=0; other combinations are unvalidated')


def validate_objective_zero2_matrix(*, backend, native_checkpointing, latent_checkpointing):
    baseline = backend == 'eager' and not native_checkpointing and not latent_checkpointing
    ours1 = backend == 'sdpa' and not native_checkpointing and latent_checkpointing
    if not (baseline or ours1):
        raise ValueError('objective ZeRO-2 requires eager without checkpointing or ours1 SDPA with selective latent checkpointing')


def validate_objective_runtime(
    config: ObjectiveConfig,
    *,
    stage: str,
    phase: str,
    backend: str,
    rollout_backend: str,
    split_attention: bool,
    microbatch_size: int,
    zero_stage: int | None,
    offload: bool,
    old_preference_enabled: bool,
    legacy_prefix: str,
    legacy_ablation: str,
    native_checkpointing: bool,
    latent_checkpointing: bool,
    cache_checkpointing: bool,
) -> None:
    validate_objective_config(config)
    if not config.enabled:
        return
    for value, name in (
        (split_attention, "split_attention"), (offload, "offload"),
        (old_preference_enabled, "old_preference_enabled"),
        (native_checkpointing, "native_checkpointing"),
        (latent_checkpointing, "latent_checkpointing"),
        (cache_checkpointing, "cache_checkpointing"),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a boolean")
    if stage != "stage2" or phase != "standard":
        raise ValueError("objective requires Stage 2 standard phase")
    if backend not in {"eager", "sdpa"}:
        raise ValueError("objective backend must be eager or SDPA")
    if rollout_backend != "no_cache":
        raise ValueError("objective requires no-cache execution")
    if split_attention and backend != "sdpa":
        raise ValueError("objective split attention requires SDPA language attention")
    if type(microbatch_size) is not int or microbatch_size != 1:
        raise ValueError("objective microbatch_size must be exactly 1")
    if zero_stage is not None and type(zero_stage) is not int:
        raise TypeError("zero_stage must be an integer or None")
    if zero_stage not in {None, 2} or offload:
        raise ValueError("objective admits only no DeepSpeed or ZeRO-2 without offload")
    if zero_stage == 2:
        validate_objective_zero2_matrix(backend=backend,
            native_checkpointing=native_checkpointing, latent_checkpointing=latent_checkpointing)
    if old_preference_enabled:
        raise ValueError("old preference objective must be disabled")
    if legacy_prefix not in {"", "none"} or legacy_ablation not in {"", "none"}:
        raise ValueError("legacy insertion and gating must be disabled")
    policies = (native_checkpointing, latent_checkpointing, cache_checkpointing)
    if sum(policies) > 1:
        raise ValueError("at most one checkpointing policy may be enabled")
    if cache_checkpointing:
        raise ValueError("cache checkpointing is unavailable for no-cache execution")
