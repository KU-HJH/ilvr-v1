"""Configuration and pre-forward checks for reference-free DIL-BCE losses."""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from step_preference.config import StepPreferenceConfig


_DIL_LOSS_TYPES = ("dil_bce_rf", "dil_bce_rf_topk", "dil_bce_rf_topk_reverse")
_TOPK_LOSS_TYPES = ("dil_bce_rf_topk", "dil_bce_rf_topk_reverse")


def _finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def uses_entropy_selection(config: StepPreferenceConfig) -> bool:
    """Whether a top-k objective selects fewer than all eligible steps."""
    return (
        config.preference_loss_type in _TOPK_LOSS_TYPES
        and config.preference_entropy_top_percent is not None
        and config.preference_entropy_top_percent < 100.0
    )


def validate_dil_config(config: StepPreferenceConfig) -> None:
    """Reject DIL options incompatible with the selected objective or run."""
    loss_type = config.preference_loss_type
    if loss_type not in ("dpo", *_DIL_LOSS_TYPES):
        raise ValueError("unknown step_preference_loss_type")

    k = config.preference_topk
    if k is not None and (type(k) is not int or k < 2):
        raise ValueError("step_preference_topk must be an integer >= 2")
    mass = _finite("step_preference_lambda_topk_mass", config.lambda_topk_mass)
    if mass < 0:
        raise ValueError("step_preference_lambda_topk_mass must be nonnegative")
    percent = config.preference_entropy_top_percent
    if percent is not None and not 0 < _finite("step_preference_entropy_top_percent", percent) <= 100:
        raise ValueError("step_preference_entropy_top_percent must be in (0, 100]")
    if config.preference_entropy_source not in ("clean", "voff"):
        raise ValueError("step_preference_entropy_source must be clean or voff")
    if percent is None and config.preference_entropy_source != "clean":
        raise ValueError("step_preference_entropy_source requires entropy selection")

    if loss_type not in _TOPK_LOSS_TYPES and (
        k is not None or mass != 0 or percent is not None
    ):
        raise ValueError("top-k options require dil_bce_rf_topk")
    if loss_type in _TOPK_LOSS_TYPES and k is None:
        raise ValueError(f"{loss_type} requires step_preference_topk")
    if loss_type == "dpo":
        return

    if not (config.is_context_preference and config.visual_only_context_preference):
        raise ValueError("DIL-BCE requires enabled Stage 2B visual-only context preference")
    if config.prefix_mode != "clean_prefix" or config.branch_mode != "independent":
        raise ValueError("DIL-BCE requires clean_prefix and independent branches")
    if not config.shared_prefix_enabled:
        raise ValueError("DIL-BCE requires shared-prefix execution")
    if config.visual_off_suppress_latent_boundaries:
        raise ValueError("DIL-BCE requires latent boundaries kept")
    if config.block_original_image:
        raise ValueError("DIL-BCE requires original image access")
    if config.lambda_entropy != 0.0 or config.preference_margin != 0.0:
        raise ValueError("DIL-BCE requires entropy=0 and preference margin=0")


def validate_dil_runtime(
    config: StepPreferenceConfig, *, vocab_size: int, supervise_latent_boundaries: bool
) -> None:
    """Check resolved model and boundary settings before any training forward."""
    validate_dil_config(config)
    if config.preference_loss_type == "dpo":
        return
    if supervise_latent_boundaries is not True:
        raise ValueError("DIL-BCE requires supervised latent boundaries")
    if type(vocab_size) is not int or vocab_size < 2:
        raise ValueError("DIL-BCE requires a valid output vocabulary size")
    if config.preference_loss_type in _TOPK_LOSS_TYPES and config.preference_topk >= vocab_size:
        raise ValueError("step_preference_topk must be smaller than output vocab size")
