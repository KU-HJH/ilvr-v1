"""Versioned objective identity for DIL-BCE checkpoints and inference artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from peft_common import resolve_qwen2_5_vl_model
from step_preference.config import StepPreferenceConfig


NAMESPACE = "ilvr_context_preference"
_FORMULAS = {"dil_bce_rf": "dil_bce_rf_v1", "dil_bce_rf_topk": "dil_bce_rf_topk_v1",
             "dil_bce_rf_topk_reverse": "dil_bce_rf_topk_reverse_v1",
             "dil_bce_rf_topk_both_upper": "dil_bce_rf_topk_both_upper_v1",
             "dil_bce_rf_topk_both_lower": "dil_bce_rf_topk_both_lower_v1"}
_FIXED = {
    "schema_version": 1,
    "mask": "latent_body_attention",
    "boundaries": "visible_supervised",
    "block_original_image": False,
    "shared_prefix": True,
    "sidecar_checkpoint": True,
    "reduction": "token_mean_then_sample_step_mean_v1",
    "entropy_selection": "sample_full_vocab_stable_ceil_original_step_denominator_v1",
}
_FLOATS = ("beta", "lambda_preference", "lambda_ce", "lambda_entropy",
           "lambda_topk_mass", "preference_margin")
_STRINGS = ("loss_type", "formula_version", "entropy_source", "target_scope",
            "prefix_mode", "branch_mode", "mask", "boundaries",
            "sidecar_attention_backend", "reduction", "entropy_selection")
_BOOLS = ("visual_only", "block_original_image", "shared_prefix", "sidecar_checkpoint")
_KEYS = frozenset((*_FIXED, *_FLOATS, *_STRINGS, *_BOOLS,
                   "topk", "entropy_top_percent", "sidecar_query_budget"))


def build_dil_identity(config: StepPreferenceConfig) -> dict[str, object] | None:
    if config.preference_loss_type == "dpo":
        return None
    from step_preference.dil_config import validate_dil_config

    validate_dil_config(config)
    return {
        **_FIXED,
        "loss_type": config.preference_loss_type,
        "formula_version": _FORMULAS[config.preference_loss_type],
        "beta": float(config.beta),
        "lambda_preference": float(config.lambda_preference),
        "lambda_ce": float(config.lambda_ce),
        "lambda_entropy": float(config.lambda_entropy),
        "lambda_topk_mass": float(config.lambda_topk_mass),
        "preference_margin": float(config.preference_margin),
        "topk": config.preference_topk,
        "entropy_top_percent": (
            None if config.preference_entropy_top_percent is None
            else float(config.preference_entropy_top_percent)
        ),
        "entropy_source": config.preference_entropy_source,
        "target_scope": config.target_scope,
        "prefix_mode": config.prefix_mode,
        "branch_mode": config.branch_mode,
        "visual_only": config.visual_only_context_preference,
        "sidecar_attention_backend": config.sidecar_attention_backend,
        "sidecar_query_budget": config.sidecar_query_budget,
    }


def read_dil_identity(config_payload: Mapping[str, object]) -> dict[str, object] | None:
    if not isinstance(config_payload, Mapping):
        raise ValueError("invalid DIL objective identity config")
    if NAMESPACE not in config_payload:
        return None
    identity = config_payload[NAMESPACE]
    if not isinstance(identity, Mapping) or set(identity) != _KEYS:
        raise ValueError("invalid DIL objective identity keys")
    if any(type(identity[key]) is not float or not math.isfinite(identity[key])
           for key in _FLOATS):
        raise ValueError("invalid DIL objective identity numeric type")
    if (identity["beta"] <= 0 or identity["lambda_preference"] < 0
        or identity["lambda_topk_mass"] < 0 or identity["lambda_ce"] < 0.0
        or identity["lambda_entropy"] != 0.0
        or identity["preference_margin"] != 0.0):
        raise ValueError("invalid DIL objective identity objective weights")
    if any(type(identity[key]) is not str for key in _STRINGS):
        raise ValueError("invalid DIL objective identity string type")
    if any(type(identity[key]) is not bool for key in _BOOLS):
        raise ValueError("invalid DIL objective identity boolean type")
    if type(identity["schema_version"]) is not int or any(
        type(identity[key]) is not type(value) or identity[key] != value
        for key, value in _FIXED.items()
    ):
        raise ValueError("invalid DIL objective identity version or invariant")
    if type(identity["sidecar_query_budget"]) is not int or identity["sidecar_query_budget"] <= 0:
        raise ValueError("invalid DIL objective identity query budget")
    loss_type = identity["loss_type"]
    if loss_type not in _FORMULAS or identity["formula_version"] != _FORMULAS[loss_type]:
        raise ValueError("invalid DIL objective identity formula")
    topk = identity["topk"]
    if loss_type in ("dil_bce_rf_topk", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_both_upper", "dil_bce_rf_topk_both_lower"):
        if type(topk) is not int or topk < 2:
            raise ValueError("invalid DIL objective identity topk")
    elif topk is not None:
        raise ValueError("invalid DIL objective identity topk")
    if loss_type == "dil_bce_rf" and identity["lambda_topk_mass"] != 0.0:
        raise ValueError("invalid DIL objective identity topk mass")
    percent = identity["entropy_top_percent"]
    if percent is not None and (type(percent) is not float
                                or not math.isfinite(percent) or not 0 < percent <= 100):
        raise ValueError("invalid DIL objective identity entropy percentage")
    if loss_type == "dil_bce_rf" and percent is not None:
        raise ValueError("invalid DIL objective identity entropy percentage")
    if identity["entropy_source"] not in ("clean", "voff") or (
        percent is None and identity["entropy_source"] != "clean"
    ):
        raise ValueError("invalid DIL objective identity entropy source")
    if identity["target_scope"] not in ("immediate_text", "until_next_latent") or (
        identity["prefix_mode"] != "clean_prefix" or identity["branch_mode"] != "independent"
        or identity["visual_only"] is not True
        or identity["sidecar_attention_backend"] not in ("dense", "flex")
    ):
        raise ValueError("invalid DIL objective identity execution profile")
    return dict(identity)


def validate_dil_resume(checkpoint: Path, requested: StepPreferenceConfig) -> None:
    config_path = Path(checkpoint) / "config.json"
    if requested.preference_loss_type == "dpo" and not config_path.exists():
        return
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        saved = read_dil_identity(payload)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("cannot read DIL objective identity from checkpoint") from error
    expected = build_dil_identity(requested)
    if saved != expected:
        raise ValueError("DIL objective identity does not match requested resume")


def attach_dil_identity(model, config: StepPreferenceConfig) -> None:
    target = resolve_qwen2_5_vl_model(model)
    identity = build_dil_identity(config)
    if identity is None:
        if hasattr(target.config, NAMESPACE):
            delattr(target.config, NAMESPACE)
    else:
        setattr(target.config, NAMESPACE, identity)
