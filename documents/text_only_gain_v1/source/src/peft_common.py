"""Stage-neutral Qwen2.5-VL PEFT and architecture contracts."""

import math
from collections.abc import Mapping, Sequence

from peft import PeftModel
from torch import nn


ALLOWED_QWEN_DECODER_SUFFIXES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


def resolve_qwen2_5_vl_model(model: nn.Module) -> nn.Module:
    """Unwrap one engine wrapper and a PEFT model to its Qwen2.5-VL base."""
    candidate = getattr(model, "module", model)
    if isinstance(candidate, PeftModel):
        candidate = candidate.get_base_model()

    required = ("model", "lm_head")
    if any(not hasattr(candidate, attribute) for attribute in required):
        raise TypeError("model does not satisfy the Qwen2.5-VL interface")
    return candidate


def normalize_decoder_target_suffixes(
    value: str | Sequence[str], *, stage_label: str
) -> tuple[str, ...]:
    """Normalize one non-empty, unique set of supported decoder suffixes."""
    raw = value.split(",") if isinstance(value, str) else list(value)
    targets = tuple(item.strip() for item in raw if item.strip())
    if not targets:
        raise ValueError(f"{stage_label} must not be empty")
    if len(set(targets)) != len(targets):
        raise ValueError(f"{stage_label} must be unique")
    invalid = sorted(set(targets) - ALLOWED_QWEN_DECODER_SUFFIXES)
    if invalid:
        raise ValueError(f"unsupported {stage_label}: {invalid}")
    return targets


def resolve_qwen_decoder_target_names(
    model: nn.Module, requested_suffixes: Sequence[str]
) -> tuple[str, ...]:
    """Return full decoder projection paths while rejecting visual-only matches."""
    suffixes = tuple(requested_suffixes)
    unknown = sorted(set(suffixes) - ALLOWED_QWEN_DECODER_SUFFIXES)
    if unknown:
        raise ValueError(f"Unsupported decoder target suffixes: {unknown}")

    qwen = resolve_qwen2_5_vl_model(model)
    decoder_module_ids = {id(module) for module in qwen.model.modules()}
    matches = tuple(
        name
        for name, module in model.named_modules()
        if id(module) in decoder_module_ids
        and any(name.endswith(f".{suffix}") for suffix in suffixes)
    )
    missing = [
        suffix
        for suffix in suffixes
        if not any(name.endswith(f".{suffix}") for name in matches)
    ]
    if missing:
        visual_matches = [
            suffix
            for suffix in missing
            if any(
                name.endswith(f".{suffix}")
                for name, _ in model.named_modules()
            )
        ]
        if visual_matches:
            raise ValueError(
                f"Requested targets must be decoder-owned: {visual_matches}"
            )
        raise ValueError(f"Requested decoder targets did not match: {missing}")
    return matches


_ARCHITECTURE_KEYS = frozenset(
    {"model_type", "architectures", "decoder", "multimodal_token_ids", "vision"}
)
_DECODER_ARCHITECTURE_KEYS = frozenset(
    {
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_act",
        "vocab_size", "max_position_embeddings", "rope_theta", "rms_norm_eps",
        "rope_scaling", "attention_dropout", "tie_word_embeddings",
        "use_sliding_window", "sliding_window", "max_window_layers",
    }
)
_MULTIMODAL_TOKEN_KEYS = frozenset(
    {"image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"}
)
_VISION_ARCHITECTURE_KEYS = frozenset(
    {
        "depth", "hidden_size", "intermediate_size", "num_heads", "in_channels",
        "patch_size", "temporal_patch_size", "spatial_merge_size", "window_size",
        "out_hidden_size", "tokens_per_second", "fullatt_block_indexes", "hidden_act",
    }
)


def _require_exact_object(value, *, path: str, keys: frozenset[str]) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{path} must contain non-empty string keys")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"invalid {path} fields: "
            f"missing={sorted(keys - actual)}, unknown={sorted(actual - keys)}"
        )
    return value


def _require_int(value, *, path: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _optional_int(value, *, path: str, minimum: int = 1) -> int | None:
    if value is None:
        return None
    return _require_int(value, path=path, minimum=minimum)


def _require_nonempty_string(value, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _require_finite_number(value, *, path: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def _normalize_json_value(value, *, path: str):
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if isinstance(value, list):
        return [
            _normalize_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError(f"{path} must contain non-empty string keys")
        return {
            key: _normalize_json_value(value[key], path=f"{path}.{key}")
            for key in sorted(value)
        }
    raise ValueError(f"{path} contains a non-JSON value: {type(value).__name__}")


def normalize_qwen2_5_vl_architecture(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and canonically order a Qwen2.5-VL architecture snapshot."""
    root = _require_exact_object(
        value, path="base_model_architecture", keys=_ARCHITECTURE_KEYS
    )
    model_type = root["model_type"]
    if model_type != "qwen2_5_vl":
        raise ValueError(
            "base_model_architecture.model_type must be supported qwen2_5_vl"
        )
    architectures = root["architectures"]
    if not isinstance(architectures, list) or any(
        not isinstance(item, str) or not item for item in architectures
    ):
        raise ValueError(
            "base_model_architecture.architectures must be a JSON string list"
        )

    decoder = _require_exact_object(
        root["decoder"], path="base_model_architecture.decoder", keys=_DECODER_ARCHITECTURE_KEYS
    )
    normalized_decoder = {
        name: _require_int(decoder[name], path=f"base_model_architecture.decoder.{name}")
        for name in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
            "max_position_embeddings",
        )
    }
    normalized_decoder["hidden_act"] = _require_nonempty_string(
        decoder["hidden_act"], path="base_model_architecture.decoder.hidden_act"
    )
    rope_theta = _require_finite_number(
        decoder["rope_theta"], path="base_model_architecture.decoder.rope_theta"
    )
    if rope_theta <= 0:
        raise ValueError("base_model_architecture.decoder.rope_theta must be finite and positive")
    normalized_decoder["rope_theta"] = rope_theta
    rms_norm_eps = _require_finite_number(
        decoder["rms_norm_eps"], path="base_model_architecture.decoder.rms_norm_eps"
    )
    if rms_norm_eps <= 0:
        raise ValueError("base_model_architecture.decoder.rms_norm_eps must be finite and positive")
    normalized_decoder["rms_norm_eps"] = rms_norm_eps
    attention_dropout = _require_finite_number(
        decoder["attention_dropout"], path="base_model_architecture.decoder.attention_dropout"
    )
    if not 0.0 <= attention_dropout < 1.0:
        raise ValueError(
            "base_model_architecture.decoder.attention_dropout must be finite and in [0, 1)"
        )
    normalized_decoder["attention_dropout"] = attention_dropout
    if type(decoder["tie_word_embeddings"]) is not bool:
        raise ValueError("base_model_architecture.decoder.tie_word_embeddings must be boolean")
    normalized_decoder["tie_word_embeddings"] = decoder["tie_word_embeddings"]
    rope_scaling = decoder["rope_scaling"]
    if rope_scaling is not None and not isinstance(rope_scaling, Mapping):
        raise ValueError(
            "base_model_architecture.decoder.rope_scaling must be a JSON object or null"
        )
    normalized_decoder["rope_scaling"] = _normalize_json_value(
        rope_scaling, path="base_model_architecture.decoder.rope_scaling"
    )
    if type(decoder["use_sliding_window"]) is not bool:
        raise ValueError("base_model_architecture.decoder.use_sliding_window must be boolean")
    normalized_decoder["use_sliding_window"] = decoder["use_sliding_window"]
    for name in ("sliding_window", "max_window_layers"):
        normalized_decoder[name] = _optional_int(
            decoder[name], path=f"base_model_architecture.decoder.{name}"
        )

    token_ids = _require_exact_object(
        root["multimodal_token_ids"],
        path="base_model_architecture.multimodal_token_ids",
        keys=_MULTIMODAL_TOKEN_KEYS,
    )
    normalized_token_ids = {
        name: _require_int(
            token_ids[name],
            path=f"base_model_architecture.multimodal_token_ids.{name}",
            minimum=0,
        )
        for name in ("image_token_id", "video_token_id")
    }
    for name in ("vision_start_token_id", "vision_end_token_id"):
        normalized_token_ids[name] = _optional_int(
            token_ids[name],
            path=f"base_model_architecture.multimodal_token_ids.{name}",
            minimum=0,
        )

    vision = _require_exact_object(
        root["vision"], path="base_model_architecture.vision", keys=_VISION_ARCHITECTURE_KEYS
    )
    normalized_vision = {
        name: _require_int(vision[name], path=f"base_model_architecture.vision.{name}")
        for name in _VISION_ARCHITECTURE_KEYS - {"fullatt_block_indexes", "hidden_act"}
    }
    normalized_vision["hidden_act"] = _require_nonempty_string(
        vision["hidden_act"], path="base_model_architecture.vision.hidden_act"
    )
    fullatt_indexes = vision["fullatt_block_indexes"]
    if not isinstance(fullatt_indexes, list):
        raise ValueError("base_model_architecture.vision.fullatt_block_indexes must be a JSON list")
    normalized_vision["fullatt_block_indexes"] = [
        _require_int(
            item,
            path=f"base_model_architecture.vision.fullatt_block_indexes[{index}]",
            minimum=0,
        )
        for index, item in enumerate(fullatt_indexes)
    ]
    return {
        "architectures": list(architectures),
        "decoder": {key: normalized_decoder[key] for key in sorted(normalized_decoder)},
        "model_type": model_type,
        "multimodal_token_ids": {key: normalized_token_ids[key] for key in sorted(normalized_token_ids)},
        "vision": {key: normalized_vision[key] for key in sorted(normalized_vision)},
    }


def capture_qwen2_5_vl_architecture(config) -> dict[str, object]:
    """Capture the normalized architecture fields used for Qwen PEFT compatibility."""
    hidden_size = int(getattr(config, "hidden_size"))
    attention_heads = int(getattr(config, "num_attention_heads"))
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        if hidden_size % attention_heads:
            raise ValueError("Qwen hidden_size must be divisible by num_attention_heads")
        head_dim = hidden_size // attention_heads
    architectures = getattr(config, "architectures", None) or []
    if isinstance(architectures, str) or not isinstance(architectures, Sequence):
        raise ValueError("Qwen architectures must be a sequence of class names")
    vision = getattr(config, "vision_config", None)
    if vision is None:
        raise ValueError("Qwen2.5-VL config is missing vision_config")
    snapshot = {
        "model_type": getattr(config, "model_type", None),
        "architectures": list(architectures),
        "decoder": {
            "hidden_size": hidden_size,
            "intermediate_size": int(getattr(config, "intermediate_size")),
            "num_hidden_layers": int(getattr(config, "num_hidden_layers")),
            "num_attention_heads": attention_heads,
            "num_key_value_heads": int(getattr(config, "num_key_value_heads")),
            "head_dim": int(head_dim), "hidden_act": getattr(config, "hidden_act", None),
            "vocab_size": int(getattr(config, "vocab_size")),
            "max_position_embeddings": int(getattr(config, "max_position_embeddings")),
            "rope_theta": float(getattr(config, "rope_theta")),
            "rms_norm_eps": getattr(config, "rms_norm_eps", None),
            "rope_scaling": getattr(config, "rope_scaling", None),
            "attention_dropout": getattr(config, "attention_dropout", None),
            "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
            "use_sliding_window": getattr(config, "use_sliding_window", False),
            "sliding_window": getattr(config, "sliding_window", None),
            "max_window_layers": getattr(config, "max_window_layers", None),
        },
        "multimodal_token_ids": {
            "image_token_id": getattr(config, "image_token_id", None),
            "video_token_id": getattr(config, "video_token_id", None),
            "vision_start_token_id": getattr(config, "vision_start_token_id", None),
            "vision_end_token_id": getattr(config, "vision_end_token_id", None),
        },
        "vision": {
            "depth": getattr(vision, "depth", None), "hidden_size": getattr(vision, "hidden_size", None),
            "intermediate_size": getattr(vision, "intermediate_size", None), "num_heads": getattr(vision, "num_heads", None),
            "in_channels": getattr(vision, "in_channels", None), "patch_size": getattr(vision, "patch_size", None),
            "temporal_patch_size": getattr(vision, "temporal_patch_size", None), "spatial_merge_size": getattr(vision, "spatial_merge_size", None),
            "window_size": getattr(vision, "window_size", None), "out_hidden_size": getattr(vision, "out_hidden_size", None),
            "tokens_per_second": getattr(vision, "tokens_per_second", None),
            "fullatt_block_indexes": list(getattr(vision, "fullatt_block_indexes", ())),
            "hidden_act": getattr(vision, "hidden_act", None),
        },
    }
    return normalize_qwen2_5_vl_architecture(snapshot)


def first_architecture_difference(expected: Mapping[str, object], actual: Mapping[str, object], path=""):
    """Return the first stable path where two architecture snapshots differ."""
    if type(expected) is not type(actual):
        return path, expected, actual
    if isinstance(expected, dict):
        for key in sorted(expected):
            result = first_architecture_difference(
                expected[key], actual[key], f"{path}.{key}" if path else key
            )
            if result is not None:
                return result
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return path, expected, actual
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            result = first_architecture_difference(
                expected_item, actual_item, f"{path}[{index}]"
            )
            if result is not None:
                return result
        return None
    if expected != actual:
        return path, expected, actual
    return None
