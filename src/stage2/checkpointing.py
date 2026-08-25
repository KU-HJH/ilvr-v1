"""Validate Stage 2 token metadata and checkpoint transitions."""

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path

import peft
from peft import PeftModel
from safetensors import safe_open
from torch import nn

from .peft import Stage2TuningConfig, resolve_stage2_target_module_names
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentTokenIds,
    validate_stage2_config,
)


STAGE2_PEFT_METADATA_FILENAME = "stage2_peft_metadata.json"
STAGE2_PEFT_SCHEMA_VERSION = 2


class Stage2ArtifactKind(str, Enum):
    """Supported local Stage 2 artifact layouts."""

    FULL = "full"
    PEFT = "peft"


_REQUIRED_PEFT_MARKERS = frozenset(
    {
        STAGE2_PEFT_METADATA_FILENAME,
        "adapter_config.json",
        "adapter_model.safetensors",
    }
)
_OTHER_PEFT_MARKERS = frozenset(
    {
        "adapter_model.bin",
        "adapter_model.bin.index.json",
        "adapter_model.safetensors.index.json",
    }
)
_FULL_MODEL_MARKERS = frozenset(
    {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
)


def classify_stage2_artifact(directory: str | Path) -> Stage2ArtifactKind:
    """Classify a local artifact from small filenames, failing closed on mixtures.

    Non-local references are left to ``from_pretrained`` and classified as full
    because their remote files cannot be inspected here. This repository's PEFT
    publication contract is the unsharded safetensors marker set above.
    """
    path = Path(directory)
    if not path.exists():
        return Stage2ArtifactKind.FULL
    if not path.is_dir():
        raise ValueError(f"Stage 2 artifact is not a directory: {path}")

    names = {entry.name for entry in path.iterdir() if entry.is_file()}
    present_peft = names & (_REQUIRED_PEFT_MARKERS | _OTHER_PEFT_MARKERS)
    present_full = names & _FULL_MODEL_MARKERS

    if present_peft and present_full:
        raise ValueError(
            "mixed Stage 2 full and PEFT artifact markers: "
            f"peft={sorted(present_peft)}, full={sorted(present_full)}"
        )
    if present_peft and present_peft != _REQUIRED_PEFT_MARKERS:
        missing = sorted(_REQUIRED_PEFT_MARKERS - present_peft)
        unsupported = sorted(present_peft & _OTHER_PEFT_MARKERS)
        raise ValueError(
            "incomplete Stage 2 PEFT artifact: "
            f"missing={missing}, unsupported={unsupported}; "
            "adapter_model.safetensors is required"
        )
    if present_peft:
        return Stage2ArtifactKind.PEFT
    if present_full:
        return Stage2ArtifactKind.FULL
    raise ValueError(
        f"unrecognized local Stage 2 artifact (missing model weights): {path}"
    )


_ARCHITECTURE_KEYS = frozenset(
    {
        "model_type",
        "architectures",
        "decoder",
        "multimodal_token_ids",
        "vision",
    }
)
_DECODER_ARCHITECTURE_KEYS = frozenset(
    {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_act",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
        "rms_norm_eps",
        "rope_scaling",
        "attention_dropout",
        "tie_word_embeddings",
        "use_sliding_window",
        "sliding_window",
        "max_window_layers",
    }
)
_MULTIMODAL_TOKEN_KEYS = frozenset(
    {
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    }
)
_VISION_ARCHITECTURE_KEYS = frozenset(
    {
        "depth",
        "hidden_size",
        "intermediate_size",
        "num_heads",
        "in_channels",
        "patch_size",
        "temporal_patch_size",
        "spatial_merge_size",
        "window_size",
        "out_hidden_size",
        "tokens_per_second",
        "fullatt_block_indexes",
        "hidden_act",
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


def _normalize_base_model_architecture(value) -> dict[str, object]:
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
        root["decoder"],
        path="base_model_architecture.decoder",
        keys=_DECODER_ARCHITECTURE_KEYS,
    )
    normalized_decoder = {
        name: _require_int(
            decoder[name], path=f"base_model_architecture.decoder.{name}"
        )
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
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
        raise ValueError(
            "base_model_architecture.decoder.rope_theta must be finite and positive"
        )
    normalized_decoder["rope_theta"] = rope_theta
    rms_norm_eps = _require_finite_number(
        decoder["rms_norm_eps"],
        path="base_model_architecture.decoder.rms_norm_eps",
    )
    if rms_norm_eps <= 0:
        raise ValueError(
            "base_model_architecture.decoder.rms_norm_eps must be finite and positive"
        )
    normalized_decoder["rms_norm_eps"] = rms_norm_eps
    attention_dropout = _require_finite_number(
        decoder["attention_dropout"],
        path="base_model_architecture.decoder.attention_dropout",
    )
    if not 0.0 <= attention_dropout < 1.0:
        raise ValueError(
            "base_model_architecture.decoder.attention_dropout must be finite "
            "and in [0, 1)"
        )
    normalized_decoder["attention_dropout"] = attention_dropout
    if type(decoder["tie_word_embeddings"]) is not bool:
        raise ValueError(
            "base_model_architecture.decoder.tie_word_embeddings must be boolean"
        )
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
        raise ValueError(
            "base_model_architecture.decoder.use_sliding_window must be boolean"
        )
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
        root["vision"],
        path="base_model_architecture.vision",
        keys=_VISION_ARCHITECTURE_KEYS,
    )
    normalized_vision = {
        name: _require_int(
            vision[name], path=f"base_model_architecture.vision.{name}"
        )
        for name in _VISION_ARCHITECTURE_KEYS
        - {"fullatt_block_indexes", "hidden_act"}
    }
    normalized_vision["hidden_act"] = _require_nonempty_string(
        vision["hidden_act"], path="base_model_architecture.vision.hidden_act"
    )
    fullatt_indexes = vision["fullatt_block_indexes"]
    if not isinstance(fullatt_indexes, list):
        raise ValueError(
            "base_model_architecture.vision.fullatt_block_indexes must be a JSON list"
        )
    normalized_vision["fullatt_block_indexes"] = [
        _require_int(
            item,
            path=(
                "base_model_architecture.vision.fullatt_block_indexes"
                f"[{index}]"
            ),
            minimum=0,
        )
        for index, item in enumerate(fullatt_indexes)
    ]
    return {
        "architectures": list(architectures),
        "decoder": {key: normalized_decoder[key] for key in sorted(normalized_decoder)},
        "model_type": model_type,
        "multimodal_token_ids": {
            key: normalized_token_ids[key] for key in sorted(normalized_token_ids)
        },
        "vision": {key: normalized_vision[key] for key in sorted(normalized_vision)},
    }


def _capture_base_model_architecture(config) -> dict[str, object]:
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
            "head_dim": int(head_dim),
            "hidden_act": getattr(config, "hidden_act", None),
            "vocab_size": int(getattr(config, "vocab_size")),
            "max_position_embeddings": int(
                getattr(config, "max_position_embeddings")
            ),
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
            "depth": getattr(vision, "depth", None),
            "hidden_size": getattr(vision, "hidden_size", None),
            "intermediate_size": getattr(vision, "intermediate_size", None),
            "num_heads": getattr(vision, "num_heads", None),
            "in_channels": getattr(vision, "in_channels", None),
            "patch_size": getattr(vision, "patch_size", None),
            "temporal_patch_size": getattr(vision, "temporal_patch_size", None),
            "spatial_merge_size": getattr(vision, "spatial_merge_size", None),
            "window_size": getattr(vision, "window_size", None),
            "out_hidden_size": getattr(vision, "out_hidden_size", None),
            "tokens_per_second": getattr(vision, "tokens_per_second", None),
            "fullatt_block_indexes": list(
                getattr(vision, "fullatt_block_indexes", ())
            ),
            "hidden_act": getattr(vision, "hidden_act", None),
        },
    }
    return _normalize_base_model_architecture(snapshot)


def _first_architecture_difference(expected, actual, path=""):
    if type(expected) is not type(actual):
        return path, expected, actual
    if isinstance(expected, dict):
        for key in sorted(expected):
            result = _first_architecture_difference(
                expected[key], actual[key], f"{path}.{key}" if path else key
            )
            if result is not None:
                return result
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return path, expected, actual
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            result = _first_architecture_difference(
                expected_item, actual_item, f"{path}[{index}]"
            )
            if result is not None:
                return result
        return None
    if expected != actual:
        return path, expected, actual
    return None


@dataclass(frozen=True)
class Stage2PeftMetadata:
    """Portable provenance and structural contract for a Stage 2 adapter."""

    schema_version: int
    stage: str
    tuning_method: str
    base_model_reference: str
    base_model_architecture: dict[str, object]
    latent_size: int
    latent_start_token_id: int
    latent_pad_token_id: int
    latent_end_token_id: int
    supervise_latent_boundaries: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    requested_target_modules: tuple[str, ...]
    resolved_target_module_names: tuple[str, ...]
    peft_version: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["requested_target_modules"] = list(self.requested_target_modules)
        payload["resolved_target_module_names"] = list(
            self.resolved_target_module_names
        )
        type(self).from_dict(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Stage2PeftMetadata":
        if not isinstance(payload, Mapping):
            raise ValueError("Stage 2 PEFT metadata must be a JSON object")
        if payload.get("schema_version") != STAGE2_PEFT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported Stage 2 PEFT metadata schema: "
                f"expected={STAGE2_PEFT_SCHEMA_VERSION}, "
                f"actual={payload.get('schema_version')}; "
                "pre-v2 sidecars lack required base architecture provenance"
            )
        expected = {field.name for field in fields(cls)}
        actual = set(payload)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(
                "invalid Stage 2 PEFT metadata fields: "
                f"missing={missing}, unknown={unknown}"
            )
        if payload["stage"] != "stage2":
            raise ValueError("Stage 2 PEFT metadata must describe stage2")
        if payload["tuning_method"] not in ("lora", "dora"):
            raise ValueError("Stage 2 PEFT metadata requires LoRA or DoRA")

        integer_fields = (
            "schema_version",
            "latent_size",
            "latent_start_token_id",
            "latent_pad_token_id",
            "latent_end_token_id",
            "lora_r",
            "lora_alpha",
        )
        if any(type(payload[name]) is not int for name in integer_fields):
            raise ValueError("Stage 2 PEFT integer metadata has an invalid type")
        if type(payload["supervise_latent_boundaries"]) is not bool:
            raise ValueError("supervise_latent_boundaries must be a boolean")
        if type(payload["lora_dropout"]) not in (int, float):
            raise ValueError("lora_dropout must be numeric")
        string_fields = (
            "stage",
            "tuning_method",
            "base_model_reference",
            "peft_version",
        )
        if any(
            not isinstance(payload[name], str) or not payload[name]
            for name in string_fields
        ):
            raise ValueError("Stage 2 PEFT string metadata must be non-empty")

        sequence_fields = (
            "requested_target_modules",
            "resolved_target_module_names",
        )
        converted: dict[str, tuple[str, ...]] = {}
        for name in sequence_fields:
            value = payload[name]
            if (
                not isinstance(value, list)
                or not value
                or any(not isinstance(item, str) or not item for item in value)
            ):
                raise ValueError(f"{name} must be a non-empty JSON string list")
            converted[name] = tuple(value)

        tuning = Stage2TuningConfig(
            method=payload["tuning_method"],
            r=payload["lora_r"],
            alpha=payload["lora_alpha"],
            dropout=float(payload["lora_dropout"]),
            target_modules=converted["requested_target_modules"],
        )
        if payload["latent_size"] <= 0:
            raise ValueError("latent_size must be positive")
        if any(payload[name] < 0 for name in integer_fields[2:5]):
            raise ValueError("latent token IDs must be non-negative")
        if len(set(converted["resolved_target_module_names"])) != len(
            converted["resolved_target_module_names"]
        ):
            raise ValueError("resolved target module names must be unique")

        return cls(
            schema_version=STAGE2_PEFT_SCHEMA_VERSION,
            stage="stage2",
            tuning_method=tuning.method,
            base_model_reference=payload["base_model_reference"],
            base_model_architecture=_normalize_base_model_architecture(
                payload["base_model_architecture"]
            ),
            latent_size=payload["latent_size"],
            latent_start_token_id=payload["latent_start_token_id"],
            latent_pad_token_id=payload["latent_pad_token_id"],
            latent_end_token_id=payload["latent_end_token_id"],
            supervise_latent_boundaries=payload["supervise_latent_boundaries"],
            lora_r=tuning.r,
            lora_alpha=tuning.alpha,
            lora_dropout=tuning.dropout,
            requested_target_modules=tuning.target_modules,
            resolved_target_module_names=converted["resolved_target_module_names"],
            peft_version=payload["peft_version"],
        )


def build_stage2_peft_metadata(
    *,
    tuning: Stage2TuningConfig,
    base_model_reference: str,
    model_config,
    token_ids: LatentTokenIds,
    resolved_target_names: Sequence[str],
) -> Stage2PeftMetadata:
    """Build the sidecar for a fresh Stage 2 PEFT run."""
    if not tuning.is_peft:
        raise ValueError("Stage 2 PEFT metadata requires LoRA or DoRA tuning")
    return Stage2PeftMetadata.from_dict(
        {
            "schema_version": STAGE2_PEFT_SCHEMA_VERSION,
            "stage": "stage2",
            "tuning_method": tuning.method,
            "base_model_reference": str(base_model_reference),
            "base_model_architecture": _capture_base_model_architecture(model_config),
            "latent_size": getattr(model_config, "latent_size", None),
            "latent_start_token_id": token_ids.start_token_id,
            "latent_pad_token_id": token_ids.pad_token_id,
            "latent_end_token_id": token_ids.end_token_id,
            "supervise_latent_boundaries": getattr(
                model_config, "supervise_latent_boundaries", None
            ),
            "lora_r": tuning.r,
            "lora_alpha": tuning.alpha,
            "lora_dropout": tuning.dropout,
            "requested_target_modules": list(tuning.target_modules),
            "resolved_target_module_names": list(resolved_target_names),
            "peft_version": peft.__version__,
        }
    )


def save_stage2_peft_metadata(
    directory: str | Path, metadata: Stage2PeftMetadata
) -> None:
    """Atomically publish one validated Stage 2 adapter sidecar."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / STAGE2_PEFT_METADATA_FILENAME
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata.to_dict(), sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def load_stage2_peft_metadata(directory: str | Path) -> Stage2PeftMetadata:
    """Load and fail closed on a missing or malformed adapter sidecar."""
    source = Path(directory) / STAGE2_PEFT_METADATA_FILENAME
    if not source.is_file():
        raise ValueError(f"missing Stage 2 PEFT metadata: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Stage 2 PEFT metadata: {source}") from error
    return Stage2PeftMetadata.from_dict(payload)


def _resolve_stage1_qwen_base(base_model):
    """Require the unmodified full Stage 1 Qwen base used before PEFT loading."""
    candidate = getattr(base_model, "module", base_model)
    if isinstance(candidate, PeftModel):
        raise ValueError("Stage 1 PEFT-only base is not supported")
    required = (
        "prepare_stage2_base_inputs_embeds",
        "get_rope_index",
        "stage2_rollout_engine",
        "model",
        "lm_head",
    )
    if (
        any(not hasattr(candidate, attribute) for attribute in required)
        or getattr(getattr(candidate, "config", None), "stage", None) != "stage1"
    ):
        raise TypeError("model does not satisfy the Stage 1 Qwen interface")
    return candidate


def _expected_qwen_projection_weight_shape(config, suffix: str) -> tuple[int, int]:
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    attention_heads = int(config.num_attention_heads)
    key_value_heads = int(config.num_key_value_heads)
    head_dim = getattr(config, "head_dim", None) or hidden_size // attention_heads
    attention_width = attention_heads * int(head_dim)
    key_value_width = key_value_heads * int(head_dim)
    return {
        "q_proj": (attention_width, hidden_size),
        "k_proj": (key_value_width, hidden_size),
        "v_proj": (key_value_width, hidden_size),
        "o_proj": (hidden_size, attention_width),
        "gate_proj": (intermediate_size, hidden_size),
        "up_proj": (intermediate_size, hidden_size),
        "down_proj": (hidden_size, intermediate_size),
    }[suffix]


def validate_stage2_peft_compatibility(
    base_model,
    tokenizer,
    metadata: Stage2PeftMetadata,
    adapter_config,
    adapter_path: str | Path,
) -> None:
    """Validate a relocated base against adapter metadata without reading weights."""
    qwen = _resolve_stage1_qwen_base(base_model)
    config = qwen.config
    current_architecture = _capture_base_model_architecture(config)
    architecture_difference = _first_architecture_difference(
        metadata.base_model_architecture, current_architecture
    )
    if architecture_difference is not None:
        path, expected, actual = architecture_difference
        raise ValueError(
            f"base architecture mismatch at {path}: "
            f"checkpoint={expected!r}, base={actual!r}"
        )
    token_ids = resolve_latent_token_ids(tokenizer, config)
    for field, actual in (
        ("latent_pad_token_id", token_ids.pad_token_id),
        ("latent_start_token_id", token_ids.start_token_id),
        ("latent_end_token_id", token_ids.end_token_id),
    ):
        expected = getattr(metadata, field)
        if actual != expected:
            raise ValueError(
                f"{field} mismatch: checkpoint={expected}, base={actual}"
            )
    for field in ("latent_size", "supervise_latent_boundaries"):
        actual = getattr(config, field, None)
        expected = getattr(metadata, field)
        if actual != expected:
            raise ValueError(f"{field} mismatch: checkpoint={expected}, base={actual}")

    source = getattr(config, "_name_or_path", "")
    if source and source != metadata.base_model_reference:
        logging.getLogger(__name__).warning(
            "validating relocated Stage 1 base %s (recorded source: %s)",
            source,
            metadata.base_model_reference,
        )

    expected_use_dora = metadata.tuning_method == "dora"
    actual_use_dora = getattr(adapter_config, "use_dora", None)
    if actual_use_dora is not expected_use_dora:
        raise ValueError(
            "adapter method mismatch: "
            f"use_dora mismatch (checkpoint={expected_use_dora}, "
            f"adapter={actual_use_dora})"
        )
    peft_type = getattr(adapter_config, "peft_type", None)
    peft_type_value = getattr(peft_type, "value", peft_type)
    if peft_type_value != "LORA":
        raise ValueError(f"adapter method mismatch: unsupported peft_type={peft_type}")
    if getattr(adapter_config, "use_rslora", None) is not False:
        raise ValueError("Stage 2 adapters require use_rslora=False")
    for pattern_name in ("rank_pattern", "alpha_pattern"):
        if getattr(adapter_config, pattern_name, None):
            raise ValueError(f"Stage 2 adapters do not support {pattern_name}")
    task_type = getattr(adapter_config, "task_type", None)
    task_type_value = getattr(task_type, "value", task_type)
    if task_type_value != "CAUSAL_LM":
        raise ValueError(
            f"Stage 2 adapters require task_type=CAUSAL_LM, got {task_type}"
        )
    if getattr(adapter_config, "bias", None) != "none":
        raise ValueError("Stage 2 adapters require bias=none")

    for label, attribute, expected in (
        ("rank", "r", metadata.lora_r),
        ("alpha", "lora_alpha", metadata.lora_alpha),
        ("dropout", "lora_dropout", metadata.lora_dropout),
    ):
        actual = getattr(adapter_config, attribute, None)
        if actual != expected:
            raise ValueError(
                f"adapter {label} mismatch: checkpoint={expected}, adapter={actual}"
            )
    if getattr(adapter_config, "modules_to_save", None) is not None:
        raise ValueError("adapter must not contain modules_to_save")

    modules = dict(qwen.named_modules())
    for target in metadata.resolved_target_module_names:
        if target not in modules:
            raise ValueError(f"missing resolved target: {target}")
        if not isinstance(modules[target], nn.Linear):
            raise ValueError(f"resolved target is not linear: {target}")
    target_modules = tuple(getattr(adapter_config, "target_modules", ()) or ())
    if set(target_modules) != set(metadata.resolved_target_module_names):
        raise ValueError("adapter resolved targets do not match metadata")
    suffixes = tuple(
        dict.fromkeys(name.rsplit(".", 1)[-1] for name in target_modules)
    )
    if set(suffixes) != set(metadata.requested_target_modules):
        raise ValueError("adapter requested targets do not match metadata")
    current_resolved_targets = resolve_stage2_target_module_names(
        qwen, metadata.requested_target_modules
    )
    if current_resolved_targets != metadata.resolved_target_module_names:
        raise ValueError(
            "resolved target topology mismatch: "
            f"checkpoint={metadata.resolved_target_module_names}, "
            f"base={current_resolved_targets}"
        )

    for target in metadata.resolved_target_module_names:
        suffix = target.rsplit(".", 1)[-1]
        expected_shape = _expected_qwen_projection_weight_shape(config, suffix)
        actual_shape = tuple(modules[target].weight.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"hidden size or target shape mismatch at {target}: "
                f"architecture={expected_shape}, target={actual_shape}"
            )

    tensor_path = Path(adapter_path) / "adapter_model.safetensors"
    if not tensor_path.is_file():
        raise ValueError(f"missing adapter safetensors: {tensor_path}")
    with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
        keys = tuple(tensors.keys())
        for target in metadata.resolved_target_module_names:
            module = modules[target]
            expected_shapes = {
                "lora_A": [metadata.lora_r, module.in_features],
                "lora_B": [module.out_features, metadata.lora_r],
            }
            if expected_use_dora:
                expected_shapes["lora_magnitude_vector"] = [module.out_features]
            for tensor_name, expected_shape in expected_shapes.items():
                parameter_suffix = (
                    "" if tensor_name == "lora_magnitude_vector" else ".weight"
                )
                suffix = f".{target}.{tensor_name}{parameter_suffix}"
                matches = [key for key in keys if key.endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError(
                        f"adapter tensor mismatch for {target}: expected {tensor_name}"
                    )
                actual_shape = tensors.get_slice(matches[0]).get_shape()
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"target shape mismatch at {target}: "
                        f"adapter={actual_shape}, expected={expected_shape}"
                    )


LATENT_TOKEN_STRINGS = {
    "pad_token_id": "<|latent_pad|>",
    "start_token_id": "<|latent_start|>",
    "end_token_id": "<|latent_end|>",
}

LATENT_CONFIG_FIELDS = {
    "pad_token_id": ("latent_pad_token_id", "latent_token_id"),
    "start_token_id": ("latent_start_token_id", "latent_start_id"),
    "end_token_id": ("latent_end_token_id", "latent_end_id"),
}


def resolve_latent_token_ids(tokenizer, config) -> LatentTokenIds:
    """Resolve latent IDs without modifying the tokenizer or its vocabulary.

    Every latent token must be present in the tokenizer. Any canonical or legacy
    config alias that is present must agree with the tokenizer's ID.
    """
    resolved: dict[str, int] = {}
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    unknown_token = getattr(tokenizer, "unk_token", None)

    for result_field, token in LATENT_TOKEN_STRINGS.items():
        tokenizer_id = tokenizer.convert_tokens_to_ids(token)
        if tokenizer_id is None or (
            unknown_id is not None
            and tokenizer_id == unknown_id
            and token != unknown_token
        ):
            raise ValueError(f"tokenizer does not contain {token}")

        tokenizer_id = int(tokenizer_id)
        resolved[result_field] = tokenizer_id

    tokenizer_ids = LatentTokenIds(**resolved)
    validate_stage2_config(
        config,
        fallback_token_ids=tokenizer_ids,
        require_canonical_ids=False,
        require_latent_size=False,
    )
    for result_field, tokenizer_id in resolved.items():
        canonical_field = LATENT_CONFIG_FIELDS[result_field][0]
        for config_field in LATENT_CONFIG_FIELDS[result_field]:
            config_id = getattr(config, config_field, None)
            if config_id is not None and int(config_id) != tokenizer_id:
                alias = "" if config_field == canonical_field else f" via {config_field}"
                raise ValueError(
                    f"{canonical_field} mismatch{alias}: "
                    f"tokenizer={tokenizer_id}, config={config_id}"
                )

    return tokenizer_ids


def configure_stage2_metadata(
    config,
    token_ids: LatentTokenIds,
    *,
    latent_size: int,
    supervise_latent_boundaries: bool,
) -> None:
    """Write the complete, backwards-compatible Stage 2 config metadata."""
    config.stage = "stage2"
    config.latent_size = latent_size
    config.supervise_latent_boundaries = supervise_latent_boundaries
    config.latent_pad_token_id = config.latent_token_id = token_ids.pad_token_id
    config.latent_start_token_id = config.latent_start_id = token_ids.start_token_id
    config.latent_end_token_id = config.latent_end_id = token_ids.end_token_id


@dataclass(frozen=True)
class Stage2LoadDecision:
    """The model, processor, and trainer sources for one Stage 2 run."""

    base_model_source: str
    processor_source: str
    adapter_source: str | None
    trainer_resume_path: str | None
    tuning: Stage2TuningConfig
    latent_size: int
    supervise_latent_boundaries: bool
    peft_metadata: Stage2PeftMetadata | None = None

    @property
    def model_source(self) -> str:
        """Backward-compatible name used by the full-checkpoint loader."""
        return self.base_model_source


def resolve_stage2_load_decision(
    *,
    load_model_path: str,
    resume_from_checkpoint_path: str | None,
    load_config,
    load_peft_metadata=load_stage2_peft_metadata,
    latent_size: int,
    supervise_latent_boundaries: bool,
    tuning_method: str | None = None,
    tuning_method_explicit: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_target_modules: str | Sequence[str] = ("q_proj",),
) -> Stage2LoadDecision:
    """Choose fresh/full/adapter Stage 2 initialization and resume sources.

    A fresh run reads model and processor artifacts from a Stage 1 source but
    intentionally returns no trainer resume path, so optimizer/scheduler/global
    state starts new. Full resume remains single-source. Adapter resume keeps an
    explicit, structurally validated Stage 1 base separate from adapter/trainer
    state and preserves the checkpoint sidecar as the provenance authority.
    """
    requested_method = tuning_method or "full"
    requested_tuning = Stage2TuningConfig(
        method=requested_method,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=tuple(
            item.strip() for item in lora_target_modules.split(",")
        )
        if isinstance(lora_target_modules, str)
        else tuple(lora_target_modules),
    )

    def require_full_stage1(source: str):
        adapter_config_file = Path(source) / "adapter_config.json"
        if adapter_config_file.is_file():
            raise ValueError("Stage 1 PEFT-only base is not supported")
        config = load_config(source)
        if getattr(config, "stage", None) != "stage1":
            raise ValueError("fresh Stage 2 requires a stage1 checkpoint")
        if getattr(config, "peft_type", None) is not None:
            raise ValueError("Stage 1 PEFT-only base is not supported")
        return config

    if resume_from_checkpoint_path is None:
        require_full_stage1(load_model_path)
        return Stage2LoadDecision(
            load_model_path,
            load_model_path,
            None,
            None,
            requested_tuning,
            latent_size,
            supervise_latent_boundaries,
        )

    resume_path = Path(resume_from_checkpoint_path)
    local_artifact_kind = (
        classify_stage2_artifact(resume_path) if resume_path.is_dir() else None
    )
    if local_artifact_kind is Stage2ArtifactKind.FULL:
        metadata = None
    else:
        try:
            metadata = load_peft_metadata(resume_from_checkpoint_path)
        except FileNotFoundError:
            metadata = None
        except ValueError as error:
            if not str(error).startswith("missing Stage 2 PEFT metadata"):
                raise
            metadata = None

    if metadata is None:
        if (Path(resume_from_checkpoint_path) / "adapter_config.json").is_file():
            raise ValueError(
                "missing Stage 2 PEFT metadata: "
                f"{Path(resume_from_checkpoint_path) / STAGE2_PEFT_METADATA_FILENAME}"
            )
        resume_config = load_config(resume_from_checkpoint_path)
        if getattr(resume_config, "stage", None) != "stage2":
            raise ValueError("expected stage2 checkpoint for resume")
        if requested_tuning.is_peft:
            raise ValueError("requested PEFT resume is missing Stage 2 PEFT metadata")
        for field, requested in (
            ("latent_size", latent_size),
            ("supervise_latent_boundaries", supervise_latent_boundaries),
        ):
            checkpoint_value = getattr(resume_config, field, None)
            if checkpoint_value != requested:
                raise ValueError(
                    f"cannot resume Stage 2 with changed {field}: "
                    f"checkpoint={checkpoint_value}, requested={requested}"
                )
        return Stage2LoadDecision(
            resume_from_checkpoint_path,
            resume_from_checkpoint_path,
            None,
            resume_from_checkpoint_path,
            requested_tuning,
            resume_config.latent_size,
            resume_config.supervise_latent_boundaries,
        )

    if tuning_method_explicit and requested_method == "full":
        raise ValueError("cannot request full tuning against a PEFT checkpoint")
    if tuning_method_explicit and requested_method != metadata.tuning_method:
        raise ValueError(
            "cannot switch Stage 2 PEFT method from "
            f"{metadata.tuning_method} to {requested_method}"
        )
    if tuning_method_explicit:
        for label, checkpoint_value, requested_value in (
            ("rank", metadata.lora_r, requested_tuning.r),
            ("alpha", metadata.lora_alpha, requested_tuning.alpha),
            ("dropout", metadata.lora_dropout, requested_tuning.dropout),
            (
                "target_modules",
                metadata.requested_target_modules,
                requested_tuning.target_modules,
            ),
        ):
            if checkpoint_value != requested_value:
                raise ValueError(
                    f"cannot resume Stage 2 with changed {label}: "
                    f"checkpoint={checkpoint_value}, requested={requested_value}"
                )
    for field, requested in (
        ("latent_size", latent_size),
        ("supervise_latent_boundaries", supervise_latent_boundaries),
    ):
        checkpoint_value = getattr(metadata, field)
        if checkpoint_value != requested:
            raise ValueError(
                f"cannot resume Stage 2 with changed {field}: "
                f"checkpoint={checkpoint_value}, requested={requested}"
            )
    require_full_stage1(load_model_path)
    metadata_tuning = Stage2TuningConfig(
        method=metadata.tuning_method,
        r=metadata.lora_r,
        alpha=metadata.lora_alpha,
        dropout=metadata.lora_dropout,
        target_modules=metadata.requested_target_modules,
    )
    return Stage2LoadDecision(
        load_model_path,
        resume_from_checkpoint_path,
        resume_from_checkpoint_path,
        resume_from_checkpoint_path,
        metadata_tuning,
        metadata.latent_size,
        metadata.supervise_latent_boundaries,
        metadata,
    )
