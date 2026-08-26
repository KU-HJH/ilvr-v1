"""Validate Stage 2 token metadata and checkpoint transitions."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Literal

import peft
from peft import PeftModel
from safetensors import SafetensorError, safe_open
from safetensors.torch import load as load_safetensors
from torch import nn

from peft_common import (
    capture_qwen2_5_vl_architecture,
    first_architecture_difference,
    normalize_qwen2_5_vl_architecture,
)
from stage1.token_rows import (
    SPARSE_TOKEN_ROWS_FILENAME,
    decode_sparse_token_rows_state,
    initialize_frozen_output_token_rows,
)
from stage1.checkpointing import (
    _NON_MODEL_BINARY_FILES,
    _RANK_RNG_STATE,
    _claims_weight_format,
    _is_full_weight_filename,
    _validate_full_serialization,
    Stage1PeftMetadataSnapshot,
    decode_stage1_peft_metadata_snapshot,
    read_stage1_peft_metadata_snapshot,
)
from .peft import Stage2TuningConfig, resolve_stage2_target_module_names
from transformers.utils.peft_utils import find_adapter_config_file
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentTokenIds,
    validate_stage2_config,
)


STAGE2_PEFT_METADATA_FILENAME = "stage2_peft_metadata.json"
STAGE2_PEFT_SCHEMA_VERSION = 3


class Stage2ArtifactKind(str, Enum):
    """Supported local Stage 2 artifact layouts."""

    FULL = "full"
    PEFT = "peft"


@dataclass(frozen=True)
class Stage2PeftMetadataSnapshot:
    """One canonical-path-bound immutable read of Stage 2 metadata."""

    artifact_dir: Path
    metadata_json: bytes


@dataclass(frozen=True)
class Stage2ArtifactInspection:
    """Classified Stage 2 artifact with retained immutable PEFT payloads."""

    artifact_kind: Stage2ArtifactKind
    peft_metadata_snapshot: Stage2PeftMetadataSnapshot | None = None
    adapter_config_json: bytes | None = None
    adapter_safetensors: bytes | None = None
    sparse_safetensors: bytes | None = None


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
        SPARSE_TOKEN_ROWS_FILENAME,
    }
)
def _canonical_stage2_artifact_dir(directory: str | Path) -> Path:
    candidate = Path(directory)
    if candidate.is_symlink():
        raise ValueError(
            f"Stage 2 artifact directory must not be a symlink: {candidate}"
        )
    try:
        canonical = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"Stage 2 artifact does not exist: {candidate}") from error
    if not canonical.is_dir():
        raise ValueError(f"Stage 2 artifact is not a directory: {candidate}")
    return canonical


def _read_stage2_regular_payload(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"invalid {label} payload file: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid {label} payload file: {path}") from error


def inspect_stage2_artifact(directory: str | Path) -> Stage2ArtifactInspection:
    """Classify once while retaining exact metadata and sparse payload bytes.

    Non-local references are left to ``from_pretrained`` and classified as full
    because their remote files cannot be inspected here. Local full artifacts
    follow Task 9's standalone/indexed-family contract; this repository's PEFT
    publication contract remains the unsharded safetensors marker set above.
    """
    candidate = Path(directory)
    if not candidate.exists():
        return Stage2ArtifactInspection(Stage2ArtifactKind.FULL)
    path = _canonical_stage2_artifact_dir(candidate)

    names = {entry.name for entry in path.iterdir()}
    present_peft = names & (_REQUIRED_PEFT_MARKERS | _OTHER_PEFT_MARKERS)
    present_full = {name for name in names if _is_full_weight_filename(name)}

    if present_peft and present_full:
        raise ValueError(
            "mixed Stage 2 full and PEFT artifact markers: "
            f"peft={sorted(present_peft)}, full={sorted(present_full)}"
        )

    recognized_weights = (
        present_full
        | (names & (_REQUIRED_PEFT_MARKERS | _OTHER_PEFT_MARKERS))
        | (names & _NON_MODEL_BINARY_FILES)
        | {name for name in names if _RANK_RNG_STATE.fullmatch(name)}
    )
    unknown_weights = sorted(
        name
        for name in names
        if _claims_weight_format(name) and name not in recognized_weights
    )
    if unknown_weights:
        raise ValueError(
            f"unknown Stage 2 model weight markers: {unknown_weights}"
        )

    required_present = present_peft & _REQUIRED_PEFT_MARKERS
    unsupported_present = present_peft & (
        _OTHER_PEFT_MARKERS - {SPARSE_TOKEN_ROWS_FILENAME}
    )
    if present_peft and (
        required_present != _REQUIRED_PEFT_MARKERS or unsupported_present
    ):
        missing = sorted(_REQUIRED_PEFT_MARKERS - present_peft)
        unsupported = sorted(unsupported_present)
        raise ValueError(
            "incomplete Stage 2 PEFT artifact: "
            f"missing={missing}, unsupported={unsupported}; "
            "adapter_model.safetensors is required"
        )
    if present_peft:
        payload_names = set(_REQUIRED_PEFT_MARKERS)
        if SPARSE_TOKEN_ROWS_FILENAME in names:
            payload_names.add(SPARSE_TOKEN_ROWS_FILENAME)
        invalid_files = sorted(
            name
            for name in payload_names
            if (path / name).is_symlink() or not (path / name).is_file()
        )
        if invalid_files:
            raise ValueError(
                "Stage 2 PEFT payloads must be regular non-symlink files: "
                f"{invalid_files}"
            )
        metadata_snapshot = read_stage2_peft_metadata_snapshot(path)
        metadata = decode_stage2_peft_metadata_snapshot(metadata_snapshot)
        adapter_config_json = _read_stage2_regular_payload(
            path / "adapter_config.json", label="Stage 2 adapter config"
        )
        adapter_safetensors = _read_stage2_regular_payload(
            path / "adapter_model.safetensors",
            label="Stage 2 adapter safetensors",
        )
        has_sparse_rows = SPARSE_TOKEN_ROWS_FILENAME in names
        expects_sparse_rows = metadata.token_row_source == "sparse_adapter"
        if expects_sparse_rows and not has_sparse_rows:
            raise ValueError(
                "incomplete Stage 2 PEFT token-row layout: "
                f"missing={SPARSE_TOKEN_ROWS_FILENAME}"
            )
        if has_sparse_rows and not expects_sparse_rows:
            raise ValueError(
                "mixed Stage 2 PEFT token-row layout: "
                "token_row_source=base but sparse token rows are present"
            )
        sparse_safetensors = None
        if expects_sparse_rows:
            sparse_path = path / SPARSE_TOKEN_ROWS_FILENAME
            try:
                sparse_safetensors = _read_stage2_regular_payload(
                    sparse_path, label="Stage 2 sparse token rows"
                )
                stored_ids, output_delta = decode_sparse_token_rows_state(
                    sparse_safetensors
                )
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"invalid sparse Stage 2 token-row artifact: {sparse_path}"
                ) from error
            expected_ids = (
                metadata.latent_pad_token_id,
                metadata.latent_start_token_id,
                metadata.latent_end_token_id,
            )
            if tuple(stored_ids.tolist()) != expected_ids:
                raise ValueError(
                    "sparse Stage 2 token IDs do not match metadata: "
                    f"stored={tuple(stored_ids.tolist())}, expected={expected_ids}"
                )
            expected_shape = (
                3,
                metadata.base_model_architecture["decoder"]["hidden_size"],
            )
            if tuple(output_delta.shape) != expected_shape:
                raise ValueError(
                    "sparse Stage 2 output shape does not match metadata: "
                    f"stored={tuple(output_delta.shape)}, expected={expected_shape}"
                )
        return Stage2ArtifactInspection(
            artifact_kind=Stage2ArtifactKind.PEFT,
            peft_metadata_snapshot=metadata_snapshot,
            adapter_config_json=adapter_config_json,
            adapter_safetensors=adapter_safetensors,
            sparse_safetensors=sparse_safetensors,
        )
    if present_full:
        _validate_full_serialization(path, names)
        return Stage2ArtifactInspection(Stage2ArtifactKind.FULL)
    raise ValueError(
        f"unrecognized local Stage 2 artifact (missing model weights): {path}"
    )


def classify_stage2_artifact(directory: str | Path) -> Stage2ArtifactKind:
    """Classify a Stage 2 artifact while preserving the public enum API."""

    return inspect_stage2_artifact(directory).artifact_kind


# Retain the fix-round-1 private name for internal compatibility.
_inspect_stage2_artifact = inspect_stage2_artifact


# Retain the Stage 2 helper names as direct compatibility wrappers.
def _normalize_base_model_architecture(value) -> dict[str, object]:
    return normalize_qwen2_5_vl_architecture(value)


def _capture_base_model_architecture(config) -> dict[str, object]:
    return capture_qwen2_5_vl_architecture(config)


def _first_architecture_difference(expected, actual, path=""):
    return first_architecture_difference(expected, actual, path)


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
    token_row_source: Literal["base", "sparse_adapter"]
    stage1_adapter_reference: str | None

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
        return migrate_stage2_metadata(payload)


_V3_STAGE2_PEFT_FIELDS = {field.name for field in fields(Stage2PeftMetadata)}
_V2_STAGE2_PEFT_FIELDS = frozenset(
    {
        "schema_version",
        "stage",
        "tuning_method",
        "base_model_reference",
        "base_model_architecture",
        "latent_size",
        "latent_start_token_id",
        "latent_pad_token_id",
        "latent_end_token_id",
        "supervise_latent_boundaries",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "requested_target_modules",
        "resolved_target_module_names",
        "peft_version",
    }
)


def _parse_stage2_metadata_v3(
    payload: Mapping[str, object],
) -> Stage2PeftMetadata:
    if set(payload) != _V3_STAGE2_PEFT_FIELDS:
        actual = set(payload)
        raise ValueError(
            "invalid Stage 2 PEFT metadata fields: "
            f"missing={sorted(_V3_STAGE2_PEFT_FIELDS - actual)}, "
            f"unknown={sorted(actual - _V3_STAGE2_PEFT_FIELDS)}"
        )
    if payload["stage"] != "stage2":
        raise ValueError("Stage 2 PEFT metadata must describe stage2")
    if payload["tuning_method"] not in ("lora", "dora"):
        raise ValueError("Stage 2 PEFT metadata requires LoRA or DoRA")
    if payload["token_row_source"] not in ("base", "sparse_adapter"):
        raise ValueError("token_row_source must be base or sparse_adapter")
    adapter_reference = payload["stage1_adapter_reference"]
    if adapter_reference is not None and (
        not isinstance(adapter_reference, str) or not adapter_reference
    ):
        raise ValueError(
            "stage1_adapter_reference must be a non-empty string or null"
        )

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

    converted: dict[str, tuple[str, ...]] = {}
    for name in ("requested_target_modules", "resolved_target_module_names"):
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

    return Stage2PeftMetadata(
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
        token_row_source=payload["token_row_source"],
        stage1_adapter_reference=adapter_reference,
    )


def migrate_stage2_metadata(raw: Mapping[str, object]) -> Stage2PeftMetadata:
    """Strictly parse v3 or migrate one exact complete v2 payload in memory."""
    if not isinstance(raw, Mapping):
        raise ValueError("Stage 2 PEFT metadata must be a JSON object")
    schema_version = raw.get("schema_version")
    if type(schema_version) is not int:
        raise ValueError("Stage 2 PEFT schema_version must be an integer")
    if schema_version == 1:
        raise ValueError(
            "unsupported Stage 2 PEFT metadata schema v1 because it lacks "
            "base architecture provenance"
        )
    if schema_version == 2:
        actual = set(raw)
        if actual != _V2_STAGE2_PEFT_FIELDS:
            raise ValueError(
                "Stage 2 PEFT migration requires an exact complete v2 payload: "
                f"missing={sorted(_V2_STAGE2_PEFT_FIELDS - actual)}, "
                f"unknown={sorted(actual - _V2_STAGE2_PEFT_FIELDS)}"
            )
        migrated = dict(raw)
        migrated.update(
            schema_version=STAGE2_PEFT_SCHEMA_VERSION,
            token_row_source="base",
            stage1_adapter_reference=None,
        )
        return _parse_stage2_metadata_v3(migrated)
    if schema_version == STAGE2_PEFT_SCHEMA_VERSION:
        return _parse_stage2_metadata_v3(raw)
    raise ValueError(
        "unsupported Stage 2 PEFT metadata schema: "
        f"expected={STAGE2_PEFT_SCHEMA_VERSION}, actual={schema_version}"
    )


def build_stage2_peft_metadata(
    *,
    tuning: Stage2TuningConfig,
    base_model_reference: str,
    model_config,
    token_ids: LatentTokenIds,
    resolved_target_names: Sequence[str],
    token_row_source: str = "base",
    stage1_adapter_reference: str | None = None,
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
            "token_row_source": token_row_source,
            "stage1_adapter_reference": stage1_adapter_reference,
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


def read_stage2_peft_metadata_snapshot(
    directory: str | Path,
) -> Stage2PeftMetadataSnapshot:
    """Read Stage 2 metadata once and bind it to a canonical artifact path."""

    artifact = _canonical_stage2_artifact_dir(directory)
    source = artifact / STAGE2_PEFT_METADATA_FILENAME
    if not source.is_file():
        raise ValueError(f"missing Stage 2 PEFT metadata: {source}")
    return Stage2PeftMetadataSnapshot(
        artifact_dir=artifact,
        metadata_json=_read_stage2_regular_payload(
            source, label="Stage 2 PEFT metadata"
        ),
    )


def decode_stage2_peft_metadata_snapshot(
    snapshot: Stage2PeftMetadataSnapshot,
) -> Stage2PeftMetadata:
    """Decode strict Stage 2 metadata only from immutable witnessed bytes."""

    if not isinstance(snapshot, Stage2PeftMetadataSnapshot):
        raise TypeError("snapshot must be a Stage2PeftMetadataSnapshot")
    try:
        payload = json.loads(snapshot.metadata_json.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Stage 2 PEFT metadata bytes") from error
    return Stage2PeftMetadata.from_dict(payload)


def load_stage2_peft_metadata(directory: str | Path) -> Stage2PeftMetadata:
    """Load and fail closed on a missing or malformed adapter sidecar."""

    return decode_stage2_peft_metadata_snapshot(
        read_stage2_peft_metadata_snapshot(directory)
    )


def _resolve_stage1_qwen_base(
    base_model, *, expected_stage: str | None = "stage1"
):
    """Require an unmodified full Qwen base with the requested stage tag."""
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
        or getattr(getattr(candidate, "config", None), "stage", None)
        != expected_stage
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


def _legacy_stage2_adapter_tensor_contract(
    qwen: nn.Module,
    metadata: Stage2PeftMetadata,
) -> dict[str, tuple[int, ...]]:
    """Return PEFT's explicit default-adapter saved-key contract.

    Inference supplies the contract produced by installed PEFT itself.  This
    explicit legacy path keeps checkpoint validation independent of a second
    model wrapping operation while using the same canonical saved-key schema.
    """

    modules = dict(qwen.named_modules())
    expected: dict[str, tuple[int, ...]] = {}
    for target in metadata.resolved_target_module_names:
        module = modules[target]
        prefix = f"base_model.model.{target}"
        expected[f"{prefix}.lora_A.weight"] = (
            metadata.lora_r,
            module.in_features,
        )
        expected[f"{prefix}.lora_B.weight"] = (
            module.out_features,
            metadata.lora_r,
        )
        if metadata.tuning_method == "dora":
            expected[f"{prefix}.lora_magnitude_vector"] = (
                module.out_features,
            )
    if metadata.token_row_source == "sparse_adapter":
        expected[
            "base_model.model.model.embed_tokens.token_adapter."
            "trainable_tokens_delta"
        ] = (
            3,
            int(metadata.base_model_architecture["decoder"]["hidden_size"]),
        )
    return expected


def _validate_stage2_adapter_tensor_shapes(
    tensor_shapes: Mapping[str, Sequence[int]],
    expected_shapes: Mapping[str, Sequence[int]],
    metadata: Stage2PeftMetadata,
) -> None:
    """Require an exact canonical PEFT saved-key and logical-shape contract."""

    actual_keys = set(tensor_shapes)
    expected_keys = set(expected_shapes)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    selective_key = (
        "base_model.model.model.embed_tokens.token_adapter."
        "trainable_tokens_delta"
    )
    if missing or unexpected:
        if missing == [selective_key] and not unexpected:
            raise ValueError(
                "Stage 2 adapter must contain exactly one selective input-row tensor; "
                "canonical adapter tensor fields mismatch"
            )
        missing_components = [
            component
            for component in ("lora_A", "lora_B", "lora_magnitude_vector")
            if any(component in key for key in missing)
        ]
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected adapter tensor fields={unexpected}")
        if missing_components:
            details.append(f"missing components={missing_components}")
        raise ValueError(
            "canonical adapter tensor fields mismatch: " + "; ".join(details)
        )

    for key in sorted(expected_keys):
        actual_shape = tuple(int(value) for value in tensor_shapes[key])
        expected_shape = tuple(int(value) for value in expected_shapes[key])
        if actual_shape == expected_shape:
            continue
        if key == selective_key:
            raise ValueError(
                "Stage 2 selective input-row shape mismatch: "
                f"adapter={list(actual_shape)}, expected={list(expected_shape)}"
            )
        target = key.removeprefix("base_model.model.").rsplit(".", 2)[0]
        raise ValueError(
            f"target shape mismatch at {target}: "
            f"adapter={list(actual_shape)}, expected={list(expected_shape)}"
        )


def validate_stage2_adapter_tensor_contract(
    adapter_safetensors: bytes,
    expected_shapes: Mapping[str, Sequence[int]],
    metadata: Stage2PeftMetadata,
) -> None:
    """Validate immutable adapter bytes against one canonical saved contract."""

    if type(adapter_safetensors) is not bytes:
        raise TypeError("adapter_safetensors must be bytes")
    try:
        adapter_state = load_safetensors(adapter_safetensors)
    except SafetensorError as error:
        raise ValueError("invalid Stage 2 adapter safetensors bytes") from error
    tensor_shapes = {key: value.shape for key, value in adapter_state.items()}
    _validate_stage2_adapter_tensor_shapes(
        tensor_shapes, expected_shapes, metadata
    )


def validate_stage2_peft_compatibility(
    base_model,
    tokenizer,
    metadata: Stage2PeftMetadata,
    adapter_config,
    adapter_path: str | Path,
    *,
    adapter_safetensors: bytes | None = None,
    expected_adapter_tensor_shapes: Mapping[str, Sequence[int]] | None = None,
    validate_adapter_tensors: bool = True,
) -> None:
    """Validate a relocated base against adapter metadata without reading weights."""
    qwen = _resolve_stage1_qwen_base(
        base_model,
        expected_stage=(
            "stage2" if metadata.token_row_source == "sparse_adapter" else "stage1"
        ),
    )
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
    trainable_token_indices = getattr(
        adapter_config, "trainable_token_indices", None
    )
    expected_sparse_ids = (
        metadata.latent_pad_token_id,
        metadata.latent_start_token_id,
        metadata.latent_end_token_id,
    )
    if metadata.token_row_source == "sparse_adapter":
        if (
            not isinstance(trainable_token_indices, dict)
            or set(trainable_token_indices) != {"model.embed_tokens"}
            or tuple(trainable_token_indices["model.embed_tokens"])
            != expected_sparse_ids
        ):
            raise ValueError(
                "sparse Stage 2 adapter selective input token IDs do not match metadata"
            )
    elif trainable_token_indices is not None:
        raise ValueError(
            "base Stage 2 token rows must not contain selective input token rows"
        )

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

    if not validate_adapter_tensors:
        return
    if expected_adapter_tensor_shapes is None:
        expected_adapter_tensor_shapes = _legacy_stage2_adapter_tensor_contract(
            qwen, metadata
        )
    if adapter_safetensors is None:
        tensor_path = Path(adapter_path) / "adapter_model.safetensors"
        if not tensor_path.is_file():
            raise ValueError(f"missing adapter safetensors: {tensor_path}")
        with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
            tensor_shapes = {
                key: tensors.get_slice(key).get_shape() for key in tensors.keys()
            }
        _validate_stage2_adapter_tensor_shapes(
            tensor_shapes, expected_adapter_tensor_shapes, metadata
        )
        return
    validate_stage2_adapter_tensor_contract(
        adapter_safetensors, expected_adapter_tensor_shapes, metadata
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

_PEFT_CONFIG_METADATA_FIELDS = frozenset(
    {
        "task_type",
        "auto_mapping",
        "base_model_name_or_path",
        "revision",
        "inference_mode",
    }
)


def _validate_stage2_sparse_original_base_model(base_model) -> None:
    """Reject every supported in-memory adapter marker before base mutation."""

    candidate = getattr(base_model, "module", base_model)
    markers = []
    if isinstance(candidate, PeftModel):
        markers.append("PeftModel")
    if getattr(candidate, "peft_config", None):
        markers.append("peft_config")
    if getattr(candidate, "_hf_peft_config_loaded", False):
        markers.append("_hf_peft_config_loaded")
    if markers:
        raise ValueError(
            "sparse Stage 2 resume requires an adapter-free original base; "
            f"adapter-bearing markers={sorted(markers)}"
        )


def prepare_stage2_sparse_token_base(
    base_model,
    tokenizer,
    metadata: Stage2PeftMetadata,
) -> LatentTokenIds:
    """Resize an original compatible base for one self-contained sparse adapter."""

    if metadata.token_row_source != "sparse_adapter":
        raise ValueError("sparse base preparation requires sparse_adapter metadata")
    _validate_stage2_sparse_original_base_model(base_model)
    qwen = _resolve_stage1_qwen_base(base_model, expected_stage=None)
    validate_stage2_sparse_original_base_config(qwen.config, metadata)
    token_ids = validate_stage2_sparse_processor(tokenizer, metadata)
    expected_vocab_size = int(
        metadata.base_model_architecture["decoder"]["vocab_size"]
    )
    ordered_ids = (
        token_ids.pad_token_id,
        token_ids.start_token_id,
        token_ids.end_token_id,
    )
    base_model.resize_token_embeddings(expected_vocab_size)
    initialize_frozen_output_token_rows(base_model, ordered_ids)
    configure_stage2_metadata(
        qwen.config,
        token_ids,
        latent_size=metadata.latent_size,
        supervise_latent_boundaries=metadata.supervise_latent_boundaries,
    )
    return token_ids


def validate_stage2_sparse_original_base_config(
    config, metadata: Stage2PeftMetadata
) -> None:
    """Validate the tag-free, pre-resize Qwen config for sparse resume."""

    if metadata.token_row_source != "sparse_adapter":
        raise ValueError("sparse base validation requires sparse_adapter metadata")
    if getattr(config, "stage", None) is not None:
        raise ValueError("sparse Stage 2 resume requires a tag-free original base")
    if getattr(config, "peft_type", None) is not None:
        raise ValueError(
            "sparse Stage 2 original base config contains PEFT peft_type metadata"
        )
    config_fields = getattr(config, "__dict__", {})
    peft_metadata_fields = sorted(
        field
        for field in _PEFT_CONFIG_METADATA_FIELDS
        if field in config_fields and config_fields[field] is not None
    )
    if peft_metadata_fields:
        raise ValueError(
            "sparse Stage 2 original base config contains PEFT metadata fields: "
            f"{peft_metadata_fields}"
        )
    expected_architecture = deepcopy(metadata.base_model_architecture)
    resized_vocab_size = int(expected_architecture["decoder"]["vocab_size"])
    expected_architecture["decoder"]["vocab_size"] = resized_vocab_size - 3
    actual_architecture = _capture_base_model_architecture(config)
    architecture_difference = _first_architecture_difference(
        expected_architecture, actual_architecture
    )
    if architecture_difference is not None:
        path, expected, actual = architecture_difference
        raise ValueError(
            f"original base architecture mismatch at {path}: "
            f"checkpoint={expected!r}, base={actual!r}"
        )


def validate_stage2_sparse_processor(
    tokenizer, metadata: Stage2PeftMetadata
) -> LatentTokenIds:
    """Validate exact appended sparse token strings, IDs, and vocabulary."""

    if metadata.token_row_source != "sparse_adapter":
        raise ValueError("sparse processor validation requires sparse_adapter metadata")
    expected_vocab_size = int(
        metadata.base_model_architecture["decoder"]["vocab_size"]
    )
    if len(tokenizer) != expected_vocab_size:
        raise ValueError(
            "sparse Stage 2 processor vocabulary mismatch: "
            f"processor={len(tokenizer)}, checkpoint={expected_vocab_size}"
        )
    token_ids = _resolve_tokenizer_latent_ids(tokenizer)
    ordered_ids = (
        token_ids.pad_token_id,
        token_ids.start_token_id,
        token_ids.end_token_id,
    )
    expected_ids = tuple(range(expected_vocab_size - 3, expected_vocab_size))
    if ordered_ids != expected_ids or ordered_ids != (
        metadata.latent_pad_token_id,
        metadata.latent_start_token_id,
        metadata.latent_end_token_id,
    ):
        raise ValueError(
            "sparse Stage 2 latent token IDs must be the exact appended rows"
        )
    for token, token_id in zip(LATENT_TOKEN_STRINGS.values(), ordered_ids):
        if tokenizer.convert_ids_to_tokens(token_id) != token:
            raise ValueError(
                "sparse Stage 2 latent token string round-trip mismatch: "
                f"token={token}, id={token_id}"
            )
    return token_ids


def _resolve_tokenizer_latent_ids(tokenizer) -> LatentTokenIds:
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

    return LatentTokenIds(**resolved)


def resolve_latent_token_ids(tokenizer, config) -> LatentTokenIds:
    """Resolve latent IDs without modifying the tokenizer or its vocabulary.

    Every latent token must be present in the tokenizer. Any canonical or legacy
    config alias that is present must agree with the tokenizer's ID.
    """
    tokenizer_ids = _resolve_tokenizer_latent_ids(tokenizer)

    validate_stage2_config(
        config,
        fallback_token_ids=tokenizer_ids,
        require_canonical_ids=False,
        require_latent_size=False,
    )
    for result_field, tokenizer_id in (
        ("pad_token_id", tokenizer_ids.pad_token_id),
        ("start_token_id", tokenizer_ids.start_token_id),
        ("end_token_id", tokenizer_ids.end_token_id),
    ):
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
    stage1_adapter_source: str | None = None
    stage1_metadata_snapshot: Stage1PeftMetadataSnapshot | None = None

    @property
    def model_source(self) -> str:
        """Backward-compatible name used by the full-checkpoint loader."""
        return self.base_model_source


def resolve_stage2_load_decision(
    *,
    load_model_path: str,
    stage1_adapter_path: str | None = None,
    resume_from_checkpoint_path: str | None,
    load_config,
    discover_adapter_config=find_adapter_config_file,
    source_provenance_kwargs: Mapping[str, object] | None = None,
    load_peft_metadata=load_stage2_peft_metadata,
    load_stage1_metadata_snapshot=read_stage1_peft_metadata_snapshot,
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

    if stage1_adapter_path is not None and resume_from_checkpoint_path is not None:
        raise ValueError(
            "Stage 1 adapter initialization and Stage 2 resume are mutually exclusive"
        )

    if stage1_adapter_path is not None:
        if requested_tuning.method == "full":
            raise ValueError("Stage 1 DoRA cannot initialize Stage 2 full tuning")
        if requested_tuning.method == "lora":
            raise ValueError("Stage 1 DoRA cannot initialize a new Stage 2 LoRA adapter")
        metadata_snapshot = load_stage1_metadata_snapshot(
            Path(stage1_adapter_path)
        )
        metadata = decode_stage1_peft_metadata_snapshot(metadata_snapshot)
        if metadata.tuning_method != "dora":
            raise ValueError(
                "fresh Stage 2 continuation requires a Stage 1 DoRA adapter"
            )
        for label, stage1_value, stage2_value in (
            ("latent_size", metadata.latent_size, latent_size),
            ("rank", metadata.rank, requested_tuning.r),
            ("alpha", metadata.alpha, requested_tuning.alpha),
            ("dropout", metadata.dropout, requested_tuning.dropout),
            (
                "requested target modules",
                metadata.requested_target_suffixes,
                requested_tuning.target_modules,
            ),
        ):
            if stage1_value != stage2_value:
                raise ValueError(
                    f"Stage 1/Stage 2 {label} mismatch: "
                    f"stage1={stage1_value}, stage2={stage2_value}"
                )
        if (Path(load_model_path) / "adapter_config.json").is_file():
            raise ValueError(
                "Stage 1 DoRA continuation requires the original full base model"
            )
        load_config(load_model_path)
        return Stage2LoadDecision(
            base_model_source=load_model_path,
            processor_source=stage1_adapter_path,
            adapter_source=None,
            trainer_resume_path=None,
            tuning=requested_tuning,
            latent_size=latent_size,
            supervise_latent_boundaries=supervise_latent_boundaries,
            stage1_adapter_source=stage1_adapter_path,
            stage1_metadata_snapshot=metadata_snapshot,
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
    if metadata.token_row_source == "sparse_adapter":
        if source_provenance_kwargs is None:
            source_provenance_kwargs = {}
        discovered_adapter_config = discover_adapter_config(
            load_model_path, **source_provenance_kwargs
        )
        if discovered_adapter_config is not None:
            raise ValueError(
                "sparse Stage 2 resume requires an adapter-free original base "
                f"model; discovered adapter config at {discovered_adapter_config}"
            )
        validate_stage2_sparse_original_base_config(
            load_config(load_model_path, **source_provenance_kwargs), metadata
        )
    else:
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
