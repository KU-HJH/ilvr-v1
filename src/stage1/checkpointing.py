"""Strict Stage 1 full-model and adapter artifact lifecycles."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Literal

import peft
import torch
from accelerate import init_empty_weights
from peft import (
    PeftConfig,
    PeftModel,
    get_peft_model,
    set_peft_model_state_dict,
)
from safetensors import SafetensorError, safe_open
from safetensors.torch import load as load_safetensors
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from peft_common import (
    capture_qwen2_5_vl_architecture,
    first_architecture_difference,
    normalize_qwen2_5_vl_architecture,
    resolve_qwen2_5_vl_model,
)
from stage1.ema import _parameter_shape
from stage1.peft import (
    _configured_target_patterns_match,
    _resolve_linear_decoder_targets,
    _validate_target_selector_modifiers_absent,
    validate_stage1_trainable_boundary,
)
from stage1.token_rows import (
    SPARSE_TOKEN_ROWS_FILENAME,
    apply_sparse_token_rows_state,
    attach_sparse_output_token_rows,
    decode_sparse_token_rows_state,
    find_sparse_output_token_rows,
    initialize_frozen_output_token_rows,
    save_sparse_token_rows,
)
from stage1.types import Stage1TuningConfig


STAGE1_PEFT_METADATA_FILENAME = "stage1_peft_metadata.json"
STAGE1_PEFT_SCHEMA_VERSION = 2
STAGE1_REPOSITORY_FORMAT_VERSION = 1

_ADAPTER_CONFIG_FILENAME = "adapter_config.json"
_ADAPTER_SAFE_FILENAME = "adapter_model.safetensors"
_ADAPTER_BINARY_FILENAME = "adapter_model.bin"
_CANONICAL_LATENT_TOKEN_STRINGS = (
    "<|latent_pad|>",
    "<|latent_start|>",
    "<|latent_end|>",
)
_INPUT_ROW_FORMAT = "peft_trainable_tokens_v1"
_OUTPUT_ROW_FORMAT = "ilvr_sparse_output_rows_v1"
_ADAPTER_NAME = "default"
_EMBEDDING_TARGET = "model.embed_tokens"

_REQUIRED_PEFT_MARKERS = frozenset(
    {
        _ADAPTER_CONFIG_FILENAME,
        _ADAPTER_SAFE_FILENAME,
        SPARSE_TOKEN_ROWS_FILENAME,
        STAGE1_PEFT_METADATA_FILENAME,
    }
)
_OTHER_PEFT_MARKERS = frozenset(
    {
        _ADAPTER_BINARY_FILENAME,
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
_NON_MODEL_BINARY_FILES = frozenset(
    {
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rng_state.pth",
        "training_args.bin",
    }
)
_FULL_SAFE_SHARD = re.compile(
    r"model-(\d{5})-of-(\d{5})\.safetensors\Z"
)
_FULL_BINARY_SHARD = re.compile(
    r"pytorch_model-(\d{5})-of-(\d{5})\.bin\Z"
)
_RANK_RNG_STATE = re.compile(r"rng_state_(0|[1-9][0-9]*)\.pth\Z")
_NUMBERED_CHECKPOINT = re.compile(r"checkpoint-(\d+)\Z")
_DEEPSPEED_TAG = re.compile(r"global_step(0|[1-9][0-9]*)\Z")
_DEEPSPEED_ZERO3_MODEL_STATE = re.compile(
    r"zero_pp_rank_(0|[1-9][0-9]*)_mp_rank_([0-9]{2,})_model_states\.pt\Z"
)
_DEEPSPEED_REPLICATED_MODEL_STATE = re.compile(
    r"mp_rank_([0-9]{2,})_model_states\.pt\Z"
)
_DEEPSPEED_OPTIMIZER_STATE = re.compile(
    r"(bf16_)?zero_pp_rank_(0|[1-9][0-9]*)_mp_rank_([0-9]{2,})_optim_states\.pt\Z"
)
_MAX_FULL_INDEX_BYTES = 4 * 1024 * 1024
_MAX_DEEPSPEED_LATEST_BYTES = 1024
_MAX_TRAINER_STATE_BYTES = 4 * 1024 * 1024


class Stage1ArtifactKind(str, Enum):
    """Supported local Stage 1 weight layouts."""

    FULL = "full"
    PEFT = "peft"


def _is_full_weight_filename(name: str) -> bool:
    return (
        name in _FULL_MODEL_MARKERS
        or _FULL_SAFE_SHARD.fullmatch(name) is not None
        or _FULL_BINARY_SHARD.fullmatch(name) is not None
    )


def _claims_weight_format(name: str) -> bool:
    return (
        name.endswith((".safetensors", ".bin", ".pt", ".pth"))
        or name.endswith((".safetensors.index.json", ".bin.index.json"))
    )


def _validate_full_shard_index(
    directory: Path,
    *,
    index_name: str,
    shard_pattern: re.Pattern[str],
    actual_shards: set[str],
) -> None:
    index_path = directory / index_name
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError(f"invalid full-model shard index: {index_path}")
    try:
        if index_path.stat().st_size > _MAX_FULL_INDEX_BYTES:
            raise ValueError(f"full-model shard index is too large: {index_path}")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid full-model shard index: {index_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("full-model shard index must be a JSON object")
    weight_map = payload.get("weight_map")
    if (
        not isinstance(weight_map, Mapping)
        or not weight_map
        or any(not isinstance(key, str) or not key for key in weight_map)
    ):
        raise ValueError("full-model shard index requires a non-empty weight_map")

    references: set[str] = set()
    for reference in weight_map.values():
        if (
            not isinstance(reference, str)
            or not reference
            or Path(reference).name != reference
            or shard_pattern.fullmatch(reference) is None
        ):
            raise ValueError(
                f"invalid full-model shard reference: {reference!r}"
            )
        references.add(reference)
    if references != actual_shards:
        raise ValueError(
            "full-model shard set does not match index: "
            f"missing={sorted(references - actual_shards)}, "
            f"orphan={sorted(actual_shards - references)}"
        )
    if any(
        (directory / shard).is_symlink()
        or not (directory / shard).is_file()
        for shard in references
    ):
        raise ValueError("full-model shard index references an invalid file")

    positions = []
    totals = set()
    for shard in references:
        match = shard_pattern.fullmatch(shard)
        if match is None:
            raise ValueError(f"invalid full-model shard name: {shard}")
        positions.append(int(match.group(1)))
        totals.add(int(match.group(2)))
    if (
        len(totals) != 1
        or next(iter(totals)) != len(references)
        or sorted(positions) != list(range(1, len(references) + 1))
    ):
        raise ValueError("full-model shard numbering is inconsistent")


def _validate_full_serialization(directory: Path, names: set[str]) -> None:
    safe_single = "model.safetensors" in names
    binary_single = "pytorch_model.bin" in names
    safe_index = "model.safetensors.index.json" in names
    binary_index = "pytorch_model.bin.index.json" in names
    safe_shards = {name for name in names if _FULL_SAFE_SHARD.fullmatch(name)}
    binary_shards = {
        name for name in names if _FULL_BINARY_SHARD.fullmatch(name)
    }
    has_safe = safe_single or safe_index or bool(safe_shards)
    has_binary = binary_single or binary_index or bool(binary_shards)
    if has_safe and has_binary:
        raise ValueError("mixed full-model serialization families")
    if safe_single:
        if safe_index or safe_shards:
            raise ValueError("mixed standalone and sharded full serialization")
        weight = directory / "model.safetensors"
        if weight.is_symlink() or not weight.is_file():
            raise ValueError(
                f"standalone full-model weight must be a regular file: {weight}"
            )
        return
    if binary_single:
        if binary_index or binary_shards:
            raise ValueError("mixed standalone and sharded full serialization")
        weight = directory / "pytorch_model.bin"
        if weight.is_symlink() or not weight.is_file():
            raise ValueError(
                f"standalone full-model weight must be a regular file: {weight}"
            )
        return
    if safe_index:
        _validate_full_shard_index(
            directory,
            index_name="model.safetensors.index.json",
            shard_pattern=_FULL_SAFE_SHARD,
            actual_shards=safe_shards,
        )
        return
    if binary_index:
        _validate_full_shard_index(
            directory,
            index_name="pytorch_model.bin.index.json",
            shard_pattern=_FULL_BINARY_SHARD,
            actual_shards=binary_shards,
        )
        return
    raise ValueError("full-model shards require their matching parsed index")


def _classify_stage1_artifact_with_metadata(
    path: Path,
    *,
    metadata_snapshot: Stage1PeftMetadataSnapshot | None = None,
) -> tuple[
    Stage1ArtifactKind,
    Stage1PeftMetadata | None,
    Stage1PeftMetadataSnapshot | None,
]:
    """Classify one local artifact from filenames and a strict PEFT sidecar.

    Classification deliberately performs no model/config ``from_pretrained``
    call. A PEFT-shaped directory is never reinterpreted as a full checkpoint.
    """

    directory = Path(path)
    if not directory.exists():
        raise ValueError(f"Stage 1 artifact does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Stage 1 artifact is not a directory: {directory}")

    names = {entry.name for entry in directory.iterdir() if entry.is_file()}
    present_peft = names & (_REQUIRED_PEFT_MARKERS | _OTHER_PEFT_MARKERS)
    present_full = {name for name in names if _is_full_weight_filename(name)}
    if present_peft and present_full:
        raise ValueError(
            "mixed Stage 1 full and PEFT artifact markers: "
            f"PEFT={sorted(present_peft)}, full={sorted(present_full)}"
        )

    recognized_weights = (
        present_full
        | (names & {_ADAPTER_SAFE_FILENAME, _ADAPTER_BINARY_FILENAME})
        | ({SPARSE_TOKEN_ROWS_FILENAME} if SPARSE_TOKEN_ROWS_FILENAME in names else set())
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
            f"unknown Stage 1 model weight markers: {unknown_weights}"
        )

    if present_peft:
        if names & _OTHER_PEFT_MARKERS:
            raise ValueError(
                "legacy binary or sharded Stage 1 adapter is unsupported; "
                "adapter_model.safetensors is required"
            )
        missing = _REQUIRED_PEFT_MARKERS - names
        if missing:
            labels = []
            if STAGE1_PEFT_METADATA_FILENAME in missing:
                labels.append("metadata")
            if SPARSE_TOKEN_ROWS_FILENAME in missing:
                labels.append("token-row artifact")
            if _ADAPTER_SAFE_FILENAME in missing:
                labels.append("adapter_model.safetensors")
            if _ADAPTER_CONFIG_FILENAME in missing:
                labels.append("adapter config")
            raise ValueError(
                "incomplete Stage 1 PEFT artifact: missing " + ", ".join(labels)
            )
        artifact = _canonical_artifact_dir(directory)
        if metadata_snapshot is None:
            metadata_snapshot = read_stage1_peft_metadata_snapshot(artifact)
        elif not isinstance(metadata_snapshot, Stage1PeftMetadataSnapshot):
            raise TypeError(
                "metadata_snapshot must be a Stage1PeftMetadataSnapshot"
            )
        if metadata_snapshot.artifact_dir != artifact:
            raise ValueError(
                "Stage 1 PEFT metadata snapshot artifact mismatch: "
                f"snapshot={metadata_snapshot.artifact_dir}, requested={artifact}"
            )
        validated_metadata = decode_stage1_peft_metadata_snapshot(
            metadata_snapshot
        )
        return Stage1ArtifactKind.PEFT, validated_metadata, metadata_snapshot

    if present_full:
        _validate_full_serialization(directory, names)
        if metadata_snapshot is not None:
            raise ValueError("full Stage 1 artifact cannot use PEFT metadata")
        return Stage1ArtifactKind.FULL, None, None
    raise ValueError(
        f"unrecognized local Stage 1 artifact (missing model weights): {directory}"
    )


def classify_stage1_artifact(path: Path) -> Stage1ArtifactKind:
    """Classify one local artifact without loading model weights."""

    return _classify_stage1_artifact_with_metadata(path)[0]


def _require_exact_string_list(
    value: object, *, name: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or any(not isinstance(item, str) or not item for item in value)
    ):
        qualifier = "a JSON string list" if allow_empty else "a non-empty JSON string list"
        raise ValueError(f"{name} must be {qualifier}")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique and ordered")
    return result


def _require_exact_int_list(
    value: object, *, name: str, length: int
) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(type(item) is not int for item in value)
    ):
        raise ValueError(f"{name} must be a length-{length} JSON integer list")
    return tuple(value)


def _require_finite_number(value: object, *, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class Stage1PeftMetadata:
    """Complete schema-v2 provenance for one Stage 1 adapter artifact."""

    schema_version: int
    stage: Literal["stage1"]
    base_model_reference: str
    base_model_architecture: dict[str, object]
    base_vocab_size: int
    pretrained_model_vocab_size: int
    resized_vocab_size: int
    latent_token_ids: tuple[int, int, int]
    latent_token_strings: tuple[str, str, str]
    latent_size: int
    ce_weight: float
    sim_weight: float
    coverage_p: float
    image_pool_k: int
    tuning_method: Literal["lora", "dora"]
    rank: int
    alpha: int
    dropout: float
    requested_target_suffixes: tuple[str, ...]
    resolved_target_names: tuple[str, ...]
    input_row_format: Literal["peft_trainable_tokens_v1"]
    output_row_format: Literal["ilvr_sparse_output_rows_v1"]
    output_row_shape: tuple[int, int]
    peft_version: str
    repository_format_version: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for name in (
            "latent_token_ids",
            "latent_token_strings",
            "requested_target_suffixes",
            "resolved_target_names",
            "output_row_shape",
        ):
            payload[name] = list(payload[name])
        type(self).from_dict(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Stage1PeftMetadata":
        if not isinstance(payload, Mapping):
            raise ValueError("Stage 1 PEFT metadata must be a JSON object")
        schema_version = payload.get("schema_version")
        if type(schema_version) is not int or schema_version not in (
            1,
            STAGE1_PEFT_SCHEMA_VERSION,
        ):
            raise ValueError(
                "unsupported Stage 1 PEFT metadata schema: "
                f"expected=1 or {STAGE1_PEFT_SCHEMA_VERSION}, "
                f"actual={schema_version}"
            )
        expected_fields = {field.name for field in fields(cls)}
        if schema_version == 1:
            expected_fields = expected_fields - {"pretrained_model_vocab_size"}
        actual_fields = set(payload)
        if actual_fields != expected_fields:
            raise ValueError(
                "invalid Stage 1 PEFT metadata fields: "
                f"missing={sorted(expected_fields - actual_fields)}, "
                f"unknown={sorted(actual_fields - expected_fields)}"
            )
        payload = dict(payload)
        if schema_version == 1:
            payload["schema_version"] = STAGE1_PEFT_SCHEMA_VERSION
            payload["pretrained_model_vocab_size"] = payload["base_vocab_size"]
        if payload["stage"] != "stage1":
            raise ValueError("Stage 1 PEFT metadata must describe stage1")
        if (
            not isinstance(payload["base_model_reference"], str)
            or not payload["base_model_reference"]
        ):
            raise ValueError("base_model_reference must be a non-empty string")

        integer_fields = (
            "base_vocab_size",
            "pretrained_model_vocab_size",
            "resized_vocab_size",
            "latent_size",
            "image_pool_k",
            "rank",
            "alpha",
            "repository_format_version",
        )
        if any(type(payload[name]) is not int for name in integer_fields):
            raise ValueError("Stage 1 PEFT integer metadata has an invalid type")
        if any(payload[name] <= 0 for name in integer_fields[:-1]):
            raise ValueError("Stage 1 PEFT size and tuning integers must be positive")
        if payload["repository_format_version"] != STAGE1_REPOSITORY_FORMAT_VERSION:
            raise ValueError(
                "unsupported Stage 1 repository format version: "
                f"{payload['repository_format_version']}"
            )

        architecture = normalize_qwen2_5_vl_architecture(
            payload["base_model_architecture"]
        )
        if (
            architecture["decoder"]["vocab_size"]
            != payload["pretrained_model_vocab_size"]
        ):
            raise ValueError(
                "base architecture vocab_size must equal "
                "pretrained_model_vocab_size"
            )
        if payload["resized_vocab_size"] != payload["base_vocab_size"] + 3:
            raise ValueError(
                "resized_vocab_size must add exactly three latent tokens"
            )
        if payload["pretrained_model_vocab_size"] < payload["base_vocab_size"]:
            raise ValueError(
                "pretrained_model_vocab_size must be at least base_vocab_size"
            )

        token_ids = _require_exact_int_list(
            payload["latent_token_ids"], name="latent_token_ids", length=3
        )
        expected_ids = tuple(
            range(payload["base_vocab_size"], payload["resized_vocab_size"])
        )
        if token_ids != expected_ids:
            raise ValueError(
                "latent token IDs must preserve appended order: "
                f"expected={expected_ids}, actual={token_ids}"
            )
        token_strings = _require_exact_string_list(
            payload["latent_token_strings"], name="latent_token_strings"
        )
        if token_strings != _CANONICAL_LATENT_TOKEN_STRINGS:
            raise ValueError(
                "latent token strings must preserve canonical pad/start/end order"
            )

        ce_weight = _require_finite_number(payload["ce_weight"], name="ce_weight")
        sim_weight = _require_finite_number(payload["sim_weight"], name="sim_weight")
        coverage_p = _require_finite_number(payload["coverage_p"], name="coverage_p")
        dropout = _require_finite_number(payload["dropout"], name="dropout")
        if ce_weight < 0 or sim_weight < 0:
            raise ValueError("Stage 1 loss weights must be non-negative")
        if not 0.0 < coverage_p <= 1.0:
            raise ValueError("coverage_p must be finite and in (0, 1]")

        requested = _require_exact_string_list(
            payload["requested_target_suffixes"],
            name="requested_target_suffixes",
        )
        resolved = _require_exact_string_list(
            payload["resolved_target_names"], name="resolved_target_names"
        )
        if payload["tuning_method"] not in ("lora", "dora"):
            raise ValueError("Stage 1 PEFT metadata requires LoRA or DoRA tuning")
        tuning = Stage1TuningConfig(
            method=payload["tuning_method"],
            rank=payload["rank"],
            alpha=payload["alpha"],
            dropout=dropout,
            target_suffixes=requested,
        )
        resolved_suffixes = tuple(name.rsplit(".", 1)[-1] for name in resolved)
        if set(resolved_suffixes) != set(requested):
            raise ValueError(
                "resolved target names must cover exactly the requested suffixes"
            )

        if payload["input_row_format"] != _INPUT_ROW_FORMAT:
            raise ValueError("unsupported input_row_format")
        if payload["output_row_format"] != _OUTPUT_ROW_FORMAT:
            raise ValueError("unsupported output_row_format")
        output_shape = _require_exact_int_list(
            payload["output_row_shape"], name="output_row_shape", length=2
        )
        if output_shape != (3, architecture["decoder"]["hidden_size"]):
            raise ValueError(
                "output_row_shape must equal (3, base hidden_size)"
            )
        if not isinstance(payload["peft_version"], str) or not payload["peft_version"]:
            raise ValueError("peft_version must be a non-empty string")

        return cls(
            schema_version=STAGE1_PEFT_SCHEMA_VERSION,
            stage="stage1",
            base_model_reference=payload["base_model_reference"],
            base_model_architecture=architecture,
            base_vocab_size=payload["base_vocab_size"],
            pretrained_model_vocab_size=payload["pretrained_model_vocab_size"],
            resized_vocab_size=payload["resized_vocab_size"],
            latent_token_ids=token_ids,
            latent_token_strings=token_strings,
            latent_size=payload["latent_size"],
            ce_weight=ce_weight,
            sim_weight=sim_weight,
            coverage_p=coverage_p,
            image_pool_k=payload["image_pool_k"],
            tuning_method=tuning.method,
            rank=tuning.rank,
            alpha=tuning.alpha,
            dropout=tuning.dropout,
            requested_target_suffixes=tuning.target_suffixes,
            resolved_target_names=resolved,
            input_row_format=_INPUT_ROW_FORMAT,
            output_row_format=_OUTPUT_ROW_FORMAT,
            output_row_shape=output_shape,
            peft_version=payload["peft_version"],
            repository_format_version=STAGE1_REPOSITORY_FORMAT_VERSION,
        )


@dataclass(frozen=True)
class Stage1PeftMetadataSnapshot:
    """One path-bound immutable read of the strict metadata sidecar."""

    artifact_dir: Path
    metadata_json: bytes


@dataclass(frozen=True)
class Stage1ArtifactInspection:
    """One classified artifact with any already-read immutable PEFT sidecar."""

    artifact_kind: Stage1ArtifactKind
    peft_metadata_snapshot: Stage1PeftMetadataSnapshot | None


def _inspect_stage1_artifact(path: Path) -> Stage1ArtifactInspection:
    kind, _, metadata_snapshot = _classify_stage1_artifact_with_metadata(path)
    return Stage1ArtifactInspection(kind, metadata_snapshot)


def _canonical_artifact_dir(directory: Path) -> Path:
    candidate = Path(directory)
    if candidate.is_symlink():
        raise ValueError(
            f"Stage 1 artifact directory must not be a symlink: {candidate}"
        )
    try:
        canonical = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"Stage 1 artifact does not exist: {candidate}") from error
    if not canonical.is_dir():
        raise ValueError(f"Stage 1 artifact is not a directory: {candidate}")
    return canonical


def _read_regular_payload(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"invalid {label} payload file: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid {label} payload file: {path}") from error


def read_stage1_peft_metadata_snapshot(
    directory: Path,
) -> Stage1PeftMetadataSnapshot:
    """Read the metadata payload once while binding its canonical directory."""

    artifact = _canonical_artifact_dir(directory)
    source = artifact / STAGE1_PEFT_METADATA_FILENAME
    if not source.is_file():
        raise ValueError(f"missing Stage 1 PEFT metadata: {source}")
    return Stage1PeftMetadataSnapshot(
        artifact_dir=artifact,
        metadata_json=_read_regular_payload(source, label="Stage 1 PEFT metadata"),
    )


def decode_stage1_peft_metadata_snapshot(
    snapshot: Stage1PeftMetadataSnapshot,
) -> Stage1PeftMetadata:
    """Reconstruct strict metadata only from immutable witnessed JSON bytes."""

    if not isinstance(snapshot, Stage1PeftMetadataSnapshot):
        raise TypeError("snapshot must be a Stage1PeftMetadataSnapshot")
    try:
        payload = json.loads(snapshot.metadata_json.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Stage 1 PEFT metadata bytes") from error
    return Stage1PeftMetadata.from_dict(payload)


def save_stage1_peft_metadata(
    directory: Path, metadata: Stage1PeftMetadata
) -> None:
    """Atomically publish one validated small Stage 1 sidecar."""

    destination_dir = Path(directory)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / STAGE1_PEFT_METADATA_FILENAME
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata.to_dict(), sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def load_stage1_peft_metadata(directory: Path) -> Stage1PeftMetadata:
    """Read one strict Stage 1 sidecar for deferred validation/migration."""

    return decode_stage1_peft_metadata_snapshot(
        read_stage1_peft_metadata_snapshot(directory)
    )


def _peft_model(model: nn.Module) -> PeftModel:
    candidate = getattr(model, "module", model)
    if not isinstance(candidate, PeftModel):
        raise ValueError("Stage 1 PEFT artifact requires a PeftModel")
    if set(candidate.peft_config) != {_ADAPTER_NAME}:
        raise ValueError("Stage 1 PEFT artifact requires one default adapter")
    return candidate


def _base_embedding(qwen: nn.Module) -> nn.Embedding:
    embedding = qwen.model.embed_tokens
    token_adapter = getattr(embedding, "token_adapter", None)
    base = getattr(token_adapter, "base_layer", embedding)
    if not isinstance(base, nn.Embedding):
        raise TypeError("Stage 1 input embedding is not torch.nn.Embedding")
    return base


def _base_head(qwen: nn.Module) -> nn.Linear:
    head = qwen.lm_head
    base = getattr(head, "base_head", head)
    if not isinstance(base, nn.Linear):
        raise TypeError("Stage 1 output head is not torch.nn.Linear")
    return base


def _validate_processor(processor, metadata: Stage1PeftMetadata):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Stage 1 processor must expose its tokenizer")
    try:
        tokenizer_size = len(tokenizer)
    except (TypeError, AttributeError) as error:
        raise ValueError("Stage 1 tokenizer must expose its vocabulary size") from error
    if tokenizer_size != metadata.resized_vocab_size:
        raise ValueError(
            "resized vocabulary mismatch: "
            f"metadata={metadata.resized_vocab_size}, tokenizer={tokenizer_size}"
        )

    unknown_id = getattr(tokenizer, "unk_token_id", None)
    unknown_token = getattr(tokenizer, "unk_token", None)
    for token, expected_id in zip(
        metadata.latent_token_strings, metadata.latent_token_ids
    ):
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id is None or (
            unknown_id is not None
            and actual_id == unknown_id
            and token != unknown_token
        ):
            raise ValueError(f"tokenizer does not contain latent token {token}")
        if type(actual_id) is bool or int(actual_id) != expected_id:
            raise ValueError(
                "latent token ID mismatch: "
                f"token={token}, metadata={expected_id}, tokenizer={actual_id}"
            )
        reverse = tokenizer.convert_ids_to_tokens(expected_id)
        if reverse != token:
            raise ValueError(
                "latent token round-trip mismatch: "
                f"id={expected_id}, metadata={token}, tokenizer={reverse}"
            )
    return tokenizer


def _validate_latent_config(config, metadata: Stage1PeftMetadata) -> None:
    for expected_id, names in zip(
        metadata.latent_token_ids,
        (
            ("latent_token_id", "latent_pad_token_id"),
            ("latent_start_id", "latent_start_token_id"),
            ("latent_end_id", "latent_end_token_id"),
        ),
    ):
        for name in names:
            actual = getattr(config, name, None)
            if actual is not None and int(actual) != expected_id:
                raise ValueError(
                    f"{name} mismatch: metadata={expected_id}, config={actual}"
                )
    latent_size = getattr(config, "latent_size", None)
    if latent_size is not None and int(latent_size) != metadata.latent_size:
        raise ValueError(
            "latent_size mismatch: "
            f"metadata={metadata.latent_size}, config={latent_size}"
        )


def _configure_latent_config(config, metadata: Stage1PeftMetadata) -> None:
    pad_id, start_id, end_id = metadata.latent_token_ids
    config.stage = "stage1"
    config.latent_size = metadata.latent_size
    config.latent_token_id = config.latent_pad_token_id = pad_id
    config.latent_start_id = config.latent_start_token_id = start_id
    config.latent_end_id = config.latent_end_token_id = end_id


def validate_stage1_vocab_alignment(
    model: nn.Module, expected_vocab_size: int
) -> None:
    """Require config, input embedding, and output head to share one vocab."""

    if type(expected_vocab_size) is not int or expected_vocab_size <= 0:
        raise ValueError("expected_vocab_size must be a positive integer")
    qwen = resolve_qwen2_5_vl_model(model)
    config_vocab = capture_qwen2_5_vl_architecture(qwen.config)["decoder"][
        "vocab_size"
    ]
    embedding = _base_embedding(qwen)
    head = _base_head(qwen)
    actual = (config_vocab, embedding.num_embeddings, head.out_features)
    expected = (expected_vocab_size,) * 3
    if actual != expected:
        raise ValueError(
            "Stage 1 post-resize vocabulary mismatch: "
            f"expected={expected_vocab_size}, config={config_vocab}, "
            f"input={embedding.num_embeddings}, output={head.out_features}"
        )


def _validate_architecture(
    qwen: nn.Module,
    metadata: Stage1PeftMetadata,
    *,
    resized: bool,
    sparse_runtime: bool = False,
) -> None:
    current = capture_qwen2_5_vl_architecture(qwen.config)
    expected_vocab = (
        metadata.resized_vocab_size
        if resized
        else metadata.pretrained_model_vocab_size
    )
    if current["decoder"]["vocab_size"] != expected_vocab:
        raise ValueError(
            "base vocabulary mismatch: "
            f"metadata={expected_vocab}, model={current['decoder']['vocab_size']}"
        )
    expected_architecture = deepcopy(metadata.base_model_architecture)
    expected_architecture["decoder"]["vocab_size"] = expected_vocab
    if sparse_runtime:
        if not resized:
            raise ValueError("sparse runtime architecture must be resized")
        expected_architecture["decoder"]["tie_word_embeddings"] = False
    difference = first_architecture_difference(
        expected_architecture, current
    )
    if difference is not None:
        path, expected, actual = difference
        raise ValueError(
            f"base architecture mismatch at {path}: "
            f"metadata={expected!r}, model={actual!r}"
        )

    validate_stage1_vocab_alignment(qwen, expected_vocab)


def _adapter_value(config, name: str):
    value = getattr(config, name, None)
    return getattr(value, "value", value)


def _decode_adapter_config(payload: bytes) -> PeftConfig:
    if type(payload) is not bytes:
        raise TypeError("Stage 1 adapter config payload must be bytes")
    try:
        values = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Stage 1 adapter config bytes") from error
    if not isinstance(values, Mapping):
        raise ValueError("Stage 1 adapter config must be a JSON object")
    values = dict(values)
    values.pop("runtime_config", None)
    try:
        checked = PeftConfig.check_kwargs(**values)
        return PeftConfig.from_peft_type(**checked)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid Stage 1 adapter config values") from error


def _validate_adapter_config(
    qwen: nn.Module,
    config,
    metadata: Stage1PeftMetadata,
) -> None:
    if _adapter_value(config, "peft_type") != "LORA":
        raise ValueError("Stage 1 adapter requires peft_type=LORA")
    if _adapter_value(config, "task_type") != "CAUSAL_LM":
        raise ValueError("Stage 1 adapter requires task_type=CAUSAL_LM")
    if bool(getattr(config, "use_dora", False)) is not (
        metadata.tuning_method == "dora"
    ):
        raise ValueError("Stage 1 adapter tuning method mismatch")
    if getattr(config, "use_rslora", None) is not False:
        raise ValueError("Stage 1 adapter requires use_rslora=False")
    if getattr(config, "rank_pattern", None):
        raise ValueError("Stage 1 adapter does not support rank_pattern")
    if getattr(config, "alpha_pattern", None):
        raise ValueError("Stage 1 adapter does not support alpha_pattern")
    if getattr(config, "bias", None) != "none":
        raise ValueError("Stage 1 adapter requires bias=none")
    if getattr(config, "modules_to_save", None) is not None:
        raise ValueError("Stage 1 adapter must not contain modules_to_save")
    _validate_target_selector_modifiers_absent(config)
    for label, attribute, expected in (
        ("rank", "r", metadata.rank),
        ("alpha", "lora_alpha", metadata.alpha),
        ("dropout", "lora_dropout", metadata.dropout),
    ):
        actual = getattr(config, attribute, None)
        if actual != expected:
            raise ValueError(
                f"Stage 1 adapter {label} mismatch: "
                f"metadata={expected}, adapter={actual}"
            )

    configured_rows = getattr(config, "trainable_token_indices", None)
    expected_rows = {_EMBEDDING_TARGET: list(metadata.latent_token_ids)}
    if configured_rows != expected_rows:
        raise ValueError(
            "Stage 1 adapter selective input-row mismatch: "
            f"metadata={expected_rows}, adapter={configured_rows}"
        )
    target_modules = getattr(config, "target_modules", None)
    if not _configured_target_patterns_match(
        qwen, target_modules, metadata.resolved_target_names
    ):
        raise ValueError("Stage 1 adapter resolved target mismatch")
    actual_resolved = _resolve_linear_decoder_targets(
        qwen, metadata.requested_target_suffixes
    )
    if actual_resolved != metadata.resolved_target_names:
        raise ValueError(
            "Stage 1 resolved target topology mismatch: "
            f"metadata={metadata.resolved_target_names}, model={actual_resolved}"
        )
    source = getattr(config, "base_model_name_or_path", "")
    if source and source != metadata.base_model_reference:
        raise ValueError(
            "Stage 1 adapter base reference mismatch: "
            f"metadata={metadata.base_model_reference}, adapter={source}"
        )


def _linear_base(module: nn.Module, target: str) -> nn.Linear:
    candidate = getattr(module, "base_layer", module)
    if not isinstance(candidate, nn.Linear):
        raise ValueError(f"Stage 1 resolved target is not linear: {target}")
    return candidate


def _validate_adapter_tensors(
    qwen: nn.Module,
    payload: bytes,
    metadata: Stage1PeftMetadata,
) -> None:
    modules = dict(qwen.named_modules())
    matched: set[str] = set()
    try:
        tensors = load_safetensors(payload)
    except SafetensorError as error:
        raise ValueError("invalid Stage 1 adapter safetensors bytes") from error
    keys = tuple(tensors)

    input_matches = [
        key
        for key in keys
        if key.endswith(
            ".model.embed_tokens.token_adapter.trainable_tokens_delta"
        )
    ]
    if len(input_matches) != 1:
        raise ValueError(
            "Stage 1 adapter must contain exactly one selective input-row tensor"
        )
    input_key = input_matches[0]
    matched.add(input_key)
    input_shape = list(tensors[input_key].shape)
    if input_shape != list(metadata.output_row_shape):
        raise ValueError(
            "Stage 1 selective input-row shape mismatch: "
            f"adapter={input_shape}, expected={list(metadata.output_row_shape)}"
        )

    for target in metadata.resolved_target_names:
        if target not in modules:
            raise ValueError(f"missing Stage 1 resolved target: {target}")
        module = _linear_base(modules[target], target)
        expected_shapes = {
            "lora_A.weight": [metadata.rank, module.in_features],
            "lora_B.weight": [module.out_features, metadata.rank],
        }
        if metadata.tuning_method == "dora":
            expected_shapes["lora_magnitude_vector"] = [module.out_features]
        for component, expected_shape in expected_shapes.items():
            suffix = f".{target}.{component}"
            component_matches = [key for key in keys if key.endswith(suffix)]
            if len(component_matches) != 1:
                raise ValueError(
                    f"Stage 1 adapter tensor mismatch at {target}: {component}"
                )
            key = component_matches[0]
            matched.add(key)
            actual_shape = list(tensors[key].shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Stage 1 adapter target shape mismatch at {target}: "
                    f"adapter={actual_shape}, expected={expected_shape}"
                )
    unexpected = sorted(set(keys) - matched)
    if unexpected:
        raise ValueError(f"unexpected adapter tensor fields: {unexpected}")


def _validate_sparse_artifact(
    payload: bytes, metadata: Stage1PeftMetadata
) -> None:
    token_ids, output_delta = decode_sparse_token_rows_state(payload)
    if token_ids.dtype != torch.int64 or token_ids.shape != torch.Size((3,)):
        raise ValueError("sparse token_ids must be an int64 tensor with shape (3,)")
    if tuple(token_ids.tolist()) != metadata.latent_token_ids:
        raise ValueError("sparse latent token IDs mismatch")
    if output_delta.dtype != torch.float32:
        raise ValueError("sparse output_delta must use float32 serialization")
    if tuple(output_delta.shape) != metadata.output_row_shape:
        raise ValueError(
            "sparse output_delta shape mismatch: "
            f"artifact={tuple(output_delta.shape)}, "
            f"metadata={metadata.output_row_shape}"
        )
    if not bool(torch.isfinite(output_delta).all().item()):
        raise ValueError("sparse output_delta must contain only finite values")


def _metadata_tuning(metadata: Stage1PeftMetadata) -> Stage1TuningConfig:
    return Stage1TuningConfig(
        method=metadata.tuning_method,
        rank=metadata.rank,
        alpha=metadata.alpha,
        dropout=metadata.dropout,
        target_suffixes=metadata.requested_target_suffixes,
    )


def _validate_stage1_peft_runtime(
    model: nn.Module,
    processor,
    metadata: Stage1PeftMetadata,
) -> PeftModel:
    validated = Stage1PeftMetadata.from_dict(metadata.to_dict())
    if validated.peft_version != peft.__version__:
        raise ValueError(
            "Stage 1 PEFT version mismatch: "
            f"metadata={validated.peft_version}, installed={peft.__version__}"
        )
    candidate = _peft_model(model)
    qwen = resolve_qwen2_5_vl_model(candidate)
    _validate_processor(processor, validated)
    _validate_latent_config(qwen.config, validated)
    _validate_architecture(
        qwen, validated, resized=True, sparse_runtime=True
    )
    _validate_adapter_config(
        qwen, candidate.peft_config[_ADAPTER_NAME], validated
    )
    rows = find_sparse_output_token_rows(candidate)
    if tuple(rows.token_ids.detach().cpu().tolist()) != validated.latent_token_ids:
        raise ValueError("sparse output-row latent token ID mismatch")
    if tuple(_parameter_shape(rows.output_delta)) != validated.output_row_shape:
        raise ValueError("sparse output-row shape mismatch")
    validate_stage1_trainable_boundary(
        candidate, _metadata_tuning(validated), validated.latent_token_ids
    )
    return candidate


def _save_stage1_peft_supplements(
    model: nn.Module,
    output_dir: Path,
    metadata: Stage1PeftMetadata,
) -> None:
    save_sparse_token_rows(model, Path(output_dir))
    save_stage1_peft_metadata(Path(output_dir), metadata)


def save_stage1_peft_artifact(
    model: nn.Module,
    processor,
    output_dir: Path,
    metadata: Stage1PeftMetadata,
) -> None:
    """Save one complete adapter-only Stage 1 artifact."""

    candidate = _validate_stage1_peft_runtime(model, processor, metadata)
    destination = Path(output_dir)
    if destination.exists():
        names = {entry.name for entry in destination.iterdir() if entry.is_file()}
        full_markers = sorted(name for name in names if _is_full_weight_filename(name))
        if full_markers:
            raise ValueError(
                f"refusing to mix Stage 1 PEFT with full weights: {full_markers}"
            )
    destination.mkdir(parents=True, exist_ok=True)
    candidate.save_pretrained(destination, safe_serialization=True)
    _save_stage1_peft_supplements(candidate, destination, metadata)
    processor.save_pretrained(destination)
    classify_stage1_artifact(destination)


@dataclass(frozen=True)
class Stage1PeftPreflightWitness:
    """Immutable path-bound payloads and scalar provenance for restoration."""

    artifact_dir: Path
    metadata_json: bytes
    adapter_config_json: bytes
    adapter_safetensors: bytes
    sparse_safetensors: bytes
    base_model_reference: str
    base_architecture_json: bytes
    base_vocab_size: int
    pretrained_model_vocab_size: int
    resized_vocab_size: int
    processor_vocab_size: int
    processor_contract: tuple[tuple[str, int], ...]
    latent_size: int
    tuning_method: str
    rank: int
    alpha: int
    dropout: float


def _build_meta_stage1_base(config) -> nn.Module:
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration(config)
    if config.tie_word_embeddings is True:
        model.tie_weights()
    return model


def _architecture_json(architecture: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            architecture,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Stage 1 base architecture is not canonical JSON"
        ) from error


def _validate_base_source(requested: str, expected: str) -> None:
    if not isinstance(requested, str) or not requested:
        raise ValueError(
            "Stage 1 PEFT requested base source must be a non-empty string"
        )
    if requested != expected:
        raise ValueError(
            "Stage 1 PEFT base source mismatch: "
            f"metadata={expected!r}, requested={requested!r}"
        )


def _decode_preflight_witness(
    witness: Stage1PeftPreflightWitness,
) -> tuple[Stage1PeftMetadata, PeftConfig]:
    metadata = decode_stage1_peft_metadata_snapshot(
        Stage1PeftMetadataSnapshot(
            artifact_dir=witness.artifact_dir,
            metadata_json=witness.metadata_json,
        )
    )
    scalar_provenance = (
        metadata.base_model_reference,
        _architecture_json(metadata.base_model_architecture),
        metadata.base_vocab_size,
        metadata.pretrained_model_vocab_size,
        metadata.resized_vocab_size,
        metadata.resized_vocab_size,
        tuple(zip(metadata.latent_token_strings, metadata.latent_token_ids)),
        metadata.latent_size,
        metadata.tuning_method,
        metadata.rank,
        metadata.alpha,
        metadata.dropout,
    )
    witnessed_provenance = (
        witness.base_model_reference,
        witness.base_architecture_json,
        witness.base_vocab_size,
        witness.pretrained_model_vocab_size,
        witness.resized_vocab_size,
        witness.processor_vocab_size,
        witness.processor_contract,
        witness.latent_size,
        witness.tuning_method,
        witness.rank,
        witness.alpha,
        witness.dropout,
    )
    if scalar_provenance != witnessed_provenance:
        raise ValueError("Stage 1 PEFT witness scalar provenance mismatch")
    return metadata, _decode_adapter_config(witness.adapter_config_json)


def preflight_stage1_peft_artifact(
    processor,
    base_config,
    artifact_dir: Path,
    *,
    requested_base_model_reference: str,
    metadata_snapshot: Stage1PeftMetadataSnapshot | None = None,
    model_factory: Callable[[object], nn.Module] = _build_meta_stage1_base,
) -> Stage1PeftPreflightWitness:
    """Validate a PEFT artifact completely without loading base weights."""

    artifact = _canonical_artifact_dir(artifact_dir)
    if metadata_snapshot is None:
        metadata_snapshot = read_stage1_peft_metadata_snapshot(artifact)
    elif not isinstance(metadata_snapshot, Stage1PeftMetadataSnapshot):
        raise TypeError("metadata_snapshot must be a Stage1PeftMetadataSnapshot")
    if metadata_snapshot.artifact_dir != artifact:
        raise ValueError(
            "Stage 1 PEFT metadata snapshot artifact mismatch: "
            f"snapshot={metadata_snapshot.artifact_dir}, requested={artifact}"
        )
    kind, validated, classified_snapshot = _classify_stage1_artifact_with_metadata(
        artifact, metadata_snapshot=metadata_snapshot
    )
    if kind is not Stage1ArtifactKind.PEFT or validated is None:
        raise ValueError("Stage 1 PEFT preflight requires a PEFT artifact")
    if classified_snapshot is not metadata_snapshot:
        raise ValueError("Stage 1 PEFT preflight did not retain metadata snapshot")
    if validated.peft_version != peft.__version__:
        raise ValueError(
            "Stage 1 PEFT version mismatch: "
            f"metadata={validated.peft_version}, installed={peft.__version__}"
        )
    tokenizer = _validate_processor(processor, validated)
    _validate_latent_config(base_config, validated)
    _validate_base_source(
        requested_base_model_reference, validated.base_model_reference
    )
    structural_model = model_factory(deepcopy(base_config))
    qwen = resolve_qwen2_5_vl_model(structural_model)
    _validate_architecture(qwen, validated, resized=False)
    structural_model.resize_token_embeddings(
        validated.resized_vocab_size, mean_resizing=False
    )
    _validate_architecture(qwen, validated, resized=True)
    initialize_frozen_output_token_rows(
        structural_model, validated.latent_token_ids
    )
    _validate_architecture(
        qwen, validated, resized=True, sparse_runtime=True
    )
    adapter_config_json = _read_regular_payload(
        artifact / _ADAPTER_CONFIG_FILENAME,
        label="Stage 1 adapter config",
    )
    adapter_safetensors = _read_regular_payload(
        artifact / _ADAPTER_SAFE_FILENAME,
        label="Stage 1 adapter safetensors",
    )
    sparse_safetensors = _read_regular_payload(
        artifact / SPARSE_TOKEN_ROWS_FILENAME,
        label="Stage 1 sparse token rows",
    )
    adapter_config = _decode_adapter_config(adapter_config_json)
    _validate_adapter_config(qwen, adapter_config, validated)
    _validate_adapter_tensors(qwen, adapter_safetensors, validated)
    _validate_sparse_artifact(sparse_safetensors, validated)
    return Stage1PeftPreflightWitness(
        artifact_dir=artifact,
        metadata_json=metadata_snapshot.metadata_json,
        adapter_config_json=adapter_config_json,
        adapter_safetensors=adapter_safetensors,
        sparse_safetensors=sparse_safetensors,
        base_model_reference=validated.base_model_reference,
        base_architecture_json=_architecture_json(
            validated.base_model_architecture
        ),
        base_vocab_size=validated.base_vocab_size,
        pretrained_model_vocab_size=validated.pretrained_model_vocab_size,
        resized_vocab_size=validated.resized_vocab_size,
        processor_vocab_size=len(tokenizer),
        processor_contract=tuple(
            zip(validated.latent_token_strings, validated.latent_token_ids)
        ),
        latent_size=validated.latent_size,
        tuning_method=validated.tuning_method,
        rank=validated.rank,
        alpha=validated.alpha,
        dropout=validated.dropout,
    )


def load_stage1_peft_artifact(
    base_model: nn.Module,
    processor,
    artifact_dir: Path,
    *,
    requested_base_model_reference: str,
    is_trainable: bool,
    preflight: Stage1PeftPreflightWitness | None = None,
) -> nn.Module:
    """Validate, resize, and restore one Stage 1 PEFT artifact in strict order."""

    if type(is_trainable) is not bool:
        raise TypeError("is_trainable must be boolean")
    artifact = Path(artifact_dir)
    candidate = getattr(base_model, "module", base_model)
    if isinstance(candidate, PeftModel):
        raise ValueError("Stage 1 PEFT loader requires an unmodified full base")
    qwen = resolve_qwen2_5_vl_model(candidate)
    if preflight is None:
        preflight = preflight_stage1_peft_artifact(
            processor,
            qwen.config,
            artifact,
            requested_base_model_reference=requested_base_model_reference,
        )
    elif not isinstance(preflight, Stage1PeftPreflightWitness):
        raise TypeError("preflight must be a Stage1PeftPreflightWitness")
    requested_artifact = artifact.resolve()
    if preflight.artifact_dir != requested_artifact:
        raise ValueError(
            "Stage 1 PEFT preflight artifact mismatch: "
            f"witness={preflight.artifact_dir}, requested={requested_artifact}"
        )
    metadata, adapter_config = _decode_preflight_witness(preflight)

    # The small artifact was already fully validated. Recheck only the actual
    # runtime objects constructed after preflight, before mutating the base.
    _validate_processor(processor, metadata)
    _validate_latent_config(qwen.config, metadata)
    _validate_architecture(qwen, metadata, resized=False)
    _validate_base_source(
        requested_base_model_reference, preflight.base_model_reference
    )
    actual_architecture_json = _architecture_json(
        capture_qwen2_5_vl_architecture(qwen.config)
    )
    if actual_architecture_json != preflight.base_architecture_json:
        raise ValueError("Stage 1 PEFT runtime architecture witness mismatch")
    _validate_adapter_config(qwen, adapter_config, metadata)

    candidate.resize_token_embeddings(metadata.resized_vocab_size)
    _validate_architecture(qwen, metadata, resized=True)
    initialize_frozen_output_token_rows(candidate, metadata.latent_token_ids)
    _validate_architecture(
        qwen, metadata, resized=True, sparse_runtime=True
    )
    _configure_latent_config(candidate.config, metadata)
    adapter_config.inference_mode = not is_trainable
    restored = get_peft_model(
        candidate,
        adapter_config,
        adapter_name=_ADAPTER_NAME,
    )
    try:
        adapter_state = load_safetensors(preflight.adapter_safetensors)
    except SafetensorError as error:
        raise ValueError("invalid witnessed Stage 1 adapter bytes") from error
    load_result = set_peft_model_state_dict(
        restored,
        adapter_state,
        adapter_name=_ADAPTER_NAME,
    )
    if load_result.unexpected_keys:
        raise ValueError(
            "unexpected Stage 1 adapter state keys after application: "
            f"{sorted(load_result.unexpected_keys)}"
        )
    if not is_trainable:
        # PEFT 0.17.1 leaves TrainableTokensModel's selective delta enabled
        # even when the adapter itself is loaded for inference.
        restored.requires_grad_(False)
    # PEFT restoration never creates the repository-owned output-row wrapper.
    rows = attach_sparse_output_token_rows(restored, metadata.latent_token_ids)
    rows.output_delta.requires_grad_(is_trainable)
    apply_sparse_token_rows_state(
        restored,
        preflight.sparse_safetensors,
        metadata.latent_token_ids,
    )
    if is_trainable:
        validate_stage1_trainable_boundary(
            restored, _metadata_tuning(metadata), metadata.latent_token_ids
        )
    elif any(parameter.requires_grad for parameter in restored.parameters()):
        raise ValueError("inference Stage 1 adapter contains trainable parameters")
    return restored


@dataclass(frozen=True)
class Stage1ResumeDecision:
    """A numbered Trainer checkpoint selected for same-stage resume."""

    checkpoint_path: Path | None
    artifact_kind: Stage1ArtifactKind | None
    explicit: bool
    peft_metadata_snapshot: Stage1PeftMetadataSnapshot | None = None


def _is_regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _validated_rng_world_size(path: Path) -> int | None:
    single = path / "rng_state.pth"
    rank_files = {
        int(match.group(1)): entry
        for entry in path.iterdir()
        if entry.is_file()
        and (match := _RANK_RNG_STATE.fullmatch(entry.name))
    }
    if _is_regular_file(single):
        return 1 if not rank_files else None
    if single.exists() or not rank_files:
        return None
    if sorted(rank_files) != list(range(len(rank_files))) or not all(
        _is_regular_file(rank_file) for rank_file in rank_files.values()
    ):
        return None
    return len(rank_files)


def _has_complete_standard_trainer_state(path: Path) -> bool:
    return all(
        _is_regular_file(path / filename)
        for filename in ("optimizer.pt", "scheduler.pt")
    )


def _load_trainer_global_step(path: Path) -> int | None:
    trainer_state = path / "trainer_state.json"
    try:
        if trainer_state.stat().st_size > _MAX_TRAINER_STATE_BYTES:
            return None
        payload = json.loads(trainer_state.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    global_step = payload.get("global_step")
    if type(global_step) is not int or global_step < 0:
        return None
    return global_step


def _canonical_mp_rank(value: str) -> int | None:
    rank = int(value)
    return rank if value == f"{rank:02d}" else None


def _has_complete_deepspeed_trainer_state(
    path: Path, *, rng_world_size: int
) -> bool:
    latest = path / "latest"
    if not _is_regular_file(latest):
        return False
    try:
        if latest.stat().st_size > _MAX_DEEPSPEED_LATEST_BYTES:
            return False
        tag_name = latest.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    tag_match = _DEEPSPEED_TAG.fullmatch(tag_name)
    if tag_match is None:
        return False
    tag_step = int(tag_match.group(1))
    if _load_trainer_global_step(path) != tag_step:
        return False
    tag = path / tag_name
    if tag.is_symlink() or not tag.is_dir() or tag.resolve().parent != path.resolve():
        return False

    zero3_model_pairs: list[tuple[int, int]] = []
    replicated_model_ranks: list[int] = []
    optimizer_pairs: list[tuple[int, int]] = []
    optimizer_bf16: set[bool] = set()
    for entry in tag.iterdir():
        if entry.is_symlink():
            return False
        zero3_match = _DEEPSPEED_ZERO3_MODEL_STATE.fullmatch(entry.name)
        replicated_match = _DEEPSPEED_REPLICATED_MODEL_STATE.fullmatch(
            entry.name
        )
        optimizer_match = _DEEPSPEED_OPTIMIZER_STATE.fullmatch(entry.name)
        if zero3_match is not None and entry.is_file():
            mp_rank = _canonical_mp_rank(zero3_match.group(2))
            if mp_rank is None:
                return False
            zero3_model_pairs.append((int(zero3_match.group(1)), mp_rank))
        elif replicated_match is not None and entry.is_file():
            mp_rank = _canonical_mp_rank(replicated_match.group(1))
            if mp_rank is None:
                return False
            replicated_model_ranks.append(mp_rank)
        elif optimizer_match is not None and entry.is_file():
            mp_rank = _canonical_mp_rank(optimizer_match.group(3))
            if mp_rank is None:
                return False
            optimizer_pairs.append((int(optimizer_match.group(2)), mp_rank))
            optimizer_bf16.add(optimizer_match.group(1) is not None)
        elif entry.name.endswith("_states.pt"):
            return False
    if (
        not optimizer_pairs
        or len(set(optimizer_pairs)) != len(optimizer_pairs)
        or len(optimizer_bf16) != 1
    ):
        return False
    mp_ranks = sorted({mp_rank for _, mp_rank in optimizer_pairs})
    if mp_ranks != list(range(len(mp_ranks))):
        return False
    dp_ranks_by_mp = {
        mp_rank: sorted(
            dp_rank
            for dp_rank, candidate_mp_rank in optimizer_pairs
            if candidate_mp_rank == mp_rank
        )
        for mp_rank in mp_ranks
    }
    expected_dp_ranks = list(range(len(dp_ranks_by_mp[0])))
    if not expected_dp_ranks or any(
        dp_ranks != expected_dp_ranks
        for dp_ranks in dp_ranks_by_mp.values()
    ):
        return False
    expected_pairs = {
        (dp_rank, mp_rank)
        for mp_rank in mp_ranks
        for dp_rank in expected_dp_ranks
    }
    if set(optimizer_pairs) != expected_pairs:
        return False
    if len(expected_pairs) != rng_world_size:
        return False

    has_zero3 = bool(zero3_model_pairs)
    has_replicated = bool(replicated_model_ranks)
    if has_zero3 == has_replicated:
        return False
    if has_zero3:
        return (
            len(set(zero3_model_pairs)) == len(zero3_model_pairs)
            and set(zero3_model_pairs) == expected_pairs
        )
    return (
        len(set(replicated_model_ranks)) == len(replicated_model_ranks)
        and set(replicated_model_ranks) == set(mp_ranks)
    )


def _validate_complete_trainer_state(path: Path) -> None:
    if not _is_regular_file(path / "trainer_state.json"):
        raise ValueError(
            f"Stage 1 resume requires complete trainer_state.json: {path}"
        )
    rng_world_size = _validated_rng_world_size(path)
    if rng_world_size is None:
        raise ValueError(f"Stage 1 resume has incomplete RNG state: {path}")

    optimizer = path / "optimizer.pt"
    scheduler = path / "scheduler.pt"
    optimizer_marker = optimizer.exists() or optimizer.is_symlink()
    scheduler_marker = scheduler.exists() or scheduler.is_symlink()
    deepspeed_markers = (path / "latest").exists() or (path / "latest").is_symlink()
    deepspeed_markers = deepspeed_markers or any(
        _DEEPSPEED_TAG.fullmatch(entry.name)
        and (entry.is_dir() or entry.is_symlink())
        for entry in path.iterdir()
    )
    if deepspeed_markers:
        if optimizer_marker:
            raise ValueError(
                f"Stage 1 resume has mixed Trainer optimizer layouts: {path}"
            )
        if scheduler_marker and not _is_regular_file(scheduler):
            raise ValueError(
                f"Stage 1 resume has unsafe external scheduler state: {scheduler}"
            )
        if _has_complete_deepspeed_trainer_state(
            path, rng_world_size=rng_world_size
        ):
            return
        raise ValueError(
            f"Stage 1 resume requires complete DeepSpeed state: {path}"
        )
    if optimizer_marker or scheduler_marker:
        if not _has_complete_standard_trainer_state(path):
            raise ValueError(
                f"Stage 1 resume has incomplete optimizer/scheduler state: {path}"
            )
        return
    raise ValueError(
        f"Stage 1 resume requires complete standard or DeepSpeed state: {path}"
    )


def _validate_peft_resume_location(output_dir: Path, checkpoint: Path) -> None:
    if checkpoint.is_symlink():
        raise ValueError(
            f"Stage 1 adapter resume rejects symlink checkpoints: {checkpoint}"
        )
    output = output_dir.resolve()
    if checkpoint.parent.resolve() != output or checkpoint.resolve().parent != output:
        raise ValueError(
            "Stage 1 adapter resume requires an immediate checkpoint in the "
            f"same output directory: output={output_dir}, checkpoint={checkpoint}"
        )


def _validated_resume_checkpoint(
    path: Path, *, output_dir: Path
) -> Stage1ArtifactInspection:
    inspection = _inspect_stage1_artifact(path)
    if inspection.artifact_kind is Stage1ArtifactKind.PEFT:
        if _NUMBERED_CHECKPOINT.fullmatch(path.name) is None:
            raise ValueError(
                f"Stage 1 adapter resume requires a numbered checkpoint: {path}"
            )
        _validate_peft_resume_location(output_dir, path)
    _validate_complete_trainer_state(path)
    return inspection


def resolve_stage1_resume(
    output_dir: Path,
    explicit_resume: str | Path | None,
) -> Stage1ResumeDecision:
    """Resolve explicit or latest valid immediate numbered Stage 1 checkpoint."""

    output = Path(output_dir)
    if explicit_resume is not None:
        checkpoint = Path(explicit_resume)
        inspection = _validated_resume_checkpoint(checkpoint, output_dir=output)
        return Stage1ResumeDecision(
            checkpoint,
            inspection.artifact_kind,
            True,
            inspection.peft_metadata_snapshot,
        )

    if not output.is_dir():
        return Stage1ResumeDecision(None, None, False)
    candidates: list[tuple[int, Path, Stage1ArtifactInspection]] = []
    for child in output.iterdir():
        if child.is_symlink() or not child.is_dir():
            continue
        match = _NUMBERED_CHECKPOINT.fullmatch(child.name)
        if match is None:
            continue
        try:
            inspection = _validated_resume_checkpoint(child, output_dir=output)
        except (OSError, ValueError):
            continue
        candidates.append((int(match.group(1)), child, inspection))
    if not candidates:
        return Stage1ResumeDecision(None, None, False)
    _, checkpoint, inspection = max(candidates, key=lambda item: item[0])
    return Stage1ResumeDecision(
        checkpoint,
        inspection.artifact_kind,
        False,
        inspection.peft_metadata_snapshot,
    )


__all__ = [
    "STAGE1_PEFT_METADATA_FILENAME",
    "STAGE1_PEFT_SCHEMA_VERSION",
    "STAGE1_REPOSITORY_FORMAT_VERSION",
    "Stage1ArtifactKind",
    "Stage1PeftMetadata",
    "Stage1PeftMetadataSnapshot",
    "Stage1PeftPreflightWitness",
    "Stage1ResumeDecision",
    "classify_stage1_artifact",
    "decode_stage1_peft_metadata_snapshot",
    "load_stage1_peft_artifact",
    "load_stage1_peft_metadata",
    "preflight_stage1_peft_artifact",
    "read_stage1_peft_metadata_snapshot",
    "resolve_stage1_resume",
    "save_stage1_peft_artifact",
    "save_stage1_peft_metadata",
    "validate_stage1_vocab_alignment",
]
