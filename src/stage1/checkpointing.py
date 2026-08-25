"""Strict Stage 1 full-model and adapter artifact lifecycles."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Literal

import peft
import torch
from peft import PeftConfig, PeftModel
from safetensors import SafetensorError, safe_open
from torch import nn

from peft_common import (
    capture_qwen2_5_vl_architecture,
    first_architecture_difference,
    normalize_qwen2_5_vl_architecture,
    resolve_qwen2_5_vl_model,
)
from stage1.ema import _parameter_shape
from stage1.peft import (
    _resolve_linear_decoder_targets,
    validate_stage1_trainable_boundary,
)
from stage1.token_rows import (
    SPARSE_TOKEN_ROWS_FILENAME,
    attach_sparse_output_token_rows,
    find_sparse_output_token_rows,
    initialize_frozen_output_token_rows,
    load_sparse_token_rows,
    save_sparse_token_rows,
)
from stage1.types import Stage1TuningConfig


STAGE1_PEFT_METADATA_FILENAME = "stage1_peft_metadata.json"
STAGE1_PEFT_SCHEMA_VERSION = 1
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
_DEEPSPEED_TAG = re.compile(r"global_step[0-9]+\Z")
_DEEPSPEED_MODEL_STATE = re.compile(
    r"(?:zero_pp_rank_[0-9]+_)?mp_rank_([0-9]{2})_model_states\.pt\Z"
)
_DEEPSPEED_OPTIMIZER_STATE = re.compile(
    r"(?:bf16_)?zero_pp_rank_[0-9]+_mp_rank_([0-9]{2})_optim_states\.pt\Z"
)
_MAX_FULL_INDEX_BYTES = 4 * 1024 * 1024
_MAX_DEEPSPEED_LATEST_BYTES = 1024


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
        return
    if binary_single:
        if binary_index or binary_shards:
            raise ValueError("mixed standalone and sharded full serialization")
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


def classify_stage1_artifact(path: Path) -> Stage1ArtifactKind:
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
        load_stage1_peft_metadata(directory)
        return Stage1ArtifactKind.PEFT

    if present_full:
        _validate_full_serialization(directory, names)
        return Stage1ArtifactKind.FULL
    raise ValueError(
        f"unrecognized local Stage 1 artifact (missing model weights): {directory}"
    )


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
    """Complete schema-v1 provenance for one Stage 1 adapter artifact."""

    schema_version: int
    stage: Literal["stage1"]
    base_model_reference: str
    base_model_architecture: dict[str, object]
    base_vocab_size: int
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
        if (
            type(payload.get("schema_version")) is not int
            or payload["schema_version"] != STAGE1_PEFT_SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported Stage 1 PEFT metadata schema: "
                f"expected={STAGE1_PEFT_SCHEMA_VERSION}, "
                f"actual={payload.get('schema_version')}"
            )
        expected_fields = {field.name for field in fields(cls)}
        actual_fields = set(payload)
        if actual_fields != expected_fields:
            raise ValueError(
                "invalid Stage 1 PEFT metadata fields: "
                f"missing={sorted(expected_fields - actual_fields)}, "
                f"unknown={sorted(actual_fields - expected_fields)}"
            )
        if payload["stage"] != "stage1":
            raise ValueError("Stage 1 PEFT metadata must describe stage1")
        if (
            not isinstance(payload["base_model_reference"], str)
            or not payload["base_model_reference"]
        ):
            raise ValueError("base_model_reference must be a non-empty string")

        integer_fields = (
            "base_vocab_size",
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
        if architecture["decoder"]["vocab_size"] != payload["base_vocab_size"]:
            raise ValueError(
                "base architecture vocab_size must equal base_vocab_size"
            )
        if payload["resized_vocab_size"] != payload["base_vocab_size"] + 3:
            raise ValueError(
                "resized_vocab_size must add exactly three latent tokens"
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
    """Read the strict schema-v1 Stage 1 sidecar."""

    source = Path(directory) / STAGE1_PEFT_METADATA_FILENAME
    if not source.is_file():
        raise ValueError(f"missing Stage 1 PEFT metadata: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Stage 1 PEFT metadata: {source}") from error
    return Stage1PeftMetadata.from_dict(payload)


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


def _validate_architecture(
    qwen: nn.Module,
    metadata: Stage1PeftMetadata,
    *,
    resized: bool,
) -> None:
    current = capture_qwen2_5_vl_architecture(qwen.config)
    expected_vocab = (
        metadata.resized_vocab_size if resized else metadata.base_vocab_size
    )
    if current["decoder"]["vocab_size"] != expected_vocab:
        raise ValueError(
            "base vocabulary mismatch: "
            f"metadata={expected_vocab}, model={current['decoder']['vocab_size']}"
        )
    comparable = deepcopy(current)
    if resized:
        comparable["decoder"]["vocab_size"] = metadata.base_vocab_size
    difference = first_architecture_difference(
        metadata.base_model_architecture, comparable
    )
    if difference is not None:
        path, expected, actual = difference
        raise ValueError(
            f"base architecture mismatch at {path}: "
            f"metadata={expected!r}, model={actual!r}"
        )

    embedding = _base_embedding(qwen)
    head = _base_head(qwen)
    if (
        embedding.num_embeddings != expected_vocab
        or head.out_features != expected_vocab
    ):
        raise ValueError(
            "input/output vocabulary mismatch: "
            f"expected={expected_vocab}, input={embedding.num_embeddings}, "
            f"output={head.out_features}"
        )


def _adapter_value(config, name: str):
    value = getattr(config, name, None)
    return getattr(value, "value", value)


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
    target_modules = tuple(getattr(config, "target_modules", ()) or ())
    if set(target_modules) != set(metadata.resolved_target_names):
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
    artifact_dir: Path,
    metadata: Stage1PeftMetadata,
) -> None:
    tensor_path = Path(artifact_dir) / _ADAPTER_SAFE_FILENAME
    modules = dict(qwen.named_modules())
    matched: set[str] = set()
    try:
        with safe_open(tensor_path, framework="pt", device="cpu") as stored:
            keys = tuple(stored.keys())

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
            input_shape = stored.get_slice(input_key).get_shape()
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
                    actual_shape = stored.get_slice(key).get_shape()
                    if actual_shape != expected_shape:
                        raise ValueError(
                            f"Stage 1 adapter target shape mismatch at {target}: "
                            f"adapter={actual_shape}, expected={expected_shape}"
                        )
            unexpected = sorted(set(keys) - matched)
            if unexpected:
                raise ValueError(
                    f"unexpected adapter tensor fields: {unexpected}"
                )
    except SafetensorError as error:
        raise ValueError(f"invalid Stage 1 adapter safetensors: {tensor_path}") from error


def _validate_sparse_artifact(
    artifact_dir: Path, metadata: Stage1PeftMetadata
) -> None:
    path = Path(artifact_dir) / SPARSE_TOKEN_ROWS_FILENAME
    try:
        with safe_open(path, framework="pt", device="cpu") as stored:
            if set(stored.keys()) != {"token_ids", "output_delta"}:
                raise ValueError("invalid sparse token-row tensor fields")
            sparse_metadata = stored.metadata()
            if (
                sparse_metadata is None
                or set(sparse_metadata) != {"format_version", "training_dtype"}
                or sparse_metadata["format_version"] != "1"
                or sparse_metadata["training_dtype"]
                not in {"float16", "bfloat16", "float32", "float64"}
            ):
                raise ValueError("invalid sparse token-row metadata")
            token_ids = stored.get_tensor("token_ids")
            output_delta = stored.get_tensor("output_delta")
    except SafetensorError as error:
        raise ValueError(f"invalid sparse token-row artifact: {path}") from error
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
    _validate_architecture(qwen, validated, resized=True)
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


def load_stage1_peft_artifact(
    base_model: nn.Module,
    processor,
    artifact_dir: Path,
    *,
    is_trainable: bool,
) -> nn.Module:
    """Validate, resize, and restore one Stage 1 PEFT artifact in strict order."""

    if type(is_trainable) is not bool:
        raise TypeError("is_trainable must be boolean")
    artifact = Path(artifact_dir)
    kind = classify_stage1_artifact(artifact)
    if kind is not Stage1ArtifactKind.PEFT:
        raise ValueError("Stage 1 PEFT loader requires a PEFT artifact")
    metadata = load_stage1_peft_metadata(artifact)
    if metadata.peft_version != peft.__version__:
        raise ValueError(
            "Stage 1 PEFT version mismatch: "
            f"metadata={metadata.peft_version}, installed={peft.__version__}"
        )

    candidate = getattr(base_model, "module", base_model)
    if isinstance(candidate, PeftModel):
        raise ValueError("Stage 1 PEFT loader requires an unmodified full base")
    qwen = resolve_qwen2_5_vl_model(candidate)
    _validate_processor(processor, metadata)
    _validate_latent_config(qwen.config, metadata)
    _validate_architecture(qwen, metadata, resized=False)

    adapter_config = PeftConfig.from_pretrained(artifact)
    _validate_adapter_config(qwen, adapter_config, metadata)
    _validate_adapter_tensors(qwen, artifact, metadata)
    _validate_sparse_artifact(artifact, metadata)

    recorded_source = metadata.base_model_reference
    current_source = getattr(qwen.config, "_name_or_path", "")
    if current_source and current_source != recorded_source:
        logging.getLogger(__name__).warning(
            "loading relocated Stage 1 base %s (recorded source: %s)",
            current_source,
            recorded_source,
        )

    candidate.resize_token_embeddings(metadata.resized_vocab_size)
    initialize_frozen_output_token_rows(candidate, metadata.latent_token_ids)
    _configure_latent_config(candidate.config, metadata)
    restored = PeftModel.from_pretrained(
        candidate, artifact, is_trainable=is_trainable
    )
    if not is_trainable:
        # PEFT 0.17.1 leaves TrainableTokensModel's selective delta enabled
        # even when the adapter itself is loaded for inference.
        restored.requires_grad_(False)
    # PEFT restoration never creates the repository-owned output-row wrapper.
    rows = attach_sparse_output_token_rows(restored, metadata.latent_token_ids)
    rows.output_delta.requires_grad_(is_trainable)
    load_sparse_token_rows(restored, artifact, metadata.latent_token_ids)
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


def _is_regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _has_complete_rng_layout(path: Path) -> bool:
    single = path / "rng_state.pth"
    rank_files = {
        int(match.group(1)): entry
        for entry in path.iterdir()
        if entry.is_file()
        and (match := _RANK_RNG_STATE.fullmatch(entry.name))
    }
    if _is_regular_file(single):
        return not rank_files
    if single.exists() or not rank_files:
        return False
    return sorted(rank_files) == list(range(len(rank_files))) and all(
        _is_regular_file(rank_file) for rank_file in rank_files.values()
    )


def _has_complete_standard_trainer_state(path: Path) -> bool:
    return all(
        _is_regular_file(path / filename)
        for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    ) and _has_complete_rng_layout(path)


def _has_complete_deepspeed_trainer_state(path: Path) -> bool:
    latest = path / "latest"
    if not _is_regular_file(latest):
        return False
    try:
        if latest.stat().st_size > _MAX_DEEPSPEED_LATEST_BYTES:
            return False
        tag_name = latest.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return False
    if _DEEPSPEED_TAG.fullmatch(tag_name) is None:
        return False
    tag = path / tag_name
    if tag.is_symlink() or not tag.is_dir() or tag.resolve().parent != path.resolve():
        return False

    model_mp_ranks: set[int] = set()
    optimizer_mp_ranks: set[int] = set()
    for entry in tag.iterdir():
        if entry.is_symlink():
            return False
        model_match = _DEEPSPEED_MODEL_STATE.fullmatch(entry.name)
        optimizer_match = _DEEPSPEED_OPTIMIZER_STATE.fullmatch(entry.name)
        if model_match is not None and entry.is_file():
            model_mp_ranks.add(int(model_match.group(1)))
        elif optimizer_match is not None and entry.is_file():
            optimizer_mp_ranks.add(int(optimizer_match.group(1)))
        elif entry.name.endswith("_states.pt"):
            return False
    return bool(model_mp_ranks) and model_mp_ranks == optimizer_mp_ranks


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
) -> Stage1ArtifactKind:
    kind = classify_stage1_artifact(path)
    if kind is Stage1ArtifactKind.PEFT:
        if _NUMBERED_CHECKPOINT.fullmatch(path.name) is None:
            raise ValueError(
                f"Stage 1 adapter resume requires a numbered checkpoint: {path}"
            )
        _validate_peft_resume_location(output_dir, path)
        if not _is_regular_file(path / "trainer_state.json"):
            raise ValueError(
                "Stage 1 adapter resume requires complete Trainer state; "
                f"missing trainer_state.json: {path}"
            )
        if not _has_complete_rng_layout(path):
            raise ValueError(
                f"Stage 1 adapter resume has incomplete RNG state: {path}"
            )
        if not (
            _has_complete_standard_trainer_state(path)
            or _has_complete_deepspeed_trainer_state(path)
        ):
            raise ValueError(
                "Stage 1 adapter resume requires complete optimizer/scheduler "
                f"state or a complete DeepSpeed checkpoint: {path}"
            )
    elif not _is_regular_file(path / "trainer_state.json"):
        raise ValueError(f"Stage 1 resume is missing trainer_state.json: {path}")
    return kind


def resolve_stage1_resume(
    output_dir: Path,
    explicit_resume: str | Path | None,
) -> Stage1ResumeDecision:
    """Resolve explicit or latest valid immediate numbered Stage 1 checkpoint."""

    output = Path(output_dir)
    if explicit_resume is not None:
        checkpoint = Path(explicit_resume)
        kind = _validated_resume_checkpoint(checkpoint, output_dir=output)
        return Stage1ResumeDecision(checkpoint, kind, True)

    if not output.is_dir():
        return Stage1ResumeDecision(None, None, False)
    candidates: list[tuple[int, Path, Stage1ArtifactKind]] = []
    for child in output.iterdir():
        if child.is_symlink() or not child.is_dir():
            continue
        match = _NUMBERED_CHECKPOINT.fullmatch(child.name)
        if match is None:
            continue
        try:
            kind = _validated_resume_checkpoint(child, output_dir=output)
        except (OSError, ValueError):
            continue
        candidates.append((int(match.group(1)), child, kind))
    if not candidates:
        return Stage1ResumeDecision(None, None, False)
    _, checkpoint, kind = max(candidates, key=lambda item: item[0])
    return Stage1ResumeDecision(checkpoint, kind, False)


__all__ = [
    "STAGE1_PEFT_METADATA_FILENAME",
    "STAGE1_PEFT_SCHEMA_VERSION",
    "STAGE1_REPOSITORY_FORMAT_VERSION",
    "Stage1ArtifactKind",
    "Stage1PeftMetadata",
    "Stage1ResumeDecision",
    "classify_stage1_artifact",
    "load_stage1_peft_artifact",
    "load_stage1_peft_metadata",
    "resolve_stage1_resume",
    "save_stage1_peft_artifact",
    "save_stage1_peft_metadata",
]
