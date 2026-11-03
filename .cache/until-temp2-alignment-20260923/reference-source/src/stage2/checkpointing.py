"""Validate Stage 2 token metadata and checkpoint transitions."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
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
    validate_stage1_vocab_alignment,
)
from artifact_paths import CANONICAL_CHECKPOINT_ROOT
from step_preference.config import (
    DEFAULT_STEP_PREFERENCE_EXECUTION_BACKEND,
    normalize_step_preference_execution_backend,
    preference_prefix_source,
    split_stage2_optimizer_budget,
)
from step_preference.gated import GatedExecutionIdentity, gated_execution_identity
from .peft import Stage2TuningConfig, resolve_stage2_target_module_names
from transformers.utils.peft_utils import find_adapter_config_file
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentTokenIds,
    normalize_stage2_rollout_backend,
    stage2_prefix_dropout_semantics,
    validate_stage2_config,
)


STAGE2_PEFT_METADATA_FILENAME = "stage2_peft_metadata.json"
STAGE2_PEFT_SCHEMA_VERSION = 7
STEP_PREFERENCE_METADATA_FILENAME = "step_preference_metadata.json"
STEP_PREFERENCE_SCHEMA_VERSION = 7


@dataclass(frozen=True)
class StepPreferenceMetadata:
    """Semantic identity for one Stage 2A or Stage 2B training phase."""

    schema_version: int
    phase: str
    total_optimizer_steps: int | None
    phase_optimizer_steps: int | None
    stage2a_ratio: float | None
    stage2a_artifact: str | None
    stage2a_metadata_digest: str | None
    stage2a_lifecycle_metadata_digest: str | None
    beta: float
    lambda_preference: float
    lambda_entropy: float
    entropy_margin: float
    ablation: str
    modalities: str = "visual_text"
    preference_margin: float = 0.0
    objective: str = "context_preference_v1"
    execution_backend: str | None = None
    sidecar_prefix_source: str | None = None
    target_scope: str = "until_next_latent"
    prefix_mode: str = "clean_prefix"
    branch_mode: str = "independent"
    on_layer_checkpoint_stride: int = 1
    gated_execution: GatedExecutionIdentity | None = None


_STEP_PREFERENCE_METADATA_FIELDS = tuple(
    field.name for field in fields(StepPreferenceMetadata)
)
_V6_STEP_PREFERENCE_METADATA_FIELDS = frozenset(
    {
        "schema_version", "phase", "total_optimizer_steps",
        "phase_optimizer_steps", "stage2a_ratio", "stage2a_artifact",
        "stage2a_metadata_digest", "stage2a_lifecycle_metadata_digest",
        "beta", "lambda_preference", "lambda_entropy", "entropy_margin",
        "ablation", "modalities", "preference_margin", "objective",
        "execution_backend", "sidecar_prefix_source", "target_scope",
        "prefix_mode", "branch_mode", "on_layer_checkpoint_stride",
    }
)
_V5_STEP_PREFERENCE_METADATA_FIELDS = frozenset(
    {
        "schema_version", "phase", "total_optimizer_steps",
        "phase_optimizer_steps", "stage2a_ratio", "stage2a_artifact",
        "stage2a_metadata_digest", "stage2a_lifecycle_metadata_digest",
        "beta", "lambda_preference", "lambda_entropy", "entropy_margin",
        "ablation", "modalities", "preference_margin", "objective",
        "execution_backend", "sidecar_prefix_source", "target_scope",
    }
)
_V4_STEP_PREFERENCE_METADATA_FIELDS = frozenset(
    {
        "schema_version", "phase", "total_optimizer_steps",
        "phase_optimizer_steps", "stage2a_ratio", "stage2a_artifact",
        "stage2a_metadata_digest", "stage2a_lifecycle_metadata_digest",
        "beta", "lambda_dpo", "lambda_entropy", "entropy_margin",
        "ablation", "execution_backend", "sidecar_prefix_source", "reference",
        "negative_manifest", "negative_manifest_digest",
        "negative_records_sha256",
    }
)
_V3_STEP_PREFERENCE_METADATA_FIELDS = frozenset(
    {
        "schema_version", "phase", "total_optimizer_steps",
        "phase_optimizer_steps", "stage2a_ratio", "stage2a_artifact",
        "stage2a_metadata_digest", "stage2a_lifecycle_metadata_digest",
        "beta", "lambda_dpo", "lambda_entropy", "entropy_margin",
        "ablation", "execution_backend", "reference", "negative_manifest",
        "negative_manifest_digest", "negative_records_sha256",
    }
)
_V2_STEP_PREFERENCE_METADATA_FIELDS = frozenset(
    {
        "schema_version", "phase", "total_optimizer_steps",
        "phase_optimizer_steps", "stage2a_ratio", "stage2a_artifact",
        "stage2a_metadata_digest", "stage2a_lifecycle_metadata_digest",
        "beta", "lambda_dpo", "lambda_entropy", "entropy_margin",
        "ablation", "reference", "negative_manifest",
        "negative_manifest_digest", "negative_records_sha256",
    }
)


def _validated_gated_execution_identity(
    identity: GatedExecutionIdentity,
    ablation: str,
) -> GatedExecutionIdentity:
    if not isinstance(identity, GatedExecutionIdentity):
        raise ValueError("gated_execution must be a JSON object")
    expected = gated_execution_identity(
        rollout_backend=identity.clean_backend,
        native_checkpointing=identity.native_checkpointing,
        latent_checkpointing=identity.latent_checkpointing,
        cache_checkpointing=identity.cache_checkpointing,
        ablation=ablation,
    )
    if identity != expected:
        raise ValueError("gated_execution contains inconsistent semantics")
    return identity


def _validate_step_preference_metadata(metadata: StepPreferenceMetadata) -> None:
    if not isinstance(metadata, StepPreferenceMetadata):
        raise TypeError("metadata must be StepPreferenceMetadata")
    if (
        type(metadata.schema_version) is not int
        or metadata.schema_version != STEP_PREFERENCE_SCHEMA_VERSION
    ):
        raise ValueError(
            "step preference schema_version must be "
            f"{STEP_PREFERENCE_SCHEMA_VERSION}"
        )
    if not isinstance(metadata.phase, str) or metadata.phase not in {
        "stage2a",
        "stage2b",
    }:
        raise ValueError("step preference phase must be stage2a or stage2b")

    for name in ("total_optimizer_steps", "phase_optimizer_steps"):
        value = getattr(metadata, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if metadata.total_optimizer_steps <= 1:
        raise ValueError("total_optimizer_steps must be greater than one")
    direct = (metadata.phase == "stage2b"
              and type(metadata.stage2a_ratio) in (int, float)
              and metadata.stage2a_ratio == 0)
    if not direct and metadata.phase_optimizer_steps >= metadata.total_optimizer_steps:
        raise ValueError("phase_optimizer_steps must be less than total_optimizer_steps")
    if (
        type(metadata.stage2a_ratio) not in (int, float)
        or not math.isfinite(metadata.stage2a_ratio)
        or (not direct and not 0.0 < metadata.stage2a_ratio < 1.0)
    ):
        raise ValueError("stage2a_ratio must be finite and strictly between zero and one")
    stage2a_steps = math.floor(
        metadata.total_optimizer_steps * metadata.stage2a_ratio
    )
    expected_phase_steps = (
        stage2a_steps
        if metadata.phase == "stage2a"
        else metadata.total_optimizer_steps - stage2a_steps
    )
    if metadata.phase_optimizer_steps != expected_phase_steps:
        raise ValueError(
            "phase_optimizer_steps does not match phase budget: "
            f"expected={expected_phase_steps}, actual={metadata.phase_optimizer_steps}"
        )

    artifact_fields = (
        "stage2a_artifact",
        "stage2a_metadata_digest",
        "stage2a_lifecycle_metadata_digest",
    )
    if metadata.phase == "stage2a":
        for name in artifact_fields:
            if getattr(metadata, name) is not None:
                raise ValueError(f"Stage 2A {name} must be null")
        if metadata.execution_backend is not None:
            raise ValueError("Stage 2A execution_backend must be null")
        if metadata.sidecar_prefix_source is not None:
            raise ValueError("Stage 2A sidecar_prefix_source must be null")
        if (
            metadata.prefix_mode != "clean_prefix"
            or metadata.branch_mode != "independent"
            or metadata.on_layer_checkpoint_stride != 1
            or metadata.gated_execution is not None
        ):
            raise ValueError("Stage 2A requires clean-prefix insertion defaults")
    elif direct:
        for name in artifact_fields:
            if getattr(metadata, name) is not None:
                raise ValueError(f"direct preference {name} must be null")
        normalize_step_preference_execution_backend(metadata.execution_backend)
    else:
        for name in artifact_fields:
            value = getattr(metadata, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Stage 2B {name} must be a non-empty string")
        for name in (
            "stage2a_lifecycle_metadata_digest",
            "stage2a_metadata_digest",
        ):
            value = getattr(metadata, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    f"Stage 2B {name} must be a lowercase SHA-256 digest"
                )
        normalize_step_preference_execution_backend(metadata.execution_backend)

    if metadata.phase == "stage2b":
        if metadata.prefix_mode not in {
            "clean_prefix", "v_off_insertion", "single_image_latent_gate",
        }:
            raise ValueError(
                "prefix_mode must be clean_prefix, v_off_insertion, or "
                "single_image_latent_gate"
            )
        if metadata.branch_mode not in {"independent", "cumulative"}:
            raise ValueError("branch_mode must be independent or cumulative")
        if (
            type(metadata.on_layer_checkpoint_stride) is not int
            or metadata.on_layer_checkpoint_stride not in {1, 2, 4}
        ):
            raise ValueError("on_layer_checkpoint_stride must be 1, 2, or 4")
        if metadata.prefix_mode == "clean_prefix" and (
            metadata.branch_mode != "independent"
            or metadata.on_layer_checkpoint_stride != 1
        ):
            raise ValueError("branch mode and ON stride require v_off_insertion")
        expected_backend = {
            "clean_prefix": DEFAULT_STEP_PREFERENCE_EXECUTION_BACKEND,
            "v_off_insertion": "v_off_insertion",
            "single_image_latent_gate": "single_image_latent_gate",
        }[metadata.prefix_mode]
        if metadata.execution_backend != expected_backend:
            raise ValueError("step preference execution backend disagrees with prefix_mode")
        if metadata.sidecar_prefix_source != preference_prefix_source(metadata.prefix_mode):
            raise ValueError("Stage 2B sidecar_prefix_source disagrees with prefix_mode")
        if metadata.prefix_mode == "single_image_latent_gate":
            if metadata.on_layer_checkpoint_stride != 1:
                raise ValueError("gated execution requires ON layer checkpoint stride 1")
            if metadata.gated_execution is None:
                raise ValueError("single_image_latent_gate requires gated_execution")
            _validated_gated_execution_identity(
                metadata.gated_execution, metadata.ablation
            )
        elif metadata.gated_execution is not None:
            raise ValueError("legacy prefix modes require gated_execution=null")

    scalars = (
        ("beta", metadata.beta),
        ("lambda_preference", metadata.lambda_preference),
        ("lambda_entropy", metadata.lambda_entropy),
        ("entropy_margin", metadata.entropy_margin),
        ("preference_margin", metadata.preference_margin),
    )
    for name, value in scalars:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if metadata.beta <= 0.0:
        raise ValueError("beta must be positive")
    for name, value in scalars[1:]:
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if metadata.ablation not in {
        "attention_mask",
        "remove_full_visual_block",
        "remove_full_latent_block_keep_image",
    }:
        raise ValueError("unsupported step preference ablation")
    if metadata.modalities not in {"visual", "visual_text"}:
        raise ValueError("modalities must be visual or visual_text")
    if metadata.target_scope not in {"until_next_latent", "immediate_text"}:
        raise ValueError("target_scope must be until_next_latent or immediate_text")
    if metadata.objective != "context_preference_v1":
        raise ValueError("objective must be context_preference_v1")


def save_step_preference_metadata(
    path: Path, metadata: StepPreferenceMetadata
) -> None:
    """Atomically write the validated preference sidecar with stable JSON."""
    _validate_step_preference_metadata(metadata)
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / STEP_PREFERENCE_METADATA_FILENAME
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(asdict(metadata), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _load_step_preference_metadata_with_bytes(
    path: Path,
) -> tuple[StepPreferenceMetadata, bytes]:
    source = Path(path) / STEP_PREFERENCE_METADATA_FILENAME
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"missing or invalid step preference metadata: {source}")
    try:
        source_bytes = source.read_bytes()
        payload = json.loads(source_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid step preference metadata: {source}") from error
    metadata = step_preference_metadata_from_dict(payload)
    return metadata, source_bytes


def step_preference_metadata_from_dict(
    raw: Mapping[str, object],
) -> StepPreferenceMetadata:
    """Parse context preference v7; migrate historical modes without gating."""
    if not isinstance(raw, Mapping):
        raise ValueError("step preference metadata must be a JSON object")
    schema_version = raw.get("schema_version")
    if type(schema_version) is not int:
        raise ValueError("step preference schema_version must be an integer")
    payload = dict(raw)
    if schema_version in (2, 3, 4) and payload.get("phase") == "stage2b":
        raise ValueError("legacy Stage 2B cannot resume as context preference; warm-start from Stage 2A")
    if schema_version == 2:
        actual = set(payload)
        if actual != _V2_STEP_PREFERENCE_METADATA_FIELDS:
            raise ValueError(
                "invalid step preference metadata fields: "
                f"missing={sorted(_V2_STEP_PREFERENCE_METADATA_FIELDS - actual)}, "
                f"unknown={sorted(actual - _V2_STEP_PREFERENCE_METADATA_FIELDS)}"
            )
        if payload["phase"] == "stage2b":
            raise ValueError(
                "legacy Stage 2B metadata requires current execution_backend"
            )
        payload["execution_backend"] = None
        payload["schema_version"] = 3
        schema_version = 3
    if schema_version == 3:
        actual = set(payload)
        if actual != _V3_STEP_PREFERENCE_METADATA_FIELDS:
            raise ValueError(
                "invalid step preference metadata fields: "
                f"missing={sorted(_V3_STEP_PREFERENCE_METADATA_FIELDS - actual)}, "
                f"unknown={sorted(actual - _V3_STEP_PREFERENCE_METADATA_FIELDS)}"
            )
        if payload["phase"] == "stage2b":
            raise ValueError(
                "Stage 2B lifecycle schema v3 predates final_clean_forward"
            )
        payload["sidecar_prefix_source"] = None
        payload["schema_version"] = 4
        schema_version = 4
    if schema_version == 4:
        if set(payload) != _V4_STEP_PREFERENCE_METADATA_FIELDS:
            raise ValueError("invalid legacy step preference metadata fields")
        for name in ("negative_manifest", "negative_manifest_digest", "negative_records_sha256"):
            if payload.pop(name) is not None:
                raise ValueError(f"Stage 2A {name} must be null")
        if payload.pop("reference") != "stage2a":
            raise ValueError("legacy reference must be stage2a")
        payload["lambda_preference"] = payload.pop("lambda_dpo")
        payload["preference_margin"] = 0.0
        payload["objective"] = "context_preference_v1"
        payload["schema_version"] = 5
        schema_version = 5
    if schema_version == 5:
        # Original v5 always executed both branches. Never infer visual-only.
        payload.setdefault("modalities", "visual_text")
        # Pre-scope v5 always scored the entire text run. Never infer immediate_text.
        payload.setdefault("target_scope", "until_next_latent")
        actual = set(payload)
        expected = set(_V5_STEP_PREFERENCE_METADATA_FIELDS)
        if actual != expected:
            raise ValueError(
                "invalid step preference metadata fields: "
                f"missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)}"
            )
        payload.update(
            prefix_mode="clean_prefix",
            branch_mode="independent",
            on_layer_checkpoint_stride=1,
            schema_version=6,
        )
        schema_version = 6
    if schema_version == 6:
        actual = set(payload)
        if actual != _V6_STEP_PREFERENCE_METADATA_FIELDS:
            raise ValueError(
                "invalid step preference metadata fields: "
                f"missing={sorted(_V6_STEP_PREFERENCE_METADATA_FIELDS - actual)}, "
                f"unknown={sorted(actual - _V6_STEP_PREFERENCE_METADATA_FIELDS)}"
            )
        if payload["prefix_mode"] == "single_image_latent_gate":
            raise ValueError("schema v6 cannot describe single_image_latent_gate")
        payload.update(gated_execution=None, schema_version=7)
        schema_version = 7
    if schema_version == STEP_PREFERENCE_SCHEMA_VERSION:
        actual = set(payload)
        expected = set(_STEP_PREFERENCE_METADATA_FIELDS)
        if actual != expected:
            raise ValueError(
                "invalid step preference metadata fields: "
                f"missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)}"
            )
        gated_payload = payload["gated_execution"]
        if gated_payload is not None:
            if not isinstance(gated_payload, Mapping):
                raise ValueError("gated_execution must be a JSON object or null")
            expected_gated_fields = {
                field.name for field in fields(GatedExecutionIdentity)
            }
            actual_gated_fields = set(gated_payload)
            if actual_gated_fields != expected_gated_fields:
                raise ValueError(
                    "invalid gated_execution fields: "
                    f"missing={sorted(expected_gated_fields - actual_gated_fields)}, "
                    f"unknown={sorted(actual_gated_fields - expected_gated_fields)}"
                )
            payload["gated_execution"] = GatedExecutionIdentity(**gated_payload)
    else:
        raise ValueError(
            "unsupported step preference metadata schema: "
            f"expected={STEP_PREFERENCE_SCHEMA_VERSION}, actual={schema_version}"
        )
    try:
        metadata = StepPreferenceMetadata(**payload)
        _validate_step_preference_metadata(metadata)
    except TypeError as error:
        raise ValueError("invalid step preference metadata values") from error
    return metadata


def load_step_preference_metadata(path: Path) -> StepPreferenceMetadata:
    """Load one strict preference sidecar without consulting model artifacts."""
    metadata, _source_bytes = _load_step_preference_metadata_with_bytes(path)
    return metadata


def _optional_step_preference_metadata(path: str | Path) -> StepPreferenceMetadata | None:
    source = Path(path) / STEP_PREFERENCE_METADATA_FILENAME
    if source.exists() or source.is_symlink():
        return load_step_preference_metadata(Path(path))
    return None


@dataclass(frozen=True)
class Stage2ALifecycleIdentity:
    """Validated Stage 2A lifecycle contract and its small-sidecar digest."""

    metadata: StepPreferenceMetadata
    path: Path
    digest: str


def load_stage2a_lifecycle_identity(
    path: str | Path,
    *,
    total_optimizer_steps: int,
    stage2a_ratio: float,
) -> Stage2ALifecycleIdentity:
    """Authenticate a Stage 2A sidecar against the requested lifecycle split."""
    budget = split_stage2_optimizer_budget(total_optimizer_steps, stage2a_ratio)
    artifact = Path(path)
    metadata, source_bytes = _load_step_preference_metadata_with_bytes(artifact)
    if metadata.phase != "stage2a":
        raise ValueError("Stage 2A lifecycle requires phase=stage2a")
    if metadata.total_optimizer_steps != budget.total_steps:
        raise ValueError(
            "Stage 2A lifecycle total budget mismatch: "
            f"metadata={metadata.total_optimizer_steps}, requested={budget.total_steps}"
        )
    if metadata.stage2a_ratio != float(stage2a_ratio):
        raise ValueError(
            "Stage 2A lifecycle ratio mismatch: "
            f"metadata={metadata.stage2a_ratio}, requested={stage2a_ratio}"
        )
    if metadata.phase_optimizer_steps != budget.stage2a_steps:
        raise ValueError(
            "Stage 2A lifecycle phase steps mismatch: "
            f"metadata={metadata.phase_optimizer_steps}, requested={budget.stage2a_steps}"
        )
    return Stage2ALifecycleIdentity(
        metadata=metadata,
        path=artifact / STEP_PREFERENCE_METADATA_FILENAME,
        digest=hashlib.sha256(source_bytes).hexdigest(),
    )


def validate_step_preference_resume_identity(
    checkpoint_metadata: StepPreferenceMetadata,
    requested_metadata: StepPreferenceMetadata,
) -> None:
    """Reject the first changed semantic field of a preference-phase resume."""
    _validate_step_preference_metadata(checkpoint_metadata)
    if not isinstance(requested_metadata, StepPreferenceMetadata):
        raise TypeError("requested_metadata must be StepPreferenceMetadata")
    for name in _STEP_PREFERENCE_METADATA_FIELDS:
        checkpoint_value = getattr(checkpoint_metadata, name)
        requested_value = getattr(requested_metadata, name)
        if checkpoint_value != requested_value:
            raise ValueError(
                f"cannot resume with changed {name}: "
                f"checkpoint={checkpoint_value!r}, requested={requested_value!r}"
            )
    _validate_step_preference_metadata(requested_metadata)


def digest_small_metadata(path: Path) -> str:
    """Hash exactly one supplied regular metadata file, never an artifact tree."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"metadata path must be a regular file: {source}")
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read metadata file: {source}") from error
    return hashlib.sha256(payload).hexdigest()


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
    is_checkpoint_alias = candidate.is_symlink()
    try:
        canonical = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"Stage 2 artifact does not exist: {candidate}") from error
    if not canonical.is_dir():
        raise ValueError(f"Stage 2 artifact is not a directory: {candidate}")
    if is_checkpoint_alias:
        checkpoint_root = CANONICAL_CHECKPOINT_ROOT.resolve(strict=False)
        if not canonical.is_relative_to(checkpoint_root):
            raise ValueError(
                "Stage 2 artifact symlink resolves outside canonical checkpoint "
                f"root: alias={candidate}, target={canonical}"
            )
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
    tokenizer_base_vocab_size: int
    pretrained_model_vocab_size: int
    resized_vocab_size: int
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
    rollout_backend: str
    prefix_dropout_semantics: str
    stage2_cache_activation_checkpointing: bool
    execution_backend: str | None = None

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


_V7_STAGE2_PEFT_FIELDS = frozenset(
    field.name for field in fields(Stage2PeftMetadata)
)
_V6_STAGE2_PEFT_FIELDS = frozenset(
    _V7_STAGE2_PEFT_FIELDS - {"execution_backend"}
)
_V5_STAGE2_PEFT_FIELDS = frozenset(
    _V6_STAGE2_PEFT_FIELDS - {"stage2_cache_activation_checkpointing"}
)
_V4_STAGE2_PEFT_FIELDS = frozenset(
    _V5_STAGE2_PEFT_FIELDS - {"rollout_backend", "prefix_dropout_semantics"}
)
_V3_STAGE2_PEFT_FIELDS = frozenset(
    _V4_STAGE2_PEFT_FIELDS
    - {
        "tokenizer_base_vocab_size",
        "pretrained_model_vocab_size",
        "resized_vocab_size",
    }
)
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


def _parse_stage2_metadata_v7(
    payload: Mapping[str, object],
) -> Stage2PeftMetadata:
    if set(payload) != _V7_STAGE2_PEFT_FIELDS:
        actual = set(payload)
        raise ValueError(
            "invalid Stage 2 PEFT metadata fields: "
            f"missing={sorted(_V7_STAGE2_PEFT_FIELDS - actual)}, "
            f"unknown={sorted(actual - _V7_STAGE2_PEFT_FIELDS)}"
        )
    if payload["stage"] != "stage2":
        raise ValueError("Stage 2 PEFT metadata must describe stage2")
    if payload["tuning_method"] not in ("lora", "dora"):
        raise ValueError("Stage 2 PEFT metadata requires LoRA or DoRA")
    if payload["token_row_source"] not in ("base", "sparse_adapter"):
        raise ValueError("token_row_source must be base or sparse_adapter")
    rollout_backend = normalize_stage2_rollout_backend(payload["rollout_backend"])
    expected_dropout_semantics = stage2_prefix_dropout_semantics(rollout_backend)
    if payload["prefix_dropout_semantics"] != expected_dropout_semantics:
        raise ValueError(
            "Stage 2 prefix dropout semantics do not match rollout backend"
        )
    adapter_reference = payload["stage1_adapter_reference"]
    if adapter_reference is not None and (
        not isinstance(adapter_reference, str) or not adapter_reference
    ):
        raise ValueError(
            "stage1_adapter_reference must be a non-empty string or null"
        )

    integer_fields = (
        "schema_version",
        "tokenizer_base_vocab_size",
        "pretrained_model_vocab_size",
        "resized_vocab_size",
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
    if type(payload["stage2_cache_activation_checkpointing"]) is not bool:
        raise ValueError("stage2_cache_activation_checkpointing must be a boolean")
    if (
        payload["stage2_cache_activation_checkpointing"]
        and rollout_backend != "differentiable_cache"
    ):
        raise ValueError(
            "stage2_cache_activation_checkpointing requires differentiable_cache"
        )
    execution_backend = payload["execution_backend"]
    if execution_backend is not None:
        execution_backend = normalize_step_preference_execution_backend(
            execution_backend
        )
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
    if any(
        payload[name] <= 0
        for name in (
            "tokenizer_base_vocab_size",
            "pretrained_model_vocab_size",
            "resized_vocab_size",
        )
    ):
        raise ValueError("Stage 2 vocabulary sizes must be positive")
    if any(
        payload[name] < 0
        for name in (
            "latent_start_token_id",
            "latent_pad_token_id",
            "latent_end_token_id",
        )
    ):
        raise ValueError("latent token IDs must be non-negative")
    if len(set(converted["resolved_target_module_names"])) != len(
        converted["resolved_target_module_names"]
    ):
        raise ValueError("resolved target module names must be unique")

    architecture = _normalize_base_model_architecture(
        payload["base_model_architecture"]
    )
    tokenizer_base_vocab_size = payload["tokenizer_base_vocab_size"]
    pretrained_model_vocab_size = payload["pretrained_model_vocab_size"]
    resized_vocab_size = payload["resized_vocab_size"]
    if resized_vocab_size != tokenizer_base_vocab_size + 3:
        raise ValueError("resized_vocab_size must add exactly three latent tokens")
    if pretrained_model_vocab_size < tokenizer_base_vocab_size:
        raise ValueError(
            "pretrained_model_vocab_size must be at least tokenizer_base_vocab_size"
        )
    if architecture["decoder"]["vocab_size"] != resized_vocab_size:
        raise ValueError(
            "Stage 2 architecture vocab_size must equal resized_vocab_size"
        )
    if payload["token_row_source"] == "sparse_adapter" and (
        payload["latent_pad_token_id"],
        payload["latent_start_token_id"],
        payload["latent_end_token_id"],
    ) != tuple(range(tokenizer_base_vocab_size, resized_vocab_size)):
        raise ValueError(
            "sparse Stage 2 latent IDs must be the exact appended tokenizer rows"
        )

    return Stage2PeftMetadata(
        schema_version=STAGE2_PEFT_SCHEMA_VERSION,
        stage="stage2",
        tuning_method=tuning.method,
        base_model_reference=payload["base_model_reference"],
        base_model_architecture=architecture,
        tokenizer_base_vocab_size=tokenizer_base_vocab_size,
        pretrained_model_vocab_size=pretrained_model_vocab_size,
        resized_vocab_size=resized_vocab_size,
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
        rollout_backend=rollout_backend,
        prefix_dropout_semantics=expected_dropout_semantics,
        stage2_cache_activation_checkpointing=payload[
            "stage2_cache_activation_checkpointing"
        ],
        execution_backend=execution_backend,
    )


def _parse_stage2_metadata_v6(
    payload: Mapping[str, object],
) -> Stage2PeftMetadata:
    if set(payload) != _V6_STAGE2_PEFT_FIELDS:
        actual = set(payload)
        raise ValueError(
            "invalid Stage 2 PEFT metadata fields: "
            f"missing={sorted(_V6_STAGE2_PEFT_FIELDS - actual)}, "
            f"unknown={sorted(actual - _V6_STAGE2_PEFT_FIELDS)}"
        )
    migrated = dict(payload)
    migrated.update(
        schema_version=STAGE2_PEFT_SCHEMA_VERSION,
        execution_backend=None,
    )
    return _parse_stage2_metadata_v7(migrated)


def _parse_stage2_metadata_v5(payload: Mapping[str, object]) -> Stage2PeftMetadata:
    if set(payload) != _V5_STAGE2_PEFT_FIELDS:
        actual = set(payload)
        raise ValueError(
            "invalid Stage 2 PEFT metadata fields: "
            f"missing={sorted(_V5_STAGE2_PEFT_FIELDS - actual)}, "
            f"unknown={sorted(actual - _V5_STAGE2_PEFT_FIELDS)}"
        )
    migrated = dict(payload)
    migrated.update(
        schema_version=6,
        stage2_cache_activation_checkpointing=False,
    )
    return _parse_stage2_metadata_v6(migrated)


def migrate_stage2_metadata(raw: Mapping[str, object]) -> Stage2PeftMetadata:
    """Strictly parse v7 or migrate exact complete v2-v6 payloads."""
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
            schema_version=3,
            token_row_source="base",
            stage1_adapter_reference=None,
        )
        raw = migrated
        schema_version = 3
    if schema_version == 3:
        actual = set(raw)
        if actual != _V3_STAGE2_PEFT_FIELDS:
            raise ValueError(
                "Stage 2 PEFT migration requires an exact complete v3 payload: "
                f"missing={sorted(_V3_STAGE2_PEFT_FIELDS - actual)}, "
                f"unknown={sorted(actual - _V3_STAGE2_PEFT_FIELDS)}"
            )
        if raw["token_row_source"] == "sparse_adapter":
            raise ValueError(
                "Stage 2 sparse schema v3 cannot be loaded or automatically "
                "migrated because it lacks pretrained padded vocabulary "
                "provenance; restart a fresh Stage 2 run from an authenticated "
                "Stage 1 schema v2 checkpoint to create schema v4. Existing v3 "
                "optimizer, scheduler, global-step, and Stage 2 weight state is "
                "not preserved; preserving it requires a future audited offline "
                "converter, which is not implemented"
            )
        migrated = dict(raw)
        architecture = _normalize_base_model_architecture(
            migrated["base_model_architecture"]
        )
        resized_vocab_size = int(architecture["decoder"]["vocab_size"])
        tokenizer_base_vocab_size = resized_vocab_size - 3
        migrated.update(
            schema_version=4,
            tokenizer_base_vocab_size=tokenizer_base_vocab_size,
            pretrained_model_vocab_size=tokenizer_base_vocab_size,
            resized_vocab_size=resized_vocab_size,
        )
        raw = migrated
        schema_version = 4
    if schema_version == 4:
        actual = set(raw)
        if actual != _V4_STAGE2_PEFT_FIELDS:
            raise ValueError(
                "Stage 2 PEFT migration requires an exact complete v4 payload: "
                f"missing={sorted(_V4_STAGE2_PEFT_FIELDS - actual)}, "
                f"unknown={sorted(actual - _V4_STAGE2_PEFT_FIELDS)}"
            )
        migrated = dict(raw)
        migrated.update(
            schema_version=5,
            rollout_backend="no_cache",
            prefix_dropout_semantics="resample_per_prefix",
        )
        return _parse_stage2_metadata_v5(migrated)
    if schema_version == 5:
        return _parse_stage2_metadata_v5(raw)
    if schema_version == 6:
        return _parse_stage2_metadata_v6(raw)
    if schema_version == STAGE2_PEFT_SCHEMA_VERSION:
        return _parse_stage2_metadata_v7(raw)
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
    tokenizer_base_vocab_size: int | None = None,
    pretrained_model_vocab_size: int | None = None,
    resized_vocab_size: int | None = None,
    source_base_model_architecture: Mapping[str, object] | None = None,
    rollout_backend: str | None = None,
    stage2_cache_activation_checkpointing: bool | None = None,
    execution_backend: str | None = None,
) -> Stage2PeftMetadata:
    """Build the sidecar for a fresh Stage 2 PEFT run."""
    if not tuning.is_peft:
        raise ValueError("Stage 2 PEFT metadata requires LoRA or DoRA tuning")
    vocab_provenance = (
        tokenizer_base_vocab_size,
        pretrained_model_vocab_size,
        resized_vocab_size,
    )
    if token_row_source == "sparse_adapter" and any(
        value is None for value in vocab_provenance
    ):
        raise ValueError(
            "sparse Stage 2 metadata requires complete Stage 1 vocabulary provenance"
        )
    if (
        token_row_source == "sparse_adapter"
        and source_base_model_architecture is None
    ):
        raise ValueError(
            "sparse Stage 2 metadata requires source architecture provenance"
        )
    if token_row_source != "sparse_adapter" and any(
        value is not None for value in vocab_provenance
    ) and any(value is None for value in vocab_provenance):
        raise ValueError("Stage 2 vocabulary provenance must be provided all-or-none")
    runtime_architecture = _capture_base_model_architecture(model_config)
    architecture_vocab_size = int(
        runtime_architecture["decoder"]["vocab_size"]
    )
    resized_vocab_size = (
        architecture_vocab_size
        if resized_vocab_size is None
        else resized_vocab_size
    )
    tokenizer_base_vocab_size = (
        resized_vocab_size - 3
        if tokenizer_base_vocab_size is None
        else tokenizer_base_vocab_size
    )
    pretrained_model_vocab_size = (
        tokenizer_base_vocab_size
        if pretrained_model_vocab_size is None
        else pretrained_model_vocab_size
    )
    if token_row_source == "sparse_adapter":
        architecture = _normalize_base_model_architecture(
            source_base_model_architecture
        )
        architecture["decoder"]["vocab_size"] = resized_vocab_size
        expected_runtime = deepcopy(architecture)
        expected_runtime["decoder"]["tie_word_embeddings"] = False
        difference = _first_architecture_difference(
            expected_runtime, runtime_architecture
        )
        if difference is not None:
            path, expected, actual = difference
            raise ValueError(
                f"sparse runtime architecture mismatch at {path}: "
                f"source-derived={expected!r}, runtime={actual!r}"
            )
    else:
        if source_base_model_architecture is not None:
            raise ValueError(
                "source_base_model_architecture is only valid for sparse_adapter"
            )
        architecture = runtime_architecture
    configured_backend = normalize_stage2_rollout_backend(
        getattr(model_config, "stage2_rollout_backend", "no_cache")
    )
    resolved_backend = (
        configured_backend
        if rollout_backend is None
        else normalize_stage2_rollout_backend(rollout_backend)
    )
    if resolved_backend != configured_backend:
        raise ValueError("Stage 2 metadata rollout backend disagrees with model config")
    configured_cache_checkpointing = getattr(
        model_config, "stage2_cache_activation_checkpointing", False
    )
    resolved_cache_checkpointing = (
        configured_cache_checkpointing
        if stage2_cache_activation_checkpointing is None
        else stage2_cache_activation_checkpointing
    )
    if type(resolved_cache_checkpointing) is not bool:
        raise ValueError("stage2_cache_activation_checkpointing must be a boolean")
    if resolved_cache_checkpointing != configured_cache_checkpointing:
        raise ValueError(
            "Stage 2 metadata cache activation checkpointing disagrees with model config"
        )
    configured_execution_backend = getattr(
        model_config, "step_preference_execution_backend", None
    )
    if configured_execution_backend is not None:
        configured_execution_backend = normalize_step_preference_execution_backend(
            configured_execution_backend
        )
    resolved_execution_backend = (
        configured_execution_backend
        if execution_backend is None
        else normalize_step_preference_execution_backend(execution_backend)
    )
    if resolved_execution_backend != configured_execution_backend:
        raise ValueError(
            "Stage 2 metadata execution backend disagrees with model config"
        )
    return Stage2PeftMetadata.from_dict(
        {
            "schema_version": STAGE2_PEFT_SCHEMA_VERSION,
            "stage": "stage2",
            "tuning_method": tuning.method,
            "base_model_reference": str(base_model_reference),
            "base_model_architecture": architecture,
            "tokenizer_base_vocab_size": tokenizer_base_vocab_size,
            "pretrained_model_vocab_size": pretrained_model_vocab_size,
            "resized_vocab_size": resized_vocab_size,
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
            "rollout_backend": resolved_backend,
            "prefix_dropout_semantics": stage2_prefix_dropout_semantics(
                resolved_backend
            ),
            "stage2_cache_activation_checkpointing": resolved_cache_checkpointing,
            "execution_backend": resolved_execution_backend,
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
    required = [
        "prepare_stage2_base_inputs_embeds",
        "get_rope_index",
        "stage2_rollout_engine",
        "model",
        "lm_head",
    ]
    if normalize_stage2_rollout_backend(
        getattr(
            getattr(candidate, "config", None),
            "stage2_rollout_backend",
            "no_cache",
        )
    ) == "differentiable_cache":
        required.append("stage2_cache_rollout_engine")
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
    expected_architecture = deepcopy(metadata.base_model_architecture)
    if metadata.token_row_source == "sparse_adapter":
        expected_architecture["decoder"]["tie_word_embeddings"] = False
    architecture_difference = _first_architecture_difference(
        expected_architecture, current_architecture
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
    configured_targets = getattr(adapter_config, "target_modules", ()) or ()
    if isinstance(configured_targets, str) or not isinstance(
        configured_targets, (list, set, tuple)
    ):
        raise ValueError("adapter resolved targets do not match metadata")
    target_modules = tuple(configured_targets)
    if not target_modules or any(
        not isinstance(target, str) or not target for target in target_modules
    ):
        raise ValueError("adapter resolved targets do not match metadata")
    resolved_adapter_targets = {
        name
        for name in modules
        if any(
            name == target or name.endswith(f".{target}")
            for target in target_modules
        )
    }
    if resolved_adapter_targets != set(metadata.resolved_target_module_names):
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
    expected_vocab_size = metadata.resized_vocab_size
    ordered_ids = (
        token_ids.pad_token_id,
        token_ids.start_token_id,
        token_ids.end_token_id,
    )
    base_model.resize_token_embeddings(expected_vocab_size)
    validate_stage1_vocab_alignment(base_model, expected_vocab_size)
    initialize_frozen_output_token_rows(base_model, ordered_ids)
    configure_stage2_metadata(
        qwen.config,
        token_ids,
        latent_size=metadata.latent_size,
        supervise_latent_boundaries=metadata.supervise_latent_boundaries,
        rollout_backend=metadata.rollout_backend,
        stage2_cache_activation_checkpointing=(
            metadata.stage2_cache_activation_checkpointing
        ),
        execution_backend=metadata.execution_backend,
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
    expected_architecture["decoder"]["vocab_size"] = (
        metadata.pretrained_model_vocab_size
    )
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
    expected_vocab_size = metadata.resized_vocab_size
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
    expected_ids = tuple(
        range(metadata.tokenizer_base_vocab_size, expected_vocab_size)
    )
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
    rollout_backend: str = "no_cache",
    stage2_cache_activation_checkpointing: bool = False,
    execution_backend: str | None = None,
) -> None:
    """Write the complete, backwards-compatible Stage 2 config metadata."""
    config.stage = "stage2"
    config.latent_size = latent_size
    config.supervise_latent_boundaries = supervise_latent_boundaries
    config.stage2_rollout_backend = normalize_stage2_rollout_backend(rollout_backend)
    config.prefix_dropout_semantics = stage2_prefix_dropout_semantics(
        config.stage2_rollout_backend
    )
    if type(stage2_cache_activation_checkpointing) is not bool:
        raise ValueError("stage2_cache_activation_checkpointing must be a boolean")
    if (
        stage2_cache_activation_checkpointing
        and config.stage2_rollout_backend != "differentiable_cache"
    ):
        raise ValueError(
            "stage2_cache_activation_checkpointing requires differentiable_cache"
        )
    config.stage2_cache_activation_checkpointing = stage2_cache_activation_checkpointing
    config.step_preference_execution_backend = (
        None
        if execution_backend is None
        else normalize_step_preference_execution_backend(execution_backend)
    )
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
    rollout_backend: str = "no_cache"
    stage2_cache_activation_checkpointing: bool = False
    execution_backend: str | None = None

    @property
    def model_source(self) -> str:
        """Backward-compatible name used by the full-checkpoint loader."""
        return self.base_model_source


@dataclass(frozen=True)
class _ExistingStage2Settings:
    load_model_path: str
    tuning: Stage2TuningConfig
    tuning_method_explicit: bool
    rollout_backend: str
    rollout_backend_explicit: bool
    cache_activation_checkpointing: bool
    cache_activation_checkpointing_explicit: bool
    execution_backend: str | None
    latent_size: int
    supervise_latent_boundaries: bool


@dataclass(frozen=True)
class _ExistingStage2Dependencies:
    load_config: object
    discover_adapter_config: object
    source_provenance_kwargs: Mapping[str, object] | None
    load_peft_metadata: object
    require_full_stage1: object


def _resolve_existing_stage2_artifact(
    path: str,
    requested_settings: _ExistingStage2Settings,
    dependencies: _ExistingStage2Dependencies,
) -> Stage2LoadDecision:
    """Validate one existing Stage 2 artifact and resolve weight sources.

    Trainer state is deliberately not selected here. The caller decides whether
    the validated artifact is an interrupted-run resume or a weight-only warm
    start, ensuring both lifecycle paths share every artifact validation.
    """
    artifact_path = Path(path)
    lifecycle = _optional_step_preference_metadata(path)
    if (
        requested_settings.execution_backend is not None
        and (lifecycle is None or lifecycle.phase != "stage2b")
    ):
        raise ValueError(
            "Stage 2B resume requires current Stage 2B lifecycle metadata"
        )
    local_artifact_kind = (
        classify_stage2_artifact(artifact_path) if artifact_path.is_dir() else None
    )
    if local_artifact_kind is Stage2ArtifactKind.FULL:
        metadata = None
    else:
        try:
            metadata = dependencies.load_peft_metadata(path)
        except FileNotFoundError:
            metadata = None
        except ValueError as error:
            if not str(error).startswith("missing Stage 2 PEFT metadata"):
                raise
            metadata = None

    if metadata is None:
        if (artifact_path / "adapter_config.json").is_file():
            raise ValueError(
                "missing Stage 2 PEFT metadata: "
                f"{artifact_path / STAGE2_PEFT_METADATA_FILENAME}"
            )
        artifact_config = dependencies.load_config(path)
        if getattr(artifact_config, "stage", None) != "stage2":
            raise ValueError("expected stage2 checkpoint for resume")
        if requested_settings.tuning.is_peft:
            raise ValueError("requested PEFT resume is missing Stage 2 PEFT metadata")
        checkpoint_rollout_backend = normalize_stage2_rollout_backend(
            getattr(artifact_config, "stage2_rollout_backend", "no_cache")
        )
        checkpoint_dropout_semantics = getattr(
            artifact_config, "prefix_dropout_semantics", None
        )
        if (
            checkpoint_dropout_semantics is not None
            and checkpoint_dropout_semantics
            != stage2_prefix_dropout_semantics(checkpoint_rollout_backend)
        ):
            raise ValueError(
                "Stage 2 prefix dropout semantics do not match rollout backend"
            )
        if (
            requested_settings.rollout_backend_explicit
            and requested_settings.rollout_backend != checkpoint_rollout_backend
        ):
            raise ValueError("cannot resume Stage 2 with changed rollout_backend")
        checkpoint_cache_activation_checkpointing = getattr(
            artifact_config, "stage2_cache_activation_checkpointing", False
        )
        if type(checkpoint_cache_activation_checkpointing) is not bool:
            raise ValueError("stage2_cache_activation_checkpointing must be a boolean")
        if (
            checkpoint_cache_activation_checkpointing
            and checkpoint_rollout_backend != "differentiable_cache"
        ):
            raise ValueError(
                "stage2_cache_activation_checkpointing requires differentiable_cache"
            )
        if (
            requested_settings.cache_activation_checkpointing_explicit
            and requested_settings.cache_activation_checkpointing
            != checkpoint_cache_activation_checkpointing
        ):
            raise ValueError(
                "cannot resume Stage 2 with changed "
                "stage2_cache_activation_checkpointing"
            )
        checkpoint_execution_backend = getattr(
            artifact_config, "step_preference_execution_backend", None
        )
        if checkpoint_execution_backend is not None:
            checkpoint_execution_backend = (
                normalize_step_preference_execution_backend(
                    checkpoint_execution_backend
                )
            )
        if (
            lifecycle is not None
            and checkpoint_execution_backend != lifecycle.execution_backend
        ):
            raise ValueError(
                "Stage 2 config execution_backend disagrees with lifecycle metadata"
            )
        for field, requested in (
            ("latent_size", requested_settings.latent_size),
            (
                "supervise_latent_boundaries",
                requested_settings.supervise_latent_boundaries,
            ),
        ):
            checkpoint_value = getattr(artifact_config, field, None)
            if checkpoint_value != requested:
                raise ValueError(
                    f"cannot resume Stage 2 with changed {field}: "
                    f"checkpoint={checkpoint_value}, requested={requested}"
                )
        return Stage2LoadDecision(
            path,
            path,
            None,
            None,
            requested_settings.tuning,
            artifact_config.latent_size,
            artifact_config.supervise_latent_boundaries,
            rollout_backend=checkpoint_rollout_backend,
            stage2_cache_activation_checkpointing=(
                checkpoint_cache_activation_checkpointing
            ),
            execution_backend=checkpoint_execution_backend,
        )

    if (
        lifecycle is not None
        and metadata.execution_backend != lifecycle.execution_backend
    ):
        raise ValueError(
            "Stage 2 PEFT execution_backend disagrees with lifecycle metadata"
        )
    requested_method = requested_settings.tuning.method
    if requested_settings.tuning_method_explicit and requested_method == "full":
        raise ValueError("cannot request full tuning against a PEFT checkpoint")
    if (
        requested_settings.tuning_method_explicit
        and requested_method != metadata.tuning_method
    ):
        raise ValueError(
            "cannot switch Stage 2 PEFT method from "
            f"{metadata.tuning_method} to {requested_method}"
        )
    if (
        requested_settings.rollout_backend_explicit
        and requested_settings.rollout_backend != metadata.rollout_backend
    ):
        raise ValueError("cannot resume Stage 2 with changed rollout_backend")
    if (
        requested_settings.cache_activation_checkpointing_explicit
        and requested_settings.cache_activation_checkpointing
        != metadata.stage2_cache_activation_checkpointing
    ):
        raise ValueError(
            "cannot resume Stage 2 with changed "
            "stage2_cache_activation_checkpointing"
        )
    if requested_settings.tuning_method_explicit:
        for label, checkpoint_value, requested_value in (
            ("rank", metadata.lora_r, requested_settings.tuning.r),
            ("alpha", metadata.lora_alpha, requested_settings.tuning.alpha),
            ("dropout", metadata.lora_dropout, requested_settings.tuning.dropout),
            (
                "target_modules",
                metadata.requested_target_modules,
                requested_settings.tuning.target_modules,
            ),
        ):
            if checkpoint_value != requested_value:
                raise ValueError(
                    f"cannot resume Stage 2 with changed {label}: "
                    f"checkpoint={checkpoint_value}, requested={requested_value}"
                )
    for field, requested in (
        ("latent_size", requested_settings.latent_size),
        (
            "supervise_latent_boundaries",
            requested_settings.supervise_latent_boundaries,
        ),
    ):
        checkpoint_value = getattr(metadata, field)
        if checkpoint_value != requested:
            raise ValueError(
                f"cannot resume Stage 2 with changed {field}: "
                f"checkpoint={checkpoint_value}, requested={requested}"
            )
    if metadata.token_row_source == "sparse_adapter":
        source_provenance_kwargs = dependencies.source_provenance_kwargs or {}
        discovered_adapter_config = dependencies.discover_adapter_config(
            requested_settings.load_model_path, **source_provenance_kwargs
        )
        if discovered_adapter_config is not None:
            raise ValueError(
                "sparse Stage 2 resume requires an adapter-free original base "
                f"model; discovered adapter config at {discovered_adapter_config}"
            )
        validate_stage2_sparse_original_base_config(
            dependencies.load_config(
                requested_settings.load_model_path, **source_provenance_kwargs
            ),
            metadata,
        )
    else:
        dependencies.require_full_stage1(requested_settings.load_model_path)
    metadata_tuning = Stage2TuningConfig(
        method=metadata.tuning_method,
        r=metadata.lora_r,
        alpha=metadata.lora_alpha,
        dropout=metadata.lora_dropout,
        target_modules=metadata.requested_target_modules,
    )
    return Stage2LoadDecision(
        requested_settings.load_model_path,
        path,
        path,
        None,
        metadata_tuning,
        metadata.latent_size,
        metadata.supervise_latent_boundaries,
        metadata,
        rollout_backend=metadata.rollout_backend,
        stage2_cache_activation_checkpointing=(
            metadata.stage2_cache_activation_checkpointing
        ),
        execution_backend=getattr(metadata, "execution_backend", None),
    )


def resolve_stage2_load_decision(
    *,
    load_model_path: str,
    stage1_adapter_path: str | None = None,
    stage2a_checkpoint_path: str | None = None,
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
    rollout_backend: str | None = None,
    rollout_backend_explicit: bool = False,
    stage2_cache_activation_checkpointing: bool | None = None,
    stage2_cache_activation_checkpointing_explicit: bool = False,
    execution_backend: str | None = None,
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
    requested_rollout_backend = (
        "no_cache"
        if rollout_backend is None
        else normalize_stage2_rollout_backend(rollout_backend)
    )
    requested_cache_activation_checkpointing = (
        False
        if stage2_cache_activation_checkpointing is None
        else stage2_cache_activation_checkpointing
    )
    if type(requested_cache_activation_checkpointing) is not bool:
        raise ValueError("stage2_cache_activation_checkpointing must be a boolean")
    requested_execution_backend = (
        None
        if execution_backend is None
        else normalize_step_preference_execution_backend(execution_backend)
    )

    exclusive_sources = tuple(
        source is not None
        for source in (
            stage1_adapter_path,
            stage2a_checkpoint_path,
            resume_from_checkpoint_path,
        )
    )
    if sum(exclusive_sources) > 1:
        raise ValueError(
            "Stage 1 initialization, Stage 2A warm-start, and Stage 2 resume "
            "are mutually exclusive"
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
        if load_model_path != metadata.base_model_reference:
            raise ValueError(
                "Stage 1 PEFT base source mismatch: "
                f"metadata={metadata.base_model_reference!r}, "
                f"requested={load_model_path!r}"
            )
        if metadata.tuning_method != "dora":
            raise ValueError(
                "fresh Stage 2 continuation requires a Stage 1 DoRA adapter"
            )
        for label, stage1_value, stage2_value in (
            ("latent_size", metadata.latent_size, latent_size),
            ("rank", metadata.rank, requested_tuning.r),
            ("alpha", metadata.alpha, requested_tuning.alpha),
            ("dropout", metadata.dropout, requested_tuning.dropout),
        ):
            if stage1_value != stage2_value:
                raise ValueError(
                    f"Stage 1/Stage 2 {label} mismatch: "
                    f"stage1={stage1_value}, stage2={stage2_value}"
                )
        if set(metadata.requested_target_suffixes) != set(
            requested_tuning.target_modules
        ):
            raise ValueError(
                "Stage 1/Stage 2 requested target modules mismatch: "
                f"stage1={metadata.requested_target_suffixes}, "
                f"stage2={requested_tuning.target_modules}"
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
            rollout_backend=requested_rollout_backend,
            stage2_cache_activation_checkpointing=requested_cache_activation_checkpointing,
            execution_backend=requested_execution_backend,
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

    if stage2a_checkpoint_path is None and resume_from_checkpoint_path is None:
        require_full_stage1(load_model_path)
        return Stage2LoadDecision(
            load_model_path,
            load_model_path,
            None,
            None,
            requested_tuning,
            latent_size,
            supervise_latent_boundaries,
            rollout_backend=requested_rollout_backend,
            stage2_cache_activation_checkpointing=requested_cache_activation_checkpointing,
            execution_backend=requested_execution_backend,
        )

    artifact_path = stage2a_checkpoint_path or resume_from_checkpoint_path
    if artifact_path is None:  # pragma: no cover - guarded by the fresh branch
        raise AssertionError("existing Stage 2 artifact path was not selected")
    decision = _resolve_existing_stage2_artifact(
        artifact_path,
        _ExistingStage2Settings(
            load_model_path=load_model_path,
            tuning=requested_tuning,
            tuning_method_explicit=tuning_method_explicit,
            rollout_backend=requested_rollout_backend,
            rollout_backend_explicit=(
                rollout_backend_explicit and resume_from_checkpoint_path is not None
            ),
            cache_activation_checkpointing=requested_cache_activation_checkpointing,
            cache_activation_checkpointing_explicit=(
                stage2_cache_activation_checkpointing_explicit
                and resume_from_checkpoint_path is not None
            ),
            execution_backend=(
                requested_execution_backend
                if resume_from_checkpoint_path is not None
                else None
            ),
            latent_size=latent_size,
            supervise_latent_boundaries=supervise_latent_boundaries,
        ),
        _ExistingStage2Dependencies(
            load_config=load_config,
            discover_adapter_config=discover_adapter_config,
            source_provenance_kwargs=source_provenance_kwargs,
            load_peft_metadata=load_peft_metadata,
            require_full_stage1=require_full_stage1,
        ),
    )
    if resume_from_checkpoint_path is not None:
        return replace(decision, trainer_resume_path=resume_from_checkpoint_path)
    return replace(
        decision,
        rollout_backend=requested_rollout_backend,
        stage2_cache_activation_checkpointing=(
            requested_cache_activation_checkpointing
        ),
        execution_backend=(
            requested_execution_backend
            if requested_execution_backend is not None
            else DEFAULT_STEP_PREFERENCE_EXECUTION_BACKEND
        ),
    )
