"""Small metadata manifests for non-shared multi-node checkpoint trees."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.multinode import ClusterSpec


_MANIFEST_SCHEMA_VERSION = 1
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,127}$")
_SAFE_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_OPTIMIZER_SHARD = re.compile(
    r"(?:bf16_|fp16_)?zero_pp_rank_(0|[1-9][0-9]*)_mp_rank_00_optim_states\.pt\Z"
)
_PARTITIONED_MODEL_SHARD = re.compile(
    r"zero_pp_rank_(0|[1-9][0-9]*)_mp_rank_00_model_states\.pt\Z"
)
_REPLICATED_MODEL_STATE = "mp_rank_00_model_states.pt"
_RNG_STATE = re.compile(r"rng_state_(0|[1-9][0-9]*)\.pth\Z")
_MANIFEST_IDENTITY_FIELDS = (
    "stage",
    "tuning_method",
    "zero_stage",
    "deepspeed_config_digest",
    "world_size",
    "rank_mapping",
    "global_step",
    "model_state_layout",
    "optimizer",
    "scheduler",
    "rng_ranks",
    "stage_sidecar",
    "commit",
    "run_id",
)
_RESUME_IDENTITY_FIELDS = _MANIFEST_IDENTITY_FIELDS
_LOCAL_REPOSITORY = Path(__file__).resolve().parents[1]
_REMOTE_HOST = "korea@172.16.11.41"
_RSYNC_SSH = "/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=8"
_MAX_COMMON_METADATA_BYTES = 4 * 1024 * 1024
_MAX_LATEST_BYTES = 1024
INFERENCE_MANIFEST_SUFFIX = ".manifest.json"
_EVALUATION_REMOTE_FILES = (
    "shard-0.jsonl",
    "shard-0.jsonl.manifest.json",
    "shard-1.jsonl",
    "shard-1.jsonl.manifest.json",
)


def _prepared_inference_relative_path(
    path: str | Path, *, label: str, allow_resolved: bool = False
) -> Path:
    staging_root = _LOCAL_REPOSITORY / "checkpoints" / "evaluation-staging"
    candidate = Path(path)
    roots = [staging_root]
    if allow_resolved:
        roots.append(
            (_LOCAL_REPOSITORY / "checkpoints").resolve(strict=True)
            / "evaluation-staging"
        )
    relative = None
    for root in roots:
        try:
            relative = candidate.relative_to(root)
            break
        except ValueError:
            continue
    if relative is None:
        raise ValueError(
            f"{label} must be under the checkpoint evaluation-staging root"
        )
    validate_coordinator_relative_path(relative, label=label)
    if len(relative.parts) < 2:
        raise ValueError(f"{label} must belong to a run")
    _validate_run_id(relative.parts[0])
    return relative
_V5_TRAINING_FILES = frozenset({Path("stage1_v5.json"), Path("stage1_v5_reference.pt")})
_MATCHED_STANDARD_TRAINING_FILES = frozenset({Path('stage1_matched_execution.json'),
                                            Path('stage1_matched_resume.json')})
_CACHE_TRAINING_FILES = frozenset({Path('stage1_run_cache.json'), Path('stage1_cache_initial_teacher.pt')})

_INFERENCE_STATIC_NAMES = frozenset(
    {
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "ilvr_token_rows.safetensors",
        "stage1_peft_metadata.json",
        "stage2_peft_metadata.json",
    }
)
_INFERENCE_SHARD = re.compile(
    r"(?:model-[0-9]{5}-of-[0-9]{5}\.safetensors|"
    r"pytorch_model-[0-9]{5}-of-[0-9]{5}\.bin)\Z"
)
_FULL_SAFE_SHARD = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors\Z")
_FULL_BINARY_SHARD = re.compile(r"pytorch_model-(\d{5})-of-(\d{5})\.bin\Z")
_REQUIRED_LOADER_ASSETS = frozenset(
    {"config.json", "preprocessor_config.json", "tokenizer_config.json"}
)
_BASE_CHECKPOINT_METADATA = frozenset(
    {"latest", "trainer_state.json", "scheduler.pt", "deepspeed-runtime.json"}
)
_STAGE_METHOD_CHECKPOINT_METADATA = {
    ("stage1", "full"): _BASE_CHECKPOINT_METADATA | {"stage1_ema_teacher.pt"},
    ("stage1", "lora"): _BASE_CHECKPOINT_METADATA
    | {"stage1_ema_teacher.pt", "stage1_peft_metadata.json"},
    ("stage1", "dora"): _BASE_CHECKPOINT_METADATA
    | {"stage1_ema_teacher.pt", "stage1_peft_metadata.json"},
    ("stage2", "full"): _BASE_CHECKPOINT_METADATA | {"stage2-sampler.json"},
    ("stage2", "lora"): _BASE_CHECKPOINT_METADATA
    | {"stage2-sampler.json", "stage2_peft_metadata.json"},
    ("stage2", "dora"): _BASE_CHECKPOINT_METADATA
    | {"stage2-sampler.json", "stage2_peft_metadata.json"},
}
_COMMON_METADATA_ALLOWLIST = {
    (stage, method, zero_stage): frozenset(Path(name) for name in names)
    for (stage, method), names in _STAGE_METHOD_CHECKPOINT_METADATA.items()
    for zero_stage in (1, 2, 3)
}


def derive_model_state_layout(zero_stage: object) -> str:
    """Return the only supported model-state layout for a canonical ZeRO stage."""
    if type(zero_stage) is not int or zero_stage not in (1, 2, 3):
        raise ValueError("zero_stage must be exactly 1, 2, or 3")
    return "partitioned" if zero_stage == 3 else "replicated"


def validate_checkpoint_directory(
    checkpoint_directory: str | Path,
    *,
    checkpoint_root: str | Path,
    run_id: str,
    global_step: int,
) -> Path:
    """Bind the logical path and open every descendant below the fixed root nofollow."""
    descriptor, physical = _open_pinned_checkpoint_directory(
        checkpoint_directory,
        checkpoint_root=checkpoint_root,
        run_id=run_id,
        global_step=global_step,
    )
    os.close(descriptor)
    return physical


def _open_pinned_checkpoint_directory(
    checkpoint_directory: str | Path,
    *,
    checkpoint_root: str | Path,
    run_id: str,
    global_step: int,
) -> tuple[int, Path]:
    """Open one canonical checkpoint inode and retain its no-follow descriptor."""
    from src.multinode_runtime import _open_child_directory, _open_trusted_directory

    run = _validate_run_id(run_id)
    if type(global_step) is not int or global_step < 0:
        raise ValueError("checkpoint global step must be a nonnegative integer")
    logical_root = Path(checkpoint_root)
    logical = Path(checkpoint_directory)
    expected = logical_root / run / f"checkpoint-{global_step}"
    if not logical.is_absolute() or logical.absolute() != expected.absolute():
        raise ValueError("checkpoint directory must be the exact canonical path")
    physical_root = logical_root.resolve(strict=True)
    physical = physical_root / run / f"checkpoint-{global_step}"
    root_descriptor = _open_trusted_directory(physical_root)
    run_descriptor: int | None = None
    checkpoint_descriptor: int | None = None
    try:
        run_descriptor = _open_child_directory(
            root_descriptor, run, create=False
        )
        checkpoint_descriptor = _open_child_directory(
            run_descriptor, f"checkpoint-{global_step}", create=False
        )
    except OSError as error:
        raise ValueError(
            "checkpoint run and checkpoint descendants must be no-follow "
            "non-symlink directories"
        ) from error
    finally:
        if run_descriptor is not None:
            os.close(run_descriptor)
        os.close(root_descriptor)
    assert checkpoint_descriptor is not None
    return checkpoint_descriptor, physical


def _pinned_directory_descriptor(path: Path) -> int | None:
    candidate = Path(path)
    if candidate.parent == Path("/proc/self/fd") and candidate.name.isdecimal():
        return int(candidate.name)
    return None


def _pin_directory_for_inspection(path: Path, *, label: str) -> tuple[Path, int | None]:
    existing = _pinned_directory_descriptor(path)
    if existing is not None:
        try:
            metadata = os.fstat(existing)
        except OSError as error:
            raise ValueError(f"{label} descriptor is unavailable") from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} descriptor must identify a directory")
        return path, None

    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        metadata = os.fstat(descriptor)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ValueError(f"{label} must be a non-symlink directory: {path}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise ValueError(f"{label} must identify a directory: {path}")
    return Path(f"/proc/self/fd/{descriptor}"), descriptor


def _read_small_regular(path: Path, *, maximum_size: int, label: str) -> bytes:
    parent_descriptor = _pinned_directory_descriptor(path.parent)
    if parent_descriptor is None:
        raise ValueError(f"{label} must be read from a pinned directory: {path}")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"{label} must be a regular non-symlink file: {path}"
            )
        if metadata.st_size > maximum_size:
            raise ValueError(f"{label} exceeds its bounded size: {path}")
        chunks: list[bytes] = []
        remaining = maximum_size + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        final_metadata = os.fstat(descriptor)
        if (
            len(raw) > maximum_size
            or len(raw) != metadata.st_size
            or final_metadata.st_size != metadata.st_size
        ):
            raise ValueError(f"{label} changed during its bounded read: {path}")
        return raw
    except OSError as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _top_level_regular_names(artifact: Path) -> set[str]:
    descriptor = _pinned_directory_descriptor(artifact)
    if descriptor is None:
        raise ValueError("loader artifact inspection requires a pinned directory")
    try:
        with os.scandir(descriptor) as children:
            return {
                child.name
                for child in children
                if stat.S_ISREG(child.stat(follow_symlinks=False).st_mode)
            }
    except OSError as error:
        raise ValueError("cannot inspect pinned loader artifact") from error


def _decode_checkpoint_step(
    checkpoint: Path, *, pinned_step: int | None = None
) -> int:
    if pinned_step is None:
        match = re.fullmatch(r"checkpoint-(0|[1-9][0-9]*)", checkpoint.name)
        if match is None:
            raise ValueError("checkpoint directory must be numbered checkpoint-N")
        numbered_step = int(match.group(1))
    else:
        if type(pinned_step) is not int or pinned_step < 0:
            raise ValueError("pinned checkpoint step must be a nonnegative integer")
        numbered_step = pinned_step
    latest = _read_small_regular(
        checkpoint / "latest", maximum_size=_MAX_LATEST_BYTES, label="DeepSpeed latest"
    )
    try:
        latest_tag = latest.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("DeepSpeed latest must be UTF-8") from error
    if latest_tag != f"global_step{numbered_step}":
        raise ValueError("checkpoint-N and latest global_stepN disagree")
    trainer_raw = _read_small_regular(
        checkpoint / "trainer_state.json",
        maximum_size=_MAX_COMMON_METADATA_BYTES,
        label="trainer state",
    )
    try:
        trainer = json.loads(trainer_raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("trainer_state.json is invalid JSON") from error
    if type(trainer) is not dict or type(trainer.get("global_step")) is not int:
        raise ValueError("trainer_state.json global_step must be an exact integer")
    if trainer["global_step"] != numbered_step:
        raise ValueError("trainer_state.json and checkpoint-N steps disagree")
    return numbered_step


def _observed_model_layout(checkpoint: Path, step: int) -> str:
    tag_name = f"global_step{step}"
    tag = checkpoint / tag_name
    descriptor: int | None = None
    try:
        if checkpoint.parent == Path("/proc/self/fd") and checkpoint.name.isdecimal():
            descriptor = os.open(
                tag_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=int(checkpoint.name),
            )
            scan_target: int | Path = descriptor
        else:
            if tag.is_symlink() or not tag.is_dir():
                raise ValueError(
                    "checkpoint global_stepN must be a non-symlink directory"
                )
            scan_target = tag
        with os.scandir(scan_target) as children:
            names = {
                entry.name
                for entry in children
                if entry.is_file(follow_symlinks=False)
            }
    except OSError as error:
        raise ValueError(
            "checkpoint global_stepN must be a no-follow directory"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    has_partitioned = any(_PARTITIONED_MODEL_SHARD.fullmatch(name) for name in names)
    has_replicated = _REPLICATED_MODEL_STATE in names
    if has_partitioned == has_replicated:
        raise ValueError("checkpoint model-state filenames have no single layout")
    return "partitioned" if has_partitioned else "replicated"


def _checkpoint_stage_and_method(
    checkpoint: Path, expected: Mapping[str, object]
) -> tuple[str, str]:
    artifact = checkpoint
    artifact_names = _top_level_regular_names(artifact)
    _require_loader_asset_names(artifact_names)
    config_raw = _read_small_regular(
        artifact / "config.json",
        maximum_size=_MAX_COMMON_METADATA_BYTES,
        label="loader config",
    )
    try:
        config = json.loads(config_raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("loader config is invalid JSON") from error
    if type(config) is not dict:
        raise ValueError("loader config must be a JSON object")
    stage1_metadata = artifact / "stage1_peft_metadata.json"
    stage2_metadata = artifact / "stage2_peft_metadata.json"
    stage1_present = stage1_metadata.name in artifact_names
    stage2_present = stage2_metadata.name in artifact_names
    if stage1_present and stage2_present:
        raise ValueError("checkpoint has ambiguous Stage PEFT metadata")
    source_root = str(Path(__file__).resolve().parent)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    if stage1_present:
        metadata_json = _read_small_regular(
            stage1_metadata,
            maximum_size=_MAX_COMMON_METADATA_BYTES,
            label="Stage 1 PEFT metadata",
        )
        from stage1.checkpointing import (
            Stage1PeftMetadataSnapshot,
            decode_stage1_peft_metadata_snapshot,
        )

        metadata = decode_stage1_peft_metadata_snapshot(
            Stage1PeftMetadataSnapshot(
                artifact_dir=artifact,
                metadata_json=metadata_json,
            )
        )
        _require_peft_payload_names(artifact_names, stage="stage1")
        return "stage1", metadata.tuning_method
    if stage2_present:
        metadata_json = _read_small_regular(
            stage2_metadata,
            maximum_size=_MAX_COMMON_METADATA_BYTES,
            label="Stage 2 PEFT metadata",
        )
        from stage2.checkpointing import (
            Stage2PeftMetadataSnapshot,
            decode_stage2_peft_metadata_snapshot,
        )

        metadata = decode_stage2_peft_metadata_snapshot(
            Stage2PeftMetadataSnapshot(
                artifact_dir=artifact,
                metadata_json=metadata_json,
            )
        )
        _require_peft_payload_names(artifact_names, stage="stage2")
        has_sparse = "ilvr_token_rows.safetensors" in artifact_names
        if (metadata.token_row_source == "sparse_adapter") != has_sparse:
            raise ValueError(
                "Stage 2 PEFT sparse token-row marker differs from strict metadata"
            )
        return "stage2", metadata.tuning_method
    if expected["tuning_method"] != "full":
        raise ValueError("PEFT checkpoint is missing strict Stage metadata")
    stage1_full = artifact / "stage1_ema_teacher.pt"
    stage2_full = artifact / "stage2-sampler.json"
    stage1_full_present = stage1_full.name in artifact_names
    stage2_full_present = stage2_full.name in artifact_names
    if stage1_full_present == stage2_full_present:
        raise ValueError("full checkpoint must have exactly one Stage sidecar")
    stage = "stage1" if stage1_full_present else "stage2"
    if stage == "stage2":
        sampler_raw = _read_small_regular(
            stage2_full,
            maximum_size=_MAX_COMMON_METADATA_BYTES,
            label="Stage 2 sampler sidecar",
        )
        try:
            sampler_payload = json.loads(sampler_raw)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid Stage 2 sampler sidecar JSON") from error
        if (
            type(sampler_payload) is not dict
            or sampler_payload.get("global_step") != expected["global_step"]
        ):
            raise ValueError("Stage 2 sampler global_step differs from checkpoint")
        from stage2.sampler import Stage2SamplerIdentity

        sampler = Stage2SamplerIdentity.from_dict(
            {
                key: value
                for key, value in sampler_payload.items()
                if key != "global_step"
            }
        )
        if sampler.world_size != expected["world_size"]:
            raise ValueError("Stage 2 sampler world_size differs from checkpoint")
    _validate_full_serialization_metadata_only(artifact, artifact_names)
    return stage, "full"


def _require_loader_asset_names(names: set[str]) -> None:
    missing = _REQUIRED_LOADER_ASSETS - names
    if missing:
        raise ValueError(f"loader artifact is missing required asset: {sorted(missing)[0]}")
    has_tokenizer_json = "tokenizer.json" in names
    has_vocab_merges = {"vocab.json", "merges.txt"} <= names
    if not has_tokenizer_json and not has_vocab_merges:
        raise ValueError(
            "loader artifact requires tokenizer.json or vocab.json plus merges.txt"
        )


def _validate_full_shard_index_metadata_only(
    artifact: Path,
    *,
    index_name: str,
    shard_pattern: re.Pattern[str],
    actual_shards: set[str],
) -> None:
    raw = _read_small_regular(
        artifact / index_name,
        maximum_size=_MAX_COMMON_METADATA_BYTES,
        label="full-model shard index",
    )
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid full-model shard index JSON") from error
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
            raise ValueError(f"invalid full-model shard reference: {reference!r}")
        references.add(reference)
    if references != actual_shards:
        raise ValueError(
            "full-model shard set does not match index: "
            f"missing={sorted(references - actual_shards)}, "
            f"orphan={sorted(actual_shards - references)}"
        )
    positions: list[int] = []
    totals: set[int] = set()
    for shard in references:
        match = shard_pattern.fullmatch(shard)
        assert match is not None
        positions.append(int(match.group(1)))
        totals.add(int(match.group(2)))
    if (
        len(totals) != 1
        or next(iter(totals)) != len(references)
        or sorted(positions) != list(range(1, len(references) + 1))
    ):
        raise ValueError("full-model shard numbering is inconsistent")


def _validate_full_serialization_metadata_only(
    artifact: Path, names: set[str]
) -> None:
    """Mirror the real Full classifier without opening any weight payload."""
    safe_single = "model.safetensors" in names
    binary_single = "pytorch_model.bin" in names
    safe_index = "model.safetensors.index.json" in names
    binary_index = "pytorch_model.bin.index.json" in names
    safe_shards = {name for name in names if _FULL_SAFE_SHARD.fullmatch(name)}
    binary_shards = {name for name in names if _FULL_BINARY_SHARD.fullmatch(name)}
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
        _validate_full_shard_index_metadata_only(
            artifact,
            index_name="model.safetensors.index.json",
            shard_pattern=_FULL_SAFE_SHARD,
            actual_shards=safe_shards,
        )
        return
    if binary_index:
        _validate_full_shard_index_metadata_only(
            artifact,
            index_name="pytorch_model.bin.index.json",
            shard_pattern=_FULL_BINARY_SHARD,
            actual_shards=binary_shards,
        )
        return
    raise ValueError("full-model shards require their matching parsed index")


def _require_peft_payload_names(names: set[str], *, stage: str) -> None:
    metadata_name = f"{stage}_peft_metadata.json"
    required = {"adapter_config.json", "adapter_model.safetensors", metadata_name}
    if stage == "stage1":
        required.add("ilvr_token_rows.safetensors")
    missing = required - names
    if missing:
        raise ValueError(f"incomplete {stage} PEFT artifact: missing={sorted(missing)}")
    full_markers = {
        name
        for name in names
        if name
        in {
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        }
        or _INFERENCE_SHARD.fullmatch(name)
    }
    if full_markers:
        raise ValueError(
            f"mixed {stage} PEFT and full serialization markers: "
            f"{sorted(full_markers)}"
        )
    allowed_weight_names = {
        "adapter_model.safetensors",
        "ilvr_token_rows.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rng_state.pth",
        "training_args.bin",
    }
    if stage == "stage1":
        allowed_weight_names.update({"stage1_ema_teacher.pt", "stage1_v5_reference.pt", "stage1_cache_initial_teacher.pt"})
    unsupported = {
        name
        for name in names
        if (
            name.endswith((".safetensors", ".bin", ".pt", ".pth"))
            or name.endswith((".safetensors.index.json", ".bin.index.json"))
        )
        and name not in allowed_weight_names
        and _RNG_STATE.fullmatch(name) is None
    }
    if unsupported:
        raise ValueError(
            f"unsupported legacy or unknown {stage} PEFT weight marker: "
            f"{sorted(unsupported)}"
        )


def validate_loader_artifact(
    checkpoint_directory: str | Path,
    runtime_identity: Mapping[str, object],
) -> None:
    """Require the classifier's metadata/marker contract without reading weights."""
    identity = _validate_identity(runtime_identity)
    checkpoint, owned_descriptor = _pin_directory_for_inspection(
        Path(checkpoint_directory), label="loader artifact"
    )
    try:
        stage, method = _checkpoint_stage_and_method(checkpoint, identity)
        if (stage, method) != (identity["stage"], identity["tuning_method"]):
            raise ValueError(
                "loader artifact stage/method differs from runtime identity: "
                f"observed={(stage, method)!r}, "
                f"expected={(identity['stage'], identity['tuning_method'])!r}"
            )
    finally:
        if owned_descriptor is not None:
            os.close(owned_descriptor)


def observe_checkpoint_identity(
    checkpoint_directory: str | Path,
    trusted_identity: Mapping[str, object],
) -> dict[str, object]:
    """Observe bounded producer sidecars and compare them with trusted plan fields."""
    from src.deepspeed_runtime import (
        DEEPSPEED_RUNTIME_FILENAME,
        _MAX_DEEPSPEED_RUNTIME_BYTES,
        decode_deepspeed_runtime_sidecar,
    )

    expected = _validate_identity(trusted_identity)
    checkpoint, owned_descriptor = _pin_directory_for_inspection(
        Path(checkpoint_directory), label="checkpoint artifact"
    )
    try:
        step = _decode_checkpoint_step(
            checkpoint, pinned_step=int(expected["global_step"])
        )
        runtime_path = checkpoint / DEEPSPEED_RUNTIME_FILENAME
        runtime_raw = _read_small_regular(
            runtime_path,
            maximum_size=_MAX_DEEPSPEED_RUNTIME_BYTES,
            label="DeepSpeed runtime identity",
        )
        profile = decode_deepspeed_runtime_sidecar(runtime_raw, path=runtime_path)
        if not profile.enabled:
            raise ValueError("checkpoint DeepSpeed runtime must be enabled")
        observed = {
            "zero_stage": profile.zero_stage,
            "deepspeed_config_digest": profile.config_digest,
            "world_size": profile.world_size,
            "global_step": step,
            "model_state_layout": _observed_model_layout(checkpoint, step),
        }
        stage, method = _checkpoint_stage_and_method(checkpoint, expected)
        observed.update({"stage": stage, "tuning_method": method})
        for field, value in observed.items():
            if expected[field] != value:
                raise ValueError(
                    f"observed checkpoint identity mismatch for {field}: "
                    f"observed={value!r}, expected={expected[field]!r}"
                )
        derived = derive_model_state_layout(expected["zero_stage"])
        if observed["model_state_layout"] != derived:
            raise ValueError("observed model-state filenames disagree with ZeRO stage")
        return {**expected, **observed}
    finally:
        if owned_descriptor is not None:
            os.close(owned_descriptor)


def _process_rank(path: Path) -> int | None:
    for pattern in (_OPTIMIZER_SHARD, _PARTITIONED_MODEL_SHARD, _RNG_STATE):
        match = pattern.fullmatch(path.name)
        if match is not None:
            return int(match.group(1))
    return None


def _safe_relative_path(path: Path) -> Path:
    return validate_coordinator_relative_path(path, label="artifact")


def validate_coordinator_relative_path(
    path: str | Path, *, label: str
) -> Path:
    """Restrict coordinator-owned remote path components to one safe grammar."""
    candidate = Path(path)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(
            part in ("", ".", "..")
            or _SAFE_COMPONENT_PATTERN.fullmatch(part) is None
            for part in candidate.parts
        )
    ):
        raise ValueError(
            f"{label} relative path must use canonical safe components: {candidate}"
        )
    return candidate


def _validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("artifact run ID is invalid")
    return run_id


@dataclass(frozen=True)
class ArtifactEntry:
    relative_path: Path
    byte_size: int
    node_rank: int

    @property
    def process_rank(self) -> int | None:
        """Return ownership encoded by a canonical DeepSpeed shard filename."""
        return _process_rank(self.relative_path)

    def to_json(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path.as_posix(),
            "byte_size": self.byte_size,
            "node_rank": self.node_rank,
            "process_rank": self.process_rank,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "ArtifactEntry":
        expected = {"relative_path", "byte_size", "node_rank", "process_rank"}
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("invalid artifact entry fields")
        if type(payload["relative_path"]) is not str:
            raise ValueError("artifact relative path must be a string")
        if type(payload["byte_size"]) is not int or payload["byte_size"] < 0:
            raise ValueError("artifact byte size must be a nonnegative integer")
        if type(payload["node_rank"]) is not int or payload["node_rank"] not in (0, 1):
            raise ValueError("artifact entry node rank is invalid")
        process_rank = payload["process_rank"]
        if process_rank is not None and (
            type(process_rank) is not int or process_rank not in range(4)
        ):
            raise ValueError("artifact process rank is invalid")
        entry = cls(
            relative_path=_safe_relative_path(Path(str(payload["relative_path"]))),
            byte_size=payload["byte_size"],
            node_rank=payload["node_rank"],
        )
        if payload["process_rank"] != entry.process_rank:
            raise ValueError("artifact process-rank ownership mismatch")
        return entry


@dataclass(frozen=True)
class NodeArtifactManifest:
    schema_version: int
    run_id: str
    node_rank: int
    process_ranks: tuple[int, ...]
    entries: tuple[ArtifactEntry, ...]
    runtime_identity: Mapping[str, object]
    completion_marker: bool = True
    checkpoint_relative: Path = Path("checkpoint-2")

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "node_rank": self.node_rank,
            "process_ranks": list(self.process_ranks),
            "entries": [entry.to_json() for entry in self.entries],
            "runtime_identity": _json_value(self.runtime_identity),
            "completion_marker": self.completion_marker,
            "checkpoint_relative": self.checkpoint_relative.as_posix(),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "NodeArtifactManifest":
        expected = {
            "schema_version",
            "run_id",
            "node_rank",
            "process_ranks",
            "entries",
            "runtime_identity",
            "completion_marker",
            "checkpoint_relative",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("invalid node artifact manifest fields")
        ranks = payload["process_ranks"]
        entries = payload["entries"]
        runtime_identity = payload["runtime_identity"]
        if type(ranks) is not list or type(entries) is not list:
            raise ValueError("invalid node artifact manifest sequence fields")
        if type(runtime_identity) is not dict:
            raise ValueError("invalid node artifact runtime identity")
        if type(payload["schema_version"]) is not int:
            raise ValueError("invalid node artifact schema version")
        if type(payload["run_id"]) is not str:
            raise ValueError("invalid node artifact run ID")
        if type(payload["node_rank"]) is not int:
            raise ValueError("invalid node artifact node rank")
        if any(type(rank) is not int for rank in ranks):
            raise ValueError("invalid node artifact process ranks")
        if type(payload["completion_marker"]) is not bool:
            raise ValueError("invalid node artifact completion marker")
        if type(payload["checkpoint_relative"]) is not str:
            raise ValueError("invalid node artifact checkpoint path")
        return cls(
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            run_id=payload["run_id"],  # type: ignore[arg-type]
            node_rank=payload["node_rank"],  # type: ignore[arg-type]
            process_ranks=tuple(ranks),  # type: ignore[arg-type]
            entries=tuple(ArtifactEntry.from_json(entry) for entry in entries),
            runtime_identity=runtime_identity,
            completion_marker=payload["completion_marker"],  # type: ignore[arg-type]
            checkpoint_relative=_safe_relative_path(
                Path(str(payload["checkpoint_relative"]))
            ),
        )


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _walk_regular_files(root: Path) -> tuple[tuple[Path, os.stat_result], ...]:
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise ValueError(f"artifact directory is unavailable: {root}") from error
    if stat.S_ISLNK(root_metadata.st_mode):
        raise ValueError("artifact directory must not be a symlink")
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("artifact root must be a directory")

    found: list[tuple[Path, os.stat_result]] = []

    def visit(directory: Path, relative_directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise ValueError(f"cannot inspect artifact directory: {directory}") from error
        for child in children:
            relative = relative_directory / child.name
            metadata = child.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"artifact tree contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                visit(Path(child.path), relative)
            elif stat.S_ISREG(metadata.st_mode):
                found.append((_safe_relative_path(relative), metadata))
            else:
                raise ValueError(
                    f"artifact tree entries must be regular files: {relative}"
                )

    visit(root, Path())
    return tuple(found)


def _walk_regular_files_fd(
    root_descriptor: int,
) -> tuple[tuple[Path, os.stat_result], ...]:
    """Enumerate one pinned checkpoint using only no-follow dirfd operations."""
    found: list[tuple[Path, os.stat_result]] = []

    def visit(directory_descriptor: int, relative_directory: Path) -> None:
        try:
            children = sorted(
                os.scandir(directory_descriptor), key=lambda item: item.name
            )
        except OSError as error:
            raise ValueError("cannot inspect pinned artifact directory") from error
        for child in children:
            relative = relative_directory / child.name
            metadata = child.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"artifact tree contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                child_descriptor: int | None = None
                try:
                    child_descriptor = os.open(
                        child.name,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                        dir_fd=directory_descriptor,
                    )
                    visit(child_descriptor, relative)
                except OSError as error:
                    raise ValueError(
                        f"cannot inspect pinned artifact directory: {relative}"
                    ) from error
                finally:
                    if child_descriptor is not None:
                        os.close(child_descriptor)
            elif stat.S_ISREG(metadata.st_mode):
                found.append((_safe_relative_path(relative), metadata))
            else:
                raise ValueError(
                    f"artifact tree entries must be regular files: {relative}"
                )

    visit(root_descriptor, Path())
    return tuple(found)


def build_node_manifest(
    run_directory: str | Path,
    *,
    node_rank: int,
    process_ranks: Sequence[int],
    run_id: str | None = None,
    runtime_identity: Mapping[str, object] | None = None,
    completion_marker: bool = True,
    checkpoint_root: str | Path = _LOCAL_REPOSITORY / "checkpoints",
    canonical_owner: bool = False,
) -> NodeArtifactManifest:
    """Inspect only lstat metadata and describe one completed node-local tree."""
    directory = Path(run_directory)
    if type(node_rank) is not int or node_rank not in (0, 1):
        raise ValueError("node rank must be 0 or 1")
    trusted_identity = _validate_identity(dict(runtime_identity or {}))
    ranks = tuple(process_ranks)
    expected_ranks = (
        (0, 1, 2, 3)
        if canonical_owner
        else ((0, 1) if node_rank == 0 else (2, 3))
    )
    if ranks != expected_ranks:
        raise ValueError("node manifest must claim only its assigned process ranks")
    checkpoint_relative = Path(f"checkpoint-{trusted_identity['global_step']}")
    checkpoint_descriptor, _ = _open_pinned_checkpoint_directory(
        directory,
        checkpoint_root=checkpoint_root,
        run_id=str(trusted_identity["run_id"]),
        global_step=int(trusted_identity["global_step"]),
    )
    try:
        pinned_directory = Path(f"/proc/self/fd/{checkpoint_descriptor}")
        entries: list[ArtifactEntry] = []
        for relative, metadata in _walk_regular_files_fd(checkpoint_descriptor):
            item = ArtifactEntry(relative, metadata.st_size, node_rank)
            if item.process_rank is not None and item.process_rank not in ranks:
                raise ValueError(
                    "artifact shard is outside the node's assigned process ranks: "
                    f"{relative}"
                )
            entries.append(item)
        identity = (
            observe_checkpoint_identity(pinned_directory, trusted_identity)
            if node_rank == 0 or canonical_owner
            else trusted_identity
        )
    finally:
        os.close(checkpoint_descriptor)
    manifest_run_id = _validate_run_id(run_id or str(identity["run_id"]))
    if manifest_run_id != identity["run_id"]:
        raise ValueError("node manifest run ID differs from its canonical ancestor")
    return NodeArtifactManifest(
        schema_version=_MANIFEST_SCHEMA_VERSION,
        run_id=manifest_run_id,
        node_rank=node_rank,
        process_ranks=ranks,
        entries=tuple(entries),
        runtime_identity=identity,
        completion_marker=completion_marker,
        checkpoint_relative=checkpoint_relative,
    )


def _validate_identity(
    identity: Mapping[str, object],
    *,
    required_fields: Sequence[str] = _MANIFEST_IDENTITY_FIELDS,
) -> dict[str, object]:
    if type(identity) is not dict:
        raise ValueError("runtime identity must be an exact JSON object")
    missing = [field for field in required_fields if field not in identity]
    if missing:
        raise ValueError(f"runtime identity is missing fields: {', '.join(missing)}")
    if set(identity) != set(required_fields):
        extra = sorted(set(identity) - set(required_fields))
        if extra:
            raise ValueError(f"runtime identity has unexpected fields: {', '.join(extra)}")
    raw_rank_mapping = identity.get("rank_mapping")
    if (
        type(raw_rank_mapping) is not dict
        or set(raw_rank_mapping) != {"0", "1"}
        or any(type(key) is not str for key in raw_rank_mapping)
        or any(type(value) is not list for value in raw_rank_mapping.values())
        or any(
            type(rank) is not int
            for value in raw_rank_mapping.values()
            for rank in value
        )
        or raw_rank_mapping != {"0": [0, 1], "1": [2, 3]}
    ):
        raise ValueError("runtime identity rank_mapping is invalid")
    normalized = _json_value(identity)
    assert isinstance(normalized, dict)
    if (
        normalized["stage"] not in ("stage1", "stage2")
        or type(normalized["stage"]) is not str
    ):
        raise ValueError("runtime identity stage is invalid")
    if (
        normalized["tuning_method"] not in ("full", "lora", "dora")
        or type(normalized["tuning_method"]) is not str
    ):
        raise ValueError("runtime identity tuning_method is invalid")
    zero_stage = normalized["zero_stage"]
    if type(zero_stage) is not int or zero_stage not in (1, 2, 3):
        raise ValueError("runtime identity zero_stage is invalid")
    world_size = normalized["world_size"]
    if type(world_size) is not int or world_size != 4:
        raise ValueError("runtime identity world_size is invalid")
    global_step = normalized["global_step"]
    if type(global_step) is not int or global_step < 0:
        raise ValueError("runtime identity global_step is invalid")
    digest = normalized["deepspeed_config_digest"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("runtime identity deepspeed_config_digest is invalid")
    commit = normalized["commit"]
    if (
        type(commit) is not str
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("runtime identity commit is invalid")
    _validate_run_id(normalized["run_id"])
    layout = normalized["model_state_layout"]
    if type(layout) is not str or layout not in ("replicated", "partitioned"):
        raise ValueError("runtime identity model_state_layout is invalid")
    if layout != derive_model_state_layout(zero_stage):
        raise ValueError("runtime identity model_state_layout conflicts with zero_stage")
    if normalized["optimizer"] is not True:
        raise ValueError("runtime identity optimizer must be true")
    if normalized["scheduler"] is not True:
        raise ValueError("runtime identity scheduler must be true")
    if (
        type(normalized["rng_ranks"]) is not list
        or any(type(rank) is not int for rank in normalized["rng_ranks"])
        or normalized["rng_ranks"] != [0, 1, 2, 3]
    ):
        raise ValueError("runtime identity rng_ranks is invalid")
    expected_sidecar = _primary_stage_sidecar(normalized)
    if (
        type(normalized["stage_sidecar"]) is not str
        or normalized["stage_sidecar"] != expected_sidecar
    ):
        raise ValueError("runtime identity stage_sidecar is invalid")
    return normalized


def validate_runtime_identity(identity: Mapping[str, object]) -> dict[str, object]:
    """Return the exact, type-strict checkpoint runtime identity."""
    return _validate_identity(identity)


def _primary_stage_sidecar(identity: Mapping[str, object]) -> str:
    stage = identity["stage"]
    method = identity["tuning_method"]
    if stage == "stage1":
        return (
            "stage1_peft_metadata.json"
            if method in ("lora", "dora")
            else "stage1_ema_teacher.pt"
        )
    return (
        "stage2_peft_metadata.json"
        if method in ("lora", "dora")
        else "stage2-sampler.json"
    )


def _required_common_paths(identity: Mapping[str, object]) -> frozenset[Path]:
    key = (identity["stage"], identity["tuning_method"], identity["zero_stage"])
    try:
        return _COMMON_METADATA_ALLOWLIST[key]
    except KeyError as error:
        raise ValueError("checkpoint metadata identity has no explicit allowlist") from error


def select_common_metadata_entries(
    manifest: NodeArtifactManifest,
) -> tuple[ArtifactEntry, ...]:
    """Select only explicitly approved metadata for this stage/method contract."""
    allowed = _required_common_paths(_validate_identity(manifest.runtime_identity))
    if manifest.runtime_identity["stage"] == "stage1":
        allowed = allowed | _V5_TRAINING_FILES | _MATCHED_STANDARD_TRAINING_FILES | _CACHE_TRAINING_FILES
    return tuple(entry for entry in manifest.entries if entry.relative_path in allowed)


def _validate_manifest(manifest: NodeArtifactManifest, *, world_size: int) -> None:
    if not isinstance(manifest, NodeArtifactManifest):
        raise TypeError("collection members must be NodeArtifactManifest values")
    if (
        type(manifest.schema_version) is not int
        or manifest.schema_version != _MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("artifact manifest schema mismatch")
    _validate_run_id(manifest.run_id)
    if type(manifest.node_rank) is not int or manifest.node_rank not in (0, 1):
        raise ValueError("artifact manifest node rank is invalid")
    expected = (0, 1) if manifest.node_rank == 0 else (2, 3)
    canonical_expected = (0, 1, 2, 3)
    if manifest.process_ranks not in (expected, canonical_expected):
        raise ValueError("node manifest process ranks do not match assigned process ranks")
    if any(type(rank) is not int for rank in manifest.process_ranks):
        raise ValueError("node manifest process ranks must be exact integers")
    if type(manifest.completion_marker) is not bool or manifest.completion_marker is not True:
        raise ValueError("node artifact completion marker is missing")
    identity = _validate_identity(manifest.runtime_identity)
    if identity["run_id"] != manifest.run_id:
        raise ValueError("runtime identity run ID differs from manifest run ID")
    if identity["world_size"] != world_size:
        raise ValueError("runtime identity world_size mismatch")
    expected_checkpoint = Path(f"checkpoint-{identity['global_step']}")
    if manifest.checkpoint_relative != expected_checkpoint:
        raise ValueError("checkpoint-N does not match runtime identity global_step")
    seen: set[Path] = set()
    for entry in manifest.entries:
        relative = _safe_relative_path(entry.relative_path)
        if relative in seen:
            raise ValueError("node manifest contains an ambiguous duplicate path")
        seen.add(relative)
        if type(entry.byte_size) is not int or entry.byte_size < 0:
            raise ValueError("artifact byte size must be nonnegative")
        if type(entry.node_rank) is not int or entry.node_rank != manifest.node_rank:
            raise ValueError("artifact entry node-rank ownership mismatch")
        if entry.process_rank is not None and entry.process_rank not in manifest.process_ranks:
            raise ValueError("artifact entry is outside assigned process ranks")


def validate_collection(
    manifests: Sequence[NodeArtifactManifest], *, world_size: int
) -> tuple[ArtifactEntry, ...]:
    """Validate two node manifests and return one unambiguous combined file set."""
    if type(world_size) is not int or world_size != 4:
        raise ValueError("artifact collection requires world_size 4")
    values = tuple(manifests)
    if not values:
        raise ValueError("artifact collection requires ranks 0 through 3")
    for manifest in values:
        _validate_manifest(manifest, world_size=world_size)
    run_ids = {manifest.run_id for manifest in values}
    if len(run_ids) != 1:
        raise ValueError("artifact manifests have different run IDs")
    identities = [_validate_identity(manifest.runtime_identity) for manifest in values]
    if any(candidate != identities[0] for candidate in identities[1:]):
        raise ValueError("artifact manifests have different runtime identity")
    node_ranks = [manifest.node_rank for manifest in values]
    if sorted(node_ranks) != [0, 1]:
        raise ValueError(
            "artifact collection requires one manifest from each node and ranks 0 through 3"
        )

    combined: dict[Path, ArtifactEntry] = {}
    identity = identities[0]
    allowed_common = _required_common_paths(identity)
    if identity["stage"] == "stage1" and any(
            entry.relative_path in _V5_TRAINING_FILES for manifest in values for entry in manifest.entries):
        allowed_common = allowed_common | _V5_TRAINING_FILES
    if identity['stage'] == 'stage1':
        # Cache-off and V5 checkpoints need no separate standard initial state.
        allowed_common = allowed_common | frozenset(entry.relative_path for manifest in values
            for entry in manifest.entries if entry.relative_path in _CACHE_TRAINING_FILES)
    step_directory = f"global_step{identity['global_step']}"
    for manifest in sorted(values, key=lambda item: item.node_rank):
        for entry in manifest.entries:
            path = entry.relative_path
            if (
                path.parts
                and path.parts[0].startswith("global_step")
                and path.parts[0] != step_directory
            ):
                raise ValueError("artifact global_step directory mismatches runtime identity")
            previous = combined.get(entry.relative_path)
            if previous is None:
                combined[entry.relative_path] = entry
                continue
            if previous.byte_size != entry.byte_size:
                raise ValueError(
                    f"duplicate artifact path has different sizes: {entry.relative_path}"
                )
            duplicate_allowed = (
                entry.relative_path in allowed_common
                or (
                    identity["model_state_layout"] == "replicated"
                    and entry.relative_path
                    == Path(step_directory) / _REPLICATED_MODEL_STATE
                )
            )
            oversized_common = (
                entry.relative_path in allowed_common
                and entry.byte_size > _MAX_COMMON_METADATA_BYTES
            )
            if not duplicate_allowed or oversized_common:
                raise ValueError(
                    "ambiguous duplicate artifact is not allowlisted common metadata: "
                    f"{entry.relative_path}"
                )
            # Equal-sized common metadata under an identical runtime identity is
            # represented once, retaining the canonical owner's node-0 entry.
    paths = set(combined)
    missing_common = sorted(allowed_common - paths, key=lambda path: path.as_posix())
    if missing_common:
        raise ValueError(f"required checkpoint sidecar is missing: {missing_common[0]}")
    optimizer_entries = [
        entry
        for entry in combined.values()
        if _OPTIMIZER_SHARD.fullmatch(entry.relative_path.name)
    ]
    optimizer_ranks = {
        entry.process_rank
        for entry in optimizer_entries
    }
    if len(optimizer_entries) != world_size or optimizer_ranks != set(range(world_size)):
        raise ValueError("checkpoint requires optimizer ranks 0 through 3")
    for entry in combined.values():
        if (
            _OPTIMIZER_SHARD.fullmatch(entry.relative_path.name)
            or _PARTITIONED_MODEL_SHARD.fullmatch(entry.relative_path.name)
        ) and entry.relative_path.parent != Path(step_directory):
            raise ValueError("checkpoint shard global_step directory mismatches runtime identity")
    rng_entries = [
        entry
        for entry in combined.values()
        if _RNG_STATE.fullmatch(entry.relative_path.name)
    ]
    rng_ranks = {
        entry.process_rank
        for entry in rng_entries
    }
    if (
        len(rng_entries) != world_size
        or rng_ranks != set(range(world_size))
        or any(entry.relative_path.parent != Path() for entry in rng_entries)
    ):
        raise ValueError("checkpoint requires RNG ranks 0 through 3")
    if identity["model_state_layout"] == "partitioned":
        model_entries = [
            entry
            for entry in combined.values()
            if _PARTITIONED_MODEL_SHARD.fullmatch(entry.relative_path.name)
        ]
        model_ranks = {
            entry.process_rank
            for entry in model_entries
        }
        if len(model_entries) != world_size or model_ranks != set(range(world_size)):
            raise ValueError("partitioned checkpoint requires model-state ranks 0 through 3")
        if Path(step_directory) / _REPLICATED_MODEL_STATE in paths:
            raise ValueError("partitioned checkpoint rejects replicated model state")
    else:
        if Path(step_directory) / _REPLICATED_MODEL_STATE not in paths:
            raise ValueError("replicated checkpoint requires common model state")
        if any(
            _PARTITIONED_MODEL_SHARD.fullmatch(entry.relative_path.name)
            for entry in combined.values()
        ):
            raise ValueError("replicated checkpoint rejects partitioned model states")
    for path in paths:
        if path.name in {"trainer_state.json", "scheduler.pt", "deepspeed-runtime.json", "latest"}:
            continue
        if path in allowed_common:
            continue
        if (
            _OPTIMIZER_SHARD.fullmatch(path.name)
            or _RNG_STATE.fullmatch(path.name)
            or _PARTITIONED_MODEL_SHARD.fullmatch(path.name)
            or path == Path(step_directory) / _REPLICATED_MODEL_STATE
        ):
            continue
        occurrences = sum(
            path == entry.relative_path
            for manifest in values
            for entry in manifest.entries
        )
        if occurrences > 1:
            raise ValueError(f"duplicate artifact is not allowlisted common metadata: {path}")
    collected = tuple(
        combined[path]
        for path in sorted(combined, key=lambda item: item.as_posix())
    )
    canonical = replace(
        values[0],
        node_rank=0,
        process_ranks=(0, 1, 2, 3),
        entries=tuple(replace(entry, node_rank=0) for entry in collected),
    )
    select_inference_entries(canonical)
    return collected


def _lstat_if_present(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _reject_symlink_ancestors(path: Path) -> None:
    candidate = path if path.is_absolute() else path.absolute()
    for ancestor in reversed(candidate.parents):
        metadata = _lstat_if_present(ancestor)
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"destination ancestor is a symlink: {ancestor}")


def validate_transfer_destination(
    destinations: Sequence[str | Path],
    *,
    validated_identical_common_metadata: Mapping[str | Path, int] | None = None,
) -> None:
    """Prove no destination will be overwritten before invoking rsync."""
    allowed = {
        Path(path): size
        for path, size in (validated_identical_common_metadata or {}).items()
    }
    for value in destinations:
        destination = Path(value)
        _reject_symlink_ancestors(destination)
        metadata = _lstat_if_present(destination)
        if metadata is None:
            continue
        expected_size = allowed.get(destination)
        if (
            expected_size is not None
            and expected_size <= _MAX_COMMON_METADATA_BYTES
            and destination.name
            in {
                "latest",
                "trainer_state.json",
                "scheduler.pt",
                "deepspeed-runtime.json",
                "stage1_peft_metadata.json",
                "stage1_v5.json",
                "stage1_v5_reference.pt",
                "stage1_matched_execution.json",
                "stage1_matched_resume.json",
                "stage1_run_cache.json",
                "stage1_cache_initial_teacher.pt",
                "stage2_peft_metadata.json",
                "stage2-sampler.json",
            }
            and stat.S_ISREG(metadata.st_mode)
            and metadata.st_size == expected_size
        ):
            continue
        raise FileExistsError(f"refusing to overwrite transfer destination: {destination}")


def _files_from_path(manifest: NodeArtifactManifest, label: str) -> Path:
    return (
        _LOCAL_REPOSITORY
        / "logs"
        / "multinode"
        / manifest.run_id
        / f"artifact-{label}-rank{manifest.node_rank}.files"
    )


def _rsync_prefix(files_from: Path) -> tuple[str, ...]:
    return (
        "/usr/bin/rsync",
        "--archive",
        "--ignore-existing",
        "--secluded-args",
        "--files-from",
        str(files_from),
        "--from0",
        "--rsh",
        _RSYNC_SSH,
        "--",
    )


def build_push_command(
    manifest: NodeArtifactManifest, *, files_from: str | Path | None = None,
    cluster: "ClusterSpec | None" = None
) -> tuple[str, ...]:
    """Build a fixed local-ranks-to-canonical-owner rsync argv."""
    _validate_run_id(manifest.run_id)
    if cluster is not None:
        from src.multinode import validate_cluster_topology
        validate_cluster_topology(cluster)
    local = cluster.local_repo_root if cluster is not None else _LOCAL_REPOSITORY
    remote = cluster.remote_repo_root if cluster is not None else _LOCAL_REPOSITORY
    source = (
        local
        / "checkpoints"
        / manifest.run_id
        / manifest.checkpoint_relative
    )
    destination = (
        remote
        / "checkpoints"
        / manifest.run_id
        / manifest.checkpoint_relative
    )
    list_path = Path(files_from) if files_from is not None else _files_from_path(manifest, "push")
    prefix = _rsync_prefix(list_path)
    return (
        *prefix[:-1],
        "--rsync-path",
        "/usr/bin/docker exec -i llm_kd_rdma /usr/bin/rsync",
        "--",
        f"{source}/",
        f"{_REMOTE_HOST}:{destination}/",
    )


def select_resume_entries(
    manifest: NodeArtifactManifest,
) -> tuple[ArtifactEntry, ...]:
    """Select the complete canonical checkpoint required by every resume node."""
    _validate_identity(manifest.runtime_identity)
    select_inference_entries(manifest)
    return manifest.entries


def select_collection_push_entries(
    manifest: NodeArtifactManifest,
) -> tuple[ArtifactEntry, ...]:
    """Select only locally owned DS shards and explicit common control metadata."""
    common = set(select_common_metadata_entries(manifest))
    return tuple(
        entry
        for entry in manifest.entries
        if entry in common or entry.process_rank in (2, 3)
    )


def resume_entry_delta(
    canonical_manifest: NodeArtifactManifest,
    local_manifest: NodeArtifactManifest,
) -> tuple[tuple[ArtifactEntry, ...], tuple[ArtifactEntry, ...]]:
    """Return the complete required local set and only paths still missing locally."""
    _validate_manifest(canonical_manifest, world_size=4)
    _validate_manifest(local_manifest, world_size=4)
    if canonical_manifest.process_ranks != (0, 1, 2, 3):
        raise ValueError("resume source must be the canonical four-rank manifest")
    if local_manifest.node_rank != 1 or local_manifest.process_ranks not in (
        (2, 3),
        (0, 1, 2, 3),
    ):
        raise ValueError(
            "resume destination must be the local rank-2/3 manifest or complete replica"
        )
    if (
        canonical_manifest.run_id != local_manifest.run_id
        or _validate_identity(canonical_manifest.runtime_identity)
        != _validate_identity(local_manifest.runtime_identity)
    ):
        raise ValueError("resume node-local manifests have different identity")
    required = select_resume_entries(canonical_manifest)
    local_by_path = {entry.relative_path: entry for entry in local_manifest.entries}
    missing: list[ArtifactEntry] = []
    for entry in required:
        present = local_by_path.get(entry.relative_path)
        if present is None:
            missing.append(entry)
        elif present.byte_size != entry.byte_size:
            raise ValueError(
                f"resume destination has conflicting existing file: {entry.relative_path}"
            )
    return required, tuple(missing)


def select_inference_entries(
    manifest: NodeArtifactManifest,
) -> tuple[ArtifactEntry, ...]:
    """Select only explicit model/processor artifacts, never training state."""
    identity = _validate_identity(manifest.runtime_identity)
    top_level_names = {
        entry.relative_path.name
        for entry in manifest.entries
        if entry.relative_path == Path(entry.relative_path.name)
    }
    _require_loader_asset_names(top_level_names)
    selected = tuple(
        entry
        for entry in manifest.entries
        if (
            entry.relative_path == Path(entry.relative_path.name)
            and (
                entry.relative_path.name in _INFERENCE_STATIC_NAMES
                or _INFERENCE_SHARD.fullmatch(entry.relative_path.name)
            )
        )
    )
    names = {entry.relative_path.name for entry in selected}
    has_adapter = bool(
        names & {"adapter_config.json", "adapter_model.safetensors"}
    )
    safe_single = "model.safetensors" in names
    binary_single = "pytorch_model.bin" in names
    safe_index = "model.safetensors.index.json" in names
    binary_index = "pytorch_model.bin.index.json" in names
    safe_shards = {
        name for name in names if re.fullmatch(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", name)
    }
    binary_shards = {
        name for name in names if re.fullmatch(r"pytorch_model-[0-9]{5}-of-[0-9]{5}\.bin", name)
    }
    has_full = any(
        (safe_single, binary_single, safe_index, binary_index, safe_shards, binary_shards)
    )
    if has_adapter and has_full:
        raise ValueError("mixed PEFT and full inference serialization")
    if has_adapter:
        if identity["tuning_method"] == "full":
            raise ValueError("full identity cannot select a PEFT inference artifact")
        metadata = f"{identity['stage']}_peft_metadata.json"
        required = {"adapter_config.json", "adapter_model.safetensors", metadata}
        if identity["stage"] == "stage1":
            required.add("ilvr_token_rows.safetensors")
        missing = required - names
        if missing:
            raise ValueError(
                f"incomplete PEFT inference metadata/token-row artifact: {sorted(missing)}"
            )
    else:
        if identity["tuning_method"] != "full":
            raise ValueError("PEFT inference artifact is missing adapter files")
        has_safe = safe_single or safe_index or bool(safe_shards)
        has_binary = binary_single or binary_index or bool(binary_shards)
        if has_safe and has_binary:
            raise ValueError("mixed full inference serialization families")
        if safe_single or binary_single:
            if safe_index or binary_index or safe_shards or binary_shards:
                raise ValueError("mixed standalone and sharded full inference weights")
        elif safe_index or safe_shards:
            _validate_manifest_shard_names(
                safe_shards,
                index_present=safe_index,
                pattern=re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors\Z"),
            )
        elif binary_index or binary_shards:
            _validate_manifest_shard_names(
                binary_shards,
                index_present=binary_index,
                pattern=re.compile(r"pytorch_model-(\d{5})-of-(\d{5})\.bin\Z"),
            )
        else:
            raise ValueError("inference manifest has no complete model weights")
    return selected


def _validate_manifest_shard_names(
    shards: set[str], *, index_present: bool, pattern: re.Pattern[str]
) -> None:
    if not index_present or not shards:
        raise ValueError("complete sharded inference weights require their index")
    positions: list[int] = []
    totals: set[int] = set()
    for name in shards:
        match = pattern.fullmatch(name)
        assert match is not None
        positions.append(int(match.group(1)))
        totals.add(int(match.group(2)))
    if (
        len(totals) != 1
        or next(iter(totals)) != len(shards)
        or sorted(positions) != list(range(1, len(shards) + 1))
    ):
        raise ValueError("inference shard set is incomplete or inconsistently numbered")


def build_inference_manifest(
    checkpoint_directory: str | Path,
    manifest: NodeArtifactManifest,
) -> NodeArtifactManifest:
    """Validate one complete loadable artifact and retain only inference files."""
    checkpoint = Path(checkpoint_directory)
    identity = _validate_identity(manifest.runtime_identity)
    validate_loader_artifact(checkpoint, identity)
    return filter_inference_manifest(manifest)


def filter_inference_manifest(
    manifest: NodeArtifactManifest,
) -> NodeArtifactManifest:
    """Filter a manifest whose loader contract was already pinned and validated."""
    return replace(manifest, entries=select_inference_entries(manifest))


def validate_inference_manifest_identity(
    manifest: NodeArtifactManifest,
    *,
    expected_identity: Mapping[str, object],
) -> tuple[ArtifactEntry, ...]:
    """Bind a filtered inference manifest to the canonical owner and identity."""
    _validate_manifest(manifest, world_size=4)
    expected = _validate_identity(expected_identity)
    if (
        manifest.node_rank != 0
        or manifest.process_ranks != (0, 1, 2, 3)
        or _validate_identity(manifest.runtime_identity) != expected
    ):
        raise ValueError("inference manifest canonical identity mismatch")
    return select_inference_entries(manifest)


def build_pull_command(
    manifest: NodeArtifactManifest,
    destination: str | Path | None = None,
    *,
    files_from: str | Path | None = None,
    cluster: "ClusterSpec | None" = None,
) -> tuple[str, ...]:
    """Build a canonical-owner-to-local rsync argv for resume artifacts."""
    _validate_run_id(manifest.run_id)
    if cluster is not None:
        from src.multinode import validate_cluster_topology
        validate_cluster_topology(cluster)
    local = cluster.local_repo_root if cluster is not None else _LOCAL_REPOSITORY
    remote = cluster.remote_repo_root if cluster is not None else _LOCAL_REPOSITORY
    source = (
        remote
        / "checkpoints"
        / manifest.run_id
        / manifest.checkpoint_relative
    )
    local_destination = (
        local / "checkpoints" / manifest.run_id
        if destination is None
        else Path(destination)
    )
    list_path = (
        Path(files_from)
        if files_from is not None
        else _files_from_path(manifest, "pull")
    )
    prefix = _rsync_prefix(list_path)
    return (
        *prefix[:-1],
        "--rsync-path",
        "/usr/bin/docker exec -i llm_kd_rdma /usr/bin/rsync",
        "--",
        f"{_REMOTE_HOST}:{source}/",
        f"{local_destination}/",
    )


def _identity_mapping(value: object) -> dict[str, object]:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint identity must be a mapping or dataclass")
    return _validate_identity(value, required_fields=_RESUME_IDENTITY_FIELDS)


def validate_resume_identity(checkpoint: object, requested: object) -> None:
    """Reject every checkpoint/runtime identity change before artifact transfer."""
    actual = _identity_mapping(checkpoint)
    expected = _identity_mapping(requested)
    for field in _RESUME_IDENTITY_FIELDS:
        if actual[field] != expected[field]:
            raise ValueError(f"resume identity mismatch for {field}")


def build_inference_pull_command(
    remote_manifest: NodeArtifactManifest,
    destination: str | Path,
    *,
    files_from: str | Path | None = None,
) -> tuple[str, ...]:
    """Build one manifest-limited pull into a fresh local inference directory."""
    target = Path(destination)
    _reject_symlink_ancestors(target)
    if _lstat_if_present(target) is not None:
        raise FileExistsError(f"fresh inference destination already exists: {target}")
    if remote_manifest.completion_marker is not True or not remote_manifest.entries:
        raise ValueError("remote inference manifest is incomplete")
    _validate_run_id(remote_manifest.run_id)
    identity = _validate_identity(remote_manifest.runtime_identity)
    if identity["run_id"] != remote_manifest.run_id:
        raise ValueError("remote inference manifest run ID mismatch")
    for entry in remote_manifest.entries:
        _safe_relative_path(entry.relative_path)
        if type(entry.byte_size) is not int or entry.byte_size < 0:
            raise ValueError("remote inference artifact size is invalid")
        if entry.node_rank != remote_manifest.node_rank:
            raise ValueError("remote inference artifact node ownership mismatch")
    select_inference_entries(remote_manifest)
    return build_pull_command(remote_manifest, target, files_from=files_from)


def reserve_inference_destination(destination: str | Path) -> Path:
    """Atomically reserve one fresh trusted inference destination directory."""
    from src.multinode_runtime import _open_child_directory, _open_trusted_directory

    target = Path(destination)
    if not target.is_absolute() or ".." in target.parts:
        raise ValueError("fresh inference destination must be an absolute trusted path")
    parent_descriptor = _open_trusted_directory(target.parent, create=True)
    child_descriptor: int | None = None
    try:
        try:
            child_descriptor = _open_child_directory(
                parent_descriptor,
                target.name,
                create=True,
                exclusive=True,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"fresh inference destination already exists: {target}"
            ) from error
    finally:
        if child_descriptor is not None:
            os.close(child_descriptor)
        os.close(parent_descriptor)
    return target


def resolve_local_checkpoint_destination(destination: str | Path) -> Path:
    """Map one fixed logical checkpoint path through its trusted storage link."""
    logical_root = _LOCAL_REPOSITORY / "checkpoints"
    target = Path(destination)
    if not target.is_absolute() or ".." in target.parts:
        raise ValueError("checkpoint destination must be an absolute canonical path")
    try:
        relative = target.relative_to(logical_root)
    except ValueError as error:
        raise ValueError("checkpoint destination is outside the canonical root") from error
    if len(relative.parts) < 2:
        raise ValueError("checkpoint destination must belong to a run subtree")
    _validate_run_id(relative.parts[0])
    physical_root = logical_root.resolve(strict=True)
    return physical_root.joinpath(*relative.parts)


def reserve_checkpoint_destination(destination: str | Path) -> Path:
    """Securely reserve a fresh run-owned directory behind the fixed root link."""
    physical = resolve_local_checkpoint_destination(destination)
    return reserve_inference_destination(physical)


def validate_reserved_inference_destination(destination: str | Path) -> None:
    """Re-open a reserved destination without symlinks and require it is empty."""
    from src.multinode_runtime import _open_trusted_directory

    descriptor = _open_trusted_directory(destination)
    try:
        if os.listdir(descriptor):
            raise FileExistsError("fresh inference destination is no longer empty")
    finally:
        os.close(descriptor)


def validate_transferred_entries(
    destination: str | Path, expected_entries: Sequence[ArtifactEntry]
) -> None:
    """Validate an exact transferred file set using lstat metadata only."""
    observed = {
        relative: metadata.st_size
        for relative, metadata in _walk_regular_files(Path(destination))
    }
    expected = {
        entry.relative_path: entry.byte_size
        for entry in expected_entries
    }
    if len(expected) != len(tuple(expected_entries)) or observed != expected:
        raise ValueError("transferred destination must match exact path and size list")


def validate_transferred_entry_subset(
    destination: str | Path, expected_entries: Sequence[ArtifactEntry]
) -> None:
    """Require each selected entry at its expected size without reading contents."""
    root = Path(destination)
    for entry in expected_entries:
        target = root / _safe_relative_path(entry.relative_path)
        try:
            metadata = target.lstat()
        except OSError as error:
            raise ValueError(
                f"transferred entry is missing: {entry.relative_path}"
            ) from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_size != entry.byte_size
        ):
            raise ValueError(
                f"transferred entry has invalid metadata: {entry.relative_path}"
            )


def validate_canonical_destination(
    expected_entries: Sequence[ArtifactEntry],
    observed_manifest: NodeArtifactManifest,
    *,
    expected_identity: Mapping[str, object],
    expected_run_id: str,
) -> None:
    """Require a fresh remote rebuild to equal the expected collection exactly."""
    identity = _validate_identity(expected_identity)
    _validate_manifest(observed_manifest, world_size=4)
    if observed_manifest.node_rank != 0 or observed_manifest.process_ranks != (0, 1, 2, 3):
        raise ValueError("canonical destination manifest rank map is invalid")
    if observed_manifest.run_id != expected_run_id or identity["run_id"] != expected_run_id:
        raise ValueError("canonical destination manifest run ID mismatch")
    if _validate_identity(observed_manifest.runtime_identity) != identity:
        raise ValueError("canonical destination runtime identity mismatch")
    expected = {
        (entry.relative_path, entry.byte_size, entry.process_rank)
        for entry in expected_entries
    }
    observed = {
        (entry.relative_path, entry.byte_size, entry.process_rank)
        for entry in observed_manifest.entries
    }
    if expected != observed or len(observed) != len(observed_manifest.entries):
        raise ValueError("canonical destination must match exact path and size union")


def validate_canonical_checkpoint_manifest(
    manifest: NodeArtifactManifest,
    *,
    expected_identity: Mapping[str, object],
) -> tuple[ArtifactEntry, ...]:
    """Validate a rank-0 canonical manifest as a complete four-rank checkpoint."""
    validate_canonical_destination(
        manifest.entries,
        manifest,
        expected_identity=expected_identity,
        expected_run_id=str(expected_identity.get("run_id")),
    )
    remote_entries = tuple(
        replace(entry, node_rank=0)
        for entry in manifest.entries
        if entry.process_rank in (None, 0, 1)
    )
    local_entries = tuple(
        replace(entry, node_rank=1)
        for entry in manifest.entries
        if entry.process_rank in (2, 3)
    )
    remote = replace(
        manifest,
        process_ranks=(0, 1),
        entries=remote_entries,
    )
    local = replace(
        manifest,
        node_rank=1,
        process_ranks=(2, 3),
        entries=local_entries,
    )
    return validate_collection((remote, local), world_size=4)


def files_list_bytes(entries: Sequence[ArtifactEntry]) -> bytes:
    """Encode the exact validated rsync file set without reading any artifact."""
    paths = sorted(
        {_safe_relative_path(entry.relative_path).as_posix() for entry in entries}
    )
    return b"".join(path.encode("utf-8") + b"\0" for path in paths)


def inference_manifest_path(model_directory: str | Path) -> Path:
    """Return the sibling Task 6 identity record for one prepared model tree."""
    model = Path(model_directory)
    return model.with_name(f"{model.name}{INFERENCE_MANIFEST_SUFFIX}")


def inference_model_identity(manifest: NodeArtifactManifest) -> str:
    """Derive one explicit evaluator identity from a validated Task 6 manifest."""
    validate_inference_manifest_identity(
        manifest, expected_identity=manifest.runtime_identity
    )
    payload = json.dumps(
        manifest.to_json(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"task6:{hashlib.sha256(payload).hexdigest()}"


def evaluation_result_files_bytes(*, include_runtime_identity: bool = False) -> bytes:
    """Encode only remote shard 0/1 JSONL and Task 7 sidecars."""
    names = _EVALUATION_REMOTE_FILES
    if include_runtime_identity:
        names += ('shard-0.runtime-identity.json', 'shard-1.runtime-identity.json')
    return b"".join(name.encode("utf-8") + b"\0" for name in names)


def build_evaluation_pull_command(
    output_directory: str | Path, *, files_from: str | Path, cluster: "ClusterSpec | None" = None
) -> tuple[str, ...]:
    """Build the one fixed, non-recursive remote evaluation result pull."""
    output = Path(output_directory)
    try:
        relative = output.relative_to(_LOCAL_REPOSITORY / "results")
    except ValueError as error:
        raise ValueError("evaluation output must be under the canonical result root") from error
    validate_coordinator_relative_path(relative, label="evaluation output")
    _validate_run_id(relative.parts[0])
    if cluster is not None:
        from src.multinode import validate_cluster_topology
        validate_cluster_topology(cluster)
    remote = (cluster.remote_repo_root if cluster is not None else _LOCAL_REPOSITORY) / "results" / relative
    prefix = _rsync_prefix(Path(files_from))
    return (
        *prefix[:-1],
        "--rsync-path",
        "/usr/bin/docker exec -i llm_kd_rdma /usr/bin/rsync",
        "--",
        f"{_REMOTE_HOST}:{remote}/",
        f"{output}/",
    )


def build_prepared_inference_pull_command(
    manifest: NodeArtifactManifest,
    source_directory: str | Path,
    destination: str | Path,
    *,
    files_from: str | Path,
    cluster: "ClusterSpec | None" = None,
) -> tuple[str, ...]:
    """Pull only Task 6 allowlisted files from one prepared remote artifact."""
    validate_inference_manifest_identity(
        manifest, expected_identity=manifest.runtime_identity
    )
    if cluster is not None:
        from src.multinode import validate_cluster_topology
        validate_cluster_topology(cluster)
    source = Path(source_directory)
    relative = _prepared_inference_relative_path(
        source, label="prepared inference source"
    )
    _prepared_inference_relative_path(
        destination,
        label="prepared inference destination",
        allow_resolved=True,
    )
    remote = (
        (cluster.remote_repo_root if cluster is not None else _LOCAL_REPOSITORY) / "checkpoints" / "evaluation-staging" / relative
    )
    prefix = _rsync_prefix(Path(files_from))
    return (
        *prefix[:-1],
        "--rsync-path",
        "/usr/bin/docker exec -i llm_kd_rdma /usr/bin/rsync",
        "--",
        f"{_REMOTE_HOST}:{remote}/",
        f"{Path(destination)}/",
    )


def prepared_inference_mirror_files_bytes(
    manifest: NodeArtifactManifest, source_directory: str | Path
) -> bytes:
    """Encode one prepared model and its sibling identity manifest exactly."""
    validate_inference_manifest_identity(
        manifest, expected_identity=manifest.runtime_identity
    )
    source = Path(source_directory)
    _prepared_inference_relative_path(
        source, label="prepared inference source"
    )
    paths = {
        (Path(source.name) / entry.relative_path).as_posix()
        for entry in select_inference_entries(manifest)
    }
    paths.add(inference_manifest_path(source).name)
    return b"".join(path.encode("utf-8") + b"\0" for path in sorted(paths))


def build_prepared_inference_push_command(
    manifest: NodeArtifactManifest,
    source_directory: str | Path,
    *,
    files_from: str | Path,
    cluster: "ClusterSpec | None" = None,
) -> tuple[str, ...]:
    """Push one exact prepared artifact to the identical remote checkpoint path."""
    if cluster is not None:
        from src.multinode import validate_cluster_topology
        validate_cluster_topology(cluster)
    source = Path(source_directory)
    prepared_inference_mirror_files_bytes(manifest, source)
    relative = _prepared_inference_relative_path(
        source, label="prepared inference source"
    )
    remote_parent = (
        (cluster.remote_repo_root if cluster is not None else _LOCAL_REPOSITORY)
        / "checkpoints"
        / "evaluation-staging"
        / relative.parent
    )
    prefix = _rsync_prefix(Path(files_from))
    return (
        *prefix[:-1],
        "--rsync-path",
        "/usr/bin/docker exec -i llm_kd_rdma /usr/bin/rsync",
        "--",
        f"{source.parent}/",
        f"{_REMOTE_HOST}:{remote_parent}/",
    )


def validate_evaluation_sidecars(
    manifest_paths: Sequence[str | Path],
    shard_jsonls: Sequence[str | Path],
    *,
    expected_model_identity: str,
    expected_sample_count: int,
    expected_generation_config: Mapping[str, object],
) -> None:
    """Require four successful Task 7 sidecars before invoking its merger."""
    from src.multinode_runtime import _read_regular_file

    manifests = tuple(Path(path) for path in manifest_paths)
    jsonls = tuple(Path(path) for path in shard_jsonls)
    if len(manifests) != 4 or len(jsonls) != 4:
        raise ValueError("all four evaluation workers require sidecars and JSONLs")
    if type(expected_sample_count) is not int or expected_sample_count < 0:
        raise ValueError("expected sample count must be a nonnegative integer")
    from src.multinode_evaluation import (
        MANIFEST_SCHEMA_VERSION,
        _validate_generation_config,
    )

    common_dataset_identity: object | None = None
    generation_config = _validate_generation_config(
        dict(expected_generation_config), include_default_vision_ablation=True
    )
    required = {
        "schema_version",
        "result_jsonl",
        "dataset_identity",
        "model_identity",
        "generation_config",
        "shard_index",
        "num_shards",
        "expected_source_indices",
        "processed_source_indices",
        "worker_exit_status",
        "successful",
    }
    for shard_index, (manifest_path, jsonl_path) in enumerate(zip(manifests, jsonls)):
        if manifest_path != Path(f"{jsonl_path}.manifest.json"):
            raise ValueError("evaluation sidecar path does not own its shard JSONL")
        try:
            payload = json.loads(_read_regular_file(manifest_path))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid evaluation sidecar for shard {shard_index}") from error
        if type(payload) is not dict or set(payload) != required:
            raise ValueError(f"invalid evaluation sidecar fields for shard {shard_index}")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != MANIFEST_SCHEMA_VERSION
            or type(payload["shard_index"]) is not int
            or payload["shard_index"] != shard_index
            or type(payload["num_shards"]) is not int
            or payload["num_shards"] != 4
            or type(payload["worker_exit_status"]) is not int
            or payload["worker_exit_status"] != 0
            or payload["successful"] is not True
        ):
            raise ValueError(f"unsuccessful worker sidecar for shard {shard_index}")
        if payload["result_jsonl"] != jsonl_path.name:
            raise ValueError(f"sidecar JSONL mismatch for shard {shard_index}")
        if payload["model_identity"] != expected_model_identity:
            raise ValueError("model identity mismatch across evaluation sidecars")
        if (
            _validate_generation_config(
                payload["generation_config"], include_default_vision_ablation=True
            )
            != generation_config
        ):
            raise ValueError("generation config mismatch across evaluation sidecars")
        dataset = payload["dataset_identity"]
        if (
            type(dataset) is not dict
            or set(dataset) != {"path", "size", "mtime_ns"}
            or type(dataset["path"]) is not str
            or not Path(dataset["path"]).is_absolute()
            or type(dataset["size"]) is not int
            or dataset["size"] < 0
            or type(dataset["mtime_ns"]) is not int
            or dataset["mtime_ns"] < 0
        ):
            raise ValueError(f"invalid dataset identity for shard {shard_index}")
        if common_dataset_identity is None:
            common_dataset_identity = dataset
        elif dataset != common_dataset_identity:
            raise ValueError("dataset identity mismatch across evaluation sidecars")
        expected_indices = payload["expected_source_indices"]
        processed_indices = payload["processed_source_indices"]
        wanted = list(range(shard_index, expected_sample_count, 4))
        if (
            type(expected_indices) is not list
            or any(type(index) is not int for index in expected_indices)
            or len(expected_indices) != len(set(expected_indices))
            or expected_indices != wanted
        ):
            raise ValueError(f"invalid expected indices for shard {shard_index}")
        if (
            type(processed_indices) is not list
            or any(type(index) is not int for index in processed_indices)
            or processed_indices != expected_indices
        ):
            raise ValueError(f"incomplete processed indices for shard {shard_index}")
        _reject_symlink_ancestors(jsonl_path)
        try:
            metadata = jsonl_path.lstat()
        except OSError as error:
            raise ValueError(f"missing shard JSONL for shard {shard_index}") from error
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"invalid shard JSONL for shard {shard_index}")


def publish_node_manifest(
    path: str | Path, manifest: NodeArtifactManifest
) -> Path:
    """Exclusively publish one bounded JSON manifest through the shared helper."""
    from src.multinode_runtime import publish_small_file

    payload = (
        json.dumps(
            manifest.to_json(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return publish_small_file(path, payload)


def publish_files_list(
    path: str | Path, entries: Sequence[ArtifactEntry]
) -> Path:
    """Exclusively publish the exact NUL-delimited rsync input list."""
    from src.multinode_runtime import publish_small_file

    return publish_small_file(path, files_list_bytes(entries))


class _ResumeTransferRunner(Protocol):
    def validate_resume_destination(
        self, entries: Sequence[ArtifactEntry], destination: Path
    ) -> None: ...

    def pull_resume_artifacts(
        self,
        manifest: NodeArtifactManifest,
        entries: Sequence[ArtifactEntry],
        destination: Path,
    ) -> None: ...


def prepare_resume_transfer(
    checkpoint_identity: object,
    requested_identity: object,
    remote_manifest: NodeArtifactManifest,
    destination: str | Path,
    runner: _ResumeTransferRunner,
) -> tuple[ArtifactEntry, ...]:
    """Validate all resume identity before destination checks or transfer."""
    validate_resume_identity(checkpoint_identity, requested_identity)
    entries = select_resume_entries(remote_manifest)
    target = Path(destination)
    runner.validate_resume_destination(entries, target)
    runner.pull_resume_artifacts(remote_manifest, entries, target)
    return entries


class _InferenceTransferRunner(Protocol):
    def validate_inference_manifest(self, manifest: NodeArtifactManifest) -> None: ...
    def pull_inference_artifacts(
        self, manifest: NodeArtifactManifest, destination: Path
    ) -> None: ...


def prepare_inference_copy(
    remote_manifest: NodeArtifactManifest,
    destination: str | Path,
    runner: _InferenceTransferRunner,
) -> Path:
    """Validate a remote inference manifest and pull it once to a fresh path."""
    target = Path(destination)
    runner.validate_inference_manifest(remote_manifest)
    select_inference_entries(remote_manifest)
    reserve_inference_destination(target)
    validate_reserved_inference_destination(target)
    runner.pull_inference_artifacts(remote_manifest, target)
    publish_node_manifest(inference_manifest_path(target), remote_manifest)
    return target
