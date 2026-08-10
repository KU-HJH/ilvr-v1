"""Download and validate repository-local model snapshots."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SnapshotValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ModelAssetSpec:
    key: str
    repo_id: str
    directory_name: str
    index_filename: str
    weight_suffix: str
    expected_shards: int


MODEL_SPEC = ModelAssetSpec(
    key="model",
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    directory_name="Qwen2.5-VL-3B-Instruct",
    index_filename="model.safetensors.index.json",
    weight_suffix=".safetensors",
    expected_shards=2,
)
CHECKPOINT_SPEC = ModelAssetSpec(
    key="checkpoint",
    repo_id="shuai22/comt_ckpt",
    directory_name="comt_ckpt",
    index_filename="pytorch_model.bin.index.json",
    weight_suffix=".bin",
    expected_shards=7,
)
REQUIRED_METADATA = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

SnapshotFetch = Callable[..., str]


def _default_snapshot_fetch(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def ensure_model_snapshot(
    spec: ModelAssetSpec,
    models_dir: Path,
    snapshot_fetch: SnapshotFetch | None = None,
) -> Path:
    destination = Path(models_dir) / spec.directory_name
    try:
        return verify_model_snapshot(destination, spec)
    except SnapshotValidationError:
        pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    fetch = snapshot_fetch or _default_snapshot_fetch
    fetch(repo_id=spec.repo_id, local_dir=destination)
    return verify_model_snapshot(destination, spec)


def _resolve_indexed_shards(
    directory: Path, weight_map: dict[str, object], spec: ModelAssetSpec
) -> set[Path]:
    root = directory.resolve()
    shards: set[Path] = set()
    for value in weight_map.values():
        if not isinstance(value, str):
            raise SnapshotValidationError(f"{spec.key}: shard names must be strings")
        relative = Path(value)
        candidate = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not candidate.is_relative_to(root)
        ):
            raise SnapshotValidationError(f"{spec.key}: unsafe shard path: {value}")
        if candidate.suffix != spec.weight_suffix:
            raise SnapshotValidationError(f"{spec.key}: unexpected shard type: {value}")
        shards.add(candidate)
    return shards


def verify_model_snapshot(directory: Path, spec: ModelAssetSpec) -> Path:
    directory = Path(directory)
    for name in (*REQUIRED_METADATA, spec.index_filename):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SnapshotValidationError(f"{spec.key}: missing or empty file: {name}")
    try:
        index = json.loads((directory / spec.index_filename).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotValidationError(f"{spec.key}: invalid weight index") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SnapshotValidationError(f"{spec.key}: weight_map must be a non-empty object")
    indexed_shards = _resolve_indexed_shards(directory, weight_map, spec)
    if len(indexed_shards) != spec.expected_shards:
        raise SnapshotValidationError(
            f"{spec.key}: expected {spec.expected_shards} indexed shards, "
            f"found {len(indexed_shards)}"
        )
    for shard in indexed_shards:
        if not shard.is_file() or shard.stat().st_size == 0:
            raise SnapshotValidationError(
                f"{spec.key}: missing or empty shard: {shard.name}"
            )
    local_weights = {
        path.resolve()
        for path in directory.rglob(f"*{spec.weight_suffix}")
        if path.is_file()
    }
    extras = sorted(path.name for path in local_weights - indexed_shards)
    if extras:
        raise SnapshotValidationError(
            f"{spec.key}: unindexed weight files: {', '.join(extras)}"
        )
    return directory
