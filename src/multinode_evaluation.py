from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from evaluation.scoring import EvaluationScoreSummary


NUM_EVALUATION_SHARDS = 4
MANIFEST_SCHEMA_VERSION = 6
VISION_ABLATION_MANIFEST_SCHEMA_VERSION = 5
REPETITION_PENALTY_MANIFEST_SCHEMA_VERSION = 4
SEQUENCE_LENGTH_MANIFEST_SCHEMA_VERSION = 3
GENERATION_CONFIG_MANIFEST_SCHEMA_VERSION = 2
LEGACY_MANIFEST_SCHEMA_VERSION = 1


def _is_int(value: object) -> bool:
    return type(value) is int


def iter_owned_samples(
    samples: Iterable[Any], num_shards: int, shard_index: int
) -> Iterator[tuple[int, Any]]:
    if (
        not _is_int(num_shards)
        or not _is_int(shard_index)
        or num_shards != NUM_EVALUATION_SHARDS
        or shard_index not in range(NUM_EVALUATION_SHARDS)
    ):
        raise ValueError("multi-node evaluation requires four shards indexed 0-3")
    for source_index, sample in enumerate(samples):
        if source_index % num_shards == shard_index:
            yield source_index, sample


def dataset_identity(dataset_path: str | os.PathLike[str]) -> dict[str, Any]:
    logical_path = Path(os.path.abspath(os.fspath(dataset_path)))
    stat = logical_path.stat()
    if not logical_path.is_file():
        raise ValueError(f"evaluation dataset is not a file: {logical_path}")
    return {
        "path": str(logical_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def dataset_content_sha256(dataset_path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(dataset_path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def authenticated_dataset_identity(dataset_path: str | os.PathLike[str], *,
                                   canonical_identity_json: str | None = None,
                                   expected_sha256: str | None = None,
                                   num_shards: int = 1) -> dict[str, Any]:
    """Authenticate mapped worker bytes before emitting the canonical identity.

    No pair means the existing physical identity. Managed callers validate this
    before model loading and again immediately before publishing a sidecar.
    """
    if (canonical_identity_json is None) != (expected_sha256 is None):
        raise ValueError('managed dataset identity inputs must be paired')
    actual = dataset_identity(dataset_path)
    if canonical_identity_json is None:
        return actual
    if num_shards != NUM_EVALUATION_SHARDS:
        raise ValueError('managed dataset identity requires four-shard evaluation')
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(character not in '0123456789abcdef' for character in expected_sha256)):
        raise ValueError('invalid expected dataset content SHA256')
    try:
        canonical = _validate_dataset_identity(json.loads(canonical_identity_json))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError('invalid canonical dataset identity') from error
    if (actual['size'] != canonical['size']
            or dataset_content_sha256(dataset_path) != expected_sha256):
        raise ValueError('evaluation dataset content mismatch')
    return canonical


def resolve_evaluation_image_paths(
    image_input: object, dataset_root: str | os.PathLike[str]
) -> list[str]:
    """Resolve one sample's safe relative image paths beneath its dataset root."""
    root = Path(dataset_root)
    if not root.is_absolute():
        raise ValueError("dataset root must be absolute")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("dataset root must be a directory")
    if image_input is None:
        values: list[object] = []
    elif isinstance(image_input, str):
        values = [image_input]
    elif isinstance(image_input, list):
        values = list(image_input)
    else:
        raise ValueError("image_input must contain safe relative paths")

    resolved_paths: list[str] = []
    for value in values:
        if (
            not isinstance(value, str)
            or not value
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in value
            )
        ):
            raise ValueError("image_input must contain safe relative paths")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("image_input must contain safe relative paths")
        candidate = root / relative
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Image not found: {value}") from error
        if not resolved.is_relative_to(root):
            raise ValueError("evaluation image must remain beneath dataset root")
        if not resolved.is_file():
            raise FileNotFoundError(f"Image not found: {value}")
        resolved_paths.append(str(resolved))
    return resolved_paths


def resolve_model_identity(
    model_dir: str | os.PathLike[str], explicit_identity: str | None
) -> str:
    if explicit_identity is not None:
        if not isinstance(explicit_identity, str) or not explicit_identity.strip():
            raise ValueError("model identity must be a non-empty string")
        return explicit_identity
    logical_path = Path(os.path.abspath(os.fspath(model_dir)))
    return f"direct:{logical_path}"


def shard_manifest_path(
    shard_jsonl: str | os.PathLike[str],
) -> Path:
    return Path(f"{os.fspath(shard_jsonl)}.manifest.json")


def _validate_index_list(value: object, field: str) -> list[int]:
    if not isinstance(value, list) or any(not _is_int(item) for item in value):
        raise ValueError(f"{field} must be a list of source indices")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} contains duplicate source indices")
    return value


def _validate_dataset_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "size", "mtime_ns"}:
        raise ValueError("invalid dataset identity")
    if not isinstance(value["path"], str) or not os.path.isabs(value["path"]):
        raise ValueError("invalid dataset identity path")
    if not _is_int(value["size"]) or value["size"] < 0:
        raise ValueError("invalid dataset identity size")
    if not _is_int(value["mtime_ns"]) or value["mtime_ns"] < 0:
        raise ValueError("invalid dataset identity mtime_ns")
    return value


def _validate_generation_config(
    value: object,
    *,
    require_sequence_length: bool = True,
    require_repetition_penalty: bool = True,
    require_vision_ablation: bool = False,
    require_generation_limit_policy: bool = True,
    include_default_vision_ablation: bool = False,
) -> dict[str, Any]:
    required = {"max_new_tokens", "temperature", "top_p"}
    if require_sequence_length:
        required.add("max_sequence_length")
    if require_repetition_penalty:
        required.add("repetition_penalty")
    if require_generation_limit_policy:
        required.add("generation_limit_policy")
    if require_vision_ablation:
        required.add("vision_ablation")
    accepted_field_sets = (
        (required,)
        if require_vision_ablation
        else (required, required | {"vision_ablation"})
    )
    accepted_field_sets += tuple(
        fields | {"disable_latent_generation"} for fields in accepted_field_sets
    )
    accepted_field_sets += tuple(
        fields | {"use_kv_cache"} for fields in accepted_field_sets
    )
    if not isinstance(value, dict) or set(value) not in accepted_field_sets:
        raise ValueError("invalid generation config")
    if "disable_latent_generation" in value and type(value["disable_latent_generation"]) is not bool:
        raise ValueError("invalid generation config disable_latent_generation")
    if "use_kv_cache" in value and type(value["use_kv_cache"]) is not bool:
        raise ValueError("invalid generation config use_kv_cache")
    max_new_tokens = value["max_new_tokens"]
    max_sequence_length = value.get("max_sequence_length")
    repetition_penalty = value.get("repetition_penalty")
    temperature = value["temperature"]
    top_p = value["top_p"]
    vision_ablation = value.get("vision_ablation", "none")
    generation_limit_policy = value.get("generation_limit_policy", "bounded")
    if not _is_int(max_new_tokens) or max_new_tokens <= 0:
        raise ValueError("invalid generation config max_new_tokens")
    if require_sequence_length and (
        not _is_int(max_sequence_length) or max_sequence_length <= 0
    ):
        raise ValueError("invalid generation config max_sequence_length")
    if require_repetition_penalty and (
        type(repetition_penalty) not in (int, float)
        or not math.isfinite(repetition_penalty)
        or repetition_penalty <= 0
    ):
        raise ValueError("invalid generation config repetition_penalty")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("invalid generation config temperature")
    if (
        type(top_p) not in (int, float)
        or not math.isfinite(top_p)
        or not 0 < top_p <= 1
    ):
        raise ValueError("invalid generation config top_p")
    if vision_ablation not in ("none", "all", "interleaved"):
        raise ValueError("invalid generation config vision_ablation")
    if generation_limit_policy not in ("bounded", "external_full_latent_block"):
        raise ValueError("invalid generation config generation_limit_policy")
    normalized = {
        "max_new_tokens": max_new_tokens,
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if require_sequence_length:
        normalized["max_sequence_length"] = max_sequence_length
    if require_repetition_penalty:
        normalized["repetition_penalty"] = float(repetition_penalty)
    if require_vision_ablation or include_default_vision_ablation or "vision_ablation" in value:
        normalized["vision_ablation"] = vision_ablation
    if require_generation_limit_policy or "generation_limit_policy" in value:
        normalized["generation_limit_policy"] = generation_limit_policy
    if "disable_latent_generation" in value:
        normalized["disable_latent_generation"] = value["disable_latent_generation"]
    if "use_kv_cache" in value:
        normalized["use_kv_cache"] = value["use_kv_cache"]
    return normalized


def write_shard_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    shard_jsonl: str | os.PathLike[str],
    dataset_identity: Mapping[str, Any],
    model_identity: str,
    generation_config: Mapping[str, Any],
    shard_index: int,
    num_shards: int,
    expected_source_indices: Sequence[int],
    processed_source_indices: Sequence[int],
    worker_exit_status: int,
) -> Path:
    manifest_path = Path(manifest_path)
    shard_jsonl = Path(shard_jsonl)
    if not shard_jsonl.is_file():
        raise ValueError(f"shard JSONL does not exist: {shard_jsonl}")
    if shard_jsonl.parent.absolute() != manifest_path.parent.absolute():
        raise ValueError("shard JSONL and manifest must share one directory")
    if not _is_int(num_shards) or num_shards != NUM_EVALUATION_SHARDS:
        raise ValueError("shard manifest requires num_shards=4")
    if not _is_int(shard_index) or shard_index not in range(num_shards):
        raise ValueError("shard manifest requires shard_index 0-3")
    if not isinstance(model_identity, str) or not model_identity.strip():
        raise ValueError("model identity must be a non-empty string")
    if not _is_int(worker_exit_status) or worker_exit_status < 0:
        raise ValueError("worker exit status must be a non-negative integer")

    identity = _validate_dataset_identity(dict(dataset_identity))
    generation_input = dict(generation_config)
    generation_input.setdefault("generation_limit_policy", "bounded")
    generation = _validate_generation_config(
        generation_input, include_default_vision_ablation=True
    )
    expected = _validate_index_list(
        list(expected_source_indices), "expected source indices"
    )
    processed = _validate_index_list(
        list(processed_source_indices), "processed source indices"
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "result_jsonl": shard_jsonl.name,
        "dataset_identity": identity,
        "model_identity": model_identity,
        "generation_config": generation,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "expected_source_indices": expected,
        "processed_source_indices": processed,
        "worker_exit_status": worker_exit_status,
        "successful": worker_exit_status == 0,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest_path


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object in {description} {path}")
    return value


def _validate_manifest(
    path: Path, expected_sample_count: int
) -> tuple[dict[str, Any], Path]:
    manifest = _read_json_object(path, description="shard manifest")
    required_fields = {
        "schema_version",
        "result_jsonl",
        "dataset_identity",
        "model_identity",
        "shard_index",
        "num_shards",
        "expected_source_indices",
        "processed_source_indices",
        "worker_exit_status",
        "successful",
    }
    schema_version = manifest.get("schema_version")
    if not _is_int(schema_version):
        raise ValueError(f"unsupported shard manifest schema in {path}")
    if schema_version in (
        MANIFEST_SCHEMA_VERSION,
        VISION_ABLATION_MANIFEST_SCHEMA_VERSION,
        REPETITION_PENALTY_MANIFEST_SCHEMA_VERSION,
        SEQUENCE_LENGTH_MANIFEST_SCHEMA_VERSION,
        GENERATION_CONFIG_MANIFEST_SCHEMA_VERSION,
    ):
        required_fields.add("generation_config")
    elif schema_version != LEGACY_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported shard manifest schema in {path}")
    if set(manifest) != required_fields:
        raise ValueError(f"invalid shard manifest fields in {path}")
    if manifest["num_shards"] != NUM_EVALUATION_SHARDS or not _is_int(
        manifest["num_shards"]
    ):
        raise ValueError(f"invalid shard count in {path}")
    shard_index = manifest["shard_index"]
    if not _is_int(shard_index) or shard_index not in range(NUM_EVALUATION_SHARDS):
        raise ValueError(f"invalid shard index in {path}")
    result_jsonl = manifest["result_jsonl"]
    if (
        not isinstance(result_jsonl, str)
        or not result_jsonl
        or Path(result_jsonl).name != result_jsonl
    ):
        raise ValueError(f"invalid shard JSONL path in {path}")
    _validate_dataset_identity(manifest["dataset_identity"])
    if not isinstance(manifest["model_identity"], str) or not manifest[
        "model_identity"
    ].strip():
        raise ValueError(f"invalid model identity in {path}")
    if schema_version in (
        MANIFEST_SCHEMA_VERSION,
        VISION_ABLATION_MANIFEST_SCHEMA_VERSION,
        REPETITION_PENALTY_MANIFEST_SCHEMA_VERSION,
        SEQUENCE_LENGTH_MANIFEST_SCHEMA_VERSION,
        GENERATION_CONFIG_MANIFEST_SCHEMA_VERSION,
    ):
        manifest["generation_config"] = _validate_generation_config(
            manifest["generation_config"],
            require_sequence_length=(
                schema_version
                in (
                    MANIFEST_SCHEMA_VERSION,
                    VISION_ABLATION_MANIFEST_SCHEMA_VERSION,
                    REPETITION_PENALTY_MANIFEST_SCHEMA_VERSION,
                    SEQUENCE_LENGTH_MANIFEST_SCHEMA_VERSION,
                )
            ),
            require_repetition_penalty=(
                schema_version in (
                    MANIFEST_SCHEMA_VERSION,
                    VISION_ABLATION_MANIFEST_SCHEMA_VERSION,
                    REPETITION_PENALTY_MANIFEST_SCHEMA_VERSION,
                )
            ),
            require_vision_ablation=(
                schema_version
                in (
                    MANIFEST_SCHEMA_VERSION,
                    VISION_ABLATION_MANIFEST_SCHEMA_VERSION,
                )
            ),
            require_generation_limit_policy=(
                schema_version == MANIFEST_SCHEMA_VERSION
            ),
        )
    exit_status = manifest["worker_exit_status"]
    if not _is_int(exit_status) or exit_status < 0:
        raise ValueError(f"invalid worker exit status in {path}")
    if type(manifest["successful"]) is not bool:
        raise ValueError(f"invalid worker success status in {path}")
    if manifest["successful"] != (exit_status == 0):
        raise ValueError(f"inconsistent worker exit status in {path}")
    if not manifest["successful"]:
        raise ValueError(f"unsuccessful worker for shard {shard_index}")

    expected = _validate_index_list(
        manifest["expected_source_indices"], "expected source indices"
    )
    processed = _validate_index_list(
        manifest["processed_source_indices"], "processed source indices"
    )
    manifest["expected_source_indices"] = expected
    manifest["processed_source_indices"] = processed
    jsonl_path = path.parent / result_jsonl
    if not jsonl_path.is_file():
        raise ValueError(f"missing shard JSONL for shard {shard_index}")
    return manifest, jsonl_path


def _read_shard_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        stream = path.open("r", encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read shard JSONL {path}: {error}") from error
    with stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid JSON in shard JSONL {path} line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"invalid JSON object in shard JSONL {path} line {line_number}"
                )
            if "error" in row:
                raise ValueError(
                    f"error row in shard JSONL {path} line {line_number}: "
                    f"{row['error']}"
                )
            if row.get("summary") is True:
                continue
            if set(("index", "task_name", "match")) - set(row):
                raise ValueError(
                    f"missing evaluation fields in shard JSONL {path} line {line_number}"
                )
            if not _is_int(row["index"]):
                raise ValueError(
                    f"invalid source index in shard JSONL {path} line {line_number}"
                )
            if not isinstance(row["task_name"], str) or type(row["match"]) is not bool:
                raise ValueError(
                    f"invalid evaluation fields in shard JSONL {path} line {line_number}"
                )
            rows.append(row)
    return rows


def _publish_outputs(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_jsonl: Path,
    summary_json: Path,
) -> None:
    if output_jsonl == summary_json:
        raise ValueError("merged JSONL and summary JSON paths must differ")
    if output_jsonl.exists():
        raise FileExistsError(output_jsonl)
    if summary_json.exists():
        raise FileExistsError(summary_json)

    output_created = False
    summary_created = False
    try:
        with output_jsonl.open("x", encoding="utf-8") as stream:
            output_created = True
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        with summary_json.open("x", encoding="utf-8") as stream:
            summary_created = True
            stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
    except BaseException:
        if summary_created:
            summary_json.unlink(missing_ok=True)
        if output_created:
            output_jsonl.unlink(missing_ok=True)
        raise


def merge_shards(
    shard_manifests: Iterable[str | os.PathLike[str]],
    *,
    expected_sample_count: int,
    output_jsonl: str | os.PathLike[str],
    summary_json: str | os.PathLike[str],
) -> dict[str, Any]:
    if not _is_int(expected_sample_count) or expected_sample_count < 0:
        raise ValueError("expected sample count must be a non-negative integer")
    manifest_paths = [Path(path) for path in shard_manifests]
    if len(manifest_paths) != NUM_EVALUATION_SHARDS:
        raise ValueError("merge requires exactly four shard manifests and JSONLs")

    manifests: dict[int, tuple[dict[str, Any], Path]] = {}
    common_dataset_identity = None
    common_model_identity = None
    common_generation_config = None
    generation_config_initialized = False
    for manifest_path in manifest_paths:
        manifest, jsonl_path = _validate_manifest(
            manifest_path, expected_sample_count
        )
        shard_index = manifest["shard_index"]
        if shard_index in manifests:
            raise ValueError(f"duplicate shard index {shard_index}")
        if common_dataset_identity is None:
            common_dataset_identity = manifest["dataset_identity"]
        elif manifest["dataset_identity"] != common_dataset_identity:
            raise ValueError("dataset identity mismatch across shards")
        if common_model_identity is None:
            common_model_identity = manifest["model_identity"]
        elif manifest["model_identity"] != common_model_identity:
            raise ValueError("model identity mismatch across shards")
        generation_config = manifest.get("generation_config")
        if not generation_config_initialized:
            common_generation_config = generation_config
            generation_config_initialized = True
        elif generation_config != common_generation_config:
            raise ValueError("generation config mismatch across shards")
        manifests[shard_index] = (manifest, jsonl_path)
    if set(manifests) != set(range(NUM_EVALUATION_SHARDS)):
        raise ValueError("merge requires shard indices 0-3 exactly once")

    rows_by_index: dict[int, dict[str, Any]] = {}
    actual_indices_by_shard: dict[int, list[int]] = {}
    for shard_index in range(NUM_EVALUATION_SHARDS):
        manifest, jsonl_path = manifests[shard_index]
        rows = _read_shard_rows(jsonl_path)
        actual_indices = []
        for row in rows:
            row_cache = row.get("use_kv_cache", False)
            manifest_cache = (manifest.get("generation_config") or {}).get("use_kv_cache", False)
            if type(row_cache) is not bool or row_cache != manifest_cache:
                raise ValueError("cache execution mode mismatch between row and manifest")
            source_index = row["index"]
            if source_index in rows_by_index:
                raise ValueError(f"duplicate source index {source_index}")
            if source_index < 0 or source_index >= expected_sample_count:
                raise ValueError(f"source index {source_index} is out of range")
            if source_index % NUM_EVALUATION_SHARDS != shard_index:
                raise ValueError(
                    f"source index {source_index} belongs to another shard"
                )
            rows_by_index[source_index] = row
            actual_indices.append(source_index)
        actual_indices_by_shard[shard_index] = actual_indices

    for source_index in range(expected_sample_count):
        if source_index not in rows_by_index:
            raise ValueError(f"missing source index {source_index}")
    for shard_index in range(NUM_EVALUATION_SHARDS):
        manifest, _jsonl_path = manifests[shard_index]
        wanted = list(
            range(shard_index, expected_sample_count, NUM_EVALUATION_SHARDS)
        )
        if manifest["expected_source_indices"] != wanted:
            raise ValueError(
                f"invalid expected source indices for shard {shard_index}"
            )
        if actual_indices_by_shard[shard_index] != manifest[
            "processed_source_indices"
        ]:
            raise ValueError(f"processed indices mismatch for shard {shard_index}")
        if manifest["processed_source_indices"] != manifest[
            "expected_source_indices"
        ]:
            raise ValueError(f"processed indices incomplete for shard {shard_index}")
    rows = [rows_by_index[index] for index in range(expected_sample_count)]
    score_summary = EvaluationScoreSummary.from_rows(
        rows, expected_sample_count
    )
    successes = score_summary.correct
    inference_times = [
        row.get("inference_time_sec")
        for row in rows
        if row.get("inference_time_sec") is not None
    ]
    if any(type(value) not in (int, float) for value in inference_times):
        raise ValueError("invalid inference time in shard output")
    total_inference_time = sum(inference_times, 0.0)
    summary = {
        "summary": True,
        "num_samples": expected_sample_count,
        "average_inference_time_sec": (
            total_inference_time / expected_sample_count
            if expected_sample_count > 0
            else 0.0
        ),
        "total_inference_time_sec": total_inference_time,
        "accuracy": score_summary.coverage_accuracy,
        "success": successes,
        "correct": score_summary.correct,
        "source_count": score_summary.source_count,
        "valid_gold_count": score_summary.valid_gold_count,
        "invalid_gold_count": score_summary.invalid_gold_count,
        "invalid_gold": score_summary.invalid_gold,
        "coverage_accuracy": score_summary.coverage_accuracy,
        "valid_gold_accuracy": score_summary.valid_gold_accuracy,
    }
    if common_generation_config is not None:
        summary["generation_config"] = common_generation_config
    _publish_outputs(rows, summary, Path(output_jsonl), Path(summary_json))
    return summary
