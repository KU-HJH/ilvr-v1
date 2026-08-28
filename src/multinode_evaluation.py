from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


NUM_EVALUATION_SHARDS = 4
MANIFEST_SCHEMA_VERSION = 1


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


def write_shard_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    shard_jsonl: str | os.PathLike[str],
    dataset_identity: Mapping[str, Any],
    model_identity: str,
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
    if set(manifest) != required_fields:
        raise ValueError(f"invalid shard manifest fields in {path}")
    if (
        not _is_int(manifest["schema_version"])
        or manifest["schema_version"] != MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported shard manifest schema in {path}")
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
            if row.get("summary") is True:
                continue
            if "error" in row:
                raise ValueError(
                    f"error row in shard JSONL {path} line {line_number}: "
                    f"{row['error']}"
                )
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
    successes = sum(row["match"] for row in rows)
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
        "accuracy": (
            successes / expected_sample_count
            if expected_sample_count > 0
            else 0.0
        ),
        "success": successes,
    }
    _publish_outputs(rows, summary, Path(output_jsonl), Path(summary_json))
    return summary
