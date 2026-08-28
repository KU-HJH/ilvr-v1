"""Strict final validation for the four-GPU ILVR result set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType

from src.multinode_runtime import (
    _canonical_json_bytes,
    _open_trusted_directory,
    _publish_bytes_no_replace,
    _read_regular_file,
)


_METHODS = ("full", "lora", "dora")
_ZERO_STAGES = (1, 2, 3)
_MATRIX_IDENTITIES = tuple(
    (stage, method, zero_stage)
    for stage in ("stage1", "stage2")
    for method in _METHODS
    for zero_stage in _ZERO_STAGES
)
_CELL_IDENTITIES = tuple(
    (method, zero_stage) for method in _METHODS for zero_stage in _ZERO_STAGES
)
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,127}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MAX_SMALL_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ValidationReport:
    matrix_cases: int
    evaluation_samples: int
    stage1_recommendations: Mapping[str, int]
    stage2_recommendations: Mapping[str, int]


def _reject_duplicates(description: str):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{description} contains a duplicate field")
            result[key] = value
        return result

    return hook


def _read_bytes(path: Path, *, description: str) -> bytes:
    try:
        return _read_regular_file(path, maximum_size=_MAX_SMALL_BYTES)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid {description}: {path}") from error


def _decode_json(raw: bytes, *, description: str) -> object:
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicates(description))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {description}") from error


def _read_json(path: Path, *, description: str) -> dict[str, object]:
    value = _decode_json(_read_bytes(path, description=description), description=description)
    if type(value) is not dict:
        raise ValueError(f"{description} must be a JSON object")
    return value


def _exact(payload: Mapping[str, object], fields: set[str], description: str) -> None:
    if set(payload) != fields:
        raise ValueError(f"{description} fields are invalid")


def _safe_id(value: object, description: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{description} is invalid")
    return value


def _finite(value: object) -> bool:
    return type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(value)


def _direct_entries(root: Path, *, description: str) -> dict[str, str]:
    """Enumerate one no-follow directory while its descriptor is pinned."""
    try:
        descriptor = _open_trusted_directory(root)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid {description} root") from error
    try:
        entries: dict[str, str] = {}
        for name in os.listdir(descriptor):
            if not name or name in (".", "..") or "/" in name or "\0" in name:
                raise ValueError(f"unsafe direct child in {description}")
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISREG(metadata.st_mode):
                entries[name] = "file"
            elif stat.S_ISDIR(metadata.st_mode):
                entries[name] = "directory"
            else:
                raise ValueError(f"unsupported direct child in {description}: {name}")
        return entries
    finally:
        os.close(descriptor)


_COMPLETION_FIELDS = {
    "schema_version", "kind", "run_id", "world_size", "first_port", "cases"
}
_COMPLETION_CASE_FIELDS = {
    "stage", "method", "zero_stage", "child_run_id", "main_process_port",
    "witness", "exit_codes",
}
_CASE_FIELDS = {
    "schema_version", "kind", "parent_run_id", "child_run_id", "stage",
    "method", "zero_stage", "world_size", "main_process_port", "machine_ranks",
    "process_ranks", "method_artifact", "checkpoint", "raw_witness", "attempts",
    "finite_loss", "finite_gradient", "worker_errors", "collected_node_manifests",
    "exit_codes",
}
_RAW_REFERENCE_FIELDS = {"path", "sha256", "size"}
_CASE_ATTEMPT_FIELDS = {"from_step", "to_step", "rank_losses", "rank_gradients"}


def _validate_raw_matrix_witness(
    raw: bytes, *, stage: str, method: str, zero_stage: int,
    child_run_id: str, checkpoint_path: Path,
) -> dict[str, object]:
    from src.multinode import (
        _STAGE1_RAW_FIELDS,
        _STAGE1_RANK_FIELDS,
        _STAGE2_RAW_FIELDS,
        _STAGE2_RANK_FIELDS,
    )

    payload = _decode_json(raw, description="matrix raw witness")
    if type(payload) is not dict:
        raise ValueError("matrix raw witness must be an object")
    fields = _STAGE1_RAW_FIELDS if stage == "stage1" else _STAGE2_RAW_FIELDS
    rank_fields = _STAGE1_RANK_FIELDS if stage == "stage1" else _STAGE2_RANK_FIELDS
    _exact(payload, fields, "matrix raw witness")
    method_field = "method" if stage == "stage1" else "tuning_method"
    if (
        not checkpoint_path.is_absolute()
        or checkpoint_path.name != "checkpoint-2"
        or checkpoint_path.parent.name != child_run_id
        or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
        or payload["stage"] != stage or type(payload["stage"]) is not str
        or payload[method_field] != method or type(payload[method_field]) is not str
        or type(payload["zero_stage"]) is not int or payload["zero_stage"] != zero_stage
        or type(payload["config_digest"]) is not str
        or _HEX64.fullmatch(payload["config_digest"]) is None
        or type(payload["world_size"]) is not int or payload["world_size"] != 4
        or payload["process_ranks"] != [0, 1, 2, 3]
        or payload["worker_errors"] != []
        or payload["experiment_id"] != child_run_id
        or payload["checkpoint"] != str(checkpoint_path)
    ):
        raise ValueError("matrix raw witness identity is invalid")
    artifact = "full" if method == "full" else f"peft-{method}"
    artifact_field = "method_artifact" if stage == "stage1" else "artifact_kind"
    if payload[artifact_field] != artifact:
        raise ValueError("matrix raw method artifact is invalid")
    records = payload["rank_records"]
    if type(records) is not list or len(records) != 4:
        raise ValueError("matrix raw witness requires four ranks")
    for rank, record in enumerate(records):
        if type(record) is not dict or set(record) != rank_fields or record["rank"] != rank:
            raise ValueError("matrix raw rank fields are invalid")
        if stage == "stage1":
            valid = (
                record["initial_global_step"] == 2
                and record["resumed_global_step"] == 3
                and record["initial_optimizer_steps"] == 2
                and record["resumed_optimizer_steps"] == 1
                and record["method_artifact"] == artifact
                and record["finite_gradient_norm"] is True
                and _finite(record["representative_gradient_norm"])
            )
        else:
            valid = (
                record["first_global_step"] == 2
                and record["resumed_global_step"] == 3
                and record["optimizer_step"] == 3
                and record["optimizer_step_at_resume"] == 2
                and record["scheduler_step_at_resume"] == 2
                and _finite(record["gradient_norm"])
            )
        if (
            not valid or record["worker_error"] is not None
            or not _finite(record["first_loss"])
            or not _finite(record["resumed_loss"])
        ):
            raise ValueError("matrix raw rank diagnostics are invalid")
    finite_gradient = (
        payload["finite_gradient_norm"] if stage == "stage1" else payload["finite_gradient"]
    )
    if payload["finite_loss"] is not True or finite_gradient is not True:
        raise ValueError("matrix raw diagnostics are invalid")
    return payload


def _validate_matrix(matrix_root: Path) -> int:
    completion = _read_json(matrix_root / "matrix-complete.json", description="matrix completion")
    _exact(completion, _COMPLETION_FIELDS, "matrix completion")
    parent = _safe_id(completion["run_id"], "matrix run ID")
    cases = completion["cases"]
    if (
        type(completion["schema_version"]) is not int or completion["schema_version"] != 1
        or completion["kind"] != "multinode-lifecycle-matrix-complete"
        or type(completion["world_size"]) is not int or completion["world_size"] != 4
        or type(completion["first_port"]) is not int
        or not 1024 <= completion["first_port"] <= 65518
        or type(cases) is not list or len(cases) != 18
    ):
        raise ValueError("matrix completion identity is invalid")
    expected_files = {"matrix-complete.json"}
    for stage, method, zero_stage in _MATRIX_IDENTITIES:
        expected_files.add(f"{parent}-{stage}-{method}-zero{zero_stage}.json")
    if _direct_entries(matrix_root, description="matrix") != {
        name: "file" for name in expected_files
    }:
        raise ValueError("matrix requires exactly 18 direct case summaries")

    for index, (summary, identity) in enumerate(zip(cases, _MATRIX_IDENTITIES, strict=True)):
        if type(summary) is not dict:
            raise ValueError("matrix completion case must be an object")
        _exact(summary, _COMPLETION_CASE_FIELDS, "matrix completion case")
        stage, method, zero_stage = identity
        child = f"{parent}-{stage}-{method}-zero{zero_stage}"
        witness_name = f"{child}.json"
        if summary != {
            "stage": stage, "method": method, "zero_stage": zero_stage,
            "child_run_id": child,
            "main_process_port": completion["first_port"] + index,
            "witness": witness_name, "exit_codes": [0, 0],
        }:
            raise ValueError("matrix completion case identity is invalid")
        witness = _read_json(matrix_root / witness_name, description="matrix case witness")
        _exact(witness, _CASE_FIELDS, "matrix case witness")
        artifact = "full" if method == "full" else f"peft-{method}"
        if (
            type(witness["schema_version"]) is not int or witness["schema_version"] != 1
            or witness["kind"] != "multinode-lifecycle-case"
            or witness["parent_run_id"] != parent or witness["child_run_id"] != child
            or witness["stage"] != stage or witness["method"] != method
            or type(witness["zero_stage"]) is not int or witness["zero_stage"] != zero_stage
            or type(witness["world_size"]) is not int or witness["world_size"] != 4
            or type(witness["main_process_port"]) is not int
            or witness["main_process_port"] != completion["first_port"] + index
            or witness["machine_ranks"] != [0, 1]
            or witness["process_ranks"] != [0, 1, 2, 3]
            or witness["method_artifact"] != artifact
            or witness["checkpoint"] != "checkpoint-2"
            or witness["finite_loss"] is not True or witness["finite_gradient"] is not True
            or witness["worker_errors"] != []
            or type(witness["collected_node_manifests"]) is not int
            or witness["collected_node_manifests"] != 2
            or witness["exit_codes"] != {"0": 0, "1": 0}
        ):
            raise ValueError("matrix case witness identity is invalid")
        reference = witness["raw_witness"]
        if type(reference) is not dict:
            raise ValueError("matrix raw witness reference is invalid")
        _exact(reference, _RAW_REFERENCE_FIELDS, "matrix raw witness reference")
        if (
            type(reference["path"]) is not str or not Path(reference["path"]).is_absolute()
            or Path(reference["path"]).name != "witness.json"
            or Path(reference["path"]).parent.name != child
            or type(reference["sha256"]) is not str
            or _HEX64.fullmatch(reference["sha256"]) is None
            or type(reference["size"]) is not int or reference["size"] <= 0
        ):
            raise ValueError("matrix raw witness reference is invalid")
        raw = _read_bytes(Path(reference["path"]), description="matrix raw witness")
        if len(raw) != reference["size"] or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("matrix raw witness digest is invalid")
        raw_payload = _validate_raw_matrix_witness(
            raw, stage=stage, method=method, zero_stage=zero_stage,
            child_run_id=child, checkpoint_path=Path(raw_payload_checkpoint(raw)),
        )
        attempts = witness["attempts"]
        if type(attempts) is not list or len(attempts) != 2:
            raise ValueError("matrix case attempts are invalid")
        records = raw_payload["rank_records"]
        gradient_key = "representative_gradient_norm" if stage == "stage1" else "gradient_norm"
        for attempt_index, (from_step, to_step, loss_key) in enumerate(
            ((0, 2, "first_loss"), (2, 3, "resumed_loss"))
        ):
            attempt = attempts[attempt_index]
            if type(attempt) is not dict:
                raise ValueError("matrix case attempt is invalid")
            _exact(attempt, _CASE_ATTEMPT_FIELDS, "matrix case attempt")
            if attempt != {
                "from_step": from_step, "to_step": to_step,
                "rank_losses": [record[loss_key] for record in records],
                "rank_gradients": [record[gradient_key] for record in records],
            }:
                raise ValueError("matrix case diagnostics differ from raw witness")
    return 18


def raw_payload_checkpoint(raw: bytes) -> str:
    value = _decode_json(raw, description="matrix raw witness")
    if type(value) is not dict or type(value.get("checkpoint")) is not str:
        raise ValueError("matrix raw checkpoint is invalid")
    return value["checkpoint"]


_SUMMARY_FIELDS = {
    "summary", "num_samples", "average_inference_time_sec",
    "total_inference_time_sec", "accuracy", "success",
}
_SIDECAR_FIELDS = {
    "schema_version", "result_jsonl", "dataset_identity", "model_identity",
    "shard_index", "num_shards", "expected_source_indices",
    "processed_source_indices", "worker_exit_status", "successful",
}


def _read_jsonl(path: Path, *, description: str) -> list[dict[str, object]]:
    raw = _read_bytes(path, description=description)
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        row = _decode_json(line, description=f"{description} line {line_number}")
        if type(row) is not dict:
            raise ValueError(f"{description} row must be an object")
        if "error" in row:
            raise ValueError(f"error row in {description}")
        if row.get("summary") is True:
            raise ValueError(f"summary row in {description}")
        if not {"index", "task_name", "match"}.issubset(row):
            raise ValueError(f"evaluation fields missing in {description}")
        if (
            type(row["index"]) is not int or type(row["task_name"]) is not str
            or type(row["match"]) is not bool
        ):
            raise ValueError(f"evaluation row types are invalid in {description}")
        inference = row.get("inference_time_sec")
        if inference is not None and (not _finite(inference) or inference < 0):
            raise ValueError("evaluation inference time is invalid")
        rows.append(row)
    return rows


def _validate_evaluation(root: Path) -> int:
    summary = _read_json(root / "summary.json", description="evaluation summary")
    _exact(summary, _SUMMARY_FIELDS, "evaluation summary")
    if (
        summary["summary"] is not True or type(summary["num_samples"]) is not int
        or summary["num_samples"] < 0 or type(summary["success"]) is not int
        or not _finite(summary["accuracy"]) or not _finite(summary["total_inference_time_sec"])
        or not _finite(summary["average_inference_time_sec"])
    ):
        raise ValueError("evaluation summary is invalid")
    sample_count = summary["num_samples"]
    common_dataset = None
    common_model = None
    rows_by_index: dict[int, dict[str, object]] = {}
    inference_times: list[float] = []
    for shard_index in range(4):
        sidecar = _read_json(
            root / f"shard-{shard_index}.jsonl.manifest.json",
            description=f"evaluation sidecar {shard_index}",
        )
        _exact(sidecar, _SIDECAR_FIELDS, f"evaluation sidecar {shard_index}")
        expected = list(range(shard_index, sample_count, 4))
        dataset = sidecar["dataset_identity"]
        model = sidecar["model_identity"]
        if (
            type(sidecar["schema_version"]) is not int or sidecar["schema_version"] != 1
            or sidecar["result_jsonl"] != f"shard-{shard_index}.jsonl"
            or type(sidecar["shard_index"]) is not int or sidecar["shard_index"] != shard_index
            or type(sidecar["num_shards"]) is not int or sidecar["num_shards"] != 4
            or sidecar["expected_source_indices"] != expected
            or sidecar["processed_source_indices"] != expected
            or type(sidecar["worker_exit_status"]) is not int
            or sidecar["worker_exit_status"] != 0 or sidecar["successful"] is not True
            or type(model) is not str or not model.startswith("task6:")
            or _HEX64.fullmatch(model.removeprefix("task6:")) is None
            or type(dataset) is not dict or set(dataset) != {"path", "size", "mtime_ns"}
            or type(dataset["path"]) is not str or not Path(dataset["path"]).is_absolute()
            or type(dataset["size"]) is not int or dataset["size"] < 0
            or type(dataset["mtime_ns"]) is not int or dataset["mtime_ns"] < 0
        ):
            raise ValueError(f"evaluation sidecar {shard_index} is invalid")
        if common_dataset is None:
            common_dataset, common_model = dataset, model
        elif dataset != common_dataset or model != common_model:
            raise ValueError("evaluation sidecar identities disagree")
        rows = _read_jsonl(root / sidecar["result_jsonl"], description=f"evaluation shard {shard_index}")
        if [row["index"] for row in rows] != expected:
            raise ValueError("evaluation shard coverage differs from sidecar")
        for row in rows:
            index = row["index"]
            if index in rows_by_index:
                raise ValueError("duplicate evaluation source index")
            rows_by_index[index] = row
            if row.get("inference_time_sec") is not None:
                inference_times.append(float(row["inference_time_sec"]))
    if sorted(rows_by_index) != list(range(sample_count)):
        raise ValueError("evaluation source indices are incomplete")
    successes = sum(int(rows_by_index[index]["match"]) for index in range(sample_count))
    total = sum(inference_times, 0.0)
    accuracy = successes / sample_count if sample_count else 0.0
    average = total / sample_count if sample_count else 0.0
    if (
        summary["success"] != successes or summary["accuracy"] != accuracy
        or summary["total_inference_time_sec"] != total
        or summary["average_inference_time_sec"] != average
    ):
        raise ValueError("evaluation accuracy or timing is inconsistent")
    return sample_count


_CAPACITY_FIELDS = {
    "schema_version", "kind", "run_id", "stage", "method", "zero_stage",
    "attempts", "recommended_batch_per_gpu",
}
_CAPACITY_ATTEMPT_FIELDS = {
    "batch_per_gpu", "child_run_id", "main_process_port", "status", "evidence_paths"
}
_TRAINING_WITNESS_FIELDS = {
    "schema_version", "run_id", "attempt_id", "stage", "method", "zero_stage",
    "batch_per_gpu", "world_size", "machine_ranks", "exit_codes", "checkpoint_path",
    "finite_loss_evidence",
}
_STAGE1_FINITE_FIELDS = {"status", "path", "sha256", "size", "finite_loss"}
_STAGE2_FINITE_FIELDS = _STAGE1_FINITE_FIELDS | {"finite_gradient", "process_ranks"}


def _validate_stage1_child(
    path: Path, *, child: str, method: str, zero_stage: int, batch: int
) -> None:
    from src.multinode_capacity import _decode_training_witness

    producer = path.parent / "initial-producer-evidence.json"
    raw_witness = _read_bytes(path, description="Stage 1 child training witness")
    witness = _decode_training_witness(
        raw_witness, child_run_id=child, zero_stage=zero_stage,
        tuning_method=method, batch_per_gpu=batch, producer_path=producer,
    )
    raw_producer = _read_bytes(producer, description="Stage 1 child producer evidence")
    finite = witness["finite_loss_evidence"]
    payload = _decode_json(raw_producer, description="Stage 1 child producer evidence")
    evidence = payload.get("benchmark_evidence", payload) if type(payload) is dict else None
    if (
        type(evidence) is not dict or evidence.get("finite_loss") is not True
        or finite["size"] != len(raw_producer)
        or finite["sha256"] != hashlib.sha256(raw_producer).hexdigest()
    ):
        raise ValueError("Stage 1 successful child witness is invalid")


def _validate_capacity_report(
    path: Path, *, stage1_root: Path
) -> tuple[tuple[str, int], int, set[str]]:
    payload = _read_json(path, description="Stage 1 capacity cell")
    _exact(payload, _CAPACITY_FIELDS, "Stage 1 capacity cell")
    run_id = _safe_id(payload["run_id"], "Stage 1 capacity run ID")
    method = payload["method"]
    zero_stage = payload["zero_stage"]
    attempts = payload["attempts"]
    recommendation = payload["recommended_batch_per_gpu"]
    if (
        type(payload["schema_version"]) is not int or payload["schema_version"] != 1
        or payload["kind"] != "stage1-multinode-capacity" or payload["stage"] != "stage1"
        or type(method) is not str or method not in _METHODS
        or type(zero_stage) is not int or zero_stage not in _ZERO_STAGES
        or type(attempts) is not list or not attempts
        or type(recommendation) is not int or recommendation not in (1, 2, 4, 8)
        or path.parent.name != run_id
    ):
        raise ValueError("Stage 1 capacity cell is invalid")
    batches: list[int] = []
    successes: set[int] = set()
    child_ids: set[str] = set()
    for attempt in attempts:
        if type(attempt) is not dict:
            raise ValueError("Stage 1 capacity attempt is invalid")
        _exact(attempt, _CAPACITY_ATTEMPT_FIELDS, "Stage 1 capacity attempt")
        batch = attempt["batch_per_gpu"]
        child = _safe_id(attempt["child_run_id"], "Stage 1 child run ID")
        evidence_paths = attempt["evidence_paths"]
        if (
            type(batch) is not int or batch <= 0
            or child != f"{run_id}-batch{batch}"
            or type(attempt["main_process_port"]) is not int
            or not 1024 <= attempt["main_process_port"] <= 65535
            or attempt["status"] not in {"success", "capacity_oom", "integration_failure"}
            or type(evidence_paths) is not list or not evidence_paths
        ):
            raise ValueError("Stage 1 capacity attempt is invalid")
        batches.append(batch)
        child_ids.add(child)
        if attempt["status"] == "success":
            expected_witness = stage1_root / child / "initial-train-witness.json"
            if evidence_paths != [{"node": "coordinator", "path": str(expected_witness)}]:
                raise ValueError("Stage 1 success does not reference its direct child witness")
            _validate_stage1_child(
                expected_witness, child=child, method=method,
                zero_stage=zero_stage, batch=batch,
            )
            successes.add(batch)
        else:
            for reference in evidence_paths:
                if (
                    type(reference) is not dict or set(reference) != {"node", "path"}
                    or reference["node"] not in {"remote", "local", "coordinator"}
                    or type(reference["path"]) is not str
                    or not Path(reference["path"]).is_absolute()
                ):
                    raise ValueError("Stage 1 failure evidence reference is invalid")
    if batches not in ([2], [2, 4], [2, 4, 8], [2, 1]):
        raise ValueError("Stage 1 capacity attempts must begin at batch 2")
    if recommendation not in successes:
        raise ValueError("Stage 1 recommendation must be a successful attempted batch")
    return (method, zero_stage), recommendation, child_ids


def _checkpoint_has_output(checkpoint: Path) -> bool:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        return False
    for root, directories, files in os.walk(checkpoint, followlinks=False):
        root_path = Path(root)
        directories[:] = [name for name in directories if not (root_path / name).is_symlink()]
        if any(
            not (root_path / name).is_symlink()
            and (root_path / name).is_file()
            and (root_path / name).stat().st_size > 0
            for name in files
        ):
            return True
    return False


def _validate_stage2_witness(path: Path) -> tuple[tuple[str, int], int]:
    witness = _read_json(path, description="Stage 2 batch-1 witness")
    _exact(witness, _TRAINING_WITNESS_FIELDS, "Stage 2 batch-1 witness")
    run_id = _safe_id(witness["run_id"], "Stage 2 batch-1 run ID")
    method = witness["method"]
    zero_stage = witness["zero_stage"]
    checkpoint_root = Path(witness["checkpoint_path"]) if type(witness["checkpoint_path"]) is str else Path()
    producer = path.parent / "initial-producer-evidence.json"
    finite = witness["finite_loss_evidence"]
    if (
        path.parent.name != run_id or witness["attempt_id"] != "initial"
        or witness["stage"] != "stage2" or type(method) is not str or method not in _METHODS
        or type(zero_stage) is not int or zero_stage not in _ZERO_STAGES
        or type(witness["schema_version"]) is not int or witness["schema_version"] != 1
        or type(witness["batch_per_gpu"]) is not int or witness["batch_per_gpu"] != 1
        or type(witness["world_size"]) is not int or witness["world_size"] != 4
        or witness["machine_ranks"] != [0, 1] or witness["exit_codes"] != {"0": 0, "1": 0}
        or not checkpoint_root.is_absolute() or checkpoint_root.name != run_id
        or type(finite) is not dict or set(finite) != _STAGE2_FINITE_FIELDS
        or finite["status"] != "validated" or finite["path"] != str(producer)
        or type(finite["sha256"]) is not str or _HEX64.fullmatch(finite["sha256"]) is None
        or type(finite["size"]) is not int or finite["size"] <= 0
        or finite["finite_loss"] is not True or finite["finite_gradient"] is not True
        or finite["process_ranks"] != [0, 1, 2, 3]
    ):
        raise ValueError("Stage 2 batch-1 witness is invalid")
    checkpoint = checkpoint_root / "checkpoint-1"
    if not _checkpoint_has_output(checkpoint):
        raise ValueError("Stage 2 batch-1 checkpoint step 1 is invalid")
    raw = _read_bytes(producer, description="Stage 2 producer evidence")
    if len(raw) != finite["size"] or hashlib.sha256(raw).hexdigest() != finite["sha256"]:
        raise ValueError("Stage 2 producer evidence digest is invalid")
    from stage2.evidence import decode_stage2_producer_evidence

    decode_stage2_producer_evidence(
        raw, expected_run_id=run_id, expected_method=method,
        expected_zero_stage=zero_stage, expected_checkpoint=checkpoint,
        expected_global_step=1,
    )
    return (method, zero_stage), 1


def _validate_capacity(root: Path) -> tuple[Mapping[str, int], Mapping[str, int]]:
    stage1_root = root / "capacity"
    entries = _direct_entries(stage1_root, description="Stage 1 capacity")
    if any(kind != "directory" or _SAFE_ID.fullmatch(name) is None for name, kind in entries.items()):
        raise ValueError("Stage 1 capacity direct children are invalid")
    reports = [
        stage1_root / name / "capacity.json"
        for name in sorted(entries)
        if (stage1_root / name / "capacity.json").is_file()
        and not (stage1_root / name / "capacity.json").is_symlink()
    ]
    if len(reports) != 9:
        raise ValueError("exactly nine Stage 1 capacity cells are required")
    stage1: dict[str, int] = {}
    parent_ids: set[str] = set()
    child_ids: set[str] = set()
    identities: set[tuple[str, int]] = set()
    for report_path in reports:
        identity, recommendation, children = _validate_capacity_report(
            report_path, stage1_root=stage1_root
        )
        if identity in identities:
            raise ValueError("duplicate Stage 1 capacity identity")
        identities.add(identity)
        parent_ids.add(report_path.parent.name)
        child_ids.update(children)
        stage1[f"{identity[0]}-zero{identity[1]}"] = recommendation
    if identities != set(_CELL_IDENTITIES) or set(entries) != parent_ids | child_ids:
        raise ValueError("exactly nine Stage 1 capacity cells are required")

    stage2_root = root / "stage2-batch1"
    stage2_entries = _direct_entries(stage2_root, description="Stage 2 batch-1")
    if len(stage2_entries) != 9 or any(
        kind != "directory" or _SAFE_ID.fullmatch(name) is None
        for name, kind in stage2_entries.items()
    ):
        raise ValueError("exactly nine Stage 2 batch-1 cells are required")
    stage2: dict[str, int] = {}
    stage2_identities: set[tuple[str, int]] = set()
    for name in sorted(stage2_entries):
        identity, recommendation = _validate_stage2_witness(
            stage2_root / name / "initial-train-witness.json"
        )
        if identity in stage2_identities:
            raise ValueError("duplicate Stage 2 batch-1 identity")
        stage2_identities.add(identity)
        stage2[f"{identity[0]}-zero{identity[1]}"] = recommendation
    if stage2_identities != set(_CELL_IDENTITIES):
        raise ValueError("exactly nine Stage 2 batch-1 cells are required")
    return MappingProxyType(stage1), MappingProxyType(stage2)


def validate_results(
    matrix_root: Path, evaluation_root: Path, capacity_root: Path
) -> ValidationReport:
    """Validate only direct Task 10 producer paths using descriptor-bound reads."""
    stage1, stage2 = _validate_capacity(Path(capacity_root))
    return ValidationReport(
        matrix_cases=_validate_matrix(Path(matrix_root)),
        evaluation_samples=_validate_evaluation(Path(evaluation_root)),
        stage1_recommendations=stage1,
        stage2_recommendations=stage2,
    )


def _report_payload(report: ValidationReport) -> dict[str, object]:
    return {
        "schema_version": 1,
        "matrix_cases": report.matrix_cases,
        "evaluation_samples": report.evaluation_samples,
        "stage1_recommendations": dict(report.stage1_recommendations),
        "stage2_recommendations": dict(report.stage2_recommendations),
    }


def _recommendation_table(title: str, values: Mapping[str, int]) -> list[str]:
    lines = [
        f"## {title}", "", "| Method | ZeRO 1 | ZeRO 2 | ZeRO 3 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in _METHODS:
        lines.append(
            f"| {method} | {values[f'{method}-zero1']} | "
            f"{values[f'{method}-zero2']} | {values[f'{method}-zero3']} |"
        )
    return lines


def _report_markdown(report: ValidationReport) -> str:
    return "\n".join([
        "# ILVR Four-GPU Validation Summary", "",
        f"Validated matrix cases: {report.matrix_cases}", "",
        f"Validated evaluation samples: {report.evaluation_samples}", "",
        *_recommendation_table("Stage 1 recommendations", report.stage1_recommendations),
        "", *_recommendation_table("Stage 2 recommendations", report.stage2_recommendations),
        "",
    ])


def _rollback_exact(path: Path, expected: bytes) -> None:
    """Remove only the exact regular bytes this call just published."""
    try:
        observed = _read_regular_file(path, maximum_size=max(_MAX_SMALL_BYTES, len(expected)))
    except (OSError, ValueError):
        return
    if observed != expected:
        return
    try:
        descriptor = _open_trusted_directory(path.parent)
        try:
            os.unlink(path.name, dir_fd=descriptor)
        finally:
            os.close(descriptor)
    except (OSError, ValueError):
        return


def publish_validation_outputs(
    report: ValidationReport, output_manifest: str | Path, output_summary: str | Path
) -> None:
    """Prepare both outputs, precheck both absent, publish exclusively, roll back."""
    if not isinstance(report, ValidationReport):
        raise TypeError("report must be a ValidationReport")
    manifest = Path(output_manifest)
    summary = Path(output_summary)
    if manifest == summary:
        raise ValueError("output manifest and summary must differ")
    manifest_bytes = _canonical_json_bytes(_report_payload(report)) + b"\n"
    summary_bytes = _report_markdown(report).encode("utf-8")
    if os.path.lexists(manifest):
        raise FileExistsError(manifest)
    if os.path.lexists(summary):
        raise FileExistsError(summary)
    manifest_created = False
    try:
        _publish_bytes_no_replace(manifest, manifest_bytes)
        manifest_created = True
        _publish_bytes_no_replace(summary, summary_bytes)
    except BaseException:
        _rollback_exact(summary, summary_bytes)
        if manifest_created:
            _rollback_exact(manifest, manifest_bytes)
        raise


def validate_results_main(argv: Sequence[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.multinode validate-results", allow_abbrev=False
    )
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--capacity-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args(argv)
    try:
        report = validate_results(
            Path(args.matrix_root), Path(args.evaluation_root), Path(args.capacity_root)
        )
        publish_validation_outputs(report, args.output_manifest, args.output_summary)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(_report_payload(report), allow_nan=False, sort_keys=True))
    return 0
