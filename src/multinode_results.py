"""Strict final validation for the four-GPU ILVR result set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from contextvars import ContextVar
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from types import MappingProxyType

from src.multinode_runtime import (
    _canonical_json_bytes,
    _open_trusted_directory,
    _read_regular_file,
    _trusted_absolute_path,
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
_MAX_SNAPSHOT_ENTRIES = 4096
_CANONICAL_RESULT_ROOT = Path(
    "/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/results"
)
_CANONICAL_CHECKPOINT_ROOT = Path(
    "/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/checkpoints"
)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_REPORT_MANIFEST_NAME = "validation-manifest.json"
_REPORT_SUMMARY_NAME = "validation-summary.md"


@dataclass(frozen=True)
class _SnapshotFile:
    raw: bytes | None
    size: int
    device: int
    inode: int


@dataclass
class _SnapshotState:
    files: dict[Path, _SnapshotFile]
    directories: set[Path]
    entries: dict[Path, dict[str, str]]

    @classmethod
    def empty(cls) -> "_SnapshotState":
        return cls(files={}, directories=set(), entries={})

    def merge(self, other: "_SnapshotState") -> None:
        owned = set(self.files) | self.directories
        incoming = set(other.files) | other.directories
        if owned & incoming:
            raise ValueError("allowed evidence snapshots overlap")
        self.files.update(other.files)
        self.directories.update(other.directories)
        self.entries.update(other.entries)


_ACTIVE_SNAPSHOT: ContextVar[_SnapshotState | None] = ContextVar(
    "multinode_result_snapshot", default=None
)


def _snapshot_tree(
    root: Path,
    *,
    description: str,
    capture_bytes: bool = True,
    maximum_depth: int = 5,
) -> _SnapshotState:
    """Snapshot one bounded tree through retained no-follow directory fds."""
    root = Path(root)
    try:
        root_descriptor = _open_trusted_directory(root)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid {description} root") from error
    try:
        return _snapshot_open_tree(
            root_descriptor,
            root,
            description=description,
            capture_bytes=capture_bytes,
            maximum_depth=maximum_depth,
        )
    finally:
        os.close(root_descriptor)


def _snapshot_open_tree(
    root_descriptor: int,
    root: Path,
    *,
    description: str,
    capture_bytes: bool,
    maximum_depth: int = 5,
) -> _SnapshotState:
    """Snapshot below one already pinned directory without closing its fd."""
    state = _SnapshotState.empty()
    count = 0

    def walk(descriptor: int, path: Path, depth: int) -> None:
        nonlocal count
        if depth > maximum_depth:
            raise ValueError(f"{description} exceeds the snapshot depth bound")
        state.directories.add(path)
        direct: dict[str, str] = {}
        names = os.listdir(descriptor)
        for name in sorted(names):
            if not name or name in (".", "..") or "/" in name or "\0" in name:
                raise ValueError(f"unsafe child in {description}")
            count += 1
            if count > _MAX_SNAPSHOT_ENTRIES:
                raise ValueError(f"{description} exceeds the snapshot entry bound")
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            child_path = path / name
            if stat.S_ISDIR(metadata.st_mode):
                direct[name] = "directory"
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=descriptor,
                )
                try:
                    observed = os.fstat(child)
                    if (
                        observed.st_dev != metadata.st_dev
                        or observed.st_ino != metadata.st_ino
                        or not stat.S_ISDIR(observed.st_mode)
                    ):
                        raise ValueError(
                            f"swapped directory in {description}: {name}"
                        )
                    walk(child, child_path, depth + 1)
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                direct[name] = "file"
                raw: bytes | None = None
                if capture_bytes:
                    if metadata.st_size > _MAX_SMALL_BYTES:
                        raise ValueError(f"oversized small file in {description}: {name}")
                    child = os.open(
                        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=descriptor,
                    )
                    try:
                        observed = os.fstat(child)
                        if (
                            observed.st_dev != metadata.st_dev
                            or observed.st_ino != metadata.st_ino
                            or not stat.S_ISREG(observed.st_mode)
                        ):
                            raise ValueError(f"swapped file in {description}: {name}")
                        chunks = []
                        remaining = _MAX_SMALL_BYTES + 1
                        while remaining:
                            chunk = os.read(child, min(65536, remaining))
                            if not chunk:
                                break
                            chunks.append(chunk)
                            remaining -= len(chunk)
                        raw = b"".join(chunks)
                        if len(raw) != observed.st_size:
                            raise ValueError(f"changed file in {description}: {name}")
                    finally:
                        os.close(child)
                state.files[child_path] = _SnapshotFile(
                    raw=raw, size=metadata.st_size,
                    device=metadata.st_dev, inode=metadata.st_ino,
                )
            else:
                raise ValueError(f"unsupported child in {description}: {name}")
        state.entries[path] = direct

    walk(root_descriptor, root, 0)
    return state


def _snapshot_checkpoint_run(
    checkpoint_root_descriptor: int, run_id: str, *, description: str
) -> _SnapshotState:
    """Open a run fd-relative below the one transaction-pinned physical root."""
    run = _safe_id(run_id, "checkpoint snapshot run ID")
    try:
        run_descriptor = os.open(run, _DIRECTORY_FLAGS, dir_fd=checkpoint_root_descriptor)
    except OSError as error:
        raise ValueError(f"invalid {description} root") from error
    logical_run = _CANONICAL_CHECKPOINT_ROOT / run
    try:
        return _snapshot_open_tree(
            run_descriptor,
            logical_run,
            description=description,
            capture_bytes=False,
        )
    finally:
        os.close(run_descriptor)


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
    snapshot = _ACTIVE_SNAPSHOT.get()
    if snapshot is not None:
        observed = snapshot.files.get(Path(path))
        if observed is None or observed.raw is None:
            raise ValueError(f"invalid {description}: {path}")
        return observed.raw
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
    snapshot = _ACTIVE_SNAPSHOT.get()
    if snapshot is not None:
        entries = snapshot.entries.get(Path(root))
        if entries is None:
            raise ValueError(f"invalid {description} root")
        return dict(entries)
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


def _path_kind(path: Path) -> str | None:
    snapshot = _ACTIVE_SNAPSHOT.get()
    if snapshot is not None:
        if path in snapshot.files:
            return "file"
        if path in snapshot.directories:
            return "directory"
        return None
    if path.is_symlink():
        return None
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    return None


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
    "process_ranks", "artifact_kind", "checkpoint", "raw_witnesses", "attempts",
    "finite_loss", "finite_gradient", "worker_errors", "collected_node_manifests",
    "redistributed_entries", "exit_codes",
}
_RAW_REFERENCE_FIELDS = {"path", "sha256", "size"}
_CASE_ATTEMPT_FIELDS = {"from_step", "to_step", "rank_losses", "rank_gradients"}


def _validate_raw_matrix_witness(
    raw: bytes, *, phase: str, stage: str, method: str, zero_stage: int,
    child_run_id: str, checkpoint_path: Path, config_digest: str,
) -> dict[str, object]:
    from src.multinode_evidence import decode_lifecycle_phase_evidence

    if checkpoint_path != _CANONICAL_CHECKPOINT_ROOT / child_run_id / "checkpoint-2":
        raise ValueError("matrix raw checkpoint is not canonical")
    return decode_lifecycle_phase_evidence(
        raw, phase=phase, stage=stage, method=method, zero_stage=zero_stage,
        config_digest=config_digest, experiment_id=child_run_id,
        checkpoint_path=checkpoint_path,
    )


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
        expected_completion_exits = {"initial": [0, 0], "resume": [0, 0]}
        if (
            type(summary["stage"]) is not str or summary["stage"] != stage
            or type(summary["method"]) is not str or summary["method"] != method
            or type(summary["zero_stage"]) is not int
            or summary["zero_stage"] != zero_stage
            or type(summary["child_run_id"]) is not str
            or summary["child_run_id"] != child
            or type(summary["main_process_port"]) is not int
            or summary["main_process_port"] != completion["first_port"] + index
            or type(summary["witness"]) is not str or summary["witness"] != witness_name
            or type(summary["exit_codes"]) is not dict
            or set(summary["exit_codes"]) != {"initial", "resume"}
            or any(
                type(values) is not list or len(values) != 2
                for values in summary["exit_codes"].values()
            )
            or summary["exit_codes"] != expected_completion_exits
            or any(
                type(code) is not int
                for values in summary["exit_codes"].values()
                for code in values
            )
        ):
            raise ValueError("matrix completion case identity is invalid")
        witness = _read_json(matrix_root / witness_name, description="matrix case witness")
        _exact(witness, _CASE_FIELDS, "matrix case witness")
        artifact = "full" if method == "full" else "peft"
        expected_phase_exits = {
            "initial": {"0": 0, "1": 0},
            "resume": {"0": 0, "1": 0},
        }
        if (
            type(witness["schema_version"]) is not int or witness["schema_version"] != 1
            or witness["kind"] != "multinode-lifecycle-case"
            or witness["parent_run_id"] != parent or witness["child_run_id"] != child
            or witness["stage"] != stage or witness["method"] != method
            or type(witness["zero_stage"]) is not int or witness["zero_stage"] != zero_stage
            or type(witness["world_size"]) is not int or witness["world_size"] != 4
            or type(witness["main_process_port"]) is not int
            or witness["main_process_port"] != completion["first_port"] + index
            or type(witness["machine_ranks"]) is not list
            or witness["machine_ranks"] != [0, 1]
            or any(type(rank) is not int for rank in witness["machine_ranks"])
            or type(witness["process_ranks"]) is not list
            or witness["process_ranks"] != [0, 1, 2, 3]
            or any(type(rank) is not int for rank in witness["process_ranks"])
            or type(witness["artifact_kind"]) is not str
            or witness["artifact_kind"] != artifact
            or witness["checkpoint"] != "checkpoint-2"
            or witness["finite_loss"] is not True or witness["finite_gradient"] is not True
            or witness["worker_errors"] != []
            or type(witness["collected_node_manifests"]) is not int
            or witness["collected_node_manifests"] != 2
            or type(witness["redistributed_entries"]) is not int
            or witness["redistributed_entries"] <= 0
            or type(witness["exit_codes"]) is not dict
            or set(witness["exit_codes"]) != {"initial", "resume"}
            or any(
                type(values) is not dict or set(values) != {"0", "1"}
                for values in witness["exit_codes"].values()
            )
            or any(
                type(code) is not int
                for values in witness["exit_codes"].values()
                for code in values.values()
            )
            or witness["exit_codes"] != expected_phase_exits
        ):
            raise ValueError("matrix case witness identity is invalid")
        raw_references = witness["raw_witnesses"]
        if type(raw_references) is not dict or set(raw_references) != {"initial", "resume"}:
            raise ValueError("matrix raw witness references are invalid")
        raw_root = _CANONICAL_RESULT_ROOT / child
        if _direct_entries(raw_root, description="matrix raw evidence") != {
            "initial-witness.json": "file",
            "resume-witness.json": "file",
        }:
            raise ValueError("matrix raw evidence has unexpected direct children")
        raw_payloads: dict[str, dict[str, object]] = {}
        config_digest: str | None = None
        for phase in ("initial", "resume"):
            reference = raw_references[phase]
            if type(reference) is not dict:
                raise ValueError("matrix raw witness reference is invalid")
            _exact(reference, _RAW_REFERENCE_FIELDS, "matrix raw witness reference")
            expected_path = (
                _CANONICAL_RESULT_ROOT / child / f"{phase}-witness.json"
            )
            if (
                type(reference["path"]) is not str
                or Path(reference["path"]) != expected_path
                or type(reference["sha256"]) is not str
                or _HEX64.fullmatch(reference["sha256"]) is None
                or type(reference["size"]) is not int or reference["size"] <= 0
            ):
                raise ValueError("matrix raw witness path is not canonical")
            raw = _read_bytes(expected_path, description="matrix raw witness")
            if (
                len(raw) != reference["size"]
                or hashlib.sha256(raw).hexdigest() != reference["sha256"]
            ):
                raise ValueError("matrix raw witness digest is invalid")
            decoded = _decode_json(raw, description="matrix raw witness")
            if type(decoded) is not dict or type(decoded.get("config_digest")) is not str:
                raise ValueError("matrix raw witness config digest is invalid")
            if config_digest is None:
                config_digest = decoded["config_digest"]
            elif decoded["config_digest"] != config_digest:
                raise ValueError("matrix phase config digests disagree")
            raw_payloads[phase] = _validate_raw_matrix_witness(
                raw, phase=phase, stage=stage, method=method,
                zero_stage=zero_stage, child_run_id=child,
                checkpoint_path=_CANONICAL_CHECKPOINT_ROOT / child / "checkpoint-2",
                config_digest=config_digest,
            )
        if not _checkpoint_has_output(
            _CANONICAL_CHECKPOINT_ROOT / child / "checkpoint-2"
        ):
            raise ValueError("matrix checkpoint-2 metadata is invalid")
        attempts = witness["attempts"]
        if type(attempts) is not list or len(attempts) != 2:
            raise ValueError("matrix case attempts are invalid")
        for attempt_index, (phase, from_step, to_step) in enumerate(
            (("initial", 0, 2), ("resume", 2, 3))
        ):
            attempt = attempts[attempt_index]
            if type(attempt) is not dict:
                raise ValueError("matrix case attempt is invalid")
            _exact(attempt, _CASE_ATTEMPT_FIELDS, "matrix case attempt")
            if (
                type(attempt["from_step"]) is not int
                or type(attempt["to_step"]) is not int
                or type(attempt["rank_losses"]) is not list
                or len(attempt["rank_losses"]) != 4
                or not all(_finite(value) for value in attempt["rank_losses"])
                or type(attempt["rank_gradients"]) is not list
                or len(attempt["rank_gradients"]) != 4
                or not all(_finite(value) for value in attempt["rank_gradients"])
            ):
                raise ValueError("matrix case attempt types are invalid")
            records = raw_payloads[phase]["ranks"]
            if attempt != {
                "from_step": from_step, "to_step": to_step,
                "rank_losses": [record["loss"] for record in records],
                "rank_gradients": [record["gradient_norm"] for record in records],
            }:
                raise ValueError("matrix case diagnostics differ from raw witness")
    return 18


_SUMMARY_FIELDS = {
    "summary", "num_samples", "average_inference_time_sec",
    "total_inference_time_sec", "accuracy", "success",
    "correct", "source_count", "valid_gold_count", "invalid_gold_count",
    "invalid_gold", "coverage_accuracy", "valid_gold_accuracy",
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
    expected_files = {"merged.jsonl", "summary.json"}
    for shard_index in range(4):
        expected_files.add(f"shard-{shard_index}.jsonl")
        expected_files.add(f"shard-{shard_index}.jsonl.manifest.json")
    if _direct_entries(root, description="evaluation") != {
        name: "file" for name in expected_files
    }:
        raise ValueError("evaluation has unexpected direct children")
    summary = _read_json(root / "summary.json", description="evaluation summary")
    _exact(summary, _SUMMARY_FIELDS, "evaluation summary")
    if (
        summary["summary"] is not True or type(summary["num_samples"]) is not int
        or summary["num_samples"] < 0 or type(summary["success"]) is not int
        or type(summary["correct"]) is not int or summary["correct"] < 0
        or type(summary["source_count"]) is not int or summary["source_count"] < 0
        or type(summary["valid_gold_count"]) is not int
        or summary["valid_gold_count"] < 0
        or type(summary["invalid_gold_count"]) is not int
        or summary["invalid_gold_count"] < 0
        or type(summary["invalid_gold"]) is not list
        or not _finite(summary["accuracy"]) or not _finite(summary["total_inference_time_sec"])
        or not _finite(summary["average_inference_time_sec"])
        or not _finite(summary["coverage_accuracy"])
        or not _finite(summary["valid_gold_accuracy"])
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
            or type(sidecar["expected_source_indices"]) is not list
            or any(
                type(index) is not int
                for index in sidecar["expected_source_indices"]
            )
            or type(sidecar["processed_source_indices"]) is not list
            or any(
                type(index) is not int
                for index in sidecar["processed_source_indices"]
            )
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
    merged_rows = _read_jsonl(
        root / "merged.jsonl", description="evaluation merged result"
    )
    if [row["index"] for row in merged_rows] != list(range(sample_count)):
        raise ValueError("evaluation merged coverage or order is invalid")
    if merged_rows != [rows_by_index[index] for index in range(sample_count)]:
        raise ValueError("evaluation merged rows differ from the ordered shards")
    from evaluation.scoring import EvaluationScoreSummary

    try:
        score_summary = EvaluationScoreSummary.from_rows(
            [rows_by_index[index] for index in range(sample_count)], sample_count
        )
    except ValueError as error:
        raise ValueError("evaluation gold accounting is invalid") from error
    successes = score_summary.correct
    total = sum(inference_times, 0.0)
    accuracy = successes / sample_count if sample_count else 0.0
    average = total / sample_count if sample_count else 0.0
    if (
        summary["success"] != successes or summary["accuracy"] != accuracy
        or summary["total_inference_time_sec"] != total
        or summary["average_inference_time_sec"] != average
        or summary["correct"] != score_summary.correct
        or summary["source_count"] != score_summary.source_count
        or summary["valid_gold_count"] != score_summary.valid_gold_count
        or summary["invalid_gold_count"] != score_summary.invalid_gold_count
        or summary["invalid_gold"] != score_summary.invalid_gold
        or summary["coverage_accuracy"] != score_summary.coverage_accuracy
        or summary["valid_gold_accuracy"] != score_summary.valid_gold_accuracy
    ):
        raise ValueError("evaluation accuracy, gold accounting, or timing is inconsistent")
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
    if _direct_entries(path.parent, description="Stage 1 child evidence") != {
        "initial-producer-evidence.json": "file",
        "initial-train-witness.json": "file",
    }:
        raise ValueError("Stage 1 child evidence has unexpected direct children")
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
    if _direct_entries(path.parent, description="Stage 1 capacity run") != {
        "capacity.json": "file"
    }:
        raise ValueError("Stage 1 capacity run has unexpected direct children")
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
    snapshot = _ACTIVE_SNAPSHOT.get()
    if snapshot is not None:
        if checkpoint not in snapshot.directories:
            return False
        prefix = checkpoint.parts
        return any(
            path.parts[: len(prefix)] == prefix and metadata.size > 0
            for path, metadata in snapshot.files.items()
        )
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
    if _direct_entries(path.parent, description="Stage 2 batch-1 run") != {
        "initial-producer-evidence.json": "file",
        "initial-train-witness.json": "file",
    }:
        raise ValueError("Stage 2 batch-1 run has unexpected direct children")
    witness = _read_json(path, description="Stage 2 batch-1 witness")
    _exact(witness, _TRAINING_WITNESS_FIELDS, "Stage 2 batch-1 witness")
    run_id = _safe_id(witness["run_id"], "Stage 2 batch-1 run ID")
    method = witness["method"]
    zero_stage = witness["zero_stage"]
    checkpoint_root = Path(witness["checkpoint_path"]) if type(witness["checkpoint_path"]) is str else Path()
    producer = path.parent / "initial-producer-evidence.json"
    finite = witness["finite_loss_evidence"]
    if (
        path.parent.name != run_id
        or type(witness["attempt_id"]) is not str or witness["attempt_id"] != "initial"
        or type(witness["stage"]) is not str or witness["stage"] != "stage2"
        or type(method) is not str or method not in _METHODS
        or type(zero_stage) is not int or zero_stage not in _ZERO_STAGES
        or type(witness["schema_version"]) is not int or witness["schema_version"] != 1
        or type(witness["batch_per_gpu"]) is not int or witness["batch_per_gpu"] != 1
        or type(witness["world_size"]) is not int or witness["world_size"] != 4
        or type(witness["machine_ranks"]) is not list
        or witness["machine_ranks"] != [0, 1]
        or any(type(rank) is not int for rank in witness["machine_ranks"])
        or type(witness["exit_codes"]) is not dict
        or witness["exit_codes"] != {"0": 0, "1": 0}
        or any(type(code) is not int for code in witness["exit_codes"].values())
        or checkpoint_root != _CANONICAL_CHECKPOINT_ROOT / run_id
        or type(finite) is not dict or set(finite) != _STAGE2_FINITE_FIELDS
        or finite["status"] != "validated" or finite["path"] != str(producer)
        or type(finite["sha256"]) is not str or _HEX64.fullmatch(finite["sha256"]) is None
        or type(finite["size"]) is not int or finite["size"] <= 0
        or finite["finite_loss"] is not True or finite["finite_gradient"] is not True
        or type(finite["process_ranks"]) is not list
        or finite["process_ranks"] != [0, 1, 2, 3]
        or any(type(rank) is not int for rank in finite["process_ranks"])
    ):
        raise ValueError("Stage 2 batch-1 witness canonical identity is invalid")
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
    if _direct_entries(root, description="capacity evidence") != {
        "capacity": "directory",
        "stage2-batch1": "directory",
    }:
        raise ValueError("capacity evidence has unexpected direct children")
    stage1_root = root / "capacity"
    entries = _direct_entries(stage1_root, description="Stage 1 capacity")
    if any(kind != "directory" or _SAFE_ID.fullmatch(name) is None for name, kind in entries.items()):
        raise ValueError("Stage 1 capacity direct children are invalid")
    reports = [
        stage1_root / name / "capacity.json"
        for name in sorted(entries)
        if _path_kind(stage1_root / name / "capacity.json") == "file"
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
    """Snapshot all allowed evidence first, then validate immutable observations."""
    matrix_root = Path(matrix_root)
    evaluation_root = Path(evaluation_root)
    capacity_root = Path(capacity_root)
    try:
        physical_checkpoint_root = _CANONICAL_CHECKPOINT_ROOT.resolve(strict=True)
        checkpoint_root_descriptor = _open_trusted_directory(physical_checkpoint_root)
    except (OSError, ValueError) as error:
        raise ValueError("invalid canonical checkpoint root") from error
    try:
        snapshot = _SnapshotState.empty()
        matrix_snapshot = _snapshot_tree(matrix_root, description="matrix evidence")
        evaluation_snapshot = _snapshot_tree(
            evaluation_root, description="evaluation evidence"
        )
        capacity_snapshot = _snapshot_tree(
            capacity_root, description="capacity evidence"
        )
        snapshot.merge(matrix_snapshot)
        snapshot.merge(evaluation_snapshot)
        snapshot.merge(capacity_snapshot)

        completion_file = matrix_snapshot.files.get(matrix_root / "matrix-complete.json")
        if completion_file is None or completion_file.raw is None:
            raise ValueError("matrix completion is missing from the pinned snapshot")
        completion = _decode_json(
            completion_file.raw, description="matrix completion snapshot"
        )
        if type(completion) is not dict:
            raise ValueError("matrix completion snapshot must be an object")
        parent = _safe_id(completion.get("run_id"), "matrix run ID")
        for stage, method, zero_stage in _MATRIX_IDENTITIES:
            child = f"{parent}-{stage}-{method}-zero{zero_stage}"
            snapshot.merge(_snapshot_tree(
                _CANONICAL_RESULT_ROOT / child,
                description=f"matrix raw evidence {child}",
            ))
            snapshot.merge(_snapshot_checkpoint_run(
                checkpoint_root_descriptor,
                child,
                description=f"matrix checkpoint metadata {child}",
            ))

        stage2_root = capacity_root / "stage2-batch1"
        stage2_entries = capacity_snapshot.entries.get(stage2_root)
        if stage2_entries is None:
            raise ValueError("Stage 2 batch-1 snapshot is missing")
        for run_id, kind in stage2_entries.items():
            _safe_id(run_id, "Stage 2 batch-1 run ID")
            if kind != "directory":
                raise ValueError("Stage 2 batch-1 snapshot child is invalid")
            snapshot.merge(_snapshot_checkpoint_run(
                checkpoint_root_descriptor,
                run_id,
                description=f"Stage 2 checkpoint metadata {run_id}",
            ))

        token = _ACTIVE_SNAPSHOT.set(snapshot)
        try:
            stage1, stage2 = _validate_capacity(capacity_root)
            return ValidationReport(
                matrix_cases=_validate_matrix(matrix_root),
                evaluation_samples=_validate_evaluation(evaluation_root),
                stage1_recommendations=stage1,
                stage2_recommendations=stage2,
            )
        finally:
            _ACTIVE_SNAPSHOT.reset(token)
    finally:
        os.close(checkpoint_root_descriptor)


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


class ReportBundleDurabilityError(OSError):
    """A complete bundle is visible but its parent sync did not complete."""


class ReportBundlePublicationAmbiguityError(OSError):
    """A rename error left the report bundle publication outcome ambiguous."""


def _rename_noreplace(
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
) -> None:
    """Atomically rename one name only when the destination is absent."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_descriptor,
        os.fsencode(source_name),
        destination_descriptor,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), source_name)


def _write_report_bundle_file(
    staging_descriptor: int, name: str, payload: bytes
) -> None:
    """Create, freeze, and fsync one fixed file inside a private staging dir."""
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=staging_descriptor,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing validation output")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_private_report_staging(
    parent_descriptor: int, staging_descriptor: int, staging_name: str
) -> None:
    """Remove only fixed children from this invocation's private directory."""
    os.fchmod(staging_descriptor, 0o700)
    for name in (_REPORT_MANIFEST_NAME, _REPORT_SUMMARY_NAME):
        try:
            os.unlink(name, dir_fd=staging_descriptor)
        except FileNotFoundError:
            pass
    os.rmdir(staging_name, dir_fd=parent_descriptor)


def _handle_report_rename_error(
    error: BaseException,
    *,
    bundle: Path,
    staging_name: str,
) -> None:
    """Preserve both namespaces after every ambiguous rename exception."""
    orphan_staging = bundle.parent / staging_name
    raise ReportBundlePublicationAmbiguityError(
        "validation report bundle publication is ambiguous; "
        f"output root: {bundle}; orphan staging path: {orphan_staging}; "
        "neither namespace was inspected, cleaned, or relocated"
    ) from error


def publish_validation_outputs(
    report: ValidationReport, output_root: str | Path
) -> None:
    """Publish both reports through one immutable no-replace directory rename."""
    if not isinstance(report, ValidationReport):
        raise TypeError("report must be a ValidationReport")
    bundle = _trusted_absolute_path(output_root)
    if not bundle.name or bundle.name in (".", ".."):
        raise ValueError("validation output root must name a bundle directory")
    manifest_bytes = _canonical_json_bytes(_report_payload(report)) + b"\n"
    summary_bytes = _report_markdown(report).encode("utf-8")
    parent_descriptor = _open_trusted_directory(bundle.parent, create=True)
    parent_metadata = os.fstat(parent_descriptor)
    if (
        parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        os.close(parent_descriptor)
        raise ValueError("validation output parent is not a private trusted directory")
    staging_name = f".{bundle.name}.{secrets.token_hex(12)}.staging"
    staging_descriptor: int | None = None
    staging_created = False
    try:
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
        staging_created = True
        staging_descriptor = os.open(
            staging_name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor
        )
        staging_metadata = os.fstat(staging_descriptor)
        if (
            staging_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(staging_metadata.st_mode) != 0o700
        ):
            raise ValueError("validation report staging directory is not private")
        _write_report_bundle_file(
            staging_descriptor, _REPORT_MANIFEST_NAME, manifest_bytes
        )
        _write_report_bundle_file(
            staging_descriptor, _REPORT_SUMMARY_NAME, summary_bytes
        )
        os.fchmod(staging_descriptor, 0o500)
        os.fsync(staging_descriptor)
    except BaseException:
        if staging_descriptor is not None:
            _cleanup_private_report_staging(
                parent_descriptor, staging_descriptor, staging_name
            )
            os.close(staging_descriptor)
        elif staging_created:
            os.rmdir(staging_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)
        raise

    assert staging_descriptor is not None
    try:
        try:
            _rename_noreplace(
                parent_descriptor,
                staging_name,
                parent_descriptor,
                bundle.name,
            )
        except BaseException as error:
            _handle_report_rename_error(
                error,
                bundle=bundle,
                staging_name=staging_name,
            )
        try:
            os.fsync(parent_descriptor)
        except OSError as error:
            raise ReportBundleDurabilityError(
                f"validation report bundle is complete at {bundle}, but parent "
                "directory durability is ambiguous"
            ) from error
    finally:
        # Closing retained descriptors is always safe. Cleanup is deliberately
        # absent here: rename completion may be ambiguous during async unwind.
        os.close(staging_descriptor)
        os.close(parent_descriptor)


def validate_results_main(argv: Sequence[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.multinode validate-results", allow_abbrev=False
    )
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--capacity-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    try:
        report = validate_results(
            Path(args.matrix_root), Path(args.evaluation_root), Path(args.capacity_root)
        )
        publish_validation_outputs(report, args.output_root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(_report_payload(report), allow_nan=False, sort_keys=True))
    return 0
