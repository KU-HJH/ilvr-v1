#!/usr/bin/env python3
"""Authenticate the frozen v13 capacity-tail evidence and publish its premise."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from step_preference.negatives import CheckpointFile, read_negative_manifest


HISTORICAL_GIT_COMMIT = "0e69b54c02f5472c60c6ed64e5eebc2b46374a6c"
HISTORICAL_V11_SUMMARY_SHA256 = (
    "7419aff8604072b57966fb4b460f99735cb6f7f2c2b386d2f46d7a28139a2c6f"
)
FROZEN_INPUT_SHA256 = {
    "row1241_capacity": "82a0117a65b997ef66bfa49d8762bc443872a72ae68cb4d921fd1abf6a4430ca",
    "row1241_audit": "5e3cd459a72c4f9a29f61a8c9f3b5c2ef06ae351f8bb7f4024f929ab342d6352",
    "row1241_manifest": "febc23b80a9953de643cd7c972e71a4beac1de5d79fad9bfc47c59a775f052d5",
    "row1314_capacity": "8785cd4283c95d0bac6c5e96531342a0b565b355cf9805037d74bc2a6ce82fff",
    "row1314_audit": "c4cbac9305439f7de50cc7503fb26e6639e2266208a16ed17725649e06a014a0",
    "row1314_manifest": "aa2f591ca323cd99af084d38ebbffb30e46187294bfd2ac760cda03ccb76d8c4",
}
PILOT_SUMMARY_FIELDS = (
    "schema_version", "kind", "status", "run_id", "git_commit",
    "tracked_worktree_clean", "candidate_cap", "rejected_v11_summary",
    "original_data_path", "original_data_sha256", "checkpoint_path",
    "checkpoint_files", "gpu", "audits", "admitted_source_indices",
    "ignored_source_indices", "capacity_reports", "errors",
)
_V11_FIELDS = {
    "schema_version", "run_id", "git_commit", "tracked_worktree_clean",
    "original_train", "checkpoint", "gpus", "probe_gpu", "audits",
    "capacity_reports", "admitted_row_ids", "ignored_over_cap_row_ids",
    "unprobed_admitted_row_ids", "failure", "validation", "overall_status",
}
_AUDIT_FIELDS = {"schema_version", "identity", "records", "winner"}
_AUDIT_IDENTITY_FIELDS = {
    "checkpoint_path", "data_path", "image_root", "negative_path",
    "negative_manifest", "negative_sha256", "max_sequence_length",
    "assistant_plan_rendering", "attention_implementation",
    "vision_attention_implementation",
}
_AUDIT_RECORD_FIELDS = {
    "source_index", "sample_id", "clean_length", "step_count", "branch_count",
    "unpadded_length", "padded_length",
}
_CAPACITY_FIELDS = {
    "schema_version", "status", "identity", "sample", "losses", "gradient_norm",
    "memory", "elapsed_seconds", "errors",
}
_CAPACITY_IDENTITY_FIELDS = {
    "checkpoint_path", "device", "dtype", "attention_implementation",
    "vision_attention_implementation", "tuning_method", "gradient_checkpointing",
    "git_commit", "negative_sha256", "sample_id",
}
_CAPACITY_SAMPLE_FIELDS = {
    "source_row_index", "sample_id", "clean_length", "step_count", "branch_count",
    "packed_unpadded_length", "packed_padded_length",
}
_MEMORY_FIELDS = {
    "device_total_bytes", "baseline_allocated_bytes", "peak_allocated_bytes",
    "baseline_reserved_bytes", "peak_reserved_bytes",
}
_LOSS_FIELDS = {"ce", "dpo", "entropy", "total"}
_GPU_OUTPUT_FIELDS = {"physical_index", "name", "uuid", "total_memory_bytes"}


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_json_object(path: str | Path, label: str) -> dict:
    """Read a regular strict JSON object without duplicate/non-finite values."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    _reject_nonfinite(value, label)
    return value


def _decode_json_bytes(raw: bytes, label: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    _reject_nonfinite(value, label)
    return value


def _reject_nonfinite(value, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite(item, label)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item, label)


def _exact(payload, fields, label: str) -> None:
    if not isinstance(payload, dict) or set(payload) != set(fields):
        raise ValueError(f"{label} has unknown or missing fields")


def _integer(value, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _text(value, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _sha256(value, label: str) -> str:
    value = _text(value, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _resolved(path: str | Path, label: str, *, directory: bool = False) -> Path:
    path = Path(path)
    if path.is_symlink() or (not path.is_dir() if directory else not path.is_file()):
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} must be a regular {kind}: {path}")
    return path.resolve(strict=True)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_file(path: str | Path, expected: str, label: str) -> Path:
    resolved = _resolved(path, label)
    if _file_sha256(resolved) != _sha256(expected, f"{label} SHA-256"):
        raise ValueError(f"{label} SHA-256 does not match bytes")
    return resolved


def _checkpoint_files(value, checkpoint: Path, label: str) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty array")
    records = []
    previous = None
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError(f"{label} entry must be an object")
        _exact(raw, {"relative_path", "size", "sha256", "load_root"}, "checkpoint file")
        record = CheckpointFile(**raw)
        if Path(record.load_root).resolve(strict=False) != checkpoint:
            raise ValueError("checkpoint inventory load root does not match checkpoint")
        if previous is not None and record.relative_path <= previous:
            raise ValueError("checkpoint inventory must be sorted by relative_path")
        previous = record.relative_path
        file_path = checkpoint / record.relative_path
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError(f"checkpoint inventory file is not regular: {file_path}")
        if file_path.stat().st_size != record.size or _file_sha256(file_path) != record.sha256:
            raise ValueError(f"checkpoint inventory does not match current file: {file_path}")
        records.append(raw)
    return records


def _validate_record(record: object, label: str) -> dict:
    _exact(record, _AUDIT_RECORD_FIELDS, label)
    result = dict(record)
    _text(result["sample_id"], f"{label} sample_id")
    for name in _AUDIT_RECORD_FIELDS - {"sample_id"}:
        _integer(result[name], f"{label} {name}")
    if result["clean_length"] <= 0 or result["unpadded_length"] <= 0:
        raise ValueError(f"{label} lengths must be positive")
    if result["padded_length"] < result["unpadded_length"]:
        raise ValueError(f"{label} padded length is smaller than unpadded length")
    return result


def _read_audit(path: Path, expected_sha: str, *, checkpoint: Path, manifest_path: Path, manifest_sha: str, expected_row: int, expected_clean: int, expected_padded: int) -> dict:
    path = _bound_file(path, expected_sha, "packed audit")
    payload = read_json_object(path, "packed audit")
    _exact(payload, _AUDIT_FIELDS, "packed audit")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported packed audit schema")
    _exact(payload["identity"], _AUDIT_IDENTITY_FIELDS, "packed audit identity")
    identity = payload["identity"]
    if Path(identity["checkpoint_path"]).resolve(strict=False) != checkpoint:
        raise ValueError("packed audit checkpoint identity differs")
    if Path(identity["negative_manifest"]).resolve(strict=False) != manifest_path:
        raise ValueError("packed audit negative manifest differs")
    if identity["negative_sha256"] != manifest_sha:
        raise ValueError("packed audit negative SHA-256 differs")
    if identity["max_sequence_length"] != 8192 or identity["assistant_plan_rendering"] != "adjacent-text-space-lf-v1" or identity["attention_implementation"] != "sdpa" or identity["vision_attention_implementation"] != "flash_attention_2":
        raise ValueError("packed audit identity is not the frozen v13 identity")
    if not isinstance(payload["records"], list) or len(payload["records"]) != 1:
        raise ValueError("packed audit must contain exactly one tail record")
    record = _validate_record(payload["records"][0], "packed audit record")
    if payload["winner"] != record:
        raise ValueError("packed audit winner does not equal its record")
    if (record["source_index"], record["step_count"], record["branch_count"], record["clean_length"], record["padded_length"]) != (0, 6 if expected_row != 1372 else 5, 24 if expected_row != 1372 else 20, expected_clean, expected_padded):
        raise ValueError("packed audit does not contain the required boundary geometry")
    return {"path": str(path), "sha256": expected_sha, "record": record, "identity": identity}


def _read_capacity(path: Path, expected_sha: str, *, checkpoint: Path, audit: dict, expected_row: int, expected_clean: int, expected_padded: int, expected_peak: int, required_status: str) -> dict:
    path = _bound_file(path, expected_sha, "capacity report")
    payload = read_json_object(path, "capacity report")
    _exact(payload, _CAPACITY_FIELDS, "capacity report")
    if payload["schema_version"] != 1 or payload["status"] != required_status:
        raise ValueError("capacity report status is not the required frozen status")
    _exact(payload["identity"], _CAPACITY_IDENTITY_FIELDS, "capacity report identity")
    identity = payload["identity"]
    if Path(identity["checkpoint_path"]).resolve(strict=False) != checkpoint or identity["device"] != "cuda:0" or identity["dtype"] != "bfloat16" or identity["attention_implementation"] != "sdpa" or identity["vision_attention_implementation"] != "flash_attention_2" or identity["tuning_method"] != "dora" or identity["gradient_checkpointing"] is not True or identity["git_commit"] != HISTORICAL_GIT_COMMIT:
        raise ValueError("capacity report identity is not the frozen v13 identity")
    if identity["negative_sha256"] != audit["identity"]["negative_sha256"] or identity["sample_id"] != audit["record"]["sample_id"]:
        raise ValueError("capacity report does not bind its packed audit")
    _exact(payload["sample"], _CAPACITY_SAMPLE_FIELDS, "capacity report sample")
    sample = payload["sample"]
    for name in _CAPACITY_SAMPLE_FIELDS - {"sample_id"}:
        _integer(sample[name], f"capacity report sample {name}")
    if (sample["source_row_index"], sample["sample_id"], sample["clean_length"], sample["step_count"], sample["branch_count"], sample["packed_padded_length"]) != (expected_row, audit["record"]["sample_id"], expected_clean, audit["record"]["step_count"], audit["record"]["branch_count"], expected_padded):
        raise ValueError("capacity report does not contain the required boundary sample")
    _exact(payload["losses"], _LOSS_FIELDS, "capacity report losses")
    _exact(payload["memory"], _MEMORY_FIELDS, "capacity report memory")
    memory = payload["memory"]
    for name in _MEMORY_FIELDS:
        _integer(memory[name], f"capacity report memory {name}")
    if memory["device_total_bytes"] != 99949674496 or memory["peak_reserved_bytes"] != expected_peak:
        raise ValueError("capacity report memory is not the frozen boundary measurement")
    if required_status == "pass":
        if payload["errors"] != []:
            raise ValueError("passing capacity report must have no errors")
        for name, value in payload["losses"].items():
            _finite(value, f"capacity loss {name}")
        if _finite(payload["gradient_norm"], "capacity gradient norm") <= 0.0:
            raise ValueError("capacity gradient norm must be positive")
    elif not isinstance(payload["errors"], list) or not payload["errors"] or not all(isinstance(item, str) and item for item in payload["errors"]):
        raise ValueError("failed capacity report must contain errors")
    _finite(payload["elapsed_seconds"], "capacity elapsed_seconds")
    return {"path": str(path), "sha256": expected_sha, "peak_reserved_fraction": memory["peak_reserved_bytes"] / memory["device_total_bytes"], "payload": payload}


def _read_manifest(path: Path, expected_sha: str, *, checkpoint: Path, checkpoint_files: list[dict], label: str) -> tuple[Path, str]:
    path = _bound_file(path, expected_sha, label)
    manifest = read_negative_manifest(path)
    if manifest.schema_version != 3 or Path(manifest.stage2a_checkpoint_path).resolve(strict=False) != checkpoint:
        raise ValueError("negative manifest does not bind the frozen checkpoint")
    manifest_files = [
        {"relative_path": item.relative_path, "size": item.size, "sha256": item.sha256, "load_root": item.load_root}
        for item in manifest.checkpoint_files
    ]
    if manifest_files != checkpoint_files:
        raise ValueError("negative manifest checkpoint inventory does not match completely")
    return path, manifest.negative_sha256


def _v11_ref(payload: dict, name: str, fields: set, row_id: int) -> dict:
    matches = [entry for entry in payload[name] if isinstance(entry, dict) and entry.get("row_id") == row_id]
    if len(matches) != 1:
        raise ValueError(f"v11 summary must contain exactly one row-{row_id} {name} reference")
    _exact(matches[0], fields, f"v11 {name} reference")
    return matches[0]


def _read_v11(path: str | Path, *, original: Path, original_sha: str, checkpoint: Path) -> tuple[Path, str, list[dict], dict]:
    path = _resolved(path, "rejected v11 summary")
    raw_sha = _file_sha256(path)
    if raw_sha != HISTORICAL_V11_SUMMARY_SHA256:
        raise ValueError("rejected v11 summary SHA-256 does not match frozen evidence")
    payload = read_json_object(path, "rejected v11 summary")
    _exact(payload, _V11_FIELDS, "rejected v11 summary")
    if payload["schema_version"] != 1 or payload["git_commit"] != HISTORICAL_GIT_COMMIT or payload["tracked_worktree_clean"] is not True or payload["overall_status"] != "fail":
        raise ValueError("v11 summary is not the frozen rejected premise")
    _exact(payload["original_train"], {"path", "sha256"}, "v11 original_train")
    if Path(payload["original_train"]["path"]).resolve(strict=False) != original or payload["original_train"]["sha256"] != original_sha:
        raise ValueError("v11 original data identity differs")
    _exact(payload["checkpoint"], {"path", "checkpoint_files", "identity_source_manifests"}, "v11 checkpoint")
    if Path(payload["checkpoint"]["path"]).resolve(strict=False) != checkpoint:
        raise ValueError("v11 checkpoint identity differs")
    checkpoint_files = _checkpoint_files(payload["checkpoint"]["checkpoint_files"], checkpoint, "v11 checkpoint_files")
    if not isinstance(payload["gpus"], list) or not payload["gpus"]:
        raise ValueError("v11 GPU inventory must be nonempty")
    for gpu in payload["gpus"]:
        _exact(gpu, {"index", "name", "uuid", "memory_total_mib", "memory_used_mib"}, "v11 GPU")
        _integer(gpu["index"], "v11 GPU index")
        _text(gpu["name"], "v11 GPU name")
        _text(gpu["uuid"], "v11 GPU uuid")
        _integer(gpu["memory_total_mib"], "v11 GPU total memory", minimum=1)
        _integer(gpu["memory_used_mib"], "v11 GPU used memory")
    _exact(payload["probe_gpu"], {"host_gpu_index", "visible_device_index", "name", "uuid"}, "v11 probe GPU")
    if payload["probe_gpu"] != {"host_gpu_index": 0, "visible_device_index": 0, "name": payload["gpus"][0]["name"], "uuid": payload["gpus"][0]["uuid"]}:
        raise ValueError("v11 probe GPU is not historical GPU 0")
    manifest_ref = _v11_ref(payload["checkpoint"], "identity_source_manifests", {"row_id", "path", "sha256", "schema_version"}, 1372)
    if manifest_ref["schema_version"] != 3:
        raise ValueError("v11 negative manifest must use schema 3")
    manifest_path, _ = _read_manifest(manifest_ref["path"], manifest_ref["sha256"], checkpoint=checkpoint, checkpoint_files=checkpoint_files, label="v11 negative manifest")
    audit_ref = _v11_ref(payload, "audits", {"row_id", "path", "sha256", "padded_length", "status"}, 1372)
    if audit_ref["status"] != "admitted" or audit_ref["padded_length"] != 6784:
        raise ValueError("v11 row-1372 audit is not admitted at 6784")
    audit = _read_audit(audit_ref["path"], audit_ref["sha256"], checkpoint=checkpoint, manifest_path=manifest_path, manifest_sha=read_negative_manifest(manifest_path).negative_sha256, expected_row=1372, expected_clean=6065, expected_padded=6784)
    capacity_ref = _v11_ref(payload, "capacity_reports", {"row_id", "path", "sha256", "status", "device_total_bytes", "peak_reserved_bytes", "reserved_ratio"}, 1372)
    if capacity_ref["status"] != "oom" or capacity_ref["device_total_bytes"] != 99949674496 or capacity_ref["peak_reserved_bytes"] != 99199483904:
        raise ValueError("v11 row-1372 capacity reference is not the OOM premise")
    _finite(capacity_ref["reserved_ratio"], "v11 reserved_ratio")
    _read_capacity(capacity_ref["path"], capacity_ref["sha256"], checkpoint=checkpoint, audit=audit, expected_row=1372, expected_clean=6065, expected_padded=6784, expected_peak=99199483904, required_status="oom")
    _exact(payload["failure"], {"row_id", "observed_status", "reason", "errors"}, "v11 failure")
    if payload["failure"]["row_id"] != 1372 or payload["failure"]["observed_status"] != "oom" or payload["failure"]["reason"] != "capacity_probe_oom" or not isinstance(payload["failure"]["errors"], list) or not payload["failure"]["errors"]:
        raise ValueError("v11 failure is not the row-1372 OOM premise")
    _exact(payload["validation"], {"all_admitted_reports_pass", "all_admitted_reports_present", "at_least_two_admitted", "reserved_ratio_at_most_0_85", "row_1582_admitted"}, "v11 validation")
    if not all(type(value) is bool for value in payload["validation"].values()):
        raise ValueError("v11 validation values must be booleans")
    for name in ("admitted_row_ids", "ignored_over_cap_row_ids", "unprobed_admitted_row_ids"):
        if not isinstance(payload[name], list) or not all(type(value) is int for value in payload[name]):
            raise ValueError(f"v11 {name} must be integer array")
    return path, raw_sha, checkpoint_files, payload["gpus"][0]


def _live_gpu_inventory() -> list[dict]:
    try:
        import torch
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"], check=True, text=True, capture_output=True).stdout.splitlines()
        result = []
        for row in rows:
            index, name, uuid = (part.strip() for part in row.split(",", 2))
            physical_index = int(index)
            if physical_index == 0:
                total = int(torch.cuda.get_device_properties(0).total_memory)
                result.append({"index": physical_index, "name": name, "uuid": uuid, "total_memory_bytes": total})
        return result
    except (OSError, ValueError, subprocess.CalledProcessError, RuntimeError) as error:
        raise ValueError(f"could not inspect live GPU inventory: {error}") from error


def _validate_live_gpu(historical: dict, inventory: object) -> dict:
    if not isinstance(inventory, list):
        raise ValueError("live GPU inventory must be a list")
    matches = [item for item in inventory if isinstance(item, dict) and item.get("index") == 0]
    if len(matches) != 1:
        raise ValueError("live GPU inventory does not contain physical GPU 0")
    gpu = matches[0]
    _exact(gpu, {"index", "name", "uuid", "total_memory_bytes"}, "live GPU")
    if gpu["name"] != historical["name"] or gpu["uuid"] != historical["uuid"] or gpu["total_memory_bytes"] != 99949674496:
        raise ValueError("live GPU 0 does not match the historical GPU identity")
    return {"physical_index": 0, "name": gpu["name"], "uuid": gpu["uuid"], "total_memory_bytes": gpu["total_memory_bytes"]}


def _tail(root: str | Path, row: int, *, checkpoint: Path, checkpoint_files: list[dict], original_sha: str) -> tuple[dict, dict]:
    root = _resolved(root, f"row-{row} root", directory=True)
    manifest_path = root / f"row-{row}-negative-manifest.json"
    audit_path = root / f"row-{row}-packed-audit.json"
    capacity_path = root / f"row-{row}-capacity.json"
    manifest_path = _resolved(manifest_path, f"row-{row} negative manifest")
    manifest_sha = FROZEN_INPUT_SHA256[f"row{row}_manifest"]
    manifest_path, negative_sha = _read_manifest(manifest_path, manifest_sha, checkpoint=checkpoint, checkpoint_files=checkpoint_files, label=f"row-{row} negative manifest")
    expected = {1241: (4000, 4864, 85115011072), 1314: (3733, 4608, 78922121216)}[row]
    audit_path = _resolved(audit_path, f"row-{row} packed audit")
    audit_sha = FROZEN_INPUT_SHA256[f"row{row}_audit"]
    audit = _read_audit(audit_path, audit_sha, checkpoint=checkpoint, manifest_path=manifest_path, manifest_sha=negative_sha, expected_row=row, expected_clean=expected[0], expected_padded=expected[1])
    capacity_path = _resolved(capacity_path, f"row-{row} capacity report")
    capacity_sha = FROZEN_INPUT_SHA256[f"row{row}_capacity"]
    capacity = _read_capacity(capacity_path, capacity_sha, checkpoint=checkpoint, audit=audit, expected_row=row, expected_clean=expected[0], expected_padded=expected[1], expected_peak=expected[2], required_status="pass")
    audit_summary = {"source_index": row, "path": str(audit_path), "sha256": audit_sha, "packed_padded_length": expected[1], "admission": "ignored_over_cap" if row == 1241 else "admitted"}
    capacity_summary = {"source_index": row, "path": str(capacity_path), "sha256": capacity_sha, "status": "pass", "peak_reserved_fraction": capacity["peak_reserved_fraction"]}
    return audit_summary, capacity_summary


def authenticate_v13_premise(*, original_data: str | Path, checkpoint: str | Path, rejected_v11_summary: str | Path, row1241_root: str | Path, row1314_root: str | Path, gpu_inventory_reader=_live_gpu_inventory, run_id: str = "step-pref-packed-premise-v13") -> dict:
    """Authenticate immutable v11/tail evidence and return the exact v13 summary."""
    original = _resolved(original_data, "original data")
    original_sha = _file_sha256(original)
    checkpoint = _resolved(checkpoint, "checkpoint", directory=True)
    v11_path, v11_sha, checkpoint_files, historical_gpu = _read_v11(rejected_v11_summary, original=original, original_sha=original_sha, checkpoint=checkpoint)
    gpu = _validate_live_gpu(historical_gpu, gpu_inventory_reader())
    audit1241, capacity1241 = _tail(row1241_root, 1241, checkpoint=checkpoint, checkpoint_files=checkpoint_files, original_sha=original_sha)
    audit1314, capacity1314 = _tail(row1314_root, 1314, checkpoint=checkpoint, checkpoint_files=checkpoint_files, original_sha=original_sha)
    if not Fraction(85115011072, 99949674496) > Fraction(85, 100) or not Fraction(78922121216, 99949674496) <= Fraction(85, 100):
        raise AssertionError("frozen v13 boundary constants are inconsistent")
    return {
        "schema_version": 1,
        "kind": "stage2b-packed-pilot",
        "status": "pass",
        "run_id": _text(run_id, "run_id"),
        "git_commit": HISTORICAL_GIT_COMMIT,
        "tracked_worktree_clean": True,
        "candidate_cap": 4608,
        "rejected_v11_summary": {"path": str(v11_path), "sha256": v11_sha},
        "original_data_path": str(original),
        "original_data_sha256": original_sha,
        "checkpoint_path": str(checkpoint),
        "checkpoint_files": checkpoint_files,
        "gpu": gpu,
        "audits": [audit1241, audit1314],
        "admitted_source_indices": [1314],
        "ignored_source_indices": [1241],
        "capacity_reports": [capacity1241, capacity1314],
        "errors": [],
    }


def decode_pilot_summary(raw: bytes) -> dict:
    payload = _decode_json_bytes(raw, "pilot summary")
    _exact(payload, PILOT_SUMMARY_FIELDS, "pilot summary")
    if payload["schema_version"] != 1 or payload["kind"] != "stage2b-packed-pilot" or payload["status"] != "pass" or payload["git_commit"] != HISTORICAL_GIT_COMMIT or payload["tracked_worktree_clean"] is not True or payload["candidate_cap"] != 4608 or payload["errors"] != []:
        raise ValueError("pilot summary is not a passing v13 premise")
    _text(payload["run_id"], "pilot summary run_id")
    _sha256(payload["original_data_sha256"], "pilot summary original data SHA-256")
    _exact(payload["rejected_v11_summary"], {"path", "sha256"}, "pilot rejected v11 summary")
    _sha256(payload["rejected_v11_summary"]["sha256"], "pilot rejected v11 SHA-256")
    _exact(payload["gpu"], _GPU_OUTPUT_FIELDS, "pilot GPU")
    if payload["gpu"]["physical_index"] != 0 or payload["gpu"]["total_memory_bytes"] != 99949674496:
        raise ValueError("pilot GPU identity is invalid")
    if payload["admitted_source_indices"] != [1314] or payload["ignored_source_indices"] != [1241]:
        raise ValueError("pilot summary partitions are not the v13 boundary")
    if not isinstance(payload["audits"], list) or not isinstance(payload["capacity_reports"], list) or len(payload["audits"]) != 2 or len(payload["capacity_reports"]) != 2:
        raise ValueError("pilot summary must contain both boundary witnesses")
    for expected_row, expected_length, expected_admission, audit in ((1241, 4864, "ignored_over_cap", payload["audits"][0]), (1314, 4608, "admitted", payload["audits"][1])):
        _exact(audit, {"source_index", "path", "sha256", "packed_padded_length", "admission"}, "pilot audit")
        if audit["source_index"] != expected_row or audit["packed_padded_length"] != expected_length or audit["admission"] != expected_admission:
            raise ValueError("pilot audit boundary is invalid")
        _sha256(audit["sha256"], "pilot audit SHA-256")
    for expected_row, report in zip((1241, 1314), payload["capacity_reports"]):
        _exact(report, {"source_index", "path", "sha256", "status", "peak_reserved_fraction"}, "pilot capacity report")
        if report["source_index"] != expected_row or report["status"] != "pass":
            raise ValueError("pilot capacity report boundary is invalid")
        fraction = _finite(report["peak_reserved_fraction"], "pilot peak reserved fraction")
        if (expected_row == 1241 and fraction <= 0.85) or (expected_row == 1314 and fraction > 0.85):
            raise ValueError("pilot capacity fraction does not establish the v13 boundary")
        _sha256(report["sha256"], "pilot capacity SHA-256")
    return payload


def read_pilot_summary(path: str | Path) -> dict:
    path = _resolved(path, "pilot summary")
    return decode_pilot_summary(path.read_bytes())


def _canonical_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def publish_pilot_summary(path: str | Path, summary: dict) -> str:
    path = Path(path)
    if path.is_symlink() or path.exists():
        raise ValueError(f"pilot summary output already exists: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(f"pilot summary parent must be a regular directory: {path.parent}")
    payload = _canonical_json_bytes(summary)
    decode_pilot_summary(payload)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise ValueError(f"pilot summary output already exists: {path}") from error
    read_pilot_summary(path)
    return _file_sha256(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "publish"):
        child = subparsers.add_parser(command)
        child.add_argument("--original-data", required=True)
        child.add_argument("--checkpoint", required=True)
        child.add_argument("--rejected-v11-summary", required=True)
        child.add_argument("--row1241-root", required=True)
        child.add_argument("--row1314-root", required=True)
        if command == "publish":
            child.add_argument("--output-path", required=True)
    return parser


def main(argv: list[str] | None = None, *, gpu_inventory_reader=_live_gpu_inventory) -> int:
    args = _parser().parse_args(argv)
    run_id = "step-pref-packed-premise-v13"
    if args.command == "publish":
        output = Path(args.output_path)
        run_id = output.parent.name
    try:
        summary = authenticate_v13_premise(
            original_data=args.original_data, checkpoint=args.checkpoint,
            rejected_v11_summary=args.rejected_v11_summary,
            row1241_root=args.row1241_root, row1314_root=args.row1314_root,
            gpu_inventory_reader=gpu_inventory_reader, run_id=run_id,
        )
        print("VERIFIED_V13_BOUNDARY rejected=4864 accepted=4608")
        if args.command == "publish":
            digest = publish_pilot_summary(args.output_path, summary)
            absolute = Path(args.output_path).resolve(strict=True)
            print("VERIFIED_V13_PREMISE status=pass cap=4608")
            print(f"PILOT_SUMMARY_PATH={absolute} PILOT_SUMMARY_SHA256={digest}")
        return 0
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
