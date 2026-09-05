import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import publish_packed_v13_premise as premise


HISTORICAL_COMMIT = "0e69b54c02f5472c60c6ed64e5eebc2b46374a6c"
HISTORICAL_V11_SHA256 = (
    "7419aff8604072b57966fb4b460f99735cb6f7f2c2b386d2f46d7a28139a2c6f"
)
GPU = {
    "index": 0,
    "name": "NVIDIA H100 NVL",
    "uuid": "GPU-adaa4170-c124-2afc-114d-e1ea2bff7e8c",
    "memory_total_mib": 95830,
    "memory_used_mib": 0,
}
TOTAL_BYTES = 99949674496
MODULE_HISTORICAL_V11_SHA256 = premise.HISTORICAL_V11_SUMMARY_SHA256


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _write(path, value):
    path.write_bytes(_canonical(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_file(checkpoint):
    checkpoint_file = checkpoint / "config.json"
    checkpoint_file.write_bytes(b"{}\n")
    return {
        "relative_path": "config.json",
        "size": checkpoint_file.stat().st_size,
        "sha256": _sha(checkpoint_file),
        "load_root": str(checkpoint),
    }


def _manifest(checkpoint_file, source_sha, negative_sha):
    return {
        "schema_version": 3,
        "source_sha256": source_sha,
        "selected_row_indices": [0],
        "stage2a_checkpoint_path": checkpoint_file["load_root"],
        "checkpoint_files": [checkpoint_file],
        "tokenizer_config_identity": "a" * 64,
        "latent_size": 8,
        "assistant_plan_rendering": "adjacent-text-space-lf-v1",
        "generation_settings": {"device": "cuda:0"},
        "negative_sha256": negative_sha,
    }


def _audit(checkpoint, data, negative_path, manifest_path, negative_sha, *, row, clean, padded):
    record = {
        "source_index": 0,
        "sample_id": ("b" if row == 1241 else "c" if row == 1314 else "d") * 64,
        "clean_length": clean,
        "step_count": 6 if row != 1372 else 5,
        "branch_count": 24 if row != 1372 else 20,
        "unpadded_length": padded - 42,
        "padded_length": padded,
    }
    return {
        "schema_version": 1,
        "identity": {
            "checkpoint_path": str(checkpoint),
            "data_path": str(data),
            "image_root": str(data.parent),
            "negative_path": str(negative_path),
            "negative_manifest": str(manifest_path),
            "negative_sha256": negative_sha,
            "max_sequence_length": 8192,
            "assistant_plan_rendering": "adjacent-text-space-lf-v1",
            "attention_implementation": "sdpa",
            "vision_attention_implementation": "flash_attention_2",
        },
        "records": [record],
        "winner": record,
    }


def _capacity(checkpoint, sample_id, negative_sha, *, row, clean, padded, peak, status="pass"):
    return {
        "schema_version": 1,
        "status": status,
        "identity": {
            "checkpoint_path": str(checkpoint),
            "device": "cuda:0",
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "vision_attention_implementation": "flash_attention_2",
            "tuning_method": "dora",
            "gradient_checkpointing": True,
            "git_commit": HISTORICAL_COMMIT,
            "negative_sha256": negative_sha,
            "sample_id": sample_id,
        },
        "sample": {
            "source_row_index": row,
            "sample_id": sample_id,
            "clean_length": clean,
            "step_count": 6 if row != 1372 else 5,
            "branch_count": 24 if row != 1372 else 20,
            "packed_unpadded_length": padded - 42,
            "packed_padded_length": padded,
        },
        "losses": ({"ce": 1.0, "dpo": 2.0, "entropy": 3.0, "total": 4.0}
                   if status == "pass" else {"ce": None, "dpo": None, "entropy": None, "total": None}),
        "gradient_norm": 1.0 if status == "pass" else None,
        "memory": {
            "device_total_bytes": TOTAL_BYTES,
            "baseline_allocated_bytes": 1,
            "peak_allocated_bytes": peak - 1,
            "baseline_reserved_bytes": 1,
            "peak_reserved_bytes": peak,
        },
        "elapsed_seconds": 1.0,
        "errors": [] if status == "pass" else ["OutOfMemoryError: expected"],
    }


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    original = tmp_path / "TRAIN.jsonl"
    original.write_bytes(b'{"row":0}\n')
    source_sha = _sha(original)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint_file = _checkpoint_file(checkpoint)
    roots = {}
    details = {1241: (4000, 4864, 85115011072), 1314: (3733, 4608, 78922121216)}
    for row, (clean, padded, peak) in details.items():
        root = tmp_path / f"row-{row}"
        root.mkdir()
        negative = root / "negatives.jsonl"
        negative.write_bytes(b'{"negative":true}\n')
        manifest = root / f"row-{row}-negative-manifest.json"
        manifest_sha = _write(manifest, _manifest(checkpoint_file, source_sha, _sha(negative)))
        audit = root / f"row-{row}-packed-audit.json"
        audit_payload = _audit(checkpoint, original, negative, manifest, _sha(negative), row=row, clean=clean, padded=padded)
        audit_sha = _write(audit, audit_payload)
        sample_id = audit_payload["winner"]["sample_id"]
        capacity = root / f"row-{row}-capacity.json"
        capacity_sha = _write(capacity, _capacity(checkpoint, sample_id, _sha(negative), row=row, clean=clean, padded=padded, peak=peak))
        roots[row] = {"root": root, "manifest": manifest, "manifest_sha": manifest_sha, "audit": audit, "audit_sha": audit_sha, "capacity": capacity, "capacity_sha": capacity_sha}

    v11_root = tmp_path / "v11"
    v11_root.mkdir()
    negative = v11_root / "negatives.jsonl"
    negative.write_bytes(b'{"negative":true}\n')
    manifest = v11_root / "row-1372-negative-manifest.json"
    manifest_sha = _write(manifest, _manifest(checkpoint_file, source_sha, _sha(negative)))
    audit = v11_root / "row-1372-packed-audit.json"
    audit_payload = _audit(checkpoint, original, negative, manifest, _sha(negative), row=1372, clean=6065, padded=6784)
    audit_sha = _write(audit, audit_payload)
    capacity = v11_root / "row-1372-capacity.json"
    capacity_sha = _write(capacity, _capacity(checkpoint, audit_payload["winner"]["sample_id"], _sha(negative), row=1372, clean=6065, padded=6784, peak=99199483904, status="oom"))
    v11 = {
        "schema_version": 1,
        "run_id": "step-pref-packed-pilot-v11",
        "git_commit": HISTORICAL_COMMIT,
        "tracked_worktree_clean": True,
        "original_train": {"path": str(original), "sha256": source_sha},
        "checkpoint": {"path": str(checkpoint), "checkpoint_files": [checkpoint_file], "identity_source_manifests": [{"row_id": 1372, "path": str(manifest), "sha256": manifest_sha, "schema_version": 3}]},
        "gpus": [GPU],
        "probe_gpu": {"host_gpu_index": 0, "visible_device_index": 0, "name": GPU["name"], "uuid": GPU["uuid"]},
        "audits": [{"row_id": 1372, "path": str(audit), "sha256": audit_sha, "padded_length": 6784, "status": "admitted"}],
        "capacity_reports": [{"row_id": 1372, "path": str(capacity), "sha256": capacity_sha, "status": "oom", "device_total_bytes": TOTAL_BYTES, "peak_reserved_bytes": 99199483904, "reserved_ratio": 99199483904 / TOTAL_BYTES}],
        "admitted_row_ids": [1372],
        "ignored_over_cap_row_ids": [],
        "unprobed_admitted_row_ids": [],
        "failure": {"row_id": 1372, "observed_status": "oom", "reason": "capacity_probe_oom", "errors": ["OutOfMemoryError: expected"]},
        "validation": {"all_admitted_reports_pass": False, "all_admitted_reports_present": False, "at_least_two_admitted": True, "reserved_ratio_at_most_0_85": False, "row_1582_admitted": True},
        "overall_status": "fail",
    }
    v11_path = tmp_path / "v11-summary.json"
    _write(v11_path, v11)
    monkeypatch.setattr(premise, "HISTORICAL_V11_SUMMARY_SHA256", _sha(v11_path))
    monkeypatch.setattr(premise, "FROZEN_INPUT_SHA256", {
        "row1241_capacity": roots[1241]["capacity_sha"], "row1241_audit": roots[1241]["audit_sha"],
        "row1241_manifest": roots[1241]["manifest_sha"], "row1314_capacity": roots[1314]["capacity_sha"],
        "row1314_audit": roots[1314]["audit_sha"], "row1314_manifest": roots[1314]["manifest_sha"],
    })
    return {"original": original, "checkpoint": checkpoint, "v11": v11_path, "row1241": roots[1241]["root"], "row1314": roots[1314]["root"], "gpu": [{"index": 0, "name": GPU["name"], "uuid": GPU["uuid"], "total_memory_bytes": TOTAL_BYTES}]}


def _authenticate(evidence):
    return premise.authenticate_v13_premise(
        original_data=evidence["original"], checkpoint=evidence["checkpoint"],
        rejected_v11_summary=evidence["v11"], row1241_root=evidence["row1241"],
        row1314_root=evidence["row1314"], gpu_inventory_reader=lambda: evidence["gpu"],
    )


def test_authenticates_exact_boundary_and_normalizes_pilot_summary(evidence):
    summary = _authenticate(evidence)
    assert premise.HISTORICAL_GIT_COMMIT == HISTORICAL_COMMIT
    assert MODULE_HISTORICAL_V11_SHA256 == HISTORICAL_V11_SHA256
    assert set(summary) == set(premise.PILOT_SUMMARY_FIELDS)
    assert summary["candidate_cap"] == 4608
    assert summary["admitted_source_indices"] == [1314]
    assert summary["ignored_source_indices"] == [1241]
    assert summary["audits"][0]["packed_padded_length"] == 4864
    assert summary["capacity_reports"][0]["peak_reserved_fraction"] > 0.85
    assert summary["capacity_reports"][1]["peak_reserved_fraction"] <= 0.85
    assert summary["errors"] == []


def test_check_writes_nothing_and_publish_is_exclusive_and_strictly_reloadable(evidence, tmp_path, capsys):
    output = tmp_path / "pilot-summary.json"
    arguments = ["check", "--original-data", str(evidence["original"]), "--checkpoint", str(evidence["checkpoint"]), "--rejected-v11-summary", str(evidence["v11"]), "--row1241-root", str(evidence["row1241"]), "--row1314-root", str(evidence["row1314"])]
    assert premise.main(arguments, gpu_inventory_reader=lambda: evidence["gpu"]) == 0
    assert not output.exists()
    publish_arguments = arguments + ["--output-path", str(output)]
    publish_arguments[0] = "publish"
    assert premise.main(publish_arguments, gpu_inventory_reader=lambda: evidence["gpu"]) == 0
    assert premise.read_pilot_summary(output)["status"] == "pass"
    assert "PILOT_SUMMARY_PATH=" in capsys.readouterr().out
    with pytest.raises(ValueError, match="already exists"):
        premise.publish_pilot_summary(output, _authenticate(evidence))


@pytest.mark.parametrize("mutation, message", [
    (lambda payload: payload.__setitem__("unexpected", 1), "unknown or missing"),
    (lambda payload: payload["capacity_reports"][1].__setitem__("peak_reserved_fraction", float("nan")), "non-finite"),
])
def test_strict_decoder_rejects_unknown_fields_and_nonfinite_values(evidence, mutation, message):
    payload = _authenticate(evidence)
    mutation(payload)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode() + b"\n"
    with pytest.raises(ValueError, match=message):
        premise.decode_pilot_summary(raw)


def test_strict_json_rejects_duplicate_key_and_stale_gpu(evidence, tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON field"):
        premise.read_json_object(duplicate, "duplicate")
    evidence["gpu"][0]["uuid"] = "GPU-not-the-historical-device"
    with pytest.raises(ValueError, match="live GPU"):
        _authenticate(evidence)


@pytest.mark.parametrize("target", ["v11", "row1241_capacity", "row1314_manifest"])
def test_mutated_historical_input_is_rejected(evidence, target):
    if target == "v11":
        path = evidence["v11"]
    elif target == "row1241_capacity":
        path = evidence["row1241"] / "row-1241-capacity.json"
    else:
        path = evidence["row1314"] / "row-1314-negative-manifest.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        _authenticate(evidence)


def test_bound_reader_rejects_file_changed_after_open(monkeypatch, tmp_path):
    """The SHA-256 must cover the same regular file descriptor that is decoded."""
    path = tmp_path / "bound.json"
    path.write_bytes(b'{"before":true}\n')
    expected = _sha(path)
    original_open = os.open
    changed = False

    def open_then_replace(candidate, flags, *args):
        nonlocal changed
        descriptor = original_open(candidate, flags, *args)
        if Path(candidate) == path and not changed:
            changed = True
            replacement = original_open(path, os.O_WRONLY | os.O_TRUNC)
            try:
                os.write(replacement, b'{"after":true}\n')
            finally:
                os.close(replacement)
        return descriptor

    monkeypatch.setattr(premise.os, "open", open_then_replace)
    with pytest.raises(ValueError, match="changed|SHA-256"):
        premise._bound_file(path, expected, "test artifact")


@pytest.mark.parametrize("mutation, message", [
    (lambda summary: json.dumps(summary, indent=2).encode() + b"\n", "canonical"),
    (lambda summary: _canonical({**summary, "checkpoint_files": []}), "checkpoint_files"),
    (lambda summary: _canonical({**summary, "audits": [{**summary["audits"][0], "path": "relative-audit.json"}, summary["audits"][1]]}), "absolute"),
])
def test_strict_pilot_reload_requires_canonical_complete_identities(evidence, mutation, message):
    summary = _authenticate(evidence)
    with pytest.raises(ValueError, match=message):
        premise.decode_pilot_summary(mutation(summary))


def test_physical_gpu_zero_rejects_cuda_visible_device_remap():
    physical = [
        {"index": 0, "name": GPU["name"], "uuid": GPU["uuid"]},
        {"index": 1, "name": GPU["name"], "uuid": "GPU-second-device"},
    ]
    cuda_zero = {"uuid": "GPU-second-device", "total_memory_bytes": TOTAL_BYTES}
    with pytest.raises(ValueError, match="CUDA device 0.*physical GPU 0"):
        premise.physical_gpu_zero_from_observations(physical, cuda_zero)
