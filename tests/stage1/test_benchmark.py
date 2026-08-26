from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from stage1.benchmark import (
    Stage1BenchmarkEvidence,
    capacity_candidates,
    fallback_candidates_after_initial_failure,
    load_verified_evidence,
    select_high_cost_probe_rows,
    write_benchmark_evidence,
    config_digest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE = REPO_ROOT / "run_stage1_batch_probe.sh"


def evidence(**overrides):
    values = {
        "status": "success",
        "method": "full",
        "micro_batch": 8,
        "world_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_global_batch": 32,
        "optimizer_steps": 3,
        "samples_seen": 96,
        "model_load_seconds": 1.0,
        "trainer_seconds": 2.0,
        "total_seconds": 4.0,
        "final_save_seconds": 1.0,
        "samples_per_second": 24.0,
        "steps_per_second": 0.75,
        "rank_peak_allocated_bytes": (10, 11),
        "rank_peak_reserved_bytes": (12, 13),
        "trainable_parameters": 20,
        "total_parameters": 100,
        "checkpoint_bytes": 200,
        "final_loss": 0.5,
        "representative_gradient_norm": 0.25,
        "git_commit": "abc1234",
        "dependency_versions": {"torch": "2.7.1"},
        "gpu_names": ("B200", "B200"),
        "config_digest": "a" * 64,
        "probe_row_indices": (7, 3),
    }
    values.update(overrides)
    return Stage1BenchmarkEvidence(**values)


def test_capacity_candidates_follow_only_the_approved_order():
    """Catches speculative lower batches or retries in the primary search."""
    assert capacity_candidates(start=8, max_per_gpu=64) == (8, 16, 32, 64)
    assert fallback_candidates_after_initial_failure(start=8) == (4, 2, 1)
    with pytest.raises(ValueError, match="initial batch 8"):
        fallback_candidates_after_initial_failure(start=16)


def test_exact_module_cli_runs_without_repository_pythonpath_override():
    """Catches package initialization preventing the documented benchmark CLI."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
            "-m",
            "src.stage1.benchmark",
            "capacity-candidates",
            "--start",
            "8",
            "--max-per-gpu",
            "64",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "8 16 32 64"


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("micro_batch", True, "micro_batch"),
        ("trainer_seconds", math.inf, "trainer_seconds"),
        ("final_loss", math.nan, "final_loss"),
        ("checkpoint_bytes", -1, "checkpoint_bytes"),
        ("rank_peak_allocated_bytes", (1,), "world_size"),
        ("gpu_names", ("B200",), "world_size"),
        ("effective_global_batch", 31, "effective_global_batch"),
    ],
)
def test_evidence_rejects_wrong_types_nonfinite_values_and_invariants(
    field, bad_value, message
):
    """Catches malformed metrics being published as reusable evidence."""
    with pytest.raises((TypeError, ValueError), match=message):
        evidence(**{field: bad_value})


def test_evidence_write_is_atomic_and_verified_reads_are_reused(tmp_path, monkeypatch):
    """Catches partial JSON, stale digests, and repeated immutable-file reads."""
    path = tmp_path / "nested" / "evidence.json"
    expected = evidence()
    write_benchmark_evidence(path, expected)
    assert not list(path.parent.glob("*.tmp"))

    import stage1.benchmark as benchmark

    benchmark._VERIFIED_EVIDENCE_CACHE.clear()
    reads = 0
    original = benchmark._read_json_once

    def counted(selected):
        nonlocal reads
        reads += 1
        return original(selected)

    monkeypatch.setattr(benchmark, "_read_json_once", counted)
    assert load_verified_evidence(path, "b" * 64) is None
    assert load_verified_evidence(path, expected.config_digest) == expected
    assert load_verified_evidence(path, expected.config_digest) is not None
    assert reads == 1


def _touch(path: Path) -> str:
    path.write_bytes(b"not an image and must never be opened")
    return str(path)


def test_high_cost_audit_ranks_metadata_and_reuses_unchanged_stat_witness(tmp_path):
    """Catches image-content reads, unstable row identity, and duplicate audits."""
    images = [_touch(tmp_path / f"image-{index}.bin") for index in range(6)]
    rows = [
        {
            "id": "low",
            "image_input": images[0],
            "text_input": "short",
            "sequence_plan": [
                {"type": "latent", "helper_image": images[1]},
            ],
        },
        {
            "id": "high",
            "image_input": [images[2], images[3]],
            "text_input": "one two three four five six",
            "sequence_plan": [
                {"type": "text", "content": "seven eight"},
                {"type": "latent", "helper_image": images[4]},
                {"type": "latent", "helper_image": images[5]},
            ],
        },
    ]
    data_path = tmp_path / "TRAIN.jsonl"
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    audit_path = tmp_path / "high-cost.json"

    selected = select_high_cost_probe_rows(data_path, 2, audit_path)
    assert selected == (1, 0)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["input_stat"]["size"] == data_path.stat().st_size
    assert [row["source_row_index"] for row in audit["selected_rows"]] == [1, 0]

    data_path.chmod(0)
    try:
        assert select_high_cost_probe_rows(data_path, 2, audit_path) == (1, 0)
    finally:
        data_path.chmod(0o600)


def test_high_cost_audit_fails_clearly_for_missing_references(tmp_path):
    """Catches malformed CoMT metadata being silently accepted by a probe."""
    data_path = tmp_path / "TRAIN.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "image_input": "missing-original.png",
                "text_input": "text",
                "sequence_plan": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match=r"source_row_index=0.*original"):
        select_high_cost_probe_rows(data_path, 1, tmp_path / "audit.json")


def _write_fake_launcher(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
micro=
for ((i=1; i<=$#; i++)); do
    if [[ ${!i} == --micro-batch ]]; then
        j=$((i + 1)); micro=${!j}
    fi
done
printf '%s\\0' "$BASHPID" "$@" > "$STUB_RECORD_DIR/$BASHPID.argv0"
if [[ ,${STUB_FAIL_BATCHES:-}, == *,$micro,* ]]; then
    exit 42
fi
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run_probe(tmp_path, *, fail_batches, extra=()):
    fake_launcher = tmp_path / "fake stage1 launcher"
    _write_fake_launcher(fake_launcher)
    records = tmp_path / "process records"
    records.mkdir()
    data_path = tmp_path / "probe.jsonl"
    data_path.write_text("{}\n", encoding="utf-8")
    image_root = tmp_path / "images"
    image_root.mkdir()
    probe_rows = tmp_path / "rows.json"
    probe_rows.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "stage1-high-cost-row-audit",
                "data_path": str(data_path.resolve()),
                "image_root": str(image_root.resolve()),
                "requested_count": 1,
                "row_count": 1,
                "input_stat": {
                    "device": data_path.stat().st_dev,
                    "inode": data_path.stat().st_ino,
                    "size": data_path.stat().st_size,
                    "mtime_ns": data_path.stat().st_mtime_ns,
                },
                "selected_rows": [
                    {
                        "source_row_index": 0,
                        "token_estimate": 0,
                        "original_image_count": 0,
                        "helper_image_count": 0,
                        "latent_block_count": 0,
                        "example": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "probe outputs;literal"
    env = os.environ.copy()
    env.update(
        {
            "STAGE1_LAUNCHER": str(fake_launcher),
            "STAGE1_BENCHMARK_PYTHON": sys.executable,
            "STUB_RECORD_DIR": str(records),
            "STUB_FAIL_BATCHES": fail_batches,
            "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'src'}",
        }
    )
    completed = subprocess.run(
        [
            str(PROBE),
            "--method",
            "full",
            "--model",
            "model path;literal",
            "--data-path",
            data_path,
            "--image-root",
            image_root,
            "--output-root",
            output_root,
            "--probe-rows",
            probe_rows,
            "--start",
            "8",
            "--max-per-gpu",
            "64",
            *extra,
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, records, output_root


def test_probe_uses_fresh_processes_unique_paths_and_stops_at_first_failure(tmp_path):
    """Catches mocked isolation, dirty-process reuse, retry, and output collision."""
    completed, records, output_root = _run_probe(tmp_path, fail_batches="32")

    assert completed.returncode != 0
    assert "requires-classification" in completed.stderr
    launches = sorted(records.glob("*.argv0"))
    assert len(launches) == 3
    parsed = [path.read_bytes().split(b"\0")[:-1] for path in launches]
    pids = {parts[0].decode() for parts in parsed}
    assert len(pids) == 3
    batches = {
        parts[parts.index(b"--micro-batch") + 1].decode() for parts in parsed
    }
    assert batches == {"8", "16", "32"}
    for parts in parsed:
        assert parts[parts.index(b"--max-steps") + 1] == b"3"
        assert b"--no-final-save" in parts
        assert b"--max-steps" in parts
    attempts = list(output_root.glob("attempts/full/micro-batch-*/attempt-*/evidence/attempt.json"))
    assert len(attempts) == 3
    assert len({path.parent.parent.parent for path in attempts}) == 3
    failed = [json.loads(path.read_text()) for path in attempts if json.loads(path.read_text())["exit_code"]]
    assert len(failed) == 1
    assert failed[0]["micro_batch"] == 32
    assert failed[0]["requires_oom_classification"] is True


def test_probe_does_not_fallback_without_a_confirmed_initial_eight_oom(tmp_path):
    """Catches an implicit batch reduction after an unclassified failure."""
    completed, records, output_root = _run_probe(
        tmp_path, fail_batches="8", extra=("--fallback", "4")
    )

    assert completed.returncode != 0
    assert "confirmed" in completed.stderr
    assert not list(records.glob("*.argv0"))
    assert not output_root.exists()


def test_prepare_timed_run_restores_full_data_source_not_probe_subset(tmp_path):
    """Catches an approved three-epoch run accidentally using only probe rows."""
    original_data = tmp_path / "complete TRAIN.jsonl"
    probe_data = tmp_path / "64 high-cost rows.jsonl"
    launch_config = {
        "method": "full",
        "micro_batch": 16,
        "world_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_global_batch": 64,
        "learning_rate": 1e-5,
        "seed": 42,
        "max_steps": 3,
        "final_save": False,
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "data_path": str(original_data),
        "image_root": str(tmp_path / "images"),
        "deepspeed_config": str(REPO_ROOT / "configs/config_stage3.json"),
        "attention_implementation": "flash_attention_2",
        "probe_rows": str(tmp_path / "high-cost.json"),
        "command": [
            str(REPO_ROOT / "run_stage1_training.sh"),
            "--model",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "--data-path",
            str(probe_data),
            "--image-root",
            str(tmp_path / "images"),
            "--deepspeed-config",
            str(REPO_ROOT / "configs/config_stage3.json"),
        ],
    }
    maximum = {
        "schema_version": 1,
        "kind": "stage1-capacity-maximum",
        "status": "success",
        "method": "full",
        "selected_micro_batch": 16,
        "world_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_global_batch": 64,
        "long_smoke_steps": 10,
        "long_smoke_attempt": "small-long-smoke.json",
        "selected_attempt": "small-attempt.json",
        "first_failure": None,
        "launch_config": launch_config,
    }
    maximum["config_digest"] = config_digest(maximum)
    maximum_path = tmp_path / "maximum.json"
    maximum_path.write_text(json.dumps(maximum), encoding="utf-8")
    output_dir = tmp_path / "timed full"

    completed = subprocess.run(
        [
            str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
            "-m",
            "src.stage1.benchmark",
            "prepare-timed-run",
            "--method",
            "full",
            "--maximum",
            str(maximum_path),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "3",
            "--learning-rate",
            "1e-5",
            "--gradient-accumulation",
            "2",
            "--world-size",
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(
        (output_dir / "launch-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["data_path"] == str(original_data)
    assert manifest["data_path"] != str(probe_data)
