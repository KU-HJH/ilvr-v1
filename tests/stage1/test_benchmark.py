from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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


def test_canonical_evidence_strings_reject_embedded_nul():
    """Catches argv/config truncation at a later NUL-delimited boundary."""
    with pytest.raises(ValueError, match="NUL"):
        config_digest({"model": "safe\0forged"})

    with pytest.raises(ValueError, match="NUL"):
        evidence(git_commit="abc\0forged")


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


def test_high_cost_audit_rejects_blank_physical_lines(tmp_path):
    """Catches compacted row IDs that no longer identify physical JSONL rows."""
    image = _touch(tmp_path / "image.bin")
    row = {"image_input": image, "text_input": "x", "sequence_plan": []}
    data_path = tmp_path / "TRAIN.jsonl"
    data_path.write_text(json.dumps(row) + "\n\n" + json.dumps(row) + "\n")

    with pytest.raises(ValueError, match=r"blank.*line 2"):
        select_high_cost_probe_rows(data_path, 2, tmp_path / "audit.json")


def test_probe_jsonl_rejects_audit_from_another_image_root(tmp_path):
    """Catches replay of the right row list against a different image tree."""
    image_root = tmp_path / "images"
    image_root.mkdir()
    other_root = tmp_path / "other-images"
    other_root.mkdir()
    image = _touch(image_root / "image.bin")
    data = tmp_path / "TRAIN.jsonl"
    data.write_text(json.dumps({"image_input": image, "sequence_plan": []}) + "\n")
    audit = tmp_path / "audit.json"
    select_high_cost_probe_rows(data, 64, audit, image_root=image_root)

    import stage1.benchmark as benchmark
    with pytest.raises(ValueError, match="source data/image root/stat"):
        benchmark._atomic_write_probe_jsonl(
            audit, tmp_path / "probe.jsonl", data_path=data, image_root=other_root
        )


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
                "requested_count": 64,
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
    probe_payload = json.loads(probe_rows.read_text())
    probe_payload["audit_digest"] = config_digest(probe_payload)
    probe_rows.write_text(json.dumps(probe_payload), encoding="utf-8")
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


def test_attempt_seal_binds_top_level_fields_and_is_append_only(tmp_path):
    """Catches post-hoc PID/method/argv edits and attempt-file replacement."""
    completed, _, output_root = _run_probe(tmp_path, fail_batches="8")
    assert completed.returncode == 42
    attempt_path = next(output_root.glob("attempts/full/**/evidence/attempt.json"))
    payload = json.loads(attempt_path.read_text())

    import stage1.benchmark as benchmark

    payload["process_id"] += 1000
    attempt_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="attempt digest"):
        benchmark._load_probe_attempt(attempt_path)

    # A recorded attempt is immutable even if a caller presents identical fields.
    attempt_path.write_text(json.dumps({**payload, "process_id": payload["process_id"] - 1000}))
    args = benchmark._build_parser().parse_args(
        [
            "record-attempt", "--attempt-dir", str(attempt_path.parents[1]),
            "--attempt-id", payload["attempt_id"], "--method", "full",
            "--micro-batch", "8", "--world-size", "2",
            "--gradient-accumulation", "2", "--learning-rate", "1e-5",
            "--seed", "42", "--model", payload["launch_config"]["model"],
            "--source-data-path", payload["launch_config"]["data_path"],
            "--image-root", payload["launch_config"]["image_root"],
            "--deepspeed-config", payload["launch_config"]["deepspeed_config"],
            "--attention-implementation", "flash_attention_2",
            "--process-id", str(payload["process_id"] - 1000), "--exit-code", "42",
            "--max-steps", "3", "--output-dir", payload["output_dir"],
            "--stdout-path", payload["stdout_path"], "--stderr-path", payload["stderr_path"],
            "--probe-rows", payload["probe_rows"], "--", *payload["command"],
        ]
    )
    with pytest.raises(FileExistsError, match="append-only"):
        benchmark._record_probe_attempt(args)


def test_capacity_chain_rejects_duplicate_and_success_after_first_failure():
    """Catches re-search and a result selected from beyond the stopping point."""
    import stage1.benchmark as benchmark

    immutable = {
        "method": "full", "world_size": 2, "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5, "seed": 42, "model": "model",
        "data_path": "/data", "data_stat": {"device": 1}, "image_root": "/images",
        "deepspeed_config": "/zero3.json", "deepspeed_config_digest": "d" * 64,
        "attention_implementation": "flash_attention_2", "probe_rows": "/audit",
        "probe_row_indices": [1], "audit_digest": "a" * 64, "tuning_config": None,
    }

    def item(batch, exit_code, recorded):
        return Path(f"attempt-{recorded}.json"), {
            "micro_batch": batch, "exit_code": exit_code,
            "recorded_at_ns": recorded, "launch_config": immutable,
        }

    with pytest.raises(ValueError, match="duplicate"):
        benchmark._validated_capacity_chain(
            [item(8, 0, 1), item(8, 0, 2)], require_success=True
        )
    with pytest.raises(ValueError, match="after its first failure"):
        benchmark._validated_capacity_chain(
            [item(8, 0, 1), item(16, 42, 2), item(32, 0, 3)],
            require_success=True,
        )


def test_timed_validation_rejects_artifactless_and_plain_metric_evidence(tmp_path):
    """Catches sealing a successful metrics blob without final model topology."""
    import stage1.benchmark as benchmark

    plain = evidence(config_digest="c" * 64, checkpoint_bytes=0)
    plain_path = tmp_path / "plain.json"
    write_benchmark_evidence(plain_path, plain)
    with pytest.raises((TypeError, ValueError), match="producer evidence"):
        benchmark._load_evidence_without_expected(plain_path)

    config = {
        "schema_version": 1, "kind": "stage1-timed-launch-manifest",
        "method": "full", "tuning_config": None, "model": "model",
        "source_data_path": "/data", "data_path": "/data",
        "data_stat": {"device": 1, "inode": 2, "size": 3, "mtime_ns": 4},
        "image_root": "/images", "output_dir": str(tmp_path), "micro_batch": 8,
        "world_size": 2, "gradient_accumulation_steps": 2,
        "effective_global_batch": 32, "epochs": 3, "learning_rate": 1e-5,
        "seed": 42, "max_steps": 0, "final_save": True,
        "deepspeed_config": "/zero3.json", "deepspeed_config_digest": "d" * 64,
        "attention_implementation": "flash_attention_2", "save_strategy": "no",
        "audit_digest": "a" * 64, "probe_rows": "/audit",
        "probe_row_indices": [7, 3], "maximum_config_digest": "b" * 64,
    }
    config["config_digest"] = config_digest(config)
    metrics = evidence(config_digest=config["config_digest"], checkpoint_bytes=0)
    artifact = {
        "schema_version": 1, "kind": "stage1-artifact-topology-witness",
        "method": "full", "artifact_kind": "none", "output_dir": str(tmp_path),
        "final_save": True, "files": [], "checkpoint_bytes": 0,
    }
    artifact["artifact_manifest_digest"] = config_digest(artifact)
    seal = {
        "schema_version": 1, "kind": "stage1-producer-evidence",
        "run_config": config,
        "benchmark_evidence": benchmark._evidence_to_dict(metrics),
        "artifact_manifest": artifact,
        "artifact_manifest_digest": artifact["artifact_manifest_digest"],
    }
    seal["producer_digest"] = config_digest(seal)
    seal_path = tmp_path / "producer.json"
    seal_path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="topology"):
        benchmark._load_producer_evidence(seal_path)


def test_compare_maxima_exposes_rank_local_ten_step_memory(monkeypatch):
    """Catches a maximum comparison that drops the required rank witnesses."""
    import stage1.benchmark as benchmark

    maxima = {
        "full.json": {"method": "full", "selected_micro_batch": 8,
                      "effective_global_batch": 32, "long_smoke_steps": 10,
                      "first_failure": None, "rank_peak_allocated_bytes": [10, 11],
                      "rank_peak_reserved_bytes": [12, 13], "config_digest": "a" * 64},
        "dora.json": {"method": "dora", "selected_micro_batch": 16,
                      "effective_global_batch": 64, "long_smoke_steps": 10,
                      "first_failure": None, "rank_peak_allocated_bytes": [20, 21],
                      "rank_peak_reserved_bytes": [22, 23], "config_digest": "b" * 64},
    }
    monkeypatch.setattr(benchmark, "_load_maximum", lambda path: maxima[path.name])
    compared = benchmark._compare_maxima(["full.json", "dora.json"])
    assert compared["methods"]["full"]["rank_peak_allocated_bytes"] == [10, 11]
    assert compared["methods"]["dora"]["rank_peak_reserved_bytes"] == [22, 23]


def test_probe_does_not_fallback_without_a_confirmed_initial_eight_oom(tmp_path):
    """Catches an implicit batch reduction after an unclassified failure."""
    completed, records, output_root = _run_probe(
        tmp_path, fail_batches="8", extra=("--fallback", "4")
    )

    assert completed.returncode != 0
    assert "confirmed" in completed.stderr
    assert not list(records.glob("*.argv0"))
    assert not output_root.exists()


def test_prepare_timed_run_restores_full_data_source_not_probe_subset(tmp_path, monkeypatch):
    """Catches an approved three-epoch run accidentally using only probe rows."""
    original_data = tmp_path / "complete TRAIN.jsonl"
    probe_data = tmp_path / "64 high-cost rows.jsonl"
    launch_config = {
        "method": "full",
        "tuning_config": None,
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
        "train_data_path": str(probe_data),
        "data_stat": {"device": 1, "inode": 2, "size": 3, "mtime_ns": 4},
        "image_root": str(tmp_path / "images"),
        "deepspeed_config": str(REPO_ROOT / "configs/config_stage3.json"),
        "deepspeed_config_digest": "d" * 64,
        "attention_implementation": "flash_attention_2",
        "probe_rows": str(tmp_path / "high-cost.json"),
        "probe_row_indices": [1, 0],
        "audit_digest": "a" * 64,
        "run_config_digest": "c" * 64,
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
    output_dir = tmp_path / "timed full"
    import stage1.benchmark as benchmark
    monkeypatch.setattr(benchmark, "_load_maximum", lambda path: maximum)
    benchmark._prepare_timed_run(
        SimpleNamespace(
            method="full", maximum="sealed-maximum.json", output_dir=str(output_dir),
            epochs=3, learning_rate=1e-5, gradient_accumulation=2, world_size=2,
        )
    )
    manifest = json.loads(
        (output_dir / "launch-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["data_path"] == str(original_data)
    assert manifest["data_path"] != str(probe_data)
