from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import hashlib
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from stage1.benchmark import (
    Stage1BenchmarkEvidence,
    archive_failed_distributed_attempt,
    capacity_candidates,
    fallback_candidates_after_initial_failure,
    load_verified_evidence,
    select_high_cost_probe_rows,
    _write_legacy_benchmark_evidence,
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

    with pytest.raises(ValueError, match="NUL"):
        config_digest({"safe\0forged": "value"})


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


def test_official_verified_loader_rejects_raw_metrics_without_producer_seals(tmp_path):
    """Catches Task 15 reuse that bypasses artifact and resume producer seals."""
    path = tmp_path / "nested" / "evidence.json"
    expected = evidence()
    _write_legacy_benchmark_evidence(path, expected)
    assert not list(path.parent.glob("*.tmp"))

    with pytest.raises(ValueError, match="producer envelope"):
        load_verified_evidence(path, expected.config_digest)


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
if [[ -n ${STUB_START_DIR:-} ]]; then
    touch "$STUB_START_DIR/$BASHPID"
fi
if [[ -n ${STUB_HOLD_SECONDS:-} ]]; then
    sleep "$STUB_HOLD_SECONDS"
fi
if [[ ,${STUB_FAIL_BATCHES:-}, == *,$micro,* ]]; then
    exit 42
fi
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run_probe(tmp_path, *, fail_batches, extra=(), poison_benchmark_python=False):
    fake_launcher = tmp_path / "fake stage1 launcher"
    _write_fake_launcher(fake_launcher)
    records = tmp_path / "process records"
    records.mkdir(exist_ok=True)
    data_path = tmp_path / "probe.jsonl"
    if not data_path.exists():
        data_path.write_text("{}\n", encoding="utf-8")
    image_root = tmp_path / "images"
    image_root.mkdir(exist_ok=True)
    probe_rows = tmp_path / "rows.json"
    if not probe_rows.exists():
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
    probe_payload = json.loads(probe_rows.read_text())
    identity = {
        "model": "model path;literal", "source_data_path": str(data_path.resolve()),
        "data_stat": probe_payload["input_stat"], "image_root": str(image_root.resolve()),
        "world_size": 2, "gradient_accumulation_steps": 2, "epochs": 3,
        "learning_rate": 1e-5, "seed": 42,
        "deepspeed_config": str((REPO_ROOT / "configs/config_stage3.json").resolve()),
        "deepspeed_config_digest": hashlib.sha256(
            (REPO_ROOT / "configs/config_stage3.json").read_bytes()
        ).hexdigest(),
        "attention_implementation": "flash_attention_2", "save_strategy": "no",
        "audit_digest": probe_payload["audit_digest"], "probe_row_indices": [0],
    }
    prerequisite = {
        "schema_version": 1, "kind": "stage1-smoke-prerequisite-chain",
        "shared_identity": identity, "shared_identity_digest": config_digest(identity),
        "producer_digests": {"full": "1" * 64, "dora": "2" * 64},
        "resume_witness_digests": {"full": "3" * 64, "dora": "4" * 64},
    }
    prerequisite["prerequisite_chain_digest"] = config_digest(prerequisite)
    prerequisite_path = tmp_path / "prerequisite-chain.json"
    if not prerequisite_path.exists():
        prerequisite_path.write_text(json.dumps(prerequisite))
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
    poison_marker = tmp_path / "wrong benchmark python used"
    if poison_benchmark_python:
        poison_bin = tmp_path / "wrong python bin"
        poison_bin.mkdir(exist_ok=True)
        for name in ("python", "accelerate"):
            executable = poison_bin / name
            executable.write_text(
                "#!/usr/bin/env bash\n"
                "touch \"$POISON_MARKER\"\n"
                "exit 97\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
        env["PATH"] = f"{poison_bin}{os.pathsep}{env['PATH']}"
        env["STAGE1_BENCHMARK_PYTHON"] = str(poison_bin / "python")
        env["POISON_MARKER"] = str(poison_marker)
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
            "--prerequisite-chain",
            prerequisite_path,
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
    return completed, records, output_root, poison_marker


def test_probe_uses_fresh_processes_unique_paths_and_stops_at_first_failure(tmp_path):
    """Catches mocked isolation, dirty-process reuse, retry, and output collision."""
    completed, records, output_root, _ = _run_probe(tmp_path, fail_batches="32")

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


def test_probe_pins_benchmark_python_when_path_and_override_are_wrong(tmp_path):
    """Catches the probe controller escaping the mandatory repository venv."""

    completed, records, output_root, poison_marker = _run_probe(
        tmp_path,
        fail_batches="8",
        poison_benchmark_python=True,
    )

    assert completed.returncode == 42, completed.stderr
    assert not poison_marker.exists()
    assert len(list(records.glob("*.argv0"))) == 1
    assert len(list(output_root.glob("attempts/full/**/evidence/attempt.json"))) == 1


@pytest.mark.parametrize("failed_batch", [8, 16])
def test_integration_failure_retries_same_batch_append_only_then_continues(
    tmp_path, failed_batch
):
    """Catches import/harness failures being treated as OOMs or overwritten."""

    failed, _, output_root, _ = _run_probe(
        tmp_path, fail_batches=str(failed_batch)
    )
    assert failed.returncode == 42
    failed_attempt = next(
        output_root.glob(
            f"attempts/full/micro-batch-{failed_batch}/*/evidence/attempt.json"
        )
    )
    original_attempt_bytes = failed_attempt.read_bytes()

    import stage1.benchmark as benchmark

    benchmark._classify_failure(failed_attempt, "integration_failure")
    classification_path = benchmark._classification_path(failed_attempt)
    classification_bytes = classification_path.read_bytes()
    benchmark._classify_failure(failed_attempt, "integration_failure")
    assert classification_path.read_bytes() == classification_bytes
    assert not benchmark._confirmed_capacity_oom(failed_attempt)

    retried, records, _, _ = _run_probe(tmp_path, fail_batches="")

    assert retried.returncode == 0, retried.stderr
    assert failed_attempt.read_bytes() == original_attempt_bytes
    attempts = [
        (path, benchmark._load_probe_attempt(path))
        for path in benchmark._attempt_files(output_root, "full")
    ]
    expected_batches = (
        [8, 8, 16, 32, 64]
        if failed_batch == 8
        else [8, 16, 16, 32, 64]
    )
    expected_exits = (
        [42, 0, 0, 0, 0]
        if failed_batch == 8
        else [0, 42, 0, 0, 0]
    )
    assert [attempt["micro_batch"] for _, attempt in attempts] == expected_batches
    assert [attempt["exit_code"] for _, attempt in attempts] == expected_exits
    selected, capacity_failure = benchmark._validated_capacity_chain(
        attempts, require_success=True
    )
    assert selected is not None and selected[1]["micro_batch"] == 64
    assert capacity_failure is None
    assert len(list(records.glob("*.argv0"))) == 5


def test_integration_failure_never_authorizes_capacity_fallback(tmp_path):
    """Catches a harness failure silently lowering the requested batch."""

    failed, records, output_root, _ = _run_probe(tmp_path, fail_batches="8")
    assert failed.returncode == 42
    failed_attempt = next(output_root.glob("attempts/full/**/evidence/attempt.json"))

    import stage1.benchmark as benchmark

    benchmark._classify_failure(failed_attempt, "integration_failure")
    before = sorted(records.glob("*.argv0"))
    fallback, _, _, _ = _run_probe(
        tmp_path, fail_batches="", extra=("--fallback", "4")
    )

    assert fallback.returncode != 0
    assert "confirmed capacity OOM" in fallback.stderr
    assert sorted(records.glob("*.argv0")) == before


def test_conflicting_failure_classifications_publish_exactly_once(
    tmp_path, monkeypatch
):
    """Catches a concurrent integration/OOM classification overwrite."""

    failed, _, output_root, _ = _run_probe(tmp_path, fail_batches="8")
    assert failed.returncode == 42
    failed_attempt = next(output_root.glob("attempts/full/**/evidence/attempt.json"))

    import stage1.benchmark as benchmark

    barrier = __import__("threading").Barrier(2)
    original_exclusive = benchmark._exclusive_write_json

    def synchronized_exclusive(path, payload):
        barrier.wait(timeout=5)
        return original_exclusive(path, payload)

    monkeypatch.setattr(
        benchmark, "_exclusive_write_json", synchronized_exclusive
    )

    def classify(value):
        try:
            benchmark._classify_failure(failed_attempt, value)
        except Exception as error:  # the losing conflicting publication must fail
            return error
        return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(classify, ("integration_failure", "capacity_oom"))
        )

    assert sum(outcome is None for outcome in outcomes) == 1
    published = benchmark._failure_classification(failed_attempt)
    successful = (
        "integration_failure" if outcomes[0] is None else "capacity_oom"
    )
    assert published == successful


def test_identical_classification_is_invisible_until_atomic_publication(
    tmp_path, monkeypatch
):
    """Catches readers observing an empty/partial final classification inode."""

    failed, _, output_root, _ = _run_probe(tmp_path, fail_batches="8")
    assert failed.returncode == 42
    failed_attempt = next(output_root.glob("attempts/full/**/evidence/attempt.json"))

    import stage1.benchmark as benchmark

    target = benchmark._classification_path(failed_attempt)
    first_link_ready = __import__("threading").Event()
    release_first_link = __import__("threading").Event()
    calls = 0
    calls_lock = __import__("threading").Lock()
    original_link = os.link

    def controlled_link(source, destination, *args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            first_link_ready.set()
            assert release_first_link.wait(timeout=5)
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(benchmark.os, "link", controlled_link)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            benchmark._classify_failure, failed_attempt, "integration_failure"
        )
        assert first_link_ready.wait(timeout=5)
        assert not target.exists()
        second = pool.submit(
            benchmark._classify_failure, failed_attempt, "integration_failure"
        )
        second.result(timeout=5)
        release_first_link.set()
        first.result(timeout=5)

    assert benchmark._failure_classification(failed_attempt) == "integration_failure"
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_concurrent_probe_controllers_cannot_launch_duplicate_retry(tmp_path):
    """Catches two controllers passing the same candidate gate concurrently."""

    failed, records, output_root, _ = _run_probe(tmp_path, fail_batches="8")
    assert failed.returncode == 42
    failed_attempt = next(output_root.glob("attempts/full/**/evidence/attempt.json"))

    import stage1.benchmark as benchmark

    benchmark._classify_failure(failed_attempt, "integration_failure")
    fake_launcher = tmp_path / "fake stage1 launcher"
    starts = tmp_path / "controller starts"
    starts.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "STAGE1_LAUNCHER": str(fake_launcher),
            "STUB_RECORD_DIR": str(records),
            "STUB_FAIL_BATCHES": "",
            "STUB_START_DIR": str(starts),
            "STUB_HOLD_SECONDS": "0.5",
            "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'src'}",
        }
    )
    command = list(failed.args)
    first = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5
    while not list(starts.iterdir()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert list(starts.iterdir()), "first controller did not reach its launcher"
    second = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    first_stdout, first_stderr = first.communicate(timeout=10)

    assert first.returncode == 0, first_stderr
    assert second.returncode != 0
    assert "probe execution is in progress" in second.stderr
    assert stat.S_IMODE((output_root / ".probe-execution.lock").stat().st_mode) == 0o600
    attempts = [
        benchmark._load_probe_attempt(path)
        for path in benchmark._attempt_files(output_root, "full")
    ]
    assert [item["micro_batch"] for item in attempts].count(8) == 2


def test_probe_execution_lock_rejects_named_inode_replacement(tmp_path):
    """Catches a replacement lock disconnecting the held controller lease."""

    import stage1.benchmark as benchmark

    lock_path = tmp_path / ".probe-execution.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        benchmark._acquire_probe_execution_lock(descriptor, lock_path)
        replaced = tmp_path / "replaced.lock"
        lock_path.rename(replaced)
        lock_path.write_bytes(b"replacement")
        lock_path.chmod(0o600)

        with pytest.raises(ValueError, match="mode-0600 regular file"):
            benchmark._validate_probe_execution_lock(descriptor, lock_path)
    finally:
        os.close(descriptor)


def test_attempt_seal_binds_top_level_fields_and_is_append_only(tmp_path):
    """Catches post-hoc PID/method/argv edits and attempt-file replacement."""
    completed, _, output_root, _ = _run_probe(tmp_path, fail_batches="8")
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


def test_attempt_rejects_data_argv_not_equal_to_materialized_audit_subset(tmp_path):
    """Catches swapping the 64-row probe input for the full source JSONL."""
    completed, _, output_root, _ = _run_probe(tmp_path, fail_batches="8")
    assert completed.returncode == 42
    attempt_path = next(output_root.glob("attempts/full/**/evidence/attempt.json"))
    payload = json.loads(attempt_path.read_text())
    command = payload["command"]
    command[command.index("--data-path") + 1] = payload["launch_config"]["data_path"]
    payload["launch_config"]["command"] = command
    payload["config_digest"] = config_digest(payload["launch_config"])
    payload["attempt_digest"] = config_digest(
        {key: value for key, value in payload.items() if key != "attempt_digest"}
    )
    attempt_path.write_text(json.dumps(payload))

    import stage1.benchmark as benchmark
    with pytest.raises(ValueError, match="materialized audited probe"):
        benchmark._load_probe_attempt(attempt_path)


def _rewrite_probe_attempt_argv(attempt_path, replacements):
    payload = json.loads(attempt_path.read_text())
    for command in (payload["command"], payload["launch_config"]["command"]):
        for flag, value in replacements.items():
            command[command.index(flag) + 1] = value
    payload["config_digest"] = config_digest(payload["launch_config"])
    payload["attempt_digest"] = config_digest(
        {key: item for key, item in payload.items() if key != "attempt_digest"}
    )
    attempt_path.write_text(json.dumps(payload))
    return payload


def _probe_output_path_expectations(payload):
    output_dir = Path(payload["output_dir"])
    return {
        "--data-path": payload["launch_config"]["train_data_path"],
        "--output-dir": payload["output_dir"],
        "--log-file": payload["stdout_path"].replace("stdout.log", "train.log"),
        "--benchmark-run-config": str(
            output_dir.parent / "evidence" / "run-config.json"
        ),
        "--producer-evidence": str(
            output_dir.parent / "evidence" / "producer-evidence.json"
        ),
    }


@pytest.fixture
def pre_fix_preserved_relative_probe_attempt(tmp_path, monkeypatch):
    """Materialize the exact relative-path shape of the retained old attempt."""

    completed, _, output_root, _ = _run_probe(tmp_path, fail_batches="8")
    assert completed.returncode == 42
    attempt_path = next(output_root.glob("attempts/full/**/evidence/attempt.json"))

    import stage1.benchmark as benchmark

    monkeypatch.setattr(benchmark, "_REPOSITORY_ROOT", tmp_path, raising=False)
    payload = json.loads(attempt_path.read_text())
    expected_paths = _probe_output_path_expectations(payload)
    relative_paths = {
        flag: str(Path(expected).relative_to(tmp_path))
        for flag, expected in expected_paths.items()
    }
    assert set(relative_paths) == {
        "--data-path",
        "--output-dir",
        "--log-file",
        "--benchmark-run-config",
        "--producer-evidence",
    }
    _rewrite_probe_attempt_argv(attempt_path, relative_paths)
    return attempt_path, relative_paths


def test_pre_fix_all_repository_relative_probe_paths_can_be_classified(
    pre_fix_preserved_relative_probe_attempt,
):
    """Catches the producer's complete old relative argv becoming incompatible."""

    attempt_path, relative_paths = pre_fix_preserved_relative_probe_attempt
    import stage1.benchmark as benchmark

    original_attempt = attempt_path.read_bytes()

    loaded = benchmark._load_probe_attempt(attempt_path)
    benchmark._classify_failure(attempt_path, "integration_failure")

    for flag, value in relative_paths.items():
        assert loaded["command"][loaded["command"].index(flag) + 1] == value
    for flag in ("--source-data-path", "--image-root", "--deepspeed-config"):
        value = loaded["command"][loaded["command"].index(flag) + 1]
        assert Path(value).is_absolute()
    assert Path(loaded["command"][0]).is_absolute()
    assert attempt_path.read_bytes() == original_attempt
    assert benchmark._failure_classification(attempt_path) == "integration_failure"


@pytest.mark.parametrize("alias_kind", ["dot", "parent", "outside", "symlink", "hardlink"])
def test_pre_fix_probe_argv_compatibility_rejects_noncanonical_aliases(
    tmp_path, monkeypatch, alias_kind
):
    """Catches path normalization reopening audited-probe swap aliases."""

    completed, _, output_root, _ = _run_probe(tmp_path, fail_batches="8")
    assert completed.returncode == 42
    attempt_path = next(output_root.glob("attempts/full/**/evidence/attempt.json"))

    import stage1.benchmark as benchmark

    monkeypatch.setattr(benchmark, "_REPOSITORY_ROOT", tmp_path, raising=False)
    expected = Path(
        json.loads(attempt_path.read_text())["launch_config"]["train_data_path"]
    )
    canonical = str(expected.relative_to(tmp_path))
    if alias_kind == "dot":
        alias = f"./{canonical}"
    elif alias_kind == "parent":
        alias = str(Path(canonical).parent / "nested" / ".." / expected.name)
    elif alias_kind == "outside":
        alias = str(Path("..") / tmp_path.name / canonical)
    else:
        alias_path = tmp_path / f"probe-{alias_kind}.jsonl"
        if alias_kind == "symlink":
            alias_path.symlink_to(expected)
        else:
            os.link(expected, alias_path)
        alias = alias_path.name
    _rewrite_probe_attempt_argv(attempt_path, {"--data-path": alias})

    with pytest.raises(ValueError, match="materialized audited probe"):
        benchmark._load_probe_attempt(attempt_path)


def test_probe_argv_compatibility_rejects_relative_expected_outside_repository(
    tmp_path, monkeypatch
):
    import stage1.benchmark as benchmark

    repository_root = tmp_path / "checkout"
    repository_root.mkdir()
    expected = tmp_path / "outside" / "probe-input.jsonl"
    monkeypatch.setattr(benchmark, "_REPOSITORY_ROOT", repository_root)

    assert not benchmark._probe_argv_path_matches_sealed_absolute(
        "../outside/probe-input.jsonl", str(expected)
    )


@pytest.mark.parametrize(
    ("flag", "sealed_field"),
    [
        ("--source-data-path", "data_path"),
        ("--image-root", "image_root"),
        ("--deepspeed-config", "deepspeed_config"),
    ],
)
def test_probe_argv_relative_compatibility_excludes_input_authorities(
    tmp_path, monkeypatch, flag, sealed_field
):
    completed, _, output_root, _ = _run_probe(tmp_path, fail_batches="8")
    assert completed.returncode == 42
    attempt_path = next(output_root.glob("attempts/full/**/evidence/attempt.json"))
    payload = json.loads(attempt_path.read_text())

    import stage1.benchmark as benchmark

    expected = Path(payload["launch_config"][sealed_field])
    repository_root = REPO_ROOT if flag == "--deepspeed-config" else tmp_path
    monkeypatch.setattr(benchmark, "_REPOSITORY_ROOT", repository_root)
    relative = str(expected.relative_to(repository_root))
    _rewrite_probe_attempt_argv(attempt_path, {flag: relative})

    with pytest.raises(ValueError, match=f"value mismatch for {flag}"):
        benchmark._load_probe_attempt(attempt_path)


def test_public_verified_loader_accepts_the_authoritative_producer_envelope(tmp_path):
    """Catches public API and CLI consumers validating different formats."""
    import stage1.benchmark as benchmark

    data = tmp_path / "TRAIN.jsonl"
    data.write_text("{}\n")
    images = tmp_path / "images"
    images.mkdir()
    config = {
        "schema_version": 1, "kind": "stage1-probe-run-manifest",
        "method": "full", "tuning_config": None, "model": "model",
        "source_data_path": str(data.resolve()), "data_path": str(data.resolve()),
        "data_stat": benchmark._stat_payload(data), "image_root": str(images.resolve()),
        "output_dir": str((tmp_path / "output").resolve()), "micro_batch": 8,
        "world_size": 2, "gradient_accumulation_steps": 2,
        "effective_global_batch": 32, "epochs": 3, "learning_rate": 1e-5,
        "seed": 42, "max_steps": 3, "final_save": False,
        "deepspeed_config": str((REPO_ROOT / "configs/config_stage3.json").resolve()),
        "deepspeed_config_digest": "d" * 64,
        "attention_implementation": "flash_attention_2", "save_strategy": "no",
        "audit_digest": "a" * 64, "probe_rows": str(tmp_path / "audit.json"),
        "probe_row_indices": [7, 3], "maximum_config_digest": None,
        "prerequisite_chain": str(tmp_path / "prerequisite-chain.json"),
        "prerequisite_chain_digest": "f" * 64,
    }
    config["config_digest"] = config_digest(config)
    config_path = tmp_path / "run-config.json"
    config_path.write_text(json.dumps(config))
    sealed_path = tmp_path / "producer-evidence.json"
    benchmark.write_stage1_producer_evidence(
        sealed_path,
        run_config_path=config_path,
        evidence=evidence(config_digest=config["config_digest"], checkpoint_bytes=0),
    )

    assert load_verified_evidence(sealed_path, config["config_digest"]) == evidence(
        config_digest=config["config_digest"], checkpoint_bytes=0
    )

    forged = json.loads(sealed_path.read_text())
    forged["benchmark_evidence"]["micro_batch"] = 16
    forged["benchmark_evidence"]["effective_global_batch"] = 64
    forged["producer_digest"] = config_digest(
        {key: value for key, value in forged.items() if key != "producer_digest"}
    )
    forged_path = tmp_path / "forged-producer-evidence.json"
    forged_path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="micro_batch"):
        benchmark._load_producer_evidence(forged_path)


def test_full_artifact_index_and_allowlisted_bytes_are_strict(tmp_path):
    """Catches dangling shard indexes and logs counted as checkpoint bytes."""
    import stage1.benchmark as benchmark

    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}})
    )
    with pytest.raises(ValueError, match="shard"):
        benchmark._build_artifact_manifest(tmp_path, method="full", final_save=True)

    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"shard")
    indexed = benchmark._build_artifact_manifest(
        tmp_path, method="full", final_save=True
    )
    assert indexed["weight_serialization"] == {
        "kind": "indexed",
        "index": "model.safetensors.index.json",
        "shards": ["model-00001-of-00001.safetensors"],
    }

    (tmp_path / "model.safetensors.index.json").unlink()
    (tmp_path / "model-00001-of-00001.safetensors").unlink()
    (tmp_path / "model.safetensors").write_bytes(b"weight")
    (tmp_path / "train.log").write_bytes(b"x" * 100)
    manifest = benchmark._build_artifact_manifest(tmp_path, method="full", final_save=True)
    assert "train.log" not in {item["name"] for item in manifest["files"]}
    assert manifest["checkpoint_bytes"] == sum(
        (tmp_path / name).stat().st_size
        for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors")
    )
    assert manifest["weight_serialization"] == {
        "kind": "standalone", "index": None, "shards": ["model.safetensors"]
    }


def test_dora_artifact_requires_positive_sidecars(tmp_path):
    """Catches empty adapter/token/processor files presented as a final save."""
    import stage1.benchmark as benchmark

    for name in (
        "adapter_config.json", "adapter_model.safetensors",
        "ilvr_token_rows.safetensors", "stage1_peft_metadata.json",
        "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
    ):
        (tmp_path / name).touch()
    with pytest.raises(ValueError, match="positive"):
        benchmark._build_artifact_manifest(tmp_path, method="dora", final_save=True)

    for path in tmp_path.iterdir():
        path.write_bytes(b"witness")
    (tmp_path / "model.safetensors").write_bytes(b"forbidden-full-weight")
    with pytest.raises(ValueError, match="full-model"):
        benchmark._build_artifact_manifest(tmp_path, method="dora", final_save=True)


def test_capacity_chain_rejects_duplicate_and_success_after_first_failure():
    """Catches re-search and a result selected from beyond the stopping point."""
    import stage1.benchmark as benchmark

    immutable = {
        "method": "full", "world_size": 2, "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5, "seed": 42, "model": "model",
        "data_path": "/data", "train_data_path": "/probe-data",
        "data_stat": {"device": 1}, "image_root": "/images",
        "deepspeed_config": "/zero3.json", "deepspeed_config_digest": "d" * 64,
        "attention_implementation": "flash_attention_2", "probe_rows": "/audit",
        "probe_row_indices": [1], "audit_digest": "a" * 64, "tuning_config": None,
        "prerequisite_chain": "/smoke/prerequisite-chain.json",
        "prerequisite_chain_digest": "f" * 64,
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


def _maximum_probe_launch_config(*, max_steps=3):
    return {
        "method": "full",
        "micro_batch": 8,
        "world_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_global_batch": 32,
        "learning_rate": 1e-5,
        "seed": 42,
        "max_steps": max_steps,
        "final_save": False,
        "model": "model",
        "tuning_config": None,
        "data_path": "/source/TRAIN.jsonl",
        "train_data_path": "/probe/probe-input.jsonl",
        "data_stat": {"device": 1, "inode": 2, "size": 3, "mtime_ns": 4},
        "image_root": "/source/images",
        "deepspeed_config": "/repo/configs/config_stage3.json",
        "deepspeed_config_digest": "d" * 64,
        "attention_implementation": "flash_attention_2",
        "probe_rows": "/probe/high-cost.json",
        "probe_row_indices": [7, 3],
        "audit_digest": "a" * 64,
        "run_config_digest": ("c" if max_steps == 3 else "b") * 64,
        "prerequisite_chain_digest": "f" * 64,
        "prerequisite_chain": "/smoke/prerequisite-chain.json",
        "command": ["launcher"],
    }


def _sealed_maximum_fixture(tmp_path, monkeypatch):
    import stage1.benchmark as benchmark

    probe_root = tmp_path / "probe"
    selected_path = (
        probe_root
        / "attempts/full/micro-batch-8/attempt-selected/evidence/attempt.json"
    )
    long_path = (
        probe_root
        / "long-smoke/full/micro-batch-8/attempt-long/evidence/attempt.json"
    )
    selected_launch = _maximum_probe_launch_config(max_steps=3)
    long_launch = _maximum_probe_launch_config(max_steps=10)
    selected = {
        "method": "full", "micro_batch": 8, "world_size": 2,
        "gradient_accumulation_steps": 2, "effective_global_batch": 32,
        "max_steps": 3, "final_save": False, "exit_code": 0,
        "recorded_at_ns": 1, "attempt_digest": "1" * 64,
        "audit_digest": "a" * 64, "data_stat": selected_launch["data_stat"],
        "probe_row_indices": [7, 3], "prerequisite_chain_digest": "f" * 64,
        "launch_config": selected_launch,
    }
    long_attempt = {
        **selected,
        "max_steps": 10,
        "recorded_at_ns": 2,
        "attempt_digest": "2" * 64,
        "output_dir": str((long_path.parents[1] / "output").resolve()),
        "launch_config": long_launch,
    }
    producer_config = {
        "schema_version": 1, "kind": "stage1-probe-run-manifest",
        "method": "full", "tuning_config": None, "model": "model",
        "source_data_path": long_launch["data_path"],
        "data_path": long_launch["train_data_path"],
        "data_stat": long_launch["data_stat"],
        "image_root": long_launch["image_root"],
        "output_dir": long_attempt["output_dir"],
        "micro_batch": 8, "world_size": 2,
        "gradient_accumulation_steps": 2, "effective_global_batch": 32,
        "epochs": 3, "learning_rate": 1e-5, "seed": 42,
        "max_steps": 10, "final_save": False,
        "deepspeed_config": long_launch["deepspeed_config"],
        "deepspeed_config_digest": "d" * 64,
        "attention_implementation": "flash_attention_2", "save_strategy": "no",
        "audit_digest": "a" * 64, "probe_rows": long_launch["probe_rows"],
        "probe_row_indices": [7, 3], "maximum_config_digest": None,
        "prerequisite_chain": long_launch["prerequisite_chain"],
        "prerequisite_chain_digest": "f" * 64,
    }
    producer_config["config_digest"] = "b" * 64
    long_evidence = evidence(
        optimizer_steps=10,
        samples_seen=320,
        config_digest=producer_config["config_digest"],
        checkpoint_bytes=0,
    )
    producer_seal = {
        "producer_digest": "9" * 64,
        "run_config": producer_config,
        "benchmark_evidence": benchmark._evidence_to_dict(long_evidence),
    }
    maximum = {
        "schema_version": 1,
        "kind": "stage1-capacity-maximum",
        "status": "success",
        "method": "full",
        "selected_micro_batch": 8,
        "world_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_global_batch": 32,
        "long_smoke_steps": 10,
        "long_smoke_attempt": str(long_path),
        "selected_attempt": str(selected_path),
        "first_failure": None,
        "launch_config": dict(selected_launch),
        "attempt_chain_digest": config_digest({"attempt_digests": ["1" * 64]}),
        "audit_digest": "a" * 64,
        "data_stat": selected_launch["data_stat"],
        "probe_row_indices": [7, 3],
        "prerequisite_chain_digest": "f" * 64,
        "long_smoke_producer_digest": "9" * 64,
        "rank_peak_allocated_bytes": [10, 11],
        "rank_peak_reserved_bytes": [12, 13],
    }
    maximum["config_digest"] = config_digest(maximum)
    maximum_path = tmp_path / "maximum.json"
    maximum_path.write_text(json.dumps(maximum))

    def load_attempt(path):
        return long_attempt if Path(path) == long_path else selected

    monkeypatch.setattr(benchmark, "_load_probe_attempt", load_attempt)
    monkeypatch.setattr(
        benchmark, "_attempt_files", lambda probe_dir, method=None: [selected_path]
    )
    monkeypatch.setattr(
        benchmark, "_load_producer_evidence",
        lambda path: (long_evidence, producer_seal),
    )
    return maximum_path, maximum, selected, long_attempt, producer_seal


def _reseal_maximum(path, payload):
    payload["config_digest"] = config_digest(
        {key: value for key, value in payload.items() if key != "config_digest"}
    )
    path.write_text(json.dumps(payload))


def _reseal_mock_long_producer(maximum_path, maximum, long_attempt, producer):
    producer_config = producer["run_config"]
    producer_config["config_digest"] = config_digest(
        {
            key: value
            for key, value in producer_config.items()
            if key != "config_digest"
        }
    )
    long_attempt["launch_config"]["run_config_digest"] = producer_config[
        "config_digest"
    ]
    producer["benchmark_evidence"]["config_digest"] = producer_config[
        "config_digest"
    ]
    producer["producer_digest"] = config_digest(
        {key: value for key, value in producer.items() if key != "producer_digest"}
    )
    maximum["long_smoke_producer_digest"] = producer["producer_digest"]
    _reseal_maximum(maximum_path, maximum)


def test_maximum_rejects_resealed_launch_config_not_equal_to_selected_attempt(
    tmp_path, monkeypatch,
):
    """Catches a maximum relabeling the selected attempt after the probe."""
    import stage1.benchmark as benchmark

    maximum_path, maximum, _, _, _ = _sealed_maximum_fixture(tmp_path, monkeypatch)
    maximum["launch_config"] = dict(maximum["launch_config"])
    maximum["launch_config"]["model"] = "forged-model"
    maximum["config_digest"] = config_digest(
        {key: value for key, value in maximum.items() if key != "config_digest"}
    )
    maximum_path.write_text(json.dumps(maximum))

    with pytest.raises(ValueError, match="selected attempt launch config"):
        benchmark._load_maximum(maximum_path)


def test_maximum_rejects_resealed_long_producer_not_bound_to_long_attempt(
    tmp_path, monkeypatch,
):
    """Catches unrelated ten-step producer evidence attached to a maximum."""
    import stage1.benchmark as benchmark

    maximum_path, _, _, _, producer = _sealed_maximum_fixture(tmp_path, monkeypatch)
    producer["run_config"] = dict(producer["run_config"])
    producer["run_config"]["source_data_path"] = "/forged/TRAIN.jsonl"
    producer["run_config"]["config_digest"] = "e" * 64
    producer["benchmark_evidence"] = dict(producer["benchmark_evidence"])
    producer["benchmark_evidence"]["config_digest"] = "e" * 64

    with pytest.raises(ValueError, match="long-smoke producer.*attempt"):
        benchmark._load_maximum(maximum_path)


def test_maximum_rejects_resealed_top_level_batch_controls_not_from_selected(
    tmp_path, monkeypatch,
):
    """Catches top-level accumulation/effective values relabeled after selection."""
    import stage1.benchmark as benchmark

    maximum_path, maximum, _, _, _ = _sealed_maximum_fixture(tmp_path, monkeypatch)
    maximum["gradient_accumulation_steps"] = 3
    maximum["effective_global_batch"] = 48
    _reseal_maximum(maximum_path, maximum)

    with pytest.raises(ValueError, match="selected attempt.*batch controls"):
        benchmark._load_maximum(maximum_path)


def test_maximum_rejects_full_selection_with_resealed_dora_long_attempt(
    tmp_path, monkeypatch,
):
    """Catches a long-smoke method/config relabeled independently of selection."""
    import stage1.benchmark as benchmark

    maximum_path, maximum, _, long_attempt, producer = _sealed_maximum_fixture(
        tmp_path, monkeypatch
    )
    dora_tuning = {
        "use_dora": True,
        "rank": 32,
        "alpha": 64,
        "dropout": 0.05,
        "target_modules": (
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        ),
    }
    long_attempt["method"] = "dora"
    long_attempt["launch_config"]["method"] = "dora"
    long_attempt["launch_config"]["tuning_config"] = dora_tuning
    producer["run_config"]["method"] = "dora"
    producer["run_config"]["tuning_config"] = dora_tuning
    producer["benchmark_evidence"]["method"] = "dora"
    _reseal_mock_long_producer(
        maximum_path, maximum, long_attempt, producer
    )

    with pytest.raises(ValueError, match="long-smoke.*selected"):
        benchmark._load_maximum(maximum_path)


def test_maximum_rejects_unrelated_long_launcher_command_change(
    tmp_path, monkeypatch,
):
    """Catches long smoke changing more than attempt-local paths and max steps."""
    import stage1.benchmark as benchmark

    maximum_path, _, _, long_attempt, _ = _sealed_maximum_fixture(
        tmp_path, monkeypatch
    )
    long_attempt["launch_config"]["command"] = ["unrelated-launcher"]

    with pytest.raises(ValueError, match="long-smoke.*command"):
        benchmark._load_maximum(maximum_path)


def test_maximum_rejects_resealed_selected_attempt_not_exactly_three_steps(
    tmp_path, monkeypatch,
):
    """Catches a capacity selection built from a non-three-step attempt."""
    import stage1.benchmark as benchmark

    maximum_path, maximum, selected, _, _ = _sealed_maximum_fixture(
        tmp_path, monkeypatch
    )
    selected["max_steps"] = 4
    selected["launch_config"]["max_steps"] = 4
    maximum["launch_config"] = dict(selected["launch_config"])
    _reseal_maximum(maximum_path, maximum)

    with pytest.raises(ValueError, match="selected attempt.*3 steps"):
        benchmark._load_maximum(maximum_path)


def test_maximum_rejects_resealed_long_attempt_not_exactly_ten_steps(
    tmp_path, monkeypatch,
):
    """Catches a capacity maximum whose long smoke is not the approved ten steps."""
    import stage1.benchmark as benchmark

    maximum_path, maximum, _, long_attempt, producer = _sealed_maximum_fixture(
        tmp_path, monkeypatch
    )
    maximum["long_smoke_steps"] = 11
    long_attempt["max_steps"] = 11
    long_attempt["launch_config"]["max_steps"] = 11
    producer["run_config"]["max_steps"] = 11
    producer["benchmark_evidence"]["optimizer_steps"] = 11
    producer["benchmark_evidence"]["samples_seen"] = 352
    _reseal_mock_long_producer(
        maximum_path, maximum, long_attempt, producer
    )

    with pytest.raises(ValueError, match="long smoke.*10 steps"):
        benchmark._load_maximum(maximum_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimizer_steps", 9, "optimizer steps"),
        ("samples_seen", 319, "samples seen"),
    ],
)
def test_maximum_rejects_resealed_long_producer_step_or_sample_count(
    tmp_path, monkeypatch, field, value, message,
):
    """Catches producer metrics not proving the exact ten-step long smoke."""
    import stage1.benchmark as benchmark

    maximum_path, maximum, _, long_attempt, producer = _sealed_maximum_fixture(
        tmp_path, monkeypatch
    )
    producer["benchmark_evidence"][field] = value
    _reseal_mock_long_producer(
        maximum_path, maximum, long_attempt, producer
    )

    with pytest.raises(ValueError, match=message):
        benchmark._load_maximum(maximum_path)


def test_timed_validation_rejects_artifactless_and_plain_metric_evidence(tmp_path):
    """Catches sealing a successful metrics blob without final model topology."""
    import stage1.benchmark as benchmark

    plain = evidence(config_digest="c" * 64, checkpoint_bytes=0)
    plain_path = tmp_path / "plain.json"
    _write_legacy_benchmark_evidence(plain_path, plain)
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
        "prerequisite_chain": "/smoke/prerequisite-chain.json",
        "prerequisite_chain_digest": "f" * 64,
    }
    config["config_digest"] = config_digest(config)
    metrics = evidence(config_digest=config["config_digest"], checkpoint_bytes=0)
    artifact = {
        "schema_version": 1, "kind": "stage1-artifact-topology-witness",
        "method": "full", "artifact_kind": "none", "output_dir": str(tmp_path),
        "final_save": True, "weight_serialization": None,
        "files": [], "checkpoint_bytes": 0,
    }
    artifact["artifact_manifest_digest"] = config_digest(artifact)
    seal = {
        "schema_version": 1, "kind": "stage1-producer-evidence",
        "run_config": config,
        "benchmark_evidence": benchmark._evidence_to_dict(metrics),
        "artifact_manifest": artifact,
        "artifact_manifest_digest": artifact["artifact_manifest_digest"],
        "smoke_resume_witness": None,
        "smoke_resume_witness_digest": None,
    }
    seal["producer_digest"] = config_digest(seal)
    seal_path = tmp_path / "producer.json"
    seal_path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="topology"):
        benchmark._load_producer_evidence(seal_path)


def test_compare_maxima_exposes_rank_local_ten_step_memory(monkeypatch):
    """Catches a maximum comparison that drops the required rank witnesses."""
    import stage1.benchmark as benchmark

    controlled = {
        "model": "model", "data_path": "/data",
        "data_stat": {"device": 1, "inode": 2, "size": 3, "mtime_ns": 4},
        "image_root": "/images", "world_size": 2,
        "gradient_accumulation_steps": 2, "learning_rate": 1e-5, "seed": 42,
        "deepspeed_config": "/zero3.json", "deepspeed_config_digest": "d" * 64,
        "attention_implementation": "flash_attention_2", "probe_rows": "/audit",
        "probe_row_indices": [7, 3], "audit_digest": "a" * 64,
        "prerequisite_chain_digest": "f" * 64,
    }
    maxima = {
        "full.json": {"method": "full", "selected_micro_batch": 8,
                      "effective_global_batch": 32, "long_smoke_steps": 10,
                      "first_failure": None, "rank_peak_allocated_bytes": [10, 11],
                      "rank_peak_reserved_bytes": [12, 13], "config_digest": "a" * 64,
                      "prerequisite_chain_digest": "f" * 64,
                      "launch_config": dict(controlled)},
        "dora.json": {"method": "dora", "selected_micro_batch": 16,
                      "effective_global_batch": 64, "long_smoke_steps": 10,
                      "first_failure": None, "rank_peak_allocated_bytes": [20, 21],
                      "rank_peak_reserved_bytes": [22, 23], "config_digest": "b" * 64,
                      "prerequisite_chain_digest": "f" * 64,
                      "launch_config": dict(controlled)},
    }
    monkeypatch.setattr(benchmark, "_load_maximum", lambda path: maxima[path.name])
    compared = benchmark._compare_maxima(["full.json", "dora.json"])
    assert compared["methods"]["full"]["rank_peak_allocated_bytes"] == [10, 11]
    assert compared["methods"]["dora"]["rank_peak_reserved_bytes"] == [22, 23]
    assert compared["prerequisite_chain_digest"] == "f" * 64

    maxima["dora.json"]["prerequisite_chain_digest"] = "e" * 64
    with pytest.raises(ValueError, match="prerequisite"):
        benchmark._compare_maxima(["full.json", "dora.json"])


def test_smoke_pair_seals_resume_identity_and_rejects_control_mismatch(monkeypatch, tmp_path):
    """Catches Task 16 accepting unrelated Task 15 smoke successes."""
    import stage1.benchmark as benchmark

    shared = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct", "source_data_path": "/data/TRAIN.jsonl",
        "data_stat": {"device": 1, "inode": 2, "size": 3, "mtime_ns": 4},
        "image_root": "/data", "world_size": 2, "gradient_accumulation_steps": 2,
        "epochs": 3, "learning_rate": 1e-5, "seed": 42,
        "deepspeed_config": "/repo/configs/config_stage3.json",
        "deepspeed_config_digest": "d" * 64,
        "attention_implementation": "flash_attention_2", "save_strategy": "no",
        "audit_digest": "a" * 64, "probe_row_indices": [7, 3],
    }
    resume = {
        "initial_final_step": 3, "resume_initial_step": 3, "resume_final_step": 4,
        "batch_size_observed": 2, "finite_loss": True,
        "checkpoint_validated": True, "resume_artifact_digest": "e" * 64,
    }
    loaded = {}
    for method in ("full", "dora"):
        metrics = evidence(method=method, micro_batch=2, effective_global_batch=8)
        loaded[f"{method}.json"] = (
            metrics,
            {
                "producer_digest": ("1" if method == "full" else "2") * 64,
                "run_config": {**shared, "kind": "stage1-smoke-run-manifest", "method": method},
                "smoke_resume_witness": resume,
                "smoke_resume_witness_digest": config_digest(resume),
            },
        )
    monkeypatch.setattr(benchmark, "_load_producer_evidence", lambda path: loaded[path.name])
    loaded["dora.json"][1]["run_config"]["model"] = "different-model"
    with pytest.raises(ValueError, match="model"):
        benchmark._require_evidence(["full.json", "dora.json"])

    loaded["dora.json"][1]["run_config"]["model"] = shared["model"]
    chain = benchmark._require_evidence(["full.json", "dora.json"])
    assert chain["kind"] == "stage1-smoke-prerequisite-chain"
    assert len(chain["prerequisite_chain_digest"]) == 64


def test_probe_does_not_fallback_without_a_confirmed_initial_eight_oom(tmp_path):
    """Catches an implicit batch reduction after an unclassified failure."""
    completed, records, output_root, _ = _run_probe(
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
        "prerequisite_chain": str(tmp_path / "prerequisite-chain.json"),
        "prerequisite_chain_digest": "f" * 64,
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
        "prerequisite_chain_digest": "f" * 64,
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
    assert manifest["prerequisite_chain"] == str(tmp_path / "prerequisite-chain.json")
    assert manifest["prerequisite_chain_digest"] == "f" * 64


def test_final_report_rejects_a_maximum_from_another_smoke_chain(tmp_path, monkeypatch):
    """Catches Task 18 joining individually valid but unrelated evidence chains."""
    import stage1.benchmark as benchmark

    evidence_by_name = {
        "smoke-full.json": (evidence(method="full", micro_batch=2,
                                      effective_global_batch=8), {}),
        "smoke-dora.json": (evidence(method="dora", micro_batch=2,
                                      effective_global_batch=8), {}),
        "timed-full.json": (evidence(method="full"), {}),
        "timed-dora.json": (evidence(method="dora"), {}),
    }
    maxima = {
        "maximum-full.json": {
            "method": "full", "prerequisite_chain_digest": "f" * 64,
        },
        "maximum-dora.json": {
            "method": "dora", "prerequisite_chain_digest": "e" * 64,
        },
    }
    monkeypatch.setattr(
        benchmark, "_load_producer_evidence",
        lambda path: evidence_by_name[path.name],
    )
    monkeypatch.setattr(benchmark, "_load_maximum", lambda path: maxima[path.name])
    monkeypatch.setattr(
        benchmark, "_build_smoke_prerequisite_chain",
        lambda loaded: {"prerequisite_chain_digest": "f" * 64},
    )

    with pytest.raises(ValueError, match="smoke prerequisite"):
        benchmark._final_report(
            SimpleNamespace(
                smoke_full="smoke-full.json", smoke_dora="smoke-dora.json",
                maximum_full="maximum-full.json", maximum_dora="maximum-dora.json",
                timed_full="timed-full.json", timed_dora="timed-dora.json",
                output=str(tmp_path / "final.json"),
            )
        )


def test_timed_comparison_is_a_standalone_canonical_prerequisite_seal(
    tmp_path, monkeypatch,
):
    """Catches a write-only comparison with no chain identity or tamper seal."""
    import stage1.benchmark as benchmark

    shared = {
        "schema_version": 1, "kind": "stage1-timed-launch-manifest",
        "model": "model", "source_data_path": "/data/TRAIN.jsonl",
        "data_path": "/data/TRAIN.jsonl",
        "data_stat": {"device": 1, "inode": 2, "size": 3, "mtime_ns": 4},
        "image_root": "/data/images", "micro_batch": 8, "world_size": 2,
        "gradient_accumulation_steps": 2, "effective_global_batch": 32,
        "epochs": 3, "learning_rate": 1e-5, "seed": 42, "max_steps": 0,
        "final_save": True, "deepspeed_config": "/repo/zero3.json",
        "deepspeed_config_digest": "d" * 64,
        "attention_implementation": "flash_attention_2", "save_strategy": "no",
        "audit_digest": "a" * 64, "probe_rows": "/probe/high-cost.json",
        "probe_row_indices": [7, 3], "prerequisite_chain": "/smoke/chain.json",
        "prerequisite_chain_digest": "f" * 64,
    }
    loaded = {}
    for method in ("full", "dora"):
        run_config = {
            **shared,
            "method": method,
            "tuning_config": None if method == "full" else {
                "use_dora": True, "rank": 32, "alpha": 64, "dropout": 0.05,
                "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            },
            "output_dir": f"/timed/{method}",
            "maximum_config_digest": ("1" if method == "full" else "2") * 64,
        }
        run_config["config_digest"] = config_digest(run_config)
        metrics = evidence(method=method, config_digest=run_config["config_digest"])
        loaded[f"{method}.json"] = (
            metrics,
            {"producer_digest": ("3" if method == "full" else "4") * 64,
             "run_config": run_config},
        )
    monkeypatch.setattr(
        benchmark, "_load_producer_evidence", lambda path: loaded[path.name]
    )
    output = tmp_path / "comparison.json"
    compared = benchmark._compare_runs(["full.json", "dora.json"], output)

    assert compared["prerequisite_chain_digest"] == "f" * 64
    assert len(compared["comparison_digest"]) == 64
    assert benchmark._load_run_comparison(output) == compared

    tampered = json.loads(output.read_text())
    tampered["methods"]["full"]["final_loss"] = 0.25
    output.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="comparison digest"):
        benchmark._load_run_comparison(output)


def _write_smoke_subprocess_launcher(
    path: Path,
    record_dir: Path,
    *,
    initial_gate: Path | None = None,
) -> None:
    python = REPO_ROOT / "venv/ilvr-venv/bin/python"
    path.write_text(
        f"""#!{python}
import json
import os
from pathlib import Path
import sys

from stage1.benchmark import (
    Stage1BenchmarkEvidence,
    _build_artifact_manifest,
    load_stage1_smoke_resume_witness,
    load_stage1_run_config_witness,
    write_stage1_producer_evidence,
)

args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]

record = Path({str(record_dir)!r}) / f"{{os.getpid()}}.json"
record.write_text(json.dumps({{
    "pid": os.getpid(), "argv": args, "path": os.environ["PATH"]
}}))
output = Path(value("--output-dir"))
output.mkdir(parents=True, exist_ok=True)
max_steps = int(value("--max-steps"))
initial_gate = {str(initial_gate) if initial_gate is not None else None!r}
if max_steps == 3 and initial_gate is not None:
    import time
    while not Path(initial_gate).exists():
        time.sleep(0.01)
if max_steps == 3:
    checkpoint = output / "checkpoint-3"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"full")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({{"global_step": 3}})
    )
    for name in ("optimizer.pt", "scheduler.pt", "rng_state_0.pth", "rng_state_1.pth"):
        (checkpoint / name).write_bytes(b"state")
elif max_steps == 4:
    config_path = Path(value("--benchmark-run-config"))
    evidence_path = Path(value("--producer-evidence"))
    marker_path = Path(value("--smoke-resume-witness"))
    config_witness = load_stage1_run_config_witness(config_path)
    config = config_witness.payload()
    smoke = load_stage1_smoke_resume_witness(
        marker_path,
        method=config["method"],
        checkpoint_path=Path(value("--resume")),
        micro_batch=int(value("--micro-batch")),
        initial_max_steps=3,
        resumed_max_steps=4,
    )
    artifact = _build_artifact_manifest(
        output, method=config["method"], final_save=False
    )
    metrics = Stage1BenchmarkEvidence(
        status="success", method=config["method"],
        micro_batch=config["micro_batch"], world_size=config["world_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        effective_global_batch=config["effective_global_batch"],
        optimizer_steps=4, samples_seen=4 * config["effective_global_batch"],
        model_load_seconds=1.0, trainer_seconds=2.0, total_seconds=3.0,
        final_save_seconds=0.0, samples_per_second=1.0,
        steps_per_second=1.0,
        rank_peak_allocated_bytes=(10, 11),
        rank_peak_reserved_bytes=(12, 13), trainable_parameters=20,
        total_parameters=100, checkpoint_bytes=0, final_loss=0.5,
        representative_gradient_norm=0.25, git_commit="abc1234",
        dependency_versions={{"torch": "test"}}, gpu_names=("B200", "B200"),
        config_digest=config["config_digest"],
        probe_row_indices=tuple(config["probe_row_indices"]),
    )
    write_stage1_producer_evidence(
        evidence_path, run_config_witness=config_witness, evidence=metrics,
        smoke_resume_witness=smoke.summary(), artifact_manifest=artifact,
    )
else:
    raise SystemExit(f"unexpected max steps: {{max_steps}}")
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _smoke_fixture(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = image_root / "original.bin"
    image.write_bytes(b"metadata-only image reference")
    data_path = tmp_path / "TRAIN.jsonl"
    data_path.write_text(
        json.dumps(
            {"image_input": str(image), "text_input": "hello", "sequence_plan": []}
        )
        + "\n",
        encoding="utf-8",
    )
    records = tmp_path / "records"
    records.mkdir()
    launcher = tmp_path / "smoke-launcher"
    _write_smoke_subprocess_launcher(launcher, records)
    return data_path, image_root, records, launcher


def _write_failing_smoke_launcher(path: Path, record_dir: Path) -> None:
    python = REPO_ROOT / "venv/ilvr-venv/bin/python"
    path.write_text(
        f"""#!{python}
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
output = Path(args[args.index("--output-dir") + 1])
output.mkdir(parents=True, exist_ok=True)
record = Path({str(record_dir)!r}) / f"{{os.getpid()}}.json"
record.write_text(json.dumps({{"pid": os.getpid(), "output": str(output)}}))
raise SystemExit(19)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_distributed_smoke_uses_two_real_processes_then_reuses_only_small_seals(
    tmp_path, monkeypatch,
):
    """Catches in-process resume, rerun-on-reuse, or checkpoint-tree rereads."""
    import shutil
    import stage1.benchmark as benchmark

    data_path, image_root, records, launcher = _smoke_fixture(tmp_path)
    evidence_dir = tmp_path / "evidence" / "full"
    kwargs = dict(
        method="full",
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        data_path=data_path,
        image_root=image_root,
        micro_batch=2,
        gradient_accumulation_steps=2,
        world_size=2,
        initial_max_steps=3,
        resumed_max_steps=4,
        deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
        evidence_dir=evidence_dir,
        launcher_path=launcher,
    )

    observed = benchmark.run_or_load_distributed_witness(**kwargs)

    attempts = sorted((evidence_dir / "attempts").iterdir())
    assert [attempt.name for attempt in attempts] == ["attempt-1"]
    attempt = attempts[0]
    canonical = json.loads((evidence_dir / "evidence.json").read_text())
    assert canonical["kind"] == "stage1-smoke-registry-evidence"
    assert canonical["checkpoint_marker"]["kind"] == (
        "stage1-smoke-checkpoint-topology"
    )
    assert canonical["attempt_path"] == "attempts/attempt-1"
    assert canonical["attempt_evidence_path"] == (
        "attempts/attempt-1/evidence.json"
    )
    assert canonical["marker_path"] == (
        "attempts/attempt-1/checkpoint-topology.json"
    )
    assert benchmark._check_evidence(
        SimpleNamespace(
            evidence_dir=str(evidence_dir),
            method="full",
            micro_batch=2,
            world_size=2,
        )
    ) == "reusable-success"
    assert benchmark._summaries([str(evidence_dir / "evidence.json")])[0][
        "optimizer_steps"
    ] == 4
    process_records = [json.loads(path.read_text()) for path in records.iterdir()]
    assert len(process_records) == 2
    assert len({item["pid"] for item in process_records}) == 2
    assert all(
        item["path"].split(os.pathsep)[0]
        == str(REPO_ROOT / "venv/ilvr-venv/bin")
        for item in process_records
    )
    commands = {int(item["argv"][item["argv"].index("--max-steps") + 1]): item["argv"] for item in process_records}
    assert commands[3][commands[3].index("--save-steps") + 1] == "3"
    assert commands[4][commands[4].index("--resume") + 1].endswith("checkpoint-3")
    assert observed.resume_initial_step == 3
    assert observed.resume_final_step == 4

    canonical_before = (evidence_dir / "evidence.json").stat()
    shutil.rmtree(attempt)
    data_path.unlink()
    shutil.rmtree(image_root)
    launcher.unlink()

    def reject_external_resolve(*args, **kwargs):
        raise AssertionError("successful canonical reuse resolved an external path")

    monkeypatch.setattr(
        Path,
        "resolve",
        reject_external_resolve,
    )
    reused = benchmark.run_or_load_distributed_witness(**kwargs)
    canonical_after = (evidence_dir / "evidence.json").stat()
    assert reused == observed
    assert len(list(records.iterdir())) == 2
    assert (canonical_after.st_ino, canonical_after.st_mtime_ns) == (
        canonical_before.st_ino,
        canonical_before.st_mtime_ns,
    )


def test_distributed_smoke_reuse_rejects_a_tampered_small_checkpoint_marker(tmp_path):
    """Catches evidence reuse that ignores its sealed checkpoint marker."""
    import stage1.benchmark as benchmark

    data_path, image_root, _, launcher = _smoke_fixture(tmp_path)
    evidence_dir = tmp_path / "evidence" / "full"
    kwargs = dict(
        method="full", model="Qwen/Qwen2.5-VL-3B-Instruct",
        data_path=data_path, image_root=image_root, micro_batch=2,
        gradient_accumulation_steps=2, world_size=2, initial_max_steps=3,
        resumed_max_steps=4,
        deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
        evidence_dir=evidence_dir, launcher_path=launcher,
    )
    benchmark.run_or_load_distributed_witness(**kwargs)
    canonical_path = evidence_dir / "evidence.json"
    payload = json.loads(canonical_path.read_text())
    payload["checkpoint_marker"]["checkpoint_files"][0]["size"] += 1
    payload["registry_digest"] = benchmark.config_digest(
        {key: value for key, value in payload.items() if key != "registry_digest"}
    )
    canonical_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="checkpoint topology.*digest"):
        benchmark.run_or_load_distributed_witness(**kwargs)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("data_path", "missing-other-data.jsonl"),
        ("image_root", "missing-other-images"),
        ("deepspeed_config", "missing-other-zero3.json"),
        ("launcher_path", "missing-other-launcher"),
    ],
)
def test_canonical_smoke_reuse_rejects_different_requested_path_without_io(
    tmp_path, field, replacement
):
    import stage1.benchmark as benchmark

    data_path, image_root, records, launcher = _smoke_fixture(tmp_path)
    evidence_dir = tmp_path / "evidence" / "full"
    kwargs = dict(
        method="full", model="Qwen/Qwen2.5-VL-3B-Instruct",
        data_path=data_path, image_root=image_root, micro_batch=2,
        gradient_accumulation_steps=2, world_size=2, initial_max_steps=3,
        resumed_max_steps=4,
        deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
        evidence_dir=evidence_dir, launcher_path=launcher,
    )
    benchmark.run_or_load_distributed_witness(**kwargs)
    before = len(list(records.iterdir()))
    kwargs[field] = tmp_path / replacement

    with pytest.raises(FileExistsError, match="different.*request"):
        benchmark.run_or_load_distributed_witness(**kwargs)

    assert len(list(records.iterdir())) == before
    assert [path.name for path in (evidence_dir / "attempts").iterdir()] == [
        "attempt-1"
    ]


def test_smoke_rejects_method_directory_mismatch_before_launch_input_io(tmp_path):
    import stage1.benchmark as benchmark

    wrong = tmp_path / "evidence" / "not-full"
    with pytest.raises(ValueError, match="evidence directory.*method"):
        benchmark.run_or_load_distributed_witness(
            method="full", model="Qwen/Qwen2.5-VL-3B-Instruct",
            data_path=tmp_path / "missing-data.jsonl",
            image_root=tmp_path / "missing-images", micro_batch=2,
            gradient_accumulation_steps=2, world_size=2,
            initial_max_steps=3, resumed_max_steps=4,
            deepspeed_config=tmp_path / "missing-zero3.json",
            evidence_dir=wrong,
            launcher_path=tmp_path / "missing-launcher",
        )

    assert not wrong.exists()
    assert not (wrong.parent / "high-cost-rows.json").exists()

    args = SimpleNamespace(
        evidence_dir=str(wrong), method="full", micro_batch=2, world_size=2
    )
    with pytest.raises(ValueError, match="evidence directory.*method"):
        benchmark._check_evidence(args)


def test_canonical_smoke_reuse_rejects_model_and_setting_mismatch(tmp_path):
    import stage1.benchmark as benchmark

    data_path, image_root, records, launcher = _smoke_fixture(tmp_path)
    evidence_dir = tmp_path / "evidence" / "full"
    kwargs = dict(
        method="full", model="Qwen/Qwen2.5-VL-3B-Instruct",
        data_path=data_path, image_root=image_root, micro_batch=2,
        gradient_accumulation_steps=2, world_size=2, initial_max_steps=3,
        resumed_max_steps=4,
        deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
        evidence_dir=evidence_dir, launcher_path=launcher,
    )
    benchmark.run_or_load_distributed_witness(**kwargs)
    before = len(list(records.iterdir()))

    with pytest.raises(FileExistsError, match="different.*request"):
        benchmark.run_or_load_distributed_witness(
            **{**kwargs, "model": "Different/Model"}
        )
    with pytest.raises(ValueError, match="distributed smoke requires"):
        benchmark.run_or_load_distributed_witness(
            **{**kwargs, "gradient_accumulation_steps": 1}
        )

    assert len(list(records.iterdir())) == before


def test_smoke_execution_lease_releases_on_close_and_is_method_scoped(tmp_path):
    import stage1.benchmark as benchmark

    full = tmp_path / "evidence" / "full"
    dora = tmp_path / "evidence" / "dora"
    with benchmark._smoke_execution_lease(full):
        with pytest.raises(RuntimeError, match="in progress"):
            with benchmark._smoke_execution_lease(full):
                raise AssertionError("contender acquired the full lease")
        with benchmark._smoke_execution_lease(dora):
            pass

    lock = full / ".execution.lock"
    before = lock.stat()
    with pytest.raises(RuntimeError, match="simulated process failure"):
        with benchmark._smoke_execution_lease(full):
            raise RuntimeError("simulated process failure")
    with benchmark._smoke_execution_lease(full):
        pass
    after = lock.stat()
    assert stat.S_IMODE(after.st_mode) == 0o600
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


def test_smoke_execution_lease_rejects_symlink_and_replaced_lock(
    tmp_path, monkeypatch
):
    import stage1.benchmark as benchmark

    registry = tmp_path / "evidence" / "full"
    registry.mkdir(parents=True)
    target = tmp_path / "foreign-lock"
    target.write_bytes(b"")
    lock = registry / ".execution.lock"
    lock.symlink_to(target)
    with pytest.raises(ValueError, match="execution lock"):
        with benchmark._smoke_execution_lease(registry):
            raise AssertionError("symlink lock was accepted")

    lock.unlink()
    os.mkfifo(lock)
    with pytest.raises(ValueError, match="execution lock"):
        with benchmark._smoke_execution_lease(registry):
            raise AssertionError("FIFO lock was accepted")

    lock.unlink()
    original = benchmark.fcntl.flock

    def replace_after_acquire(fd, operation):
        original(fd, operation)
        lock.unlink()
        lock.write_bytes(b"replacement")
        lock.chmod(0o600)

    monkeypatch.setattr(benchmark.fcntl, "flock", replace_after_acquire)
    with pytest.raises(ValueError, match="execution lock changed"):
        with benchmark._smoke_execution_lease(registry):
            raise AssertionError("replaced lock was accepted")


def test_smoke_execution_lease_is_released_when_holder_process_is_killed(tmp_path):
    import stage1.benchmark as benchmark

    registry = tmp_path / "evidence" / "full"
    ready = tmp_path / "holder-ready"
    code = """
import sys
import time
from pathlib import Path
from stage1.benchmark import _smoke_execution_lease
with _smoke_execution_lease(Path(sys.argv[1])):
    Path(sys.argv[2]).write_text('ready', encoding='utf-8')
    time.sleep(60)
"""
    holder = subprocess.Popen(
        [sys.executable, "-c", code, str(registry), str(ready)],
        cwd=REPO_ROOT,
    )
    try:
        for _ in range(500):
            if ready.is_file():
                break
            if holder.poll() is not None:
                raise AssertionError(f"lease holder exited early: {holder.returncode}")
            time.sleep(0.01)
        else:
            raise AssertionError("lease holder did not acquire the lock")
        with pytest.raises(RuntimeError, match="in progress"):
            with benchmark._smoke_execution_lease(registry):
                raise AssertionError("contender acquired a live process lease")
    finally:
        holder.kill()
        holder.wait(timeout=10)

    with benchmark._smoke_execution_lease(registry):
        pass


def test_concurrent_smoke_wrapper_never_launches_after_winner_publishes_success(
    tmp_path,
):
    import stage1.benchmark as benchmark

    data_path, image_root, records, _ = _smoke_fixture(tmp_path)
    gate = tmp_path / "release-initial"
    launcher = tmp_path / "blocking-smoke-launcher"
    _write_smoke_subprocess_launcher(
        launcher, records, initial_gate=gate
    )
    evidence_dir = tmp_path / "evidence" / "full"
    kwargs = dict(
        method="full", model="Qwen/Qwen2.5-VL-3B-Instruct",
        data_path=data_path, image_root=image_root, micro_batch=2,
        gradient_accumulation_steps=2, world_size=2, initial_max_steps=3,
        resumed_max_steps=4,
        deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
        evidence_dir=evidence_dir, launcher_path=launcher,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        winner = executor.submit(benchmark.run_or_load_distributed_witness, **kwargs)
        for _ in range(500):
            if list(records.iterdir()):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("winner did not reach its first subprocess")

        with pytest.raises(RuntimeError, match="in progress"):
            benchmark.run_or_load_distributed_witness(**kwargs)
        assert len(list(records.iterdir())) == 1
        assert [path.name for path in (evidence_dir / "attempts").iterdir()] == [
            "attempt-1"
        ]
        gate.write_text("release", encoding="utf-8")
        observed = winner.result(timeout=20)

    assert observed.resume_final_step == 4
    assert len(list(records.iterdir())) == 2
    assert benchmark.run_or_load_distributed_witness(**kwargs) == observed
    assert len(list(records.iterdir())) == 2


def test_smoke_wrapper_detects_execution_lock_replacement_between_launches(
    tmp_path, monkeypatch
):
    import stage1.benchmark as benchmark

    data_path, image_root, records, launcher = _smoke_fixture(tmp_path)
    evidence_dir = tmp_path / "evidence" / "full"
    lock = evidence_dir / ".execution.lock"
    original = benchmark._run_smoke_launcher
    calls = 0

    def replace_lock_after_first_launch(*args, **kwargs):
        nonlocal calls
        original(*args, **kwargs)
        calls += 1
        if calls == 1:
            lock.unlink()
            lock.write_bytes(b"replacement")
            lock.chmod(0o600)

    monkeypatch.setattr(
        benchmark, "_run_smoke_launcher", replace_lock_after_first_launch
    )
    with pytest.raises(ValueError, match="execution lock changed"):
        benchmark.run_or_load_distributed_witness(
            method="full", model="Qwen/Qwen2.5-VL-3B-Instruct",
            data_path=data_path, image_root=image_root, micro_batch=2,
            gradient_accumulation_steps=2, world_size=2,
            initial_max_steps=3, resumed_max_steps=4,
            deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
            evidence_dir=evidence_dir, launcher_path=launcher,
        )

    assert len(list(records.iterdir())) == 1
    assert not (evidence_dir / "evidence.json").exists()


def test_smoke_wrapper_detects_method_directory_replacement_before_resume(
    tmp_path, monkeypatch
):
    import stage1.benchmark as benchmark

    data_path, image_root, records, launcher = _smoke_fixture(tmp_path)
    evidence_dir = tmp_path / "evidence" / "full"
    displaced = tmp_path / "evidence" / "full-displaced"
    original = benchmark._run_smoke_launcher
    calls = 0

    def replace_method_after_first_launch(*args, **kwargs):
        nonlocal calls
        original(*args, **kwargs)
        calls += 1
        if calls == 1:
            evidence_dir.rename(displaced)
            evidence_dir.mkdir()

    monkeypatch.setattr(
        benchmark, "_run_smoke_launcher", replace_method_after_first_launch
    )
    with pytest.raises(ValueError, match="method registry changed"):
        benchmark.run_or_load_distributed_witness(
            method="full", model="Qwen/Qwen2.5-VL-3B-Instruct",
            data_path=data_path, image_root=image_root, micro_batch=2,
            gradient_accumulation_steps=2, world_size=2,
            initial_max_steps=3, resumed_max_steps=4,
            deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
            evidence_dir=evidence_dir, launcher_path=launcher,
        )

    assert len(list(records.iterdir())) == 1
    assert not (evidence_dir / "evidence.json").exists()
    assert not (displaced / "evidence.json").exists()


def test_canonical_publication_never_succeeds_through_replaced_method_name(
    tmp_path, monkeypatch
):
    import stage1.benchmark as benchmark

    data_path, image_root, records, launcher = _smoke_fixture(tmp_path)
    evidence_dir = tmp_path / "evidence" / "full"
    displaced = tmp_path / "evidence" / "full-displaced"
    original = benchmark._exclusive_write_json_at

    def replace_method_at_publication(directory_fd, name, payload):
        evidence_dir.rename(displaced)
        evidence_dir.mkdir()
        original(directory_fd, name, payload)

    monkeypatch.setattr(
        benchmark, "_exclusive_write_json_at", replace_method_at_publication
    )
    with pytest.raises(ValueError, match="method registry changed"):
        benchmark.run_or_load_distributed_witness(
            method="full", model="Qwen/Qwen2.5-VL-3B-Instruct",
            data_path=data_path, image_root=image_root, micro_batch=2,
            gradient_accumulation_steps=2, world_size=2,
            initial_max_steps=3, resumed_max_steps=4,
            deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
            evidence_dir=evidence_dir, launcher_path=launcher,
        )

    assert len(list(records.iterdir())) == 2
    assert not (evidence_dir / "evidence.json").exists()
    assert (displaced / "evidence.json").is_file()


@pytest.mark.parametrize(
    "topology",
    ["failure", "rename_recreate", "symlink", "success", "lock"],
)
def test_failed_smoke_archive_is_unsupported_without_target_io(
    tmp_path, monkeypatch, topology
):
    import stage1.benchmark as benchmark

    evidence_root = tmp_path / "evidence"
    evidence_dir = evidence_root / "full"
    evidence_root.mkdir()
    if topology == "symlink":
        retained = tmp_path / "retained-failure"
        retained.mkdir()
        (retained / "initial.log").write_bytes(b"retained symlink target")
        evidence_dir.symlink_to(retained, target_is_directory=True)
    else:
        evidence_dir.mkdir()
        (evidence_dir / "initial.log").write_bytes(b"retained active failure")
        if topology == "rename_recreate":
            displaced = evidence_root / "full-displaced"
            evidence_dir.rename(displaced)
            evidence_dir.mkdir()
            (evidence_dir / "initial.log").write_bytes(b"replacement failure")
        elif topology == "success":
            (evidence_dir / "evidence.json").write_bytes(b"sealed success")
        elif topology == "lock":
            (evidence_dir / ".execution.lock").write_bytes(b"persistent lease")

    def snapshot():
        result = []
        for path in sorted(tmp_path.rglob("*"), key=lambda item: str(item)):
            status = path.lstat()
            payload = None
            if path.is_symlink():
                payload = os.readlink(path)
            elif path.is_file():
                payload = path.read_bytes()
            result.append(
                (
                    str(path.relative_to(tmp_path)),
                    status.st_ino,
                    status.st_mode,
                    status.st_size,
                    status.st_mtime_ns,
                    payload,
                )
            )
        return result

    before = snapshot()

    def forbidden(*args, **kwargs):
        raise AssertionError("archive command touched the target filesystem")

    with monkeypatch.context() as context:
        for name in ("open", "stat", "lstat", "rename", "mkdir", "unlink", "listdir"):
            context.setattr(benchmark.os, name, forbidden)
        with pytest.raises(RuntimeError, match="unsupported.*append-only"):
            archive_failed_distributed_attempt(evidence_dir)

    assert snapshot() == before


def test_check_evidence_rejects_every_noncanonical_file_topology(tmp_path):
    import stage1.benchmark as benchmark

    data_path, image_root, _, launcher = _smoke_fixture(tmp_path)
    evidence_dir = tmp_path / "evidence" / "full"
    kwargs = dict(
        method="full", model="Qwen/Qwen2.5-VL-3B-Instruct",
        data_path=data_path, image_root=image_root, micro_batch=2,
        gradient_accumulation_steps=2, world_size=2, initial_max_steps=3,
        resumed_max_steps=4,
        deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
        evidence_dir=evidence_dir, launcher_path=launcher,
    )
    benchmark.run_or_load_distributed_witness(**kwargs)
    canonical = evidence_dir / "evidence.json"
    backing = evidence_dir / "backing.json"
    canonical.rename(backing)
    args = SimpleNamespace(
        evidence_dir=str(evidence_dir), method="full", micro_batch=2, world_size=2
    )

    canonical.symlink_to(backing.name)
    with pytest.raises(ValueError, match="canonical.*regular"):
        benchmark._check_evidence(args)
    canonical.unlink()

    os.link(backing, canonical)
    with pytest.raises(ValueError, match="canonical.*alias"):
        benchmark._check_evidence(args)
    canonical.unlink()

    canonical.mkdir()
    with pytest.raises(ValueError, match="canonical.*regular"):
        benchmark._check_evidence(args)
    canonical.rmdir()

    os.mkfifo(canonical)
    with pytest.raises(ValueError, match="canonical.*regular"):
        benchmark._check_evidence(args)


def test_canonical_reader_rejects_fifo_swap_without_blocking(tmp_path, monkeypatch):
    import stage1.benchmark as benchmark

    data_path, image_root, _, launcher = _smoke_fixture(tmp_path)
    evidence_dir = tmp_path / "evidence" / "full"
    benchmark.run_or_load_distributed_witness(
        method="full", model="Qwen/Qwen2.5-VL-3B-Instruct",
        data_path=data_path, image_root=image_root, micro_batch=2,
        gradient_accumulation_steps=2, world_size=2, initial_max_steps=3,
        resumed_max_steps=4,
        deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
        evidence_dir=evidence_dir, launcher_path=launcher,
    )
    canonical = evidence_dir / "evidence.json"
    original_open = os.open
    swapped = False

    def swap_to_fifo_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "evidence.json" and kwargs.get("dir_fd") is not None:
            if not flags & os.O_NONBLOCK:
                raise AssertionError("canonical FIFO would be opened in blocking mode")
            if not swapped:
                swapped = True
                canonical.unlink()
                os.mkfifo(canonical)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(benchmark.os, "open", swap_to_fifo_before_open)
    with pytest.raises(ValueError, match="canonical evidence (changed|must be)"):
        benchmark._check_evidence(
            SimpleNamespace(
                evidence_dir=str(evidence_dir), method="full",
                micro_batch=2, world_size=2,
            )
        )


def test_canonical_reader_rejects_name_replacement_during_read(
    tmp_path, monkeypatch
):
    import stage1.benchmark as benchmark

    data_path, image_root, _, launcher = _smoke_fixture(tmp_path)
    evidence_dir = tmp_path / "evidence" / "full"
    benchmark.run_or_load_distributed_witness(
        method="full", model="Qwen/Qwen2.5-VL-3B-Instruct",
        data_path=data_path, image_root=image_root, micro_batch=2,
        gradient_accumulation_steps=2, world_size=2, initial_max_steps=3,
        resumed_max_steps=4,
        deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
        evidence_dir=evidence_dir, launcher_path=launcher,
    )
    canonical = evidence_dir / "evidence.json"
    displaced = evidence_dir / "evidence-original.json"
    replacement = canonical.read_bytes()
    original_read = os.read
    swapped = False

    def replace_name_after_read(fd, size):
        nonlocal swapped
        data = original_read(fd, size)
        if data and not swapped:
            swapped = True
            canonical.rename(displaced)
            canonical.write_bytes(replacement)
        return data

    monkeypatch.setattr(benchmark.os, "read", replace_name_after_read)
    with pytest.raises(ValueError, match="canonical evidence changed"):
        benchmark._check_evidence(
            SimpleNamespace(
                evidence_dir=str(evidence_dir), method="full",
                micro_batch=2, world_size=2,
            )
        )


def test_legacy_flat_failure_is_retained_while_registry_retry_uses_attempt_2(
    tmp_path,
):
    import stage1.benchmark as benchmark

    data_path, image_root, records, launcher = _smoke_fixture(tmp_path)
    evidence_dir = tmp_path / "evidence" / "full"
    failed_log = evidence_dir / "initial.log"
    failed_log.parent.mkdir(parents=True)
    failed_log.write_bytes(b"original failed attempt\n")
    before = failed_log.stat()
    kwargs = dict(
        method="full", model="Qwen/Qwen2.5-VL-3B-Instruct",
        data_path=data_path, image_root=image_root, micro_batch=2,
        gradient_accumulation_steps=2, world_size=2, initial_max_steps=3,
        resumed_max_steps=4,
        deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
        evidence_dir=evidence_dir, launcher_path=launcher,
    )

    observed = benchmark.run_or_load_distributed_witness(**kwargs)

    after = failed_log.stat()
    assert observed.resume_final_step == 4
    assert failed_log.read_bytes() == b"original failed attempt\n"
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino, before.st_size, before.st_mtime_ns
    )
    attempts = sorted((evidence_dir / "attempts").iterdir())
    assert [attempt.name for attempt in attempts] == ["attempt-2"]
    assert (attempts[0] / "initial.log").is_file()
    assert (attempts[0] / "resume.log").is_file()
    assert len(list(records.iterdir())) == 2


def test_two_failed_smoke_runs_append_distinct_attempts_without_overwrite(tmp_path):
    import stage1.benchmark as benchmark

    data_path, image_root, records, _ = _smoke_fixture(tmp_path)
    launcher = tmp_path / "failing-smoke-launcher"
    _write_failing_smoke_launcher(launcher, records)
    evidence_dir = tmp_path / "evidence" / "full"
    kwargs = dict(
        method="full",
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        data_path=data_path,
        image_root=image_root,
        micro_batch=2,
        gradient_accumulation_steps=2,
        world_size=2,
        initial_max_steps=3,
        resumed_max_steps=4,
        deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
        evidence_dir=evidence_dir,
        launcher_path=launcher,
    )

    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError):
            benchmark.run_or_load_distributed_witness(**kwargs)

    attempts = sorted((evidence_dir / "attempts").iterdir())
    assert [attempt.name for attempt in attempts] == ["attempt-1", "attempt-2"]
    logs = [attempt / "initial.log" for attempt in attempts]
    assert all(log.is_file() for log in logs)
    assert logs[0].stat().st_ino != logs[1].stat().st_ino
    process_records = [json.loads(path.read_text()) for path in records.iterdir()]
    assert len(process_records) == 2
    assert len({record["pid"] for record in process_records}) == 2
    assert {Path(record["output"]).parent.name for record in process_records} == {
        "attempt-1",
        "attempt-2",
    }


def test_concurrent_smoke_attempt_allocators_never_share_or_replace_a_target(
    tmp_path,
):
    import stage1.benchmark as benchmark

    registry = tmp_path / "evidence" / "full"

    def allocate_once():
        try:
            return benchmark._allocate_smoke_attempt(registry)
        except RuntimeError as error:
            assert "execution is in progress" in str(error)
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: allocate_once(), range(2)))
    attempts = [attempt for attempt in results if attempt is not None]
    while len(attempts) < 2:
        attempts.append(benchmark._allocate_smoke_attempt(registry))

    assert {attempt.name for attempt in attempts} == {"attempt-1", "attempt-2"}
    assert len({attempt.stat().st_ino for attempt in attempts}) == 2


def test_smoke_attempt_allocator_never_overwrites_precreated_entries(tmp_path):
    import stage1.benchmark as benchmark

    registry = tmp_path / "evidence" / "full"
    prior = registry / "attempts" / "attempt-1"
    prior.mkdir(parents=True)
    sentinel = prior / "initial.log"
    sentinel.write_bytes(b"untrusted prior attempt")
    before = sentinel.stat()

    allocated = benchmark._allocate_smoke_attempt(registry)

    after = sentinel.stat()
    assert allocated.name == "attempt-2"
    assert sentinel.read_bytes() == b"untrusted prior attempt"
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )


@pytest.mark.parametrize("bad_name", ["attempt-0", "attempt-01", "foreign"])
def test_smoke_attempt_allocator_rejects_noncanonical_registry_entries(
    tmp_path, bad_name
):
    import stage1.benchmark as benchmark

    attempts = tmp_path / "evidence" / "full" / "attempts"
    attempts.mkdir(parents=True)
    (attempts / bad_name).mkdir()

    with pytest.raises(ValueError, match="noncanonical"):
        benchmark._allocate_smoke_attempt(attempts.parent)

    assert not (attempts / "attempt-1").exists()


def test_malicious_canonical_smoke_evidence_is_never_overwritten_or_launched(
    tmp_path,
):
    import stage1.benchmark as benchmark

    data_path, image_root, records, launcher = _smoke_fixture(tmp_path)
    evidence_dir = tmp_path / "evidence" / "full"
    evidence_dir.mkdir(parents=True)
    canonical = evidence_dir / "evidence.json"
    canonical.write_bytes(b"{}\n")
    before = canonical.stat()

    with pytest.raises(ValueError, match="registry envelope"):
        benchmark.run_or_load_distributed_witness(
            method="full", model="Qwen/Qwen2.5-VL-3B-Instruct",
            data_path=data_path, image_root=image_root, micro_batch=2,
            gradient_accumulation_steps=2, world_size=2,
            initial_max_steps=3, resumed_max_steps=4,
            deepspeed_config=REPO_ROOT / "configs/config_stage3.json",
            evidence_dir=evidence_dir, launcher_path=launcher,
        )

    after = canonical.stat()
    assert canonical.read_bytes() == b"{}\n"
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    assert list(records.iterdir()) == []


@pytest.mark.parametrize("topology", ["failure", "missing", "symlink"])
def test_archive_failed_attempt_cli_is_unsupported_and_preserves_history(
    tmp_path, topology
):
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    evidence_dir = evidence_root / "full"
    if topology == "failure":
        evidence_dir.mkdir()
        (evidence_dir / "initial.log").write_bytes(b"retained failure")
    elif topology == "symlink":
        retained = tmp_path / "retained-failure"
        retained.mkdir()
        (retained / "initial.log").write_bytes(b"retained symlink target")
        evidence_dir.symlink_to(retained, target_is_directory=True)

    def snapshot():
        result = []
        for path in sorted(tmp_path.rglob("*"), key=lambda item: str(item)):
            status = path.lstat()
            payload = None
            if path.is_symlink():
                payload = os.readlink(path)
            elif path.is_file():
                payload = path.read_bytes()
            result.append(
                (
                    str(path.relative_to(tmp_path)),
                    status.st_ino,
                    status.st_mode,
                    status.st_size,
                    status.st_mtime_ns,
                    payload,
                )
            )
        return result

    before = snapshot()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
            "-m",
            "src.stage1.benchmark",
            "archive-failed-attempt",
            "--evidence-dir",
            str(evidence_dir),
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "unsupported/deprecated" in completed.stderr
    assert "append-only attempts preserve failures in place" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert snapshot() == before
def test_invalid_distributed_method_environment_fails_collection_without_cuda(tmp_path):
    environment = dict(os.environ)
    environment.update(
        ILVR_DISTRIBUTED_METHOD="invalid-method",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join(
            (str(REPO_ROOT), str(REPO_ROOT / "transformers/src"), str(REPO_ROOT / "src"))
        ),
    )
    completed = subprocess.run(
        [
            str(REPO_ROOT / "venv/ilvr-venv/bin/pytest"),
            "tests/stage1/test_distributed_smoke.py",
            "--collect-only",
            "-q",
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "invalid ILVR_DISTRIBUTED_METHOD" in completed.stderr + completed.stdout
    assert "cuda_initialized=True" not in completed.stderr + completed.stdout
