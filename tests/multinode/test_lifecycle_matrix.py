from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess

import pytest

from src.multinode import (
    build_lifecycle_matrix,
    build_lifecycle_plans,
    run_lifecycle_matrix,
)


EXPECTED_IDENTITIES = [
    (stage, method, zero_stage)
    for stage in ("stage1", "stage2")
    for method in ("full", "lora", "dora")
    for zero_stage in (1, 2, 3)
]
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_matrix_contains_exactly_eighteen_cases_in_binding_order():
    """Catches set/dict iteration changing operator-visible matrix order."""
    cases = build_lifecycle_matrix()

    assert len(cases) == len(set(cases)) == 18
    assert [
        (case.stage, case.method, case.zero_stage) for case in cases
    ] == EXPECTED_IDENTITIES
    assert {
        (case.initial_steps, case.resumed_steps) for case in cases
    } == {(2, 3)}


def test_lifecycle_plans_bind_safe_child_ids_ports_and_two_attempt_arrays(tmp_path):
    """Catches a case changing identity or port between initial and resume."""
    result_root = tmp_path / "results"
    plans = build_lifecycle_plans(
        parent_run_id="four-gpu-matrix",
        first_port=29700,
        result_root=result_root,
        python_executable="/repo/venv/ilvr-venv/bin/python",
    )

    assert len(plans) == 18
    assert [plan.main_process_port for plan in plans] == list(range(29700, 29718))
    for index, plan in enumerate(plans):
        stage, method, zero_stage = EXPECTED_IDENTITIES[index]
        child = f"four-gpu-matrix-{stage}-{method}-zero{zero_stage}"
        assert plan.child_run_id == child
        assert plan.initial.attempt_id == "initial"
        assert (plan.initial.from_step, plan.initial.to_step) == (0, 2)
        assert plan.resume.attempt_id == "resume"
        assert (plan.resume.from_step, plan.resume.to_step) == (2, 3)
        for attempt in (plan.initial, plan.resume):
            command = list(attempt.command)
            assert command[command.index("--run-id") + 1] == child
            assert command[command.index("--main-process-port") + 1] == str(
                29700 + index
            )
            assert command[command.index("--attempt-id") + 1] == attempt.attempt_id
        assert "checkpoint-2" in " ".join(plan.resume.command)


@pytest.mark.parametrize("first_port", [1023, 65519])
def test_lifecycle_plans_validate_the_complete_port_range_before_use(
    tmp_path, first_port
):
    """Catches the eighteenth port failing only after earlier cases ran."""
    with pytest.raises(ValueError, match="18 ports"):
        build_lifecycle_plans(
            parent_run_id="four-gpu-matrix",
            first_port=first_port,
            result_root=tmp_path / "results",
            python_executable="/repo/python",
        )


def test_lifecycle_dry_run_has_no_filesystem_or_runner_side_effect(tmp_path):
    """Catches dry-run reserving the parent namespace or starting a command."""
    result_root = tmp_path / "missing-results"

    def forbidden_runner(_command):
        raise AssertionError("dry-run called the external runner")

    plans = run_lifecycle_matrix(
        parent_run_id="four-gpu-matrix-dry",
        first_port=29700,
        result_root=result_root,
        python_executable="/repo/python",
        command_runner=forbidden_runner,
        dry_run=True,
    )

    assert len(plans) == 18
    assert not result_root.exists()


def _write_training_witness(
    result_root: Path,
    *,
    child_run_id: str,
    attempt_id: str,
    stage: str,
    method: str,
    zero_stage: int,
    final_step: int,
) -> Path:
    run_root = result_root / child_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        result_root.parent / "checkpoints" / child_run_id / f"checkpoint-{final_step}"
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "rank-shard.bin").write_bytes(b"bounded-fake-shard")
    finite = run_root / f"{attempt_id}-producer-evidence.json"
    finite.write_text(
        json.dumps({"benchmark_evidence": {"finite_loss": True}}) + "\n",
        encoding="utf-8",
    )
    finite_bytes = finite.read_bytes()
    witness = run_root / f"{attempt_id}-train-witness.json"
    witness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": child_run_id,
                "attempt_id": attempt_id,
                "stage": stage,
                "method": method,
                "zero_stage": zero_stage,
                "batch_per_gpu": 2 if stage == "stage1" else 1,
                "world_size": 4,
                "machine_ranks": [0, 1],
                "exit_codes": {"0": 0, "1": 0},
                "checkpoint_path": str(checkpoint.parent),
                "finite_loss_evidence": {
                    "status": "validated",
                    "path": str(finite),
                    "sha256": hashlib.sha256(finite_bytes).hexdigest(),
                    "size": len(finite_bytes),
                    "finite_loss": True,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return witness


def test_matrix_runs_initial_then_resume_and_completes_only_after_all_cases(tmp_path):
    """Catches retry/interleaving or early parent completion publication."""
    result_root = tmp_path / "results"
    seen: list[tuple[str, str]] = []

    def runner(command):
        values = list(command)
        child = values[values.index("--run-id") + 1]
        attempt = values[values.index("--attempt-id") + 1]
        stage = values[values.index("-m") + 2].removeprefix("train-")
        zero_stage = int(values[values.index("--zero-stage") + 1])
        method = child.split("-")[-2]
        final_step = 2 if attempt == "initial" else 3
        if attempt == "resume":
            assert seen[-1] == (child, "initial")
        assert not (result_root / "four-gpu-matrix" / "matrix-complete.json").exists()
        witness = _write_training_witness(
            result_root,
            child_run_id=child,
            attempt_id=attempt,
            stage=stage,
            method=method,
            zero_stage=zero_stage,
            final_step=final_step,
        )
        seen.append((child, attempt))
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"training_witness": str(witness)}), stderr=""
        )

    plans = run_lifecycle_matrix(
        parent_run_id="four-gpu-matrix",
        first_port=29700,
        result_root=result_root,
        python_executable="/repo/python",
        command_runner=runner,
    )

    assert len(seen) == 36
    assert seen[:4] == [
        ("four-gpu-matrix-stage1-full-zero1", "initial"),
        ("four-gpu-matrix-stage1-full-zero1", "resume"),
        ("four-gpu-matrix-stage1-full-zero2", "initial"),
        ("four-gpu-matrix-stage1-full-zero2", "resume"),
    ]
    completion = json.loads(
        (result_root / "four-gpu-matrix" / "matrix-complete.json").read_text()
    )
    assert len(completion["cases"]) == len(plans) == 18


def test_matrix_stops_without_retry_or_completion_on_bad_attempt_witness(tmp_path):
    """Catches a malformed initial attempt being retried or followed by resume."""
    result_root = tmp_path / "results"
    calls = 0

    def runner(command):
        nonlocal calls
        calls += 1
        values = list(command)
        child = values[values.index("--run-id") + 1]
        witness = result_root / child / "initial-train-witness.json"
        witness.parent.mkdir(parents=True)
        witness.write_text('{"schema_version":1}\n', encoding="utf-8")
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"training_witness": str(witness)}), stderr=""
        )

    with pytest.raises(ValueError, match="training witness"):
        run_lifecycle_matrix(
            parent_run_id="four-gpu-matrix",
            first_port=29700,
            result_root=result_root,
            python_executable="/repo/python",
            command_runner=runner,
        )

    assert calls == 1
    assert not (result_root / "four-gpu-matrix" / "matrix-complete.json").exists()


def test_matrix_prechecks_parent_namespace_before_any_attempt(tmp_path):
    """Catches stale case evidence being mixed into a new matrix run."""
    result_root = tmp_path / "results"
    parent = result_root / "four-gpu-matrix"
    parent.mkdir(parents=True)
    (parent / "stale.json").write_text("{}\n", encoding="utf-8")
    calls = 0

    def runner(_command):
        nonlocal calls
        calls += 1
        raise AssertionError("stale namespace reached runner")

    with pytest.raises(FileExistsError, match="parent result namespace"):
        run_lifecycle_matrix(
            parent_run_id="four-gpu-matrix",
            first_port=29700,
            result_root=result_root,
            python_executable="/repo/python",
            command_runner=runner,
        )

    assert calls == 0


def test_wrapper_dry_run_prints_exactly_eighteen_lifecycle_plan_objects(tmp_path):
    """Catches the operator wrapper mutating files or emitting attempt fragments."""
    result_root = tmp_path / "missing-results"
    completed = subprocess.run(
        [
            str(REPO_ROOT / "run_multinode_lifecycle_matrix.sh"),
            "--run-id",
            "four-gpu-matrix-dry",
            "--first-port",
            "29700",
            "--result-root",
            str(result_root),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    plans = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(plans) == 18
    assert all(set(plan) == {
        "stage",
        "method",
        "zero_stage",
        "child_run_id",
        "main_process_port",
        "initial",
        "resume",
    } for plan in plans)
    assert [plan["main_process_port"] for plan in plans] == list(range(29700, 29718))
    assert not result_root.exists()
