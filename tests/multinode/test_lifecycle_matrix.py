from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from src.multinode import (
    build_lifecycle_matrix,
    build_lifecycle_plans,
    run_lifecycle_matrix,
)
from src.multinode_runtime import build_owned_lifecycle_harness_commands


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


def test_matrix_plans_one_exact_approved_harness_job_per_cell(tmp_path):
    """Catches matrix cells reverting to incomplete normal training commands."""
    plans = build_lifecycle_plans(
        parent_run_id="four-gpu-matrix",
        first_port=29700,
        result_root=tmp_path / "results",
        python_executable="/repo/venv/ilvr-venv/bin/python",
        commit="a" * 40,
    )

    assert len(plans) == 18
    for plan in plans:
        assert not hasattr(plan, "initial")
        assert not hasattr(plan, "resume")
        assert len(plan.commands.node_specs) == 2
        expected_harness = (
            "/root/workspace/ILVR/tests/stage1/zero_compat_diagnostic.py"
            if plan.case.stage == "stage1"
            else "/root/workspace/ILVR/tests/stage2/integration/tiny_stage2_train.py"
        )
        for machine_rank, node in enumerate(plan.commands.node_specs):
            command = node.command
            child = command[command.index("--") + 1 :]
            assert child[0] == "/root/workspace/ILVR/venv/ilvr-venv/bin/accelerate"
            assert child[1] == "launch"
            assert child[child.index("--num_machines") + 1] == "2"
            assert child[child.index("--num_processes") + 1] == "4"
            assert child[child.index("--machine_rank") + 1] == str(machine_rank)
            assert child[child.index("--main_process_ip") + 1] == "10.1.102.41"
            assert child[child.index("--main_process_port") + 1] == str(
                plan.main_process_port
            )
            assert expected_harness in child
            harness_index = child.index(expected_harness)
            harness_args = child[harness_index + 1 :]
            method_flag = "--method" if plan.case.stage == "stage1" else "--tuning-method"
            assert harness_args == (
                "--zero-stage",
                str(plan.case.zero_stage),
                method_flag,
                plan.case.method,
                "--deepspeed",
                f"/root/workspace/ILVR/configs/deepspeed_zero{plan.case.zero_stage}.json",
                "--experiment-id",
                plan.child_run_id,
                "--initial-steps",
                "2",
                "--resumed-steps",
                "3",
            )


def test_owned_harness_builder_rejects_arbitrary_path_or_arguments():
    """Catches the lifecycle coordinator becoming an arbitrary command escape."""
    launch_plan = build_lifecycle_plans(
        parent_run_id="four-gpu-matrix",
        first_port=29700,
        result_root=Path("/root/workspace/ILVR/results"),
        python_executable="/root/workspace/ILVR/venv/ilvr-venv/bin/python",
        commit="a" * 40,
    )[0].launch_plan

    with pytest.raises(ValueError, match="approved harness"):
        build_owned_lifecycle_harness_commands(
            launch_plan,
            commit="a" * 40,
            harness_path=Path("/tmp/arbitrary.py"),
        )
    with pytest.raises(ValueError, match="exact harness arguments"):
        build_owned_lifecycle_harness_commands(
            launch_plan,
            commit="a" * 40,
            harness_arguments=("--zero-stage", "3"),
        )


def test_lifecycle_plans_bind_safe_child_ids_ports_and_one_lifecycle_job(tmp_path):
    """Catches a case changing identity or port inside its owned harness job."""
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
        assert plan.launch_plan.run_id == child
        assert plan.launch_plan.attempt_id == "lifecycle"
        assert plan.launch_plan.main_process_port == 29700 + index
        assert len(plan.commands.node_specs) == 2
        assert plan.checkpoint_path.name == "checkpoint-2"


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

    plans = run_lifecycle_matrix(
        parent_run_id="four-gpu-matrix-dry",
        first_port=29700,
        result_root=result_root,
        python_executable="/repo/python",
        dry_run=True,
    )

    assert len(plans) == 18
    assert not result_root.exists()


def test_matrix_prechecks_parent_namespace_before_any_attempt(tmp_path):
    """Catches stale case evidence being mixed into a new matrix run."""
    result_root = tmp_path / "results"
    parent = result_root / "four-gpu-matrix"
    parent.mkdir(parents=True)
    (parent / "stale.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="parent result namespace"):
        run_lifecycle_matrix(
            parent_run_id="four-gpu-matrix",
            first_port=29700,
            result_root=result_root,
            python_executable="/repo/python",
        )



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
        "lifecycle",
    } for plan in plans)
    assert all(set(plan["lifecycle"]) == {"remote", "local", "steps"} for plan in plans)
    assert all(plan["lifecycle"]["steps"] == [0, 2, 3] for plan in plans)
    assert [plan["main_process_port"] for plan in plans] == list(range(29700, 29718))
    assert not result_root.exists()
