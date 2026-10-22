from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from src.multinode import (
    _execute_lifecycle_attempts,
    build_lifecycle_matrix,
    build_lifecycle_plans,
    run_lifecycle_matrix,
)
from src.multinode_evidence import (
    build_lifecycle_phase_evidence,
    decode_lifecycle_phase_evidence,
)
from src.multinode_runtime import (
    SubprocessCoordinatorRunner,
    build_owned_lifecycle_harness_commands,
    canonical_command_digest,
)
from src.multinode_artifacts import build_inference_manifest
from tests.multinode.test_artifacts import loader_manifest, write_loader_artifact


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


def test_matrix_plans_exact_initial_and_resume_harness_jobs_per_cell(tmp_path):
    """Catches a phase running before Task 6 makes the checkpoint node-local."""
    plans = build_lifecycle_plans(
        parent_run_id="four-gpu-matrix",
        first_port=29700,
        result_root=tmp_path / "results",
        python_executable="/repo/venv/ilvr-venv/bin/python",
        commit="a" * 40,
    )

    assert len(plans) == 18
    for plan in plans:
        assert plan.initial_launch_plan.attempt_id == "initial"
        assert plan.resume_launch_plan.attempt_id == "resume"
        expected_harness = (
            "/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/tests/stage1/zero_compat_diagnostic.py"
            if plan.case.stage == "stage1"
            else "/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/tests/stage2/integration/tiny_stage2_train.py"
        )
        for phase, commands in (
            ("initial", plan.initial_commands),
            ("resume", plan.resume_commands),
        ):
            assert len(commands.node_specs) == 2
            for machine_rank, node in enumerate(commands.node_specs):
                command = node.command
                child = command[command.index("--") + 1 :]
                assert child[0] == "/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/venv/ilvr-venv/bin/accelerate"
                assert child[1] == "launch"
                assert child[child.index("--num_machines") + 1] == "2"
                assert child[child.index("--num_processes") + 1] == "4"
                assert child[child.index("--machine_rank") + 1] == str(machine_rank)
                assert child[child.index("--main_process_ip") + 1] == "172.16.11.41"
                assert child[child.index("--main_process_port") + 1] == str(
                    plan.main_process_port
                )
                assert child[child.index("--deepspeed_multinode_launcher") + 1] == (
                    "standard"
                )
                assert expected_harness in child
                harness_index = child.index(expected_harness)
                harness_args = child[harness_index + 1 :]
                method_flag = (
                    "--method" if plan.case.stage == "stage1" else "--tuning-method"
                )
                assert harness_args == (
                    "--phase",
                    phase,
                    "--zero-stage",
                    str(plan.case.zero_stage),
                        method_flag,
                        plan.case.method,
                        "--deepspeed",
                        str(plan.initial_launch_plan.deepspeed_config),
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
    )[0].initial_launch_plan

    with pytest.raises(ValueError, match="approved harness"):
        build_owned_lifecycle_harness_commands(
            launch_plan,
            commit="a" * 40,
            phase="initial",
            harness_path=Path("/tmp/arbitrary.py"),
        )
    with pytest.raises(ValueError, match="exact harness arguments"):
        build_owned_lifecycle_harness_commands(
            launch_plan,
            commit="a" * 40,
            phase="initial",
            harness_arguments=("--zero-stage", "3"),
        )
    with pytest.raises(ValueError, match="phase"):
        build_owned_lifecycle_harness_commands(
            launch_plan,
            commit="a" * 40,
            phase="lifecycle",
        )


def test_lifecycle_worker_environment_resolves_canonical_venv_deepspeed():
    """Catches Accelerate failing because its sealed PATH cannot find DeepSpeed."""
    plan = build_lifecycle_plans(
        parent_run_id="four-gpu-matrix",
        first_port=29700,
        result_root=Path("/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/results"),
        python_executable="/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/venv/ilvr-venv/bin/python",
        commit="a" * 40,
    )[0]

    for node in plan.initial_commands.node_specs:
        path = dict(node.environment)["PATH"]
        assert path.split(":")[0] == "/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/venv/ilvr-venv/bin"
        checkout_bin = str(Path(__file__).resolve().parents[2] / "venv/ilvr-venv/bin")
        resolved = shutil.which("deepspeed", path=checkout_bin + ":" + path.split(":", 1)[1])
        assert resolved == checkout_bin + "/deepspeed"
        assert canonical_command_digest(
            node.command, dict(node.environment)
        ) == plan.initial_commands.command_digests[node.machine_rank]


@pytest.mark.parametrize(
    "relative_harness",
    (
        "tests/stage1/zero_compat_diagnostic.py",
        "tests/stage2/integration/tiny_stage2_train.py",
    ),
)
def test_lifecycle_harness_imports_with_python_build_marker(relative_harness):
    """Catches packaging rejecting the host's process-local Python suffix."""
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{REPO_ROOT / 'transformers/src'}:{REPO_ROOT}:{REPO_ROOT / 'src'}",
    }

    completed = subprocess.run(
        [
            str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
            str(REPO_ROOT / relative_harness),
            "--help",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_lifecycle_harness_processors_save_required_loader_assets(tmp_path):
    """Catches four-rank artifact collection rejecting a tiny checkpoint."""
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{REPO_ROOT / 'transformers/src'}:{REPO_ROOT}:{REPO_ROOT / 'src'}",
    }
    for name, save_expression in (
        (
            "stage1",
            "from tests.stage1.zero_compat_diagnostic import _tiny_dependencies; "
            "_tiny_dependencies().load_processor('/nonexistent').save_pretrained({path!r})",
        ),
        (
            "stage2",
            "from tests.stage2.integration.tiny_stage2_train import TinyStage2Processor; "
            "TinyStage2Processor().save_pretrained({path!r})",
        ),
    ):
        destination = tmp_path / name
        completed = subprocess.run(
            [
                str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
                "-c",
                save_expression.format(path=str(destination)),
            ],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert (destination / "preprocessor_config.json").is_file()
        assert (destination / "tokenizer_config.json").is_file()
        assert (destination / "tokenizer.json").is_file()


@pytest.mark.parametrize("method", ("full", "lora", "dora"))
def test_stage2_tiny_processor_assets_pass_loader_and_manifest_selection(
    tmp_path, method
):
    """Catches any Stage 2 lifecycle method publishing an unloadable artifact."""
    destination = tmp_path / method
    write_loader_artifact(destination, stage="stage2", method=method)
    for name in (
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ):
        (destination / name).unlink()
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": (
            f"{REPO_ROOT / 'transformers/src'}:{REPO_ROOT}:{REPO_ROOT / 'src'}"
        ),
    }
    completed = subprocess.run(
        [
            str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
            "-c",
            (
                "from tests.stage2.integration.tiny_stage2_train "
                "import TinyStage2Processor; "
                f"TinyStage2Processor().save_pretrained({str(destination)!r})"
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    manifest = build_inference_manifest(
        destination,
        loader_manifest(destination, stage="stage2", method=method),
    )
    selected = {entry.relative_path.name for entry in manifest.entries}
    assert {
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    } <= selected
    expected_weight = (
        "model.safetensors" if method == "full" else "adapter_model.safetensors"
    )
    assert expected_weight in selected


def test_lifecycle_plans_bind_safe_child_ids_ports_and_two_ordered_phase_jobs(tmp_path):
    """Catches either phase changing the shared child identity or port."""
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
        assert plan.initial_launch_plan.run_id == child
        assert plan.resume_launch_plan.run_id == child
        assert plan.initial_launch_plan.attempt_id == "initial"
        assert plan.resume_launch_plan.attempt_id == "resume"
        assert plan.initial_launch_plan.main_process_port == 29700 + index
        assert plan.resume_launch_plan.main_process_port == 29700 + index
        assert len(plan.initial_commands.node_specs) == 2
        assert len(plan.resume_commands.node_specs) == 2
        assert plan.checkpoint_path.name == "checkpoint-2"
        assert plan.checkpoint_path == Path("/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/checkpoints") / child / "checkpoint-2"
        assert plan.initial_witness_path == Path("/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/results") / child / "initial-witness.json"
        assert plan.resume_witness_path == Path("/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/results") / child / "resume-witness.json"


def _phase_bytes(plan, phase, *, gradient):
    global_step = 2 if phase == "initial" else 3
    optimizer_step = 2 if phase == "initial" else (
        1 if plan.case.stage == "stage1" else 3
    )
    return (json.dumps(build_lifecycle_phase_evidence(
        phase=phase,
        stage=plan.case.stage,
        method=plan.case.method,
        zero_stage=plan.case.zero_stage,
        config_digest="a" * 64,
        experiment_id=plan.child_run_id,
        checkpoint_path=plan.checkpoint_path,
        artifact_kind="full",
        rank_records=[{
            "rank": rank,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "loss": 0.25 + rank,
            "gradient_norm": gradient + rank,
            "worker_error": None,
        } for rank in range(4)],
    ), sort_keys=True, separators=(",", ":")) + "\n").encode()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("schema_version", True),
        ("world_size", 4.0),
        ("process_ranks", [False, 1, 2, 3]),
        ("global_step", 2.0),
        ("finite_loss", 1),
    ),
)
def test_lifecycle_phase_decoder_rejects_top_level_bool_float_aliases(
    tmp_path, field, replacement
):
    plan = build_lifecycle_plans(
        parent_run_id="matrix-parent", first_port=29700,
        result_root=tmp_path / "results", python_executable="/repo/python",
        commit="a" * 40,
        config_digests={1: "a" * 64, 2: "a" * 64, 3: "a" * 64},
    )[0]
    payload = json.loads(_phase_bytes(plan, "initial", gradient=1.0))
    payload[field] = replacement

    with pytest.raises(ValueError, match="lifecycle phase"):
        decode_lifecycle_phase_evidence(
            json.dumps(payload).encode("utf-8"),
            phase="initial", stage="stage1", method="full", zero_stage=1,
            config_digest="a" * 64, experiment_id=plan.child_run_id,
            checkpoint_path=plan.checkpoint_path,
        )


def test_resume_never_starts_when_checkpoint_redistribution_fails(tmp_path, monkeypatch):
    """Catches local ranks 2/3 loading before their loader artifacts arrive."""
    plan = build_lifecycle_plans(
        parent_run_id="matrix-parent", first_port=29700,
        result_root=tmp_path / "results", python_executable="/repo/python",
        commit="a" * 40, config_digests={1: "a" * 64, 2: "a" * 64, 3: "a" * 64},
    )[0]
    events = []

    class Runner:
        def __init__(self, phase):
            self.phase = phase

        def read_lifecycle_phase_evidence(self, machine_rank, phase):
            events.append(f"read-{phase}-rank{machine_rank}")
            return _phase_bytes(plan, phase, gradient=1.0 if phase == "initial" else 9.0)

    initial_runner = Runner("initial")
    resume_runner = Runner("resume")
    monkeypatch.setattr("src.multinode.start_rank0_then_rank1", lambda launch, runner, **kwargs: events.append(f"start-{launch.attempt_id}"))
    monkeypatch.setattr("src.multinode.monitor_owned_jobs", lambda launch, runner, **kwargs: events.append(f"monitor-{launch.attempt_id}") or {0: 0, 1: 0})
    monkeypatch.setattr("src.multinode.collect_checkpoint_artifacts", lambda *args, **kwargs: events.append("collect") or (object(), object()))

    def fail_transfer(*args, **kwargs):
        events.append("redistribute")
        raise RuntimeError("transfer failed")

    monkeypatch.setattr("src.multinode.redistribute_resume_checkpoint", fail_transfer)
    with pytest.raises(RuntimeError, match="transfer failed"):
        _execute_lifecycle_attempts(
            plan, initial_runner=initial_runner, resume_runner=resume_runner,
            commit="a" * 40,
        )
    assert events.index("collect") < events.index("redistribute")
    assert "start-resume" not in events


def test_phase_evidence_keeps_distinct_initial_and_resume_gradients(
    tmp_path, monkeypatch
):
    """Catches a case summary copying the initial gradient into resume evidence."""
    plan = build_lifecycle_plans(
        parent_run_id="matrix-parent", first_port=29700,
        result_root=tmp_path / "results", python_executable="/repo/python",
        commit="a" * 40, config_digests={1: "a" * 64, 2: "a" * 64, 3: "a" * 64},
    )[0]

    class Runner:
        def read_lifecycle_phase_evidence(self, machine_rank, phase):
            del machine_rank
            return _phase_bytes(
                plan, phase, gradient=1.0 if phase == "initial" else 9.0
            )

    monkeypatch.setattr("src.multinode.start_rank0_then_rank1", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.multinode.monitor_owned_jobs", lambda *args, **kwargs: {0: 0, 1: 0})
    monkeypatch.setattr("src.multinode.collect_checkpoint_artifacts", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr("src.multinode.redistribute_resume_checkpoint", lambda *args, **kwargs: (object(),))
    observed = _execute_lifecycle_attempts(
        plan, initial_runner=Runner(), resume_runner=Runner(), commit="a" * 40,
    )
    assert observed["initial_payload"]["ranks"][0]["gradient_norm"] == 1.0
    assert observed["resume_payload"]["ranks"][0]["gradient_norm"] == 9.0


def test_remote_phase_evidence_uses_only_the_fixed_bounded_node_cli(tmp_path):
    plan = build_lifecycle_plans(
        parent_run_id="matrix-parent", first_port=29700,
        result_root=tmp_path / "results", python_executable="/repo/python",
        commit="a" * 40,
        config_digests={1: "a" * 64, 2: "a" * 64, 3: "a" * 64},
    )[0]
    calls = []
    expected = _phase_bytes(plan, "initial", gradient=1.0)

    def command_runner(command, *, input_bytes=None, timeout_seconds=None):
        calls.append((tuple(command), input_bytes, timeout_seconds))
        return subprocess.CompletedProcess(
            command, 0, stdout=expected.decode("utf-8"), stderr=""
        )

    runner = SubprocessCoordinatorRunner(
        plan.initial_launch_plan,
        commands=plan.initial_commands,
        commit="a" * 40,
        log_root=tmp_path / "logs",
        command_runner=command_runner,
    )
    assert runner.read_lifecycle_phase_evidence(0, "initial") == expected
    assert calls[0][0][:6] == (
        "/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        "korea@172.16.11.41",
    )
    assert len(calls[0][0]) == 7
    remote = shlex.split(calls[0][0][6])
    assert remote[:7] == [
        "/usr/bin/docker", "exec", "--workdir", "/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022",
        "llm_kd_rdma", "/usr/bin/bash", "-lc",
    ]
    assert len(remote) == 8
    assert shlex.split(remote[7]) == [
        "env", "PYTHONPATH=/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022:/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/src",
        "/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/venv/ilvr-venv/bin/python",
        "-m", "src.multinode", "lifecycle-evidence-show",
        "--run-id", plan.child_run_id, "--phase", "initial",
    ]
    assert calls[0][2] == 30


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
    assert all(set(plan["lifecycle"]) == {"initial", "redistribute", "resume"} for plan in plans)
    assert all(plan["lifecycle"]["initial"]["steps"] == [0, 2] for plan in plans)
    assert all(plan["lifecycle"]["redistribute"]["checkpoint"] == "checkpoint-2" for plan in plans)
    assert all(plan["lifecycle"]["resume"]["steps"] == [2, 3] for plan in plans)
    assert [plan["main_process_port"] for plan in plans] == list(range(29700, 29718))
    assert not result_root.exists()
