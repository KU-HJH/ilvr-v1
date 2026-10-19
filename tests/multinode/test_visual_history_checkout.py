"""Execution-checkout bindings for the approved visual-history experiment."""

import json
import os
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import shlex
import subprocess

import pytest

import src.multinode_results as results_module
from src.artifact_paths import (
    require_canonical_artifact_path,
    resolve_run_artifact_paths,
)
from src.multinode import ClusterSpec, _node_info, build_launch_plan, build_lifecycle_plans
from src.multinode_artifacts import (
    NodeArtifactManifest,
    build_evaluation_pull_command,
    build_push_command,
    resolve_local_checkpoint_destination,
)
from src.multinode_capacity import _decode_training_witness
from src.multinode_evidence import build_lifecycle_phase_evidence
from src.multinode_runtime import (
    OwnershipRecord,
    _local_repository_module_command,
    _remote_repository_module_command,
    build_owned_tmux_commands,
    build_preflight_commands,
    build_remote_exec,
    validate_resume_path,
    validate_ownership,
)


REPOSITORY = Path(__file__).resolve().parents[2]
EXECUTION_ROOT = Path(
    "/root/workspace/ILVR/.worktrees/stage1-visual-history-only-20261019"
)
HOST_ROOT = Path(
    "/home/korea/Workspace/hjh/workspace/ILVR/.worktrees/"
    "stage1-visual-history-only-20261019"
)
ORIGINAL_ROOT = Path("/root/workspace/ILVR")
RUN_ID = "visual-history-checkout-contract"
INPUT_ARGUMENTS = (
    "--method", "dora",
    "--model", str(ORIGINAL_ROOT / "checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct"),
    "--data-path", str(ORIGINAL_ROOT / "data/TRAIN.jsonl"),
    "--image-root", str(ORIGINAL_ROOT / "data/images_comt"),
)


def launch_plan():
    return build_launch_plan(
        stage="stage1", tuning_method="dora", zero_stage=2,
        run_id=RUN_ID, main_process_port=29641,
    )


def remote_argv(command):
    docker = shlex.split(command[-1])
    return docker, shlex.split(docker[-1])


def test_plan_binds_both_nodes_to_fixed_checkout_without_changing_topology():
    plan = launch_plan()
    cluster = ClusterSpec.default()

    assert plan.cluster == cluster
    assert cluster.remote_repo_root == EXECUTION_ROOT
    assert cluster.remote_host_repo_root == HOST_ROOT
    assert (cluster.remote_host, cluster.remote_container) == (
        "korea@10.1.102.41", "llm_kd"
    )
    assert (cluster.remote_rank, cluster.local_rank) == (0, 1)
    assert (cluster.num_machines, cluster.gpus_per_node, plan.world_size) == (2, 2, 4)
    assert (cluster.master_addr, cluster.socket_ifname) == ("10.1.102.41", "bond1.102")
    assert plan.deepspeed_config == EXECUTION_ROOT / "configs/deepspeed_zero2.json"
    assert (EXECUTION_ROOT / "configs/deepspeed_zero2.json").read_bytes() == (
        REPOSITORY / "configs/deepspeed_zero2.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="fixed ILVR topology"):
        replace(plan, cluster=replace(cluster, remote_repo_root=ORIGINAL_ROOT))


def test_both_owned_commands_bind_checkout_scripts_python_outputs_and_environment():
    plan = launch_plan()
    commands = build_owned_tmux_commands(
        plan, commit="a" * 40, stage_arguments=INPUT_ARGUMENTS
    )
    docker, remote = remote_argv(commands.remote)
    assert docker[docker.index("--workdir") + 1] == str(EXECUTION_ROOT)
    assert commands.remote[5] == "korea@10.1.102.41"
    for rank, spec in enumerate(commands.node_specs):
        environment = dict(spec.environment)
        assert spec.machine_rank == rank
        assert spec.command[:4] == (
            str(EXECUTION_ROOT / "venv/ilvr-venv/bin/python"),
            "-m", "src.multinode", "node-run",
        )
        assert environment["PYTHONPATH"] == f"{EXECUTION_ROOT}:{EXECUTION_ROOT / 'src'}"
        assert environment["PATH"].split(":")[0] == str(EXECUTION_ROOT / "venv/ilvr-venv/bin")
        assert environment["ILVR_MACHINE_RANK"] == str(rank)
        assert environment["ILVR_NUM_MACHINES"] == "2"
        assert environment["ILVR_NUM_PROCESSES"] == "4"
        assert environment["ILVR_MAIN_PROCESS_IP"] == "10.1.102.41"
        assert environment["ILVR_SOCKET_IFNAME"] == "bond1.102"
        assert spec.ownership_path == EXECUTION_ROOT / f"logs/multinode/{RUN_ID}/initial/ownership-rank{rank}.json"
        assert spec.exit_path == EXECUTION_ROOT / f"logs/multinode/{RUN_ID}/initial/exit-rank{rank}.json"
        child = spec.command[spec.command.index("--") + 1:]
        assert child[:2] == ("/usr/bin/bash", str(EXECUTION_ROOT / "run_stage1_training.sh"))
        assert child[2:2 + len(INPUT_ARGUMENTS)] == INPUT_ARGUMENTS
        for flag, expected in (
            ("--output-dir", EXECUTION_ROOT / "checkpoints" / RUN_ID),
            ("--log-file", EXECUTION_ROOT / f"logs/zero-stage-compatibility/{RUN_ID}/initial-train.log"),
            ("--deepspeed-config", plan.deepspeed_config),
            ("--training-evidence", EXECUTION_ROOT / f"results/{RUN_ID}/initial-producer-evidence.json"),
        ):
            assert child[child.index(flag) + 1] == str(expected)
        actual = remote if rank == 0 else list(commands.local)
        assert tuple(actual[-len(child):]) == child


def test_preflight_and_child_module_commands_use_fixed_checkout_on_both_nodes():
    plan = launch_plan()
    inputs = tuple(INPUT_ARGUMENTS[index] for index in (3, 5, 7))
    commands = build_preflight_commands(
        plan, input_paths=inputs,
        artifact_roots=(EXECUTION_ROOT / "checkpoints", EXECUTION_ROOT / "results"),
        remote_run_paths=(), local_run_paths=(),
        remote_tmux_sessions=(), local_tmux_sessions=(),
    )
    _, remote = remote_argv(commands.remote_node_info)
    for command in (commands.local_node_info, remote):
        assert str(EXECUTION_ROOT / "venv/ilvr-venv/bin/python") in command
        assert f"PYTHONPATH={EXECUTION_ROOT}:{EXECUTION_ROOT / 'src'}" in command
        assert command[command.index("--repository") + 1] == str(EXECUTION_ROOT)
        assert command[command.index("--deepspeed-config") + 1] == str(plan.deepspeed_config)
        assert all(path in command for path in inputs)
    local = _local_repository_module_command("ownership-show", ("--path", "/unused"))
    assert local[:3] == ("/usr/bin/env", "-C", str(EXECUTION_ROOT))
    docker, remote = remote_argv(
        _remote_repository_module_command("ownership-show", ("--path", "/unused"))
    )
    assert docker[docker.index("--workdir") + 1] == str(EXECUTION_ROOT)
    assert list(local[3:]) == remote[1:]


def test_artifact_and_transfer_paths_stay_checkout_owned_and_use_physical_checkpoint_storage():
    paths = resolve_run_artifact_paths(RUN_ID)
    assert paths.checkpoint_dir == EXECUTION_ROOT / "checkpoints" / RUN_ID
    assert paths.result_dir == EXECUTION_ROOT / "results" / RUN_ID
    assert paths.log_dir == EXECUTION_ROOT / "logs/zero-stage-compatibility" / RUN_ID
    logical = paths.checkpoint_dir / "checkpoint-2"
    assert (EXECUTION_ROOT / "checkpoints").is_symlink()
    assert (EXECUTION_ROOT / "checkpoints").resolve(strict=True) == (
        ORIGINAL_ROOT / "checkpoints"
    ).resolve(strict=True)
    assert resolve_local_checkpoint_destination(logical) == (
        ORIGINAL_ROOT / "checkpoints"
    ).resolve(strict=True) / RUN_ID / "checkpoint-2"
    assert require_canonical_artifact_path(logical, kind="checkpoint") == logical
    manifest = NodeArtifactManifest(
        schema_version=1, run_id=RUN_ID, node_rank=1,
        process_ranks=(2, 3), entries=(), runtime_identity={},
    )
    transfer = build_push_command(manifest)
    assert transfer[-2:] == (f"{logical}/", f"korea@10.1.102.41:{logical}/")
    assert transfer[transfer.index("--files-from") + 1] == str(
        EXECUTION_ROOT / f"logs/multinode/{RUN_ID}/artifact-push-rank1.files"
    )
    assert "--ignore-existing" in transfer
    result_transfer = build_evaluation_pull_command(
        paths.result_dir, files_from=EXECUTION_ROOT / "logs/evaluation.files"
    )
    assert result_transfer[-2:] == (
        f"korea@10.1.102.41:{paths.result_dir}/", f"{paths.result_dir}/"
    )


@pytest.mark.parametrize("foreign", (ORIGINAL_ROOT, EXECUTION_ROOT.parent / "foreign"))
def test_foreign_repositories_are_rejected_before_remote_execution_or_artifact_transfer(foreign):
    with pytest.raises(ValueError, match="working directory must be canonical"):
        build_remote_exec(("true",), working_directory=foreign)
    with pytest.raises(ValueError, match="outside the canonical root"):
        resolve_local_checkpoint_destination(foreign / f"checkpoints/{RUN_ID}/checkpoint-2")
    with pytest.raises(ValueError, match="canonical result root"):
        build_evaluation_pull_command(
            foreign / "results" / RUN_ID,
            files_from=EXECUTION_ROOT / "logs/evaluation.files",
        )


@pytest.mark.parametrize("checkpoint", (
    EXECUTION_ROOT / f"checkpoints/foreign-run/checkpoint-2",
    EXECUTION_ROOT / f"checkpoints/{RUN_ID}/../foreign-run/checkpoint-2",
))
def test_resume_keeps_same_run_and_traversal_guards(checkpoint):
    with pytest.raises(ValueError, match="same run ID|without|absolute canonical"):
        validate_resume_path(RUN_ID, checkpoint)


def test_checkout_rebinding_does_not_authorize_a_foreign_process_identity():
    spec = build_owned_tmux_commands(launch_plan(), commit="a" * 40).node_specs[0]
    ownership = OwnershipRecord.create(
        run_id=RUN_ID, attempt_id="initial", machine_rank=0,
        pid=4312, process_start_ticks=9981, commit="a" * 40,
        command=spec.command, environment=dict(spec.environment),
    )
    with pytest.raises(ValueError, match="PID"):
        validate_ownership(
            ownership, expected_run_id=RUN_ID, expected_attempt_id="initial",
            expected_machine_rank=0, expected_pid=4313,
            expected_commit="a" * 40,
            expected_command_digest=ownership.command_digest,
            observed_start_ticks=9981,
        )


def test_capacity_witness_identity_binds_to_execution_checkout():
    producer = EXECUTION_ROOT / f"results/{RUN_ID}/initial-producer-evidence.json"
    payload = {
        "schema_version": 1, "run_id": RUN_ID, "attempt_id": "initial",
        "stage": "stage1", "method": "dora", "zero_stage": 2,
        "batch_per_gpu": 1, "world_size": 4, "machine_ranks": [0, 1],
        "exit_codes": {"0": 0, "1": 0},
        "checkpoint_path": str(EXECUTION_ROOT / "checkpoints" / RUN_ID),
        "finite_loss_evidence": {
            "status": "validated", "path": str(producer),
            "sha256": "a" * 64, "size": 1, "finite_loss": True,
        },
    }
    def decode(value):
        return _decode_training_witness(
            json.dumps(value).encode(), child_run_id=RUN_ID, zero_stage=2,
            tuning_method="dora", batch_per_gpu=1, producer_path=producer,
        )
    assert decode(payload) == payload
    with pytest.raises(ValueError, match="text identity"):
        decode({**payload, "checkpoint_path": str(ORIGINAL_ROOT / "checkpoints" / RUN_ID)})


def test_actual_stage1_coordinator_dry_run_binds_fixed_checkout_without_writing_run_outputs():
    command = (
        "/usr/bin/bash", str(REPOSITORY / "run_multinode_stage1_training.sh"),
        "--run-id", RUN_ID, "--zero-stage", "2",
        "--main-process-port", "29641", "--dry-run", *INPUT_ARGUMENTS,
    )
    generated = tuple(EXECUTION_ROOT / kind / RUN_ID for kind in ("checkpoints", "results"))
    before = tuple(path.exists() for path in generated)
    completed = subprocess.run(
        command, cwd=REPOSITORY, env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["plan"]["cluster"]["remote_repo_root"] == str(EXECUTION_ROOT)
    assert payload["plan"]["cluster"]["remote_host_repo_root"] == str(HOST_ROOT)
    assert payload["plan"]["world_size"] == 4
    docker, remote = remote_argv(payload["remote"])
    assert docker[docker.index("--workdir") + 1] == str(EXECUTION_ROOT)
    for command in (remote, payload["local"]):
        assert str(EXECUTION_ROOT / "run_stage1_training.sh") in command
        assert str(EXECUTION_ROOT / "venv/ilvr-venv/bin/python") in command
        assert command[command.index("--output-dir") + 1] == str(generated[0])
        assert command[command.index("--deepspeed-config") + 1] == str(EXECUTION_ROOT / "configs/deepspeed_zero2.json")
        assert all(argument in command for argument in INPUT_ARGUMENTS)
    assert tuple(path.exists() for path in generated) == before


@pytest.mark.parametrize("subcommand", ("train-stage1", "capacity-stage1"))
def test_coordinator_foreign_result_rejection_reports_execution_root(subcommand):
    arguments = (
        ("--method", "dora", "--main-process-port", "29641")
        if subcommand == "train-stage1"
        else (
            "--tuning-method", "dora",
            "--main-process-port", "29641", "--main-process-port", "29642",
            "--main-process-port", "29643", "--main-process-port", "29644",
        )
    )
    completed = subprocess.run(
        (
            str(REPOSITORY / "venv/ilvr-venv/bin/python"), "-m", "src.multinode",
            subcommand, "--run-id", RUN_ID, "--zero-stage", "2", "--dry-run",
            "--result-root", str(ORIGINAL_ROOT / "results"), *arguments,
        ),
        cwd=REPOSITORY, env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert completed.returncode == 2
    assert f"result_root must be inside {EXECUTION_ROOT / 'results'}" in completed.stderr


def test_node_info_foreign_repository_rejection_reports_execution_root():
    with pytest.raises(ValueError) as rejected:
        _node_info(Namespace(repository=str(ORIGINAL_ROOT)))
    assert str(rejected.value) == f"node-info repository must be {EXECUTION_ROOT}"


def test_lifecycle_evidence_and_final_validator_accept_the_planned_checkout():
    lifecycle = build_lifecycle_plans(
        parent_run_id=RUN_ID, first_port=29641,
        result_root=EXECUTION_ROOT / "results",
        python_executable=str(EXECUTION_ROOT / "venv/ilvr-venv/bin/python"),
        commit="a" * 40,
    )[0]
    payload = build_lifecycle_phase_evidence(
        phase="initial", stage="stage1", method="full", zero_stage=1,
        config_digest="a" * 64, experiment_id=lifecycle.child_run_id,
        checkpoint_path=lifecycle.checkpoint_path, artifact_kind="full",
        rank_records=[
            {
                "rank": rank, "global_step": 2, "optimizer_step": 2,
                "loss": 0.5, "gradient_norm": 1.0, "worker_error": None,
            }
            for rank in range(4)
        ],
    )
    raw = json.dumps(payload).encode()
    arguments = dict(
        phase="initial", stage="stage1", method="full", zero_stage=1,
        child_run_id=lifecycle.child_run_id, config_digest="a" * 64,
    )
    assert results_module._validate_raw_matrix_witness(
        raw, checkpoint_path=lifecycle.checkpoint_path, **arguments
    ) == payload
    assert results_module._CANONICAL_RESULT_ROOT / lifecycle.child_run_id == (
        lifecycle.initial_witness_path.parent
    )
    with pytest.raises(ValueError, match="checkpoint is not canonical"):
        results_module._validate_raw_matrix_witness(
            raw,
            checkpoint_path=ORIGINAL_ROOT / "checkpoints" / lifecycle.child_run_id / "checkpoint-2",
            **arguments,
        )
