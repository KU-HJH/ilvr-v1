"""Consumer contracts for the preserved Stage 1 RDMA operator checkout."""

import json
from pathlib import Path
import shlex
import subprocess

import pytest

from src.multinode import build_launch_plan
from src.multinode_runtime import (
    _remote_repository_module_stdin_command,
    build_owned_tmux_commands,
    build_preflight_commands,
)

ROOT = "/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022"
CHECKOUT = Path(__file__).resolve().parents[2]


def stage1_commands():
    plan = build_launch_plan(
        stage="stage1", tuning_method="dora", zero_stage=1,
        run_id="stage1-baseline-10ep-ib-contract", main_process_port=29671,
    )
    return plan, build_owned_tmux_commands(
        plan, commit="c21b4e5c311204df4281317219b3f4a6b95e4d3b",
        stage_arguments=(
            "--method", "dora", "--model", "/models/original-qwen",
            "--data-path", "/data/train.jsonl", "--image-root", "/data/images",
            "--micro-batch", "1", "--gradient-accumulation", "2",
            "--epochs", "10", "--max-steps", "4320", "--ema-tau", "1",
        ),
    )


@pytest.mark.parametrize("machine_rank", [0, 1])
def test_owned_stage1_worker_consumes_sanitized_ib_environment(machine_rank):
    # Dropping IB from either isolated worker would silently select socket transport.
    _, commands = stage1_commands()
    spec = commands.node_specs[machine_rank]
    environment = dict(spec.environment)
    observed = subprocess.run(
        ["/usr/bin/env", "-i", *[f"{k}={v}" for k, v in environment.items()],
         str(CHECKOUT / "venv/ilvr-venv/bin/python"), "-c",
         "import json, os; print(json.dumps(dict(os.environ)))"],
        text=True, capture_output=True, check=True,
    )
    actual = json.loads(observed.stdout)
    assert {k: actual.get(k) for k in (
        "NCCL_NET", "NCCL_IB_DISABLE", "NCCL_IB_HCA", "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS",
    )} == {
        "NCCL_NET": "IB", "NCCL_IB_DISABLE": "0", "NCCL_IB_HCA": "mlx5_0:1",
        "NCCL_SOCKET_IFNAME": "bond0.11", "GLOO_SOCKET_IFNAME": "bond0.11",
        "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET",
    }
    assert actual["ILVR_MACHINE_RANK"] == str(machine_rank)
    assert actual["PATH"].split(":")[0] == ROOT + "/venv/ilvr-venv/bin"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in actual


@pytest.mark.parametrize("machine_rank", [0, 1])
def test_resolved_stage1_shell_dry_run_preserves_historical_training(machine_rank):
    # Catch a stale root/network or console shebang before Accelerate consumes it.
    plan, commands = stage1_commands()
    spec = commands.node_specs[machine_rank]
    stage_command = spec.command[spec.command.index("--") + 1:]
    assert stage_command[:2] == ("/usr/bin/bash", ROOT + "/run_stage1_training.sh")
    assert plan.cluster.remote_host_repo_root == Path(
        "/home/korea/Workspace/hjh/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022"
    )
    completed = subprocess.run(
        ["/usr/bin/bash", str(CHECKOUT / "run_stage1_training.sh"),
         *stage_command[2:], "--dry-run"], env=dict(spec.environment),
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout)
    assert command[:3] == [str(CHECKOUT / "venv/ilvr-venv/bin/python"),
                           str(CHECKOUT / "venv/ilvr-venv/bin/accelerate"), "launch"]
    for flag, want in {
        "--num_machines": "2", "--num_processes": "4",
        "--machine_rank": str(machine_rank), "--main_process_ip": "172.16.11.41",
        "--deepspeed_config_file": ROOT + "/configs/deepspeed_zero1.json",
        "--save_model_path": ROOT + "/checkpoints/stage1-baseline-10ep-ib-contract",
        "--per_device_train_batch_size": "1", "--gradient_accumulation_steps": "2",
        "--epochs": "10", "--max_steps": "4320", "--ema_tau": "1",
        "--lora_r": "32", "--lora_alpha": "64", "--lora_dropout": "0.05",
        "--lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    }.items():
        assert command[command.index(flag) + 1] == want


def test_remote_operator_boundaries_use_same_rdma_container():
    # Catch a split launch/control/transfer identity before any SSH is attempted.
    plan, commands = stage1_commands()
    preflight = build_preflight_commands(
        plan, input_paths=(), artifact_roots=(), remote_run_paths=(),
        local_run_paths=(), remote_tmux_sessions=(), local_tmux_sessions=(),
    )
    stdin = _remote_repository_module_stdin_command("receive-launch-plan")
    for command in (commands.remote, preflight.remote_node_info, stdin):
        assert command[-2] == "korea@172.16.11.41"
        docker = shlex.split(command[-1])
        assert docker[docker.index("/usr/bin/bash") - 1] == "llm_kd_rdma"
    assert shlex.split(preflight.container_inspect[-1])[-1] == "llm_kd_rdma"
