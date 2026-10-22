"""Transport regressions for the dedicated, isolated RDMA runtime."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from src.multinode import build_launch_plan
from src.multinode_artifacts import (
    NodeArtifactManifest,
    build_evaluation_pull_command,
    build_pull_command,
    build_push_command,
)
from src.multinode_runtime import (
    _build_remote_stdin_exec,
    build_owned_tmux_commands,
    build_remote_exec,
    validated_stage2_environment,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("builder", [build_remote_exec, _build_remote_stdin_exec])
def test_remote_operations_enter_the_rdma_container(builder):
    """A leftover exec boundary would run workers/control in the old container."""
    command = builder(["/usr/bin/true"])
    assert command[-2] == "korea@172.16.11.41"
    docker = shlex.split(command[-1])
    assert docker[docker.index("/usr/bin/bash") - 1] == "llm_kd_rdma"


@pytest.mark.parametrize("direction", ["push", "pull", "evaluation"])
def test_artifact_transfers_enter_the_same_rdma_container(direction, tmp_path):
    """A leftover rsync boundary would collect another container's files."""
    manifest = NodeArtifactManifest(
        schema_version=1,
        run_id="rdma-transport-test",
        node_rank=1,
        process_ranks=(2, 3),
        entries=(),
        runtime_identity={},
    )
    if direction == "push":
        command = build_push_command(manifest, files_from=tmp_path / "files")
    elif direction == "pull":
        command = build_pull_command(manifest, files_from=tmp_path / "files")
    else:
        command = build_evaluation_pull_command(
            ROOT / "results/rdma-transport-test", files_from=tmp_path / "files"
        )
    assert shlex.split(command[command.index("--rsync-path") + 1]) == [
        "/usr/bin/docker", "exec", "-i", "llm_kd_rdma", "/usr/bin/rsync"
    ]
    assert any(part.startswith("korea@172.16.11.41:") for part in command)


@pytest.mark.parametrize("rank", [0, 1])
def test_sanitized_stage2_worker_process_uses_ib_despite_ambient_tcp(rank, monkeypatch):
    """Forgetting the fixed transport after env -i permits silent TCP fallback."""
    monkeypatch.setenv("NCCL_NET", "Socket")
    plan = build_launch_plan(
        stage="stage2", tuning_method="dora", zero_stage=2,
        run_id="rdma-transport-test", main_process_port=39761,
    )
    environment = validated_stage2_environment(
        {"STAGE1_CHECKPOINT": "/models/base", "DATA_PATH": "/data/train.jsonl",
         "IMAGE_ROOT": "/data/images", "TUNING_METHOD": "dora", "NCCL_NET": "Socket"},
        output_dir=str(ROOT / "checkpoints/rdma-transport-test"),
        log_file=str(ROOT / "logs/rdma-transport-test/train.log"),
        deepspeed_config=str(ROOT / "configs/deepspeed_zero2.json"),
    )
    commands = build_owned_tmux_commands(plan, commit="a" * 40, stage_environment=environment)
    spec = commands.node_specs[rank]
    probe = "import json,os; print(json.dumps(dict(os.environ)))"
    result = subprocess.run(
        ["/usr/bin/env", "-i", *(f"{k}={v}" for k, v in spec.environment),
         sys.executable, "-c", probe], capture_output=True, text=True, check=True,
    )
    effective = json.loads(result.stdout)
    for key, value in {
        "NCCL_NET": "IB", "NCCL_IB_DISABLE": "0", "NCCL_IB_HCA": "mlx5_0:1",
        "NCCL_SOCKET_IFNAME": "bond0.11", "GLOO_SOCKET_IFNAME": "bond0.11",
        "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "ILVR_MAIN_PROCESS_IP": "172.16.11.41", "ILVR_NUM_PROCESSES": "4",
        "ILVR_MACHINE_RANK": str(rank),
    }.items():
        assert effective.get(key) == value, key
