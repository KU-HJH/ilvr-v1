import dataclasses
import json
from pathlib import Path
import subprocess

import pytest

from src.multinode import build_launch_plan
from src.multinode_runtime import (
    NodeSnapshot,
    build_remote_exec,
    collect_node_snapshot,
    validate_preflight,
)


def snapshot(node, rank):
    return NodeSnapshot(
        node=node,
        machine_rank=rank,
        commit="a" * 40,
        tracked_clean=True,
        repository=Path("/root/workspace/ILVR"),
        container_running=True,
        python_version="3.11",
        torch_version="2.7.1+cu128",
        accelerate_version="1.10.1",
        deepspeed_version="0.18.0",
        torchrun_ok=True,
        gpu_names=("NVIDIA H100 NVL", "NVIDIA H100 NVL"),
        interface="bond1.102",
        input_paths_ok=True,
        artifact_roots_writable=True,
        artifact_free_bytes=500_000_000_000,
        run_paths_fresh=True,
        tmux_sessions_fresh=True,
        master_port_free=True,
    )


def test_preflight_accepts_symmetric_clean_nodes():
    report = validate_preflight(snapshot("remote", 0), snapshot("local", 1))

    assert report.world_size == 4
    assert report.commit == "a" * 40


def test_preflight_rejects_commit_mismatch_before_launch():
    remote = snapshot("remote", 0)
    local = dataclasses.replace(snapshot("local", 1), commit="b" * 40)

    with pytest.raises(ValueError, match="same tracked commit"):
        validate_preflight(remote, local)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tracked_clean": False}, "tracked files"),
        ({"container_running": False}, "llm_kd"),
        ({"torch_version": "2.7.0+cu126"}, "environment versions"),
        ({"gpu_names": ("NVIDIA H100 NVL",)}, "exactly two GPUs"),
        ({"gpu_names": ("NVIDIA A100", "NVIDIA A100")}, "H100"),
        ({"interface": "eth0"}, "bond1.102"),
        ({"input_paths_ok": False}, "input paths"),
        ({"artifact_roots_writable": False}, "artifact roots"),
        ({"artifact_free_bytes": 1024}, "free space"),
        ({"run_paths_fresh": False}, "fresh run paths"),
        ({"tmux_sessions_fresh": False}, "tmux session"),
        ({"master_port_free": False}, "master port"),
    ],
)
def test_preflight_rejects_invalid_remote_snapshot(changes, message):
    remote = dataclasses.replace(snapshot("remote", 0), **changes)

    with pytest.raises(ValueError, match=message):
        validate_preflight(remote, snapshot("local", 1))


def test_preflight_does_not_require_a_local_container():
    local = dataclasses.replace(snapshot("local", 1), container_running=None)

    assert validate_preflight(snapshot("remote", 0), local).world_size == 4


def test_preflight_accepts_python_311_build_marker():
    local = dataclasses.replace(snapshot("local", 1), python_version="3.11.15+")

    assert validate_preflight(snapshot("remote", 0), local).world_size == 4


def plan():
    return build_launch_plan(
        stage="stage1",
        tuning_method="lora",
        zero_stage=2,
        run_id="stage1-lora-zero2-four-gpu",
        main_process_port=29613,
    )


def test_remote_exec_shell_quotes_each_argument():
    command = build_remote_exec(("python", "-c", "print('safe value')"))

    assert command == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "korea@10.1.102.41",
        "docker",
        "exec",
        "llm_kd",
        "bash",
        "-lc",
        """python -c 'print('"'"'safe value'"'"')'""",
    ]


def test_remote_collection_checks_container_then_parses_probe_json():
    calls = []

    def runner(command):
        calls.append(list(command))
        if "inspect" in command:
            return subprocess.CompletedProcess(command, 0, "true\n", "")
        return subprocess.CompletedProcess(
            command, 0, json.dumps(snapshot("remote", 0).to_json()), ""
        )

    result = collect_node_snapshot(
        plan(),
        node="remote",
        input_paths=("/input",),
        artifact_roots=("/artifacts",),
        run_paths=("/fresh",),
        tmux_sessions=("ilvr-run-rank0",),
        runner=runner,
    )

    assert result.container_running is True
    assert calls[0][-5:] == [
        "docker",
        "inspect",
        "--format",
        "{{.State.Running}}",
        "llm_kd",
    ]
    assert calls[1][:8] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "korea@10.1.102.41",
        "docker",
        "exec",
    ]


def test_local_collection_never_uses_ssh_or_docker():
    calls = []

    def runner(command):
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command, 0, json.dumps(snapshot("local", 1).to_json()), ""
        )

    result = collect_node_snapshot(
        plan(),
        node="local",
        input_paths=(),
        artifact_roots=(),
        run_paths=(),
        tmux_sessions=(),
        runner=runner,
        repository="/node/repository",
    )

    assert result.container_running is None
    assert len(calls) == 1
    assert calls[0][0] == "/node/repository/venv/ilvr-venv/bin/python"
    assert "ssh" not in calls[0]
    assert "docker" not in calls[0]


def test_preflight_dry_run_prints_commands_without_probing(monkeypatch, capsys):
    import src.multinode as multinode

    def forbidden_probe(*args, **kwargs):
        raise AssertionError("dry run must not collect node info")

    monkeypatch.setattr(multinode, "collect_node_snapshot", forbidden_probe)

    assert multinode._main(
        [
            "preflight",
            "--stage",
            "stage1",
            "--tuning-method",
            "lora",
            "--zero-stage",
            "2",
            "--run-id",
            "stage1-lora-zero2-four-gpu",
            "--main-process-port",
            "29613",
            "--dry-run",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["world_size"] == 4
    assert payload["local_command"][:3] == ["node-info", "--node", "local"]
    assert payload["remote_command"][:2] == ["ssh", "-o"]


def test_preflight_uses_each_node_repository_artifact_roots(monkeypatch, capsys):
    import src.multinode as multinode

    calls = []

    def collect(_plan, **kwargs):
        calls.append(kwargs)
        return snapshot(kwargs["node"], 0 if kwargs["node"] == "remote" else 1)

    monkeypatch.setattr(multinode, "collect_node_snapshot", collect)

    assert multinode._main(
        [
            "preflight",
            "--stage",
            "stage1",
            "--tuning-method",
            "lora",
            "--zero-stage",
            "2",
            "--run-id",
            "stage1-lora-zero2-four-gpu",
            "--main-process-port",
            "29613",
        ]
    ) == 0

    capsys.readouterr()
    assert calls[0]["artifact_roots"] == (
        Path("/root/workspace/ILVR/checkpoints"),
        Path("/root/workspace/ILVR/results"),
        Path("/root/workspace/ILVR/logs"),
    )
    assert calls[1]["artifact_roots"] == (
        Path.cwd() / "checkpoints",
        Path.cwd() / "results",
        Path.cwd() / "logs",
    )


def test_missing_artifact_root_is_reported_without_creating_it(tmp_path):
    from src.multinode import _artifact_root_evidence

    missing = tmp_path / "not-created"

    assert _artifact_root_evidence((missing,)) == (False, 0)
    assert not missing.exists()
