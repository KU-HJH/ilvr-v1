import dataclasses
import json
import re
import shlex
from pathlib import Path
import subprocess

import pytest

import src.multinode as multinode
from src.multinode import build_launch_plan
from src.multinode_runtime import (
    NodeSnapshot,
    PreflightCommands,
    build_remote_exec,
    build_preflight_commands,
    collect_node_snapshot,
    validate_preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def snapshot(node, rank):
    return NodeSnapshot(
        node=node,
        machine_rank=rank,
        commit="a" * 40,
        tracked_clean=True,
        repository=REPO_ROOT,
        container_running=True,
        python_version="3.11",
        torch_version="2.7.1+cu128",
        accelerate_version="1.10.1",
        deepspeed_version="0.18.7",
        torchrun_ok=True,
        gpu_names=("NVIDIA H100 NVL", "NVIDIA H100 NVL"),
        interface="bond0.11",
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


@pytest.mark.parametrize("stale_node", ["remote", "local"])
def test_preflight_rejects_deepspeed_0180_on_either_node(stale_node):
    """Catches either node silently retaining the pre-recovery DeepSpeed version."""
    remote = snapshot("remote", 0)
    local = snapshot("local", 1)
    if stale_node == "remote":
        remote = dataclasses.replace(remote, deepspeed_version="0.18.0")
    else:
        local = dataclasses.replace(local, deepspeed_version="0.18.0")

    with pytest.raises(ValueError, match=f"{stale_node}: environment versions"):
        validate_preflight(remote, local)


def test_stage2_peft_zero3_preflight_uses_dedicated_config_on_both_nodes():
    """Catches node-info validating a different config than the GPU workers."""
    launch_plan = build_launch_plan(
        stage="stage2",
        tuning_method="lora",
        zero_stage=3,
        run_id="stage2-lora-zero3-dedicated",
        main_process_port=29633,
    )
    commands = build_preflight_commands(
        launch_plan,
        input_paths=("/input",),
        artifact_roots=("/root/workspace/ILVR/checkpoints",),
        remote_run_paths=("/root/workspace/ILVR/checkpoints/remote",),
        local_run_paths=("/root/workspace/ILVR/checkpoints/local",),
        remote_tmux_sessions=("ilvr-remote",),
        local_tmux_sessions=("ilvr-local",),
    )

    expected = str(REPO_ROOT / "configs/deepspeed_zero3_stage2_peft.json")
    assert commands.local_node_info[
        commands.local_node_info.index("--deepspeed-config") + 1
    ] == expected
    remote = shlex.split(shlex.split(commands.remote_node_info[-1])[-1])
    assert remote[remote.index("--deepspeed-config") + 1] == expected


def test_stage2_peft_zero3_node_info_decodes_selected_policy(monkeypatch):
    """Catches selected-policy fields disappearing at the node-info parser boundary."""
    launch_plan = build_launch_plan(
        stage="stage2",
        tuning_method="lora",
        zero_stage=3,
        run_id="stage2-lora-zero3-node-info",
        main_process_port=29634,
    )
    commands = build_preflight_commands(
        launch_plan,
        input_paths=("/input",),
        artifact_roots=("/root/workspace/ILVR/checkpoints",),
        remote_run_paths=("/root/workspace/ILVR/checkpoints/remote",),
        local_run_paths=("/root/workspace/ILVR/checkpoints/local",),
        remote_tmux_sessions=("ilvr-remote",),
        local_tmux_sessions=("ilvr-local",),
    )
    decoded = {}

    def observe(args):
        decoded.update(vars(args))
        return snapshot("local", 1)

    monkeypatch.setattr(multinode, "_node_info", observe)
    node_argv = list(commands.local_node_info)
    node_argv = node_argv[node_argv.index("node-info") :]

    assert multinode._main(node_argv) == 0
    assert decoded["stage"] == "stage2"
    assert decoded["tuning_method"] == "lora"


def test_canonical_git_query_tolerates_different_repository_owner(monkeypatch):
    """Catches node-info omitting the canonical Git safe-directory argument."""
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")

    commit = multinode._repository_git_stdout(("rev-parse", "HEAD"))

    assert re.fullmatch(r"[0-9a-f]{40}", commit)


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
        ({"interface": "eth0"}, "bond0.11"),
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

    assert command[:6] == [
        "/usr/bin/ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "korea@172.16.11.41",
    ]
    assert len(command) == 7
    remote = shlex.split(command[6])
    assert remote[:3] == ["/usr/bin/docker", "exec", "llm_kd"]
    assert remote[3:5] == ["/usr/bin/bash", "-lc"]
    assert shlex.split(remote[5]) == ["python", "-c", "print('safe value')"]


def test_remote_collection_checks_container_then_parses_probe_json():
    calls = []

    def runner(command):
        calls.append(list(command))
        if any("inspect" in token for token in command):
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
    assert shlex.split(calls[0][-1]) == [
        "/usr/bin/docker",
        "inspect",
        "--format",
        "{{.State.Running}}",
        "llm_kd",
    ]
    assert calls[1][:6] == [
        "/usr/bin/ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "korea@172.16.11.41",
    ]
    assert shlex.split(calls[1][6])[:2] == ["/usr/bin/docker", "exec"]


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
    )

    assert result.container_running is None
    assert len(calls) == 1
    assert calls[0][:3] == [
        "env",
        f"PYTHONPATH={REPO_ROOT}:{REPO_ROOT / 'src'}",
        str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
    ]
    assert "/usr/bin/ssh" not in calls[0]
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
    assert payload["local_node_info"][:6] == [
        "env",
        f"PYTHONPATH={REPO_ROOT}:{REPO_ROOT / 'src'}",
        str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
        "-m",
        "src.multinode",
        "node-info",
    ]
    assert payload["remote_node_info"][:2] == ["/usr/bin/ssh", "-o"]


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
        REPO_ROOT / "checkpoints",
        REPO_ROOT / "results",
        REPO_ROOT / "logs",
    )
    assert calls[1]["artifact_roots"] == (
        REPO_ROOT / "checkpoints",
        REPO_ROOT / "results",
        REPO_ROOT / "logs",
    )


def test_missing_artifact_root_is_reported_without_creating_it(tmp_path):
    from src.multinode import _artifact_root_evidence

    missing = tmp_path / "not-created"

    assert _artifact_root_evidence((missing,)) == (False, 0)
    assert not missing.exists()


def test_preflight_commands_use_the_canonical_repo_venv_and_config():
    commands = build_preflight_commands(
        plan(),
        input_paths=("/input",),
        artifact_roots=("/root/workspace/ILVR/checkpoints",),
        remote_run_paths=("/root/workspace/ILVR/checkpoints/fresh-remote",),
        local_run_paths=("/root/workspace/ILVR/checkpoints/fresh-local",),
        remote_tmux_sessions=("ilvr-remote",),
        local_tmux_sessions=("ilvr-local",),
    )

    expected_prefix = [
        "env",
        f"PYTHONPATH={REPO_ROOT}:{REPO_ROOT / 'src'}",
        str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
        "-m",
        "src.multinode",
        "node-info",
    ]
    assert list(commands.local_node_info[:6]) == expected_prefix
    assert "--repository" in commands.local_node_info
    assert commands.local_node_info[
        commands.local_node_info.index("--repository") + 1
    ] == str(REPO_ROOT)
    assert commands.local_node_info[
        commands.local_node_info.index("--deepspeed-config") + 1
    ] == str(REPO_ROOT / "configs/deepspeed_zero2.json")
    remote_docker = shlex.split(commands.remote_node_info[-1])
    assert remote_docker[:3] == ["/usr/bin/docker", "exec", "llm_kd"]
    assert shlex.split(remote_docker[-1])[:2] == [
        "env",
        f"PYTHONPATH={REPO_ROOT}:{REPO_ROOT / 'src'}",
    ]


def test_live_collection_uses_the_exact_prebuilt_dry_run_commands():
    commands = build_preflight_commands(
        plan(),
        input_paths=("/input",),
        artifact_roots=("/root/workspace/ILVR/checkpoints",),
        remote_run_paths=("/root/workspace/ILVR/checkpoints/fresh-remote",),
        local_run_paths=("/root/workspace/ILVR/checkpoints/fresh-local",),
        remote_tmux_sessions=("ilvr-remote",),
        local_tmux_sessions=("ilvr-local",),
    )
    calls = []

    def runner(command):
        calls.append(tuple(command))
        if tuple(command) == commands.container_inspect:
            return subprocess.CompletedProcess(command, 0, "true\n", "")
        node = "remote" if tuple(command) == commands.remote_node_info else "local"
        return subprocess.CompletedProcess(command, 0, json.dumps(snapshot(node, 0 if node == "remote" else 1).to_json()), "")

    collect_node_snapshot(plan(), node="remote", commands=commands, runner=runner)
    collect_node_snapshot(plan(), node="local", commands=commands, runner=runner)

    assert calls == [
        commands.container_inspect,
        commands.remote_node_info,
        commands.local_node_info,
    ]


def test_dry_run_prints_the_same_three_executable_commands(monkeypatch, capsys):
    import src.multinode as multinode

    def forbidden_probe(*args, **kwargs):
        raise AssertionError("dry run must not collect node info")

    monkeypatch.setattr(multinode, "collect_node_snapshot", forbidden_probe)

    assert multinode._main(
        [
            "preflight", "--stage", "stage1", "--tuning-method", "lora",
            "--zero-stage", "2", "--run-id", "stage1-lora-zero2-four-gpu",
            "--main-process-port", "29613", "--dry-run",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"container_inspect", "local_node_info", "plan", "remote_node_info"}
    assert payload["local_node_info"][:6] == [
        "env", f"PYTHONPATH={REPO_ROOT}:{REPO_ROOT / 'src'}",
        str(REPO_ROOT / "venv/ilvr-venv/bin/python"), "-m", "src.multinode", "node-info",
    ]


def test_preflight_cannot_lower_the_immutable_training_free_space_limit():
    remote = dataclasses.replace(snapshot("remote", 0), artifact_free_bytes=1024)

    with pytest.raises(TypeError, match="required_free_bytes"):
        validate_preflight(remote, snapshot("local", 1), required_free_bytes=1)
