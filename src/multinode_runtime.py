"""Pure validation and command construction for the fixed ILVR cluster."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from src.deepspeed_runtime import load_deepspeed_runtime_profile

if TYPE_CHECKING:
    from src.multinode import LaunchPlan


_REMOTE_HOST = "korea@10.1.102.41"
_REMOTE_CONTAINER = "llm_kd"
_REQUIRED_FREE_BYTES = 100 * 1024**3


@dataclass(frozen=True)
class NodeSnapshot:
    """Small, serializable evidence gathered before a node starts CUDA work."""

    node: str
    machine_rank: int
    commit: str
    tracked_clean: bool
    repository: Path
    container_running: bool | None
    python_version: str
    torch_version: str
    accelerate_version: str
    deepspeed_version: str
    torchrun_ok: bool
    gpu_names: tuple[str, ...]
    interface: str
    input_paths_ok: bool
    artifact_roots_writable: bool
    artifact_free_bytes: int
    run_paths_fresh: bool
    tmux_sessions_fresh: bool
    master_port_free: bool

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "NodeSnapshot":
        """Decode the JSON emitted by ``python -m src.multinode node-info``."""
        values = dict(payload)
        values["repository"] = Path(str(values["repository"]))
        values["gpu_names"] = tuple(str(name) for name in values["gpu_names"])
        return cls(**values)  # type: ignore[arg-type]

    def to_json(self) -> dict[str, object]:
        values = asdict(self)
        values["repository"] = str(self.repository)
        values["gpu_names"] = list(self.gpu_names)
        return values


@dataclass(frozen=True)
class PreflightReport:
    commit: str
    world_size: int


@dataclass(frozen=True)
class PreflightCommands:
    """The three external commands a live preflight is permitted to run."""

    container_inspect: tuple[str, ...]
    remote_node_info: tuple[str, ...]
    local_node_info: tuple[str, ...]


def build_remote_exec(argv: Sequence[str]) -> list[str]:
    """Build the fixed, noninteractive remote container command safely."""
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        _REMOTE_HOST,
        "docker",
        "exec",
        _REMOTE_CONTAINER,
        "bash",
        "-lc",
        shlex.join(argv),
    ]


def validate_preflight(remote: NodeSnapshot, local: NodeSnapshot) -> PreflightReport:
    """Reject all launch blockers before either machine initializes CUDA."""
    if (remote.node, remote.machine_rank) != ("remote", 0):
        raise ValueError("remote snapshot must be machine rank 0")
    if (local.node, local.machine_rank) != ("local", 1):
        raise ValueError("local snapshot must be machine rank 1")
    if remote.commit != local.commit:
        raise ValueError("nodes must have the same tracked commit")

    for node in (remote, local):
        _validate_snapshot(node)
    if remote.container_running is not True:
        raise ValueError("remote llm_kd container is not running")
    return PreflightReport(commit=remote.commit, world_size=4)


def _validate_snapshot(node: NodeSnapshot) -> None:
    if not node.tracked_clean:
        raise ValueError(f"{node.node}: tracked files are dirty")
    if (
        not _is_python_311(node.python_version)
        or node.torch_version != "2.7.1+cu128"
        or node.accelerate_version != "1.10.1"
        or node.deepspeed_version != "0.18.0"
        or not node.torchrun_ok
    ):
        raise ValueError(f"{node.node}: environment versions or torchrun mismatch")
    if len(node.gpu_names) != 2:
        raise ValueError(f"{node.node}: exactly two GPUs are required")
    if any("H100" not in name for name in node.gpu_names):
        raise ValueError(f"{node.node}: H100 GPUs are required")
    if node.interface != "bond1.102":
        raise ValueError(f"{node.node}: bond1.102 interface is required")
    if not node.input_paths_ok:
        raise ValueError(f"{node.node}: input paths are missing")
    if not node.artifact_roots_writable:
        raise ValueError(f"{node.node}: artifact roots are not writable")
    if node.artifact_free_bytes < _REQUIRED_FREE_BYTES:
        raise ValueError(f"{node.node}: insufficient free space")
    if not node.run_paths_fresh:
        raise ValueError(f"{node.node}: fresh run paths are required")
    if not node.tmux_sessions_fresh:
        raise ValueError(f"{node.node}: tmux session already exists")
    if not node.master_port_free:
        raise ValueError(f"{node.node}: master port is unavailable")


def _is_python_311(version: str) -> bool:
    """Accept the interpreter's build marker while enforcing its major/minor ABI."""
    fields = version.split(".", 2)
    return len(fields) >= 2 and fields[:2] == ["3", "11"]


def validate_node_config_digest(
    config_path: str | Path, *, zero_stage: int, expected_digest: str
) -> None:
    """Validate a node's exact canonical config bytes before CUDA inspection."""
    profile = load_deepspeed_runtime_profile(config_path, world_size=4)
    if profile.zero_stage != zero_stage:
        raise ValueError("node-local DeepSpeed zero_optimization.stage mismatch")
    if profile.config_digest != expected_digest:
        raise ValueError("node-local DeepSpeed config digest mismatch")


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def build_preflight_commands(
    plan: "LaunchPlan",
    *,
    input_paths: Sequence[str | Path],
    artifact_roots: Sequence[str | Path],
    remote_run_paths: Sequence[str | Path],
    local_run_paths: Sequence[str | Path],
    remote_tmux_sessions: Sequence[str],
    local_tmux_sessions: Sequence[str],
) -> PreflightCommands:
    """Build once the exact arrays shared by dry-run and live preflight."""
    repository = plan.cluster.remote_repo_root
    remote_argv = _node_info_command(
        plan,
        node="remote",
        machine_rank=plan.cluster.remote_rank,
        repository=repository,
        input_paths=input_paths,
        artifact_roots=artifact_roots,
        run_paths=remote_run_paths,
        tmux_sessions=remote_tmux_sessions,
    )
    local_argv = _node_info_command(
        plan,
        node="local",
        machine_rank=plan.cluster.local_rank,
        repository=repository,
        input_paths=input_paths,
        artifact_roots=artifact_roots,
        run_paths=local_run_paths,
        tmux_sessions=local_tmux_sessions,
    )
    return PreflightCommands(
        container_inspect=(
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            plan.cluster.remote_host,
            "docker",
            "inspect",
            "--format",
            "{{.State.Running}}",
            plan.cluster.remote_container,
        ),
        remote_node_info=tuple(build_remote_exec(remote_argv)),
        local_node_info=tuple(local_argv),
    )


def collect_node_snapshot(
    plan: "LaunchPlan",
    *,
    node: str,
    runner: Runner,
    commands: PreflightCommands | None = None,
    input_paths: Sequence[str | Path] = (),
    artifact_roots: Sequence[str | Path] = (),
    run_paths: Sequence[str | Path] = (),
    tmux_sessions: Sequence[str] = (),
    repository: str | Path | None = None,
) -> NodeSnapshot:
    """Run exactly one node-info probe, using SSH only for the remote node."""
    if node not in {"remote", "local"}:
        raise ValueError("node must be 'remote' or 'local'")
    repository = Path(repository) if repository is not None else plan.cluster.remote_repo_root
    if repository != plan.cluster.remote_repo_root:
        raise ValueError("node-info repository must be the canonical ILVR checkout")
    if commands is None:
        commands = build_preflight_commands(
            plan,
            input_paths=input_paths,
            artifact_roots=artifact_roots,
            remote_run_paths=run_paths,
            local_run_paths=run_paths,
            remote_tmux_sessions=tmux_sessions,
            local_tmux_sessions=tmux_sessions,
        )
    if node == "remote":
        inspect = runner(commands.container_inspect)
        if inspect.returncode != 0:
            raise RuntimeError("remote docker inspect failed")
        completed = runner(commands.remote_node_info)
        container_running: bool | None = inspect.stdout.strip() == "true"
    else:
        completed = runner(commands.local_node_info)
        container_running = None
    if completed.returncode != 0:
        raise RuntimeError(f"{node} node-info failed: {completed.stderr.strip()}")
    try:
        snapshot = NodeSnapshot.from_json(json.loads(completed.stdout))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{node} node-info returned invalid JSON") from error
    return replace(snapshot, container_running=container_running)


def _node_info_command(
    plan: "LaunchPlan",
    *,
    node: str,
    machine_rank: int,
    repository: Path,
    input_paths: Sequence[str | Path],
    artifact_roots: Sequence[str | Path],
    run_paths: Sequence[str | Path],
    tmux_sessions: Sequence[str],
) -> list[str]:
    command = [
        "env",
        f"PYTHONPATH={repository}",
        str(repository / "venv" / "ilvr-venv" / "bin" / "python"),
        "-m",
        "src.multinode",
        "node-info",
        "--node",
        node,
        "--machine-rank",
        str(machine_rank),
        "--repository",
        str(repository),
        "--main-process-port",
        str(plan.main_process_port),
        "--deepspeed-config",
        str(plan.deepspeed_config),
        "--deepspeed-config-digest",
        plan.deepspeed_config_digest,
        "--zero-stage",
        str(plan.zero_stage),
    ]
    for flag, paths in (
        ("--input-path", input_paths),
        ("--artifact-root", artifact_roots),
        ("--run-path", run_paths),
    ):
        for path in paths:
            command.extend((flag, str(path)))
    for session in tmux_sessions:
        command.extend(("--tmux-session", session))
    return command
