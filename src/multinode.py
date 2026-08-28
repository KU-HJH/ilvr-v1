"""Pure fixed topology and launch planning for ILVR multi-node runs."""

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Literal

from src.deepspeed_runtime import (
    canonical_deepspeed_config_path,
    load_deepspeed_runtime_profile,
)
from src.multinode_runtime import (
    NodeSnapshot,
    build_preflight_commands,
    collect_node_snapshot,
    validate_node_config_digest,
    validate_preflight,
)


_REPOSITORY_ROOT = Path("/root/workspace/ILVR")
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,127}$")
_TRAINING_REQUIRED_FREE_BYTES = 100 * 1024**3


@dataclass(frozen=True)
class ClusterSpec:
    """The approved, non-discoverable two-node ILVR topology."""

    remote_host: str
    remote_container: str
    remote_repo_root: Path
    remote_host_repo_root: Path
    master_addr: str
    socket_ifname: str
    remote_rank: int = 0
    local_rank: int = 1
    num_machines: int = 2
    gpus_per_node: int = 2
    num_processes: int = 4

    @classmethod
    def default(cls) -> "ClusterSpec":
        return cls(
            remote_host="korea@10.1.102.41",
            remote_container="llm_kd",
            remote_repo_root=_REPOSITORY_ROOT,
            remote_host_repo_root=Path("/home/korea/Workspace/hjh/workspace/ILVR"),
            master_addr="10.1.102.41",
            socket_ifname="bond1.102",
        )


@dataclass(frozen=True)
class LaunchPlan:
    """The immutable coordinator-controlled identity of one training attempt."""

    stage: Literal["stage1", "stage2"]
    tuning_method: Literal["full", "lora", "dora"]
    zero_stage: Literal[1, 2, 3]
    run_id: str
    main_process_port: int
    world_size: int
    deepspeed_config: Path
    deepspeed_config_digest: str
    cluster: ClusterSpec
    attempt_id: str = "initial"
    required_free_bytes: int = _TRAINING_REQUIRED_FREE_BYTES

    def __post_init__(self) -> None:
        if self.stage not in ("stage1", "stage2"):
            raise ValueError("stage must be 'stage1' or 'stage2'")
        if self.tuning_method not in ("full", "lora", "dora"):
            raise ValueError("tuning_method must be 'full', 'lora', or 'dora'")
        if type(self.zero_stage) is not int or self.zero_stage not in (1, 2, 3):
            raise ValueError("zero_stage must be 1, 2, or 3")
        _validate_identifier("run_id", self.run_id)
        _validate_identifier("attempt_id", self.attempt_id)
        if (
            type(self.main_process_port) is not int
            or not 1024 <= self.main_process_port <= 65535
        ):
            raise ValueError("main_process_port must be between 1024 and 65535")
        if self.cluster != ClusterSpec.default():
            raise ValueError("cluster must match the fixed ILVR topology")
        if self.world_size != self.cluster.num_processes:
            raise ValueError("world_size must equal the fixed four-process topology")
        if self.required_free_bytes != _TRAINING_REQUIRED_FREE_BYTES:
            raise ValueError("required_free_bytes must be 100 GiB for training")
        expected_config = canonical_deepspeed_config(self.zero_stage)
        if self.deepspeed_config != expected_config:
            raise ValueError("deepspeed_config must be the canonical ZeRO config")
        if (
            not isinstance(self.deepspeed_config_digest, str)
            or len(self.deepspeed_config_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.deepspeed_config_digest
            )
        ):
            raise ValueError("deepspeed_config_digest must be a SHA-256 digest")


def _validate_identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not _RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must match {_RUN_ID_PATTERN.pattern}")


def canonical_deepspeed_config(zero_stage: int) -> Path:
    """Return the lexical canonical config path for one supported ZeRO stage."""
    if type(zero_stage) is not int or zero_stage not in (1, 2, 3):
        raise ValueError("zero_stage must be 1, 2, or 3")
    approved_path = canonical_deepspeed_config_path(zero_stage)
    expected_name = f"deepspeed_zero{zero_stage}.json"
    if approved_path.name != expected_name:
        raise ValueError("canonical DeepSpeed config selection is inconsistent")
    return _REPOSITORY_ROOT / "configs" / expected_name


def build_launch_plan(
    *,
    stage: Literal["stage1", "stage2"],
    tuning_method: Literal["full", "lora", "dora"],
    zero_stage: Literal[1, 2, 3],
    run_id: str,
    main_process_port: int,
    attempt_id: str = "initial",
) -> LaunchPlan:
    """Validate and build the fixed four-process training launch plan."""
    cluster = ClusterSpec.default()
    config_path = canonical_deepspeed_config(zero_stage)
    profile = load_deepspeed_runtime_profile(
        canonical_deepspeed_config_path(zero_stage), world_size=cluster.num_processes
    )
    if profile.zero_stage != zero_stage:
        raise ValueError(
            "zero_optimization.stage must match the requested zero_stage"
        )
    plan = LaunchPlan(
        stage=stage,
        tuning_method=tuning_method,
        zero_stage=zero_stage,
        run_id=run_id,
        main_process_port=main_process_port,
        world_size=cluster.num_processes,
        deepspeed_config=config_path,
        deepspeed_config_digest=profile.config_digest,
        cluster=cluster,
        attempt_id=attempt_id,
    )
    return plan


def validate_node_local_deepspeed_config(
    plan: LaunchPlan, node_config_path: str | Path
) -> None:
    """Require a node's canonical config to match the validated launch plan."""
    if not isinstance(plan, LaunchPlan):
        raise TypeError("plan must be a LaunchPlan")
    profile = load_deepspeed_runtime_profile(
        node_config_path, world_size=plan.world_size
    )
    if profile.zero_stage != plan.zero_stage:
        raise ValueError("node-local DeepSpeed zero_optimization.stage mismatch")
    if profile.config_digest != plan.deepspeed_config_digest:
        raise ValueError("node-local DeepSpeed config digest mismatch")


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument(
        "--tuning-method", choices=("full", "lora", "dora"), required=True
    )
    parser.add_argument("--zero-stage", choices=(1, 2, 3), type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", default="initial")
    parser.add_argument("--main-process-port", type=int, required=True)


def _plan_from_args(args: argparse.Namespace) -> LaunchPlan:
    return build_launch_plan(
        stage=args.stage,
        tuning_method=args.tuning_method,
        zero_stage=args.zero_stage,
        run_id=args.run_id,
        main_process_port=args.main_process_port,
        attempt_id=args.attempt_id,
    )


def _node_info(args: argparse.Namespace) -> NodeSnapshot:
    """Collect node-local evidence after validating the exact config bytes."""
    repository = Path(args.repository)
    if repository != _REPOSITORY_ROOT:
        raise ValueError("node-info repository must be /root/workspace/ILVR")
    if Path(args.deepspeed_config) != canonical_deepspeed_config(args.zero_stage):
        raise ValueError("node-info config must be the canonical DeepSpeed config")
    validate_node_config_digest(
        args.deepspeed_config,
        zero_stage=args.zero_stage,
        expected_digest=args.deepspeed_config_digest,
    )

    commit = _command_stdout(("git", "rev-parse", "HEAD"), cwd=repository)
    tracked_clean = not _command_stdout(
        ("git", "status", "--porcelain", "--untracked-files=no"), cwd=repository
    )
    interface = _interface_name()
    input_paths = tuple(Path(path) for path in args.input_path)
    artifact_roots = tuple(Path(path) for path in args.artifact_root)
    run_paths = tuple(Path(path) for path in args.run_path)
    free_roots = artifact_roots or (repository,)
    artifact_roots_writable, artifact_free_bytes = _artifact_root_evidence(free_roots)

    # Importing torch and asking CUDA only occurs after the policy/digest check above.
    import accelerate
    import deepspeed
    import torch

    return NodeSnapshot(
        node=args.node,
        machine_rank=args.machine_rank,
        commit=commit,
        tracked_clean=tracked_clean,
        repository=repository,
        container_running=None,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        torch_version=torch.__version__,
        accelerate_version=accelerate.__version__,
        deepspeed_version=deepspeed.__version__,
        torchrun_ok=shutil.which("torchrun") is not None,
        gpu_names=tuple(
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ),
        interface=interface,
        input_paths_ok=all(path.exists() for path in input_paths),
        artifact_roots_writable=artifact_roots_writable,
        artifact_free_bytes=artifact_free_bytes,
        run_paths_fresh=all(not path.exists() for path in run_paths),
        tmux_sessions_fresh=all(_tmux_session_is_fresh(name) for name in args.tmux_session),
        master_port_free=(
            _remote_master_port_is_free(args.main_process_port)
            if (args.node, args.machine_rank) == ("remote", 0)
            else True
        ),
    )


def os_access_writable(path: Path) -> bool:
    """Keep the write check explicit and side-effect free for existing roots."""
    import os

    return os.access(path, os.W_OK)


def _artifact_root_evidence(roots: tuple[Path, ...]) -> tuple[bool, int]:
    """Inspect existing artifact roots only; preflight never creates them."""
    if not all(root.is_dir() and root.exists() and os_access_writable(root) for root in roots):
        return False, 0
    return True, min(shutil.disk_usage(root).free for root in roots)


def _command_stdout(command: tuple[str, ...], *, cwd: Path) -> str:
    completed = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}")
    return completed.stdout.strip()


def _interface_name() -> str:
    completed = subprocess.run(
        ("ip", "-o", "link", "show", "bond1.102"),
        capture_output=True,
        text=True,
        check=False,
    )
    return "bond1.102" if completed.returncode == 0 else ""


def _tmux_session_is_fresh(name: str) -> bool:
    completed = subprocess.run(
        ("tmux", "has-session", "-t", name),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode != 0


def _remote_master_port_is_free(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((ClusterSpec.default().master_addr, port))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    _add_plan_arguments(plan_parser)
    preflight_parser = subparsers.add_parser("preflight")
    _add_plan_arguments(preflight_parser)
    preflight_parser.add_argument("--input-path", action="append", default=[])
    preflight_parser.add_argument("--artifact-root", action="append", default=[])
    preflight_parser.add_argument("--dry-run", action="store_true")
    node_info_parser = subparsers.add_parser("node-info")
    node_info_parser.add_argument("--node", choices=("remote", "local"), required=True)
    node_info_parser.add_argument("--machine-rank", type=int, required=True)
    node_info_parser.add_argument("--repository", required=True)
    node_info_parser.add_argument("--main-process-port", type=int, required=True)
    node_info_parser.add_argument("--deepspeed-config", required=True)
    node_info_parser.add_argument("--deepspeed-config-digest", required=True)
    node_info_parser.add_argument("--zero-stage", choices=(1, 2, 3), type=int, required=True)
    node_info_parser.add_argument("--input-path", action="append", default=[])
    node_info_parser.add_argument("--artifact-root", action="append", default=[])
    node_info_parser.add_argument("--run-path", action="append", default=[])
    node_info_parser.add_argument("--tmux-session", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if args.command == "node-info":
            print(json.dumps(_node_info(args).to_json(), sort_keys=True))
            return 0
        plan = _plan_from_args(args)
        if args.command == "preflight":
            if args.artifact_root:
                artifact_roots = tuple(Path(path) for path in args.artifact_root)
            else:
                artifact_roots = (
                    plan.cluster.remote_repo_root / "checkpoints",
                    plan.cluster.remote_repo_root / "results",
                    plan.cluster.remote_repo_root / "logs",
                )
            remote_run_paths = tuple(
                root / plan.run_id / plan.attempt_id for root in artifact_roots
            )
            local_run_paths = tuple(
                root / plan.run_id / plan.attempt_id for root in artifact_roots
            )
            sessions = (
                f"ilvr-{plan.run_id}-{plan.attempt_id}-rank0",
                f"ilvr-{plan.run_id}-{plan.attempt_id}-rank1",
            )
            commands = build_preflight_commands(
                plan,
                input_paths=args.input_path,
                artifact_roots=artifact_roots,
                remote_run_paths=remote_run_paths,
                local_run_paths=local_run_paths,
                remote_tmux_sessions=(sessions[0],),
                local_tmux_sessions=(sessions[1],),
            )
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "container_inspect": commands.container_inspect,
                            "local_node_info": commands.local_node_info,
                            "plan": _json_ready(plan),
                            "remote_node_info": commands.remote_node_info,
                        },
                        sort_keys=True,
                    )
                )
                return 0

            def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
                return subprocess.run(command, capture_output=True, text=True, check=False)

            remote = collect_node_snapshot(
                plan,
                node="remote",
                input_paths=args.input_path,
                artifact_roots=artifact_roots,
                run_paths=remote_run_paths,
                tmux_sessions=(sessions[0],),
                runner=runner,
                commands=commands,
                repository=plan.cluster.remote_repo_root,
            )
            local = collect_node_snapshot(
                plan,
                node="local",
                input_paths=args.input_path,
                artifact_roots=artifact_roots,
                run_paths=local_run_paths,
                tmux_sessions=(sessions[1],),
                runner=runner,
                commands=commands,
                repository=plan.cluster.remote_repo_root,
            )
            report = validate_preflight(remote, local)
            print(json.dumps(_json_ready(report), sort_keys=True))
            return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(_json_ready(plan), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
