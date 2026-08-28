"""Pure fixed topology and launch planning for ILVR multi-node runs."""

import argparse
import json
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Literal

if __name__ == "__main__":
    # Keep runtime type checks stable when this module is executed with ``-m``.
    sys.modules.setdefault("src.multinode", sys.modules[__name__])

from src.deepspeed_runtime import (
    canonical_deepspeed_config_path,
    load_deepspeed_runtime_profile,
)
from src.multinode_runtime import (
    NodeSnapshot,
    OwnershipRecord,
    SubprocessCoordinatorRunner,
    _nul_fields,
    _proc_start_ticks,
    build_owned_tmux_commands,
    build_preflight_commands,
    collect_checkpoint_artifacts,
    collect_node_snapshot,
    prepare_attempt_directory,
    receive_launch_plan,
    redistribute_resume_checkpoint,
    read_ownership,
    monitor_owned_jobs,
    owned_listener_is_ready,
    start_rank0_then_rank1,
    validate_owned_process,
    validate_ownership,
    validate_node_config_digest,
    validate_preflight,
    validate_stage1_arguments,
    validate_stage2_arguments,
    validate_resume_path,
    validated_stage2_environment,
    write_ownership,
    write_incomplete_attempt,
    write_training_witness,
)


_REPOSITORY_ROOT = Path("/root/workspace/ILVR")
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,127}$")
_TRAINING_REQUIRED_FREE_BYTES = 100 * 1024**3
_ATTEMPT_TIMEOUT_SECONDS = 24 * 60 * 60
_CONTROL_TIMEOUT_SECONDS = 30
_PROC_ROOT = Path("/proc")


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
    attempt_timeout_seconds: int = _ATTEMPT_TIMEOUT_SECONDS

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
        if self.attempt_timeout_seconds != _ATTEMPT_TIMEOUT_SECONDS:
            raise ValueError("attempt_timeout_seconds must be the fixed 24-hour bound")
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
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=_CONTROL_TIMEOUT_SECONDS,
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
        timeout=_CONTROL_TIMEOUT_SECONDS,
    )
    return "bond1.102" if completed.returncode == 0 else ""


def _tmux_session_is_fresh(name: str) -> bool:
    completed = subprocess.run(
        ("tmux", "has-session", "-t", name),
        capture_output=True,
        text=True,
        check=False,
        timeout=_CONTROL_TIMEOUT_SECONDS,
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


def _environment_assignments(values: list[str]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for assignment in values:
        name, separator, value = assignment.partition("=")
        if not separator or not name or name in environment:
            raise ValueError("--env requires a unique NAME=VALUE assignment")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or "\0" in value:
            raise ValueError("--env contains an invalid environment assignment")
        environment[name] = value
    return environment


def _node_run(argv: list[str]) -> int:
    if "--" not in argv:
        raise ValueError("node-run requires -- followed by the Stage launcher command")
    separator = argv.index("--")
    parser = argparse.ArgumentParser(prog="python -m src.multinode node-run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--machine-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--ownership-path", required=True)
    parser.add_argument("--exit-path", required=True)
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args(argv[:separator])
    command = tuple(argv[separator + 1 :])
    if not command or not Path(command[0]).is_absolute():
        raise ValueError("owned Stage command must use an absolute executable")
    environment = _environment_assignments(args.env)
    if dict(os.environ) != environment:
        raise ValueError("node-run environment differs from the complete launch environment")
    supervisor_command = tuple(sys.orig_argv)
    expected_suffix = ("-m", "src.multinode", "node-run", *argv)
    if (
        not supervisor_command
        or not Path(supervisor_command[0]).is_absolute()
        or tuple(supervisor_command[1:]) != expected_suffix
    ):
        raise ValueError("node-run supervisor argv differs from the immutable launch")
    ownership = OwnershipRecord.create(
        run_id=args.run_id,
        attempt_id=args.attempt_id,
        machine_rank=args.machine_rank,
        pid=os.getpid(),
        process_start_ticks=_proc_start_ticks(os.getpid()),
        commit=args.commit,
        command=supervisor_command,
        environment=environment,
    )
    write_ownership(args.ownership_path, ownership)
    child_pidfd: int | None = None
    pending_signals: list[int] = []

    def forward_signal(signum: int, _frame: object) -> None:
        if child_pidfd is None:
            pending_signals.append(signum)
            return
        try:
            signal.pidfd_send_signal(child_pidfd, signum)
        except ProcessLookupError:
            pass

    handled_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    previous_handlers = {
        signum: signal.signal(signum, forward_signal) for signum in handled_signals
    }
    exit_code: int
    try:
        child = subprocess.Popen(
            command,
            env=environment,
            close_fds=True,
            start_new_session=False,
        )
        child_pidfd = os.pidfd_open(child.pid)
        for signum in pending_signals:
            try:
                signal.pidfd_send_signal(child_pidfd, signum)
            except ProcessLookupError:
                break
        pending_signals.clear()
        exit_code = child.wait()
    finally:
        if child_pidfd is not None:
            descriptor = child_pidfd
            child_pidfd = None
            os.close(descriptor)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    _publish_exit_record(
        ownership=ownership,
        exit_path=Path(args.exit_path),
        exit_code=exit_code,
    )
    if exit_code < 0:
        return 128 + min(-exit_code, 127)
    return min(exit_code, 255)


def _publish_exit_record(
    *, ownership: OwnershipRecord, exit_path: Path, exit_code: int
) -> None:
    from src.multinode_runtime import _canonical_json_bytes, _publish_bytes_no_replace

    if type(exit_code) is not int:
        raise ValueError("exit code must be an integer")
    payload = {
        "schema_version": 1,
        **ownership.to_json(),
        "exit_code": exit_code,
    }
    _publish_bytes_no_replace(
        exit_path, _canonical_json_bytes(payload) + b"\n"
    )


def _write_exit_record(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.multinode node-exit")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--machine-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--exit-path", required=True)
    parser.add_argument("--ownership-path", required=True)
    parser.add_argument("--command-digest", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args(argv)
    ownership = read_ownership(args.ownership_path)
    validate_ownership(
        ownership,
        expected_run_id=args.run_id,
        expected_attempt_id=args.attempt_id,
        expected_machine_rank=args.machine_rank,
        expected_pid=ownership.pid,
        expected_commit=args.commit,
        expected_command_digest=args.command_digest,
        observed_start_ticks=ownership.process_start_ticks,
    )
    exit_path = Path(args.exit_path)
    _publish_exit_record(
        ownership=ownership,
        exit_path=exit_path,
        exit_code=args.exit_code,
    )
    return 0


def _ownership_show(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.multinode ownership-show")
    parser.add_argument("--ownership-path", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(read_ownership(args.ownership_path).to_json(), sort_keys=True))
    return 0


def _owned_process_info(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.multinode owned-process-info"
    )
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args(argv)
    process_directory = _PROC_ROOT / str(args.pid)
    command = _nul_fields(process_directory / "cmdline")
    environment: dict[str, str] = {}
    for assignment in _nul_fields(process_directory / "environ"):
        name, separator, value = assignment.partition("=")
        if not separator or name in environment:
            raise ValueError("owned process environment is malformed")
        environment[name] = value
    print(
        json.dumps(
            {
                "command": command,
                "environment": environment,
                "process_start_ticks": _proc_start_ticks(args.pid, proc_root=_PROC_ROOT),
            },
            sort_keys=True,
        )
    )
    return 0


def _owned_validation_arguments(
    argv: list[str], *, prog: str, include_signal_identity: bool
) -> tuple[argparse.Namespace, tuple[str, ...], dict[str, str]]:
    if "--" not in argv:
        raise ValueError(f"{prog} requires -- followed by the exact Stage command")
    separator = argv.index("--")
    parser = argparse.ArgumentParser(prog=f"python -m src.multinode {prog}")
    parser.add_argument("--ownership-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--machine-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--command-digest", required=True)
    parser.add_argument("--env", action="append", default=[])
    if include_signal_identity:
        parser.add_argument("--pid", type=int, required=True)
        parser.add_argument("--process-start-ticks", type=int, required=True)
    else:
        parser.add_argument("--master-addr", required=True)
        parser.add_argument("--main-process-port", type=int, required=True)
    args = parser.parse_args(argv[:separator])
    return args, tuple(argv[separator + 1 :]), _environment_assignments(args.env)


def _ownership_ready(argv: list[str]) -> int:
    args, command, environment = _owned_validation_arguments(
        argv, prog="ownership-ready", include_signal_identity=False
    )
    ownership = read_ownership(args.ownership_path)
    if ownership.command_digest != args.command_digest:
        raise ValueError("ownership command digest mismatch")
    if args.master_addr != ClusterSpec.default().master_addr:
        raise ValueError("readiness master address must match the fixed topology")
    if not 1024 <= args.main_process_port <= 65535:
        raise ValueError("readiness port must be between 1024 and 65535")
    pidfd = os.pidfd_open(ownership.pid)
    try:
        validate_owned_process(
            ownership,
            expected_run_id=args.run_id,
            expected_attempt_id=args.attempt_id,
            expected_machine_rank=args.machine_rank,
            expected_commit=args.commit,
            expected_command=command,
            expected_environment=environment,
            proc_root=_PROC_ROOT,
        )
        if not owned_listener_is_ready(
            ownership.pid,
            master_addr=args.master_addr,
            port=args.main_process_port,
            proc_root=_PROC_ROOT,
            owner_pidfd=pidfd,
            expected_owner_start_ticks=ownership.process_start_ticks,
        ):
            raise ValueError("owned rendezvous listener is not ready")
    finally:
        os.close(pidfd)
    return 0


def _signal_owned(argv: list[str]) -> int:
    args, command, environment = _owned_validation_arguments(
        argv, prog="signal-owned", include_signal_identity=True
    )
    ownership = read_ownership(args.ownership_path)
    validate_ownership(
        ownership,
        expected_run_id=args.run_id,
        expected_attempt_id=args.attempt_id,
        expected_machine_rank=args.machine_rank,
        expected_pid=args.pid,
        expected_commit=args.commit,
        expected_command_digest=args.command_digest,
        observed_start_ticks=args.process_start_ticks,
    )
    pidfd = os.pidfd_open(ownership.pid)
    try:
        validate_owned_process(
            ownership,
            expected_run_id=args.run_id,
            expected_attempt_id=args.attempt_id,
            expected_machine_rank=args.machine_rank,
            expected_commit=args.commit,
            expected_command=command,
            expected_environment=environment,
            proc_root=_PROC_ROOT,
        )
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
    finally:
        os.close(pidfd)
    return 0


def _exit_show(argv: list[str]) -> int:
    from src.multinode_runtime import _read_regular_file

    parser = argparse.ArgumentParser(prog="python -m src.multinode exit-show")
    parser.add_argument("--exit-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--machine-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--command-digest", required=True)
    args = parser.parse_args(argv)
    path = Path(args.exit_path)
    try:
        raw = _read_regular_file(path)
    except (FileNotFoundError, OSError, ValueError):
        return 3
    payload = json.loads(raw)
    expected_fields = {
        "schema_version",
        "run_id",
        "attempt_id",
        "machine_rank",
        "pid",
        "process_start_ticks",
        "commit",
        "command_digest",
        "exit_code",
    }
    if type(payload) is not dict or set(payload) != expected_fields:
        raise ValueError("invalid exit record fields")
    ownership = OwnershipRecord.from_json(
        {name: value for name, value in payload.items() if name != "exit_code"}
    )
    validate_ownership(
        ownership,
        expected_run_id=args.run_id,
        expected_attempt_id=args.attempt_id,
        expected_machine_rank=args.machine_rank,
        expected_pid=ownership.pid,
        expected_commit=args.commit,
        expected_command_digest=args.command_digest,
        observed_start_ticks=ownership.process_start_ticks,
    )
    if type(payload["exit_code"]) is not int:
        raise ValueError("exit code must be an integer")
    print(json.dumps({"exit_code": payload["exit_code"]}, sort_keys=True))
    return 0


def _artifact_manifest(argv: list[str]) -> int:
    from src.multinode_artifacts import build_node_manifest

    parser = argparse.ArgumentParser(prog="python -m src.multinode artifact-manifest")
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--checkpoint-root", default=str(_REPOSITORY_ROOT / "checkpoints")
    )
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--process-rank", type=int, action="append", required=True)
    parser.add_argument("--runtime-identity-json", required=True)
    parser.add_argument("--completion-marker", action="store_true")
    parser.add_argument("--canonical-owner", action="store_true")
    args = parser.parse_args(argv)
    try:
        identity = json.loads(args.runtime_identity_json)
        if type(identity) is not dict:
            raise ValueError("runtime identity must be a JSON object")
        manifest = build_node_manifest(
            args.run_directory,
            node_rank=args.node_rank,
            process_ranks=tuple(args.process_rank),
            run_id=args.run_id,
            runtime_identity=identity,
            completion_marker=args.completion_marker,
            checkpoint_root=args.checkpoint_root,
            canonical_owner=args.canonical_owner,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(manifest.to_json(), sort_keys=True))
    return 0


def _artifact_destination_check(argv: list[str]) -> int:
    from src.multinode_artifacts import (
        ArtifactEntry,
        NodeArtifactManifest,
        select_common_metadata_entries,
        validate_runtime_identity,
        validate_transfer_destination,
    )

    parser = argparse.ArgumentParser(
        prog="python -m src.multinode artifact-destination-check"
    )
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--checkpoint-root", default=str(_REPOSITORY_ROOT / "checkpoints")
    )
    args = parser.parse_args(argv)
    try:
        run_directory = Path(args.run_directory)
        if not run_directory.is_absolute() or ".." in run_directory.parts:
            raise ValueError("artifact destination must be the run-owned directory")
        raw = sys.stdin.buffer.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("artifact destination request exceeds the size limit")
        payload = json.loads(raw)
        if type(payload) is not dict or set(payload) != {
            "entries",
            "runtime_identity",
        }:
            raise ValueError("invalid artifact destination request fields")
        if (
            type(payload["entries"]) is not list
            or type(payload["runtime_identity"]) is not dict
        ):
            raise ValueError("invalid artifact destination request values")
        identity = validate_runtime_identity(payload["runtime_identity"])
        checkpoint_root = Path(args.checkpoint_root)
        expected_directory = (
            checkpoint_root
            / args.run_id
            / f"checkpoint-{identity['global_step']}"
        )
        if run_directory != expected_directory:
            raise ValueError("artifact destination must be the run-owned checkpoint")
        if identity["run_id"] != args.run_id:
            raise ValueError("artifact destination request run ID mismatch")
        entries = tuple(ArtifactEntry.from_json(item) for item in payload["entries"])
        physical_checkpoint = (
            checkpoint_root.resolve(strict=True)
            / args.run_id
            / f"checkpoint-{identity['global_step']}"
        )
        destinations = tuple(
            physical_checkpoint / entry.relative_path for entry in entries
        )
        request_manifest = NodeArtifactManifest(
            schema_version=1,
            run_id=args.run_id,
            node_rank=1,
            process_ranks=(2, 3),
            entries=entries,
            runtime_identity=identity,
            completion_marker=True,
            checkpoint_relative=Path(f"checkpoint-{identity['global_step']}"),
        )
        allowed_common = {
            physical_checkpoint / entry.relative_path: entry.byte_size
            for entry in select_common_metadata_entries(request_manifest)
        }
        validate_transfer_destination(
            destinations,
            validated_identical_common_metadata=allowed_common,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


def _artifact_inference_manifest(argv: list[str]) -> int:
    from src.multinode_artifacts import filter_inference_manifest, build_node_manifest

    parser = argparse.ArgumentParser(
        prog="python -m src.multinode artifact-inference-manifest"
    )
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-identity-json", required=True)
    parser.add_argument("--completion-marker", action="store_true")
    parser.add_argument(
        "--checkpoint-root", default=str(_REPOSITORY_ROOT / "checkpoints")
    )
    args = parser.parse_args(argv)
    try:
        identity = json.loads(args.runtime_identity_json)
        if type(identity) is not dict:
            raise ValueError("runtime identity must be a JSON object")
        manifest = build_node_manifest(
            args.run_directory,
            node_rank=0,
            process_ranks=(0, 1, 2, 3),
            run_id=args.run_id,
            runtime_identity=identity,
            completion_marker=args.completion_marker,
            checkpoint_root=args.checkpoint_root,
            canonical_owner=True,
        )
        # build_node_manifest validated the complete loader contract while its
        # canonical checkpoint dirfd was pinned. Filtering uses only that proof.
        filtered = filter_inference_manifest(manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(filtered.to_json(), sort_keys=True))
    return 0


def _argument_value(arguments: tuple[str, ...], option: str) -> str | None:
    result: str | None = None
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == option:
            result = arguments[index + 1]
            index += 2
            continue
        if token.startswith(f"{option}="):
            result = token.split("=", 1)[1]
        index += 1
    return result


def _replace_argument_value(
    arguments: tuple[str, ...], option: str, replacement: str
) -> tuple[str, ...]:
    values = list(arguments)
    index = 0
    while index < len(values):
        if values[index] == option:
            values[index + 1] = replacement
            index += 2
            continue
        if values[index].startswith(f"{option}="):
            values[index] = f"{option}={replacement}"
        index += 1
    return tuple(values)


def _checkpoint_artifact_identity(
    plan: LaunchPlan,
    *,
    commit: str,
    global_step: int,
) -> dict[str, object]:
    """Build the strict checkpoint identity used on both non-shared nodes."""
    if type(global_step) is not int or global_step < 0:
        raise ValueError("checkpoint global step must be a nonnegative integer")
    from src.multinode_artifacts import derive_model_state_layout

    model_state_layout = derive_model_state_layout(plan.zero_stage)
    if plan.stage == "stage1":
        sidecar = (
            "stage1_peft_metadata.json"
            if plan.tuning_method in ("lora", "dora")
            else "stage1_ema_teacher.pt"
        )
    else:
        sidecar = (
            "stage2_peft_metadata.json"
            if plan.tuning_method in ("lora", "dora")
            else "stage2-sampler.json"
        )
    return {
        "stage": plan.stage,
        "tuning_method": plan.tuning_method,
        "zero_stage": plan.zero_stage,
        "deepspeed_config_digest": plan.deepspeed_config_digest,
        "world_size": plan.world_size,
        "rank_mapping": {"0": [0, 1], "1": [2, 3]},
        "global_step": global_step,
        "model_state_layout": model_state_layout,
        "optimizer": True,
        "scheduler": True,
        "rng_ranks": [0, 1, 2, 3],
        "stage_sidecar": sidecar,
        "commit": commit,
        "run_id": plan.run_id,
    }


def _training_main(stage: Literal["stage1", "stage2"], argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=f"python -m src.multinode train-{stage}", allow_abbrev=False
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", default="initial")
    parser.add_argument("--zero-stage", choices=(1, 2, 3), type=int, required=True)
    parser.add_argument("--main-process-port", type=int, required=True)
    parser.add_argument("--result-root", default=str(_REPOSITORY_ROOT / "results"))
    parser.add_argument("--dry-run", action="store_true")
    args, stage_arguments = parser.parse_known_args(argv)
    try:
        result_root = Path(args.result_root)
        canonical_result_root = _REPOSITORY_ROOT / "results"
        if not result_root.is_absolute() or ".." in result_root.parts:
            raise ValueError("result_root must be an absolute path without traversal")
        try:
            result_root.relative_to(canonical_result_root)
        except ValueError as error:
            raise ValueError(
                "result_root must be inside /root/workspace/ILVR/results"
            ) from error
        forwarded = tuple(stage_arguments)
        if stage == "stage1":
            forwarded = validate_stage1_arguments(forwarded)
            tuning_method = _argument_value(forwarded, "--method")
            if tuning_method not in ("full", "lora", "dora"):
                raise ValueError("Stage 1 requires --method full, lora, or dora")
            stage_environment: dict[str, str] = {}
            resume_requested = _argument_value(forwarded, "--resume") is not None
        else:
            forwarded = validate_stage2_arguments(forwarded)
            tuning_method = os.environ.get("TUNING_METHOD", "full")
            if tuning_method not in ("full", "lora", "dora"):
                raise ValueError("TUNING_METHOD must be full, lora, or dora")
            output_dir = _REPOSITORY_ROOT / "checkpoints" / args.run_id
            log_file = (
                _REPOSITORY_ROOT
                / "logs"
                / "zero-stage-compatibility"
                / args.run_id
                / f"{args.attempt_id}-train.log"
            )
            stage_environment = validated_stage2_environment(
                os.environ,
                output_dir=str(output_dir),
                log_file=str(log_file),
                deepspeed_config=str(canonical_deepspeed_config(args.zero_stage)),
            )
            resume_requested = "RESUME_CHECKPOINT" in stage_environment
        if resume_requested and args.attempt_id == "initial":
            raise ValueError("resume requires a fresh --attempt-id, not 'initial'")
        if stage == "stage1" and resume_requested:
            resume = _argument_value(forwarded, "--resume")
            assert resume is not None
            canonical_resume = validate_resume_path(args.run_id, resume)
            forwarded = _replace_argument_value(
                forwarded, "--resume", str(canonical_resume)
            )
        elif stage == "stage2" and resume_requested:
            stage_environment["RESUME_CHECKPOINT"] = str(
                validate_resume_path(
                    args.run_id, stage_environment["RESUME_CHECKPOINT"]
                )
            )
        plan = build_launch_plan(
            stage=stage,
            tuning_method=tuning_method,
            zero_stage=args.zero_stage,
            run_id=args.run_id,
            main_process_port=args.main_process_port,
            attempt_id=args.attempt_id,
        )
        source_repository = Path(__file__).resolve().parents[1]
        commit = _command_stdout(("git", "rev-parse", "HEAD"), cwd=source_repository)
        commands = build_owned_tmux_commands(
            plan,
            commit=commit,
            stage_arguments=forwarded,
            stage_environment=stage_environment,
            result_root=result_root,
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "command_digests": commands.command_digests,
                        "local": commands.local,
                        "plan": _json_ready(plan),
                        "remote": commands.remote,
                    },
                    sort_keys=True,
                )
            )
            return 0
        checkpoint_root = _REPOSITORY_ROOT / "checkpoints"
        log_root = _REPOSITORY_ROOT / "logs" / "multinode"
        stage_log_root = _REPOSITORY_ROOT / "logs" / "zero-stage-compatibility"
        checkpoint_run_path = checkpoint_root / plan.run_id
        control_directory = log_root / plan.run_id / plan.attempt_id
        witness_path = result_root / plan.run_id / f"{plan.attempt_id}-train-witness.json"
        stage_log_file = (
            stage_log_root / plan.run_id / f"{plan.attempt_id}-train.log"
        )
        if stage == "stage1":
            input_paths = tuple(
                Path(value)
                for option in (
                    "--data-path",
                    "--source-data-path",
                    "--image-root",
                    "--resume",
                )
                if (value := _argument_value(forwarded, option)) is not None
            )
            model = _argument_value(forwarded, "--model")
            if model is not None and Path(model).is_absolute():
                input_paths += (Path(model),)
        else:
            input_paths = tuple(
                Path(stage_environment[name])
                for name in ("STAGE1_CHECKPOINT", "DATA_PATH", "IMAGE_ROOT")
            )
            if "RESUME_CHECKPOINT" in stage_environment:
                input_paths += (Path(stage_environment["RESUME_CHECKPOINT"]),)
        artifact_roots = (checkpoint_root, result_root, _REPOSITORY_ROOT / "logs")
        run_paths = [control_directory, witness_path, stage_log_file]
        if not resume_requested:
            run_paths.append(checkpoint_run_path)
        sessions = (
            f"ilvr-{plan.run_id}-{plan.attempt_id}-rank0",
            f"ilvr-{plan.run_id}-{plan.attempt_id}-rank1",
        )
        preflight_commands = build_preflight_commands(
            plan,
            input_paths=input_paths,
            artifact_roots=artifact_roots,
            remote_run_paths=run_paths,
            local_run_paths=run_paths,
            remote_tmux_sessions=(sessions[0],),
            local_tmux_sessions=(sessions[1],),
        )

        def subprocess_runner(command: list[str] | tuple[str, ...]):
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_CONTROL_TIMEOUT_SECONDS,
            )

        remote_snapshot = collect_node_snapshot(
            plan,
            node="remote",
            runner=subprocess_runner,
            commands=preflight_commands,
            input_paths=input_paths,
            artifact_roots=artifact_roots,
            run_paths=run_paths,
            tmux_sessions=(sessions[0],),
            repository=plan.cluster.remote_repo_root,
        )
        local_snapshot = collect_node_snapshot(
            plan,
            node="local",
            runner=subprocess_runner,
            commands=preflight_commands,
            input_paths=input_paths,
            artifact_roots=artifact_roots,
            run_paths=run_paths,
            tmux_sessions=(sessions[1],),
            repository=plan.cluster.remote_repo_root,
        )
        report = validate_preflight(remote_snapshot, local_snapshot)
        if report.commit != commit:
            raise ValueError("validated preflight commit changed before launch")
        runner = SubprocessCoordinatorRunner(
            plan,
            commands=commands,
            commit=commit,
            log_root=log_root,
        )
        if resume_requested:
            resume_source = (
                Path(_argument_value(forwarded, "--resume"))
                if stage == "stage1"
                else Path(stage_environment["RESUME_CHECKPOINT"])
            )
            match = re.fullmatch(r"checkpoint-(0|[1-9][0-9]*)", resume_source.name)
            if match is None:
                raise ValueError("resume source must be a numbered checkpoint-N")
            resume_step = int(match.group(1))
            resume_identity = _checkpoint_artifact_identity(
                plan,
                commit=commit,
                global_step=resume_step,
            )
            redistributed = checkpoint_run_path / f"checkpoint-{resume_step}"
            redistribute_resume_checkpoint(
                runner,
                checkpoint_path=checkpoint_run_path / f"checkpoint-{resume_step}",
                checkpoint_identity=resume_identity,
                requested_identity=resume_identity,
                destination=redistributed,
            )
        start_rank0_then_rank1(plan, runner, commands=commands)
        exit_codes = monitor_owned_jobs(
            plan,
            runner,
            expected_commit=commit,
            expected_command_digests={
                0: commands.command_digests[0],
                1: commands.command_digests[1],
            },
        )
        if exit_codes != {0: 0, 1: 0}:
            raise RuntimeError(f"multi-node training failed: exit_codes={exit_codes}")
        step_option = "--max-steps" if stage == "stage1" else "--max_steps"
        step_value = _argument_value(forwarded, step_option)
        if step_value is None or not step_value.isdecimal() or int(step_value) <= 0:
            raise ValueError(f"{step_option} must identify the final checkpoint step")
        final_step = int(step_value)
        checkpoint_path = checkpoint_run_path / f"checkpoint-{final_step}"
        artifact_identity = _checkpoint_artifact_identity(
            plan,
            commit=commit,
            global_step=final_step,
        )
        collect_checkpoint_artifacts(
            runner,
            checkpoint_path=checkpoint_path,
            runtime_identity=artifact_identity,
            exit_codes=exit_codes,
            world_size=plan.world_size,
        )
        if stage == "stage1":
            batch_value = _argument_value(forwarded, "--micro-batch") or "8"
            batch_per_gpu = int(batch_value)
        else:
            batch_per_gpu = 1
        try:
            witness = write_training_witness(
                plan,
                result_root=result_root,
                exit_codes=exit_codes,
                checkpoint_path=checkpoint_run_path,
                batch_per_gpu=batch_per_gpu,
                finite_loss_evidence_path=commands.finite_loss_evidence_path,
            )
        except ValueError as error:
            incomplete = write_incomplete_attempt(
                plan,
                result_root=result_root,
                reason=str(error),
                exit_codes=exit_codes,
            )
            raise RuntimeError(
                f"training attempt incomplete; record={incomplete}: {error}"
            ) from error
        print(json.dumps({"training_witness": str(witness)}, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


def _prepare_inference_main(argv: list[str]) -> int:
    """Prepare a complete local inference artifact without starting evaluators."""
    from src.multinode_runtime import prepare_inference_artifacts

    parser = argparse.ArgumentParser(
        prog="python -m src.multinode prepare-inference", allow_abbrev=False
    )
    _add_plan_arguments(parser)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    try:
        if args.checkpoint_step < 0:
            raise ValueError("checkpoint step must be nonnegative")
        plan = _plan_from_args(args)
        destination = Path(args.destination)
        result_root = _REPOSITORY_ROOT / "results"
        if not destination.is_absolute() or ".." in destination.parts:
            raise ValueError("inference destination must be an absolute trusted path")
        try:
            destination.relative_to(result_root / plan.run_id)
        except ValueError as error:
            raise ValueError(
                "inference destination must belong to the run result tree"
            ) from error
        source_repository = Path(__file__).resolve().parents[1]
        commit = _command_stdout(("git", "rev-parse", "HEAD"), cwd=source_repository)
        commands = build_owned_tmux_commands(plan, commit=commit)
        runner = SubprocessCoordinatorRunner(
            plan,
            commands=commands,
            commit=commit,
            log_root=_REPOSITORY_ROOT / "logs" / "multinode",
        )
        runtime_identity = _checkpoint_artifact_identity(
            plan,
            commit=commit,
            global_step=args.checkpoint_step,
        )
        prepared = prepare_inference_artifacts(
            runner,
            checkpoint_path=(
                _REPOSITORY_ROOT
                / "checkpoints"
                / plan.run_id
                / f"checkpoint-{args.checkpoint_step}"
            ),
            runtime_identity=runtime_identity,
            destination=destination,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({"inference_artifact": str(prepared)}, sort_keys=True))
    return 0


def _prepared_inference_show(argv: list[str]) -> int:
    """Validate one Task 6 prepared artifact without importing an evaluator."""
    from src.multinode_artifacts import (
        NodeArtifactManifest,
        inference_manifest_path,
        inference_model_identity,
        validate_transferred_entries,
    )
    from src.multinode_runtime import _read_regular_file

    parser = argparse.ArgumentParser(
        prog="python -m src.multinode prepared-inference-show", allow_abbrev=False
    )
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args(argv)
    try:
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute() or ".." in model_dir.parts:
            raise ValueError("prepared inference model path must be absolute")
        model_dir.relative_to(_REPOSITORY_ROOT / "results")
        payload = json.loads(_read_regular_file(inference_manifest_path(model_dir)))
        manifest = NodeArtifactManifest.from_json(payload)
        inference_model_identity(manifest)
        validate_transferred_entries(model_dir, manifest.entries)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(manifest.to_json(), sort_keys=True))
    return 0


def _prepare_evaluation_output(argv: list[str]) -> int:
    from src.multinode_artifacts import reserve_inference_destination

    parser = argparse.ArgumentParser(
        prog="python -m src.multinode prepare-evaluation-output", allow_abbrev=False
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        _validate_identifier("run_id", args.run_id)
        output = Path(args.output_dir)
        relative = output.relative_to(_REPOSITORY_ROOT / "results")
        if not relative.parts or relative.parts[-1] != args.run_id:
            raise ValueError("evaluation output must belong to the run ID")
        reserve_inference_destination(output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


def _evaluate_main(argv: list[str]) -> int:
    """Coordinate four independent Task 6-validated evaluation workers."""
    from src.multinode_runtime import (
        EvaluationPlan,
        EvaluationSubprocessRunner,
        _build_evaluation_commands_for_stage,
        run_evaluation,
    )

    parser = argparse.ArgumentParser(
        prog="python -m src.multinode evaluate", allow_abbrev=False
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", default="initial")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-data-path", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model-dir")
    parser.add_argument("--cache-dir", default=str(_REPOSITORY_ROOT / "cache"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = EvaluationPlan(
            run_id=args.run_id,
            attempt_id=args.attempt_id,
            model_dir=Path(args.model_dir),
            test_data_path=Path(args.test_data_path),
            task_name=args.task_name,
            output_dir=Path(args.output_dir),
            base_model_dir=(Path(args.base_model_dir) if args.base_model_dir else None),
            cache_dir=Path(args.cache_dir),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if args.dry_run:
            dry_identity = f"task6:{'0' * 64}"
            variants = {
                stage: _build_evaluation_commands_for_stage(
                    plan,
                    stage=stage,
                    model_identity=dry_identity,
                    commit="0" * 40,
                )
                for stage in ("stage1", "stage2")
            }
            first = variants["stage1"]
            print(
                json.dumps(
                    {
                        "model_identity_source": "validated Task 6 inference manifest",
                        "plan": _json_ready(plan),
                        "remote_result_files": [
                            value.decode("utf-8")
                            for value in first.remote_files_payload.split(b"\0")
                            if value
                        ],
                        "remote_result_pull": list(first.remote_pull),
                        "worker_commands_by_observed_stage": {
                            stage: [list(worker.launch) for worker in commands.workers]
                            for stage, commands in variants.items()
                        },
                    },
                    sort_keys=True,
                )
            )
            return 0
        source_repository = Path(__file__).resolve().parents[1]
        commit = _command_stdout(("git", "rev-parse", "HEAD"), cwd=source_repository)
        runner = EvaluationSubprocessRunner(plan, commit=commit)
        summary = run_evaluation(plan, runner)
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(_json_ready(summary), sort_keys=True))
    return 0


def _main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "capacity-stage1":
        from src.multinode_capacity import capacity_stage1_main

        return capacity_stage1_main(argv[1:])
    if argv and argv[0] == "capacity-evidence-show":
        from src.multinode_capacity import _capacity_evidence_main

        return _capacity_evidence_main(argv[1:])
    if argv and argv[0] in ("train-stage1", "train-stage2"):
        return _training_main(argv[0].removeprefix("train-"), argv[1:])  # type: ignore[arg-type]
    if argv and argv[0] == "prepare-inference":
        return _prepare_inference_main(argv[1:])
    if argv and argv[0] == "evaluate":
        return _evaluate_main(argv[1:])
    if argv and argv[0] == "prepared-inference-show":
        return _prepared_inference_show(argv[1:])
    if argv and argv[0] == "prepare-evaluation-output":
        return _prepare_evaluation_output(argv[1:])
    if argv and argv[0] == "node-run":
        return _node_run(argv[1:])
    if argv and argv[0] == "node-exit":
        return _write_exit_record(argv[1:])
    if argv and argv[0] == "ownership-show":
        return _ownership_show(argv[1:])
    if argv and argv[0] == "owned-process-info":
        try:
            return _owned_process_info(argv[1:])
        except (OSError, ValueError):
            return 3
    if argv and argv[0] == "ownership-ready":
        try:
            return _ownership_ready(argv[1:])
        except (OSError, ValueError, json.JSONDecodeError):
            return 1
    if argv and argv[0] == "signal-owned":
        return _signal_owned(argv[1:])
    if argv and argv[0] == "exit-show":
        return _exit_show(argv[1:])
    if argv and argv[0] == "artifact-manifest":
        return _artifact_manifest(argv[1:])
    if argv and argv[0] == "artifact-destination-check":
        return _artifact_destination_check(argv[1:])
    if argv and argv[0] == "artifact-inference-manifest":
        return _artifact_inference_manifest(argv[1:])
    if argv and argv[0] == "prepare-attempt":
        internal = argparse.ArgumentParser(prog="python -m src.multinode prepare-attempt")
        internal.add_argument("--log-root", required=True)
        internal.add_argument("--run-id", required=True)
        internal.add_argument("--attempt-id", required=True)
        parsed = internal.parse_args(argv[1:])
        prepare_attempt_directory(
            log_root=parsed.log_root,
            run_id=parsed.run_id,
            attempt_id=parsed.attempt_id,
        )
        return 0
    if argv and argv[0] == "receive-plan":
        internal = argparse.ArgumentParser(prog="python -m src.multinode receive-plan")
        internal.add_argument("--path", required=True)
        internal.add_argument("--size", type=int, required=True)
        internal.add_argument("--sha256", required=True)
        parsed = internal.parse_args(argv[1:])
        raw = sys.stdin.buffer.read(1024 * 1024 + 1)
        receive_launch_plan(
            path=parsed.path,
            raw=raw,
            expected_size=parsed.size,
            expected_sha256=parsed.sha256,
        )
        return 0
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
                return subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=_CONTROL_TIMEOUT_SECONDS,
                )

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
