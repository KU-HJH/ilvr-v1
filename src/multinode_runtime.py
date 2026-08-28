"""Pure validation and command construction for the fixed ILVR cluster."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, Sequence

from src.deepspeed_runtime import load_deepspeed_runtime_profile

if TYPE_CHECKING:
    from src.multinode import LaunchPlan


_REMOTE_HOST = "korea@10.1.102.41"
_REMOTE_CONTAINER = "llm_kd"
_REQUIRED_FREE_BYTES = 100 * 1024**3
_REPOSITORY_ROOT = Path("/root/workspace/ILVR")
_CHECKPOINT_ROOT = _REPOSITORY_ROOT / "checkpoints"
_RESULT_ROOT = _REPOSITORY_ROOT / "results"
_LOG_ROOT = _REPOSITORY_ROOT / "logs" / "multinode"
_STAGE_LOG_ROOT = _REPOSITORY_ROOT / "logs" / "zero-stage-compatibility"
_OWNERSHIP_SCHEMA_VERSION = 1
_WITNESS_SCHEMA_VERSION = 1
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,127}$")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must match {_IDENTIFIER_PATTERN.pattern}")
    return value


def _require_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validated_command(
    command: Sequence[str], environment: Mapping[str, str]
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    argv = tuple(command)
    if not argv:
        raise ValueError("command must not be empty")
    for value in argv:
        if not isinstance(value, str) or not value or "\0" in value:
            raise ValueError("command arguments must be non-empty strings without NUL")
    pairs: list[tuple[str, str]] = []
    for name, value in environment.items():
        if not isinstance(name, str) or not _ENVIRONMENT_NAME_PATTERN.fullmatch(name):
            raise ValueError("environment variable names must be portable identifiers")
        if not isinstance(value, str) or "\0" in value:
            raise ValueError("environment values must be strings without NUL")
        pairs.append((name, value))
    return argv, tuple(sorted(pairs))


def canonical_command_digest(
    command: Sequence[str], environment: Mapping[str, str] | None = None
) -> str:
    """Digest the exact argv boundaries and explicit environment overlay."""
    argv, env_pairs = _validated_command(command, environment or {})
    payload = {"argv": list(argv), "environment": [list(pair) for pair in env_pairs]}
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class OwnershipRecord:
    """Exact process identity required before any run-owned signal."""

    schema_version: int
    run_id: str
    attempt_id: str
    machine_rank: int
    pid: int
    process_start_ticks: int
    commit: str
    command_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != _OWNERSHIP_SCHEMA_VERSION
        ):
            raise ValueError("invalid ownership schema version")
        _require_identifier("run_id", self.run_id)
        _require_identifier("attempt_id", self.attempt_id)
        if type(self.machine_rank) is not int or self.machine_rank not in (0, 1):
            raise ValueError("machine_rank must be 0 or 1")
        if type(self.pid) is not int or self.pid <= 1:
            raise ValueError("pid must identify a non-init process")
        if type(self.process_start_ticks) is not int or self.process_start_ticks <= 0:
            raise ValueError("process_start_ticks must be positive")
        if (
            not isinstance(self.commit, str)
            or len(self.commit) != 40
            or any(character not in "0123456789abcdef" for character in self.commit)
        ):
            raise ValueError("commit must be a lowercase 40-character Git object ID")
        _require_digest("command_digest", self.command_digest)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        attempt_id: str,
        machine_rank: int,
        pid: int,
        process_start_ticks: int,
        commit: str,
        command: Sequence[str],
        environment: Mapping[str, str] | None = None,
    ) -> "OwnershipRecord":
        return cls(
            schema_version=_OWNERSHIP_SCHEMA_VERSION,
            run_id=run_id,
            attempt_id=attempt_id,
            machine_rank=machine_rank,
            pid=pid,
            process_start_ticks=process_start_ticks,
            commit=commit,
            command_digest=canonical_command_digest(command, environment),
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OwnershipRecord":
        expected = {
            "schema_version",
            "run_id",
            "attempt_id",
            "machine_rank",
            "pid",
            "process_start_ticks",
            "commit",
            "command_digest",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("invalid ownership record fields")
        return cls(**dict(payload))  # type: ignore[arg-type]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def validate_ownership(
    record: OwnershipRecord,
    *,
    expected_run_id: str,
    expected_attempt_id: str,
    expected_commit: str,
    observed_start_ticks: int,
    expected_machine_rank: int | None = None,
    expected_pid: int | None = None,
    expected_command_digest: str | None = None,
    observed_command_digest: str | None = None,
    observed_commit: str | None = None,
) -> bool:
    """Validate an ownership record and any live observations exactly."""
    if not isinstance(record, OwnershipRecord):
        raise TypeError("record must be an OwnershipRecord")
    if record.schema_version != _OWNERSHIP_SCHEMA_VERSION:
        raise ValueError("ownership schema version mismatch")
    if record.run_id != expected_run_id:
        raise ValueError("ownership run ID mismatch")
    if record.attempt_id != expected_attempt_id:
        raise ValueError("ownership attempt ID mismatch")
    if expected_machine_rank is not None and record.machine_rank != expected_machine_rank:
        raise ValueError("ownership machine rank mismatch")
    if expected_pid is not None and record.pid != expected_pid:
        raise ValueError("ownership PID mismatch")
    if record.process_start_ticks != observed_start_ticks:
        raise ValueError("ownership process start time mismatch")
    if record.commit != expected_commit:
        raise ValueError("ownership commit mismatch")
    if observed_commit is not None and observed_commit != record.commit:
        raise ValueError("live ownership commit mismatch")
    if (
        expected_command_digest is not None
        and record.command_digest != expected_command_digest
    ):
        raise ValueError("ownership command digest mismatch")
    if (
        observed_command_digest is not None
        and observed_command_digest != record.command_digest
    ):
        raise ValueError("live ownership command digest mismatch")
    return True


def write_ownership(path: str | Path, record: OwnershipRecord) -> Path:
    """Atomically publish one mode-0600 ownership record without replacement."""
    if not isinstance(record, OwnershipRecord):
        raise TypeError("record must be an OwnershipRecord")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace ownership: {destination}")
    payload = _canonical_json_bytes(record.to_json()) + b"\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to replace ownership: {destination}")
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    except BaseException:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


def read_ownership(path: str | Path) -> OwnershipRecord:
    ownership_path = Path(path)
    if ownership_path.is_symlink() or not ownership_path.is_file():
        raise ValueError("ownership path must be a regular file")
    return OwnershipRecord.from_json(json.loads(ownership_path.read_bytes()))


def _proc_start_ticks(pid: int, *, proc_root: str | Path = "/proc") -> int:
    stat_text = (Path(proc_root) / str(pid) / "stat").read_text(encoding="utf-8")
    closing_parenthesis = stat_text.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("process stat has no command terminator")
    trailing_fields = stat_text[closing_parenthesis + 2 :].split()
    if len(trailing_fields) <= 19:
        raise ValueError("process stat is missing start time")
    try:
        start_ticks = int(trailing_fields[19])
    except ValueError as error:
        raise ValueError("process stat start time is not an integer") from error
    if start_ticks <= 0:
        raise ValueError("process stat start time must be positive")
    return start_ticks


def _nul_fields(path: Path) -> tuple[str, ...]:
    raw = path.read_bytes()
    try:
        return tuple(
            field.decode("utf-8") for field in raw.split(b"\0") if field
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"process {path.name} is not valid UTF-8") from error


def validate_owned_process(
    record: OwnershipRecord,
    *,
    expected_run_id: str,
    expected_attempt_id: str,
    expected_machine_rank: int,
    expected_commit: str,
    expected_command: Sequence[str],
    expected_environment: Mapping[str, str],
    proc_root: str | Path = "/proc",
) -> bool:
    """Bind an ownership record to the current PID, start ticks, argv, and env."""
    process_directory = Path(proc_root) / str(record.pid)
    observed_command = _nul_fields(process_directory / "cmdline")
    if not observed_command:
        raise ValueError("owned process command is empty")
    raw_environment = _nul_fields(process_directory / "environ")
    observed_environment: dict[str, str] = {}
    expected_names = {*expected_environment, "ILVR_OWNED_COMMAND_DIGEST"}
    for assignment in raw_environment:
        name, separator, value = assignment.partition("=")
        if separator and name in expected_names:
            observed_environment[name] = value
    expected_digest = canonical_command_digest(expected_command, expected_environment)
    observed_digest = observed_environment.pop("ILVR_OWNED_COMMAND_DIGEST", None)
    if observed_environment != dict(expected_environment):
        raise ValueError("live ownership environment mismatch")
    observed_commit = observed_environment.get("ILVR_REPOSITORY_COMMIT")
    return validate_ownership(
        record,
        expected_run_id=expected_run_id,
        expected_attempt_id=expected_attempt_id,
        expected_machine_rank=expected_machine_rank,
        expected_pid=record.pid,
        expected_commit=expected_commit,
        expected_command_digest=expected_digest,
        observed_start_ticks=_proc_start_ticks(record.pid, proc_root=proc_root),
        observed_command_digest=observed_digest,
        observed_commit=observed_commit,
    )


@dataclass(frozen=True)
class SerializedLaunchPlan:
    path: Path
    size: int
    sha256: str


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def serialize_launch_plan(
    plan: "LaunchPlan", *, log_root: str | Path = _LOG_ROOT
) -> SerializedLaunchPlan:
    """Exclusively serialize the small plan inside its attempt directory."""
    from src.multinode import LaunchPlan

    if not isinstance(plan, LaunchPlan):
        raise TypeError("plan must be a LaunchPlan")
    root = Path(log_root)
    root.mkdir(parents=True, exist_ok=True)
    run_directory = root / plan.run_id
    run_directory.mkdir(mode=0o700, exist_ok=True)
    attempt_directory = run_directory / plan.attempt_id
    attempt_directory.mkdir(mode=0o700, exist_ok=False)
    path = attempt_directory / "launch-plan.json"
    payload = _canonical_json_bytes(_json_ready(plan)) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)
    return SerializedLaunchPlan(
        path=path,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


_STAGE1_VALUE_OPTIONS = {
    "--method",
    "--model",
    "--data-path",
    "--source-data-path",
    "--image-root",
    "--micro-batch",
    "--epochs",
    "--learning-rate",
    "--gradient-accumulation",
    "--seed",
    "--attention-implementation",
    "--save-strategy",
    "--save-steps",
    "--resume",
    "--max-steps",
    "--benchmark-run-config",
}
_STAGE1_BOOLEAN_OPTIONS = {"--final-save", "--no-final-save", "--use_lora"}
_STAGE1_COORDINATOR_OPTIONS = {
    "--output-dir",
    "--log-file",
    "--deepspeed-config",
    "--manifest",
    "--producer-evidence",
    "--smoke-resume-witness",
}


def _validate_allowlisted_arguments(
    arguments: Sequence[str],
    *,
    value_options: set[str],
    boolean_options: set[str],
    owned_options: set[str],
    label: str,
) -> tuple[str, ...]:
    values = tuple(arguments)
    index = 0
    while index < len(values):
        token = values[index]
        if not isinstance(token, str) or "\0" in token:
            raise ValueError(f"{label} arguments must be strings without NUL")
        option, separator, inline_value = token.partition("=")
        if option in owned_options:
            raise ValueError(f"coordinator-owned option cannot be overridden: {option}")
        if option in boolean_options:
            if separator:
                raise ValueError(f"{option} does not accept a value")
            index += 1
            continue
        if option not in value_options:
            raise ValueError(f"{option} is not an approved {label} argument")
        if separator:
            if not inline_value:
                raise ValueError(f"{option} requires a value")
            index += 1
            continue
        if index + 1 >= len(values) or not values[index + 1]:
            raise ValueError(f"{option} requires a value")
        if "\0" in values[index + 1]:
            raise ValueError(f"{option} value must not contain NUL")
        index += 2
    return values


def validate_stage1_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    """Forward only flags already accepted by the Stage 1 launcher."""
    return _validate_allowlisted_arguments(
        arguments,
        value_options=_STAGE1_VALUE_OPTIONS,
        boolean_options=_STAGE1_BOOLEAN_OPTIONS,
        owned_options=_STAGE1_COORDINATOR_OPTIONS,
        label="Stage 1",
    )


_STAGE2_ENVIRONMENT_ALLOWLIST = {
    "STAGE1_CHECKPOINT",
    "DATA_PATH",
    "IMAGE_ROOT",
    "TUNING_METHOD",
    "LORA_R",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_TARGET_MODULES",
    "LATENT_SIZE",
    "SUPERVISE_LATENT_BOUNDARIES",
    "LEARNING_RATE",
    "GRADIENT_ACCUMULATION_STEPS",
    "ATTENTION_IMPLEMENTATION",
    "GRADIENT_CHECKPOINTING",
    "MAX_SEQUENCE_LENGTH",
    "RESUME_CHECKPOINT",
}
_STAGE2_REQUIRED_ENVIRONMENT = {"STAGE1_CHECKPOINT", "DATA_PATH", "IMAGE_ROOT"}


def validated_stage2_environment(
    source: Mapping[str, str],
    *,
    output_dir: str,
    log_file: str,
    deepspeed_config: str,
) -> dict[str, str]:
    """Copy only the existing approved Stage 2 knobs and owned artifact values."""
    result = {
        name: source[name]
        for name in sorted(_STAGE2_ENVIRONMENT_ALLOWLIST)
        if name in source
    }
    missing = sorted(
        name for name in _STAGE2_REQUIRED_ENVIRONMENT if not result.get(name)
    )
    if missing:
        raise ValueError(f"missing required Stage 2 environment: {', '.join(missing)}")
    result.update(
        {
            "OUTPUT_DIR": output_dir,
            "LOG_FILE": log_file,
            "DEEPSPEED_CONFIG": deepspeed_config,
        }
    )
    _validated_command(("stage2-environment",), result)
    return result


_STAGE2_VALUE_OPTIONS = {
    "--evaluation_output_path",
    "--model",
    "--task",
    "--test_data_path",
    "--cache_dir",
    "--ce_weight",
    "--warm_up_steps",
    "--sim_weight",
    "--ema_tau",
    "--coverage_p",
    "--image_pool_k",
    "--latent_ce_ratio",
    "--stage1_adapter_path",
    "--helper_group_L",
    "--latent_num_segments_train",
    "--save_strategy",
    "--max_train_samples",
    "--max_steps",
    "--seed",
}
_STAGE2_BOOLEAN_OPTIONS = {
    "--save_final_model",
    "--no-save_final_model",
    "--final_save",
    "--no_final_save",
    "--use_cpu",
}
_STAGE2_COORDINATOR_OPTIONS = {
    "--stage",
    "--load_model_path",
    "--data_path",
    "--image_root",
    "--save_model_path",
    "--log_file",
    "--latent_size",
    "--supervise_latent_boundaries",
    "--no-supervise_latent_boundaries",
    "--learning_rate",
    "--gradient_accumulation_steps",
    "--per_device_train_batch_size",
    "--attention_implementation",
    "--gradient_checkpointing",
    "--no-gradient_checkpointing",
    "--max_seq_length_train",
    "--deepspeed_config",
    "--resume_from_checkpoint_path",
    "--tuning_method",
    "--lora_r",
    "--lora_alpha",
    "--lora_dropout",
    "--lora_target_modules",
    "--benchmark_evidence_path",
    "--benchmark_smoke_resume_witness",
}


def validate_stage2_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    return _validate_allowlisted_arguments(
        arguments,
        value_options=_STAGE2_VALUE_OPTIONS,
        boolean_options=_STAGE2_BOOLEAN_OPTIONS,
        owned_options=_STAGE2_COORDINATOR_OPTIONS,
        label="Stage 2",
    )


def _distributed_environment(plan: "LaunchPlan", machine_rank: int) -> dict[str, str]:
    return {
        "ILVR_NUM_MACHINES": str(plan.cluster.num_machines),
        "ILVR_NUM_PROCESSES": str(plan.cluster.num_processes),
        "ILVR_MACHINE_RANK": str(machine_rank),
        "ILVR_MAIN_PROCESS_IP": plan.cluster.master_addr,
        "ILVR_MAIN_PROCESS_PORT": str(plan.main_process_port),
        "ILVR_SOCKET_IFNAME": plan.cluster.socket_ifname,
    }


@dataclass(frozen=True)
class OwnedTmuxCommands:
    remote: tuple[str, ...]
    local: tuple[str, ...]
    command_digests: tuple[str, str]
    node_specs: tuple["OwnedNodeCommand", "OwnedNodeCommand"]
    finite_loss_evidence_path: Path | None


@dataclass(frozen=True)
class OwnedNodeCommand:
    machine_rank: int
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    ownership_path: Path
    exit_path: Path


def _node_shell_command(
    plan: "LaunchPlan",
    *,
    machine_rank: int,
    commit: str,
    stage_command: tuple[str, ...],
    environment: Mapping[str, str],
    log_root: Path,
) -> tuple[str, str, OwnedNodeCommand]:
    repository = plan.cluster.remote_repo_root
    attempt_directory = log_root / plan.run_id / plan.attempt_id
    ownership_path = attempt_directory / f"ownership-rank{machine_rank}.json"
    exit_path = attempt_directory / f"exit-rank{machine_rank}.json"
    digest = canonical_command_digest(stage_command, environment)
    node_run = [
        str(repository / "venv" / "ilvr-venv" / "bin" / "python"),
        "-m",
        "src.multinode",
        "node-run",
        "--run-id",
        plan.run_id,
        "--attempt-id",
        plan.attempt_id,
        "--machine-rank",
        str(machine_rank),
        "--commit",
        commit,
        "--ownership-path",
        str(ownership_path),
    ]
    for name, value in sorted(environment.items()):
        node_run.extend(("--env", f"{name}={value}"))
    node_run.extend(("--", *stage_command))
    node_exit = [
        str(repository / "venv" / "ilvr-venv" / "bin" / "python"),
        "-m",
        "src.multinode",
        "node-exit",
        "--run-id",
        plan.run_id,
        "--attempt-id",
        plan.attempt_id,
        "--machine-rank",
        str(machine_rank),
        "--exit-path",
        str(exit_path),
        "--ownership-path",
        str(ownership_path),
        "--command-digest",
        digest,
        "--commit",
        commit,
    ]
    # This fixed shell layer exists only to persist the exec'd launcher's exit code.
    # All variable values enter through shlex.join and never as shell syntax.
    payload = (
        f"{shlex.join(node_run)}; exit_status=$?; "
        f"{shlex.join(node_exit)} --exit-code \"$exit_status\"; "
        'exit "$exit_status"'
    )
    return (
        payload,
        digest,
        OwnedNodeCommand(
            machine_rank=machine_rank,
            command=stage_command,
            environment=tuple(sorted(environment.items())),
            ownership_path=ownership_path,
            exit_path=exit_path,
        ),
    )


def build_owned_tmux_commands(
    plan: "LaunchPlan",
    *,
    commit: str,
    stage_arguments: Sequence[str] = (),
    stage_environment: Mapping[str, str] | None = None,
    checkpoint_root: str | Path = _CHECKPOINT_ROOT,
    result_root: str | Path = _RESULT_ROOT,
    log_root: str | Path = _LOG_ROOT,
    stage_log_root: str | Path = _STAGE_LOG_ROOT,
) -> OwnedTmuxCommands:
    """Construct remote and local tmux launches without executing either."""
    from src.multinode import LaunchPlan

    if not isinstance(plan, LaunchPlan):
        raise TypeError("plan must be a LaunchPlan")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("commit must be a lowercase 40-character Git object ID")
    repository = plan.cluster.remote_repo_root
    checkpoint = Path(checkpoint_root) / plan.run_id
    attempt_log = Path(stage_log_root) / plan.run_id / f"{plan.attempt_id}-train.log"
    base_environment = dict(stage_environment or {})
    if plan.stage == "stage1":
        forwarded = validate_stage1_arguments(stage_arguments)
        stage_owned_arguments: tuple[str, ...] = ()
        if any(
            argument == "--benchmark-run-config"
            or argument.startswith("--benchmark-run-config=")
            for argument in forwarded
        ):
            finite_loss_evidence_path = (
                Path(result_root)
                / plan.run_id
                / f"{plan.attempt_id}-producer-evidence.json"
            )
            stage_owned_arguments = (
                "--producer-evidence",
                str(finite_loss_evidence_path),
            )
            if any(
                argument == "--resume" or argument.startswith("--resume=")
                for argument in forwarded
            ):
                stage_owned_arguments += (
                    "--smoke-resume-witness",
                    str(
                        Path(result_root)
                        / plan.run_id
                        / f"{plan.attempt_id}-smoke-resume-witness.json"
                    ),
                )
        else:
            finite_loss_evidence_path = None
        stage_command = (
            "/usr/bin/bash",
            str(repository / "run_stage1_training.sh"),
            *forwarded,
            "--output-dir",
            str(checkpoint),
            "--log-file",
            str(attempt_log),
            "--deepspeed-config",
            str(plan.deepspeed_config),
            *stage_owned_arguments,
        )
    else:
        finite_loss_evidence_path = None
        forwarded = validate_stage2_arguments(stage_arguments)
        stage_command = (
            "/usr/bin/bash",
            str(repository / "run_stage2_training.sh"),
            *forwarded,
        )
    node_commands: list[tuple[str, ...]] = []
    digests: list[str] = []
    node_specs: list[OwnedNodeCommand] = []
    for machine_rank in (plan.cluster.remote_rank, plan.cluster.local_rank):
        environment = {
            **base_environment,
            **_distributed_environment(plan, machine_rank),
            "ILVR_REPOSITORY_COMMIT": commit,
        }
        payload, digest, node_spec = _node_shell_command(
            plan,
            machine_rank=machine_rank,
            commit=commit,
            stage_command=stage_command,
            environment=environment,
            log_root=Path(log_root),
        )
        session = f"ilvr-{plan.run_id}-{plan.attempt_id}-rank{machine_rank}"
        node_commands.append(
            ("tmux", "new-session", "-d", "-s", session, "--", "bash", "-lc", payload)
        )
        digests.append(digest)
        node_specs.append(node_spec)
    return OwnedTmuxCommands(
        remote=tuple(build_remote_exec(node_commands[0])),
        local=node_commands[1],
        command_digests=(digests[0], digests[1]),
        node_specs=(node_specs[0], node_specs[1]),
        finite_loss_evidence_path=finite_loss_evidence_path,
    )


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _subprocess_command_runner(
    command: Sequence[str], *, input_bytes: bytes | None = None
) -> subprocess.CompletedProcess[str]:
    if input_bytes is None:
        return subprocess.run(command, capture_output=True, text=True, check=False)
    completed = subprocess.run(
        command, input=input_bytes, capture_output=True, text=False, check=False
    )
    return subprocess.CompletedProcess(
        completed.args,
        completed.returncode,
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


class SubprocessCoordinatorRunner:
    """Execute fixed coordinator commands through an injectable subprocess boundary."""

    def __init__(
        self,
        plan: "LaunchPlan",
        *,
        commands: OwnedTmuxCommands,
        commit: str,
        log_root: str | Path = _LOG_ROOT,
        command_runner: CommandRunner = _subprocess_command_runner,
    ) -> None:
        self.plan = plan
        self.commands = commands
        self.commit_value = commit
        self.log_root = Path(log_root)
        self.command_runner = command_runner
        self.serialized_plan: SerializedLaunchPlan | None = None
        self._seen_ownership: set[int] = set()

    def _run(
        self, command: Sequence[str], *, input_bytes: bytes | None = None
    ) -> subprocess.CompletedProcess[str]:
        completed = self.command_runner(command, input_bytes=input_bytes)
        if completed.returncode != 0:
            raise RuntimeError(
                f"coordinator command failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
        return completed

    def _node_spec(self, machine_rank: int) -> OwnedNodeCommand:
        if machine_rank not in (0, 1):
            raise ValueError("machine_rank must be 0 or 1")
        spec = self.commands.node_specs[machine_rank]
        if spec.machine_rank != machine_rank:
            raise ValueError("owned node command rank mismatch")
        return spec

    def _node_command(
        self, machine_rank: int, arguments: Sequence[str]
    ) -> tuple[str, ...]:
        if machine_rank == self.plan.cluster.remote_rank:
            return tuple(build_remote_exec(arguments))
        if machine_rank == self.plan.cluster.local_rank:
            return tuple(arguments)
        raise ValueError("machine_rank must be 0 or 1")

    def _internal_command(
        self, machine_rank: int, subcommand: str, arguments: Sequence[str]
    ) -> tuple[str, ...]:
        python = (
            self.plan.cluster.remote_repo_root
            / "venv"
            / "ilvr-venv"
            / "bin"
            / "python"
        )
        return self._node_command(
            machine_rank,
            (str(python), "-m", "src.multinode", subcommand, *arguments),
        )

    def _process_snapshot(self, ownership: OwnershipRecord) -> dict[str, object]:
        spec = self._node_spec(ownership.machine_rank)
        arguments = ["--pid", str(ownership.pid)]
        for name, _value in spec.environment:
            arguments.extend(("--environment-name", name))
        completed = self._run(
            self._internal_command(
                ownership.machine_rank, "owned-process-info", arguments
            )
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("owned process info returned invalid JSON") from error
        if type(payload) is not dict:
            raise ValueError("owned process info must be a JSON object")
        return payload

    def prepare_plan(self, plan: "LaunchPlan") -> None:
        if plan != self.plan:
            raise ValueError("runner plan does not match launch plan")
        serialized = serialize_launch_plan(plan, log_root=self.log_root)
        if serialized.size > 1024 * 1024:
            raise ValueError("launch plan exceeds the one-MiB transfer limit")
        remote_directory = self.log_root / plan.run_id / plan.attempt_id
        repository_python = (
            plan.cluster.remote_repo_root / "venv" / "ilvr-venv" / "bin" / "python"
        )
        mkdir_script = (
            "from pathlib import Path; import sys; "
            "Path(sys.argv[1]).mkdir(parents=True, mode=0o700, exist_ok=False)"
        )
        self._run(
            build_remote_exec(
                (str(repository_python), "-c", mkdir_script, str(remote_directory))
            )
        )
        transfer_script = (
            "import hashlib, os, pathlib, sys; "
            "path=pathlib.Path(sys.argv[1]); size=int(sys.argv[2]); digest=sys.argv[3]; "
            "raw=sys.stdin.buffer.read(1048577); "
            "(len(raw)==size and len(raw)<=1048576 and hashlib.sha256(raw).hexdigest()==digest) "
            "or sys.exit('launch plan transfer mismatch'); "
            "fd=os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600); "
            "stream=os.fdopen(fd, 'wb'); stream.write(raw); stream.flush(); "
            "os.fsync(stream.fileno()); stream.close(); stored=path.read_bytes(); "
            "(len(stored)==size and hashlib.sha256(stored).hexdigest()==digest) "
            "or sys.exit('stored launch plan mismatch')"
        )
        remote_path = remote_directory / "launch-plan.json"
        self._run(
            _build_remote_stdin_exec(
                (
                    str(repository_python),
                    "-c",
                    transfer_script,
                    str(remote_path),
                    str(serialized.size),
                    serialized.sha256,
                )
            ),
            input_bytes=serialized.path.read_bytes(),
        )
        self.serialized_plan = serialized

    def start_remote(self, command: Sequence[str]) -> None:
        if tuple(command) != self.commands.remote:
            raise ValueError("remote tmux command differs from the immutable launch")
        self._run(command)

    def rank0_is_ready(self, plan: "LaunchPlan") -> bool:
        if plan != self.plan:
            raise ValueError("runner plan does not match readiness plan")
        spec = self.commands.node_specs[0]
        command = [
            str(
                plan.cluster.remote_repo_root
                / "venv"
                / "ilvr-venv"
                / "bin"
                / "python"
            ),
            "-m",
            "src.multinode",
            "ownership-ready",
            "--ownership-path",
            str(spec.ownership_path),
            "--run-id",
            plan.run_id,
            "--attempt-id",
            plan.attempt_id,
            "--machine-rank",
            "0",
            "--commit",
            self.commit_value,
            "--command-digest",
            self.commands.command_digests[0],
            "--master-addr",
            plan.cluster.master_addr,
            "--main-process-port",
            str(plan.main_process_port),
        ]
        for name, value in spec.environment:
            command.extend(("--env", f"{name}={value}"))
        command.extend(("--", *spec.command))
        completed = self.command_runner(
            build_remote_exec(command), input_bytes=None
        )
        return completed.returncode == 0

    def start_local(self, command: Sequence[str]) -> None:
        if tuple(command) != self.commands.local:
            raise ValueError("local tmux command differs from the immutable launch")
        self._run(command)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def read_ownership(self, machine_rank: int) -> OwnershipRecord:
        spec = self._node_spec(machine_rank)
        completed = self._run(
            self._internal_command(
                machine_rank,
                "ownership-show",
                ("--ownership-path", str(spec.ownership_path)),
            )
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("ownership-show returned invalid JSON") from error
        ownership = OwnershipRecord.from_json(payload)
        self._seen_ownership.add(machine_rank)
        return ownership

    def process_start_ticks(self, ownership: OwnershipRecord) -> int:
        payload = self._process_snapshot(ownership)
        value = payload.get("process_start_ticks")
        if type(value) is not int:
            raise ValueError("owned process start ticks are invalid")
        return value

    def command_digest(self, ownership: OwnershipRecord) -> str:
        payload = self._process_snapshot(ownership)
        environment = payload.get("environment")
        if type(environment) is not dict or not all(
            type(key) is str and type(value) is str
            for key, value in environment.items()
        ):
            raise ValueError("owned process environment is invalid")
        digest = environment.get("ILVR_OWNED_COMMAND_DIGEST")
        if type(digest) is not str:
            raise ValueError("owned process command digest is missing")
        return _require_digest("owned process command digest", digest)

    def commit(self, ownership: OwnershipRecord) -> str:
        payload = self._process_snapshot(ownership)
        environment = payload.get("environment")
        if type(environment) is not dict:
            raise ValueError("owned process environment is invalid")
        commit = environment.get("ILVR_REPOSITORY_COMMIT")
        if type(commit) is not str:
            raise ValueError("owned process commit is missing")
        return commit

    def signal(self, ownership: OwnershipRecord) -> None:
        spec = self._node_spec(ownership.machine_rank)
        arguments = [
            "--ownership-path",
            str(spec.ownership_path),
            "--run-id",
            ownership.run_id,
            "--attempt-id",
            ownership.attempt_id,
            "--machine-rank",
            str(ownership.machine_rank),
            "--pid",
            str(ownership.pid),
            "--process-start-ticks",
            str(ownership.process_start_ticks),
            "--commit",
            ownership.commit,
            "--command-digest",
            ownership.command_digest,
        ]
        for name, value in spec.environment:
            arguments.extend(("--env", f"{name}={value}"))
        arguments.extend(("--", *spec.command))
        self._run(
            self._internal_command(
                ownership.machine_rank, "signal-owned", arguments
            )
        )

    def poll_exit(self, machine_rank: int) -> int | None:
        spec = self._node_spec(machine_rank)
        arguments = (
            "--exit-path",
            str(spec.exit_path),
            "--run-id",
            self.plan.run_id,
            "--attempt-id",
            self.plan.attempt_id,
            "--machine-rank",
            str(machine_rank),
            "--commit",
            self.commit_value,
            "--command-digest",
            self.commands.command_digests[machine_rank],
        )
        completed = self.command_runner(
            self._internal_command(machine_rank, "exit-show", arguments),
            input_bytes=None,
        )
        if completed.returncode == 0:
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                raise ValueError("exit-show returned invalid JSON") from error
            exit_code = payload.get("exit_code") if type(payload) is dict else None
            if type(exit_code) is not int:
                raise ValueError("owned exit code is invalid")
            return exit_code
        if completed.returncode != 3:
            raise RuntimeError(f"exit-show failed: {completed.stderr.strip()}")
        try:
            ownership = self.read_ownership(machine_rank)
        except (RuntimeError, ValueError):
            if machine_rank not in self._seen_ownership:
                return None
            return 125
        try:
            self._process_snapshot(ownership)
        except (RuntimeError, ValueError):
            return 125
        return None


class CoordinatorRunner(Protocol):
    def prepare_plan(self, plan: "LaunchPlan") -> None: ...
    def start_remote(self, command: Sequence[str]) -> None: ...
    def rank0_is_ready(self, plan: "LaunchPlan") -> bool: ...
    def start_local(self, command: Sequence[str]) -> None: ...
    def sleep(self, seconds: float) -> None: ...


def start_rank0_then_rank1(
    plan: "LaunchPlan",
    runner: CoordinatorRunner,
    *,
    commands: OwnedTmuxCommands | None = None,
    commit: str = "0" * 40,
    stage_arguments: Sequence[str] = (),
    stage_environment: Mapping[str, str] | None = None,
    ready_timeout_seconds: float = 30,
    dry_run: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Start remote rank 0, poll its launch-specific readiness, then rank 1."""
    if not math.isfinite(ready_timeout_seconds) or ready_timeout_seconds <= 0:
        raise ValueError("ready_timeout_seconds must be positive and finite")
    commands = commands or build_owned_tmux_commands(
        plan,
        commit=commit,
        stage_arguments=stage_arguments,
        stage_environment=stage_environment,
    )
    result = (commands.remote, commands.local)
    if dry_run:
        return result
    runner.prepare_plan(plan)
    runner.start_remote(commands.remote)
    checks = max(1, math.ceil(ready_timeout_seconds / 0.25))
    deadline = time.monotonic() + ready_timeout_seconds
    for check in range(checks):
        if check and time.monotonic() >= deadline:
            break
        if runner.rank0_is_ready(plan):
            runner.start_local(commands.local)
            return result
        if check + 1 < checks and time.monotonic() < deadline:
            runner.sleep(0.25)
    raise TimeoutError("rank 0 readiness was not observed before timeout")


class OwnedProcessRunner(Protocol):
    def read_ownership(self, machine_rank: int) -> OwnershipRecord: ...
    def process_start_ticks(self, ownership: OwnershipRecord) -> int: ...
    def command_digest(self, ownership: OwnershipRecord) -> str: ...
    def commit(self, ownership: OwnershipRecord) -> str: ...
    def signal(self, ownership: OwnershipRecord) -> None: ...
    def poll_exit(self, machine_rank: int) -> int | None: ...
    def sleep(self, seconds: float) -> None: ...


def stop_owned_peer(record: OwnershipRecord, runner: OwnedProcessRunner) -> None:
    """Re-read and validate the live process immediately before signaling it."""
    current = runner.read_ownership(record.machine_rank)
    validate_ownership(
        current,
        expected_run_id=record.run_id,
        expected_attempt_id=record.attempt_id,
        expected_machine_rank=record.machine_rank,
        expected_pid=record.pid,
        expected_commit=record.commit,
        expected_command_digest=record.command_digest,
        observed_start_ticks=runner.process_start_ticks(current),
        observed_command_digest=runner.command_digest(current),
        observed_commit=runner.commit(current),
    )
    runner.signal(current)


def monitor_owned_jobs(
    plan: "LaunchPlan",
    runner: OwnedProcessRunner,
    *,
    expected_commit: str,
    expected_command_digests: Mapping[int, str],
    poll_interval_seconds: float = 0.25,
) -> dict[int, int]:
    """Collect both exits and stop only a revalidated live peer after failure."""
    exits: dict[int, int] = {}
    ranks = (plan.cluster.remote_rank, plan.cluster.local_rank)
    while len(exits) < len(ranks):
        for machine_rank in ranks:
            if machine_rank in exits:
                continue
            exit_code = runner.poll_exit(machine_rank)
            if exit_code is None:
                continue
            exits[machine_rank] = exit_code
            if exit_code != 0:
                for peer_rank in ranks:
                    if peer_rank == machine_rank or peer_rank in exits:
                        continue
                    peer = runner.read_ownership(peer_rank)
                    validate_ownership(
                        peer,
                        expected_run_id=plan.run_id,
                        expected_attempt_id=plan.attempt_id,
                        expected_machine_rank=peer_rank,
                        expected_pid=peer.pid,
                        expected_commit=expected_commit,
                        expected_command_digest=expected_command_digests[peer_rank],
                        observed_start_ticks=runner.process_start_ticks(peer),
                        observed_command_digest=runner.command_digest(peer),
                        observed_commit=runner.commit(peer),
                    )
                    stop_owned_peer(peer, runner)
                    exits[peer_rank] = -15
                return exits
        if len(exits) < len(ranks):
            runner.sleep(poll_interval_seconds)
    return exits


def _finite_loss_reference(path: Path | None) -> dict[str, object]:
    if path is None:
        return {
            "status": "pending-producer-evidence",
            "path": None,
            "finite_loss": None,
        }
    if path.is_symlink() or not path.is_file():
        raise ValueError("finite loss evidence must be a regular file")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("finite loss evidence is invalid JSON") from error
    evidence = payload.get("benchmark_evidence", payload) if type(payload) is dict else None
    if type(evidence) is not dict or evidence.get("finite_loss") is not True:
        raise ValueError("finite loss evidence does not prove finite loss")
    return {
        "status": "validated",
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "finite_loss": True,
    }


def write_training_witness(
    plan: "LaunchPlan",
    *,
    result_root: str | Path,
    exit_codes: Mapping[int, int],
    checkpoint_path: str | Path,
    batch_per_gpu: int,
    finite_loss_evidence_path: str | Path | None = None,
) -> Path:
    """Exclusively publish one attempt witness without inventing loss evidence."""
    if dict(exit_codes) != {0: 0, 1: 0}:
        raise ValueError("training witness requires exit 0 from both machine ranks")
    if type(batch_per_gpu) is not int or batch_per_gpu <= 0:
        raise ValueError("batch_per_gpu must be positive")
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("training witness checkpoint must be an existing directory")
    finite_loss = _finite_loss_reference(
        Path(finite_loss_evidence_path)
        if finite_loss_evidence_path is not None
        else None
    )
    run_directory = Path(result_root) / plan.run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    path = run_directory / f"{plan.attempt_id}-train-witness.json"
    payload = {
        "schema_version": _WITNESS_SCHEMA_VERSION,
        "run_id": plan.run_id,
        "attempt_id": plan.attempt_id,
        "stage": plan.stage,
        "method": plan.tuning_method,
        "zero_stage": plan.zero_stage,
        "batch_per_gpu": batch_per_gpu,
        "world_size": plan.world_size,
        "machine_ranks": [plan.cluster.remote_rank, plan.cluster.local_rank],
        "exit_codes": {str(rank): code for rank, code in sorted(exit_codes.items())},
        "checkpoint_path": str(checkpoint),
        "finite_loss_evidence": finite_loss,
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)
    return path


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


def _build_remote_stdin_exec(argv: Sequence[str]) -> list[str]:
    """Build the fixed remote container boundary with stdin attached."""
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        _REMOTE_HOST,
        "docker",
        "exec",
        "-i",
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
