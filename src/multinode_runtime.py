"""Pure validation and command construction for the fixed ILVR cluster."""

from __future__ import annotations

import json
import hashlib
import ipaddress
import math
import os
import re
import secrets
import shlex
import signal
import stat
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, Sequence

from src.artifact_paths import CANONICAL_EVALUATION_STAGING_ROOT
from src.deepspeed_runtime import load_deepspeed_runtime_profile

if TYPE_CHECKING:
    from src.multinode import LaunchPlan


_REMOTE_HOST = "korea@10.1.102.41"
_REMOTE_CONTAINER = "llm_kd"
_REQUIRED_FREE_BYTES = 100 * 1024**3
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CHECKPOINT_ROOT = _REPOSITORY_ROOT / "checkpoints"
_EVALUATION_STAGING_ROOT = CANONICAL_EVALUATION_STAGING_ROOT
_RESULT_ROOT = _REPOSITORY_ROOT / "results"
_LOG_ROOT = _REPOSITORY_ROOT / "logs" / "multinode"
_STAGE_LOG_ROOT = _REPOSITORY_ROOT / "logs" / "zero-stage-compatibility"
_OWNERSHIP_SCHEMA_VERSION = 1
_WITNESS_SCHEMA_VERSION = 1
_INCOMPLETE_SCHEMA_VERSION = 1
_CONTROL_TIMEOUT_SECONDS = 30
_ARTIFACT_TRANSFER_TIMEOUT_SECONDS = 60 * 60
_MAX_TRANSFER_BYTES = 1024 * 1024
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,127}$")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MINIMAL_EXEC_ENVIRONMENT = {
    "HOME": "/root",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": (
        f"{_REPOSITORY_ROOT / 'venv/ilvr-venv/bin'}:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PYTHONUNBUFFERED": "1",
}
_STAGE2_FIXED_EXEC_ENVIRONMENT = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _trusted_absolute_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or any(part in ("", ".", "..") for part in candidate.parts[1:]):
        raise ValueError("trusted publication path must be absolute without traversal")
    return candidate


def _open_trusted_directory(
    path: str | Path, *, create: bool = False, mode: int = 0o700
) -> int:
    """Open an absolute directory chain without following any symlink ancestor."""
    directory = _trusted_absolute_path(path)
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in directory.parts[1:]:
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=mode, dir_fd=descriptor)
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as error:
                if error.errno in (getattr(os, "ELOOP", 40), 20):
                    raise ValueError("trusted directory chain contains a symlink") from error
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_child_directory(
    parent_descriptor: int,
    name: str,
    *,
    create: bool,
    exclusive: bool = False,
    mode: int = 0o700,
) -> int:
    if not name or name in (".", "..") or "/" in name:
        raise ValueError("directory component is invalid")
    if create:
        try:
            os.mkdir(name, mode=mode, dir_fd=parent_descriptor)
        except FileExistsError:
            if exclusive:
                raise
    return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)


def _publish_bytes_no_replace(path: str | Path, payload: bytes) -> Path:
    """Atomically publish bytes via a trusted dirfd and an exclusive hard link."""
    destination = _trusted_absolute_path(path)
    parent_descriptor = _open_trusted_directory(destination.parent, create=True)
    temporary_name = f".{destination.name}.{secrets.token_hex(12)}.tmp"
    file_descriptor: int | None = None
    try:
        file_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=parent_descriptor,
        )
        os.fchmod(file_descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(file_descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing attempt record")
            view = view[written:]
        os.fsync(file_descriptor)
        os.close(file_descriptor)
        file_descriptor = None
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace attempt record: {destination}") from error
        os.fsync(parent_descriptor)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)
    return destination


def publish_small_file(path: str | Path, payload: bytes) -> Path:
    """Publish bounded control metadata through the trusted no-replace path."""
    if not isinstance(payload, bytes):
        raise TypeError("small publication payload must be bytes")
    if len(payload) > _MAX_TRANSFER_BYTES:
        raise ValueError("small publication payload exceeds the size limit")
    return _publish_bytes_no_replace(path, payload)


def _read_regular_file(path: str | Path, *, maximum_size: int = _MAX_TRANSFER_BYTES) -> bytes:
    source = _trusted_absolute_path(path)
    parent_descriptor = _open_trusted_directory(source.parent)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            source.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("attempt record must be a regular file")
        if metadata.st_size > maximum_size:
            raise ValueError("attempt record exceeds the size limit")
        chunks: list[bytes] = []
        remaining = maximum_size + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > maximum_size:
            raise ValueError("attempt record exceeds the size limit")
        return raw
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


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
    payload = _canonical_json_bytes(record.to_json()) + b"\n"
    return _publish_bytes_no_replace(path, payload)


def read_ownership(path: str | Path) -> OwnershipRecord:
    return OwnershipRecord.from_json(json.loads(_read_regular_file(path)))


def _proc_start_ticks(pid: int, *, proc_root: str | Path = "/proc") -> int:
    return _proc_process_identity(pid, proc_root=proc_root)[1]


def _proc_process_identity(
    pid: int, *, proc_root: str | Path = "/proc"
) -> tuple[int, int]:
    """Read one stat snapshot as (parent PID, process start ticks)."""
    stat_text = (Path(proc_root) / str(pid) / "stat").read_text(encoding="utf-8")
    closing_parenthesis = stat_text.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("process stat has no command terminator")
    trailing_fields = stat_text[closing_parenthesis + 2 :].split()
    if len(trailing_fields) <= 19:
        raise ValueError("process stat is missing identity fields")
    try:
        parent_pid = int(trailing_fields[1])
        start_ticks = int(trailing_fields[19])
    except ValueError as error:
        raise ValueError("process stat identity is not an integer") from error
    if parent_pid < 0 or start_ticks <= 0:
        raise ValueError("process stat identity values are invalid")
    return parent_pid, start_ticks


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
    expected_argv, expected_pairs = _validated_command(
        expected_command, expected_environment
    )
    if observed_command != expected_argv:
        raise ValueError("live ownership command mismatch")
    raw_environment = _nul_fields(process_directory / "environ")
    observed_environment: dict[str, str] = {}
    for assignment in raw_environment:
        name, separator, value = assignment.partition("=")
        if not separator or name in observed_environment:
            raise ValueError("live ownership environment is malformed")
        observed_environment[name] = value
    expected_environment_dict = dict(expected_pairs)
    expected_digest = canonical_command_digest(expected_command, expected_environment)
    if observed_environment != expected_environment_dict:
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
        observed_command_digest=expected_digest,
        observed_commit=observed_commit,
    )


@dataclass(frozen=True)
class _PinnedProcess:
    pid: int
    pidfd: int
    parent_pid: int
    start_ticks: int
    close_pidfd: bool


def _process_children(pid: int, *, proc_root: Path) -> tuple[int, ...]:
    tasks = tuple((proc_root / str(pid) / "task").iterdir())
    children: set[int] = set()
    for task in tasks:
        values = (task / "children").read_text(encoding="ascii").split()
        for value in values:
            try:
                child_pid = int(value)
            except ValueError as error:
                raise ValueError("process children list contains a non-PID") from error
            if child_pid <= 1:
                raise ValueError("process children list contains an invalid PID")
            children.add(child_pid)
    return tuple(sorted(children))


def _pin_process_tree(
    owner_pid: int,
    *,
    proc_root: Path,
    owner_pidfd: int | None,
    expected_owner_start_ticks: int | None,
) -> dict[int, _PinnedProcess]:
    if type(owner_pid) is not int or owner_pid <= 1:
        raise ValueError("owner PID must identify a non-init process")
    root_pidfd = os.pidfd_open(owner_pid) if owner_pidfd is None else owner_pidfd
    pinned: dict[int, _PinnedProcess] = {}
    try:
        parent_pid, start_ticks = _proc_process_identity(owner_pid, proc_root=proc_root)
        if (
            expected_owner_start_ticks is not None
            and start_ticks != expected_owner_start_ticks
        ):
            raise ValueError("owner process start ticks changed")
        pinned[owner_pid] = _PinnedProcess(
            pid=owner_pid,
            pidfd=root_pidfd,
            parent_pid=parent_pid,
            start_ticks=start_ticks,
            close_pidfd=owner_pidfd is None,
        )
        pending = [owner_pid]
        while pending:
            parent = pinned[pending.pop(0)]
            for child_pid in _process_children(parent.pid, proc_root=proc_root):
                if child_pid in pinned:
                    if pinned[child_pid].parent_pid != parent.pid:
                        raise ValueError("pinned process ancestry changed")
                    continue
                child_pidfd = os.pidfd_open(child_pid)
                try:
                    child_parent, child_start = _proc_process_identity(
                        child_pid, proc_root=proc_root
                    )
                    if child_parent != parent.pid:
                        raise ValueError("descendant parent is not pinned")
                    pinned[child_pid] = _PinnedProcess(
                        pid=child_pid,
                        pidfd=child_pidfd,
                        parent_pid=child_parent,
                        start_ticks=child_start,
                        close_pidfd=True,
                    )
                except BaseException:
                    os.close(child_pidfd)
                    raise
                pending.append(child_pid)
        return pinned
    except BaseException:
        for process in pinned.values():
            if process.close_pidfd:
                os.close(process.pidfd)
        if not pinned and owner_pidfd is None:
            os.close(root_pidfd)
        raise


def _listening_socket_inodes(
    *, master_addr: str, port: int, proc_root: str | Path = "/proc"
) -> set[str]:
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("readiness port must be between 1024 and 65535")
    expected_port = f"{port:04X}"
    address = ipaddress.ip_address(master_addr)
    if not isinstance(address, ipaddress.IPv4Address):
        raise ValueError("readiness master address must be IPv4")
    expected_ipv4 = address.packed[::-1].hex().upper()
    inodes: set[str] = set()
    for table_name in ("tcp", "tcp6"):
        table = Path(proc_root) / "net" / table_name
        try:
            lines = table.read_text(encoding="ascii").splitlines()[1:]
        except FileNotFoundError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10:
                continue
            local = fields[1].rsplit(":", 1)
            if len(local) != 2 or local[1].upper() != expected_port:
                continue
            local_address = local[0].upper()
            if table_name == "tcp":
                if local_address not in ("00000000", expected_ipv4):
                    continue
            elif local_address != "0" * 32 and not local_address.endswith(expected_ipv4):
                continue
            if fields[3].upper() == "0A":
                inodes.add(fields[9])
    return inodes


def owned_listener_is_ready(
    owner_pid: int,
    *,
    master_addr: str,
    port: int,
    proc_root: str | Path = "/proc",
    owner_pidfd: int | None = None,
    expected_owner_start_ticks: int | None = None,
) -> bool:
    """Prove that the requested LISTEN socket belongs to the owned process tree."""
    if master_addr != "10.1.102.41":
        raise ValueError("readiness master address must match the fixed topology")
    listening = _listening_socket_inodes(
        master_addr=master_addr, port=port, proc_root=proc_root
    )
    if not listening:
        return False
    root = Path(proc_root)
    pinned: dict[int, _PinnedProcess] = {}
    try:
        pinned = _pin_process_tree(
            owner_pid,
            proc_root=root,
            owner_pidfd=owner_pidfd,
            expected_owner_start_ticks=expected_owner_start_ticks,
        )
        found_listener = False
        for process in pinned.values():
            entries = tuple((root / str(process.pid) / "fd").iterdir())
            for entry in entries:
                target = os.readlink(entry)
                if target.startswith("socket:[") and target.endswith("]"):
                    found_listener = found_listener or target[8:-1] in listening
        for process in pinned.values():
            parent_pid, start_ticks = _proc_process_identity(
                process.pid, proc_root=root
            )
            if (parent_pid, start_ticks) != (
                process.parent_pid,
                process.start_ticks,
            ):
                return False
            if process.pid != owner_pid and parent_pid not in pinned:
                return False
            signal.pidfd_send_signal(process.pidfd, 0)
        return found_listener
    except (OSError, ValueError, ProcessLookupError):
        return False
    finally:
        for process in pinned.values():
            if process.close_pidfd:
                os.close(process.pidfd)


def validate_resume_path(
    run_id: str,
    resume_path: str | Path,
    *,
    checkpoint_root: str | Path = _CHECKPOINT_ROOT,
) -> Path:
    """Bind resume to the immediate logical checkpoint path shared by both nodes."""
    _require_identifier("run_id", run_id)
    root = Path(checkpoint_root)
    candidate = Path(resume_path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("resume checkpoint must be an absolute canonical path")
    if re.fullmatch(r"checkpoint-(0|[1-9][0-9]*)", candidate.name) is None:
        raise ValueError("resume checkpoint must be an immediate checkpoint-N")
    expected = root / run_id / candidate.name
    if candidate.absolute() != expected.absolute():
        raise ValueError(
            "resume checkpoint must be the immediate checkpoint-N for the same run ID"
        )
    return expected.absolute()


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
    root = _trusted_absolute_path(log_root)
    root_descriptor = _open_trusted_directory(root, create=True)
    run_descriptor: int | None = None
    attempt_descriptor: int | None = None
    try:
        run_descriptor = _open_child_directory(
            root_descriptor, plan.run_id, create=True
        )
        attempt_descriptor = _open_child_directory(
            run_descriptor, plan.attempt_id, create=True, exclusive=True
        )
    finally:
        if attempt_descriptor is not None:
            os.close(attempt_descriptor)
        if run_descriptor is not None:
            os.close(run_descriptor)
        os.close(root_descriptor)
    attempt_directory = root / plan.run_id / plan.attempt_id
    path = attempt_directory / "launch-plan.json"
    payload = _canonical_json_bytes(_json_ready(plan)) + b"\n"
    _publish_bytes_no_replace(path, payload)
    return SerializedLaunchPlan(
        path=path,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def prepare_attempt_directory(
    *, log_root: str | Path, run_id: str, attempt_id: str
) -> Path:
    """Create a fresh attempt directory through trusted directory descriptors."""
    _require_identifier("run_id", run_id)
    _require_identifier("attempt_id", attempt_id)
    root = _trusted_absolute_path(log_root)
    root_descriptor = _open_trusted_directory(root, create=True)
    run_descriptor: int | None = None
    attempt_descriptor: int | None = None
    try:
        run_descriptor = _open_child_directory(root_descriptor, run_id, create=True)
        attempt_descriptor = _open_child_directory(
            run_descriptor, attempt_id, create=True, exclusive=True
        )
    finally:
        if attempt_descriptor is not None:
            os.close(attempt_descriptor)
        if run_descriptor is not None:
            os.close(run_descriptor)
        os.close(root_descriptor)
    return root / run_id / attempt_id


def receive_launch_plan(
    *,
    path: str | Path,
    raw: bytes,
    expected_size: int,
    expected_sha256: str,
) -> Path:
    """Validate and exclusively publish the one permitted stdin transfer."""
    if type(expected_size) is not int or not 0 < expected_size <= _MAX_TRANSFER_BYTES:
        raise ValueError("launch plan transfer size is invalid")
    _require_digest("launch plan transfer digest", expected_sha256)
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("launch plan transfer mismatch")
    destination = _publish_bytes_no_replace(path, raw)
    stored = _read_regular_file(destination)
    if len(stored) != expected_size or hashlib.sha256(stored).hexdigest() != expected_sha256:
        raise ValueError("stored launch plan mismatch")
    return destination


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
    "--ema-tau",
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
    "--training-evidence",
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
    "CE_WEIGHT",
    "GRADIENT_ACCUMULATION_STEPS",
    "ATTENTION_IMPLEMENTATION",
    "VISION_ATTENTION_IMPLEMENTATION",
    "GRADIENT_CHECKPOINTING",
    "MAX_SEQUENCE_LENGTH",
    "DATALOADER_NUM_WORKERS",
    "SEED",
    "STEP_PREFERENCE_ENABLED",
    "STEP_PREFERENCE_MODE",
    "STEP_PREFERENCE_OBJECTIVE",
    "STEP_PREFERENCE_TARGET_SCOPE",
    "STEP_PREFERENCE_BETA",
    "STEP_PREFERENCE_WEIGHT",
    "STEP_PREFERENCE_MARGIN",
    "STEP_VISUAL_ONLY_CONTEXT_PREFERENCE",
    "STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES",
    "STAGE2B_DIRECT_FROM_STAGE1",
    "STAGE1_ADAPTER_PATH",
    "STAGE2A_CHECKPOINT_PATH",
    "STEP_NEGATIVE_PATH",
    "STEP_NEGATIVE_MANIFEST",
    "STEP_DPO_BETA",
    "STEP_DPO_WEIGHT",
    "STEP_ENTROPY_WEIGHT",
    "STEP_ENTROPY_MARGIN",
    "STAGE2A_RATIO",
    "STAGE2_TOTAL_STEPS",
    "STEP_SHARED_PREFIX_ENABLED",
    "STEP_SIDECAR_ATTENTION_BACKEND",
    "STEP_SIDECAR_QUERY_BUDGET",
    "RESUME_CHECKPOINT",
    "BENCHMARK_EVIDENCE_PATH",
}
_STAGE2_REQUIRED_ENVIRONMENT = {"STAGE1_CHECKPOINT", "DATA_PATH", "IMAGE_ROOT"}


def validated_stage2_environment(
    source: Mapping[str, str],
    *,
    output_dir: str,
    log_file: str,
    deepspeed_config: str,
    benchmark_evidence_path: str | None = None,
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
    if benchmark_evidence_path is not None:
        evidence = Path(benchmark_evidence_path)
        if not evidence.is_absolute() or ".." in evidence.parts:
            raise ValueError("Stage 2 producer evidence path must be absolute")
        result["BENCHMARK_EVIDENCE_PATH"] = str(evidence)
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
    "--save_steps",
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
    validated = _validate_allowlisted_arguments(
        arguments,
        value_options=_STAGE2_VALUE_OPTIONS,
        boolean_options=_STAGE2_BOOLEAN_OPTIONS,
        owned_options=_STAGE2_COORDINATOR_OPTIONS,
        label="Stage 2",
    )
    save_steps: list[str] = []
    index = 0
    while index < len(validated):
        option, separator, inline_value = validated[index].partition("=")
        if option in _STAGE2_BOOLEAN_OPTIONS:
            index += 1
            continue
        value = inline_value if separator else validated[index + 1]
        if option == "--save_steps":
            save_steps.append(value)
        index += 1 if separator else 2
    if len(save_steps) > 1:
        raise ValueError("--save_steps may be provided exactly once")
    if save_steps and re.fullmatch(r"[1-9][0-9]*", save_steps[0]) is None:
        raise ValueError("--save_steps must be a canonical positive integer")
    return validated


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


@dataclass(frozen=True)
class EvaluationAssignment:
    """One fixed independent evaluator process and its owned control names."""

    node: str
    machine_rank: int
    gpu: int
    shard_index: int
    session_name: str
    ownership_name: str
    exit_name: str


@dataclass(frozen=True)
class EvaluationPlan:
    """Immutable user inputs for one four-worker evaluation attempt."""

    run_id: str
    model_dir: Path
    test_data_path: Path
    task_name: str
    output_dir: Path
    attempt_id: str = "initial"
    base_model_dir: Path | None = None
    cache_dir: Path = _REPOSITORY_ROOT / "cache"
    max_new_tokens: int = 4096
    max_sequence_length: int = 32768
    repetition_penalty: float = 1.05
    temperature: float = 0.0
    top_p: float = 1.0
    vision_ablation: str = "none"
    attempt_timeout_seconds: int = 24 * 60 * 60

    def __post_init__(self) -> None:
        from src.multinode_artifacts import validate_coordinator_relative_path

        _require_identifier("run_id", self.run_id)
        _require_identifier("attempt_id", self.attempt_id)
        for label, path in (
            ("model_dir", self.model_dir),
            ("test_data_path", self.test_data_path),
            ("output_dir", self.output_dir),
            ("cache_dir", self.cache_dir),
        ):
            if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{label} must be an absolute path without traversal")
        if self.base_model_dir is not None and (
            not isinstance(self.base_model_dir, Path)
            or not self.base_model_dir.is_absolute()
            or ".." in self.base_model_dir.parts
        ):
            raise ValueError("base_model_dir must be an absolute path without traversal")
        try:
            relative_output = self.output_dir.relative_to(_RESULT_ROOT)
        except ValueError as error:
            raise ValueError("output_dir must be under the canonical result root") from error
        if not relative_output.parts or relative_output.parts[-1] != self.run_id:
            raise ValueError("output_dir must belong to the evaluation run ID")
        validate_coordinator_relative_path(
            relative_output, label="evaluation output"
        )
        try:
            relative_model = self.model_dir.relative_to(
                _EVALUATION_STAGING_ROOT
            )
        except ValueError as error:
            raise ValueError(
                "model_dir must be a Task 6 artifact under the canonical "
                "checkpoint evaluation-staging root"
            ) from error
        validate_coordinator_relative_path(
            relative_model, label="prepared inference model"
        )
        if (
            not isinstance(self.task_name, str)
            or not self.task_name.strip()
            or "\0" in self.task_name
        ):
            raise ValueError("task_name must be a non-empty string without NUL")
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if (
            type(self.max_sequence_length) is not int
            or self.max_sequence_length <= 0
        ):
            raise ValueError("max_sequence_length must be positive")
        if (
            type(self.repetition_penalty) not in (int, float)
            or not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty <= 0
        ):
            raise ValueError("repetition_penalty must be finite and positive")
        if (
            type(self.temperature) not in (int, float)
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be finite and nonnegative")
        if (
            type(self.top_p) not in (int, float)
            or not math.isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be finite and in (0, 1]")
        if self.vision_ablation not in ("none", "all", "interleaved"):
            raise ValueError("vision_ablation must be none, all, or interleaved")
        if type(self.attempt_timeout_seconds) is not int or self.attempt_timeout_seconds <= 0:
            raise ValueError("attempt_timeout_seconds must be positive")

    @property
    def local_model_dir(self) -> Path:
        return _EVALUATION_STAGING_ROOT / self.run_id / "local-model"


@dataclass(frozen=True)
class EvaluationWorkerCommand:
    assignment: EvaluationAssignment
    launch: tuple[str, ...]
    command_digest: str
    node_spec: OwnedNodeCommand


@dataclass(frozen=True)
class EvaluationCommands:
    workers: tuple[EvaluationWorkerCommand, ...]
    model_identity: str
    shard_jsonls: tuple[Path, ...]
    shard_manifests: tuple[Path, ...]
    remote_files_path: Path
    remote_files_payload: bytes
    remote_pull: tuple[str, ...]
    merged_jsonl: Path
    summary_json: Path


def build_evaluation_assignments(
    *, run_id: str, attempt_id: str = "initial"
) -> tuple[EvaluationAssignment, ...]:
    """Assign remote GPUs to shards 0/1 and local GPUs to shards 2/3."""
    _require_identifier("run_id", run_id)
    _require_identifier("attempt_id", attempt_id)
    layout = (
        ("remote", 0, 0, 0),
        ("remote", 0, 1, 1),
        ("local", 1, 0, 2),
        ("local", 1, 1, 3),
    )
    return tuple(
        EvaluationAssignment(
            node=node,
            machine_rank=machine_rank,
            gpu=gpu,
            shard_index=shard,
            session_name=f"ilvr-{run_id}-{attempt_id}-eval-shard{shard}",
            ownership_name=f"ownership-eval-shard{shard}.json",
            exit_name=f"exit-eval-shard{shard}.json",
        )
        for node, machine_rank, gpu, shard in layout
    )


def _build_evaluation_commands_for_stage(
    plan: EvaluationPlan,
    *,
    stage: str,
    model_identity: str,
    commit: str,
    log_root: Path = _LOG_ROOT,
) -> EvaluationCommands:
    from src.multinode_artifacts import (
        build_evaluation_pull_command,
        evaluation_result_files_bytes,
    )

    if stage not in ("stage1", "stage2"):
        raise ValueError("observed inference stage must be stage1 or stage2")
    if stage == "stage1" and plan.vision_ablation != "none":
        raise ValueError("vision ablation is only supported for Stage 2 evaluation")
    _require_digest("model_identity", model_identity.removeprefix("task6:"))
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("commit must be a lowercase 40-character Git object ID")
    repository = _REPOSITORY_ROOT
    evaluator = repository / ("eval.py" if stage == "stage1" else "src/evaluate_deepseed.py")
    python = repository / "venv/ilvr-venv/bin/python"
    attempt_directory = log_root / plan.run_id / plan.attempt_id
    assignments = build_evaluation_assignments(
        run_id=plan.run_id, attempt_id=plan.attempt_id
    )
    workers: list[EvaluationWorkerCommand] = []
    shard_jsonls = tuple(plan.output_dir / f"shard-{index}.jsonl" for index in range(4))
    for assignment in assignments:
        model_dir = plan.model_dir if assignment.node == "remote" else plan.local_model_dir
        evaluator_command = [
            str(python),
            str(evaluator),
            "--model_dir", str(model_dir),
            "--test_data_path", str(plan.test_data_path),
            "--task_name", plan.task_name,
            "--output_json_path", str(shard_jsonls[assignment.shard_index]),
            "--cache_dir", str(plan.cache_dir),
            "--max_new_tokens", str(plan.max_new_tokens),
            "--max_sequence_length", str(plan.max_sequence_length),
            "--repetition_penalty", str(plan.repetition_penalty),
            "--temperature", str(plan.temperature),
            "--top_p", str(plan.top_p),
        ]
        if plan.base_model_dir is not None:
            evaluator_command.extend(("--base_model_dir", str(plan.base_model_dir)))
        if stage == "stage2":
            evaluator_command.extend(("--vision-ablation", plan.vision_ablation))
        evaluator_command.extend(
            (
                "--model-identity", model_identity,
                "--num-shards", "4",
                "--shard-index", str(assignment.shard_index),
            )
        )
        environment = {
            **_MINIMAL_EXEC_ENVIRONMENT,
            "CUDA_VISIBLE_DEVICES": str(assignment.gpu),
            "ILVR_REPOSITORY_COMMIT": commit,
            "PYTHONPATH": (
                f"{repository / 'transformers/src'}:{repository}:{repository / 'src'}"
            ),
        }
        ownership_path = attempt_directory / assignment.ownership_name
        exit_path = attempt_directory / assignment.exit_name
        supervisor = [
            str(python),
            "-m", "src.multinode", "node-run",
            "--run-id", plan.run_id,
            "--attempt-id", plan.attempt_id,
            "--machine-rank", str(assignment.machine_rank),
            "--evaluation-shard-index", str(assignment.shard_index),
            "--commit", commit,
            "--ownership-path", str(ownership_path),
            "--exit-path", str(exit_path),
        ]
        for name, value in sorted(environment.items()):
            supervisor.extend(("--env", f"{name}={value}"))
        supervisor.extend(("--", *evaluator_command))
        owned_command = tuple(supervisor)
        digest = canonical_command_digest(owned_command, environment)
        isolated = (
            "/usr/bin/env", "-i",
            *(f"{name}={value}" for name, value in sorted(environment.items())),
            *owned_command,
        )
        tmux = (
            "/usr/bin/tmux", "new-session", "-d", "-s",
            assignment.session_name, "--", *isolated,
        )
        launch = tuple(build_remote_exec(tmux)) if assignment.node == "remote" else tmux
        workers.append(
            EvaluationWorkerCommand(
                assignment=assignment,
                launch=launch,
                command_digest=digest,
                node_spec=OwnedNodeCommand(
                    machine_rank=assignment.machine_rank,
                    command=owned_command,
                    environment=tuple(sorted(environment.items())),
                    ownership_path=ownership_path,
                    exit_path=exit_path,
                ),
            )
        )
    shard_manifests = tuple(Path(f"{path}.manifest.json") for path in shard_jsonls)
    remote_files_path = attempt_directory / "evaluation-pull.files"
    return EvaluationCommands(
        workers=tuple(workers),
        model_identity=model_identity,
        shard_jsonls=shard_jsonls,
        shard_manifests=shard_manifests,
        remote_files_path=remote_files_path,
        remote_files_payload=evaluation_result_files_bytes(),
        remote_pull=build_evaluation_pull_command(
            plan.output_dir, files_from=remote_files_path
        ),
        merged_jsonl=plan.output_dir / "merged.jsonl",
        summary_json=plan.output_dir / "summary.json",
    )


def build_evaluation_commands(
    plan: EvaluationPlan,
    *,
    inference_manifest: object,
    commit: str,
) -> EvaluationCommands:
    """Build four immutable owned workers from a Task 6 validated identity."""
    from src.multinode_artifacts import (
        NodeArtifactManifest,
        inference_model_identity,
        validate_inference_manifest_identity,
    )

    if not isinstance(plan, EvaluationPlan):
        raise TypeError("plan must be an EvaluationPlan")
    if not isinstance(inference_manifest, NodeArtifactManifest):
        raise TypeError("inference_manifest must be a NodeArtifactManifest")
    validate_inference_manifest_identity(
        inference_manifest, expected_identity=inference_manifest.runtime_identity
    )
    return _build_evaluation_commands_for_stage(
        plan,
        stage=str(inference_manifest.runtime_identity["stage"]),
        model_identity=inference_model_identity(inference_manifest),
        commit=commit,
    )


class EvaluationRunner(Protocol):
    def prepare_inference(self, plan: EvaluationPlan) -> object: ...
    def repository_commit(self) -> str: ...
    def start_worker(self, worker: EvaluationWorkerCommand) -> None: ...
    def poll_worker(self, worker: EvaluationWorkerCommand) -> int | None: ...
    def cleanup_worker(self, worker: EvaluationWorkerCommand) -> bool: ...
    def pull_remote_results(self, commands: EvaluationCommands) -> None: ...
    def validate_sidecars(self, commands: EvaluationCommands) -> None: ...
    def merge(self, commands: EvaluationCommands) -> Mapping[str, object]: ...
    def sleep(self, seconds: float) -> None: ...


def run_evaluation(
    plan: EvaluationPlan,
    runner: EvaluationRunner,
    *,
    poll_interval_seconds: float = 0.25,
) -> Mapping[str, object]:
    """Prepare, own, monitor, transfer, validate, and merge four evaluators."""
    manifest = runner.prepare_inference(plan)
    commands = build_evaluation_commands(
        plan, inference_manifest=manifest, commit=runner.repository_commit()
    )
    possibly_live: dict[int, EvaluationWorkerCommand] = {}
    try:
        for worker in commands.workers:
            possibly_live[worker.assignment.shard_index] = worker
            runner.start_worker(worker)
        exits: dict[int, int] = {}
        deadline = time.monotonic() + plan.attempt_timeout_seconds
        while possibly_live:
            for shard_index, worker in tuple(possibly_live.items()):
                exit_code = runner.poll_worker(worker)
                if exit_code is None:
                    continue
                exits[shard_index] = exit_code
                del possibly_live[shard_index]
                if exit_code != 0:
                    raise RuntimeError(
                        "all four evaluation workers must exit zero before merge"
                    )
            if possibly_live:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "all four evaluation workers must complete before timeout"
                    )
                runner.sleep(poll_interval_seconds)
    except BaseException as primary_error:
        cleanup_errors: list[tuple[int, BaseException]] = []
        for worker in possibly_live.values():
            try:
                runner.cleanup_worker(worker)
            except BaseException as cleanup_error:
                cleanup_errors.append(
                    (worker.assignment.shard_index, cleanup_error)
                )
        for shard_index, cleanup_error in cleanup_errors:
            primary_error.add_note(
                f"evaluation cleanup failed for shard {shard_index}: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise
    if exits != {0: 0, 1: 0, 2: 0, 3: 0}:
        raise RuntimeError("all four evaluation workers must exit zero before merge")
    runner.pull_remote_results(commands)
    try:
        runner.validate_sidecars(commands)
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            "all four evaluation workers require valid successful sidecars"
        ) from error
    return runner.merge(commands)


def _node_shell_command(
    plan: "LaunchPlan",
    *,
    machine_rank: int,
    commit: str,
    stage_command: tuple[str, ...],
    environment: Mapping[str, str],
    log_root: Path,
    require_bash_launcher: bool = True,
) -> tuple[tuple[str, ...], str, OwnedNodeCommand]:
    repository = plan.cluster.remote_repo_root
    attempt_directory = log_root / plan.run_id / plan.attempt_id
    ownership_path = attempt_directory / f"ownership-rank{machine_rank}.json"
    exit_path = attempt_directory / f"exit-rank{machine_rank}.json"
    if require_bash_launcher and (
        len(stage_command) < 2 or stage_command[0] != "/usr/bin/bash"
    ):
        raise ValueError("Stage launcher must be an explicit /usr/bin/bash command")
    if not require_bash_launcher and not stage_command:
        raise ValueError("owned lifecycle harness command must be nonempty")
    supervisor_command = [
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
        "--exit-path",
        str(exit_path),
    ]
    for name, value in sorted(environment.items()):
        supervisor_command.extend(("--env", f"{name}={value}"))
    supervisor_command.extend(("--", *stage_command))
    owned_command = tuple(supervisor_command)
    digest = canonical_command_digest(owned_command, environment)
    isolated_supervisor = (
        "/usr/bin/env",
        "-i",
        *(f"{name}={value}" for name, value in sorted(environment.items())),
        *owned_command,
    )
    return (
        isolated_supervisor,
        digest,
        OwnedNodeCommand(
            machine_rank=machine_rank,
            command=owned_command,
            environment=tuple(sorted(environment.items())),
            ownership_path=ownership_path,
            exit_path=exit_path,
        ),
    )


def build_owned_lifecycle_harness_commands(
    plan: "LaunchPlan",
    *,
    commit: str,
    phase: str | None = None,
    harness_path: str | Path | None = None,
    harness_arguments: Sequence[str] | None = None,
    log_root: str | Path = _LOG_ROOT,
) -> OwnedTmuxCommands:
    """Build one exact two-node Accelerate job for an approved tiny harness."""
    from src.multinode import LaunchPlan

    if not isinstance(plan, LaunchPlan):
        raise TypeError("plan must be a LaunchPlan")
    selected_phase = plan.attempt_id if phase is None else phase
    if selected_phase not in ("initial", "resume"):
        raise ValueError("lifecycle harness phase must be initial or resume")
    if plan.attempt_id != selected_phase:
        raise ValueError("lifecycle harness phase must match the attempt ID")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("commit must be a lowercase 40-character Git object ID")
    repository = plan.cluster.remote_repo_root
    expected_harness = repository / (
        "tests/stage1/zero_compat_diagnostic.py"
        if plan.stage == "stage1"
        else "tests/stage2/integration/tiny_stage2_train.py"
    )
    selected_harness = expected_harness if harness_path is None else Path(harness_path)
    if selected_harness != expected_harness:
        raise ValueError("lifecycle command must use the approved harness path")
    method_flag = "--method" if plan.stage == "stage1" else "--tuning-method"
    expected_arguments = (
        "--phase",
        selected_phase,
        "--zero-stage",
        str(plan.zero_stage),
        method_flag,
        plan.tuning_method,
        "--deepspeed",
        str(plan.deepspeed_config),
        "--experiment-id",
        plan.run_id,
        "--initial-steps",
        "2",
        "--resumed-steps",
        "3",
    )
    selected_arguments = (
        expected_arguments
        if harness_arguments is None
        else tuple(harness_arguments)
    )
    if selected_arguments != expected_arguments:
        raise ValueError("lifecycle command requires the exact harness arguments")

    node_commands: list[tuple[str, ...]] = []
    digests: list[str] = []
    node_specs: list[OwnedNodeCommand] = []
    for machine_rank in (plan.cluster.remote_rank, plan.cluster.local_rank):
        environment = {
            **_MINIMAL_EXEC_ENVIRONMENT,
            **_distributed_environment(plan, machine_rank),
            "MASTER_ADDR": plan.cluster.master_addr,
            "GLOO_SOCKET_IFNAME": plan.cluster.socket_ifname,
            "NCCL_SOCKET_IFNAME": plan.cluster.socket_ifname,
            "PYTHONPATH": (
                f"{repository / 'transformers/src'}:{repository}:{repository / 'src'}"
            ),
            "ILVR_REPOSITORY_COMMIT": commit,
        }
        harness_command = (
            str(repository / "venv/ilvr-venv/bin/accelerate"),
            "launch",
            "--num_machines",
            str(plan.cluster.num_machines),
            "--num_processes",
            str(plan.cluster.num_processes),
            "--machine_rank",
            str(machine_rank),
            "--main_process_ip",
            plan.cluster.master_addr,
            "--main_process_port",
            str(plan.main_process_port),
            "--mixed_precision",
            "bf16",
            "--use_deepspeed",
            "--deepspeed_multinode_launcher",
            "standard",
            "--deepspeed_config_file",
            str(plan.deepspeed_config),
            str(selected_harness),
            *selected_arguments,
        )
        supervisor, digest, node_spec = _node_shell_command(
            plan,
            machine_rank=machine_rank,
            commit=commit,
            stage_command=harness_command,
            environment=environment,
            log_root=Path(log_root),
            require_bash_launcher=False,
        )
        session = f"ilvr-{plan.run_id}-{plan.attempt_id}-rank{machine_rank}"
        tmux = (
            "/usr/bin/tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "--",
            *supervisor,
        )
        node_commands.append(
            tuple(build_remote_exec(tmux)) if machine_rank == 0 else tmux
        )
        digests.append(digest)
        node_specs.append(node_spec)
    return OwnedTmuxCommands(
        remote=node_commands[0],
        local=node_commands[1],
        command_digests=(digests[0], digests[1]),
        node_specs=(node_specs[0], node_specs[1]),
        finite_loss_evidence_path=None,
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
    supplied_environment = dict(stage_environment or {})
    if plan.stage == "stage1":
        if supplied_environment:
            raise ValueError("Stage 1 does not accept coordinator environment overrides")
        base_environment: dict[str, str] = {}
        forwarded = validate_stage1_arguments(stage_arguments)
        stage_owned_arguments: tuple[str, ...] = ()
        finite_loss_evidence_path = (
            Path(result_root)
            / plan.run_id
            / f"{plan.attempt_id}-producer-evidence.json"
        )
        if any(
            argument == "--benchmark-run-config"
            or argument.startswith("--benchmark-run-config=")
            for argument in forwarded
        ):
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
            stage_owned_arguments = (
                "--training-evidence",
                str(finite_loss_evidence_path),
            )
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
        forwarded = validate_stage2_arguments(stage_arguments)
        approved_stage2_names = _STAGE2_ENVIRONMENT_ALLOWLIST | {
            "OUTPUT_DIR",
            "LOG_FILE",
            "DEEPSPEED_CONFIG",
        }
        unexpected = sorted(set(supplied_environment) - approved_stage2_names)
        if unexpected:
            raise ValueError(
                f"unapproved Stage 2 environment: {', '.join(unexpected)}"
            )
        missing = sorted(
            name for name in _STAGE2_REQUIRED_ENVIRONMENT if not supplied_environment.get(name)
        )
        if missing:
            raise ValueError(f"missing required Stage 2 environment: {', '.join(missing)}")
        base_environment = supplied_environment
        evidence_value = supplied_environment.get("BENCHMARK_EVIDENCE_PATH")
        finite_loss_evidence_path = (
            Path(evidence_value) if evidence_value is not None else None
        )
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
            **_MINIMAL_EXEC_ENVIRONMENT,
            **(_STAGE2_FIXED_EXEC_ENVIRONMENT if plan.stage == "stage2" else {}),
            **base_environment,
            **_distributed_environment(plan, machine_rank),
            "PYTHONPATH": f"{repository}:{repository / 'src'}",
            "ILVR_REPOSITORY_COMMIT": commit,
        }
        supervisor, digest, node_spec = _node_shell_command(
            plan,
            machine_rank=machine_rank,
            commit=commit,
            stage_command=stage_command,
            environment=environment,
            log_root=Path(log_root),
        )
        session = f"ilvr-{plan.run_id}-{plan.attempt_id}-rank{machine_rank}"
        node_commands.append(
            (
                "/usr/bin/tmux",
                "new-session",
                "-d",
                "-s",
                session,
                "--",
                *supervisor,
            )
        )
        digests.append(digest)
        node_specs.append(node_spec)
    return OwnedTmuxCommands(
        remote=tuple(
            build_remote_exec(
                node_commands[0], working_directory=repository
            )
        ),
        local=node_commands[1],
        command_digests=(digests[0], digests[1]),
        node_specs=(node_specs[0], node_specs[1]),
        finite_loss_evidence_path=finite_loss_evidence_path,
    )


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _subprocess_command_runner(
    command: Sequence[str],
    *,
    input_bytes: bytes | None = None,
    timeout_seconds: float = _CONTROL_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    if input_bytes is None:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    completed = subprocess.run(
        command,
        input=input_bytes,
        capture_output=True,
        text=False,
        check=False,
        timeout=timeout_seconds,
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
        commands: OwnedTmuxCommands | None,
        commit: str,
        log_root: str | Path = _LOG_ROOT,
        checkpoint_root: str | Path = _CHECKPOINT_ROOT,
        command_runner: CommandRunner = _subprocess_command_runner,
    ) -> None:
        self.plan = plan
        self.commands = commands
        self.commit_value = commit
        self.log_root = Path(log_root)
        self.checkpoint_root = Path(checkpoint_root)
        if self.checkpoint_root not in (
            _CHECKPOINT_ROOT,
            _CHECKPOINT_ROOT / "zero-stage-compatibility",
        ):
            raise ValueError("coordinator checkpoint root is not approved")
        self.command_runner = command_runner
        self.serialized_plan: SerializedLaunchPlan | None = None
        self._seen_ownership: set[int] = set()
        self._artifact_checkpoint_paths: dict[int, Path] = {}
        self._reserved_transfer_destinations: dict[Path, Path] = {}

    def _run(
        self, command: Sequence[str], *, input_bytes: bytes | None = None
    ) -> subprocess.CompletedProcess[str]:
        completed = self.command_runner(
            command,
            input_bytes=input_bytes,
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"coordinator command failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
        return completed

    def _run_artifact_transfer(
        self, command: Sequence[str]
    ) -> subprocess.CompletedProcess[str]:
        completed = self.command_runner(
            command,
            input_bytes=None,
            timeout_seconds=_ARTIFACT_TRANSFER_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"coordinator artifact transfer failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
        return completed

    def _node_spec(self, machine_rank: int) -> OwnedNodeCommand:
        if machine_rank not in (0, 1):
            raise ValueError("machine_rank must be 0 or 1")
        commands = self._launch_commands()
        spec = commands.node_specs[machine_rank]
        if spec.machine_rank != machine_rank:
            raise ValueError("owned node command rank mismatch")
        return spec

    def _launch_commands(self) -> OwnedTmuxCommands:
        if self.commands is None:
            raise RuntimeError("transfer-only coordinator cannot launch workers")
        return self.commands

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
        if machine_rank == self.plan.cluster.remote_rank:
            return _remote_repository_module_command(subcommand, arguments)
        if machine_rank == self.plan.cluster.local_rank:
            return _local_repository_module_command(subcommand, arguments)
        raise ValueError("machine_rank must be 0 or 1")

    def _process_snapshot(self, ownership: OwnershipRecord) -> dict[str, object]:
        arguments = ["--pid", str(ownership.pid)]
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
        self._run(
            _remote_repository_module_command(
                "prepare-attempt",
                (
                    "--log-root",
                    str(self.log_root),
                    "--run-id",
                    plan.run_id,
                    "--attempt-id",
                    plan.attempt_id,
                ),
            )
        )
        remote_path = self.log_root / plan.run_id / plan.attempt_id / "launch-plan.json"
        self._run(
            _remote_repository_module_stdin_command(
                "receive-plan",
                (
                    "--path",
                    str(remote_path),
                    "--size",
                    str(serialized.size),
                    "--sha256",
                    serialized.sha256,
                ),
            ),
            input_bytes=_read_regular_file(serialized.path),
        )
        self.serialized_plan = serialized

    def start_remote(self, command: Sequence[str]) -> None:
        if tuple(command) != self._launch_commands().remote:
            raise ValueError("remote tmux command differs from the immutable launch")
        self._run(command)

    def rank0_is_ready(self, plan: "LaunchPlan") -> bool:
        if plan != self.plan:
            raise ValueError("runner plan does not match readiness plan")
        commands = self._launch_commands()
        spec = commands.node_specs[0]
        arguments = [
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
            commands.command_digests[0],
            "--master-addr",
            plan.cluster.master_addr,
            "--main-process-port",
            str(plan.main_process_port),
        ]
        for name, value in spec.environment:
            arguments.extend(("--env", f"{name}={value}"))
        arguments.extend(("--", *spec.command))
        completed = self.command_runner(
            _remote_repository_module_command("ownership-ready", arguments),
            input_bytes=None,
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
        )
        return completed.returncode == 0

    def start_local(self, command: Sequence[str]) -> None:
        if tuple(command) != self._launch_commands().local:
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
        command = payload.get("command")
        environment = payload.get("environment")
        if type(command) is not list or not all(type(value) is str for value in command):
            raise ValueError("owned process command is invalid")
        if type(environment) is not dict or not all(
            type(key) is str and type(value) is str
            for key, value in environment.items()
        ):
            raise ValueError("owned process environment is invalid")
        spec = self._node_spec(ownership.machine_rank)
        if tuple(command) != spec.command or environment != dict(spec.environment):
            raise ValueError("owned process argv or environment mismatch")
        return canonical_command_digest(command, environment)

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
        commands = self._launch_commands()
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
            commands.command_digests[machine_rank],
        )
        exit_command = self._internal_command(machine_rank, "exit-show", arguments)

        def decode_exit(completed: subprocess.CompletedProcess[str]) -> int | None:
            if completed.returncode == 3:
                return None
            if completed.returncode != 0:
                raise RuntimeError(f"exit-show failed: {completed.stderr.strip()}")
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                raise ValueError("exit-show returned invalid JSON") from error
            exit_code = payload.get("exit_code") if type(payload) is dict else None
            if type(exit_code) is not int:
                raise ValueError("owned exit code is invalid")
            return exit_code

        def retry_exit_after_process_disappears() -> int:
            completed = self.command_runner(
                exit_command,
                input_bytes=None,
                timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
            )
            try:
                retry_exit_code = decode_exit(completed)
            except (RuntimeError, ValueError):
                return 125
            return 125 if retry_exit_code is None else retry_exit_code

        completed = self.command_runner(
            exit_command,
            input_bytes=None,
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
        )
        exit_code = decode_exit(completed)
        if exit_code is not None:
            return exit_code
        try:
            ownership = self.read_ownership(machine_rank)
        except (RuntimeError, ValueError):
            if machine_rank not in self._seen_ownership:
                return None
            return retry_exit_after_process_disappears()
        try:
            self.command_digest(ownership)
        except (RuntimeError, ValueError):
            return retry_exit_after_process_disappears()
        return None

    def read_lifecycle_phase_evidence(
        self, machine_rank: int, phase: str
    ) -> bytes:
        """Read one fixed canonical node-local phase witness without path input."""
        if phase not in ("initial", "resume") or self.plan.attempt_id != phase:
            raise ValueError("lifecycle evidence phase must match the owned attempt")
        path = (
            self.plan.cluster.remote_repo_root
            / "results"
            / self.plan.run_id
            / f"{phase}-witness.json"
        )
        if machine_rank == self.plan.cluster.local_rank:
            return _read_regular_file(path, maximum_size=1024 * 1024)
        repository = _REPOSITORY_ROOT
        completed = self._run(
            build_remote_exec(
                (
                    "env",
                    f"PYTHONPATH={repository}:{repository / 'src'}",
                    str(repository / "venv/ilvr-venv/bin/python"),
                    "-m",
                    "src.multinode",
                    "lifecycle-evidence-show",
                    "--run-id",
                    self.plan.run_id,
                    "--phase",
                    phase,
                ),
                working_directory=repository,
            )
        )
        raw = completed.stdout.encode("utf-8")
        if not raw or len(raw) > 1024 * 1024:
            raise ValueError("remote lifecycle evidence size is invalid")
        return raw

    def read_stage1_training_evidence(self, path: Path) -> bytes:
        """Read the fixed remote ordinary Stage 1 producer after owned exit zero."""
        expected = self._launch_commands().finite_loss_evidence_path
        if self.plan.stage != "stage1" or expected is None or path != expected:
            raise ValueError("Stage 1 training evidence path differs from the launch")
        repository = _REPOSITORY_ROOT
        completed = self._run(
            build_remote_exec(
                (
                    "env",
                    f"PYTHONPATH={repository}:{repository / 'src'}",
                    str(repository / "venv/ilvr-venv/bin/python"),
                    "-m",
                    "src.multinode",
                    "stage1-training-evidence-show",
                    "--path",
                    str(path),
                    "--run-id",
                    self.plan.run_id,
                    "--attempt-id",
                    self.plan.attempt_id,
                ),
                working_directory=repository,
            )
        )
        raw = completed.stdout.encode("utf-8")
        if not raw or len(raw) > 1024 * 1024:
            raise ValueError("remote Stage 1 training evidence size is invalid")
        return raw

    def cleanup_owned(self, machine_rank: int, *, timeout_seconds: float) -> bool:
        """Signal only an exactly revalidated owner, then boundedly observe exit."""
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("cleanup timeout must be positive and finite")
        try:
            ownership = self.read_ownership(machine_rank)
            spec = self._node_spec(machine_rank)
            validate_ownership(
                ownership,
                expected_run_id=self.plan.run_id,
                expected_attempt_id=self.plan.attempt_id,
                expected_machine_rank=machine_rank,
                expected_pid=ownership.pid,
                expected_commit=self.commit_value,
                expected_command_digest=(
                    self._launch_commands().command_digests[machine_rank]
                ),
                observed_start_ticks=self.process_start_ticks(ownership),
                observed_command_digest=self.command_digest(ownership),
                observed_commit=self.commit(ownership),
            )
            if ownership.command_digest != canonical_command_digest(
                spec.command, dict(spec.environment)
            ):
                raise ValueError("owned cleanup command digest mismatch")
            stop_owned_peer(ownership, self)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            return False
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            exit_code = self.poll_exit(machine_rank)
            if exit_code is not None:
                return True
            self.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        return False

    def read_artifact_manifest(
        self,
        machine_rank: int,
        checkpoint_path: Path,
        runtime_identity: Mapping[str, object],
        *,
        canonical_owner: bool = False,
    ) -> object:
        """Collect one bounded node-local lstat manifest through the node CLI."""
        from src.multinode_artifacts import (
            NodeArtifactManifest,
            validate_runtime_identity,
        )

        identity = validate_runtime_identity(runtime_identity)
        if (
            identity["run_id"] != self.plan.run_id
            or identity["stage"] != self.plan.stage
            or identity["tuning_method"] != self.plan.tuning_method
            or identity["zero_stage"] != self.plan.zero_stage
            or identity["world_size"] != self.plan.world_size
        ):
            raise ValueError("artifact identity differs from the coordinator plan")
        expected_checkpoint = (
            self.checkpoint_root
            / self.plan.run_id
            / f"checkpoint-{identity['global_step']}"
        )
        if checkpoint_path != expected_checkpoint:
            raise ValueError("artifact manifest path is not the canonical checkpoint")
        ranks = (0, 1, 2, 3) if canonical_owner else (
            (0, 1) if machine_rank == 0 else (2, 3)
        )
        arguments = [
            "--run-directory",
            str(checkpoint_path),
            "--run-id",
            self.plan.run_id,
            "--checkpoint-root",
            str(self.checkpoint_root),
            "--node-rank",
            str(machine_rank),
            "--runtime-identity-json",
            json.dumps(
                runtime_identity,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "--completion-marker",
        ]
        for rank in ranks:
            arguments.extend(("--process-rank", str(rank)))
        if canonical_owner:
            arguments.append("--canonical-owner")
        completed = self._run(
            self._internal_command(machine_rank, "artifact-manifest", arguments)
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("artifact-manifest returned invalid JSON") from error
        manifest = NodeArtifactManifest.from_json(payload)
        if manifest.node_rank != machine_rank or manifest.run_id != self.plan.run_id:
            raise ValueError("artifact manifest differs from requested node identity")
        self._artifact_checkpoint_paths[machine_rank] = checkpoint_path
        return manifest

    def read_canonical_artifact_manifest(
        self,
        checkpoint_path: Path,
        runtime_identity: Mapping[str, object],
    ) -> object:
        """Rebuild the canonical owner manifest after rsync for exact comparison."""
        from src.multinode_artifacts import (
            NodeArtifactManifest,
            validate_runtime_identity,
        )

        identity = validate_runtime_identity(runtime_identity)
        if identity["run_id"] != self.plan.run_id:
            raise ValueError("canonical artifact identity run ID mismatch")
        expected_checkpoint = (
            self.checkpoint_root
            / self.plan.run_id
            / f"checkpoint-{identity['global_step']}"
        )
        if checkpoint_path != expected_checkpoint:
            raise ValueError("canonical artifact path is not the fixed checkpoint")

        arguments = [
            "--run-directory",
            str(checkpoint_path),
            "--run-id",
            self.plan.run_id,
            "--checkpoint-root",
            str(self.checkpoint_root),
            "--node-rank",
            "0",
            "--runtime-identity-json",
            json.dumps(runtime_identity, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "--completion-marker",
            "--canonical-owner",
        ]
        for rank in (0, 1, 2, 3):
            arguments.extend(("--process-rank", str(rank)))
        completed = self._run(
            self._internal_command(0, "artifact-manifest", arguments)
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("canonical artifact manifest returned invalid JSON") from error
        manifest = NodeArtifactManifest.from_json(payload)
        if (
            manifest.node_rank != 0
            or manifest.process_ranks != (0, 1, 2, 3)
            or manifest.run_id != self.plan.run_id
        ):
            raise ValueError("canonical artifact manifest owner identity mismatch")
        return manifest

    def read_inference_artifact_manifest(
        self,
        checkpoint_path: Path,
        runtime_identity: Mapping[str, object],
    ) -> object:
        """Ask rank 0 to validate and filter one complete inference artifact."""
        from src.multinode_artifacts import (
            NodeArtifactManifest,
            validate_inference_manifest_identity,
            validate_runtime_identity,
        )

        identity = validate_runtime_identity(runtime_identity)
        expected = (
            self.checkpoint_root
            / self.plan.run_id
            / f"checkpoint-{identity['global_step']}"
        )
        if checkpoint_path != expected or identity["run_id"] != self.plan.run_id:
            raise ValueError("inference checkpoint is not the fixed canonical path")
        arguments = (
            "--run-directory",
            str(checkpoint_path),
            "--run-id",
            self.plan.run_id,
            "--checkpoint-root",
            str(self.checkpoint_root),
            "--runtime-identity-json",
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "--completion-marker",
        )
        completed = self._run(
            self._internal_command(0, "artifact-inference-manifest", arguments)
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValueError(
                "inference artifact manifest returned invalid JSON"
            ) from error
        manifest = NodeArtifactManifest.from_json(payload)
        validate_inference_manifest_identity(manifest, expected_identity=identity)
        return manifest

    def validate_push_destination(
        self, manifest: object, entries: Sequence[object]
    ) -> None:
        """Ask the canonical owner to prove every selected destination is fresh."""
        from src.multinode_artifacts import ArtifactEntry, NodeArtifactManifest

        if not isinstance(manifest, NodeArtifactManifest) or manifest.node_rank != 1:
            raise ValueError("push validation requires the local node manifest")
        if not all(isinstance(entry, ArtifactEntry) for entry in entries):
            raise TypeError("push entries must be ArtifactEntry values")
        destination = self._artifact_checkpoint_paths.get(0)
        if destination is None:
            raise ValueError("remote artifact manifest must be collected first")
        payload = {
            "entries": [entry.to_json() for entry in entries],
            "runtime_identity": manifest.runtime_identity,
        }
        encoded = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        if len(encoded) > _MAX_TRANSFER_BYTES:
            raise ValueError("artifact destination request exceeds the size limit")
        arguments = (
            "--run-directory",
            str(destination),
            "--run-id",
            self.plan.run_id,
            "--checkpoint-root",
            str(self.checkpoint_root),
        )
        self._run(
            _remote_repository_module_stdin_command(
                "artifact-destination-check", arguments
            ),
            input_bytes=encoded,
        )

    def push_artifacts(
        self, manifest: object, entries: Sequence[object]
    ) -> None:
        """Push only the prevalidated local manifest list; never move sources."""
        from src.multinode_artifacts import (
            ArtifactEntry,
            NodeArtifactManifest,
            build_push_command,
            publish_files_list,
        )

        if not isinstance(manifest, NodeArtifactManifest) or manifest.node_rank != 1:
            raise ValueError("artifact push requires the local node manifest")
        if not all(isinstance(entry, ArtifactEntry) for entry in entries):
            raise TypeError("push entries must be ArtifactEntry values")
        files_path = (
            self.log_root
            / self.plan.run_id
            / self.plan.attempt_id
            / "artifact-push-rank1.files"
        )
        publish_files_list(files_path, entries)  # type: ignore[arg-type]
        self._run_artifact_transfer(
            build_push_command(manifest, files_from=files_path)
        )

    def validate_resume_destination(
        self, entries: Sequence[object], destination: Path
    ) -> None:
        """Reserve one fresh local checkpoint directory before redistribution."""
        from src.multinode_artifacts import (
            ArtifactEntry,
            validate_checkpoint_directory,
            validate_transfer_destination,
        )

        if not all(isinstance(entry, ArtifactEntry) for entry in entries):
            raise TypeError("resume entries must be ArtifactEntry values")
        match = re.fullmatch(r"checkpoint-(0|[1-9][0-9]*)", destination.name)
        if match is None:
            raise ValueError("resume destination must be immediate checkpoint-N")
        physical = validate_checkpoint_directory(
            destination,
            checkpoint_root=self.checkpoint_root,
            run_id=self.plan.run_id,
            global_step=int(match.group(1)),
        )
        validate_transfer_destination(
            tuple(physical / entry.relative_path for entry in entries)  # type: ignore[attr-defined]
        )
        self._reserved_transfer_destinations[destination] = physical

    def pull_resume_artifacts(
        self, manifest: object, entries: Sequence[object], destination: Path
    ) -> None:
        """Pull the exact rank-2/3 resume list, then validate local lstat metadata."""
        from src.multinode_artifacts import (
            ArtifactEntry,
            NodeArtifactManifest,
            build_pull_command,
            publish_files_list,
            validate_transfer_destination,
            validate_transferred_entry_subset,
            resolve_local_checkpoint_destination,
        )

        if not isinstance(manifest, NodeArtifactManifest):
            raise TypeError("resume manifest must be a NodeArtifactManifest")
        if not all(isinstance(entry, ArtifactEntry) for entry in entries):
            raise TypeError("resume entries must be ArtifactEntry values")
        physical = self._reserved_transfer_destinations.get(destination)
        if (
            physical is None
            or physical != resolve_local_checkpoint_destination(destination)
        ):
            raise ValueError("resume destination reservation changed before transfer")
        validate_transfer_destination(
            tuple(physical / entry.relative_path for entry in entries)  # type: ignore[attr-defined]
        )
        files_path = (
            self.log_root
            / self.plan.run_id
            / self.plan.attempt_id
            / "artifact-resume-rank1.files"
        )
        publish_files_list(files_path, entries)  # type: ignore[arg-type]
        self._run_artifact_transfer(
            build_pull_command(manifest, physical, files_from=files_path)
        )
        validate_transferred_entry_subset(physical, entries)  # type: ignore[arg-type]

    def validate_inference_manifest(self, manifest: object) -> None:
        """Bind a complete canonical manifest to this fixed run and identity."""
        from src.multinode_artifacts import (
            NodeArtifactManifest,
            validate_inference_manifest_identity,
        )

        if not isinstance(manifest, NodeArtifactManifest):
            raise TypeError("inference manifest must be a NodeArtifactManifest")
        if manifest.run_id != self.plan.run_id:
            raise ValueError("inference manifest run ID mismatch")
        validate_inference_manifest_identity(
            manifest, expected_identity=manifest.runtime_identity
        )

    def pull_inference_artifacts(
        self, manifest: object, destination: Path
    ) -> None:
        """Pull only allowlisted inference files into the reserved directory."""
        from src.multinode_artifacts import (
            NodeArtifactManifest,
            build_pull_command,
            publish_files_list,
            select_inference_entries,
            validate_reserved_inference_destination,
            validate_transferred_entries,
        )

        if not isinstance(manifest, NodeArtifactManifest):
            raise TypeError("inference manifest must be a NodeArtifactManifest")
        entries = select_inference_entries(manifest)
        validate_reserved_inference_destination(destination)
        files_path = (
            self.log_root
            / self.plan.run_id
            / self.plan.attempt_id
            / "artifact-inference.files"
        )
        publish_files_list(files_path, entries)
        self._run_artifact_transfer(
            build_pull_command(manifest, destination, files_from=files_path)
        )
        validate_transferred_entries(destination, entries)

    def mirror_prepared_inference(
        self, manifest: object, destination: Path
    ) -> object:
        """Publish and revalidate the identical prepared artifact remotely."""
        from src.multinode_artifacts import (
            NodeArtifactManifest,
            build_prepared_inference_push_command,
            prepared_inference_mirror_files_bytes,
            validate_inference_manifest_identity,
        )

        if not isinstance(manifest, NodeArtifactManifest):
            raise TypeError("inference manifest must be a NodeArtifactManifest")
        validate_inference_manifest_identity(
            manifest, expected_identity=manifest.runtime_identity
        )
        files_path = (
            self.log_root
            / self.plan.run_id
            / self.plan.attempt_id
            / "artifact-inference-mirror.files"
        )
        publish_small_file(
            files_path,
            prepared_inference_mirror_files_bytes(manifest, destination),
        )
        repository = self.plan.cluster.remote_repo_root
        python = repository / "venv" / "ilvr-venv" / "bin" / "python"

        def remote_repository_command(
            subcommand: str, arguments: Sequence[str]
        ) -> tuple[str, ...]:
            return tuple(build_remote_exec(
                (
                    "/usr/bin/env",
                    f"PYTHONPATH={repository}:{repository / 'src'}",
                    str(python),
                    "-m",
                    "src.multinode",
                    subcommand,
                    *arguments,
                ),
                working_directory=repository,
            ))

        self._run(remote_repository_command(
            "prepare-inference-destination",
            ("--destination", str(destination)),
        ))
        self._run_artifact_transfer(
            build_prepared_inference_push_command(
                manifest, destination, files_from=files_path
            )
        )
        shown = self._run(remote_repository_command(
            "prepared-inference-show",
            ("--model-dir", str(destination)),
        ))
        try:
            remote_manifest = NodeArtifactManifest.from_json(json.loads(shown.stdout))
        except json.JSONDecodeError as error:
            raise ValueError("remote prepared inference manifest is invalid JSON") from error
        validate_inference_manifest_identity(
            remote_manifest, expected_identity=manifest.runtime_identity
        )
        if remote_manifest != manifest:
            raise ValueError("remote prepared inference manifest differs from local")
        return remote_manifest


class EvaluationSubprocessRunner:
    """Execute the fixed four-worker evaluation through injectable subprocesses."""

    def __init__(
        self,
        plan: EvaluationPlan,
        *,
        commit: str,
        log_root: str | Path = _LOG_ROOT,
        command_runner: CommandRunner = _subprocess_command_runner,
        attempt_preparer: Callable[..., Path] = prepare_attempt_directory,
    ) -> None:
        if not isinstance(plan, EvaluationPlan):
            raise TypeError("plan must be an EvaluationPlan")
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError("commit must be a lowercase 40-character Git object ID")
        self.plan = plan
        self.commit_value = commit
        self.log_root = Path(log_root)
        self.command_runner = command_runner
        self.attempt_preparer = attempt_preparer
        self._manifest: object | None = None
        self._started: dict[int, EvaluationWorkerCommand] = {}
        self._controls_prepared = False

    def _run(
        self, command: Sequence[str], *, input_bytes: bytes | None = None
    ) -> subprocess.CompletedProcess[str]:
        completed = self.command_runner(
            command,
            input_bytes=input_bytes,
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"evaluation coordinator command failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
        return completed

    def _run_artifact_transfer(
        self, command: Sequence[str]
    ) -> subprocess.CompletedProcess[str]:
        completed = self.command_runner(
            command,
            input_bytes=None,
            timeout_seconds=_ARTIFACT_TRANSFER_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"evaluation artifact transfer failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
        return completed

    def _node_command(
        self, worker: EvaluationWorkerCommand, arguments: Sequence[str]
    ) -> tuple[str, ...]:
        if worker.assignment.node == "remote":
            return tuple(build_remote_exec(arguments))
        if worker.assignment.node == "local":
            return tuple(arguments)
        raise ValueError("evaluation worker node is invalid")

    def _internal_command(
        self,
        worker: EvaluationWorkerCommand,
        subcommand: str,
        arguments: Sequence[str],
    ) -> tuple[str, ...]:
        if worker.assignment.node == "remote":
            return _remote_repository_module_command(subcommand, arguments)
        if worker.assignment.node == "local":
            return _local_repository_module_command(subcommand, arguments)
        raise ValueError("evaluation worker node is invalid")

    def repository_commit(self) -> str:
        return self.commit_value

    def prepare_control_namespaces(self) -> None:
        """Exclusively reserve local then remote Task 5 attempt directories."""
        if self._controls_prepared:
            raise RuntimeError("evaluation control namespaces are already prepared")
        self.attempt_preparer(
            log_root=self.log_root,
            run_id=self.plan.run_id,
            attempt_id=self.plan.attempt_id,
        )
        self._run(
            _remote_repository_module_command(
                "prepare-attempt",
                (
                    "--log-root",
                    str(self.log_root),
                    "--run-id",
                    self.plan.run_id,
                    "--attempt-id",
                    self.plan.attempt_id,
                ),
            )
        )
        self._controls_prepared = True

    def prepare_inference(self, plan: EvaluationPlan) -> object:
        """Validate the remote Task 6 artifact and make one fresh verified copy."""
        from src.multinode_artifacts import (
            NodeArtifactManifest,
            prepare_inference_copy,
            reserve_inference_destination,
            resolve_local_checkpoint_destination,
            validate_inference_manifest_identity,
        )

        if plan != self.plan:
            raise ValueError("runner evaluation plan mismatch")
        shown = self._run(
            _remote_repository_module_command(
                "prepared-inference-show",
                (
                    "--model-dir",
                    str(plan.model_dir),
                ),
            )
        )
        try:
            payload = json.loads(shown.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("prepared inference manifest returned invalid JSON") from error
        manifest = NodeArtifactManifest.from_json(payload)
        validate_inference_manifest_identity(
            manifest, expected_identity=manifest.runtime_identity
        )
        self.prepare_control_namespaces()
        reserve_inference_destination(plan.output_dir)
        self._manifest = manifest
        physical_local_model = resolve_local_checkpoint_destination(
            plan.local_model_dir
        )
        prepare_inference_copy(manifest, physical_local_model, self)
        self._run(
            _remote_repository_module_command(
                "prepare-evaluation-output",
                (
                    "--output-dir",
                    str(plan.output_dir),
                    "--run-id",
                    plan.run_id,
                ),
            )
        )
        return manifest

    def validate_inference_manifest(self, manifest: object) -> None:
        from src.multinode_artifacts import (
            NodeArtifactManifest,
            validate_inference_manifest_identity,
        )

        if not isinstance(manifest, NodeArtifactManifest):
            raise TypeError("inference manifest must be a NodeArtifactManifest")
        if self._manifest is not None and manifest != self._manifest:
            raise ValueError("inference manifest changed before local copy")
        validate_inference_manifest_identity(
            manifest, expected_identity=manifest.runtime_identity
        )

    def pull_inference_artifacts(self, manifest: object, destination: Path) -> None:
        from src.multinode_artifacts import (
            NodeArtifactManifest,
            build_prepared_inference_pull_command,
            publish_files_list,
            select_inference_entries,
            validate_reserved_inference_destination,
            validate_transferred_entries,
        )

        if not isinstance(manifest, NodeArtifactManifest):
            raise TypeError("inference manifest must be a NodeArtifactManifest")
        if manifest != self._manifest:
            raise ValueError("inference manifest changed before transfer")
        entries = select_inference_entries(manifest)
        validate_reserved_inference_destination(destination)
        files_path = (
            self.log_root
            / self.plan.run_id
            / self.plan.attempt_id
            / "evaluation-inference.files"
        )
        publish_files_list(files_path, entries)
        self._run_artifact_transfer(
            build_prepared_inference_pull_command(
                manifest,
                self.plan.model_dir,
                destination,
                files_from=files_path,
            )
        )
        validate_transferred_entries(destination, entries)

    def start_worker(self, worker: EvaluationWorkerCommand) -> None:
        if not self._controls_prepared:
            raise RuntimeError("evaluation control namespaces are not prepared")
        shard = worker.assignment.shard_index
        if shard in self._started:
            raise ValueError("evaluation worker was already started")
        self._run(worker.launch)
        self._started[shard] = worker

    def poll_worker(self, worker: EvaluationWorkerCommand) -> int | None:
        shard = worker.assignment.shard_index
        if self._started.get(shard) != worker:
            raise ValueError("evaluation worker is not the exact started command")
        spec = worker.node_spec
        arguments = (
            "--exit-path", str(spec.exit_path),
            "--run-id", self.plan.run_id,
            "--attempt-id", self.plan.attempt_id,
            "--machine-rank", str(worker.assignment.machine_rank),
            "--commit", self.commit_value,
            "--command-digest", worker.command_digest,
        )
        completed = self.command_runner(
            self._internal_command(worker, "exit-show", arguments),
            input_bytes=None,
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
        )
        if completed.returncode == 3:
            return None
        if completed.returncode != 0:
            raise RuntimeError(f"evaluation exit-show failed: {completed.stderr.strip()}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("evaluation exit record returned invalid JSON") from error
        exit_code = payload.get("exit_code") if type(payload) is dict else None
        if type(exit_code) is not int:
            raise ValueError("evaluation worker exit code is invalid")
        return exit_code

    def cleanup_worker(self, worker: EvaluationWorkerCommand) -> bool:
        """Ask node-side pidfd validation to signal only the exact live owner."""
        spec = worker.node_spec
        shown = self.command_runner(
            self._internal_command(
                worker,
                "ownership-show",
                ("--ownership-path", str(spec.ownership_path)),
            ),
            input_bytes=None,
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
        )
        if shown.returncode != 0:
            return False
        try:
            ownership = OwnershipRecord.from_json(json.loads(shown.stdout))
            validate_ownership(
                ownership,
                expected_run_id=self.plan.run_id,
                expected_attempt_id=self.plan.attempt_id,
                expected_machine_rank=worker.assignment.machine_rank,
                expected_pid=ownership.pid,
                expected_commit=self.commit_value,
                expected_command_digest=worker.command_digest,
                observed_start_ticks=ownership.process_start_ticks,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        arguments = [
            "--ownership-path", str(spec.ownership_path),
            "--run-id", ownership.run_id,
            "--attempt-id", ownership.attempt_id,
            "--machine-rank", str(ownership.machine_rank),
            "--pid", str(ownership.pid),
            "--process-start-ticks", str(ownership.process_start_ticks),
            "--commit", ownership.commit,
            "--command-digest", ownership.command_digest,
        ]
        for name, value in spec.environment:
            arguments.extend(("--env", f"{name}={value}"))
        arguments.extend(("--", *spec.command))
        signaled = self.command_runner(
            self._internal_command(worker, "signal-owned", arguments),
            input_bytes=None,
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
        )
        if signaled.returncode != 0:
            raise RuntimeError(
                f"evaluation signal-owned failed ({signaled.returncode}): "
                f"{signaled.stderr.strip()}"
            )
        return True

    def pull_remote_results(self, commands: EvaluationCommands) -> None:
        from src.multinode_runtime import publish_small_file

        publish_small_file(commands.remote_files_path, commands.remote_files_payload)
        self._run(commands.remote_pull)

    def validate_sidecars(self, commands: EvaluationCommands) -> None:
        from src.multinode_artifacts import validate_evaluation_sidecars

        validate_evaluation_sidecars(
            commands.shard_manifests,
            commands.shard_jsonls,
            expected_model_identity=commands.model_identity,
            expected_sample_count=self._expected_sample_count(),
            expected_generation_config={
                "max_new_tokens": self.plan.max_new_tokens,
                "max_sequence_length": self.plan.max_sequence_length,
                "repetition_penalty": self.plan.repetition_penalty,
                "temperature": self.plan.temperature,
                "top_p": self.plan.top_p,
                "vision_ablation": self.plan.vision_ablation,
            },
        )

    def _expected_sample_count(self) -> int:
        expected_sample_count = 0
        with self.plan.test_data_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    expected_sample_count += 1
        return expected_sample_count

    def merge(self, commands: EvaluationCommands) -> Mapping[str, object]:
        from src.multinode_evaluation import merge_shards

        return merge_shards(
            commands.shard_manifests,
            expected_sample_count=self._expected_sample_count(),
            output_jsonl=commands.merged_jsonl,
            summary_json=commands.summary_json,
        )

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class CoordinatorRunner(Protocol):
    def prepare_plan(self, plan: "LaunchPlan") -> None: ...
    def start_remote(self, command: Sequence[str]) -> None: ...
    def rank0_is_ready(self, plan: "LaunchPlan") -> bool: ...
    def start_local(self, command: Sequence[str]) -> None: ...
    def sleep(self, seconds: float) -> None: ...
    def cleanup_owned(self, machine_rank: int, *, timeout_seconds: float) -> bool: ...


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
    possibly_live: list[int] = []
    try:
        possibly_live.append(plan.cluster.remote_rank)
        runner.start_remote(commands.remote)
        checks = max(1, math.ceil(ready_timeout_seconds / 0.25))
        deadline = time.monotonic() + ready_timeout_seconds
        for check in range(checks):
            if check and time.monotonic() >= deadline:
                break
            if runner.rank0_is_ready(plan):
                possibly_live.append(plan.cluster.local_rank)
                runner.start_local(commands.local)
                return result
            if check + 1 < checks and time.monotonic() < deadline:
                runner.sleep(0.25)
    except BaseException as primary:
        _cleanup_possibly_live(runner, possibly_live, primary=primary)
        raise
    primary = TimeoutError("rank 0 readiness was not observed before timeout")
    _cleanup_possibly_live(runner, possibly_live, primary=primary)
    raise primary


class OwnedProcessRunner(Protocol):
    def read_ownership(self, machine_rank: int) -> OwnershipRecord: ...
    def process_start_ticks(self, ownership: OwnershipRecord) -> int: ...
    def command_digest(self, ownership: OwnershipRecord) -> str: ...
    def commit(self, ownership: OwnershipRecord) -> str: ...
    def signal(self, ownership: OwnershipRecord) -> None: ...
    def poll_exit(self, machine_rank: int) -> int | None: ...
    def sleep(self, seconds: float) -> None: ...
    def cleanup_owned(self, machine_rank: int, *, timeout_seconds: float) -> bool: ...


def _cleanup_possibly_live(
    runner: OwnedProcessRunner,
    machine_ranks: Sequence[int],
    *,
    primary: BaseException,
) -> None:
    """Attempt each exact-owned cleanup independently without hiding primary."""
    for machine_rank in dict.fromkeys(machine_ranks):
        try:
            runner.cleanup_owned(
                machine_rank, timeout_seconds=_CONTROL_TIMEOUT_SECONDS
            )
        except BaseException as cleanup_error:
            primary.add_note(
                f"cleanup rank {machine_rank} failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )


class ArtifactCollectionRunner(Protocol):
    def read_artifact_manifest(
        self,
        machine_rank: int,
        checkpoint_path: Path,
        runtime_identity: Mapping[str, object],
    ) -> object: ...

    def validate_push_destination(
        self, manifest: object, entries: Sequence[object]
    ) -> None: ...

    def push_artifacts(
        self, manifest: object, entries: Sequence[object]
    ) -> None: ...
    def read_canonical_artifact_manifest(
        self, checkpoint_path: Path, runtime_identity: Mapping[str, object]
    ) -> object: ...


def collect_checkpoint_artifacts(
    runner: ArtifactCollectionRunner,
    *,
    checkpoint_path: str | Path,
    runtime_identity: Mapping[str, object],
    exit_codes: Mapping[int, int],
    world_size: int,
) -> tuple[object, ...]:
    """Collect local shards only after both node supervisors prove exit zero."""
    from src.multinode_artifacts import (
        select_collection_push_entries,
        validate_canonical_destination,
        validate_collection,
    )

    if dict(exit_codes) != {0: 0, 1: 0}:
        raise ValueError("checkpoint collection requires zero exits from both nodes")
    checkpoint = Path(checkpoint_path)
    remote = runner.read_artifact_manifest(0, checkpoint, runtime_identity)
    local = runner.read_artifact_manifest(1, checkpoint, runtime_identity)
    combined = validate_collection((remote, local), world_size=world_size)
    local_entries = select_collection_push_entries(local)
    runner.validate_push_destination(local, local_entries)
    runner.push_artifacts(local, local_entries)
    observed = runner.read_canonical_artifact_manifest(checkpoint, runtime_identity)
    validate_canonical_destination(
        combined,
        observed,
        expected_identity=runtime_identity,
        expected_run_id=str(runtime_identity["run_id"]),
    )
    return combined


def redistribute_resume_checkpoint(
    runner: object,
    *,
    checkpoint_path: str | Path,
    checkpoint_identity: Mapping[str, object],
    requested_identity: Mapping[str, object],
    destination: str | Path,
) -> tuple[object, ...]:
    """Validate the remote canonical checkpoint and restore local ranks before launch."""
    from src.multinode_artifacts import (
        resume_entry_delta,
        validate_canonical_checkpoint_manifest,
        validate_resume_identity,
    )

    validate_resume_identity(checkpoint_identity, requested_identity)
    checkpoint = Path(checkpoint_path)
    target = Path(destination)
    if target != checkpoint:
        raise ValueError("resume must use the same immediate canonical checkpoint path")
    expected_name = f"checkpoint-{checkpoint_identity['global_step']}"
    if (
        checkpoint.name != expected_name
        or checkpoint.parent.name != checkpoint_identity["run_id"]
    ):
        raise ValueError("resume path is not the immediate run-owned checkpoint-N")
    read_manifest = getattr(runner, "read_canonical_artifact_manifest")
    manifest = read_manifest(checkpoint, checkpoint_identity)
    validate_canonical_checkpoint_manifest(
        manifest, expected_identity=checkpoint_identity
    )
    local = getattr(runner, "read_artifact_manifest")(
        1, checkpoint, checkpoint_identity
    )
    required, missing = resume_entry_delta(manifest, local)
    if missing:
        runner.validate_resume_destination(missing, target)
        runner.pull_resume_artifacts(manifest, missing, target)
    observed = getattr(runner, "read_artifact_manifest")(
        1, checkpoint, checkpoint_identity, canonical_owner=True
    )
    observed_required, observed_missing = resume_entry_delta(manifest, observed)
    if observed_missing or observed_required != required:
        raise ValueError("local resume checkpoint is incomplete after redistribution")
    return required


def prepare_inference_artifacts(
    runner: object,
    *,
    checkpoint_path: str | Path,
    runtime_identity: Mapping[str, object],
    destination: str | Path,
) -> Path:
    """Prepare one fresh inference-only copy before an evaluator is started."""
    from src.multinode_artifacts import (
        prepare_inference_copy,
        resolve_local_checkpoint_destination,
        validate_canonical_checkpoint_manifest,
        validate_inference_manifest_identity,
    )

    read_manifest = getattr(runner, "read_canonical_artifact_manifest")
    manifest = read_manifest(Path(checkpoint_path), runtime_identity)
    validate_canonical_checkpoint_manifest(
        manifest, expected_identity=runtime_identity
    )
    read_inference = getattr(runner, "read_inference_artifact_manifest")
    inference_manifest = read_inference(Path(checkpoint_path), runtime_identity)
    validate_inference_manifest_identity(
        inference_manifest, expected_identity=runtime_identity
    )
    logical_destination = Path(destination)
    physical_destination = resolve_local_checkpoint_destination(
        logical_destination
    )
    prepare_inference_copy(
        inference_manifest, physical_destination, runner
    )
    mirrored = getattr(runner, "mirror_prepared_inference")(
        inference_manifest, logical_destination
    )
    if mirrored != inference_manifest:
        raise ValueError("remote prepared inference identity differs from local")
    return logical_destination


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


def _decode_process_environment(process_directory: Path) -> dict[str, str]:
    environment: dict[str, str] = {}
    for assignment in _nul_fields(process_directory / "environ"):
        name, separator, value = assignment.partition("=")
        if not separator or name in environment:
            raise ValueError("live ownership environment is malformed")
        environment[name] = value
    return environment


def stop_exact_owned_local_process(
    record: OwnershipRecord,
    *,
    expected_run_id: str,
    expected_attempt_id: str,
    expected_machine_rank: int,
    proc_root: str | Path = "/proc",
) -> bool:
    """Signal one still-live exact owner through a validated pidfd only."""
    if not isinstance(record, OwnershipRecord):
        raise TypeError("record must be an OwnershipRecord")
    try:
        pidfd = os.pidfd_open(record.pid)
    except ProcessLookupError:
        return False
    try:
        process_directory = Path(proc_root) / str(record.pid)
        try:
            observed_start_ticks = _proc_start_ticks(
                record.pid, proc_root=proc_root
            )
            observed_command = _nul_fields(process_directory / "cmdline")
            observed_environment = _decode_process_environment(process_directory)
        except (FileNotFoundError, ProcessLookupError):
            return False
        observed_digest = canonical_command_digest(
            observed_command, observed_environment
        )
        validate_ownership(
            record,
            expected_run_id=expected_run_id,
            expected_attempt_id=expected_attempt_id,
            expected_machine_rank=expected_machine_rank,
            expected_pid=record.pid,
            expected_commit=record.commit,
            expected_command_digest=record.command_digest,
            observed_start_ticks=observed_start_ticks,
            observed_command_digest=observed_digest,
            observed_commit=observed_environment.get("ILVR_REPOSITORY_COMMIT"),
        )
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(pidfd)


def _read_ownership_at(attempt_descriptor: int, name: str) -> OwnershipRecord:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=attempt_descriptor,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise ValueError("ownership record is not a private owned regular file")
        if metadata.st_size > _MAX_TRANSFER_BYTES:
            raise ValueError("ownership record exceeds the size limit")
        chunks: list[bytes] = []
        remaining = _MAX_TRANSFER_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > _MAX_TRANSFER_BYTES:
            raise ValueError("ownership record exceeds the size limit")
        return OwnershipRecord.from_json(json.loads(raw))
    finally:
        os.close(descriptor)


def stop_owned_attempts(*, run_id: str, machine_rank: int) -> dict[str, object]:
    """Stop every exact live attempt in one canonical node-local run namespace."""
    requested_run_id = _require_identifier("run_id", run_id)
    if type(machine_rank) is not int or machine_rank not in (0, 1):
        raise ValueError("machine_rank must be 0 or 1")
    ownership_names = {
        0: (
            "ownership-rank0.json",
            "ownership-eval-shard0.json",
            "ownership-eval-shard1.json",
        ),
        1: (
            "ownership-rank1.json",
            "ownership-eval-shard2.json",
            "ownership-eval-shard3.json",
        ),
    }[machine_rank]
    root_descriptor = _open_trusted_directory(_LOG_ROOT)
    run_descriptor: int | None = None
    records: list[tuple[str, OwnershipRecord]] = []
    try:
        run_descriptor = _open_child_directory(
            root_descriptor, requested_run_id, create=False
        )
        run_metadata = os.fstat(run_descriptor)
        if (
            run_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(run_metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise ValueError("owned run control directory is not private")
        attempt_names = sorted(os.listdir(run_descriptor))
        for attempt_name in attempt_names:
            _require_identifier("attempt_id", attempt_name)
            attempt_descriptor = _open_child_directory(
                run_descriptor, attempt_name, create=False
            )
            try:
                attempt_metadata = os.fstat(attempt_descriptor)
                if (
                    attempt_metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(attempt_metadata.st_mode)
                    & (stat.S_IWGRP | stat.S_IWOTH)
                ):
                    raise ValueError("owned attempt control directory is not private")
                for ownership_name in ownership_names:
                    try:
                        record = _read_ownership_at(
                            attempt_descriptor, ownership_name
                        )
                    except FileNotFoundError:
                        continue
                    validate_ownership(
                        record,
                        expected_run_id=requested_run_id,
                        expected_attempt_id=attempt_name,
                        expected_machine_rank=machine_rank,
                        expected_pid=record.pid,
                        expected_commit=record.commit,
                        expected_command_digest=record.command_digest,
                        observed_start_ticks=record.process_start_ticks,
                    )
                    records.append((attempt_name, record))
            finally:
                os.close(attempt_descriptor)
    finally:
        if run_descriptor is not None:
            os.close(run_descriptor)
        os.close(root_descriptor)

    outcomes: dict[str, list[bool]] = {}
    for attempt_name, record in records:
        stopped = stop_exact_owned_local_process(
            record,
            expected_run_id=requested_run_id,
            expected_attempt_id=record.attempt_id,
            expected_machine_rank=machine_rank,
        )
        outcomes.setdefault(attempt_name, []).append(stopped)
    return {
        "run_id": requested_run_id,
        "machine_rank": machine_rank,
        "stopped_attempts": [
            attempt_name
            for attempt_name, attempt_outcomes in outcomes.items()
            if any(attempt_outcomes)
        ],
        "already_exited_attempts": [
            attempt_name
            for attempt_name, attempt_outcomes in outcomes.items()
            if not any(attempt_outcomes)
        ],
    }


def monitor_owned_jobs(
    plan: "LaunchPlan",
    runner: OwnedProcessRunner,
    *,
    expected_commit: str,
    expected_command_digests: Mapping[int, str],
    poll_interval_seconds: float = 0.25,
    attempt_timeout_seconds: float | None = None,
) -> dict[int, int]:
    """Collect both exits and stop only a revalidated live peer after failure."""
    exits: dict[int, int] = {}
    ranks = (plan.cluster.remote_rank, plan.cluster.local_rank)
    timeout = (
        plan.attempt_timeout_seconds
        if attempt_timeout_seconds is None
        else attempt_timeout_seconds
    )
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("attempt timeout must be positive and finite")
    deadline = time.monotonic() + timeout
    try:
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
                        if runner.cleanup_owned(
                            peer_rank, timeout_seconds=_CONTROL_TIMEOUT_SECONDS
                        ):
                            exits[peer_rank] = -15
                    return exits
            if len(exits) < len(ranks):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "training attempt exceeded its 24-hour deadline"
                    )
                runner.sleep(poll_interval_seconds)
    except BaseException as primary:
        _cleanup_possibly_live(
            runner,
            (rank for rank in ranks if rank not in exits),
            primary=primary,
        )
        raise
    return exits


def _finite_loss_reference(
    path: Path | None,
    *,
    plan: "LaunchPlan" | None = None,
    checkpoint_root: Path | None = None,
    expected_global_step: int | None = None,
) -> dict[str, object]:
    if path is None:
        raise ValueError("finite loss evidence is required for a successful attempt")
    try:
        raw = _read_regular_file(path)
    except (OSError, ValueError) as error:
        raise ValueError("finite loss evidence must be a trusted regular file") from error
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("finite loss evidence contains a duplicate field")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise ValueError("finite loss evidence is invalid JSON") from error
    if plan is not None and plan.stage == "stage2":
        if type(payload) is not dict:
            raise ValueError("Stage 2 producer evidence must be a JSON object")
        global_step = payload.get("global_step")
        producer_checkpoint = payload.get("checkpoint_path")
        if type(global_step) is not int or type(producer_checkpoint) is not str:
            raise ValueError("Stage 2 producer evidence identity is invalid")
        from stage2.evidence import decode_stage2_producer_evidence

        decoded = decode_stage2_producer_evidence(
            raw,
            expected_run_id=plan.run_id,
            expected_method=plan.tuning_method,
            expected_zero_stage=plan.zero_stage,
            expected_checkpoint=producer_checkpoint,
            expected_global_step=global_step,
        )
        if (
            checkpoint_root is None
            or Path(decoded["checkpoint_path"]).parent != checkpoint_root
        ):
            raise ValueError("Stage 2 producer checkpoint differs from training output")
        return {
            "status": "validated",
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
            "finite_loss": True,
            "finite_gradient": True,
            "process_ranks": [0, 1, 2, 3],
        }
    if (
        plan is not None
        and plan.stage == "stage1"
        and type(payload) is dict
        and payload.get("kind") == "stage1-training-evidence"
    ):
        if checkpoint_root is None or expected_global_step is None:
            raise ValueError("Stage 1 training evidence requires exact checkpoint step")
        from stage1.training_evidence import decode_stage1_training_evidence

        decoded = decode_stage1_training_evidence(
            raw,
            expected_run_id=plan.run_id,
            expected_method=plan.tuning_method,
            expected_zero_stage=plan.zero_stage,
            expected_world_size=plan.world_size,
            expected_checkpoint=(
                checkpoint_root / f"checkpoint-{expected_global_step}"
            ),
            expected_global_step=expected_global_step,
        )
        return {
            "status": "validated",
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
            "finite_loss": decoded["finite_loss"],
        }
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


def _checkpoint_has_concrete_output(checkpoint: Path) -> bool:
    for root, directories, files in os.walk(checkpoint, followlinks=False):
        root_path = Path(root)
        directories[:] = [
            name for name in directories if not (root_path / name).is_symlink()
        ]
        for name in files:
            candidate = root_path / name
            if not candidate.is_symlink() and candidate.is_file() and candidate.stat().st_size > 0:
                return True
    return False


def write_incomplete_attempt(
    plan: "LaunchPlan",
    *,
    result_root: str | Path,
    reason: str,
    exit_codes: Mapping[int, int] | None = None,
) -> Path:
    """Publish a distinct non-success record when witness evidence is incomplete."""
    if not isinstance(reason, str) or not reason or "\0" in reason:
        raise ValueError("incomplete attempt reason must be a nonempty string")
    payload = {
        "schema_version": _INCOMPLETE_SCHEMA_VERSION,
        "status": "incomplete",
        "run_id": plan.run_id,
        "attempt_id": plan.attempt_id,
        "stage": plan.stage,
        "reason": reason,
        "exit_codes": {
            str(rank): code for rank, code in sorted((exit_codes or {}).items())
        },
    }
    path = (
        _trusted_absolute_path(result_root)
        / plan.run_id
        / f"{plan.attempt_id}-train-incomplete.json"
    )
    return _publish_bytes_no_replace(path, _canonical_json_bytes(payload) + b"\n")


def write_training_witness(
    plan: "LaunchPlan",
    *,
    result_root: str | Path,
    exit_codes: Mapping[int, int],
    checkpoint_path: str | Path,
    batch_per_gpu: int,
    finite_loss_evidence_path: str | Path | None = None,
    expected_global_step: int | None = None,
) -> Path:
    """Exclusively publish one attempt witness without inventing loss evidence."""
    if dict(exit_codes) != {0: 0, 1: 0}:
        raise ValueError("training witness requires exit 0 from both machine ranks")
    if type(batch_per_gpu) is not int or batch_per_gpu <= 0:
        raise ValueError("batch_per_gpu must be positive")
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("training witness checkpoint must be an existing directory")
    if checkpoint.name != plan.run_id:
        raise ValueError("training witness checkpoint must be owned by the run ID")
    canonical_checkpoint = checkpoint.resolve(strict=False)
    canonical_repository_checkpoint_root = _CHECKPOINT_ROOT.resolve(strict=False)
    try:
        checkpoint.relative_to(_CHECKPOINT_ROOT)
    except ValueError:
        checkpoint_scope = checkpoint.parent.resolve(strict=False)
    else:
        checkpoint_scope = canonical_repository_checkpoint_root
    if canonical_checkpoint != checkpoint_scope / plan.run_id:
        raise ValueError("training witness checkpoint escaped its run-owned root")
    if not _checkpoint_has_concrete_output(checkpoint):
        raise ValueError("training witness checkpoint must contain nonempty output")
    finite_loss = _finite_loss_reference(
        Path(finite_loss_evidence_path)
        if finite_loss_evidence_path is not None
        else None,
        plan=plan,
        checkpoint_root=checkpoint,
        expected_global_step=expected_global_step,
    )
    run_directory = _trusted_absolute_path(result_root) / plan.run_id
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
    return _publish_bytes_no_replace(path, _canonical_json_bytes(payload) + b"\n")


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


def build_remote_exec(
    argv: Sequence[str], *, working_directory: str | Path | None = None
) -> list[str]:
    """Build the fixed, noninteractive remote container command safely."""
    docker = ["/usr/bin/docker", "exec"]
    if working_directory is not None:
        directory = Path(working_directory)
        if directory != _REPOSITORY_ROOT:
            raise ValueError("remote container working directory must be canonical")
        docker.extend(("--workdir", str(directory)))
    docker.extend((_REMOTE_CONTAINER, "/usr/bin/bash", "-lc", shlex.join(argv)))
    remote_command = shlex.join(
        docker
    )
    return [
        "/usr/bin/ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        _REMOTE_HOST,
        remote_command,
    ]


def _repository_module_command(
    subcommand: str, arguments: Sequence[str] = ()
) -> tuple[str, ...]:
    """Build an import-stable command for the fixed canonical repository."""
    repository = _REPOSITORY_ROOT
    return (
        "/usr/bin/env",
        f"PYTHONPATH={repository}:{repository / 'src'}",
        str(repository / "venv/ilvr-venv/bin/python"),
        "-m",
        "src.multinode",
        subcommand,
        *arguments,
    )


def _local_repository_module_command(
    subcommand: str, arguments: Sequence[str] = ()
) -> tuple[str, ...]:
    repository_command = _repository_module_command(subcommand, arguments)
    return (
        repository_command[0],
        "-C",
        str(_REPOSITORY_ROOT),
        *repository_command[1:],
    )


def _remote_repository_module_command(
    subcommand: str, arguments: Sequence[str] = ()
) -> tuple[str, ...]:
    return tuple(
        build_remote_exec(
            _repository_module_command(subcommand, arguments),
            working_directory=_REPOSITORY_ROOT,
        )
    )


def _build_remote_stdin_exec(
    argv: Sequence[str], *, working_directory: str | Path | None = None
) -> list[str]:
    """Build the fixed remote container boundary with stdin attached."""
    docker = ["/usr/bin/docker", "exec", "-i"]
    if working_directory is not None:
        directory = Path(working_directory)
        if directory != _REPOSITORY_ROOT:
            raise ValueError("remote container working directory must be canonical")
        docker.extend(("--workdir", str(directory)))
    docker.extend((_REMOTE_CONTAINER, "/usr/bin/bash", "-lc", shlex.join(argv)))
    remote_command = shlex.join(docker)
    return [
        "/usr/bin/ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        _REMOTE_HOST,
        remote_command,
    ]


def _remote_repository_module_stdin_command(
    subcommand: str, arguments: Sequence[str] = ()
) -> tuple[str, ...]:
    return tuple(
        _build_remote_stdin_exec(
            _repository_module_command(subcommand, arguments),
            working_directory=_REPOSITORY_ROOT,
        )
    )


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
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            plan.cluster.remote_host,
            shlex.join(
                (
                    "/usr/bin/docker",
                    "inspect",
                    "--format",
                    "{{.State.Running}}",
                    plan.cluster.remote_container,
                )
            ),
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
        f"PYTHONPATH={repository}:{repository / 'src'}",
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
        "--stage",
        plan.stage,
        "--tuning-method",
        plan.tuning_method,
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
