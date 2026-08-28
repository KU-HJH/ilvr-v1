"""Bounded Stage 1 multi-node capacity probing."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Literal, Protocol, Sequence

from src.multinode_runtime import (
    _open_trusted_directory,
    _read_regular_file,
    build_remote_exec,
    publish_small_file,
    validate_stage1_arguments,
)


AttemptStatus = Literal["success", "capacity_oom", "integration_failure"]

_INTEGRATION_FAILURE_SUBJECTS = ("checkpoint", "ssh", "docker", "rendezvous")
_FAILURE_TOKEN_PATTERN = re.compile(
    r"\b(?:error|fail|failed|failure|invalid|mismatch|missing|refused|timeout|unavailable)\b"
)
_FAILURE_PHRASES = ("not found", "not running", "timed out")
_CUDA_OOM_EXCEPTION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"torch\.outofmemoryerror",
        r"cuda(?:\s+error\s*:)?\s+out\s+of\s+memory",
        r"cuda\s+(?:allocator|allocation)\s+(?:error|failed|failure)"
        r"(?:\s+(?:allocation|allocating)(?:\s+device\s+memory)?)?",
        r"cudamalloc\s+(?:error|failed|failure)"
        r"(?:\s+allocating(?:\s+device\s+memory)?)?",
    )
)
_CAPACITY_MARKERS = (
    "torch.outofmemoryerror",
    "cuda out of memory",
)
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,127}$")
_MAX_EVIDENCE_BYTES = 256 * 1024
_REPOSITORY_ROOT = Path("/root/workspace/ILVR")
_RESULT_ROOT = _REPOSITORY_ROOT / "results"
_LOG_ROOT = _REPOSITORY_ROOT / "logs" / "multinode"
_STAGE_LOG_ROOT = _REPOSITORY_ROOT / "logs" / "zero-stage-compatibility"
_CAPACITY_LOG_ROOT = _REPOSITORY_ROOT / "logs" / "capacity"
_TRAINING_WITNESS_FIELDS = {
    "schema_version",
    "run_id",
    "attempt_id",
    "stage",
    "method",
    "zero_stage",
    "batch_per_gpu",
    "world_size",
    "machine_ranks",
    "exit_codes",
    "checkpoint_path",
    "finite_loss_evidence",
}
_FINITE_LOSS_FIELDS = {"status", "path", "sha256", "size", "finite_loss"}


@dataclass(frozen=True)
class CapacityEvidence:
    """One bounded exact-child status or log excerpt used for classification."""

    node: Literal["remote", "local", "coordinator"]
    path: Path
    text: str

    def __post_init__(self) -> None:
        if self.node not in ("remote", "local", "coordinator"):
            raise ValueError("capacity evidence node is invalid")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("capacity evidence path must be absolute")
        if not isinstance(self.text, str):
            raise TypeError("capacity evidence text must be text")
        if len(self.text.encode("utf-8")) > _MAX_EVIDENCE_BYTES:
            raise ValueError("capacity evidence exceeds the bounded size")


@dataclass(frozen=True)
class ChildExecution:
    """Bounded outcome returned by one existing Stage 1 coordinator launch."""

    returncode: int
    evidence: tuple[CapacityEvidence, ...]

    def __post_init__(self) -> None:
        if type(self.returncode) is not int:
            raise TypeError("child returncode must be an integer")
        if type(self.evidence) is not tuple or not all(
            isinstance(item, CapacityEvidence) for item in self.evidence
        ):
            raise TypeError("child evidence must be a tuple of CapacityEvidence")


class CapacityRunner(Protocol):
    def launch(
        self,
        command: Sequence[str],
        *,
        child_run_id: str,
        batch_per_gpu: int,
    ) -> ChildExecution: ...


def _read_bounded_tail(path: Path) -> str:
    parent_descriptor = _open_trusted_directory(path.parent)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("capacity evidence must be a regular file")
        offset = max(0, metadata.st_size - _MAX_EVIDENCE_BYTES)
        os.lseek(descriptor, offset, os.SEEK_SET)
        raw = os.read(descriptor, _MAX_EVIDENCE_BYTES)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
    return raw.decode("utf-8", errors="replace")


def _attempt_evidence_paths(
    *,
    run_id: str,
    machine_rank: int,
    result_root: Path,
    log_root: Path,
    stage_log_root: Path,
) -> tuple[Path, ...]:
    return (
        stage_log_root / run_id / "initial-train.log",
        log_root / run_id / "initial" / f"exit-rank{machine_rank}.json",
        result_root / run_id / "initial-train-incomplete.json",
    )


def collect_attempt_evidence(
    *,
    run_id: str,
    node: Literal["remote", "local"],
    machine_rank: int,
    result_root: str | Path = _RESULT_ROOT,
    log_root: str | Path = _LOG_ROOT,
    stage_log_root: str | Path = _STAGE_LOG_ROOT,
) -> tuple[CapacityEvidence, ...]:
    """Read only the exact child's bounded node-local log and status files."""
    child_run_id = _validate_run_id(run_id)
    if (node, machine_rank) not in (("remote", 0), ("local", 1)):
        raise ValueError("capacity node and machine rank do not match")
    paths = _attempt_evidence_paths(
        run_id=child_run_id,
        machine_rank=machine_rank,
        result_root=Path(result_root),
        log_root=Path(log_root),
        stage_log_root=Path(stage_log_root),
    )
    evidence: list[CapacityEvidence] = []
    for path in paths:
        try:
            text = _read_bounded_tail(path)
        except FileNotFoundError:
            continue
        evidence.append(CapacityEvidence(node=node, path=path, text=text))
    return tuple(evidence)


def _capacity_evidence_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.multinode capacity-evidence-show", allow_abbrev=False
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node", choices=("remote", "local"), required=True)
    parser.add_argument("--machine-rank", type=int, required=True)
    parser.add_argument("--result-root", default=str(_RESULT_ROOT))
    args = parser.parse_args(argv)
    try:
        result_root = Path(args.result_root)
        if not result_root.is_absolute() or ".." in result_root.parts:
            raise ValueError("result_root must be an absolute path without traversal")
        result_root.relative_to(_RESULT_ROOT)
        evidence = collect_attempt_evidence(
            run_id=args.run_id,
            node=args.node,
            machine_rank=args.machine_rank,
            result_root=result_root,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            [
                {"node": item.node, "path": str(item.path), "text": item.text}
                for item in evidence
            ],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


class SubprocessCapacityRunner:
    """Launch the existing Stage 1 coordinator and gather bounded child evidence."""

    def __init__(
        self,
        *,
        parent_run_id: str,
        result_root: str | Path,
        training_runner=subprocess.run,
        control_runner=subprocess.run,
        log_root: str | Path = _CAPACITY_LOG_ROOT,
    ) -> None:
        self.parent_run_id = _validate_run_id(parent_run_id)
        self.result_root = Path(result_root)
        self.training_runner = training_runner
        self.control_runner = control_runner
        self.log_root = Path(log_root)

    def _open_log(self, directory_descriptor: int, name: str):
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_descriptor,
        )
        return os.fdopen(descriptor, "w", encoding="utf-8")

    def _remote_evidence(self, child_run_id: str) -> tuple[CapacityEvidence, ...]:
        python = _REPOSITORY_ROOT / "venv" / "ilvr-venv" / "bin" / "python"
        command = build_remote_exec(
            (
                "env",
                f"PYTHONPATH={_REPOSITORY_ROOT}",
                str(python),
                "-m",
                "src.multinode",
                "capacity-evidence-show",
                "--run-id",
                child_run_id,
                "--node",
                "remote",
                "--machine-rank",
                "0",
                "--result-root",
                str(self.result_root),
            )
        )
        completed = self.control_runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"SSH remote evidence query failed: {completed.stderr[-4096:]}"
            )
        payload = json.loads(completed.stdout)
        if type(payload) is not list:
            raise ValueError("remote capacity evidence must be a JSON array")
        allowed = set(
            _attempt_evidence_paths(
                run_id=child_run_id,
                machine_rank=0,
                result_root=self.result_root,
                log_root=_LOG_ROOT,
                stage_log_root=_STAGE_LOG_ROOT,
            )
        )
        evidence: list[CapacityEvidence] = []
        seen: set[Path] = set()
        for item in payload:
            if type(item) is not dict or item.get("node") != "remote":
                raise ValueError("remote capacity evidence identity is invalid")
            path = Path(item.get("path", ""))
            if path not in allowed or path in seen:
                raise ValueError("remote capacity evidence path is not exact")
            evidence.append(
                CapacityEvidence(node="remote", path=path, text=item.get("text"))
            )
            seen.add(path)
        return tuple(evidence)

    def launch(
        self,
        command: Sequence[str],
        *,
        child_run_id: str,
        batch_per_gpu: int,
    ) -> ChildExecution:
        _validate_run_id(child_run_id)
        if type(batch_per_gpu) is not int or batch_per_gpu not in (1, 2, 4, 8):
            raise ValueError("capacity child batch is invalid")
        attempt_log_directory = self.log_root / self.parent_run_id
        directory_descriptor = _open_trusted_directory(
            attempt_log_directory, create=True
        )
        stdout_path = attempt_log_directory / f"{child_run_id}.stdout.log"
        stderr_path = attempt_log_directory / f"{child_run_id}.stderr.log"
        launch_failed = False
        try:
            with self._open_log(directory_descriptor, stdout_path.name) as stdout_handle:
                with self._open_log(directory_descriptor, stderr_path.name) as stderr_handle:
                    try:
                        completed = self.training_runner(
                            tuple(command),
                            stdout=stdout_handle,
                            stderr=stderr_handle,
                            text=True,
                            check=False,
                        )
                    except (OSError, subprocess.SubprocessError) as error:
                        launch_failed = True
                        stderr_handle.write(
                            f"integration failure launching child coordinator: {error}\n"
                        )
                        stderr_handle.flush()
                        completed = subprocess.CompletedProcess(tuple(command), 125)
        finally:
            os.close(directory_descriptor)
        evidence = [
            CapacityEvidence(
                node="coordinator", path=stdout_path, text=_read_bounded_tail(stdout_path)
            ),
            CapacityEvidence(
                node="coordinator", path=stderr_path, text=_read_bounded_tail(stderr_path)
            ),
        ]
        if completed.returncode != 0 and not launch_failed:
            evidence.extend(
                collect_attempt_evidence(
                    run_id=child_run_id,
                    node="local",
                    machine_rank=1,
                    result_root=self.result_root,
                )
            )
            try:
                evidence.extend(self._remote_evidence(child_run_id))
            except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
                query_path = attempt_log_directory / f"{child_run_id}.remote-evidence.log"
                query_text = f"SSH remote evidence query failed: {error}"
                publish_small_file(
                    query_path,
                    query_text.encode("utf-8")[:_MAX_EVIDENCE_BYTES],
                )
                evidence.append(
                    CapacityEvidence(
                        node="coordinator", path=query_path, text=query_text
                    )
                )
        return ChildExecution(
            returncode=completed.returncode,
            evidence=tuple(evidence),
        )


def capacity_candidates(*, start: int, maximum: int) -> tuple[int, ...]:
    """Return the fixed primary candidate sequence."""
    if type(start) is not int or start != 2:
        raise ValueError("primary capacity probing must start at batch 2")
    if type(maximum) is not int or maximum != 8:
        raise ValueError("primary capacity probing must end at batch 8")
    return (2, 4, 8)


def fallback_candidates(
    *, classifications: Mapping[int, AttemptStatus]
) -> tuple[int, ...]:
    """Return batch one only after candidate two has a CUDA capacity OOM."""
    if classifications.get(2) != "capacity_oom":
        raise ValueError("batch 1 requires candidate 2 CUDA capacity OOM")
    return (1,)


def classify_failure(evidence: str) -> AttemptStatus:
    """Classify bounded child evidence, with integration markers taking priority."""
    if not isinstance(evidence, str):
        raise TypeError("failure evidence must be text")
    normalized = evidence.casefold()
    nonfinite_diagnostic = re.search(
        r"\b(?:non[- ]?finite)\s+(?:loss|gradient)\b"
        r"|\b(?:loss|gradient)\b[^\n]{0,64}\bnon[- ]?finite\b",
        normalized,
    )
    if (
        "nccl" in normalized
        or "collective mismatch" in normalized
        or nonfinite_diagnostic is not None
    ):
        return "integration_failure"
    for line in normalized.splitlines():
        conditional_line = line
        for pattern in _CUDA_OOM_EXCEPTION_PATTERNS:
            conditional_line = pattern.sub(" ", conditional_line)
        if any(
            re.search(rf"\b{subject}\b", conditional_line)
            for subject in _INTEGRATION_FAILURE_SUBJECTS
        ) and (
            _FAILURE_TOKEN_PATTERN.search(conditional_line) is not None
            or any(phrase in conditional_line for phrase in _FAILURE_PHRASES)
        ):
            return "integration_failure"
    if any(marker in normalized for marker in _CAPACITY_MARKERS):
        return "capacity_oom"
    allocator_oom = (
        "cuda" in normalized
        and "allocat" in normalized
        and any(marker in normalized for marker in ("failed", "failure", "out of memory"))
    )
    return "capacity_oom" if allocator_oom else "integration_failure"


def _validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"run_id must match {_RUN_ID_PATTERN.pattern}")
    return run_id


def _validate_ports(ports: Sequence[int]) -> tuple[int, int, int, int]:
    values = tuple(ports)
    if len(values) != 4:
        raise ValueError("capacity probing requires four distinct explicit ports")
    if any(type(port) is not int or not 1024 <= port <= 65535 for port in values):
        raise ValueError("capacity ports must be between 1024 and 65535")
    if len(set(values)) != 4:
        raise ValueError("capacity probing requires four distinct explicit ports")
    return values  # type: ignore[return-value]


def _validate_stage_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    values = tuple(arguments)
    if not all(isinstance(value, str) and "\0" not in value for value in values):
        raise ValueError("Stage 1 arguments must be NUL-free text")
    if any(
        value in ("--micro-batch", "--method")
        or value.startswith("--micro-batch=")
        or value.startswith("--method=")
        for value in values
    ):
        raise ValueError("capacity controller owns --method and --micro-batch")
    return values


def _decode_training_witness(
    raw: bytes,
    *,
    child_run_id: str,
    zero_stage: int,
    tuning_method: str,
    batch_per_gpu: int,
    producer_path: Path,
) -> dict[str, object]:
    def exact_object(pairs):
        payload = {}
        for name, value in pairs:
            if name in payload:
                raise ValueError("training witness contains a duplicate field")
            payload[name] = value
        return payload

    payload = json.loads(raw, object_pairs_hook=exact_object)
    if type(payload) is not dict or set(payload) != _TRAINING_WITNESS_FIELDS:
        raise ValueError("training witness fields are invalid")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("training witness schema version is invalid")
    expected_text = {
        "run_id": child_run_id,
        "attempt_id": "initial",
        "stage": "stage1",
        "method": tuning_method,
        "checkpoint_path": f"/root/workspace/ILVR/checkpoints/{child_run_id}",
    }
    if any(
        type(payload[name]) is not str or payload[name] != expected
        for name, expected in expected_text.items()
    ):
        raise ValueError("training witness text identity is invalid")
    expected_integers = {
        "zero_stage": zero_stage,
        "batch_per_gpu": batch_per_gpu,
        "world_size": 4,
    }
    if any(
        type(payload[name]) is not int or payload[name] != expected
        for name, expected in expected_integers.items()
    ):
        raise ValueError("training witness integer identity is invalid")
    ranks = payload["machine_ranks"]
    if (
        type(ranks) is not list
        or len(ranks) != 2
        or any(type(rank) is not int for rank in ranks)
        or ranks != [0, 1]
    ):
        raise ValueError("training witness machine ranks are invalid")
    exit_codes = payload["exit_codes"]
    if (
        type(exit_codes) is not dict
        or set(exit_codes) != {"0", "1"}
        or any(type(code) is not int or code != 0 for code in exit_codes.values())
    ):
        raise ValueError("training witness exit codes are invalid")
    finite_loss = payload["finite_loss_evidence"]
    if type(finite_loss) is not dict or set(finite_loss) != _FINITE_LOSS_FIELDS:
        raise ValueError("training witness finite-loss fields are invalid")
    if (
        type(finite_loss["status"]) is not str
        or finite_loss["status"] != "validated"
        or type(finite_loss["path"]) is not str
        or finite_loss["path"] != str(producer_path)
        or type(finite_loss["sha256"]) is not str
        or len(finite_loss["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in finite_loss["sha256"])
        or type(finite_loss["size"]) is not int
        or finite_loss["size"] < 0
        or type(finite_loss["finite_loss"]) is not bool
        or finite_loss["finite_loss"] is not True
    ):
        raise ValueError("training witness finite-loss evidence is invalid")
    return payload


def _valid_training_witness(
    path: Path,
    *,
    child_run_id: str,
    zero_stage: int,
    tuning_method: str,
    batch_per_gpu: int,
) -> bool:
    producer_path = path.parent / "initial-producer-evidence.json"
    try:
        payload = _decode_training_witness(
            _read_regular_file(path),
            child_run_id=child_run_id,
            zero_stage=zero_stage,
            tuning_method=tuning_method,
            batch_per_gpu=batch_per_gpu,
            producer_path=producer_path,
        )
        producer = _read_regular_file(producer_path)
        producer_payload = json.loads(producer)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return False
    finite_loss = payload["finite_loss_evidence"]
    assert type(finite_loss) is dict
    producer_evidence = (
        producer_payload.get("benchmark_evidence", producer_payload)
        if type(producer_payload) is dict
        else None
    )
    return (
        type(producer_evidence) is dict
        and producer_evidence.get("finite_loss") is True
        and finite_loss["size"] == len(producer)
        and finite_loss["sha256"] == hashlib.sha256(producer).hexdigest()
    )


def _child_command(
    *,
    python_executable: str,
    child_run_id: str,
    zero_stage: int,
    port: int,
    result_root: Path,
    tuning_method: str,
    stage_arguments: tuple[str, ...],
    batch_per_gpu: int,
) -> tuple[str, ...]:
    return (
        python_executable,
        "-m",
        "src.multinode",
        "train-stage1",
        "--run-id",
        child_run_id,
        "--zero-stage",
        str(zero_stage),
        "--main-process-port",
        str(port),
        "--result-root",
        str(result_root),
        "--method",
        tuning_method,
        *stage_arguments,
        "--micro-batch",
        str(batch_per_gpu),
    )


def _evidence_paths(evidence: tuple[CapacityEvidence, ...]) -> list[dict[str, str]]:
    return [{"node": item.node, "path": str(item.path)} for item in evidence]


def run_capacity_probe(
    *,
    run_id: str,
    zero_stage: int,
    tuning_method: str,
    ports: Sequence[int],
    result_root: str | Path,
    stage_arguments: Sequence[str],
    runner: CapacityRunner,
    python_executable: str,
    dry_run: bool = False,
) -> dict[str, object]:
    """Run the strict 2→4→8 search and exclusively publish its bounded report."""
    parent_run_id = _validate_run_id(run_id)
    if type(zero_stage) is not int or zero_stage not in (1, 2, 3):
        raise ValueError("zero_stage must be 1, 2, or 3")
    if tuning_method not in ("full", "lora", "dora"):
        raise ValueError("tuning_method must be full, lora, or dora")
    port_sequence = _validate_ports(ports)
    root = Path(result_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("result_root must be an absolute path without traversal")
    forwarded = _validate_stage_arguments(stage_arguments)
    if not isinstance(python_executable, str) or not python_executable:
        raise ValueError("python_executable must be nonempty")

    primary = capacity_candidates(start=2, maximum=8)
    planned = tuple(zip(primary, port_sequence[:3]))
    for candidate in (2, 4, 8, 1):
        _validate_run_id(f"{parent_run_id}-batch{candidate}")
    if dry_run:
        return {
            "schema_version": 1,
            "kind": "stage1-multinode-capacity-dry-run",
            "run_id": parent_run_id,
            "attempts": [
                {
                    "batch_per_gpu": batch,
                    "child_run_id": f"{parent_run_id}-batch{batch}",
                    "main_process_port": port,
                    "command": list(
                        _child_command(
                            python_executable=python_executable,
                            child_run_id=f"{parent_run_id}-batch{batch}",
                            zero_stage=zero_stage,
                            port=port,
                            result_root=root,
                            tuning_method=tuning_method,
                            stage_arguments=forwarded,
                            batch_per_gpu=batch,
                        )
                    ),
                }
                for batch, port in planned
            ],
            "fallback": {
                "batch_per_gpu": 1,
                "child_run_id": f"{parent_run_id}-batch1",
                "main_process_port": port_sequence[3],
                "condition": "candidate 2 capacity_oom only",
            },
        }

    report_path = root / parent_run_id / "capacity.json"
    if os.path.lexists(report_path):
        raise FileExistsError(f"refusing to replace capacity report: {report_path}")
    attempts: list[dict[str, object]] = []
    last_success: int | None = None
    queue = list(planned)
    index = 0
    while index < len(queue):
        batch, port = queue[index]
        index += 1
        child_run_id = f"{parent_run_id}-batch{batch}"
        command = _child_command(
            python_executable=python_executable,
            child_run_id=child_run_id,
            zero_stage=zero_stage,
            port=port,
            result_root=root,
            tuning_method=tuning_method,
            stage_arguments=forwarded,
            batch_per_gpu=batch,
        )
        execution = runner.launch(
            command,
            child_run_id=child_run_id,
            batch_per_gpu=batch,
        )
        witness_path = root / child_run_id / "initial-train-witness.json"
        if execution.returncode == 0:
            if _valid_training_witness(
                witness_path,
                child_run_id=child_run_id,
                zero_stage=zero_stage,
                tuning_method=tuning_method,
                batch_per_gpu=batch,
            ):
                status: AttemptStatus = "success"
                evidence_paths = [
                    {"node": "coordinator", "path": str(witness_path)}
                ]
                last_success = batch
            else:
                status = "integration_failure"
                evidence_paths = _evidence_paths(execution.evidence)
        else:
            status = classify_failure("\n".join(item.text for item in execution.evidence))
            evidence_paths = _evidence_paths(execution.evidence)
        attempts.append(
            {
                "batch_per_gpu": batch,
                "child_run_id": child_run_id,
                "main_process_port": port,
                "status": status,
                "evidence_paths": evidence_paths,
            }
        )
        if status == "integration_failure":
            last_success = None
            break
        if status == "capacity_oom":
            if batch == 2:
                fallback = fallback_candidates(classifications={2: status})
                queue = [(fallback[0], port_sequence[3])]
                index = 0
                continue
            break

    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "stage1-multinode-capacity",
        "run_id": parent_run_id,
        "stage": "stage1",
        "method": tuning_method,
        "zero_stage": zero_stage,
        "attempts": attempts,
        "recommended_batch_per_gpu": last_success,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    publish_small_file(report_path, encoded + b"\n")
    return payload


def capacity_stage1_main(argv: Sequence[str]) -> int:
    """Public ``capacity-stage1`` CLI used by the shell wrapper."""
    parser = argparse.ArgumentParser(
        prog="python -m src.multinode capacity-stage1", allow_abbrev=False
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--zero-stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument(
        "--tuning-method", choices=("full", "lora", "dora"), required=True
    )
    parser.add_argument(
        "--main-process-port",
        type=int,
        action="append",
        required=True,
        help="repeat exactly four times for batches 2, 4, 8, and fallback 1",
    )
    parser.add_argument("--result-root", default=str(_RESULT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args, stage_arguments = parser.parse_known_args(argv)
    try:
        result_root = Path(args.result_root)
        if not result_root.is_absolute() or ".." in result_root.parts:
            raise ValueError("result_root must be an absolute path without traversal")
        try:
            result_root.relative_to(_RESULT_ROOT)
        except ValueError as error:
            raise ValueError(
                "result_root must be inside /root/workspace/ILVR/results"
            ) from error
        forwarded = validate_stage1_arguments(stage_arguments)
        runner = (
            None
            if args.dry_run
            else SubprocessCapacityRunner(
                parent_run_id=args.run_id,
                result_root=result_root,
            )
        )
        payload = run_capacity_probe(
            run_id=args.run_id,
            zero_stage=args.zero_stage,
            tuning_method=args.tuning_method,
            ports=args.main_process_port,
            result_root=result_root,
            stage_arguments=forwarded,
            runner=runner,  # type: ignore[arg-type]
            python_executable=sys.executable,
            dry_run=args.dry_run,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
    if args.dry_run:
        return 0
    return 0 if payload["recommended_batch_per_gpu"] is not None else 1
