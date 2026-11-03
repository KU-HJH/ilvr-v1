"""Pure fixed topology and launch planning for ILVR multi-node runs."""

import argparse
import hashlib
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
from typing import Callable, Literal, Sequence

if __name__ == "__main__":
    # Keep runtime type checks stable when this module is executed with ``-m``.
    sys.modules.setdefault("src.multinode", sys.modules[__name__])

from src.deepspeed_runtime import (
    canonical_deepspeed_config_path,
    load_deepspeed_runtime_profile,
)
from src.multinode_runtime import (
    NodeSnapshot,
    OwnedTmuxCommands,
    OwnershipRecord,
    SubprocessCoordinatorRunner,
    _open_trusted_directory,
    _nul_fields,
    _proc_start_ticks,
    build_owned_tmux_commands,
    build_owned_lifecycle_harness_commands,
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


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REMOTE_CONTAINER_REPOSITORY_BASE = Path("/root/workspace/ILVR")
_REMOTE_HOST_REPOSITORY_BASE = Path(
    "/home/korea/Workspace/hjh/workspace/ILVR"
)
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,127}$")
_TRAINING_REQUIRED_FREE_BYTES = 100 * 1024**3
_ATTEMPT_TIMEOUT_SECONDS = 24 * 60 * 60
_CONTROL_TIMEOUT_SECONDS = 30
_PROC_ROOT = Path("/proc")


def _evaluation_staging_root() -> Path:
    return _REPOSITORY_ROOT / "checkpoints" / "evaluation-staging"


def _remote_host_repository_root(repository_root: Path) -> Path:
    relative = repository_root.relative_to(_REMOTE_CONTAINER_REPOSITORY_BASE)
    return _REMOTE_HOST_REPOSITORY_BASE / relative


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
            remote_host="korea@172.16.11.41",
            remote_container="llm_kd",
            remote_repo_root=_REPOSITORY_ROOT,
            remote_host_repo_root=_remote_host_repository_root(_REPOSITORY_ROOT),
            master_addr="172.16.11.41",
            socket_ifname="bond0.11",
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
        expected_config = selected_deepspeed_config(
            self.stage, self.tuning_method, self.zero_stage
        )
        if self.deepspeed_config != expected_config:
            raise ValueError("deepspeed_config must match the selected ZeRO policy")
        if (
            not isinstance(self.deepspeed_config_digest, str)
            or len(self.deepspeed_config_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.deepspeed_config_digest
            )
        ):
            raise ValueError("deepspeed_config_digest must be a SHA-256 digest")


@dataclass(frozen=True)
class LifecycleCase:
    """One deterministic cell in the four-rank save/resume matrix."""

    stage: Literal["stage1", "stage2"]
    method: Literal["full", "lora", "dora"]
    zero_stage: Literal[1, 2, 3]
    initial_steps: int = 2
    resumed_steps: int = 3


@dataclass(frozen=True)
class LifecyclePlan:
    """Two owned phase jobs separated by mandatory checkpoint redistribution."""

    case: LifecycleCase
    child_run_id: str
    main_process_port: int
    witness_path: Path
    initial_witness_path: Path
    resume_witness_path: Path
    checkpoint_path: Path
    initial_launch_plan: LaunchPlan
    resume_launch_plan: LaunchPlan
    initial_commands: OwnedTmuxCommands
    resume_commands: OwnedTmuxCommands


def build_lifecycle_matrix() -> tuple[LifecycleCase, ...]:
    """Return the binding stage → method → ZeRO Cartesian order."""
    return tuple(
        LifecycleCase(stage=stage, method=method, zero_stage=zero_stage)
        for stage in ("stage1", "stage2")
        for method in ("full", "lora", "dora")
        for zero_stage in (1, 2, 3)
    )


def build_lifecycle_plans(
    *,
    parent_run_id: str,
    first_port: int,
    result_root: str | Path,
    python_executable: str,
    commit: str = "0" * 40,
    config_digests: dict[int, str] | None = None,
) -> tuple[LifecyclePlan, ...]:
    """Validate all 18 identities and ports before constructing any side effect."""
    _validate_identifier("parent_run_id", parent_run_id)
    if type(first_port) is not int:
        raise ValueError("the 18 ports require an integer first_port")
    ports = tuple(first_port + offset for offset in range(18))
    if (
        len(set(ports)) != 18
        or any(type(port) is not int or not 1024 <= port <= 65535 for port in ports)
    ):
        raise ValueError("all 18 ports must be unique and between 1024 and 65535")
    root = Path(result_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("result_root must be an absolute path without traversal")
    if not isinstance(python_executable, str) or not python_executable:
        raise ValueError("python_executable must be nonempty")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("commit must be a lowercase 40-character Git object ID")
    digests = config_digests or {1: "0" * 64, 2: "0" * 64, 3: "0" * 64}
    if set(digests) != {1, 2, 3}:
        raise ValueError("config digests must cover ZeRO stages 1, 2, and 3")
    stage2_peft_zero3_digest = (
        "0" * 64
        if config_digests is None
        else load_deepspeed_runtime_profile(
            canonical_deepspeed_config_path(3).with_name(
                "deepspeed_zero3_stage2_peft.json"
            ),
            world_size=4,
        ).config_digest
    )

    plans: list[LifecyclePlan] = []
    for case, port in zip(build_lifecycle_matrix(), ports, strict=True):
        child_run_id = (
            f"{parent_run_id}-{case.stage}-{case.method}-zero{case.zero_stage}"
        )
        _validate_identifier("child_run_id", child_run_id)
        phase_plans = {
            phase: LaunchPlan(
                stage=case.stage,
                tuning_method=case.method,
                zero_stage=case.zero_stage,
                run_id=child_run_id,
                main_process_port=port,
                world_size=4,
                deepspeed_config=selected_deepspeed_config(
                    case.stage, case.method, case.zero_stage
                ),
                deepspeed_config_digest=(
                    stage2_peft_zero3_digest
                    if case.stage == "stage2"
                    and case.method in ("lora", "dora")
                    and case.zero_stage == 3
                    else digests[case.zero_stage]
                ),
                cluster=ClusterSpec.default(),
                attempt_id=phase,
            )
            for phase in ("initial", "resume")
        }
        phase_commands = {
            phase: build_owned_lifecycle_harness_commands(
                phase_plans[phase], commit=commit, phase=phase
            )
            for phase in ("initial", "resume")
        }
        raw_root = _REPOSITORY_ROOT / "results" / child_run_id
        plans.append(
            LifecyclePlan(
                case=case,
                child_run_id=child_run_id,
                main_process_port=port,
                witness_path=root / parent_run_id / f"{child_run_id}.json",
                initial_witness_path=raw_root / "initial-witness.json",
                resume_witness_path=raw_root / "resume-witness.json",
                checkpoint_path=(
                    _REPOSITORY_ROOT
                    / "checkpoints"
                    / child_run_id
                    / f"checkpoint-{case.initial_steps}"
                ),
                initial_launch_plan=phase_plans["initial"],
                resume_launch_plan=phase_plans["resume"],
                initial_commands=phase_commands["initial"],
                resume_commands=phase_commands["resume"],
            )
        )
    return tuple(plans)


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


def selected_deepspeed_config(
    stage: Literal["stage1", "stage2"],
    tuning_method: Literal["full", "lora", "dora"],
    zero_stage: Literal[1, 2, 3],
) -> Path:
    """Select the one dedicated GPU-only policy for Stage 2 PEFT ZeRO-3."""
    if stage not in ("stage1", "stage2"):
        raise ValueError("stage must be 'stage1' or 'stage2'")
    if tuning_method not in ("full", "lora", "dora"):
        raise ValueError("tuning_method must be 'full', 'lora', or 'dora'")
    canonical = canonical_deepspeed_config(zero_stage)
    if stage == "stage2" and tuning_method in ("lora", "dora") and zero_stage == 3:
        return canonical.with_name("deepspeed_zero3_stage2_peft.json")
    return canonical


def validate_stage2_checkpointing_policy(
    *, tuning_method: str, zero_stage: int, gradient_checkpointing: str
) -> None:
    """Fail before launch when the unsupported PEFT ZeRO-3 GC path is requested."""
    if gradient_checkpointing not in ("true", "false"):
        raise ValueError("GRADIENT_CHECKPOINTING must be true or false")
    if (
        tuning_method in ("lora", "dora")
        and zero_stage == 3
        and gradient_checkpointing == "true"
    ):
        raise ValueError(
            "Stage 2 PEFT ZeRO-3 requires GRADIENT_CHECKPOINTING=false"
        )


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
    config_path = selected_deepspeed_config(stage, tuning_method, zero_stage)
    source_config_path = (
        canonical_deepspeed_config_path(zero_stage)
        if config_path.name == f"deepspeed_zero{zero_stage}.json"
        else canonical_deepspeed_config_path(zero_stage).with_name(config_path.name)
    )
    profile = load_deepspeed_runtime_profile(
        source_config_path, world_size=cluster.num_processes
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


def _validate_lifecycle_phase_witness(
    plan: LifecyclePlan, *, phase: Literal["initial", "resume"], raw: bytes
) -> dict[str, object]:
    """Validate one separately observed phase witness from either node."""
    from src.multinode_evidence import decode_lifecycle_phase_evidence

    return decode_lifecycle_phase_evidence(
        raw,
        phase=phase,
        stage=plan.case.stage,
        method=plan.case.method,
        zero_stage=plan.case.zero_stage,
        config_digest=plan.initial_launch_plan.deepspeed_config_digest,
        experiment_id=plan.child_run_id,
        checkpoint_path=plan.checkpoint_path,
    )


def _execute_lifecycle_attempts(
    plan: LifecyclePlan,
    *,
    initial_runner: object,
    resume_runner: object,
    commit: str,
) -> dict[str, object]:
    """Run initial, collect/redistribute, then and only then run resume."""
    phase_results: dict[str, object] = {}
    initial = plan.initial_launch_plan
    start_rank0_then_rank1(initial, initial_runner, commands=plan.initial_commands)
    initial_exits = monitor_owned_jobs(
        initial,
        initial_runner,
        expected_commit=commit,
        expected_command_digests={
            0: plan.initial_commands.command_digests[0],
            1: plan.initial_commands.command_digests[1],
        },
    )
    if initial_exits != {0: 0, 1: 0}:
        raise RuntimeError(f"initial lifecycle phase failed: {initial_exits}")
    initial_copies = tuple(
        initial_runner.read_lifecycle_phase_evidence(rank, "initial")
        for rank in (0, 1)
    )
    if initial_copies[0] != initial_copies[1]:
        raise ValueError("initial lifecycle phase differs between node filesystems")
    initial_payload = _validate_lifecycle_phase_witness(
        plan, phase="initial", raw=initial_copies[0]
    )
    artifact_identity = _checkpoint_artifact_identity(
        initial, commit=commit, global_step=2
    )
    manifests = collect_checkpoint_artifacts(
        initial_runner,
        checkpoint_path=plan.checkpoint_path,
        runtime_identity=artifact_identity,
        exit_codes=initial_exits,
        world_size=4,
    )
    redistributed = redistribute_resume_checkpoint(
        initial_runner,
        checkpoint_path=plan.checkpoint_path,
        checkpoint_identity=artifact_identity,
        requested_identity=artifact_identity,
        destination=plan.checkpoint_path,
    )

    resume = plan.resume_launch_plan
    start_rank0_then_rank1(resume, resume_runner, commands=plan.resume_commands)
    resume_exits = monitor_owned_jobs(
        resume,
        resume_runner,
        expected_commit=commit,
        expected_command_digests={
            0: plan.resume_commands.command_digests[0],
            1: plan.resume_commands.command_digests[1],
        },
    )
    if resume_exits != {0: 0, 1: 0}:
        raise RuntimeError(f"resume lifecycle phase failed: {resume_exits}")
    resume_copies = tuple(
        resume_runner.read_lifecycle_phase_evidence(rank, "resume")
        for rank in (0, 1)
    )
    if resume_copies[0] != resume_copies[1]:
        raise ValueError("resume lifecycle phase differs between node filesystems")
    resume_payload = _validate_lifecycle_phase_witness(
        plan, phase="resume", raw=resume_copies[0]
    )
    phase_results.update({
        "initial_payload": initial_payload,
        "initial_bytes": initial_copies[0],
        "initial_exit_codes": initial_exits,
        "resume_payload": resume_payload,
        "resume_bytes": resume_copies[0],
        "resume_exit_codes": resume_exits,
        "manifests": manifests,
        "redistributed": redistributed,
    })
    return phase_results


def run_lifecycle_matrix(
    *,
    parent_run_id: str,
    first_port: int,
    result_root: str | Path,
    python_executable: str,
    dry_run: bool = False,
) -> tuple[LifecyclePlan, ...]:
    """Run each fixed tiny harness serially and complete only after all 18."""
    plans = build_lifecycle_plans(
        parent_run_id=parent_run_id,
        first_port=first_port,
        result_root=result_root,
        python_executable=python_executable,
    )
    if dry_run:
        return plans
    root = Path(result_root)
    matrix_root = root / parent_run_id
    completion_path = matrix_root / "matrix-complete.json"
    if os.path.lexists(matrix_root):
        raise FileExistsError(
            f"parent result namespace must be fresh: {matrix_root}"
        )
    if os.path.lexists(completion_path):
        raise FileExistsError(f"matrix completion must be fresh: {completion_path}")
    commit = _repository_git_stdout(("rev-parse", "HEAD"))
    digests = {
        stage: load_deepspeed_runtime_profile(
            canonical_deepspeed_config_path(stage), world_size=4
        ).config_digest
        for stage in (1, 2, 3)
    }
    plans = build_lifecycle_plans(
        parent_run_id=parent_run_id,
        first_port=first_port,
        result_root=result_root,
        python_executable=python_executable,
        commit=commit,
        config_digests=digests,
    )
    matrix_root.mkdir(parents=True, exist_ok=False)

    from src.multinode_runtime import _canonical_json_bytes, _publish_bytes_no_replace

    completed_cases: list[dict[str, object]] = []
    for plan in plans:
        launch = plan.initial_launch_plan
        checkpoint_root = _REPOSITORY_ROOT / "checkpoints"
        control_root = _REPOSITORY_ROOT / "logs" / "multinode" / plan.child_run_id
        harness = (
            _REPOSITORY_ROOT / "tests/stage1/zero_compat_diagnostic.py"
            if plan.case.stage == "stage1"
            else _REPOSITORY_ROOT / "tests/stage2/integration/tiny_stage2_train.py"
        )
        run_paths = (
            control_root / "initial",
            control_root / "resume",
            plan.initial_witness_path.parent,
            plan.checkpoint_path.parent,
            plan.witness_path,
        )
        remote_sessions = (
            f"ilvr-{plan.child_run_id}-initial-rank0",
            f"ilvr-{plan.child_run_id}-resume-rank0",
        )
        local_sessions = (
            f"ilvr-{plan.child_run_id}-initial-rank1",
            f"ilvr-{plan.child_run_id}-resume-rank1",
        )
        preflight = build_preflight_commands(
            launch,
            input_paths=(harness, launch.deepspeed_config),
            artifact_roots=(
                checkpoint_root, _REPOSITORY_ROOT / "results", _REPOSITORY_ROOT / "logs"
            ),
            remote_run_paths=run_paths,
            local_run_paths=run_paths,
            remote_tmux_sessions=remote_sessions,
            local_tmux_sessions=local_sessions,
        )

        def preflight_runner(command):
            return subprocess.run(
                command, capture_output=True, text=True, check=False,
                timeout=_CONTROL_TIMEOUT_SECONDS,
            )

        remote = collect_node_snapshot(
            launch, node="remote", runner=preflight_runner, commands=preflight,
            input_paths=(harness, launch.deepspeed_config),
            artifact_roots=(
                checkpoint_root, _REPOSITORY_ROOT / "results", _REPOSITORY_ROOT / "logs"
            ),
            run_paths=run_paths, tmux_sessions=remote_sessions,
            repository=launch.cluster.remote_repo_root,
        )
        local = collect_node_snapshot(
            launch, node="local", runner=preflight_runner, commands=preflight,
            input_paths=(harness, launch.deepspeed_config),
            artifact_roots=(
                checkpoint_root, _REPOSITORY_ROOT / "results", _REPOSITORY_ROOT / "logs"
            ),
            run_paths=run_paths, tmux_sessions=local_sessions,
            repository=launch.cluster.remote_repo_root,
        )
        report = validate_preflight(remote, local)
        if report.commit != commit:
            raise ValueError("validated lifecycle preflight commit changed")
        initial_runner = SubprocessCoordinatorRunner(
            plan.initial_launch_plan, commands=plan.initial_commands, commit=commit,
            log_root=_REPOSITORY_ROOT / "logs" / "multinode",
            checkpoint_root=checkpoint_root,
        )
        resume_runner = SubprocessCoordinatorRunner(
            plan.resume_launch_plan, commands=plan.resume_commands, commit=commit,
            log_root=_REPOSITORY_ROOT / "logs" / "multinode",
            checkpoint_root=checkpoint_root,
        )
        observed = _execute_lifecycle_attempts(
            plan, initial_runner=initial_runner, resume_runner=resume_runner,
            commit=commit,
        )
        case = plan.case
        initial_payload = observed["initial_payload"]
        resume_payload = observed["resume_payload"]
        initial_bytes = observed["initial_bytes"]
        resume_bytes = observed["resume_bytes"]
        initial_exits = observed["initial_exit_codes"]
        resume_exits = observed["resume_exit_codes"]
        manifests = observed["manifests"]
        redistributed = observed["redistributed"]
        if (
            type(initial_payload) is not dict or type(resume_payload) is not dict
            or type(initial_bytes) is not bytes or type(resume_bytes) is not bytes
            or type(initial_exits) is not dict or type(resume_exits) is not dict
        ):
            raise TypeError("validated lifecycle phase result types are invalid")
        if initial_payload["artifact_kind"] != resume_payload["artifact_kind"]:
            raise ValueError("lifecycle phase artifact observations disagree")
        case_payload = {
            "schema_version": 1,
            "kind": "multinode-lifecycle-case",
            "parent_run_id": parent_run_id,
            "child_run_id": plan.child_run_id,
            "stage": case.stage,
            "method": case.method,
            "zero_stage": case.zero_stage,
            "world_size": 4,
            "main_process_port": plan.main_process_port,
            "machine_ranks": [0, 1],
            "process_ranks": initial_payload["process_ranks"],
            "artifact_kind": initial_payload["artifact_kind"],
            "checkpoint": "checkpoint-2",
            "raw_witnesses": {
                "initial": {
                    "path": str(plan.initial_witness_path),
                    "sha256": hashlib.sha256(initial_bytes).hexdigest(),
                    "size": len(initial_bytes),
                },
                "resume": {
                    "path": str(plan.resume_witness_path),
                    "sha256": hashlib.sha256(resume_bytes).hexdigest(),
                    "size": len(resume_bytes),
                },
            },
            "attempts": [
                {
                    "from_step": 0, "to_step": 2,
                    "rank_losses": [record["loss"] for record in initial_payload["ranks"]],
                    "rank_gradients": [record["gradient_norm"] for record in initial_payload["ranks"]],
                },
                {
                    "from_step": 2, "to_step": 3,
                    "rank_losses": [record["loss"] for record in resume_payload["ranks"]],
                    "rank_gradients": [record["gradient_norm"] for record in resume_payload["ranks"]],
                },
            ],
            "finite_loss": True,
            "finite_gradient": True,
            "worker_errors": [],
            "collected_node_manifests": len(manifests),
            "redistributed_entries": len(redistributed),
            "exit_codes": {
                "initial": {str(rank): initial_exits[rank] for rank in (0, 1)},
                "resume": {str(rank): resume_exits[rank] for rank in (0, 1)},
            },
        }
        _publish_bytes_no_replace(
            plan.witness_path,
            _canonical_json_bytes(case_payload) + b"\n",
        )
        completed_cases.append(
            {
                "stage": case.stage,
                "method": case.method,
                "zero_stage": case.zero_stage,
                "child_run_id": plan.child_run_id,
                "main_process_port": plan.main_process_port,
                "witness": plan.witness_path.name,
                "exit_codes": {"initial": [0, 0], "resume": [0, 0]},
            }
        )
    completion = {
        "schema_version": 1,
        "kind": "multinode-lifecycle-matrix-complete",
        "run_id": parent_run_id,
        "world_size": 4,
        "first_port": first_port,
        "cases": completed_cases,
    }
    _publish_bytes_no_replace(
        completion_path,
        _canonical_json_bytes(completion) + b"\n",
    )
    return plans


def _lifecycle_plan_json(plan: LifecyclePlan) -> dict[str, object]:
    return {
        "stage": plan.case.stage,
        "method": plan.case.method,
        "zero_stage": plan.case.zero_stage,
        "child_run_id": plan.child_run_id,
        "main_process_port": plan.main_process_port,
        "lifecycle": {
            "initial": {
                "remote": list(plan.initial_commands.remote),
                "local": list(plan.initial_commands.local),
                "steps": [0, 2],
            },
            "redistribute": {
                "checkpoint": "checkpoint-2",
                "remote_path": str(plan.checkpoint_path),
                "local_path": str(plan.checkpoint_path),
            },
            "resume": {
                "remote": list(plan.resume_commands.remote),
                "local": list(plan.resume_commands.local),
                "steps": [2, 3],
            },
        },
    }


def _lifecycle_matrix_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.multinode lifecycle-matrix", allow_abbrev=False
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--first-port", required=True, type=int)
    parser.add_argument("--result-root", default=str(_REPOSITORY_ROOT / "results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        plans = run_lifecycle_matrix(
            parent_run_id=args.run_id,
            first_port=args.first_port,
            result_root=args.result_root,
            python_executable=sys.executable,
            dry_run=args.dry_run,
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.dry_run:
        for plan in plans:
            print(json.dumps(_lifecycle_plan_json(plan), sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "matrix_complete": str(
                        Path(args.result_root) / args.run_id / "matrix-complete.json"
                    )
                },
                sort_keys=True,
            )
        )
    return 0


def _lifecycle_evidence_show(argv: list[str]) -> int:
    """Emit one bounded canonical node-local phase witness for comparison."""
    parser = argparse.ArgumentParser(
        prog="python -m src.multinode lifecycle-evidence-show",
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=("initial", "resume"), required=True)
    args = parser.parse_args(argv)
    try:
        _validate_identifier("run_id", args.run_id)
        from src.multinode_runtime import _read_regular_file

        raw = _read_regular_file(
            _REPOSITORY_ROOT / "results" / args.run_id
            / f"{args.phase}-witness.json",
            maximum_size=1024 * 1024,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.buffer.write(raw)
    return 0


def _stage1_training_evidence_show(argv: list[str]) -> int:
    """Emit one bounded canonical ordinary Stage 1 producer from remote rank zero."""
    parser = argparse.ArgumentParser(
        prog="python -m src.multinode stage1-training-evidence-show",
        allow_abbrev=False,
    )
    parser.add_argument("--path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args(argv)
    try:
        _validate_identifier("run_id", args.run_id)
        _validate_identifier("attempt_id", args.attempt_id)
        path = Path(args.path)
        result_root = _REPOSITORY_ROOT / "results"
        if (
            not path.is_absolute()
            or ".." in path.parts
            or path.parent.name != args.run_id
            or path.name != f"{args.attempt_id}-producer-evidence.json"
        ):
            raise ValueError("Stage 1 training evidence path identity is invalid")
        path.relative_to(result_root)
        from src.multinode_runtime import _read_regular_file

        raw = _read_regular_file(path, maximum_size=1024 * 1024)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.buffer.write(raw)
    return 0


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
        raise ValueError(f"node-info repository must be {_REPOSITORY_ROOT}")
    expected_config = selected_deepspeed_config(
        args.stage, args.tuning_method, args.zero_stage
    )
    if Path(args.deepspeed_config) != expected_config:
        raise ValueError("node-info config must match the selected DeepSpeed policy")
    validate_node_config_digest(
        args.deepspeed_config,
        zero_stage=args.zero_stage,
        expected_digest=args.deepspeed_config_digest,
    )

    commit = _repository_git_stdout(("rev-parse", "HEAD"))
    tracked_clean = not _repository_git_stdout(
        ("status", "--porcelain", "--untracked-files=no")
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


def _repository_git_stdout(arguments: tuple[str, ...]) -> str:
    """Run Git only against the fixed canonical repository."""
    return _command_stdout(
        (
            "git",
            "-c",
            f"safe.directory={_REPOSITORY_ROOT}",
            "-C",
            str(_REPOSITORY_ROOT),
            *arguments,
        ),
        cwd=_REPOSITORY_ROOT,
    )


def _interface_name() -> str:
    completed = subprocess.run(
        ("ip", "-o", "link", "show", "bond0.11"),
        capture_output=True,
        text=True,
        check=False,
        timeout=_CONTROL_TIMEOUT_SECONDS,
    )
    return "bond0.11" if completed.returncode == 0 else ""


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


def _node_run_control_paths(
    *,
    run_id: str,
    attempt_id: str,
    machine_rank: int,
    ownership_path: str,
    exit_path: str,
    evaluation_shard_index: int | None = None,
) -> tuple[Path, Path]:
    """Bind child logs to the immutable canonical run/rank identity."""
    _validate_identifier("run_id", run_id)
    _validate_identifier("attempt_id", attempt_id)
    attempt_directory = (
        _REPOSITORY_ROOT / "logs" / "multinode" / run_id / attempt_id
    )
    if evaluation_shard_index is None:
        control_identity = f"rank{machine_rank}"
    else:
        if (
            type(evaluation_shard_index) is not int
            or evaluation_shard_index not in range(4)
        ):
            raise ValueError("evaluation shard index must be 0, 1, 2, or 3")
        expected_machine_rank = 0 if evaluation_shard_index < 2 else 1
        if machine_rank != expected_machine_rank:
            raise ValueError("evaluation shard does not belong to the machine rank")
        control_identity = f"eval-shard{evaluation_shard_index}"
    expected_ownership = attempt_directory / f"ownership-{control_identity}.json"
    expected_exit = attempt_directory / f"exit-{control_identity}.json"
    if Path(ownership_path) != expected_ownership or Path(exit_path) != expected_exit:
        raise ValueError("node-run control paths must use the canonical attempt directory")
    return (
        attempt_directory / f"stdout-{control_identity}.log",
        attempt_directory / f"stderr-{control_identity}.log",
    )


def _open_owned_child_log(path: Path):
    """Open one fresh rank log through a trusted parent without following links."""
    parent_descriptor = _open_trusted_directory(path.parent)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | os.O_CLOEXEC,
            0o600,
            dir_fd=parent_descriptor,
        )
        os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "wb", buffering=0)
        descriptor = None
        return stream
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _node_run(argv: list[str]) -> int:
    if "--" not in argv:
        raise ValueError("node-run requires -- followed by the Stage launcher command")
    separator = argv.index("--")
    parser = argparse.ArgumentParser(prog="python -m src.multinode node-run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--machine-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--evaluation-shard-index", type=int, choices=range(4))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--ownership-path", required=True)
    parser.add_argument("--exit-path", required=True)
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args(argv[:separator])
    command = tuple(argv[separator + 1 :])
    if not command or not Path(command[0]).is_absolute():
        raise ValueError("owned Stage command must use an absolute executable")
    stdout_path, stderr_path = _node_run_control_paths(
        run_id=args.run_id,
        attempt_id=args.attempt_id,
        machine_rank=args.machine_rank,
        ownership_path=args.ownership_path,
        exit_path=args.exit_path,
        evaluation_shard_index=args.evaluation_shard_index,
    )
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
    with _open_owned_child_log(stdout_path) as child_stdout, _open_owned_child_log(
        stderr_path
    ) as child_stderr:
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
                stdout=child_stdout,
                stderr=child_stderr,
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


def _stop_owned_main(argv: list[str]) -> int:
    """Stop all exact live attempts for one run on this machine rank."""
    from src.multinode_runtime import stop_owned_attempts

    parser = argparse.ArgumentParser(
        prog="python -m src.multinode stop-owned", allow_abbrev=False
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--machine-rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args(argv)
    try:
        summary = stop_owned_attempts(
            run_id=args.run_id,
            machine_rank=args.machine_rank,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))
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


def _dil_full_model_probe_main(argv: list[str]) -> int:
    """Fixed diagnostic terminal; ordinary training outputs are never consulted."""
    from scripts.probe_dil_full_model import load_manifest, validate_rank_records
    from src.multinode_runtime import build_dil_full_model_probe_commands, publish_small_file, _publish_bytes_no_replace

    parser = argparse.ArgumentParser(prog="train-stage2 --dil-full-model-probe", allow_abbrev=False)
    parser.add_argument("--dil-full-model-probe", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", choices=("initial",), default="initial")
    parser.add_argument("--zero-stage", type=int, choices=(2,), required=True)
    parser.add_argument("--main-process-port", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if any(os.environ.get(name) for name in ("RESUME_CHECKPOINT", "RESUME_FROM_CHECKPOINT", "STAGE2_STOP_AFTER_CHECKPOINT")):
            raise ValueError("full-model diagnostic rejects training resume/update settings")
        manifest_path = Path(args.dil_full_model_probe)
        manifest = load_manifest(manifest_path, _REPOSITORY_ROOT)
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        plan = build_launch_plan(stage="stage2", tuning_method="dora", zero_stage=2, run_id=args.run_id,
                                 main_process_port=args.main_process_port)
        commit = _repository_git_stdout(("rev-parse", "HEAD"))
        commands = build_dil_full_model_probe_commands(plan, commit=commit, manifest=manifest_path, manifest_sha256=digest)
        from scripts.probe_dil_full_model import SOURCE_LOCAL, SOURCE_REMOTE
        selected_source = SOURCE_LOCAL if manifest["variant"] != "portTorch" else _REPOSITORY_ROOT
        remote_source = SOURCE_REMOTE if manifest["variant"] != "portTorch" else _REPOSITORY_ROOT
        common_inputs = (manifest_path, _REPOSITORY_ROOT / "scripts/probe_dil_full_model.py",
                         *((_REPOSITORY_ROOT / "scripts/dil_boundary_observation.py",) if manifest.get("boundary_observation", False) else ()),
                         plan.deepspeed_config,
                         Path(manifest["data_path"]), Path(manifest["image_root"]),
                         Path(manifest["initializer"]), Path(manifest["base"]))
        roots = (_REPOSITORY_ROOT / "checkpoints", _REPOSITORY_ROOT / "results", _REPOSITORY_ROOT / "logs")
        run_paths = (_REPOSITORY_ROOT / "checkpoints" / plan.run_id,
                     _REPOSITORY_ROOT / "results" / plan.run_id,
                     _REPOSITORY_ROOT / "logs/multinode" / plan.run_id / plan.attempt_id)
        sessions = tuple(f"ilvr-{plan.run_id}-initial-rank{rank}" for rank in (0, 1))
        # Common ordinary node resource preflight; source root identity has a
        # node-specific authenticated preflight immediately before owned launch.
        preflight = build_preflight_commands(plan, input_paths=common_inputs, artifact_roots=roots,
                                            remote_run_paths=run_paths, local_run_paths=run_paths,
                                            remote_tmux_sessions=(sessions[0],), local_tmux_sessions=(sessions[1],))
        preflight_commands = []
        from src.multinode_runtime import build_remote_exec
        for machine_rank in (0, 1):
            command = (str(_REPOSITORY_ROOT / "venv/ilvr-venv/bin/python"),
                       str(_REPOSITORY_ROOT / "scripts/probe_dil_full_model.py"), "preflight",
                       "--repository", str(_REPOSITORY_ROOT), "--manifest", str(manifest_path),
                       "--manifest-sha256", digest, "--machine-rank", str(machine_rank))
            preflight_commands.append(tuple(build_remote_exec(command)) if machine_rank == 0 else command)
        if args.dry_run:
            print(json.dumps({"kind": "dil-full-model-probe", "manifest_sha256": digest, "variant": manifest["variant"],
                              "profile": manifest["profile"], "comparator": manifest["comparator"],
                              "local": commands.local, "remote": commands.remote, "plan": _json_ready(plan),
                              "local_source": str(selected_source), "remote_source": str(remote_source),
                              "identity_preflight": preflight_commands, "resource_preflight": _json_ready(preflight)}, sort_keys=True))
            return 0
        def run(command):
            return subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)
        remote = collect_node_snapshot(plan, node="remote", runner=run, commands=preflight,
                                       input_paths=common_inputs, artifact_roots=roots, run_paths=run_paths,
                                       tmux_sessions=(sessions[0],), repository=plan.cluster.remote_repo_root)
        local = collect_node_snapshot(plan, node="local", runner=run, commands=preflight,
                                      input_paths=common_inputs, artifact_roots=roots, run_paths=run_paths,
                                      tmux_sessions=(sessions[1],), repository=plan.cluster.remote_repo_root)
        if validate_preflight(remote, local).commit != commit:
            raise ValueError("probe preflight commit changed")
        for command in preflight_commands:
            completed = run(command)
            if completed.returncode:
                raise RuntimeError(f"probe identity preflight failed: {completed.stderr}")
            observed = json.loads(completed.stdout)
            if observed["manifest_sha256"] != digest or observed["variant"] != manifest["variant"]:
                raise ValueError("probe identity preflight changed")
        runner = SubprocessCoordinatorRunner(plan, commands=commands, commit=commit,
                                             log_root=_REPOSITORY_ROOT / "logs/multinode")
        start_rank0_then_rank1(plan, runner, commands=commands, ready_timeout_seconds=600)
        exits = monitor_owned_jobs(plan, runner, expected_commit=commit,
                                   expected_command_digests={0: commands.command_digests[0], 1: commands.command_digests[1]})
        if exits != {0: 0, 1: 0}:
            raise RuntimeError(f"full-model diagnostic failed: {exits}")
        records = [json.loads(runner.read_dil_full_model_probe_record(rank)) for rank in range(4)]
        witness = validate_rank_records(records, plan.run_id, digest, manifest["variant"],
                                        boundary_observation=manifest.get("boundary_observation", False))
        result_dir = _REPOSITORY_ROOT / "results" / plan.run_id
        for rank in (0, 1):
            _publish_bytes_no_replace(result_dir / f"probe-rank-{rank}.json", json.dumps(records[rank], sort_keys=True).encode())
        witness_path = result_dir / "initial-probe-witness.json"
        # Four rank files remain the full evidence; keep the terminal metadata bounded.
        witness["ranks"] = [{key: record[key] for key in ("rank", "host", "device", "machine_rank", "local_rank")}
                            for record in witness["ranks"]]
        publish_small_file(witness_path, json.dumps(witness, sort_keys=True).encode())
        print(json.dumps({"probe_witness": str(witness_path)}, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


def _training_main(stage: Literal["stage1", "stage2"], argv: list[str]) -> int:
    if stage == "stage2" and any(value == "--dil-full-model-probe" or value.startswith("--dil-full-model-probe=") for value in argv):
        return _dil_full_model_probe_main(argv)
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
                f"result_root must be inside {_REPOSITORY_ROOT / 'results'}"
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
            validate_stage2_checkpointing_policy(
                tuning_method=tuning_method,
                zero_stage=args.zero_stage,
                gradient_checkpointing=os.environ.get(
                    "GRADIENT_CHECKPOINTING", "false"
                ),
            )
            output_dir = _REPOSITORY_ROOT / "checkpoints" / args.run_id
            log_file = (
                _REPOSITORY_ROOT
                / "logs"
                / "zero-stage-compatibility"
                / args.run_id
                / f"{args.attempt_id}-train.log"
            )
            producer_evidence = (
                result_root
                / args.run_id
                / f"{args.attempt_id}-producer-evidence.json"
            )
            stage_environment = validated_stage2_environment(
                os.environ,
                output_dir=str(output_dir),
                log_file=str(log_file),
                deepspeed_config=str(
                    selected_deepspeed_config(
                        stage, tuning_method, args.zero_stage
                    )
                ),
                benchmark_evidence_path=str(producer_evidence),
            )
            resume_requested = "RESUME_CHECKPOINT" in stage_environment
        planned_stop = _argument_value(forwarded, "--stage2_stop_after_checkpoint") if stage == "stage2" else None
        planned_stop_max_steps = None
        if planned_stop is not None:
            budget = _argument_value(forwarded, "--max_steps")
            if not (planned_stop.isdecimal() and budget is not None and budget.isdecimal()
                    and 0 < int(planned_stop) < int(budget)):
                raise ValueError("stage2_stop_after_checkpoint requires 0 < stop < explicit max_steps")
            if resume_requested:
                raise ValueError("stage2_stop_after_checkpoint is fresh-run only")
            planned_stop_max_steps = int(budget)
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
        commit = _repository_git_stdout(("rev-parse", "HEAD"))
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
            if stage_environment.get("STEP_PREFERENCE_ENABLED") == "true":
                input_paths += tuple(
                    Path(stage_environment[name])
                    for name in (
                        "STAGE1_ADAPTER_PATH", "STAGE2A_CHECKPOINT_PATH",
                        "STEP_NEGATIVE_PATH", "STEP_NEGATIVE_MANIFEST"
                    )
                    if stage_environment.get(name)
                )
        artifact_roots = (checkpoint_root, result_root, _REPOSITORY_ROOT / "logs")
        run_paths = [control_directory, witness_path, stage_log_file]
        if commands.finite_loss_evidence_path is not None:
            run_paths.append(commands.finite_loss_evidence_path)
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
        # Stage 2 measures the full dataset before opening the rendezvous port.
        start_rank0_then_rank1(
            plan, runner, commands=commands,
            ready_timeout_seconds=600 if stage == "stage2" else 30,
        )
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
        ordinary_stage1 = (
            stage == "stage1"
            and _argument_value(forwarded, "--benchmark-run-config") is None
        )
        if ordinary_stage1:
            evidence_path = commands.finite_loss_evidence_path
            if evidence_path is None:
                raise ValueError("ordinary Stage 1 producer path is missing")
            from src.multinode_runtime import publish_small_file
            from src.stage1.training_evidence import decode_stage1_training_evidence

            raw_evidence = runner.read_stage1_training_evidence(evidence_path)
            step_value = _argument_value(forwarded, "--max-steps")
            if step_value is None:
                evidence_payload = json.loads(raw_evidence)
                evidence_step = (
                    evidence_payload.get("global_step")
                    if type(evidence_payload) is dict
                    else None
                )
                if type(evidence_step) is not int or evidence_step <= 0:
                    raise ValueError(
                        "Stage 1 training evidence must identify the final checkpoint step"
                    )
                final_step = evidence_step
            elif not step_value.isdecimal() or int(step_value) <= 0:
                raise ValueError(
                    "--max-steps must identify the final checkpoint step"
                )
            else:
                final_step = int(step_value)
            decode_stage1_training_evidence(
                raw_evidence,
                expected_run_id=plan.run_id,
                expected_method=plan.tuning_method,
                expected_zero_stage=plan.zero_stage,
                expected_world_size=plan.world_size,
                expected_checkpoint=(
                    checkpoint_run_path / f"checkpoint-{final_step}"
                ),
                expected_global_step=final_step,
                expected_micro_batch=int(
                    _argument_value(forwarded, "--micro-batch") or "1"
                ),
                expected_gradient_accumulation_steps=int(
                    _argument_value(forwarded, "--gradient-accumulation") or "2"
                ),
                expected_ema_tau=float(
                    _argument_value(forwarded, "--ema-tau") or "1.0"
                ),
                expected_seed=int(_argument_value(forwarded, "--seed") or "42"),
            )
            publish_small_file(
                evidence_path,
                raw_evidence,
            )
        else:
            step_option = "--max-steps" if stage == "stage1" else "--max_steps"
            step_value = _argument_value(forwarded, step_option)
            if (
                step_value is None
                or not step_value.isdecimal()
                or int(step_value) <= 0
            ):
                raise ValueError(
                    f"{step_option} must identify the final checkpoint step"
                )
            final_step = int(planned_stop) if planned_stop is not None else int(step_value)
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
            batch_value = _argument_value(forwarded, "--micro-batch") or "1"
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
                expected_global_step=final_step,
                **({"planned_stop_max_steps": planned_stop_max_steps} if planned_stop is not None else {}),
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
        staging_root = _evaluation_staging_root()
        if not destination.is_absolute() or ".." in destination.parts:
            raise ValueError("inference destination must be an absolute trusted path")
        try:
            destination.relative_to(staging_root / plan.run_id)
        except ValueError as error:
            raise ValueError(
                "inference destination must belong to the run checkpoint "
                "evaluation-staging tree"
            ) from error
        commit = _repository_git_stdout(("rev-parse", "HEAD"))
        runner = SubprocessCoordinatorRunner(
            plan,
            commands=None,
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
        resolve_local_checkpoint_destination,
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
        try:
            model_dir.relative_to(_evaluation_staging_root())
        except ValueError as error:
            raise ValueError(
                "prepared inference model must be under the checkpoint "
                "evaluation-staging root"
            ) from error
        physical_model_dir = resolve_local_checkpoint_destination(model_dir)
        payload = json.loads(
            _read_regular_file(inference_manifest_path(physical_model_dir))
        )
        manifest = NodeArtifactManifest.from_json(payload)
        inference_model_identity(manifest)
        validate_transferred_entries(physical_model_dir, manifest.entries)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(manifest.to_json(), sort_keys=True))
    return 0


def _prepare_inference_destination_main(argv: list[str]) -> int:
    """Reserve the identical fresh remote checkpoint path before a bounded push."""
    from src.multinode_artifacts import (
        inference_manifest_path,
        reserve_checkpoint_destination,
        validate_coordinator_relative_path,
    )

    parser = argparse.ArgumentParser(
        prog="python -m src.multinode prepare-inference-destination",
        allow_abbrev=False,
    )
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    try:
        destination = Path(args.destination)
        try:
            relative = destination.relative_to(_evaluation_staging_root())
        except ValueError as error:
            raise ValueError(
                "prepared inference destination must be under the checkpoint "
                "evaluation-staging root"
            ) from error
        if len(relative.parts) < 2:
            raise ValueError("prepared inference destination must belong to a run")
        validate_coordinator_relative_path(
            relative, label="prepared inference destination"
        )
        _validate_identifier("run_id", relative.parts[0])
        if os.path.lexists(inference_manifest_path(destination)):
            raise FileExistsError(
                "prepared inference identity manifest already exists"
            )
        reserve_checkpoint_destination(destination)
    except (OSError, ValueError) as error:
        parser.error(str(error))
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
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-sequence-length", type=int, default=32768)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--generation-limit-policy",
        choices=("bounded", "external_full_latent_block"),
        default="bounded",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--vision-ablation", choices=("none", "all", "interleaved"), default="none"
    )
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
            max_sequence_length=args.max_sequence_length,
            repetition_penalty=args.repetition_penalty,
            generation_limit_policy=args.generation_limit_policy,
            temperature=args.temperature,
            top_p=args.top_p,
            vision_ablation=args.vision_ablation,
        )
        if args.dry_run:
            dry_identity = f"task6:{'0' * 64}"
            variants = {}
            for stage in ("stage1", "stage2"):
                try:
                    variants[stage] = _build_evaluation_commands_for_stage(
                        plan,
                        stage=stage,
                        model_identity=dry_identity,
                        commit="0" * 40,
                    )
                except ValueError as error:
                    variants[stage] = {"unsupported": str(error)}
            first = variants["stage2"] if args.vision_ablation != "none" else variants["stage1"]
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
                            stage: (
                                commands
                                if isinstance(commands, dict)
                                else [list(worker.launch) for worker in commands.workers]
                            )
                            for stage, commands in variants.items()
                        },
                    },
                    sort_keys=True,
                )
            )
            return 0
        commit = _repository_git_stdout(("rev-parse", "HEAD"))
        runner = EvaluationSubprocessRunner(plan, commit=commit)
        summary = run_evaluation(plan, runner)
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(_json_ready(summary), sort_keys=True))
    return 0


def _main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "lifecycle-matrix":
        return _lifecycle_matrix_main(argv[1:])
    if argv and argv[0] == "validate-results":
        from src.multinode_results import validate_results_main

        return validate_results_main(argv[1:])
    if argv and argv[0] == "capacity-stage1":
        from src.multinode_capacity import capacity_stage1_main

        return capacity_stage1_main(argv[1:])
    if argv and argv[0] == "capacity-evidence-show":
        from src.multinode_capacity import _capacity_evidence_main

        return _capacity_evidence_main(argv[1:])
    if argv and argv[0] == "lifecycle-evidence-show":
        return _lifecycle_evidence_show(argv[1:])
    if argv and argv[0] == "stage1-training-evidence-show":
        return _stage1_training_evidence_show(argv[1:])
    if argv and argv[0] in ("train-stage1", "train-stage2"):
        return _training_main(argv[0].removeprefix("train-"), argv[1:])  # type: ignore[arg-type]
    if argv and argv[0] == "prepare-inference":
        return _prepare_inference_main(argv[1:])
    if argv and argv[0] == "evaluate":
        return _evaluate_main(argv[1:])
    if argv and argv[0] == "prepared-inference-show":
        return _prepared_inference_show(argv[1:])
    if argv and argv[0] == "prepare-inference-destination":
        return _prepare_inference_destination_main(argv[1:])
    if argv and argv[0] == "prepare-evaluation-output":
        return _prepare_evaluation_output(argv[1:])
    if argv and argv[0] == "node-run":
        return _node_run(argv[1:])
    if argv and argv[0] == "node-exit":
        return _write_exit_record(argv[1:])
    if argv and argv[0] == "ownership-show":
        return _ownership_show(argv[1:])
    if argv and argv[0] == "stop-owned":
        return _stop_owned_main(argv[1:])
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
    node_info_parser.add_argument(
        "--stage", choices=("stage1", "stage2"), required=True
    )
    node_info_parser.add_argument(
        "--tuning-method", choices=("full", "lora", "dora"), required=True
    )
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
