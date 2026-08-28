"""Bounded Stage 1 ZeRO capacity-matrix coordination.

This module handles only small configuration, audit, and producer-evidence
witnesses.  It never opens model weights, checkpoint shards, images, or dataset
contents and it never initializes CUDA.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Literal, Mapping, Sequence

from src.deepspeed_runtime import (
    DeepSpeedRuntimeProfile,
    canonical_deepspeed_config_path,
    load_deepspeed_runtime_profile,
    read_deepspeed_runtime_sidecar,
)
from src.artifact_paths import resolve_run_artifact_paths
from src.stage1.benchmark import (
    STAGE1_PRODUCER_SCHEMA_VERSION,
    Stage1BenchmarkEvidence,
    Stage1ProducerIdentity,
    _load_producer_evidence,
    config_digest,
)


AttemptStatus = Literal["success", "capacity_oom", "integration_failure"]
_MATRIX_STAGES = (3, 2, 1)
_MATRIX_METHODS = ("full", "dora")
_DIGEST_LENGTH = 64
_MAX_STDERR_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ZeroBatchCandidate:
    micro_batch: int
    gradient_accumulation_steps: int
    effective_global_batch: int = 16


ZERO_MATRIX_CANDIDATES = (
    ZeroBatchCandidate(8, 1),
    ZeroBatchCandidate(4, 2),
    ZeroBatchCandidate(2, 4),
    ZeroBatchCandidate(1, 8),
)


@dataclass(frozen=True)
class ZeroMatrixAttempt:
    identity: Stage1ProducerIdentity
    status: AttemptStatus
    samples_per_second: float | None
    evidence_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.identity, Stage1ProducerIdentity):
            raise TypeError("identity must be Stage1ProducerIdentity")
        if self.status not in {"success", "capacity_oom", "integration_failure"}:
            raise ValueError("invalid zero matrix attempt status")
        if self.status == "success":
            if (
                type(self.samples_per_second) is not float
                or not math.isfinite(self.samples_per_second)
                or self.samples_per_second < 0
            ):
                raise ValueError("successful attempt requires finite samples_per_second")
        elif self.samples_per_second is not None:
            raise ValueError("failed attempt cannot report samples_per_second")
        if not isinstance(self.evidence_path, Path):
            raise TypeError("evidence_path must be a Path")

    @property
    def micro_batch(self) -> int:
        return self.identity.micro_batch


@dataclass(frozen=True)
class SampleAuditWitness:
    path: Path
    audit_digest: str
    input_stat: dict[str, int]
    row_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("sample audit path must be absolute")
        _require_digest("audit_digest", self.audit_digest)
        if set(self.input_stat) != {"device", "inode", "size", "mtime_ns"}:
            raise ValueError("sample audit requires an exact input stat witness")
        if any(type(value) is not int or value < 0 for value in self.input_stat.values()):
            raise ValueError("sample audit stat values must be non-negative integers")
        if type(self.row_indices) is not tuple or not self.row_indices:
            raise ValueError("sample audit requires selected row indices")
        if len(set(self.row_indices)) != len(self.row_indices) or any(
            type(value) is not int or value < 0 for value in self.row_indices
        ):
            raise ValueError("sample audit row indices must be unique non-negative integers")


def _require_digest(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _atomic_write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> object:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def path_stat_witness(path: Path) -> dict[str, int]:
    """Return the small identity witness for a named source without reading it."""
    stat = Path(path).stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _sample_audit_from_payload(path: Path, payload: object) -> SampleAuditWitness:
    if type(payload) is not dict:
        raise ValueError("sample audit must be a JSON object")
    required = {
        "schema_version",
        "kind",
        "data_path",
        "image_root",
        "requested_count",
        "row_count",
        "input_stat",
        "selected_rows",
        "audit_digest",
    }
    if set(payload) != required:
        raise ValueError("sample audit has incomplete or unexpected fields")
    if payload.get("schema_version") != 1 or payload.get("kind") != "stage1-high-cost-row-audit":
        raise ValueError("unsupported sample audit witness")
    claimed_digest = _require_digest("audit_digest", payload.get("audit_digest"))
    digest_input = dict(payload)
    digest_input.pop("audit_digest")
    if config_digest(digest_input) != claimed_digest:
        raise ValueError("sample audit digest mismatch")
    selected = payload.get("selected_rows")
    if type(selected) is not list or not selected:
        raise ValueError("sample audit has no selected rows")
    indices = []
    for row in selected:
        if type(row) is not dict or type(row.get("source_row_index")) is not int:
            raise ValueError("sample audit selected row is malformed")
        indices.append(row["source_row_index"])
    return SampleAuditWitness(
        path=path.resolve(strict=True),
        audit_digest=payload["audit_digest"],
        input_stat=dict(payload["input_stat"]),
        row_indices=tuple(indices),
    )


def load_sample_audit(path: Path) -> SampleAuditWitness:
    """Load one already-produced small audit witness without following its data."""
    return _sample_audit_from_payload(Path(path), _read_json(Path(path)))


def resolve_sample_audit(
    candidate_audits: Sequence[Path],
    *,
    data_path: Path,
    image_root: Path,
    count: int = 64,
) -> SampleAuditWitness:
    """Return the first exact path/stat/count audit, canonical candidate first."""
    if type(count) is not int or count < 1:
        raise ValueError("sample audit count must be positive")
    expected_data = str(Path(data_path).resolve(strict=True))
    expected_images = str(Path(image_root).resolve(strict=True))
    expected_stat = path_stat_witness(Path(data_path))
    for candidate in candidate_audits:
        path = Path(candidate)
        try:
            payload = _read_json(path)
            witness = _sample_audit_from_payload(path, payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            payload["data_path"] == expected_data
            and payload["image_root"] == expected_images
            and payload["requested_count"] == count
            and payload["input_stat"] == expected_stat
        ):
            return witness
    raise FileNotFoundError("no candidate sample audit has exact path/stat/count identity")


def classify_zero_matrix_attempt(
    *,
    returncode: int,
    stderr: str,
    producer: Stage1BenchmarkEvidence | None = None,
) -> AttemptStatus:
    """Classify one bounded producer exit without interpreting integration errors as OOM."""
    if type(returncode) is not int:
        raise TypeError("returncode must be an integer")
    if type(stderr) is not str:
        raise TypeError("stderr must be text")
    if returncode == 0:
        return (
            "success"
            if producer is not None and producer.status == "success"
            else "integration_failure"
        )
    normalized = stderr.casefold()
    oom_markers = (
        "torch.outofmemoryerror",
        "cuda out of memory",
        "cuda error: out of memory",
        "cudamalloc failed",
        "cuda allocator out of memory",
    )
    return "capacity_oom" if any(marker in normalized for marker in oom_markers) else "integration_failure"


def select_fastest_success(rows: Sequence[ZeroMatrixAttempt]) -> ZeroMatrixAttempt:
    successes = [
        row
        for row in rows
        if row.status == "success" and row.samples_per_second is not None
    ]
    if not successes:
        raise RuntimeError("No successful Stage 1 batch candidate")
    return max(successes, key=lambda row: (row.samples_per_second, row.micro_batch))


def _candidate_key(identity: Stage1ProducerIdentity) -> tuple[int, int]:
    return identity.micro_batch, identity.gradient_accumulation_steps


def select_zero_matrix(rows: Sequence[ZeroMatrixAttempt]) -> ZeroMatrixAttempt:
    """Require one exact official matrix and return its fastest successful row."""
    if len(rows) != len(ZERO_MATRIX_CANDIDATES):
        raise ValueError("selection requires an exact four-candidate matrix")
    expected = [
        (candidate.micro_batch, candidate.gradient_accumulation_steps)
        for candidate in ZERO_MATRIX_CANDIDATES
    ]
    if [_candidate_key(row.identity) for row in rows] != expected:
        raise ValueError("selection requires an exact four-candidate matrix")
    first = rows[0].identity
    shared_names = (
        "method",
        "zero_stage",
        "deepspeed_config_digest",
        "world_size",
        "effective_global_batch",
        "sample_stream_digest",
        "model_id",
        "seed",
    )
    if any(
        any(getattr(row.identity, name) != getattr(first, name) for name in shared_names)
        for row in rows[1:]
    ):
        raise ValueError("selection requires exact shared producer identity")
    return select_fastest_success(rows)


def _attempt_payload(
    row: ZeroMatrixAttempt, *, evidence_digest: str | None
) -> dict[str, object]:
    if evidence_digest is not None:
        _require_digest("evidence_digest", evidence_digest)
    return {
        "identity": row.identity.to_dict(),
        "status": row.status,
        "samples_per_second": row.samples_per_second,
        "evidence_path": str(row.evidence_path),
        "evidence_digest": evidence_digest,
    }


def write_zero_matrix_report(
    matrix_result_dir: Path,
    rows: Sequence[ZeroMatrixAttempt],
    *,
    evidence_digests: Mapping[str, str | None],
) -> Path:
    """Atomically publish the selection from existing authenticated witnesses."""
    selected = select_zero_matrix(rows)
    row_payloads = [
        _attempt_payload(row, evidence_digest=evidence_digests.get(str(row.evidence_path)))
        for row in rows
    ]
    selected_payload = next(
        payload for row, payload in zip(rows, row_payloads) if row == selected
    )
    output = Path(matrix_result_dir) / "zero-matrix-report.json"
    matrices: dict[str, object] = {}
    if output.is_file():
        existing = _read_json(output)
        if type(existing) is not dict or existing.get("kind") != "stage1-zero-matrix-report":
            raise ValueError("existing zero matrix report has invalid identity")
        claimed = _require_digest("report_digest", existing.get("report_digest"))
        digest_input = dict(existing)
        digest_input.pop("report_digest")
        if config_digest(digest_input) != claimed:
            raise ValueError("existing zero matrix report digest mismatch")
        existing_matrices = existing.get("matrices")
        if type(existing_matrices) is dict:
            matrices.update(existing_matrices)
        elif type(existing.get("rows")) is list and type(existing.get("selected")) is dict:
            old_identity = existing["selected"].get("identity")
            if type(old_identity) is not dict:
                raise ValueError("existing zero matrix report selection is malformed")
            old_key = f"zero{old_identity.get('zero_stage')}-{old_identity.get('method')}"
            matrices[old_key] = {
                "rows": existing["rows"],
                "selected": existing["selected"],
            }
        else:
            raise ValueError("existing zero matrix report matrices are malformed")
    key = f"zero{selected.identity.zero_stage}-{selected.identity.method}"
    matrices[key] = {"rows": row_payloads, "selected": selected_payload}
    ordered_keys = [
        f"zero{stage}-{method}"
        for stage in _MATRIX_STAGES
        for method in _MATRIX_METHODS
        if f"zero{stage}-{method}" in matrices
    ]
    aggregate_rows = [
        row
        for matrix_key in ordered_keys
        for row in matrices[matrix_key]["rows"]
    ]
    report = {
        "schema_version": 1,
        "kind": "stage1-zero-matrix-report",
        "matrices": matrices,
        "rows": aggregate_rows,
        "selected": selected_payload,
    }
    report["report_digest"] = config_digest(report)
    _atomic_write_json(output, report)
    return output


def _directory_stat_witness(path: Path) -> dict[str, int]:
    stat = Path(path).stat()
    if not Path(path).is_dir() or Path(path).is_symlink():
        raise ValueError(f"selected run path is not a real directory: {path}")
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


def _checkpoint_step_witness(
    checkpoint_path: Path,
    *,
    expected_step: int,
    identity: Stage1ProducerIdentity,
) -> dict[str, object]:
    checkpoint = Path(checkpoint_path)
    if checkpoint.name != f"checkpoint-{expected_step}":
        raise ValueError(
            f"selected run requires exact checkpoint-{expected_step}: {checkpoint}"
        )
    checkpoint_stat = _directory_stat_witness(checkpoint)
    trainer_state = checkpoint / "trainer_state.json"
    if not trainer_state.is_file() or trainer_state.is_symlink():
        raise ValueError(
            f"selected run checkpoint requires real trainer_state.json: {checkpoint}"
        )
    state = _read_json(trainer_state)
    if type(state) is not dict or state.get("global_step") != expected_step:
        raise ValueError(
            f"selected run checkpoint global_step must be {expected_step}: {checkpoint}"
        )
    # This validator examines only checkpoint metadata/layout; it never hashes or
    # loads model, optimizer, EMA, or RNG tensors.
    from src.stage1.checkpointing import _validate_complete_trainer_state

    _validate_complete_trainer_state(checkpoint)
    runtime = read_deepspeed_runtime_sidecar(checkpoint)
    if (
        runtime.zero_stage != identity.zero_stage
        or runtime.config_digest != identity.deepspeed_config_digest
        or runtime.world_size != identity.world_size
    ):
        raise ValueError("selected checkpoint runtime identity does not match producer")
    runtime_path = checkpoint / "deepspeed-runtime.json"
    return {
        "checkpoint_stat": checkpoint_stat,
        "trainer_state_stat": path_stat_witness(trainer_state),
        "runtime_stat": path_stat_witness(runtime_path),
    }


def write_selected_run_witness(
    path: Path,
    *,
    run_kind: Literal["selected-save", "selected-resume"],
    identity: Stage1ProducerIdentity,
    experiment_id: str,
    checkpoint_dir: Path,
    checkpoint_path: Path,
    result_dir: Path,
    log_dir: Path,
    initial_step: int,
    final_step: int,
    resume_from: Path | None,
) -> Path:
    """Seal one selected numbered checkpoint and its bounded step transition."""
    if run_kind not in {"selected-save", "selected-resume"}:
        raise ValueError("invalid selected run kind")
    if not isinstance(identity, Stage1ProducerIdentity):
        raise TypeError("identity must be Stage1ProducerIdentity")
    if f"-{run_kind}-" not in experiment_id:
        raise ValueError("selected run experiment ID must name its run kind")
    directories = {
        "checkpoint_dir": Path(checkpoint_dir).resolve(strict=True),
        "result_dir": Path(result_dir).resolve(strict=True),
        "log_dir": Path(log_dir).resolve(strict=True),
    }
    if any(directory.name != experiment_id for directory in directories.values()):
        raise ValueError("selected run directories must share one experiment ID")
    owned = resolve_run_artifact_paths(experiment_id)
    if any(
        directories[name] != getattr(owned, name)
        for name in ("checkpoint_dir", "result_dir", "log_dir")
    ):
        raise ValueError("selected run directories must use canonical ownership")
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    if checkpoint.parent != directories["checkpoint_dir"]:
        raise ValueError("selected checkpoint must be immediate in its experiment root")
    if run_kind == "selected-save" and resume_from is not None:
        raise ValueError("selected-save cannot name a resume source")
    if run_kind == "selected-resume" and resume_from is None:
        raise ValueError("selected-resume requires the selected-save checkpoint")
    if run_kind == "selected-save" and (initial_step, final_step) != (0, 3):
        raise ValueError("selected-save must prove optimizer steps 0 to 3")
    if run_kind == "selected-resume" and (initial_step, final_step) != (3, 4):
        raise ValueError("selected-resume must prove optimizer step 3 to 4")
    checkpoint_witness = _checkpoint_step_witness(
        checkpoint, expected_step=final_step, identity=identity
    )
    resolved_resume = None
    resume_witness = None
    if resume_from is not None:
        resolved_resume = Path(resume_from).resolve(strict=True)
        if (
            resolved_resume.parent == directories["checkpoint_dir"]
            or "-selected-save-" not in resolved_resume.parent.name
        ):
            raise ValueError(
                "selected-resume requires checkpoint-3 from a distinct selected-save run"
            )
        resume_witness = _checkpoint_step_witness(
            resolved_resume, expected_step=initial_step, identity=identity
        )
    payload = {
        "schema_version": 2,
        "kind": "stage1-selected-run-witness",
        "run_kind": run_kind,
        "identity": identity.to_dict(),
        "experiment_id": experiment_id,
        **{name: str(directory) for name, directory in directories.items()},
        "checkpoint_path": str(checkpoint),
        "initial_step": initial_step,
        "final_step": final_step,
        "resume_from": None if resolved_resume is None else str(resolved_resume),
        "checkpoint_root_stat": _directory_stat_witness(directories["checkpoint_dir"]),
        "checkpoint_stat": checkpoint_witness,
        "resume_stat": resume_witness,
        "log_stat": _directory_stat_witness(directories["log_dir"]),
    }
    payload["witness_digest"] = config_digest(payload)
    _atomic_write_json(Path(path), payload)
    return Path(path)


def load_reusable_selected_run(
    path: Path,
    *,
    run_kind: Literal["selected-save", "selected-resume"],
    identity: Stage1ProducerIdentity,
) -> dict[str, object] | None:
    """Return an exact selected-run witness without scanning checkpoint contents."""
    try:
        payload = _read_json(Path(path))
        expected = {
            "schema_version",
            "kind",
            "run_kind",
            "identity",
            "experiment_id",
            "checkpoint_dir",
            "checkpoint_path",
            "result_dir",
            "log_dir",
            "initial_step",
            "final_step",
            "resume_from",
            "checkpoint_root_stat",
            "checkpoint_stat",
            "resume_stat",
            "log_stat",
            "witness_digest",
        }
        if type(payload) is not dict or set(payload) != expected:
            return None
        claimed = _require_digest("witness_digest", payload["witness_digest"])
        digest_input = dict(payload)
        digest_input.pop("witness_digest")
        if config_digest(digest_input) != claimed:
            return None
        if (
            payload["schema_version"] != 2
            or payload["kind"] != "stage1-selected-run-witness"
            or payload["run_kind"] != run_kind
            or payload["identity"] != identity.to_dict()
        ):
            return None
        experiment_id = payload["experiment_id"]
        if type(experiment_id) is not str or f"-{run_kind}-" not in experiment_id:
            return None
        directories = {
            name: Path(payload[name]).resolve(strict=True)
            for name in ("checkpoint_dir", "result_dir", "log_dir")
        }
        if any(directory.name != experiment_id for directory in directories.values()):
            return None
        owned = resolve_run_artifact_paths(experiment_id)
        if any(
            directories[name] != getattr(owned, name)
            for name in ("checkpoint_dir", "result_dir", "log_dir")
        ):
            return None
        checkpoint = Path(payload["checkpoint_path"]).resolve(strict=True)
        if checkpoint.parent != directories["checkpoint_dir"]:
            return None
        if payload["checkpoint_root_stat"] != _directory_stat_witness(
            directories["checkpoint_dir"]
        ):
            return None
        if payload["checkpoint_stat"] != _checkpoint_step_witness(
            checkpoint, expected_step=payload["final_step"], identity=identity
        ):
            return None
        if payload["log_stat"] != _directory_stat_witness(directories["log_dir"]):
            return None
        if run_kind == "selected-save":
            if (
                payload["resume_from"] is not None
                or payload["resume_stat"] is not None
                or (payload["initial_step"], payload["final_step"]) != (0, 3)
            ):
                return None
        if run_kind == "selected-resume":
            resume_from = payload["resume_from"]
            if (
                type(resume_from) is not str
                or (payload["initial_step"], payload["final_step"]) != (3, 4)
                or Path(resume_from).resolve(strict=True).parent
                == directories["checkpoint_dir"]
                or "-selected-save-"
                not in Path(resume_from).resolve(strict=True).parent.name
                or payload["resume_stat"]
                != _checkpoint_step_witness(
                    Path(resume_from),
                    expected_step=payload["initial_step"],
                    identity=identity,
                )
            ):
                return None
        return payload
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def load_zero_matrix_profiles(
    config_paths: Mapping[int, Path], *, world_size: int = 2
) -> dict[int, DeepSpeedRuntimeProfile]:
    """Load the three canonical runtime profiles exactly once in reverse-stage order."""
    if set(config_paths) != set(_MATRIX_STAGES):
        raise ValueError("zero matrix requires exact ZeRO stages 3, 2, and 1")
    return {
        stage: load_deepspeed_runtime_profile(config_paths[stage], world_size=world_size)
        for stage in _MATRIX_STAGES
    }


def runtime_profile_from_witness(
    *,
    config_path: Path,
    config_digest: str,
    zero_stage: int,
    world_size: int,
) -> DeepSpeedRuntimeProfile:
    """Rehydrate an already-authenticated small profile without opening its config."""
    if zero_stage not in _MATRIX_STAGES:
        raise ValueError("zero stage must be 1, 2, or 3")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world size must be positive")
    _require_digest("deepspeed_config_digest", config_digest)
    return DeepSpeedRuntimeProfile(
        enabled=True,
        zero_stage=zero_stage,
        config_path=Path(config_path).resolve(strict=False),
        config_digest=config_digest,
        world_size=world_size,
    )


def _candidate_for_micro_batch(micro_batch: int) -> ZeroBatchCandidate:
    for candidate in ZERO_MATRIX_CANDIDATES:
        if candidate.micro_batch == micro_batch:
            return candidate
    raise ValueError("micro batch must be one of 8, 4, 2, or 1")


def prepare_zero_matrix_run(
    *,
    run_config_path: Path,
    method: str,
    model: str,
    source_data_path: Path,
    data_path: Path,
    image_root: Path,
    output_dir: Path,
    candidate_micro_batch: int,
    profile: DeepSpeedRuntimeProfile,
    sample_audit: SampleAuditWitness,
    prerequisite_chain_path: Path,
    prerequisite_chain_digest: str,
    final_save: bool,
    seed: int = 42,
) -> dict[str, object]:
    """Write one small schema-2 matrix run config from immutable witnesses."""
    if method not in _MATRIX_METHODS:
        raise ValueError("method must be full or dora")
    if not isinstance(profile, DeepSpeedRuntimeProfile) or not profile.enabled:
        raise ValueError("zero matrix prepare requires an enabled runtime profile")
    if profile.world_size != 2 or profile.zero_stage not in _MATRIX_STAGES:
        raise ValueError("zero matrix prepare requires a two-rank ZeRO profile")
    if profile.config_path is None or profile.config_digest is None:
        raise ValueError("zero matrix profile is incomplete")
    _require_digest("prerequisite_chain_digest", prerequisite_chain_digest)
    candidate = _candidate_for_micro_batch(candidate_micro_batch)
    config = {
        "schema_version": STAGE1_PRODUCER_SCHEMA_VERSION,
        "kind": "stage1-probe-run-manifest",
        "producer_kind": "zero_matrix",
        "optimizer_steps": 3,
        "method": method,
        "tuning_config": None if method == "full" else {
            "use_dora": True,
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        },
        "model": model,
        "source_data_path": str(Path(source_data_path).resolve(strict=False)),
        "data_path": str(Path(data_path).resolve(strict=False)),
        "data_stat": dict(sample_audit.input_stat),
        "image_root": str(Path(image_root).resolve(strict=False)),
        "output_dir": str(Path(output_dir).resolve(strict=False)),
        "micro_batch": candidate.micro_batch,
        "world_size": 2,
        "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
        "effective_global_batch": candidate.effective_global_batch,
        "epochs": 3,
        "learning_rate": 1e-5,
        "seed": seed,
        "max_steps": 3,
        "final_save": final_save,
        "deepspeed_config": str(profile.config_path),
        "deepspeed_config_digest": profile.config_digest,
        "attention_implementation": "flash_attention_2",
        "save_strategy": "no",
        "audit_digest": sample_audit.audit_digest,
        "probe_rows": str(sample_audit.path),
        "probe_row_indices": list(sample_audit.row_indices),
        "maximum_config_digest": None,
        "prerequisite_chain_digest": prerequisite_chain_digest,
        "prerequisite_chain": str(Path(prerequisite_chain_path).resolve(strict=False)),
    }
    config["config_digest"] = config_digest(config)
    path = Path(run_config_path)
    if path.exists():
        if _read_json(path) == config:
            return config
        raise FileExistsError(f"refusing to replace zero matrix run config: {path}")
    _atomic_write_json(path, config)
    return config


def _identity_from_run_config(config: Mapping[str, object]) -> Stage1ProducerIdentity:
    return Stage1ProducerIdentity(
        method=config["method"],
        zero_stage=int(Path(config["deepspeed_config"]).stem[-1]),
        deepspeed_config_digest=config["deepspeed_config_digest"],
        world_size=config["world_size"],
        micro_batch=config["micro_batch"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        effective_global_batch=config["effective_global_batch"],
        sample_stream_digest=config["audit_digest"],
        model_id=config["model"],
        seed=config["seed"],
    )


def _row_filename(identity: Stage1ProducerIdentity) -> str:
    return (
        f"zero{identity.zero_stage}-{identity.method}-mb{identity.micro_batch}"
        f"-ga{identity.gradient_accumulation_steps}.json"
    )


def _write_attempt_record(
    path: Path,
    row: ZeroMatrixAttempt,
    *,
    evidence_digest: str | None,
) -> Path:
    payload = {
        "schema_version": 1,
        "kind": "stage1-zero-matrix-attempt",
        **_attempt_payload(row, evidence_digest=evidence_digest),
    }
    payload["attempt_digest"] = config_digest(payload)
    _atomic_write_json(path, payload)
    return path


def _load_attempt_record(path: Path) -> tuple[ZeroMatrixAttempt, str | None]:
    payload = _read_json(path)
    expected = {
        "schema_version",
        "kind",
        "identity",
        "status",
        "samples_per_second",
        "evidence_path",
        "evidence_digest",
        "attempt_digest",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("invalid zero matrix attempt record")
    claimed = payload["attempt_digest"]
    _require_digest("attempt_digest", claimed)
    digest_input = dict(payload)
    digest_input.pop("attempt_digest")
    if config_digest(digest_input) != claimed:
        raise ValueError("zero matrix attempt digest mismatch")
    identity_raw = payload["identity"]
    if type(identity_raw) is not dict or set(identity_raw) != {
        field.name for field in fields(Stage1ProducerIdentity)
    }:
        raise ValueError("invalid zero matrix producer identity")
    row = ZeroMatrixAttempt(
        identity=Stage1ProducerIdentity(**identity_raw),
        status=payload["status"],
        samples_per_second=payload["samples_per_second"],
        evidence_path=Path(payload["evidence_path"]),
    )
    digest = payload["evidence_digest"]
    if digest is not None:
        _require_digest("evidence_digest", digest)
    return row, digest


def _producer_matches_row(
    row: ZeroMatrixAttempt, expected_digest: str | None
) -> bool:
    if row.status != "success" or expected_digest is None:
        return row.status in {"capacity_oom", "integration_failure"}
    try:
        evidence, envelope = _load_producer_evidence(row.evidence_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        envelope.get("producer_digest") == expected_digest
        and evidence.status == "success"
        and evidence.identity == row.identity
        and evidence.optimizer_steps == 3
        and evidence.samples_seen == 3 * evidence.effective_global_batch
    )


def _identity_for_candidate(
    *,
    method: str,
    profile: DeepSpeedRuntimeProfile,
    candidate: ZeroBatchCandidate,
    audit: SampleAuditWitness,
    model: str,
    seed: int,
) -> Stage1ProducerIdentity:
    if profile.zero_stage is None or profile.config_digest is None:
        raise ValueError("runtime profile is incomplete")
    return Stage1ProducerIdentity(
        method=method,
        zero_stage=profile.zero_stage,
        deepspeed_config_digest=profile.config_digest,
        world_size=profile.world_size,
        micro_batch=candidate.micro_batch,
        gradient_accumulation_steps=candidate.gradient_accumulation_steps,
        effective_global_batch=candidate.effective_global_batch,
        sample_stream_digest=audit.audit_digest,
        model_id=model,
        seed=seed,
    )


def reuse_status_rows(
    *,
    matrix_result_dir: Path,
    profiles: Mapping[int, DeepSpeedRuntimeProfile],
    methods: Sequence[str],
    audit: SampleAuditWitness,
    model: str,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Return fixed-order reuse decisions from exact small identity witnesses."""
    decisions = []
    rows_dir = Path(matrix_result_dir) / "rows"
    for stage in _MATRIX_STAGES:
        if stage not in profiles:
            continue
        for method in methods:
            for candidate in ZERO_MATRIX_CANDIDATES:
                identity = _identity_for_candidate(
                    method=method,
                    profile=profiles[stage],
                    candidate=candidate,
                    audit=audit,
                    model=model,
                    seed=seed,
                )
                path = rows_dir / _row_filename(identity)
                reused = False
                status = None
                evidence_path = None
                if path.is_file():
                    try:
                        row, digest = _load_attempt_record(path)
                        reused = row.identity == identity and _producer_matches_row(row, digest)
                    except (OSError, TypeError, ValueError, json.JSONDecodeError):
                        reused = False
                    if reused:
                        status = row.status
                        evidence_path = str(row.evidence_path)
                decisions.append(
                    {
                        "identity": identity.to_dict(),
                        "reuse": reused,
                        "status": status,
                        "attempt_record": str(path),
                        "evidence_path": evidence_path,
                    }
                )
    return decisions


def _read_bounded_stderr(path: Path) -> str:
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > _MAX_STDERR_BYTES:
            handle.seek(size - _MAX_STDERR_BYTES)
        data = handle.read(_MAX_STDERR_BYTES)
    return data.decode("utf-8", errors="replace")


def _load_prerequisite_digest(path: Path) -> str:
    payload = _read_json(path)
    if type(payload) is not dict:
        raise ValueError("prerequisite chain must be a JSON object")
    return _require_digest(
        "prerequisite_chain_digest", payload.get("prerequisite_chain_digest")
    )


def _add_shared_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=_MATRIX_METHODS, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed-config", required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--sample-audit", required=True)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit-status")
    audit_parser.add_argument("--candidate-audit", action="append", required=True)
    audit_parser.add_argument("--data-path", required=True)
    audit_parser.add_argument("--image-root", required=True)
    audit_parser.add_argument("--count", type=int, default=64)

    prepare_parser = subparsers.add_parser("prepare")
    _add_shared_identity_arguments(prepare_parser)
    prepare_parser.add_argument("--run-config", required=True)
    prepare_parser.add_argument("--source-data-path", required=True)
    prepare_parser.add_argument("--data-path", required=True)
    prepare_parser.add_argument("--image-root", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--micro-batch", type=int, required=True)
    prepare_parser.add_argument("--prerequisite-chain", required=True)
    prepare_parser.add_argument("--final-save", action="store_true")
    prepare_parser.add_argument("--deepspeed-config-digest")
    prepare_parser.add_argument("--zero-stage", type=int)

    reuse_parser = subparsers.add_parser("reuse-status")
    reuse_parser.add_argument("--matrix-result-dir", required=True)
    reuse_parser.add_argument("--method", choices=(*_MATRIX_METHODS, "all"), default="all")
    reuse_parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    reuse_parser.add_argument("--seed", type=int, default=42)
    reuse_parser.add_argument("--deepspeed-config", action="append")
    reuse_parser.add_argument("--world-size", type=int, default=2)
    reuse_parser.add_argument("--sample-audit", required=True)

    classify_parser = subparsers.add_parser("classify")
    classify_parser.add_argument("--returncode", type=int, required=True)
    classify_parser.add_argument("--stderr-file", required=True)
    classify_parser.add_argument("--run-config", required=True)
    classify_parser.add_argument("--producer-evidence", required=True)
    classify_parser.add_argument("--matrix-result-dir", required=True)

    select_parser = subparsers.add_parser("select")
    _add_shared_identity_arguments(select_parser)
    select_parser.add_argument("--matrix-result-dir", required=True)
    select_parser.add_argument("--deepspeed-config-digest")
    select_parser.add_argument("--zero-stage", type=int)

    record_selected_parser = subparsers.add_parser("record-selected")
    record_selected_parser.add_argument("--witness", required=True)
    record_selected_parser.add_argument(
        "--run-kind", choices=("selected-save", "selected-resume"), required=True
    )
    record_selected_parser.add_argument("--identity-json", required=True)
    record_selected_parser.add_argument("--experiment-id", required=True)
    record_selected_parser.add_argument("--checkpoint-dir", required=True)
    record_selected_parser.add_argument("--checkpoint-path", required=True)
    record_selected_parser.add_argument("--result-dir", required=True)
    record_selected_parser.add_argument("--log-dir", required=True)
    record_selected_parser.add_argument("--initial-step", type=int, required=True)
    record_selected_parser.add_argument("--final-step", type=int, required=True)
    record_selected_parser.add_argument("--resume-from")

    selected_status_parser = subparsers.add_parser("selected-status")
    selected_status_parser.add_argument("--witness", required=True)
    selected_status_parser.add_argument(
        "--run-kind", choices=("selected-save", "selected-resume"), required=True
    )
    selected_status_parser.add_argument("--identity-json", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "audit-status":
            witness = resolve_sample_audit(
                [Path(value) for value in args.candidate_audit],
                data_path=Path(args.data_path),
                image_root=Path(args.image_root),
                count=args.count,
            )
            print(json.dumps({
                "path": str(witness.path),
                "audit_digest": witness.audit_digest,
                "input_stat": witness.input_stat,
                "row_indices": list(witness.row_indices),
            }, separators=(",", ":")))
        elif args.command == "prepare":
            if (args.deepspeed_config_digest is None) != (args.zero_stage is None):
                raise ValueError(
                    "prepare requires both --deepspeed-config-digest and --zero-stage"
                )
            profile = (
                load_deepspeed_runtime_profile(
                    args.deepspeed_config, world_size=args.world_size
                )
                if args.deepspeed_config_digest is None
                else runtime_profile_from_witness(
                    config_path=Path(args.deepspeed_config),
                    config_digest=args.deepspeed_config_digest,
                    zero_stage=args.zero_stage,
                    world_size=args.world_size,
                )
            )
            audit = load_sample_audit(Path(args.sample_audit))
            config = prepare_zero_matrix_run(
                run_config_path=Path(args.run_config),
                method=args.method,
                model=args.model,
                source_data_path=Path(args.source_data_path),
                data_path=Path(args.data_path),
                image_root=Path(args.image_root),
                output_dir=Path(args.output_dir),
                candidate_micro_batch=args.micro_batch,
                profile=profile,
                sample_audit=audit,
                prerequisite_chain_path=Path(args.prerequisite_chain),
                prerequisite_chain_digest=_load_prerequisite_digest(
                    Path(args.prerequisite_chain)
                ),
                final_save=args.final_save,
                seed=args.seed,
            )
            print(json.dumps(config, separators=(",", ":")))
        elif args.command == "reuse-status":
            config_values = args.deepspeed_config or [
                str(canonical_deepspeed_config_path(stage)) for stage in _MATRIX_STAGES
            ]
            if len(config_values) == 1:
                profile = load_deepspeed_runtime_profile(
                    config_values[0], world_size=args.world_size
                )
                profiles = {profile.zero_stage: profile}
            elif len(config_values) == 3:
                provisional = {}
                for value in config_values:
                    stage = int(Path(value).stem[-1])
                    provisional[stage] = Path(value)
                profiles = load_zero_matrix_profiles(
                    provisional, world_size=args.world_size
                )
            else:
                raise ValueError("reuse-status requires one or three DeepSpeed configs")
            methods = _MATRIX_METHODS if args.method == "all" else (args.method,)
            decisions = reuse_status_rows(
                matrix_result_dir=Path(args.matrix_result_dir),
                profiles=profiles,
                methods=methods,
                audit=load_sample_audit(Path(args.sample_audit)),
                model=args.model,
                seed=args.seed,
            )
            print(json.dumps(decisions, separators=(",", ":")))
        elif args.command == "classify":
            config = _read_json(Path(args.run_config))
            if type(config) is not dict:
                raise ValueError("run config must be an object")
            identity = _identity_from_run_config(config)
            evidence = None
            producer_digest = None
            if args.returncode == 0:
                try:
                    loaded, envelope = _load_producer_evidence(
                        Path(args.producer_evidence)
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    loaded = None
                if (
                    loaded is not None
                    and loaded.identity == identity
                    and loaded.optimizer_steps == 3
                    and loaded.samples_seen == 3 * loaded.effective_global_batch
                ):
                    evidence = loaded
                    producer_digest = envelope["producer_digest"]
            status = classify_zero_matrix_attempt(
                returncode=args.returncode,
                stderr=_read_bounded_stderr(Path(args.stderr_file)),
                producer=evidence,
            )
            row_path = (
                Path(args.matrix_result_dir) / "rows" / _row_filename(identity)
            )
            row = ZeroMatrixAttempt(
                identity=identity,
                status=status,
                samples_per_second=(
                    evidence.samples_per_second if status == "success" else None
                ),
                evidence_path=(
                    Path(args.producer_evidence) if status == "success" else row_path
                ),
            )
            _write_attempt_record(
                row_path, row, evidence_digest=producer_digest
            )
            print(json.dumps({
                **_attempt_payload(row, evidence_digest=producer_digest),
                "attempt_record": str(row_path),
            }, separators=(",", ":")))
        elif args.command == "select":
            if (args.deepspeed_config_digest is None) != (args.zero_stage is None):
                raise ValueError(
                    "select requires both --deepspeed-config-digest and --zero-stage"
                )
            profile = (
                load_deepspeed_runtime_profile(
                    args.deepspeed_config, world_size=args.world_size
                )
                if args.deepspeed_config_digest is None
                else runtime_profile_from_witness(
                    config_path=Path(args.deepspeed_config),
                    config_digest=args.deepspeed_config_digest,
                    zero_stage=args.zero_stage,
                    world_size=args.world_size,
                )
            )
            audit = load_sample_audit(Path(args.sample_audit))
            rows = []
            digests = {}
            rows_dir = Path(args.matrix_result_dir) / "rows"
            for candidate in ZERO_MATRIX_CANDIDATES:
                identity = _identity_for_candidate(
                    method=args.method,
                    profile=profile,
                    candidate=candidate,
                    audit=audit,
                    model=args.model,
                    seed=args.seed,
                )
                row, digest = _load_attempt_record(rows_dir / _row_filename(identity))
                if row.identity != identity or not _producer_matches_row(row, digest):
                    raise ValueError("matrix row does not have exact reusable evidence")
                rows.append(row)
                digests[str(row.evidence_path)] = digest
            report_path = write_zero_matrix_report(
                Path(args.matrix_result_dir), rows, evidence_digests=digests
            )
            print(report_path.read_text(encoding="utf-8"), end="")
        elif args.command == "record-selected":
            identity_payload = json.loads(args.identity_json)
            if type(identity_payload) is not dict:
                raise ValueError("selected identity must be an object")
            owned = resolve_run_artifact_paths(args.experiment_id)
            if (
                Path(args.checkpoint_dir).resolve(strict=True) != owned.checkpoint_dir
                or Path(args.result_dir).resolve(strict=True) != owned.result_dir
                or Path(args.log_dir).resolve(strict=True) != owned.log_dir
            ):
                raise ValueError(
                    "selected checkpoint, result, and log roots must use canonical ownership"
                )
            path = write_selected_run_witness(
                Path(args.witness),
                run_kind=args.run_kind,
                identity=Stage1ProducerIdentity(**identity_payload),
                experiment_id=args.experiment_id,
                checkpoint_dir=Path(args.checkpoint_dir),
                checkpoint_path=Path(args.checkpoint_path),
                result_dir=Path(args.result_dir),
                log_dir=Path(args.log_dir),
                initial_step=args.initial_step,
                final_step=args.final_step,
                resume_from=None if args.resume_from is None else Path(args.resume_from),
            )
            print(path)
        else:
            identity_payload = json.loads(args.identity_json)
            if type(identity_payload) is not dict:
                raise ValueError("selected identity must be an object")
            reusable = load_reusable_selected_run(
                Path(args.witness),
                run_kind=args.run_kind,
                identity=Stage1ProducerIdentity(**identity_payload),
            )
            print(json.dumps({
                "reuse": reusable is not None,
                "witness": reusable,
            }, separators=(",", ":")))
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
