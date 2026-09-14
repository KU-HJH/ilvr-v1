"""Strict bounded Stage 2 per-rank producer evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.distributed as dist

from src.multinode_runtime import _canonical_json_bytes, _publish_bytes_no_replace


_RANK_FIELDS = {
    "rank",
    "global_step",
    "optimizer_step",
    "loss",
    "gradient_norm",
    "worker_error",
}
_EVIDENCE_FIELDS = {
    "schema_version",
    "kind",
    "run_id",
    "stage",
    "method",
    "zero_stage",
    "world_size",
    "process_ranks",
    "global_step",
    "optimizer_step",
    "checkpoint_path",
    "artifact_kind",
    "finite_loss",
    "finite_gradient",
    "worker_errors",
    "ranks",
}


def _reject_duplicate_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Stage 2 producer evidence contains a duplicate field")
        result[key] = value
    return result


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(
        value
    )


def _validate_decoded_types(payload: dict[str, object]) -> None:
    """Reject JSON scalar aliases before equality-based reconstruction."""
    if (
        type(payload["schema_version"]) is not int
        or type(payload["kind"]) is not str
        or type(payload["run_id"]) is not str
        or type(payload["stage"]) is not str
        or type(payload["method"]) is not str
        or (payload.get("execution_backend") != "ddp" and type(payload["zero_stage"]) is not int)
        or type(payload["world_size"]) is not int
        or type(payload["process_ranks"]) is not list
        or any(type(rank) is not int for rank in payload["process_ranks"])
        or type(payload["global_step"]) is not int
        or type(payload["optimizer_step"]) is not int
        or type(payload["checkpoint_path"]) is not str
        or type(payload["artifact_kind"]) is not str
        or type(payload["finite_loss"]) is not bool
        or type(payload["finite_gradient"]) is not bool
        or type(payload["worker_errors"]) is not list
        or type(payload["ranks"]) is not list
    ):
        raise ValueError("Stage 2 producer evidence types are invalid")


def _validate_identity(
    *,
    run_id: object,
    method: object,
    zero_stage: object,
    checkpoint_path: object,
    artifact_kind: object,
    execution_backend: str = "deepspeed",
) -> tuple[str, str, int | None, Path, str]:
    from src.multinode import _RUN_ID_PATTERN

    if type(run_id) is not str or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("Stage 2 producer run ID is invalid")
    if type(method) is not str or method not in ("full", "lora", "dora"):
        raise ValueError("Stage 2 producer method is invalid")
    if execution_backend not in ("deepspeed", "ddp"):
        raise ValueError("Stage 2 producer execution backend is invalid")
    if execution_backend == "ddp" and zero_stage is not None:
        raise ValueError("native Stage 2 producer must have null ZeRO stage")
    if execution_backend == "deepspeed" and (type(zero_stage) is not int or zero_stage not in (1, 2, 3)):
        raise ValueError("Stage 2 producer ZeRO stage is invalid")
    if type(checkpoint_path) is not str and not isinstance(checkpoint_path, Path):
        raise ValueError("Stage 2 producer checkpoint path is invalid")
    checkpoint = Path(checkpoint_path)
    if (
        not checkpoint.is_absolute()
        or ".." in checkpoint.parts
        or checkpoint.parent.name != run_id
        or not checkpoint.name.startswith("checkpoint-")
    ):
        raise ValueError("Stage 2 producer checkpoint identity is invalid")
    if type(artifact_kind) is not str or artifact_kind not in ("full", "peft"):
        raise ValueError("Stage 2 producer method artifact is invalid")
    if (method == "full") != (artifact_kind == "full"):
        raise ValueError("Stage 2 producer observed artifact disagrees with method")
    return run_id, method, zero_stage, checkpoint, artifact_kind


def build_stage2_producer_evidence(
    *,
    run_id: str,
    method: str,
    zero_stage: int | None,
    checkpoint_path: str | Path,
    artifact_kind: str,
    rank_records: Sequence[dict[str, object]],
    execution_backend: str = "deepspeed",
    objective_identity_sha256: str | None = None,
) -> dict[str, object]:
    """Validate four observed rank records and derive one exact evidence object."""
    run_id, method, zero_stage, checkpoint, artifact_kind = _validate_identity(
        run_id=run_id,
        method=method,
        zero_stage=zero_stage,
        checkpoint_path=checkpoint_path,
        artifact_kind=artifact_kind,
        execution_backend=execution_backend,
    )
    if objective_identity_sha256 is not None and (
        execution_backend != "ddp" or type(objective_identity_sha256) is not str
        or len(objective_identity_sha256) != 64
        or any(c not in "0123456789abcdef" for c in objective_identity_sha256)
    ):
        raise ValueError("Stage 2 producer objective digest is invalid")
    if type(rank_records) not in (list, tuple) or len(rank_records) != 4:
        raise ValueError("Stage 2 producer requires exactly four rank records")
    normalized: list[dict[str, object]] = []
    for record in rank_records:
        if type(record) is not dict or set(record) != _RANK_FIELDS:
            raise ValueError("Stage 2 producer rank evidence fields are invalid")
        if type(record["rank"]) is not int:
            raise ValueError("Stage 2 producer rank is invalid")
        if type(record["global_step"]) is not int or record["global_step"] <= 0:
            raise ValueError("Stage 2 producer global step is invalid")
        if (
            type(record["optimizer_step"]) is not int
            or record["optimizer_step"] <= 0
        ):
            raise ValueError("Stage 2 producer optimizer step is invalid")
        if not _finite_number(record["loss"]):
            raise ValueError("Stage 2 producer requires finite loss on every rank")
        if not _finite_number(record["gradient_norm"]):
            raise ValueError("Stage 2 producer requires finite gradient on every rank")
        if record["worker_error"] is not None:
            raise ValueError("Stage 2 producer contains a worker error")
        normalized.append(dict(record))
    normalized.sort(key=lambda item: item["rank"])
    if [record["rank"] for record in normalized] != [0, 1, 2, 3]:
        raise ValueError("Stage 2 producer process ranks are incomplete")
    global_steps = {record["global_step"] for record in normalized}
    optimizer_steps = {record["optimizer_step"] for record in normalized}
    if len(global_steps) != 1 or len(optimizer_steps) != 1:
        raise ValueError("Stage 2 producer ranks disagree on training steps")
    global_step = next(iter(global_steps))
    optimizer_step = next(iter(optimizer_steps))
    if optimizer_step != global_step or checkpoint.name != f"checkpoint-{global_step}":
        raise ValueError("Stage 2 producer checkpoint and optimizer steps disagree")
    payload = {
        "schema_version": 1,
        "kind": "stage2-producer-evidence",
        "run_id": run_id,
        "stage": "stage2",
        "method": method,
        "zero_stage": zero_stage,
        "world_size": 4,
        "process_ranks": [0, 1, 2, 3],
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "checkpoint_path": str(checkpoint),
        "artifact_kind": artifact_kind,
        "finite_loss": True,
        "finite_gradient": True,
        "worker_errors": [],
        "ranks": normalized,
    }
    if execution_backend == "ddp":
        payload.update(schema_version=2, execution_backend="ddp",
                       objective_identity_sha256=objective_identity_sha256)
    return payload


def decode_stage2_producer_evidence(
    raw: bytes,
    *,
    expected_run_id: str,
    expected_method: str,
    expected_zero_stage: int | None,
    expected_checkpoint: str | Path,
    expected_global_step: int,
    expected_execution_backend: str = "deepspeed",
) -> dict[str, object]:
    """Decode one duplicate-free bounded producer object and revalidate identity."""
    if type(raw) is not bytes or not raw or len(raw) > 1024 * 1024:
        raise ValueError("Stage 2 producer evidence size is invalid")
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_fields)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage 2 producer evidence is invalid JSON") from error
    fields = _EVIDENCE_FIELDS | ({"execution_backend", "objective_identity_sha256"}
                               if expected_execution_backend == "ddp" else set())
    if type(payload) is not dict or set(payload) != fields:
        raise ValueError("Stage 2 producer evidence fields are invalid")
    _validate_decoded_types(payload)
    rebuilt = build_stage2_producer_evidence(
        run_id=payload["run_id"],
        method=payload["method"],
        zero_stage=payload["zero_stage"],
        checkpoint_path=payload["checkpoint_path"],
        artifact_kind=payload["artifact_kind"],
        rank_records=payload["ranks"],
        execution_backend=payload.get("execution_backend", "deepspeed"),
        objective_identity_sha256=payload.get("objective_identity_sha256"),
    )
    if payload != rebuilt:
        raise ValueError("Stage 2 producer evidence contains unobserved claims")
    if (
        payload["run_id"] != expected_run_id
        or payload["method"] != expected_method
        or payload["zero_stage"] != expected_zero_stage
        or payload["checkpoint_path"] != str(Path(expected_checkpoint))
        or payload["global_step"] != expected_global_step
        or payload.get("execution_backend", "deepspeed") != expected_execution_backend
    ):
        raise ValueError("Stage 2 producer evidence identity is inconsistent")
    return payload


def _gather_records(local_record: dict[str, object]) -> list[dict[str, object]]:
    if not dist.is_available() or not dist.is_initialized():
        return [local_record]
    gathered: list[dict[str, object] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_record)
    if any(record is None for record in gathered):
        raise RuntimeError("Stage 2 producer collective returned an empty rank")
    return [record for record in gathered if record is not None]


def _failure_count(local_failure: int) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return local_failure
    device = (torch.device("cuda", torch.cuda.current_device())
              if dist.get_backend() == "nccl" else torch.device("cpu"))
    failed = torch.tensor(local_failure, dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    return int(failed.item())


def _synchronize() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _broadcast_artifact(value: str | None) -> str:
    if not dist.is_available() or not dist.is_initialized():
        if type(value) is not str:
            raise ValueError("rank zero did not observe a Stage 2 artifact")
        return value
    values: list[str | None] = [value]
    dist.broadcast_object_list(values, src=0)
    if type(values[0]) is not str:
        raise ValueError("rank zero broadcast an invalid Stage 2 artifact")
    return values[0]


def observe_stage2_checkpoint_artifact(
    checkpoint_path: str | Path,
    *,
    method: str,
    is_rank_zero: bool | None = None,
    classifier: Callable[[Path], object] | None = None,
    peft_method_reader: Callable[[Path], str] | None = None,
    broadcaster: Callable[[str | None], str] = _broadcast_artifact,
) -> str:
    """Classify on global rank zero, bind PEFT metadata, then broadcast."""
    checkpoint = Path(checkpoint_path)
    rank_zero = (
        not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        if is_rank_zero is None
        else is_rank_zero
    )
    observed: str | None = None
    if rank_zero:
        if classifier is None:
            from stage2.checkpointing import classify_stage2_artifact

            classifier = classify_stage2_artifact
        classified = classifier(checkpoint)
        observed = getattr(classified, "value", classified)
        if observed == "peft":
            if peft_method_reader is None:
                from stage2.checkpointing import load_stage2_peft_metadata

                peft_method_reader = lambda path: load_stage2_peft_metadata(path).tuning_method
            if peft_method_reader(checkpoint) != method:
                raise ValueError("observed Stage 2 PEFT method is inconsistent")
        elif observed != "full" or method != "full":
            raise ValueError("observed Stage 2 artifact kind is inconsistent")
    broadcast = broadcaster(observed)
    if broadcast not in ("full", "peft"):
        raise ValueError("broadcast Stage 2 artifact kind is invalid")
    return broadcast


def _is_machine_local_leader() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    try:
        return int(__import__("os").environ["LOCAL_RANK"]) == 0
    except (KeyError, ValueError) as error:
        raise ValueError("LOCAL_RANK is required for dual-node publication") from error


def publish_stage2_producer_evidence(
    path: str | Path,
    *,
    run_id: str,
    method: str,
    zero_stage: int | None,
    checkpoint_path: str | Path,
    artifact_kind: str,
    local_rank_record: dict[str, object],
    execution_backend: str = "deepspeed",
    objective_identity_sha256: str | None = None,
    gather_records: Callable[[dict[str, object]], list[dict[str, object]]] = _gather_records,
    is_machine_leader: bool | None = None,
    failure_counter: Callable[[int], int] = _failure_count,
    synchronize: Callable[[], None] = _synchronize,
) -> dict[str, object]:
    """Collectively validate and exclusively publish Stage 2 producer evidence."""
    records = gather_records(local_rank_record)
    payload = build_stage2_producer_evidence(
        run_id=run_id,
        method=method,
        zero_stage=zero_stage,
        checkpoint_path=checkpoint_path,
        artifact_kind=artifact_kind,
        rank_records=records,
        execution_backend=execution_backend,
        objective_identity_sha256=objective_identity_sha256,
    )
    writer = (
        _is_machine_local_leader()
        if is_machine_leader is None
        else is_machine_leader
    )
    local_error: Exception | None = None
    if writer:
        try:
            _publish_bytes_no_replace(
                Path(path), _canonical_json_bytes(payload) + b"\n"
            )
        except Exception as error:
            local_error = error
    if failure_counter(int(local_error is not None)):
        if local_error is not None:
            raise local_error
        raise RuntimeError("Stage 2 producer publication failed on a machine-local leader")
    synchronize()
    return payload


def validate_stage2_objective_identity(raw: bytes | None, *, config) -> str | None:
    """Bind the complete canonical sidecar to the frozen resolved configuration."""
    enabled = config is not None and config.enable_stage2_objective
    if not enabled:
        if raw is not None:
            raise ValueError("disabled Stage 2 objective must not have an identity")
        return None
    if type(raw) is not bytes or not raw or len(raw) > 4 * 1024 * 1024:
        raise ValueError("Stage 2 objective identity is missing or exceeds its bound")
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_fields)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage 2 objective identity is invalid JSON") from error
    if (type(payload) is not dict
            or set(payload) != {"schema_version", "config", "reference_identity", "vocabulary_identity"}
            or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
            or type(payload["config"]) is not dict
            or payload["config"].get("enable_stage2_objective") is not True):
        raise ValueError("Stage 2 objective identity schema is invalid")
    expected = config.to_dict()
    if (set(payload["config"]) != set(expected)
            or any(type(payload["config"][key]) is not type(value)
                   or payload["config"][key] != value for key, value in expected.items())):
        raise ValueError("Stage 2 objective configuration differs from requested config")
    if any(value is not None and type(value) is not dict
           for value in (payload["reference_identity"], payload["vocabulary_identity"])):
        raise ValueError("Stage 2 objective reference/vocabulary identity is invalid")
    if not config.active and (payload["reference_identity"] is not None
                              or payload["vocabulary_identity"] is not None):
        raise ValueError("CE-only objective must have null reference/vocabulary identities")
    return hashlib.sha256(raw).hexdigest()
