"""Strict bounded Stage 2 per-rank producer evidence."""

from __future__ import annotations

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


def _validate_identity(
    *,
    run_id: object,
    method: object,
    zero_stage: object,
    checkpoint_path: object,
    artifact_kind: object,
) -> tuple[str, str, int, Path, str]:
    from src.multinode import _RUN_ID_PATTERN

    if type(run_id) is not str or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("Stage 2 producer run ID is invalid")
    if type(method) is not str or method not in ("full", "lora", "dora"):
        raise ValueError("Stage 2 producer method is invalid")
    if type(zero_stage) is not int or zero_stage not in (1, 2, 3):
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
    expected_artifact = "full" if method == "full" else f"peft-{method}"
    if type(artifact_kind) is not str or artifact_kind != expected_artifact:
        raise ValueError("Stage 2 producer method artifact is invalid")
    return run_id, method, zero_stage, checkpoint, artifact_kind


def build_stage2_producer_evidence(
    *,
    run_id: str,
    method: str,
    zero_stage: int,
    checkpoint_path: str | Path,
    artifact_kind: str,
    rank_records: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Validate four observed rank records and derive one exact evidence object."""
    run_id, method, zero_stage, checkpoint, artifact_kind = _validate_identity(
        run_id=run_id,
        method=method,
        zero_stage=zero_stage,
        checkpoint_path=checkpoint_path,
        artifact_kind=artifact_kind,
    )
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
    return {
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


def decode_stage2_producer_evidence(
    raw: bytes,
    *,
    expected_run_id: str,
    expected_method: str,
    expected_zero_stage: int,
    expected_checkpoint: str | Path,
    expected_global_step: int,
) -> dict[str, object]:
    """Decode one duplicate-free bounded producer object and revalidate identity."""
    if type(raw) is not bytes or not raw or len(raw) > 1024 * 1024:
        raise ValueError("Stage 2 producer evidence size is invalid")
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_fields)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage 2 producer evidence is invalid JSON") from error
    if type(payload) is not dict or set(payload) != _EVIDENCE_FIELDS:
        raise ValueError("Stage 2 producer evidence fields are invalid")
    rebuilt = build_stage2_producer_evidence(
        run_id=payload["run_id"],
        method=payload["method"],
        zero_stage=payload["zero_stage"],
        checkpoint_path=payload["checkpoint_path"],
        artifact_kind=payload["artifact_kind"],
        rank_records=payload["ranks"],
    )
    if payload != rebuilt:
        raise ValueError("Stage 2 producer evidence contains unobserved claims")
    if (
        payload["run_id"] != expected_run_id
        or payload["method"] != expected_method
        or payload["zero_stage"] != expected_zero_stage
        or payload["checkpoint_path"] != str(Path(expected_checkpoint))
        or payload["global_step"] != expected_global_step
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
    device = torch.device("cuda", int(__import__("os").environ["LOCAL_RANK"]))
    failed = torch.tensor(local_failure, dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    return int(failed.item())


def _synchronize() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def publish_stage2_producer_evidence(
    path: str | Path,
    *,
    run_id: str,
    method: str,
    zero_stage: int,
    checkpoint_path: str | Path,
    artifact_kind: str,
    local_rank_record: dict[str, object],
    gather_records: Callable[[dict[str, object]], list[dict[str, object]]] = _gather_records,
    is_writer: bool | None = None,
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
    )
    writer = (
        not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        if is_writer is None
        else is_writer
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
        raise RuntimeError("Stage 2 producer publication failed on rank zero")
    synchronize()
    return payload
