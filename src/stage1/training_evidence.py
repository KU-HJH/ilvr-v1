"""Strict bounded producer evidence for ordinary Stage 1 training."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.multinode_runtime import _canonical_json_bytes, _publish_bytes_no_replace


_FIELDS = {
    "schema_version",
    "kind",
    "run_id",
    "stage",
    "method",
    "zero_stage",
    "world_size",
    "process_ranks",
    "global_step",
    "checkpoint_path",
    "loss",
    "finite_loss",
}


def _finite_number(value: object) -> bool:
    return (
        type(value) in (int, float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_identity(
    *,
    run_id: object,
    method: object,
    zero_stage: object,
    world_size: object,
    global_step: object,
    checkpoint_path: object,
) -> tuple[str, str, int, int, int, Path]:
    from src.multinode import _RUN_ID_PATTERN

    if type(run_id) is not str or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("Stage 1 training evidence run identity is invalid")
    if type(method) is not str or method not in ("full", "lora", "dora"):
        raise ValueError("Stage 1 training evidence method identity is invalid")
    if type(zero_stage) is not int or zero_stage not in (1, 2, 3):
        raise ValueError("Stage 1 training evidence ZeRO identity is invalid")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("Stage 1 training evidence world identity is invalid")
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("Stage 1 training evidence step identity is invalid")
    if type(checkpoint_path) is not str and not isinstance(checkpoint_path, Path):
        raise ValueError("Stage 1 training evidence checkpoint identity is invalid")
    checkpoint = Path(checkpoint_path)
    if (
        not checkpoint.is_absolute()
        or ".." in checkpoint.parts
        or checkpoint.name != f"checkpoint-{global_step}"
        or checkpoint.parent.name != run_id
    ):
        raise ValueError("Stage 1 training evidence checkpoint identity is invalid")
    return run_id, method, zero_stage, world_size, global_step, checkpoint


def build_stage1_training_evidence(
    *,
    run_id: str,
    method: str,
    zero_stage: int,
    world_size: int,
    global_step: int,
    checkpoint_path: str | Path,
    loss: float,
) -> dict[str, object]:
    """Bind one observed finite Trainer loss to its exact numbered checkpoint."""
    run_id, method, zero_stage, world_size, global_step, checkpoint = (
        _validate_identity(
            run_id=run_id,
            method=method,
            zero_stage=zero_stage,
            world_size=world_size,
            global_step=global_step,
            checkpoint_path=checkpoint_path,
        )
    )
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Stage 1 training evidence checkpoint must exist")
    if not _finite_number(loss):
        raise ValueError("Stage 1 training evidence requires observed finite loss")
    return {
        "schema_version": 1,
        "kind": "stage1-training-evidence",
        "run_id": run_id,
        "stage": "stage1",
        "method": method,
        "zero_stage": zero_stage,
        "world_size": world_size,
        "process_ranks": list(range(world_size)),
        "global_step": global_step,
        "checkpoint_path": str(checkpoint),
        "loss": float(loss),
        "finite_loss": True,
    }


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Stage 1 training evidence contains a duplicate field")
        result[key] = value
    return result


def decode_stage1_training_evidence(
    raw: bytes,
    *,
    expected_run_id: str,
    expected_method: str,
    expected_zero_stage: int,
    expected_world_size: int,
    expected_checkpoint: str | Path,
    expected_global_step: int,
) -> dict[str, object]:
    """Decode and revalidate one exact ordinary Stage 1 producer object."""
    if type(raw) is not bytes or not raw or len(raw) > 1024 * 1024:
        raise ValueError("Stage 1 training evidence size is invalid")
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage 1 training evidence is invalid JSON") from error
    if type(payload) is not dict or set(payload) != _FIELDS:
        raise ValueError("Stage 1 training evidence fields are invalid")
    if (
        type(payload["schema_version"]) is not int
        or type(payload["kind"]) is not str
        or type(payload["stage"]) is not str
        or type(payload["process_ranks"]) is not list
        or any(type(rank) is not int for rank in payload["process_ranks"])
        or type(payload["finite_loss"]) is not bool
    ):
        raise ValueError("Stage 1 training evidence types are invalid")
    rebuilt = build_stage1_training_evidence(
        run_id=payload["run_id"],
        method=payload["method"],
        zero_stage=payload["zero_stage"],
        world_size=payload["world_size"],
        global_step=payload["global_step"],
        checkpoint_path=payload["checkpoint_path"],
        loss=payload["loss"],
    )
    if payload != rebuilt:
        raise ValueError("Stage 1 training evidence contains unobserved claims")
    if (
        payload["run_id"] != expected_run_id
        or payload["method"] != expected_method
        or payload["zero_stage"] != expected_zero_stage
        or payload["world_size"] != expected_world_size
        or payload["global_step"] != expected_global_step
        or payload["checkpoint_path"] != str(Path(expected_checkpoint))
    ):
        raise ValueError("Stage 1 training evidence identity is inconsistent")
    return payload


def publish_stage1_training_evidence(
    path: str | Path,
    **evidence: object,
) -> dict[str, object]:
    """Exclusively publish one already observed ordinary Stage 1 result."""
    payload = build_stage1_training_evidence(**evidence)
    _publish_bytes_no_replace(Path(path), _canonical_json_bytes(payload) + b"\n")
    return payload
