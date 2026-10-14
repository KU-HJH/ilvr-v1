"""Strict bounded producer evidence for ordinary Stage 1 training."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.multinode_runtime import _canonical_json_bytes, _publish_bytes_no_replace


_FIELDS_V1 = {
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
_FIELDS_V2 = _FIELDS_V1 | {
    "micro_batch",
    "gradient_accumulation_steps",
    "effective_global_batch",
    "ema_tau",
    "teacher_update_mode",
    "seed",
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
    micro_batch: int,
    gradient_accumulation_steps: int,
    ema_tau: float,
    seed: int,
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
    if type(micro_batch) is not int or micro_batch <= 0:
        raise ValueError("Stage 1 training evidence micro batch is invalid")
    if (
        type(gradient_accumulation_steps) is not int
        or gradient_accumulation_steps <= 0
    ):
        raise ValueError("Stage 1 training evidence accumulation is invalid")
    if type(seed) is not int or seed < 0:
        raise ValueError("Stage 1 training evidence seed is invalid")
    if not _finite_number(ema_tau) or not 0.0 <= float(ema_tau) <= 1.0:
        raise ValueError("Stage 1 training evidence EMA decay is invalid")
    normalized_ema_tau = float(ema_tau)
    return {
        "schema_version": 2,
        "kind": "stage1-training-evidence",
        "run_id": run_id,
        "stage": "stage1",
        "method": method,
        "zero_stage": zero_stage,
        "world_size": world_size,
        "process_ranks": list(range(world_size)),
        "micro_batch": micro_batch,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_global_batch": (
            micro_batch * world_size * gradient_accumulation_steps
        ),
        "ema_tau": normalized_ema_tau,
        "teacher_update_mode": (
            "frozen" if normalized_ema_tau == 1.0 else "moving"
        ),
        "seed": seed,
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
    expected_producer_checkpoint: str | Path | None = None,
    expected_micro_batch: int | None = None,
    expected_gradient_accumulation_steps: int | None = None,
    expected_ema_tau: float | None = None,
    expected_seed: int | None = None,
) -> dict[str, object]:
    """Decode and revalidate one exact ordinary Stage 1 producer object."""
    if type(raw) is not bytes or not raw or len(raw) > 1024 * 1024:
        raise ValueError("Stage 1 training evidence size is invalid")
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage 1 training evidence is invalid JSON") from error
    if type(payload) is not dict:
        raise ValueError("Stage 1 training evidence fields are invalid")
    schema_version = payload.get("schema_version")
    if schema_version == 1:
        expected_fields = _FIELDS_V1
    elif schema_version == 2:
        expected_fields = _FIELDS_V2
    else:
        raise ValueError("Stage 1 training evidence schema is invalid")
    if set(payload) != expected_fields:
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
    common = {
        "run_id": payload["run_id"],
        "method": payload["method"],
        "zero_stage": payload["zero_stage"],
        "world_size": payload["world_size"],
        "global_step": payload["global_step"],
        "checkpoint_path": payload["checkpoint_path"],
        "loss": payload["loss"],
    }
    if expected_producer_checkpoint is not None:
        if payload["checkpoint_path"] != str(Path(expected_producer_checkpoint)):
            raise ValueError("Stage 1 training evidence producer identity is inconsistent")
        common["checkpoint_path"] = expected_checkpoint
    if schema_version == 1:
        run_id, method, zero_stage, world_size, global_step, checkpoint = (
            _validate_identity(
                run_id=common["run_id"],
                method=common["method"],
                zero_stage=common["zero_stage"],
                world_size=common["world_size"],
                global_step=common["global_step"],
                checkpoint_path=common["checkpoint_path"],
            )
        )
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ValueError("Stage 1 training evidence checkpoint must exist")
        if not _finite_number(common["loss"]):
            raise ValueError(
                "Stage 1 training evidence requires observed finite loss"
            )
        rebuilt = {
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
            "loss": float(common["loss"]),
            "finite_loss": True,
        }
    else:
        rebuilt = build_stage1_training_evidence(
            **common,
            micro_batch=payload["micro_batch"],
            gradient_accumulation_steps=payload[
                "gradient_accumulation_steps"
            ],
            ema_tau=payload["ema_tau"],
            seed=payload["seed"],
        )
    if expected_producer_checkpoint is not None:
        rebuilt["checkpoint_path"] = str(Path(expected_producer_checkpoint))
    if payload != rebuilt:
        raise ValueError("Stage 1 training evidence contains unobserved claims")
    expected_config = (
        expected_micro_batch,
        expected_gradient_accumulation_steps,
        expected_ema_tau,
        expected_seed,
    )
    if any(value is not None for value in expected_config):
        if any(value is None for value in expected_config):
            raise ValueError(
                "Stage 1 training evidence expected configuration is incomplete"
            )
        if schema_version == 1:
            raise ValueError(
                "Stage 1 training evidence configuration is unavailable"
            )
        assert expected_micro_batch is not None
        assert expected_gradient_accumulation_steps is not None
        assert expected_ema_tau is not None
        assert expected_seed is not None
        if (
            payload["micro_batch"] != expected_micro_batch
            or payload["gradient_accumulation_steps"]
            != expected_gradient_accumulation_steps
            or payload["effective_global_batch"]
            != expected_micro_batch
            * expected_world_size
            * expected_gradient_accumulation_steps
            or payload["ema_tau"] != float(expected_ema_tau)
            or payload["seed"] != expected_seed
        ):
            raise ValueError(
                "Stage 1 training evidence configuration is inconsistent"
            )
    if (
        payload["run_id"] != expected_run_id
        or payload["method"] != expected_method
        or payload["zero_stage"] != expected_zero_stage
        or payload["world_size"] != expected_world_size
        or payload["global_step"] != expected_global_step
        or payload["checkpoint_path"] != str(Path(expected_producer_checkpoint or expected_checkpoint))
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
