"""Strict small evidence shared by the approved lifecycle harnesses."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_CHECKPOINT_ROOT = _REPOSITORY_ROOT / "checkpoints"
_RANK_FIELDS = {
    "rank", "global_step", "optimizer_step", "loss", "gradient_norm",
    "worker_error",
}
_PHASE_FIELDS = {
    "schema_version", "kind", "phase", "stage", "method", "zero_stage",
    "config_digest", "experiment_id", "world_size", "process_ranks",
    "global_step", "optimizer_step", "checkpoint_path", "artifact_kind",
    "finite_loss", "finite_gradient", "worker_errors", "ranks",
}


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("lifecycle phase evidence contains a duplicate field")
        result[key] = value
    return result


def _expected_steps(stage: str, phase: str) -> tuple[int, int]:
    if phase == "initial":
        return 2, 2
    if phase == "resume":
        return 3, 1 if stage == "stage1" else 3
    raise ValueError("lifecycle phase must be initial or resume")


def _validate_decoded_types(payload: dict[str, object]) -> None:
    """Reject JSON scalar aliases before any equality-based reconstruction."""
    if (
        type(payload["schema_version"]) is not int
        or type(payload["kind"]) is not str
        or type(payload["phase"]) is not str
        or type(payload["stage"]) is not str
        or type(payload["method"]) is not str
        or type(payload["zero_stage"]) is not int
        or type(payload["config_digest"]) is not str
        or type(payload["experiment_id"]) is not str
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
        raise ValueError("lifecycle phase evidence types are invalid")


def _validate_identity(
    *, phase: object, stage: object, method: object, zero_stage: object,
    config_digest: object, experiment_id: object, checkpoint_path: object,
    artifact_kind: object,
) -> tuple[str, str, str, int, str, str, Path, str]:
    from src.multinode import _RUN_ID_PATTERN

    if type(phase) is not str or phase not in ("initial", "resume"):
        raise ValueError("lifecycle phase is invalid")
    if type(stage) is not str or stage not in ("stage1", "stage2"):
        raise ValueError("lifecycle phase stage is invalid")
    if type(method) is not str or method not in ("full", "lora", "dora"):
        raise ValueError("lifecycle phase method is invalid")
    if type(zero_stage) is not int or zero_stage not in (1, 2, 3):
        raise ValueError("lifecycle phase ZeRO stage is invalid")
    if (
        type(config_digest) is not str or len(config_digest) != 64
        or any(character not in "0123456789abcdef" for character in config_digest)
    ):
        raise ValueError("lifecycle phase config digest is invalid")
    if type(experiment_id) is not str or not _RUN_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("lifecycle phase experiment ID is invalid")
    if type(checkpoint_path) is not str and not isinstance(checkpoint_path, Path):
        raise ValueError("lifecycle phase checkpoint path is invalid")
    checkpoint = Path(checkpoint_path)
    expected_checkpoint = _CANONICAL_CHECKPOINT_ROOT / experiment_id / "checkpoint-2"
    if checkpoint != expected_checkpoint:
        raise ValueError("lifecycle phase checkpoint is not canonical checkpoint-2")
    if type(artifact_kind) is not str or artifact_kind not in ("full", "peft"):
        raise ValueError("lifecycle phase artifact kind is invalid")
    if (method == "full") != (artifact_kind == "full"):
        raise ValueError("lifecycle phase artifact kind disagrees with observed method")
    return (
        phase, stage, method, zero_stage, config_digest, experiment_id,
        checkpoint, artifact_kind,
    )


def build_lifecycle_phase_evidence(
    *, phase: str, stage: str, method: str, zero_stage: int,
    config_digest: str, experiment_id: str, checkpoint_path: str | Path,
    artifact_kind: str, rank_records: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Build one phase only from four separately observed rank records."""
    (
        phase, stage, method, zero_stage, config_digest, experiment_id,
        checkpoint, artifact_kind,
    ) = _validate_identity(
        phase=phase, stage=stage, method=method, zero_stage=zero_stage,
        config_digest=config_digest, experiment_id=experiment_id,
        checkpoint_path=checkpoint_path, artifact_kind=artifact_kind,
    )
    if type(rank_records) not in (list, tuple) or len(rank_records) != 4:
        raise ValueError("lifecycle phase requires exactly four rank records")
    expected_global, expected_optimizer = _expected_steps(stage, phase)
    normalized: list[dict[str, object]] = []
    for record in rank_records:
        if type(record) is not dict or set(record) != _RANK_FIELDS:
            raise ValueError("lifecycle phase rank fields are invalid")
        if (
            type(record["rank"]) is not int
            or type(record["global_step"]) is not int
            or record["global_step"] != expected_global
            or type(record["optimizer_step"]) is not int
            or record["optimizer_step"] != expected_optimizer
            or not _finite(record["loss"])
            or not _finite(record["gradient_norm"])
            or record["worker_error"] is not None
        ):
            raise ValueError("lifecycle phase rank diagnostics are invalid")
        normalized.append(dict(record))
    normalized.sort(key=lambda item: item["rank"])
    if [record["rank"] for record in normalized] != [0, 1, 2, 3]:
        raise ValueError("lifecycle phase process ranks are incomplete")
    return {
        "schema_version": 1,
        "kind": "multinode-lifecycle-phase",
        "phase": phase,
        "stage": stage,
        "method": method,
        "zero_stage": zero_stage,
        "config_digest": config_digest,
        "experiment_id": experiment_id,
        "world_size": 4,
        "process_ranks": [0, 1, 2, 3],
        "global_step": expected_global,
        "optimizer_step": expected_optimizer,
        "checkpoint_path": str(checkpoint),
        "artifact_kind": artifact_kind,
        "finite_loss": True,
        "finite_gradient": True,
        "worker_errors": [],
        "ranks": normalized,
    }


def decode_lifecycle_phase_evidence(
    raw: bytes, *, phase: str, stage: str, method: str, zero_stage: int,
    config_digest: str, experiment_id: str, checkpoint_path: str | Path,
) -> dict[str, object]:
    """Decode duplicate-free bounded bytes and reject every unobserved claim."""
    if type(raw) is not bytes or not raw or len(raw) > 1024 * 1024:
        raise ValueError("lifecycle phase evidence size is invalid")
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("lifecycle phase evidence is invalid JSON") from error
    if type(payload) is not dict or set(payload) != _PHASE_FIELDS:
        raise ValueError("lifecycle phase evidence fields are invalid")
    _validate_decoded_types(payload)
    rebuilt = build_lifecycle_phase_evidence(
        phase=payload["phase"], stage=payload["stage"], method=payload["method"],
        zero_stage=payload["zero_stage"], config_digest=payload["config_digest"],
        experiment_id=payload["experiment_id"],
        checkpoint_path=payload["checkpoint_path"],
        artifact_kind=payload["artifact_kind"], rank_records=payload["ranks"],
    )
    if payload != rebuilt:
        raise ValueError("lifecycle phase evidence contains unobserved claims")
    if (
        payload["phase"] != phase or payload["stage"] != stage
        or payload["method"] != method or payload["zero_stage"] != zero_stage
        or payload["config_digest"] != config_digest
        or payload["experiment_id"] != experiment_id
        or payload["checkpoint_path"] != str(Path(checkpoint_path))
    ):
        raise ValueError("lifecycle phase evidence identity is inconsistent")
    return payload


__all__ = ["build_lifecycle_phase_evidence", "decode_lifecycle_phase_evidence"]
