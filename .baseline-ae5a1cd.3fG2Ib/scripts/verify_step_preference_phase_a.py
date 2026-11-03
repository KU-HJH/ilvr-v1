#!/usr/bin/env python3
"""Verify the focused one-step Stage 2B Phase A integration gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from transformers import AutoConfig
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    validate_stage2_config,
)

from stage2.checkpointing import (
    Stage2ArtifactKind,
    decode_stage2_peft_metadata_snapshot,
    inspect_stage2_artifact,
)
from step_preference.negatives import (
    _negative_store,
    _parse_json,
    _read_regular_file,
    _row_from_payload,
    _verify_checkpoint_files,
    read_negative_manifest,
)


PHASE_A_SUMMARY_PREFIX = "STEP_PREFERENCE_PHASE_A_SUMMARY "
_SUMMARY_FIELDS = {
    "schema_version",
    "global_step",
    "optimizer_step",
    "training_loss",
    "total_loss",
    "gradient_norm",
    "metrics",
}
_METRIC_FIELDS = (
    "loss/ce",
    "loss/dpo_visual",
    "loss/dpo_text",
    "loss/entropy_visual",
    "loss/entropy_text",
    "stats/valid_visual_pairs",
    "stats/valid_textual_pairs",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative_manifest", type=Path, required=True)
    parser.add_argument("--negative_path", type=Path, required=True)
    parser.add_argument("--training_log", type=Path, required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser


def _finite_number(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _pair_count(value, label: str) -> float:
    count = _finite_number(value, label)
    if count < 0 or not count.is_integer():
        raise ValueError(f"{label} must be a nonnegative integer-valued number")
    return count


def _read_training_summary(path: Path) -> dict:
    raw = _read_regular_file(path, "Stage 2B training log")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Stage 2B training log must be UTF-8") from error
    summary_lines = [
        line for line in text.splitlines() if PHASE_A_SUMMARY_PREFIX in line
    ]
    if len(summary_lines) != 1:
        raise ValueError(
            "Stage 2B training log must contain exactly one canonical Phase A summary"
        )
    encoded = summary_lines[0].split(PHASE_A_SUMMARY_PREFIX, 1)[1]
    payload = _parse_json(encoded.encode("utf-8"), "Phase A training summary")
    if not isinstance(payload, dict) or set(payload) != _SUMMARY_FIELDS:
        raise ValueError("Phase A training summary has unknown or missing fields")
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise ValueError("unsupported Phase A training summary schema_version")
    global_step = payload["global_step"]
    optimizer_step = payload["optimizer_step"]
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("Phase A training summary requires a positive global step")
    if type(optimizer_step) is not int or optimizer_step <= 0:
        raise ValueError("Phase A training summary requires a positive optimizer step")
    if optimizer_step != global_step:
        raise ValueError("Phase A optimizer and global steps disagree")
    training_loss = _finite_number(payload["training_loss"], "training loss")
    total_loss = _finite_number(payload["total_loss"], "total loss")
    gradient_norm = _finite_number(payload["gradient_norm"], "gradient norm")
    metrics = payload["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("Phase A training metrics have unknown or missing fields")
    normalized_metrics = {
        name: _finite_number(metrics[name], f"training metric {name}")
        for name in _METRIC_FIELDS
    }
    visual_pairs = _pair_count(
        metrics["stats/valid_visual_pairs"], "training visual pair count"
    )
    textual_pairs = _pair_count(
        metrics["stats/valid_textual_pairs"], "training textual pair count"
    )
    if visual_pairs + textual_pairs < 1:
        raise ValueError("Phase A training did not observe a valid preference pair")
    return {
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "training_loss": training_loss,
        "total_loss": total_loss,
        "gradient_norm": gradient_norm,
        "metrics": normalized_metrics,
        "valid_pairs": visual_pairs + textual_pairs,
    }


def _read_negative_evidence(negative_path: Path, manifest_path: Path) -> dict:
    manifest = read_negative_manifest(manifest_path)
    _verify_checkpoint_files(manifest)
    raw = _read_regular_file(negative_path, "negative artifact")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest.negative_sha256:
        raise ValueError("negative artifact SHA-256 does not match its manifest")
    rows = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ValueError(f"negative JSONL row {line_number} is empty")
        rows.append(
            _row_from_payload(
                _parse_json(line, f"negative JSONL row {line_number}")
            )
        )
    _negative_store(rows)
    visual_pairs = sum(row.valid_visual_negative for row in rows)
    textual_pairs = sum(row.valid_textual_negative for row in rows)
    if visual_pairs + textual_pairs <= 0:
        raise ValueError("authenticated negative artifact contains no valid pair")
    return {
        "negative_path": str(negative_path.resolve(strict=True)),
        "negative_sha256": digest,
        "row_count": len(rows),
        "valid_visual_pairs": visual_pairs,
        "valid_textual_pairs": textual_pairs,
        "valid_pairs": visual_pairs + textual_pairs,
    }


def _validate_pair_agreement(negative: dict, training: dict) -> None:
    metrics = training["metrics"]
    for modality in ("visual", "textual"):
        trained = metrics[f"stats/valid_{modality}_pairs"]
        available = negative[f"valid_{modality}_pairs"]
        if trained > available:
            raise ValueError(
                "training pair counts exceed authenticated negative artifact: "
                f"{modality} trained={trained}, available={available}"
            )


def _validate_checkpoint(path: Path) -> dict:
    inspection = inspect_stage2_artifact(path)
    canonical = path.resolve(strict=True)
    if inspection.artifact_kind is Stage2ArtifactKind.PEFT:
        snapshot = inspection.peft_metadata_snapshot
        if snapshot is None:
            raise ValueError("Stage 2 PEFT inspection lost its metadata snapshot")
        config = decode_stage2_peft_metadata_snapshot(snapshot)
    else:
        config = AutoConfig.from_pretrained(
            str(canonical), local_files_only=True
        )
    if getattr(config, "stage", None) != "stage2":
        raise ValueError("saved checkpoint config must describe stage2")
    validate_stage2_config(config)
    if type(getattr(config, "supervise_latent_boundaries", None)) is not bool:
        raise ValueError(
            "saved checkpoint supervise_latent_boundaries must be a boolean"
        )
    return {
        "path": str(canonical),
        "artifact_kind": inspection.artifact_kind.value,
    }


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        negative = _read_negative_evidence(
            args.negative_path, args.negative_manifest
        )
        training = _read_training_summary(args.training_log)
        _validate_pair_agreement(negative, training)
        checkpoint = _validate_checkpoint(args.checkpoint_path)
        report = {
            "schema_version": 1,
            "status": "pass",
            "negative_artifact": negative,
            "training": training,
            "checkpoint": checkpoint,
            "errors": [],
        }
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "fail",
            "errors": [str(error)],
        }
        _write_report(args.output_path, report)
        return 1
    _write_report(args.output_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
