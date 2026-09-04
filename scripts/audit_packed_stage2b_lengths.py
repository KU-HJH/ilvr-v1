#!/usr/bin/env python3
"""Audit authenticated Stage 2B packed lengths without loading a model."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.select_jsonl_rows import select_jsonl_rows


_RECORD_KEYS = (
    "source_index",
    "sample_id",
    "clean_length",
    "step_count",
    "branch_count",
    "unpadded_length",
    "padded_length",
)


def _validated_record(record: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(record, Mapping) or set(record) != set(_RECORD_KEYS):
        raise ValueError("packed length records must contain the exact audit fields")
    result = {name: record[name] for name in _RECORD_KEYS}
    if not isinstance(result["sample_id"], str) or not result["sample_id"]:
        raise ValueError("packed length sample_id must be nonempty text")
    for name in _RECORD_KEYS:
        if name == "sample_id":
            continue
        if type(result[name]) is not int or result[name] < 0:
            raise ValueError(f"packed length {name} must be a nonnegative integer")
    if result["clean_length"] <= 0 or result["unpadded_length"] <= 0:
        raise ValueError("packed and clean lengths must be positive")
    if result["padded_length"] < result["unpadded_length"]:
        raise ValueError("padded length must not be less than unpadded length")
    return result


def select_packed_length_winner(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Select Python tuple maximum in the declared packed-capacity order."""
    validated = tuple(_validated_record(record) for record in records)
    if not validated:
        raise ValueError("packed length audit requires at least one retained record")
    return max(
        validated,
        key=lambda record: (
            record["padded_length"],
            record["unpadded_length"],
            record["clean_length"],
            record["source_index"],
        ),
    )


def publish_packed_length_audit(
    *,
    input_path: Path,
    report_path: Path,
    subset_output_path: Path,
    identity: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> None:
    """Publish one exclusive audit report and its exact winning source row."""
    from step_preference.equivalence import write_evidence_exclusive

    report_path = Path(report_path)
    subset_output_path = Path(subset_output_path)
    if report_path == subset_output_path:
        raise ValueError("report and subset output paths must be distinct")
    for path in (report_path, subset_output_path):
        if path.exists():
            raise ValueError(f"output path already exists: {path}")
    validated = tuple(_validated_record(record) for record in records)
    winner = select_packed_length_winner(validated)
    report = {
        "schema_version": 1,
        "identity": dict(identity),
        "records": list(validated),
        "winner": winner,
    }
    json.dumps(report, allow_nan=False)
    write_evidence_exclusive(report_path, report)
    select_jsonl_rows(
        Path(input_path), subset_output_path, (int(winner["source_index"]),)
    )


def collect_packed_length_audit(args) -> tuple[dict[str, object], tuple[dict, ...]]:
    """Run strict Stage 2B preflight, filtering, collation, and pure planning."""
    from main import preflight_stage2b_artifacts
    from stage2.checkpointing import resolve_latent_token_ids
    from stage2.data import (
        Stage2DataCollator,
        build_prepared_example_length_measure,
        filter_stage2_examples,
    )
    from step_preference.negatives import read_negative_manifest
    from step_preference.packing import build_packed_tree_plan
    from transformers import AutoConfig, AutoProcessor
    from utils_deepseed import parse_args

    checkpoint = Path(args.checkpoint_path).resolve(strict=True)
    source = Path(args.data_path).resolve(strict=True)
    images = Path(args.image_root).resolve(strict=True)
    negatives = Path(args.negative_path).resolve(strict=True)
    manifest_path = Path(args.negative_manifest).resolve(strict=True)
    production_args = parse_args(
        [
            "--stage",
            "stage2",
            "--load_model_path",
            str(checkpoint),
            "--data_path",
            str(source),
            "--image_root",
            str(images),
            "--step_preference_enabled",
            "--step_preference_mode",
            "stage2b",
            "--stage2a_checkpoint_path",
            str(checkpoint),
            "--step_negative_path",
            str(negatives),
            "--step_negative_manifest",
            str(manifest_path),
            "--attention_implementation",
            "sdpa",
            "--tuning_method",
            "full",
            "--no-gradient_checkpointing",
        ]
    )
    preflight = preflight_stage2b_artifacts(production_args)
    manifest = read_negative_manifest(manifest_path)
    processor = AutoProcessor.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    model_config = AutoConfig.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    token_ids = resolve_latent_token_ids(processor.tokenizer, model_config)
    kept, _stats = filter_stage2_examples(
        examples=preflight.prepared_examples,
        measure_length=build_prepared_example_length_measure(processor),
        max_length=production_args.max_seq_length_train,
    )
    source_indices = {
        example.sample_id: source_index
        for source_index, example in enumerate(preflight.prepared_examples)
    }
    collator = Stage2DataCollator(
        processor,
        token_ids,
        production_args.latent_size,
        production_args.supervise_latent_boundaries,
        step_preference_enabled=True,
        negative_store=preflight.negative_store,
    )
    records = []
    for example in kept:
        batch = collator([example])
        plan = build_packed_tree_plan(
            clean_input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            reasoning_spans=batch["reasoning_step_spans"],
            joined_steps=batch["step_preference"]["steps"],
        )
        records.append(
            {
                "source_index": source_indices[example.sample_id],
                "sample_id": example.sample_id,
                "clean_length": int(plan.root_length),
                "step_count": len(batch["reasoning_step_spans"]),
                "branch_count": len(plan.branches),
                "unpadded_length": int(plan.unpadded_length),
                "padded_length": int(plan.padded_length),
            }
        )
    identity = {
        "checkpoint_path": str(checkpoint),
        "data_path": str(source),
        "image_root": str(images),
        "negative_path": str(negatives),
        "negative_manifest": str(manifest_path),
        "negative_sha256": manifest.negative_sha256,
        "assistant_plan_rendering": manifest.assistant_plan_rendering,
        "max_sequence_length": production_args.max_seq_length_train,
    }
    return identity, tuple(records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--negative_path", required=True)
    parser.add_argument("--negative_manifest", required=True)
    parser.add_argument("--report_path", type=Path, required=True)
    parser.add_argument("--subset_output_path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.report_path.exists():
        parser.error(f"output path already exists: {args.report_path}")
    if args.subset_output_path.exists():
        parser.error(f"output path already exists: {args.subset_output_path}")
    try:
        identity, records = collect_packed_length_audit(args)
        publish_packed_length_audit(
            input_path=Path(args.data_path),
            report_path=args.report_path,
            subset_output_path=args.subset_output_path,
            identity=identity,
            records=records,
        )
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
