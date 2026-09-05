#!/usr/bin/env python3
"""Benchmark naive and tree-packed Stage 2B training on one fixed sample."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from step_preference.shared_prefix_measurement import (
    add_shared_measurement_arguments, validate_measurement_arguments, run_shared_measurement_cli,
)

from step_preference.benchmark import (
    CudaBenchmarkRuntime,
    Stage2BTrainingStep,
    benchmark_training_path,
    evaluate_benchmark_gate,
)
from step_preference.equivalence import (
    load_production_context,
    write_evidence_exclusive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--data_path")
    parser.add_argument("--image_root")
    parser.add_argument("--negative_path")
    parser.add_argument("--negative_manifest")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_path", type=Path, required=True)
    add_shared_measurement_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    validate_measurement_arguments(parser, args)
    if args.output_path.exists():
        parser.error(f"output path already exists: {args.output_path}")
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    context = None
    try:
        if args.shared_prefix:
            return run_shared_measurement_cli(args, benchmark=True, git_commit=git_commit)
        context = load_production_context(
            checkpoint_path=args.checkpoint_path,
            data_path=args.data_path,
            image_root=args.image_root,
            negative_path=args.negative_path,
            negative_manifest=args.negative_manifest,
            device=args.device,
        )
        runtime = CudaBenchmarkRuntime(args.device)
        torch.manual_seed(307)
        naive_step = Stage2BTrainingStep(
            context.model, context.batch, context.naive_config
        )
        packed_step = Stage2BTrainingStep(
            context.model, context.batch, context.packed_config
        )
        if naive_step.autocast_dtype != packed_step.autocast_dtype:
            raise ValueError("benchmark paths must use the same autocast dtype")
        naive = benchmark_training_path(naive_step, runtime)
        context.model.zero_grad(set_to_none=True)
        torch.manual_seed(307)
        packed = benchmark_training_path(packed_step, runtime)
        gate = evaluate_benchmark_gate(
            naive_ms=naive.median_ms,
            packed_ms=packed.median_ms,
            naive_bytes=naive.peak_allocated_bytes,
            packed_bytes=packed.peak_allocated_bytes,
        )
        identity = dict(context.identity)
        identity["dtype"] = str(naive_step.autocast_dtype).removeprefix("torch.")
        identity["git_commit"] = git_commit
        report = {
            "schema_version": 1,
            "status": "pass" if gate.passed else "fail",
            "identity": identity,
            "naive": naive.to_dict(),
            "packed": packed.to_dict(),
            "time_reduction": gate.time_reduction,
            "memory_reduction": gate.memory_reduction,
            "thresholds": {
                "time_reduction": gate.time_threshold,
                "memory_reduction": gate.memory_threshold,
            },
            "errors": [] if gate.passed else ["performance threshold not met"],
        }
        write_evidence_exclusive(args.output_path, report)
        return 0 if gate.passed else 1
    except Exception as error:
        if context is not None:
            identity = dict(context.identity)
            identity["git_commit"] = git_commit
            write_evidence_exclusive(
                args.output_path,
                {
                    "schema_version": 1,
                    "status": "fail",
                    "identity": identity,
                    "errors": [str(error)],
                },
            )
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
