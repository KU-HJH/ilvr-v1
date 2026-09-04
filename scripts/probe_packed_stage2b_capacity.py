#!/usr/bin/env python3
"""Run one bounded packed Stage 2B forward/backward capacity probe."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

from step_preference.benchmark import Stage2BTrainingStep
from step_preference.equivalence import (
    load_production_context,
    write_evidence_exclusive,
)
from step_preference.packing import build_packed_tree_plan


_LOSS_METRICS = {
    "ce": "loss/ce",
    "dpo": "loss/dpo",
    "entropy": "loss/entropy",
    "total": "loss/total",
}


class CudaCapacityRuntime:
    """The CUDA measurements required around exactly one training step."""

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def device_total_bytes(self) -> int:
        return int(torch.cuda.get_device_properties(self.device).total_memory)

    def memory_allocated(self) -> int:
        return int(torch.cuda.memory_allocated(self.device))

    def memory_reserved(self) -> int:
        return int(torch.cuda.memory_reserved(self.device))

    def reset_peak_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats(self.device)

    def max_memory_allocated(self) -> int:
        return int(torch.cuda.max_memory_allocated(self.device))

    def max_memory_reserved(self) -> int:
        return int(torch.cuda.max_memory_reserved(self.device))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--negative_path", required=True)
    parser.add_argument("--negative_manifest", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--source_row_index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--tuning_method", choices=("full", "lora", "dora"), default="full"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def _sample_report(context, plan, source_row_index: int) -> dict[str, object]:
    return {
        "source_row_index": source_row_index,
        "sample_id": context.batch["sample_ids"][0],
        "clean_length": int(plan.root_length),
        "step_count": len(context.batch["reasoning_step_spans"]),
        "branch_count": len(plan.branches),
        "packed_unpadded_length": int(plan.unpadded_length),
        "packed_padded_length": int(plan.padded_length),
    }


def _empty_losses() -> dict[str, None]:
    return {name: None for name in _LOSS_METRICS}


def _memory_baseline(runtime) -> dict[str, int]:
    return {
        "device_total_bytes": int(runtime.device_total_bytes()),
        "baseline_allocated_bytes": int(runtime.memory_allocated()),
        "peak_allocated_bytes": 0,
        "baseline_reserved_bytes": int(runtime.memory_reserved()),
        "peak_reserved_bytes": 0,
    }


def _capture_peaks(runtime, memory: dict[str, int]) -> None:
    memory["peak_allocated_bytes"] = int(runtime.max_memory_allocated())
    memory["peak_reserved_bytes"] = int(runtime.max_memory_reserved())


def _losses(step) -> dict[str, float]:
    batches = getattr(step.trainer, "_step_preference_metric_batches", None)
    totals = getattr(step.trainer, "_step_preference_metric_totals", None)
    if batches != 1 or not isinstance(totals, dict):
        raise ValueError("capacity probe requires exactly one Stage 2B metric batch")
    losses = {}
    for output_name, metric_name in _LOSS_METRICS.items():
        try:
            value = float(totals[metric_name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"capacity probe is missing finite raw metric {metric_name}"
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"capacity probe raw metric {metric_name} must be finite")
        losses[output_name] = value
    return losses


def _trainable_gradient_norm(model) -> float:
    squared_norm = 0.0
    observed = False
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm(2).cpu().item())
        if not math.isfinite(norm):
            raise ValueError("capacity probe trainable gradient must be finite")
        squared_norm += norm * norm
        observed = True
    result = math.sqrt(squared_norm)
    if not observed or not math.isfinite(result) or result <= 0.0:
        raise ValueError("capacity probe trainable gradient norm must be finite and nonzero")
    return result


def run_capacity_probe(
    context,
    runtime,
    *,
    source_row_index: int,
    step_factory=Stage2BTrainingStep,
    clock=time.perf_counter,
) -> dict[str, object]:
    """Build one layout witness and execute one packed forward/backward."""
    plan = build_packed_tree_plan(
        clean_input_ids=context.batch["input_ids"],
        attention_mask=context.batch["attention_mask"],
        reasoning_spans=context.batch["reasoning_step_spans"],
        joined_steps=context.batch["step_preference"]["steps"],
    )
    report = {
        "schema_version": 1,
        "status": "error",
        "identity": dict(context.identity),
        "sample": _sample_report(context, plan, source_row_index),
        "losses": _empty_losses(),
        "gradient_norm": None,
        "memory": None,
        "elapsed_seconds": None,
        "errors": [],
    }
    started = None
    step = step_factory(context.model, context.batch, context.packed_config)
    try:
        step.zero_grad()
        runtime.synchronize()
        report["memory"] = _memory_baseline(runtime)
        runtime.reset_peak_memory_stats()
        started = clock()
        step()
        runtime.synchronize()
        report["elapsed_seconds"] = float(clock() - started)
        _capture_peaks(runtime, report["memory"])
        report["losses"] = _losses(step)
        report["gradient_norm"] = _trainable_gradient_norm(context.model)
        report["status"] = "pass"
    except Exception as error:
        if started is not None and report["elapsed_seconds"] is None:
            report["elapsed_seconds"] = float(clock() - started)
        if report["memory"] is not None:
            _capture_peaks(runtime, report["memory"])
        report["status"] = (
            "oom" if isinstance(error, torch.OutOfMemoryError) else "error"
        )
        report["errors"] = [f"{type(error).__name__}: {error}"]
    return report


def _error_report(
    args, git_commit: str | None, error: Exception
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "oom" if isinstance(error, torch.OutOfMemoryError) else "error",
        "identity": {
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "device": args.device,
            "dtype": "bfloat16",
            "tuning_method": args.tuning_method,
            "gradient_checkpointing": args.gradient_checkpointing,
            "git_commit": git_commit,
        },
        "sample": {
            "source_row_index": args.source_row_index,
            "sample_id": None,
            "clean_length": None,
            "step_count": None,
            "branch_count": None,
            "packed_unpadded_length": None,
            "packed_padded_length": None,
        },
        "losses": _empty_losses(),
        "gradient_norm": None,
        "memory": None,
        "elapsed_seconds": None,
        "errors": [f"{type(error).__name__}: {error}"],
    }


def _exit_code(status: str) -> int:
    if status == "pass":
        return 0
    if status == "oom":
        return 1
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output_path.exists():
        parser.error(f"output path already exists: {args.output_path}")
    if args.source_row_index < 0:
        parser.error("source_row_index must be nonnegative")
    git_commit = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        context = load_production_context(
            checkpoint_path=args.checkpoint_path,
            data_path=args.data_path,
            image_root=args.image_root,
            negative_path=args.negative_path,
            negative_manifest=args.negative_manifest,
            device=args.device,
            tuning_method=args.tuning_method,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        report = run_capacity_probe(
            context,
            CudaCapacityRuntime(args.device),
            source_row_index=args.source_row_index,
        )
        report["identity"]["git_commit"] = git_commit
    except Exception as error:
        report = _error_report(args, git_commit, error)
    try:
        write_evidence_exclusive(args.output_path, report)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2
    if report["errors"]:
        print(report["errors"][0], file=sys.stderr)
    return _exit_code(report["status"])


if __name__ == "__main__":
    raise SystemExit(main())
