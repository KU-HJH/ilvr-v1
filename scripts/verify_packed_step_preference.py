#!/usr/bin/env python3
"""Verify tiny FP32 and production BF16 packed Stage 2B equivalence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from step_preference.equivalence import (
    EquivalenceResult,
    load_production_context,
    run_production_bf16_equivalence,
    run_tiny_fp32_equivalence,
    write_evidence_exclusive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--negative_path", required=True)
    parser.add_argument("--negative_manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_path", type=Path, required=True)
    return parser


def _section(result: EquivalenceResult, *, production: bool) -> dict:
    gradients = dict(result.gradients)
    if production:
        gradients = {
            suffix.removesuffix(".weight"): next(
                value for name, value in result.gradients.items() if name.endswith(suffix)
            )
            for suffix in ("q_proj.weight", "o_proj.weight", "down_proj.weight")
        }
    return {
        "loss_errors": dict(result.loss_errors),
        "gradient_relative_l2": gradients,
        "epsilon_loss": result.epsilon_loss,
        "epsilon_grad": result.epsilon_grad,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
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
        tiny = run_tiny_fp32_equivalence(args.device)
        torch.cuda.empty_cache()
        context = load_production_context(
            checkpoint_path=args.checkpoint_path,
            data_path=args.data_path,
            image_root=args.image_root,
            negative_path=args.negative_path,
            negative_manifest=args.negative_manifest,
            device=args.device,
        )
        production = run_production_bf16_equivalence(
            context.model,
            context.batch,
            naive_config=context.naive_config,
            packed_config=context.packed_config,
        )
        errors = list(tiny.errors) + list(production.errors)
        identity = dict(context.identity)
        identity["git_commit"] = git_commit
        report = {
            "schema_version": 1,
            "status": "pass" if not errors else "fail",
            "identity": identity,
            "tiny_fp32": _section(tiny, production=False),
            "production_bf16": _section(production, production=True),
            "errors": errors,
        }
        write_evidence_exclusive(args.output_path, report)
        return 0 if not errors else 1
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
