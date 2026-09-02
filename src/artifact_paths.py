"""Canonical ownership boundaries for generated run artifacts."""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


CANONICAL_REPOSITORY_ROOT = Path(
    "/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR"
)
CANONICAL_CHECKPOINT_ROOT = (
    CANONICAL_REPOSITORY_ROOT / "checkpoints" / "zero-stage-compatibility"
)
CANONICAL_RESULT_ROOT = (
    CANONICAL_REPOSITORY_ROOT / "results" / "zero-stage-compatibility"
)
CANONICAL_LOG_ROOT = CANONICAL_REPOSITORY_ROOT / "logs" / "zero-stage-compatibility"
EXPERIMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,127}$")


@dataclass(frozen=True)
class RunArtifactPaths:
    experiment_id: str
    checkpoint_dir: Path
    result_dir: Path
    log_dir: Path


def resolve_run_artifact_paths(experiment_id: str) -> RunArtifactPaths:
    """Return the three canonical directories for a safe run identity."""
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(
            "experiment ID must be 3-128 lowercase letters, digits, or hyphens"
        )
    return RunArtifactPaths(
        experiment_id=experiment_id,
        checkpoint_dir=CANONICAL_CHECKPOINT_ROOT / experiment_id,
        result_dir=CANONICAL_RESULT_ROOT / experiment_id,
        log_dir=CANONICAL_LOG_ROOT / experiment_id,
    )


def require_canonical_artifact_path(
    path: str | Path,
    *,
    kind: Literal["checkpoint", "result", "log"],
) -> Path:
    """Resolve and require a path strictly beneath the canonical root for *kind*."""
    roots = {
        "checkpoint": CANONICAL_CHECKPOINT_ROOT,
        "result": CANONICAL_RESULT_ROOT,
        "log": CANONICAL_LOG_ROOT,
    }
    if path is None:
        raise ValueError(f"path must be below the canonical {kind} root: {roots[kind]}")
    candidate = Path(path).expanduser().resolve(strict=False)
    root = roots[kind].resolve(strict=False)
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(f"path must be below the canonical {kind} root: {root}")
    return candidate


def validate_generated_artifact_paths(args: argparse.Namespace) -> RunArtifactPaths:
    """Require generated paths to belong to one canonical experiment identity."""
    checkpoint_dir = require_canonical_artifact_path(
        args.save_model_path, kind="checkpoint"
    )
    log_path = require_canonical_artifact_path(args.log_file, kind="log")
    experiment_id = checkpoint_dir.name
    paths = resolve_run_artifact_paths(experiment_id)
    if checkpoint_dir != paths.checkpoint_dir or log_path.parent != paths.log_dir:
        raise ValueError("checkpoint and log paths must share one experiment ID")
    for attribute in (
        "benchmark_run_config",
        "benchmark_evidence_path",
        "benchmark_smoke_resume_witness",
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            result = require_canonical_artifact_path(value, kind="result")
            if result.parent != paths.result_dir:
                raise ValueError(
                    f"{attribute} must use result experiment ID {experiment_id}"
                )
    manifest = getattr(args, "stage2_length_manifest", None)
    if manifest is not None:
        manifest_path = require_canonical_artifact_path(manifest, kind="result")
        expected_manifest = paths.result_dir / "stage2-length-manifest.json"
        if manifest_path != expected_manifest:
            raise ValueError(
                "stage2_length_manifest must use the run stage2-length-manifest.json"
            )
    return paths


def _json_paths(paths: RunArtifactPaths) -> str:
    return json.dumps({key: str(value) for key, value in asdict(paths).items()})


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve_parser = subparsers.add_parser("resolve")
    resolve_parser.add_argument("--experiment-id", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--kind", choices=("checkpoint", "result", "log"), required=True)
    validate_parser.add_argument("--path", required=True)
    run_parser = subparsers.add_parser("validate-run")
    run_parser.add_argument("--checkpoint", required=True)
    run_parser.add_argument("--log", required=True)
    run_parser.add_argument("--benchmark-run-config")
    run_parser.add_argument("--benchmark-evidence-path")
    run_parser.add_argument("--benchmark-smoke-resume-witness")
    run_parser.add_argument("--stage2-length-manifest")
    args = parser.parse_args(argv)
    try:
        if args.command == "resolve":
            print(_json_paths(resolve_run_artifact_paths(args.experiment_id)))
        elif args.command == "validate":
            print(require_canonical_artifact_path(args.path, kind=args.kind))
        else:
            paths = validate_generated_artifact_paths(
                argparse.Namespace(
                    save_model_path=args.checkpoint,
                    log_file=args.log,
                    benchmark_run_config=args.benchmark_run_config,
                    benchmark_evidence_path=args.benchmark_evidence_path,
                    benchmark_smoke_resume_witness=args.benchmark_smoke_resume_witness,
                    stage2_length_manifest=args.stage2_length_manifest,
                )
            )
            print(_json_paths(paths))
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
