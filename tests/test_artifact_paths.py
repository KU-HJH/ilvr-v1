from argparse import Namespace
from pathlib import Path

import pytest

from src.artifact_paths import (
    resolve_run_artifact_paths,
    require_canonical_artifact_path,
    validate_generated_artifact_paths,
)


@pytest.fixture
def stage1_args():
    return Namespace(
        save_model_path=None,
        log_file=None,
        benchmark_run_config=None,
        benchmark_evidence_path=None,
        benchmark_smoke_resume_witness=None,
        training_evidence_path=None,
    )


def test_run_paths_share_one_experiment_identifier():
    """Catches a run writing related artifacts beneath different experiment IDs."""
    paths = resolve_run_artifact_paths("stage1-zero2-dora-mb4-ga2-candidate")

    assert paths.checkpoint_dir == Path(
        "/root/workspace/ILVR/checkpoints/stage1-zero2-dora-mb4-ga2-candidate"
    )
    assert paths.result_dir == Path(
        "/root/workspace/ILVR/results/stage1-zero2-dora-mb4-ga2-candidate"
    )
    assert paths.log_dir == Path(
        "/root/workspace/ILVR/logs/zero-stage-compatibility/"
        "stage1-zero2-dora-mb4-ga2-candidate"
    )
    assert paths.result_dir.name == paths.checkpoint_dir.name
    assert paths.log_dir.name == paths.checkpoint_dir.name


@pytest.mark.parametrize("experiment_id", ["../escape", "has/slash", "UPPER", ""])
def test_experiment_identifier_rejects_unsafe_values(experiment_id):
    """Catches unsafe experiment IDs escaping the canonical artifact roots."""
    with pytest.raises(ValueError, match="experiment ID"):
        resolve_run_artifact_paths(experiment_id)


def test_worktree_local_outputs_are_not_canonical():
    """Catches worktree-local result files bypassing repository ownership."""
    with pytest.raises(ValueError, match="canonical result root"):
        require_canonical_artifact_path(
            Path.cwd() / "outputs" / "evidence.json", kind="result"
        )


def test_canonical_checkpoint_path_preserves_repository_symlink_spelling():
    """Catches validation returning the checkpoint symlink's storage target."""
    canonical = Path(
        "/root/workspace/ILVR/checkpoints/example-run"
    )

    assert require_canonical_artifact_path(canonical, kind="checkpoint") == canonical


def test_generated_checkpoint_result_and_log_share_experiment_id(stage1_args):
    """Catches generated checkpoint, result, and log paths with mismatched IDs."""
    paths = resolve_run_artifact_paths("stage1-zero2-dora-mb4-ga2-candidate")
    stage1_args.save_model_path = str(paths.checkpoint_dir)
    stage1_args.log_file = str(paths.log_dir / "train.log")
    stage1_args.benchmark_evidence_path = str(
        paths.result_dir / "producer-evidence.json"
    )

    assert validate_generated_artifact_paths(stage1_args) == paths


def test_owned_training_evidence_supports_nested_capacity_result_root(stage1_args):
    run_id = "multinode-four-gpu-stage1-full-zero1-r4-batch2"
    paths = resolve_run_artifact_paths(run_id)
    stage1_args.save_model_path = str(paths.checkpoint_dir)
    stage1_args.log_file = str(paths.log_dir / "initial-train.log")
    stage1_args.training_evidence_path = str(
        Path("/root/workspace/ILVR/results/hardware/capacity")
        / run_id
        / "initial-producer-evidence.json"
    )

    assert validate_generated_artifact_paths(stage1_args) == paths

    stage1_args.training_evidence_path = str(
        Path("/root/workspace/ILVR/results/hardware/capacity/wrong-run")
        / "initial-producer-evidence.json"
    )
    with pytest.raises(ValueError, match="experiment ID"):
        validate_generated_artifact_paths(stage1_args)


def test_multinode_direct_artifact_tuple_rejects_legacy_nested_checkpoint(
    stage1_args,
):
    run_id = "multinode-four-gpu-stage2-full-zero1"
    paths = resolve_run_artifact_paths(run_id)
    stage1_args.save_model_path = str(Path("/root/workspace/ILVR/checkpoints") / run_id)
    stage1_args.log_file = str(
        Path("/root/workspace/ILVR/logs/zero-stage-compatibility")
        / run_id
        / "initial-train.log"
    )
    stage1_args.benchmark_evidence_path = str(
        Path("/root/workspace/ILVR/results")
        / run_id
        / "initial-producer-evidence.json"
    )

    assert validate_generated_artifact_paths(stage1_args) == paths

    stage1_args.save_model_path = str(
        Path("/root/workspace/ILVR/checkpoints/zero-stage-compatibility") / run_id
    )
    with pytest.raises(ValueError, match="share one experiment ID"):
        validate_generated_artifact_paths(stage1_args)


def test_generated_paths_reject_different_canonical_experiment_ids(stage1_args):
    """Catches launcher preflight accepting separate canonical run directories."""
    checkpoint = resolve_run_artifact_paths("stage1-zero2-dora-mb4-ga2-candidate")
    other = resolve_run_artifact_paths("stage1-zero3-dora-mb4-ga2-candidate")
    stage1_args.save_model_path = str(checkpoint.checkpoint_dir)
    stage1_args.log_file = str(other.log_dir / "train.log")

    with pytest.raises(ValueError, match="share one experiment ID"):
        validate_generated_artifact_paths(stage1_args)
