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
    )


def test_run_paths_share_one_experiment_identifier():
    """Catches a run writing related artifacts beneath different experiment IDs."""
    paths = resolve_run_artifact_paths("stage1-zero2-dora-mb4-ga2-candidate")

    assert paths.checkpoint_dir == Path(
        "/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/"
        "checkpoints/zero-stage-compatibility/stage1-zero2-dora-mb4-ga2-candidate"
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


def test_generated_checkpoint_result_and_log_share_experiment_id(stage1_args):
    """Catches generated checkpoint, result, and log paths with mismatched IDs."""
    paths = resolve_run_artifact_paths("stage1-zero2-dora-mb4-ga2-candidate")
    stage1_args.save_model_path = str(paths.checkpoint_dir)
    stage1_args.log_file = str(paths.log_dir / "train.log")
    stage1_args.benchmark_evidence_path = str(
        paths.result_dir / "producer-evidence.json"
    )

    assert validate_generated_artifact_paths(stage1_args) == paths
