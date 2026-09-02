from argparse import Namespace
from pathlib import Path

import pytest

import src.artifact_paths as artifact_paths
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
        "/root/workspace/ILVR/"
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


def test_canonical_checkpoint_path_preserves_repository_symlink_spelling():
    """Catches validation returning the checkpoint symlink's storage target."""
    canonical = Path(
        "/root/workspace/ILVR/checkpoints/zero-stage-compatibility/example-run"
    )

    assert require_canonical_artifact_path(canonical, kind="checkpoint") == canonical


def test_canonical_path_normalizes_descendant_symlink_alias(tmp_path, monkeypatch):
    """Catches a run alias silently targeting another canonical experiment."""
    storage_root = tmp_path / "checkpoint-storage"
    resolved_root = storage_root / "zero-stage-compatibility"
    actual_run = resolved_root / "actual-run"
    actual_run.mkdir(parents=True)
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    (repository_root / "checkpoints").symlink_to(
        storage_root, target_is_directory=True
    )
    canonical_root = repository_root / "checkpoints" / "zero-stage-compatibility"
    (canonical_root / "alias-run").symlink_to(actual_run, target_is_directory=True)
    monkeypatch.setattr(
        artifact_paths, "CANONICAL_CHECKPOINT_ROOT", canonical_root
    )

    requested = canonical_root / "alias-run" / "model.safetensors"

    assert require_canonical_artifact_path(requested, kind="checkpoint") == (
        canonical_root / "actual-run" / "model.safetensors"
    )


def test_canonical_path_normalizes_parent_segments(tmp_path, monkeypatch):
    """Catches lexical parent segments surviving canonical validation."""
    canonical_root = tmp_path / "checkpoints" / "zero-stage-compatibility"
    (canonical_root / "actual-run").mkdir(parents=True)
    monkeypatch.setattr(
        artifact_paths, "CANONICAL_CHECKPOINT_ROOT", canonical_root
    )

    requested = canonical_root / "discarded-run" / ".." / "actual-run" / "model"

    assert require_canonical_artifact_path(requested, kind="checkpoint") == (
        canonical_root / "actual-run" / "model"
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


def test_generated_paths_accept_only_the_run_stage2_length_manifest(stage1_args):
    """Catches a Stage 2 manifest being published outside its run result directory."""
    paths = resolve_run_artifact_paths("stage1-zero2-dora-mb4-ga2-candidate")
    stage1_args.save_model_path = str(paths.checkpoint_dir)
    stage1_args.log_file = str(paths.log_dir / "train.log")
    stage1_args.stage2_length_manifest = str(
        paths.result_dir / "stage2-length-manifest.json"
    )

    assert validate_generated_artifact_paths(stage1_args) == paths

    stage1_args.stage2_length_manifest = str(paths.result_dir / "other.json")
    with pytest.raises(ValueError, match="stage2_length_manifest"):
        validate_generated_artifact_paths(stage1_args)


def test_generated_paths_reject_different_canonical_experiment_ids(stage1_args):
    """Catches launcher preflight accepting separate canonical run directories."""
    checkpoint = resolve_run_artifact_paths("stage1-zero2-dora-mb4-ga2-candidate")
    other = resolve_run_artifact_paths("stage1-zero3-dora-mb4-ga2-candidate")
    stage1_args.save_model_path = str(checkpoint.checkpoint_dir)
    stage1_args.log_file = str(other.log_dir / "train.log")

    with pytest.raises(ValueError, match="share one experiment ID"):
        validate_generated_artifact_paths(stage1_args)
