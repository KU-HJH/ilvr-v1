import json

import pytest

from src.deepspeed_runtime import (
    DeepSpeedRuntimeProfile,
    canonical_deepspeed_config_path,
    load_deepspeed_runtime_profile,
)


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
def test_canonical_profile_reports_exact_stage_and_one_digest(zero_stage):
    """Catches canonical ZeRO configs losing their immutable stage identity."""
    path = canonical_deepspeed_config_path(zero_stage)
    profile = load_deepspeed_runtime_profile(path, world_size=2)

    assert profile.enabled is True
    assert profile.zero_stage == zero_stage
    assert profile.config_path == path.resolve()
    assert len(profile.config_digest) == 64
    assert profile.world_size == 2


def test_two_gpu_profile_rejects_fixed_eight_rank_mesh(tmp_path):
    """Catches a config imposing an incompatible fixed distributed topology."""
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "bf16": {"enabled": True},
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": "auto",
                "train_batch_size": "auto",
                "zero_optimization": {"stage": 2},
                "data_parallel_size": 4,
                "sequence_parallel_size": 2,
            }
        )
    )

    with pytest.raises(ValueError, match="requires world size 8"):
        load_deepspeed_runtime_profile(path, world_size=2)


def test_disabled_profile_has_no_config_identity():
    """Catches non-DeepSpeed runs retaining a stale config identity."""
    profile = load_deepspeed_runtime_profile(None, world_size=1)

    assert profile == DeepSpeedRuntimeProfile(
        enabled=False,
        zero_stage=None,
        config_path=None,
        config_digest=None,
        world_size=1,
    )


@pytest.mark.parametrize("stage", [True, 1.0, "1", 0, 4])
def test_profile_rejects_non_integer_or_unsupported_zero_stage(tmp_path, stage):
    """Catches JSON coercion accepting an invalid ZeRO stage."""
    path = tmp_path / "bad-stage.json"
    path.write_text(
        json.dumps(
            {
                "bf16": {"enabled": True},
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": "auto",
                "train_batch_size": "auto",
                "zero_optimization": {"stage": stage},
            }
        )
    )

    with pytest.raises(ValueError, match="zero_optimization.stage"):
        load_deepspeed_runtime_profile(path, world_size=1)


@pytest.mark.parametrize("dimension", [True, 1.0, "1", 0, -1, 1.5])
def test_profile_rejects_non_positive_integer_mesh_dimensions(tmp_path, dimension):
    """Catches mesh dimensions being coerced, truncated, or non-positive."""
    path = tmp_path / "bad-mesh.json"
    path.write_text(
        json.dumps(
            {
                "bf16": {"enabled": True},
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": "auto",
                "train_batch_size": "auto",
                "zero_optimization": {"stage": 2},
                "data_parallel_size": dimension,
            }
        )
    )

    with pytest.raises(ValueError, match="data_parallel_size"):
        load_deepspeed_runtime_profile(path, world_size=1)
