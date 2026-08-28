import json

import pytest

from src.deepspeed_runtime import (
    DEEPSPEED_RUNTIME_FILENAME,
    DeepSpeedRuntimeProfile,
    canonical_deepspeed_config_path,
    load_deepspeed_runtime_profile,
    read_deepspeed_runtime_sidecar,
    validate_deepspeed_resume_identity,
    write_deepspeed_runtime_sidecar,
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


def test_runtime_sidecar_round_trip(tmp_path):
    """Catches checkpoint identity losing an exact runtime profile field."""
    profile = load_deepspeed_runtime_profile(
        canonical_deepspeed_config_path(2), world_size=2
    )

    path = write_deepspeed_runtime_sidecar(tmp_path, profile)

    assert path.name == "deepspeed-runtime.json"
    assert read_deepspeed_runtime_sidecar(tmp_path) == profile


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("enabled", False),
        ("zero_stage", 1),
        ("config_digest", "0" * 64),
        ("world_size", 4),
    ],
)
def test_resume_rejects_runtime_identity_mismatch(
    tmp_path, field, replacement
):
    """Catches a numbered checkpoint crossing an incompatible runtime."""
    expected = load_deepspeed_runtime_profile(
        canonical_deepspeed_config_path(2), world_size=2
    )
    payload = {"schema_version": 1, **expected.to_dict(), field: replacement}
    (tmp_path / DEEPSPEED_RUNTIME_FILENAME).write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=field):
        validate_deepspeed_resume_identity(tmp_path, expected)


def test_resume_identity_uses_digest_not_old_config_location(tmp_path):
    """Catches portable resumes requiring the producer's absolute config path."""
    expected = load_deepspeed_runtime_profile(
        canonical_deepspeed_config_path(2), world_size=2
    )
    payload = {
        "schema_version": 1,
        **expected.to_dict(),
        "config_path": "/deleted/producer/config.json",
    }
    (tmp_path / DEEPSPEED_RUNTIME_FILENAME).write_text(json.dumps(payload))

    validate_deepspeed_resume_identity(tmp_path, expected)


def test_resume_rejects_missing_runtime_identity(tmp_path):
    """Catches legacy or incomplete numbered checkpoints reaching restore."""
    expected = load_deepspeed_runtime_profile(
        canonical_deepspeed_config_path(2), world_size=2
    )

    with pytest.raises(ValueError, match="missing DeepSpeed runtime identity"):
        validate_deepspeed_resume_identity(tmp_path, expected)


def test_runtime_sidecar_refuses_to_replace_different_identity(tmp_path):
    """Catches concurrent or repeated writers silently changing run identity."""
    first = load_deepspeed_runtime_profile(
        canonical_deepspeed_config_path(1), world_size=2
    )
    second = load_deepspeed_runtime_profile(
        canonical_deepspeed_config_path(2), world_size=2
    )
    write_deepspeed_runtime_sidecar(tmp_path, first)

    with pytest.raises(FileExistsError, match="different runtime identity"):
        write_deepspeed_runtime_sidecar(tmp_path, second)

    assert read_deepspeed_runtime_sidecar(tmp_path) == first


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
