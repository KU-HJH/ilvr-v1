from pathlib import Path

import pytest

from evidence_window.stage2_config import (
    Stage2EvidenceWindowConfig,
    read_deepspeed_zero_stage,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPOSITORY_ROOT / "configs" / "evidence_window" / "stage2"


@pytest.mark.parametrize(
    "name,mode,outputs,active",
    [
        ("baseline.json", "none", frozenset(), frozenset()),
        (
            "ew_reference_free.json",
            "none",
            frozenset({"token_logps"}),
            frozenset({"ew"}),
        ),
        (
            "ew_precomputed_dpo.json",
            "precomputed",
            frozenset({"token_logps"}),
            frozenset({"ew"}),
        ),
        (
            "relation_kd.json",
            "precomputed",
            frozenset({"token_logps"}),
            frozenset({"relation"}),
        ),
        (
            "hidden_residual_kd.json",
            "precomputed",
            frozenset({"token_logps", "hidden_states"}),
            frozenset({"residual"}),
        ),
        (
            "online_frozen_zero1.json",
            "online_frozen",
            frozenset({"token_logps"}),
            frozenset({"ew"}),
        ),
        (
            "online_frozen_zero2.json",
            "online_frozen",
            frozenset({"token_logps"}),
            frozenset({"ew"}),
        ),
    ],
)
def test_stage2_variant_enables_only_its_named_objective(name, mode, outputs, active):
    """A copied experiment config must not silently add another objective."""
    config = Stage2EvidenceWindowConfig.from_json(CONFIG_ROOT / name)

    enabled = frozenset(
        objective
        for objective in ("ew", "relation", "residual")
        if config.effective_weight(objective, 1.0) > 0.0
    )
    assert config.reference.mode == mode
    assert config.required_outputs == outputs
    assert enabled == active
    assert config.loo.intervention == "window_attention_block"
    assert config.preference.contribution_reduction == "mean"


@pytest.mark.parametrize(
    "name,deepspeed_name,expected_stage",
    [
        ("online_frozen_zero1.json", "config_stage1.json", 1),
        ("online_frozen_zero2.json", "config_stage2.json", 2),
    ],
)
def test_online_variants_validate_only_with_the_documented_zero_stage(
    name, deepspeed_name, expected_stage
):
    """Confusing ILVR stages with DeepSpeed filenames can duplicate a 7B model unsafely."""
    config = Stage2EvidenceWindowConfig.from_json(CONFIG_ROOT / name)
    zero_stage = read_deepspeed_zero_stage(
        REPOSITORY_ROOT / "configs" / deepspeed_name
    )

    assert zero_stage == expected_stage
    config.validate_runtime(zero_stage)
    with pytest.raises(ValueError, match="ZeRO-3"):
        config.validate_runtime(3)
