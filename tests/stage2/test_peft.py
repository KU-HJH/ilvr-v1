import pytest

from stage2.peft import Stage2TuningConfig, normalize_stage2_target_modules


def test_stage2_tuning_config_normalizes_the_approved_decoder_targets():
    """Catches decoder PEFT target validation accepting the wrong modules."""
    targets = normalize_stage2_target_modules(
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    config = Stage2TuningConfig(
        method="dora", r=8, alpha=16, dropout=0.1, target_modules=targets
    )

    assert config.method == "dora"
    assert config.use_dora is True
    assert config.target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@pytest.mark.parametrize("value", ["", "q_proj,visual", "q_proj,q_proj"])
def test_stage2_tuning_config_rejects_invalid_target_lists(value):
    """Catches empty, unsupported, or duplicate decoder target selections."""
    with pytest.raises(ValueError):
        normalize_stage2_target_modules(value)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"method": "adapter"}, "unsupported Stage 2 tuning method"),
        ({"r": 0}, "LoRA rank and alpha must be positive"),
        ({"alpha": 0}, "LoRA rank and alpha must be positive"),
        ({"dropout": 1.0}, "LoRA dropout must be in \\[0, 1\\)"),
    ],
)
def test_stage2_tuning_config_rejects_invalid_hyperparameters(overrides, message):
    """Catches invalid Stage 2 method and LoRA hyperparameter values."""
    config = {
        "method": "lora",
        "r": 8,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": ("q_proj",),
    }
    config.update(overrides)

    with pytest.raises(ValueError, match=message):
        Stage2TuningConfig(**config)
