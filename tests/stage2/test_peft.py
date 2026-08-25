from types import SimpleNamespace

import pytest
from torch import nn

from helpers import tiny_stage2_qwen
from stage2.peft import (
    ALLOWED_STAGE2_TARGET_SUFFIXES,
    Stage2TuningConfig,
    build_stage2_tuned_model,
    normalize_stage2_target_modules,
    resolve_stage2_qwen_model,
    validate_stage2_parameter_boundary,
)


def settings(method: str) -> Stage2TuningConfig:
    return Stage2TuningConfig(
        method=method,
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
    )


def lora_settings() -> Stage2TuningConfig:
    return settings("lora")


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


@pytest.mark.parametrize("method,use_dora", [("lora", False), ("dora", True)])
def test_stage2_peft_targets_only_decoder_modules(method, use_dora):
    """Catches PEFT matching visual-tower projections by short suffix."""
    base = tiny_stage2_qwen()
    tuning = settings(method)

    wrapped = build_stage2_tuned_model(base, tuning)
    qwen = resolve_stage2_qwen_model(wrapped)
    config = wrapped.peft_config["default"]

    assert qwen is base
    assert config.use_dora is use_dora
    assert config.modules_to_save is None
    assert all(name.startswith("model.layers.") for name in config.target_modules)
    assert not any("visual" in name for name in config.target_modules)


def test_wrapper_resolver_handles_deepspeed_then_peft():
    """Catches wrapper traversal that stops before the underlying Stage 2 Qwen."""
    wrapped = build_stage2_tuned_model(tiny_stage2_qwen(), lora_settings())
    engine = SimpleNamespace(module=wrapped)

    assert resolve_stage2_qwen_model(engine) is wrapped.get_base_model()


def test_wrapper_resolver_rejects_non_stage2_base():
    """Catches silent acceptance of a model without the Stage 2 Qwen contract."""
    with pytest.raises(TypeError, match="Stage 2 Qwen interface"):
        resolve_stage2_qwen_model(nn.Linear(2, 2))


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_only_adapter_parameters_are_trainable(method):
    """Catches training that leaks outside decoder adapter parameters."""
    wrapped = build_stage2_tuned_model(tiny_stage2_qwen(), settings(method))
    report = validate_stage2_parameter_boundary(wrapped, settings(method))
    trainable = [name for name, parameter in wrapped.named_parameters() if parameter.requires_grad]

    assert report.trainable_parameters > 0
    assert all("lora_" in name for name in trainable)
    assert not any("visual" in name for name in trainable)
    assert not any("embed_tokens" in name or "lm_head" in name for name in trainable)


def test_parameter_boundary_fails_if_visual_is_reenabled():
    """Catches a frozen vision tower being accidentally made trainable."""
    wrapped = build_stage2_tuned_model(tiny_stage2_qwen(), lora_settings())
    next(resolve_stage2_qwen_model(wrapped).visual.parameters()).requires_grad_(True)

    with pytest.raises(ValueError, match="visual parameter is trainable"):
        validate_stage2_parameter_boundary(wrapped, lora_settings())


def test_inference_boundary_requires_frozen_adapter_parameters():
    """Catches inference validation accepting a trainable adapter or no topology."""
    loaded = build_stage2_tuned_model(tiny_stage2_qwen(), lora_settings())
    with pytest.raises(ValueError, match="inference model has trainable parameters"):
        validate_stage2_parameter_boundary(
            loaded, lora_settings(), expect_trainable=False
        )

    loaded.requires_grad_(False)
    report = validate_stage2_parameter_boundary(
        loaded, lora_settings(), expect_trainable=False
    )

    assert report.trainable_parameters == 0
    assert report.resolved_target_names
    assert all(name.startswith("model.layers.") for name in report.resolved_target_names)


def test_full_tuning_keeps_the_vision_tower_frozen_but_trains_language():
    """Catches full tuning accidentally freezing all language parameters."""
    model = build_stage2_tuned_model(tiny_stage2_qwen(), settings("full"))

    report = validate_stage2_parameter_boundary(model, settings("full"))

    assert report.trainable_parameters > 0
    assert any("embed_tokens" in name for name in report.trainable_names)
    assert not any("visual" in name for name in report.trainable_names)
