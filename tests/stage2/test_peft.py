from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from .helpers import activate_stage2_adapter_deltas, tiny_stage2_qwen
from src.peft_common import resolve_qwen_decoder_target_names
from stage2.data import build_stage2_labels
from stage2.peft import (
    ALLOWED_STAGE2_TARGET_SUFFIXES,
    Stage2TuningConfig,
    build_stage2_tuned_model,
    normalize_stage2_target_modules,
    resolve_stage2_qwen_model,
    validate_stage2_parameter_boundary,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


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


def one_block_stage2_batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[1, 91, 90, 90, 92, 2]])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": torch.tensor([[-100, 91, -100, -100, 92, 2]]),
    }


def original_image_stage2_batch(*, supervise_latent_boundaries: bool):
    input_ids = torch.tensor(
        [[1, 81, 80, 80, 80, 80, 2, 91, 90, 90, 92, 3]]
    )
    attention_mask = torch.ones_like(input_ids)
    labels = build_stage2_labels(
        input_ids=input_ids,
        attention_mask=attention_mask,
        assistant_content_start=7,
        token_ids=LatentTokenIds(90, 91, 92),
        latent_size=2,
        supervise_latent_boundaries=supervise_latent_boundaries,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": torch.ones((4, 12)),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }


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


def test_shared_target_resolution_preserves_stage2_contract():
    """Catches shared decoder resolution changing Stage 2 target paths or order."""
    requested = ("q_proj", "k_proj", "v_proj", "o_proj")

    assert resolve_qwen_decoder_target_names(tiny_stage2_qwen(), requested) == (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
    )


def test_shared_target_resolution_rejects_visual_projection():
    """Catches a visual-only suffix being accepted as a decoder PEFT target."""
    model = tiny_stage2_qwen()
    delattr(model.model.layers[0].self_attn, "q_proj")
    model.visual.q_proj = nn.Linear(8, 8)

    with pytest.raises(ValueError, match="decoder-owned"):
        resolve_qwen_decoder_target_names(model, ("q_proj",))


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


def test_stage2_peft_accepts_optimized_target_selectors_for_multilayer_decoder():
    """Catches treating PEFT's compact target selectors as resolved module names."""
    base = tiny_stage2_qwen()
    base.model.layers = nn.ModuleList(
        [deepcopy(base.model.layers[0]) for _ in range(3)]
    )
    base.config.num_hidden_layers = 3

    wrapped = build_stage2_tuned_model(base, settings("dora"))
    report = validate_stage2_parameter_boundary(wrapped, settings("dora"))

    assert len(report.resolved_target_names) == 21
    assert all(
        name.startswith("model.layers.") for name in report.resolved_target_names
    )


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


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_adapter_executes_in_every_recurrent_prefix_and_final_forward(method):
    """Catches Stage 2 recurrence bypassing decoder adapters during PEFT training."""
    wrapped = build_stage2_tuned_model(tiny_stage2_qwen(), settings(method))
    activate_stage2_adapter_deltas(wrapped)
    qwen = resolve_stage2_qwen_model(wrapped)
    calls = []
    handle = qwen.model.layers[0].self_attn.q_proj.register_forward_hook(
        lambda *unused: calls.append(1)
    )
    try:
        outputs = wrapped(**one_block_stage2_batch(), return_stage2_latent_states=True)
    finally:
        handle.remove()

    assert outputs.loss.isfinite()
    assert len(outputs.stage2_latent_states) == 2
    assert len(calls) == 3


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_adapter_latents_and_only_adapter_parameters_receive_ce_gradients(method):
    """Catches detached Stage 2 latent states or gradients leaking into frozen modules."""
    wrapped = build_stage2_tuned_model(tiny_stage2_qwen(), settings(method))
    activate_stage2_adapter_deltas(wrapped)
    qwen = resolve_stage2_qwen_model(wrapped)
    outputs = wrapped(**one_block_stage2_batch(), return_stage2_latent_states=True)
    for state in outputs.stage2_latent_states:
        state.retain_grad()

    outputs.loss.backward()

    assert all(
        state.grad is not None and torch.isfinite(state.grad).all()
        for state in outputs.stage2_latent_states
    )
    assert all(state.grad.abs().sum() > 0 for state in outputs.stage2_latent_states)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for name, parameter in wrapped.named_parameters()
        if "lora_" in name
    )
    assert all(
        parameter.grad is None
        for name, parameter in qwen.model.named_parameters()
        if "lora_" not in name
    )
    assert all(parameter.grad is None for parameter in qwen.model.embed_tokens.parameters())
    assert all(parameter.grad is None for parameter in qwen.lm_head.parameters())
    assert all(parameter.grad is None for parameter in qwen.visual.parameters())


@pytest.mark.parametrize("method", ["lora", "dora"])
@pytest.mark.parametrize(
    ("supervise_latent_boundaries", "expected_boundaries"),
    [(True, [91, 92]), (False, [-100, -100])],
)
def test_peft_original_image_forward_keeps_visual_frozen_and_pairs_boundary_labels(
    method, supervise_latent_boundaries, expected_boundaries
):
    """Catches PEFT paths that re-encode vision or ignore paired label policy."""
    wrapped = build_stage2_tuned_model(tiny_stage2_qwen(), settings(method))
    activate_stage2_adapter_deltas(wrapped)
    qwen = resolve_stage2_qwen_model(wrapped)
    batch = original_image_stage2_batch(
        supervise_latent_boundaries=supervise_latent_boundaries
    )
    visual_grad_modes = []
    handle = qwen.visual.register_forward_hook(
        lambda *unused: visual_grad_modes.append(torch.is_grad_enabled())
    )
    try:
        outputs = wrapped(**batch, return_stage2_latent_states=True)
        outputs.loss.backward()
    finally:
        handle.remove()

    assert batch["labels"][0, 8:10].tolist() == [-100, -100]
    assert [batch["labels"][0, 7].item(), batch["labels"][0, 10].item()] == expected_boundaries
    assert visual_grad_modes == [False]
    assert all(parameter.grad is None for parameter in qwen.visual.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for name, parameter in wrapped.named_parameters()
        if "lora_" in name
    )


def test_parameter_boundary_fails_if_visual_is_reenabled():
    """Catches a frozen vision tower being accidentally made trainable."""
    wrapped = build_stage2_tuned_model(tiny_stage2_qwen(), lora_settings())
    next(resolve_stage2_qwen_model(wrapped).visual.parameters()).requires_grad_(True)

    with pytest.raises(ValueError, match="visual parameter is trainable"):
        validate_stage2_parameter_boundary(wrapped, lora_settings())


def test_fresh_inference_adapter_is_constructed_with_frozen_parameters():
    """Catches fresh inference construction leaving LoRA parameters trainable."""
    loaded = build_stage2_tuned_model(
        tiny_stage2_qwen(), lora_settings(), is_trainable=False
    )
    report = validate_stage2_parameter_boundary(
        loaded, lora_settings(), expect_trainable=False
    )

    assert report.trainable_parameters == 0
    assert report.resolved_target_names
    assert all(name.startswith("model.layers.") for name in report.resolved_target_names)


def test_loaded_inference_adapter_is_frozen_and_matches_tuning(tmp_path):
    """Catches loading that bypasses inference freezing or target validation."""
    trained = build_stage2_tuned_model(tiny_stage2_qwen(), lora_settings())
    with pytest.warns(UserWarning, match="Could not find a config file"):
        trained.save_pretrained(tmp_path)

    loaded = build_stage2_tuned_model(
        tiny_stage2_qwen(),
        lora_settings(),
        adapter_path=tmp_path,
        is_trainable=False,
    )
    report = validate_stage2_parameter_boundary(
        loaded, lora_settings(), expect_trainable=False
    )

    assert report.trainable_parameters == 0
    assert report.resolved_target_names == tuple(
        loaded.peft_config["default"].target_modules
    )


def test_loaded_adapter_rejects_unapproved_decoder_linear_target(tmp_path):
    """Catches a loaded adapter targeting a non-approved decoder projection."""
    target_name = "model.layers.0.unapproved_proj"
    source = tiny_stage2_qwen()
    source.model.layers[0].add_module("unapproved_proj", nn.Linear(8, 8))
    adapter = get_peft_model(
        source,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=[target_name],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    with pytest.warns(UserWarning, match="Could not find a config file"):
        adapter.save_pretrained(tmp_path)

    base = tiny_stage2_qwen()
    base.model.layers[0].add_module("unapproved_proj", nn.Linear(8, 8))
    with pytest.raises(ValueError, match="unapproved Stage 2 decoder targets"):
        build_stage2_tuned_model(
            base,
            lora_settings(),
            adapter_path=tmp_path,
            is_trainable=False,
        )


def test_loaded_adapter_rejects_targets_that_do_not_match_tuning(tmp_path):
    """Catches a loaded adapter whose approved targets differ from the request."""
    source = tiny_stage2_qwen()
    adapter = get_peft_model(
        source,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["model.layers.0.self_attn.q_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    with pytest.warns(UserWarning, match="Could not find a config file"):
        adapter.save_pretrained(tmp_path)

    with pytest.raises(ValueError, match="do not match Stage 2 tuning"):
        build_stage2_tuned_model(
            tiny_stage2_qwen(),
            lora_settings(),
            adapter_path=tmp_path,
            is_trainable=False,
        )


def test_full_tuning_keeps_the_vision_tower_frozen_but_trains_language():
    """Catches full tuning accidentally freezing all language parameters."""
    model = build_stage2_tuned_model(tiny_stage2_qwen(), settings("full"))

    report = validate_stage2_parameter_boundary(model, settings("full"))

    assert report.trainable_parameters > 0
    assert any("embed_tokens" in name for name in report.trainable_names)
    assert not any("visual" in name for name in report.trainable_names)
