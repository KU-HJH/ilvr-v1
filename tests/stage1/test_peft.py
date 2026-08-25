from __future__ import annotations

import pytest
import torch
from peft import PeftModel
from torch import nn

from peft_common import resolve_qwen2_5_vl_model
from stage1 import (
    Stage1ParameterReport,
    Stage1TuningConfig,
    build_stage1_tuned_model,
    validate_stage1_trainable_boundary,
)
from stage1.token_rows import SparseOutputTokenRows
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


LATENT_TOKEN_IDS = (125, 126, 127)


@pytest.fixture
def latent_token_ids() -> tuple[int, int, int]:
    return LATENT_TOKEN_IDS


@pytest.fixture
def tiny_qwen() -> Qwen2_5_VLForConditionalGeneration:
    """Return the real local Qwen fork at a CPU-testable scale."""
    config = Qwen2_5_VLConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 2,
            "out_hidden_size": 8,
            "fullatt_block_indexes": [0],
        },
        image_token_id=80,
        vision_start_token_id=81,
        video_token_id=82,
        stage="stage1",
        latent_size=2,
        latent_token_id=126,
        attn_implementation="eager",
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 0]},
    )
    return Qwen2_5_VLForConditionalGeneration(config)


def tuning(method: str, *, targets=None) -> Stage1TuningConfig:
    return Stage1TuningConfig(
        method=method,
        rank=4,
        alpha=8,
        dropout=0.0,
        target_suffixes=targets
        or ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    )


def test_dora_targets_decoder_and_exact_latent_rows(tiny_qwen, latent_token_ids):
    """Catches DoRA touching non-decoder modules or omitting either sparse row side."""
    configured = tuning("dora")

    model = build_stage1_tuned_model(tiny_qwen, configured, latent_token_ids)
    report = validate_stage1_trainable_boundary(
        model, configured, latent_token_ids
    )
    peft_config = model.peft_config["default"]

    assert isinstance(model, PeftModel)
    assert isinstance(report, Stage1ParameterReport)
    assert peft_config.use_dora is True
    assert peft_config.r == 4
    assert peft_config.lora_alpha == 8
    assert peft_config.lora_dropout == 0.0
    assert peft_config.bias == "none"
    assert peft_config.task_type == "CAUSAL_LM"
    assert peft_config.trainable_token_indices == {
        "model.embed_tokens": list(latent_token_ids)
    }
    assert set(peft_config.target_modules) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    }
    assert report.categories["visual"] == 0
    assert report.categories["base_decoder"] == 0
    assert report.categories["ordinary_embedding_rows"] == 0
    assert report.categories["ordinary_lm_head_rows"] == 0
    assert report.categories["latent_input_rows"] == 3
    assert report.categories["latent_output_rows"] == 3
    assert report.categories["latent_input_parameters"] == 24
    assert report.categories["latent_output_parameters"] == 24
    assert any("lora_magnitude_vector" in name for name in report.trainable_names)
    assert all("visual" not in name.split(".") for name in report.trainable_names)


def test_full_freezes_visual_but_trains_all_language(tiny_qwen, latent_token_ids):
    """Catches full tuning freezing language or installing unnecessary row wrappers."""
    tiny_qwen.model.norm.weight.requires_grad_(False)

    model = build_stage1_tuned_model(
        tiny_qwen, tuning("full"), latent_token_ids
    )
    qwen = resolve_qwen2_5_vl_model(model)
    report = validate_stage1_trainable_boundary(
        model, tuning("full"), latent_token_ids
    )

    assert model is tiny_qwen
    assert not any(parameter.requires_grad for parameter in qwen.visual.parameters())
    assert all(parameter.requires_grad for parameter in qwen.model.parameters())
    assert all(parameter.requires_grad for parameter in qwen.lm_head.parameters())
    assert not hasattr(qwen.model.embed_tokens, "token_adapter")
    assert not isinstance(qwen.lm_head, SparseOutputTokenRows)
    assert report.categories["visual"] == 0
    assert report.categories["base_decoder"] > 0
    assert report.categories["ordinary_embedding_rows"] == 125
    assert report.categories["ordinary_lm_head_rows"] == 125
    assert report.categories["latent_input_rows"] == 3
    assert report.categories["latent_output_rows"] == 3


def test_legacy_lora_remains_lora_and_keeps_exact_latent_rows(
    tiny_qwen, latent_token_ids
):
    """Catches legacy LoRA being silently promoted to DoRA or losing token rows."""
    configured = tuning("lora")

    model = build_stage1_tuned_model(tiny_qwen, configured, latent_token_ids)
    report = validate_stage1_trainable_boundary(
        model, configured, latent_token_ids
    )

    assert model.peft_config["default"].use_dora is False
    assert any("lora_A" in name.split(".") for name, _ in model.named_parameters())
    assert not any(
        "lora_magnitude_vector" in name.split(".")
        for name, _ in model.named_parameters()
    )
    assert report.categories["latent_input_rows"] == 3
    assert report.categories["latent_output_rows"] == 3


def test_latent_ce_backward_reaches_only_selective_input_rows(
    tiny_qwen, latent_token_ids
):
    """Catches detached selective embeddings or gradients leaking to ordinary rows."""
    model = build_stage1_tuned_model(
        tiny_qwen, tuning("dora"), latent_token_ids
    )
    qwen = resolve_qwen2_5_vl_model(model)
    token_adapter = qwen.model.embed_tokens.token_adapter
    selective = token_adapter.trainable_tokens_delta["default"]
    base_embedding = token_adapter.base_layer.weight
    ordinary_before = base_embedding[7].detach().clone()
    with torch.no_grad():
        selective.add_(0.05)
    input_ids = torch.tensor([[1, 125, 4, 126, 5, 127, 6]])

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
    )
    outputs.loss.backward()

    assert selective.grad is not None
    assert torch.isfinite(selective.grad).all()
    assert selective.grad.abs().sum() > 0
    assert token_adapter.token_indices["default"] == list(latent_token_ids)
    assert base_embedding.requires_grad is False
    assert base_embedding.grad is None
    torch.testing.assert_close(base_embedding[7], ordinary_before)


def test_parameter_report_uses_zero3_logical_sparse_row_shapes(
    tiny_qwen, latent_token_ids
):
    """Catches empty local ZeRO shards being mistaken for missing sparse rows."""
    configured = tuning("dora")
    model = build_stage1_tuned_model(tiny_qwen, configured, latent_token_ids)
    qwen = resolve_qwen2_5_vl_model(model)
    input_delta = qwen.model.embed_tokens.token_adapter.trainable_tokens_delta[
        "default"
    ]
    output_delta = qwen.lm_head.output_delta
    for index, parameter in enumerate((input_delta, output_delta)):
        parameter.ds_id = index
        parameter.ds_shape = torch.Size((3, 8))
        parameter.data = torch.empty(0)

    report = validate_stage1_trainable_boundary(
        model, configured, latent_token_ids
    )

    assert report.categories["latent_input_rows"] == 3
    assert report.categories["latent_output_rows"] == 3
    assert report.categories["latent_input_parameters"] == 24
    assert report.categories["latent_output_parameters"] == 24


@pytest.mark.parametrize(
    ("leak", "message"),
    [
        ("visual", "visual parameter is trainable"),
        ("decoder", "base decoder parameter is trainable"),
        ("embedding", "ordinary embedding parameter is trainable"),
        ("lm_head", "ordinary LM-head parameter is trainable"),
    ],
)
def test_adapter_validation_fails_closed_on_parameter_leakage(
    tiny_qwen, latent_token_ids, leak, message
):
    """Catches any PEFT trainability outside exact adapter and latent-row owners."""
    configured = tuning("dora")
    model = build_stage1_tuned_model(tiny_qwen, configured, latent_token_ids)
    qwen = resolve_qwen2_5_vl_model(model)
    if leak == "visual":
        parameter = next(qwen.visual.parameters())
    elif leak == "decoder":
        parameter = qwen.model.layers[0].self_attn.q_proj.base_layer.weight
    elif leak == "embedding":
        parameter = qwen.model.embed_tokens.token_adapter.base_layer.weight
    else:
        parameter = qwen.lm_head.base_head.weight
    parameter.requires_grad_(True)

    with pytest.raises(ValueError, match=message):
        validate_stage1_trainable_boundary(model, configured, latent_token_ids)


def test_adapter_validation_rejects_wrong_selective_token_identity(
    tiny_qwen, latent_token_ids
):
    """Catches three arbitrary input rows being mistaken for the latent rows."""
    configured = tuning("dora")
    model = build_stage1_tuned_model(tiny_qwen, configured, latent_token_ids)
    qwen = resolve_qwen2_5_vl_model(model)
    qwen.model.embed_tokens.token_adapter.token_indices["default"] = [124, 125, 126]

    with pytest.raises(ValueError, match="selective input token IDs"):
        validate_stage1_trainable_boundary(model, configured, latent_token_ids)


def test_adapter_validation_rejects_wrong_selective_row_count(
    tiny_qwen, latent_token_ids
):
    """Catches selective input state containing fewer than three logical rows."""
    configured = tuning("dora")
    model = build_stage1_tuned_model(tiny_qwen, configured, latent_token_ids)
    qwen = resolve_qwen2_5_vl_model(model)
    token_adapter = qwen.model.embed_tokens.token_adapter
    token_adapter.trainable_tokens_delta["default"] = nn.Parameter(
        token_adapter.trainable_tokens_delta["default"][:2].detach().clone()
    )
    token_adapter.token_indices["default"] = list(latent_token_ids[:2])

    with pytest.raises(ValueError, match="exactly three selective input rows"):
        validate_stage1_trainable_boundary(model, configured, latent_token_ids)


def test_dora_validation_rejects_missing_magnitude_vector(
    tiny_qwen, latent_token_ids
):
    """Catches a direction-only LoRA target being mislabeled as DoRA."""
    configured = tuning("dora")
    model = build_stage1_tuned_model(tiny_qwen, configured, latent_token_ids)
    target = resolve_qwen2_5_vl_model(model).model.layers[0].self_attn.q_proj
    del target.lora_magnitude_vector["default"]

    with pytest.raises(ValueError, match="missing DoRA magnitude vector"):
        validate_stage1_trainable_boundary(model, configured, latent_token_ids)


@pytest.mark.parametrize("visual_match", [False, True])
def test_build_rejects_unmatched_or_visual_only_decoder_target(
    tiny_qwen, latent_token_ids, visual_match
):
    """Catches target suffixes being resolved outside the language decoder."""
    del tiny_qwen.model.layers[0].self_attn.q_proj
    if visual_match:
        tiny_qwen.visual.q_proj = nn.Linear(8, 8)
    configured = tuning("dora", targets=("q_proj",))
    message = "decoder-owned" if visual_match else "did not match"

    with pytest.raises(ValueError, match=message):
        build_stage1_tuned_model(tiny_qwen, configured, latent_token_ids)


def test_build_rejects_non_linear_decoder_target(tiny_qwen, latent_token_ids):
    """Catches a matching decoder path whose component cannot host LoRA."""
    tiny_qwen.model.layers[0].self_attn.q_proj = nn.Identity()

    with pytest.raises(ValueError, match="linear decoder modules"):
        build_stage1_tuned_model(
            tiny_qwen,
            tuning("dora", targets=("q_proj",)),
            latent_token_ids,
        )


@pytest.mark.parametrize("bad_ids", [(125, 126), (125, 125, 127), (125, 126, 128)])
def test_build_rejects_invalid_latent_token_ids(tiny_qwen, bad_ids):
    """Catches malformed selective rows before adapter training can begin."""
    with pytest.raises(ValueError, match="latent token IDs"):
        build_stage1_tuned_model(tiny_qwen, tuning("dora"), bad_ids)
