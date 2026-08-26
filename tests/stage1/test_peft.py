from __future__ import annotations

from copy import deepcopy

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
    torch.manual_seed(17)
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


def assert_finite_nonzero_tensor(value: torch.Tensor) -> None:
    assert torch.isfinite(value).all()
    assert value.abs().sum() > 0


def assert_finite_nonzero_gradient(parameter: nn.Parameter) -> None:
    assert parameter.grad is not None
    assert_finite_nonzero_tensor(parameter.grad)


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


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_multilayer_qwen_validates_exact_injected_targets_after_peft_optimization(
    tiny_qwen, latent_token_ids, method
):
    config = deepcopy(tiny_qwen.config)
    config.num_hidden_layers = 4
    config.tie_word_embeddings = True
    model = Qwen2_5_VLForConditionalGeneration(config)
    model.tie_weights()
    configured = tuning(method)

    tuned = build_stage1_tuned_model(model, configured, latent_token_ids)
    qwen = resolve_qwen2_5_vl_model(tuned)
    expected = {
        f"model.layers.{layer}.{owner}.{suffix}"
        for layer in range(4)
        for owner, suffixes in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for suffix in suffixes
    }

    assert set(tuned.base_model.targeted_module_names) == expected
    assert set(tuned.peft_config["default"].target_modules) != expected
    report = validate_stage1_trainable_boundary(
        tuned, configured, latent_token_ids
    )
    assert report.categories["decoder_adapters"] > 0
    assert qwen.config.tie_word_embeddings is False


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_adapter_boundary_rejects_inexact_injected_target_witness(
    tiny_qwen, latent_token_ids, mutation
):
    configured = tuning("dora")
    model = build_stage1_tuned_model(
        tiny_qwen, configured, latent_token_ids
    )
    targeted = model.base_model.targeted_module_names
    if mutation == "missing":
        targeted.pop()
    else:
        targeted.append("visual.blocks.0.attn.qkv")

    with pytest.raises(ValueError, match="adapter targets do not match"):
        validate_stage1_trainable_boundary(
            model, configured, latent_token_ids
        )


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


def test_full_parameter_report_counts_tied_embeddings_once(
    tiny_qwen, latent_token_ids
):
    tiny_qwen.config.tie_word_embeddings = True
    tiny_qwen.tie_weights()

    model = build_stage1_tuned_model(
        tiny_qwen, tuning("full"), latent_token_ids
    )
    qwen = resolve_qwen2_5_vl_model(model)
    report = validate_stage1_trainable_boundary(
        model, tuning("full"), latent_token_ids
    )
    expected_trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert qwen.lm_head.weight is qwen.model.embed_tokens.weight
    assert report.trainable_parameters == expected_trainable
    assert len(report.trainable_names) == len(set(report.trainable_names))


def test_full_ce_backward_reaches_language_but_not_visual(
    tiny_qwen, latent_token_ids
):
    """Catches full-mode language gradients being detached or leaking to vision."""
    model = build_stage1_tuned_model(
        tiny_qwen, tuning("full"), latent_token_ids
    )
    qwen = resolve_qwen2_5_vl_model(model)
    input_ids = torch.tensor([[1, 125, 4, 126, 5, 127, 6]])

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
    )
    outputs.loss.backward()

    expected_language_parameters = (
        qwen.model.embed_tokens.weight,
        qwen.model.layers[0].self_attn.q_proj.weight,
        qwen.model.layers[0].mlp.down_proj.weight,
        qwen.model.norm.weight,
        qwen.lm_head.weight,
    )
    for parameter in expected_language_parameters:
        assert_finite_nonzero_gradient(parameter)
    assert all(parameter.grad is None for parameter in qwen.visual.parameters())


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


def test_dora_ce_backward_reaches_every_adapter_and_latent_row_only(
    tiny_qwen, latent_token_ids
):
    """Catches missing DoRA/token-row gradients or leakage into frozen Qwen state."""
    model = build_stage1_tuned_model(
        tiny_qwen, tuning("dora"), latent_token_ids
    )
    qwen = resolve_qwen2_5_vl_model(model)
    token_adapter = qwen.model.embed_tokens.token_adapter
    input_delta = token_adapter.trainable_tokens_delta["default"]
    output_delta = qwen.lm_head.output_delta
    base_embedding = token_adapter.base_layer.weight
    with torch.no_grad():
        input_delta.add_(0.05)
        for name, parameter in model.named_parameters():
            if "lora_B" in name.split("."):
                parameter.fill_(0.05)
    input_ids = torch.tensor([[1, 125, 4, 126, 5, 127, 6]])

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
    )
    outputs.loss.backward()

    adapter_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if {"lora_A", "lora_B", "lora_magnitude_vector"}
        & set(name.split("."))
    }
    for component in ("lora_A", "lora_B", "lora_magnitude_vector"):
        selected = [
            parameter
            for name, parameter in adapter_parameters.items()
            if component in name.split(".")
        ]
        assert selected
        for parameter in selected:
            assert_finite_nonzero_gradient(parameter)

    assert token_adapter.token_indices["default"] == list(latent_token_ids)
    assert tuple(qwen.lm_head.token_ids.tolist()) == latent_token_ids
    assert input_delta.grad is not None
    assert output_delta.grad is not None
    for row in range(3):
        assert_finite_nonzero_tensor(input_delta.grad[row])
        assert_finite_nonzero_tensor(output_delta.grad[row])

    adapter_ids = {id(parameter) for parameter in adapter_parameters.values()}
    frozen_decoder = [
        parameter
        for parameter in qwen.model.parameters()
        if id(parameter) not in adapter_ids | {id(input_delta)}
    ]
    assert frozen_decoder
    assert all(parameter.grad is None for parameter in frozen_decoder)
    assert base_embedding.requires_grad is False
    assert base_embedding.grad is None
    assert qwen.lm_head.base_head.weight.requires_grad is False
    assert qwen.lm_head.base_head.weight.grad is None
    assert all(parameter.grad is None for parameter in qwen.visual.parameters())


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


def test_build_rejects_suffix_match_outside_canonical_decoder_layer(
    tiny_qwen, latent_token_ids
):
    """Catches unrelated qwen.model descendants being accepted by suffix alone."""
    tiny_qwen.model.norm.q_proj = nn.Linear(8, 8)

    with pytest.raises(ValueError, match="canonical decoder-layer projection"):
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


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_adapter_build_unties_declared_qwen_head_for_sparse_output_rows(
    tiny_qwen, latent_token_ids, method
):
    tiny_qwen.config.tie_word_embeddings = True
    tiny_qwen.tie_weights()
    tied_weight = tiny_qwen.model.embed_tokens.weight
    assert tiny_qwen.lm_head.weight is tiny_qwen.model.embed_tokens.weight

    model = build_stage1_tuned_model(
        tiny_qwen, tuning(method), latent_token_ids
    )
    qwen = resolve_qwen2_5_vl_model(model)
    rows = qwen.lm_head

    assert qwen.config.tie_word_embeddings is False
    assert isinstance(rows, SparseOutputTokenRows)
    assert rows.base_head.weight is not tied_weight
    assert rows.base_head.weight is not qwen.model.embed_tokens.token_adapter.base_layer.weight
    assert rows.base_head.weight.requires_grad is False
    assert qwen.model.embed_tokens.token_adapter.base_layer.weight.requires_grad is False
    assert rows.output_delta.requires_grad is True
