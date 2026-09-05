from copy import deepcopy

import pytest
import torch
from torch import nn
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLSdpaAttention,
    Qwen2_5_VLVisionFlashAttention2,
)


def _tiny_qwen_config() -> Qwen2_5_VLConfig:
    return Qwen2_5_VLConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vision_config={
            "depth": 2,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 2,
            "out_hidden_size": 8,
            "fullatt_block_indexes": [0, 1],
        },
        image_token_id=80,
        vision_start_token_id=81,
        video_token_id=82,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 0]},
    )


def _load_tiny_mixed_model():
    from stage2.attention_backend import load_qwen2_5_vl_with_attention

    config = _tiny_qwen_config()
    config_calls = []
    model_calls = []

    def load_config(source, **kwargs):
        config_calls.append((source, kwargs))
        return config

    def load_model(source, **kwargs):
        model_calls.append((source, kwargs.copy()))
        prepared = kwargs.pop("config")
        assert kwargs == {
            "device_map": "cuda:0",
            "local_files_only": True,
            "torch_dtype": torch.bfloat16,
        }
        return Qwen2_5_VLForConditionalGeneration(prepared)

    model = load_qwen2_5_vl_with_attention(
        "tiny/qwen",
        attention_implementation="sdpa",
        vision_attention_implementation="flash_attention_2",
        config_load_kwargs={"revision": "tested"},
        model_load_kwargs={
            "device_map": "cuda:0",
            "local_files_only": True,
            "torch_dtype": torch.bfloat16,
        },
        load_config=load_config,
        load_model=load_model,
    )
    return model, config, config_calls, model_calls


def test_mixed_loader_constructs_real_flash_vision_and_sdpa_language_attention():
    """Catches Transformers flattening the requested mixed pair onto one backend."""
    model, config, config_calls, model_calls = _load_tiny_mixed_model()

    assert config_calls == [("tiny/qwen", {"revision": "tested"})]
    assert model_calls[0][0] == "tiny/qwen"
    assert model_calls[0][1]["config"] is config
    assert "attn_implementation" not in model_calls[0][1]
    assert all(
        type(block.attn) is Qwen2_5_VLVisionFlashAttention2
        for block in model.visual.blocks
    )
    assert all(
        type(layer.self_attn) is Qwen2_5_VLSdpaAttention
        for layer in model.model.layers
    )


def test_mixed_loader_binds_default_transformers_loaders_at_call_time(monkeypatch):
    """Catches import-time loader binding defeating production/test replacement seams."""
    import stage2.attention_backend as backend

    config = _tiny_qwen_config()
    calls = []
    monkeypatch.setattr(
        backend.AutoConfig,
        "from_pretrained",
        lambda source, **kwargs: calls.append(("config", source, kwargs)) or config,
    )

    def load_model(source, **kwargs):
        calls.append(("model", source, kwargs.copy()))
        return Qwen2_5_VLForConditionalGeneration(kwargs.pop("config"))

    monkeypatch.setattr(
        backend.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        load_model,
    )

    backend.load_qwen2_5_vl_with_attention(
        "tiny/qwen",
        attention_implementation="sdpa",
        vision_attention_implementation="flash_attention_2",
    )

    assert calls[0] == ("config", "tiny/qwen", {})
    assert calls[1][0:2] == ("model", "tiny/qwen")
    assert calls[1][2]["config"] is config


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda model: setattr(model.visual.blocks[0], "attn", nn.Identity()),
            "vision attention backend mismatch",
        ),
        (
            lambda model: setattr(model.model.layers[0], "self_attn", nn.Identity()),
            "language attention backend mismatch",
        ),
        (
            lambda model: setattr(model.visual, "blocks", nn.ModuleList()),
            "nonempty vision attention blocks",
        ),
        (
            lambda model: setattr(model.model, "layers", nn.ModuleList()),
            "nonempty language attention layers",
        ),
    ],
    ids=("wrong-vision", "wrong-language", "missing-blocks", "missing-layers"),
)
def test_mixed_validator_rejects_runtime_attention_class_mismatch(mutate, message):
    """Catches authentication based on config/class-name strings instead of modules."""
    from stage2.attention_backend import validate_qwen2_5_vl_attention_backends

    model, _config, _config_calls, _model_calls = _load_tiny_mixed_model()
    mutate(model)

    with pytest.raises(ValueError, match=message):
        validate_qwen2_5_vl_attention_backends(
            model,
            attention_implementation="sdpa",
            vision_attention_implementation="flash_attention_2",
        )


@pytest.mark.parametrize(
    ("language", "vision"),
    [("eager", "flash_attention_2"), ("sdpa", "eager")],
)
def test_mixed_loader_rejects_unsupported_distinct_backend_pairs(language, vision):
    """Catches silently constructing an unauthenticated distinct backend pair."""
    from stage2.attention_backend import load_qwen2_5_vl_with_attention

    with pytest.raises(ValueError, match="supported mixed attention contract"):
        load_qwen2_5_vl_with_attention(
            "tiny/qwen",
            attention_implementation=language,
            vision_attention_implementation=vision,
            load_config=lambda _source: deepcopy(_tiny_qwen_config()),
            load_model=lambda *_args, **_kwargs: pytest.fail("model load must not start"),
        )


def test_attention_counters_increment_at_real_kernel_dispatch(monkeypatch):
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as module
    model, *_ = _load_tiny_mixed_model()
    language=model.model.layers[0].self_attn
    vision=model.visual.blocks[0].attn
    observed=[]
    def sdpa(q,k,v,**kw):
        observed.append(("sdpa", getattr(language, "_sdpa_forward_count", 0)))
        return q
    def flex(owner,q,k,v,mask,**kw):
        observed.append(("flex", getattr(language, "_flex_attention_forward_count", 0)))
        return q.transpose(1,2), None
    def flash(q,k,v,*a):
        observed.append(("vision", getattr(vision, "_vision_flash_attention_forward_count", 0)))
        return q
    monkeypatch.setattr(torch.nn.functional,"scaled_dot_product_attention",sdpa)
    monkeypatch.setattr(module,"flex_attention_forward",flex)
    monkeypatch.setattr(module,"flash_attn_varlen_func",flash)
    embeddings=(torch.ones(3,1,2,4),torch.zeros(3,1,2,4))
    language(torch.zeros(1,2,8),position_embeddings=embeddings)
    language.forward_flex(torch.zeros(1,2,8),block_mask=object(),position_embeddings=embeddings)
    vision(torch.zeros(2,8),torch.tensor([0,2]),position_embeddings=(torch.ones(2,4),torch.zeros(2,4)))
    assert observed == [("sdpa",1),("flex",1),("vision",1)]
