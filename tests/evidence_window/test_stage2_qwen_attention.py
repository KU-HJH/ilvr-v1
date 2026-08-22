from types import MethodType

import pytest
import torch

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
)


def _tiny_config(*, stage=None):
    config = Qwen2_5_VLConfig(
        vocab_size=32,
        hidden_size=24,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        attention_dropout=0.0,
        use_cache=False,
        pad_token_id=0,
        latent_token_id=9,
        vision_start_token_id=29,
        image_token_id=30,
        video_token_id=31,
        stage=stage,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 2, 2]},
        vision_config={
            "depth": 1,
            "hidden_size": 24,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": 24,
            "fullatt_block_indexes": [0],
        },
    )
    config._attn_implementation = "eager"
    return config


@pytest.fixture
def tiny_qwen():
    torch.manual_seed(101)
    return Qwen2_5_VLModel(_tiny_config()).eval()


@pytest.fixture
def text_batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "use_cache": False,
        "return_dict": True,
    }


@pytest.fixture
def block():
    mask = torch.zeros((1, 6, 6), dtype=torch.bool)
    mask[0, 4, 1:3] = True
    return mask


def test_block_leaves_prior_hidden_states_unchanged(tiny_qwen, text_batch, block):
    """Applying a block before its query row would rewrite retained history."""
    with torch.no_grad():
        full = tiny_qwen(**text_batch, attention_backend_override="sdpa").last_hidden_state
        loo = tiny_qwen(
            **text_batch,
            attention_backend_override="sdpa",
            attention_block_mask=block,
        ).last_hidden_state

    torch.testing.assert_close(full[:, :4], loo[:, :4], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(full[:, 4], loo[:, 4], atol=1e-7, rtol=1e-7)


def test_block_requires_explicit_sdpa_override(tiny_qwen, text_batch, block):
    """Sending an intervention through the configured backend would change its meaning."""
    with pytest.raises(ValueError, match="SDPA"):
        tiny_qwen(**text_batch, attention_block_mask=block)


def test_two_unblocked_sdpa_calls_match(tiny_qwen, text_batch):
    """Nondeterministic matched-full scores would contaminate the LOO contrast."""
    with torch.no_grad():
        first = tiny_qwen(**text_batch, attention_backend_override="sdpa").last_hidden_state
        second = tiny_qwen(**text_batch, attention_backend_override="sdpa").last_hidden_state

    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


def test_disabled_sliding_window_leaves_distant_causal_keys_allowed():
    """A configured-but-disabled window must not deprive distant evidence."""
    config = _tiny_config()
    config.use_sliding_window = False
    config.sliding_window = 2
    model = Qwen2_5_VLModel(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    inputs_embeds = model.embed_tokens(input_ids)

    mask = model._build_auxiliary_sdpa_mask(
        attention_mask=torch.ones_like(input_ids),
        attention_block_mask=None,
        input_tensor=inputs_embeds,
        cache_position=torch.arange(input_ids.shape[1]),
    )

    assert mask[0, 0, 5, 0].item() == 0.0


def test_enabled_sliding_window_rejects_auxiliary_sdpa(tiny_qwen, text_batch):
    """One shared auxiliary mask cannot preserve layer-selective sliding semantics."""
    tiny_qwen.config.use_sliding_window = True
    tiny_qwen.config.sliding_window = 2
    tiny_qwen.config.max_window_layers = 1

    with pytest.raises(ValueError, match="sliding-window"):
        tiny_qwen(**text_batch, attention_backend_override="sdpa")


def test_default_forward_does_not_enter_sdpa_override(tiny_qwen, text_batch):
    """A default CE forward entering the auxiliary path would break compatibility."""

    def fail_if_called(_self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("default forward entered auxiliary SDPA")

    for layer in tiny_qwen.layers:
        layer.self_attn.forward_sdpa = MethodType(fail_if_called, layer.self_attn)

    with torch.no_grad():
        output = tiny_qwen(**text_batch)

    assert output.last_hidden_state.shape == (1, 6, 24)


@pytest.mark.parametrize(
    ("override", "make_mask", "use_cache", "message"),
    [
        ("eager", lambda value: value, False, "attention_backend_override"),
        ("sdpa", lambda value: value.to(dtype=torch.float32), False, "boolean"),
        ("sdpa", lambda value: value[:, :-1], False, "shape"),
        ("sdpa", lambda value: value.to(device="meta"), False, "device"),
        ("sdpa", lambda value: value, True, "use_cache"),
    ],
)
def test_sdpa_override_rejects_invalid_inputs(
    tiny_qwen,
    text_batch,
    block,
    override,
    make_mask,
    use_cache,
    message,
):
    """Malformed intervention inputs must fail instead of changing LOO semantics."""
    batch = dict(text_batch)
    batch["use_cache"] = use_cache
    with pytest.raises(ValueError, match=message):
        tiny_qwen(
            **batch,
            attention_backend_override=override,
            attention_block_mask=make_mask(block),
        )


def test_stage2_prefix_calls_slice_padding_and_both_block_axes():
    """Passing full key columns to a prefix would misalign its pair intervention."""
    torch.manual_seed(202)
    model = Qwen2_5_VLForConditionalGeneration(_tiny_config(stage="stage2")).eval()
    input_ids = torch.tensor([[1, 2, 9, 3, 4, 9, 5]], dtype=torch.long)
    padding_mask = torch.ones_like(input_ids)
    block_mask = torch.zeros((1, 7, 7), dtype=torch.bool)
    block_mask[0, 6, 2] = True
    observed = []

    def record_model_inputs(_module, args, kwargs):
        del args
        observed.append(
            (
                kwargs["attention_mask"].detach().clone(),
                kwargs["attention_block_mask"].detach().clone(),
            )
        )

    handle = model.model.register_forward_pre_hook(record_model_inputs, with_kwargs=True)
    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=padding_mask,
                use_cache=False,
                return_dict=True,
                attention_backend_override="sdpa",
                attention_block_mask=block_mask,
            )
    finally:
        handle.remove()

    assert [tuple(mask.shape) for mask, _ in observed] == [(1, 2), (1, 5), (1, 7)]
    assert [tuple(mask.shape) for _, mask in observed] == [
        (1, 2, 2),
        (1, 5, 5),
        (1, 7, 7),
    ]
    for prefix_length, (seen_padding, seen_block) in zip((2, 5, 7), observed):
        assert torch.equal(seen_padding, padding_mask[:, :prefix_length])
        assert torch.equal(seen_block, block_mask[:, :prefix_length, :prefix_length])
