"""Production Stage 1 artifact handoff must preserve sparse rows in packed M."""

from copy import deepcopy

import pytest
import torch

from peft_common import resolve_qwen2_5_vl_model
from stage2.objective import Stage2ObjectiveConfig
from stage2.objective_runtime import Stage2ObjectiveRuntime
from stage2.objective_windows import TextVocabulary
from stage2.packed_m import PackedMPrefixCapture
from stage2.rng import capture_rng_state, replay_rng_state
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds

from .packed_m_helpers import make_packed_inputs
from .test_objective_reference import (
    LATENT_TOKEN_IDS, _student_from_dora, _write_schema2_dora,
    checkpoint_directory,
)
from .test_packed_m_runtime import _run_runtime
from .test_packed_m import _captured_outputs_with_latent_delta


@pytest.fixture
def sparse_model(checkpoint_directory):
    sources = _write_schema2_dora(checkpoint_directory, dropout=0.05)
    model = _student_from_dora(*sources, dropout=0.05)
    model.train()
    return model


def sparse_inputs():
    inputs = make_packed_inputs(blocks=2)
    inputs.pop("pixel_values")
    inputs.pop("image_grid_thw")
    for key in ("input_ids", "labels"):
        for old, new in zip((90, 91, 92), LATENT_TOKEN_IDS):
            inputs[key][inputs[key] == old] = new
    return inputs


@pytest.mark.parametrize("bf16_checkpointing", [False, True])
def test_loaded_stage1_sparse_rows_match_independent_replay(sparse_model, bf16_checkpointing):
    """Rejecting sparse wrappers or bypassing either sparse row path breaks this oracle."""
    base = resolve_qwen2_5_vl_model(sparse_model)
    embed = base.model.embed_tokens
    ids = torch.tensor(LATENT_TOKEN_IDS)
    assert not torch.equal(embed(ids), embed.original_module(ids))
    hidden = torch.ones(1, 1, base.config.hidden_size)
    assert not torch.equal(base.lm_head(hidden), base.lm_head.base_head(hidden))
    if bf16_checkpointing:
        sparse_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    full, packed = deepcopy(sparse_model), deepcopy(sparse_model)
    rng = capture_rng_state()
    results = []
    for model, mode in ((full, "full_replay"), (packed, "packed_suffix")):
        with replay_rng_state(rng), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16_checkpointing):
            results.append(_run_runtime(
                model, sparse_inputs(), mode=mode, retain_latents=True,
                tokens=LatentTokenIds(*LATENT_TOKEN_IDS),
                text_vocabulary=TextVocabulary(tuple(range(1, 125)), {}, 128),
            ))
    f, p = results
    tolerance = dict(atol=1e-3, rtol=1e-2) if bf16_checkpointing else dict(atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(p[0].outputs.logits, f[0].outputs.logits, atol=0, rtol=0)
    torch.testing.assert_close(p[0].ce, f[0].ce, atol=0, rtol=0)
    torch.testing.assert_close(p[0].vis_sum, f[0].vis_sum, **tolerance)
    torch.testing.assert_close(torch.stack(p[1]), torch.stack(f[1]), **tolerance)
    torch.testing.assert_close(p[2], f[2], **tolerance)
    assert all(not score.requires_grad for score in p[1])
    for name, grad in p[3].items():
        assert (grad is None) == (f[3][name] is None), name
        if grad is not None:
            assert torch.isfinite(grad).all(), name
            torch.testing.assert_close(grad, f[3][name], **tolerance)
    for suffix in ("trainable_tokens_delta.default", "lm_head.output_delta"):
        grads = [grad for name, grad in p[3].items() if name.endswith(suffix)]
        assert len(grads) == 1 and grads[0] is not None and grads[0].abs().sum() > 0
    assert p[4] and len(p[4]) == len(f[4])
    for actual, expected in zip(p[4], f[4], strict=True):
        assert actual.grad is not None and actual.grad.abs().sum() > 0
        torch.testing.assert_close(actual.grad, expected.grad, **tolerance)
    torch.testing.assert_close(p[5].cpu_state, f[5].cpu_state, atol=0, rtol=0)
    assert p[0].count == 2
    assert p[0].metric_sums["decoder_forwards_M"] == 1
    assert p[0].metric_sums["model_forwards_M"] == 0
    for name, value in packed.state_dict().items():
        torch.testing.assert_close(value, full.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize("mutation", [
    "input_ids", "output_ids", "config_ids", "disabled", "merged", "tied",
    "active_adapter", "max_norm", "extra_wrapper", "not_dora", "head_bypass",
    "projection_disabled", "projection_not_dora",
])
def test_sparse_exception_rejects_noncanonical_layouts(sparse_model, mutation):
    """A sparse-row exception must not become a generic auxiliary-tuner bypass."""
    base = resolve_qwen2_5_vl_model(sparse_model)
    wrapper = base.model.embed_tokens
    layer = wrapper.token_adapter
    if mutation == "input_ids":
        layer.token_indices["default"][0] = 1
    elif mutation == "output_ids":
        base.lm_head.token_ids[0] = 1
    elif mutation == "config_ids":
        base.config.latent_start_token_id = 1
    elif mutation == "disabled":
        wrapper.enable_adapters(False)
    elif mutation == "merged":
        layer.merge()
    elif mutation == "tied":
        layer._tied_adapter = [deepcopy(layer)]
    elif mutation == "active_adapter":
        layer._active_adapter = "other"
    elif mutation == "max_norm":
        layer.base_layer.max_norm = 1.0
    elif mutation == "extra_wrapper":
        base.extra_wrapper = deepcopy(wrapper)
    elif mutation == "not_dora":
        sparse_model.peft_config["default"].use_dora = False
    elif mutation == "head_bypass":
        base.lm_head = base.lm_head.base_head
    elif mutation == "projection_disabled":
        base.model.layers[0].self_attn.q_proj.enable_adapters(False)
    elif mutation == "projection_not_dora":
        base.model.layers[0].self_attn.q_proj.use_dora["default"] = False
    with pytest.raises(ValueError):
        with PackedMPrefixCapture(sparse_model, sparse_inputs()["input_ids"].shape[1]):
            pass


def test_sparse_packed_m_does_not_leak_selected_latents(sparse_model):
    """Perturbing selected generated slots changes F but not that block's M logits."""
    inputs = sparse_inputs()
    runtime = Stage2ObjectiveRuntime(
        Stage2ObjectiveConfig(True, 0, 1, post_window=2, stage2_m_execution="packed_suffix"),
        LatentTokenIds(*LATENT_TOKEN_IDS), 2,
        TextVocabulary(tuple(range(1, 125)), {}, 128), None,
    )
    blocks = runtime.select(inputs).blocks
    rng = capture_rng_state()
    clean = _captured_outputs_with_latent_delta(
        sparse_model, inputs, blocks, rng, torch.zeros(8)
    )
    changed = _captured_outputs_with_latent_delta(
        sparse_model, inputs, blocks, rng,
        torch.tensor([90., -80., 70., -60., 50., -40., 30., -20.]),
    )
    count = len(blocks[0].post_positions)
    torch.testing.assert_close(clean[1][:, :count], changed[1][:, :count], atol=1e-7, rtol=0)
    assert (clean[0] - changed[0]).abs().max() > 1e-4
