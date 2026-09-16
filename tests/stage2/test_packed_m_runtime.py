"""Runtime contracts for opt-in one-call packed Stage 2 M scoring."""

from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from stage2.objective import Stage2ObjectiveConfig
from stage2.objective_runtime import Stage2ObjectiveRuntime
from stage2.reference import FrozenReference
from stage2.rng import capture_rng_state, replay_rng_state
from peft_common import resolve_qwen2_5_vl_model
from transformers.models.qwen2_5_vl import stage2_latent_rollout

from .packed_m_helpers import (
    TEXT_VOCABULARY,
    TOKENS,
    make_packed_inputs,
    make_packed_model,
)


def test_packed_execution_is_an_explicit_opt_in_runtime_mode():
    """Catches routing packed M by default instead of retaining full replay."""
    config = Stage2ObjectiveConfig(enable_stage2_objective=True, lambda_vis=1.0)
    assert config.to_dict().get("stage2_m_execution") == "full_replay"

    packed = replace(config, stage2_m_execution="packed_suffix")
    model = make_packed_model()
    runtime = Stage2ObjectiveRuntime(
        packed, TOKENS, 2, TEXT_VOCABULARY, reference=None
    )
    inputs = make_packed_inputs(blocks=2)
    selection = runtime.select(inputs)

    result = runtime.forward(model, inputs, selection)

    assert result.metric_sums["model_forwards_M"] == 0
    assert result.metric_sums["decoder_forwards_M"] == 1
    assert result.metric_sums["m_branches"] == len(selection.blocks)
    assert result.metric_sums["m_query_rows"] == sum(
        max(block.post_positions) - block.end_position + 1
        for block in selection.blocks
    )
    assert torch.isfinite(result.vis_sum)


def _run_runtime(model, inputs, *, mode, retain_latents=False, post_only=False,
                 tokens=TOKENS, text_vocabulary=TEXT_VOCABULARY):
    import stage2.objective_runtime as objective_runtime

    config = Stage2ObjectiveConfig(
        enable_stage2_objective=True,
        lambda_vis=1.0,
        stage2_m_execution=mode,
        post_window=2,
    )
    runtime = Stage2ObjectiveRuntime(config, tokens, 2, text_vocabulary, None)
    selection = runtime.select(inputs)
    scores = []
    latents = []
    original_gain = objective_runtime.vision_gain
    original_replace = stage2_latent_rollout._replace_embedding
    lm_head_outputs = []

    def observe_gain(*args, **kwargs):
        metrics = original_gain(*args, **kwargs)
        scores.append(metrics["S_M"])
        return metrics

    def retain(embeds, position, value):
        if retain_latents and torch.is_grad_enabled():
            value.retain_grad()
            latents.append(value)
        return original_replace(embeds, position, value)

    lm_head_handle = resolve_qwen2_5_vl_model(model).lm_head.register_forward_hook(
        lambda _module, _args, value: lm_head_outputs.append(value.detach().clone())
    )
    try:
        with patch.object(objective_runtime, "vision_gain", side_effect=observe_gain), patch.object(
            stage2_latent_rollout, "_replace_embedding", side_effect=retain
        ):
            result = runtime.forward(model, inputs, selection)
    finally:
        lm_head_handle.remove()
    assert len(lm_head_outputs) == 1 + (len(selection.blocks) if mode == "full_replay" else 1)
    if mode == "full_replay":
        m_logits = torch.cat(
            [
                output[block.batch_index, list(block.post_positions)]
                for output, block in zip(lm_head_outputs[1:], selection.blocks, strict=True)
            ],
            dim=0,
        )
    else:
        m_logits = lm_head_outputs[-1][0]
    total = result.vis_sum if post_only else result.ce + result.vis_sum / result.count
    total.backward()
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    return result, tuple(scores), m_logits, gradients, latents, capture_rng_state()


@pytest.mark.parametrize(
    ("tuning", "checkpointing", "bf16"),
    [
        ("dora", False, False),
        ("lora", False, False),
        ("full", False, False),
        ("dora", True, True),
    ],
)
def test_actual_runtime_packed_matches_full_loss_scores_and_gradients(
    tuning, checkpointing, bf16
):
    """Catches packed routing changing real image/mRoPE objective values or gradients."""
    torch.manual_seed(307)
    initial = make_packed_model(tuning=tuning, checkpointing=checkpointing)
    full = deepcopy(initial)
    packed = deepcopy(initial)
    inputs = make_packed_inputs(blocks=2)
    paired_rng = capture_rng_state()
    autocast = torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16)

    with replay_rng_state(paired_rng), autocast:
        full_result, full_scores, full_m_logits, full_grads, full_latents, full_rng = _run_runtime(
            full, inputs, mode="full_replay", retain_latents=True
        )
    with replay_rng_state(paired_rng), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        packed_result, packed_scores, packed_m_logits, packed_grads, packed_latents, packed_rng = _run_runtime(
            packed, inputs, mode="packed_suffix", retain_latents=True
        )

    tolerance = {"atol": 1e-3, "rtol": 1e-2} if bf16 else {"atol": 1e-6, "rtol": 1e-5}
    torch.testing.assert_close(packed_result.outputs.logits, full_result.outputs.logits, atol=0, rtol=0)
    torch.testing.assert_close(packed_m_logits, full_m_logits, **tolerance)
    torch.testing.assert_close(packed_result.vis_sum, full_result.vis_sum, **tolerance)
    torch.testing.assert_close(torch.stack(packed_scores), torch.stack(full_scores), **tolerance)
    assert packed_grads.keys() == full_grads.keys()
    assert {name for name, value in packed_grads.items() if value is None} == {
        name for name, value in full_grads.items() if value is None
    }
    for name, packed_gradient in packed_grads.items():
        if packed_gradient is not None:
            assert torch.isfinite(packed_gradient).all(), name
            torch.testing.assert_close(packed_gradient, full_grads[name], **tolerance)
    assert full_latents and len(full_latents) == len(packed_latents)
    for index, (full_latent, packed_latent) in enumerate(
        zip(full_latents, packed_latents, strict=True)
    ):
        assert (full_latent.grad is None) == (packed_latent.grad is None), index
        assert full_latent.grad is not None and packed_latent.grad is not None, index
        assert torch.isfinite(full_latent.grad).all(), index
        assert torch.isfinite(packed_latent.grad).all(), index
        torch.testing.assert_close(packed_latent.grad, full_latent.grad, **tolerance)
    torch.testing.assert_close(packed_rng.cpu_state, full_rng.cpu_state, atol=0, rtol=0)
    assert packed_result.metric_sums["model_forwards_M"] == 0
    assert packed_result.metric_sums["decoder_forwards_M"] == 1
    assert packed_result.metric_sums["m_branches"] == full_result.count
    expected_query_rows = sum(
        max(block.post_positions) - block.end_position + 1
        for block in Stage2ObjectiveRuntime(
            Stage2ObjectiveConfig(True, 0, 1, stage2_m_execution="packed_suffix", post_window=2),
            TOKENS, 2, TEXT_VOCABULARY, None,
        ).select(inputs).blocks
    )
    assert packed_result.metric_sums["m_query_rows"] == expected_query_rows
    assert packed_result.metric_sums["decoder_forwards_F"] == full_result.metric_sums["decoder_forwards_F"]
    for name, value in packed.state_dict().items():
        torch.testing.assert_close(value, full.state_dict()[name], atol=0, rtol=0)


def test_post_only_packed_loss_has_independent_retained_latent_gradients():
    """Catches checking merely any parameter gradient while latent paths are detached."""
    model = make_packed_model(tuning="dora")
    inputs = make_packed_inputs(blocks=2)
    result, _, _, _, latents, _ = _run_runtime(
        model, inputs, mode="packed_suffix", retain_latents=True, post_only=True
    )
    assert result.vis_sum.requires_grad
    assert latents
    assert all(
        value.grad is not None
        and torch.isfinite(value.grad).all()
        and value.grad.abs().sum() > 0
        for value in latents
    )


@pytest.mark.parametrize("blocks", [1, 3])
def test_packed_runtime_keeps_normal_f_and_reference_counts_while_m_is_one(blocks):
    """Catches routing M through normal F/reference paths or adding a decoder per block."""
    inputs = make_packed_inputs(blocks=blocks)

    def run(mode):
        model = make_packed_model()
        reference = FrozenReference(
            deepcopy(model).requires_grad_(False).eval(), {"source": "frozen"}
        )
        config = Stage2ObjectiveConfig(
            enable_stage2_objective=True,
            lambda_dyn=1.0,
            lambda_vis=1.0,
            reference_model_path="frozen",
            stage2_m_execution=mode,
            post_window=2,
        )
        runtime = Stage2ObjectiveRuntime(config, TOKENS, 2, TEXT_VOCABULARY, reference)
        with torch.no_grad():
            result = runtime.forward(model, inputs, runtime.select(inputs))
        assert not any(parameter.grad is not None for parameter in reference.model.parameters())
        return result

    full = run("full_replay")
    packed = run("packed_suffix")
    assert packed.metric_sums["decoder_forwards_F"] == full.metric_sums["decoder_forwards_F"]
    assert packed.metric_sums["decoder_forwards_reference"] == full.metric_sums["decoder_forwards_reference"]
    assert full.metric_sums["model_forwards_M"] == blocks
    assert packed.metric_sums["model_forwards_M"] == 0
    assert packed.metric_sums["decoder_forwards_M"] == 1


@pytest.mark.parametrize("empty_blocks", [False, True])
def test_packed_capture_is_never_constructed_when_m_is_disabled_or_empty(
    monkeypatch, empty_blocks
):
    """Catches capture work leaking into zero-M paths that retain legacy execution."""
    import stage2.objective_runtime as objective_runtime

    inputs = make_packed_inputs(blocks=1)
    if empty_blocks:
        inputs["labels"][:] = -100
    config = Stage2ObjectiveConfig(
        enable_stage2_objective=True,
        lambda_vis=1.0 if empty_blocks else 0.0,
        stage2_m_execution="packed_suffix",
    )
    runtime = Stage2ObjectiveRuntime(config, TOKENS, 2, TEXT_VOCABULARY, None)
    monkeypatch.setattr(
        objective_runtime,
        "PackedMPrefixCapture",
        lambda *_args, **_kwargs: pytest.fail("packed capture must not be constructed"),
    )
    result = runtime.forward(make_packed_model(), inputs, runtime.select(inputs))
    assert "decoder_forwards_M" not in result.metric_sums
