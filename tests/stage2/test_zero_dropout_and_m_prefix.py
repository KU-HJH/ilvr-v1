"""ours1 zero-dropout policy and the score-preserving full-replay M prefix cut."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from peft_common import resolve_qwen2_5_vl_model
from stage2.dropout_policy import apply_zero_dropout_policy, has_active_dropout
from stage2.objective import Stage2ObjectiveConfig
from stage2.objective_runtime import Stage2ObjectiveRuntime, m_prefix_inputs, model_inputs
from stage2.objective_windows import WindowBlock

from .packed_m_helpers import TEXT_VOCABULARY, TOKENS, make_packed_inputs, make_packed_model


def test_zero_dropout_replaces_adapter_dropout_with_identity_and_keeps_config():
    """Catches a Dropout(p=0) left in place (different DoRA path) or a rewritten adapter config."""
    model = make_packed_model(tuning="dora")
    assert has_active_dropout(model)
    config_dropout = model.peft_config["default"].lora_dropout
    originals = apply_zero_dropout_policy(model)
    assert not has_active_dropout(model)
    adapter_sites = {name: value for name, value in originals.items() if ".lora_dropout." in name}
    assert adapter_sites and set(adapter_sites.values()) == {0.05}
    assert all(isinstance(dropout, nn.Identity) and dropout.training
               for module in model.modules() if hasattr(module, "lora_dropout")
               for dropout in module.lora_dropout.values())
    assert model.peft_config["default"].lora_dropout == config_dropout == 0.05


def _runtime():
    config = Stage2ObjectiveConfig(enable_stage2_objective=True, lambda_vis=0.3, m_V=0.0,
                                   stage2_m_execution="full_replay", post_window=1,
                                   stage2_m_prefix_cut=True)
    return Stage2ObjectiveRuntime(config, TOKENS, 2, TEXT_VOCABULARY, None)


def _m_input_lengths(model, run):
    lengths = []
    decoder = resolve_qwen2_5_vl_model(model).model
    handle = decoder.register_forward_pre_hook(
        lambda _m, args, kwargs: lengths.append(kwargs["inputs_embeds"].shape[1]), with_kwargs=True)
    try:
        result = run()
    finally:
        handle.remove()
    return result, lengths


def test_m_prefix_cut_preserves_scores_and_gradients_without_dropout():
    """Catches the cut changing S_M/loss/gradients, or silently not cutting at all."""
    torch.manual_seed(503)
    base = make_packed_model(tuning="dora")
    apply_zero_dropout_policy(base)
    with torch.no_grad():
        resolve_qwen2_5_vl_model(base).lm_head.weight.mul_(50.0)
    inputs = make_packed_inputs(blocks=3)
    runtime = _runtime()
    selection = runtime.select(inputs)
    assert len(selection.blocks) == 3

    import stage2.objective_runtime as objective_runtime
    results = {}
    for mode in ("cut", "full"):
        model = deepcopy(base)
        original = objective_runtime.m_prefix_inputs
        if mode == "full":
            objective_runtime.m_prefix_inputs = lambda inputs, block, **_: inputs
        try:
            result, lengths = _m_input_lengths(model, lambda: runtime.forward(model, inputs, selection))
        finally:
            objective_runtime.m_prefix_inputs = original
        result.vis_sum.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        results[mode] = (result, lengths, grads)
    (cut, cut_lengths, cut_grads), (full, full_lengths, full_grads) = results["cut"], results["full"]
    torch.testing.assert_close(cut.metric_sums["S_M"], full.metric_sums["S_M"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cut.vis_sum, full.vis_sum, atol=1e-5, rtol=1e-5)
    for name, value in cut_grads.items():
        torch.testing.assert_close(value, full_grads[name], atol=1e-5, rtol=1e-4)
    length = inputs["input_ids"].shape[1]
    assert max(cut_lengths) == length                       # F is never cut
    assert sum(cut_lengths) < sum(full_lengths)             # M prefixes really were cut
    expected = [max(b.post_positions) + 1 for b in selection.blocks]
    assert all(any(n == e for n in cut_lengths) for e in expected)


def test_m_prefix_cut_is_skipped_when_unsafe():
    """Catches cutting with active dropout, a later image, or a hidden span past the cut."""
    inputs = model_inputs(make_packed_inputs(blocks=2))
    block = WindowBlock(0, 0, 5, 8, (4,), (8,), (9,))
    cut = m_prefix_inputs(inputs, block, image_token_id=80, dropout_active=False)
    assert cut["input_ids"].shape[1] == 9 and cut["attention_mask"].shape[1] == 9
    assert m_prefix_inputs(inputs, block, image_token_id=80, dropout_active=True) is inputs
    later_image = dict(inputs, input_ids=inputs["input_ids"].clone())
    later_image["input_ids"][0, -1] = 80
    assert m_prefix_inputs(later_image, block, image_token_id=80, dropout_active=False) is later_image
    spanning = WindowBlock(0, 0, 5, 8, (), (8,), (9,), hidden_spans=((5, 8), (12, 15)))
    assert m_prefix_inputs(inputs, spanning, image_token_id=80, dropout_active=False) is inputs
