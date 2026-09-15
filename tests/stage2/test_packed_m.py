"""One-call packed Stage 2 M suffix scorer contracts."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from peft_common import resolve_qwen2_5_vl_model
from stage2.objective import target_log_likelihood
from stage2.objective_runtime import model_inputs
from stage2 import packed_m
from stage2.packed_m import PackedMPrefixCapture, score_packed_m
from stage2.rng import capture_rng_state, replay_rng_state
from transformers.models.qwen2_5_vl import stage2_latent_rollout
from .packed_m_helpers import (
    autocast_for,
    decoder,
    independent_full_replay_scores,
    make_packed_inputs,
    make_packed_model,
    select_packed_blocks,
)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_packed_m_one_decoder_call_matches_independent_full_replays(
    backend, dtype, checkpointing
):
    """Catches packed branches changing full-replay scores or adding decoder calls."""
    torch.manual_seed(107)
    model = make_packed_model(backend=backend, checkpointing=checkpointing)
    inputs = make_packed_inputs()
    blocks = select_packed_blocks(inputs)
    pre_f_rng = capture_rng_state()
    with autocast_for(dtype):
        with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
            model(**model_inputs(inputs), collect_stage2_diagnostics=True)

        calls = 0

        def observed(_module, _args):
            nonlocal calls
            calls += 1

        handle = decoder(model).register_forward_pre_hook(observed)
        try:
            scored = score_packed_m(model, inputs, blocks, captured)
        finally:
            handle.remove()
        oracle = independent_full_replay_scores(model, inputs, blocks, pre_f_rng)

    tolerance = (
        {"atol": 1e-3, "rtol": 1e-2}
        if dtype == torch.bfloat16
        else {"atol": 1e-6, "rtol": 1e-5}
    )
    assert scored.decoder_calls == calls == 1
    assert scored.query_rows == sum(
        max(block.post_positions) - block.end_position + 1 for block in blocks
    )
    assert len(scored.scores) == len(blocks)
    assert all(not score.requires_grad for score in scored.scores)
    torch.testing.assert_close(torch.stack(scored.scores), oracle, **tolerance)


@pytest.mark.parametrize("tuning", ["dora", "lora", "full"])
@pytest.mark.parametrize("padding", [None, "left", "right"])
def test_packed_m_supports_tuning_and_padding_contracts(tuning, padding):
    """Catches adapter-only assumptions or padding becoming visible cache state."""
    torch.manual_seed(109)
    model = make_packed_model(tuning=tuning)
    inputs = make_packed_inputs(padding=padding)
    blocks = select_packed_blocks(inputs, post_window=1)
    pre_f_rng = capture_rng_state()
    with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
        model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    scored = score_packed_m(model, inputs, blocks, captured)
    oracle = independent_full_replay_scores(model, inputs, blocks, pre_f_rng)
    torch.testing.assert_close(
        torch.stack(scored.scores), oracle, atol=1e-6, rtol=1e-5
    )


def test_packed_m_zero_blocks_skips_decoder_and_preserves_rng():
    """Catches empty M work mutating state or pretending a decoder ran."""
    model = make_packed_model()
    inputs = make_packed_inputs(blocks=1)
    before = capture_rng_state()
    scored = score_packed_m(
        model,
        inputs,
        (),
        PackedMPrefixCapture(model, inputs["input_ids"].shape[1]),
    )
    after = capture_rng_state()
    assert scored.scores == ()
    assert scored.decoder_calls == scored.query_rows == 0
    torch.testing.assert_close(after.cpu_state, before.cpu_state)


def test_capture_is_observational_and_removed_before_backward():
    """Catches capture altering F, leaking hooks, or surviving into recomputation."""
    torch.manual_seed(113)
    baseline = make_packed_model(checkpointing=True)
    captured_model = make_packed_model(checkpointing=True)
    captured_model.load_state_dict(baseline.state_dict())
    inputs = make_packed_inputs(blocks=1)
    paired = capture_rng_state()
    torch.set_rng_state(paired.cpu_state)
    plain = baseline(**model_inputs(inputs), collect_stage2_diagnostics=True)
    plain_rng = capture_rng_state()
    torch.set_rng_state(paired.cpu_state)
    parameters_before = {
        name: value.detach().clone() for name, value in captured_model.state_dict().items()
    }
    hooks_before = sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in captured_model.modules())
    with PackedMPrefixCapture(captured_model, inputs["input_ids"].shape[1]) as captured:
        observed = captured_model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    hooks_after = sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in captured_model.modules())
    torch.testing.assert_close(observed.logits, plain.logits, atol=0, rtol=0)
    torch.testing.assert_close(capture_rng_state().cpu_state, plain_rng.cpu_state)
    for name, before in parameters_before.items():
        torch.testing.assert_close(captured_model.state_dict()[name], before, atol=0, rtol=0)
    assert hooks_after == hooks_before
    observed.loss.backward()
    assert any(parameter.grad is not None for parameter in captured_model.parameters())
    assert captured.ready


def test_packed_m_preserves_outer_rng_and_temporary_forward_attributes():
    """Catches replay masks consuming RNG or leaving patched dropout methods behind."""
    torch.manual_seed(127)
    model = make_packed_model()
    inputs = make_packed_inputs()
    blocks = select_packed_blocks(inputs)
    with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
        model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    dropouts = [
        module
        for module in resolve_qwen2_5_vl_model(model).model.modules()
        if isinstance(module, torch.nn.Dropout)
    ]
    attributes_before = [("forward" in module.__dict__, module.__dict__.get("forward")) for module in dropouts]
    rng_before = capture_rng_state()
    score_packed_m(model, inputs, blocks, captured)
    rng_after = capture_rng_state()
    attributes_after = [("forward" in module.__dict__, module.__dict__.get("forward")) for module in dropouts]
    torch.testing.assert_close(rng_after.cpu_state, rng_before.cpu_state)
    assert attributes_after == attributes_before


def test_capture_rejects_reuse_and_cleans_up_after_f_exception():
    """Catches overwritten capture state and dangling hooks on exceptional F."""
    model = make_packed_model()
    inputs = make_packed_inputs(blocks=1)
    capture = PackedMPrefixCapture(model, inputs["input_ids"].shape[1])
    hooks_before = sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules())
    with pytest.raises(RuntimeError, match="capture failure"):
        with capture:
            raise RuntimeError("capture failure")
    hooks_after = sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules())
    assert hooks_after == hooks_before
    with pytest.raises(RuntimeError, match="cannot be reused"):
        with capture:
            pass


def test_capture_cleans_hooks_after_decoder_exception_and_normal_f_still_backprops():
    """Catches an exceptional F decoder call leaving observational hooks active."""
    model = make_packed_model()
    inputs = make_packed_inputs(blocks=1)
    hooks_before = sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules())
    failure = decoder(model).layers[0].register_forward_pre_hook(
        lambda _module, _args: (_ for _ in ()).throw(RuntimeError("F decoder failure"))
    )
    with pytest.raises(RuntimeError, match="F decoder failure"):
        with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]):
            model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    failure.remove()
    hooks_after = sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules())
    assert hooks_after == hooks_before
    output = model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_capture_rejects_a_second_full_length_f_without_overwriting_state():
    """Catches a capture silently replacing the first final-F prefix state."""
    model = make_packed_model()
    inputs = make_packed_inputs(blocks=1)
    capture = PackedMPrefixCapture(model, inputs["input_ids"].shape[1])
    with capture:
        model(**model_inputs(inputs), collect_stage2_diagnostics=True)
        first_embeds = capture._inputs_embeds.clone()
        with pytest.raises(RuntimeError, match="more than one final F"):
            model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    torch.testing.assert_close(capture._inputs_embeds, first_embeds, atol=0, rtol=0)


def test_capture_rejects_noncontiguous_final_f_layout_and_removes_hooks():
    """Catches an unsupported final-F embedding layout entering cache capture."""
    model = make_packed_model()
    inputs = make_packed_inputs(blocks=1)
    root_length = inputs["input_ids"].shape[1]

    def make_noncontiguous(_module, args, kwargs):
        embeds = kwargs.get("inputs_embeds")
        if isinstance(embeds, torch.Tensor) and embeds.shape[1] == root_length:
            kwargs = dict(kwargs)
            kwargs["inputs_embeds"] = embeds.transpose(1, 2).contiguous().transpose(1, 2)
            assert not kwargs["inputs_embeds"].is_contiguous()
            return args, kwargs
        return None

    decoder_handle = decoder(model).register_forward_pre_hook(
        make_noncontiguous, with_kwargs=True
    )
    hooks_with_injector = sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in model.modules()
    )
    try:
        with pytest.raises(ValueError, match="contiguous final F"):
            with PackedMPrefixCapture(model, root_length):
                model(**model_inputs(inputs), collect_stage2_diagnostics=True)
        hooks_after = sum(
            len(module._forward_hooks) + len(module._forward_pre_hooks)
            for module in model.modules()
        )
        assert hooks_after == hooks_with_injector
    finally:
        decoder_handle.remove()


def test_capture_supports_unique_full_length_f_during_no_grad_evaluation():
    """Catches inherited Trainer evaluation missing F capture under no_grad."""
    model = make_packed_model().eval()
    inputs = make_packed_inputs(blocks=1)
    blocks = select_packed_blocks(inputs)
    with torch.no_grad():
        with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
            model(**model_inputs(inputs), collect_stage2_diagnostics=True)
        scored = score_packed_m(model, inputs, blocks, captured)
    assert captured.ready
    assert scored.decoder_calls == 1
    assert len(scored.scores) == 1


def test_packed_m_builds_exact_branch_mask_and_original_mrope_positions():
    """Catches physical cache positions replacing mRoPE or cross-branch visibility."""
    model = make_packed_model(tuning="full")
    inputs = make_packed_inputs()
    blocks = select_packed_blocks(inputs)
    visual_calls = 0

    def observed_visual(_module, _args, _output):
        nonlocal visual_calls
        visual_calls += 1

    visual_handle = resolve_qwen2_5_vl_model(model).visual.register_forward_hook(
        observed_visual
    )
    with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
        model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    visual_handle.remove()
    decoder_kwargs = []

    def observed_decoder(_module, _args, kwargs):
        decoder_kwargs.append(kwargs)

    handle = decoder(model).register_forward_pre_hook(observed_decoder, with_kwargs=True)
    try:
        scored = score_packed_m(model, inputs, blocks, captured)
    finally:
        handle.remove()

    assert visual_calls == 1
    assert len(decoder_kwargs) == 1
    values = decoder_kwargs[0]
    logical_rows = torch.tensor(
        [
            position
            for block in blocks
            for position in range(block.end_position, max(block.post_positions) + 1)
        ]
    )
    torch.testing.assert_close(
        values["position_ids"], captured._position_ids.index_select(2, logical_rows)
    )
    # The actual image path produced nontrivial full-sequence mRoPE; packed text
    # suffixes intentionally gather their original (equal-axis) coordinates.
    assert not torch.equal(captured._position_ids[0], captured._position_ids[2])
    root_length = inputs["input_ids"].shape[1]
    torch.testing.assert_close(
        values["cache_position"], torch.arange(root_length, root_length + scored.query_rows)
    )
    additive = values["attention_mask"]
    assert additive.shape == (1, 1, scored.query_rows, root_length + scored.query_rows)
    allowed = additive[0, 0] == 0
    offset = 0
    for block in blocks:
        length = max(block.post_positions) - block.end_position + 1
        root = allowed[offset : offset + length, :root_length]
        assert torch.all(root[:, : block.start_position + 1])
        assert not torch.any(root[:, block.start_position + 1 :])
        own = allowed[
            offset : offset + length,
            root_length + offset : root_length + offset + length,
        ]
        torch.testing.assert_close(own, torch.ones_like(own).tril())
        assert not torch.any(allowed[offset : offset + length, root_length : root_length + offset])
        assert not torch.any(allowed[offset : offset + length, root_length + offset + length :])
        offset += length


def test_packed_m_replays_skipped_intermediate_rows_and_larger_k_once():
    """Catches selector skips shortening causal suffixes or K increasing call count."""
    torch.manual_seed(131)
    model = make_packed_model()
    inputs = make_packed_inputs(blocks=3)
    # Skip the first post target of the middle block; later predictors still
    # require its teacher-forced row in the contiguous suffix.
    middle_end = select_packed_blocks(inputs, post_window=3)[1].end_position
    inputs["labels"][0, middle_end + 1] = -100
    blocks = select_packed_blocks(inputs, post_window=3)
    assert len(blocks) == 3
    assert blocks[1].post_positions[0] == middle_end + 1
    pre_f_rng = capture_rng_state()
    with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
        model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    scored = score_packed_m(model, inputs, blocks, captured)
    oracle = independent_full_replay_scores(model, inputs, blocks, pre_f_rng)
    assert scored.decoder_calls == 1
    assert scored.query_rows == sum(
        max(block.post_positions) - block.end_position + 1 for block in blocks
    )
    torch.testing.assert_close(torch.stack(scored.scores), oracle, atol=1e-6, rtol=1e-5)


def _captured_outputs_with_latent_delta(model, inputs, blocks, paired_rng, delta):
    original_replace = stage2_latent_rollout._replace_embedding

    def replace_latent(embeds, position, value):
        if blocks[0].start_position < position < blocks[0].end_position:
            value = value + delta.to(value)
        return original_replace(embeds, position, value)

    with replay_rng_state(paired_rng), patch.object(
        stage2_latent_rollout, "_replace_embedding", side_effect=replace_latent
    ):
        with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
            output = model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    selected_logits = []
    handle = resolve_qwen2_5_vl_model(model).lm_head.register_forward_hook(
        lambda _module, _args, value: selected_logits.append(value.detach().clone())
    )
    try:
        scored = score_packed_m(model, inputs, blocks, captured)
    finally:
        handle.remove()
    return output.logits[0, list(blocks[0].post_positions)].detach(), selected_logits[-1], scored, captured


def test_packed_m_blocks_selected_latents_and_exposed_end_control_detects_leakage():
    """Catches selected-slot or clean-END cache leakage into its M branch."""
    torch.manual_seed(137)
    model = make_packed_model()
    inputs = make_packed_inputs()
    blocks = select_packed_blocks(inputs)
    paired = capture_rng_state()
    zero = torch.zeros(model.config.hidden_size)
    delta = torch.tensor([90.0, -80.0, 70.0, -60.0, 50.0, -40.0, 30.0, -20.0])
    f_clean, m_clean, _, clean_capture = _captured_outputs_with_latent_delta(
        model, inputs, blocks, paired, zero
    )
    f_changed, m_changed, _, changed_capture = _captured_outputs_with_latent_delta(
        model, inputs, blocks, paired, delta
    )
    first_count = len(blocks[0].post_positions)
    safe_delta = float((m_clean[:, :first_count] - m_changed[:, :first_count]).abs().max())
    f_delta = float((f_clean - f_changed).abs().max())
    assert safe_delta <= 1e-7
    assert f_delta > 1e-4

    original_layout = packed_m._packed_layout

    def exposed_end(values, selected, capture):
        rows, visible, branches = original_layout(values, selected, capture)
        offset = 0
        for block, branch in zip(selected, branches, strict=True):
            visible[offset : offset + len(branch), block.end_position] = True
            offset += len(branch)
        return rows, visible, branches

    leaked_logits = []
    changed_leaked_logits = []
    with patch.object(packed_m, "_packed_layout", side_effect=exposed_end):
        handle = resolve_qwen2_5_vl_model(model).lm_head.register_forward_hook(
            lambda _module, _args, value: leaked_logits.append(value.detach().clone())
        )
        score_packed_m(model, inputs, blocks, clean_capture)
        handle.remove()
        handle = resolve_qwen2_5_vl_model(model).lm_head.register_forward_hook(
            lambda _module, _args, value: changed_leaked_logits.append(value.detach().clone())
        )
        score_packed_m(model, inputs, blocks, changed_capture)
        handle.remove()
    assert float(
        (
            leaked_logits[-1][:, :first_count]
            - changed_leaked_logits[-1][:, :first_count]
        ).abs().max()
    ) > 1e-6


def test_packed_m_cross_branch_suffixes_are_isolated():
    """Catches a packed branch attending to another branch's recomputed suffix."""
    model = make_packed_model(tuning="full")
    inputs = make_packed_inputs()
    blocks = select_packed_blocks(inputs)
    with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
        model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    clean = score_packed_m(model, inputs, blocks, captured)
    first = blocks[0]
    rows = list(range(first.end_position, max(first.post_positions) + 1))
    captured._inputs_embeds[:, rows] += 50.0
    changed = score_packed_m(model, inputs, blocks, captured)
    assert not torch.allclose(clean.scores[0], changed.scores[0])
    torch.testing.assert_close(clean.scores[1], changed.scores[1], atol=0, rtol=0)


def test_no_intervention_packed_suffix_matches_final_f_score():
    """Catches detached cache replay changing an otherwise fully visible F path."""
    model = make_packed_model()
    inputs = make_packed_inputs(blocks=1)
    selected = select_packed_blocks(inputs)[0]
    no_intervention = replace(
        selected, start_position=selected.end_position - 1
    )
    with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
        output = model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    f_logits = output.logits[0, list(selected.post_positions)]
    targets = torch.tensor(selected.target_ids, device=f_logits.device)
    expected = target_log_likelihood(f_logits, targets).mean()
    scored = score_packed_m(model, inputs, (no_intervention,), captured)
    torch.testing.assert_close(scored.scores[0], expected, atol=1e-6, rtol=1e-5)


def test_packed_m_unmatched_dropout_negative_control_is_observable():
    """Catches tests passing when full-shape final-F dropout replay is removed."""
    model = make_packed_model()
    inputs = make_packed_inputs()
    blocks = select_packed_blocks(inputs)
    with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
        model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    matched = score_packed_m(model, inputs, blocks, captured)
    for index, (site_id, record) in enumerate(captured._dropout_records.items()):
        wrong_state = torch.Generator().manual_seed(991 + index).get_state()
        captured._dropout_records[site_id] = replace(record, rng_state=wrong_state)
    unmatched = score_packed_m(model, inputs, blocks, captured)
    assert float((torch.stack(matched.scores) - torch.stack(unmatched.scores)).abs().max()) > 1e-6


@pytest.mark.parametrize("invalid", ["inplace", "multiple", "merged", "nonstandard"])
def test_packed_m_rejects_unsupported_adapter_layouts(invalid):
    """Catches stochastic replay silently accepting an unverified PEFT layout."""
    model = make_packed_model()
    first = next(
        module
        for module in resolve_qwen2_5_vl_model(model).model.modules()
        if hasattr(module, "lora_dropout")
    )
    if invalid == "inplace":
        first.lora_dropout["default"].inplace = True
    elif invalid == "multiple":
        model.add_adapter("second", model.peft_config["default"])
    elif invalid == "merged":
        first.merged_adapters.append("default")
    else:
        first.lora_dropout["default"] = torch.nn.AlphaDropout(0.05)
    with pytest.raises((ValueError, RuntimeError), match="packed M"):
        with PackedMPrefixCapture(model, make_packed_inputs(blocks=1)["input_ids"].shape[1]):
            pass


def test_packed_m_rejects_repeated_stochastic_site_visitation():
    """Catches a shared/repeated adapter dropout site overwriting captured RNG."""
    model = make_packed_model()
    modules = [
        module
        for module in resolve_qwen2_5_vl_model(model).model.modules()
        if hasattr(module, "lora_dropout")
    ]
    modules[1].lora_dropout["default"] = modules[0].lora_dropout["default"]
    inputs = make_packed_inputs(blocks=1)
    with pytest.raises(RuntimeError, match="one final F invocation"):
        with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]):
            model(**model_inputs(inputs), collect_stage2_diagnostics=True)


def test_packed_m_restores_dropout_and_decoder_hooks_after_m_exception():
    """Catches exceptional packed execution leaking instance methods or hooks."""
    model = make_packed_model()
    inputs = make_packed_inputs(blocks=1)
    blocks = select_packed_blocks(inputs)
    with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
        model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    dropouts = tuple(record.module for record in captured._dropout_records.values())
    attributes_before = [("forward" in module.__dict__, module.__dict__.get("forward")) for module in dropouts]
    hooks_before = sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules())
    failure = decoder(model).layers[0].register_forward_pre_hook(
        lambda _module, _args: (_ for _ in ()).throw(RuntimeError("M failure"))
    )
    with pytest.raises(RuntimeError, match="M failure"):
        score_packed_m(model, inputs, blocks, captured)
    failure.remove()
    attributes_after = [("forward" in module.__dict__, module.__dict__.get("forward")) for module in dropouts]
    hooks_after = sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules())
    assert attributes_after == attributes_before
    assert hooks_after == hooks_before
    output = model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_packed_m_rejects_changed_dropout_identity_before_rng_can_advance():
    """Catches an unmatched replay site running stochastically before rejection."""
    model = make_packed_model()
    inputs = make_packed_inputs(blocks=1)
    blocks = select_packed_blocks(inputs)
    with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
        model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    first = next(
        module
        for module in resolve_qwen2_5_vl_model(model).model.modules()
        if hasattr(module, "lora_dropout")
    )
    first.lora_dropout["default"] = torch.nn.Dropout(0.05).train()
    rng_before = capture_rng_state()
    with pytest.raises(ValueError, match="stochastic site identities"):
        score_packed_m(model, inputs, blocks, captured)
    torch.testing.assert_close(capture_rng_state().cpu_state, rng_before.cpu_state)


def test_packed_m_cpu_path_does_not_initialize_cuda(monkeypatch):
    """Catches CPU capture/replay querying an unused CUDA generator."""
    if torch.cuda.is_initialized():
        pytest.skip("CUDA was already initialized by this process")
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda *args, **kwargs: pytest.fail("unused CUDA RNG was queried"),
    )
    model = make_packed_model()
    inputs = make_packed_inputs(blocks=1)
    blocks = select_packed_blocks(inputs)
    with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
        model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    score_packed_m(model, inputs, blocks, captured)
    assert not torch.cuda.is_initialized()
