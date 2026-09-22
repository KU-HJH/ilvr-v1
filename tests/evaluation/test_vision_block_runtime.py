"""Runtime witnesses for observer capture and independent local attention edits."""
from dataclasses import replace
import importlib
import importlib.util

import pytest
import torch

from evaluation.vision_block_inputs import PreparedSample, Site, _sample_identity
from step_preference.equivalence import _tiny_model
from tests.evaluation.vision_block_reference import reference_logits


@pytest.fixture
def runtime():
    if importlib.util.find_spec("evaluation.vision_block_runtime") is None:
        return None
    return importlib.import_module("evaluation.vision_block_runtime")


def image_case(two_sites=False):
    torch.manual_seed(481)
    config = _tiny_model().config
    config.num_hidden_layers = 2
    config.latent_size = 8
    config.vision_start_token_id = 78
    config.vision_end_token_id = 79
    config._attn_implementation = "sdpa"
    model = type(_tiny_model())(config).eval()
    tokens = [11, 78, 80, 80, 79, 12, 91, *([90] * 8), 92, 13, 14]
    sites = [Site(0, 1, 6, 7, 15, 15, (16, 17), (13, 14))]
    if two_sites:
        tokens += [91, *([90] * 8), 92, 15, 16, 17]
        sites.append(Site(0, 3, 18, 19, 27, 27, (28, 29, 30), (15, 16, 17)))
    ids = torch.tensor([tokens])
    p = torch.arange(len(tokens))[None, None, :]
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  position_ids=torch.cat((p, 2*p, 3*p)),
                  pixel_values=torch.linspace(-1, 1, 24).reshape(2, 12),
                  image_grid_thw=torch.tensor([[1, 1, 2]]))
    return model, PreparedSample(0, inputs, tuple(sites))


@torch.inference_mode()
def production_logits(runtime, model, state, site, condition):
    embeds, positions, mask = runtime.build_replay_inputs(state, site, condition)
    hidden = model.model(inputs_embeds=embeds, position_ids=positions,
                         attention_mask=mask, stage2_prefix_key_values=state.prefix_kv,
                         generate_mode=True, use_cache=False, return_dict=True).last_hidden_state
    return model.lm_head(hidden).float()


@pytest.mark.parametrize("condition,root", [
    ("clean", list(range(11))), ("body_mask", [0, 1, 2]),
    ("boundary_mask", [0, 1, *range(3, 11)]), ("full_mask", [0, 1]),
])
def test_literal_visibility_keeps_axes_and_only_replayed_end(runtime, condition, root):
    assert runtime is not None, "capture/replay runtime is missing"
    # Catches wrong boundaries, root END leakage, renumbered mRoPE and shared mutation.
    p = torch.arange(14)[None, None, :]
    state = runtime.CleanState(torch.ones(1, 14, 8), torch.cat((p, 2*p, 3*p)), (), {}, {})
    site = Site(0, 1, 2, 3, 11, 11, (12, 13), (7, 8))
    embeds, positions, mask = runtime.build_replay_inputs(state, site, condition)
    assert torch.equal(positions, state.positions[:, :, 11:13])
    visible = mask[0, 0] == 0
    assert visible[0, :14].nonzero().flatten().tolist() == root
    assert visible[:, 14:].tolist() == [[True, False], [True, True]]
    assert bool(torch.count_nonzero(embeds[:, 0]) == 0) == (condition in ("boundary_mask", "full_mask"))
    assert torch.equal(state.embeds, torch.ones_like(state.embeds))


@pytest.mark.parametrize("two_sites", [False, True])
def test_real_capture_matches_ordinary_oracle_and_existing_body_path(runtime, two_sites):
    assert runtime is not None, "capture/replay runtime is missing"
    # Catches activating capture in the reference and incorrect layer/position dispatch.
    from step_preference.sidecar import build_context_sidecar_plan, sidecar_dense_visibility
    from step_preference.spans import ReasoningStepSpan
    from transformers.models.qwen2_5_vl.stage2_shared_prefix import _branch_inputs
    model, sample = image_case(two_sites)
    identity = _sample_identity(sample)
    calls = []
    handle = model.model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(dict(kwargs)), with_kwargs=True)
    try:
        state = runtime.capture_clean(model, sample)
    finally:
        handle.remove()
    full = [call for call in calls if call["inputs_embeds"].shape[1] == sample.model_inputs["input_ids"].shape[1]]
    assert len(full) == 2
    assert not full[0].get("stage2_capture_kv", False)
    assert full[1]["stage2_capture_kv"] is True
    assert torch.equal(state.positions, sample.model_inputs["position_ids"])
    assert not state.embeds.requires_grad
    assert all(not tensor.requires_grad for pair in state.prefix_kv for tensor in pair)
    assert _sample_identity(sample) == identity
    for site in sample.sites:
        for condition in ("clean", "body_mask", "boundary_mask", "full_mask"):
            actual = production_logits(runtime, model, state, site, condition)
            expected = reference_logits(model, state, site, condition)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            rows = runtime.replay_site(model, state, site, condition)
            for row, target, token_id in zip(rows, site.target_positions, site.target_ids):
                logits = expected[0, target - 1 - site.end]
                assert row["argmax_id"] == int(logits.argmax())
                assert row["nll"] == pytest.approx(float(-logits.log_softmax(-1)[token_id]), abs=1e-5)
        span = ReasoningStepSpan(0, 5 if site.plan_index == 1 else 16,
                                 site.start, site.body_start, site.body_end,
                                 site.target_positions[0], site.target_positions[-1]+1, None)
        plan = build_context_sidecar_plan(clean_input_ids=sample.model_inputs["input_ids"],
            attention_mask=sample.model_inputs["attention_mask"], spans=(span,),
            visual_only=True, suppress_latent_boundaries=False)
        branch = plan.groups[0][0]
        embeds, positions = _branch_inputs(model, branch, sample.model_inputs["input_ids"], state.embeds, state.positions)
        visibility = sidecar_dense_visibility(plan.root_length, (branch,))
        mask = torch.zeros((1, 1, *visibility.shape)).masked_fill(~visibility, float("-inf"))
        with torch.inference_mode():
            hidden = model.model(inputs_embeds=embeds, position_ids=positions, attention_mask=mask,
                stage2_prefix_key_values=state.prefix_kv, use_cache=False, return_dict=True).last_hidden_state
            expected = model.lm_head(hidden).float()
        torch.testing.assert_close(production_logits(runtime, model, state, site, "body_mask"), expected, atol=1e-5, rtol=1e-5)
    rows, witness = runtime.analyze_sample(model, sample, repeat_clean=True)
    assert len(rows) == 4 * sum(len(site.target_ids) for site in sample.sites)
    assert [row["condition"] for row in rows] == [c for c in ("clean", "body_mask", "boundary_mask", "full_mask") for site in sample.sites for _ in site.target_ids]
    for name in ("normal_vs_captured", "normal_vs_replay", "captured_vs_replay", "repeat_clean"):
        assert witness[name]["argmax_mismatches"] == 0
        assert witness[name]["max_nll_abs_diff"] < 1e-5
    assert witness["finite"] is True


@pytest.mark.parametrize("condition,source", [
    ("body_mask", "body"), ("full_mask", "body"),
    ("boundary_mask", "start"), ("full_mask", "start"),
    *[(condition, "tail") for condition in ("clean", "body_mask", "boundary_mask", "full_mask")],
])
def test_blocked_captured_sources_cannot_change_logits(runtime, condition, source):
    assert runtime is not None, "capture/replay runtime is missing"
    # Decisive first nail: a leaked blocked key/value changes off logits.
    model, sample = image_case(True)
    state = runtime.capture_clean(model, sample)
    site = sample.sites[0]
    before = production_logits(runtime, model, state, site, condition)
    bank = tuple(tuple(t.clone() for t in pair) for pair in state.prefix_kv)
    region = {"body": slice(site.body_start, site.body_end),
              "start": slice(site.start, site.start+1), "tail": slice(site.end, None)}[source]
    for key, value in bank:
        key[:, :, region] += 37
        value[:, :, region] -= 53
    changed = replace(state, prefix_kv=bank)
    torch.testing.assert_close(production_logits(runtime, model, changed, site, condition), before, atol=0, rtol=0)
    if source == "body":
        visible_before = production_logits(runtime, model, state, site, "boundary_mask")
        visible_after = production_logits(runtime, model, changed, site, "boundary_mask")
        assert float((visible_before-visible_after).abs().max()) > 1e-3
    second_before = production_logits(runtime, model, state, sample.sites[1], "clean")
    runtime.replay_site(model, state, site, "full_mask")
    torch.testing.assert_close(production_logits(runtime, model, state, sample.sites[1], "clean"), second_before, atol=0, rtol=0)


def test_capture_removes_observers_after_decoder_error(runtime):
    assert runtime is not None, "capture/replay runtime is missing"
    model, sample = image_case()
    def fail(module, args, kwargs):
        raise RuntimeError("decoder witness failure")
    handle = model.model.layers[0].register_forward_pre_hook(fail, with_kwargs=True)
    before = (len(model.model._forward_pre_hooks), len(model.model._forward_hooks))
    try:
        with pytest.raises(RuntimeError, match="decoder witness failure"):
            runtime.capture_clean(model, sample)
    finally:
        handle.remove()
    assert (len(model.model._forward_pre_hooks), len(model.model._forward_hooks)) == before
    runtime.capture_clean(model, sample)


def test_unscored_suffix_rows_are_retained(runtime):
    assert runtime is not None, "capture/replay runtime is missing"
    model, sample = image_case()
    state = runtime.capture_clean(model, sample)
    site = replace(sample.sites[0], target_positions=(17,), target_ids=(14,))
    embeds, positions, _ = runtime.build_replay_inputs(state, site, "clean")
    assert embeds.shape[1] == 2
    assert torch.equal(positions, state.positions[:, :, 15:17])
    row = runtime.replay_site(model, state, site, "clean")[0]
    assert row["nll"] == pytest.approx(state.normal_scores[(site.plan_index, 17)][1], abs=1e-5)


def test_each_sample_rejects_nonfinite_or_unfaithful_clean(runtime, monkeypatch):
    assert runtime is not None, "capture/replay runtime is missing"
    model, sample = image_case()
    original = runtime.capture_clean
    def corrupted(*args, **kwargs):
        state = original(*args, **kwargs)
        scores = dict(state.captured_scores)
        key = next(iter(scores))
        scores[key] = (scores[key][0], scores[key][1] + .02)
        return replace(state, captured_scores=scores)
    monkeypatch.setattr(runtime, "capture_clean", corrupted)
    with pytest.raises(ValueError, match="fidelity"):
        runtime.analyze_sample(model, sample)
    monkeypatch.setattr(runtime, "capture_clean", original)
    with torch.no_grad():
        model.lm_head.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runtime.analyze_sample(model, sample)


@pytest.mark.parametrize("phase", ["ordinary_final", "separate_capture"])
def test_full_length_errors_leave_no_capture_observers(runtime, phase):
    model, sample = image_case()
    length = sample.model_inputs["input_ids"].shape[1]
    def fail(module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        captured = kwargs.get("stage2_capture_kv", False)
        if hidden.shape[1] == length and captured == (phase == "separate_capture"):
            raise RuntimeError("full-length injected failure")
    handle = model.model.layers[0].register_forward_pre_hook(fail, with_kwargs=True)
    before = (len(model.model._forward_pre_hooks), len(model.model._forward_hooks))
    try:
        with pytest.raises(RuntimeError, match="full-length injected failure"):
            runtime.capture_clean(model, sample)
    finally:
        handle.remove()
    assert (len(model.model._forward_pre_hooks), len(model.model._forward_hooks)) == before


def test_projected_chunks_bound_vocab_memory_and_ties_choose_first(runtime):
    # Catches projecting a long target in one allocation or nondeterministic ties.
    model, sample = image_case()
    ids = torch.cat((sample.model_inputs["input_ids"], torch.full((1, 260), 13)), dim=1)
    p = torch.arange(ids.shape[1])[None, None, :]
    inputs = dict(sample.model_inputs, input_ids=ids, attention_mask=torch.ones_like(ids),
                  position_ids=torch.cat((p, 2*p, 3*p)))
    site = replace(sample.sites[0], target_positions=tuple(range(16, ids.shape[1])),
                   target_ids=tuple(int(x) for x in ids[0, 16:]))
    sample = replace(sample, model_inputs=inputs, sites=(site,))
    with torch.no_grad():
        model.lm_head.weight.zero_()
    projected_rows = []
    handle = model.lm_head.register_forward_pre_hook(
        lambda module, args: projected_rows.append(args[0].numel()//args[0].shape[-1]))
    try:
        state = runtime.capture_clean(model, sample)
        rows = runtime.replay_site(model, state, site, "clean")
    finally:
        handle.remove()
    assert projected_rows == [1, 128, 128, 6, 128, 128, 6, 128, 128, 6]
    assert len(rows) == 262
    assert all(row["argmax_id"] == 0 for row in rows)
    assert all(row["nll"] == pytest.approx(float(torch.tensor(128.).log())) for row in rows)


@pytest.mark.parametrize("argmax_delta,nll_delta", [(1, 0), (0, .003)])
def test_fidelity_enforces_argmax_and_mean_thresholds(runtime, monkeypatch, argmax_delta, nll_delta):
    model, sample = image_case()
    original = runtime.capture_clean
    def corrupted(*args, **kwargs):
        state = original(*args, **kwargs)
        scores = dict(state.captured_scores)
        key = next(iter(scores))
        argmax, nll = scores[key]
        scores[key] = (argmax + argmax_delta, nll + nll_delta)
        return replace(state, captured_scores=scores)
    monkeypatch.setattr(runtime, "capture_clean", corrupted)
    with pytest.raises(ValueError, match="fidelity"):
        runtime.analyze_sample(model, sample)


def test_repeat_clean_rejects_argmax_change(runtime, monkeypatch):
    model, sample = image_case()
    original = runtime.replay_site
    clean_calls = 0
    def corrupted(*args, **kwargs):
        nonlocal clean_calls
        rows = original(*args, **kwargs)
        if rows[0]["condition"] == "clean":
            clean_calls += 1
            if clean_calls == 2:
                rows[0]["argmax_id"] += 1
        return rows
    monkeypatch.setattr(runtime, "replay_site", corrupted)
    with pytest.raises(ValueError, match="fidelity"):
        runtime.analyze_sample(model, sample, repeat_clean=True)


def test_capture_rejects_extra_full_length_call_and_cleans_hooks(runtime):
    model, sample = image_case()
    def duplicate(module, args, kwargs):
        with torch.inference_mode():
            model.model(inputs_embeds=model.model.embed_tokens(kwargs["input_ids"]),
                position_ids=kwargs["position_ids"], attention_mask=kwargs["attention_mask"],
                use_cache=False, return_dict=True, generate_mode=True)
    handle = model.register_forward_pre_hook(duplicate, with_kwargs=True)
    before = (len(model.model._forward_pre_hooks), len(model.model._forward_hooks))
    try:
        with pytest.raises(ValueError, match="exactly one"):
            runtime.capture_clean(model, sample)
    finally:
        handle.remove()
    assert (len(model.model._forward_pre_hooks), len(model.model._forward_hooks)) == before


def test_capture_calls_loaded_dora_wrapper_and_preserves_adapter_outputs(runtime):
    from peft import LoraConfig, get_peft_model
    from stage2.peft import resolve_stage2_qwen_model
    model, sample = image_case()
    wrapped = get_peft_model(model, LoraConfig(r=2, lora_alpha=4,
        target_modules=["model.layers.0.self_attn.q_proj"], use_dora=True))
    with torch.no_grad():
        for name, parameter in wrapped.named_parameters():
            if "lora_B" in name:
                parameter.fill_(.2)
    calls = []
    handle = wrapped.register_forward_pre_hook(lambda module, args: calls.append(1))
    try:
        rows, witness = runtime.analyze_sample(wrapped, sample, repeat_clean=True)
    finally:
        handle.remove()
    assert len(calls) == 2
    assert len(rows) == 8
    assert witness["normal_vs_replay"]["argmax_mismatches"] == 0
    assert resolve_stage2_qwen_model(wrapped) is model
