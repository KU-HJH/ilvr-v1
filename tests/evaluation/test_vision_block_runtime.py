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


@pytest.mark.parametrize('condition', ['clean', 'body_mask', 'boundary_mask', 'full_mask'])
def test_full_length_v2_geometry(runtime, condition):
    # Catches shortened query scope, rebased axes and mutation of the clean inputs.
    p = torch.arange(14)[None, None, :]
    state = runtime.CleanState(torch.ones(1, 14, 8), torch.cat((p, 2*p, 3*p)), (), {}, {})
    site = Site(0, 1, 2, 3, 11, 11, (12, 13), (7, 8))
    embeds, positions, mask = runtime.build_replay_inputs(state, site, condition)
    assert embeds.shape == state.embeds.shape
    assert torch.equal(positions, state.positions)
    blocked = set(range(3, 11)) if condition in ('body_mask', 'full_mask') else set()
    if condition in ('boundary_mask', 'full_mask'):
        blocked.add(2)
    expected = torch.tensor([[k <= q and (q < 11 or k not in blocked)
                              for k in range(14)] for q in range(14)])
    assert torch.equal(torch.isfinite(mask[0, 0]), expected)
    expected_embeds = state.embeds.clone()
    if condition in ('boundary_mask', 'full_mask'):
        expected_embeds[:, 11] = 0
    assert torch.equal(embeds, expected_embeds)
    assert torch.equal(state.embeds, torch.ones_like(state.embeds))


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
    """Observe logits from the actual public replay call."""
    outputs = []
    handle = model.model.register_forward_hook(
        lambda module, args, output: outputs.append(model.lm_head(output.last_hidden_state).float()))
    try:
        runtime.replay_site(model, state, site, condition)
    finally:
        handle.remove()
    assert len(outputs) == 1
    return outputs[0]


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
            torch.testing.assert_close(actual[:, site.end:site.target_positions[-1]], expected,
                                       atol=1e-5, rtol=1e-5)
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
        torch.testing.assert_close(
            production_logits(runtime, model, state, site, "body_mask")[:, site.end:site.target_positions[-1]],
            expected, atol=1e-5, rtol=1e-5)
    rows, witness = runtime.analyze_sample(model, sample, repeat_clean=True)
    assert len(rows) == 4 * sum(len(site.target_ids) for site in sample.sites)
    assert [row["condition"] for row in rows] == [c for c in ("clean", "body_mask", "boundary_mask", "full_mask") for site in sample.sites for _ in site.target_ids]
    for name in ("normal_vs_captured", "normal_vs_replay", "captured_vs_replay", "repeat_clean"):
        assert witness[name]["argmax_mismatches"] == 0
        assert witness[name]["max_nll_abs_diff"] < 1e-5
    assert witness["finite"] is True
    assert witness["execution_strategy"] == "full_length_v2"
    assert witness["numerical_policy"] == MATH_POLICY
    expected = [{"plan_index": site.plan_index, "condition": condition,
                 "prefix_length": site.end, "layers": 2, "kv_comparisons": 4, "exact": True}
                for condition in ("clean", "body_mask", "boundary_mask", "full_mask")
                for site in sample.sites]
    assert witness["prefix_identity"] == expected
    assert witness["repeat_prefix_identity"] == [row for row in expected if row["condition"] == "clean"]


@pytest.mark.parametrize("two_sites", [False, True])
def test_actual_full_decoder_and_attention_edges(runtime, two_sites):
    # Catches a correct builder wired to a shortened/cached or wrongly masked decoder.
    model, sample = image_case(two_sites)
    state = runtime.capture_clean(model, sample)
    length = state.embeds.shape[1]
    for site in sample.sites:
        for condition in ("clean", "body_mask", "boundary_mask", "full_mask"):
            decoder_calls, attention_calls = [], []
            handles = [model.model.register_forward_pre_hook(
                lambda module, args, kwargs: decoder_calls.append(dict(kwargs)), with_kwargs=True)]
            handles += [layer.self_attn.register_forward_pre_hook(
                lambda module, args, kwargs: attention_calls.append(dict(kwargs)), with_kwargs=True)
                for layer in model.model.layers]
            try:
                runtime.replay_site(model, state, site, condition)
            finally:
                for handle in handles:
                    handle.remove()
            assert len(decoder_calls) == 1
            call = decoder_calls[0]
            assert call["inputs_embeds"].shape == state.embeds.shape
            assert torch.equal(call["position_ids"], state.positions)
            assert call["stage2_capture_kv"] is True
            assert call.get("stage2_prefix_key_values") is None
            assert len(attention_calls) == 2
            for attention in attention_calls:
                assert attention["hidden_states"].shape == state.embeds.shape
                assert torch.equal(attention["position_ids"], state.positions)
                assert attention.get("stage2_prefix_key_value") is None
                mask = attention["attention_mask"][0, 0]
                assert mask.shape == (length, length)
                for query in range(length):
                    for key in range(length):
                        visible = key <= query
                        if query >= site.end:
                            if condition in ("body_mask", "full_mask") and site.body_start <= key < site.body_end:
                                visible = False
                            if condition in ("boundary_mask", "full_mask") and key == site.start:
                                visible = False
                        assert bool(torch.isfinite(mask[query, key])) == visible


@pytest.mark.parametrize("condition,source", [
    ("body_mask", "body"), ("full_mask", "body"),
    ("boundary_mask", "start"), ("full_mask", "start"),
    *[(condition, "future") for condition in ("clean", "body_mask", "boundary_mask", "full_mask")],
    ("boundary_mask", "body"),
])
def test_live_last_layer_values_obey_direct_edges(runtime, condition, source):
    # Catches leaked direct edges and a disconnected visible-Body positive control.
    # Inspect direct decoder logits: a live K/V intervention intentionally violates
    # the separate prefix-fidelity guard, so do not mistake rejection for no influence.
    model, sample = image_case(True)
    state = runtime.capture_clean(model, sample)
    site = sample.sites[0]
    embeds, positions, mask = runtime.build_replay_inputs(state, site, condition)

    @torch.inference_mode()
    def logits():
        output = model.model(inputs_embeds=embeds, position_ids=positions,
            attention_mask=mask, stage2_capture_kv=True,
            generate_mode=True, use_cache=False, return_dict=True)
        return model.lm_head(output.last_hidden_state[:, [p-1 for p in site.target_positions]]).float()

    with runtime.numerical_policy():
        before = logits()
        region = {"body": slice(site.body_start, site.body_end),
                  "start": slice(site.start, site.start+1),
                  "future": slice(site.target_positions[-1], None)}[source]
        calls = []
        def intervene(module, args, output):
            calls.append(output.shape)
            changed = output.clone()
            changed[:, region] += torch.linspace(-53, 37, output.shape[-1])
            return changed
        handle = model.model.layers[-1].self_attn.v_proj.register_forward_hook(intervene)
        try:
            after = logits()
        finally:
            handle.remove()
    assert calls == [torch.Size((1, state.embeds.shape[1], model.model.layers[-1].self_attn.v_proj.out_features))]
    if condition == "boundary_mask" and source == "body":
        targets = torch.tensor(site.target_ids)[None, :, None]
        before_nll = -before.log_softmax(-1).gather(-1, targets)
        after_nll = -after.log_softmax(-1).gather(-1, targets)
        assert float((before_nll-after_nll).abs().max()) > 1e-3
    else:
        torch.testing.assert_close(after, before, atol=0, rtol=0)


@pytest.mark.parametrize("condition", ["clean", "body_mask", "boundary_mask", "full_mask"])
def test_corrupt_clean_prefix_is_rejected_even_without_witness(runtime, condition):
    # This tests reference checking only, not causal influence from cached inputs.
    model, sample = image_case()
    state = runtime.capture_clean(model, sample)
    bank = tuple(tuple(t.clone() for t in pair) for pair in state.prefix_kv)
    bank[0][0][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="full-length prefix K/V changed"):
        runtime.replay_site(model, replace(state, prefix_kv=bank), sample.sites[0], condition)


def test_site_order_and_witness_coverage_preserve_shared_clean_state(runtime):
    # Catches state mutation, site contamination and records with wrong coverage.
    model, sample = image_case(True)
    state = runtime.capture_clean(model, sample)
    tensors = (state.embeds, state.positions, *(t for pair in state.prefix_kv for t in pair))
    saved = tuple(t.clone() for t in tensors)
    def passes(sites):
        witness, rows = [], {}
        for site in sites:
            for condition in ("clean", "body_mask", "boundary_mask", "full_mask"):
                rows[(site.plan_index, condition)] = runtime.replay_site(
                    model, state, site, condition, prefix_witness=witness)
        return rows, witness
    forward, witness = passes(sample.sites)
    reverse, reverse_witness = passes(reversed(sample.sites))
    assert forward == reverse
    expected = [{"plan_index": site.plan_index, "condition": condition,
                 "prefix_length": site.end, "layers": 2, "kv_comparisons": 4, "exact": True}
                for site in sample.sites for condition in ("clean", "body_mask", "boundary_mask", "full_mask")]
    assert witness == expected
    assert sorted(reverse_witness, key=lambda item: (item["plan_index"], item["condition"])) == sorted(
        expected, key=lambda item: (item["plan_index"], item["condition"]))
    assert all(torch.equal(t, original) for t, original in zip(tensors, saved))


def active_numerical_flags():
    return dict(sdpa_math=torch.backends.cuda.math_sdp_enabled(),
                sdpa_flash=torch.backends.cuda.flash_sdp_enabled(),
                sdpa_mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                sdpa_cudnn=torch.backends.cuda.cudnn_sdp_enabled(),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32)


MATH_POLICY = dict(sdpa_math=True, sdpa_flash=False, sdpa_mem_efficient=False,
                   sdpa_cudnn=False, tf32_matmul=False, tf32_cudnn=False)


@pytest.mark.parametrize("entrypoint", ["capture", "replay", "analyze"])
@pytest.mark.parametrize("fail", [False, True])
def test_public_calls_apply_and_restore_numerical_policy(runtime, entrypoint, fail):
    # Catches missing policy on ordinary rollout, direct replay, scoring or exceptions.
    from torch.nn.attention import sdpa_kernel, SDPBackend
    model, sample = image_case()
    state = runtime.capture_clean(model, sample)
    flags, lengths, scored = [], [], []
    def observe(module, args, kwargs):
        flags.append(active_numerical_flags())
        lengths.append(kwargs["inputs_embeds"].shape[1])
        if fail:
            raise RuntimeError("policy injected failure")
    handle = model.model.register_forward_pre_hook(observe, with_kwargs=True)
    head = model.lm_head.register_forward_pre_hook(lambda module, args: scored.append(active_numerical_flags()))
    saved = active_numerical_flags()
    try:
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            ambient = active_numerical_flags()
            def invoke():
                if entrypoint == "capture":
                    return runtime.capture_clean(model, sample)
                if entrypoint == "replay":
                    return runtime.replay_site(model, state, sample.sites[0], "clean")
                return runtime.analyze_sample(model, sample, repeat_clean=True)
            if fail:
                with pytest.raises(RuntimeError, match="policy injected failure"):
                    invoke()
            else:
                invoke()
                assert scored and all(value == MATH_POLICY for value in scored)
                if entrypoint != "replay":
                    assert any(length < state.embeds.shape[1] for length in lengths)
            assert flags and all(value == MATH_POLICY for value in flags)
            assert active_numerical_flags() == ambient
    finally:
        handle.remove()
        head.remove()
        torch.backends.cuda.matmul.allow_tf32 = saved["tf32_matmul"]
        torch.backends.cudnn.allow_tf32 = saved["tf32_cudnn"]
    assert active_numerical_flags() == saved


def test_numerical_policy_nesting_restores_outer_context_on_exception(runtime):
    saved = active_numerical_flags()
    with runtime.numerical_policy():
        assert active_numerical_flags() == MATH_POLICY
        assert runtime.numerical_policy_identity() == MATH_POLICY
        with pytest.raises(RuntimeError, match="nested"):
            with runtime.numerical_policy():
                assert active_numerical_flags() == MATH_POLICY
                raise RuntimeError("nested")
        assert active_numerical_flags() == MATH_POLICY
    assert active_numerical_flags() == saved


@pytest.mark.parametrize("defect", ["layers", "pair", "rank", "sequence", "heads", "head_dim", "dtype", "nonfinite"])
def test_malformed_prefix_bank_fails_closed(runtime, defect):
    # Catches broadcasted/truncated comparisons that could certify an invalid bank.
    model, sample = image_case()
    state = runtime.capture_clean(model, sample)
    bank = [list(pair) for pair in state.prefix_kv]
    if defect == "layers":
        bank.pop()
    elif defect == "pair":
        bank[0].pop()
    elif defect == "rank":
        bank[0][0] = bank[0][0][0]
    elif defect == "sequence":
        bank[0][0] = bank[0][0][:, :, :-1]
    elif defect == "heads":
        bank[0][0] = bank[0][0][:, :0]
    elif defect == "head_dim":
        bank[0][0] = bank[0][0][..., :-1]
    elif defect == "dtype":
        bank[0][0] = bank[0][0].double()
    else:
        bank[0][0] = bank[0][0].clone()
        bank[0][0][0, 0, 0, 0] = float("nan")
    witness = []
    with pytest.raises(ValueError, match="K/V"):
        runtime.replay_site(model, replace(state, prefix_kv=tuple(tuple(p) for p in bank)),
                            sample.sites[0], "clean", prefix_witness=witness)
    assert witness == []


def test_capture_rejects_config_layer_count_mismatch(runtime):
    model, sample = image_case()
    model.config.num_hidden_layers += 1
    with pytest.raises(ValueError, match="layer count"):
        runtime.capture_clean(model, sample)


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
    assert embeds.shape == state.embeds.shape
    assert torch.equal(positions, state.positions)
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
