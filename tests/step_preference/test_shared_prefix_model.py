"""Independent full-prefix oracles for differentiable shared sidecars."""

import copy
from dataclasses import replace

import pytest
import torch

from step_preference.equivalence import _tiny_model
from step_preference.negatives import StepNegative
from step_preference.sidecar import build_sidecar_plan
from step_preference.spans import ReasoningStepSpan


def fixture(backend="eager", dropout=0.):
    torch.manual_seed(731)
    config = copy.deepcopy(_tiny_model().config)
    config.num_hidden_layers = 2
    config._attn_implementation = backend
    config.attention_dropout = dropout
    model = type(_tiny_model())(config).float().train()
    model.visual.requires_grad_(False)
    ids = torch.tensor([[11, 91, 90, 90, 92, 12, 13, 91, 90, 90, 92, 14, 15]])
    spans = (ReasoningStepSpan(0, 0, 1, 2, 4, 5, 7, None),
             ReasoningStepSpan(1, 5, 7, 8, 10, 11, 13, None))
    steps = []
    for span in spans:
        start = span.target_text_start
        reject_ids = torch.cat((ids[:, :start], torch.tensor([[41, 42, 43, 44]])), 1)
        steps.append(dict(span=span, negative=StepNegative(
            sample_id="shared", step_id=span.step_id, chosen_text="chosen",
            rejected_visual_text="reject", rejected_textual_text=None,
            valid_visual_negative=True, valid_textual_negative=False,
            reference_chosen_score=-1., reference_visual_rejected_score=-2.,
            reference_textual_rejected_score=None), visual_rejected=dict(
                model_inputs=dict(input_ids=reject_ids, attention_mask=torch.ones_like(reject_ids)),
                completion_token_positions=tuple(range(start, start+4)),
                completion_token_ids=(41, 42, 43, 44)), textual_rejected=None))
    plan = build_sidecar_plan(clean_input_ids=ids, attention_mask=torch.ones_like(ids), spans=spans, steps=steps)
    return model, ids, spans, steps, plan


def score(logits, targets):
    lp = logits.float().log_softmax(-1)
    ids = torch.as_tensor(targets, device=lp.device)
    return lp.gather(-1, ids[:, None]).mean(), -(lp.exp()*lp).sum(-1)


def oracle(model, ids, spans, steps):
    """Literal causal+rectangle masks; no planner visibility or branch metadata."""
    results = {}
    selected_logits = {}
    for span, step in zip(spans, steps):
        a, b = span.target_text_start, span.target_text_end
        for modality, blocked in (("visual", (span.latent_start, span.latent_end)),
                                  ("textual", (span.prev_text_start, span.prev_text_end))):
            truncated = ids[:, :b]
            visible = torch.ones((b, b), dtype=torch.bool).tril()
            visible[a-1:b-1, blocked[0]:blocked[1]] = False
            mask = torch.zeros((b, b)).masked_fill(~visible, torch.finfo(torch.float32).min)[None, None]
            out = model(input_ids=truncated, attention_mask=torch.ones_like(truncated),
                        stage2_final_attention_mask=mask, use_cache=False)
            results[f"step-{span.step_id}-{modality}-off"] = score(out.logits[0, a-1:b-1], ids[0,a:b])
            selected_logits[f"step-{span.step_id}-{modality}-off"] = out.logits[0, a-1:b-1]
        rejected = step["visual_rejected"]
        out = model(**rejected["model_inputs"], use_cache=False)
        p = torch.tensor(rejected["completion_token_positions"])-1
        results[f"step-{span.step_id}-visual-rejected"] = score(out.logits[0,p], rejected["completion_token_ids"])
        selected_logits[f"step-{span.step_id}-visual-rejected"] = out.logits[0,p]
    return results, selected_logits


def objective(out, results):
    return out.logits.square().mean() + sum(lp + ent.mean()*.13 for lp, ent in results.values())


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_dense_shared_matches_independent_two_layer_oracle_all_gradients(checkpoint, backend):
    """Catches detached prefix/latent edges, wrong boundary mask, and rejected conditioning."""
    model, ids, spans, steps, plan = fixture(backend)
    reference = copy.deepcopy(model)
    projected_logits = []
    hook = model.lm_head.register_forward_hook(lambda _m, _a, out: projected_logits.append(out))
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                stage2_shared_prefix_inputs=dict(plan=plan, backend="dense", query_budget=20, checkpoint=checkpoint))
    hook.remove()
    expected, expected_logits = oracle(reference, ids, spans, steps)
    # The first projection is clean; each subsequent projection is one sidecar group.
    assert len(projected_logits) == 1 + len(plan.groups)
    torch.testing.assert_close(projected_logits[0], out.logits, atol=0, rtol=0)
    logit_errors = []
    for group, projected in zip(plan.groups, projected_logits[1:]):
        offset = 0
        for branch in group:
            length = len(branch.target_token_ids)
            got = projected[0, offset:offset+length]
            want = expected_logits[branch.name]
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
            logit_errors.append(float((got.detach()-want.detach()).abs().max()))
            offset += length
    actual = {r["name"]: (r["mean_logp"], r["entropies"]) for r in out.stage2_sidecar_results}
    assert actual.keys() == expected.keys()
    score_errors = []
    for name in expected:
        for got, want in zip(actual[name], expected[name]):
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
            score_errors.append(float((got.detach()-want.detach()).abs().max()))
    clean = reference(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    torch.testing.assert_close(out.logits, clean.logits, atol=1e-5, rtol=1e-4)
    objective(out, actual).backward()
    objective(clean, expected).backward()
    errors = []
    near_zero = []
    near_zero_elements = 0
    for (name, got), (_, want) in zip(model.named_parameters(), reference.named_parameters()):
        if not got.requires_grad:
            continue
        assert got.grad is not None and want.grad is not None, name
        torch.testing.assert_close(got.grad, want.grad, atol=1e-5, rtol=1e-4, msg=name)
        errors.append(float((got.grad-want.grad).abs().max()))
        if float(want.grad.norm()) < 1e-8:
            near_zero.append(name)
        near_zero_elements += int((want.grad.abs() < 1e-8).sum())
    assert model.model.embed_tokens.weight.grad[11].norm() > 0
    print(dict(backend=backend, checkpoint=checkpoint, parameters=len(errors),
               max_score_abs=max(score_errors), max_gradient_abs=max(errors),
               max_logit_abs=max(logit_errors),
               near_zero=near_zero, near_zero_elements=near_zero_elements))


def test_noop_replay_boundary_and_branch_isolation():
    """Catches using clean boundary states, cross-branch attention, or mutating clean K/V."""
    model, ids, _, _, plan = fixture()
    base = model(input_ids=ids, use_cache=False)
    branches = tuple(replace(b, blocked_start=0, blocked_end=0) for b in plan.groups[0] if b.kind == "visual_off")
    noop = replace(plan, groups=(branches, ()))
    out = model(input_ids=ids, use_cache=False, stage2_shared_prefix_inputs=dict(plan=noop, checkpoint=True, query_budget=20))
    for branch, result in zip(branches, out.stage2_sidecar_results):
        lp, ent = score(base.logits[0, list(branch.logical_positions)], branch.target_token_ids)
        torch.testing.assert_close(result["mean_logp"], lp, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(result["entropies"], ent, atol=1e-5, rtol=1e-4)
    masked = model(input_ids=ids, use_cache=False, stage2_shared_prefix_inputs=dict(plan=plan, checkpoint=False, query_budget=20))
    assert not torch.equal(masked.stage2_sidecar_results[0]["entropies"][0], out.stage2_sidecar_results[0]["entropies"][0])
    torch.testing.assert_close(masked.logits, base.logits, atol=0, rtol=0)
    solo = replace(plan, groups=((plan.groups[0][-2],), ()))
    independent = model(input_ids=ids, use_cache=False, stage2_shared_prefix_inputs=dict(plan=solo, checkpoint=False, query_budget=1))
    torch.testing.assert_close(independent.stage2_sidecar_results[0]["mean_logp"], masked.stage2_sidecar_results[2]["mean_logp"], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("layer_checkpoint", [False, True])
def test_checkpoint_preserves_active_dropout_rng_and_prefix_latent_gradients(layer_checkpoint):
    """Catches whole-chunk RNG drift, detached K/V, and nested suffix layer replay."""
    model, ids, _, _, plan = fixture("sdpa", dropout=.17)
    if layer_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference = copy.deepcopy(model)
    assert all(layer.self_attn.attention_dropout == .17 for layer in model.model.layers)
    torch.manual_seed(912)
    expected = reference(input_ids=ids, use_cache=False, return_stage2_latent_states=True,
                         retain_stage2_latent_grads=True,
                         stage2_shared_prefix_inputs=dict(plan=plan, checkpoint=False, query_budget=3))
    expected_rng = torch.get_rng_state().clone()
    # Hooks observe actual capture output; they do not create or inject K/V.
    captures = []
    snapshots = []
    projection_rows = []
    mlp_rows = []
    head_rows = []
    def witness(_module, _args, kwargs, output):
        if kwargs.get("stage2_capture_kv"):
            captures.append(output.stage2_key_values)
            snapshots.extend(t.clone() for pair in output.stage2_key_values for t in pair)
    handle = model.model.register_forward_hook(witness, with_kwargs=True)
    ph = model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(lambda _m, args: projection_rows.append(args[0].shape[1]))
    mh = model.model.layers[0].mlp.register_forward_pre_hook(lambda _m, args: mlp_rows.append(args[0].shape[1]))
    hh = model.lm_head.register_forward_pre_hook(lambda _m, args: head_rows.append(args[0].shape[-2]))
    torch.manual_seed(912)
    out = model(input_ids=ids, use_cache=False, return_stage2_latent_states=True,
                retain_stage2_latent_grads=True,
                stage2_shared_prefix_inputs=dict(plan=plan, checkpoint=True, query_budget=3))
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
    assert len(captures) == 1
    for pair in captures[0]:
        for tensor in pair:
            tensor.retain_grad()
            assert tensor.requires_grad and tensor.shape == (1, 2, 13, 16)
    assert projection_rows == [2, 3, 8, 9, 13, 2, 4, 2, 4, 2, 2]
    assert mlp_rows == projection_rows
    initial_head_calls = len(head_rows)
    for got, want in zip(out.stage2_sidecar_results, expected.stage2_sidecar_results):
        torch.testing.assert_close(got["mean_logp"], want["mean_logp"], atol=0, rtol=0)
        torch.testing.assert_close(got["entropies"], want["entropies"], atol=0, rtol=0)
    # A branch-only objective must reach root-only input11 and generated latent slots.
    sum(r["mean_logp"] + .13*r["entropies"].mean() for r in out.stage2_sidecar_results).backward()
    after_backward_rng = torch.get_rng_state().clone()
    sum(r["mean_logp"] + .13*r["entropies"].mean() for r in expected.stage2_sidecar_results).backward()
    torch.testing.assert_close(after_backward_rng, expected_rng, atol=0, rtol=0)
    assert len(head_rows) == initial_head_calls + 6
    assert model.model.embed_tokens.weight.grad[11].norm() > 0
    assert all(latent.grad is not None and latent.grad.norm() > 0 for latent in out.stage2_latent_states)
    assert any(t.grad is not None and t.grad.norm() > 0 for pair in captures[0] for t in pair)
    for tensor, before in zip((t for pair in captures[0] for t in pair), snapshots):
        torch.testing.assert_close(tensor, before, atol=0, rtol=0)
    for (name, got), (_, want) in zip(model.named_parameters(), reference.named_parameters()):
        if got.requires_grad:
            assert got.grad is not None and want.grad is not None, name
            torch.testing.assert_close(got.grad, want.grad, atol=1e-5, rtol=1e-4, msg=name)
    handle.remove()
    ph.remove()
    hh.remove()
    mh.remove()
    torch.manual_seed(913)
    changed = model(input_ids=ids, use_cache=False, stage2_shared_prefix_inputs=dict(plan=plan, checkpoint=False, query_budget=3))
    assert not torch.equal(out.logits, changed.logits)


def test_right_padding_and_empty_sidecars_preserve_clean_route():
    """Catches padded root K/V visibility and unwanted sidecar work on no-step inputs."""
    model, ids, _, _, plan = fixture("sdpa")
    padded = torch.cat((ids, torch.zeros((1, 3), dtype=torch.long)), dim=1)
    mask = torch.cat((torch.ones_like(ids), torch.zeros((1, 3), dtype=torch.long)), dim=1)
    out = model(input_ids=padded, attention_mask=mask, use_cache=False,
                stage2_shared_prefix_inputs=dict(plan=plan))
    plain = model(input_ids=ids, use_cache=False, stage2_shared_prefix_inputs=dict(plan=plan))
    for got, want in zip(out.stage2_sidecar_results, plain.stage2_sidecar_results):
        torch.testing.assert_close(got["mean_logp"], want["mean_logp"], atol=1e-5, rtol=1e-4)
    empty = replace(plan, root_selected_positions=(), groups=((), ()))
    captured = []
    handle = model.model.register_forward_pre_hook(lambda _m, _a, kw: captured.append(kw.get("stage2_capture_kv", False)), with_kwargs=True)
    out = model(input_ids=ids, use_cache=False, stage2_shared_prefix_inputs=dict(plan=empty))
    handle.remove()
    assert not any(captured)
    clean = model(input_ids=ids, use_cache=False)
    torch.testing.assert_close(out.logits, clean.logits, atol=0, rtol=0)
    assert out.stage2_sidecar_results == ()
    assert out.stage2_sidecar_stats["chunks"] == 0
    assert out.stage2_sidecar_stats["logical_groups"] == 1
    assert clean.stage2_sidecar_results is None


def test_rejected_rope_must_match_ancestors_and_shared_options_are_exclusive():
    """Catches sharing positional mismatches and accidentally accepting legacy combinations."""
    model, ids, _, _, plan = fixture()
    with pytest.raises(ValueError, match="matching mRoPE"):
        model(input_ids=ids, position_ids=torch.arange(ids.shape[1])[None] + 1, use_cache=False,
              stage2_shared_prefix_inputs=dict(plan=plan))
    with pytest.raises(ValueError, match="cannot be combined"):
        model(input_ids=ids, use_cache=False, stage2_packed_inputs={}, stage2_shared_prefix_inputs=dict(plan=plan))
    model.config.stage = "stage1"
    with pytest.raises(ValueError, match="only in Stage 2"):
        model(input_ids=ids, use_cache=False, stage2_shared_prefix_inputs=dict(plan=plan))


def test_dora_dropout_checkpoint_matches_same_shared_graph():
    """Catches bypassed adapter projection hooks and lost dropout RNG during replay."""
    from peft import LoraConfig, get_peft_model

    model, ids, _, _, plan = fixture("sdpa")
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], use_dora=True,
        task_type="CAUSAL_LM"))
    # A trained adapter has nonzero B; zero initialization would hide dropout.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.05)
    reference = copy.deepcopy(model)
    dropout_outputs = []
    adapter = model.base_model.model.model.layers[0].self_attn.q_proj
    hook = adapter.lora_dropout["default"].register_forward_hook(
        lambda _m, args, output: dropout_outputs.append(bool((output != args[0]).any())))
    assert adapter.lora_dropout["default"].p == .05
    outputs = []
    for network, use_checkpoint in ((reference, False), (model, True)):
        torch.manual_seed(185)
        out = network(input_ids=ids, use_cache=False,
                      stage2_shared_prefix_inputs=dict(plan=plan, checkpoint=use_checkpoint, query_budget=3))
        reduced = {r["name"]: (r["mean_logp"], r["entropies"]) for r in out.stage2_sidecar_results}
        loss = objective(out, reduced)
        loss.backward()
        outputs.append(loss.detach())
    hook.remove()
    assert dropout_outputs and any(dropout_outputs)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    compared = 0
    for (name, got), (_, want) in zip(model.named_parameters(), reference.named_parameters()):
        if got.requires_grad:
            assert got.grad is not None and want.grad is not None, name
            assert torch.isfinite(got.grad).all()
            torch.testing.assert_close(got.grad, want.grad, atol=1e-5, rtol=1e-4, msg=name)
            compared += 1
    assert compared == 24


def test_shared_clean_loss_only_and_selected_logits_keep_existing_contract():
    """Catches sidecar reduction replacing clean CE or changing selected-row order."""
    model, ids, _, _, plan = fixture()
    labels = ids.clone()
    labels[:, :5] = -100
    positions = torch.tensor([10, 4, 5])
    clean = model(input_ids=ids, labels=labels, use_cache=False,
                  stage2_loss_only=True, stage2_logit_positions=positions)
    shared = model(input_ids=ids, labels=labels, use_cache=False,
                   stage2_loss_only=True, stage2_logit_positions=positions,
                   stage2_shared_prefix_inputs=dict(plan=plan))
    assert shared.logits is None
    torch.testing.assert_close(shared.loss, clean.loss, atol=0, rtol=0)
    torch.testing.assert_close(shared.stage2_selected_logits, clean.stage2_selected_logits, atol=0, rtol=0)
