"""Native full-prefix witnesses for same-image evidence replacement.

Each literal replay below deliberately bypasses the scorer: changing the
replacement block, predictor row, or native attention call makes it fail.
"""

import copy
import os

import pytest
import torch

from step_preference.equivalence import _tiny_model
from step_preference.replacement import build_replacement_plan
from replacement_helpers import three_step_planner_case
from step_preference.sidecar import build_context_sidecar_plan
from step_preference.spans import ReasoningStepSpan


def _case():
    torch.manual_seed(907)
    config = copy.deepcopy(_tiny_model().config)
    config.num_hidden_layers = 2
    config._attn_implementation = "sdpa"
    model = type(_tiny_model())(config).float().train()
    model.visual.requires_grad_(False)
    ids, sidecar_plan = three_step_planner_case()
    positions = torch.arange(ids.shape[1])[None, None].expand(3, 1, -1).clone()
    base = model.model.embed_tokens(ids).detach().requires_grad_()
    clean = (base.detach() + torch.randn_like(base) * 0.1).requires_grad_()
    options = dict(
        plan=sidecar_plan,
        prefix_mode="evidence_replacement",
        backend="dense",
        checkpoint=False,
        sample_id="replacement-runtime",
        replacement_seed=42,
        replacement_version="ilvr-evidence-replacement-v1",
    )
    return model, ids, positions, base, clean, options


def _literal_score(model, ids, positions, base, clean, branch):
    """Independent native replay with one detached, hand-assembled replacement."""
    prefix = clean[:, : branch.prefix_length].clone()
    source = clean if branch.source_kind == "donor" else base
    indexes = torch.tensor(branch.source_positions, device=prefix.device)
    prefix[:, branch.latent_start : branch.latent_end] = source.index_select(1, indexes).detach()
    hidden = model.model(
        inputs_embeds=prefix,
        position_ids=positions[:, :, : branch.prefix_length],
        attention_mask=torch.ones((1, branch.prefix_length), dtype=torch.long),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    rows = torch.tensor(branch.predictor_positions, device=prefix.device)
    target = torch.tensor(branch.target_token_ids, device=prefix.device)[None]
    expected = model.lm_head(hidden.index_select(1, rows)).float().log_softmax(-1)
    return expected.gather(-1, target.unsqueeze(-1)).mean()


def test_native_replacement_matches_literal_prefix_and_detaches_copied_block():
    """Catches replaying a stale KV, wrong predictor row, or live replacement copy."""
    from transformers.models.qwen2_5_vl.stage2_evidence_replacement import (
        score_evidence_replacement,
    )

    model, ids, positions, base, clean, options = _case()
    plan = build_replacement_plan(
        input_ids=ids, sidecar_plan=options["plan"], image_token_id=80,
        image_count=1, latent_size=2, sample_id=options["sample_id"],
        seed=options["replacement_seed"],
    )
    results, stats = score_evidence_replacement(
        model, options, ids, base, clean, positions, image_grid_thw=None
    )
    actual = {(result["step_id"], result["kind"]): result for result in results}
    assert set(actual) == {
        (step, kind) for step in range(3)
        for kind in ("visual_original", "visual_replacement")
    }
    branch = plan.branches[1]
    torch.testing.assert_close(
        actual[(1, "visual_replacement")]["mean_logp"],
        _literal_score(model, ids, positions, base, clean, branch),
        atol=1e-5,
        rtol=1e-4,
    )
    assert actual[(1, "visual_replacement")]["source_kind"] == "donor"
    assert actual[(1, "visual_replacement")]["donor_step_id"] == branch.donor_step_id
    assert stats["replacement_donor_count"] == 2
    assert stats["replacement_image_count"] == 1

    replacement = actual[(1, "visual_replacement")]["mean_logp"]
    replacement_grad = torch.autograd.grad(replacement, clean, retain_graph=True)[0]
    assert replacement_grad[:, branch.latent_start : branch.latent_end].count_nonzero() == 0
    donor = plan.branches[branch.donor_step_id]
    assert replacement_grad[:, donor.latent_start : donor.latent_end].count_nonzero() > 0
    total = sum(result["mean_logp"] for result in results)
    total.backward()
    assert model.lm_head.weight.grad is not None and model.lm_head.weight.grad.count_nonzero() > 0
    assert clean.grad is not None and clean.grad.count_nonzero() > 0


def test_replacement_ignores_current_latent_and_responds_to_its_selected_source():
    """Catches accidentally retaining the current latent or selecting the wrong source."""
    from transformers.models.qwen2_5_vl.stage2_evidence_replacement import (
        score_evidence_replacement,
    )

    model, ids, positions, base, clean, options = _case()
    plan = build_replacement_plan(
        input_ids=ids, sidecar_plan=options["plan"], image_token_id=80,
        image_count=1, latent_size=2, sample_id=options["sample_id"],
        seed=options["replacement_seed"],
    )
    before, _ = score_evidence_replacement(model, options, ids, base, clean, positions, None)
    current = plan.branches[1]
    changed_clean = clean.detach().clone()
    changed_clean[:, current.latent_start : current.latent_end] += 4
    after, _ = score_evidence_replacement(model, options, ids, base, changed_clean, positions, None)
    before_map = {(r["step_id"], r["kind"]): r["mean_logp"] for r in before}
    after_map = {(r["step_id"], r["kind"]): r["mean_logp"] for r in after}
    torch.testing.assert_close(
        before_map[(1, "visual_replacement")],
        after_map[(1, "visual_replacement")],
        atol=0,
        rtol=0,
    )
    image_branch = plan.branches[0]
    changed_base = base.detach().clone()
    changed_base[:, image_branch.source_positions[0], 0] += 4
    sourced, _ = score_evidence_replacement(model, options, ids, changed_base, clean, positions, None)
    sourced_map = {(r["step_id"], r["kind"]): r["mean_logp"] for r in sourced}
    assert not torch.equal(
        before_map[(0, "visual_replacement")],
        sourced_map[(0, "visual_replacement")],
    )
    linked = base.detach().clone()
    image_visible, _ = score_evidence_replacement(model, options, ids, linked, linked, positions, None)
    image_changed = linked.clone()
    image_changed[:, image_branch.source_positions[0], 0] += 4
    image_changed_scores, _ = score_evidence_replacement(
        model, options, ids, image_changed, image_changed, positions, None
    )
    linked_map = {(r["step_id"], r["kind"]): r["mean_logp"] for r in image_visible}
    image_changed_map = {(r["step_id"], r["kind"]): r["mean_logp"] for r in image_changed_scores}
    assert not torch.equal(
        linked_map[(0, "visual_original")],
        image_changed_map[(0, "visual_original")],
    )


@pytest.mark.parametrize("checkpoint_enabled", [False, True])
def test_paired_dropout_reuses_native_original_draws_and_preserves_post_original_rng(checkpoint_enabled):
    """Catches replacement dropout advancing global RNG or diverging from native replay."""
    from transformers.models.qwen2_5_vl.stage2_evidence_replacement import (
        score_evidence_replacement,
    )

    model, ids, positions, base, clean, options = _case()
    for layer in model.model.layers:
        layer.self_attn.attention_dropout = 0.19
    options["checkpoint"] = checkpoint_enabled
    plan = build_replacement_plan(
        input_ids=ids, sidecar_plan=options["plan"], image_token_id=80,
        image_count=1, latent_size=2, sample_id=options["sample_id"],
        seed=options["replacement_seed"],
    )
    paired_clean = clean.detach().clone()
    self_branch = plan.branches[1]
    paired_clean[:, self_branch.latent_start : self_branch.latent_end] = clean.index_select(
        1, torch.tensor(self_branch.source_positions)
    )
    torch.manual_seed(818)
    paired, _ = score_evidence_replacement(model, options, ids, base, paired_clean, positions, None)
    paired_rng = torch.get_rng_state().clone()
    torch.manual_seed(818)
    expected = {}
    for branch in plan.branches:
        prefix = paired_clean[:, : branch.prefix_length]
        hidden = model.model(
            inputs_embeds=prefix,
            position_ids=positions[:, :, : branch.prefix_length],
            attention_mask=torch.ones((1, branch.prefix_length), dtype=torch.long),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        rows = torch.tensor(branch.predictor_positions)
        targets = torch.tensor(branch.target_token_ids)
        log_probs = model.lm_head(hidden.index_select(1, rows)).float().log_softmax(-1)
        expected[branch.step_id] = log_probs.gather(-1, targets[None, :, None]).mean()
    torch.testing.assert_close(paired_rng, torch.get_rng_state(), atol=0, rtol=0)
    for result in paired:
        if result["kind"] == "visual_original":
            torch.testing.assert_close(result["mean_logp"], expected[result["step_id"]], atol=0, rtol=0)
    paired_scores = {(r["step_id"], r["kind"]): r["mean_logp"] for r in paired}
    torch.testing.assert_close(
        paired_scores[(1, "visual_original")],
        paired_scores[(1, "visual_replacement")], atol=0, rtol=0,
    )


def test_self_substitution_is_an_exact_score_identity_and_first_predictor_matters():
    """Catches shifting the first target row or changing an equal replacement prefix."""
    from transformers.models.qwen2_5_vl.stage2_evidence_replacement import (
        score_evidence_replacement,
    )

    model, ids, positions, base, clean, options = _case()
    plan = build_replacement_plan(
        input_ids=ids, sidecar_plan=options["plan"], image_token_id=80,
        image_count=1, latent_size=2, sample_id=options["sample_id"],
        seed=options["replacement_seed"],
    )
    branch = plan.branches[1]
    self_substitution = clean.detach().clone()
    self_substitution[:, branch.latent_start : branch.latent_end] = clean.index_select(
        1, torch.tensor(branch.source_positions)
    )
    equal, _ = score_evidence_replacement(model, options, ids, base, self_substitution, positions, None)
    equal_scores = {(r["step_id"], r["kind"]): r["mean_logp"] for r in equal}
    torch.testing.assert_close(
        equal_scores[(branch.step_id, "visual_original")],
        equal_scores[(branch.step_id, "visual_replacement")], atol=0, rtol=0,
    )
    changed = self_substitution.clone()
    changed[:, branch.predictor_positions[0], 0] += 3
    perturbed, _ = score_evidence_replacement(model, options, ids, base, changed, positions, None)
    perturbed_scores = {(r["step_id"], r["kind"]): r["mean_logp"] for r in perturbed}
    assert not torch.equal(
        equal_scores[(branch.step_id, "visual_original")],
        perturbed_scores[(branch.step_id, "visual_original")],
    )


def test_checkpoint_replays_fixed_sources_with_the_same_outputs_and_gradients():
    """Catches delayed backward rebinding a branch's source indices or detached live prefix."""
    from transformers.models.qwen2_5_vl.stage2_evidence_replacement import (
        score_evidence_replacement,
    )

    model, ids, positions, base, clean, options = _case()
    reference = copy.deepcopy(model)
    fixed_plan = build_replacement_plan(
        input_ids=ids, sidecar_plan=options["plan"], image_token_id=80,
        image_count=1, latent_size=2, sample_id=options["sample_id"],
        seed=options["replacement_seed"],
    )
    assert fixed_plan == build_replacement_plan(
        input_ids=ids, sidecar_plan=options["plan"], image_token_id=80,
        image_count=1, latent_size=2, sample_id=options["sample_id"],
        seed=options["replacement_seed"],
    )
    checkpoint_base = base.detach().clone().requires_grad_()
    checkpoint_clean = clean.detach().clone().requires_grad_()
    plain_base = base.detach().clone().requires_grad_()
    plain_clean = clean.detach().clone().requires_grad_()
    checkpoint_results, _ = score_evidence_replacement(
        model, dict(options, checkpoint=True), ids, checkpoint_base,
        checkpoint_clean, positions, None,
    )
    plain_results, _ = score_evidence_replacement(
        reference, dict(options, checkpoint=False), ids, plain_base,
        plain_clean, positions, None,
    )
    checkpoint_loss = sum(r["mean_logp"] + .03 * r["entropies"].mean() for r in checkpoint_results)
    plain_loss = sum(r["mean_logp"] + .03 * r["entropies"].mean() for r in plain_results)
    torch.testing.assert_close(checkpoint_loss, plain_loss, atol=0, rtol=0)
    checkpoint_loss.backward()
    plain_loss.backward()
    torch.testing.assert_close(checkpoint_clean.grad, plain_clean.grad, atol=1e-5, rtol=1e-4)
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        if actual.requires_grad:
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-5, rtol=1e-4, msg=name)


def test_variable_target_lengths_keep_each_native_entropy_vector():
    """Catches flattening two steps' targets or reusing the prior step's row count."""
    from transformers.models.qwen2_5_vl.stage2_evidence_replacement import (
        score_evidence_replacement,
    )

    model = _tiny_model().float().train()
    ids = torch.tensor([[11, 81, 80, 80, 80, 83, 14, 91, 90, 90, 92, 21,
                         91, 90, 90, 92, 22, 23, 24, 25]])
    spans = (
        ReasoningStepSpan(0, 0, 7, 8, 10, 11, 12, None),
        ReasoningStepSpan(1, 11, 12, 13, 15, 16, 19, None),
    )
    sidecar = build_context_sidecar_plan(
        clean_input_ids=ids, attention_mask=torch.ones_like(ids), spans=spans,
        visual_only=True, suppress_latent_boundaries=False,
    )
    positions = torch.arange(ids.shape[1])[None, None].expand(3, 1, -1).clone()
    base = model.model.embed_tokens(ids).detach().requires_grad_()
    clean = (base.detach() + .1).requires_grad_()
    results, _ = score_evidence_replacement(
        model,
        dict(plan=sidecar, prefix_mode="evidence_replacement", backend="dense",
             checkpoint=False, sample_id="variable-targets", replacement_seed=42,
             replacement_version="ilvr-evidence-replacement-v1"),
        ids, base, clean, positions, None,
    )
    assert [result["entropies"].numel() for result in results] == [1, 1, 3, 3]


def test_zero_steps_returns_ce_only_scoring_shape_without_replay():
    """Catches requiring an image donor or decoder replay for a valid CE-only sample."""
    from transformers.models.qwen2_5_vl.stage2_evidence_replacement import (
        score_evidence_replacement,
    )

    model = _tiny_model().float().train()
    ids = torch.tensor([[11, 12, 13]])
    sidecar = build_context_sidecar_plan(
        clean_input_ids=ids, attention_mask=torch.ones_like(ids), spans=(),
        visual_only=True, suppress_latent_boundaries=False,
    )
    base = model.model.embed_tokens(ids)
    positions = torch.arange(ids.shape[1])[None, None].expand(3, 1, -1).clone()
    results, stats = score_evidence_replacement(
        model,
        dict(plan=sidecar, prefix_mode="evidence_replacement", backend="dense",
             checkpoint=True, sample_id="zero-steps", replacement_seed=42,
             replacement_version="ilvr-evidence-replacement-v1"),
        ids, base, base, positions, None,
    )
    assert results == ()
    assert stats["chunks"] == stats["query_rows"] == stats["entropy_scored_rows"] == 0


@pytest.mark.parametrize("length", [3, 8])
def test_flex_visibility_matches_actual_native_sliding_mask_even_when_flag_is_false(length):
    """Catches Flex honoring configured support instead of Qwen's native no-cache rule."""
    from transformers.models.qwen2_5_vl.stage2_evidence_replacement import _native_mask

    model, _, _, _, _, _ = _case()
    model.config.sliding_window = 4
    model.config.use_sliding_window = False
    embeds = torch.zeros((1, length, model.config.hidden_size))
    flex = _native_mask(model, embeds, "flex")
    rows = torch.arange(max(length, 16))[:, None]
    keys = torch.arange(max(length, 16))[None, :]
    actual = flex.mask_mod(torch.tensor(0), torch.tensor(0), rows, keys)
    native = model.model._prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask=None, sequence_length=length, target_length=length,
        dtype=embeds.dtype, device=embeds.device,
        cache_position=torch.arange(length), batch_size=1,
        config=model.config, past_key_values=None,
    )[0, 0]
    expected = torch.zeros_like(actual)
    expected[:length, :length] = native == 0
    assert torch.equal(actual, expected)


def test_model_dispatch_requires_one_real_image_grid_then_routes_active_plan():
    """Catches bypassing Stage 2's image preconditions or leaving the new mode undispatched."""
    model, ids, positions, _, _, options = _case()
    with pytest.raises(ValueError, match="requires pixel_values"):
        model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=positions,
              use_cache=False, stage2_shared_prefix_inputs=options)
    pixels = torch.zeros((3, 12))
    with pytest.raises(ValueError, match="exactly one image grid"):
        model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=positions,
              pixel_values=pixels, image_grid_thw=torch.ones((2, 3), dtype=torch.long),
              use_cache=False, stage2_shared_prefix_inputs=options)
    output = model(
        input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=positions,
        pixel_values=pixels, image_grid_thw=torch.tensor([[1, 1, 3]]),
        use_cache=False, stage2_shared_prefix_inputs=options,
    )
    assert {result["kind"] for result in output.stage2_sidecar_results} == {
        "visual_original", "visual_replacement"
    }


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("RUN_REPLACEMENT_CUDA") != "1",
    reason="set RUN_REPLACEMENT_CUDA=1 for native CUDA Flex replay",
)
def test_cuda_flex_dispatch_matches_dense_outputs_and_gradients():
    """Catches configured-but-unused Flex, or a Flex causal/sliding mask mismatch."""
    from transformers.models.qwen2_5_vl.stage2_evidence_replacement import (
        score_evidence_replacement,
    )

    model, ids, positions, base, clean, options = _case()
    model.config.sliding_window = 4
    model.config.use_sliding_window = False
    dense = copy.deepcopy(model).cuda()
    flex = copy.deepcopy(model).cuda()
    ids, positions = ids.cuda(), positions.cuda()
    dense_base = base.detach().cuda().requires_grad_()
    dense_clean = clean.detach().cuda().requires_grad_()
    flex_base = base.detach().cuda().requires_grad_()
    flex_clean = clean.detach().cuda().requires_grad_()
    dense_options = dict(options, backend="dense")
    flex_options = dict(options, backend="flex")
    dense_results, _ = score_evidence_replacement(
        dense, dense_options, ids, dense_base, dense_clean, positions, None
    )
    flex_results, _ = score_evidence_replacement(
        flex, flex_options, ids, flex_base, flex_clean, positions, None
    )
    assert sum(layer.self_attn._flex_attention_forward_count for layer in flex.model.layers) > 0
    for actual, expected in zip(flex_results, dense_results):
        torch.testing.assert_close(actual["mean_logp"], expected["mean_logp"], atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(actual["entropies"], expected["entropies"], atol=2e-5, rtol=2e-4)
    sum(r["mean_logp"] for r in dense_results).backward()
    sum(r["mean_logp"] for r in flex_results).backward()
    for (name, actual), (_, expected) in zip(flex.named_parameters(), dense.named_parameters()):
        if actual.requires_grad:
            torch.testing.assert_close(actual.grad, expected.grad, atol=3e-5, rtol=3e-4, msg=name)
