"""Literal sequence and native runtime witnesses for image-free OFF insertion."""
from dataclasses import replace

import pytest
import torch

from step_preference.sidecar import build_context_sidecar_plan
from step_preference.spans import ReasoningStepSpan


def two_step_inputs():
    ids = torch.tensor([[11, 78, 80, 79, 12, 91, 90, 90, 92, 13, 14, 91, 90, 90, 92, 15, 16]])
    spans = (ReasoningStepSpan(0, 0, 5, 6, 8, 9, 11, None),
             ReasoningStepSpan(1, 9, 11, 12, 14, 15, 17, 0))
    plan = build_context_sidecar_plan(clean_input_ids=ids, attention_mask=torch.ones_like(ids),
                                    spans=spans, visual_only=True)
    return dict(input_ids=ids, sidecar_plan=plan, image_token_id=80,
                vision_start_token_id=78, vision_end_token_id=79,
                latent_start_token_id=91, latent_pad_token_id=90, latent_end_token_id=92)


def planner(**kwargs):
    from step_preference.insertion import build_insertion_plan
    return build_insertion_plan(**kwargs)


@pytest.mark.parametrize('mode', ['independent', 'cumulative'])
def test_off_removes_all_visual_slots_but_keeps_boundaries(mode):
    plan = planner(**two_step_inputs(), branch_mode=mode)
    assert plan.off_token_ids == (11, 12, 91, 92, 13, 14, 91, 92, 15, 16)
    assert plan.off_source_indices == (0, 4, 5, 8, 9, 10, 11, 14, 15, 16)
    assert [t.target_token_ids for t in plan.off_targets] == [(13, 14), (15, 16)]
    assert [t.selected_positions for t in plan.off_targets] == [(3, 4), (7, 8)]


def test_independent_second_branch_starts_from_empty_first_block():
    plan = planner(**two_step_inputs(), branch_mode='independent')
    assert [branch.prefix_length for branch in plan.branches] == [2, 6]
    assert plan.branches[1].token_ids == (11, 12, 91, 92, 13, 14, 78, 80, 79, 91, 90, 90, 92, 15, 16)
    assert plan.branches[1].targets[0].selected_positions == (12, 13)
    assert plan.branches[1].active_spans == ((9, (10, 11), 12),)


def test_cumulative_keeps_both_insertions_in_one_branch():
    plan = planner(**two_step_inputs(), branch_mode='cumulative')
    assert len(plan.branches) == 1
    branch = plan.branches[0]
    assert branch.prefix_length == 2
    assert branch.token_ids == (11, 12, 78, 80, 79, 91, 90, 90, 92, 13, 14, 78, 80, 79, 91, 90, 90, 92, 15, 16)
    assert [t.selected_positions for t in branch.targets] == [(8, 9), (17, 18)]
    assert branch.active_spans == ((5, (6, 7), 8), (14, (15, 16), 17))


@pytest.mark.parametrize('position,value', [(2, 17), (3, 17), (6, 17), (8, 90)])
def test_malformed_image_or_latent_blocks_fail(position, value):
    kwargs = two_step_inputs()
    kwargs['input_ids'][0, position] = value
    with pytest.raises(ValueError):
        planner(**kwargs, branch_mode='independent')


def test_empty_plan_has_no_invented_targets_or_active_latents():
    kwargs = two_step_inputs()
    kwargs['sidecar_plan'] = replace(kwargs['sidecar_plan'], groups=((), ()), root_selected_positions=())
    plan = planner(**kwargs, branch_mode='cumulative')
    assert plan.branches == ()
    assert plan.off_targets == ()
    assert 90 not in plan.off_token_ids


def test_unselected_latent_block_stays_empty():
    kwargs = two_step_inputs()
    original = kwargs['sidecar_plan']
    kwargs['sidecar_plan'] = replace(original, groups=((original.groups[0][1],), ()))
    plan = planner(**kwargs, branch_mode='cumulative')
    assert plan.branches[0].token_ids == (11, 12, 91, 92, 13, 14, 78, 80, 79, 91, 90, 90, 92, 15, 16)


def test_unknown_branch_mode_fails():
    with pytest.raises(ValueError):
        planner(**two_step_inputs(), branch_mode='other')


def test_nonempty_visual_request_requires_a_source_image_block():
    kwargs = two_step_inputs()
    kwargs['input_ids'][0, 1:4] = torch.tensor([21, 22, 23])
    with pytest.raises(ValueError, match='original image'):
        planner(**kwargs, branch_mode='independent')


def scorer(*args):
    from transformers.models.qwen2_5_vl.stage2_v_off_insertion import score_v_off_insertion
    return score_v_off_insertion(*args)


def test_native_off_first_and_exact_pre_s_kv_reuse():
    from tests.step_preference.insertion_helpers import native_case
    model, ids, base, grid, options, _ = native_case()
    calls = []
    handle = model.model.register_forward_hook(lambda module, args, kw, out: calls.append((kw, out)), with_kwargs=True)
    result, _ = scorer(model, options, ids, base, grid)
    handle.remove()
    first, off = calls[0]
    assert first.get('stage2_capture_kv') is True
    assert first['inputs_embeds'].shape[1] == 10
    torch.testing.assert_close(first['position_ids'], torch.arange(10)[None, None].expand(3, 1, -1))
    assert len(result) == 4
    for kwargs, _ in calls[1:]:
        prefix = kwargs['stage2_prefix_key_values']
        length = prefix[0][0].shape[2]
        assert length in (2, 6)
        for (key, value), (off_key, off_value) in zip(prefix, off.stage2_key_values):
            torch.testing.assert_close(key, off_key[:, :, :length], atol=0, rtol=0)
            torch.testing.assert_close(value, off_value[:, :, :length], atol=0, rtol=0)
            assert key.untyped_storage().data_ptr() == off_key.untyped_storage().data_ptr()


def test_off_is_independent_of_raw_image_and_clean_latent_values():
    from tests.step_preference.insertion_helpers import native_case
    model, ids, base, grid, options, _ = native_case()
    raw = base.detach().requires_grad_()
    original, _ = scorer(model, options, ids, raw, grid)
    changed = raw.detach().clone()
    changed[:, 2, 0] += 5
    changed[:, [6, 7, 12, 13]] = 1000
    perturbed, _ = scorer(model, options, ids, changed, grid)
    for a, b in zip(original, perturbed):
        if a['kind'] == 'visual_off':
            torch.testing.assert_close(a['mean_logp'], b['mean_logp'], atol=0, rtol=0)
            derivative = torch.autograd.grad(a['mean_logp'], raw, retain_graph=True, allow_unused=True)[0]
            assert derivative is None or derivative.count_nonzero() == 0
        else:
            assert not torch.allclose(a['mean_logp'], b['mean_logp'], atol=1e-7, rtol=0)
            derivative = torch.autograd.grad(a['mean_logp'], raw, retain_graph=True)[0]
            assert derivative[:, 2].isfinite().all() and derivative[:, 2].count_nonzero() > 0
            assert derivative[:, [6, 7, 12, 13]].count_nonzero() == 0
            start, end = (5, 8) if a['step_id'] == 0 else (11, 14)
            assert derivative[:, start].count_nonzero() > 0
            assert derivative[:, end].count_nonzero() > 0
        marker_grad = torch.autograd.grad(a['mean_logp'], model.model.embed_tokens.weight, retain_graph=True)[0]
        assert marker_grad[[91, 92]].isfinite().all()
        # ON text after the fork uses raw embeddings; START contributes through
        # the earlier OFF prefix for step 2, and directly through raw for step 1.
        if a['kind'] == 'visual_off':
            assert marker_grad[91].count_nonzero() and marker_grad[92].count_nonzero()


@pytest.mark.parametrize('mode', ['independent', 'cumulative'])
@pytest.mark.parametrize('checkpointed', [False, True])
@pytest.mark.parametrize('window_settings', [(None, False), (4, False), (4, True)],
                         ids=['full', 'sliding_disabled', 'sliding_enabled'])
def test_native_prefix_matches_literal_full_sequence_oracle(mode, checkpointed, window_settings, monkeypatch):
    import copy
    from tests.step_preference.insertion_helpers import native_case, literal_oracle, preference
    model, ids, _, grid, options, pixels = native_case()
    model.config.sliding_window, model.config.use_sliding_window = window_settings
    oracle = copy.deepcopy(model)
    if checkpointed:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    options.update(branch_mode=mode, checkpoint=checkpointed)
    base = model.prepare_stage2_base_inputs_embeds(input_ids=ids, pixel_values=pixels, image_grid_thw=grid)
    states, rows = [], []
    original_rollout = model.stage2_rollout_engine.rollout
    def observed_rollout(**kwargs):
        value = original_rollout(**kwargs)
        states.extend(value.latent_states)
        return value
    monkeypatch.setattr(model.stage2_rollout_engine, 'rollout', observed_rollout)
    handle = model.lm_head.register_forward_hook(lambda module, args, out: rows.append(out))
    values, _ = scorer(model, options, ids, base, grid)
    handle.remove()
    actual = {(r['kind'], r['step_id']): r['mean_logp'] for r in values}
    expected, expected_logits, expected_states = literal_oracle(oracle,
        oracle.prepare_stage2_base_inputs_embeds(input_ids=ids, pixel_values=pixels, image_grid_thw=grid), grid, mode)
    assert len(states) == len(expected_states) == 4
    for a, b in zip(states, expected_states):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-5)
    # Scorer projects each context's selected rows once in target order.
    all_rows = torch.cat([r.reshape(-1, 128) for r in rows])
    expected_rows = torch.cat([expected_logits[('visual_off', 0)], expected_logits[('visual_off', 1)],
                               expected_logits[('visual_on', 0)], expected_logits[('visual_on', 1)]])
    torch.testing.assert_close(all_rows, expected_rows, atol=1e-5, rtol=1e-5)
    preference(actual).backward()
    preference(expected).backward()
    assert model.model.embed_tokens.weight.grad.norm() > 0
    for (name, a), (_, b) in zip(model.named_parameters(), oracle.named_parameters()):
        if a.requires_grad:
            assert a.grad is not None and b.grad is not None, name
            assert a.grad.isfinite().all(), name
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-3, msg=name)


def test_independent_reset_vs_cumulative_history_and_future_causality():
    from tests.step_preference.insertion_helpers import native_case
    model, ids, base, grid, options, _ = native_case()
    first, _ = scorer(model, options, ids, base, grid)
    options['branch_mode'] = 'cumulative'
    second, _ = scorer(model, options, ids, base, grid)
    left = {(r['kind'], r['step_id']): r['mean_logp'] for r in first}
    right = {(r['kind'], r['step_id']): r['mean_logp'] for r in second}
    torch.testing.assert_close(left[('visual_on', 0)], right[('visual_on', 0)], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(left[('visual_on', 1)], right[('visual_on', 1)], atol=1e-7, rtol=0)
    changed = base.clone()
    changed[:, 15:] += 20
    future, _ = scorer(model, options, ids, changed, grid)
    for item in future:
        if item['step_id'] == 0:
            torch.testing.assert_close(item['mean_logp'], right[(item['kind'], 0)], atol=0, rtol=0)


def insertion_config(**kwargs):
    from step_preference.config import StepPreferenceConfig
    values = dict(enabled=True, mode='stage2b', stage2a_checkpoint_path='tiny-stage2a',
        shared_prefix_enabled=True, visual_only_context_preference=True,
        visual_off_suppress_latent_boundaries=False, prefix_mode='v_off_insertion',
        beta=.1, lambda_ce=1., preference_margin=0.,
        lambda_preference=10., lambda_entropy=0., sidecar_attention_backend='dense')
    values.update(kwargs)
    return StepPreferenceConfig(**values)


@pytest.mark.parametrize('change', [dict(shared_prefix_enabled=False),
    dict(visual_only_context_preference=False), dict(visual_off_suppress_latent_boundaries=True),
    dict(block_original_image=True), dict(packing_enabled=True), dict(objective='legacy_dpo'),
    dict(branch_mode='unknown'), dict(enabled=False)])
def test_insertion_rejects_methodology_conflicts(change):
    with pytest.raises(ValueError):
        insertion_config(**change)


@pytest.mark.parametrize('mode', ['independent', 'cumulative'])
def test_native_trainer_reduces_true_evidence_scores_and_emits_metrics(mode, monkeypatch):
    from tests.step_preference.insertion_helpers import native_case, preference
    from step_preference.equivalence import _trainer
    from transformers import Trainer
    model, ids, _, grid, _, pixels = native_case()
    spans = (ReasoningStepSpan(0, 0, 5, 6, 8, 9, 11, None),
             ReasoningStepSpan(1, 9, 11, 12, 14, 15, 17, 0))
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
        pixel_values=pixels, image_grid_thw=grid, sample_ids=('native-insertion',),
        reasoning_step_spans=spans, step_preference={'steps': tuple({'span': s} for s in spans)})
    trainer = _trainer(insertion_config(branch_mode=mode))
    clean = model(input_ids=ids, attention_mask=batch['attention_mask'], labels=ids.clone(),
                  pixel_values=pixels, image_grid_thw=grid, use_cache=False, return_dict=True)
    total, output = trainer.compute_loss(model, batch, return_outputs=True)
    torch.testing.assert_close(output.loss, clean.loss, atol=0, rtol=0)
    scores = {(r['kind'], r['step_id']): r['mean_logp'] for r in output.stage2_sidecar_results}
    torch.testing.assert_close(total, output.loss + 10 * preference(scores))
    logged = []
    monkeypatch.setattr(Trainer, 'log', lambda self, logs, **kwargs: logged.append(logs))
    trainer.log({'loss': float(total.detach())})
    assert logged[0]['stats/logp_evidence_on'] == pytest.approx(float(torch.stack([scores[('visual_on', i)] for i in (0, 1)]).mean().detach()))
    assert logged[0]['stats/logp_evidence_off'] == pytest.approx(float(torch.stack([scores[('visual_off', i)] for i in (0, 1)]).mean().detach()))
    total.backward()
    assert model.model.embed_tokens.weight.grad.isfinite().all()


@pytest.mark.parametrize('active', [True, False])
def test_native_insertion_requires_pixels_only_for_active_visual_branches(active):
    from tests.step_preference.insertion_helpers import native_case
    model, ids, _, grid, options, _ = native_case()
    if not active:
        options['plan'] = replace(options['plan'], groups=((), ()), root_selected_positions=())
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
        image_grid_thw=grid, pixel_values=None, use_cache=False, return_dict=True,
        stage2_shared_prefix_inputs=options)
    if active:
        with pytest.raises(ValueError, match='pixel_values'):
            model(**kwargs)
    else:
        output = model(**kwargs)
        assert torch.isfinite(output.loss)
        assert output.stage2_sidecar_results == ()


@pytest.mark.parametrize('scope,wanted', [('immediate_text', [(13,), (15,)]),
                                         ('until_next_latent', [(13, 14), (15, 16)])])
def test_adjacent_text_entry_targets_retain_the_selected_scope(scope, wanted):
    from tests.step_preference.insertion_helpers import native_case
    model, ids, base, grid, options, _ = native_case()
    end1, end2 = (10, 16) if scope == 'immediate_text' else (11, 17)
    spans = (ReasoningStepSpan(0, 0, 5, 6, 8, 9, end1, None),
             ReasoningStepSpan(1, 10, 11, 12, 14, 15, end2, 0))
    sidecar = build_context_sidecar_plan(clean_input_ids=ids, attention_mask=torch.ones_like(ids), spans=spans, visual_only=True)
    config = insertion_config(target_scope=scope)
    assert config.target_scope == scope
    kwargs = two_step_inputs()
    kwargs['sidecar_plan'] = sidecar
    plan = planner(**kwargs, branch_mode='cumulative')
    assert [target.target_token_ids for target in plan.off_targets] == wanted
    assert [target.target_token_ids for target in plan.branches[0].targets] == wanted
    result, _ = scorer(model, {**options, 'plan': sidecar, 'branch_mode': 'cumulative'}, ids, base, grid)
    for item in result:
        assert len(item['entropies']) == len(wanted[item['step_id']])


def test_native_off_kv_logits_and_positions_ignore_image_resolution_and_pixels():
    from tests.step_preference.insertion_helpers import native_case, image_case
    model, _, _, _, options, _ = native_case()
    observed = []
    handle = model.model.register_forward_hook(
        lambda module, args, kw, out: observed.append((kw, out)) if kw.get('stage2_capture_kv') else None,
        with_kwargs=True)
    results = []
    for grids in ([[1, 1, 1]], [[1, 2, 2]]):
        ids, base, grid, plan, _ = image_case(model, grids)
        results.append(scorer(model, {**options, 'plan': plan}, ids, base, grid)[0])
    handle.remove()
    (left, a), (right, b) = observed
    torch.testing.assert_close(left['inputs_embeds'], right['inputs_embeds'], atol=0, rtol=0)
    torch.testing.assert_close(left['position_ids'], right['position_ids'], atol=0, rtol=0)
    torch.testing.assert_close(a.last_hidden_state, b.last_hidden_state, atol=0, rtol=0)
    for pa, pb in zip(a.stage2_key_values, b.stage2_key_values):
        for ta, tb in zip(pa, pb):
            torch.testing.assert_close(ta, tb, atol=0, rtol=0)
    for a, b in zip(*results):
        if a['kind'] == 'visual_off':
            torch.testing.assert_close(a['mean_logp'], b['mean_logp'], atol=0, rtol=0)
        else:
            assert not torch.allclose(a['mean_logp'], b['mean_logp'], atol=1e-7, rtol=0)
    # Real native encoder pixel perturbation, preserving its frozen/no_grad policy.
    ids, base, grid, plan, pixels = image_case(model, [[1, 1, 1]])
    changed = model.prepare_stage2_base_inputs_embeds(input_ids=ids, pixel_values=pixels.flip(-1), image_grid_thw=grid)
    perturbed, _ = scorer(model, {**options, 'plan': plan}, ids, changed, grid)
    for a, b in zip(results[0], perturbed):
        if a['kind'] == 'visual_off':
            torch.testing.assert_close(a['mean_logp'], b['mean_logp'], atol=0, rtol=0)
        else:
            assert not torch.allclose(a['mean_logp'], b['mean_logp'], atol=1e-7, rtol=0)


def test_multiple_images_repeat_in_original_order_and_use_matching_positions():
    from tests.step_preference.insertion_helpers import native_case, image_case
    model, _, _, _, options, _ = native_case()
    ids, base, grid, sidecar, _ = image_case(model, [[1, 1, 1], [1, 1, 2]])
    kwargs = two_step_inputs()
    kwargs.update(input_ids=ids, sidecar_plan=sidecar)
    plan = planner(**kwargs, branch_mode='cumulative')
    branch = plan.branches[0]
    assert branch.token_ids[:13] == (11, 12, 78, 80, 79, 78, 80, 80, 79, 91, 90, 90, 92)
    calls = []
    handle = model.model.register_forward_pre_hook(lambda module, args, kw: calls.append(kw), with_kwargs=True)
    result, _ = scorer(model, {**options, 'plan': sidecar, 'branch_mode': 'cumulative'}, ids, base, grid)
    handle.remove()
    # Final suffix embeds contain both image sets in their original order.
    final_embeds = calls[-1]['inputs_embeds']
    torch.testing.assert_close(final_embeds[:, [1, 4, 5]], base[:, [2, 5, 6]], atol=0, rtol=0)
    torch.testing.assert_close(final_embeds[:, [14, 17, 18]], base[:, [2, 5, 6]], atol=0, rtol=0)
    assert len(result) == 4
    sum(r['mean_logp'] for r in result).backward()
    assert model.model.embed_tokens.weight.grad.isfinite().all()


def test_future_inserted_image_cannot_change_earlier_cumulative_score():
    from tests.step_preference.insertion_helpers import native_case
    model, ids, base, grid, options, _ = native_case()
    options['branch_mode'] = 'cumulative'
    original, _ = scorer(model, options, ids, base, grid)
    def perturb_future_image(module, args, kwargs):
        if kwargs.get('stage2_prefix_key_values') is not None and kwargs['inputs_embeds'].shape[1] > 10:
            embeds = kwargs['inputs_embeds'].clone()
            embeds[:, 10, 0] += 5  # second inserted image: full index12, suffix index10
            return args, {**kwargs, 'inputs_embeds': embeds}
    handle = model.model.register_forward_pre_hook(perturb_future_image, with_kwargs=True)
    changed, _ = scorer(model, options, ids, base, grid)
    handle.remove()
    for a, b in zip(original, changed):
        if a['step_id'] == 0 or a['kind'] == 'visual_off':
            torch.testing.assert_close(a['mean_logp'], b['mean_logp'], atol=0, rtol=0)
        else:
            assert not torch.allclose(a['mean_logp'], b['mean_logp'], atol=1e-7, rtol=0)


def test_future_teacher_forced_token_ids_cannot_change_earlier_on_or_off_scores():
    from tests.step_preference.insertion_helpers import native_case
    model, ids, base, grid, options, pixels = native_case()
    options['branch_mode'] = 'cumulative'
    original, _ = scorer(model, options, ids, base, grid)
    changed_ids = ids.clone()
    changed_ids[:, 15:] = torch.tensor([23, 24])
    spans = (ReasoningStepSpan(0, 0, 5, 6, 8, 9, 11, None),
             ReasoningStepSpan(1, 9, 11, 12, 14, 15, 17, 0))
    changed_plan = build_context_sidecar_plan(clean_input_ids=changed_ids,
        attention_mask=torch.ones_like(changed_ids), spans=spans, visual_only=True)
    changed_base = model.prepare_stage2_base_inputs_embeds(input_ids=changed_ids,
        pixel_values=pixels, image_grid_thw=grid)
    changed, _ = scorer(model, {**options, 'plan': changed_plan}, changed_ids, changed_base, grid)
    for a, b in zip(original, changed):
        if a['step_id'] == 0:
            torch.testing.assert_close(a['mean_logp'], b['mean_logp'], atol=0, rtol=0)
        else:
            assert not torch.allclose(a['mean_logp'], b['mean_logp'], atol=1e-7, rtol=0)


@pytest.mark.parametrize('change', ['missing_image', 'bad_grid', 'bad_backend'])
def test_native_rejects_missing_visual_inputs_or_invalid_backend(change):
    from tests.step_preference.insertion_helpers import native_case
    model, ids, base, grid, options, _ = native_case()
    if change == 'missing_image':
        grid = None
    elif change == 'bad_grid':
        grid = torch.tensor([[1, 1, 2]])
    else:
        options['backend'] = 'unknown'
    with pytest.raises(ValueError):
        scorer(model, options, ids, base, grid)


def test_empty_native_plan_executes_no_off_or_latent_forward():
    from tests.step_preference.insertion_helpers import native_case
    model, ids, base, grid, options, _ = native_case()
    options['plan'] = replace(options['plan'], groups=((), ()), root_selected_positions=())
    calls = []
    handle = model.model.register_forward_hook(lambda *args: calls.append(args))
    result, stats = scorer(model, options, ids, base, grid)
    handle.remove()
    assert result == () and calls == []
    assert stats['entropy_scored_rows'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Flex native insertion probe needs CUDA')
@pytest.mark.parametrize('mode', ['independent', 'cumulative'])
@pytest.mark.parametrize('window_settings', [(None, False), (4, False), (4, True)],
                         ids=['full', 'sliding_disabled', 'sliding_enabled'])
def test_cuda_flex_insertion_forward_backward_matches_dense(mode, window_settings):
    import copy
    from tests.step_preference.insertion_helpers import native_case, preference
    model, ids, _, grid, options, pixels = native_case()
    model.config.sliding_window, model.config.use_sliding_window = window_settings
    model = model.cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    dense = copy.deepcopy(model)
    ids, grid, pixels = ids.cuda(), grid.cuda(), pixels.cuda()
    options.update(branch_mode=mode, checkpoint=True)
    outputs = []
    for candidate, backend in ((dense, 'dense'), (model, 'flex')):
        base = candidate.prepare_stage2_base_inputs_embeds(input_ids=ids, pixel_values=pixels, image_grid_thw=grid)
        results, _ = scorer(candidate, {**options, 'backend': backend}, ids, base, grid)
        scores = {(r['kind'], r['step_id']): r['mean_logp'] for r in results}
        outputs.append(scores)
        preference(scores).backward()
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=1e-5, rtol=1e-5)
    for (name, a), (_, b) in zip(model.named_parameters(), dense.named_parameters()):
        if a.requires_grad:
            assert a.grad.isfinite().all(), name
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-3, msg=name)
