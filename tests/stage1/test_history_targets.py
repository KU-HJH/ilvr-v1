"""Runtime target parity, injected-prefix causality and fail-closed ownership."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from peft_common import resolve_qwen2_5_vl_model
from stage1.query import Stage1QueryConfig
from stage1.targets import Stage1TeacherTargetBuilder
from stage1.types import Stage1LatentSpan
from tests.stage1.history_fixtures import (
    load_baseline_target_builder, make_history_case,
)

BASE = '148b0e8ead14456d02cc595f0cc32a0752ee22e8'


class RecordingBuilder(Stage1TeacherTargetBuilder):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.queries = []
        self.candidates = []
        self.selections = []

    def _select(self, candidates, query, max_count):
        self.queries.append(query.clone())
        self.candidates.append(candidates.clone())
        result = super()._select(candidates, query, max_count)
        self.selections.append(result[0].clone())
        return result


def builder(case, **kwargs):
    return RecordingBuilder(case.tokenizer, query_config=Stage1QueryConfig(mode='adaptive'),
                            coverage_p=.1, helper_group_L=0, **kwargs)


def observe_build(case, target_builder):
    """Capture actual decoder inputs and actual original/helper vision calls."""
    qwen = resolve_qwen2_5_vl_model(case.teacher)
    prefixes, visuals = [], []
    handles = [
        qwen.model.register_forward_pre_hook(
            lambda m, a, kw: prefixes.append({
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in kw.items()}), with_kwargs=True),
        qwen.visual.register_forward_pre_hook(
            lambda m, a: visuals.append(a[0].clone())),
    ]
    rng = torch.random.get_rng_state().clone()
    try:
        result = target_builder.build(case.teacher, dict(case.inputs), case.metadata)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(rng, torch.random.get_rng_state())
    return result, prefixes, visuals


@pytest.mark.parametrize('textual_states', ['previous', 'all'])
@pytest.mark.parametrize('explicit_legacy', [False, True])
def test_legacy_runtime_matches_pinned_product_blob(textual_states, explicit_legacy):
    """A changed query, selector, call count, or RNG breaks exact product parity."""
    case = make_history_case()
    cloned = deepcopy(case)
    baseline = load_baseline_target_builder(BASE)(
        case.tokenizer, textual_states=textual_states, helper_group_L=0, coverage_p=.6)
    options = {'query_config': Stage1QueryConfig()} if explicit_legacy else {}
    current = Stage1TeacherTargetBuilder(
        cloned.tokenizer, textual_states=textual_states, helper_group_L=0,
        coverage_p=.6, **options)
    old, old_prefixes, old_visuals = observe_build(case, baseline)
    new, new_prefixes, new_visuals = observe_build(cloned, current)
    torch.testing.assert_close(old.latent_values, new.latent_values, rtol=0, atol=0)
    assert torch.equal(old.latent_mask, new.latent_mask)
    assert old.selected_counts == new.selected_counts
    assert old.selected_candidate_indices == new.selected_candidate_indices
    assert old.selection_margins == new.selection_margins
    assert len(old_prefixes) == len(new_prefixes) == 1
    assert len(old_visuals) == len(new_visuals) == 2
    for old_call, new_call in zip(old_prefixes, new_prefixes):
        for key in ('inputs_embeds', 'position_ids', 'attention_mask'):
            torch.testing.assert_close(old_call[key], new_call[key], rtol=0, atol=0)
    for old_pixels, new_pixels in zip(old_visuals, new_visuals):
        torch.testing.assert_close(old_pixels, new_pixels, rtol=0, atol=0)


@pytest.mark.parametrize('diagnostics', [False, True])
def test_adaptive_actual_prefixes_inject_only_past_first_kstar_raw_targets(diagnostics):
    """Injecting means, diagnostic indices, inactive pads or future blocks fails."""
    case = make_history_case()
    target_builder = builder(case, collect_diagnostics=diagnostics, validate=False)
    result, prefixes, visuals = observe_build(case, target_builder)
    assert [call['inputs_embeds'].shape[1] for call in prefixes] == [11, 13, 19]
    assert len(visuals) == 2  # one original encode, one helper encode
    assert result.selected_counts == ((1, 1),)
    assert result.latent_mask.nonzero().tolist() == [[0, 14], [0, 20]]
    assert result.latent_values.shape == (2, 8)
    qwen = resolve_qwen2_5_vl_model(case.teacher)
    with torch.no_grad():
        token_values = qwen.model.embed_tokens(case.inputs['input_ids'])
    for prefix in prefixes:
        assert not prefix['inputs_embeds'].requires_grad
        torch.testing.assert_close(prefix['inputs_embeds'][0, :11],
                                   prefixes[0]['inputs_embeds'][0], rtol=0, atol=0)
    torch.testing.assert_close(prefixes[2]['inputs_embeds'][0, 14],
                               result.latent_values[0], rtol=0, atol=0)
    torch.testing.assert_close(prefixes[2]['inputs_embeds'][0, 15:17],
                               token_values[0, 15:17], rtol=0, atol=0)
    if not diagnostics:
        assert result.selected_candidate_indices == (((), ()),)
        assert result.diagnostics == {}
    assert all(not p.requires_grad and p.grad is None for p in case.teacher.parameters())


def test_multiple_helpers_preserve_source_order_and_existing_pooling():
    """Reordering helper images or grouping across image boundaries fails."""
    case = make_history_case()
    case.inputs['pixel_values_latent'] = torch.cat((
        case.inputs['pixel_values_latent'], case.inputs['pixel_values_latent'][:4] + 2))
    case.inputs['image_grid_thw_latent'] = case.inputs['image_grid_thw_latent'].repeat(2, 1)[:3]
    sample = replace(case.metadata.samples[0], helper_to_block=(0, 1, 0),
                     helper_grid_range=(0, 3), helper_pixel_range=(0, 12))
    case = replace(case, metadata=replace(case.metadata, samples=(sample,)))
    target_builder = RecordingBuilder(case.tokenizer,
        query_config=Stage1QueryConfig(mode='adaptive'), helper_group_L=2, coverage_p=1)
    qwen = resolve_qwen2_5_vl_model(case.teacher)
    with torch.no_grad():
        raw = qwen.visual(case.inputs['pixel_values_latent'],
                          grid_thw=case.inputs['image_grid_thw_latent'])
    pooled = raw.reshape(3, 2, 2, 8).mean(2)
    result, prefixes, visuals = observe_build(case, target_builder)
    torch.testing.assert_close(target_builder.candidates[0],
                               torch.cat((pooled[0], pooled[2])), rtol=0, atol=0)
    torch.testing.assert_close(target_builder.candidates[1], pooled[1], rtol=0, atol=0)
    assert result.selected_counts == ((2, 2),)
    torch.testing.assert_close(prefixes[2]['inputs_embeds'][0, 14:16],
                               result.latent_values[:2], rtol=0, atol=0)
    assert len(visuals) == 2


@pytest.mark.parametrize('mutation', ['future_text', 'future_latent', 'future_helper',
                                     'current_helper'])
def test_current_query_cannot_observe_current_helpers_or_future_trajectory(mutation):
    """A full-trajectory teacher pass or helper-conditioned intent leaks here."""
    case = make_history_case()
    changed = deepcopy(case)
    if mutation == 'future_text':
        changed.inputs['input_ids'][0, 17:19] = torch.tensor([44, 45])
    elif mutation == 'future_latent':
        # Extend a valid future block with one inactive pad, preserving metadata.
        x = changed.inputs
        x['input_ids'] = torch.cat((x['input_ids'][:, :22],
                                   torch.tensor([[125]]), x['input_ids'][:, 22:]), 1)
        x['attention_mask'] = torch.ones_like(x['input_ids'])
        sample = changed.metadata.samples[0]
        span = Stage1LatentSpan(19, 23, (20, 21, 22))
        sample = replace(sample, latent_spans=(sample.latent_spans[0], span),
                         unpadded_length=26)
        changed = replace(changed, metadata=replace(changed.metadata,
                                                   samples=(sample,), sequence_length=26))
    else:
        start = 4 if mutation == 'future_helper' else 0
        changed.inputs['pixel_values_latent'][start:start+4].add_(100)
    before, after = builder(case), builder(changed)
    result, prefixes, unused = observe_build(case, before)
    perturbed, changed_prefixes, unused = observe_build(changed, after)
    torch.testing.assert_close(before.queries[0], after.queries[0], rtol=0, atol=0)
    for a, b in zip(prefixes[:2], changed_prefixes[:2]):
        torch.testing.assert_close(a['inputs_embeds'], b['inputs_embeds'], rtol=0, atol=0)
    if mutation != 'current_helper':
        torch.testing.assert_close(result.latent_values[0], perturbed.latent_values[0],
                                   rtol=0, atol=0)
        assert result.selected_candidate_indices[0][0] == perturbed.selected_candidate_indices[0][0]


def test_previous_raw_target_intervention_changes_native_context_and_downstream_query():
    """Using stored means without injecting the native teacher prefix fails."""
    case = make_history_case()
    changed = deepcopy(case)

    class IntervenedBuilder(RecordingBuilder):
        def _select(self, candidates, query, max_count):
            selected, indices, margin = super()._select(candidates, query, max_count)
            if len(self.queries) == 1:
                # Fixed synthetic intervention; everything except past target held fixed.
                selected = torch.arange(1, 9, device=selected.device).float()[None]
            return selected, indices, margin

    before = builder(case)
    after = IntervenedBuilder(changed.tokenizer,
        query_config=Stage1QueryConfig(mode='adaptive'), coverage_p=.1, helper_group_L=0)
    _, original_prefixes, _ = observe_build(case, before)
    _, changed_prefixes, _ = observe_build(changed, after)
    torch.testing.assert_close(before.queries[0], after.queries[0], rtol=0, atol=0)
    assert not torch.equal(before.queries[1], after.queries[1])
    torch.testing.assert_close(after.candidates[1], before.candidates[1], rtol=0, atol=0)
    torch.testing.assert_close(changed_prefixes[2]['inputs_embeds'][0, 14],
                               torch.arange(1, 9).float(), rtol=0, atol=0)
    assert not torch.equal(original_prefixes[2]['inputs_embeds'][0, 14],
                           changed_prefixes[2]['inputs_embeds'][0, 14])


def test_empty_text_first_block_and_empty_later_segment_have_no_attention_mass():
    """Empty current text must not reuse older query positions for latent attention."""
    case = make_history_case()
    # Mark all text as special without changing valid token/latent ownership.
    case.inputs['input_ids'][0, [11, 12, 17, 18]] = 7
    target_builder = builder(case)
    observe_build(case, target_builder)
    torch.testing.assert_close(target_builder.queries[0], target_builder.queries[1],
                               rtol=0, atol=0)


def test_no_latent_valid_sample_returns_empty_without_teacher_work():
    case = make_history_case()
    x = {key: value.clone() for key, value in case.inputs.items()}
    x['input_ids'] = x['input_ids'][:, :13]
    x['attention_mask'] = x['attention_mask'][:, :13]
    x['pixel_values_latent'] = x['pixel_values_latent'][:0]
    x['image_grid_thw_latent'] = x['image_grid_thw_latent'][:0]
    sample = replace(case.metadata.samples[0], latent_spans=(), helper_to_block=(),
                     helper_grid_range=(0, 0), helper_pixel_range=(0, 0), unpadded_length=13)
    case = replace(case, inputs=x, metadata=replace(case.metadata, samples=(sample,), sequence_length=13))
    result, prefixes, visuals = observe_build(case, builder(case))
    assert result.latent_values.shape == (0, 8)
    assert result.selected_counts == ((),)
    assert not result.latent_mask.any()
    assert not prefixes and not visuals


@pytest.mark.parametrize('mutation', ['missing_assistant', 'two_assistants', 'late_image',
    'metadata_span', 'marker', 'missing_candidates', 'mask_hole', 'mask_shape',
    'masked_span', 'metadata_length', 'batch2', 'trainable_teacher'])
def test_malformed_adaptive_inputs_fail_even_when_validation_disabled(mutation):
    """Debug-off must not permit trajectory drift or a malformed CE-only fallback."""
    case = make_history_case()
    sample = case.metadata.samples[0]
    if mutation == 'missing_assistant':
        case.inputs['input_ids'][0, 8] = 1
    elif mutation == 'two_assistants':
        case.inputs['input_ids'][0, 17:19] = torch.tensor([31, 32])
    elif mutation == 'late_image':
        # Move header before original image; adapter alone accepts this structure.
        case.inputs['input_ids'][0, :8] = torch.tensor([1, 31, 32, 81, 80, 80, 80, 80])
        case.inputs['input_ids'][0, 8:11] = torch.tensor([83, 10, 10])
    elif mutation == 'metadata_span':
        sample = replace(sample, latent_spans=(replace(sample.latent_spans[0], start=12),
                                              sample.latent_spans[1]))
    elif mutation == 'marker':
        case.inputs['input_ids'][0, 15] = 126
    elif mutation == 'missing_candidates':
        sample = replace(sample, helper_to_block=(0, 0))
    elif mutation == 'mask_hole':
        case.inputs['attention_mask'][0, 11] = 0
    elif mutation == 'mask_shape':
        case.inputs['attention_mask'] = case.inputs['attention_mask'][:, :-1]
    elif mutation == 'masked_span':
        case.inputs['attention_mask'][0, 14] = 0
    elif mutation == 'metadata_length':
        sample = replace(sample, unpadded_length=24)
    elif mutation == 'batch2':
        case = replace(case, metadata=replace(case.metadata, batch_size=2))
    else:
        next(case.teacher.parameters()).requires_grad_(True)
    if mutation != 'batch2':
        case = replace(case, metadata=replace(case.metadata, samples=(sample,)))
    with pytest.raises((ValueError, FloatingPointError)):
        observe_build(case, builder(case, validate=False, collect_diagnostics=False))


@pytest.mark.parametrize('window', [10, 12, 18])
def test_every_adaptive_prefix_is_full_causal_before_any_language_call(window):
    """Prompt/first-block empty observer sets cannot bypass native window checks."""
    case = make_history_case()
    qwen = resolve_qwen2_5_vl_model(case.teacher)
    qwen.model.config.sliding_window = window
    calls = []
    handle = qwen.model.register_forward_pre_hook(lambda m, a: calls.append(a))
    try:
        with pytest.raises(ValueError, match='full.causal|sliding_window'):
            observe_build(case, builder(case))
    finally:
        handle.remove()
    assert not calls
    assert qwen.model.config.sliding_window == window


def test_exact_window_boundary_preserves_native_config():
    case = make_history_case()
    qwen = resolve_qwen2_5_vl_model(case.teacher)
    qwen.model.config.sliding_window = 19
    result, prefixes, unused = observe_build(case, builder(case))
    assert result.selected_counts == ((1, 1),)
    assert len(prefixes) == 3
    assert qwen.model.config.sliding_window == 19


@pytest.mark.parametrize('mutation', ['orphan_image', 'video', 'missing_vision_start'])
def test_no_latent_empty_return_still_rejects_malformed_original_layout(mutation):
    """The no-work return must not bypass original-image or video invariants."""
    case = make_history_case()
    x = {key: value.clone() for key, value in case.inputs.items()}
    x['input_ids'] = x['input_ids'][:, :13]
    x['attention_mask'] = x['attention_mask'][:, :13]
    x['pixel_values_latent'] = x['pixel_values_latent'][:0]
    x['image_grid_thw_latent'] = x['image_grid_thw_latent'][:0]
    if mutation == 'video':
        x['input_ids'][0, 7] = 82
    elif mutation == 'missing_vision_start':
        x['input_ids'][0, 1] = 10
    else:
        x['input_ids'][0, 2:7] = torch.tensor([10, 80, 80, 80, 80])
    sample = replace(case.metadata.samples[0], latent_spans=(), helper_to_block=(),
                     helper_grid_range=(0, 0), helper_pixel_range=(0, 0), unpadded_length=13)
    case = replace(case, inputs=x, metadata=replace(case.metadata, samples=(sample,), sequence_length=13))
    with pytest.raises(ValueError):
        observe_build(case, builder(case, validate=False))


def test_builder_query_matches_independent_native_hidden_and_mass_reduction(monkeypatch):
    """Wrong text discount, latent raw values, query positions or mass fusion fails."""
    import stage1.targets as module
    case = make_history_case(layers=2)
    case.inputs['input_ids'][0, 18] = 7  # second segment contains one valid text token
    qwen = resolve_qwen2_5_vl_model(case.teacher)
    hidden, reduced = [], []
    handle = qwen.model.register_forward_hook(lambda m, a, out: hidden.append(
        out.last_hidden_state.detach().clone()))
    real_observer = module.ReducedHistoryAttentionObserver

    class CaptureObserver(real_observer):
        def scores(self):
            scores = super().scores()
            reduced.append((self.query_positions, self.latent_positions, scores.clone()))
            return scores

    class CaptureIntent(RecordingBuilder):
        def _user_query(self, **kwargs):
            self.intent = super()._user_query(**kwargs).clone()
            return self.intent

    monkeypatch.setattr(module, 'ReducedHistoryAttentionObserver', CaptureObserver)
    target_builder = CaptureIntent(case.tokenizer,
        query_config=Stage1QueryConfig(mode='adaptive', attention_layers='all'),
        coverage_p=.1, helper_group_L=0)
    try:
        observe_build(case, target_builder)
    finally:
        handle.remove()
    assert len(hidden) == 3
    assert reduced[0][:2] == ((11, 12), ())
    assert reduced[1][:2] == ((17,), (14,))
    ell0 = hidden[1][0, [11, 12]].float().mean(0, keepdim=True)
    ell1 = hidden[2][0, [17]].float().mean(0, keepdim=True)
    scores = reduced[1][2]
    assert 0 < float(scores.sum()) < 1
    expected0 = (target_builder.intent.float() + ell0) / 2
    body = (hidden[2][0, [14]].float() * scores[:, None]).sum(0, keepdim=True)
    expected1 = (target_builder.intent.float() + (.9 * ell0 + ell1) / 1.9 + body) / (2 + scores.sum())
    torch.testing.assert_close(target_builder.queries[0], expected0, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(target_builder.queries[1], expected1, rtol=1e-6, atol=1e-7)


def test_known_cumulative_prefix_dependency_changes_downstream_selection(monkeypatch):
    """A deterministic causal teacher separates injected state from stored query math."""
    import stage1.targets as module
    from stage1.history_teacher import HistoryPrefixObservation
    case = make_history_case()
    qwen = resolve_qwen2_5_vl_model(case.teacher)
    # A causal cumulative-sum native-boundary fixture: first target is the only
    # nonzero context input. Current segment and active-pad hidden both equal it.
    with torch.no_grad():
        for parameter in qwen.model.embed_tokens.parameters():
            parameter.zero_()
    e0, e1 = torch.eye(8)[:2]

    def visual(pixels, *, grid_thw):
        if pixels.shape[0] == 4:
            return torch.zeros((4, 8))
        return torch.stack((e0, e1, e0, e1, e0, e1, e0, e1))

    monkeypatch.setattr(qwen.visual, 'forward', visual)
    monkeypatch.setattr(module.HistoryPrefixAdapter, 'forward',
        lambda adapter, prefix: HistoryPrefixObservation(prefix.inputs_embeds.cumsum(1)))

    class KnownMass:
        def __init__(self, backbone, *, latent_positions, **kwargs):
            self.n = len(latent_positions)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def scores(self):
            return torch.full((self.n,), .25)

    monkeypatch.setattr(module, 'ReducedHistoryAttentionObserver', KnownMass)

    class ControlledPreviousTarget(RecordingBuilder):
        def __init__(self, value):
            super().__init__(case.tokenizer, query_config=Stage1QueryConfig(mode='adaptive'),
                             coverage_p=.1, helper_group_L=0)
            self.value = value
        def _user_query(self, **kwargs):
            return e0[None]
        def _select(self, candidates, query, max_count):
            selected, indices, margin = super()._select(candidates, query, max_count)
            return (self.value[None] if len(self.queries) == 1 else selected), indices, margin

    first = ControlledPreviousTarget(4 * e0)
    second = ControlledPreviousTarget(4 * e1)
    a, _, _ = observe_build(case, first)
    b, _, _ = observe_build(case, second)
    torch.testing.assert_close(first.queries[0], e0[None] / 2, rtol=0, atol=0)
    torch.testing.assert_close(second.queries[0], first.queries[0], rtol=0, atol=0)
    torch.testing.assert_close(first.queries[1],
        (e0 + 4 * e0 / 1.9 + e0)[None] / 2.25)
    torch.testing.assert_close(second.queries[1],
        (e0 + 4 * e1 / 1.9 + e1)[None] / 2.25)
    assert a.selected_candidate_indices[0][1] == (0,)
    assert b.selected_candidate_indices[0][1] == (1,)
    torch.testing.assert_close(a.latent_values[1], e0, rtol=0, atol=0)
    torch.testing.assert_close(b.latent_values[1], e1, rtol=0, atol=0)


def test_builder_does_not_reuse_text_or_selection_state_across_builds():
    """A stored text state or stale target ledger changes the second build."""
    case = make_history_case()
    target_builder = builder(case)
    first, first_prefixes, first_visuals = observe_build(case, target_builder)
    second, second_prefixes, second_visuals = observe_build(case, target_builder)
    torch.testing.assert_close(first.latent_values, second.latent_values, rtol=0, atol=0)
    assert torch.equal(first.latent_mask, second.latent_mask)
    assert len(first_visuals) == len(second_visuals) == 2
    for before, after in zip(first_prefixes, second_prefixes):
        torch.testing.assert_close(before['inputs_embeds'], after['inputs_embeds'], rtol=0, atol=0)
    for before, after in zip(target_builder.queries[:2], target_builder.queries[2:]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
