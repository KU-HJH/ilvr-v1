"""Runtime target parity, placeholder observation and external-history causality and fail-closed ownership."""
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


def test_adaptive_observation_equals_literal_ilvr_teacher_call():
    """Observe the actual wrapper and decoder, independently of builder helpers."""
    case = make_history_case()
    qwen = resolve_qwen2_5_vl_model(case.teacher)
    inputs_before = {key: value.clone() for key, value in case.inputs.items()}
    rng = torch.random.get_rng_state().clone()
    decoder_calls, wrapper_hidden = [], []

    def decoder_input(module, args, kwargs):
        decoder_calls.append({key: kwargs[key].detach().clone()
                              for key in ('inputs_embeds', 'attention_mask', 'position_ids')})

    def wrapper_output(module, args, output):
        hidden = output.hidden_states
        wrapper_hidden.append((hidden[-1] if isinstance(hidden, (tuple, list))
                               else hidden).detach().clone())

    handles = [qwen.model.register_forward_pre_hook(decoder_input, with_kwargs=True),
               case.teacher.register_forward_hook(wrapper_output)]
    try:
        # Literal pre-change ILVR call: no builder or prefix utility participates.
        with torch.no_grad():
            case.teacher(input_ids=case.inputs['input_ids'],
                         attention_mask=case.inputs['attention_mask'],
                         pixel_values=case.inputs['pixel_values'],
                         image_grid_thw=case.inputs['image_grid_thw'],
                         output_hidden_states=True, return_dict=True)
        assert len(decoder_calls) == len(wrapper_hidden) == 1
        literal_input, literal_hidden = decoder_calls.pop(), wrapper_hidden.pop()
        result, prefixes, visuals = observe_build(case, builder(case))
        assert len(decoder_calls) == len(wrapper_hidden) == len(prefixes) == 1
        assert decoder_calls[0]['inputs_embeds'].shape[1] == case.inputs['input_ids'].shape[1]
        for key in literal_input:
            torch.testing.assert_close(decoder_calls[0][key], literal_input[key], rtol=0, atol=0)
        torch.testing.assert_close(wrapper_hidden[0], literal_hidden, rtol=0, atol=0)
        with torch.no_grad():
            token_rows = qwen.model.embed_tokens(case.inputs['input_ids'])
        for span in case.metadata.samples[0].latent_spans:
            positions = list(span.latent_positions)
            torch.testing.assert_close(decoder_calls[0]['inputs_embeds'][0, positions],
                                       token_rows[0, positions], rtol=0, atol=0)
        assert len(visuals) == 2
        assert not result.latent_values.requires_grad and result.latent_values.grad_fn is None
        assert all(not p.requires_grad and p.grad is None for p in case.teacher.parameters())
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in inputs_before.items():
        torch.testing.assert_close(case.inputs[key], value, rtol=0, atol=0)


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
def test_adaptive_full_sequence_keeps_all_latent_pad_embeddings(diagnostics):
    """Raw selected rows occupy first-K target slots, never Teacher input slots."""
    case = make_history_case()
    target_builder = builder(case, collect_diagnostics=diagnostics, validate=False)
    result, prefixes, visuals = observe_build(case, target_builder)
    assert [call['inputs_embeds'].shape[1] for call in prefixes] == [25]
    assert len(visuals) == 2  # one original encode, one helper encode
    assert result.selected_counts == ((1, 1),)
    assert result.latent_mask.nonzero().tolist() == [[0, 14], [0, 20]]
    assert result.latent_values.shape == (2, 8)
    qwen = resolve_qwen2_5_vl_model(case.teacher)
    with torch.no_grad():
        token_values = qwen.model.embed_tokens(case.inputs['input_ids'])
    assert not prefixes[0]['inputs_embeds'].requires_grad
    for span in case.metadata.samples[0].latent_spans:
        positions = list(span.latent_positions)
        torch.testing.assert_close(prefixes[0]['inputs_embeds'][0, positions],
                                   token_values[0, positions], rtol=0, atol=0)
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
    assert result.latent_mask.nonzero().tolist() == [[0, 14], [0, 15], [0, 20], [0, 21]]
    for selected, packed in zip(target_builder.selections, result.latent_values.split(2)):
        torch.testing.assert_close(selected, packed, rtol=0, atol=0)
    with torch.no_grad():
        pads = qwen.model.embed_tokens(case.inputs['input_ids'])[0, [14, 15, 20, 21]]
    torch.testing.assert_close(prefixes[0]['inputs_embeds'][0, [14, 15, 20, 21]], pads, rtol=0, atol=0)
    assert len(visuals) == 2


@pytest.mark.parametrize('mutation', ['future_text', 'future_latent', 'future_helper',
                                     'current_helper'])
def test_current_query_cannot_observe_current_helpers_or_future_trajectory(mutation):
    """Native causal Teacher rows before START exclude future text and all helpers."""
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
    hidden = []
    def capture(module, args, out):
        value = out.hidden_states
        hidden.append((value[-1] if isinstance(value, (tuple, list)) else value).clone())
    handles = [case.teacher.register_forward_hook(capture), changed.teacher.register_forward_hook(capture)]
    try:
        result, prefixes, unused = observe_build(case, before)
        perturbed, changed_prefixes, unused = observe_build(changed, after)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(before.queries[0], after.queries[0], rtol=0, atol=0)
    # Full observations may differ at future positions; only earlier causal rows
    # support the first decision, and must remain bit-identical.
    torch.testing.assert_close(prefixes[0]['inputs_embeds'][:, :13],
                               changed_prefixes[0]['inputs_embeds'][:, :13], rtol=0, atol=0)
    torch.testing.assert_close(hidden[0][:, :13], hidden[1][:, :13], rtol=0, atol=0)
    if mutation in ('current_helper', 'future_helper'):
        torch.testing.assert_close(hidden[0], hidden[1], rtol=0, atol=0)
    if mutation != 'current_helper':
        torch.testing.assert_close(result.latent_values[0], perturbed.latent_values[0],
                                   rtol=0, atol=0)
        assert result.selected_candidate_indices[0][0] == perturbed.selected_candidate_indices[0][0]


def test_previous_raw_target_intervention_changes_only_external_history_and_query(monkeypatch):
    """Hold every Teacher input/weight/state fixed and intervene only on past raw values."""
    import stage1.targets as module
    case = make_history_case()
    e0, e1 = torch.eye(8)[:2]
    histories, hidden = [], []
    actual_metric = module.content_history
    def capture_metric(text, values, **kwargs):
        result = actual_metric(text, values, **kwargs)
        histories.append((text.clone(), values.clone(), result))
        return result
    monkeypatch.setattr(module, 'content_history', capture_metric)
    def capture_hidden(module, args, output):
        value = output.hidden_states
        hidden.append((value[-1] if isinstance(value, (tuple, list)) else value).clone())

    class IntervenedBuilder(RecordingBuilder):
        def __init__(self, value):
            super().__init__(case.tokenizer, query_config=Stage1QueryConfig(mode='adaptive'),
                             coverage_p=.1, helper_group_L=0)
            self.value = value
        def _select(self, candidates, query, max_count):
            selected, indices, margin = super()._select(candidates, query, max_count)
            if len(self.queries) == 1:
                selected = self.value[None]
            return selected, indices, margin

    first, second = IntervenedBuilder(4 * e0), IntervenedBuilder(4 * e1)
    handle = case.teacher.register_forward_hook(capture_hidden)
    try:
        a, original, _ = observe_build(case, first)
        b, changed, _ = observe_build(case, second)
    finally:
        handle.remove()
    torch.testing.assert_close(a.latent_values[0], 4 * e0, rtol=0, atol=0)
    torch.testing.assert_close(b.latent_values[0], 4 * e1, rtol=0, atol=0)
    assert len(original) == len(changed) == len(hidden) // 2 == 1
    for key in ('inputs_embeds', 'attention_mask', 'position_ids'):
        torch.testing.assert_close(original[0][key], changed[0][key], rtol=0, atol=0)
    torch.testing.assert_close(hidden[0], hidden[1], rtol=0, atol=0)
    torch.testing.assert_close(first.queries[0], second.queries[0], rtol=0, atol=0)
    torch.testing.assert_close(histories[1][0], histories[3][0], rtol=0, atol=0)
    torch.testing.assert_close(histories[1][1], (4 * e0)[None], rtol=0, atol=0)
    torch.testing.assert_close(histories[3][1], (4 * e1)[None], rtol=0, atol=0)
    assert histories[1][2].present and histories[3][2].present
    assert not torch.equal(histories[1][2].summary, histories[3][2].summary)
    assert not torch.equal(first.queries[1], second.queries[1])
    torch.testing.assert_close(first.candidates[1], second.candidates[1], rtol=0, atol=0)


def test_empty_text_first_block_and_empty_later_segment_omit_external_history():
    """Empty current text must not reuse older query positions for external history."""
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


@pytest.mark.parametrize('window', [10, 12, 18, 19])
def test_adaptive_preserves_literal_native_sliding_window_masks(window):
    case = make_history_case()
    qwen = resolve_qwen2_5_vl_model(case.teacher)
    qwen.model.config.sliding_window = window
    literal = []
    handle = qwen.model.register_forward_pre_hook(
        lambda m, a, kw: literal.append(kw['attention_mask'].clone()), with_kwargs=True)
    try:
        with torch.no_grad():
            case.teacher(input_ids=case.inputs['input_ids'], attention_mask=case.inputs['attention_mask'],
                pixel_values=case.inputs['pixel_values'], image_grid_thw=case.inputs['image_grid_thw'],
                output_hidden_states=True, return_dict=True)
    finally:
        handle.remove()
    result, calls, _ = observe_build(case, builder(case))
    assert result.selected_counts == ((1, 1),) and len(calls) == 1
    torch.testing.assert_close(calls[0]['attention_mask'], literal[0], rtol=0, atol=0)
    assert qwen.model.config.sliding_window == window


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


def test_builder_query_matches_independent_placeholder_hidden_and_raw_cosine_reduction():
    """Literal normalized current text, segment votes and raw past rows define queries."""
    case = make_history_case(layers=2)
    case.inputs['input_ids'][0, 18] = 7  # second segment has one valid text token
    hidden = []
    def capture(module, args, out):
        value = out.hidden_states
        hidden.append((value[-1] if isinstance(value, (tuple, list)) else value).clone())

    class CaptureIntent(RecordingBuilder):
        def _user_query(self, **kwargs):
            self.intent = super()._user_query(**kwargs).clone()
            return self.intent

    target_builder = CaptureIntent(case.tokenizer,
        query_config=Stage1QueryConfig(mode='adaptive'), coverage_p=1, helper_group_L=0)
    handle = case.teacher.register_forward_hook(capture)
    try:
        result, _, _ = observe_build(case, target_builder)
    finally:
        handle.remove()
    assert len(hidden) == 1
    ell0 = hidden[0][0, [11, 12]].float().mean(0, keepdim=True)
    text1 = hidden[0][0, [17]].float()
    ell1 = text1.mean(0, keepdim=True)
    previous = result.latent_values[:result.selected_counts[0][0]].float()
    assert previous.shape[0] > 1  # distinguishes relevance-weighting from a raw mean
    weights = torch.softmax(torch.nn.functional.normalize(text1, dim=1) @
                            torch.nn.functional.normalize(previous, dim=1).T, dim=1).mean(0)
    body = (weights[:, None] * previous).sum(0, keepdim=True)
    expected0 = (target_builder.intent.float() + ell0) / 2
    expected1 = (target_builder.intent.float() + (.9 * ell0 + ell1) / 1.9 + body) / 3
    torch.testing.assert_close(target_builder.queries[0], expected0, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(target_builder.queries[1], expected1, rtol=1e-6, atol=1e-7)


def test_adaptive_never_invokes_historical_prefix_or_attention_utilities(monkeypatch):
    from stage1.history_teacher import HistoryPrefixAdapter
    from stage1.history_attention import ReducedHistoryAttentionObserver
    def forbidden(*args, **kwargs):
        pytest.fail('historical native attention utility reached from current adaptive')
    monkeypatch.setattr(HistoryPrefixAdapter, '__init__', forbidden)
    monkeypatch.setattr(ReducedHistoryAttentionObserver, '__init__', forbidden)
    case = make_history_case()
    result, calls, visuals = observe_build(case, builder(case))
    assert result.selected_counts == ((1, 1),) and len(calls) == 1 and len(visuals) == 2


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


@pytest.mark.parametrize('selector', ['last4', 'all'])
def test_direct_builder_rejects_nonfinal_adaptive_before_teacher_work(selector):
    case = make_history_case()
    calls = []
    handle = case.teacher.register_forward_pre_hook(lambda *args: calls.append(args))
    try:
        with pytest.raises(ValueError, match='final|last.only'):
            Stage1TeacherTargetBuilder(case.tokenizer,
                query_config=Stage1QueryConfig('adaptive', .9, selector))
    finally:
        handle.remove()
    assert calls == []


def test_empty_current_text_omits_history_but_keeps_discounted_prior_text(monkeypatch):
    import stage1.targets as module
    case = make_history_case()
    case.inputs['input_ids'][0, [17, 18]] = 7
    calls = []
    actual = module.content_history
    def capture(text, values, **kwargs):
        result = actual(text, values, **kwargs)
        calls.append((text.clone(), values.clone(), result))
        return result
    monkeypatch.setattr(module, 'content_history', capture)
    target_builder = builder(case)
    observe_build(case, target_builder)
    assert calls[0][0].shape == (2, 8) and calls[0][1].shape == (0, 8)
    assert calls[1][0].shape == (0, 8) and calls[1][1].shape == (1, 8)
    assert not calls[0][2].present and not calls[1][2].present
    torch.testing.assert_close(target_builder.queries[0], target_builder.queries[1])
