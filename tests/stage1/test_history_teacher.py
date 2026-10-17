"""Catch lost effective DoRA, image ownership, positions and prefix injection."""
import importlib
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from peft_common import resolve_qwen2_5_vl_model
from tests.stage1.history_fixtures import make_history_case


def api():
    assert importlib.util.find_spec('stage1.history_teacher') is not None, (
        'missing effective DoRA history-prefix adapter')
    return importlib.import_module('stage1.history_teacher')


def adapter_for(case):
    x = case.inputs
    return api().HistoryPrefixAdapter(
        case.teacher, input_ids=x['input_ids'], attention_mask=x['attention_mask'],
        original_pixels=x['pixel_values'], original_grid=x['image_grid_thw'])


def add_second_original(case):
    x = {key: value.clone() for key, value in case.inputs.items()}
    x['input_ids'] = torch.cat((x['input_ids'][:, :7],
                               torch.tensor([[81, 80, 80, 80, 80, 83]]),
                               x['input_ids'][:, 7:]), dim=1)
    x['attention_mask'] = torch.ones_like(x['input_ids'])
    x['pixel_values'] = torch.cat((x['pixel_values'], x['pixel_values'] + .25))
    x['image_grid_thw'] = x['image_grid_thw'].repeat(2, 1)
    sample = case.metadata.samples[0]
    spans = tuple(replace(span, start=span.start+6, end=span.end+6,
                          latent_positions=tuple(p+6 for p in span.latent_positions))
                  for span in sample.latent_spans)
    sample = replace(sample, latent_spans=spans, unpadded_length=31,
                     original_grid_range=(0, 2), original_pixel_range=(0, 8))
    return replace(case, inputs=x, metadata=replace(
        case.metadata, samples=(sample,), sequence_length=31))


@pytest.mark.parametrize(('selected', 'images'), [(0, 1), (1, 1), (2, 1), (1, 2)])
def test_prefix_matches_native_effective_dora_wrapper(selected, images):
    """A plain projection, helper injection, full trajectory or arange RoPE fails."""
    h = api()
    case = make_history_case(layers=2)
    if images == 2:
        case = add_second_original(case)
    x = case.inputs
    snapshots = {name: value.clone() for name, value in x.items()}
    qwen = resolve_qwen2_5_vl_model(case.teacher)
    projection = qwen.model.layers[0].self_attn.q_proj
    probe = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 7
    with torch.no_grad():
        assert not torch.allclose(projection(probe), projection.base_layer(probe))
    spans = case.metadata.samples[0].latent_spans
    stop = spans[0 if selected == 0 else 1].start
    positions = spans[0].latent_positions[:selected]
    values = torch.arange(selected*8, dtype=torch.float32).reshape(selected, 8) / 10
    ledger = (() if not selected else (h.HistorySelection(
        0, positions, values.requires_grad_(True), tuple(range(selected))),))
    native_mask = torch.zeros((1, stop), dtype=torch.bool)
    native_mask[0, list(positions)] = True
    captured = []
    handle = qwen.model.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append({
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()}), with_kwargs=True)
    try:
        with torch.no_grad():
            native = case.teacher(
                input_ids=x['input_ids'][:, :stop],
                attention_mask=x['attention_mask'][:, :stop],
                pixel_values=x['pixel_values'], image_grid_thw=x['image_grid_thw'],
                latent_hidden_states=values.detach(), image_out_mask=native_mask,
                use_cache=False, output_hidden_states=False,
                output_attentions=False, return_dict=True)
    finally:
        handle.remove()
    assert len(captured) == 1
    native_kwargs = captured[0]
    qwen.rope_deltas = torch.tensor([[777]])
    rope_before = qwen.rope_deltas
    rng_before = torch.random.get_rng_state().clone()
    visual_calls, lm_calls, decoder_calls, grad_modes = [], [], [], []
    hooks = [
        qwen.visual.register_forward_pre_hook(lambda m, a: visual_calls.append(a[0].clone())),
        qwen.lm_head.register_forward_pre_hook(lambda m, a: lm_calls.append(a)),
        qwen.model.register_forward_pre_hook(
            lambda m, a, kw: decoder_calls.append(kw), with_kwargs=True),
        qwen.model.register_forward_pre_hook(
            lambda m, a: grad_modes.append(torch.is_grad_enabled())),
    ]
    try:
        adapter = adapter_for(case)
        prefix = adapter.prepare(stop, ledger)
        result = adapter.forward(prefix)
        again = adapter.prepare(stop, ledger)
    finally:
        for hook in hooks:
            hook.remove()
    assert adapter.backbone is qwen.model
    assert len(visual_calls) == 1
    torch.testing.assert_close(visual_calls[0], x['pixel_values'], rtol=0, atol=0)
    assert not lm_calls
    assert len(decoder_calls) == 1
    assert grad_modes == [False]
    for key, want in {'input_ids': None, 'use_cache': False,
                      'output_hidden_states': False, 'output_attentions': False,
                      'return_dict': True}.items():
        assert decoder_calls[0][key] is want
    assert prefix.input_ids.shape == (1, stop)
    assert prefix.inputs_embeds.shape == (1, stop, 8)
    assert prefix.position_ids.shape == (3, 1, stop)
    assert not torch.equal(prefix.position_ids[0], prefix.position_ids[2])
    for key in ('inputs_embeds', 'position_ids', 'attention_mask'):
        torch.testing.assert_close(getattr(prefix, key), native_kwargs[key],
                                   atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(result.last_hidden_state, native.hidden_states,
                               atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(again.inputs_embeds, prefix.inputs_embeds, rtol=0, atol=0)
    assert qwen.rope_deltas is rope_before
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert not prefix.inputs_embeds.requires_grad
    assert not result.last_hidden_state.requires_grad
    assert values.grad is None
    assert all(not p.requires_grad and p.grad is None for p in case.teacher.parameters())
    for key, before in snapshots.items():
        torch.testing.assert_close(x[key], before, rtol=0, atol=0)
    # Prefix tensors own storage; consumer edits cannot corrupt inputs or cache.
    prefix.inputs_embeds.zero_()
    torch.testing.assert_close(adapter.prepare(stop, ledger).inputs_embeds,
                               again.inputs_embeds, rtol=0, atol=0)
    prefix.input_ids.zero_()
    prefix.attention_mask.zero_()
    assert torch.equal(x['input_ids'], snapshots['input_ids'])
    assert torch.equal(x['attention_mask'], snapshots['attention_mask'])
    with pytest.raises(FrozenInstanceError):
        result.last_hidden_state = torch.empty(0)


def test_helper_changes_cannot_change_original_prefix():
    """Helpers belong to candidate extraction, never language context."""
    case = make_history_case()
    before = adapter_for(case).prepare(19, ())
    case.inputs['pixel_values_latent'].add_(1000)
    after = adapter_for(case).prepare(19, ())
    torch.testing.assert_close(before.inputs_embeds, after.inputs_embeds, rtol=0, atol=0)


@pytest.mark.parametrize('mutation', ['4d', 'nonbinary', 'hole', 'batch2',
                                     'future_image', 'feature_count', 'bad_grid',
                                     'nested_latent', 'video'])
def test_unsupported_input_profile_is_rejected(mutation):
    """Malformed image ownership/masks must fail rather than silently realign."""
    case = make_history_case()
    x = case.inputs
    if mutation == '4d':
        x['attention_mask'] = torch.ones((1, 1, 25, 25))
    elif mutation == 'nonbinary':
        x['attention_mask'][0, 1] = 2
    elif mutation == 'hole':
        x['attention_mask'][0, 3] = 0
    elif mutation == 'batch2':
        x['input_ids'] = x['input_ids'].repeat(2, 1)
        x['attention_mask'] = x['attention_mask'].repeat(2, 1)
    elif mutation == 'future_image':
        x['input_ids'][0, 17] = 80
    elif mutation == 'feature_count':
        x['input_ids'][0, 5] = 10
    elif mutation == 'bad_grid':
        x['image_grid_thw'] = torch.tensor([[1, 0, 4]])
    elif mutation == 'nested_latent':
        x['input_ids'][0, 15] = 126
    else:
        x['input_ids'][0, 7] = 82
    with pytest.raises(ValueError):
        adapter_for(case).prepare(19, ())


@pytest.mark.parametrize('mutation', ['current', 'text', 'second_pad_only',
                                     'wrong_block', 'duplicate', 'shape',
                                     'nonfinite', 'candidate_count', 'stop_inside'])
def test_selection_must_be_raw_first_kstar_pads_of_a_complete_past_block(mutation):
    h = api()
    case = make_history_case()
    stop, positions, block = 19, (14,), 0
    values, candidates = torch.ones((1, 8)), (0,)
    if mutation == 'current':
        positions, block = (20,), 1
    elif mutation == 'text':
        positions = (17,)
    elif mutation == 'second_pad_only':
        positions = (15,)
    elif mutation == 'wrong_block':
        block = 1
    elif mutation == 'shape':
        values = torch.ones((1, 7))
    elif mutation == 'nonfinite':
        values[0, 0] = float('nan')
    elif mutation == 'candidate_count':
        candidates = (0, 1)
    elif mutation == 'stop_inside':
        stop = 15
    selection = h.HistorySelection(block, positions, values, candidates)
    ledger = (selection, selection) if mutation == 'duplicate' else (selection,)
    with pytest.raises((ValueError, FloatingPointError)):
        adapter_for(case).prepare(stop, ledger)


def test_selected_values_do_not_require_optional_candidate_diagnostics():
    """Debug-off _select returns () even with nonempty target values."""
    h = api()
    case = make_history_case()
    values = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    prefix = adapter_for(case).prepare(19, (h.HistorySelection(0, (14,), values, ()),))
    torch.testing.assert_close(prefix.inputs_embeds[0, 14], values[0], rtol=0, atol=0)


def test_empty_selection_is_a_noop_without_fake_active_history():
    h = api()
    case = make_history_case()
    adapter = adapter_for(case)
    empty = h.HistorySelection(0, (), torch.empty((0, 8)), ())
    before = adapter.prepare(19, ())
    after = adapter.prepare(19, (empty,))
    torch.testing.assert_close(after.inputs_embeds, before.inputs_embeds, rtol=0, atol=0)
    torch.testing.assert_close(adapter.forward(after).last_hidden_state,
                               adapter.forward(before).last_hidden_state, rtol=0, atol=0)
    for invalid in (
        (h.HistorySelection(1, (), torch.empty((0, 8)), ()),),
        (h.HistorySelection(0, (), torch.empty((0, 7)), ()),),
        (empty, empty),
    ):
        with pytest.raises(ValueError):
            adapter.prepare(19, invalid)


@pytest.mark.parametrize('state', ['train', 'requires_grad', 'stale_grad'])
def test_nonfrozen_teacher_is_rejected_without_changing_it(state):
    case = make_history_case()
    parameter = next(case.teacher.parameters())
    if state == 'train':
        case.teacher.train()
    elif state == 'requires_grad':
        parameter.requires_grad_(True)
    else:
        parameter.grad = torch.ones_like(parameter)
    with pytest.raises(ValueError, match='frozen|eval'):
        adapter_for(case)
    assert case.teacher.training == (state == 'train')
    assert parameter.requires_grad == (state == 'requires_grad')
    assert (parameter.grad is not None) == (state == 'stale_grad')
