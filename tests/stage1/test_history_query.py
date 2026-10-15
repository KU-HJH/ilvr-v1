"""History queries preserve segment weights and raw latent attention mass."""
from dataclasses import FrozenInstanceError, asdict
import importlib

import pytest
import torch

from stage1.types import Stage1LatentSpan


def api():
    return importlib.import_module('stage1.query')


def test_public_cli_fields_resolve_and_defaults_stay_legacy():
    q = api()
    config = q.resolve_query_config({})
    assert asdict(config) == {
        'mode': 'legacy', 'text_gamma': .9, 'attention_layers': 'last',
        'query_chunk_size': 64, 'key_chunk_size': 2048, 'head_chunk_size': 4,
    }
    adaptive = q.resolve_query_config({
        'query_mode': 'adaptive', 'history_text_gamma': .4,
        'history_attention_layers': 'last4',
    })
    assert (adaptive.mode, adaptive.text_gamma, adaptive.attention_layers) == (
        'adaptive', .4, 'last4')
    with pytest.raises(FrozenInstanceError):
        adaptive.mode = 'legacy'


@pytest.mark.parametrize('values', [
    {'history_query_mode': 'adaptive'}, {'mode': 'adaptive'},
    {'query_mode': 'unknown'}, {'history_attention_layers': 'middle'},
    *({'history_text_gamma': gamma} for gamma in (
        -.01, 1.01, float('nan'), float('inf'), -float('inf'), True, '0.9')),
])
def test_invalid_public_config_is_rejected(values):
    with pytest.raises(ValueError):
        api().resolve_query_config(values)


@pytest.mark.parametrize(('selector', 'count', 'expected'), [
    ('last', 6, (5,)), ('last4', 6, (2, 3, 4, 5)),
    ('last4', 4, (0, 1, 2, 3)), ('all', 3, (0, 1, 2)),
])
def test_layer_selection_resolves_actual_decoder_indices(selector, count, expected):
    assert api().resolve_query_layers(selector, count) == expected


@pytest.mark.parametrize(('selector', 'count'), [
    ('last4', 1), ('last4', 2), ('last4', 3),
    ('middle', 4), ('last', 0), ('last', -1), ('last', True), ('all', 2.5),
])
def test_invalid_layer_selection_fails(selector, count):
    with pytest.raises(ValueError):
        api().resolve_query_layers(selector, count)


@pytest.mark.parametrize('indices', [(0,), (0, 1), (0, 1, 2), (3, 4, 5)])
def test_last4_cannot_serialize_fewer_than_four_resolved_layers(indices):
    q = api()
    with pytest.raises(ValueError, match='last4|four|4'):
        q.query_execution_config(q.Stage1QueryConfig('adaptive', .55, 'last4'), indices)


@pytest.mark.parametrize('indices', [[0], [0, 1], [0, 1, 2], [3, 4, 5]])
def test_saved_last4_cannot_restore_fewer_than_four_resolved_layers(indices):
    saved = adaptive_saved()
    saved.update(attention_layers='last4', resolved_layer_indices=indices)
    with pytest.raises(ValueError, match='last4|four|4'):
        api().resolve_query_config({}, saved=saved)


def adaptive_saved():
    return {
        'schema_version': 1, 'mode': 'adaptive',
        'formula_id': 'stage1_injected_prefix_discounted_segments_attention_mass_v1',
        'text_mode': 'discounted_segments', 'latent_mode': 'attention_mass',
        'text_gamma': .9, 'attention_layers': 'last', 'resolved_layer_indices': [5],
        'query_chunk_size': 64, 'key_chunk_size': 2048, 'head_chunk_size': 4,
        'mask_mode': 'full_causal_2d_v1',
    }


def test_execution_schema_and_resume_preserve_formula_and_explicit_values():
    q = api()
    legacy = {'schema_version': 1, 'mode': 'legacy', 'formula_id': 'stage1_legacy_query_v1'}
    assert q.query_execution_config(q.Stage1QueryConfig(), (5,)) == legacy
    assert q.resolve_query_config({}, saved=legacy).mode == 'legacy'
    saved = adaptive_saved()
    config = q.resolve_query_config({}, saved=saved)
    assert q.query_execution_config(config, (5,)) == saved
    assert q.resolve_query_config({'query_mode': 'adaptive', 'history_text_gamma': .9}, saved=saved) == config
    with pytest.raises(ValueError, match='saved|resume|conflict'):
        q.resolve_query_config({'history_text_gamma': .5}, saved=saved)
    with pytest.raises(ValueError, match='saved|resume|conflict'):
        q.resolve_query_config({'query_mode': 'adaptive'}, saved=legacy)


@pytest.mark.parametrize(('key', 'value'), [
    ('schema_version', 2), ('schema_version', True), ('formula_id', 'other'),
    ('text_mode', 'token_mean'), ('latent_mode', 'normalized'),
    ('mask_mode', 'other'), ('resolved_layer_indices', []),
    ('resolved_layer_indices', [1, 1]), ('resolved_layer_indices', [-1]),
    ('resolved_layer_indices', [1.5]), ('resolved_layer_indices', [True]),
    ('resolved_layer_indices', (5,)), ('query_chunk_size', 0),
    ('key_chunk_size', True), ('head_chunk_size', 1.5), ('extra', 1),
])
def test_invalid_saved_execution_schema_fails(key, value):
    saved = adaptive_saved()
    saved[key] = value
    with pytest.raises(ValueError):
        api().resolve_query_config({}, saved=saved)


@pytest.mark.parametrize('key', list(adaptive_saved()))
def test_incomplete_saved_execution_schema_fails(key):
    saved = adaptive_saved()
    del saved[key]
    with pytest.raises(ValueError):
        api().resolve_query_config({}, saved=saved)


@pytest.mark.parametrize('values', [
    {'mode': 'other'}, {'text_gamma': float('nan')}, {'attention_layers': 'middle'},
    {'query_chunk_size': 0}, {'key_chunk_size': True}, {'head_chunk_size': 2.5},
])
def test_direct_config_cannot_bypass_validation(values):
    with pytest.raises(ValueError):
        api().Stage1QueryConfig(**values)


def test_history_text_uses_content_before_each_start_and_excludes_specials_and_padding():
    ids = torch.tensor([90, 91, 92, 10, 99, 11, 80, 81, 82, 12, 99, 13, 80, 81, 82, 14, 0])
    mask = torch.tensor([1] * 11 + [0] + [1] * 4 + [0])
    spans = (Stage1LatentSpan(6, 8, (7,)), Stage1LatentSpan(12, 14, (13,)))
    assert api().history_text_positions(ids, mask, spans, 3, frozenset({99, 80, 81, 82})) == (
        (3, 5), (9,))
    assert api().history_text_positions(ids, mask, (), 3, frozenset()) == ()


def test_adjacent_latent_blocks_produce_empty_text_segments():
    spans = (Stage1LatentSpan(1, 3, (2,)), Stage1LatentSpan(4, 6, (5,)))
    assert api().history_text_positions(torch.arange(8), torch.ones(8), spans, 1, frozenset()) == ((), ())


@pytest.mark.parametrize(('ids', 'mask', 'spans', 'start'), [
    (torch.arange(8).unsqueeze(0), torch.ones(8), (), 0),
    (torch.arange(8), torch.ones(7), (), 0),
    (torch.arange(8), torch.tensor([1, 1, 2, 1, 1, 1, 1, 1]), (), 0),
    (torch.arange(8), torch.ones(8), (), -1),
    (torch.arange(8), torch.ones(8), (Stage1LatentSpan(1, 3, (2,)),), 2),
    (torch.arange(8), torch.ones(8), (Stage1LatentSpan(6, 8, (7,)),), 0),
    (torch.arange(8), torch.ones(8), (Stage1LatentSpan(1, 4, (2,)),), 0),
    (torch.arange(8), torch.ones(8), (Stage1LatentSpan(1, 4, (2, 3)), Stage1LatentSpan(3, 5, (4,))), 0),
])
def test_invalid_text_layout_fails(ids, mask, spans, start):
    with pytest.raises(ValueError):
        api().history_text_positions(ids, mask, spans, start, frozenset())


@pytest.mark.parametrize(('gamma', 'expected'), [
    (0., [[0., 1.]]), (.9, [[.9 / 1.9, 1. / 1.9]]), (1., [[.5, .5]]),
])
def test_text_history_weights_segments_rather_than_tokens(gamma, expected):
    q = api()
    # A one-token first segment and a three-token second segment each get one vote.
    first = torch.tensor([[1., 0.]], dtype=torch.bfloat16).mean(0, keepdim=True)
    second = torch.tensor([[0., 1.]] * 3, dtype=torch.bfloat16).mean(0, keepdim=True)
    state, _ = q.update_text_history(q.DiscountedTextState(), first, gamma=gamma)
    state, summary = q.update_text_history(state, second, gamma=gamma)
    torch.testing.assert_close(summary, torch.tensor(expected))
    assert summary.dtype == state.total.dtype == torch.float32
    assert state.weight == pytest.approx(1 + gamma)


def test_empty_step_ages_existing_text_before_next_segment():
    q = api()
    initial = q.DiscountedTextState()
    empty, summary = q.update_text_history(initial, None, gamma=.9)
    assert empty.total is None and empty.weight == 0 and summary is None
    state, _ = q.update_text_history(initial, torch.tensor([[1., 0.]]), gamma=.9)
    aged, summary = q.update_text_history(state, None, gamma=.9)
    torch.testing.assert_close(summary, torch.tensor([[1., 0.]]))
    torch.testing.assert_close(aged.total, torch.tensor([[.9, 0.]]))
    assert aged.weight == pytest.approx(.9)
    _, summary = q.update_text_history(aged, torch.tensor([[0., 1.]]), gamma=.9)
    torch.testing.assert_close(summary, torch.tensor([[.81 / 1.81, 1. / 1.81]]))
    # Updating a frozen state must not mutate the previous tensor in place.
    torch.testing.assert_close(state.total, torch.tensor([[1., 0.]]))


def test_gamma_zero_empty_step_forgets_history():
    q = api()
    state, _ = q.update_text_history(q.DiscountedTextState(), torch.ones(1, 2), gamma=0.)
    state, summary = q.update_text_history(state, None, gamma=0.)
    assert summary is None and state.weight == 0


def test_latent_sum_preserves_small_attention_mass():
    b, mass = api().aggregate_history_latents(torch.eye(2), torch.tensor([.001, .002]))
    torch.testing.assert_close(b, torch.tensor([[.001, .002]]))
    torch.testing.assert_close(mass, torch.tensor(.003))


def test_empty_latents_and_bfloat_inputs_aggregate_in_fp32():
    q = api()
    b, mass = q.aggregate_history_latents(torch.empty(0, 3), torch.empty(0))
    torch.testing.assert_close(b, torch.zeros(1, 3))
    torch.testing.assert_close(mass, torch.tensor(0.))
    b, mass = q.aggregate_history_latents(
        torch.eye(2, dtype=torch.bfloat16), torch.tensor([.25, .5], dtype=torch.bfloat16))
    assert b.dtype == mass.dtype == torch.float32
    torch.testing.assert_close(b, torch.tensor([[.25, .5]]))
    torch.testing.assert_close(mass, torch.tensor(.75))


@pytest.mark.parametrize(('text', 'expected', 'present'), [
    (None, [[1.001 / 1.003, .002 / 1.003]], False),
    (torch.tensor([[0., 1.]]), [[1.001 / 2.003, 1.002 / 2.003]], True),
])
def test_query_denominator_counts_present_text_and_unscaled_latent_mass(text, expected, present):
    result = api().compose_history_query(
        torch.tensor([[1., 0.]], dtype=torch.bfloat16), text,
        torch.tensor([[.001, .002]]), torch.tensor(.003))
    torch.testing.assert_close(result.query, torch.tensor(expected))
    torch.testing.assert_close(result.latent_mass, torch.tensor(.003))
    assert result.text_present is present and result.query.dtype == torch.float32


@pytest.mark.parametrize('global_intent', [
    torch.zeros(1, 2), torch.tensor([[1e-9, 0.]]),
    torch.tensor([[float('nan'), 0.]]), torch.tensor([[float('inf'), 0.]]),
])
def test_nonfinite_or_degenerate_query_fails(global_intent):
    with pytest.raises((ValueError, FloatingPointError)):
        api().compose_history_query(global_intent, None, torch.zeros(1, 2), torch.tensor(0.))


@pytest.mark.parametrize(('hidden', 'scores'), [
    (torch.ones(2, 3), torch.ones(3)),
    (torch.ones(2, 3), torch.tensor([-.1, .2])),
    (torch.ones(2, 3), torch.tensor([float('nan'), .2])),
    (torch.tensor([[float('inf'), 0.]]), torch.ones(1)),
])
def test_invalid_latent_aggregates_fail_explicitly(hidden, scores):
    with pytest.raises((ValueError, FloatingPointError)):
        api().aggregate_history_latents(hidden, scores)
