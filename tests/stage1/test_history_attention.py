"""Native probability witnesses catch lost DoRA/MRoPE/GQA or latent renormalization."""
import importlib
import weakref
from dataclasses import replace

import pytest
import torch

from stage1.history_teacher import HistoryPrefixAdapter, HistorySelection
from stage1.query import Stage1QueryConfig, resolve_query_layers
from tests.stage1.history_fixtures import make_history_case


def observer_type():
    assert importlib.util.find_spec('stage1.history_attention') is not None, (
        'missing reduced native history-attention observer')
    return importlib.import_module('stage1.history_attention').ReducedHistoryAttentionObserver


def prepared(case, stop=19):
    x = case.inputs
    adapter = HistoryPrefixAdapter(
        case.teacher, input_ids=x['input_ids'], attention_mask=x['attention_mask'],
        original_pixels=x['pixel_values'], original_grid=x['image_grid_thw'])
    hidden_size = adapter.backbone.config.hidden_size
    selection = HistorySelection(0, (14, 15),
        torch.arange(2 * hidden_size).reshape(2, hidden_size).float()/9, (0, 1))
    return adapter, adapter.prepare(stop, (selection,))


def native(backbone, prefix, *, attentions=False):
    with torch.no_grad():
        return backbone(inputs_embeds=prefix.inputs_embeds,
                        attention_mask=prefix.attention_mask,
                        position_ids=prefix.position_ids, use_cache=False,
                        output_attentions=attentions, return_dict=True)


def hook_counts(backbone):
    return tuple((len(m._forward_pre_hooks), len(m._forward_hooks)) for m in backbone.modules())


@pytest.mark.parametrize('selector', ['last', 'last4', 'all'])
@pytest.mark.parametrize('chunks', [(64, 2048, 4), (2, 3, 1), (3, 7, 3)])
def test_scores_match_native_nonzero_dora_image_mrope_gqa(selector, chunks):
    """Base-only projections, wrong axes/heads or partial denominator fail this comparison."""
    obs_type = observer_type()
    eager = make_history_case(layers=5, attention_backend='eager')
    sdpa = make_history_case(layers=5)
    # Left padding is outside all requested query/key positions, but inside the prefix.
    for case in (eager, sdpa):
        case.inputs['attention_mask'][0, 0] = 0
    eager_adapter, expected_prefix = prepared(eager)
    adapter, prefix = prepared(sdpa)
    expected_output = native(eager_adapter.backbone, expected_prefix, attentions=True)
    layers = resolve_query_layers(selector, 5)
    queries, latents = (12, 17, 18), (14, 15)
    expected = torch.stack([expected_output.attentions[i][0, :, list(queries)]
                            [:, :, list(latents)].float().mean((0, 1)) for i in layers]).mean(0)
    # The early query cannot see the future latent keys.
    assert torch.equal(expected_output.attentions[0][0, :, 12, 14:16], torch.zeros(2, 2))
    baseline = adapter.forward(prefix).last_hidden_state
    counts = hook_counts(adapter.backbone)
    rng = torch.random.get_rng_state().clone()
    config = Stage1QueryConfig(mode='adaptive', attention_layers=selector,
                              query_chunk_size=chunks[0], key_chunk_size=chunks[1],
                              head_chunk_size=chunks[2])
    with obs_type(adapter.backbone, layer_indices=layers, query_positions=queries,
                  latent_positions=latents, attention_mask=prefix.attention_mask,
                  config=config) as observer:
        actual_output = adapter.forward(prefix).last_hidden_state
        actual = observer.scores()
        assert actual.dtype == torch.float32 and not actual.requires_grad
    error = (actual-expected).abs().max().item()
    print(f'native_max_abs_error selector={selector} chunks={chunks}: {error:.12g}')
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual_output, baseline, atol=0, rtol=0)
    assert hook_counts(adapter.backbone) == counts
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(p.grad is None and not p.requires_grad for p in sdpa.teacher.parameters())
    assert all(layer.attn_type == 'sdpa' for layer in adapter.backbone.layers)
    # Result remains available after caches/hooks are cleared on exit.
    torch.testing.assert_close(observer.scores(), actual, atol=0, rtol=0)


def test_distinct_kv_groups_match_native_across_head_chunk_boundary():
    """Mapping every query head to KV0 or alternating KV groups breaks this witness."""
    eager = make_history_case(attention_backend='eager', num_attention_heads=4,
                              num_key_value_heads=2)
    sdpa = make_history_case(num_attention_heads=4, num_key_value_heads=2)
    for case in (eager, sdpa):
        case.inputs['attention_mask'][0, 0] = 0
    eager_adapter, expected_prefix = prepared(eager)
    adapter, prefix = prepared(sdpa)
    attention = adapter.backbone.layers[0].self_attn
    assert (attention.num_heads, attention.num_key_value_heads,
            attention.num_key_value_groups) == (4, 2, 2)
    # Read actual effective DoRA K outputs, rather than assuming random groups differ.
    keys = []
    handle = eager_adapter.backbone.layers[0].self_attn.k_proj.register_forward_hook(
        lambda _m, _a, output: keys.append(output.detach().clone()))
    try:
        result = native(eager_adapter.backbone, expected_prefix, attentions=True)
    finally:
        handle.remove()
    grouped_keys = keys[0].reshape(1, 19, 2, 4)
    assert not torch.allclose(grouped_keys[:, :, 0], grouped_keys[:, :, 1])
    assert any('lora_B' in name and bool(parameter.ne(0).any())
               for name, parameter in sdpa.teacher.named_parameters())
    queries, latents = (12, 17, 18), (14, 15)
    probabilities = result.attentions[0][0, :, list(queries)][:, :, list(latents)].float()
    assert not torch.allclose(probabilities[:2], probabilities[2:])
    assert torch.equal(probabilities[:, 0], torch.zeros(4, 2))
    expected = probabilities.mean((0, 1))
    assert expected.sum() < 1  # Full causal keys remain in the denominator.
    config = Stage1QueryConfig(mode='adaptive', query_chunk_size=2,
                              key_chunk_size=3, head_chunk_size=3)
    # Chunk [0,1,2] spans the native [0,0,1,1] KV mapping's boundary.
    with observer_type()(adapter.backbone, layer_indices=(0,), query_positions=queries,
                         latent_positions=latents, attention_mask=prefix.attention_mask,
                         config=config) as observer:
        adapter.forward(prefix)
        actual = observer.scores()
    error = (actual - expected).abs().max().item()
    print(f'native_multigroup_gqa_max_abs_error heads=4 kv=2 head_chunk=3: {error:.12g}')
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_uniform_logits_keep_full_causal_denominator_and_padding():
    """Renormalizing the two selected keys would give [.5,.5], which is wrong."""
    obs_type = observer_type()
    case = make_history_case(attention_backend='eager')
    adapter, prefix = prepared(case)
    attn = adapter.backbone.layers[0].self_attn
    with torch.no_grad():
        for projection in (attn.q_proj, attn.k_proj):
            projection.lora_magnitude_vector['default'].weight.zero_()
            projection.base_layer.bias.zero_()
    # Query 5 has exactly four visible keys (positions 2..5); future keys stay masked.
    mask = prefix.attention_mask.clone()
    mask[:, :2] = 0
    prefix = replace(prefix, attention_mask=mask)
    with obs_type(adapter.backbone, layer_indices=(0,), query_positions=(5,),
                  latent_positions=(2, 4), attention_mask=mask,
                  config=Stage1QueryConfig(mode='adaptive', key_chunk_size=3)) as observer:
        result = native(adapter.backbone, prefix, attentions=True)
        scores = observer.scores()
    torch.testing.assert_close(result.attentions[0][0, :, 5, [2, 4]],
                               torch.full((2, 2), .25), atol=0, rtol=0)
    torch.testing.assert_close(scores, torch.tensor([.25, .25]), atol=1e-7, rtol=0)
    torch.testing.assert_close(scores.sum(), torch.tensor(.5), atol=1e-7, rtol=0)


@pytest.mark.parametrize(('queries', 'latents', 'want'),
                         [((), (14, 15), [0., 0.]), ((17,), (), []), ((), (), [])])
def test_empty_history_installs_no_hooks(queries, latents, want):
    obs_type = observer_type()
    adapter, prefix = prepared(make_history_case())
    before = hook_counts(adapter.backbone)
    with obs_type(adapter.backbone, layer_indices=(0,), query_positions=queries,
                  latent_positions=latents, attention_mask=prefix.attention_mask,
                  config=Stage1QueryConfig(mode='adaptive')) as observer:
        assert hook_counts(adapter.backbone) == before
        torch.testing.assert_close(observer.scores(), torch.tensor(want), atol=0, rtol=0)
    assert hook_counts(adapter.backbone) == before


def test_missing_capture_errors_instead_of_uniform_fallback():
    obs_type = observer_type()
    adapter, prefix = prepared(make_history_case())
    before = hook_counts(adapter.backbone)
    with obs_type(adapter.backbone, layer_indices=(0,), query_positions=(17,),
                  latent_positions=(14,), attention_mask=prefix.attention_mask,
                  config=Stage1QueryConfig(mode='adaptive')) as observer:
        with pytest.raises(RuntimeError, match='missing|capture'):
            observer.scores()
    assert hook_counts(adapter.backbone) == before


def test_second_forward_is_duplicate_capture_and_cleans_up():
    obs_type = observer_type()
    adapter, prefix = prepared(make_history_case())
    before = hook_counts(adapter.backbone)
    with pytest.raises(RuntimeError, match='duplicate|once'):
        with obs_type(adapter.backbone, layer_indices=(0,), query_positions=(17,),
                      latent_positions=(14,), attention_mask=prefix.attention_mask,
                      config=Stage1QueryConfig(mode='adaptive')) as observer:
            adapter.forward(prefix)
            observer.scores()
            adapter.forward(prefix)
    assert hook_counts(adapter.backbone) == before


def test_forward_exception_removes_observer_hooks_and_projection_cache():
    obs_type = observer_type()
    adapter, prefix = prepared(make_history_case(layers=2))
    # Force failure after q/k were captured but before the layer has returned.
    def fail(module, args):
        raise RuntimeError('forced native failure')
    projections = []
    witness = adapter.backbone.layers[0].self_attn.q_proj.register_forward_hook(
        lambda m, a, out: projections.append(weakref.ref(out)))
    failure = adapter.backbone.layers[0].self_attn.v_proj.register_forward_pre_hook(fail)
    before = hook_counts(adapter.backbone)
    try:
        with pytest.raises(RuntimeError, match='forced native failure'):
            with obs_type(adapter.backbone, layer_indices=(0, 1), query_positions=(17,),
                          latent_positions=(14,), attention_mask=prefix.attention_mask,
                          config=Stage1QueryConfig(mode='adaptive')) as observer:
                adapter.forward(prefix)
        assert hook_counts(adapter.backbone) == before
        assert len(projections) == 1 and projections[0]() is None
        # Removing the fault restores ordinary native execution, without stale hooks.
    finally:
        failure.remove()
        witness.remove()
    assert torch.isfinite(adapter.forward(prefix).last_hidden_state).all()
    with pytest.raises(RuntimeError, match='missing|capture|failed'):
        observer.scores()


@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
def test_native_window_crossing_rejects_full_causal_observer_even_when_flag_off(backend):
    """Native sliding_window=4 excludes key14 at query18 despite use_sliding_window=False."""
    obs_type = observer_type()
    adapter, prefix = prepared(make_history_case(attention_backend=backend))
    config = adapter.backbone.config
    config.sliding_window = 4
    assert config.use_sliding_window is False
    with torch.no_grad():
        for projection in (adapter.backbone.layers[0].self_attn.q_proj,
                           adapter.backbone.layers[0].self_attn.k_proj):
            projection.lora_magnitude_vector['default'].weight.zero_()
            projection.base_layer.bias.zero_()
    # Read native eager probabilities, and also the actual mask entering each backend.
    masks = []
    capture = adapter.backbone.layers[0].register_forward_pre_hook(
        lambda m, a, kw: masks.append(kw['attention_mask'].detach().clone()),
        with_kwargs=True)
    try:
        result = native(adapter.backbone, prefix, attentions=backend == 'eager')
    finally:
        capture.remove()
    assert len(masks) == 1
    assert masks[0][0, 0, 18, 14] < -1e20
    assert torch.equal(masks[0][0, 0, 18, 15:19], torch.zeros(4))
    if backend == 'eager':
        torch.testing.assert_close(result.attentions[0][0, :, 18, [14, 15]],
                                   torch.tensor([[0., .25], [0., .25]]), atol=0, rtol=0)
    before = hook_counts(adapter.backbone)
    try:
        with pytest.raises(ValueError, match='sliding.window|full.causal'):
            with obs_type(adapter.backbone, layer_indices=(0,), query_positions=(18,),
                          latent_positions=(14, 15), attention_mask=prefix.attention_mask,
                          config=Stage1QueryConfig(mode='adaptive')):
                pass
        assert hook_counts(adapter.backbone) == before
    finally:
        assert config.sliding_window == 4 and config.use_sliding_window is False


@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
@pytest.mark.parametrize('window', [19, 20, None])
@pytest.mark.parametrize('flag', [False, True])
def test_short_prefix_retains_native_full_causal_support_at_window_boundary(backend, window, flag):
    """At S==window the last query is window-1, so no key is excluded yet."""
    obs_type = observer_type()
    adapter, prefix = prepared(make_history_case(attention_backend=backend))
    adapter.backbone.config.sliding_window = window
    adapter.backbone.config.use_sliding_window = flag
    expected = native(adapter.backbone, prefix).last_hidden_state
    before = hook_counts(adapter.backbone)
    with obs_type(adapter.backbone, layer_indices=(0,), query_positions=(17, 18),
                  latent_positions=(14, 15), attention_mask=prefix.attention_mask,
                  config=Stage1QueryConfig(mode='adaptive')) as observer:
        actual = native(adapter.backbone, prefix).last_hidden_state
        assert torch.isfinite(observer.scores()).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert hook_counts(adapter.backbone) == before
    assert adapter.backbone.config.sliding_window == window
    assert adapter.backbone.config.use_sliding_window is flag
