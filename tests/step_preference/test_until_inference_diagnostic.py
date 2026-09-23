"""Independent small CPU witnesses for the inference-only source differential."""
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def probe():
    path = Path(__file__).parents[2] / 'scripts/verify_until_inference.py'
    assert path.is_file(), 'inference diagnostic is missing'
    spec = importlib.util.spec_from_file_location('until_inference_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_witness_casts_explicitly_and_detects_wrong_restoration():
    d = probe()
    saved = {'input_rows': torch.tensor([[.13, .24]]), 'output_delta': torch.tensor([[.7, .8]])}
    restored = {key: value.bfloat16() for key, value in saved.items()}
    witness = d.checkpoint_tensors(restored, saved)
    assert witness['all_equal_after_explicit_cast']
    assert witness['tensors']['input_rows']['checkpoint_dtype'] == 'torch.float32'
    assert witness['tensors']['input_rows']['restored_dtype'] == 'torch.bfloat16'
    restored['output_delta'][0, 1] = 0
    assert not d.checkpoint_tensors(restored, saved)['all_equal_after_explicit_cast']
    del restored['input_rows']
    assert not d.checkpoint_tensors(restored, saved)['all_equal_after_explicit_cast']


def test_metrics_are_site_and_token_weighted_with_signed_delta():
    d = probe()
    a = torch.tensor([[2., 0.], [0., 2.]])
    b = torch.tensor([[0., 2.], [0., 2.]])
    result = d.five_metrics([a, a[:1]], [b, b[:1]], [torch.tensor([0, 1]), torch.tensor([0])])
    assert (result['first_changed'], result['any_changed'], result['tokens_changed']) == (2, 2, 2)
    assert (result['S'], result['N']) == (2, 3)
    assert result['mean_nll'] == pytest.approx(1.46026134)
    assert result['delta_nll'] == pytest.approx(4/3)


def test_native_factor_records_effective_policy_and_both_input_comparisons():
    """Catches labelling a requested backend without its effective native policy."""
    d = probe()
    left = {'embeds': torch.tensor([1.]), 'positions': torch.tensor([2])}
    native_other = {'embeds': torch.tensor([3.]), 'positions': torch.tensor([4])}
    common_other = {'embeds': torch.tensor([1.]), 'positions': torch.tensor([2])}

    result = d.native_factor(
        'math', {'sdpa_math': True, 'sdpa_flash': False},
        left, native_other, common_other)

    assert result['backend'] == 'math'
    assert result['effective_numerics'] == {
        'sdpa_math': True,
        'sdpa_flash': False,
    }
    assert not result['native_input_rollout']['embeds']['exact_equal']
    assert result['common_input_rollout']['embeds']['exact_equal']


def test_external_model_load_options_enable_its_split_attention_protocol():
    """Catches restoring the external factorial model without its protocol flag."""
    d = probe()

    local = d.diagnostic_model_load_kwargs(external=False)
    external = d.diagnostic_model_load_kwargs(external=True)

    assert 'stage2_split_attention' not in local
    assert external['stage2_split_attention'] is True
    assert local['local_files_only'] is external['local_files_only'] is True
    assert local['torch_dtype'] is external['torch_dtype'] is torch.bfloat16


def test_selection_is_first_two_manifest_entries_not_sorted_by_effect():
    d = probe()
    assert d.selected_indices({'samples': [{'source_index': 57}, {'source_index': 4}, {'source_index': 1}]}) == [57, 4]
    with pytest.raises(ValueError):
        d.selected_indices({'samples': [{'source_index': 4}]})


def test_preflight_rejects_cpu_before_any_model_loading(tmp_path):
    d = probe()
    args = SimpleNamespace(device='cpu', output_dir=tmp_path/'result')
    with pytest.raises(ValueError, match='CUDA'):
        d.preflight(args)


@pytest.mark.parametrize('one_site', [False, True])
def test_actual_mask_paths_and_layer_witnesses_on_separately_initialized_models(one_site):
    d = probe()
    from scripts.verify_until_native import ExternalSource
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    source_path = Path('/raid/workspace/temp/ILVR/.worktrees/ours1')
    with ExternalSource(source_path) as source, d.ce_source(d.CE_ROOT) as ce:
        external = d.external_modules(source)
        runtime = ce.module('evaluation.vision_block_runtime')
        inputs = ce.module('evaluation.vision_block_inputs')
        config = dict(vocab_size=128, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, stage='stage2', latent_size=8,
            latent_pad_token_id=90, latent_start_token_id=91, latent_end_token_id=92,
            image_token_id=80, vision_start_token_id=81, video_token_id=82,
            vision_config=dict(depth=1, hidden_size=8, intermediate_size=16, num_heads=2,
                patch_size=2, spatial_merge_size=1, temporal_patch_size=1, window_size=2,
                out_hidden_size=8, fullatt_block_indexes=[0]),
            attn_implementation='sdpa', rope_scaling={'type':'mrope','mrope_section':[1,1,0]})
        torch.manual_seed(42)
        local = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(**config)).eval()
        torch.manual_seed(42)
        other = external.model.Qwen2_5_VLForConditionalGeneration(external.model.Qwen2_5_VLConfig(**config)).eval()
        ids = torch.tensor([[1, 91, *([90]*8), 92, 13, 14, 91, *([90]*8), 92, 15, 16]])
        sites = (inputs.Site(57, 1, 1, 2, 10, 10, (11,12), (13,14)),
                 inputs.Site(57, 3, 13, 14, 22, 22, (23,24), (15,16)))
        if one_site:
            sites = sites[:1]
        sample = inputs.PreparedSample(57, {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}, sites)
        with torch.inference_mode():
            local_native = d.native_rollout(local, sample, 'local', external, runtime)
            other_native = d.native_rollout(other, sample, 'external', external, runtime)
            assert local_native['embeds'].shape == other_native['embeds'].shape == (1,25,8)
            trace = d.compare_native_traces(
                local_native['trace'], other_native['trace'],
                image_token_id=80, latent_pad_token_id=90)
            assert len(trace['decoder_calls']) == 17
            assert trace['decoder_calls'][0]['kind'] == 'latent_prefix'
            assert trace['decoder_calls'][0]['latent_ordinal'] == 0
            assert trace['decoder_calls'][-1]['kind'] == 'final'
            assert trace['earliest_difference'] is None
            assert trace['image_embeddings']['status'] == 'not_applicable'
            missing_left = dict(local_native['trace'], image_expected=True)
            missing_right = dict(other_native['trace'], image_expected=True,
                                 image_embeddings=torch.ones(1, 8))
            missing = d.compare_native_traces(
                missing_left, missing_right,
                image_token_id=80, latent_pad_token_id=90)
            assert missing['earliest_difference'] == {
                'stage': 'image_embeddings', 'reason': 'observation_missing'}
            assert missing['image_embeddings']['observation_present'] == [False, True]
            nonfinite = dict(local_native['trace'])
            nonfinite['token_embeddings'] = nonfinite['token_embeddings'].clone()
            nonfinite['token_embeddings'][0, 0, 0] = float('nan')
            with pytest.raises(ValueError, match='nonfinite native trace observation'):
                d.compare_native_traces(
                    nonfinite, other_native['trace'],
                    image_token_id=80, latent_pad_token_id=90)
            assert trace['token_embeddings']['exact_equal']
            assert trace['reconstructed_base_inputs']['exact_equal']
            assert set(trace['final_embedding_regions']) == {
                'image', 'latent', 'nonlatent_nonimage'}
            with runtime.numerical_policy():
                states = d.common_state(local, local_native, sample, runtime)
                result = d.compare_paths(local, other, sample, states, runtime, external, backend='math')
        assert result['factors'] == {'backend': 'math', 'external_batching': ['per_site', 'packed_sites']}
        assert result['local']['clean_repeat']['logits']['exact_equal']
        assert result['external']['clean_repeat']['logits']['exact_equal']
        assert result['local']['body_mask']['five_metrics']['S'] == len(sites)
        assert result['external']['body_mask']['five_metrics']['N'] == 2*len(sites)
        assert result['external']['batching']['clean']['batch_size'] == 2
        assert result['external']['batching']['clean']['duplicate_control_only'] is one_site
        mask = result['cross']['body_mask']['sites'][0]['mask_geometry']
        assert mask['local_blocked_rows_for_body_key'] == list(range(10,25))
        assert mask['external_blocked_rows_for_body_key'] == [10,11]
        assert result['cross']['clean']['sites'][0]['local_dispatch']['capture_kv'] is True
        assert result['cross']['clean']['sites'][0]['external_dispatch']['capture_kv'] is False
        assert result['cross']['clean']['sites'][0]['layers']
        assert local.model._forward_hooks == {}
        assert other.model._forward_hooks == {}
        assert local.model.embed_tokens._forward_hooks == {}
        assert other.model.embed_tokens._forward_hooks == {}
        assert local.visual._forward_hooks == {}
        assert other.visual._forward_hooks == {}
        import json
        json.dumps(result, allow_nan=False)


def test_default_control_records_exact_prefix_guard_and_finishes_external_paths():
    """Catches the known local default guard aborting the supported diagnostic."""
    d = probe()
    from scripts.verify_until_native import ExternalSource
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    source_path = Path('/raid/workspace/temp/ILVR/.worktrees/ours1')
    with ExternalSource(source_path) as source, d.ce_source(d.CE_ROOT) as ce:
        external = d.external_modules(source)
        runtime = ce.module('evaluation.vision_block_runtime')
        inputs = ce.module('evaluation.vision_block_inputs')
        config = dict(vocab_size=128, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, stage='stage2', latent_size=8,
            latent_pad_token_id=90, latent_start_token_id=91, latent_end_token_id=92,
            image_token_id=80, vision_start_token_id=81, video_token_id=82,
            vision_config=dict(depth=1, hidden_size=8, intermediate_size=16, num_heads=2,
                patch_size=2, spatial_merge_size=1, temporal_patch_size=1, window_size=2,
                out_hidden_size=8, fullatt_block_indexes=[0]),
            attn_implementation='sdpa', rope_scaling={'type':'mrope','mrope_section':[1,1,0]})
        torch.manual_seed(42)
        local = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(**config)).eval()
        torch.manual_seed(42)
        other = external.model.Qwen2_5_VLForConditionalGeneration(
            external.model.Qwen2_5_VLConfig(**config)).eval()
        ids = torch.tensor([[1, 91, *([90]*8), 92, 13, 14]])
        site = inputs.Site(57, 1, 1, 2, 10, 10, (11,12), (13,14))
        sample = inputs.PreparedSample(
            57, {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}, (site,))
        with torch.inference_mode():
            bank = d.native_rollout(local, sample, 'local', external, runtime)
            state = d.common_state(local, bank, sample, runtime)
            prefix = tuple(tuple(t.clone() for t in pair) for pair in state.prefix_kv)
            prefix[0][0][0, 0, 0, 0] += 1
            with pytest.raises(ValueError, match='^full-length prefix K/V changed$'):
                d.compare_paths(local, other, sample,
                    replace(state, prefix_kv=prefix), runtime, external, backend='math')
            result = d.compare_paths(local, other, sample,
                replace(state, prefix_kv=prefix), runtime, external, backend='default')

    assert result['status'] == 'unsupported_control'
    assert result['local'] == {
        'status': 'unsupported',
        'error': 'ValueError: full-length prefix K/V changed',
    }
    assert result['cross'] == {
        'status': 'unsupported',
        'error': 'ValueError: full-length prefix K/V changed',
    }
    assert result['external']['clean']['five_metrics']['N'] == 2
    assert result['external']['body_mask']['five_metrics']['N'] == 2
    assert result['external']['clean_repeat']['logits']['exact_equal']
    assert result['external']['batching']['clean']['batch_size'] == 2
