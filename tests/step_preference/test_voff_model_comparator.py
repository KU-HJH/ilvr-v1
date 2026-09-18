"""Contracts that prevent a source differential from falsely reporting zero."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('voff_model_probe', Path(__file__).parents[2] / 'scripts/verify_voff_model_path.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_allclose_does_not_hide_nonzero_coordinate_error():
    result = probe.comparison(torch.tensor([1.]), torch.tensor([1.0000001]))
    assert result['pass']
    assert result['exact_equal'] is False
    assert result['max_abs_error'] > 0


def test_disconnected_and_nonfinite_are_not_zero_gradient_evidence():
    disconnected = probe.comparison(None, None)
    assert disconnected['exact_equal'] is None
    assert disconnected['connection'] == 'both_disconnected'
    mismatch = probe.comparison(None, torch.zeros(1))
    assert not mismatch['pass']
    assert mismatch['connection'] == 'mismatch'
    nonfinite = probe.comparison(torch.tensor([float('inf')]), torch.tensor([float('inf')]))
    assert not nonfinite['pass']
    assert nonfinite['max_abs_error'] is None
    assert nonfinite['finite'] is False


def test_short_case_selection_uses_retained_indices_and_both_block_groups():
    manifest = {'retained_indices': [4, 1, 7, 9],
                'lengths': [1, 20, 10, 10, 30, 10, 10, 50, 10, 40],
                'latent_block_counts': [1, 1, 2, 2, 1, 2, 2, 2, 2, 3]}
    assert probe.select_short_cases(manifest) == [(1, 20, 1), (7, 50, 2)]
    with pytest.raises(ValueError, match='2'):
        probe.select_short_cases({'retained_indices': [0], 'lengths': [20], 'latent_block_counts': [1]})


def test_identity_gate_rejects_trainability_and_weight_mismatches():
    left, right = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    right.load_state_dict(left.state_dict())
    assert probe.verify_model_identity(left, right)['all_state_exact_equal']
    right.weight.requires_grad_(False)
    with pytest.raises(ValueError, match='trainable'):
        probe.verify_model_identity(left, right)
    right.weight.requires_grad_(True)
    with torch.no_grad():
        right.weight.add_(1)
    with pytest.raises(ValueError, match='state'):
        probe.verify_model_identity(left, right)


def test_runtime_backend_gate_rejects_sdpa_vision_for_production_profile():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    config = Qwen2_5_VLConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        vision_config=dict(depth=1, hidden_size=8, intermediate_size=16, num_heads=2,
            patch_size=2, spatial_merge_size=1, temporal_patch_size=1, window_size=2,
            out_hidden_size=8, fullatt_block_indexes=[0]),
        attn_implementation='sdpa', rope_scaling={'type': 'mrope', 'mrope_section': [1, 1, 0]})
    model = Qwen2_5_VLForConditionalGeneration(config)
    assert probe.runtime_model_metadata(model, production=False)['decoder_attention'].endswith('Qwen2_5_VLSdpaAttention')
    with pytest.raises(ValueError, match='vision'):
        probe.runtime_model_metadata(model, production=True)


def test_preference_gradient_witness_does_not_mistake_ce_signal_for_preference():
    model = torch.nn.Linear(1, 1, bias=False)
    preference = (model.weight * 0).sum()
    ce = model.weight.square().sum()
    assert torch.autograd.grad(ce, model.weight, retain_graph=True)[0].abs().sum() > 0
    assert probe.preference_gradient_nonzero(preference, model) is False
    assert probe.preference_gradient_nonzero(model.weight.sum(), model) is True
