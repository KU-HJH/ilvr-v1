"""Causal CPU contracts for the passive OFF/cache comparison diagnostic."""
import importlib

import pytest
import torch
import torch.nn.functional as F

from evaluation.continuous_latent import ContinuousLatentController, ContinuousLatentGenerationConfig
from tests.evaluation.test_continuous_latent import TOKENS
from tests.evaluation.test_off_kv_cache import image_prompt
from tests.stage2.helpers import tiny_stage2_qwen


def probe():
    return importlib.import_module('scripts.probe_stage1_off_kv')


def test_rejects_observer_interference_even_when_common_prefix_matches():
    # Removing full-suffix checking would wrongly admit this changed final token.
    p = probe()
    frozen = {'generated_suffix_token_ids': list(range(32)), 'prompt_identity': 'same'}
    actual = dict(frozen, generated_suffix_token_ids=list(range(31))+[99])
    with pytest.raises(ValueError, match='observer fidelity'):
        p.require_fidelity(actual, frozen)
    assert p.require_fidelity(frozen, frozen) is None


def test_alignment_includes_branch_prediction_but_never_compares_its_children():
    p = probe()
    assert p.comparison_steps([3, 4, 5], [3, 9, 5]) == 2
    assert p.comparison_steps([3, 4], [3, 4]) == 2
    # Shape differences must not conceal equal newest logical rows.
    d = p.tensor_difference(torch.tensor([[[1., 2.]]]), torch.tensor([[[1., 3.]]]))
    assert d['max_abs'] == 1 and d['unequal_count'] == 1
    assert p.last_query_visibility(None, 3, 3, True).tolist() == [True]*3
    assert p.last_query_visibility(None, 1, 3, False).tolist() == [True]*3
    mask = torch.tensor([[[[0., -float('inf'), 0.]]]])
    assert p.last_query_visibility(mask, 1, 3, False).tolist() == [True, False, True]


def generate(model, cached):
    return ContinuousLatentController().generate(model=model, token_ids=TOKENS,
        generation_config=ContinuousLatentGenerationConfig(max_new_tokens=3, repetition_penalty=1.05),
        disable_latent_generation=True, use_kv_cache=cached, **image_prompt())


def test_native_observer_collects_effective_operands_and_preserves_generation():
    p = probe()
    torch.manual_seed(12)
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    with torch.inference_mode():
        expected_off = generate(model, False)
        expected_on = generate(model, True)
        with p.PassiveTrace(model, max_steps=3) as off:
            actual_off = generate(model, False)
        with p.PassiveTrace(model, max_steps=3) as on:
            actual_on = generate(model, True)
    assert actual_off.generated_suffix_ids == expected_off.generated_suffix_ids
    assert actual_on.generated_suffix_ids == expected_on.generated_suffix_ids
    assert len(off.steps) == len(on.steps) == 3
    assert off.steps[1]['history'] == on.steps[1]['history']
    assert off.steps[1]['decoder']['query_length'] == 9
    assert on.steps[1]['decoder']['query_length'] == 1
    assert on.steps[1]['decoder']['past_length'] == 8
    for trace in (off, on):
        events = trace.steps[1]['events']
        assert events['layer.0.sdpa.k']['value'].shape == (1, 2, 9, 4)
        assert events['layer.0.sdpa.visibility']['value'].tolist() == [False]+[True]*8
        assert 'layer.0.q_proj' in events and 'logits.raw' in events and 'logits.post_penalty' in events
    compared = p.compare_traces(off, on)
    assert compared['compared_steps'] >= 1
    # A later generation sees no retained cache or observation scope.
    with torch.inference_mode():
        again = generate(model, True)
    assert again.generated_suffix_ids == expected_on.generated_suffix_ids


def test_restores_hooks_and_wrappers_on_exception_and_replays_real_linear():
    p = probe()
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
    original_sdpa = F.scaled_dot_product_attention
    original_penalty = RepetitionPenaltyLogitsProcessor.__call__
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    hook_counts = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    with pytest.raises(RuntimeError, match='deliberate'):
        with p.PassiveTrace(model, max_steps=2):
            raise RuntimeError('deliberate')
    assert F.scaled_dot_product_attention is original_sdpa
    assert RepetitionPenaltyLogitsProcessor.__call__ is original_penalty
    assert hook_counts == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    linear = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
    replay = p.replay_row_operation(linear, torch.tensor([[[1., 2., 3.]]]).repeat(1, 7, 1))
    assert replay['difference']['max_abs'] == 0
    assert replay['full_shape'] == [1, 7, 3]


def test_published_sample_image_is_resolved_before_existing_runner_opens_it():
    # Omitting eval.main's image resolution makes direct run_one_example fail.
    sample = probe().selected_sample()
    from pathlib import Path
    assert Path(sample['image_input'][0]).is_absolute()
    assert Path(sample['image_input'][0]).is_file()
    assert sample['text_input'].startswith('In △ABC')


def test_output_reservation_rejects_overwrite_and_outside_results(tmp_path, monkeypatch):
    p = probe()
    monkeypatch.chdir(tmp_path)
    output = p.output_directory('results/probe')
    (output/'report.json').write_text('original')
    with pytest.raises(FileExistsError):
        p.output_directory('results/probe')
    assert (output/'report.json').read_text() == 'original'
    with pytest.raises(ValueError, match='results'):
        p.output_directory('logs/probe')
