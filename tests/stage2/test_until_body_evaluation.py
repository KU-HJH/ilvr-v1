"""Causal contracts for the full-GT fixed-bank evaluation."""
import math
from types import SimpleNamespace

import pytest
import torch

from scripts.evaluate_until_body_mask import (
    extract_sites, validate_sites, summarize, capture_bank, replay_scores,
    verify_identity, numerical_policy,
)
from tests.stage2.helpers import tiny_stage2_qwen
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


def test_all_gt_until_next_start_includes_final_answer_and_checks_independent_ids():
    ids = torch.tensor([[1, 91, 90, 90, 92, 2, 3, 91, 90, 90, 92, 4, 5, 6, 7]])
    sites = extract_sites(ids, 1, 14, LatentTokenIds(90, 91, 92), 2)
    oracle = [dict(block_index=0, body_positions=[2, 3], target_start=5, target_end=7,
                   target_count=2, target_ids=[2, 3]),
              dict(block_index=1, body_positions=[8, 9], target_start=11, target_end=14,
                   target_count=3, target_ids=[4, 5, 6])]
    validate_sites(sites, oracle)
    assert sites[-1]['target_ids'] == [4, 5, 6]
    wrong = [dict(oracle[0], target_ids=[2, 8]), oracle[1]]
    with pytest.raises(ValueError, match='target_ids'):
        validate_sites(sites, wrong)
    with pytest.raises(ValueError, match='site count'):
        validate_sites(sites, oracle[:1])


def test_metrics_weight_each_token_and_keep_paired_site_denominators():
    records = [dict(sample_id='a', block_index=0, clean_target_logps=[-1.], target_logps=[-3.],
                    clean_top1_ids=[2], top1_ids=[3]),
               dict(sample_id='b', block_index=0, clean_target_logps=[-2., -2., -2.],
                    target_logps=[-2., -4., -3.], clean_top1_ids=[4, 4, 4], top1_ids=[4, 5, 4])]
    got = summarize(records)
    assert got == dict(sites=2, tokens=4, first_changed=1, any_changed=2, tokens_changed=2,
                      first_changed_rate=.5, any_changed_rate=1., tokens_changed_rate=.5,
                      clean_mean_nll=1.75, mean_nll=3., delta_nll=1.25)
    with pytest.raises(ValueError, match='duplicate'):
        summarize(records + records[:1])
    with pytest.raises(ValueError, match='finite'):
        summarize([dict(records[0], target_logps=[math.nan])])


def test_native_bank_noop_and_body_key_dispatch_with_real_decoder():
    torch.manual_seed(29)
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    ids = torch.tensor([[1, 91, 90, 90, 92, 2, 3, 4]])
    sites = extract_sites(ids, 1, 8, LatentTokenIds(90, 91, 92), 2)
    sample = SimpleNamespace(model_inputs={'input_ids':ids, 'attention_mask':torch.ones_like(ids)}, sites=sites)
    with torch.inference_mode(), numerical_policy():
        bank, native = capture_bank(model, sample)
        saved_embeds, saved_positions = bank['embeds'].clone(), bank['positions'].clone()
        clean = replay_scores(model, bank, sample, sites[0])
        noop = replay_scores(model, bank, sample, sites[0], body_positions=[])
        assert verify_identity(native, clean)['max_logp_error'] <= 1e-3
        assert verify_identity(clean, noop, exact=True)['max_logp_error'] == 0
        masks = []
        hook = model.model.layers[0].self_attn.register_forward_pre_hook(
            lambda m, a, k: masks.append(k['attention_mask'].clone()), with_kwargs=True)
        try:
            changed = replay_scores(model, bank, sample, sites[0], body_positions=[2, 3])
        finally:
            hook.remove()
    # Actual dispatched decoder mask blocks only body KEY columns; START/END visible.
    assert masks[0].shape == (1, 1, 8, 8)
    assert bool((masks[0][0, 0, 4:, 2:4] < -1e20).all())
    assert bool((masks[0][0, 0, 4:, 1] == 0).all())
    assert bool((masks[0][0, 0, 4:, 4] == 0).all())
    assert torch.equal(bank['embeds'], saved_embeds)
    assert torch.equal(bank['positions'], saved_positions)
    assert not torch.equal(changed['logps'], clean['logps'])
