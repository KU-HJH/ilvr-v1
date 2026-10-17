"""Literal arithmetic/gradient witnesses for the v5 objective composition."""
from dataclasses import replace
import math

import pytest
import torch
from torch.testing import assert_close

from stage1.v5_config import Stage1V5Config
from stage1.v5_metadata import V5BatchMetadata, V5Block


def compute(*args):
    from stage1.v5_objective import compute_v5_loss_sums
    return compute_v5_loss_sums(*args)


def metadata():
    return V5BatchMetadata((V5Block(0, 1, 0, 1, (2, 3), (0, 3), ()),
                            V5Block(0, 3, 4, 5, (), (), (2, 3)),
                            V5Block(1, 1, 0, 1, (2,), (1,), ())))


def fixture():
    clean = torch.tensor([[.4, -.7, .8, .1], [1., -.3, .2, -.5],
                          [-.2, .8, -.1, .3]], requires_grad=True)
    ref = torch.tensor([[.2, .9, -.1, -.3], [-.2, .5, .3, 1.],
                        [.1, -.6, .4, .7]], requires_grad=True)
    off = torch.tensor([[1., .5, -.4, -.8], [.2, 1., .5, -.3],
                        [1., -.4, .7, -.9]], requires_grad=True)
    text = torch.tensor([[-.1, .2, .6, .9], [.5, -.4, .6, -.2],
                         [.9, -.1, -.2, .8]], requires_grad=True)
    return clean, ref, off, text


def literal_dil(clean, ref, targets):
    p = clean.float().log_softmax(-1)
    mu = ref.detach().float().log_softmax(-1)
    ratio = p - mu
    return torch.nn.functional.softplus(-ratio.gather(1, targets[:, None]).squeeze(1)) + (
        mu.exp() * torch.nn.functional.softplus(ratio)).sum(-1)


def literal_support_terms(clean, off, supports):
    entropy, mass = [], []
    for i, ids in enumerate(supports):
        logq = off[i, ids].log_softmax(-1)
        entropy.append(math.log(len(ids)) + (logq.exp() * logq).sum())
        a = clean[i].detach().softmax(-1)[ids].sum()
        b = off[i].softmax(-1)[ids].sum()
        mass.append(a * (a.log() - b.log()) + (1-a) * ((1-a).log() - (1-b).log()))
    return torch.stack(entropy), torch.stack(mass)


def test_dil_token_sum_matches_full_vocab_literal_and_gradient():
    clean, ref, _, _ = fixture()
    config = Stage1V5Config(text_kl_weight=0.)
    sums = compute(clean, ref, None, None, metadata(), config)
    literal = literal_dil(clean, ref, torch.tensor([0, 3, 1]))
    block0, block2 = literal[:2], literal[2:]
    zero = clean.sum() * 0
    assert sums.original_steps == 3
    assert_close(sums.dil_sum, block0.sum() + zero + block2.sum())
    actual_grad = torch.autograd.grad(sums.dil_sum, clean, retain_graph=True)[0]
    literal_grad = torch.autograd.grad(literal.sum(), clean)[0]
    assert_close(actual_grad, literal_grad)
    assert ref.grad is None
    assert_close(sums.diagnostics['dil_raw_sum'], sums.dil_sum.detach())
    assert sums.diagnostics['eligible_tokens'].item() == 3
    assert sums.diagnostics['original_steps'].item() == 3


@pytest.mark.parametrize('mode,supports', [
    ('independent', [[0, 1], [1, 2, 3], [0, 2, 1]]),
    ('sync_topk', [[2, 0], [0, 2, 3], [1, 3]]),
])
def test_ragged_gt_union_entropy_and_mass_match_literal_gradients(mode, supports):
    clean, ref, off, _ = fixture()
    config = Stage1V5Config(lambda_topk_entropy=.1, preference_topk_k=2,
        preference_topk_mode=mode, topk_include_gt=True, lambda_topk_mass=.2,
        text_kl_weight=0.)
    sums = compute(clean, ref, off, None, metadata(), config)
    u, mass = literal_support_terms(clean, off, supports)
    literal_u, literal_mass = u[:2].mean() + u[2], mass[:2].mean() + mass[2]
    assert_close(sums.entropy_sum, literal_u)
    assert_close(sums.mass_sum, literal_mass)
    for actual, expected in [(sums.entropy_sum, literal_u), (sums.mass_sum, literal_mass)]:
        assert_close(torch.autograd.grad(actual, off, retain_graph=True)[0],
                     torch.autograd.grad(expected, off, retain_graph=True)[0])
    # Full-vocabulary normalizer gives complement entries a nonzero mass gradient.
    mass_grad = torch.autograd.grad(sums.mass_sum, off)[0]
    assert mass_grad[0, 3].abs() > 0
    assert torch.autograd.grad(sums.entropy_sum, clean, allow_unused=True)[0] is None
    assert_close(sums.diagnostics['topk_entropy_weighted_sum'], .1 * literal_u.detach())
    assert_close(sums.diagnostics['topk_mass_weighted_sum'], .2 * literal_mass.detach())


def test_text_kl_t_squared_detaches_clean_and_averages_queries_per_valid_sample():
    clean, ref, _, off = fixture()
    sums = compute(clean, ref, None, off, metadata(), Stage1V5Config())
    detached_clean_T2 = (clean.detach() / 2).log_softmax(-1)
    textoff_T2 = (off / 2).log_softmax(-1)
    literal_kl = (detached_clean_T2.exp() * (detached_clean_T2 - textoff_T2)).sum(-1)
    text_kl = sums.text_kl
    assert_close(text_kl, 4 * (literal_kl[:2].mean() + literal_kl[2]) / 2)
    actual_grad = torch.autograd.grad(text_kl, off, retain_graph=True)[0]
    assert_close(actual_grad, torch.autograd.grad(
        4 * (literal_kl[:2].mean() + literal_kl[2]) / 2, off)[0])
    text_kl.backward()
    assert clean.grad is None  # isolated Text-KL target
    assert off.grad.abs().sum() > 0
    assert ref.grad is None


def test_same_percentile_mask_for_dil_entropy_mass_but_not_text_kl():
    clean, ref, off, text = fixture()
    config = Stage1V5Config(selection='clean', preference_entropy_top_percent=50,
        lambda_topk_entropy=.1, preference_topk_k=2, lambda_topk_mass=.2)
    # In sample0 position0 has higher full-vocab clean entropy; sample1 always retained.
    sums = compute(clean, ref, off, text, metadata(), config)
    d = literal_dil(clean, ref, torch.tensor([0, 3, 1]))
    u, mass = literal_support_terms(clean, off, [[0, 1], [1, 2], [0, 2]])
    assert_close(sums.dil_sum, d[0] + d[2])
    assert_close(sums.entropy_sum, u[0] + u[2])
    assert_close(sums.mass_sum, mass[0] + mass[2])
    assert sums.original_steps == 3
    assert sums.diagnostics['selected_tokens'].item() == 2
    assert_close(sums.text_kl, compute(clean, ref, off, text, metadata(),
        replace(config, selection='all', preference_entropy_top_percent=None)).text_kl)


def test_intersection_can_empty_a_block_without_removing_its_denominator():
    clean = torch.tensor([[0., 0., 0.], [3., 0., 0.]], requires_grad=True)
    off = clean.detach().flip(0).requires_grad_()
    meta = V5BatchMetadata((V5Block(0, 1, 0, 1, (2, 3), (0, 0), ()),))
    config = Stage1V5Config(selection='intersection', preference_entropy_top_percent=50,
        lambda_topk_entropy=.1, preference_topk_k=2, text_kl_weight=0.)
    sums = compute(clean, clean.detach(), off, None, meta, config)
    assert sums.original_steps == 1
    assert sums.dil_sum.item() == sums.entropy_sum.item() == 0
    (sums.dil_sum + sums.entropy_sum).backward()
    assert_close(clean.grad, torch.zeros_like(clean))
    assert_close(off.grad, torch.zeros_like(off))


def test_empty_batch_terms_are_graph_connected_and_have_no_division_by_zero():
    clean = torch.empty(0, 4, requires_grad=True)
    off = torch.empty(0, 4, requires_grad=True)
    sums = compute(clean, clean.detach(), off, off, V5BatchMetadata(()),
        Stage1V5Config(lambda_topk_entropy=.1, preference_topk_k=2))
    assert sums.original_steps == 0
    total = sums.dil_sum + sums.entropy_sum + sums.mass_sum + sums.text_kl
    assert total.item() == 0
    total.backward()
    assert clean.grad is not None and off.grad is not None


def test_chunk_remainder_keeps_token_weight_with_more_than_128_queries():
    torch.manual_seed(3)
    clean = torch.randn(129, 4, requires_grad=True)
    ref = torch.randn(129, 4)
    meta = V5BatchMetadata((V5Block(0, 1, 0, 1, tuple(range(2, 131)), (0,) * 129, ()),))
    sums = compute(clean, ref, None, None, meta, Stage1V5Config(text_kl_weight=0.))
    literal = literal_dil(clean, ref, torch.zeros(129, dtype=torch.long)).sum()
    assert_close(sums.dil_sum, literal)
    assert_close(torch.autograd.grad(sums.dil_sum, clean, retain_graph=True)[0],
                 torch.autograd.grad(literal, clean)[0])


def test_conditional_entropy_does_not_claim_full_vocab_uniformity():
    clean = torch.zeros(1, 4, requires_grad=True)
    off = torch.tensor([[2., 2., -3., -5.]], requires_grad=True)
    meta = V5BatchMetadata((V5Block(0, 1, 0, 1, (2,), (0,), ()),))
    sums = compute(clean, clean.detach(), off, None, meta,
        Stage1V5Config(lambda_topk_entropy=.1, preference_topk_k=2, text_kl_weight=0.))
    assert_close(sums.entropy_sum, torch.tensor(0.))
    logp = off.log_softmax(-1)
    full_vocab_kl = math.log(4) + (logp.exp() * logp).sum()
    assert full_vocab_kl > .6
    grad = torch.autograd.grad(sums.entropy_sum, off)[0]
    assert_close(grad, torch.zeros_like(off))


@pytest.mark.parametrize('branch', ['reference', 'V-off', 'Text-off'])
def test_required_branch_alignment_rejects_before_arithmetic(branch):
    clean, ref, off, text = fixture()
    config = Stage1V5Config(lambda_topk_entropy=.1, preference_topk_k=2)
    if branch == 'reference':
        ref = ref[:1]
    elif branch == 'V-off':
        off = None
    else:
        text = None
    with pytest.raises(ValueError, match=branch):
        compute(clean, ref, off, text, metadata(), config)
