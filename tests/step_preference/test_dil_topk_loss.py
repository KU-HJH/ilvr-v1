import math

import pytest
import torch

from step_preference.dil_topk_loss import topk_negative_and_mass, topk_token_terms


def _oracle(clean, off, target_ids, beta, k):
    """Small independent FP64 probability-space oracle for ordinary logits."""
    ids = off.detach().topk(k, dim=-1).indices
    off_p = off.softmax(-1)
    clean_p = clean.detach().softmax(-1)
    q = off_p.gather(-1, ids)
    q = q / q.sum(-1, keepdim=True)
    negative = (q * torch.log1p(q.pow(beta))).sum(-1)
    clean_mass = clean_p.gather(-1, ids).sum(-1)
    off_mass = off_p.gather(-1, ids).sum(-1)
    mass = clean_mass * (clean_mass / off_mass).log() + (1-clean_mass) * ((1-clean_mass)/(1-off_mass)).log()
    entropy = -(q * q.log()).sum(-1)
    coverage = (ids == target_ids[:, None]).any(-1).to(clean.dtype)
    return negative, mass, entropy, clean_mass, off_mass, (clean_mass-off_mass).abs(), coverage


def test_topk_live_q_and_detached_clean_mass_match_independent_oracle():
    """Catches detached q weights, detached OFF complement, or live clean mass targets."""
    clean = torch.tensor([[.2, -.7, 1.1, -.3], [.4, .9, -.5, .1]], dtype=torch.float64, requires_grad=True)
    off = torch.tensor([[1.3, -.2, .5, -.8], [-.4, .2, 1.0, .6]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 1])
    actual = topk_token_terms(clean, off, targets, beta=.5, k=2, chunk_size=1)
    for item in actual:
        assert item.dtype == torch.float32 and item.shape == (2,)
    expected = _oracle(clean, off, targets, .5, 2)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want.float(), atol=1e-6, rtol=1e-5)

    actual_objective = (actual.negative_loss + actual.mass_loss).sum()
    actual_objective.backward()
    actual_off_grad = off.grad.clone()
    assert clean.grad is None
    off.grad = None
    oracle_objective = (expected[0] + expected[1]).sum()
    oracle_objective.backward()
    torch.testing.assert_close(actual_off_grad, off.grad, atol=1e-5, rtol=1e-4)
    assert clean.grad is None


def test_uniform_topk_floor_equal_mass_and_wrapper_means():
    """Catches wrong top-k floor or wrapper summing instead of averaging tokens."""
    clean = torch.zeros((3, 4), requires_grad=True)
    off = torch.zeros((3, 4), requires_grad=True)
    ids = torch.tensor([0, 1, 2])
    terms = topk_token_terms(clean, off, ids, beta=.5, k=2, chunk_size=2)
    floor = math.log(1 + 2 ** -.5)
    torch.testing.assert_close(terms.negative_loss, torch.full((3,), floor), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(terms.mass_loss, torch.zeros(3), atol=1e-6, rtol=1e-5)
    step = topk_negative_and_mass(clean, off, ids, beta=.5, k=2, chunk_size=2)
    for field in terms._fields:
        torch.testing.assert_close(getattr(step, field), getattr(terms, field).mean(), atol=1e-6, rtol=1e-5)


def test_no_mass_skips_mass_terms_and_keeps_negative():
    """Catches mass calculations leaking into the disabled path."""
    clean = torch.tensor([[.1, .2, -.4]])
    off = torch.tensor([[.5, .3, -.1]])
    terms = topk_token_terms(clean, off, torch.tensor([0]), beta=.5, k=2, compute_mass=False)
    assert terms.negative_loss.item() > 0
    for field in ('mass_loss', 'clean_mass', 'off_mass', 'mass_abs_gap'):
        assert getattr(terms, field).item() == 0.


def test_extreme_fp32_zero_probability_and_bf16_match_mathematical_oracles():
    """Catches finite but wrong group KL or BF16 token terms."""
    clean = torch.tensor([[1000., -1000., 0., -900.]], dtype=torch.float32)
    off = torch.tensor([[-1000., 1000., 0., -900.]], dtype=torch.float32, requires_grad=True)
    terms = topk_token_terms(clean, off, torch.tensor([1]), beta=.5, k=2)
    assert all(bool(torch.isfinite(value).all()) for value in terms)
    # OFF support is {1, 2}. Its conditional q is (1, 0), clean mass is 0,
    # and the OFF complement sits 1900 logit units below the selected group.
    extreme_expected = (math.log(2.), 1900., 0., 0., 1., 1., 1.)
    for got, want in zip(terms, extreme_expected, strict=True):
        torch.testing.assert_close(got, torch.tensor([want]), atol=1e-4, rtol=1e-6)
    terms.mass_loss.sum().backward()
    assert bool(torch.isfinite(off.grad).all())
    bf_clean = torch.tensor([[.25, -.75, 1., -.1]], dtype=torch.bfloat16)
    bf_off = torch.tensor([[.5, .1, -.2, .3]], dtype=torch.bfloat16)
    bf_terms = topk_token_terms(bf_clean, bf_off, torch.tensor([0]), beta=.5, k=2)
    assert all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in bf_terms)
    # Compare against FP64 probability arithmetic on the actual BF16-quantized
    # values, rather than the unrepresentable decimal constructor literals.
    bf_expected = _oracle(bf_clean.double(), bf_off.double(), torch.tensor([0]), .5, 2)
    for got, want in zip(bf_terms, bf_expected, strict=True):
        torch.testing.assert_close(got.double(), want, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('k', [0, 1, 4, 1.5, True])
def test_topk_rejects_invalid_k(k):
    """Catches empty/full support and noninteger top-k selections."""
    with pytest.raises(ValueError, match='k'):
        topk_token_terms(torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, dtype=torch.long), beta=.5, k=k)


@pytest.mark.parametrize('beta', [0., -1., float('nan'), float('inf')])
def test_topk_rejects_invalid_beta(beta):
    """Catches invalid top-k objective temperature."""
    with pytest.raises(ValueError, match='beta'):
        topk_token_terms(torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, dtype=torch.long), beta=beta, k=2)


def test_topk_rejects_bad_shapes_targets_and_fp32_overflow():
    """Catches invalid token alignment and finite FP64 values overflowing in FP32."""
    good = torch.zeros(1, 4)
    ids = torch.zeros(1, dtype=torch.long)
    for clean, off, target in ((torch.zeros(0, 4), torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)),
                               (good, torch.zeros(2, 4), ids),
                               (good, good, torch.zeros(2, dtype=torch.long))):
        with pytest.raises(ValueError):
            topk_token_terms(clean, off, target, beta=.5, k=2)
    with pytest.raises(ValueError, match='range'):
        topk_token_terms(good, good, torch.tensor([4]), beta=.5, k=2)
    with pytest.raises(ValueError, match='finite'):
        topk_token_terms(torch.tensor([[1e100, 0., 0., 0.]]), torch.zeros(1, 4, dtype=torch.float64), ids, beta=.5, k=2)
    with pytest.raises(ValueError, match='device'):
        topk_token_terms(good, torch.zeros(1, 4, device='meta'), ids, beta=.5, k=2)
