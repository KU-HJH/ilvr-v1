import math

import pytest
import torch

import step_preference.dil_topk_loss as module


def gt_oracle(clean, off, targets, k):
    """Independent FP64 probability arithmetic with a per-row set union."""
    clean_p = clean.detach().double().softmax(-1)
    off_p = off.double().softmax(-1)
    fields = [[] for _ in range(7)]
    for row, target in enumerate(targets.tolist()):
        original = set(off[row].detach().topk(k).indices.tolist())
        support = sorted(original | {target})
        clean_mass = clean_p[row, support].sum()
        off_mass = off_p[row, support].sum()
        q = off_p[row, support] / off_mass
        gt_logq = q[support.index(target)].log()
        entropy = -(q * q.log()).sum()
        if len(support) == off.shape[-1]:
            mass = off[row, 0].double() * 0
        else:
            mass = (
                clean_mass * (clean_mass / off_mass).log()
                + (1 - clean_mass) * ((1 - clean_mass) / (1 - off_mass)).log()
            )
        values = (gt_logq, mass, entropy, clean_mass, off_mass,
                  (clean_mass - off_mass).abs(), off_mass.new_tensor(float(target in original)))
        for field, value in zip(fields, values, strict=True):
            field.append(value)
    return tuple(torch.stack(field) for field in fields)


def _terms(clean, off, targets, **kwargs):
    assert callable(getattr(module, "topk_gt_token_terms", None))
    return module.topk_gt_token_terms(clean, off, targets, **kwargs)


def test_gt_union_values_and_live_gradients():
    """Catches omitted/duplicated GT, a clean-selected mass set, or detached OFF."""
    clean = torch.tensor([[0., 1., 3., 2.], [0., 1., 3., 2.]], requires_grad=True)
    off = torch.tensor([[3., 2., 0., 1.], [3., 2., 0., 1.]], requires_grad=True)
    targets = torch.tensor([0, 2])  # K={0,1}; S={0,1}, then {0,1,2}
    assert callable(getattr(module, "topk_gt_token_terms", None))
    got = module.topk_gt_token_terms(clean, off, targets, k=2, chunk_size=1)
    want = gt_oracle(clean.double(), off.double(), targets, 2)
    assert got._fields == ("gt_logq", "mass_loss", "entropy", "clean_mass", "off_mass",
                           "mass_abs_gap", "gt_coverage")
    for actual, expected in zip(got, want, strict=True):
        assert actual.dtype == torch.float32 and actual.shape == (2,)
        torch.testing.assert_close(actual, expected.float(), atol=1e-6, rtol=1e-5)
    assert got.gt_coverage.tolist() == [1., 0.]
    grad = torch.autograd.grad((got.gt_logq + got.mass_loss).sum(), (clean, off), allow_unused=True)
    oracle_grad = torch.autograd.grad((want[0] + want[1]).sum(), off)[0]
    assert grad[0] is None
    torch.testing.assert_close(grad[1], oracle_grad, atol=1e-5, rtol=1e-4)


def test_gt_ties_share_one_support():
    """Catches boundary-tie support disagreement and counting duplicate GT."""
    clean = torch.zeros(1, 4)
    off = torch.tensor([[2., 1., 1., 0.]], requires_grad=True)
    original = set(off.detach().topk(2, dim=-1).indices[0].tolist())
    target = next(token for token in (1, 2) if token not in original)
    got = _terms(clean, off, torch.tensor([target]), k=2)
    assert got.gt_coverage.item() == 0.
    assert got.clean_mass.item() == pytest.approx(3 / 4)
    assert got.gt_logq.exp().item() == pytest.approx(math.exp(1) / (math.exp(2) + 2 * math.exp(1)))


def test_gt_mixed_full_support():
    """Catches empty-complement NaNs or full-row gradients leaking from partial rows."""
    clean = torch.tensor([[0., 1., 2., 3.], [0., 1., 2., 3.]], requires_grad=True)
    off = torch.tensor([[3., 2., 1., 0.], [3., 2., 1., 0.]], requires_grad=True)
    got = _terms(clean, off, torch.tensor([3, 0]), k=3)
    assert got.clean_mass[0].item() == got.off_mass[0].item() == 1.
    assert got.mass_loss[0].item() == got.mass_abs_gap[0].item() == 0.
    grad = torch.autograd.grad(got.mass_loss.sum(), off)[0]
    assert torch.equal(grad[0], torch.zeros(4))
    assert bool(torch.isfinite(grad[1]).all()) and bool((grad[1] != 0).any())
    want = gt_oracle(clean.double(), off.double(), torch.tensor([3, 0]), 3)
    for actual, expected in zip(got, want, strict=True):
        torch.testing.assert_close(actual, expected.float(), atol=1e-6, rtol=1e-5)


def test_gt_extreme_and_bf16():
    """Catches probability-underflow logq and low-precision statistics/gradients."""
    clean = torch.tensor([[0., -900., 1000., -1000.]], requires_grad=True)
    off = torch.tensor([[1000., 0., -1000., -900.]], requires_grad=True)
    got = _terms(clean, off, torch.tensor([2]), k=2)
    assert got.gt_logq.item() == pytest.approx(-2000.)
    assert all(bool(torch.isfinite(value).all()) for value in got)
    grad = torch.autograd.grad((got.gt_logq + got.mass_loss + got.entropy).sum(), off)[0]
    assert bool(torch.isfinite(grad).all())

    bf_clean = torch.tensor([[.25, -.75, 1., -.1]], dtype=torch.bfloat16, requires_grad=True)
    bf_off = torch.tensor([[.5, .1, -.2, .3]], dtype=torch.bfloat16, requires_grad=True)
    got = _terms(bf_clean, bf_off, torch.tensor([2]), k=2)
    want = gt_oracle(bf_clean.double(), bf_off.double(), torch.tensor([2]), 2)
    for actual, expected in zip(got, want, strict=True):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-4)
    actual_grad = torch.autograd.grad((got.gt_logq + got.mass_loss).sum(), bf_off)[0]
    expected_grad = torch.autograd.grad((want[0] + want[1]).sum(), bf_off)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-3, rtol=2e-3)

    # Full-support zero must not sum huge finite logits before multiplying by zero.
    huge = torch.full((1, 4), 2e38, requires_grad=True)
    original = set(huge.detach().topk(3, dim=-1).indices[0].tolist())
    target = next(token for token in range(4) if token not in original)
    full = _terms(torch.zeros_like(huge), huge, torch.tensor([target]), k=3)
    assert all(bool(torch.isfinite(value).all()) for value in full)
    full_grad = torch.autograd.grad(full.mass_loss.sum(), huge)[0]
    assert torch.equal(full_grad, torch.zeros_like(huge))


def test_gt_chunking_and_mass_disabled():
    """Catches chunk-boundary changes and mass leaking into the disabled path."""
    clean = torch.tensor([[0., 1., 3., 2.], [0., 1., 3., 2.]], requires_grad=True)
    off = torch.tensor([[3., 2., 0., 1.], [3., 2., 0., 1.]], requires_grad=True)
    targets = torch.tensor([0, 2])
    small = _terms(clean, off, targets, k=2, chunk_size=1)
    large = _terms(clean, off, targets, k=2, chunk_size=128)
    for actual, expected in zip(small, large, strict=True):
        torch.testing.assert_close(actual, expected)
    small_grad = torch.autograd.grad((small.gt_logq + small.mass_loss).sum(), off)[0]
    large_grad = torch.autograd.grad((large.gt_logq + large.mass_loss).sum(), off)[0]
    torch.testing.assert_close(small_grad, large_grad)
    no_mass = _terms(clean, off, targets, k=2, compute_mass=False)
    torch.testing.assert_close(no_mass.gt_logq, large.gt_logq)
    torch.testing.assert_close(no_mass.entropy, large.entropy)
    for field in ("mass_loss", "clean_mass", "off_mass", "mass_abs_gap"):
        assert torch.equal(getattr(no_mass, field), torch.zeros(2))


def test_gt_invalid_inputs():
    """Catches invalid support size, target alignment, dtype, device, or finite range."""
    clean = torch.zeros(1, 4)
    off = torch.zeros(1, 4)
    targets = torch.tensor([0])
    for k in (1, 4, True):
        with pytest.raises(ValueError):
            _terms(clean, off, targets, k=k)
    for target in (-1, 4):
        with pytest.raises(ValueError):
            _terms(clean, off, torch.tensor([target]), k=2)
    invalid = [
        (torch.zeros(0, 4), torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)),
        (clean, off, torch.tensor([[0]])),
        (clean, off, torch.tensor([0.])),
        (clean.long(), off, targets),
        (clean, torch.zeros(2, 4), targets),
        (clean, torch.zeros(1, 4, device="meta"), targets),
        (torch.tensor([[1e100, 0., 0., 0.]], dtype=torch.float64), off, targets),
    ]
    for bad in (float("nan"), float("inf")):
        invalid.extend(((torch.tensor([[bad, 0., 0., 0.]]), off, targets),
                        (clean, torch.tensor([[bad, 0., 0., 0.]]), targets)))
    for bad_clean, bad_off, bad_targets in invalid:
        with pytest.raises(ValueError):
            _terms(bad_clean, bad_off, bad_targets, k=2)
    with pytest.raises(ValueError):
        _terms(clean, off, targets, k=2, chunk_size=0)
