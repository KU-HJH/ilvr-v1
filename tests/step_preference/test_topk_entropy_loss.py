"""Behavioral tests distinguish live conditional entropy from full mass KL."""

import importlib
import importlib.util
import math

import pytest
import torch
from torch.utils.checkpoint import checkpoint


def kernels():
    name = "step_preference.topk_entropy_loss"
    assert importlib.util.find_spec(name) is not None, "missing top-k entropy kernel"
    return importlib.import_module(name)


def close(actual, expected):
    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype), atol=1e-5, rtol=1e-4)


def support(ids, valid=None):
    ids = torch.tensor(ids, dtype=torch.long)
    if valid is None: valid = torch.ones_like(ids, dtype=torch.bool)
    return kernels().CandidateSupport(ids, valid, torch.zeros(ids.shape[0], dtype=torch.bool))


def test_uniform_and_saturated_entropy():
    mod = kernels()
    candidates = support([[0, 1, 2]])
    z = torch.zeros(1, 4, requires_grad=True)
    result = mod.topk_entropy_components(z, candidates)
    close(result.kl, [0.])
    close(result.entropy, [math.log(3)])
    close(result.normalized_entropy, [1.])
    close(result.support_size, [3])
    close(torch.autograd.grad(result.kl.sum(), z)[0], torch.zeros_like(z))
    z = torch.tensor([[1000., -1000., -1000., 2.]], requires_grad=True)
    result = mod.topk_entropy_components(z, candidates)
    close(result.kl, [math.log(3)])
    assert torch.isfinite(torch.autograd.grad(result.kl.sum(), z)[0]).all()


def test_live_q_analytic_gradient():
    mod = kernels()
    z = torch.tensor([[.2, -1., .9, .7]], requires_grad=True)
    candidates = support([[0, 2, 3]])
    result = mod.topk_entropy_components(z, candidates)
    actual_grad, = torch.autograd.grad(result.kl.sum(), z)
    selected = z.detach().double()[0, [0, 2, 3]]
    logq = selected.log_softmax(-1)
    q = logq.exp()
    entropy = -(q * logq).sum()
    close(actual_grad[0, [0, 2, 3]], (q * (logq + entropy)).float())
    assert actual_grad.abs().sum() > 0
    detached_z = z.detach().clone().requires_grad_()
    logq_bad = detached_z[0, [0, 2, 3]].log_softmax(-1)
    detached_q_variant = math.log(3) + (logq_bad.exp().detach() * logq_bad).sum()
    detached_q_variant_grad, = torch.autograd.grad(detached_q_variant, detached_z)
    close(detached_q_variant_grad, torch.zeros_like(z))
    # FP32 production requires a finite-difference step above FP32 rounding.
    for j in (0, 2, 3):
        plus, minus = z.detach().clone(), z.detach().clone()
        plus[0, j] += .002
        minus[0, j] -= .002
        numerical = (mod.topk_entropy_components(plus, candidates).kl - mod.topk_entropy_components(minus, candidates).kl) / .004
        torch.testing.assert_close(actual_grad[0, j], numerical[0], atol=1e-4, rtol=1e-3)


def test_valid_saturated_support_masks_zero_probability_times_infinity():
    # Finite FP32 logits can differ by more than FP32 can represent. Valid
    # support entries then have logq=-inf and q=0; their entropy limit is zero.
    z = torch.tensor([[3e38, -3e38, 0.]], requires_grad=True)
    result = kernels().topk_entropy_components(z, support([[0, 1]]))
    close(result.kl, [math.log(2)])
    close(result.entropy, [0.])
    grad, = torch.autograd.grad(result.kl.sum(), z)
    assert torch.isfinite(grad).all()
    close(grad, torch.zeros_like(z))


def test_candidate_modes_and_ragged_gt_union():
    mod = kernels()
    clean = torch.tensor([[5., 4., 3., 2.], [5., 4., 3., 2.]], requires_grad=True)
    off = torch.tensor([[1., 2., 4., 3.], [1., 2., 4., 3.]], requires_grad=True)
    y = torch.tensor([2, 0])
    independent = mod.build_entropy_support(clean, off, y, k=2, mode="independent", include_gt=True)
    assert independent.ids.dtype == torch.long and independent.valid.dtype == torch.bool
    assert independent.ids.shape == (2, 3)
    assert independent.ids[0, independent.valid[0]].tolist() == [2, 3]
    assert independent.ids[1, independent.valid[1]].tolist() == [2, 3, 0]
    assert independent.gt_coverage.tolist() == [True, False]
    synced = mod.build_entropy_support(clean, off, y, k=2, mode="sync_topk", include_gt=False)
    assert synced.ids.tolist() == [[0, 1], [0, 1]]
    assert synced.gt_coverage.tolist() == [False, True]
    assert not independent.ids.requires_grad
    result = mod.topk_entropy_components(off, independent)
    close(result.support_size, [2, 3])
    assert torch.isfinite(result.kl).all()


@pytest.mark.parametrize("dtype,vocab", [(torch.uint8, 256), (torch.int8, 128), (torch.int16, 65536), (torch.int32, 65536), (torch.int64, 65536)])
@pytest.mark.parametrize("mode", ["independent", "sync_topk"])
def test_support_integral_target_bounds_do_not_wrap_at_dtype_boundary(dtype, vocab, mode):
    y = torch.tensor([0, min(vocab - 1, torch.iinfo(dtype).max)], dtype=dtype)
    z = torch.zeros(2, vocab)
    z[:, 0], z[:, 1] = 3., 2.
    candidates = kernels().build_entropy_support(z, z, y, k=2, mode=mode, include_gt=True)
    assert candidates.ids[0, candidates.valid[0]].tolist() == [0, 1]
    assert candidates.ids[1, candidates.valid[1]].tolist() == [0, 1, int(y[1])]
    assert candidates.gt_coverage.tolist() == [True, False]
    assert candidates.ids.dtype == torch.long


def test_entropy_only_support_gradient():
    z = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
    result = kernels().topk_entropy_components(z, support([[0, 2]]))
    grad, = torch.autograd.grad(result.kl.sum(), z)
    close(grad[0, [1, 3]], [0., 0.])
    assert grad[0, [0, 2]].abs().sum() > 0


def test_checkpointed_ragged_mass_keeps_complement_gradient():
    mod = kernels()
    clean = torch.tensor([[3., 1., 0.], [1., 2., 3.]], requires_grad=True)
    off = torch.tensor([[0., 1., 3.], [3., 2., 1.]], requires_grad=True)
    candidates = support([[0, 1, 999], [0, 1, 2]], torch.tensor([[True, True, False], [True, True, True]]))
    entropy_loss = checkpoint(lambda z: mod.topk_entropy_components(z, candidates).kl, off, use_reentrant=False)
    entropy_grad, = torch.autograd.grad(entropy_loss.sum(), off, retain_graph=True)
    close(entropy_grad[0, 2], 0.)
    ragged_loss = checkpoint(lambda c, o: mod.entropy_support_mass_components(c, o, candidates).loss, clean, off, use_reentrant=False)
    mass_grad, = torch.autograd.grad(ragged_loss.sum(), off)
    assert mass_grad[0, 2].abs() > 0
    close(ragged_loss[1], 0.)
    close(mass_grad[1], [0., 0., 0.])
    assert torch.isfinite(ragged_loss).all() and torch.isfinite(mass_grad).all()
    assert clean.grad is None
    result = mod.entropy_support_mass_components(clean, off, candidates)
    a = clean.detach().double().softmax(-1)[0, :2].sum()
    b = off.detach().double().softmax(-1)[0, :2].sum()
    expected = a * (a / b).log() + (1-a) * ((1-a)/(1-b)).log()
    close(result.loss[0], expected.float())
    close(result.clean_mass[0], a.float())
    close(result.off_mass[0], b.float())
    close(result.mass_abs_gap[0], (a-b).abs().float())
    assert not result.clean_mass.requires_grad


def test_empty_positions_have_live_zero_graph():
    mod = kernels()
    z = torch.empty(0, 4, requires_grad=True)
    candidates = mod.build_entropy_support(z, z, torch.empty(0, dtype=torch.long), k=2, mode="independent", include_gt=True)
    assert candidates.ids.shape == (0, 3)
    entropy = mod.topk_entropy_components(z, candidates)
    mass = mod.entropy_support_mass_components(z, z, candidates)
    grad, = torch.autograd.grad(entropy.kl.sum() + mass.loss.sum(), z)
    assert grad.shape == (0, 4)


@pytest.mark.parametrize("kind", ["duplicate", "range", "dtype", "mask_dtype", "shape", "coverage", "empty_row", "one_candidate", "device", "nan", "overflow"])
def test_malformed_support_and_logits(kind):
    mod = kernels()
    z = torch.zeros(1, 4)
    ids = torch.tensor([[0, 1]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    coverage = torch.zeros(1, dtype=torch.bool)
    if kind == "duplicate": ids[0, 1] = 0
    if kind == "range": ids[0, 1] = 4
    if kind == "dtype": ids = ids.float()
    if kind == "mask_dtype": valid = valid.long()
    if kind == "shape": valid = torch.ones(1, 3, dtype=torch.bool)
    if kind == "coverage": coverage = torch.zeros(2, dtype=torch.bool)
    if kind == "empty_row": valid[:] = False
    if kind == "one_candidate": valid[0, 1] = False
    if kind == "device": ids = ids.to("meta")
    if kind == "nan": z[0, 0] = float("nan")
    if kind == "overflow": z = torch.full((1, 4), 1e300, dtype=torch.float64)
    candidates = mod.CandidateSupport(ids, valid, coverage)
    for fn in (lambda: mod.topk_entropy_components(z, candidates), lambda: mod.entropy_support_mass_components(z, z, candidates)):
        with pytest.raises(ValueError): fn()


@pytest.mark.parametrize("k,mode,include_gt", [(1, "independent", False), (4, "independent", False), (2.5, "independent", False), (True, "independent", False), (2, "wrong", False), (2, "independent", 1)])
def test_candidate_option_validation(k, mode, include_gt):
    with pytest.raises(ValueError):
        kernels().build_entropy_support(torch.zeros(1, 4), torch.zeros(1, 4), torch.tensor([0]), k=k, mode=mode, include_gt=include_gt)
