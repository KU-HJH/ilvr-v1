"""Independent oracles catch truncated expectations, wrong gradients and chunks."""

import importlib
import importlib.util
import math

import pytest
import torch


def kernels():
    name = "step_preference.dil_bce_orig_loss"
    assert importlib.util.find_spec(name) is not None, "missing orig BCE kernel"
    return importlib.import_module(name)


def close(actual, expected):
    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype), atol=1e-5, rtol=1e-4)


def literal(clean, reference, targets):
    # Independent FP64 expression; reference realization is fixed.
    p = clean.double().softmax(-1)
    mu = reference.detach().double().softmax(-1)
    ratio = p / mu
    chosen = torch.log1p(1 / ratio.gather(1, targets[:, None]).squeeze(1))
    rejected = (mu * torch.log1p(ratio)).sum(-1)
    return chosen, rejected


def test_full_vocab_literal_oracle():
    mod = kernels()
    p = torch.tensor([[.5, .3, .2]], dtype=torch.float64).log().requires_grad_()
    mu = torch.tensor([[.2, .5, .3]], dtype=torch.float64).log().requires_grad_()
    result = mod.dil_bce_orig_components(p, mu, torch.tensor([0]))
    close(result.chosen, [0.336472236621213])
    close(result.reference, [0.638802095451739])
    close(result.chosen + result.reference, [0.975274332072952])
    close(result.policy_gt_logp, [math.log(.5)])
    close(result.reference_gt_logp, [math.log(.2)])
    close(result.gt_logratio, [math.log(2.5)])
    close(result.reference_prob_sum_error, [0.])
    (result.chosen + result.reference).sum().backward()
    assert mu.grad is None
    assert not result.reference_gt_logp.requires_grad


def test_equal_policy_reference_has_nonzero_single_gt_gradient():
    mu = torch.tensor([[.2, .5, .3]])
    z = mu.log().requires_grad_()
    result = kernels().dil_bce_orig_components(z, z.detach(), torch.tensor([0]))
    close(result.chosen, [math.log(2)])
    close(result.reference, [math.log(2)])
    grad, = torch.autograd.grad((result.chosen + result.reference).sum(), z)
    torch.testing.assert_close(grad, (mu - torch.tensor([[1., 0., 0.]])) / 2, atol=1e-5, rtol=1e-4)


def test_population_gradient_at_policy_reference_is_zero():
    z = torch.tensor([[.2, .5, .3]]).log().repeat(3, 1).requires_grad_()
    result = kernels().dil_bce_orig_components(z, z.detach(), torch.arange(3))
    grad, = torch.autograd.grad(((result.chosen + result.reference) * torch.tensor([.2, .5, .3])).sum(), z)
    close(grad.sum(0), [0., 0., 0.])


def test_wrong_expectations_and_mean_score_are_distinguishable():
    z = torch.tensor([[.5, .3, .2], [.1, .6, .3]], dtype=torch.float64).log().requires_grad_()
    ref = torch.tensor([[.2, .5, .3], [.4, .2, .4]], dtype=torch.float64).log()
    targets = torch.tensor([0, 2])
    chosen, expected = literal(z, ref, targets)
    p, mu = z.softmax(-1), ref.softmax(-1)
    ratio = p / mu
    gt_only = mu[:, 0] * torch.log1p(ratio[:, 0])
    truncated_reference = (mu[:, :2] * torch.log1p(ratio[:, :2])).sum(-1)
    expected_gradient, = torch.autograd.grad(expected.sum(), z, retain_graph=True)
    live_policy_weighted_gradient, = torch.autograd.grad((p * torch.log1p(ratio)).sum(), z, retain_graph=True)
    token_mean = chosen.mean()
    softplus_of_mean_score = torch.nn.functional.softplus(-(p.log() - mu.log()).gather(1, targets[:, None]).mean())
    assert not torch.allclose(expected, gt_only)
    assert not torch.allclose(expected, truncated_reference)
    assert not torch.allclose(expected_gradient, live_policy_weighted_gradient)
    assert not torch.allclose(token_mean, softplus_of_mean_score)
    result = kernels().dil_bce_orig_components(z, ref, targets)
    close(result.chosen, chosen.detach().float())
    close(result.reference, expected.detach().float())
    actual, = torch.autograd.grad(result.reference.sum(), z)
    close(actual, expected_gradient.float())


@pytest.mark.parametrize("kind", ["shape", "target_shape", "target_float", "negative_gt", "large_gt", "nan", "inf", "fp32_overflow", "integer_logits", "empty_vocab", "device"])
def test_invalid_inputs(kind):
    z = torch.zeros(2, 3)
    ref = z.clone()
    y = torch.tensor([0, 1])
    if kind == "shape": ref = torch.zeros(2, 4)
    if kind == "target_shape": y = y[:, None]
    if kind == "target_float": y = y.float()
    if kind == "negative_gt": y[0] = -1
    if kind == "large_gt": y[0] = 3
    if kind == "nan": z[0, 0] = float("nan")
    if kind == "inf": ref[0, 0] = float("inf")
    if kind == "fp32_overflow": z = torch.full((2, 3), 1e300, dtype=torch.float64)
    if kind == "integer_logits": z = z.long()
    if kind == "empty_vocab": z = ref = torch.empty(2, 0)
    if kind == "device": ref = torch.empty(2, 3, device="meta")
    mod = kernels()
    with pytest.raises(ValueError):
        mod.dil_bce_orig_components(z, ref, y)


def test_empty_positions_stay_student_connected():
    z = torch.empty(0, 3, requires_grad=True)
    ref = torch.empty(0, 3, requires_grad=True)
    mod = kernels()
    for fn in (mod.dil_bce_orig_components, mod.orig_components_chunked):
        result = fn(z, ref, torch.empty(0, dtype=torch.long))
        assert result.chosen.shape == result.reference.shape == (0,)
        grad, = torch.autograd.grad((result.chosen + result.reference).sum(), z)
        assert grad.shape == z.shape
    assert ref.grad is None


@pytest.mark.parametrize("branch", ["clean", "reference"])
@pytest.mark.parametrize("api", ["dil_bce_orig_components", "orig_components_chunked"])
def test_unrepresentable_fp32_normalization_fails_actionably(branch, api):
    mod = kernels()
    z = torch.zeros(1, 3, requires_grad=True)
    ref = torch.zeros(1, 3)
    extreme = torch.tensor([[3e38, -3e38, 0.]])
    if branch == "clean": z = extreme.requires_grad_()
    else: ref = extreme
    with pytest.raises(ValueError, match="FP32 log probabilities.*logit magnitude"):
        getattr(mod, api)(z, ref, torch.tensor([0]))


def test_chunk_1_128_all_loss_and_gradient():
    mod = kernels()
    generator = torch.Generator().manual_seed(42)
    z0 = torch.randn(259, 11, generator=generator)
    ref = torch.randn(259, 11, generator=generator, requires_grad=True)
    y = torch.randint(11, (259,), generator=generator)
    expected_z = z0.double().requires_grad_()
    chosen, rejected = literal(expected_z, ref, y)
    expected_grad, = torch.autograd.grad((chosen + rejected).mean(), expected_z)
    for chunk in (1, 128, 259):
        for use_checkpoint in (False, True):
            z = z0.clone().requires_grad_()
            result = mod.orig_components_chunked(z, ref, y, chunk_positions=chunk, checkpoint=use_checkpoint)
            close(result.chosen, chosen.detach().float())
            close(result.reference, rejected.detach().float())
            grad, = torch.autograd.grad((result.chosen + result.reference).mean(), z)
            close(grad, expected_grad.float())
    assert ref.grad is None


def test_checkpoint_does_not_save_full_fp32_loss_intermediates():
    mod = kernels()
    def saved_shapes(checkpoint):
        saved = []
        z = torch.randn(259, 17, requires_grad=True)
        ref = torch.randn_like(z)
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append((t.shape, t.data_ptr())) or t, lambda t: t):
            result = mod.orig_components_chunked(z, ref, torch.zeros(259, dtype=torch.long), chunk_positions=128, checkpoint=checkpoint)
        # Views of retained input logits are allowed; new dense FP32 loss intermediates are not.
        pointers = {z.data_ptr() + start * 17 * z.element_size() for start in (0, 128, 256)}
        pointers.update(ref.data_ptr() + start * 17 * ref.element_size() for start in (0, 128, 256))
        intermediates = [shape for shape, ptr in saved if len(shape) == 2 and shape[1] == 17 and ptr not in pointers]
        (result.chosen + result.reference).sum().backward()
        return intermediates
    assert saved_shapes(False)
    assert saved_shapes(True) == []


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5])
def test_chunk_size_validation(chunk):
    with pytest.raises(ValueError):
        kernels().orig_components_chunked(torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor([0]), chunk_positions=chunk)
