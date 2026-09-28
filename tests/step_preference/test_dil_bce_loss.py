import math

import pytest
import torch

from step_preference.dil_bce_loss import dil_bce_loss, dil_bce_positive


def test_zero_score_is_two_log_two_and_signed_gradients():
    """Catches a paired margin objective or an incorrect clean/OFF sign."""
    clean = torch.zeros(1, requires_grad=True)
    off = torch.zeros(1, requires_grad=True)
    result = dil_bce_loss(clean, off, .5)
    assert result.loss.item() == pytest.approx(2 * math.log(2))
    torch.testing.assert_close(result.pair_losses, torch.tensor([2 * math.log(2)]))
    torch.testing.assert_close(result.policy_margins, torch.tensor([0.]))
    result.loss.backward()
    torch.testing.assert_close(clean.grad, torch.tensor([-.25]))
    torch.testing.assert_close(off.grad, torch.tensor([.25]))


def test_positive_is_independent_and_fp32_for_bf16():
    """Catches a negative term leaking into the clean-only positive path."""
    clean = torch.tensor([-2., 0.], dtype=torch.bfloat16, requires_grad=True)
    result = dil_bce_positive(clean, .5)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([math.log1p(math.e), math.log(2.)]), atol=1e-6, rtol=1e-5)
    result.sum().backward()
    assert clean.grad is not None
    assert bool((clean.grad < 0).all())


def test_empty_bce_has_differentiable_zero():
    """Catches an empty mean producing NaN or a disconnected constant."""
    clean = torch.empty(0, requires_grad=True)
    off = torch.empty(0, requires_grad=True)
    result = dil_bce_loss(clean, off, .5)
    assert result.loss.dtype == torch.float32
    assert result.loss.item() == 0.
    result.loss.backward()
    assert clean.grad.numel() == off.grad.numel() == 0


@pytest.mark.parametrize('beta', [0., -1., float('nan'), float('inf'), True])
def test_bce_rejects_invalid_beta(beta):
    """Catches undefined or inverted objective temperatures."""
    with pytest.raises(ValueError, match='beta'):
        dil_bce_loss(torch.zeros(1), torch.zeros(1), beta)


def test_bce_rejects_shape_device_and_fp32_overflow():
    """Catches silent score broadcasting and finite FP64 overflow on cast."""
    with pytest.raises(ValueError, match='shape'):
        dil_bce_loss(torch.zeros(1), torch.zeros(2), .5)
    with pytest.raises(ValueError, match='finite'):
        dil_bce_loss(torch.tensor([1e100]), torch.zeros(1), .5)
    with pytest.raises(ValueError, match='device'):
        dil_bce_loss(torch.zeros(1), torch.zeros(1, device='meta'), .5)
