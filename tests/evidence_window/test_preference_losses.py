import torch
import pytest

from evidence_window.losses import (
    evidence_window_preference_loss,
    global_preference_loss,
    window_preference_loss,
)
from evidence_window.scores import evidence_window_margins


def test_prefix_cancellation_equals_direct_window_sum():
    delta = torch.tensor([[[0.0, 3.0, 5.0, 7.0, 11.0]]])
    masks = torch.tensor(
        [[[[0, 1, 1, 0, 0], [0, 0, 0, 1, 1]]]], dtype=torch.bool
    )
    direct = evidence_window_margins(delta, masks)
    cumulative = delta.cumsum(-1)
    assert direct[0, 0, 0] == cumulative[0, 0, 2] - cumulative[0, 0, 0]
    assert direct[0, 0, 1] == cumulative[0, 0, 4] - cumulative[0, 0, 2]


def test_global_saturation_suppresses_m2_but_window_loss_retains_signal():
    margins = torch.tensor([[20.0, 0.0]], requires_grad=True)
    valid = torch.ones_like(margins, dtype=torch.bool)
    global_preference_loss(margins, valid, beta=1.0).backward(retain_graph=True)
    global_grad = margins.grad[0, 1].abs().item()
    margins.grad.zero_()
    window_preference_loss(margins, valid, beta=1.0).backward()
    local_grad = margins.grad[0, 1].abs().item()
    assert global_grad < 1e-7
    assert local_grad > 0.2


def test_invalid_global_rows_and_empty_window_masks_are_differentiable_zero():
    margins = torch.tensor([[3.0, -1.0], [2.0, 5.0]], requires_grad=True)
    valid = torch.tensor([[False, False], [True, False]])
    global_loss = global_preference_loss(margins, valid, beta=1.0)
    window_loss = window_preference_loss(margins[:1], valid[:1], beta=1.0)
    assert torch.isfinite(global_loss)
    assert torch.isfinite(window_loss)
    assert window_loss == 0
    (global_loss + window_loss).backward()
    assert torch.isfinite(margins.grad).all()
    assert margins.grad[0].abs().sum() == 0


def test_evidence_window_loss_uses_pair_weights_without_length_normalization():
    margins = torch.tensor([[[2.0, 0.0], [1.0, -1.0]]])
    pair_mask = torch.tensor([[[True, False], [True, False]]])
    weights = torch.tensor([[[2.0, 0.0], [1.0, 0.0]]])
    expected = (
        2 * torch.nn.functional.softplus(torch.tensor(-2.0))
        + torch.nn.functional.softplus(torch.tensor(-1.0))
    ) / 3
    assert torch.allclose(
        evidence_window_preference_loss(margins, pair_mask, beta=1.0, weights=weights),
        expected,
    )


def test_invalid_loss_shapes_raise_value_error():
    with pytest.raises(ValueError):
        global_preference_loss(torch.zeros(2, 3), torch.zeros(2, 2, dtype=torch.bool), 1.0)
    with pytest.raises(ValueError):
        evidence_window_preference_loss(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, 2, dtype=torch.bool), 1.0
        )
