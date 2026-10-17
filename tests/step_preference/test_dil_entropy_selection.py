import pytest
import torch

from step_preference.entropy_selection import select_entropy_masks


def test_mask_quota_and_ties():
    """Catches float-ceil drift, unstable ties, zero tiny quotas, or cross-sample pooling."""
    mask, = select_entropy_masks([torch.ones(25)], 28.)
    assert torch.equal(mask, torch.arange(25) < 7)
    tiny, = select_entropy_masks([torch.ones(25)], 1e-8)
    assert torch.equal(tiny, torch.arange(25) < 1)
    first, second = select_entropy_masks([torch.tensor([1., 3.]), torch.tensor([3., 2.])], 50.)
    assert torch.equal(first, torch.tensor([False, True]))
    assert torch.equal(second, torch.tensor([True, False]))
    one_sample, = select_entropy_masks([torch.tensor([1., 3.])], 50.)
    other_sample, = select_entropy_masks([torch.tensor([3., 2.])], 50.)
    assert torch.equal(one_sample, torch.tensor([False, True]))
    assert torch.equal(other_sample, torch.tensor([True, False]))


def test_empty_vectors_keep_shapes_and_detach_ranking():
    """Catches empty-step loss or entropy ranking that retains a gradient graph."""
    entropies = torch.tensor([.1, .9], requires_grad=True)
    empty, selected = select_entropy_masks([torch.empty(0), entropies], 50.)
    assert empty.shape == (0,) and empty.dtype == torch.bool
    assert torch.equal(selected, torch.tensor([False, True]))
    assert not selected.requires_grad
    assert select_entropy_masks([], 50.) == ()
    all_empty = select_entropy_masks([torch.empty(0), torch.empty(0)], 50.)
    assert len(all_empty) == 2
    assert all(value.shape == (0,) and value.dtype == torch.bool for value in all_empty)


@pytest.mark.parametrize('percent', [0., -1., 101., float('nan'), float('inf'), True])
def test_entropy_selection_rejects_invalid_percent(percent):
    """Catches invalid or ambiguous percentage quotas."""
    with pytest.raises(ValueError, match='percent'):
        select_entropy_masks([torch.ones(1)], percent)
