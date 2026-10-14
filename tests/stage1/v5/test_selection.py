"""Selection fixtures catch temperature, causal-mass and percentile mistakes."""
from dataclasses import replace

import pytest
import torch

from stage1.v5_config import Stage1V5Config
from stage1.v5_metadata import V5BatchMetadata, V5Block


def selectors():
    from stage1.v5_selection import select_v5_orig_positions, select_v5_text_keys
    return select_v5_orig_positions, select_v5_text_keys


def test_text_shortest_prefix_uses_original_sequence_positions():
    _, select = selectors()
    metadata = V5BatchMetadata((V5Block(0, 1, 8, 9, (10,), (1,), tuple(range(8))),))
    weights = torch.full((12,), .03)
    weights[2], weights[4] = .12, .10
    logits = torch.zeros(1, 3, requires_grad=True)
    selected_keys = select(metadata, {(0, 9): weights}, logits, logits, Stage1V5Config())
    assert selected_keys == {(0, 9): (2, 4)}
    assert logits.grad is None


def test_text_t_squared_precedes_epsilon_and_sample_centering():
    _, select = selectors()
    metadata = V5BatchMetadata((V5Block(0, 1, 10, 11, (12, 13), (0, 0), tuple(range(10))),))
    clean = torch.zeros(2, 2)
    reference = torch.tensor([[.004, 0.], [0., 0.]])
    weights = torch.zeros(14)
    weights[0] = .225  # 37.5% of candidate mass (.6).
    weights[1:10] = .375 / 9
    # Independently derive rho; omitting T² changes the first decision.
    lp = (clean.double() / 2).log_softmax(-1)
    lr = (reference.double() / 2).log_softmax(-1)
    d = 4 * (lp.exp() * (lp - lr)).sum(-1)
    u = -(d.clamp_min(0) + 1e-6).log()
    rho = .3 + .2 * (u - u.mean()).sigmoid()
    unscaled_u = -(d / 4 + 1e-6).log()
    wrong_rho = .3 + .2 * (unscaled_u - unscaled_u.mean()).sigmoid()
    assert rho[0] < .375 < wrong_rho[0]
    assert select(metadata, {(0, 11): weights, (0, 12): weights}, clean, reference,
                  Stage1V5Config()) == {(0, 11): (0,), (0, 12): (0, 1)}


@pytest.mark.parametrize('count,want', [(8, ()), (10, (2,)), (20, (2, 4))])
def test_text_low_mass_fallback_floor_and_stable_position_ties(count, want):
    _, select = selectors()
    metadata = V5BatchMetadata((V5Block(0, 1, count, count + 1,
        (count + 2,), (0,), tuple(range(count))),))
    weights = torch.zeros(count + 3)
    weights[2], weights[4] = .03, .03
    logits = torch.zeros(1, 3)
    assert select(metadata, {(0, count + 1): weights}, logits, logits,
                  Stage1V5Config()) == {(0, count + 1): want}


def test_text_empty_queries_and_cap_are_valid():
    _, select = selectors()
    metadata = V5BatchMetadata((V5Block(0, 1, 2, 3, (4,), (0,), (0, 1)),))
    logits = torch.zeros(1, 3)
    assert select(metadata, {(0, 3): torch.tensor([.3, .3, 0., 0.])}, logits, logits,
                  Stage1V5Config()) == {(0, 3): ()}
    assert select(V5BatchMetadata(()), {}, logits[:0], logits[:0], Stage1V5Config()) == {}


@pytest.mark.parametrize('mode,want', [
    ('clean', [True, True, False, False]),
    ('voff', [False, True, True, False]),
    ('gap', [True, False, True, False]),
    ('intersection', [False, True, False, False]),
])
def test_orig_percentile_masks_and_intersection_without_refill(mode, want):
    select, _ = selectors()
    metadata = V5BatchMetadata((V5Block(0, 1, 0, 1, (2, 3), (0, 0), ()),
                               V5Block(0, 3, 4, 5, (6, 7), (0, 0), (2, 3))))
    clean = torch.tensor([[0., 0.], [0., 1.], [0., 4.], [0., 8.]], requires_grad=True)
    off = torch.tensor([[0., 4.], [0., 1.], [0., 0.], [0., 8.]], requires_grad=True)
    mask = select(metadata, clean, off, Stage1V5Config(
        selection=mode, preference_entropy_top_percent=50))
    assert mask.tolist() == want
    assert mask.dtype == torch.bool and not mask.requires_grad


def test_orig_ranking_is_sample_local_and_ties_use_original_positions():
    select, _ = selectors()
    # Deliberately reversed block order: position 1 wins a tied ranking.
    metadata = V5BatchMetadata((V5Block(0, 3, 4, 5, (6,), (0,), ()),
                               V5Block(1, 1, 0, 1, (2,), (0,), ()),
                               V5Block(0, 1, 0, 1, (2,), (0,), ())))
    logits = torch.zeros(3, 3)
    config = Stage1V5Config(selection='clean', preference_entropy_top_percent=50)
    assert select(metadata, logits, None, config).tolist() == [False, True, True]
    assert select(metadata, logits, None, replace(config, selection='all',
        preference_entropy_top_percent=None)).tolist() == [True, True, True]


def test_off_dependent_selection_rejects_missing_branch():
    select, _ = selectors()
    metadata = V5BatchMetadata((V5Block(0, 1, 0, 1, (2,), (0,), ()),))
    with pytest.raises(ValueError, match='V-off'):
        select(metadata, torch.zeros(1, 3), None,
               Stage1V5Config(selection='gap', preference_entropy_top_percent=50))


@pytest.mark.parametrize('mode', ['voff', 'gap', 'intersection'])
def test_empty_orig_queries_do_not_require_an_unneeded_off_call(mode):
    select, _ = selectors()
    metadata = V5BatchMetadata((V5Block(0, 1, 0, 1, (), (), ()),))
    result = select(metadata, torch.empty(0, 4), None,
        Stage1V5Config(selection=mode, preference_entropy_top_percent=50))
    assert result.shape == (0,) and result.dtype == torch.bool
