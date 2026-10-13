"""Native CPU AdamW oracles and exact current-step projection limits."""
from copy import deepcopy
import importlib
import pytest
import torch


def api():
    return importlib.import_module('stage1.v3_adamw_update')


@pytest.mark.parametrize('warmup', [0, 3])
@pytest.mark.parametrize('decay', [0., .01])
@pytest.mark.parametrize('lr', [0., .03])
def test_counterfactual_matches_native_without_touching_live_state(warmup, decay, lr):
    p = torch.nn.Parameter(torch.tensor([2., -.4, 1.]))
    opt = torch.optim.AdamW([p], lr=lr, weight_decay=decay, fused=False)
    for _ in range(warmup):
        p.grad = torch.tensor([.1, -.3, .7])
        opt.step()
    before = p.detach().clone()
    state = deepcopy(opt.state.get(p, {}))
    p.grad = torch.tensor([-.2, .5, .3])
    candidate = api().adamw_base_candidate(p, p.grad, opt.state.get(p, {}), opt.param_groups[0])
    assert torch.equal(p, before)
    assert set(opt.state.get(p, {})) == set(state)
    for key, value in state.items():
        assert torch.equal(opt.state[p][key], value)
    opt.step()
    torch.testing.assert_close(candidate, p, rtol=0, atol=0)


def test_base_clipping_matches_zero1_divisor():
    g = torch.tensor([3., 4.])
    clipped = api().clip_base_gradient(g, global_norm=torch.tensor(5.), max_norm=1.)
    torch.testing.assert_close(clipped, g * (1 / (torch.tensor(5.) + 1e-6)), rtol=0, atol=0)
    same = api().clip_base_gradient(g, global_norm=torch.tensor(5.), max_norm=0.)
    assert torch.equal(same, g) and same.data_ptr() != g.data_ptr()


def project(base, joint, h, *, lrs=None, masks=None, reduce=lambda x: x):
    return api().project_adamw_increment(tuple(base), tuple(joint), tuple(h),
        valid_masks=tuple(torch.ones_like(t, dtype=torch.bool) for t in base) if masks is None else tuple(masks),
        lrs=tuple(.03 for _ in base) if lrs is None else tuple(lrs), sum_fp64=reduce)


def test_projection_does_not_move_zero_lr_group():
    result = project([torch.zeros(1), torch.zeros(1)], [torch.zeros(1), -torch.ones(1)],
                     [torch.ones(1), torch.ones(1)], lrs=[0., .03])
    assert all(torch.equal(p, torch.zeros(1)) for p in result.parameters)
    assert result.dot_before == -1 and result.dot_after == 0
    with pytest.raises(ValueError, match='zero.LR'):
        project([torch.zeros(1)], [torch.ones(1)], [torch.ones(1)], lrs=[0.])


@pytest.mark.parametrize('h', [[0., 0.], [1., 1.]])
def test_unconstrained_keeps_native_joint_values(h):
    joint = torch.tensor([.25, -.125])
    result = project([torch.tensor([1e9, -1e9])], [joint], [torch.tensor(h)])
    assert torch.equal(result.parameters[0], joint)
    assert not result.projected and not result.rounding_fallback


def test_projection_global_stats_and_padding():
    calls = []
    def two_identical_ranks(x):
        assert x.dtype == torch.float64
        calls.append(x.clone())
        return x * 2
    result = project([torch.zeros(3)], [torch.tensor([-2., 0., 7.])],
        [torch.tensor([1., 1., 1e10])], masks=[torch.tensor([True, True, False])], reduce=two_identical_ranks)
    torch.testing.assert_close(result.parameters[0], torch.tensor([-1., 1., 7.]), rtol=0, atol=0)
    assert result.dot_before == -4. and result.dot_after == 0.
    assert result.gain_norm_before == (8. ** .5)
    assert calls


def test_tiny_nonzero_protection_suppresses_increment():
    result = project([torch.tensor([1.])], [torch.tensor([2.])], [torch.tensor([1e-16])])
    assert torch.equal(result.parameters[0], torch.tensor([1.]))
    assert result.tiny_h and result.dot_after == 0.


def test_negative_rounded_dot_falls_back_to_whole_base_increment():
    base = torch.tensor([2., 1.])
    joint = torch.tensor([2., 1. - 2**-23])
    result = project([base], [joint], [torch.ones(2)])
    assert result.projected and result.rounding_fallback
    assert torch.equal(result.parameters[0], base)
    assert result.dot_after == 0.


def test_fp32_protection_does_not_imply_bf16_protection():
    base = torch.tensor([2., 1.])
    joint = base + torch.tensor([1/256, -1/256])
    result = project([base], [joint], [torch.ones(2)])
    assert result.dot_after == 0.
    assert (result.parameters[0].bfloat16().float() - base.bfloat16().float()).sum() < 0


@pytest.mark.parametrize('damage', ['nan', 'dtype', 'shape', 'lr', 'mask'])
def test_projection_rejects_invalid_owned_values(damage):
    base, joint, h = torch.zeros(2), torch.ones(2), torch.ones(2)
    lr, mask = .03, torch.ones(2, dtype=torch.bool)
    if damage == 'nan': joint[0] = float('nan')
    if damage == 'dtype': h = h.double()
    if damage == 'shape': h = h[:1]
    if damage == 'lr': lr = -1.
    if damage == 'mask': mask = mask.float()
    with pytest.raises(ValueError):
        project([base], [joint], [h], lrs=[lr], masks=[mask])
