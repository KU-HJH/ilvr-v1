import importlib
from copy import deepcopy
import pytest
import torch
from stage1.v3_config import Stage1V3Config
from .v3_zero1_helpers import Collective, ControlledZero, engine_for, install_cpu_collectives, run_ranks


def make(monkeypatch, *, weight=.25, lrs=None):
    api = importlib.import_module('stage1.v3_zero1_update')
    install_cpu_collectives(monkeypatch, api)
    zero = ControlledZero([[[.5, 1.]]], collective=Collective(1), lrs=lrs)
    config = Stage1V3Config(gain_update_mode='adamw_projected', gain_optimizer_weight=weight)
    adapter = api.Zero1AdamWGainAdapter(engine_for(zero), zero.all_parameters, config)
    return api, zero, adapter


def test_owned_fragments_cross_parameter_and_padding(monkeypatch):
    api = importlib.import_module('stage1.v3_zero1_update')
    install_cpu_collectives(monkeypatch, api)
    collective = Collective(4)
    zeros = [ControlledZero([[[.5]*5, [1.]*6], [[2.]*3]], collective=collective, rank=r) for r in range(4)]
    config = Stage1V3Config(gain_update_mode='adamw_projected', gain_optimizer_weight=.25)
    dictionaries = [z.averaged_gradients for z in zeros]
    def work(zero):
        adapter = api.Zero1AdamWGainAdapter(engine_for(zero), zero.all_parameters, config)
        g = tuple(torch.full_like(p, .001, dtype=torch.float32) for p in zero.all_parameters)
        adapter.prepare(g, tuple(torch.zeros_like(t) for t in g), block_count=3)
        observed = [torch.cat([g.flatten() for g in zero.averaged_gradients[i]]) for i in range(2)]
        zero.step()
        metrics = adapter.finish(successful=True)
        return observed, metrics
    results = run_ranks([lambda z=z: work(z) for z in zeros])
    for gi, length in enumerate([11, 3]):
        joined = torch.cat([r[0][gi] for r in results])
        assert joined.dtype == torch.float32
        torch.testing.assert_close(joined[:length], torch.full((length,), 1. - .25*.001), rtol=0, atol=0)
        assert joined[length:].count_nonzero() == 0
    for z, dictionary in zip(zeros, dictionaries):
        assert z.averaged_gradients is dictionary
        assert len(z.optimizer._optimizer_step_pre_hooks) == len(z.optimizer._optimizer_step_post_hooks) == 0
    assert all(trace == collective.traces[0] for trace in collective.traces)


@pytest.mark.parametrize('bad', ['cast', 'nan_moment', 'negative_moment'])
def test_native_guard_prevents_group_copy(monkeypatch, bad):
    _, zero, adapter = make(monkeypatch)
    master = zero.single_partition_of_fp32_groups[0]
    def corrupt(opt, args, kwargs):
        if bad == 'cast': master.data.fill_(torch.finfo(torch.float32).max)
        elif bad == 'nan_moment': opt.state[master]['exp_avg'].fill_(float('nan'))
        else: opt.state[master]['exp_avg_sq'].fill_(-1)
    damage = zero.optimizer.register_step_post_hook(corrupt)
    adapter.prepare((torch.ones(2),), (torch.ones(2),), block_count=1)
    with pytest.raises(RuntimeError, match='native'):
        zero.step()
    adapter.abort()
    damage.remove()
    assert 'group_bf16_copy' not in zero.events
    assert not zero.optimizer._optimizer_step_pre_hooks
    assert not zero.optimizer._optimizer_step_post_hooks


def test_remote_native_cast_failure_prevents_group_copy(monkeypatch):
    api = importlib.import_module('stage1.v3_zero1_update')
    install_cpu_collectives(monkeypatch, api)
    collective = Collective(4)
    zeros = [ControlledZero([[[1.]*8]], collective=collective, rank=r) for r in range(4)]
    config = Stage1V3Config(gain_update_mode='adamw_projected', gain_optimizer_weight=.25)
    def work(zero):
        adapter = api.Zero1AdamWGainAdapter(engine_for(zero), zero.all_parameters, config)
        handle = None
        if zero.dp_process_group.rank == 2:
            handle = zero.optimizer.register_step_post_hook(lambda opt, a, k:
                zero.single_partition_of_fp32_groups[0].data.fill_(torch.finfo(torch.float32).max))
        adapter.prepare((torch.ones(8),), (torch.zeros(8),), block_count=1)
        try:
            with pytest.raises(RuntimeError, match='native'):
                zero.step()
        finally:
            adapter.abort()
            if handle: handle.remove()
        assert 'group_bf16_copy' not in zero.events
    run_ranks([lambda z=z: work(z) for z in zeros])
    assert all(t == collective.traces[0] for t in collective.traces)


@pytest.mark.parametrize('damage', ['amsgrad', 'maximize', 'capturable', 'differentiable', 'foreach',
    'offload', 'loss_scale', 'version', 'missing_gradient', 'mapping_gap', 'gradient_dictionary'])
def test_reject_unsupported_profile_or_mapping_before_mutation(monkeypatch, damage):
    api, zero, adapter = make(monkeypatch)
    before = zero.single_partition_of_fp32_groups[0].clone()
    if damage in ('amsgrad', 'maximize', 'capturable', 'differentiable', 'foreach'):
        zero.optimizer.param_groups[0][damage] = True
    elif damage == 'offload': zero.cpu_offload = True
    elif damage == 'loss_scale': zero.loss_scale = 2.
    elif damage == 'version': monkeypatch.setattr(api, 'version', lambda name: '0.0')
    elif damage == 'missing_gradient': zero.averaged_gradients[0] = []
    elif damage == 'mapping_gap': zero.all_parameters[0]._hp_mapping.hp_fragment_address.start = 1
    else: zero.all_parameters[0]._hp_mapping.gradient_dict = {}
    with pytest.raises(RuntimeError):
        adapter.prepare((torch.ones(2),), (torch.ones(2),), block_count=1)
    adapter.abort()
    assert torch.equal(before, zero.single_partition_of_fp32_groups[0])
    assert not zero.optimizer.state
    assert not zero.optimizer._optimizer_step_post_hooks


@pytest.mark.parametrize('weight,blocks,reason', [(0., 1, 'zero_weight'), (.25, 0, 'zero_blocks')])
def test_disabled_gain_uses_native_step_without_counterfactual(monkeypatch, weight, blocks, reason):
    api, zero, adapter = make(monkeypatch, weight=weight)
    monkeypatch.setattr(api, 'adamw_base_candidate', lambda *a, **k: pytest.fail('unexpected base calculation'))
    adapter.prepare((torch.zeros(2),), (torch.ones(2),), block_count=blocks)
    zero.step()
    metrics = adapter.finish(successful=True)
    assert metrics['bypass_reason'] == reason
    assert metrics['rv_dot_before'] is None
    assert metrics['applied'] and metrics['projection_count'] == 0
    assert 'projection_sync' not in zero.events


def test_skip_cleans_hooks_without_parameter_or_state_change(monkeypatch):
    _, zero, adapter = make(monkeypatch)
    before = zero.single_partition_of_fp32_groups[0].clone()
    adapter.prepare((torch.ones(2),), (torch.ones(2),), block_count=1)
    metrics = adapter.finish(successful=False)
    assert not metrics['applied']
    assert torch.equal(before, zero.single_partition_of_fp32_groups[0])
    assert not zero.optimizer.state
    assert not zero.optimizer._optimizer_step_pre_hooks
    assert not zero.optimizer._optimizer_step_post_hooks
