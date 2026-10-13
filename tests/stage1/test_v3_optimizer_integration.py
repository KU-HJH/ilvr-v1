"""The actual engine adapter must route Gain through native Adam moments."""
from copy import deepcopy
from types import SimpleNamespace
import math

import pytest
import torch
from trl import SFTTrainer
from stage1.trainer import CustomTrainerStage1, _DeepSpeedStepAdapter
from stage1.v3_config import Stage1V3Config
from stage1.v3_gradients import V3Gradients
from stage1.v3_update import V3GainController
from stage1 import v3_zero1_update, v3_update
from .v3_zero1_helpers import Collective, ControlledZero, engine_for, install_cpu_collectives


def harness(monkeypatch, *, weighting='none', weight=1., groups=None, lrs=None, skip=False, fail=False, scheduler_lr=None):
    install_cpu_collectives(monkeypatch, v3_zero1_update)
    zero = ControlledZero(groups or [[[.5, 1.]]], collective=Collective(1), lrs=lrs)
    engine = engine_for(zero)
    engine.is_gradient_accumulation_boundary = lambda: True
    engine.was_step_applied = lambda: not skip
    config = Stage1V3Config(gain_update_mode='adamw_projected', gain_optimizer_weight=weight, gain_weighting=weighting)
    controller = V3GainController(zero.all_parameters, config)
    trainer = object.__new__(CustomTrainerStage1)
    trainer.v3_gain_controller = controller
    trainer.model = torch.nn.Linear(1, 1)
    trainer._successful_optimizer_steps = 0
    trainer.ema_teacher = SimpleNamespace(decay=.9, update=lambda model: zero.events.append('ema'))
    trainer._latest_stage1_diagnostics = {'ordinary': torch.tensor(3.)}
    trainer.collect_stage1_diagnostics = True
    logged = []
    monkeypatch.setattr(SFTTrainer, 'log', lambda self, logs, **kw: logged.append(logs))
    def norm_guard(engine):
        zero.events.append('joint_norm_check')
        assert torch.isfinite(zero.scaled_global_norm())
    trainer._validate_deepspeed_gradient_norm = norm_guard
    def step():
        if fail: raise RuntimeError('injected native failure')
        if not skip: zero.step()
        if scheduler_lr is not None:
            for group in zero.optimizer.param_groups: group['lr'] = scheduler_lr
    engine.step = _DeepSpeedStepAdapter(trainer, engine, step)
    engine.fixture_trainer = trainer  # Keep the weakly referenced real adapter owner alive.
    monkeypatch.setattr(v3_update, 'safe_get_full_fp32_param', lambda *a: pytest.fail('new mode gathered full masters'))
    monkeypatch.setattr(v3_update, 'safe_set_full_fp32_param', lambda *a: pytest.fail('new mode set full masters'))
    return zero, engine, controller, trainer, logged


def accumulate(controller, gains, hs, count=1):
    g = tuple(torch.as_tensor(v, dtype=torch.float32) for v in gains)
    h = tuple(torch.as_tensor(v, dtype=torch.float32) for v in hs)
    scalar = torch.tensor(0.)
    if controller.config.gain_weighting == 'token_entropy':
        result = V3Gradients(None, tuple(t*count for t in h), count, scalar, scalar,
                             gain_sum_grads=tuple(t*count for t in g))
    else:
        result = V3Gradients(tuple((a+b)*count for a,b in zip(g,h)), tuple(t*count for t in h), count, scalar, scalar)
    controller.accumulate(result)


def native_oracle(zero, gradients):
    params = [torch.nn.Parameter(p.detach().clone()) for p in zero.single_partition_of_fp32_groups]
    groups = [dict(group, params=[p]) for group, p in zip(zero.optimizer.param_groups, params)]
    opt = torch.optim.AdamW(groups, fused=False)
    opt.load_state_dict(deepcopy(zero.optimizer.state_dict()))
    norm = torch.stack([g.double().norm().square().float().sqrt() for g in gradients]).norm()
    divisor = ((norm+1e-6)/zero.clip_grad).clamp(min=1.) if zero.clip_grad else 1.
    for p, g in zip(params, gradients): p.grad = g.clone() * (1./divisor)
    opt.step()
    return params, opt


@pytest.mark.parametrize('weighting', ['none', 'token_entropy'])
def test_base_and_joint_clip_with_their_own_norms(monkeypatch, weighting):
    zero, engine, controller, trainer, logs = harness(monkeypatch, weighting=weighting)
    zero.averaged_gradients[0] = [torch.tensor([3., 4.], dtype=torch.bfloat16)]
    base, _ = native_oracle(zero, [torch.tensor([3., 4.])])
    joint, opt = native_oracle(zero, [torch.tensor([-1., 4.])])
    start = zero.single_partition_of_fp32_groups[0].detach().clone()
    accumulate(controller, [[4., 0.]], [[1., 1.]])
    engine.step()
    master = zero.single_partition_of_fp32_groups[0]
    torch.testing.assert_close(master, joint[0], rtol=0, atol=0)
    for name in ('step', 'exp_avg', 'exp_avg_sq'):
        assert torch.equal(zero.optimizer.state[master][name], opt.state[joint[0]][name])
    metrics = controller.last_metrics
    assert metrics['base_gradient_norm'] == 5.
    assert metrics['joint_gradient_norm'] == pytest.approx(math.sqrt(17))
    assert metrics['base_displacement_norm'] == pytest.approx(float((base[0].double()-start.double()).norm()))
    assert metrics['rv_dot_after'] >= 0
    assert zero.events == ['joint_norm_check', 'native_step', 'group_bf16_copy', 'projection_sync', 'ema']
    assert controller.state == 'IDLE' and trainer._successful_optimizer_steps == 1
    assert not zero.optimizer._optimizer_step_pre_hooks and not zero.optimizer._optimizer_step_post_hooks
    assert logs[-1]['stage1_v3/successful_steps'] == 1
    assert trainer._latest_stage1_diagnostics is not None


@pytest.mark.parametrize('weighting', ['none', 'token_entropy'])
def test_new_mode_keeps_unprojected_gain_and_raw_global_rv(monkeypatch, weighting):
    _, _, controller, _, _ = harness(monkeypatch, weighting=weighting)
    accumulate(controller, [[1., -2.]], [[1., 1.]], count=3)
    controller.prepare()
    assert torch.equal(controller.raw_gain[0], torch.tensor([1., -2.]))
    assert torch.equal(controller.raw_rv[0], torch.ones(2))
    assert controller.global_block_count == 3


def test_rejected_gain_retains_joint_momentum_on_next_zero_gain_step(monkeypatch):
    zero, engine, controller, _, _ = harness(monkeypatch, groups=[[[0.]]])
    zero.reset_gradients()
    zero.averaged_gradients[0][0].zero_()
    _, native = native_oracle(zero, [torch.tensor([1., 0.])])
    accumulate(controller, [[-1.]], [[1.]])
    engine.step()
    master = zero.single_partition_of_fp32_groups[0]
    assert master[0].item() == 0.
    assert controller.last_metrics['projection_count'] == 1
    for name in ('step', 'exp_avg', 'exp_avg_sq'):
        assert torch.equal(zero.optimizer.state[master][name], next(iter(native.state.values()))[name])
    zero.reset_gradients()
    zero.averaged_gradients[0][0].zero_()
    accumulate(controller, [[0.]], [[1.]])
    engine.step()
    assert master[0].item() < 0  # Shared pre-step state already carries historical Gain.
    assert controller.last_metrics['gain_displacement_norm_after'] == 0.


@pytest.mark.parametrize('lrs', [[0., .03], [0., 0.]])
def test_captured_group_lr_survives_scheduler_advancement(monkeypatch, lrs):
    zero, engine, controller, _, _ = harness(monkeypatch, groups=[[[0.]], [[0.]]], lrs=lrs, scheduler_lr=.04)
    for gradients in zero.averaged_gradients.values():
        for g in gradients: g.zero_()
    accumulate(controller, [[-1.], [-1.]], [[1.], [1.]])
    engine.step()
    for index, master in enumerate(zero.single_partition_of_fp32_groups):
        assert master.count_nonzero() == 0  # LR0 frozen; active group projected to base zero.
        assert zero.optimizer.state[master]['step'].item() == 1
        assert zero.optimizer.state[master]['exp_avg'][0] > 0
        assert controller.last_metrics[f'lr_group_{index}'] == lrs[index]
        assert zero.optimizer.param_groups[index]['lr'] == .04


@pytest.mark.parametrize('skip,fail', [(True,False), (False,True)])
def test_skip_or_failure_does_not_advance_ema_and_cleans_hooks(monkeypatch, skip, fail):
    zero, engine, controller, trainer, logs = harness(monkeypatch, skip=skip, fail=fail)
    before = zero.single_partition_of_fp32_groups[0].clone()
    accumulate(controller, [[1., 1.]], [[1., 1.]])
    if fail:
        with pytest.raises(RuntimeError, match='injected'): engine.step()
        with pytest.raises(RuntimeError, match='checkpoint'): controller.assert_checkpointable()
        assert controller.state == 'FAILED'
    else:
        engine.step()
        controller.assert_checkpointable()
        assert logs[-1]['stage1_v3/applied'] is False
        assert 'stage1_v3/rv_dot_after' not in logs[-1]
    assert trainer._successful_optimizer_steps == 0 and 'ema' not in zero.events
    assert torch.equal(before, zero.single_partition_of_fp32_groups[0])
    assert not zero.optimizer.state
    assert not zero.optimizer._optimizer_step_pre_hooks and not zero.optimizer._optimizer_step_post_hooks


def test_cpu_saved_optimizer_trajectory_matches_uninterrupted(monkeypatch, tmp_path):
    zero, engine, controller, _, _ = harness(monkeypatch)
    accumulate(controller, [[1., -2.]], [[1., 1.]])
    engine.step()
    saved = tmp_path/'state.pt'
    torch.save({'master': [p.detach().clone() for p in zero.single_partition_of_fp32_groups],
                'optimizer': zero.optimizer.state_dict()}, saved)
    zero.reset_gradients()
    accumulate(controller, [[-3., 2.]], [[1., 1.]])
    engine.step()
    resumed, engine2, controller2, _, _ = harness(monkeypatch)
    state = torch.load(saved, weights_only=True)
    for p, value in zip(resumed.single_partition_of_fp32_groups, state['master']): p.data.copy_(value)
    resumed.optimizer.load_state_dict(state['optimizer'])
    accumulate(controller2, [[-3., 2.]], [[1., 1.]])
    engine2.step()
    for p, q in zip(zero.single_partition_of_fp32_groups, resumed.single_partition_of_fp32_groups):
        assert torch.equal(p,q)
        for name in zero.optimizer.state[p]:
            assert torch.equal(zero.optimizer.state[p][name],resumed.optimizer.state[q][name])
