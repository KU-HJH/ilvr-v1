"""Production skips audits while optimizer/EMA semantics and live noise survive."""
import importlib

import pytest
import torch

from tests.stage1.test_trainer import (
    CountingStudent, CountingTargetBuilder, make_batch, make_deepspeed_trainer,
    make_trainer,
)


def test_policy_preserves_saved_behavior_and_restricts_validation():
    assert importlib.util.find_spec('stage1.run_cache'), 'check policy is missing'
    from stage1.run_cache import resolve_stage1_check_policy as resolve
    assert resolve(requested=None, saved=None, validation=False) == 'production'
    assert resolve(requested=None, saved={}, validation=False) == 'legacy'
    assert resolve(requested=None, saved={'check_policy': 'production'}, validation=False) == 'production'
    assert resolve(requested='validation', saved=None, validation=True) == 'validation'
    with pytest.raises(ValueError, match='validation'):
        resolve(requested='validation', saved=None, validation=False)
    with pytest.raises(ValueError, match='resume'):
        resolve(requested='production', saved={}, validation=False)


@pytest.mark.parametrize('applied', [False, True])
def test_production_skips_optional_scans_but_keeps_applied_step(applied, monkeypatch):
    trainer, engine, _ = make_deepspeed_trainer(norm_groups=[torch.tensor(1.)], step_applied=applied)
    trainer.stage1_check_policy = 'production'
    monkeypatch.setattr(trainer, '_validate_deepspeed_gradient_norm',
                        lambda *_: pytest.fail('diagnostic norm executed in production'))
    engine.set_gradient_accumulation_boundary(True)
    engine.step()
    assert trainer._successful_optimizer_steps == int(applied)
    assert trainer.ema_teacher.update_count == int(applied)
    if applied:
        torch.testing.assert_close(trainer.ema_teacher.values_at_update[0],
                                   trainer.model.language_scale.detach())


def test_production_keeps_alignment_noise_and_omits_target_audits(monkeypatch):
    mask = torch.tensor([[False, True, False]])
    trainer = make_trainer(CountingStudent(hidden_size=2),
                          CountingTargetBuilder(torch.tensor([[1., 2.]]), mask, ((1,),)))
    trainer.stage1_check_policy = 'production'
    for name in ('_validate_targets', '_validate_ema_teacher_device'):
        monkeypatch.setattr(trainer, name, lambda *_: pytest.fail('optional target audit executed'))
    before = torch.random.get_rng_state()
    loss = trainer.compute_loss(trainer.model, make_batch(mask))
    assert not torch.equal(before, torch.random.get_rng_state())
    loss.backward()
    assert trainer.model.language_scale.grad is not None
    assert trainer._latest_stage1_diagnostics is None
