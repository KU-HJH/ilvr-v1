from __future__ import annotations

import pytest
import torch
from transformers import TrainerControl, TrainerState

from evidence_window.config import (
    EvidenceWindowConfig,
    PreferenceConfig,
    ReferenceConfig,
)


TOKEN_IDS = {
    "start": 90,
    "end": 91,
    "pad": 92,
    "padding": 0,
    "special": (0, 90, 91, 92, 99),
}


def _assert_snapshot_equal(snapshot, module):
    current = dict(module.named_parameters())
    assert snapshot.keys() == current.keys()
    assert all(torch.equal(before, current[name]) for name, before in snapshot.items())


@pytest.mark.parametrize("enabled", [False, True], ids=["baseline", "frozen-reference"])
def test_trainer_optimizer_callback_updates_ema_once_per_accumulated_step(
    enabled,
    tiny_trainer_factory,
    legacy_latent_case,
    snapshot_parameters,
):
    """Removing the optimizer callback must leave the EMA observably stale."""
    config = (
        EvidenceWindowConfig(
            reference=ReferenceConfig(mode="frozen_copy"),
            preference=PreferenceConfig(lambda_ew=1.0),
        )
        if enabled
        else EvidenceWindowConfig.disabled()
    )
    trainer, model = tiny_trainer_factory(
        evidence_window_config=config,
        latent_token_ids=TOKEN_IDS,
        ema_tau=0.5,
    )
    callbacks = [
        callback
        for callback in trainer.callback_handler.callbacks
        if callback.__class__.__name__ == "_EMAOptimizerStepCallback"
    ]
    assert len(callbacks) == 1
    assert trainer.ema_update_count == 0

    student_before = snapshot_parameters(model)
    ema_before = snapshot_parameters(trainer._ema.teacher)
    reference_before = (
        None
        if trainer._reference is None
        else snapshot_parameters(trainer._reference.model)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer.callback_handler.optimizer = optimizer
    optimizer.zero_grad(set_to_none=True)

    first_loss = None
    for microbatch in range(2):
        torch.manual_seed(1234)
        loss = trainer.compute_loss(
            model,
            dict(legacy_latent_case["inputs"]),
        )
        assert torch.isfinite(loss)
        if first_loss is None:
            first_loss = loss.detach()
        (loss / 2).backward()
        if microbatch == 0:
            trainer.callback_handler.on_substep_end(
                trainer.args,
                TrainerState(),
                TrainerControl(),
            )
            assert trainer.ema_update_count == 0
            _assert_snapshot_equal(ema_before, trainer._ema.teacher)

    if not enabled:
        assert torch.allclose(
            first_loss,
            legacy_latent_case["recorded_legacy_loss"],
            atol=1e-6,
            rtol=1e-6,
        )
        assert model.forward_count == 2

    optimizer.step()
    student_after = dict(model.named_parameters())
    assert any(
        not torch.equal(before, student_after[name])
        for name, before in student_before.items()
    )
    trainer.callback_handler.on_optimizer_step(
        trainer.args,
        TrainerState(),
        TrainerControl(),
    )

    assert trainer.ema_update_count == 1
    ema_after = dict(trainer._ema.teacher.named_parameters())
    for name, before in ema_before.items():
        expected = 0.5 * before + 0.5 * student_after[name].detach()
        assert torch.allclose(ema_after[name], expected, atol=1e-7, rtol=1e-7)
    assert all(parameter.grad is None for parameter in trainer._ema.teacher.parameters())
    if reference_before is not None:
        _assert_snapshot_equal(reference_before, trainer._reference.model)
        assert all(parameter.grad is None for parameter in trainer._reference.parameters())
