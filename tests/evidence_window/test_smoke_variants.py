from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch
from transformers import TrainerControl, TrainerState

from evidence_window.config import EvidenceWindowConfig, ReferenceConfig


CONFIG_DIRECTORY = Path("configs/evidence_window")
TOKEN_IDS = {
    "start": 90,
    "end": 91,
    "pad": 92,
    "padding": 0,
    "special": (0, 90, 91, 92, 99),
}


@dataclass(frozen=True)
class SmokeCase:
    name: str
    config_name: str
    active_losses: tuple[str, ...]
    policy_forwards: tuple[int, int, int]
    reference_forwards: tuple[int, int, int] | None
    ema_forwards: tuple[int, int, int]
    preference_overrides: tuple[tuple[str, float], ...] = ()
    relation_overrides: tuple[tuple[str, float], ...] = ()
    expect_global_diagnostic: bool = False


SMOKE_CASES = (
    SmokeCase("baseline", "baseline.json", (), (1, 0, 0), None, (0, 0, 0)),
    SmokeCase(
        "diagnostic-global",
        "diagnostic_global.json",
        (),
        (1, 1, 0),
        (1, 1, 0),
        (0, 0, 0),
        expect_global_diagnostic=True,
    ),
    SmokeCase(
        "global-preference",
        "diagnostic_global.json",
        ("global_preference",),
        (1, 1, 0),
        (1, 1, 0),
        (0, 0, 0),
        preference_overrides=(("lambda_global", 1.0),),
        expect_global_diagnostic=True,
    ),
    SmokeCase(
        "window-preference",
        "window_preference.json",
        ("window_preference",),
        (1, 1, 0),
        (1, 1, 0),
        (0, 0, 0),
    ),
    SmokeCase(
        "evidence-window-preference",
        "loo_window_preference.json",
        ("ew_dpo",),
        (1, 0, 2),
        (1, 0, 2),
        (0, 0, 0),
    ),
    SmokeCase(
        "relation-jsd",
        "relation_kd.json",
        ("relation_jsd",),
        (1, 0, 2),
        None,
        (1, 0, 2),
        relation_overrides=(("lambda_magnitude", 0.0),),
    ),
    SmokeCase(
        "relation-magnitude",
        "relation_kd.json",
        ("relation_magnitude",),
        (1, 0, 2),
        None,
        (1, 0, 2),
        relation_overrides=(("lambda_jsd", 0.0),),
    ),
    SmokeCase(
        "relation-jsd-magnitude",
        "relation_kd.json",
        ("relation_jsd", "relation_magnitude"),
        (1, 0, 2),
        None,
        (1, 0, 2),
    ),
    SmokeCase(
        "logit-residual",
        "logit_residual.json",
        ("logit_residual",),
        (1, 0, 2),
        None,
        (1, 0, 2),
    ),
    SmokeCase(
        "combined",
        "loo_relation_combined.json",
        ("ew_dpo", "relation_jsd", "relation_magnitude"),
        (1, 0, 2),
        (1, 0, 2),
        (1, 0, 2),
    ),
    SmokeCase(
        "frozen-reference-zero2-config",
        "frozen_reference_zero2.json",
        ("ew_dpo",),
        (1, 1, 2),
        (1, 1, 2),
        (0, 0, 0),
        expect_global_diagnostic=True,
    ),
)


def _load_tiny_smoke_config(case: SmokeCase) -> EvidenceWindowConfig:
    config = EvidenceWindowConfig.from_json(CONFIG_DIRECTORY / case.config_name)
    preference = replace(config.preference, **dict(case.preference_overrides))
    relation = replace(config.relation, **dict(case.relation_overrides))
    reference = config.reference
    if reference.mode == "adapter_disabled":
        # TinyLatentLM is not a PEFT wrapper; the provider contract itself has
        # dedicated adapter-disabled tests. The optimizer smoke uses the other
        # fixed-reference implementation while preserving condition semantics.
        reference = ReferenceConfig(mode="frozen_copy")
    return replace(
        config,
        reference=reference,
        preference=preference,
        relation=relation,
    )


def _forward_counts(model) -> tuple[int, int, int]:
    return (
        model.full_forward_count,
        model.all_null_forward_count,
        model.loo_forward_count,
    )


def _assert_snapshot_equal(snapshot, module) -> None:
    current = dict(module.named_parameters())
    assert snapshot.keys() == current.keys()
    assert all(torch.equal(before, current[name]) for name, before in snapshot.items())


@pytest.mark.parametrize("case", SMOKE_CASES, ids=lambda case: case.name)
def test_one_tiny_optimizer_step_for_each_training_variant(
    case,
    tiny_trainer_factory,
    legacy_latent_case,
    snapshot_parameters,
    monkeypatch,
):
    """A missing loss/condition/update or leaked fixed-role graph must fail the matrix."""
    config = _load_tiny_smoke_config(case)
    trainer, model = tiny_trainer_factory(
        evidence_window_config=config,
        latent_token_ids=TOKEN_IDS,
        ema_tau=0.5,
    )
    captured = []
    if trainer._ew_engine is not None:
        original_compute = trainer._ew_engine.compute

        def capture_compute(*args, **kwargs):
            result = original_compute(*args, **kwargs)
            captured.append(result)
            return result

        monkeypatch.setattr(trainer._ew_engine, "compute", capture_compute)

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

    torch.manual_seed(1234)
    loss = trainer.compute_loss(model, dict(legacy_latent_case["inputs"]))
    assert torch.isfinite(loss)
    if case.name == "baseline":
        assert torch.allclose(
            loss,
            legacy_latent_case["recorded_legacy_loss"],
            atol=1e-6,
            rtol=1e-6,
        )
        assert captured == []
    else:
        assert len(captured) == 1
        assert tuple(captured[0].losses) == case.active_losses
        if case.expect_global_diagnostic:
            assert "preference/global_margin" in captured[0].metric_sums
        if case.active_losses:
            auxiliary_grads = torch.autograd.grad(
                captured[0].total_loss,
                tuple(model.parameters()),
                retain_graph=True,
                allow_unused=True,
            )
            assert any(gradient is not None for gradient in auxiliary_grads)
            assert all(
                gradient is None or torch.isfinite(gradient).all()
                for gradient in auxiliary_grads
            )

    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert all(parameter.grad is None for parameter in trainer._ema.teacher.parameters())
    if trainer._reference is not None:
        assert all(parameter.grad is None for parameter in trainer._reference.parameters())

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
    assert any(not torch.equal(before, ema_after[name]) for name, before in ema_before.items())
    for name, before in ema_before.items():
        expected = 0.5 * before + 0.5 * student_after[name].detach()
        assert torch.allclose(ema_after[name], expected, atol=1e-7, rtol=1e-7)
    assert all(parameter.grad is None for parameter in trainer._ema.teacher.parameters())
    if reference_before is not None:
        _assert_snapshot_equal(reference_before, trainer._reference.model)
        assert all(parameter.grad is None for parameter in trainer._reference.parameters())

    assert _forward_counts(model) == case.policy_forwards
    assert _forward_counts(trainer._ema.teacher) == case.ema_forwards
    if case.reference_forwards is None:
        assert trainer._reference is None
    else:
        assert _forward_counts(trainer._reference.model) == case.reference_forwards
