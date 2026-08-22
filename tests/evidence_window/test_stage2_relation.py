from dataclasses import replace

import pytest
import torch

import trainer_stage2 as trainer_stage2_module
from evidence_window.artifacts import ReferenceBatchTargets
from evidence_window.stage2_config import (
    Stage2EvidenceWindowConfig,
    Stage2ReferenceConfig,
    Stage2RelationConfig,
    Stage2ScheduleConfig,
)
from evidence_window.stage2_effects import Stage2ConditionOutput, Stage2EvidenceEffects
from evidence_window.stage2_engine import (
    Stage2EvidenceWindowEngine,
    Stage2EvidenceWindowResult,
)
from evidence_window.stage2_layout import Stage2InterleavedLayout
from trainer_stage2 import CustomTrainerStage2


def _two_evidence_layout() -> Stage2InterleavedLayout:
    return Stage2InterleavedLayout(
        evidence_token_mask=torch.tensor(
            [[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]], dtype=torch.bool
        ),
        window_token_mask=torch.tensor([[[0, 0, 0, 1, 1]]], dtype=torch.bool),
        evidence_valid=torch.tensor([[True, True]]),
        window_valid=torch.tensor([[True]]),
        causal_pair_mask=torch.tensor([[[True], [True]]]),
    )


def _relation_config(**changes) -> Stage2EvidenceWindowConfig:
    base = Stage2EvidenceWindowConfig(
        reference=Stage2ReferenceConfig(mode="precomputed", artifact="/fixed/reference"),
        relation=Stage2RelationConfig(enabled=True, lambda_rel=2.0, temperature=0.7),
    )
    return replace(base, **changes)


def _reference(contributions: torch.Tensor) -> ReferenceBatchTargets:
    return ReferenceBatchTargets(
        sample_indices=(17,),
        contributions=contributions,
        pair_mask=torch.ones_like(contributions, dtype=torch.bool),
    )


def test_softmax_uses_only_valid_evidence_and_always_accumulates_in_fp32():
    """Invalid evidence must get no probability, even when its logit dominates."""
    from evidence_window.losses import stage2_relation_distribution

    contributions = torch.tensor([[[1.0], [2.0], [100.0]]], dtype=torch.bfloat16)
    mask = torch.tensor([[[True], [True], [False]]])

    distribution, valid = stage2_relation_distribution(contributions, mask, 1.0)

    assert distribution.dtype is torch.float32
    assert distribution[0, 2, 0].item() == 0.0
    torch.testing.assert_close(distribution[:, :, 0].sum(1), torch.ones(1))
    assert valid.tolist() == [[True]]


def test_relation_jsd_excludes_empty_windows_and_single_candidate_is_exact_zero():
    """Degenerate evidence sets must remain finite and add no relational target."""
    from evidence_window.losses import stage2_relation_jsd

    student = torch.tensor([[[0.2, 3.0], [9.0, -4.0]]], requires_grad=True)
    teacher = torch.tensor([[[3.0, -2.0], [-8.0, 7.0]]], requires_grad=True)
    mask = torch.tensor([[[True, False], [False, False]]])

    loss = stage2_relation_jsd(student, teacher, mask, 0.7)

    assert loss.dtype is torch.float32
    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad).item() == 0
    assert teacher.grad is None


def test_relation_jsd_is_symmetric_finite_for_variable_candidate_sets_and_stops_teacher():
    """A wrong mask axis or unstopped target would corrupt relation distillation."""
    from evidence_window.losses import stage2_relation_jsd

    student = torch.tensor(
        [
            [[0.1, 4.0], [0.9, -3.0], [50.0, 0.5]],
            [[-0.4, 2.0], [0.2, 1.0], [1.3, -1.0]],
        ],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [
            [[0.8, -2.0], [-0.2, 8.0], [-50.0, 0.1]],
            [[1.0, -2.0], [-0.7, 0.2], [0.4, 4.0]],
        ],
        requires_grad=True,
    )
    mask = torch.tensor(
        [
            [[True, True], [True, False], [False, False]],
            [[True, False], [True, False], [True, False]],
        ]
    )

    loss = stage2_relation_jsd(student, teacher, mask, 0.5)
    reverse = stage2_relation_jsd(teacher.detach(), student.detach(), mask, 0.5)

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss.detach(), reverse)
    loss.backward()
    assert student.grad is not None and torch.count_nonzero(student.grad).item() > 0
    assert teacher.grad is None


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [
        (torch.float16, 65_000.0),
        (torch.bfloat16, 65_000.0),
        (torch.float32, torch.finfo(torch.float32).max * 0.75),
    ],
)
def test_extreme_relation_logits_keep_distribution_loss_and_policy_gradient_finite(
    dtype, magnitude
):
    """Overflow before softmax or probability-space log(0) must not poison gradients."""
    from evidence_window.losses import (
        stage2_relation_distribution,
        stage2_relation_jsd,
    )

    policy = torch.tensor(
        [[[magnitude], [-magnitude], [magnitude]]], dtype=dtype, requires_grad=True
    )
    reference = torch.tensor(
        [[[-magnitude], [magnitude], [-magnitude]]],
        dtype=dtype,
        requires_grad=True,
    )
    mask = torch.tensor([[[True], [True], [False]]])

    distribution, valid = stage2_relation_distribution(policy, mask, 1e-4)
    loss = stage2_relation_jsd(policy, reference, mask, 1e-4)
    reverse = stage2_relation_jsd(reference.detach(), policy.detach(), mask, 1e-4)

    assert distribution.dtype is torch.float32
    assert torch.isfinite(distribution).all()
    assert distribution[0, 2, 0].item() == 0.0
    torch.testing.assert_close(distribution.sum(dim=1), torch.ones((1, 1)))
    assert valid.tolist() == [[True]]
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 0.693148
    torch.testing.assert_close(loss.detach(), reverse)

    loss.backward()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()
    assert reference.grad is None


def test_extreme_single_candidate_keeps_exact_graph_connected_zero():
    """Creating the degenerate zero must not sum huge policy values first."""
    from evidence_window.losses import stage2_relation_jsd

    magnitude = torch.finfo(torch.float32).max * 0.75
    policy = torch.tensor(
        [[[magnitude], [magnitude]]], dtype=torch.float32, requires_grad=True
    )
    reference = torch.tensor([[[-magnitude], [-magnitude]]], dtype=torch.float32)
    mask = torch.tensor([[[True], [False]]])

    loss = stage2_relation_jsd(policy, reference, mask, 1e-4)

    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.grad is not None
    assert torch.equal(policy.grad, torch.zeros_like(policy))


def test_float64_single_candidate_forms_graph_zero_before_fp32_narrowing():
    """A finite FP64 policy value above FP32 range must not become inf times zero."""
    from evidence_window.losses import stage2_relation_jsd

    magnitude = torch.finfo(torch.float64).max * 0.75
    policy = torch.tensor([[[magnitude]]], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([[[-magnitude]]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones_like(policy, dtype=torch.bool)

    loss = stage2_relation_jsd(policy, reference, mask, 1e-4)

    assert loss.dtype is torch.float32
    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.grad is not None
    assert torch.equal(policy.grad, torch.zeros_like(policy))
    assert reference.grad is None


def test_relation_only_engine_runs_policy_effects_uses_fixed_target_and_own_schedule():
    """Relation-only training must not be suppressed by a zero EW coefficient."""
    from evidence_window.losses import stage2_relation_jsd

    config = _relation_config(
        schedule=Stage2ScheduleConfig(
            relation_warmup_ratio=0.2,
            relation_ramp_ratio=0.2,
        )
    )
    engine = Stage2EvidenceWindowEngine(config)
    full = torch.zeros((1, 5), requires_grad=True)
    loo_scales = [
        torch.tensor(-0.1, requires_grad=True),
        torch.tensor(-0.4, requires_grad=True),
    ]
    calls = []

    def policy_forward(block_mask, need_hidden):
        assert need_hidden is False
        calls.append(block_mask)
        if block_mask is None:
            return Stage2ConditionOutput(full, None)
        scale = loo_scales[len(calls) - 2]
        return Stage2ConditionOutput(scale * torch.ones((1, 5)), None)

    reference_values = torch.tensor([[[0.3], [-0.2]]], requires_grad=True)
    targets = _reference(reference_values)

    before = engine.compute(
        _two_evidence_layout(), policy_forward, progress=0.2, reference_targets=targets
    )
    assert before.effects.forward_count == 0
    assert calls == []

    result = engine.compute(
        _two_evidence_layout(), policy_forward, progress=0.3, reference_targets=targets
    )
    expected = stage2_relation_jsd(
        result.effects.contributions,
        reference_values.detach(),
        result.effects.pair_mask,
        0.7,
    )
    torch.testing.assert_close(result.losses["relation"], expected)
    torch.testing.assert_close(result.auxiliary_loss, expected)
    assert len(calls) == 3
    assert "ew/mean_contribution" in result.metric_sums
    assert "loss/relation" in result.metric_sums
    assert "loss/ew_pref" not in result.metric_sums
    assert result.metric_sums["loss/relation"][1].item() == 1.0
    assert result.metric_sums["schedule/relation_weight"][0].item() == pytest.approx(1.0)

    result.auxiliary_loss.backward()
    assert full.grad is not None
    assert all(value.grad is not None for value in loo_scales)
    assert reference_values.grad is None


def test_relation_engine_never_silently_runs_without_fixed_reference_targets():
    """Relation KD is undefined without a fixed target relation."""
    engine = Stage2EvidenceWindowEngine(_relation_config())

    with pytest.raises(ValueError, match="fixed-reference.*requires reference targets"):
        engine.compute(
            _two_evidence_layout(),
            lambda _mask, _hidden: Stage2ConditionOutput(torch.zeros((1, 5)), None),
            progress=1.0,
        )


def test_distributed_objectives_scale_each_global_mean_by_its_own_population(monkeypatch):
    """Relation windows and EW edges must not share the same distributed denominator."""
    calls = []

    monkeypatch.setattr(trainer_stage2_module.distributed, "is_available", lambda: True)
    monkeypatch.setattr(trainer_stage2_module.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_stage2_module.distributed, "get_world_size", lambda: 2)

    def add_remote_population(value, op):
        calls.append(op)
        value.add_(4.0 if len(calls) == 1 else 3.0)

    monkeypatch.setattr(
        trainer_stage2_module.distributed, "all_reduce", add_remote_population
    )
    ew = torch.tensor(2.0, requires_grad=True)
    relation = torch.tensor(5.0, requires_grad=True)
    effects = Stage2EvidenceEffects(
        contributions=torch.zeros((1, 2, 1)),
        pair_mask=torch.ones((1, 2, 1), dtype=torch.bool),
        hidden_residuals=None,
        forward_count=3,
    )
    result = Stage2EvidenceWindowResult(
        auxiliary_loss=ew + relation,
        losses={"ew_pref": ew, "relation": relation},
        metric_sums={
            "loss/ew_pref": (ew.detach() * 2.0, torch.tensor(2.0)),
            "loss/relation": (relation.detach(), torch.tensor(1.0)),
        },
        effects=effects,
        weighted_objectives={"ew_pref": ew, "relation": relation},
    )

    scaled = CustomTrainerStage2._globally_scaled_auxiliary(torch.tensor(0.0), result)

    torch.testing.assert_close(scaled, ew * (4.0 / 6.0) + relation * (2.0 / 4.0))
    assert len(calls) == 2


def test_zero_relation_window_rank_still_joins_relation_population_collective(monkeypatch):
    """A rank with no eligible relation window must not skip the shared reduction."""
    calls = []
    monkeypatch.setattr(trainer_stage2_module.distributed, "is_available", lambda: True)
    monkeypatch.setattr(trainer_stage2_module.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_stage2_module.distributed, "get_world_size", lambda: 2)

    def add_remote_windows(value, op):
        calls.append(op)
        value.add_(2.0)

    monkeypatch.setattr(trainer_stage2_module.distributed, "all_reduce", add_remote_windows)
    policy_scale = torch.tensor(1.0, requires_grad=True)
    policy_zero = policy_scale * 0.0
    effects = Stage2EvidenceEffects(
        contributions=torch.zeros((1, 1, 1)),
        pair_mask=torch.ones((1, 1, 1), dtype=torch.bool),
        hidden_residuals=None,
        forward_count=2,
    )
    result = Stage2EvidenceWindowResult(
        auxiliary_loss=policy_zero,
        losses={"relation": policy_zero},
        metric_sums={"loss/relation": (torch.tensor(0.0), torch.tensor(0.0))},
        effects=effects,
        weighted_objectives={"relation": policy_zero},
    )

    scaled = CustomTrainerStage2._globally_scaled_auxiliary(torch.tensor(0.0), result)

    assert scaled.item() == 0.0
    assert len(calls) == 1
    scaled.backward()
    assert policy_scale.grad is not None and policy_scale.grad.item() == 0.0
