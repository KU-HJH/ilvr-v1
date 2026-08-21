from __future__ import annotations

import math

import torch

from evidence_window.config import (
    EvidenceWindowConfig,
    LOOConfig,
    PreferenceConfig,
    ReferenceConfig,
    RelationConfig,
)
from evidence_window.engine import (
    Condition,
    EvidenceWindowEngine,
    ModelConditionOutput,
)
from evidence_window.layout import InterleavedLayout
from evidence_window.losses import (
    masked_relation_huber,
    masked_relation_jsd,
    relation_matrix,
)


def _layout(
    evidence_token_mask: torch.Tensor,
    window_token_mask: torch.Tensor,
    causal_pair_mask: torch.Tensor,
    training_pair_mask: torch.Tensor,
) -> InterleavedLayout:
    batch_size, _, _ = evidence_token_mask.shape
    window_count = window_token_mask.shape[1]
    predecessor = torch.full((batch_size, window_count), -1, dtype=torch.long)
    for batch_index, evidence_index, window_index in training_pair_mask.nonzero().tolist():
        predecessor[batch_index, window_index] = evidence_index
    return InterleavedLayout(
        evidence_token_mask=evidence_token_mask,
        window_token_mask=window_token_mask,
        evidence_valid=evidence_token_mask.any(dim=-1),
        window_valid=window_token_mask.any(dim=-1),
        causal_pair_mask=causal_pair_mask,
        training_pair_mask=training_pair_mask,
        native_predecessor=predecessor,
    )


def _two_evidence_layout() -> InterleavedLayout:
    return _layout(
        torch.tensor(
            [[[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]], dtype=torch.bool
        ),
        torch.tensor(
            [[[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1]]], dtype=torch.bool
        ),
        torch.tensor([[[1, 1], [0, 1]]], dtype=torch.bool),
        torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool),
    )


def test_relation_is_length_normalized_and_invalid_pairs_are_zero():
    """A missing length normalization or pair mask must change the literal matrix."""
    full = torch.tensor([[6.0, 9.0]])
    loo = torch.tensor([[[2.0, 5.0], [4.0, 3.0]]])
    lengths = torch.tensor([[2.0, 3.0]])
    pairs = torch.tensor([[[1, 1], [0, 1]]], dtype=torch.bool)

    actual = relation_matrix(full, loo, lengths, pairs)

    assert torch.allclose(actual, torch.tensor([[[2.0, 4.0 / 3.0], [0.0, 2.0]]]))


def test_jsd_excludes_single_candidate_window_and_detaches_teacher():
    """Singleton candidates and EMA targets must not receive structure gradients."""
    teacher = torch.tensor(
        [[[2.0, 1.0], [0.0, 3.0]]],
        requires_grad=True,
    )
    student = torch.tensor(
        [[[1.0, 1.0], [9.0, 2.0]]],
        requires_grad=True,
    )
    pairs = torch.tensor([[[1, 1], [0, 1]]], dtype=torch.bool)

    loss, count = masked_relation_jsd(student, teacher, pairs, 1.0)

    assert count.item() == 1
    loss.backward()
    assert student.grad is not None
    assert student.grad[0, 1, 0].item() == 0.0
    assert teacher.grad is None


def test_bfloat16_jsd_identity_is_exactly_zero_and_within_theoretical_bounds():
    """Low-precision cancellation must not make identical distributions negative."""
    values = torch.tensor(
        [6.15625, -1.171875, -8.6875, 2.28125, -4.34375, -5.59375],
        dtype=torch.bfloat16,
    ).view(1, 6, 1)
    student = values.clone().requires_grad_()
    mask = torch.ones_like(values, dtype=torch.bool)

    loss, count = masked_relation_jsd(student, values, mask, 1.0)

    assert count.item() == 1
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= math.log(2.0)
    assert loss.item() == 0.0


def test_bfloat16_jsd_is_symmetric_and_gradient_descent_moves_toward_teacher():
    """A quantized local perturbation must retain JSD symmetry and useful direction."""
    teacher = torch.tensor(
        [2.0625, -1.015625, -0.203125, 2.71875, -6.15625, -1.53125],
        dtype=torch.bfloat16,
    ).view(1, 6, 1)
    student = teacher.clone()
    student[0, 0, 0] = 2.078125
    student.requires_grad_()
    reverse_student = teacher.clone().requires_grad_()
    mask = torch.ones_like(teacher, dtype=torch.bool)

    loss, _ = masked_relation_jsd(student, teacher, mask, 1.0)
    reverse_loss, _ = masked_relation_jsd(
        reverse_student,
        student.detach(),
        mask,
        1.0,
    )

    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= math.log(2.0)
    assert torch.allclose(loss, reverse_loss, atol=1e-7, rtol=1e-5)
    loss.backward()
    displacement = student.detach().float() - teacher.float()
    gradient_alignment = (displacement * student.grad.float()).sum()
    assert gradient_alignment.item() > 0.0


def test_float16_extreme_finite_logits_at_low_temperature_have_finite_jsd():
    """Temperature scaling must not overflow before stable softmax arithmetic."""
    student = torch.tensor(
        [65000.0, -65000.0, 32000.0, -32000.0, 1.0, -1.0],
        dtype=torch.float16,
    ).view(1, 6, 1).requires_grad_()
    teacher = torch.tensor(
        [-65000.0, 65000.0, -32000.0, 32000.0, -1.0, 1.0],
        dtype=torch.float16,
    ).view(1, 6, 1)
    mask = torch.ones_like(student, dtype=torch.bool)

    loss, count = masked_relation_jsd(student, teacher, mask, 1e-4)

    assert count.item() == 1
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= math.log(2.0)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_huber_retains_negative_relations_and_detaches_teacher():
    """Signed relation magnitude, including negative effects, must be distilled."""
    student = torch.tensor([[[-1.0, 2.0]]], requires_grad=True)
    teacher = torch.tensor([[[1.0, -2.0]]], requires_grad=True)
    mask = torch.ones_like(student, dtype=torch.bool)

    loss, count = masked_relation_huber(student, teacher, mask, 1.0)

    assert count.item() == 2
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_empty_relation_losses_are_finite_graph_connected_zeros():
    """An empty causal-pair batch must remain safe to add to a trainable loss."""
    student = torch.randn((1, 2, 2), requires_grad=True)
    teacher = torch.randn((1, 2, 2), requires_grad=True)
    mask = torch.zeros_like(student, dtype=torch.bool)

    jsd, jsd_count = masked_relation_jsd(student, teacher, mask, 1.0)
    huber, huber_count = masked_relation_huber(student, teacher, mask, 1.0)
    loss = jsd + huber

    assert jsd_count.item() == 0
    assert huber_count.item() == 0
    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))
    assert teacher.grad is None


def test_engine_builds_all_causal_student_teacher_relations_with_cached_conditions():
    """Relation KD must use one policy/EMA LOO forward per span and every causal pair."""
    layout = _two_evidence_layout()
    config = EvidenceWindowConfig(
        relation=RelationConfig(
            lambda_jsd=2.0,
            lambda_magnitude=3.0,
            temperature=1.0,
            huber_delta=1.0,
        )
    )
    policy_full = torch.tensor(
        [[0.0, 6.0, 0.0, 0.0, 9.0, 0.0]],
        requires_grad=True,
    )
    outputs = {
        ("policy", "loo", 0): torch.tensor(
            [[0.0, 2.0, 0.0, 0.0, 5.0, 0.0]], requires_grad=True
        ),
        ("policy", "loo", 1): torch.tensor(
            [[0.0, 4.0, 0.0, 0.0, 3.0, 0.0]], requires_grad=True
        ),
        ("ema", "full", None): torch.tensor(
            [[0.0, 2.0, 0.0, 0.0, 2.0, 0.0]], requires_grad=True
        ),
        ("ema", "loo", 0): torch.tensor(
            [[0.0, 4.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True
        ),
        ("ema", "loo", 1): torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 4.0, 0.0]], requires_grad=True
        ),
    }
    calls: list[tuple[str, str, int | None]] = []

    def forward(role, condition, values, need_hidden):
        del need_hidden
        calls.append((role, condition.kind, condition.evidence_index))
        if role == "ema":
            assert not torch.is_grad_enabled()
        if condition.kind == "loo":
            expected = torch.ones_like(values)
            expected[:, condition.evidence_index] = 0
            assert torch.equal(values, expected)
        return ModelConditionOutput(outputs[calls[-1]])

    result = EvidenceWindowEngine(config).compute(
        layout,
        torch.ones((1, 2, 1)),
        ModelConditionOutput(policy_full),
        forward,
    )

    assert calls.count(("policy", "loo", 0)) == 1
    assert calls.count(("policy", "loo", 1)) == 1
    assert calls.count(("ema", "full", None)) == 1
    assert calls.count(("ema", "loo", 0)) == 1
    assert calls.count(("ema", "loo", 1)) == 1
    assert not any(role == "reference" for role, _, _ in calls)
    assert torch.equal(result.relation_pair_mask, layout.causal_pair_mask)
    assert torch.allclose(
        result.relations["student"],
        torch.tensor([[[2.0, 2.0], [0.0, 3.0]]]),
    )
    assert torch.allclose(
        result.relations["teacher"],
        torch.tensor([[[-1.0, 1.0], [0.0, -1.0]]]),
    )
    assert set(result.losses) == {"relation_jsd", "relation_magnitude"}
    assert torch.allclose(
        result.total_loss,
        2.0 * result.losses["relation_jsd"]
        + 3.0 * result.losses["relation_magnitude"],
    )
    assert {
        "relation/student/top1_ratio",
        "relation/student/normalized_entropy",
        "relation/student/above_threshold_count",
        "relation/teacher/top1_ratio",
        "relation/teacher/normalized_entropy",
        "relation/teacher/above_threshold_count",
        "loo/selected_span_count",
    }.issubset(result.metric_sums)

    result.total_loss.backward()
    assert policy_full.grad is not None and policy_full.grad.abs().sum() > 0
    assert all(
        tensor.grad is not None and tensor.grad.abs().sum() > 0
        for key, tensor in outputs.items()
        if key[0] == "policy"
    )
    assert all(tensor.grad is None for key, tensor in outputs.items() if key[0] == "ema")


def test_relation_structure_and_magnitude_are_independently_configurable():
    """A singleton causal window belongs to magnitude but cannot enter JSD."""
    layout = _layout(
        torch.ones((1, 1, 1), dtype=torch.bool),
        torch.ones((1, 1, 1), dtype=torch.bool),
        torch.ones((1, 1, 1), dtype=torch.bool),
        torch.ones((1, 1, 1), dtype=torch.bool),
    )

    def run(relation: RelationConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        def forward(role, condition, values, need_hidden):
            del values, need_hidden
            if role == "ema" and condition == Condition("full"):
                return ModelConditionOutput(torch.tensor([[3.0]], requires_grad=True))
            if role == "ema":
                return ModelConditionOutput(torch.tensor([[0.0]], requires_grad=True))
            return ModelConditionOutput(torch.tensor([[1.0]], requires_grad=True))

        result = EvidenceWindowEngine(EvidenceWindowConfig(relation=relation)).compute(
            layout,
            torch.ones((1, 1, 1)),
            ModelConditionOutput(torch.tensor([[2.0]], requires_grad=True)),
            forward,
        )
        return result.total_loss, result.losses

    jsd_total, jsd_losses = run(RelationConfig(lambda_jsd=1.0))
    magnitude_total, magnitude_losses = run(RelationConfig(lambda_magnitude=1.0))

    assert set(jsd_losses) == {"relation_jsd"}
    assert jsd_total.item() == 0.0
    assert jsd_total.requires_grad
    assert set(magnitude_losses) == {"relation_magnitude"}
    assert magnitude_total.item() > 0.0


def test_combined_preference_and_relation_reuses_policy_loo_without_reference_adjustment():
    """Shared policy conditions are cached while reference scores affect EW-DPO only."""
    layout = _two_evidence_layout()
    config = EvidenceWindowConfig(
        reference=ReferenceConfig(mode="adapter_disabled"),
        preference=PreferenceConfig(beta=1.0, lambda_ew=1.0),
        relation=RelationConfig(lambda_magnitude=1.0),
    )
    calls: list[tuple[str, str, int | None]] = []

    def forward(role, condition, values, need_hidden):
        del values, need_hidden
        key = (role, condition.kind, condition.evidence_index)
        calls.append(key)
        if role == "policy":
            score = 1.0
        elif role == "reference":
            score = 1000.0 if condition.kind == "full" else -1000.0
        else:
            score = 0.0
        return ModelConditionOutput(torch.full((1, 6), score))

    result = EvidenceWindowEngine(config).compute(
        layout,
        torch.ones((1, 2, 1)),
        ModelConditionOutput(torch.full((1, 6), 2.0, requires_grad=True)),
        forward,
    )

    assert calls.count(("policy", "loo", 0)) == 1
    assert calls.count(("policy", "loo", 1)) == 1
    assert torch.allclose(
        result.relations["student"],
        torch.tensor([[[1.0, 1.0], [0.0, 1.0]]]),
    )


def test_relation_loo_cap_selects_identical_policy_and_ema_spans():
    """The deterministic cap must define one shared selected condition set."""
    identity = torch.eye(3, dtype=torch.bool).unsqueeze(0)
    layout = _layout(
        identity.clone(),
        identity.clone(),
        torch.triu(torch.ones((1, 3, 3), dtype=torch.bool)),
        identity.clone(),
    )
    config = EvidenceWindowConfig(
        relation=RelationConfig(lambda_magnitude=1.0),
        loo=LOOConfig(max_spans_per_example=2),
    )
    calls: list[tuple[str, str, int | None, torch.Tensor]] = []

    def forward(role, condition, values, need_hidden):
        del need_hidden
        calls.append((role, condition.kind, condition.evidence_index, values.clone()))
        return ModelConditionOutput(torch.zeros((1, 3)))

    result = EvidenceWindowEngine(config).compute(
        layout,
        torch.ones((1, 3, 1)),
        ModelConditionOutput(torch.zeros((1, 3), requires_grad=True)),
        forward,
    )

    policy_indices = {
        index
        for role, kind, index, _ in calls
        if role == "policy" and kind == "loo"
    }
    ema_indices = {
        index
        for role, kind, index, _ in calls
        if role == "ema" and kind == "loo"
    }
    assert policy_indices == ema_indices == {0, 2}
    for evidence_index in policy_indices:
        policy_values = next(
            values
            for role, kind, index, values in calls
            if role == "policy" and kind == "loo" and index == evidence_index
        )
        ema_values = next(
            values
            for role, kind, index, values in calls
            if role == "ema" and kind == "loo" and index == evidence_index
        )
        assert torch.equal(policy_values, ema_values)
    assert result.relation_pair_mask[0, 1].sum().item() == 0
