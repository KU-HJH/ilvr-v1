from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from evidence_window.config import EvidenceWindowConfig, ResidualConfig
from evidence_window.engine import (
    Condition,
    EvidenceWindowEngine,
    ModelConditionOutput,
)
from evidence_window.layout import InterleavedLayout
from evidence_window.losses import (
    chunked_logit_residual_loss,
    select_residual_tokens,
)


def test_chunked_huber_matches_full_projection():
    """Chunk reduction must preserve the full elementwise Huber mean."""
    student = torch.tensor([[1.0, 2.0], [0.0, 1.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    student_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    teacher_weight = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    expected = F.huber_loss(
        student @ student_weight.T,
        teacher @ teacher_weight.T,
        delta=0.75,
    )

    actual = chunked_logit_residual_loss(
        student,
        teacher,
        student_weight,
        teacher_weight,
        "huber",
        2,
        0.75,
    )

    assert torch.allclose(actual, expected)


def test_chunked_cosine_matches_full_projection():
    """Cosine dot products and both norms must accumulate over all chunks."""
    student = torch.tensor([[1.0, 2.0], [2.0, -1.0]], requires_grad=True)
    teacher = torch.tensor([[2.0, 1.0], [-1.0, 3.0]])
    student_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    teacher_weight = torch.tensor([[0.5, 0.0], [0.0, 2.0], [1.0, -1.0]])
    expected = 1.0 - F.cosine_similarity(
        student @ student_weight.T,
        teacher @ teacher_weight.T,
    ).mean()

    actual = chunked_logit_residual_loss(
        student,
        teacher,
        student_weight,
        teacher_weight,
        "cosine",
        2,
        1.0,
    )

    assert torch.allclose(actual, expected)


def test_token_cap_includes_endpoints_and_zero_selects_every_eligible_token():
    """A positive temporal cap includes endpoints; zero means full coverage."""
    mask = torch.tensor([False, True, True, True, True, True])

    assert select_residual_tokens(mask, 3).tolist() == [1, 3, 5]
    assert select_residual_tokens(mask, 0).tolist() == [1, 2, 3, 4, 5]


def test_chunked_projection_never_allocates_the_full_residual_matrix(monkeypatch):
    """Every projection allocation must be bounded by the vocabulary chunk."""
    original_matmul = torch.matmul
    projected_shapes: list[tuple[int, ...]] = []

    def traced_matmul(left, right, *args, **kwargs):
        result = original_matmul(left, right, *args, **kwargs)
        projected_shapes.append(tuple(result.shape))
        return result

    monkeypatch.setattr(torch, "matmul", traced_matmul)
    student = torch.randn((4, 3), requires_grad=True)
    teacher = torch.randn((4, 3))
    student_weight = torch.randn((7, 3), requires_grad=True)
    teacher_weight = torch.randn((7, 3), requires_grad=True)

    chunked_logit_residual_loss(
        student,
        teacher,
        student_weight,
        teacher_weight,
        "huber",
        3,
        1.0,
    )

    assert projected_shapes
    assert all(shape[0] == 4 and shape[1] <= 3 for shape in projected_shapes)
    assert (4, 7) not in projected_shapes


def test_student_hidden_and_head_receive_gradients_but_teacher_is_detached():
    """Residual KD trains both student factors without entering the EMA graph."""
    student = torch.tensor([[1.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[2.0, 1.0]], requires_grad=True)
    student_weight = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True
    )
    teacher_weight = torch.tensor(
        [[2.0, 0.0], [0.0, 1.0], [1.0, -1.0]], requires_grad=True
    )

    loss = chunked_logit_residual_loss(
        student,
        teacher,
        student_weight,
        teacher_weight,
        "huber",
        2,
        1.0,
    )
    loss.backward()

    assert student.grad is not None and student.grad.abs().sum() > 0
    assert student_weight.grad is not None and student_weight.grad.abs().sum() > 0
    assert teacher.grad is None
    assert teacher_weight.grad is None


def test_empty_selected_tokens_return_a_graph_connected_finite_zero():
    """An empty pair batch remains safe in a trainable composite objective."""
    student = torch.empty((0, 2), requires_grad=True)
    teacher = torch.empty((0, 2), requires_grad=True)
    student_weight = torch.randn((3, 2), requires_grad=True)
    teacher_weight = torch.randn((3, 2), requires_grad=True)

    loss = chunked_logit_residual_loss(
        student,
        teacher,
        student_weight,
        teacher_weight,
        "cosine",
        2,
        1.0,
    )

    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert student.grad is not None
    assert torch.equal(student_weight.grad, torch.zeros_like(student_weight))
    assert teacher.grad is None
    assert teacher_weight.grad is None


def test_low_precision_projection_accumulates_safely_without_masking_nonfinite_input():
    """Finite FP16 operands must not overflow before residual comparison."""
    student = torch.tensor([[60000.0, -60000.0]], dtype=torch.float16, requires_grad=True)
    teacher = student.detach().clone()
    weight = torch.tensor(
        [[2.0, 2.0], [2.0, -2.0], [-2.0, 2.0]], dtype=torch.float16
    )

    finite_loss = chunked_logit_residual_loss(
        student,
        teacher,
        weight,
        weight,
        "huber",
        2,
        1.0,
    )
    nonfinite_loss = chunked_logit_residual_loss(
        torch.tensor([[float("inf"), 0.0]]),
        torch.zeros((1, 2)),
        weight.float(),
        weight.float(),
        "huber",
        2,
        1.0,
    )

    assert torch.isfinite(finite_loss)
    assert finite_loss.item() == 0.0
    assert not torch.isfinite(nonfinite_loss)


def test_bfloat16_cosine_identity_stays_within_loss_bounds():
    """Cosine roundoff must not turn identical projected residuals negative."""
    hidden = torch.tensor(
        [[-0.71875, -0.404296875]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.tensor(
        [
            [-0.59765625, 0.181640625],
            [-0.85546875, 1.1015625],
            [-1.0703125, 0.12255859375],
        ],
        dtype=torch.bfloat16,
    )

    loss = chunked_logit_residual_loss(
        hidden,
        hidden.detach(),
        weight,
        weight,
        "cosine",
        2,
        1.0,
    )

    assert 0.0 <= loss.item() <= 2.0
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(hidden.grad).all()


def test_vocabularies_must_align_before_chunk_iteration():
    """Mismatched LM heads must fail instead of silently truncating zip()."""
    with pytest.raises(ValueError, match="vocabulary"):
        chunked_logit_residual_loss(
            torch.ones((1, 2)),
            torch.ones((1, 2)),
            torch.ones((3, 2)),
            torch.ones((2, 2)),
            "huber",
            2,
            1.0,
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


def _two_pair_layout() -> InterleavedLayout:
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


def test_engine_uses_predictor_shift_own_heads_and_cached_adjacent_conditions():
    """Deleting the t-1 shift, own heads, adjacency, or cache must change behavior."""
    config = EvidenceWindowConfig(
        residual=ResidualConfig(
            lambda_logit=2.0,
            metric="huber",
            huber_delta=1.0,
            vocab_chunk_size=1,
            max_tokens_per_window=1,
        )
    )
    policy_full_hidden = torch.tensor(
        [[[2.0, 0.0], [10.0, 10.0], [0.0, 0.0], [0.0, 3.0],
          [20.0, 20.0], [0.0, 0.0]]],
        requires_grad=True,
    )
    policy_loo_0 = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 3.0],
          [20.0, 20.0], [0.0, 0.0]]],
        requires_grad=True,
    )
    policy_loo_1 = torch.tensor(
        [[[2.0, 0.0], [10.0, 10.0], [0.0, 0.0], [0.0, 0.0],
          [0.0, 0.0], [0.0, 0.0]]],
        requires_grad=True,
    )
    ema_full_hidden = torch.tensor(
        [[[0.5, 0.0], [4.0, 4.0], [0.0, 0.0], [0.0, 1.0],
          [7.0, 7.0], [0.0, 0.0]]],
        requires_grad=True,
    )
    ema_loo_0 = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0],
          [7.0, 7.0], [0.0, 0.0]]],
        requires_grad=True,
    )
    ema_loo_1 = torch.tensor(
        [[[0.5, 0.0], [4.0, 4.0], [0.0, 0.0], [0.0, 0.0],
          [0.0, 0.0], [0.0, 0.0]]],
        requires_grad=True,
    )
    hidden_by_condition = {
        ("policy", "loo", 0): policy_loo_0,
        ("policy", "loo", 1): policy_loo_1,
        ("ema", "full", None): ema_full_hidden,
        ("ema", "loo", 0): ema_loo_0,
        ("ema", "loo", 1): ema_loo_1,
    }
    calls: list[tuple[str, str, int | None, bool]] = []

    def forward(role, condition, values, need_hidden):
        del values
        calls.append((role, condition.kind, condition.evidence_index, need_hidden))
        if role == "ema":
            assert not torch.is_grad_enabled()
        hidden = hidden_by_condition[(role, condition.kind, condition.evidence_index)]
        return ModelConditionOutput(torch.zeros((1, 6)), hidden)

    student_weight = torch.eye(2, requires_grad=True)
    teacher_weight = (2.0 * torch.eye(2)).requires_grad_()
    weight_calls: list[str] = []

    def lm_head_weight(role):
        weight_calls.append(role)
        return student_weight if role == "policy" else teacher_weight

    policy_full = ModelConditionOutput(
        torch.zeros((1, 6), requires_grad=True),
        policy_full_hidden,
    )
    result = EvidenceWindowEngine(config).compute(
        _two_pair_layout(),
        torch.ones((1, 2, 1)),
        policy_full,
        forward,
        lm_head_weight=lm_head_weight,
    )

    assert torch.allclose(result.losses["logit_residual"], torch.tensor(0.25))
    assert torch.allclose(result.total_loss, torch.tensor(0.5))
    assert result.metric_sums["residual/selected_token_count"][0].item() == 2
    assert result.metric_sums["residual/selected_token_count"][1].item() == 1
    assert weight_calls == ["policy", "ema"]
    assert calls.count(("policy", "loo", 0, True)) == 1
    assert calls.count(("policy", "loo", 1, True)) == 1
    assert calls.count(("ema", "full", None, True)) == 1
    assert calls.count(("ema", "loo", 0, True)) == 1
    assert calls.count(("ema", "loo", 1, True)) == 1
    assert not any(role == "reference" for role, _, _, _ in calls)

    result.total_loss.backward()
    assert policy_full_hidden.grad is not None
    assert policy_full_hidden.grad[0, [0, 3]].abs().sum() > 0
    assert policy_full_hidden.grad[0, [1, 4]].abs().sum() == 0
    assert policy_loo_0.grad is not None and policy_loo_0.grad.abs().sum() > 0
    assert policy_loo_1.grad is not None and policy_loo_1.grad.abs().sum() > 0
    assert student_weight.grad is not None and student_weight.grad.abs().sum() > 0
    assert ema_full_hidden.grad is None
    assert ema_loo_0.grad is None
    assert ema_loo_1.grad is None
    assert teacher_weight.grad is None


def test_engine_rejects_missing_role_specific_head_provider_when_residual_is_active():
    """Active residual KD must not silently reuse an unspecified model head."""
    layout = _layout(
        torch.ones((1, 1, 2), dtype=torch.bool),
        torch.tensor([[[0, 1]]], dtype=torch.bool),
        torch.ones((1, 1, 1), dtype=torch.bool),
        torch.ones((1, 1, 1), dtype=torch.bool),
    )
    config = EvidenceWindowConfig(residual=ResidualConfig(lambda_logit=1.0))

    def forward(role, condition, values, need_hidden):
        del role, condition, values, need_hidden
        return ModelConditionOutput(
            torch.zeros((1, 2)),
            torch.zeros((1, 2, 2)),
        )

    with pytest.raises(ValueError, match="LM-head weight provider"):
        EvidenceWindowEngine(config).compute(
            layout,
            torch.ones((1, 2, 1)),
            ModelConditionOutput(
                torch.zeros((1, 2), requires_grad=True),
                torch.zeros((1, 2, 2), requires_grad=True),
            ),
            forward,
        )
