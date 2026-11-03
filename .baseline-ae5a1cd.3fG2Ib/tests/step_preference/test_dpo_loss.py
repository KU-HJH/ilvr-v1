import math

import pytest
import torch

from step_preference.dpo_loss import (
    aggregate_step_dpo,
    dpo_pair_losses,
    mean_completion_logps,
)


def _dpo(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta):
    margin = (policy_chosen - policy_rejected) - (
        reference_chosen - reference_rejected
    )
    return math.log1p(math.exp(-beta * margin))


def test_mean_completion_logps_uses_only_named_next_token_positions_and_token_mean():
    """Catches scoring a completion start token, wrong causal logits, or summing tokens."""
    logits = torch.log(
        torch.tensor(
            [
                [
                    [0.1, 0.2, 0.7],
                    [0.6, 0.3, 0.1],
                    [0.5, 0.25, 0.25],
                    [0.2, 0.2, 0.6],
                ],
                [
                    [0.25, 0.25, 0.5],
                    [0.7, 0.2, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.4, 0.4, 0.2],
                ],
            ]
        )
    )
    input_ids = torch.tensor([[0, 2, 1, 0], [1, 0, 1, 2]])
    target_mask = torch.tensor([[False, True, True, False], [False, False, True, False]])

    actual = mean_completion_logps(logits, input_ids, target_mask)

    assert actual.dtype == torch.float32
    assert torch.allclose(
        actual,
        torch.tensor([(math.log(0.7) + math.log(0.3)) / 2, math.log(0.2)]),
        atol=1e-6,
    )


def test_dpo_pair_losses_match_direct_scalar_formula_and_detach_reference_inputs():
    """Catches a reversed DPO margin or reference tensors receiving policy gradients."""
    policy_chosen = torch.tensor([0.8, -0.3], requires_grad=True)
    policy_rejected = torch.tensor([0.1, -0.9], requires_grad=True)
    reference_chosen = torch.tensor([0.5, -0.1], requires_grad=True)
    reference_rejected = torch.tensor([0.4, -0.5], requires_grad=True)

    actual = dpo_pair_losses(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.7
    )
    actual.sum().backward()

    expected = torch.tensor(
        [
            _dpo(0.8, 0.1, 0.5, 0.4, 0.7),
            _dpo(-0.3, -0.9, -0.1, -0.5, 0.7),
        ]
    )
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.isfinite(policy_chosen.grad).all()
    assert torch.isfinite(policy_rejected.grad).all()
    assert reference_chosen.grad is None
    assert reference_rejected.grad is None


def test_aggregate_step_dpo_uses_one_global_pair_denominator_and_omits_invalid_pairs():
    """Catches modality-mean averaging or a rejected score from an invalid pair leaking in."""
    clean_ce = torch.tensor(2.0, requires_grad=True)
    policy_chosen = torch.tensor([0.9, -0.4, 0.2], requires_grad=True)
    visual_rejected = torch.tensor([0.2, float("nan"), -0.1], requires_grad=True)
    textual_rejected = torch.tensor([0.0, 3.0, -5.0], requires_grad=True)
    reference_chosen = torch.tensor([0.4, 0.1, -0.2])
    visual_reference_rejected = torch.tensor([0.1, float("nan"), -0.4])
    textual_reference_rejected = torch.tensor([-0.2, 4.0, -6.0])

    result = aggregate_step_dpo(
        clean_ce=clean_ce,
        policy_chosen=policy_chosen,
        policy_visual_rejected=visual_rejected,
        policy_textual_rejected=textual_rejected,
        reference_chosen=reference_chosen,
        reference_visual_rejected=visual_reference_rejected,
        reference_textual_rejected=textual_reference_rejected,
        valid_visual=torch.tensor([True, False, True]),
        valid_textual=torch.tensor([True, False, False]),
        beta=0.5,
    )

    visual_values = [_dpo(0.9, 0.2, 0.4, 0.1, 0.5), _dpo(0.2, -0.1, -0.2, -0.4, 0.5)]
    textual_values = [_dpo(0.9, 0.0, 0.4, -0.2, 0.5)]
    assert result.visual_pair_count == 2
    assert result.textual_pair_count == 1
    assert torch.allclose(result.visual_loss, torch.tensor(sum(visual_values) / 2), atol=1e-6)
    assert torch.allclose(result.textual_loss, torch.tensor(textual_values[0]), atol=1e-6)
    assert torch.allclose(
        result.loss, torch.tensor((sum(visual_values) + textual_values[0]) / 3), atol=1e-6
    )
    result.loss.backward()
    assert torch.isfinite(policy_chosen.grad).all()
    assert torch.isfinite(visual_rejected.grad[[0, 2]]).all()
    assert torch.isfinite(textual_rejected.grad[[0]]).all()


def test_aggregate_step_dpo_returns_differentiable_clean_ce_zero_when_no_pairs():
    """Catches an empty pair set returning a detached CPU scalar instead of CE-shaped zero."""
    clean_ce = torch.tensor(1.25, dtype=torch.bfloat16, requires_grad=True)
    scores = torch.tensor([0.0, 0.0], dtype=torch.bfloat16)

    result = aggregate_step_dpo(
        clean_ce=clean_ce,
        policy_chosen=scores,
        policy_visual_rejected=scores,
        policy_textual_rejected=scores,
        reference_chosen=scores,
        reference_visual_rejected=scores,
        reference_textual_rejected=scores,
        valid_visual=torch.tensor([False, False]),
        valid_textual=torch.tensor([False, False]),
        beta=0.1,
    )
    result.loss.backward()

    assert result.loss.dtype == clean_ce.dtype
    assert result.loss.device == clean_ce.device
    assert result.visual_pair_count == result.textual_pair_count == 0
    assert result.visual_loss.item() == result.textual_loss.item() == 0.0
    assert clean_ce.grad.item() == 0.0


@pytest.mark.parametrize(
    "beta", [0.0, -0.1, float("inf"), float("nan")]
)
def test_dpo_pair_losses_rejects_nonpositive_or_nonfinite_beta(beta):
    """Catches silently accepting an objective with undefined or inverted temperature."""
    values = torch.tensor([0.0])
    with pytest.raises(ValueError, match="beta"):
        dpo_pair_losses(values, values, values, values, beta=beta)


def test_mean_completion_logps_rejects_empty_or_noncausal_target_selection():
    """Catches silently producing a NaN or ignoring target coordinates without prior logits."""
    logits = torch.zeros((1, 3, 2))
    input_ids = torch.tensor([[0, 1, 0]])

    with pytest.raises(ValueError, match="at least one"):
        mean_completion_logps(logits, input_ids, torch.tensor([[False, False, False]]))
    with pytest.raises(ValueError, match="position 0"):
        mean_completion_logps(logits, input_ids, torch.tensor([[True, False, True]]))


def test_mean_completion_logps_ignores_unselected_invalid_ids_and_nonfinite_logits():
    """Catches gathering every sequence ID or multiplying unselected NaNs by zero."""
    logits = torch.tensor(
        [[
            [math.log(0.25), math.log(0.75)],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
        ]]
    )
    input_ids = torch.tensor([[0, 1, 0, 99]])
    target_mask = torch.tensor([[False, True, False, False]])

    actual = mean_completion_logps(logits, input_ids, target_mask)

    assert torch.allclose(actual, torch.tensor([math.log(0.75)]), atol=1e-6)
