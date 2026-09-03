import math

import pytest
import torch

from step_preference.entropy_loss import entropy_margin_loss, token_entropies


def _entropy(probabilities):
    return -sum(probability * math.log(probability) for probability in probabilities if probability)


def _logits(*probabilities):
    return torch.log(torch.tensor([probabilities], dtype=torch.float32))


def test_token_entropies_selects_next_token_logits_and_promotes_bfloat16_to_float32():
    """Catches entropy using target-position logits or retaining BF16 reductions."""
    logits = _logits(
        [0.5, 0.5],
        [0.8, 0.2],
        [0.9, 0.1],
        [0.3, 0.7],
    ).to(torch.bfloat16)
    target_mask = torch.tensor([[False, True, False, True]])

    actual = token_entropies(logits, target_mask)

    assert actual.dtype == torch.float32
    assert torch.allclose(
        actual,
        torch.tensor([_entropy([0.5, 0.5]), _entropy([0.9, 0.1])]),
        atol=4e-3,
    )


def test_entropy_margin_loss_is_token_weighted_across_steps_and_modalities():
    """Catches averaging step means instead of every selected target token."""
    clean = _logits(
        [0.5, 0.5],
        [0.8, 0.2],
        [0.9, 0.1],
        [0.5, 0.5],
    )
    visual_ablation = _logits(
        [0.5, 0.5],
        [0.8, 0.2],
        [0.9, 0.1],
        [0.5, 0.5],
    )
    textual_ablation = _logits(
        [0.5, 0.5],
        [0.5, 0.5],
        [1.0 - 1e-6, 1e-6],
        [0.5, 0.5],
    )
    visual_mask = torch.tensor([[False, True, False, False]])
    textual_mask = torch.tensor([[False, False, True, True]])

    result = entropy_margin_loss(
        clean_logits=clean,
        visual_ablated_logits=(visual_ablation,),
        visual_target_masks=(visual_mask,),
        textual_ablated_logits=(textual_ablation,),
        textual_target_masks=(textual_mask,),
        entropy_margin=0.1,
    )

    clean_1 = _entropy([0.5, 0.5])
    clean_2 = _entropy([0.8, 0.2])
    clean_3 = _entropy([0.9, 0.1])
    ablated_2 = _entropy([0.5, 0.5])
    ablated_3 = _entropy([1.0 - 1e-6, 1e-6])
    visual_value = 0.1
    text_values = [max(0.0, clean_2 + 0.1 - ablated_2), max(0.0, clean_3 + 0.1 - ablated_3)]

    assert result.visual_token_count == 1
    assert result.textual_token_count == 2
    assert torch.allclose(result.visual_loss, torch.tensor(visual_value), atol=1e-6)
    assert torch.allclose(result.textual_loss, torch.tensor(sum(text_values) / 2), atol=1e-6)
    assert torch.allclose(
        result.loss, torch.tensor((visual_value + sum(text_values)) / 3), atol=1e-6
    )
    assert torch.allclose(result.clean_entropy, torch.tensor((clean_1 + clean_2 + clean_3) / 3), atol=1e-6)


def test_entropy_margin_loss_returns_zero_for_inactive_margin_with_finite_gradients():
    """Catches an inactive hinge producing a negative value or unusable backward pass."""
    clean = _logits([1.0 - 1e-6, 1e-6], [0.5, 0.5], [0.5, 0.5]).requires_grad_()
    ablated = _logits([0.5, 0.5], [0.5, 0.5], [0.5, 0.5]).requires_grad_()
    target_mask = torch.tensor([[False, True, False]])

    result = entropy_margin_loss(
        clean_logits=clean,
        visual_ablated_logits=(ablated,),
        visual_target_masks=(target_mask,),
        textual_ablated_logits=(),
        textual_target_masks=(),
        entropy_margin=0.1,
    )
    result.loss.backward()

    assert result.loss.item() == 0.0
    assert torch.isfinite(clean.grad).all()
    assert torch.isfinite(ablated.grad).all()
    assert result.textual_token_count == 0
    assert result.textual_loss.item() == 0.0


@pytest.mark.parametrize("margin", [-0.1, float("inf"), float("nan")])
def test_entropy_margin_loss_rejects_invalid_margin(margin):
    """Catches a nonfinite or negative entropy threshold changing the objective silently."""
    logits = torch.zeros((1, 3, 2))
    target_mask = torch.tensor([[False, True, False]])
    with pytest.raises(ValueError, match="entropy_margin"):
        entropy_margin_loss(
            clean_logits=logits,
            visual_ablated_logits=(logits,),
            visual_target_masks=(target_mask,),
            textual_ablated_logits=(),
            textual_target_masks=(),
            entropy_margin=margin,
        )


def test_token_entropies_rejects_empty_or_noncausal_target_selection():
    """Catches silently dropping an invalid target mask before entropy aggregation."""
    logits = torch.zeros((1, 3, 2))
    with pytest.raises(ValueError, match="at least one"):
        token_entropies(logits, torch.tensor([[False, False, False]]))
    with pytest.raises(ValueError, match="position 0"):
        token_entropies(logits, torch.tensor([[True, False, True]]))


def test_token_entropies_ignores_nonfinite_logits_outside_the_target_mask():
    """Catches selected entropy being contaminated by unrelated sequence positions."""
    logits = torch.tensor(
        [[
            [math.log(0.25), math.log(0.75)],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
        ]]
    )
    target_mask = torch.tensor([[False, True, False, False]])

    actual = token_entropies(logits, target_mask)

    assert torch.allclose(actual, torch.tensor([_entropy([0.25, 0.75])]), atol=1e-6)


def test_entropy_margin_loss_promotes_bfloat16_logits_and_preserves_finite_gradients():
    """Catches BF16 entropy reduction losing precision or disconnecting either policy branch."""
    clean = _logits([0.5, 0.5], [0.8, 0.2], [0.5, 0.5]).to(torch.bfloat16).requires_grad_()
    ablated = _logits([0.9, 0.1], [0.5, 0.5], [0.5, 0.5]).to(torch.bfloat16).requires_grad_()
    target_mask = torch.tensor([[False, True, False]])

    result = entropy_margin_loss(
        clean_logits=clean,
        visual_ablated_logits=(ablated,),
        visual_target_masks=(target_mask,),
        textual_ablated_logits=(),
        textual_target_masks=(),
        entropy_margin=0.1,
    )
    result.loss.backward()

    assert result.loss.dtype == torch.float32
    assert torch.isfinite(clean.grad).all()
    assert torch.isfinite(ablated.grad).all()


def test_entropy_margin_loss_rejects_invalid_clean_logits_even_with_no_steps():
    """Catches an empty aggregation bypassing the clean-logit input contract."""
    with pytest.raises(ValueError, match="clean_logits"):
        entropy_margin_loss(
            clean_logits=torch.ones((1, 3, 2), dtype=torch.long),
            visual_ablated_logits=(),
            visual_target_masks=(),
            textual_ablated_logits=(),
            textual_target_masks=(),
            entropy_margin=0.1,
        )
