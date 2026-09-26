import importlib
import math

import pytest
import torch


def _objective():
    module = importlib.import_module("stage2.objective")
    required = (
        "entropy_transition",
        "vision_gain",
        "text_entropy",
        "target_log_likelihood",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    assert not missing, f"missing objective callables: {missing}"
    return module


def test_objective_math_api_exists():
    try:
        module = importlib.import_module("stage2.objective")
    except ModuleNotFoundError:
        module = None
    assert module is not None, "stage2.objective must implement the fixed objective math"


def test_entropy_transition_has_hand_checked_value_and_gradient_directions():
    objective = _objective()
    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)

    result = objective.entropy_transition(
        a, b, a.detach(), b.detach(), eps=1e-6, gamma_H=0.0
    )

    assert result["r_pre"].item() == pytest.approx(1.0)
    assert result["r_post"].item() == pytest.approx(1.0)
    assert result["D"].item() == pytest.approx(0.0)
    assert result["loss_dyn"].item() == pytest.approx(-0.5)
    result["loss_dyn"].sum().backward()
    assert a.grad.item() < 0 < b.grad.item()


def test_entropy_transition_detaches_reference_and_is_per_block_nonlinear():
    objective = _objective()
    h_pre = torch.tensor([0.0, 2.0], requires_grad=True)
    h_post = torch.tensor([2.0, 1.0], requires_grad=True)
    ref_pre = torch.ones(2, requires_grad=True)
    ref_post = torch.ones(2, requires_grad=True)

    result = objective.entropy_transition(
        h_pre, h_post, ref_pre, ref_post, eps=1e-6, gamma_H=0.0
    )
    observed = result["loss_dyn"].mean()
    # D is [-2, 1]: mean of the two losses is -0.42513, whereas applying
    # sigmoid to mean(D) gives -0.37754. Both shape and values are required.
    assert result["loss_dyn"].shape == (2,)
    torch.testing.assert_close(result["loss_dyn"], torch.tensor([-0.11920292, -0.73105858]),
                               atol=2e-6, rtol=0)
    assert observed.item() == pytest.approx(-0.42513075, abs=2e-6)
    assert abs(observed.item() - (-0.37754067)) > 0.04
    observed.backward()
    assert ref_pre.grad is None
    assert ref_post.grad is None


def test_vision_gain_has_hand_checked_value_and_detached_m_branch():
    objective = _objective()
    s_f = torch.tensor([0.0], requires_grad=True)
    s_m = torch.tensor([0.0], requires_grad=True)

    scores = objective.vision_gain(s_f, s_m, m_V=0.0)

    assert scores["S_F"] is s_f
    assert scores["G"].item() == pytest.approx(0.0)
    assert scores["loss_vis"].item() == pytest.approx(math.log(2))
    scores["loss_vis"].sum().backward()
    assert s_f.grad.item() == pytest.approx(-0.5)
    assert s_m.grad is None


def test_vision_gain_tokens_applies_softplus_per_token_before_the_mean():
    """Catches averaging token scores inside softplus (score_mean would give log 2)."""
    objective = _objective()
    s_f = torch.tensor([0.0, -2.0], requires_grad=True)
    s_m = torch.tensor([-1.0, -1.0], requires_grad=True)

    scores = objective.vision_gain_tokens(s_f, s_m, m_V=0.0)

    assert scores["loss_vis"].shape == ()
    assert scores["loss_vis"].item() == pytest.approx(0.81326169, abs=1e-6)
    assert scores["S_F"].item() == pytest.approx(-1.0)
    assert scores["S_M"].item() == pytest.approx(-1.0)
    assert scores["G"].item() == pytest.approx(0.0)
    assert not scores["S_M"].requires_grad
    scores["loss_vis"].backward()
    torch.testing.assert_close(
        s_f.grad, torch.tensor([-0.13447071, -0.36552929]), atol=1e-6, rtol=0
    )
    assert s_m.grad is None


def test_vision_gain_tokens_rejects_mismatched_or_empty_token_scores():
    objective = _objective()
    with pytest.raises(ValueError, match="identical shapes"):
        objective.vision_gain_tokens(torch.zeros(2), torch.zeros(3), m_V=0.0)
    with pytest.raises(ValueError, match="at least one token"):
        objective.vision_gain_tokens(torch.zeros(0), torch.zeros(0), m_V=0.0)
    with pytest.raises(ValueError, match="s_m"):
        objective.vision_gain_tokens(
            torch.zeros(2), torch.tensor([0.0, float("nan")]), m_V=0.0
        )


def test_text_entropy_renormalizes_selected_rows_and_ignores_control_logits():
    objective = _objective()
    logits = torch.tensor(
        [[[0.0, 0.0, -7.0], [0.0, 0.0, 100.0]]], requires_grad=True
    )
    original_logits = logits.detach().clone()

    entropy = objective.text_entropy(logits, (0, 1))

    assert entropy.dtype == torch.float32
    torch.testing.assert_close(entropy, torch.full((1, 2), math.log(2)))
    torch.testing.assert_close(logits, original_logits)
    entropy.sum().backward()
    assert torch.equal(logits.grad[..., 2], torch.zeros((1, 2)))


def test_target_log_likelihood_uses_full_vocabulary_and_never_gathers_ignore_index():
    objective = _objective()
    logits = torch.zeros((2, 4), requires_grad=True)

    score = objective.target_log_likelihood(logits, torch.tensor([1, 3]))

    assert score.dtype == torch.float32
    torch.testing.assert_close(score, torch.full((2,), -math.log(4)))
    score.mean().backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="target.*-100"):
        objective.target_log_likelihood(logits.detach(), torch.tensor([1, -100]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_extreme_objective_inputs_keep_losses_and_gradients_finite(dtype):
    objective = _objective()
    logits = torch.tensor(
        [[[-1000.0, 1000.0, 0.0], [1000.0, -1000.0, 0.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    entropy = objective.text_entropy(logits, (0, 1))
    score = objective.target_log_likelihood(logits[:, 0], torch.tensor([1]))
    h_post = entropy[:, 1].clone().requires_grad_(True)
    dyn = objective.entropy_transition(
        entropy[:, 0], h_post, torch.zeros(1), torch.zeros(1), eps=1e-6, gamma_H=0.0
    )
    vis = objective.vision_gain(score, torch.zeros_like(score), m_V=0.0)
    loss = dyn["loss_dyn"].sum() + vis["loss_vis"].sum()

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "call, location",
    [
        (lambda o: o.text_entropy(torch.tensor([[float("nan"), 0.0]]), (0, 1)), "logits"),
        (
            lambda o: o.target_log_likelihood(
                torch.tensor([[0.0, float("inf")]]), torch.tensor([0])
            ),
            "logits",
        ),
        (
            lambda o: o.entropy_transition(
                torch.tensor([float("nan")]),
                torch.tensor([1.0]),
                torch.tensor([1.0]),
                torch.tensor([1.0]),
                eps=1e-6,
                gamma_H=0.0,
            ),
            "h_pre",
        ),
    ],
)
def test_nonfinite_objective_inputs_report_their_location(call, location):
    with pytest.raises(ValueError, match=location):
        call(_objective())
