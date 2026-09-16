"""Fixed tolerance contract and tiny CPU native A/A consumer coverage."""
import copy
import math

import pytest
import torch

from step_preference.numerical_tolerance import compare_numerical_tolerance


LOSSES = {"loss/ce": 2.0, "loss/context_preference": 0.5}


def compare(reference=None, candidate=None, *, losses=None, candidate_losses=None):
    return compare_numerical_tolerance(
        LOSSES if losses is None else losses,
        LOSSES if candidate_losses is None else candidate_losses,
        {"weight": torch.tensor([3., 4.])} if reference is None else reference,
        {"weight": torch.tensor([3., 4.])} if candidate is None else candidate,
    )


@pytest.mark.parametrize("ce,pref,passed", [
    (2.0200, .5050, True), (2.0202, .5, False), (2., .5052, False),
    (2.03, .47, False),  # Equal total loss cannot hide opposite component errors.
])
def test_loss_components_use_reference_relative_limits_separately(ce, pref, passed):
    result = compare(candidate_losses={"loss/ce": ce, "loss/context_preference": pref})
    assert result.get("passed") is passed
    assert result["losses"]["loss/ce"]["limit"] == pytest.approx(.0201)
    assert result["losses"]["loss/context_preference"]["limit"] == pytest.approx(.0051)
    assert result["losses"]["loss/ce"]["absolute_error"] == pytest.approx(abs(ce - 2.))
    assert result["losses"]["loss/context_preference"]["absolute_error"] == pytest.approx(abs(pref - .5))


@pytest.mark.parametrize("candidate,passed", [(.0001, True), (.00010001, False)])
def test_loss_absolute_boundary_is_inclusive_at_zero_reference(candidate, passed):
    result = compare(losses={**LOSSES, "loss/ce": 0.},
                     candidate_losses={**LOSSES, "loss/ce": candidate})
    assert result.get("passed") is passed


@pytest.mark.parametrize("side", ["reference", "candidate"])
@pytest.mark.parametrize("key", list(LOSSES))
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf")])
def test_missing_or_nonfinite_loss_cannot_pass(side, key, bad):
    values = dict(LOSSES)
    if bad is None:
        values.pop(key)
    else:
        values[key] = bad
    result = compare(**{"losses" if side == "reference" else "candidate_losses": values})
    assert result.get("passed") is False
    assert any(key in error for error in result["errors"])


def test_equal_norm_wrong_gradient_direction_is_rejected():
    result = compare(candidate={"weight": torch.tensor([-3., -4.])})
    assert result.get("passed") is False
    assert result["gradients"]["reference_l2"] == 5.
    assert result["gradients"]["difference_l2"] == 10.
    assert result["gradients"]["relative_l2"] == 2.


@pytest.mark.parametrize("delta,passed", [(.005, True), (.01, True), (.0101, False)])
def test_gradient_one_percent_boundary_uses_full_vector(delta, passed):
    result = compare({"w": torch.tensor([1., 0.], dtype=torch.float64)},
                     {"w": torch.tensor([1., delta], dtype=torch.float64)})
    assert result.get("passed") is passed
    assert result["gradients"]["relative_l2"] == delta


def test_gradient_ratio_weights_all_elements_not_mean_of_layer_ratios():
    result = compare(
        {"large": torch.tensor([100.], dtype=torch.float64), "small": torch.tensor([1.], dtype=torch.float64)},
        {"large": torch.tensor([100.], dtype=torch.float64), "small": torch.tensor([1.1], dtype=torch.float64)},
    )
    assert result.get("passed") is True
    assert result["gradients"]["relative_l2"] == pytest.approx(.1 / math.sqrt(10001))


def test_fp64_accumulation_prevents_half_precision_square_overflow():
    result = compare({"w": torch.tensor([60000.], dtype=torch.float16)},
                     {"w": torch.tensor([60032.], dtype=torch.float16)})
    assert result.get("passed") is True
    assert result["gradients"]["reference_l2"] == 60000.
    assert result["gradients"]["difference_l2"] == 32.
    assert result["gradients"]["relative_l2"] == pytest.approx(32 / 60000)


@pytest.mark.parametrize("reference,candidate,error", [
    ({"a": torch.ones(1)}, {"b": torch.ones(1)}, "keys"),
    ({"a": torch.ones(1)}, {"a": torch.ones(1), "b": torch.ones(1)}, "keys"),
    ({"a": torch.ones(1)}, {"a": torch.ones(1, 1)}, "shape"),
    ({"a": None}, {"a": torch.ones(1)}, "None"),
    ({"a": torch.ones(1)}, {"a": None}, "None"),
    ({}, {}, "active"),
    ({"a": None}, {"a": None}, "active"),
    ({"a": torch.empty(0)}, {"a": torch.empty(0)}, "active"),
])
def test_incomplete_or_incompatible_gradient_mapping_fails(reference, candidate, error):
    result = compare(reference, candidate)
    assert result.get("passed") is False
    assert any(error in item for item in result["errors"])


@pytest.mark.parametrize("side", ["reference", "candidate"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_gradient_cannot_pass(side, bad):
    inputs = {"reference": {"w": torch.ones(1)}, "candidate": {"w": torch.ones(1)}}
    inputs[side]["w"][0] = bad
    result = compare(**inputs)
    assert result.get("passed") is False
    assert any("finite" in item for item in result["errors"])


@pytest.mark.parametrize("value,passed", [(0., True), (1e-12, False), (1e-200, False)])
def test_zero_reference_requires_exact_zero_difference_without_epsilon(value, passed):
    result = compare({"w": torch.zeros(1, dtype=torch.float64)},
                     {"w": torch.tensor([value], dtype=torch.float64)})
    assert result.get("passed") is passed
    assert result["gradients"]["relative_l2"] == (0. if passed else math.inf)


def test_both_none_parameters_are_recorded_without_hiding_active_gradients():
    result = compare({"w": torch.ones(1), "unused": None},
                     {"unused": None, "w": torch.ones(1)})
    assert result.get("passed") is True
    assert result["gradients"]["inactive_names"] == ("unused",)
    assert result["gradients"]["active_tensors"] == 1
    assert result["errors"] == ()


def test_native_cpu_unchanged_loss_only_pair_checks_every_trainable_gradient():
    """Catch a comparator that cannot consume real checkpointed trainer outputs.

    Tiny native CPU A/A only: no production GPU or accuracy-equivalence claim.
    """
    from step_preference.equivalence import _trainer
    from step_preference.spans import ReasoningStepSpan
    from tests.step_preference.insertion_helpers import native_case
    from tests.step_preference.test_v_off_insertion import insertion_config

    with torch.random.fork_rng(devices=[]):
        model, ids, _, grid, _, pixels = native_case()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        candidate = copy.deepcopy(model)
        spans = (ReasoningStepSpan(0, 0, 5, 6, 8, 9, 11, None),
                 ReasoningStepSpan(1, 9, 11, 12, 14, 15, 17, 0))
        batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
                     pixel_values=pixels, image_grid_thw=grid, sample_ids=("tolerance-aa",),
                     reasoning_step_spans=spans,
                     step_preference={"steps": tuple({"span": s} for s in spans)})
        config = insertion_config(lambda_preference=1., branch_mode="independent", target_scope="immediate_text")
        rng = torch.get_rng_state().clone()
        observations = []
        for current in (model, candidate):
            trainer = _trainer(config)
            torch.set_rng_state(rng)
            loss = trainer.compute_loss(current, batch, return_outputs=False)
            loss.backward()
            gradients = {name: None if p.grad is None else p.grad.detach().clone()
                         for name, p in current.named_parameters() if p.requires_grad}
            observations.append((dict(trainer._step_preference_metric_totals), gradients))
        result = compare_numerical_tolerance(observations[0][0], observations[1][0],
                                             observations[0][1], observations[1][1])
        assert result.get("passed") is True, result
        assert result["gradients"]["relative_l2"] == 0.
        assert result["gradients"]["reference_l2"] > 0.
        assert result["gradients"]["active_tensors"] + len(result["gradients"]["inactive_names"]) == len(observations[0][1])
        assert result["losses"]["loss/ce"]["absolute_error"] == 0.
        assert result["losses"]["loss/context_preference"]["absolute_error"] == 0.
