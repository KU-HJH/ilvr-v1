from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from step_preference.replacement import build_replacement_plan
from step_preference.sidecar import SidecarPlan, build_context_sidecar_plan
from step_preference.spans import ReasoningStepSpan
from tests.step_preference.replacement_helpers import (
    planner_case,
    three_step_planner_case,
)


def _build(ids, sidecar, **overrides):
    arguments = {
        "input_ids": ids,
        "sidecar_plan": sidecar,
        "image_token_id": 80,
        "image_count": 1,
        "latent_size": 2,
        "sample_id": "example",
        "seed": 42,
    }
    arguments.update(overrides)
    return build_replacement_plan(**arguments)


def test_first_step_samples_image_and_next_step_uses_only_earlier_donor():
    """Catches self/future donors, wrong absolute sources, or shifted targets."""
    ids, sidecar = planner_case()

    plan = _build(ids, sidecar)

    a, b = plan.branches
    assert plan.root_length == 19
    assert (a.source_kind, a.source_positions, a.prefix_length) == (
        "image", (3, 4), 12,
    )
    assert (a.predictor_positions, a.target_token_ids) == ((10, 11), (21, 22))
    assert (b.source_kind, b.donor_step_id, b.source_positions) == (
        "donor", 0, (8, 9),
    )
    assert (b.predictor_positions, b.target_token_ids) == ((16, 17), (23, 24))
    with pytest.raises(FrozenInstanceError):
        a.step_id = 9


def test_one_step_plan_uses_image_source():
    """Catches requiring a donor before planning the first scored step."""
    ids, sidecar = planner_case()
    sidecar = replace(
        sidecar,
        root_selected_positions=(10, 11),
        groups=((sidecar.groups[0][0],), ()),
    )

    plan = _build(ids, sidecar)

    assert len(plan.branches) == 1
    assert plan.branches[0].source_kind == "image"
    assert plan.branches[0].donor_step_id is None
    assert plan.branches[0].source_positions == (3, 4)


def test_zero_step_ce_plan_is_empty_without_requiring_an_image():
    """Catches rejecting valid CE-only samples because replacement is unused."""
    ids = torch.tensor([[11, 12, 13]])
    sidecar = build_context_sidecar_plan(
        clean_input_ids=ids,
        attention_mask=torch.ones_like(ids),
        spans=(),
        visual_only=True,
    )

    plan = _build(ids, sidecar, image_count=0)

    assert plan.root_length == 3
    assert plan.branches == ()


def test_multiple_earlier_donors_are_selected_by_local_seed():
    """Catches always taking the first donor instead of seeded candidate selection."""
    ids, sidecar = three_step_planner_case()

    plan = _build(ids, sidecar)

    assert plan.branches[2].donor_step_id == 1
    assert plan.branches[2].source_positions == (14, 15)


def test_selection_is_deterministic_despite_changed_global_rng():
    """Catches replacement choices depending on ambient torch RNG state."""
    ids, sidecar = three_step_planner_case()
    torch.manual_seed(1)
    first = _build(ids, sidecar)
    torch.manual_seed(987654)
    second = _build(ids, sidecar)

    assert first == second


def test_planning_does_not_consume_global_rng():
    """Catches sampling through the process-global torch generator."""
    ids, sidecar = planner_case()
    torch.manual_seed(314159)
    expected = torch.rand(4)
    torch.manual_seed(314159)

    _build(ids, sidecar)
    observed = torch.rand(4)

    torch.testing.assert_close(observed, expected)


def test_too_few_image_tokens_samples_with_replacement_and_sorts_positions():
    """Catches rejecting M < L or returning unsorted duplicate image sources."""
    ids, sidecar = planner_case()
    ids = ids.clone()
    ids[0, 3:5] = torch.tensor([79, 79])

    plan = _build(ids, sidecar, sample_id="short", seed=7)

    assert plan.branches[0].source_positions == (2, 2)


@pytest.mark.parametrize(
    ("ids_transform", "image_count", "message"),
    [
        (lambda ids: ids.masked_fill(ids == 80, 79), 1, "image_token_id"),
        (lambda ids: ids, 0, "image_count"),
        (lambda ids: ids, 2, "image_count"),
    ],
)
def test_active_plan_rejects_absent_or_multiple_images(
    ids_transform, image_count, message
):
    """Catches ambiguous or missing same-image fallback metadata."""
    ids, sidecar = planner_case()

    with pytest.raises(ValueError, match=message):
        _build(ids_transform(ids.clone()), sidecar, image_count=image_count)


def test_malformed_earlier_donor_length_is_rejected_instead_of_filtered():
    """Catches silently bypassing malformed earlier latent metadata."""
    ids, sidecar = planner_case()
    first = replace(sidecar.groups[0][0], blocked_end=9)
    malformed = replace(sidecar, groups=((first, sidecar.groups[0][1]), ()))

    with pytest.raises(ValueError, match="latent.*length"):
        _build(ids, malformed)


def test_suppressed_latent_boundaries_are_rejected():
    """Catches accepting a plan whose blocked interval includes boundary tokens."""
    ids, _ = planner_case()
    spans = (
        ReasoningStepSpan(0, 0, 7, 8, 10, 11, 13, None),
        ReasoningStepSpan(1, 11, 13, 14, 16, 17, 19, None),
    )
    suppressed = build_context_sidecar_plan(
        clean_input_ids=ids,
        attention_mask=torch.ones_like(ids),
        spans=spans,
        visual_only=True,
        suppress_latent_boundaries=True,
    )

    with pytest.raises(ValueError, match="suppress"):
        _build(ids, suppressed)


def test_textual_and_rejected_branches_are_rejected():
    """Catches planning from non-visual or rejected sidecar branches."""
    ids, _ = planner_case()
    spans = (
        ReasoningStepSpan(0, 0, 7, 8, 10, 11, 13, None),
        ReasoningStepSpan(1, 11, 13, 14, 16, 17, 19, None),
    )
    textual = build_context_sidecar_plan(
        clean_input_ids=ids,
        attention_mask=torch.ones_like(ids),
        spans=spans,
        visual_only=False,
    )
    _, visual = planner_case()
    rejected_branch = replace(visual.groups[0][0], kind="visual_rejected")
    rejected = replace(
        visual,
        groups=((rejected_branch, visual.groups[0][1]), ()),
    )

    with pytest.raises(ValueError, match="visual-only"):
        _build(ids, textual)
    with pytest.raises(ValueError, match="visual_off"):
        _build(ids, rejected)


@pytest.mark.parametrize(
    ("sample_id", "seed", "message"),
    [
        ("", 42, "sample_id"),
        (None, 42, "sample_id"),
        ("example", True, "seed"),
        ("example", 42.0, "seed"),
        ("example", "42", "seed"),
    ],
)
def test_sample_identity_and_seed_types_are_validated(sample_id, seed, message):
    """Catches unstable serialization inputs entering deterministic seeding."""
    ids, sidecar = planner_case()

    with pytest.raises(ValueError, match=message):
        _build(ids, sidecar, sample_id=sample_id, seed=seed)


def test_input_batch_and_root_bounds_are_validated():
    """Catches planning positions against a different row or truncated root."""
    ids, sidecar = planner_case()
    two_rows = ids.repeat(2, 1)
    oversized_root = replace(sidecar, root_length=20)

    with pytest.raises(ValueError, match="shape"):
        _build(two_rows, sidecar)
    with pytest.raises(ValueError, match="root_length"):
        _build(ids, oversized_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda branch: replace(branch, target_token_ids=(999, 22)), "target_token_ids"),
        (lambda branch: replace(branch, selected_positions=(0, 2)), "predictor"),
        (lambda branch: replace(branch, step_id=3), "step_id"),
        (lambda branch: replace(branch, blocked_start=10, blocked_end=12), "latent.*range"),
        (lambda branch: replace(branch, token_ids=(999, 21)), "token_ids"),
    ],
)
def test_branch_targets_predictors_and_step_order_are_validated(mutation, message):
    """Catches trusting shifted targets or noncausal branch ordering."""
    ids, sidecar = planner_case()
    first = mutation(sidecar.groups[0][0])
    malformed = replace(sidecar, groups=((first, sidecar.groups[0][1]), ()))

    with pytest.raises(ValueError, match=message):
        _build(ids, malformed)


def test_sidecar_must_be_active_context_plan():
    """Catches accepting unrelated or inactive sidecar metadata."""
    ids, sidecar = planner_case()
    inactive = replace(sidecar, context_preference=False)

    with pytest.raises(ValueError, match="context_preference"):
        _build(ids, inactive)
