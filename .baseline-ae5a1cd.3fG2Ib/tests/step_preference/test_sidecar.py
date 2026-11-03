from dataclasses import FrozenInstanceError

import pytest
import torch

from step_preference.negatives import StepNegative
from step_preference.sidecar import (
    SidecarBranch,
    build_sidecar_plan,
    chunk_sidecar_branches,
    sidecar_dense_visibility,
)
from step_preference.spans import ReasoningStepSpan


def _negative(*, visual: bool = True, textual: bool = True) -> StepNegative:
    return StepNegative(
        sample_id="sidecar-fixture",
        step_id=0,
        chosen_text="chosen",
        rejected_visual_text="visual" if visual else None,
        rejected_textual_text="textual" if textual else None,
        valid_visual_negative=visual,
        valid_textual_negative=textual,
        reference_chosen_score=-0.1,
        reference_visual_rejected_score=-0.2 if visual else None,
        reference_textual_rejected_score=-0.3 if textual else None,
    )


def _rejected(
    ids: tuple[int, ...],
    positions: tuple[int, ...],
    *,
    valid_length: int | None = None,
) -> dict[str, object]:
    valid_length = len(ids) if valid_length is None else valid_length
    attention_mask = torch.zeros((1, len(ids)), dtype=torch.long)
    attention_mask[:, :valid_length] = 1
    return {
        "model_inputs": {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": attention_mask,
        },
        "completion_token_positions": positions,
        "completion_token_ids": tuple(ids[position] for position in positions),
        "rendered_text": "fixture",
    }


def _branch(name: str, length: int) -> SidecarBranch:
    return SidecarBranch(
        name=name,
        kind="visual_off",
        step_id=0,
        token_ids=tuple(range(length)),
        clean_embedding_sources=tuple(range(length)),
        logical_positions=tuple(range(length)),
        clean_ancestor_end=0,
        blocked_start=0,
        blocked_end=0,
        selected_positions=tuple(range(length)),
        target_token_ids=tuple(range(length)),
        rejected_model_inputs=None,
    )


def test_dense_visibility_uses_only_unblocked_root_ancestors_and_local_causality():
    """Catches reopening a blocked root edge or making suffix attention noncausal."""
    expected = torch.zeros((2, 7), dtype=torch.bool)
    expected[0, [0, 1, 5]] = True
    expected[1, [0, 1, 5, 6]] = True
    branch = SidecarBranch(
        name="s0_visual_off",
        kind="visual_off",
        step_id=0,
        token_ids=(92, 12),
        clean_embedding_sources=(3, 4),
        logical_positions=(3, 4),
        clean_ancestor_end=3,
        blocked_start=2,
        blocked_end=3,
        selected_positions=(0, 1),
        target_token_ids=(12, 13),
        rejected_model_inputs=None,
    )

    torch.testing.assert_close(sidecar_dense_visibility(5, (branch,)), expected)


def test_plan_replays_latent_end_when_first_rejected_target_token_differs():
    """Catches sharing the boundary query hidden state for a first-token reject."""
    clean_ids = torch.tensor([[10, 11, 90, 90, 92, 20, 21, 99]])
    span = ReasoningStepSpan(0, 0, 2, 2, 4, 5, 7, None)
    rejected = _rejected((10, 11, 90, 90, 92, 70, 71, 99), (5, 6))
    steps = ({
        "span": span,
        "negative": _negative(visual=True, textual=False),
        "visual_rejected": rejected,
        "textual_rejected": None,
    },)

    plan = build_sidecar_plan(
        clean_input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        spans=(span,),
        steps=steps,
    )

    assert plan.root_length == 8
    assert plan.root_selected_positions == (4, 5)
    assert [[branch.name for branch in group] for group in plan.groups] == [
        ["step-0-visual-off", "step-0-visual-rejected"],
        ["step-0-textual-off"],
    ]
    chosen = plan.groups[0][0]
    assert chosen.token_ids == (92, 20)
    assert chosen.clean_embedding_sources == (4, 5)
    assert chosen.logical_positions == (4, 5)
    assert chosen.selected_positions == (0, 1)
    assert chosen.target_token_ids == (20, 21)
    assert (chosen.clean_ancestor_end, chosen.blocked_start, chosen.blocked_end) == (4, 2, 4)
    rejected_branch = plan.groups[0][1]
    assert rejected_branch.token_ids == (92, 70)
    assert rejected_branch.clean_embedding_sources == (-1, -1)
    assert rejected_branch.logical_positions == (4, 5)
    assert rejected_branch.clean_ancestor_end == 4
    assert rejected_branch.selected_positions == (0, 1)
    assert rejected_branch.target_token_ids == (70, 71)
    assert rejected_branch.rejected_model_inputs is rejected["model_inputs"]
    with pytest.raises(FrozenInstanceError):
        rejected_branch.step_id = 4


def test_rejected_replay_starts_at_lcp_divergence_before_prediction_row():
    """Catches assuming every rejected replay begins one row before its first target."""
    clean_ids = torch.tensor([[10, 11, 90, 90, 92, 20, 21, 22, 99]])
    span = ReasoningStepSpan(0, 0, 2, 2, 4, 5, 8, None)
    rejected = _rejected((10, 11, 90, 90, 93, 20, 70, 71, 99), (6, 7))

    plan = build_sidecar_plan(
        clean_input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        spans=(span,),
        steps=({
            "span": span,
            "negative": _negative(visual=True, textual=False),
            "visual_rejected": rejected,
            "textual_rejected": None,
        },),
    )

    branch = plan.groups[0][1]
    assert branch.token_ids == (93, 20, 70)
    assert branch.logical_positions == (4, 5, 6)
    assert branch.clean_ancestor_end == 4
    assert branch.selected_positions == (1, 2)


def test_rejected_replay_can_extend_beyond_clean_root():
    """Catches retaining the legacy packed-tree root-length restriction."""
    clean_ids = torch.tensor([[10, 11, 90, 90, 92, 20, 21, 99]])
    span = ReasoningStepSpan(0, 0, 2, 2, 4, 5, 7, None)
    rejected = _rejected((10, 11, 90, 90, 92, 20, 21, 99, 70, 71), (8, 9))

    plan = build_sidecar_plan(
        clean_input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        spans=(span,),
        steps=({
            "span": span,
            "negative": _negative(visual=True, textual=False),
            "visual_rejected": rejected,
            "textual_rejected": None,
        },),
    )

    branch = plan.groups[0][1]
    assert branch.token_ids == (99, 70)
    assert branch.logical_positions == (7, 8)
    assert branch.selected_positions == (0, 1)
    assert branch.target_token_ids == (70, 71)


def test_invalid_negatives_are_skipped_but_both_entropy_branches_remain():
    """Catches invalid DPO pairs removing the chosen entropy sidecars."""
    clean_ids = torch.tensor([[10, 11, 90, 90, 92, 20, 21]])
    span = ReasoningStepSpan(0, 0, 2, 2, 4, 5, 7, None)

    plan = build_sidecar_plan(
        clean_input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        spans=(span,),
        steps=({
            "span": span,
            "negative": _negative(visual=False, textual=False),
            "visual_rejected": None,
            "textual_rejected": None,
        },),
    )

    assert tuple(branch.kind for branch in plan.groups[0]) == ("visual_off",)
    assert tuple(branch.kind for branch in plan.groups[1]) == ("textual_off",)


def test_no_step_plan_is_empty_and_ignores_clean_right_padding():
    """Catches importing packed planning's nonempty-step and padded-root behavior."""
    clean_ids = torch.tensor([[10, 11, 12, 999, 999]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

    plan = build_sidecar_plan(
        clean_input_ids=clean_ids,
        attention_mask=attention_mask,
        spans=(),
        steps=(),
    )

    assert plan.root_length == 3
    assert plan.root_selected_positions == ()
    assert plan.groups == ((), ())


def test_rejected_replay_ignores_its_right_padding():
    """Catches padded rejected rows becoming replay queries or score targets."""
    clean_ids = torch.tensor([[10, 11, 90, 90, 92, 20, 21]])
    span = ReasoningStepSpan(0, 0, 2, 2, 4, 5, 7, None)
    rejected = _rejected(
        (10, 11, 90, 90, 92, 70, 71, 999, 999),
        (5, 6),
        valid_length=7,
    )

    plan = build_sidecar_plan(
        clean_input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        spans=(span,),
        steps=({
            "span": span,
            "negative": _negative(visual=True, textual=False),
            "visual_rejected": rejected,
            "textual_rejected": None,
        },),
    )

    assert plan.groups[0][1].token_ids == (92, 70)
    assert 999 not in plan.groups[0][1].token_ids


def test_plan_rejects_empty_chosen_target():
    """Catches constructing a branch with no prediction or target rows."""
    clean_ids = torch.tensor([[10, 11, 90, 90, 92, 20]])
    span = ReasoningStepSpan(0, 0, 2, 2, 4, 5, 5, None)

    with pytest.raises(ValueError, match="nonempty"):
        build_sidecar_plan(
            clean_input_ids=clean_ids,
            attention_mask=torch.ones_like(clean_ids),
            spans=(span,),
            steps=({
                "span": span,
                "negative": _negative(visual=False, textual=False),
                "visual_rejected": None,
                "textual_rejected": None,
            },),
        )


@pytest.mark.parametrize("target_start", [4, 5])
def test_chosen_replay_requires_completed_latent_but_allows_end_token_boundary(target_start):
    """Catches entropy-only replay entering a latent pad or rejecting the legal end token."""
    clean_ids = torch.tensor([[10, 11, 90, 90, 92, 20, 21]])
    span = ReasoningStepSpan(0, 0, 2, 2, 4, target_start, 7, None)
    arguments = dict(clean_input_ids=clean_ids, attention_mask=torch.ones_like(clean_ids),
        spans=(span,), steps=({"span": span, "negative": _negative(visual=False, textual=False),
                              "visual_rejected": None, "textual_rejected": None},))
    if target_start == 4:
        with pytest.raises(ValueError, match="completed latent"):
            build_sidecar_plan(**arguments)
    else:
        plan = build_sidecar_plan(**arguments)
        for group in plan.groups:
            assert len(group) == 1
            assert group[0].logical_positions == (4, 5)
            assert group[0].token_ids == (92, 20)


def test_plan_rejects_empty_rejected_target_and_unfinished_latent_replay():
    """Catches unscoreable rejects and replay beginning inside required latent pads."""
    clean_ids = torch.tensor([[10, 11, 90, 90, 92, 20, 21]])
    span = ReasoningStepSpan(0, 0, 2, 2, 4, 5, 7, None)
    base_step = {
        "span": span,
        "negative": _negative(visual=True, textual=False),
        "textual_rejected": None,
    }

    with pytest.raises(ValueError, match="completion_token_positions"):
        build_sidecar_plan(
            clean_input_ids=clean_ids,
            attention_mask=torch.ones_like(clean_ids),
            spans=(span,),
            steps=(dict(base_step, visual_rejected=_rejected(tuple(clean_ids[0].tolist()), ())),),
        )

    unfinished = _rejected((10, 11, 90, 80, 92, 70, 71), (5, 6))
    with pytest.raises(ValueError, match="latent"):
        build_sidecar_plan(
            clean_input_ids=clean_ids,
            attention_mask=torch.ones_like(clean_ids),
            spans=(span,),
            steps=(dict(base_step, visual_rejected=unfinished),),
        )


def test_plan_rejects_mismatched_rejected_target_ids_and_positions():
    """Catches scoring IDs other than the tokens selected from the own encoding."""
    clean_ids = torch.tensor([[10, 11, 90, 90, 92, 20, 21]])
    span = ReasoningStepSpan(0, 0, 2, 2, 4, 5, 7, None)
    rejected = _rejected((10, 11, 90, 90, 92, 70, 71), (5, 6))
    rejected["completion_token_ids"] = (70, 999)

    with pytest.raises(ValueError, match="match rejected input_ids"):
        build_sidecar_plan(
            clean_input_ids=clean_ids,
            attention_mask=torch.ones_like(clean_ids),
            spans=(span,),
            steps=({
                "span": span,
                "negative": _negative(visual=True, textual=False),
                "visual_rejected": rejected,
                "textual_rejected": None,
            },),
        )


def test_chunking_retains_order_without_splitting_and_isolates_oversized_branch():
    """Catches token loss, reordering, or splitting to satisfy the query budget."""
    first = _branch("first", 1)
    oversized = _branch("oversized", 3)
    last = _branch("last", 1)

    chunks = chunk_sidecar_branches((first, oversized, last), query_budget=2)

    assert chunks == ((first,), (oversized,), (last,))
    assert tuple(branch for chunk in chunks for branch in chunk) == (first, oversized, last)
    with pytest.raises(ValueError, match="positive integer"):
        chunk_sidecar_branches((first,), query_budget=0)
