import pytest
import torch

from step_preference.negatives import StepNegative
from step_preference.packing import (
    build_flex_block_mask,
    build_packed_tree_plan,
    dense_visibility,
)
from step_preference.spans import ReasoningStepSpan


def _negative(step_id: int, *, visual: bool, textual: bool) -> StepNegative:
    return StepNegative(
        sample_id="packed-fixture",
        step_id=step_id,
        chosen_text="chosen",
        rejected_visual_text="visual" if visual else None,
        rejected_textual_text="textual" if textual else None,
        valid_visual_negative=visual,
        valid_textual_negative=textual,
        reference_chosen_score=-0.1,
        reference_visual_rejected_score=-0.2 if visual else None,
        reference_textual_rejected_score=-0.3 if textual else None,
    )


def _rejected(ids: tuple[int, ...], positions: tuple[int, ...]) -> dict:
    return {
        "model_inputs": {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        },
        "completion_token_positions": positions,
        "completion_token_ids": tuple(ids[position] for position in positions),
        "rendered_text": "fixture",
    }


def _two_step_fixture():
    # Keep the clean trajectory longer than all current-step completions and
    # cross a second Flex block once the sidecars are appended.
    clean_ids = torch.arange(10, 130, dtype=torch.long).unsqueeze(0)
    spans = (
        ReasoningStepSpan(0, 0, 2, 2, 3, 4, 6, None),
        ReasoningStepSpan(1, 6, 7, 7, 8, 9, 11, None),
    )
    joined = (
        {
            "span": spans[0],
            "negative": _negative(0, visual=True, textual=True),
            # The common prefix extends through input row 3; selected targets
            # begin at row 4, so the packed branch starts at row 3.
            "visual_rejected": _rejected((10, 11, 12, 13, 90, 91, 99), (4, 5)),
            "textual_rejected": _rejected((10, 11, 12, 13, 80, 81, 99), (4, 5)),
        },
        {
            "span": spans[1],
            "negative": _negative(1, visual=True, textual=False),
            # Common prefix through row 8 is valid; its prediction starts at 8.
            "visual_rejected": _rejected(
                (10, 11, 12, 13, 14, 15, 16, 17, 18, 70, 71, 99), (9, 10)
            ),
            "textual_rejected": None,
        },
    )
    return clean_ids, spans, joined


def test_tree_layout_has_exact_branch_bounds_targets_and_logical_sources():
    """Catches packing target rows or rejected common prefixes at wrong offsets."""
    clean_ids, spans, joined = _two_step_fixture()

    plan = build_packed_tree_plan(
        clean_input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        reasoning_spans=spans,
        joined_steps=joined,
        pad_to_multiple=128,
    )

    assert plan.root_length == 120
    assert plan.unpadded_length == 134
    assert plan.padded_length == 256
    assert plan.root_selected_positions == (3, 4, 8, 9)
    assert [(branch.name, branch.packed_start, branch.packed_end) for branch in plan.branches] == [
        ("step-0-visual-off", 120, 122),
        ("step-0-textual-off", 122, 124),
        ("step-0-visual-rejected", 124, 126),
        ("step-0-textual-rejected", 126, 128),
        ("step-1-visual-off", 128, 130),
        ("step-1-textual-off", 130, 132),
        ("step-1-visual-rejected", 132, 134),
    ]
    visual = plan.branch("step-0-visual-off")
    assert visual.logical_query_positions == (3, 4)
    assert visual.selected_positions == (120, 121)
    assert visual.target_token_ids == (14, 15)
    assert (visual.clean_ancestor_end, visual.blocked_start, visual.blocked_end) == (3, 2, 3)
    rejected = plan.branch("step-1-visual-rejected")
    assert rejected.logical_query_positions == (8, 9)
    assert rejected.selected_positions == (132, 133)
    assert rejected.target_token_ids == (70, 71)
    assert tuple(plan.logical_position_sources[132:134].tolist()) == (8, 9)
    assert tuple(plan.clean_embedding_sources[124:126].tolist()) == (-1, -1)
    assert tuple(plan.token_ids[124:126].tolist()) == (13, 90)
    assert not plan.valid_slots[134:].any()
    assert torch.equal(plan.token_ids[134:], torch.zeros(122, dtype=torch.long))
    assert torch.equal(plan.clean_embedding_sources[134:], torch.full((122,), -1))
    assert torch.equal(plan.logical_position_sources[134:], torch.zeros(122, dtype=torch.long))


def test_tree_visibility_preserves_ancestors_blocks_one_edge_and_isolates_sidecars():
    """Catches reopening an ablation edge or allowing cross-sidecar attention."""
    clean_ids, spans, joined = _two_step_fixture()
    plan = build_packed_tree_plan(
        clean_input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        reasoning_spans=spans,
        joined_steps=joined,
        pad_to_multiple=128,
    )

    visible = dense_visibility(plan)
    visual = plan.branch("step-0-visual-off")
    textual = plan.branch("step-0-textual-off")
    rejected = plan.branch("step-0-visual-rejected")
    assert not visible[visual.selected_positions[0], 2]
    assert visible[visual.selected_positions[0], 1]
    assert visible[visual.selected_positions[1], visual.selected_positions[0]]
    assert not visible[textual.selected_positions[0], 0]
    assert not visible[visual.selected_positions[0], rejected.packed_start]
    assert not visible[rejected.selected_positions[0], visual.packed_start]
    assert visible[3, 2] and not visible[2, 3]
    assert not visible[visual.selected_positions[0], 3]
    assert visible[254, 254] and not visible[254, 253]


@pytest.mark.gpu
def test_flex_block_mask_is_cuda_only_and_skips_at_least_one_block():
    """Catches replacing the packed tree with a dense additive attention mask."""
    clean_ids, spans, joined = _two_step_fixture()
    plan = build_packed_tree_plan(
        clean_input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        reasoning_spans=spans,
        joined_steps=joined,
        pad_to_multiple=128,
    )
    with pytest.raises(ValueError, match="CUDA"):
        build_flex_block_mask(plan, torch.device("cpu"))
    block_mask = build_flex_block_mask(plan, torch.device("cuda"))
    assert block_mask.sparsity() > 0


def test_tree_plan_rejects_noncontiguous_masks_and_pre_latent_rejected_prefixes():
    """Catches packed decoding silently accepting inputs it cannot represent safely."""
    clean_ids, spans, joined = _two_step_fixture()
    noncontiguous_mask = torch.ones_like(clean_ids)
    noncontiguous_mask[0, 2] = 0
    with pytest.raises(ValueError, match="contiguous"):
        build_packed_tree_plan(
            clean_input_ids=clean_ids,
            attention_mask=noncontiguous_mask,
            reasoning_spans=spans,
            joined_steps=joined,
        )
    invalid = list(joined)
    invalid[0] = dict(invalid[0])
    invalid[0]["visual_rejected"] = _rejected((10, 11, 90, 91, 99), (3, 4))
    with pytest.raises(ValueError, match="latent"):
        build_packed_tree_plan(
            clean_input_ids=clean_ids,
            attention_mask=torch.ones_like(clean_ids),
            reasoning_spans=spans,
            joined_steps=tuple(invalid),
        )


def test_tree_plan_requires_128_token_flex_alignment():
    """Catches a caller creating a plan that cannot use Flex's fixed blocks."""
    clean_ids, spans, joined = _two_step_fixture()

    with pytest.raises(ValueError, match="128"):
        build_packed_tree_plan(
            clean_input_ids=clean_ids,
            attention_mask=torch.ones_like(clean_ids),
            reasoning_spans=spans,
            joined_steps=joined,
            pad_to_multiple=1,
        )


def test_tree_plan_accepts_adjacent_steps_that_share_target_and_previous_text():
    """Catches rejecting the canonical text-latent-text-latent-text trajectory."""
    clean_ids, spans, joined = _two_step_fixture()
    shared_step = ReasoningStepSpan(1, 4, 6, 7, 8, 9, 11, None)
    shared_joined = list(joined)
    shared_joined[1] = dict(shared_joined[1], span=shared_step)

    plan = build_packed_tree_plan(
        clean_input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        reasoning_spans=(spans[0], shared_step),
        joined_steps=tuple(shared_joined),
    )

    assert plan.root_selected_positions == (3, 4, 8, 9)


@pytest.mark.parametrize(
    "malformed_step",
    (
        ReasoningStepSpan(1, 5, 7, 7, 8, 9, 11, None),
        ReasoningStepSpan(1, 2, 3, 7, 8, 9, 11, None),
    ),
    ids=("partial-overlap", "reordered"),
)
def test_tree_plan_rejects_partial_or_reordered_consecutive_trajectories(
    malformed_step,
):
    """Catches treating any prior-text overlap as canonical step adjacency."""
    clean_ids, spans, joined = _two_step_fixture()
    malformed_joined = list(joined)
    malformed_joined[1] = dict(malformed_joined[1], span=malformed_step)

    with pytest.raises(ValueError, match="trajectory"):
        build_packed_tree_plan(
            clean_input_ids=clean_ids,
            attention_mask=torch.ones_like(clean_ids),
            reasoning_spans=(spans[0], malformed_step),
            joined_steps=tuple(malformed_joined),
        )
