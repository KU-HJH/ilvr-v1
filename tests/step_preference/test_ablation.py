import pytest
import torch

from step_preference.ablation import (
    StepAblationSpec,
    build_additive_step_attention_mask,
)


@pytest.mark.parametrize(
    ("spec", "expected_rows"),
    [
        (
            StepAblationSpec(
                modality="visual",
                target_start=4,
                target_end=6,
                source_start=1,
                source_end=3,
            ),
            (
                (0, 1, 1, 1, 1, 1),
                (0, 0, 1, 1, 1, 1),
                (0, 0, 0, 1, 1, 1),
                (0, 1, 1, 0, 1, 1),
                (0, 1, 1, 0, 0, 1),
                (0, 0, 0, 0, 0, 1),
            ),
        ),
        (
            StepAblationSpec(
                modality="textual",
                target_start=3,
                target_end=5,
                source_start=0,
                source_end=1,
            ),
            (
                (0, 1, 1, 1, 1, 1),
                (0, 0, 1, 1, 1, 1),
                (1, 0, 0, 1, 1, 1),
                (1, 0, 0, 0, 1, 1),
                (0, 0, 0, 0, 0, 1),
                (0, 0, 0, 0, 0, 1),
            ),
        ),
    ],
)
def test_ablation_blocks_shifted_visual_and_textual_rectangles(spec, expected_rows):
    """Catches blocking target rows [a,b) instead of prediction rows [a-1,b-1)."""
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    minimum = torch.finfo(torch.float32).min
    expected = torch.tensor(expected_rows, dtype=torch.float32)
    expected = expected.masked_fill(expected.bool(), minimum).view(1, 1, 6, 6)

    actual = build_additive_step_attention_mask(
        attention_mask,
        spec,
        dtype=torch.float32,
        device=torch.device("cpu"),
        attn_implementation="eager",
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ablation_preserves_causal_and_padding_blocks_inside_rectangle():
    """Catches an overwrite that reopens future edges or a padded source key."""
    attention_mask = torch.tensor([[1, 1, 0, 1]])
    spec = StepAblationSpec(
        modality="visual",
        target_start=2,
        target_end=4,
        source_start=2,
        source_end=4,
    )

    actual = build_additive_step_attention_mask(
        attention_mask,
        spec,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        attn_implementation="sdpa",
    )

    minimum = torch.finfo(torch.bfloat16).min
    blocked = torch.tensor(
        (
            (False, True, True, True),
            (False, False, True, True),
            (False, False, True, True),
            (False, False, True, False),
        )
    )
    expected = torch.zeros((4, 4), dtype=torch.bfloat16).masked_fill(blocked, minimum)
    expected = expected.view(1, 1, 4, 4)

    assert actual.dtype == torch.bfloat16
    assert actual.device == torch.device("cpu")
    assert torch.equal(actual, expected)
    assert torch.all(actual[actual != 0] == minimum)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_start": 0, "target_end": 1, "source_start": 0, "source_end": 1},
        {"target_start": 2, "target_end": 2, "source_start": 0, "source_end": 1},
        {"target_start": 2, "target_end": 3, "source_start": 1, "source_end": 1},
        {"target_start": 2, "target_end": 3, "source_start": -1, "source_end": 1},
    ],
)
def test_ablation_spec_rejects_empty_or_noncausal_intervals(kwargs):
    """Catches malformed public token spans reaching tensor slicing."""
    with pytest.raises(ValueError):
        StepAblationSpec(modality="visual", **kwargs)


@pytest.mark.parametrize(
    "spec",
    [
        StepAblationSpec("visual", 4, 7, 0, 1),
        StepAblationSpec("textual", 2, 3, 0, 7),
    ],
)
def test_ablation_rejects_intervals_outside_sequence(spec):
    """Catches truncated or oversized spans silently producing partial blocks."""
    with pytest.raises(ValueError, match="sequence length"):
        build_additive_step_attention_mask(
            torch.ones(1, 6, dtype=torch.long),
            spec,
            dtype=torch.float32,
            device=torch.device("cpu"),
            attn_implementation="eager",
        )


def test_ablation_rejects_flash_attention_2():
    """Catches silently passing an additive edge mask to an unsupported backend."""
    spec = StepAblationSpec("visual", 2, 3, 0, 1)

    with pytest.raises(ValueError, match="flash_attention_2"):
        build_additive_step_attention_mask(
            torch.ones(1, 3, dtype=torch.long),
            spec,
            dtype=torch.float32,
            device=torch.device("cpu"),
            attn_implementation="flash_attention_2",
        )
