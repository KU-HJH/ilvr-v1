from __future__ import annotations

import pytest
import torch

from tests.stage1.test_peft import tiny_qwen as tiny_qwen_fixture


def make_tiny_qwen():
    """Reuse Task 7's real local-Qwen fixture without duplicating composition."""
    return tiny_qwen_fixture.__wrapped__()


def test_batched_latent_values_are_scattered_in_row_major_mask_order():
    """Catches per-row reshaping or a row-zero-only latent injection mask."""
    model = make_tiny_qwen().eval()
    input_ids = torch.tensor(
        [
            [1, 126, 2, 3, 4],
            [5, 126, 6, 126, 7],
        ]
    )
    image_out_mask = torch.tensor(
        [
            [False, True, False, False, False],
            [False, True, False, True, False],
        ]
    )
    latent_values = torch.tensor(
        [
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0],
            [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0],
        ]
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        latent_hidden_states=latent_values,
        image_out_mask=image_out_mask,
    )

    torch.testing.assert_close(
        outputs.inputs_embeds[image_out_mask], latent_values
    )


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (torch.zeros((2, 4), dtype=torch.bool), r"image_out_mask must have shape \[B, T\]"),
        (torch.zeros((2, 5), dtype=torch.long), "image_out_mask must be a BoolTensor"),
        ([[False] * 5, [False] * 5], "image_out_mask must be a BoolTensor"),
    ],
)
def test_latent_mask_shape_and_dtype_are_validated_before_scatter(mask, message):
    """Catches broadcast errors and silent integer-to-boolean mask coercion."""
    model = make_tiny_qwen().eval()
    input_ids = torch.tensor([[1, 126, 2, 3, 4], [5, 126, 6, 126, 7]])
    latent_values = torch.ones((3, model.config.hidden_size))

    with pytest.raises((TypeError, ValueError), match=message):
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            latent_hidden_states=latent_values,
            image_out_mask=mask,
        )


@pytest.mark.parametrize(
    "latent_values",
    [
        torch.ones((2, 8)),
        torch.ones((1, 3, 8)),
        torch.ones((3, 7)),
    ],
)
def test_latent_value_rows_rank_and_hidden_width_are_validated(latent_values):
    """Catches malformed packed targets reaching PyTorch masked_scatter errors."""
    model = make_tiny_qwen().eval()
    input_ids = torch.tensor([[1, 126, 2, 3, 4], [5, 126, 6, 126, 7]])
    image_out_mask = torch.tensor(
        [
            [False, True, False, False, False],
            [False, True, False, True, False],
        ]
    )

    with pytest.raises(ValueError, match="latent_hidden_states"):
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            latent_hidden_states=latent_values,
            image_out_mask=image_out_mask,
        )


def test_precomputed_latents_require_an_explicit_mask():
    """Catches a missing teacher mask falling back to every latent token."""
    model = make_tiny_qwen().eval()
    input_ids = torch.tensor([[1, 126, 2, 3]])

    with pytest.raises(ValueError, match="image_out_mask must have shape"):
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            latent_hidden_states=torch.ones((1, model.config.hidden_size)),
        )


def test_latent_values_must_use_a_floating_dtype():
    """Catches integer teacher targets being silently cast at the model boundary."""
    model = make_tiny_qwen().eval()
    input_ids = torch.tensor([[1, 126, 2, 3]])
    mask = torch.tensor([[False, True, False, False]])

    with pytest.raises(TypeError, match="floating-point"):
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            latent_hidden_states=torch.ones(
                (1, model.config.hidden_size), dtype=torch.long
            ),
            image_out_mask=mask,
        )
