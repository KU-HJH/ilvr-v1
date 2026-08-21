import pytest
import torch

from evidence_window.interventions import (
    apply_all_null,
    apply_zero_latent_loo,
    select_loo_spans,
)


def test_loo_zeros_only_selected_compact_rows():
    values = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    active = torch.tensor([[0, 1, 1, 0, 1]], dtype=torch.bool)
    evidence = torch.tensor(
        [[[0, 1, 1, 0, 0], [0, 0, 0, 0, 1]]], dtype=torch.bool
    )

    result = apply_zero_latent_loo(values, active, evidence, 0)

    assert result.shape == values.shape
    assert result.tolist() == [[[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]]]
    assert values.tolist() == [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]


def test_all_null_zeros_all_compact_rows():
    assert torch.count_nonzero(apply_all_null(torch.ones(1, 3, 2))) == 0


def test_cap_is_deterministic_and_includes_endpoints():
    valid = torch.tensor([[True, True, True, True, True]])

    assert select_loo_spans(valid, 3) == [[0, 2, 4]]
    assert select_loo_spans(valid, 0) == [[0, 1, 2, 3, 4]]


def test_cap_handles_heterogeneous_rows_and_rounding_collisions():
    valid = torch.tensor(
        [
            [True, False, True, False, True, False],
            [False, True, False, False, False, True],
            [False, False, False, False, False, False],
        ]
    )

    assert select_loo_spans(valid, 4) == [[0, 2, 4], [1, 5], []]


def test_intervention_preserves_metadata_and_unrelated_rows():
    values = torch.tensor([[[1.0], [2.0], [3.0]]])
    active = torch.tensor([[False, True, True, False, True]])
    evidence = torch.tensor(
        [[[False, True, False, False, False], [False, False, True, False, False]]]
    )
    input_ids = torch.tensor([[10, 90, 92, 91, 20]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    original = (input_ids.clone(), attention_mask.clone(), position_ids.clone())
    result = apply_zero_latent_loo(values, active, evidence, 0)

    assert result.tolist() == [[[0.0], [2.0], [3.0]]]
    assert values.tolist() == [[[1.0], [2.0], [3.0]]]
    assert input_ids.equal(original[0])
    assert attention_mask.equal(original[1])
    assert position_ids.equal(original[2])
    assert result.shape == values.shape


def test_loo_requires_exact_compact_row_count():
    values = torch.ones(1, 2, 3)
    active = torch.tensor([[True, False, False]])
    evidence = torch.tensor([[[True, False, False]]])

    with pytest.raises(ValueError, match="compact latent order"):
        apply_zero_latent_loo(values, active, evidence, 0)


def test_loo_rejects_invalid_evidence_index():
    values = torch.ones(1, 1, 2)
    active = torch.tensor([[True, False]])
    evidence = torch.tensor([[[True, False]]])

    with pytest.raises(IndexError):
        apply_zero_latent_loo(values, active, evidence, 1)


def test_cap_rejects_negative_limit():
    with pytest.raises(ValueError, match="non-negative"):
        select_loo_spans(torch.ones(1, 2, dtype=torch.bool), -1)
