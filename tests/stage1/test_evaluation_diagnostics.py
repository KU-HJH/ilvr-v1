import math

import pytest
import torch

from stage1.evaluation_diagnostics import (
    compare_forced_prefix,
    cosine_similarity,
    parameter_delta_summary,
    rescore_historical_rows,
    symmetric_relative_l2,
    token_rank,
)


def test_tensor_reducers_report_hand_checked_values_and_reject_nonfinite_inputs():
    """Catches a reducer that hides non-finite evidence or mis-ranks a token."""
    vector = torch.tensor([3.0, 4.0])

    assert symmetric_relative_l2(vector, vector.clone()) == 0.0
    assert cosine_similarity(vector, vector.clone()) == pytest.approx(1.0)
    assert token_rank(torch.tensor([3.0, 1.0, 2.0]), 2) == 2

    with pytest.raises(ValueError, match="finite"):
        symmetric_relative_l2(vector, torch.tensor([math.nan, 4.0]))
    with pytest.raises(ValueError, match="finite"):
        token_rank(torch.tensor([3.0, math.inf, 2.0]), 2)


def test_historical_rescore_joins_unique_indices_and_marks_legacy_limitations():
    """Catches rescoring by row order or pretending old raw-generation evidence exists."""
    rows = [
        {"index": 1, "prediction": "Final answer: 5", "match": False},
        {"index": 0, "prediction": "Final answer: 2", "match": False},
    ]
    dataset = [
        {"original_final_answer": "2"},
        {"original_final_answer": "4"},
    ]

    report = rescore_historical_rows(rows, dataset)

    assert report["source_count"] == 2
    assert report["legacy_raw_id_limitation"] == "unavailable"
    assert report["old_correct"] == 0
    assert report["new_correct"] == 1
    assert report["match_changes"] == [
        {"index": 0, "old_match": False, "new_match": True},
    ]
    assert all(row["evidence_status"] == "legacy_assistant_only" for row in report["rows"])
    assert all(row["generated_suffix_token_ids"] is None for row in report["rows"])
    assert all(row["stop_reason"] is None for row in report["rows"])
    assert all(row["latent_evidence"] is None for row in report["rows"])

    with pytest.raises(ValueError, match="duplicate"):
        rescore_historical_rows(rows + [dict(rows[0])], dataset)
    with pytest.raises(ValueError, match="out of range"):
        rescore_historical_rows([{ "index": 2, "prediction": "2"}], dataset)


def test_forced_prefix_comparison_requires_exact_discrete_evidence_and_thresholds():
    """Catches a recurrence gate that accepts a scaffold mismatch or a loose tensor delta."""
    reference = {
        "sequences": torch.tensor([[1, 91, 90, 92, 3]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "position_ids": torch.arange(5).view(1, 1, 5).expand(3, 1, 5),
        "latent_states": (torch.tensor([[1.0, 0.0]]),),
        "post_end_logits": torch.tensor([[0.0, 2.0]]),
    }
    result = compare_forced_prefix(reference, reference)

    assert result["pass"] is True
    assert result["discrete_match"] is True
    assert result["latent_state_count"] == 1
    assert result["max_symmetric_relative_l2"] == 0.0
    assert result["min_cosine_similarity"] == pytest.approx(1.0)

    changed = dict(reference, sequences=torch.tensor([[1, 7, 90, 92, 3]]))
    mismatch = compare_forced_prefix(reference, changed)
    assert mismatch["pass"] is False
    assert mismatch["discrete_match"] is False

    drifted = dict(reference, latent_states=(torch.tensor([[1.0, 1e-3]]),))
    with pytest.raises(ValueError, match="threshold"):
        compare_forced_prefix(reference, drifted, require_pass=True)


def test_parameter_delta_summary_reports_trainable_tensors_with_finite_distances():
    """Catches parameter evidence that omits a trainable row or accepts NaN deltas."""
    before = {
        "latent.output_delta": torch.tensor([[1.0, 2.0]]),
        "adapter.lora_B": torch.tensor([[0.0]]),
    }
    after = {
        "latent.output_delta": torch.tensor([[1.0, 2.5]]),
        "adapter.lora_B": torch.tensor([[0.1]]),
    }

    summary = parameter_delta_summary(before, after)

    assert summary["parameter_count"] == 2
    assert summary["parameters"]["latent.output_delta"]["changed_elements"] == 1
    assert summary["parameters"]["adapter.lora_B"]["l2_delta"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="finite"):
        parameter_delta_summary(before, {**after, "adapter.lora_B": torch.tensor([[math.nan]])})
