from __future__ import annotations

import math

import pytest

from evaluation.visual_metrics import (
    aggregate_rows,
    matched_group_summary,
    matched_image_groups,
    paired_summary,
)


def _row(
    index: int,
    match: bool,
    *,
    gold: str = "A",
    prediction: str | None = None,
    tokens: list[int] | None = None,
    category: str = "selection",
    error: str | None = None,
    invalid_gold_reason: str | None = None,
) -> dict:
    return {
        "index": index,
        "match": match,
        "invalid_gold_reason": invalid_gold_reason,
        "prediction_normalized": prediction if prediction is not None else (gold if match else "Z"),
        "generated_suffix_token_ids": tokens if tokens is not None else [index, int(match)],
        "latent_block_count": index % 3,
        "inference_time_sec": float(index + 1),
        "category": category,
        "error": error,
        "gold_normalized": gold,
        "gold_parse_status": "invalid_gold" if invalid_gold_reason else "parsed",
        "prediction_parse_status": "inference_error" if error else "parsed",
    }


def test_paired_summary_reports_hand_checked_transition_table() -> None:
    baseline = [_row(i, value) for i, value in enumerate([True, True, False, False])]
    condition = [_row(i, value) for i, value in enumerate([True, False, True, False])]

    result = paired_summary(baseline, condition, bootstrap_samples=200)

    assert result["paired_count"] == 4
    assert result["transitions"] == {
        "correct_to_correct": 1,
        "correct_to_incorrect": 1,
        "incorrect_to_correct": 1,
        "incorrect_to_incorrect": 1,
    }
    assert result["accuracy_delta_pp"] == 0.0
    assert result["mcnemar_pvalue"] == 1.0


def test_identical_correctness_has_zero_width_paired_bootstrap() -> None:
    baseline = [_row(i, value) for i, value in enumerate([True, False, True])]
    condition = [_row(i, value, prediction=f"changed-{i}") for i, value in enumerate([True, False, True])]

    result = paired_summary(baseline, condition, seed=9, bootstrap_samples=101)

    assert result["bootstrap_ci95_pp"] == [0.0, 0.0]
    assert result["prediction_change_count"] == 3
    assert result["prediction_change_rate"] == 1.0


def test_eight_losses_have_exact_binomial_mcnemar_probability() -> None:
    baseline = [_row(i, True) for i in range(8)]
    condition = [_row(i, False) for i in range(8)]

    result = paired_summary(baseline, condition, bootstrap_samples=50)

    assert result["transitions"]["correct_to_incorrect"] == 8
    assert result["transitions"]["incorrect_to_correct"] == 0
    assert result["accuracy_delta_pp"] == -100.0
    assert result["mcnemar_pvalue"] == pytest.approx(2 / 256)


def test_bootstrap_is_deterministic_and_reversing_conditions_negates_interval() -> None:
    baseline = [_row(i, value) for i, value in enumerate([True, True, True, False, False])]
    condition = [_row(i, value) for i, value in enumerate([True, False, False, True, False])]

    forward = paired_summary(baseline, condition, seed=123, bootstrap_samples=999)
    repeated = paired_summary(baseline, condition, seed=123, bootstrap_samples=999)
    reverse = paired_summary(condition, baseline, seed=123, bootstrap_samples=999)

    assert forward["bootstrap_ci95_pp"] == repeated["bootstrap_ci95_pp"]
    assert reverse["accuracy_delta_pp"] == -forward["accuracy_delta_pp"]
    assert reverse["bootstrap_ci95_pp"] == pytest.approx(
        [-forward["bootstrap_ci95_pp"][1], -forward["bootstrap_ci95_pp"][0]]
    )


@pytest.mark.parametrize(
    ("baseline", "condition", "message"),
    [
        ([_row(0, True)], [_row(1, True)], "same index set"),
        ([_row(0, True), _row(0, False)], [_row(0, True)], "duplicate index"),
        ([_row(0, True, gold="A")], [_row(0, True, gold="B")], "gold mismatch"),
    ],
)
def test_paired_summary_rejects_invalid_pair_identity(baseline, condition, message) -> None:
    with pytest.raises(ValueError, match=message):
        paired_summary(baseline, condition, bootstrap_samples=10)


def test_pair_exclusions_keep_errors_and_invalid_gold_out_of_correctness() -> None:
    baseline = [
        _row(0, True),
        _row(1, False, error="oom"),
        _row(2, False, invalid_gold_reason="blank"),
    ]
    condition = [
        _row(0, False),
        _row(1, True),
        _row(2, True, invalid_gold_reason="blank"),
    ]

    result = paired_summary(baseline, condition, bootstrap_samples=20)

    assert result["paired_count"] == 1
    assert result["excluded_counts"] == {"invalid_gold": 1, "inference_error": 1}
    assert result["transitions"]["correct_to_incorrect"] == 1
    assert result["transitions"]["incorrect_to_correct"] == 0


def test_prediction_and_raw_token_changes_are_separate_from_correctness() -> None:
    baseline = [_row(0, True, prediction="A", tokens=[1]), _row(1, False, prediction="B", tokens=[2])]
    condition = [_row(0, True, prediction="A", tokens=[9]), _row(1, False, prediction="C", tokens=[2])]

    result = paired_summary(baseline, condition, bootstrap_samples=20)

    assert result["accuracy_delta_pp"] == 0.0
    assert result["prediction_change_count"] == 1
    assert result["prediction_change_rate"] == 0.5
    assert result["raw_token_change_count"] == 1
    assert result["raw_token_change_rate"] == 0.5


def test_aggregate_rows_separates_coverage_valid_gold_and_accuracy_denominators() -> None:
    rows = [
        _row(0, True),
        _row(1, True, error="failed"),
        _row(2, False, invalid_gold_reason="unparseable"),
        _row(3, False),
    ]

    result = aggregate_rows(rows)

    assert result["num_samples"] == 4
    assert result["coverage_accuracy"] == 0.25
    assert result["valid_gold_accuracy"] == pytest.approx(1 / 3)
    assert result["valid_gold_count"] == 3
    assert result["invalid_gold_count"] == 1
    assert result["error_count"] == 1


def test_matched_groups_require_identical_full_question_and_differing_valid_golds() -> None:
    samples = [
        {"index": 10, "text_input": "Which?  (A) red\n(B) blue", "original_final_answer": "A"},
        {"index": 11, "text_input": " Which? (A) red (B) blue ", "original_final_answer": "B"},
        {"index": 12, "text_input": "Which? (A) red (B) green", "original_final_answer": "A"},
        {"index": 13, "text_input": "Same answer", "original_final_answer": "C"},
        {"index": 14, "text_input": " Same   answer ", "original_final_answer": "C"},
    ]

    assert matched_image_groups(samples) == [[10, 11]]


def test_matched_groups_use_list_positions_as_raw_dataset_indices() -> None:
    samples = [
        {"text_input": "same question", "original_final_answer": "Yes"},
        {"text_input": " same   question ", "original_final_answer": "No"},
    ]

    assert matched_image_groups(samples) == [[0, 1]]


def test_three_member_e5_group_is_one_uncertainty_unit() -> None:
    originals = [
        _row(10, True, gold="A", prediction="A"),
        _row(11, True, gold="B", prediction="B"),
        _row(12, False, gold="A", prediction="Z"),
    ]
    result = matched_group_summary(originals, [[10, 11, 12]])

    assert result["group_count"] == 1
    assert result["member_count"] == 3
    assert result["all_members_correct_groups"] == 0
    assert result["mean_group_member_accuracy"] == pytest.approx(2 / 3)
    assert result["directed_pair_count"] == 4
    assert result["donor_correct_switches"] == 2
    assert result["prediction_change_rate"] == 1.0
    assert result["uncertainty_unit"] == "group"
    assert result["group_member_accuracy_ci95"] == pytest.approx([2 / 3, 2 / 3])


def test_aggregates_are_json_safe_for_missing_or_nonfinite_diagnostics() -> None:
    row = _row(0, True)
    row["inference_time_sec"] = math.nan
    row["latent_block_count"] = None

    result = aggregate_rows([row])

    assert result["mean_inference_time_sec"] is None
    assert result["latent_block_distribution"] == {"unavailable": 1}
