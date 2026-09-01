import math

import pytest

from evaluation.scoring import (
    AnswerParseResult,
    EvaluationScoreSummary,
    extract_final_answer,
    score_answer,
)


@pytest.mark.parametrize(
    ("text", "extracted", "status"),
    [
        ("Final answer: \\boxed{\\frac{1}{2}}<|im_end|>", r"\frac{1}{2}", "parsed"),
        (r"Final answer: \\boxed{\frac{3}{4}}", r"\frac{3}{4}", "parsed"),
        (r"Answer: **{A}**.", "{A}", "parsed"),
        (r"Final answer: \\boxed{\frac{1}{2}", "", "malformed"),
        ("<|latent_start|><|latent_pad|><|latent_end|>", "", "empty_prediction"),
    ],
)
def test_answer_parser_golden(text, extracted, status):
    """Dropping nested braces or hiding malformed boxes must fail this test."""
    result = extract_final_answer(text)
    assert (result.extracted, result.status) == (extracted, status)


def test_last_balanced_box_wins_but_unclosed_final_box_is_malformed():
    """Falling back from a final malformed box must fail this test."""
    assert extract_final_answer(r"\boxed{first} then \\boxed{second}").extracted == "second"
    assert extract_final_answer(r"\boxed{first} then \\boxed{second").status == "malformed"


def test_blank_or_missing_gold_never_matches_blank_prediction():
    """Treating two blank values as equal must fail this test."""
    for gold in (None, "", "   "):
        score = score_answer("", gold)
        assert score.match is False
        assert score.invalid_gold_reason in {"missing", "blank"}


def test_normalization_does_not_apply_symbolic_equivalence():
    """Adding general math equivalence must fail this exact-match test."""
    score = score_answer("1/2", "0.5")
    assert score.match is False
    assert score.prediction.normalized == "1/2"
    assert score.gold.normalized == "0.5"


def test_summary_has_distinct_coverage_and_valid_gold_denominators():
    """Using valid-gold count for coverage accuracy must fail this test."""
    rows = [
        {
            "index": index,
            "match": index < 200,
            "invalid_gold_reason": "blank" if index in {57, 93} else None,
        }
        for index in range(400)
    ]

    summary = EvaluationScoreSummary.from_rows(rows, source_count=400)

    assert summary.correct == 198
    assert summary.source_count == 400
    assert summary.valid_gold_count == 398
    assert summary.invalid_gold_count == 2
    assert summary.invalid_gold == [57, 93]
    assert summary.coverage_accuracy == pytest.approx(198 / 400)
    assert summary.valid_gold_accuracy == pytest.approx(198 / 398)
    assert summary.coverage_accuracy != summary.valid_gold_accuracy


def test_summary_rejects_non_finite_accuracy_values():
    """Accepting NaN or infinity in a score summary must fail this test."""
    with pytest.raises(ValueError):
        EvaluationScoreSummary(
            correct=1,
            source_count=1,
            valid_gold_count=1,
            invalid_gold_count=0,
            invalid_gold=[],
            coverage_accuracy=math.nan,
            valid_gold_accuracy=1.0,
        )
