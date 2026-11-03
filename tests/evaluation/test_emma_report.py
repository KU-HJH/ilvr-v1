from dataclasses import replace
from pathlib import Path

import pytest

from evaluation.report import evaluate
from evaluation.schema import (
    ContractError,
    EmmaTarget,
    EvalConfig,
    ModelInput,
    Prediction,
    PreparedEvaluation,
    Provenance,
    RunMetadata,
    Sample,
    Score,
    VspTarget,
)


def _sample(
    sample_id,
    subject="Coding",
    question_type="multiple_choice",
    category="Grid;Grid;Line",
    task="Choose",
):
    return Sample(
        1,
        sample_id,
        "emma",
        subject,
        ModelInput("Q", (), ("first", "second")),
        EmmaTarget(question_type, "A", 0, "first", category, task),
        Provenance("synthetic:emma", "fixture", subject, "fixture", sample_id, {}),
    )


def _score(sample_id, subject="Coding", correctness=True, status="parsed", error=None):
    return Score(
        sample_id,
        subject,
        "emma",
        "emma_safe_rules_v1",
        "1",
        status,
        "A" if correctness is not None else None,
        "upper_label" if correctness is not None else None,
        correctness,
        None,
        error,
    )


def test_secondary_denominators_and_missing():
    from evaluation.report import emma_summary

    samples = (
        _sample("a"),
        _sample("b"),
        _sample("c", "Physics", "open_ended", "Case Preserved", "   "),
        _sample("d", "Chemistry", category="  ", task="Solve"),
    )
    scores = [
        _score("a"),
        _score("b", correctness=None, status="missing", error="prediction not received"),
        _score("c", "Physics", correctness=False, status="invalid_format", error="unsupported_expression"),
        _score("d", "Chemistry", correctness=False, status="invalid_format", error="empty_extraction"),
    ]

    result = emma_summary(samples, scores)

    assert result["categories"]["Coding"]["Grid"] == dict(
        count=2, observed=1, correct=1, complete=False, accuracy=None
    )
    assert result["categories"]["Coding"]["Line"]["count"] == 2
    assert result["categories"]["Physics"]["Case Preserved"] == dict(
        count=1, observed=1, correct=0, complete=True, accuracy=0.0
    )
    assert "Chemistry" not in result["categories"]
    assert result["tasks"]["Coding_Choose"]["observed"] == 1
    assert "Physics_" not in result["tasks"]
    assert result["question_types"]["multiple_choice"]["accuracy"] is None
    assert result["question_types"]["open_ended"]["count"] == 1
    assert result["question_types"]["open_ended"]["accuracy"] == 0.0
    assert result["comparison_diagnostics"] == {
        "unsupported_expression": 1,
        "non_finite_value": 0,
        "math_limit_exceeded": 0,
    }


def test_all_missing_and_empty_canonical_bucket_are_incomplete():
    from evaluation.report import emma_summary

    samples = (_sample("a"), _sample("b", "Coding", category="Grid", task="Choose"))
    scores = [
        _score("a", correctness=None, status="missing", error="prediction not received"),
        _score("b", correctness=None, status="infrastructure_error", error="worker failed"),
    ]

    result = emma_summary(samples, scores)

    assert result["question_types"]["multiple_choice"] == dict(
        count=2, observed=0, correct=0, complete=False, accuracy=None
    )
    assert result["question_types"]["open_ended"] == dict(
        count=0, observed=0, correct=0, complete=True, accuracy=None
    )
    assert result["comparison_diagnostics"] == {
        "unsupported_expression": 0,
        "non_finite_value": 0,
        "math_limit_exceeded": 0,
    }


@pytest.mark.parametrize(
    "samples,scores",
    [
        ((_sample("a"), _sample("a")), [_score("a"), _score("b")]),
        ((_sample("a"),), [_score("a"), _score("a")]),
        ((_sample("a"),), [_score("b")]),
    ],
)
def test_direct_summary_rejects_nonmatching_or_duplicate_populations(samples, scores):
    from evaluation.report import emma_summary

    with pytest.raises(ContractError):
        emma_summary(samples, scores)


def test_evaluate_adds_emma_details_without_filtering_subjects_or_changing_base_counts():
    samples = (
        _sample("a"),
        _sample("b", "Biology", "open_ended", "Mixed Case", "Inspect"),
    )
    config = EvalConfig(
        1,
        "emma",
        "emma_safe_rules_v1",
        ("Coding", "Biology", "Math"),
        Path("data/evaluation/fixture"),
        None,
        RunMetadata("fixture", "fixture", {}),
    )
    prepared = PreparedEvaluation(samples, {}, "manifest", "run")

    _, summary = evaluate(prepared, config)

    assert summary["N_expected"] == 2
    assert summary["observed"] == 0
    assert summary["accuracy"] is None
    assert summary["categories"]["Coding"]["count"] == 1
    assert summary["categories"]["Coding"]["accuracy"] is None
    assert summary["categories"]["Math"] == dict(
        count=0, observed=0, correct=0, complete=True, accuracy=None
    )
    assert summary["emma"]["categories"]["Biology"]["Mixed Case"]["count"] == 1
    assert summary["emma"]["tasks"]["Biology_Inspect"]["count"] == 1

    vsp_sample = replace(
        samples[0],
        benchmark="vsp",
        subtask="planning",
        target=VspTarget(((1, 2),), 1),
    )
    vsp_config = replace(
        config,
        benchmark="vsp",
        profile="vsp_validated_v1",
        categories=("planning",),
    )
    _, vsp_summary = evaluate(
        PreparedEvaluation((vsp_sample,), {}, "manifest", "run"), vsp_config
    )
    assert "emma" not in vsp_summary


def test_evaluate_keeps_empty_extraction_in_status_counts_not_diagnostics():
    sample = _sample("a", category="Grid", task="Choose")
    config = EvalConfig(
        1,
        "emma",
        "emma_safe_rules_v1",
        ("Coding",),
        Path("data/evaluation/fixture"),
        None,
        RunMetadata("fixture", "fixture", {}),
    )
    prediction = Prediction("a", "manifest", "run", "answer:", "ok")

    scores, summary = evaluate(
        PreparedEvaluation((sample,), {"a": prediction}, "manifest", "run"), config
    )

    assert scores[0].error == "empty_extraction"
    assert summary["status_counts"]["invalid_format"] == 1
    assert summary["emma"]["comparison_diagnostics"] == {
        "unsupported_expression": 0,
        "non_finite_value": 0,
        "math_limit_exceeded": 0,
    }


def test_evaluate_excludes_infrastructure_error_codes_from_comparison_diagnostics():
    samples = tuple(_sample(sample_id) for sample_id in ('a', 'b', 'c'))
    diagnostics = ('unsupported_expression', 'non_finite_value', 'math_limit_exceeded')
    predictions = {
        sample.id: Prediction(sample.id, 'manifest', 'run', None, 'infrastructure_error', error)
        for sample, error in zip(samples, diagnostics)
    }
    config = EvalConfig(1, 'emma', 'emma_safe_rules_v1', ('Coding',),
                        Path('data/evaluation/fixture'), None,
                        RunMetadata('fixture', 'fixture', {}))
    scores, summary = evaluate(PreparedEvaluation(samples, predictions, 'manifest', 'run'), config)
    assert [score.error for score in scores] == list(diagnostics)
    assert summary['status_counts'] == {'infrastructure_error': 3}
    assert (summary['N_expected'], summary['observed'], summary['accuracy']) == (3, 0, None)
    assert summary['emma']['question_types']['multiple_choice'] == dict(
        count=3, observed=0, correct=0, complete=False, accuracy=None)
    assert summary['emma']['comparison_diagnostics'] == {
        'unsupported_expression': 0, 'non_finite_value': 0, 'math_limit_exceeded': 0,
    }
