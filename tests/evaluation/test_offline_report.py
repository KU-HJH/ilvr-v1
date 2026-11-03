import pytest

from offline_fixtures import make_case, save_manifest, save_rows
from evaluation.io import parse_config, read_inputs
from evaluation.registry import get_scorer
from evaluation.report import evaluate
from evaluation.schema import ContractError


def run(case):
    config = parse_config(case['config'], case['root'])
    prepared = read_inputs(case['manifest'], case['prediction_path'], config, case['root'])
    return evaluate(prepared, config)


def test_missing_predictions_do_not_shrink_denominator():
    case = make_case(2)
    save_rows(case['prediction_path'], case['predictions'][:1])
    scores, summary = run(case)
    assert [s.id for s in scores] == ['vsp/planning/s0', 'vsp/planning/s1']
    assert summary['N_expected'] == 2 and summary['correct'] == 1
    assert summary['coverage'] == 0.5 and summary['accuracy'] is None
    assert summary['complete'] is False
    assert summary['categories']['planning']['accuracy'] is None
    assert summary['categories']['planning-empty'] == dict(count=0, observed=0, correct=0, complete=True, accuracy=None)


def test_empty_prediction_file_is_incomplete():
    case = make_case(2)
    save_rows(case['prediction_path'], [])
    _, summary = run(case)
    assert summary['N_expected'] == 2 and summary['coverage'] == 0
    assert summary['complete'] is False and summary['accuracy'] is None


def test_empty_and_invalid_are_model_failures_but_infrastructure_is_not():
    case = make_case(3)
    rows = case['predictions']
    rows[0]['response'] = ''
    rows[1]['response'] = 'RX'
    rows[2].update(status='infrastructure_error', error='worker failed', response='R')
    save_rows(case['prediction_path'], rows)
    scores, summary = run(case)
    assert [s.correctness for s in scores] == [False, False, None]
    assert [s.status for s in scores] == ['empty', 'invalid_format', 'infrastructure_error']
    assert summary['coverage'] == pytest.approx(2/3)
    assert summary['accuracy'] is None
    rows[2].update(status='ok', error=None)
    save_rows(case['prediction_path'], rows)
    _, summary = run(case)
    assert summary['complete'] is True and summary['accuracy'] == pytest.approx(1/3)


@pytest.mark.parametrize('benchmark,profile', [('unknown','vsp_validated_v1'), ('vsp','unknown'), ('vsp','vsp_ilvr_legacy')])
def test_unknown_registry_values_fail(benchmark, profile):
    with pytest.raises(ContractError):
        get_scorer(benchmark, profile)


def test_complete_category_survives_another_incomplete_category():
    case = make_case(2)
    second = case['samples'][1]
    second['subtask'] = 'planning-empty'
    second['id'] = 'vsp/planning-empty/s1'
    case['predictions'][1]['id'] = second['id']
    save_manifest(case)
    save_rows(case['prediction_path'], case['predictions'][:1])
    _, summary = run(case)
    assert summary['categories']['planning']['accuracy'] == 1.0
    assert summary['categories']['planning-empty']['accuracy'] is None
    assert summary['accuracy'] is None and not summary['complete']


def test_prediction_order_does_not_change_population_order():
    case = make_case(2)
    save_rows(case['prediction_path'], list(reversed(case['predictions'])))
    scores, _ = run(case)
    assert [row.id for row in scores] == [row['id'] for row in case['samples']]
