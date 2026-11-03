import copy
import pytest

from evaluation.schema import ContractError, ModelInput
from visulogic_fixtures import make_case, write_rows


# Literal expectations observed in pinned rules, not derived from our extractor.
@pytest.mark.parametrize('response,value,status,method', [
    ('A', 'A', 'parsed', 'box'), ('a', 'Z', 'invalid_format', 'raw'),
    (' A ', 'A', 'parsed', 'box'), (' B ', 'B', 'parsed', 'box'),
    (r'\boxed{B}', 'B', 'parsed', 'box'), (r'\boxed{b}', 'Z', 'invalid_format', 'raw'),
    (r'\boxed{\text{C}}', 'C', 'parsed', 'box'),
    (r'\boxed{A} then \boxed{D}', 'D', 'parsed', 'box'),
    (r'\boxed{A} then \boxed{D', 'A', 'parsed', 'box'),
    (r'\boxed{A or B}', 'Z', 'invalid_format', 'raw'),
    ('The answer is b.', 'B', 'parsed', 'lang'), ('Final answer: C', 'C', 'parsed', 'lang'),
    ('answer: A\nThe correct answer is B.', 'B', 'parsed', 'lang'),
    ('A\nreason B.', 'A', 'parsed', 'lang'), ('(B)', 'Z', 'invalid_format', 'raw'),
    ('foo B.', 'B', 'parsed', 'lang'), (' B.', 'B', 'parsed', 'lang'),
    ('None', 'Z', 'invalid_format', 'raw'), ('', '', 'empty', 'raw'),
    ('   ', '', 'empty', 'raw'), ('<answer>B</answer>', 'Z', 'invalid_format', 'raw'),
    (r'\boxed{{A}}', 'Z', 'invalid_format', 'raw'), ('A<|endoftext|>', 'A', 'parsed', 'lang'),
    ('A B C D', 'Z', 'invalid_format', 'raw'), ('AB', 'Z', 'invalid_format', 'raw'),
    ('foo  b', 'Z', 'invalid_format', 'raw'), ('answer: d', 'D', 'parsed', 'lang'),
])
def test_source_rule_boundaries(response, value, status, method):
    from evaluation.benchmarks.visulogic import extract_answer
    parsed = extract_answer(response, ModelInput('question', (), ('1', '2', '3', '4')))
    assert (parsed.value, parsed.status, parsed.method) == (value, status, method)


def evaluate_case(case):
    from evaluation.io import parse_config, read_inputs
    from evaluation.report import evaluate
    config = parse_config(case['config'], case['root'])
    return evaluate(read_inputs(case['manifest'], case['predictions'], config, case['root']), config)


def test_registered_mcq_path_and_gold_independent_extraction():
    from evaluation.io import parse_sample
    from evaluation.registry import get_scorer
    from evaluation.schema import Prediction
    case = make_case()
    sample = parse_sample(case['rows'][0])
    scorer = get_scorer('visulogic', 'visulogic_vlmeval_rules_v1')
    prediction = Prediction(**case['prediction_rows'][0])
    first = scorer(sample, prediction)
    row = copy.deepcopy(case['rows'][0]); row['target'] = dict(index=0, label='A', content='first')
    second = scorer(parse_sample(row), prediction)
    assert first.correctness is True and second.correctness is False
    assert (first.parsed_answer, first.extraction_method) == (second.parsed_answer, second.extraction_method)
    scores, summary = evaluate_case(case)
    assert len(scores) == 3 and summary['accuracy'] == 1.0
    assert summary['categories']['Style']['count'] == 1
    assert summary['protocol_kind'] == 'modified' and summary['verification']['paper_protocol_verified'] is False


@pytest.mark.parametrize('field,value', [('index', True), ('index', 4), ('label', 'AB'), ('content', 'wrong')])
def test_invalid_target_rejected(field, value):
    from evaluation.io import parse_sample
    case = make_case(1); case['rows'][0]['target'][field] = value
    with pytest.raises(ContractError): parse_sample(case['rows'][0])


@pytest.mark.parametrize('categories,steps', [(['Other'], None), (['Position','Position'], None), ([], None), (['Style'], 1)])
def test_invalid_config_rejected(categories, steps):
    from evaluation.io import parse_config
    case = make_case(); case['config'].update(categories=categories, max_steps=steps)
    with pytest.raises(ContractError): parse_config(case['config'], case['root'])


def test_missing_infrastructure_and_empty_category_keep_contract():
    case = make_case(2)
    case['prediction_rows'][0].update(status='infrastructure_error', response=None, error='offline')
    write_rows(case['predictions'], case['prediction_rows'][:1])
    scores, summary = evaluate_case(case)
    assert [s.status for s in scores] == ['infrastructure_error', 'missing']
    assert summary['N_expected'] == 2 and summary['accuracy'] is None and not summary['complete']
    assert summary['categories']['Style'] == dict(count=0, observed=0, correct=0, complete=True, accuracy=None)


def test_empty_invalid_answers_are_observed_wrong_and_llm_profile_rejected():
    from evaluation.registry import get_scorer
    case = make_case(2)
    for row, response in zip(case['prediction_rows'], ['', 'None']): row['response'] = response
    write_rows(case['predictions'], case['prediction_rows'])
    scores, summary = evaluate_case(case)
    assert [s.status for s in scores] == ['empty', 'invalid_format']
    assert summary['observed'] == 2 and summary['accuracy'] == 0.0 and summary['complete']
    with pytest.raises(ContractError): get_scorer('visulogic', 'visulogic_llm')
