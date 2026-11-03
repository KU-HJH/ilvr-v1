from dataclasses import replace
import pytest


def test_extraction_is_gold_independent():
    from evaluation.benchmarks.emma import extract_answer, compare_answer
    from evaluation.schema import EmmaTarget, ModelInput
    model_input = ModelInput('Q', (), ('first', 'second', 'third', 'fourth', 'fifth'))
    gold = EmmaTarget('multiple_choice', 'b', 1, 'second', '', '')
    other = replace(gold, answer='A', index=0, content='first')
    parsed = extract_answer(r'\boxed{B}', model_input)
    assert parsed.value == 'B'
    assert compare_answer(parsed, gold, model_input).correctness is True
    assert compare_answer(parsed, other, model_input).correctness is False
    assert extract_answer(r'\boxed{B}', model_input) == parsed


@pytest.mark.parametrize('response,value', [
    ('The final answer is 0.5', '0.5'), ('The final answer is -2.75.', '-2.75'),
    ('answer: 1.5e-3', '1.5e-3'), (r'\boxed{\frac{1}{2}}', r'\frac{1}{2}'),
    (r'\boxed{3} then \boxed{4', '3'), ('twelve', 'twelve'),
])
def test_decimal_and_boxed_extraction(response, value):
    from evaluation.benchmarks.emma import extract_answer
    from evaluation.schema import ModelInput
    assert extract_answer(response, ModelInput('Q', (), ())).value == value


def test_wrong_valid_label_does_not_fall_back_to_content():
    from evaluation.benchmarks.emma import extract_answer, compare_answer
    from evaluation.schema import EmmaTarget, ModelInput
    model_input = ModelInput('Q', (), ('B', 'other'))
    target = EmmaTarget('multiple_choice', 'A', 0, 'B', '', '')
    assert compare_answer(extract_answer('B', model_input), target, model_input).correctness is False


def test_general_prose_starting_with_choice_letter_remains_raw():
    from evaluation.benchmarks.emma import extract_answer
    from evaluation.schema import ModelInput
    parsed = extract_answer('A reasonable explanation follows', ModelInput('Q', (), ('one', 'two')))
    assert (parsed.value, parsed.status, parsed.method) == (
        'A reasonable explanation follows', 'parsed', 'raw')


@pytest.mark.parametrize('response,value,status,method,error', [
    ('b', 'b', 'parsed', 'lower_label', None), ('C: because', 'C', 'parsed', 'upper_label', None),
    ('D. because', 'D', 'parsed', 'upper_label', None),
    (r'\boxed{\text{x}}', 'x', 'parsed', 'boxed', None),
    ('correct answer is X.\nignored', 'X', 'parsed', 'marker', None),
    ('answer:', '', 'invalid_format', 'marker', 'empty_extraction'),
    ('NaN', 'NaN', 'invalid_format', 'raw', 'non_finite_value'),
    ('', '', 'empty', 'raw', None),
])
def test_extraction_contract(response, value, status, method, error):
    from evaluation.benchmarks.emma import extract_answer
    from evaluation.schema import ModelInput
    parsed = extract_answer(response, ModelInput('Q', (), ('a','b','c','d')))
    assert (parsed.value, parsed.status, parsed.method, parsed.error) == (value,status,method,error)


def test_registered_scorer_returns_existing_score_shape():
    from evaluation.io import parse_sample
    from evaluation.registry import get_scorer
    from evaluation.schema import Prediction
    from emma_fixtures import make_case
    case = make_case(1)
    score = get_scorer('emma', 'emma_safe_rules_v1')(
        parse_sample(case['rows'][0]), Prediction(**case['prediction_rows'][0]))
    assert (score.correctness, score.parsed_answer, score.profile, score.scorer_version) == (True, 'B', 'emma_safe_rules_v1', '1')


@pytest.mark.parametrize('response,answer,correctness,status,error', [
    ('$0.5$', '0.5', True, 'parsed', None),
    (r'\(0.5\)', '0.5', True, 'parsed', None),
    ('$$0.5$$', '0.5', False, 'parsed', 'unsupported_expression'),
    ('0.5', '$$0.5$$', False, 'parsed', 'unsupported_expression'),
    ('$NaN$', '0', False, 'invalid_format', 'non_finite_value'),
    (r'\(-inf\)', '0', False, 'invalid_format', 'non_finite_value'),
    ('$ $', '0', False, 'invalid_format', 'empty_extraction'),
])
def test_score_preserves_math_wrappers_and_normalizes_one_layer(response, answer, correctness, status, error):
    from evaluation.benchmarks.emma import score_emma
    from evaluation.schema import EmmaTarget, ModelInput, Prediction, Provenance, Sample
    sample = Sample(1, 'math_0', 'emma', 'Math', ModelInput('Q', (), ()),
                    EmmaTarget('open_ended', answer, None, answer, '', ''),
                    Provenance('synthetic:emma', 'fixture', 'Math', 'test', 'math_0', {}))
    score = score_emma(sample, Prediction('math_0', 'manifest', 'run', response, 'ok'))
    assert (score.correctness, score.status, score.error) == (correctness, status, error)
    assert score.parsed_answer == response


@pytest.mark.parametrize('response', ['$B$', r'\(B\)'])
def test_wrapped_wrong_label_remains_decisive_at_score_boundary(response):
    from evaluation.benchmarks.emma import score_emma
    from evaluation.schema import EmmaTarget, ModelInput, Prediction, Provenance, Sample
    sample = Sample(1, 'math_0', 'emma', 'Math', ModelInput('Q', (), ('B', 'other')),
                    EmmaTarget('multiple_choice', 'A', 0, 'B', '', ''),
                    Provenance('synthetic:emma', 'fixture', 'Math', 'test', 'math_0', {}))
    score = score_emma(sample, Prediction('math_0', 'manifest', 'run', response, 'ok'))
    assert score.correctness is False
    assert score.error is None
    assert score.parsed_answer == response


@pytest.mark.parametrize('response,answer', [
    (r'\boxed{\text{a}+\text{b}}', r'\text{a}+\text{b}'),
    (r'\boxed{\text{a{b}c}}', 'a{b}c'),
    (r'\boxed{\text{\text{a}}}', r'\text{a}'),
])
def test_box_text_wrapper_must_enclose_whole_candidate(response, answer):
    from evaluation.benchmarks.emma import score_emma
    from evaluation.schema import EmmaTarget, ModelInput, Prediction, Provenance, Sample
    sample = Sample(1, 'math_0', 'emma', 'Math', ModelInput('Q', (), ()),
                    EmmaTarget('open_ended', answer, None, answer, '', ''),
                    Provenance('synthetic:emma', 'fixture', 'Math', 'test', 'math_0', {}))
    score = score_emma(sample, Prediction('math_0', 'manifest', 'run', response, 'ok'))
    assert score.parsed_answer == answer
    assert score.correctness is True
    assert score.error is None
