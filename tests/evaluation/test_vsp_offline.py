import inspect

import pytest

from evaluation.schema import ContractError, ModelInput, VspTarget, ParsedAnswer
from evaluation.benchmarks.vsp import extract_answer, validate_path, normalize_map, compare_answer


INPUT = ModelInput('Synthetic question', (), ())
TARGET = VspTarget(((1, 2), (0, 0)), 100)


def parse(text):
    return validate_path(extract_answer(text, INPUT))


@pytest.mark.parametrize('text', ['R', 'RD', 'final answer: R', r'\boxed{R}', 'Action Plan： r'])
def test_goal_ends_episode(text):
    result = compare_answer(parse(text), TARGET)
    assert (result.success, result.position, result.steps) == (True, (0, 1), 1)
    assert result.terminated and not result.truncated


@pytest.mark.parametrize('text', ['XR', 'RX', 'R.', '"R"', 'RIGHT', ',R', 'R,', 'R,,D', 'R\u00a0'])
def test_unknown_or_empty_action_group_is_not_erased(text):
    assert parse(text).status == 'invalid_format'


def test_extraction_has_no_gold_and_does_not_hide_malformed_last_box():
    assert 'target' not in inspect.signature(extract_answer).parameters
    assert parse('final answer: R').value == 'R'
    assert parse(r'\boxed{R} then \boxed{R').status == 'invalid_format'
    assert parse('Action plan: R\nX').status == 'invalid_format'
    assert parse('Reasoning\nFinal answer: R').value == 'R'
    assert parse('final answer: R\nFinal answer: D').value == 'D'
    assert parse(' \t\n').status == 'empty'
    assert parse('R, d\nL').value == 'RDL'


def test_hole_boundary_and_time_limit():
    hole = VspTarget(((1, -1), (0, 2)), 100)
    result = compare_answer(parse('RD'), hole)
    assert (result.success, result.steps, result.reason) == (False, 1, 'hole')
    result = compare_answer(parse('LR'), VspTarget(((1, 2),), 1))
    assert (result.position, result.steps, result.reason) == ((0, 0), 1, 'step_limit')
    assert result.truncated and not result.terminated
    result = compare_answer(parse('L' * 99 + 'R'), TARGET)
    assert result.success and result.terminated and result.truncated and result.steps == 100
    result = compare_answer(parse('L' * 100 + 'R'), TARGET)
    assert not result.success and result.truncated and result.steps == 100
    result = compare_answer(parse('D'), TARGET)
    assert result.reason == 'path_exhausted' and not result.terminated and not result.truncated


@pytest.mark.parametrize('grid', [(), ((1, 2), (0,)), ((True, 2),), ((1, 1, 2),), ((1, 0),), ((1, 2, 3),)])
def test_invalid_map_is_contract_error(grid):
    with pytest.raises(ContractError):
        normalize_map(VspTarget(grid, 100))


def test_character_map_and_missing_gold_path_are_valid():
    assert normalize_map(TARGET) == (('S', 'G'), ('F', 'F'))
    assert normalize_map(VspTarget((('S', 'G'),), 100)) == (('S', 'G'),)


def test_direct_compare_cannot_bypass_grammar_and_box_is_balanced():
    with pytest.raises(ContractError):
        compare_answer(ParsedAnswer('RX', 'parsed', 'raw'), TARGET)
    assert parse(r'\boxed{}').status == 'empty'
    assert parse(r'\boxed{{R}}').status == 'invalid_format'
    assert parse(r'\boxed{D} then \boxed{R}').value == 'R'


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_step_limit(limit):
    with pytest.raises(ContractError):
        normalize_map(VspTarget(((1, 2),), limit))
