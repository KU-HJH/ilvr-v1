"""Target-independent extraction and bounded scoring for EMMA."""
import re

from evaluation.emma_math import (
    ComparisonResult, compare_values, is_nonfinite_literal,
    require_emma_math_runtime, unwrap_math,
)
from evaluation.schema import ContractError, EmmaTarget, ModelInput, ParsedAnswer, Prediction, Sample, Score

CATEGORIES = ('Chemistry', 'Coding', 'Math', 'Physics')
PROFILE = 'emma_safe_rules_v1'
_NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
_MARKERS = ('final answer is', 'correct answer is', 'answer should be', 'answer is', 'answer:')


def validate_target(target: EmmaTarget, model_input: ModelInput) -> None:
    if target.question_type not in ('multiple_choice', 'open_ended'):
        raise ContractError('EMMA question_type must be multiple_choice or open_ended')
    if not isinstance(target.answer, str) or not target.answer:
        raise ContractError('EMMA answer must be a nonempty string')
    if not isinstance(target.content, str) or not target.content:
        raise ContractError('EMMA content must be a nonempty string')
    if not isinstance(target.category, str) or not isinstance(target.task, str):
        raise ContractError('EMMA category and task must be strings')
    if target.question_type == 'multiple_choice':
        if not 2 <= len(model_input.choices) <= 8 or any(not isinstance(choice, str) or not choice for choice in model_input.choices):
            raise ContractError('EMMA multiple_choice requires 2–8 nonempty choices')
        if type(target.index) is not int or target.index not in range(len(model_input.choices)):
            raise ContractError('EMMA multiple_choice index is invalid')
        label = chr(ord('A') + target.index)
        if target.answer.upper() != label or target.content != model_input.choices[target.index]:
            raise ContractError('EMMA target must match indexed answer and content')
    elif target.index is not None or model_input.choices or target.content != target.answer:
        raise ContractError('EMMA open-ended target requires null index, no choices, and matching content')


def _last_box(response: str) -> str | None:
    complete = None
    start = 0
    while True:
        opening = response.find(r'\boxed{', start)
        if opening < 0:
            return complete
        content_start = opening + len(r'\boxed{')
        depth = 1
        position = content_start
        while position < len(response) and depth:
            if response[position] == '{': depth += 1
            elif response[position] == '}': depth -= 1
            position += 1
        if depth == 0:
            complete = response[content_start:position - 1]
        start = content_start


def _finish(candidate: str, method: str, *, box: bool = False) -> ParsedAnswer:
    candidate = candidate.strip()
    if box and candidate.startswith(r'\text{'):
        depth = 1
        position = len(r'\text{')
        while position < len(candidate) and depth:
            if candidate[position] == '{': depth += 1
            elif candidate[position] == '}': depth -= 1
            position += 1
        if depth == 0 and position == len(candidate):
            candidate = candidate[6:-1].strip()
    normalized = unwrap_math(candidate)
    if not normalized.strip():
        return ParsedAnswer(candidate, 'invalid_format', method, 'empty_extraction')
    if is_nonfinite_literal(normalized):
        return ParsedAnswer(candidate, 'invalid_format', method, 'non_finite_value')
    return ParsedAnswer(candidate, 'parsed', method)


def extract_answer(response: str, model_input: ModelInput) -> ParsedAnswer:
    if not isinstance(response, str):
        raise ContractError('response must be a string')
    raw = response.strip()
    if not raw:
        return ParsedAnswer('', 'empty', 'raw')
    labels = tuple(chr(ord('A') + i) for i in range(len(model_input.choices)))
    if len(raw) == 1 and raw.upper() in labels:
        return _finish(raw, 'upper_label' if raw.isupper() else 'lower_label')
    prefix = re.match(r'([A-Z])[:.]', raw)
    if prefix and prefix.group(1) in labels:
        return _finish(prefix.group(1), 'upper_label')
    if _NUMBER.fullmatch(raw):
        return _finish(raw, 'raw')
    boxed = _last_box(raw)
    if boxed is not None:
        return _finish(boxed, 'boxed', box=True)
    lowered = raw.lower()
    for marker in _MARKERS:
        position = lowered.rfind(marker)
        if position >= 0:
            tail = raw[position + len(marker):]
            candidate = (tail.splitlines()[0] if tail else '').strip()
            if candidate.endswith('.') and not _NUMBER.fullmatch(candidate):
                candidate = candidate[:-1].rstrip()
            return _finish(candidate, 'marker')
    return _finish(raw, 'raw')


def compare_answer(parsed: ParsedAnswer, target: EmmaTarget, model_input: ModelInput) -> ComparisonResult:
    if parsed.status != 'parsed':
        return ComparisonResult(False, parsed.error)
    candidate = unwrap_math(parsed.value)
    if target.question_type == 'multiple_choice':
        labels = tuple(chr(ord('A') + i) for i in range(len(model_input.choices)))
        if len(candidate) == 1 and candidate.upper() in labels:
            return ComparisonResult(candidate.upper() == chr(ord('A') + target.index))
        expected = target.content
    else:
        expected = target.answer
    # compare_values normalizes each original operand once; do not pass the
    # separately normalized label candidate and remove a second wrapper.
    return compare_values(parsed.value, expected)


def score_emma(sample: Sample, prediction: Prediction) -> Score:
    require_emma_math_runtime()
    if prediction.status != 'ok' or not isinstance(sample.target, EmmaTarget):
        raise ContractError('EMMA requires an ok prediction and EMMA target')
    validate_target(sample.target, sample.model_input)
    parsed = extract_answer(prediction.response, sample.model_input)
    comparison = compare_answer(parsed, sample.target, sample.model_input)
    return Score(sample.id, sample.subtask, 'emma', PROFILE, '1', parsed.status,
                 parsed.value, parsed.method, comparison.correctness, None,
                 parsed.error or comparison.error)
