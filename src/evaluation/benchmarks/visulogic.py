"""Independent implementation of the pinned VLMEvalKit VisuLogic rules."""
import re

from evaluation.schema import ContractError, McqTarget, ModelInput, ParsedAnswer, Prediction, Sample, Score

CATEGORIES = ('Position', 'Quantity', 'Style')
LABELS = tuple('ABCD')
FLAGS = (
    'answer:', 'the final answer is:', 'the answer is option:', 'the answer is:',
    'the correct answer is option:', 'the correct answer is:', 'the answer should be:',
    'the final answer is', 'the answer is option', 'the answer is',
    'the correct answer is option', 'the correct answer is', 'the answer should be',
)


def validate_target(target: McqTarget, model_input: ModelInput) -> None:
    choices = model_input.choices
    if len(choices) != 4 or any(not isinstance(c, str) or not c.strip() for c in choices):
        raise ContractError('VisuLogic requires four nonempty string choices')
    if type(target.index) is not int or target.index not in range(4):
        raise ContractError('VisuLogic target index must be an integer from 0 to 3')
    if target.label != LABELS[target.index] or target.content != choices[target.index]:
        raise ContractError('VisuLogic target must match indexed label and content')


def _boxed_candidate(response: str) -> str:
    if len(response) < 3:
        return response
    starts = []
    content = None
    # This is the source token grammar, not a general balanced-LaTeX parser.
    for match in re.finditer(r'\\boxed\{|[^\\]\}', response):
        if match.group().endswith(r'\boxed{'):
            starts.append(match.end())
        elif starts:
            start = starts.pop()
            if not starts:
                content = response[start:match.start() + 1]
    if content is None:
        return response  # VLMEvalKit's no-box exception fallback.
    if content:
        for marker in (r'\text{', r'\rm{', r'\mathbf{', '$', '}'):
            content = content.replace(marker, '')
    if 'letter' in content.lower() or 'or' in content or len(content) > 2:
        return response
    return content


def _language_candidate(response: str) -> str:
    candidate = response.replace('<|endoftext|>', '')
    for label in LABELS:
        endings = (f' {label}.', f' ({label}).')
        beginnings = (f'{label}\n', f'({label})\n', f'({label}) {label}\n')
        if candidate.endswith(endings) or candidate.startswith(beginnings):
            return label
    tail = candidate.lower()
    for marker in FLAGS:
        if marker in tail:
            tail = tail.split(marker)[-1].strip().split('\n')[0].split('.')[0]
            if tail.upper() in LABELS:
                return tail.upper()
    return candidate


def extract_answer(response: str, model_input: ModelInput) -> ParsedAnswer:
    if not isinstance(response, str):
        raise ContractError('response must be a string')
    if not response.strip():
        return ParsedAnswer('', 'empty', 'raw')
    boxed = _boxed_candidate(response).strip()
    if boxed in LABELS:
        return ParsedAnswer(boxed, 'parsed', 'box')
    language = _language_candidate(response)
    if language in LABELS:
        return ParsedAnswer(language, 'parsed', 'lang')
    return ParsedAnswer('Z', 'invalid_format', 'raw', 'unparseable_choice')


def compare_answer(parsed: ParsedAnswer, target: McqTarget) -> bool:
    return parsed.status == 'parsed' and parsed.value == target.label


def score_visulogic(sample: Sample, prediction: Prediction) -> Score:
    if prediction.status != 'ok' or not isinstance(sample.target, McqTarget):
        raise ContractError('VisuLogic requires an ok prediction and MCQ target')
    parsed = extract_answer(prediction.response, sample.model_input)
    return Score(sample.id, sample.subtask, 'visulogic', 'visulogic_vlmeval_rules_v1', '1',
                 parsed.status, parsed.value, parsed.method, compare_answer(parsed, sample.target),
                 None, parsed.error)
