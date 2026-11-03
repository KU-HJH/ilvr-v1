import re

from evaluation.schema import (
    ContractError,
    McqTarget,
    ModelInput,
    ParsedAnswer,
    Prediction,
    Sample,
    Score,
)


CATEGORIES = ("creation", "deletion", "update", "selection")
LABELS = tuple("ABCD")
LOWER_LABELS = tuple(label.lower() for label in LABELS)
PARENTHESIZED = re.compile(r"\(([A-Za-z])\)")
TOKEN_SEPARATORS = str.maketrans({char: " " for char in "\n.,!?"})


def validate_target(target: McqTarget, model_input: ModelInput) -> None:
    if len(model_input.choices) != 4:
        raise ContractError("CoMT requires exactly four choices")
    if any(not isinstance(choice, str) or not choice.strip() for choice in model_input.choices):
        raise ContractError("CoMT choices must be nonempty strings")
    if type(target.index) is not int or target.index not in range(4):
        raise ContractError("CoMT target index must be an integer from 0 through 3")
    if target.label != LABELS[target.index]:
        raise ContractError("CoMT target label does not match index")
    if not isinstance(target.content, str) or not target.content.strip():
        raise ContractError("CoMT target content must be nonempty")
    if target.content != model_input.choices[target.index]:
        raise ContractError("CoMT target content does not match indexed choice")


def _label_answer(value: str, method: str) -> ParsedAnswer:
    if value in LABELS:
        return ParsedAnswer(value, "parsed", method)
    return ParsedAnswer(value, "invalid_format", method, "invalid_choice_label")


def extract_answer(response: str, model_input: ModelInput) -> ParsedAnswer:
    if not isinstance(response, str):
        raise ContractError("response must be a string")
    if not response.strip():
        return ParsedAnswer("", "empty", "raw")

    candidate = response.split("ANSWER:")[-1] if "[Answer]" in response else response
    parenthesized = PARENTHESIZED.findall(candidate)
    if parenthesized:
        return _label_answer(parenthesized[-1].upper(), "parenthesized")

    lowered = candidate.lower()
    content_index = None
    for index, choice in enumerate(model_input.choices):
        if choice.lower() in lowered:
            content_index = index
    if content_index is not None:
        return ParsedAnswer(LABELS[content_index], "parsed", "choice_content")

    tokens = candidate.translate(TOKEN_SEPARATORS).split(" ")
    upper = [token for token in tokens if token in LABELS]
    if upper:
        return ParsedAnswer(max(upper), "parsed", "upper_label")
    lower = [token for token in tokens if token in LOWER_LABELS]
    if lower:
        return ParsedAnswer(max(lower).upper(), "parsed", "lower_label")
    return ParsedAnswer(candidate, "invalid_format", "raw", "unparseable_choice")


def compare_answer(parsed: ParsedAnswer, target: McqTarget) -> bool:
    return parsed.status == "parsed" and parsed.value == target.label


def score_comt(sample: Sample, prediction: Prediction) -> Score:
    if prediction.status != "ok":
        raise ContractError("score_comt requires ok prediction")
    if not isinstance(sample.target, McqTarget):
        raise ContractError("score_comt requires an MCQ target")
    parsed = extract_answer(prediction.response, sample.model_input)
    return Score(
        sample.id,
        sample.subtask,
        "comt",
        "comt_reference_rules_v1",
        "1",
        parsed.status,
        parsed.value,
        parsed.method,
        compare_answer(parsed, sample.target),
        None,
        parsed.error,
    )
