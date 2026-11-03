"""Conservative, target-independent offline scoring for Zebra-CoT 2D tasks."""
from dataclasses import replace
import re

from evaluation.schema import (
    ContractError,
    ModelInput,
    ParsedAnswer,
    Prediction,
    Sample,
    Score,
    ZebraTarget,
)

CATEGORIES = ("Jigsaw", "Search")
PROFILE = "zebra_2d_rules_v1"

_MARKER = re.compile(
    r"^[ \t]*(?:the[ \t]+)?(?:final[ \t]+answer(?:[ \t]+is(?:[:：]|[ \t]+)|[:：])|answer[:：])",
    re.IGNORECASE | re.MULTILINE,
)
_JIGSAW = re.compile(r"(?:[A-Da-d]|\([A-Da-d]\))\.?\Z")


def normalize_search(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    if normalized.endswith(".") and not normalized.endswith(".."):
        normalized = normalized[:-1].rstrip()
    return normalized


def validate_target(target: ZebraTarget, model_input: ModelInput, subtask: str) -> None:
    if not isinstance(target, ZebraTarget):
        raise ContractError("Zebra target must be ZebraTarget")
    if subtask not in CATEGORIES:
        raise ContractError("Zebra subtask must be canonical")
    if (
        not isinstance(model_input.text, str)
        or not model_input.text.strip()
        or len(model_input.images) != 1
        or bool(model_input.choices)
    ):
        raise ContractError("Zebra requires nonempty text, one image, and no choices")
    if not isinstance(target.answer, str):
        raise ContractError("Zebra target answer must be a string")
    if subtask == "Jigsaw":
        if not re.fullmatch(r"[A-D]", target.answer):
            raise ContractError("Jigsaw target must be one uppercase A-D label")
    elif not normalize_search(target.answer):
        raise ContractError("Search target must be nonempty after normalization")


def _top_level_boxes(value: str) -> tuple[list[tuple[int, int]], int | None]:
    token = r"\boxed{"
    boxes = []
    start = None
    content_start = None
    depth = 0
    index = 0
    while index < len(value):
        if depth == 0 and value.startswith(token, index):
            start = index
            content_start = index + len(token)
            depth = 1
            index = content_start
            continue
        if depth:
            if value[index] == "{":
                depth += 1
            elif value[index] == "}":
                depth -= 1
                if depth == 0:
                    boxes.append((content_start, index))
                    start = None
                    content_start = None
        index += 1
    return boxes, start


def _unwrap_whole(value: str) -> str:
    value = value.strip()
    while True:
        unwrapped = None
        for token in (r"\boxed{", r"\text{"):
            if not value.startswith(token):
                continue
            depth = 1
            index = len(token)
            while index < len(value) and depth:
                if value[index] == "{":
                    depth += 1
                elif value[index] == "}":
                    depth -= 1
                index += 1
            if depth == 0 and not value[index:].strip():
                unwrapped = value[len(token):index - 1].strip()
            break
        if unwrapped is None:
            return value
        value = unwrapped


def extract_answer(response: str, model_input: ModelInput) -> ParsedAnswer:
    del model_input
    stripped = response.strip()
    if not stripped:
        return ParsedAnswer("", "empty", "raw")
    boxes, unclosed = _top_level_boxes(stripped)
    if unclosed is not None:
        return ParsedAnswer("", "invalid_format", "boxed", "unclosed_box")
    if boxes:
        start, end = boxes[-1]
        candidate = _unwrap_whole(stripped[start:end])
        if not candidate:
            return ParsedAnswer("", "invalid_format", "boxed", "empty_extraction")
        return ParsedAnswer(candidate, "parsed", "boxed")
    markers = list(_MARKER.finditer(stripped))
    if markers:
        candidate = stripped[markers[-1].end():].strip()
        if not candidate:
            return ParsedAnswer("", "invalid_format", "marker", "empty_extraction")
        return ParsedAnswer(candidate, "parsed", "marker")
    return ParsedAnswer(stripped, "parsed", "raw")


def validate_candidate(parsed: ParsedAnswer, subtask: str) -> ParsedAnswer:
    if subtask not in CATEGORIES:
        raise ContractError("Zebra subtask must be canonical")
    if parsed.status != "parsed":
        return parsed
    if subtask == "Jigsaw" and not _JIGSAW.fullmatch(parsed.value.strip()):
        return replace(parsed, status="invalid_format", error="invalid_jigsaw_label")
    if subtask == "Search" and not normalize_search(parsed.value):
        return replace(parsed, status="invalid_format", error="empty_extraction")
    return parsed


def _jigsaw_label(value: str) -> str:
    value = value.strip()
    if value.endswith("."):
        value = value[:-1]
    if value.startswith("("):
        value = value[1:-1]
    return value.upper()


def compare_answer(parsed: ParsedAnswer, target: ZebraTarget, subtask: str) -> bool:
    if parsed.status != "parsed":
        return False
    if subtask == "Jigsaw":
        return _jigsaw_label(parsed.value) == target.answer
    if subtask == "Search":
        return normalize_search(parsed.value) == normalize_search(target.answer)
    raise ContractError("Zebra subtask must be canonical")


def score_zebra_2d(sample: Sample, prediction: Prediction) -> Score:
    if sample.benchmark != "zebra_2d":
        raise ContractError("Zebra scorer requires zebra_2d sample")
    if prediction.status != "ok" or not isinstance(prediction.response, str):
        raise ContractError("Zebra scorer requires ok prediction")
    validate_target(sample.target, sample.model_input, sample.subtask)
    parsed = extract_answer(prediction.response, sample.model_input)
    parsed = validate_candidate(parsed, sample.subtask)
    correct = compare_answer(parsed, sample.target, sample.subtask)
    return Score(sample.id, sample.subtask, "zebra_2d", PROFILE, "1",
                 parsed.status, parsed.value, parsed.method, correct, None, parsed.error)
