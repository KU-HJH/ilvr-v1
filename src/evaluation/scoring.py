"""Answer parsing and exact-match scoring shared by evaluation entry points."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Mapping


_SPECIAL_TOKENS = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end|>",
    "</s>",
    "<s>",
    "[/INST]",
    "[INST]",
    "<|latent_start|>",
    "<|latent_end|>",
    "<|latent_pad|>",
)
_BOXED_MARKER = re.compile(r"(?<!\\)(?:\\\\|\\)boxed\{")
_FINAL_MARKER = re.compile(
    r"final\s*answer\s*(?:is)?\s*[:：]?\s*(.*)", re.IGNORECASE
)
_ANSWER_MARKER = re.compile(
    r"answer\s*(?:is)?\s*[:：]\s*(.*)", re.IGNORECASE
)
_DECIMAL_NUMBER = re.compile(
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\Z"
)
_YES_ALIASES = {"yes", "true", "a"}
_NO_ALIASES = {"no", "false", "b"}
_TERMINAL_PUNCTUATION = ".,!?;:。，，!！?？;；:："


@dataclass(frozen=True)
class AnswerParseResult:
    extracted: str
    normalized: str
    status: str
    detail: str | None


@dataclass(frozen=True)
class AnswerScore:
    prediction: AnswerParseResult
    gold: AnswerParseResult
    match: bool
    invalid_gold_reason: str | None


@dataclass(frozen=True)
class EvaluationScoreSummary:
    correct: int
    source_count: int
    valid_gold_count: int
    invalid_gold_count: int
    invalid_gold: list[int]
    coverage_accuracy: float
    valid_gold_accuracy: float

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (
                self.correct,
                self.source_count,
                self.valid_gold_count,
                self.invalid_gold_count,
            )
        ):
            raise ValueError("score counts must be non-negative integers")
        if self.valid_gold_count + self.invalid_gold_count != self.source_count:
            raise ValueError("gold counts must add up to source_count")
        if self.correct > self.valid_gold_count:
            raise ValueError("correct cannot exceed valid_gold_count")
        if len(self.invalid_gold) != self.invalid_gold_count:
            raise ValueError("invalid_gold must list each invalid gold row")
        if not all(math.isfinite(value) for value in (
            self.coverage_accuracy,
            self.valid_gold_accuracy,
        )):
            raise ValueError("score accuracies must be finite")

    @classmethod
    def from_rows(
        cls, rows: Iterable[Mapping[str, object]], source_count: int
    ) -> "EvaluationScoreSummary":
        if not isinstance(source_count, int) or isinstance(source_count, bool):
            raise ValueError("source_count must be an integer")
        if source_count < 0:
            raise ValueError("source_count must be non-negative")

        materialized_rows = list(rows)
        if len(materialized_rows) != source_count:
            raise ValueError("row count must equal source_count")

        invalid_gold = [
            row.get("index")
            for row in materialized_rows
            if row.get("invalid_gold_reason") is not None
        ]
        if not all(isinstance(index, int) and not isinstance(index, bool) for index in invalid_gold):
            raise ValueError("invalid gold rows require integer indices")

        invalid_gold_count = len(invalid_gold)
        valid_gold_count = source_count - invalid_gold_count
        correct = sum(
            bool(row.get("match"))
            for row in materialized_rows
            if row.get("invalid_gold_reason") is None
        )
        coverage_accuracy = correct / source_count if source_count else 0.0
        valid_gold_accuracy = (
            correct / valid_gold_count if valid_gold_count else 0.0
        )
        return cls(
            correct=correct,
            source_count=source_count,
            valid_gold_count=valid_gold_count,
            invalid_gold_count=invalid_gold_count,
            invalid_gold=invalid_gold,
            coverage_accuracy=coverage_accuracy,
            valid_gold_accuracy=valid_gold_accuracy,
        )


def _strip_special_tokens(text: str) -> str:
    for marker in _SPECIAL_TOKENS:
        text = text.replace(marker, "")
    return text.strip()


def _clean_answer_span(candidate: str) -> str:
    span = candidate.splitlines()[0].strip() if candidate else ""
    span = _strip_special_tokens(span)
    while span:
        trimmed = span.rstrip(_TERMINAL_PUNCTUATION).strip()
        if trimmed != span:
            span = trimmed
            continue
        for opening, closing in (("**", "**"), ("`", "`"), ('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")):
            if span.startswith(opening) and span.endswith(closing):
                inner = span[len(opening):len(span) - len(closing)].strip()
                if inner != span:
                    span = inner
                    break
        else:
            break
    return span.strip()


def _normalize(span: str) -> str:
    lowered = span.lower()
    if lowered in _YES_ALIASES:
        return "Yes"
    if lowered in _NO_ALIASES:
        return "No"
    if _DECIMAL_NUMBER.fullmatch(span):
        value = float(span.replace(",", ""))
        if value.is_integer():
            return str(int(value))
        return str(value)
    return lowered


def _parsed(span: str, status: str = "parsed", detail: str | None = None) -> AnswerParseResult:
    return AnswerParseResult(span, _normalize(span) if span else "", status, detail)


def extract_final_answer(text: str) -> AnswerParseResult:
    """Extract a final answer without applying symbolic or algebraic equivalence."""
    source = _strip_special_tokens(text if isinstance(text, str) else "")
    markers = list(_BOXED_MARKER.finditer(source))
    if markers:
        start = markers[-1].end()
        depth = 1
        for position in range(start, len(source)):
            if source[position] == "{":
                depth += 1
            elif source[position] == "}":
                depth -= 1
                if depth == 0:
                    return _parsed(_clean_answer_span(source[start:position]))
        return _parsed("", "malformed", "unclosed_box")

    for marker in (_FINAL_MARKER, _ANSWER_MARKER):
        matches = list(marker.finditer(source))
        if matches:
            span = _clean_answer_span(matches[-1].group(1))
            return _parsed(span, "parsed" if span else "empty_prediction")

    span = _clean_answer_span(source)
    return _parsed(span, "parsed" if span else "empty_prediction")


def parse_gold_answer(value: object) -> AnswerParseResult:
    """Parse a gold answer while making missing and blank labels explicit."""
    if value is None:
        return AnswerParseResult("", "", "invalid_gold", "missing")
    if not isinstance(value, str) or not value.strip():
        return AnswerParseResult("", "", "invalid_gold", "blank")

    result = extract_final_answer(value)
    if result.status == "empty_prediction":
        return AnswerParseResult("", "", "invalid_gold", "blank")
    return result


def score_answer(prediction_text: str, gold_value: object) -> AnswerScore:
    prediction = extract_final_answer(prediction_text)
    gold = parse_gold_answer(gold_value)
    invalid_gold_reason = (
        gold.detail if gold.status == "invalid_gold" else
        gold.status if gold.status != "parsed" else None
    )
    match = (
        prediction.status == "parsed"
        and gold.status == "parsed"
        and prediction.normalized == gold.normalized
    )
    return AnswerScore(prediction, gold, match, invalid_gold_reason)
