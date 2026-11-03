from dataclasses import replace

import pytest

from evaluation.schema import ModelInput, Prediction, ZebraTarget
from evaluation.benchmarks.zebra_2d import extract_answer, normalize_search, score_zebra_2d
from zebra_fixtures import sample


@pytest.mark.parametrize("value,expected", [
    (" Knife. ", "knife"), ("3.14", "3.14"), ("word..", "word.."),
    (" $ 11  billion ", "$ 11 billion"), ("9/25/87", "9/25/87"),
])
def test_search_normalization(value, expected):
    assert normalize_search(value) == expected


def test_changing_gold_does_not_change_extraction():
    first = sample("Jigsaw", "A")
    second = replace(first, target=ZebraTarget("B"))
    pred = Prediction(first.id, "m", "r", r"why\n\boxed{\text{B}}", "ok")
    a, b = score_zebra_2d(first, pred), score_zebra_2d(second, pred)
    assert (a.parsed_answer, a.extraction_method, a.status) == (b.parsed_answer, b.extraction_method, b.status)
    assert a.correctness is False and b.correctness is True


def test_last_unclosed_box_does_not_rescue_earlier_answer():
    parsed = extract_answer(r"\boxed{A} later \boxed{B", ModelInput("q", ("images/x.jpg",), ()))
    assert parsed.status == "invalid_format"


@pytest.mark.parametrize("response,value,status,method", [
    ("answer", "answer", "parsed", "raw"),
    ("Answer: first\nsecond", "first\nsecond", "parsed", "marker"),
    (r"\boxed{a{b}c}", "a{b}c", "parsed", "boxed"),
    ("Answer: A\nThe final answer is B", "B", "parsed", "marker"),
    ("The Answer： C", "C", "parsed", "marker"),
    ("\tFinal\tanswer\tis D", "D", "parsed", "marker"),
    ("Final answer is: line one\nline two", "line one\nline two", "parsed", "marker"),
    (r"\boxed{\boxed{\text{B}}}", "B", "parsed", "boxed"),
    (r"\boxed{x \text{B}}", r"x \text{B}", "parsed", "boxed"),
    (r"\boxed{$B$}", "$B$", "parsed", "boxed"),
    ("", "", "empty", "raw"),
    ("Answer:   ", "", "invalid_format", "marker"),
    (r"\boxed{}", "", "invalid_format", "boxed"),
])
def test_extraction_grammar(response, value, status, method):
    parsed = extract_answer(response, ModelInput("q", ("images/x.jpg",), ()))
    assert (parsed.value, parsed.status, parsed.method) == (value, status, method)


@pytest.mark.parametrize("response,answer,correct", [
    ("a", "A", True), ("(B).", "B", True), ("D.", "C", False),
    ("( A )", "A", False), ("Option B", "B", False), ("B because", "B", False),
    ("B..", "B", False),
])
def test_jigsaw_candidate_grammar_and_comparison(response, answer, correct):
    result = score_zebra_2d(sample("Jigsaw", answer), Prediction("id", "m", "r", response, "ok"))
    assert result.correctness is correct
    assert result.status == ("parsed" if response in {"a", "(B).", "D."} else "invalid_format")


@pytest.mark.parametrize("response,answer,correct", [
    ("A", "A", True), ("yes", "yes", True), ("A", "yes", False),
    ("1/2", "0.5", False), ("9/25/87", "1987-09-25", False),
    ("$ 11 billion", "$11 billion", False), ("A knife.", "knife", False),
    ("The bicycle is red.", "red", False), ("red bicycle", "red", False),
    ("word..", "word", False),
])
def test_search_compares_the_normalized_full_operand(response, answer, correct):
    result = score_zebra_2d(sample("Search", answer), Prediction("id", "m", "r", response, "ok"))
    assert result.correctness is correct
