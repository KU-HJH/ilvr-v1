import copy
import inspect

import pytest

from comt_fixtures import CATEGORIES, make_comt_case
from offline_fixtures import make_case


def test_choice_mapping_reads_this_question():
    from evaluation.benchmarks.comt import compare_answer, extract_answer
    from evaluation.schema import McqTarget, ModelInput

    parsed = extract_answer("B", ModelInput("Q", (), ("5", "10", "15", "18")))
    assert parsed.value == "B"
    assert compare_answer(parsed, McqTarget(1, "B", "10")) is True
    assert compare_answer(parsed, McqTarget(0, "A", "5")) is False


@pytest.mark.parametrize(
    ("response", "choices", "value", "status", "method"),
    [
        ("(A) then (c)", ("5", "10", "15", "18"), "C", "parsed", "parenthesized"),
        (
            "ANSWER: (A) [Answer] ANSWER: B",
            ("5", "10", "15", "18"),
            "B",
            "parsed",
            "upper_label",
        ),
        ("[Answer] (A)", ("5", "10", "15", "18"), "A", "parsed", "parenthesized"),
        ("ANSWER: D", ("red", "green", "blue", "black"), "D", "parsed", "upper_label"),
        ("15", ("5", "10", "15", "18"), "C", "parsed", "choice_content"),
        ("18 then 10", ("5", "10", "15", "18"), "D", "parsed", "choice_content"),
        ("d a", ("5", "10", "15", "18"), "D", "parsed", "lower_label"),
        ("A D", ("5", "10", "15", "18"), "D", "parsed", "upper_label"),
        ("(Z)", ("5", "10", "15", "18"), "Z", "invalid_format", "parenthesized"),
        (r"\boxed{B}", ("5", "10", "15", "18"), r"\boxed{B}", "invalid_format", "raw"),
        ("( A )", ("5", "10", "15", "18"), "A", "parsed", "upper_label"),
        ("A\tD", ("5", "10", "15", "18"), "A\tD", "invalid_format", "raw"),
        (" \t\r\n", ("5", "10", "15", "18"), "", "empty", "raw"),
    ],
)
def test_reference_extraction_precedence(response, choices, value, status, method):
    from evaluation.benchmarks.comt import extract_answer
    from evaluation.schema import ModelInput

    parsed = extract_answer(response, ModelInput("Q", (), choices))
    assert (parsed.value, parsed.status, parsed.method) == (value, status, method)


@pytest.mark.parametrize(
    ("response", "value", "status", "method"),
    [
        ("foo  b", "B", "parsed", "lower_label"),
        ("b ", "B", "parsed", "lower_label"),
        ("foo  bar", "foo  bar", "invalid_format", "raw"),
        ("A AB", "A", "parsed", "upper_label"),
        ("A BC", "A", "parsed", "upper_label"),
        ("A d", "A", "parsed", "upper_label"),
        ("a D", "D", "parsed", "upper_label"),
    ],
)
def test_token_fallback_requires_exact_letters_and_prefers_uppercase(
    response, value, status, method
):
    from evaluation.benchmarks.comt import extract_answer
    from evaluation.schema import ModelInput

    parsed = extract_answer(response, ModelInput("Q", (), ("5", "10", "15", "18")))
    assert (parsed.value, parsed.status, parsed.method) == (value, status, method)


@pytest.mark.parametrize(
    ("response", "value", "status", "method"),
    [
        ("STRASSE", "STRASSE", "invalid_format", "raw"),
        ("Straße", "A", "parsed", "choice_content"),
        ("straße", "A", "parsed", "choice_content"),
    ],
)
def test_choice_content_uses_source_lowercase_semantics(
    response, value, status, method
):
    from evaluation.benchmarks.comt import extract_answer
    from evaluation.schema import ModelInput

    choices = ("Straße", "green", "yellow", "purple")
    parsed = extract_answer(response, ModelInput("Q", (), choices))
    assert (parsed.value, parsed.status, parsed.method) == (value, status, method)


def test_duplicate_choice_content_uses_highest_choice_index():
    from evaluation.benchmarks.comt import extract_answer
    from evaluation.schema import ModelInput

    parsed = extract_answer("same", ModelInput("Q", (), ("same", "same", "x", "y")))
    assert (parsed.value, parsed.method) == ("B", "choice_content")


def test_extraction_is_independent_of_gold_and_content_is_not_an_oracle():
    from evaluation.benchmarks.comt import compare_answer, extract_answer
    from evaluation.schema import McqTarget, ModelInput

    assert "target" not in inspect.signature(extract_answer).parameters
    model_input = ModelInput("Q", (), ("same", "same", "15", "18"))
    parsed = extract_answer("same", model_input)
    assert parsed.value == "B"
    assert compare_answer(parsed, McqTarget(1, "B", "same")) is True
    assert compare_answer(parsed, McqTarget(0, "A", "same")) is False


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("bool_index", "index"),
        ("out_of_range", "index"),
        ("wrong_label", "label"),
        ("empty_content", "content"),
        ("wrong_content", "content"),
        ("empty_target", "fields"),
        ("three_choices", "four"),
        ("empty_choice", "nonempty"),
    ],
)
def test_comt_sample_target_and_choice_contract_rejects_malformed_values(kind, match):
    from evaluation.io import parse_sample
    from evaluation.schema import ContractError

    row = copy.deepcopy(make_comt_case(1)["samples"][0])
    if kind == "bool_index":
        row["target"]["index"] = True
    elif kind == "out_of_range":
        row["target"]["index"] = 4
    elif kind == "wrong_label":
        row["target"]["label"] = "A"
    elif kind == "empty_content":
        row["target"]["content"] = ""
    elif kind == "wrong_content":
        row["target"]["content"] = "eleven"
    elif kind == "empty_target":
        row["target"] = {}
    elif kind == "three_choices":
        row["model_input"]["choices"].pop()
    else:
        row["model_input"]["choices"][2] = "  "
    with pytest.raises(ContractError, match=match):
        parse_sample(row)


def test_parse_sample_dispatches_comt_target_and_preserves_vsp_map_only_target():
    from evaluation.io import parse_sample
    from evaluation.schema import McqTarget, VspTarget

    comt = parse_sample(make_comt_case(1)["samples"][0])
    vsp = parse_sample(make_case(1)["samples"][0])
    assert comt.target == McqTarget(1, "B", "10")
    assert isinstance(vsp.target, VspTarget)


def test_parse_sample_rejects_unknown_target_benchmark():
    from evaluation.io import parse_sample
    from evaluation.schema import ContractError

    row = copy.deepcopy(make_comt_case(1)["samples"][0])
    row["benchmark"] = "other"
    with pytest.raises(ContractError, match="unsupported sample benchmark"):
        parse_sample(row)


def test_comt_config_accepts_omitted_or_null_max_steps():
    from evaluation.io import parse_config

    case = make_comt_case(1)
    omitted = parse_config(case["config"], case["root"])
    case["config"]["max_steps"] = None
    explicit_null = parse_config(case["config"], case["root"])
    assert omitted.max_steps is None and explicit_null.max_steps is None


def test_comt_config_rejects_nonnull_max_steps_and_bad_categories():
    from evaluation.io import parse_config
    from evaluation.schema import ContractError

    case = make_comt_case(1)
    case["config"]["max_steps"] = 100
    with pytest.raises(ContractError, match="max_steps"):
        parse_config(case["config"], case["root"])
    for categories in ([], ["creation", "creation"], ["creation", "unknown"]):
        raw = copy.deepcopy(case["config"])
        raw["max_steps"] = None
        raw["categories"] = categories
        with pytest.raises(ContractError, match="categories"):
            parse_config(raw, case["root"])


def test_vsp_config_still_requires_positive_max_steps():
    from evaluation.io import parse_config
    from evaluation.schema import ContractError

    case = make_case(1)
    assert parse_config(case["config"], case["root"]).max_steps == 100
    del case["config"]["max_steps"]
    with pytest.raises(ContractError):
        parse_config(case["config"], case["root"])


def test_read_manifest_applies_step_limit_only_to_vsp():
    from evaluation.io import parse_config, read_manifest

    comt_case = make_comt_case(1)
    comt_config = parse_config(comt_case["config"], comt_case["root"])
    assert read_manifest(comt_case["manifest"], comt_config, comt_case["root"]).samples


def test_registry_rejects_unknown_combinations_and_registers_comt():
    from evaluation.registry import get_scorer
    from evaluation.schema import ContractError

    assert callable(get_scorer("comt", "comt_reference_rules_v1"))
    for pair in (("comt", "unknown"), ("unknown", "comt_reference_rules_v1")):
        with pytest.raises(ContractError, match="unsupported benchmark/profile"):
            get_scorer(*pair)


def test_canonical_comt_categories_are_all_allowed_as_a_unique_subset():
    from evaluation.io import parse_config

    case = make_comt_case(1)
    for category in CATEGORIES:
        raw = copy.deepcopy(case["config"])
        raw["categories"] = [category]
        assert parse_config(raw, case["root"]).categories == (category,)
