from dataclasses import asdict, replace
import copy

import pytest

from evaluation.schema import ContractError, Prediction, ZebraTarget
from zebra_fixtures import sample, zebra_case_factory


def test_zebra_target_is_not_a_choice_in_model_input():
    value = sample("Jigsaw", "D")
    assert value.target.answer == "D"
    assert value.model_input.choices == ()
    assert len(value.model_input.images) == 1


def test_registry_rejects_unknown_zebra_profile():
    from evaluation.registry import get_scorer
    with pytest.raises(ContractError):
        get_scorer("zebra_2d", "unknown")


@pytest.mark.parametrize("change", ["extra", "unknown", "type", "empty", "label", "choices", "images", "text", "subtask"])
def test_reader_rejects_invalid_zebra_manifest_contract(change, zebra_case_factory):
    from evaluation.io import parse_sample
    row = copy.deepcopy(zebra_case_factory(("Jigsaw",))["samples"][0])
    if change == "extra": row["target"]["extra"] = 1
    elif change == "unknown": row["benchmark"] = "unknown"
    elif change == "type": row["target"]["answer"] = 1
    elif change == "empty": row["target"]["answer"] = ""
    elif change == "label": row["target"]["answer"] = "a"
    elif change == "choices": row["model_input"]["choices"] = ["A"]
    elif change == "images": row["model_input"]["images"] = []
    elif change == "text": row["model_input"]["text"] = ""
    else: row["subtask"] = "Other"
    with pytest.raises(ContractError):
        parse_sample(row)


def test_search_reader_preserves_target_and_rejects_normalized_empty(zebra_case_factory):
    from evaluation.io import parse_sample
    row = zebra_case_factory(("Search",))["samples"][0]
    parsed = parse_sample(row)
    assert isinstance(parsed.target, ZebraTarget) and asdict(parsed.target) == {"answer": "knife"}
    row["target"]["answer"] = " . "
    with pytest.raises(ContractError): parse_sample(row)


def test_zebra_config_categories_and_max_steps_contract(zebra_case_factory):
    from evaluation.io import parse_config
    case = zebra_case_factory()
    for categories in ([], ["Jigsaw", "Jigsaw"], ["Other"]):
        raw = copy.deepcopy(case["config"]); raw["categories"] = categories
        with pytest.raises(ContractError): parse_config(raw, case["root"])
    raw = copy.deepcopy(case["config"]); raw["max_steps"] = 1
    with pytest.raises(ContractError): parse_config(raw, case["root"])
    case["config"]["max_steps"] = None
    assert parse_config(case["config"], case["root"]).max_steps is None


@pytest.mark.parametrize("change", ["target", "subtask", "input", "prediction"])
def test_direct_scorer_rejects_wrong_contract_before_extraction(change):
    from evaluation.benchmarks.zebra_2d import score_zebra_2d
    value = sample("Jigsaw", "A")
    prediction = Prediction(value.id, "m", "r", "A", "ok")
    if change == "target": value = replace(value, target=object())
    elif change == "subtask": value = replace(value, subtask="Other")
    elif change == "input": value = replace(value, model_input=replace(value.model_input, choices=("A",)))
    else: prediction = replace(prediction, status="infrastructure_error", error="worker")
    with pytest.raises(ContractError): score_zebra_2d(value, prediction)
