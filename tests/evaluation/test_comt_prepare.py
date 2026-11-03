import hashlib
import json
from pathlib import Path

import pytest

from evaluation.schema import ContractError


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_source(case: dict) -> None:
    case["source_sha256"] = _write_jsonl(
        case["source_root"] / "comt/data.jsonl", case["source_rows"]
    )


def _rewrite_selection(case: dict) -> None:
    case["selection_sha256"] = _write_jsonl(
        case["selection_manifest"], case["selection_rows"]
    )


def _prepare(monkeypatch, case):
    from comt_source_fixtures import pin_fixture
    from evaluation.comt_prepare import prepare_comt

    pin_fixture(monkeypatch, case)
    return prepare_comt(
        case["source_root"],
        case["selection_manifest"],
        case["image_root"],
        case["output"],
        repo_root=case["root"],
    )


def test_prepare_preserves_population_and_applies_exact_repairs(monkeypatch):
    from comt_source_fixtures import make_comt_source, pin_fixture
    from evaluation.comt_prepare import prepare_comt

    case = make_comt_source()
    pin_fixture(monkeypatch, case)
    source_before = (case["source_root"] / "comt/data.jsonl").read_bytes()
    selection_before = case["selection_manifest"].read_bytes()
    summary = prepare_comt(
        case["source_root"],
        case["selection_manifest"],
        case["image_root"],
        case["output"],
        repo_root=case["root"],
    )

    assert summary["N_expected"] == 400
    assert summary["correction_count"] == 2
    assert summary["categories"] == {
        "creation": 100,
        "deletion": 100,
        "update": 100,
        "selection": 100,
    }
    payload = (case["output"] / "manifest.jsonl").read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    assert len(rows) == 400
    assert [row["provenance"]["source_id"] for row in rows] == case[
        "selected_source_ids"
    ]
    by_id = {row["provenance"]["source_id"]: row for row in rows}
    assert by_id["creation-12125"]["model_input"]["choices"] == [
        "17",
        "18",
        "22",
        "34",
    ]
    assert by_id["creation-12125"]["target"] == {
        "index": 1,
        "label": "B",
        "content": "18",
    }
    assert by_id["creation-12164"]["target"] == {
        "index": 3,
        "label": "D",
        "content": "24",
    }
    assert all(set(row["model_input"]) == {"text", "images", "choices"} for row in rows)
    assert all(len(row["model_input"]["images"]) == 1 for row in rows)
    assert not any(
        marker in json.dumps(row["model_input"])
        for row in rows
        for marker in (
            "RATIONALE_SENTINEL",
            "ANNOTATIONS_SENTINEL",
            "HELPER_IMAGE_SENTINEL",
            "PLAN_SENTINEL",
            "HELPER_PLAN_SENTINEL",
        )
    )
    assert (case["source_root"] / "comt/data.jsonl").read_bytes() == source_before
    assert case["selection_manifest"].read_bytes() == selection_before
    assert summary["manifest_sha256"] == hashlib.sha256(payload).hexdigest()

    unchanged = [
        row["model_input"]["text"] == selected["text_input"]
        for row, selected in zip(rows, case["selection_rows"], strict=True)
    ]
    assert unchanged.count(False) == 2
    assert {
        rows[index]["provenance"]["source_id"]
        for index, is_unchanged in enumerate(unchanged)
        if not is_unchanged
    } == {"creation-12125", "creation-12164"}
    assert {path.name for path in case["output"].iterdir()} == {
        "manifest.jsonl",
        "source-selection.json",
        "corrections.json",
        "SOURCE.json",
    }
    corrections = json.loads((case["output"] / "corrections.json").read_text())
    assert [record["source_id"] for record in corrections["corrections"]] == [
        "creation-12125",
        "creation-12164",
    ]
    assert corrections["corrections"][0]["old"] == [
        "17",
        "",
        "18",
        "",
        "22",
        "",
        "34",
        "",
    ]
    assert corrections["corrections"][0]["new"] == ["17", "18", "22", "34"]
    assert summary["dataset_root"] == "data"
    assert summary["dataset_profile"] == "ilvr-test400-corrected-v1"
    assert summary["paper_protocol_verified"] is False
    assert summary["ood_verified"] is False


def test_prepared_manifest_is_accepted_by_shared_reader(monkeypatch):
    from comt_source_fixtures import make_comt_source
    from evaluation.io import parse_config, read_manifest

    case = make_comt_source()
    summary = _prepare(monkeypatch, case)
    config = parse_config(
        {
            "schema_version": 1,
            "benchmark": "comt",
            "profile": "comt_reference_rules_v1",
            "categories": ["creation", "deletion", "update", "selection"],
            "dataset_root": "data",
            "run_metadata": {
                "model_identity": "synthetic:model",
                "prompt_template_identity": "synthetic:prompt",
                "generation_config": {},
            },
        },
        case["root"],
    )
    prepared = read_manifest(case["output"] / "manifest.jsonl", config, case["root"])
    assert len(prepared.samples) == summary["N_expected"] == 400
    assert prepared.samples[0].model_input.images[0].startswith("images_comt/")


def test_integer_choices_are_stringified_without_rewriting_legacy_text(monkeypatch):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    source = case["source_rows"][11]
    source["option"] = [10, 20, 30, 40]
    case["selection_rows"][11]["text_input"] = (
        source["question"].replace("[IMAGE0]", "").strip()
        + " option:"
        + str(source["option"])
    )
    case["selection_rows"][11]["original_final_answer"] = str(
        source["option"]["ABCD".index(source["answer"])]
    )
    _rewrite_source(case)
    _rewrite_selection(case)
    _prepare(monkeypatch, case)
    rows = [
        json.loads(line)
        for line in (case["output"] / "manifest.jsonl").read_text().splitlines()
    ]
    assert rows[11]["model_input"]["text"] == case["selection_rows"][11]["text_input"]
    assert rows[11]["model_input"]["choices"] == ["10", "20", "30", "40"]


@pytest.mark.parametrize("field", ["id", "sha"])
def test_unknown_source_identity_rejects_without_output(monkeypatch, field):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    metadata_path = case["source_root"] / "dataset-info.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = "czh-up/comt" if field == "id" else "0" * 40
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ContractError, match="identity|revision"):
        _prepare(monkeypatch, case)
    assert not case["output"].exists()


@pytest.mark.parametrize("kind", ["source", "selection"])
def test_pinned_hash_mismatch_rejects_without_output(monkeypatch, kind):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    case[f"{kind}_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="SHA-256"):
        _prepare(monkeypatch, case)
    assert not case["output"].exists()


@pytest.mark.parametrize("kind", ["source_id", "image_token"])
def test_source_identity_or_image_token_ambiguity_rejects_after_hash_check(
    monkeypatch, kind
):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    if kind == "source_id":
        case["source_rows"][1]["id"] = case["source_rows"][0]["id"]
    else:
        case["source_rows"][1]["type"] = case["source_rows"][0]["type"]
        case["source_rows"][1]["image"]["IMAGE0"] = case["source_rows"][0][
            "image"
        ]["IMAGE0"]
    _rewrite_source(case)
    with pytest.raises(ContractError, match="duplicate|ambiguous"):
        _prepare(monkeypatch, case)
    assert not case["output"].exists()


@pytest.mark.parametrize("kind", ["count", "duplicate", "missing"])
def test_selection_population_and_join_fail_closed_after_hash_check(monkeypatch, kind):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    if kind == "count":
        case["selection_rows"].pop()
    elif kind == "duplicate":
        case["selection_rows"][-1] = dict(case["selection_rows"][0])
    else:
        case["selection_rows"][0]["image_input"] = [
            "images_comt/creation/missing.png"
        ]
    _rewrite_selection(case)
    with pytest.raises(ContractError, match="400|duplicate|source"):
        _prepare(monkeypatch, case)
    assert not case["output"].exists()


@pytest.mark.parametrize("kind", ["text", "gold"])
def test_selection_text_and_legacy_gold_are_checked_after_join(monkeypatch, kind):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    field = "text_input" if kind == "text" else "original_final_answer"
    case["selection_rows"][11][field] = "MISMATCH"
    _rewrite_selection(case)
    with pytest.raises(ContractError, match="text|gold"):
        _prepare(monkeypatch, case)
    assert not case["output"].exists()


@pytest.mark.parametrize("kind", ["old_array", "label"])
def test_repair_requires_exact_pinned_old_row(monkeypatch, kind):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    repair = next(row for row in case["source_rows"] if row["id"] == "creation-12125")
    if kind == "old_array":
        repair["option"][0] = "altered"
        case["selection_rows"][57]["text_input"] = (
            repair["question"].replace("[IMAGE0]", "").strip()
            + " option:"
            + str(repair["option"])
        )
    else:
        repair["answer"] = "C"
        case["selection_rows"][57]["original_final_answer"] = "18"
    _rewrite_source(case)
    _rewrite_selection(case)
    with pytest.raises(ContractError, match="correction"):
        _prepare(monkeypatch, case)
    assert not case["output"].exists()


def test_unknown_source_category_rejects_after_hash_check(monkeypatch):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    extra = dict(case["source_rows"][0])
    extra.update(id="foreign-1", type="foreign", image={"IMAGE0": "foreign-1"})
    case["source_rows"].append(extra)
    _rewrite_source(case)
    with pytest.raises(ContractError, match="category"):
        _prepare(monkeypatch, case)


@pytest.mark.parametrize("kind", ["float_option", "helper_placeholder"])
def test_selected_source_rejects_noncanonical_options_or_helper_placeholder(
    monkeypatch, kind
):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    source = case["source_rows"][11]
    if kind == "float_option":
        source["option"][0] = 1.5
    else:
        source["question"] += " [IMAGE1]"
    case["selection_rows"][11]["text_input"] = (
        source["question"].replace("[IMAGE0]", "").strip()
        + " option:"
        + str(source["option"])
    )
    _rewrite_source(case)
    _rewrite_selection(case)
    with pytest.raises(ContractError, match="option|helper|IMAGE"):
        _prepare(monkeypatch, case)
    assert not case["output"].exists()


@pytest.mark.parametrize("kind", ["extra", "foreign", "traversal"])
def test_selection_rejects_extra_or_unsafe_image_paths(monkeypatch, kind):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    if kind == "extra":
        case["selection_rows"][0]["image_input"].append(
            "images_comt/creation/helper.png"
        )
    elif kind == "foreign":
        case["selection_rows"][0]["image_input"] = [
            "images_comt/deletion/10000.png"
        ]
    else:
        case["selection_rows"][0]["image_input"] = [
            "images_comt/creation/../creation/10000.png"
        ]
    _rewrite_selection(case)
    with pytest.raises(ContractError, match="image|category|path"):
        _prepare(monkeypatch, case)
    assert not case["output"].exists()


@pytest.mark.parametrize("argument", ["source_root", "selection_manifest", "image_root", "output"])
def test_every_path_must_be_inside_repository_data(monkeypatch, argument):
    from comt_source_fixtures import make_comt_source, pin_fixture
    from evaluation.comt_prepare import prepare_comt

    case = make_comt_source()
    pin_fixture(monkeypatch, case)
    values = {
        "source_root": case["source_root"],
        "selection_manifest": case["selection_manifest"],
        "image_root": case["image_root"],
        "output": case["output"],
    }
    values[argument] = case["root"] / f"outside-{argument}"
    with pytest.raises(ContractError, match="data"):
        prepare_comt(**values, repo_root=case["root"])
    assert not case["output"].exists()


@pytest.mark.parametrize("kind", ["metadata", "source_jsonl"])
def test_source_files_must_not_escape_source_root_through_symlinks(monkeypatch, kind):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    relative = "dataset-info.json" if kind == "metadata" else "comt/data.jsonl"
    source_file = case["source_root"] / relative
    outside = case["root"].parent / ("outside-" + Path(relative).name)
    outside.write_bytes(source_file.read_bytes())
    source_file.unlink()
    source_file.symlink_to(outside)
    with pytest.raises(ContractError, match="source root"):
        _prepare(monkeypatch, case)
    assert not case["output"].exists()


@pytest.mark.parametrize("kind", ["absent", "corrupt", "ambiguous", "non_png"])
def test_selected_image_must_be_one_decodable_file(monkeypatch, kind):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    image = case["image_root"] / "creation/10000.png"
    if kind == "absent":
        image.unlink()
    elif kind == "corrupt":
        image.write_bytes(b"not a PNG")
    else:
        alternate = case["image_root"] / "creation/10000.jpg"
        alternate.write_bytes(image.read_bytes())
        if kind == "non_png":
            image.unlink()
            case["selection_rows"][0]["image_input"] = [
                "images_comt/creation/10000.jpg"
            ]
            _rewrite_selection(case)
    with pytest.raises(ContractError, match="image"):
        _prepare(monkeypatch, case)
    assert not case["output"].exists()


def test_existing_destination_rejects_without_overwrite(monkeypatch):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    case["output"].mkdir(parents=True)
    sentinel = case["output"] / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ContractError, match="exists"):
        _prepare(monkeypatch, case)
    assert sentinel.read_text() == "keep"


def test_unselected_malformed_options_do_not_become_samples(monkeypatch):
    from comt_source_fixtures import make_comt_source

    case = make_comt_source()
    extra = dict(case["source_rows"][0])
    extra.update(
        id="creation-unselected-malformed",
        option=["only", "two"],
        image={"IMAGE0": "unselected-malformed"},
    )
    case["source_rows"].append(extra)
    _rewrite_source(case)
    summary = _prepare(monkeypatch, case)
    assert summary["N_expected"] == 400
    assert "unselected-malformed" not in (case["output"] / "manifest.jsonl").read_text()


@pytest.mark.parametrize("kind", ["source", "selection"])
def test_strict_json_rejects_duplicate_keys_or_nonfinite_values(monkeypatch, kind):
    from comt_source_fixtures import make_comt_source, pin_fixture

    case = make_comt_source()
    if kind == "source":
        path = case["source_root"] / "comt/data.jsonl"
        path.write_text('{"id":"one","id":"two"}\n', encoding="utf-8")
        case["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        path = case["selection_manifest"]
        path.write_text('{"text_input":NaN}\n', encoding="utf-8")
        case["selection_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    pin_fixture(monkeypatch, case)
    with pytest.raises(ContractError, match="JSON|duplicate|400"):
        _prepare(monkeypatch, case)
    assert not case["output"].exists()
