import copy
import hashlib
import json

import pytest

from emma_source_fixtures import make_source, pin_source, rewrite_all
from evaluation.schema import ContractError


def prepare(case):
    from evaluation.emma_prepare import prepare_emma
    return prepare_emma(case["source"], case["output"], repo_root=case["root"])


def test_preparation_preserves_raw_ids_and_slots(monkeypatch):
    from evaluation.io import parse_config, read_manifest
    case = make_source(); pin_source(monkeypatch, case)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in case["source"].rglob("*") if p.is_file()}
    summary = prepare(case)
    assert summary["N_expected"] == 400
    assert summary["categories"] == {subject: 100 for subject in ("Chemistry", "Coding", "Math", "Physics")}
    assert summary["question_types"] == {"multiple_choice": 341, "open_ended": 59}
    assert summary["image_count"] == 614
    rows = [json.loads(line) for line in (case["output"] / "manifest.jsonl").read_text().splitlines()]
    by_id = {row["id"]: row for row in rows}
    assert by_id["phy_6"]["provenance"]["source_id"] == "phy_6"
    assert by_id["phy_6"]["model_input"]["images"] == ["images/phy_6/image_2.png", "images/phy_6/image_3.png"]
    assert "<image_2><image_3>" in by_id["phy_6"]["model_input"]["text"]
    assert by_id["che_0"]["target"]["content"] == "x" * 513
    assert by_id["che_78"]["target"]["question_type"] == "open_ended"
    assert {row["target"]["question_type"] for row in rows} == {"multiple_choice", "open_ended"}
    selection_rows = json.loads((case["output"] / "source-selection.json").read_text())["rows"]
    assert next(row for row in selection_rows if row["source_id"] == "che_78")["type"] == "open-ended"
    assert summary["excluded_rows"] == 0
    assert summary["paper_protocol_verified"] is False
    source_record = json.loads((case["output"] / "SOURCE.json").read_text())
    assert source_record["schema_version"] == 1 and source_record["manifest_sha256"] == summary["manifest_sha256"]
    selection = json.loads((case["output"] / "source-selection.json").read_text())
    assert set(selection["rows"][0]) == {"source_id", "subject", "source_row", "question", "context",
        "options", "answer", "type", "category", "task", "source", "solution", "image_slots"}
    config = parse_config({"schema_version": 1, "benchmark": "emma", "profile": "emma_safe_rules_v1",
        "categories": ["Chemistry", "Coding", "Math", "Physics"], "dataset_root": summary["dataset_root"],
        "run_metadata": {"model_identity": "synthetic:emma", "prompt_template_identity": "synthetic:question", "generation_config": {}}}, case["root"])
    assert len(read_manifest(case["output"] / "manifest.jsonl", config, case["root"]).samples) == 400
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in case["source"].rglob("*") if p.is_file()}


@pytest.mark.parametrize("defect", ["metadata_id", "revision", "hash", "size", "duplicate", "unsafe_pid",
    "subject", "type", "population", "missing_field", "extra_field", "empty_question", "category", "task",
    "answer", "empty_answer", "choices_one", "choices_nine", "open_options", "missing_slot", "missing_image", "image_bytes",
    "corrupt_image", "zero_images", "source_escape", "output_escape"])
def test_invalid_source_fails_before_output_and_preserves_source(monkeypatch, defect):
    case = make_source()
    metadata = case["source"] / "EMMA-mini-dataset-info.json"
    if defect in ("metadata_id", "revision"):
        value = json.loads(metadata.read_text()); value["id" if defect == "metadata_id" else "sha"] = "wrong"; metadata.write_text(json.dumps(value))
    elif defect == "hash":
        pin_source(monkeypatch, case); case["rows"]["Coding"][0]["question"] += " changed"; rewrite_all(case)
    elif defect == "size":
        from evaluation import emma_prepare
        pin_source(monkeypatch, case); facts = copy.deepcopy(emma_prepare.PARQUET_FILES); facts["Coding"]["bytes"] += 1; monkeypatch.setattr(emma_prepare, "PARQUET_FILES", facts)
    elif defect == "source_escape": case["source"] = case["root"] / "results"
    elif defect == "output_escape": case["output"] = case["root"] / "results" / case["run_id"]
    else:
        row = case["rows"]["Chemistry"][0]
        if defect == "duplicate": case["rows"]["Coding"][0]["pid"] = row["pid"]
        elif defect == "unsafe_pid": row["pid"] = "../bad"
        elif defect == "subject": row["subject"] = "Biology"
        elif defect == "type": row["type"] = "essay"
        elif defect == "population": case["rows"]["Chemistry"].pop()
        elif defect == "missing_field": del row["task"]
        elif defect == "extra_field": row["gold"] = "secret"
        elif defect == "empty_question": row["question"] = " "
        elif defect == "category":
            for item in case["rows"]["Chemistry"]: item["category"] = 3
        elif defect == "task":
            for item in case["rows"]["Chemistry"]: item["task"] = 3
        elif defect == "answer": row["answer"] = None
        elif defect == "empty_answer": row["answer"] = ""
        elif defect == "choices_one": row["options"] = ["only"]
        elif defect == "choices_nine": row["options"] = [str(i) for i in range(9)]
        elif defect == "open_options": row = case["rows"]["Chemistry"][-1]; row["options"] = ["bad", "worse"]
        elif defect == "missing_slot":
            row = case["rows"]["Chemistry"][99]; row["question"] += " <image_5>"
        elif defect == "missing_image": row["image_1"] = None
        elif defect == "image_bytes": row["image_1"] = {"bytes": None, "path": "ignored"}
        elif defect == "corrupt_image": row["image_1"] = {"bytes": b"broken", "path": "ignored"}
        elif defect == "zero_images":
            for rows in case["rows"].values():
                for item in rows:
                    for slot in range(1, 6): item[f"image_{slot}"] = None
                    item["question"] = item["question"].split(" <image_")[0]
        rewrite_all(case); pin_source(monkeypatch, case)
    before = {p: p.read_bytes() for p in case["source"].rglob("*") if p.is_file()} if case["source"].is_dir() else {}
    with pytest.raises(ContractError): prepare(case)
    assert not case["output"].exists()
    assert before == {p: p.read_bytes() for p in case["source"].rglob("*") if p.is_file()} if case["source"].is_dir() else True


def test_existing_output_is_rejected(monkeypatch):
    case = make_source(); pin_source(monkeypatch, case); case["output"].mkdir()
    with pytest.raises(ContractError): prepare(case)


def test_null_solution_is_preserved_only_in_source_selection(monkeypatch):
    case = make_source()
    case["rows"]["Math"][54]["solution"] = None
    rewrite_all(case); pin_source(monkeypatch, case)

    prepare(case)

    selection = json.loads((case["output"] / "source-selection.json").read_text())["rows"]
    assert next(row for row in selection if row["source_id"] == "mat_54")["solution"] is None
    manifest = [json.loads(line) for line in (case["output"] / "manifest.jsonl").read_text().splitlines()]
    prepared = next(row for row in manifest if row["id"] == "mat_54")
    assert set(prepared["model_input"]) == {"text", "images", "choices"}
    assert "solution" not in prepared["model_input"]
