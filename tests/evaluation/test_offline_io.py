import copy
import hashlib
import json

import pytest

from offline_fixtures import make_case, save_manifest, save_rows
import evaluation.io as evaluation_io
from evaluation.schema import ContractError
from evaluation.io import parse_config, parse_prediction, parse_sample, read_inputs


def load(case):
    config = parse_config(case["config"], case["root"])
    return read_inputs(
        case["manifest"], case["prediction_path"], config, case["root"]
    )


def test_map_only_gold_and_partial_population():
    case = make_case()
    save_rows(case["prediction_path"], case["predictions"][:1])
    prepared = load(case)
    assert len(prepared.samples) == 2 and len(prepared.predictions) == 1
    save_rows(case["prediction_path"], [])
    assert load(case).predictions == {}


@pytest.mark.parametrize("kind", ["duplicate", "unknown", "manifest", "run"])
def test_bad_prediction_join_fails_closed(kind):
    case = make_case()
    rows = copy.deepcopy(case["predictions"])
    if kind == "duplicate":
        rows.append(rows[0])
    if kind == "unknown":
        rows[0]["id"] = "vsp/planning/not-in-manifest"
    if kind == "manifest":
        rows[0]["manifest_sha256"] = "0" * 64
    if kind == "run":
        rows[0]["run_fingerprint"] = "0" * 64
    save_rows(case["prediction_path"], rows)
    with pytest.raises(ContractError):
        load(case)


@pytest.mark.parametrize(
    "field", ["answer", "Text Reasoning Trace", "reasoning_image_1"]
)
def test_gold_or_trace_cannot_enter_model_input(field):
    case = make_case(1)
    row = case["samples"][0]
    row["model_input"][field] = "leak"
    with pytest.raises(ContractError):
        parse_sample(row)


def test_infrastructure_status_requires_error_and_preserves_raw_text():
    case = make_case(1)
    row = case["predictions"][0] | dict(
        status="infrastructure_error", error="worker failed", response="R"
    )
    assert parse_prediction(row).response == "R"
    row["error"] = ""
    with pytest.raises(ContractError):
        parse_prediction(row)


@pytest.mark.parametrize(
    "kind", ["version", "target", "id", "duplicate", "limit", "category"]
)
def test_invalid_manifest_contract(kind):
    case = make_case(1)
    row = case["samples"][0]
    if kind == "version":
        row["schema_version"] = True
    if kind == "target":
        del row["target"]
    if kind == "id":
        row["id"] = "wrong"
    if kind == "duplicate":
        case["samples"].append(copy.deepcopy(row))
    if kind == "limit":
        row["target"]["max_steps"] = 99
    if kind == "category":
        case["config"]["categories"] = ["other"]
    save_manifest(case)
    with pytest.raises(ContractError):
        load(case)


@pytest.mark.parametrize(
    "payload", [b"", b"[]\n", b"null\n", b'{"x":NaN}\n', b'{"x":1,"x":2}\n', b"{"]
)
def test_bad_manifest_json_or_empty_manifest(payload):
    case = make_case(1)
    case["manifest"].write_bytes(payload)
    with pytest.raises(ContractError):
        load(case)


@pytest.mark.parametrize("separator", ["\u2028", "\u0085"])
def test_jsonl_preserves_unicode_line_characters_inside_strings(separator):
    row = {"text": f"before{separator}after"}
    payload = (json.dumps(row, ensure_ascii=False) + "\r\n").encode("utf-8")

    assert evaluation_io.read_jsonl_bytes(payload) == [row]


@pytest.mark.parametrize("key", ["manifest", "prediction_path"])
def test_missing_input_file(key):
    case = make_case(1)
    case[key].unlink()
    with pytest.raises(ContractError):
        load(case)


@pytest.mark.parametrize(
    "kind",
    ["good", "contained_parent", "missing", "hash", "absolute", "parent", "symlink"],
)
def test_asset_containment_and_hash(kind):
    case = make_case(1)
    # All asset bytes stay under data/, including the out-of-dataset probe.
    asset = case["data_root"] / "image.bin"
    asset.write_bytes(b"fixture-image")
    outside = case["data_root"].parent / (case["run_id"] + "-outside.bin")
    outside.write_bytes(b"fixture-image")
    name = "image.bin"
    if kind == "contained_parent":
        name = f"../{case['data_root'].name}/image.bin"
    if kind == "missing":
        name = "missing.bin"
    if kind == "absolute":
        name = str(asset)
    if kind == "parent":
        name = "../" + outside.name
    if kind == "symlink":
        (case["data_root"] / "escape.bin").symlink_to(outside)
        name = "escape.bin"
    row = case["samples"][0]
    row["model_input"]["images"] = [name]
    row["provenance"]["asset_sha256"] = {
        name: (
            "0" * 64
            if kind == "hash"
            else hashlib.sha256(asset.read_bytes()).hexdigest()
        )
    }
    save_manifest(case)
    if kind in ("good", "contained_parent"):
        assert len(load(case).samples) == 1
    else:
        with pytest.raises(ContractError):
            load(case)


@pytest.mark.parametrize(
    "kind", ["root", "identity", "unknown_field", "generation_nan"]
)
def test_config_boundary(kind):
    case = make_case(1)
    if kind == "root":
        case["config"]["dataset_root"] = str(case["result_root"])
    if kind == "identity":
        del case["config"]["run_metadata"]["model_identity"]
    if kind == "unknown_field":
        case["config"]["fallback"] = True
    if kind == "generation_nan":
        case["config"]["run_metadata"]["generation_config"][
            "temperature"
        ] = float("nan")
    with pytest.raises(ContractError):
        load(case)


def test_sample_prompt_identity_does_not_change_run_identity():
    case = make_case(2)
    for index, row in enumerate(case["predictions"]):
        row["prompt_identity"] = f"prompt-{index}"
    save_rows(case["prediction_path"], case["predictions"])
    assert len(load(case).predictions) == 2


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_dataset_root_must_be_an_existing_directory(kind):
    case = make_case(1)
    candidate = case["data_root"] / "invalid-dataset-root"
    if kind == "file":
        candidate.write_bytes(b"not-a-directory")
    case["config"]["dataset_root"] = str(candidate)
    with pytest.raises(ContractError):
        parse_config(case["config"], case["root"])


@pytest.mark.parametrize(
    "kind",
    ["manifest_direct", "manifest_symlink", "prediction_direct", "prediction_symlink"],
)
def test_input_paths_reject_external_or_symlink_escape_before_read(kind, monkeypatch):
    case = make_case(1)
    manifest_path = case["manifest"]
    predictions_path = case["prediction_path"]
    rejected_kind = "manifest" if kind.startswith("manifest") else "predictions"
    if kind == "manifest_direct":
        manifest_path = case["prediction_path"]
    if kind == "manifest_symlink":
        manifest_path = case["data_root"] / "manifest-escape.jsonl"
        manifest_path.symlink_to(case["prediction_path"])
    if kind == "prediction_direct":
        predictions_path = case["manifest"]
    if kind == "prediction_symlink":
        predictions_path = case["result_root"] / "predictions-escape.jsonl"
        predictions_path.symlink_to(case["manifest"])

    original_read = evaluation_io._read_bytes

    def reject_offending_read(path, input_kind):
        if input_kind == rejected_kind:
            raise AssertionError("escaped input was read before containment validation")
        return original_read(path, input_kind)

    monkeypatch.setattr(evaluation_io, "_read_bytes", reject_offending_read)
    config = parse_config(case["config"], case["root"])
    with pytest.raises(ContractError):
        read_inputs(manifest_path, predictions_path, config, case["root"])
