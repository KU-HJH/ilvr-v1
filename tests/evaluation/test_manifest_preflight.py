import copy
import hashlib

import pytest

import evaluation.io as io
from evaluation.schema import ContractError
from offline_fixtures import make_case, save_rows


def _config(case):
    return io.parse_config(case["config"], case["root"])


def test_manifest_preflight_does_not_need_predictions():
    case = make_case(1)
    case["prediction_path"].unlink()

    assert hasattr(io, "read_manifest"), "manifest-only preflight is missing"

    prepared = io.read_manifest(case["manifest"], _config(case), case["root"])
    assert [sample.id for sample in prepared.samples] == ["vsp/planning/s0"]
    assert prepared.predictions == {}
    assert prepared.manifest_sha256 == hashlib.sha256(
        case["manifest"].read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("kind", "expected_error"),
    [
        ("malformed_map", "map must be a nonempty rectangle"),
        ("mismatched_id", "sample id does not match provenance identity"),
        ("duplicate_id", "duplicate sample id"),
        ("empty_manifest", "manifest must contain at least one row"),
    ],
)
def test_manifest_preflight_rejects_invalid_manifest_contract(kind, expected_error):
    case = make_case(1)
    rows = copy.deepcopy(case["samples"])
    if kind == "malformed_map":
        rows[0]["target"]["map_desc"] = [[1, 2], [0]]
    elif kind == "mismatched_id":
        rows[0]["id"] = "vsp/planning/wrong"
    elif kind == "duplicate_id":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows = []
    save_rows(case["manifest"], rows)

    with pytest.raises(ContractError, match=expected_error):
        io.read_manifest(case["manifest"], _config(case), case["root"])


def test_manifest_preflight_rejects_manifest_outside_data():
    case = make_case(1)
    outside_manifest = case["result_root"] / "outside-manifest.jsonl"
    outside_manifest.write_bytes(case["manifest"].read_bytes())

    with pytest.raises(
        ContractError, match="manifest path must resolve inside repo_root/data"
    ):
        io.read_manifest(outside_manifest, _config(case), case["root"])


def test_manifest_preflight_rejects_changed_image_bytes():
    case = make_case(1)
    asset = case["data_root"] / "image.bin"
    asset.write_bytes(b"original fixture image")
    row = case["samples"][0]
    row["model_input"]["images"] = ["image.bin"]
    row["provenance"]["asset_sha256"] = {
        "image.bin": hashlib.sha256(b"original fixture image").hexdigest()
    }
    save_rows(case["manifest"], case["samples"])
    asset.write_bytes(b"changed fixture image")

    with pytest.raises(ContractError, match="asset SHA-256 mismatch"):
        io.read_manifest(case["manifest"], _config(case), case["root"])


def test_manifest_preflight_matches_prediction_reader_contracts():
    case = make_case(2)
    config = _config(case)

    manifest_only = io.read_manifest(case["manifest"], config, case["root"])
    with_predictions = io.read_inputs(
        case["manifest"], case["prediction_path"], config, case["root"]
    )

    assert manifest_only.samples == with_predictions.samples
    assert manifest_only.manifest_sha256 == with_predictions.manifest_sha256
    assert manifest_only.run_fingerprint == with_predictions.run_fingerprint
    assert manifest_only.predictions == {}
    assert set(with_predictions.predictions) == {
        "vsp/planning/s0",
        "vsp/planning/s1",
    }
