import hashlib
import json
from pathlib import Path

import pytest

from evaluation.schema import ContractError
from tests.evaluation.zebra_source_fixtures import (
    BROKEN_JPEG,
    SourceFactory,
    zebra_source_factory,
    write_source,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load(source):
    return (json.loads((source / "selected-ids.json").read_text()),
            [json.loads(line) for line in (source / "source-records.jsonl").read_text().splitlines()])


def _pin(monkeypatch, module, source):
    monkeypatch.setattr(module, "SELECTION_SHA256", hashlib.sha256((source / "selected-ids.json").read_bytes()).hexdigest())
    monkeypatch.setattr(module, "RECORDS_SHA256", hashlib.sha256((source / "source-records.jsonl").read_bytes()).hexdigest())


def test_prepare_source_preserves_gold_separation(repo_root, monkeypatch, zebra_source_factory):
    from evaluation import zebra_2d_prepare as module
    source = zebra_source_factory()
    _pin(monkeypatch, module, source)
    output = repo_root / "data/evaluation" / (source.name + "-prepared")
    zebra_source_factory.track(output)
    summary = module.prepare_zebra_2d(source, output, repo_root=repo_root)
    assert summary["N_expected"] == 400
    assert summary["categories"] == {"Jigsaw": 200, "Search": 200}
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    original = [json.loads(line) for line in (source / "source-records.jsonl").read_text().splitlines()]
    by_id = {f"zebra_2d/{raw['task']}/{raw['global_row']:05d}": raw for raw in original}
    assert {row["id"] for row in rows} == set(by_id)
    for row in rows:
        raw = by_id[row["id"]]
        assert row["model_input"]["text"] == raw["Question"]
        assert row["model_input"]["choices"] == []
        assert row["target"] == {"answer": raw["Final Answer"]}
        assert set(row["model_input"]) == {"text", "images", "choices"}
    assert summary["image_count"] == 400 and summary["excluded_rows"] == 0
    assert hashlib.sha256((output / "manifest.jsonl").read_bytes()).hexdigest() == summary["manifest_sha256"]


@pytest.mark.parametrize("defect", ["record_hash", "selection_hash", "mismatch", "task", "config", "split",
    "bool_row", "negative_row", "target", "missing_image", "corrupt_image", "truncated_pixels", "traversal", "absolute_inside_source", "symlink",
    "duplicate", "population", "record_keys", "selection_keys", "image_keys", "bad_image_metadata"])
def test_invalid_source_is_rejected_before_output(repo_root, monkeypatch, defect, zebra_source_factory):
    from evaluation import zebra_2d_prepare as module
    source = zebra_source_factory()
    selections, records = _load(source)
    output = repo_root / "data/evaluation" / (source.name + "-prepared")
    zebra_source_factory.track(output)
    if defect == "record_hash":
        _pin(monkeypatch, module, source); records[0]["Question"] += " changed"; write_source(source, selections, records)
    elif defect == "selection_hash":
        _pin(monkeypatch, module, source); selections[0]["selection_hash"] = "0" * 64; write_source(source, selections, records)
    else:
        row = records[0]
        if defect == "mismatch": selections[0]["shard_row"] += 1
        elif defect == "task": row["task"] = "Other"
        elif defect == "config": row["config"] = "wrong"
        elif defect == "split": row["split"] = "test"
        elif defect == "bool_row": row["global_row"] = True
        elif defect == "negative_row": row["shard_row"] = -1
        elif defect == "target": row["Final Answer"] = "E"
        elif defect == "missing_image": (source / row["problem_image_1"]["path"]).unlink()
        elif defect == "corrupt_image": (source / row["problem_image_1"]["path"]).write_bytes(b"bad")
        elif defect == "truncated_pixels":
            image = source / row["problem_image_1"]["path"]
            image.write_bytes(BROKEN_JPEG)
            row["problem_image_1"]["sha256"] = hashlib.sha256(BROKEN_JPEG).hexdigest()
            row["problem_image_1"]["bytes"] = len(BROKEN_JPEG)
        elif defect == "traversal": row["problem_image_1"]["path"] = "../outside.jpg"
        elif defect == "absolute_inside_source": row["problem_image_1"]["path"] = str((source / row["problem_image_1"]["path"]).resolve())
        elif defect == "symlink":
            image = source / row["problem_image_1"]["path"]; image.unlink(); image.symlink_to("/etc/hosts")
        elif defect == "duplicate": records[1] = dict(row)
        elif defect == "population": records.pop()
        elif defect == "record_keys": row["extra"] = 1
        elif defect == "selection_keys": selections[0]["extra"] = 1
        elif defect == "image_keys": row["problem_image_1"]["extra"] = 1
        elif defect == "bad_image_metadata": row["problem_image_1"]["bytes"] = True
        write_source(source, selections, records); _pin(monkeypatch, module, source)
    with pytest.raises(ContractError):
        module.prepare_zebra_2d(source, output, repo_root=repo_root)
    assert not output.exists()


def test_source_and_output_containment_and_existing_output(repo_root, monkeypatch, zebra_source_factory):
    from evaluation import zebra_2d_prepare as module
    source = zebra_source_factory()
    _pin(monkeypatch, module, source)
    contained_output = source.parent / (source.name + "-contained")
    zebra_source_factory.track(contained_output)
    with pytest.raises(ContractError): module.prepare_zebra_2d(repo_root / "results", contained_output, repo_root=repo_root)
    with pytest.raises(ContractError): module.prepare_zebra_2d(source, repo_root / "results/x", repo_root=repo_root)
    output = source.parent / (source.name + "-existing")
    zebra_source_factory.track(output)
    output.mkdir()
    with pytest.raises(ContractError): module.prepare_zebra_2d(source, output, repo_root=repo_root)


def test_source_factory_rejects_unowned_cleanup_path(repo_root):
    factory = SourceFactory(repo_root)
    foreign = repo_root / "data/evaluation" / "x"
    with pytest.raises(ValueError, match="not owned"):
        factory.track(foreign)
