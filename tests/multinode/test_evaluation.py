from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.multinode_evaluation import (
    iter_owned_samples,
    merge_shards,
    write_shard_manifest,
)


def test_four_shards_cover_every_source_index_once():
    samples = [{"id": i} for i in range(11)]
    owned = [
        [index for index, _ in iter_owned_samples(samples, 4, shard)]
        for shard in range(4)
    ]
    assert owned == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7]]
    assert sorted(index for shard in owned for index in shard) == list(range(11))


@pytest.mark.parametrize(
    ("num_shards", "shard_index"),
    [(1, 0), (2, 0), (4, -1), (4, 4), (True, 0), (4, True)],
)
def test_owned_samples_rejects_every_non_four_worker_topology(
    num_shards, shard_index
):
    with pytest.raises(
        ValueError, match="multi-node evaluation requires four shards indexed 0-3"
    ):
        list(iter_owned_samples([{}], num_shards, shard_index))


def _write_fake_shards(
    tmp_path: Path,
    *,
    indices=((0,), (1,), (2,), (3,)),
    dataset_sizes=(10, 10, 10, 10),
    model_ids=("model-a", "model-a", "model-a", "model-a"),
    raw_rows: dict[int, str] | None = None,
    row_overrides: dict[int, dict] | None = None,
    successful=(True, True, True, True),
    processed_indices=None,
):
    manifests = []
    raw_rows = raw_rows or {}
    row_overrides = row_overrides or {}
    if processed_indices is None:
        processed_indices = indices
    for shard_index in range(4):
        shard_jsonl = tmp_path / f"shard-{shard_index}.jsonl"
        if shard_index in raw_rows:
            shard_jsonl.write_text(raw_rows[shard_index], encoding="utf-8")
        else:
            rows = []
            for source_index in indices[shard_index]:
                rows.append(
                    row_overrides.get(
                        source_index,
                        {
                            "index": source_index,
                            "task_name": "task",
                            "match": source_index % 2 == 0,
                            "inference_time_sec": 0.5,
                        },
                    )
                )
            shard_jsonl.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
        manifest = tmp_path / f"shard-{shard_index}.manifest.json"
        write_shard_manifest(
            manifest,
            shard_jsonl=shard_jsonl,
            dataset_identity={
                "path": "/logical/data/test.jsonl",
                "size": dataset_sizes[shard_index],
                "mtime_ns": 123456789,
            },
            model_identity=model_ids[shard_index],
            shard_index=shard_index,
            num_shards=4,
            expected_source_indices=list(indices[shard_index]),
            processed_source_indices=list(processed_indices[shard_index]),
            worker_exit_status=0 if successful[shard_index] else 1,
        )
        manifests.append(manifest)
    return manifests


def _merge(tmp_path: Path, shards, expected_sample_count: int):
    return merge_shards(
        shards,
        expected_sample_count=expected_sample_count,
        output_jsonl=tmp_path / "merged.jsonl",
        summary_json=tmp_path / "summary.json",
    )


def test_merge_sorts_rows_and_computes_accuracy_once(tmp_path):
    shards = _write_fake_shards(
        tmp_path,
        indices=((0, 4), (1, 5), (2, 6), (3,)),
    )

    summary = _merge(tmp_path, shards, expected_sample_count=7)

    rows = [
        json.loads(line)
        for line in (tmp_path / "merged.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["index"] for row in rows] == list(range(7))
    assert all("summary" not in row for row in rows)
    assert summary == {
        "summary": True,
        "num_samples": 7,
        "average_inference_time_sec": 0.5,
        "total_inference_time_sec": 3.5,
        "accuracy": 4 / 7,
        "success": 4,
    }
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_merge_rejects_duplicate_source_index(tmp_path):
    shards = _write_fake_shards(
        tmp_path, indices=((0, 4), (1, 5), (2, 6), (3, 4))
    )
    with pytest.raises(ValueError, match="duplicate source index 4"):
        _merge(tmp_path, shards, expected_sample_count=7)


@pytest.mark.parametrize(
    ("kwargs", "expected_sample_count", "message"),
    [
        ({"indices": ((0,), (1,), (2,), ())}, 4, "missing source index 3"),
        ({"dataset_sizes": (10, 10, 11, 10)}, 4, "dataset identity"),
        ({"model_ids": ("a", "a", "b", "a")}, 4, "model identity"),
        ({"raw_rows": {0: "{not-json}\n"}}, 4, "invalid JSON"),
        (
            {
                "row_overrides": {
                    0: {"index": 0, "task_name": "task", "error": "OOM"}
                }
            },
            4,
            "error row",
        ),
        ({"indices": ((0, 4), (1,), (2,), (3,))}, 4, "out of range"),
        ({"successful": (True, False, True, True)}, 4, "unsuccessful worker"),
        (
            {"processed_indices": ((0,), (), (2,), (3,))},
            4,
            "processed indices",
        ),
    ],
)
def test_merge_rejects_invalid_worker_output(
    tmp_path, kwargs, expected_sample_count, message
):
    shards = _write_fake_shards(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=message):
        _merge(tmp_path, shards, expected_sample_count)
    assert not (tmp_path / "merged.jsonl").exists()
    assert not (tmp_path / "summary.json").exists()


def test_merge_requires_exactly_four_unique_shard_manifests(tmp_path):
    shards = _write_fake_shards(tmp_path)
    with pytest.raises(ValueError, match="exactly four"):
        _merge(tmp_path, shards[:3], expected_sample_count=4)

    duplicate_manifest = json.loads(shards[3].read_text(encoding="utf-8"))
    duplicate_manifest["shard_index"] = 2
    shards[3].unlink()
    shards[3].write_text(json.dumps(duplicate_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate shard index 2"):
        _merge(tmp_path, shards, expected_sample_count=4)


def test_merge_rejects_boolean_manifest_schema_version(tmp_path):
    shards = _write_fake_shards(tmp_path)
    manifest = json.loads(shards[0].read_text(encoding="utf-8"))
    manifest["schema_version"] = True
    shards[0].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        _merge(tmp_path, shards, expected_sample_count=4)


def test_merge_excludes_legacy_summary_rows(tmp_path):
    shards = _write_fake_shards(tmp_path)
    with (tmp_path / "shard-0.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"summary": True, "accuracy": 1.0}) + "\n")

    _merge(tmp_path, shards, expected_sample_count=4)

    merged = (tmp_path / "merged.jsonl").read_text(encoding="utf-8")
    assert '"summary"' not in merged


def test_merge_never_overwrites_or_partially_publishes_outputs(tmp_path):
    shards = _write_fake_shards(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("owned-by-another-run\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _merge(tmp_path, shards, expected_sample_count=4)

    assert not (tmp_path / "merged.jsonl").exists()
    assert summary_path.read_text(encoding="utf-8") == "owned-by-another-run\n"
