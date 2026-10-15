from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.multinode_evaluation as evaluation_module
from src.multinode_evaluation import (
    iter_owned_samples,
    merge_shards,
    write_shard_manifest,
)


@pytest.mark.parametrize(
    "relative_path",
    ("images_comt/creation/12003.png", "creation/12003.png"),
)
def test_evaluation_image_paths_resolve_beneath_dataset_root(
    tmp_path, relative_path
):
    image = tmp_path / relative_path
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")

    assert evaluation_module.resolve_evaluation_image_paths(
        [relative_path], tmp_path
    ) == [str(image.resolve())]


def test_evaluation_image_paths_report_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image not found"):
        evaluation_module.resolve_evaluation_image_paths(
            "images_comt/creation/missing.png", tmp_path
        )


@pytest.mark.parametrize(
    "unsafe_path",
    ("../escape.png", "/tmp/absolute.png", "images_comt/control\n.png"),
)
def test_evaluation_image_paths_reject_unsafe_input(tmp_path, unsafe_path):
    with pytest.raises(ValueError, match="safe relative"):
        evaluation_module.resolve_evaluation_image_paths(unsafe_path, tmp_path)


def test_evaluation_image_paths_reject_symlink_escape(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (dataset_root / "linked.png").symlink_to(outside)

    with pytest.raises(ValueError, match="beneath dataset root"):
        evaluation_module.resolve_evaluation_image_paths(
            "linked.png", dataset_root
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
    generation_configs=None,
):
    manifests = []
    raw_rows = raw_rows or {}
    row_overrides = row_overrides or {}
    if processed_indices is None:
        processed_indices = indices
    if generation_configs is None:
        generation_configs = (
            {
                "max_new_tokens": 4096,
                "max_sequence_length": 32768,
                "repetition_penalty": 1.05,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        ) * 4
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
            generation_config=generation_configs[shard_index],
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
        "correct": 4,
        "source_count": 7,
        "valid_gold_count": 7,
        "invalid_gold_count": 0,
        "invalid_gold": [],
        "coverage_accuracy": 4 / 7,
        "valid_gold_accuracy": 4 / 7,
        "generation_config": {
            "max_new_tokens": 4096,
            "max_sequence_length": 32768,
            "repetition_penalty": 1.05,
            "temperature": 0.0,
            "top_p": 1.0,
            "vision_ablation": "none",
            "generation_limit_policy": "bounded",
        },
    }
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_merge_preserves_gold_denominators_and_invalid_gold_records(tmp_path):
    """Gold validity must remain auditable after four shards are combined."""
    invalid = {57, 93}
    matching = set()
    for index in range(400):
        if index not in invalid and len(matching) < 65:
            matching.add(index)
    indices = tuple(tuple(range(shard, 400, 4)) for shard in range(4))
    overrides = {
        index: {
            "index": index,
            "task_name": "zebra-cot",
            "match": index in matching,
            "inference_time_sec": 0.5,
            "invalid_gold_reason": "blank" if index in invalid else None,
        }
        for index in range(400)
    }
    shards = _write_fake_shards(
        tmp_path,
        indices=indices,
        row_overrides=overrides,
    )

    summary = _merge(tmp_path, shards, expected_sample_count=400)

    assert summary == {
        "summary": True,
        "num_samples": 400,
        "average_inference_time_sec": 0.5,
        "total_inference_time_sec": 200.0,
        "accuracy": 0.1625,
        "success": 65,
        "correct": 65,
        "source_count": 400,
        "valid_gold_count": 398,
        "invalid_gold_count": 2,
        "invalid_gold": [
            {"index": 57, "reason": "blank"},
            {"index": 93, "reason": "blank"},
        ],
        "coverage_accuracy": 0.1625,
        "valid_gold_accuracy": 65 / 398,
        "generation_config": {
            "max_new_tokens": 4096,
            "max_sequence_length": 32768,
            "repetition_penalty": 1.05,
            "temperature": 0.0,
            "top_p": 1.0,
            "vision_ablation": "none",
            "generation_limit_policy": "bounded",
        },
    }


def test_merge_rejects_duplicate_source_index(tmp_path):
    shards = _write_fake_shards(
        tmp_path, indices=((0, 4), (1, 5), (2, 6), (3, 4))
    )
    with pytest.raises(ValueError, match="duplicate source index 4"):
        _merge(tmp_path, shards, expected_sample_count=7)


def test_merge_rejects_generation_config_mismatch(tmp_path):
    configs = [
        {
            "max_new_tokens": 4096,
            "max_sequence_length": 32768,
            "repetition_penalty": 1.05,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        for _ in range(4)
    ]
    configs[3] = {
        "max_new_tokens": 4096,
        "max_sequence_length": 65536,
        "repetition_penalty": 1.05,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    shards = _write_fake_shards(tmp_path, generation_configs=configs)

    with pytest.raises(ValueError, match="generation config"):
        _merge(tmp_path, shards, expected_sample_count=4)


def test_merge_rejects_mixed_vision_ablation_modes(tmp_path):
    configs = [
        {
            "max_new_tokens": 4096,
            "max_sequence_length": 32768,
            "repetition_penalty": 1.05,
            "temperature": 0.0,
            "top_p": 1.0,
            "vision_ablation": "all",
        }
        for _ in range(4)
    ]
    configs[3] = {**configs[3], "vision_ablation": "interleaved"}

    shards = _write_fake_shards(tmp_path, generation_configs=configs)

    with pytest.raises(ValueError, match="generation config mismatch"):
        _merge(tmp_path, shards, expected_sample_count=4)


@pytest.mark.parametrize("condition", [None, False, True], ids=["absent", "on", "off"])
def test_latent_condition_roundtrips_through_manifest_and_merge(tmp_path, condition):
    """Normalization must retain explicit conditions without inventing legacy fields."""
    config = {
        "max_new_tokens": 4096, "max_sequence_length": 32768,
        "repetition_penalty": 1.05, "temperature": 0.0, "top_p": 1.0,
    }
    if condition is not None:
        config["disable_latent_generation"] = condition
    shards = _write_fake_shards(tmp_path, generation_configs=[config] * 4)
    summary = _merge(tmp_path, shards, expected_sample_count=4)

    configs = [json.loads(path.read_text())["generation_config"] for path in shards]
    configs.append(summary["generation_config"])
    configs.append(json.loads((tmp_path / "summary.json").read_text())["generation_config"])
    for saved in configs:
        if condition is None:
            assert "disable_latent_generation" not in saved
        else:
            assert saved["disable_latent_generation"] is condition


@pytest.mark.parametrize("other_condition", [None, False], ids=["legacy-on", "explicit-on"])
def test_merge_rejects_mixed_latent_conditions(tmp_path, other_condition):
    """Dropping OFF metadata during reading would incorrectly permit this merge."""
    config = {
        "max_new_tokens": 4096, "max_sequence_length": 32768,
        "repetition_penalty": 1.05, "temperature": 0.0, "top_p": 1.0,
        "disable_latent_generation": True,
    }
    other = dict(config)
    if other_condition is None:
        other.pop("disable_latent_generation")
    else:
        other["disable_latent_generation"] = other_condition
    shards = _write_fake_shards(tmp_path, generation_configs=[config] * 3 + [other])

    with pytest.raises(ValueError, match="generation config mismatch"):
        _merge(tmp_path, shards, expected_sample_count=4)
    assert not (tmp_path / "merged.jsonl").exists()
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("boundary", ["writer", "reader"])
@pytest.mark.parametrize("invalid", [0, 1, None, "true", [], {}])
def test_latent_condition_rejects_non_boolean_values(tmp_path, boundary, invalid):
    """Truthy values, including integer bool aliases, are not condition identities."""
    config = {
        "max_new_tokens": 4096, "max_sequence_length": 32768,
        "repetition_penalty": 1.05, "temperature": 0.0, "top_p": 1.0,
    }
    if boundary == "writer":
        config["disable_latent_generation"] = invalid
        with pytest.raises(ValueError, match="generation config"):
            _write_fake_shards(tmp_path, generation_configs=[config] * 4)
        assert not (tmp_path / "shard-0.manifest.json").exists()
    else:
        shards = _write_fake_shards(tmp_path)
        manifest = json.loads(shards[0].read_text())
        manifest["generation_config"]["disable_latent_generation"] = invalid
        shards[0].write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="generation config"):
            _merge(tmp_path, shards, expected_sample_count=4)
        assert not (tmp_path / "merged.jsonl").exists()


@pytest.mark.parametrize("boundary", ["writer", "reader"])
def test_latent_condition_does_not_allow_unknown_config_fields(tmp_path, boundary):
    config = {
        "max_new_tokens": 4096, "max_sequence_length": 32768,
        "repetition_penalty": 1.05, "temperature": 0.0, "top_p": 1.0,
        "disable_latent_generation": True, "unknown_condition": True,
    }
    if boundary == "writer":
        with pytest.raises(ValueError, match="generation config"):
            _write_fake_shards(tmp_path, generation_configs=[config] * 4)
    else:
        shards = _write_fake_shards(tmp_path)
        manifest = json.loads(shards[0].read_text())
        manifest["generation_config"].update(config)
        shards[0].write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="generation config"):
            _merge(tmp_path, shards, expected_sample_count=4)


def test_merge_keeps_schema_one_sidecar_compatibility(tmp_path):
    shards = _write_fake_shards(tmp_path)
    for path in shards:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 1
        manifest.pop("generation_config", None)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = _merge(tmp_path, shards, expected_sample_count=4)

    assert "generation_config" not in summary


def test_merge_keeps_schema_two_generation_config_compatibility(tmp_path):
    """Reading a pre-sequence-cap sidecar must not require rewriting history."""
    shards = _write_fake_shards(tmp_path)
    for path in shards:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 2
        manifest["generation_config"].pop("max_sequence_length")
        manifest["generation_config"].pop("repetition_penalty")
        manifest["generation_config"].pop("vision_ablation")
        manifest["generation_config"].pop("generation_limit_policy")
        path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = _merge(tmp_path, shards, expected_sample_count=4)

    assert summary["generation_config"] == {
        "max_new_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def test_merge_keeps_pre_policy_manifest_compatibility(tmp_path):
    """A schema-five result remains readable without inventing a saved policy."""
    shards = _write_fake_shards(tmp_path)
    for path in shards:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 5
        manifest["generation_config"].pop("generation_limit_policy", None)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = _merge(tmp_path, shards, expected_sample_count=4)

    assert "generation_limit_policy" not in summary["generation_config"]


def test_merge_keeps_schema_three_sequence_cap_compatibility(tmp_path):
    """Reading a pre-penalty sidecar must not require rewriting history."""
    shards = _write_fake_shards(tmp_path)
    for path in shards:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 3
        manifest["generation_config"].pop("repetition_penalty")
        manifest["generation_config"].pop("vision_ablation")
        manifest["generation_config"].pop("generation_limit_policy")
        path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = _merge(tmp_path, shards, expected_sample_count=4)

    assert summary["generation_config"] == {
        "max_new_tokens": 4096,
        "max_sequence_length": 32768,
        "temperature": 0.0,
        "top_p": 1.0,
    }


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


def test_merge_rejects_error_row_even_when_shaped_as_legacy_summary(tmp_path):
    shards = _write_fake_shards(
        tmp_path,
        row_overrides={
            0: {
                "summary": True,
                "index": 0,
                "task_name": "task",
                "match": False,
                "error": "OOM",
            }
        },
    )

    with pytest.raises(ValueError, match="error row"):
        _merge(tmp_path, shards, expected_sample_count=4)
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
