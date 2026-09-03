import json

import pytest

from step_preference.negatives import (
    CheckpointFile,
    NegativeManifest,
    StepNegative,
    filter_negative,
    load_negative_store,
    write_negative_artifact,
)


SOURCE_SHA256 = "c805f1e47450a7698e2e0b9f59b08d1f8945598a99ef24b5f25dbe046567307f"
CHECKPOINT_SHA256 = "f51ebaa08e836519e3f22e9edef16a398c2c1539c1a4f373097ad6335d8d9a1d"


def _negative(**changes):
    values = {
        "sample_id": "sample-a",
        "step_id": 0,
        "chosen_text": "chosen completion",
        "rejected_visual_text": "visual negative",
        "rejected_textual_text": "different negative",
        "valid_visual_negative": True,
        "valid_textual_negative": True,
        "reference_chosen_score": -0.25,
        "reference_visual_rejected_score": -1.25,
        "reference_textual_rejected_score": -0.75,
    }
    values.update(changes)
    return StepNegative(**values)


def _identity(tmp_path):
    checkpoint = tmp_path / "stage2a"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(b"checkpoint-content\n")
    return {
        "source_sha256": SOURCE_SHA256,
        "selected_row_indices": (0,),
        "stage2a_checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_files": (CheckpointFile("config.json", 19, CHECKPOINT_SHA256),),
        "tokenizer_config_identity": "tokenizer-config-v1",
        "latent_size": 8,
        "generation_settings": {"max_new_tokens": 32, "min_tokens": 2},
    }


def _expected_identity(identity):
    return {
        "expected_source_sha256": identity["source_sha256"],
        "expected_selected_row_indices": identity["selected_row_indices"],
        "expected_stage2a_checkpoint_path": identity["stage2a_checkpoint_path"],
        "expected_checkpoint_files": identity["checkpoint_files"],
        "expected_tokenizer_config_identity": identity["tokenizer_config_identity"],
        "expected_latent_size": identity["latent_size"],
        "expected_generation_settings": identity["generation_settings"],
    }


@pytest.mark.parametrize(
    ("rejected_text", "token_ids"),
    [
        ("chosen completion", (11, 12)),
        (" chosen\n completion ", (11, 12)),
        ("\t ", (11, 12)),
        ("different", (11,)),
    ],
)
def test_filter_negative_rejects_unusable_candidate_pairs(rejected_text, token_ids):
    """Catches accepting exact, normalized, empty, or too-short negatives."""
    assert not filter_negative(
        chosen_text="chosen completion",
        rejected_text=rejected_text,
        scoreable_token_ids=token_ids,
        min_tokens=2,
    )


def test_filter_negative_accepts_distinct_normalized_completion_at_minimum_length():
    """Catches rejecting the boundary candidate used for a valid preference pair."""
    assert filter_negative(
        chosen_text="chosen completion",
        rejected_text="different completion",
        scoreable_token_ids=(51, 52),
        min_tokens=2,
    )


def test_step_negative_represents_invalid_modalities_and_is_immutable():
    """Catches treating an unavailable generated pair as an exception or score."""
    row = _negative(
        rejected_visual_text=None,
        valid_visual_negative=False,
        reference_visual_rejected_score=None,
    )

    assert row.rejected_visual_text is None
    with pytest.raises(Exception):
        row.step_id = 1


@pytest.mark.parametrize(
    "changes",
    [
        {"reference_chosen_score": float("nan")},
        {"reference_visual_rejected_score": float("inf")},
        {
            "valid_textual_negative": False,
            "reference_textual_rejected_score": -0.75,
            "rejected_textual_text": None,
        },
        {
            "valid_textual_negative": True,
            "reference_textual_rejected_score": None,
            "rejected_textual_text": "different negative",
        },
    ],
)
def test_step_negative_rejects_nonfinite_or_structurally_inconsistent_scores(changes):
    """Catches rows that cannot safely contribute reference DPO terms."""
    with pytest.raises(ValueError):
        _negative(**changes)


def test_write_and_load_artifact_bind_raw_bytes_and_every_expected_identity(tmp_path):
    """Catches training against rows not frozen to the selected source and checkpoint."""
    source = tmp_path / "source.jsonl"
    source.write_bytes(b'{"row":1}\n')
    negatives_path = tmp_path / "negatives.jsonl"
    manifest_path = tmp_path / "negative-manifest.json"
    identity = _identity(tmp_path)

    manifest = write_negative_artifact(
        negative_path=negatives_path,
        manifest_path=manifest_path,
        negatives=(_negative(),),
        source_path=source,
        **identity,
    )
    store = load_negative_store(
        negative_path=negatives_path,
        manifest_path=manifest_path,
        expected_keys=(("sample-a", 0),),
        **_expected_identity(identity),
    )

    assert manifest.source_sha256 == SOURCE_SHA256
    assert manifest.negative_sha256 == (
        "3323476b34b66fdff82c8814e925828fa08ef4131dfe54df4c55c59efdf5a399"
    )
    assert store == {("sample-a", 0): _negative()}
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_load_negative_store_rejects_duplicate_join_identity(tmp_path):
    """Catches two artifact rows competing for the same sample reasoning step."""
    source = tmp_path / "source.jsonl"
    source.write_bytes(b'{"row":1}\n')
    negatives_path = tmp_path / "negatives.jsonl"
    manifest_path = tmp_path / "negative-manifest.json"
    identity = _identity(tmp_path)
    manifest = write_negative_artifact(
        negative_path=negatives_path,
        manifest_path=manifest_path,
        negatives=(_negative(),),
        source_path=source,
        **identity,
    )
    duplicate = negatives_path.read_text(encoding="utf-8")
    negatives_path.write_text(duplicate + duplicate, encoding="utf-8")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["negative_sha256"] = (
        "3f4029a1c685587839a572eefa510d757b81191a7edf6be7e3567fc428676c48"
    )
    manifest_path.write_text(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate negative join key"):
        load_negative_store(
            negative_path=negatives_path,
            manifest_path=manifest_path,
            expected_keys=(("sample-a", 0),),
            **_expected_identity(identity),
        )
    assert manifest.negative_sha256 != ""


def test_load_negative_store_rejects_manifest_identity_and_missing_rows(tmp_path):
    """Catches stale manifests and datasets whose required step row is absent."""
    source = tmp_path / "source.jsonl"
    source.write_bytes(b'{"row":1}\n')
    negatives_path = tmp_path / "negatives.jsonl"
    manifest_path = tmp_path / "negative-manifest.json"
    identity = _identity(tmp_path)
    write_negative_artifact(
        negative_path=negatives_path,
        manifest_path=manifest_path,
        negatives=(_negative(),),
        source_path=source,
        **identity,
    )

    with pytest.raises(ValueError, match="source SHA-256 does not match"):
        load_negative_store(
            negative_path=negatives_path,
            manifest_path=manifest_path,
            expected_source_sha256="b" * 64,
            expected_selected_row_indices=(0,),
            expected_stage2a_checkpoint_path=identity["stage2a_checkpoint_path"],
            expected_checkpoint_files=identity["checkpoint_files"],
            expected_tokenizer_config_identity="tokenizer-config-v1",
            expected_latent_size=8,
            expected_generation_settings=identity["generation_settings"],
            expected_keys=(("sample-a", 0),),
        )
    with pytest.raises(ValueError, match="missing or unexpected negative join keys"):
        load_negative_store(
            negative_path=negatives_path,
            manifest_path=manifest_path,
            expected_keys=(("sample-a", 0), ("sample-a", 1)),
            **_expected_identity(identity),
        )


def test_load_negative_store_rejects_changed_checkpoint_file_or_jsonl_bytes(tmp_path):
    """Catches artifacts whose recorded Stage 2A files or negative bytes went stale."""
    source = tmp_path / "source.jsonl"
    source.write_bytes(b'{"row":1}\n')
    negatives_path = tmp_path / "negatives.jsonl"
    manifest_path = tmp_path / "negative-manifest.json"
    identity = _identity(tmp_path)
    write_negative_artifact(
        negative_path=negatives_path,
        manifest_path=manifest_path,
        negatives=(_negative(),),
        source_path=source,
        **identity,
    )
    (tmp_path / "stage2a" / "config.json").write_bytes(b"different checkpoint\n")

    with pytest.raises(ValueError, match="checkpoint file hash or size does not match"):
        load_negative_store(
            negative_path=negatives_path,
            manifest_path=manifest_path,
            expected_keys=(("sample-a", 0),),
            **_expected_identity(identity),
        )

    (tmp_path / "stage2a" / "config.json").write_bytes(b"checkpoint-content\n")
    negatives_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="negative artifact SHA-256 does not match"):
        load_negative_store(
            negative_path=negatives_path,
            manifest_path=manifest_path,
            expected_keys=(("sample-a", 0),),
            **_expected_identity(identity),
        )


def test_negative_manifest_is_immutable():
    """Catches a frozen reference identity being mutable after validation."""
    manifest = NegativeManifest(
        schema_version=1,
        source_sha256=SOURCE_SHA256,
        selected_row_indices=(0,),
        stage2a_checkpoint_path="/checkpoint",
        checkpoint_files=(CheckpointFile("config.json", 19, CHECKPOINT_SHA256),),
        tokenizer_config_identity="tokenizer-config-v1",
        latent_size=8,
        generation_settings=(("max_new_tokens", 32),),
        negative_sha256="c" * 64,
    )

    with pytest.raises(Exception):
        manifest.latent_size = 4
