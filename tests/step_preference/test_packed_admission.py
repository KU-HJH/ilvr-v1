"""Contracts for strict packed-admission records and boundary witnesses."""

from __future__ import annotations

import json
import os

import pytest

from step_preference.packed_admission import (
    PACKED_ADMISSION_MANIFEST_KIND,
    PACKED_ADMISSION_SCHEMA_VERSION,
    PackedAdmissionCounts,
    PackedAdmissionIdentity,
    PackedAdmissionManifest,
    PackedAdmissionMeasurement,
    PackedAdmissionRecord,
    PackedAdmissionVerification,
    PackedBoundaryWitnesses,
    build_boundary_witnesses,
    read_packed_admission_manifest,
    select_capacity_witness_indices,
    select_packed_admission,
    write_packed_admission_manifest,
)


HASH = "a" * 64


def measurement(source_index: int, **changes) -> PackedAdmissionMeasurement:
    values = dict(
        source_index=source_index,
        sample_id=f"sample-{source_index}",
        clean_length=3600,
        packed_unpadded_length=4000,
        packed_padded_length=4096,
        vision_token_count=100,
        image_count=1,
        reasoning_block_count=2,
        branch_count=4,
        scored_token_count=100,
    )
    values.update(changes)
    return PackedAdmissionMeasurement(**values)


def record(source_index: int, **changes) -> PackedAdmissionRecord:
    values = dict(measurement(source_index).__dict__, admission_reason="retained")
    values.update(changes)
    return PackedAdmissionRecord(**values)


def identity() -> PackedAdmissionIdentity:
    return PackedAdmissionIdentity(
        original_data_path="/original.jsonl",
        original_data_sha256=HASH,
        clean_data_path="/clean.jsonl",
        clean_data_sha256=HASH,
        source_map_path="/map.json",
        source_map_sha256=HASH,
        retained_data_path="/retained.jsonl",
        retained_data_sha256=HASH,
        clean_length_manifest_path="/length.json",
        clean_length_manifest_sha256=HASH,
        image_root_path="/images",
        checkpoint_path="/checkpoint",
        checkpoint_files=(),
        tokenizer_config_identity=HASH,
        negative_path="/negatives.jsonl",
        negative_sha256=HASH,
        negative_manifest_path="/negative-manifest.json",
        negative_manifest_sha256=HASH,
        assistant_plan_rendering="adjacent-text-space-lf-v1",
        attention_implementation="sdpa",
        vision_attention_implementation="flash_attention_2",
        latent_size=8,
        clean_model_limit=8192,
        clean_prefilter_limit=4608,
        max_packed_sequence_length=4608,
        git_commit="b" * 40,
        tracked_worktree_clean=True,
    )


def manifest() -> PackedAdmissionManifest:
    retained = record(1)
    filtered = record(
        2,
        sample_id="sample-2",
        packed_padded_length=4736,
        admission_reason="packed_length_exceeds_limit",
    )
    witnesses = build_boundary_witnesses((retained,))
    return PackedAdmissionManifest(
        schema_version=PACKED_ADMISSION_SCHEMA_VERSION,
        kind=PACKED_ADMISSION_MANIFEST_KIND,
        identity=identity(),
        records=(retained, filtered),
        retained_source_indices=(1,),
        retained_sample_ids=("sample-1",),
        filtered_source_indices=(2,),
        filtered_sample_ids=("sample-2",),
        boundary_witnesses=witnesses,
        counts=PackedAdmissionCounts(
            original=3450, clean_admissible=3354, mapped=2, retained=1, filtered=1
        ),
    )


def failed_verification(**changes) -> PackedAdmissionVerification:
    values = dict(
        schema_version=PACKED_ADMISSION_SCHEMA_VERSION,
        kind="stage2b-packed-admission-verification",
        status="fail",
        manifest_path="/verification-manifest.json",
        manifest_sha256=None,
        clean_admissible_sha256=None,
        packed_retained_sha256=None,
        negative_sha256=None,
        negative_manifest_sha256=None,
        verified_record_count=0,
        retained_source_indices=(),
        filtered_source_indices=(),
        errors=("manifest is invalid",),
    )
    values.update(changes)
    return PackedAdmissionVerification(**values)


def test_select_packed_admission_classifies_in_order_without_other_reasons():
    retained, filtered = select_packed_admission(
        (measurement(1), measurement(2, packed_padded_length=4736)),
        max_packed_length=4608,
    )

    assert [row.admission_reason for row in retained] == ["retained"]
    assert [row.admission_reason for row in filtered] == ["packed_length_exceeds_limit"]


def test_select_packed_admission_sorts_each_partition_by_source_index():
    retained, filtered = select_packed_admission(
        (
            measurement(4, packed_padded_length=4736),
            measurement(3),
            measurement(2, packed_padded_length=4736),
            measurement(1),
        ),
        max_packed_length=4608,
    )

    assert tuple(row.source_index for row in retained) == (1, 3)
    assert tuple(row.source_index for row in filtered) == (2, 4)


@pytest.mark.parametrize(
    "changes",
    [
        {"packed_unpadded_length": 3599},
        {"packed_padded_length": 4097},
    ],
)
def test_measurement_rejects_invalid_integer_and_length_invariants(changes):
    with pytest.raises(ValueError):
        measurement(1, **changes)
    with pytest.raises(ValueError):
        PackedAdmissionMeasurement(**{**measurement(1).__dict__, "source_index": True})


def test_boundary_witnesses_keep_pareto_representative_and_dominated_global_maximum():
    winner = record(
        4, packed_padded_length=4608, packed_unpadded_length=4590,
        clean_length=3733, vision_token_count=100, branch_count=8,
        scored_token_count=400,
    )
    dominator = record(
        3, packed_padded_length=4608, packed_unpadded_length=4550,
        clean_length=3600, vision_token_count=120, branch_count=9,
        scored_token_count=450,
    )

    witnesses = build_boundary_witnesses((winner, dominator))

    assert witnesses.pareto_source_indices == (3,)
    assert witnesses.global_max_source_index == 4
    assert witnesses.final_probe_source_indices == (3, 4)
    assert select_capacity_witness_indices((winner, dominator)) == (3, 4)


def test_boundary_witnesses_use_lowest_tie_representative_but_keep_higher_global_maximum():
    low = record(2, packed_unpadded_length=4000, clean_length=3600)
    high = record(5, packed_unpadded_length=4090, clean_length=3700)

    witnesses = build_boundary_witnesses((high, low))

    assert witnesses.tie_classes[0].member_source_indices == (2, 5)
    assert witnesses.tie_classes[0].representative_source_index == 2
    assert witnesses.pareto_source_indices == (2,)
    assert witnesses.final_probe_source_indices == (2, 5)


def test_boundary_witnesses_reject_filtered_records_and_more_than_eight_final_probes():
    with pytest.raises(ValueError, match="retained"):
        build_boundary_witnesses((record(1, admission_reason="packed_length_exceeds_limit"),))

    rows = tuple(
        record(
            index,
            packed_padded_length=128 * (index + 1),
            packed_unpadded_length=128 * (index + 1),
            clean_length=128 * (index + 1),
            vision_token_count=10 - index,
            branch_count=10 - index,
            scored_token_count=10 - index,
        )
        for index in range(1, 10)
    )
    with pytest.raises(ValueError, match="eight"):
        build_boundary_witnesses(rows)


def test_manifest_reader_rejects_noncanonical_payload_symlink_and_unknown_fields(tmp_path):
    path = tmp_path / "manifest.json"
    write_packed_admission_manifest(path, manifest())
    raw = path.read_bytes()
    assert read_packed_admission_manifest(path) == manifest()

    path.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="canonical"):
        read_packed_admission_manifest(path)

    link = tmp_path / "manifest-link.json"
    os.symlink(path, link)
    with pytest.raises(ValueError, match="regular"):
        read_packed_admission_manifest(link)


def test_manifest_writer_is_exclusive_and_manifest_constructor_rejects_partition_drift(tmp_path):
    value = manifest()
    path = tmp_path / "manifest.json"
    write_packed_admission_manifest(path, value)
    with pytest.raises(ValueError, match="already exists"):
        write_packed_admission_manifest(path, value)
    with pytest.raises(ValueError):
        PackedAdmissionManifest(
            **{**value.__dict__, "retained_source_indices": (2,)}
        )


def test_manifest_rejects_exact_schema_hash_duplicate_reason_and_count_drift(tmp_path):
    value = manifest()
    path = tmp_path / "manifest.json"
    write_packed_admission_manifest(path, value)
    payload = json.loads(path.read_bytes())
    payload["unexpected"] = None
    path.write_bytes((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode())
    with pytest.raises(ValueError, match="unknown or missing"):
        read_packed_admission_manifest(path)

    with pytest.raises(ValueError, match="SHA-256"):
        PackedAdmissionIdentity(**{**identity().__dict__, "negative_sha256": "A" * 64})
    with pytest.raises(ValueError, match="duplicate"):
        PackedAdmissionManifest(**{**value.__dict__, "records": (value.records[0], value.records[0])})
    with pytest.raises(ValueError, match="exceeds"):
        PackedAdmissionManifest(
            **{
                **value.__dict__,
                "records": (record(1, packed_padded_length=4736), value.records[1]),
            }
        )
    with pytest.raises(ValueError, match="counts"):
        PackedAdmissionManifest(
            **{
                **value.__dict__,
                "counts": PackedAdmissionCounts(
                    original=3450, clean_admissible=3354, mapped=2, retained=0, filtered=2
                ),
            }
        )


def test_identity_requires_exact_latent_size_and_resolved_canonical_paths(tmp_path):
    with pytest.raises(ValueError, match="latent_size"):
        PackedAdmissionIdentity(**{**identity().__dict__, "latent_size": 7})


def test_identity_and_verification_reject_noncanonical_absolute_paths(tmp_path):
    with pytest.raises(ValueError, match="resolved"):
        PackedAdmissionIdentity(
            **{**identity().__dict__, "original_data_path": "/tmp/../original.jsonl"}
        )
    with pytest.raises(ValueError, match="resolved"):
        failed_verification(manifest_path="/tmp/../verification-manifest.json")

    path = tmp_path / "manifest.json"
    write_packed_admission_manifest(path, manifest())
    payload = json.loads(path.read_bytes())
    payload["identity"]["original_data_path"] = "/tmp/../original.jsonl"
    path.write_bytes((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode())
    with pytest.raises(ValueError, match="resolved"):
        read_packed_admission_manifest(path)


def test_failing_verification_hashes_must_form_an_authenticated_prefix():
    with pytest.raises(ValueError, match="prefix"):
        failed_verification(clean_admissible_sha256=HASH)
    with pytest.raises(ValueError, match="prefix"):
        failed_verification(manifest_sha256=HASH, packed_retained_sha256=HASH)
