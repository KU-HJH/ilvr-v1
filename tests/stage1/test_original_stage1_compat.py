"""Compatibility for the historical schema-3 Stage 1 DoRA artifact."""

import json
from pathlib import Path

import pytest

from stage1.checkpointing import (
    Stage1PeftMetadata,
    decode_stage1_peft_metadata_snapshot,
    read_stage1_peft_metadata_snapshot,
)
from tests.stage1.test_checkpointing import sample_metadata


def historical_payload():
    payload = sample_metadata().to_dict()
    payload.update(
        schema_version=3,
        ema_setting="frozen",
        teacher_target_mode="sample_isolated",
        training_implementation_version=1,
    )
    return payload


def test_schema3_decodes_and_round_trips_teacher_provenance():
    payload = historical_payload()
    decoded = Stage1PeftMetadata.from_dict(payload)
    assert decoded.to_dict() == payload
    assert decoded.ema_setting == "frozen"
    assert decoded.teacher_target_mode == "sample_isolated"
    assert decoded.training_implementation_version == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("ema_setting", "other"),
        ("teacher_target_mode", "other"),
        ("training_implementation_version", 0),
        ("training_implementation_version", True),
    ],
)
def test_schema3_rejects_invalid_teacher_provenance(field, value):
    payload = historical_payload()
    payload[field] = value
    with pytest.raises(ValueError):
        Stage1PeftMetadata.from_dict(payload)


def test_schema3_requires_complete_field_set():
    payload = historical_payload()
    payload.pop("ema_setting")
    with pytest.raises(ValueError, match="fields"):
        Stage1PeftMetadata.from_dict(payload)


def test_schema2_write_does_not_gain_schema3_fields():
    payload = sample_metadata().to_dict()
    assert payload["schema_version"] == 2
    assert "ema_setting" not in payload
    assert Stage1PeftMetadata.from_dict(payload).to_dict() == payload


def test_real_original_metadata_decodes_without_model_load():
    original = Path(
        "/raid/workspace/temp2/ILVR/checkpoints/zero-stage-compatibility/"
        "stage1-dora-frozen-b1-lr5e5-zero1-1epoch-exact32768-20260902/"
        "checkpoint-431"
    )
    if not original.is_dir():
        pytest.skip("historical artifact unavailable")
    decoded = decode_stage1_peft_metadata_snapshot(
        read_stage1_peft_metadata_snapshot(original)
    )
    assert decoded.schema_version == 3
    assert (decoded.ema_setting, decoded.teacher_target_mode,
            decoded.training_implementation_version) == ("frozen", "sample_isolated", 1)
    assert json.loads((original / "stage1_peft_metadata.json").read_bytes()) == decoded.to_dict()
