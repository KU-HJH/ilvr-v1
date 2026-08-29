import json
from pathlib import Path

import pytest

from stage1.training_evidence import (
    build_stage1_training_evidence,
    decode_stage1_training_evidence,
    publish_stage1_training_evidence,
)


RUN_ID = "capacity-full-zero1-r4-batch2"


def evidence_kwargs(tmp_path):
    checkpoint = tmp_path / "checkpoints" / RUN_ID / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    return {
        "run_id": RUN_ID,
        "method": "full",
        "zero_stage": 1,
        "world_size": 4,
        "global_step": 1,
        "checkpoint_path": checkpoint,
        "loss": 0.125,
    }


def test_stage1_training_evidence_round_trips_exact_runtime_identity(tmp_path):
    kwargs = evidence_kwargs(tmp_path)
    payload = build_stage1_training_evidence(**kwargs)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    decoded = decode_stage1_training_evidence(
        raw,
        expected_run_id=RUN_ID,
        expected_method="full",
        expected_zero_stage=1,
        expected_world_size=4,
        expected_checkpoint=kwargs["checkpoint_path"],
        expected_global_step=1,
    )

    assert decoded == payload
    assert decoded["process_ranks"] == [0, 1, 2, 3]
    assert decoded["loss"] == 0.125
    assert decoded["finite_loss"] is True


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), True, "0.1"])
def test_stage1_training_evidence_rejects_nonfinite_or_loose_loss(tmp_path, loss):
    kwargs = evidence_kwargs(tmp_path)
    kwargs["loss"] = loss

    with pytest.raises(ValueError, match="finite loss"):
        build_stage1_training_evidence(**kwargs)


@pytest.mark.parametrize(
    ("expected", "value"),
    [
        ("expected_run_id", "other-run"),
        ("expected_method", "lora"),
        ("expected_zero_stage", 2),
        ("expected_world_size", 2),
        ("expected_global_step", 2),
    ],
)
def test_stage1_training_evidence_decoder_rejects_wrong_identity(
    tmp_path, expected, value
):
    kwargs = evidence_kwargs(tmp_path)
    payload = build_stage1_training_evidence(**kwargs)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    expected_kwargs = {
        "expected_run_id": RUN_ID,
        "expected_method": "full",
        "expected_zero_stage": 1,
        "expected_world_size": 4,
        "expected_checkpoint": kwargs["checkpoint_path"],
        "expected_global_step": 1,
    }
    expected_kwargs[expected] = value

    with pytest.raises(ValueError, match="identity"):
        decode_stage1_training_evidence(raw, **expected_kwargs)


def test_stage1_training_evidence_decoder_rejects_wrong_checkpoint(tmp_path):
    kwargs = evidence_kwargs(tmp_path)
    payload = build_stage1_training_evidence(**kwargs)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(ValueError, match="identity"):
        decode_stage1_training_evidence(
            raw,
            expected_run_id=RUN_ID,
            expected_method="full",
            expected_zero_stage=1,
            expected_world_size=4,
            expected_checkpoint=kwargs["checkpoint_path"].parent / "checkpoint-2",
            expected_global_step=1,
        )


def test_stage1_training_evidence_publish_is_exclusive(tmp_path):
    kwargs = evidence_kwargs(tmp_path)
    output = tmp_path / "initial-producer-evidence.json"

    publish_stage1_training_evidence(output, **kwargs)

    assert json.loads(output.read_text()) == build_stage1_training_evidence(**kwargs)
    with pytest.raises(FileExistsError):
        publish_stage1_training_evidence(output, **kwargs)
