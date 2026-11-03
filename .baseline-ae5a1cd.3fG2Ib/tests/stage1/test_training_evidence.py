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
        "micro_batch": 2,
        "gradient_accumulation_steps": 2,
        "ema_tau": 1.0,
        "seed": 42,
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
    assert decoded["micro_batch"] == 2
    assert decoded["gradient_accumulation_steps"] == 2
    assert decoded["effective_global_batch"] == 16
    assert decoded["ema_tau"] == 1.0
    assert decoded["teacher_update_mode"] == "frozen"
    assert decoded["seed"] == 42
    assert decoded["loss"] == 0.125
    assert decoded["finite_loss"] is True


def test_stage1_training_evidence_records_moving_teacher_mode(tmp_path):
    kwargs = evidence_kwargs(tmp_path)
    kwargs["ema_tau"] = 0.999

    payload = build_stage1_training_evidence(**kwargs)

    assert payload["teacher_update_mode"] == "moving"
    assert payload["ema_tau"] == 0.999


def test_stage1_training_evidence_keeps_schema_one_read_compatibility(tmp_path):
    kwargs = evidence_kwargs(tmp_path)
    legacy = {
        "schema_version": 1,
        "kind": "stage1-training-evidence",
        "run_id": kwargs["run_id"],
        "stage": "stage1",
        "method": kwargs["method"],
        "zero_stage": kwargs["zero_stage"],
        "world_size": kwargs["world_size"],
        "process_ranks": [0, 1, 2, 3],
        "global_step": kwargs["global_step"],
        "checkpoint_path": str(kwargs["checkpoint_path"]),
        "loss": kwargs["loss"],
        "finite_loss": True,
    }

    decoded = decode_stage1_training_evidence(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode(),
        expected_run_id=RUN_ID,
        expected_method="full",
        expected_zero_stage=1,
        expected_world_size=4,
        expected_checkpoint=kwargs["checkpoint_path"],
        expected_global_step=1,
    )

    assert decoded == legacy


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


@pytest.mark.parametrize(
    ("expected", "value"),
    [
        ("expected_micro_batch", 4),
        ("expected_gradient_accumulation_steps", 4),
        ("expected_ema_tau", 0.999),
        ("expected_seed", 7),
    ],
)
def test_stage1_training_evidence_decoder_rejects_wrong_experiment_config(
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
        "expected_micro_batch": 2,
        "expected_gradient_accumulation_steps": 2,
        "expected_ema_tau": 1.0,
        "expected_seed": 42,
    }
    expected_kwargs[expected] = value

    with pytest.raises(ValueError, match="configuration"):
        decode_stage1_training_evidence(raw, **expected_kwargs)


def test_stage1_training_evidence_publish_is_exclusive(tmp_path):
    kwargs = evidence_kwargs(tmp_path)
    output = tmp_path / "initial-producer-evidence.json"

    publish_stage1_training_evidence(output, **kwargs)

    assert json.loads(output.read_text()) == build_stage1_training_evidence(**kwargs)
    with pytest.raises(FileExistsError):
        publish_stage1_training_evidence(output, **kwargs)
