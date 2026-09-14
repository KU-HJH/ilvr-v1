import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "run_stage2_objective_examples.sh"


@pytest.fixture
def example_inputs():
    checkpoint_root = REPO_ROOT / "checkpoints"
    data_root = REPO_ROOT / "data"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    stage1 = Path(tempfile.mkdtemp(prefix="objective-example-stage1-", dir=checkpoint_root))
    data_dir = Path(tempfile.mkdtemp(prefix="objective-example-data-", dir=data_root))
    data_path = data_dir / "retained samples.jsonl"
    image_root = data_dir / "images"
    data_path.write_text("{}\n", encoding="utf-8")
    image_root.mkdir()
    try:
        yield stage1, data_path, image_root
    finally:
        shutil.rmtree(stage1)
        shutil.rmtree(data_dir)


@pytest.fixture
def recording_coordinator(tmp_path):
    coordinator = tmp_path / "record-coordinator.sh"
    coordinator.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\0' \"$@\" > \"$OBJECTIVE_EXAMPLE_ARGV\"\n"
        "env -0 > \"$OBJECTIVE_EXAMPLE_ENVIRONMENT\"\n",
        encoding="utf-8",
    )
    coordinator.chmod(0o755)
    return coordinator


def _run_wrapper(
    mode,
    *,
    example_inputs,
    recording_coordinator,
    tmp_path,
    run_id=None,
    port="29871",
    max_steps="423",
    extra=(),
    action=(),
):
    stage1, data_path, image_root = example_inputs
    argv_path = tmp_path / f"argv-{uuid.uuid4().hex}"
    environment_path = tmp_path / f"env-{uuid.uuid4().hex}"
    run_id = run_id or f"objective-example-{uuid.uuid4().hex}"
    command = [
        "bash",
        str(WRAPPER),
        mode,
        "--stage1",
        str(stage1),
        "--data",
        str(data_path),
        "--image-root",
        str(image_root),
        "--run-id",
        run_id,
        "--max-steps",
        max_steps,
        "--main-process-port",
        port,
        *extra,
        *action,
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={
            "PATH": os.environ["PATH"],
            "ILVR_STAGE2_OBJECTIVE_COORDINATOR": str(recording_coordinator),
            "OBJECTIVE_EXAMPLE_ARGV": str(argv_path),
            "OBJECTIVE_EXAMPLE_ENVIRONMENT": str(environment_path),
        },
        capture_output=True,
        text=True,
    )
    forwarded = (
        argv_path.read_bytes().split(b"\0")[:-1] if argv_path.exists() else []
    )
    forwarded = [item.decode() for item in forwarded]
    observed_environment = {}
    if environment_path.exists():
        for item in environment_path.read_bytes().split(b"\0"):
            if item:
                name, value = item.decode().split("=", 1)
                observed_environment[name] = value
    return result, forwarded, observed_environment, run_id


def _option_value(arguments, option):
    return arguments[arguments.index(option) + 1]


def test_three_modes_forward_the_same_baseline_and_only_required_objective_branches(
    example_inputs, recording_coordinator, tmp_path
):
    common = dict(
        example_inputs=example_inputs,
        recording_coordinator=recording_coordinator,
        tmp_path=tmp_path,
    )
    ce_result, ce_args, ce_environment, _ = _run_wrapper("ce", **common)
    entropy_result, entropy_args, entropy_environment, _ = _run_wrapper(
        "entropy", extra=("--lambda-dyn", "0.4"), **common
    )
    combined_result, combined_args, combined_environment, _ = _run_wrapper(
        "combined",
        extra=("--lambda-dyn", "0.4", "--lambda-vis", "0.6"),
        **common,
    )

    assert ce_result.returncode == 0, ce_result.stderr
    assert entropy_result.returncode == 0, entropy_result.stderr
    assert combined_result.returncode == 0, combined_result.stderr

    assert _option_value(ce_args, "--lambda_dyn") == "0"
    assert _option_value(ce_args, "--lambda_vis") == "0"
    assert "--reference_model_path" not in ce_args
    assert _option_value(entropy_args, "--lambda_dyn") == "0.4"
    assert _option_value(entropy_args, "--lambda_vis") == "0"
    assert _option_value(combined_args, "--lambda_dyn") == "0.4"
    assert _option_value(combined_args, "--lambda_vis") == "0.6"

    stage1, data_path, image_root = example_inputs
    for arguments, observed_environment in (
        (ce_args, ce_environment),
        (entropy_args, entropy_environment),
        (combined_args, combined_environment),
    ):
        assert _option_value(arguments, "--backend") == "ddp"
        assert "--enable_stage2_objective" in arguments
        assert _option_value(arguments, "--seed") == "42"
        assert _option_value(arguments, "--warm_up_steps") == "10"
        assert _option_value(arguments, "--max_steps") == "423"
        assert _option_value(arguments, "--save_steps") == "423"
        assert _option_value(arguments, "--save_strategy") == "steps"
        assert "--dry-run" in arguments
        assert observed_environment["STAGE1_CHECKPOINT"] == str(stage1)
        assert observed_environment["DATA_PATH"] == str(data_path)
        assert observed_environment["IMAGE_ROOT"] == str(image_root)
        assert observed_environment["GRADIENT_ACCUMULATION_STEPS"] == "2"
        assert observed_environment["LEARNING_RATE"] == "5e-5"
        assert observed_environment["TUNING_METHOD"] == "dora"
        assert observed_environment["LORA_R"] == "32"
        assert observed_environment["LORA_ALPHA"] == "64"
        assert observed_environment["LORA_DROPOUT"] == "0.05"
        assert observed_environment["LATENT_SIZE"] == "8"
        assert observed_environment["SUPERVISE_LATENT_BOUNDARIES"] == "true"
        assert observed_environment["ATTENTION_IMPLEMENTATION"] == "sdpa"
        assert observed_environment["GRADIENT_CHECKPOINTING"] == "true"

    assert _option_value(entropy_args, "--reference_model_path") == str(stage1)
    assert _option_value(combined_args, "--reference_model_path") == str(stage1)


def test_default_dry_run_creates_no_run_artifacts(
    example_inputs, recording_coordinator, tmp_path
):
    run_id = f"objective-dry-{uuid.uuid4().hex}"
    destinations = (
        REPO_ROOT / "checkpoints" / run_id,
        REPO_ROOT / "results" / run_id,
        REPO_ROOT / "logs" / "multinode" / run_id,
        REPO_ROOT / "logs" / "zero-stage-compatibility" / run_id,
    )
    assert all(not path.exists() for path in destinations)

    result, arguments, _, _ = _run_wrapper(
        "ce",
        example_inputs=example_inputs,
        recording_coordinator=recording_coordinator,
        tmp_path=tmp_path,
        run_id=run_id,
    )

    assert result.returncode == 0, result.stderr
    assert "--dry-run" in arguments
    assert all(not path.exists() for path in destinations)


def test_launch_is_explicit_and_cannot_be_combined_with_dry_run(
    example_inputs, recording_coordinator, tmp_path
):
    result, arguments, _, _ = _run_wrapper(
        "ce",
        example_inputs=example_inputs,
        recording_coordinator=recording_coordinator,
        tmp_path=tmp_path,
        action=("--launch",),
    )
    assert result.returncode == 0, result.stderr
    assert "--dry-run" not in arguments

    rejected, rejected_arguments, _, _ = _run_wrapper(
        "ce",
        example_inputs=example_inputs,
        recording_coordinator=recording_coordinator,
        tmp_path=tmp_path,
        action=("--dry-run", "--launch"),
    )
    assert rejected.returncode == 2
    assert not rejected_arguments
    assert "mutually exclusive" in rejected.stderr


@pytest.mark.parametrize("mode", ("unknown", "CE", ""))
def test_invalid_mode_is_rejected_before_the_coordinator(
    mode, example_inputs, recording_coordinator, tmp_path
):
    result, arguments, _, _ = _run_wrapper(
        mode,
        example_inputs=example_inputs,
        recording_coordinator=recording_coordinator,
        tmp_path=tmp_path,
    )
    assert result.returncode == 2
    assert not arguments
    assert "mode must be ce, entropy, or combined" in result.stderr


@pytest.mark.parametrize(
    ("mode", "extra", "message"),
    (
        ("entropy", (), "--lambda-dyn is required"),
        ("entropy", ("--lambda-dyn", "0"), "--lambda-dyn must be positive"),
        ("entropy", ("--lambda-dyn", "nan"), "--lambda-dyn must be positive"),
        ("combined", ("--lambda-dyn", "0.4"), "--lambda-vis is required"),
        (
            "combined",
            ("--lambda-dyn", "0.4", "--lambda-vis", "-1"),
            "--lambda-vis must be positive",
        ),
        ("ce", ("--lambda-dyn", "0.4"), "does not accept auxiliary weights"),
    ),
)
def test_invalid_mode_weights_are_rejected_before_the_coordinator(
    mode, extra, message, example_inputs, recording_coordinator, tmp_path
):
    result, arguments, _, _ = _run_wrapper(
        mode,
        example_inputs=example_inputs,
        recording_coordinator=recording_coordinator,
        tmp_path=tmp_path,
        extra=extra,
    )
    assert result.returncode == 2
    assert not arguments
    assert message in result.stderr


@pytest.mark.parametrize("port", ("", "1023", "65536", "29.5", "port"))
def test_invalid_port_is_rejected_before_the_coordinator(
    port, example_inputs, recording_coordinator, tmp_path
):
    result, arguments, _, _ = _run_wrapper(
        "ce",
        example_inputs=example_inputs,
        recording_coordinator=recording_coordinator,
        tmp_path=tmp_path,
        port=port,
    )
    assert result.returncode == 2
    assert not arguments
    assert "--main-process-port must be an integer between 1024 and 65535" in result.stderr


@pytest.mark.parametrize("max_steps", ("", "0", "-1", "42.3", "steps"))
def test_invalid_update_budget_is_rejected_before_the_coordinator(
    max_steps, example_inputs, recording_coordinator, tmp_path
):
    result, arguments, _, _ = _run_wrapper(
        "ce",
        example_inputs=example_inputs,
        recording_coordinator=recording_coordinator,
        tmp_path=tmp_path,
        max_steps=max_steps,
    )
    assert result.returncode == 2
    assert not arguments
    assert "--max-steps must be a positive integer" in result.stderr


def test_missing_required_option_is_rejected_before_the_coordinator(
    example_inputs, recording_coordinator, tmp_path
):
    stage1, _, image_root = example_inputs
    argv_path = tmp_path / "missing-option-argv"
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "ce",
            "--stage1",
            str(stage1),
            "--image-root",
            str(image_root),
            "--run-id",
            f"objective-example-{uuid.uuid4().hex}",
            "--max-steps",
            "423",
            "--main-process-port",
            "29871",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": os.environ["PATH"],
            "ILVR_STAGE2_OBJECTIVE_COORDINATOR": str(recording_coordinator),
            "OBJECTIVE_EXAMPLE_ARGV": str(argv_path),
            "OBJECTIVE_EXAMPLE_ENVIRONMENT": str(tmp_path / "missing-option-env"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--data is required" in result.stderr
    assert not argv_path.exists()


def test_missing_or_unreadable_inputs_are_rejected_before_the_coordinator(
    example_inputs, recording_coordinator, tmp_path
):
    stage1, data_path, image_root = example_inputs
    missing_data = data_path.with_name("missing.jsonl")
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "ce",
            "--stage1",
            str(stage1),
            "--data",
            str(missing_data),
            "--image-root",
            str(image_root),
            "--run-id",
            f"objective-example-{uuid.uuid4().hex}",
            "--max-steps",
            "423",
            "--main-process-port",
            "29871",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": os.environ["PATH"],
            "ILVR_STAGE2_OBJECTIVE_COORDINATOR": str(recording_coordinator),
            "OBJECTIVE_EXAMPLE_ARGV": str(tmp_path / "missing-input-argv"),
            "OBJECTIVE_EXAMPLE_ENVIRONMENT": str(tmp_path / "missing-input-env"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--data must be a readable file" in result.stderr
    assert not (tmp_path / "missing-input-argv").exists()


def test_existing_run_destination_is_rejected_before_the_coordinator(
    example_inputs, recording_coordinator, tmp_path
):
    run_id = f"objective-collision-{uuid.uuid4().hex}"
    collision = REPO_ROOT / "checkpoints" / run_id
    collision.mkdir()
    try:
        result, arguments, _, _ = _run_wrapper(
            "ce",
            example_inputs=example_inputs,
            recording_coordinator=recording_coordinator,
            tmp_path=tmp_path,
            run_id=run_id,
        )
        assert result.returncode == 2
        assert not arguments
        assert "fresh run destination already exists" in result.stderr
    finally:
        collision.rmdir()


def test_dangling_run_destination_symlink_is_a_collision(
    example_inputs, recording_coordinator, tmp_path
):
    run_id = f"objective-symlink-{uuid.uuid4().hex}"
    collision = REPO_ROOT / "checkpoints" / run_id
    collision.symlink_to(tmp_path / "absent-target")
    try:
        result, arguments, _, _ = _run_wrapper(
            "ce",
            example_inputs=example_inputs,
            recording_coordinator=recording_coordinator,
            tmp_path=tmp_path,
            run_id=run_id,
        )
        assert result.returncode == 2
        assert not arguments
        assert "fresh run destination already exists" in result.stderr
    finally:
        collision.unlink()
