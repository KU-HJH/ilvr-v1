import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from main import (
    build_stage2_training_arguments,
    validate_stage2_distributed_settings,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _stage2_args(**overrides):
    values = {
        "save_model_path": "out",
        "epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "warm_up_steps": 0,
        "learning_rate": 1e-4,
        "save_steps": 1,
        "max_seq_length_train": 32,
        "max_steps": 2,
        "gradient_checkpointing": False,
        "deepspeed_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("local_batch_size", [0, 2])
def test_stage2_rejects_non_unit_rank_local_batch(local_batch_size):
    """Catches a rank silently receiving zero or multiple Stage 2 samples."""
    with pytest.raises(
        ValueError,
        match="train_micro_batch_size_per_gpu must resolve to 1",
    ):
        validate_stage2_distributed_settings(local_batch_size)


def test_stage2_accepts_unit_rank_local_batch_with_accumulation():
    """Catches validation confusing effective global batch with rank-local batch."""
    validate_stage2_distributed_settings(1)

    args = build_stage2_training_arguments(
        _stage2_args(gradient_accumulation_steps=4)
    )

    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 4
    assert args.deepspeed is None


def test_stage2_training_arguments_enable_deepspeed_only_when_selected():
    """Catches Stage 2 forcing ZeRO-3 or selecting a non-reloadable ZeRO config."""
    args = build_stage2_training_arguments(
        _stage2_args(deepspeed_config=str(REPO_ROOT / "configs/config_stage3.json"))
    )

    assert args.deepspeed == str(REPO_ROOT / "configs/config_stage3.json")
    assert (
        args.hf_deepspeed_config.get_value(
            "zero_optimization.stage3_gather_16bit_weights_on_model_save"
        )
        is True
    )


def _dry_run_launcher(*, deepspeed_config=None):
    env = os.environ.copy()
    env.update(
        {
            "STAGE1_CHECKPOINT": "/tmp/stage1",
            "DATA_PATH": "/tmp/train.jsonl",
            "IMAGE_ROOT": "/tmp/images",
            "OUTPUT_DIR": "/tmp/out",
            "LOG_FILE": "/tmp/train.log",
            "STAGE2_DRY_RUN": "true",
        }
    )
    if deepspeed_config is not None:
        env["DEEPSPEED_CONFIG"] = deepspeed_config
    else:
        env.pop("DEEPSPEED_CONFIG", None)
    return subprocess.run(
        [str(REPO_ROOT / "run_stage2_training.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_stage2_launcher_passes_optional_deepspeed_selection_by_behavior():
    """Catches the launcher forcing or ignoring the optional DeepSpeed config."""
    without_deepspeed = _dry_run_launcher()
    assert without_deepspeed.returncode == 0, without_deepspeed.stderr
    assert "--deepspeed_config" not in without_deepspeed.stdout

    config = str(REPO_ROOT / "configs/config_stage3.json")
    with_deepspeed = _dry_run_launcher(deepspeed_config=config)
    assert with_deepspeed.returncode == 0, with_deepspeed.stderr
    assert f"--deepspeed_config {config}" in with_deepspeed.stdout


@pytest.mark.skipif(
    os.environ.get("RUN_STAGE2_TINY_TRAIN") != "1",
    reason="set RUN_STAGE2_TINY_TRAIN=1 for the save/resume witness",
)
def test_tiny_stage2_harness_restores_full_training_state(tmp_path):
    """Catches resume restarting optimizer, scheduler, or global step from zero."""
    witness_path = tmp_path / "witness.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tests/stage2/integration/tiny_stage2_train.py"),
            "--output-dir",
            str(tmp_path / "train"),
            "--witness-path",
            str(witness_path),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    witness = json.loads(witness_path.read_text())
    assert witness["first_global_step"] == 1
    assert witness["resumed_global_step"] == 2
    assert witness["optimizer_state_restored"] is True
    assert witness["scheduler_state_restored"] is True
    assert witness["optimizer_step_at_resume"] == 1
    assert witness["scheduler_step_at_resume"] == 1
    assert witness["observed_local_batch_sizes"]
    assert set(witness["observed_local_batch_sizes"]) == {1}
    assert math.isfinite(witness["first_loss"])
    assert math.isfinite(witness["resumed_loss"])
