import os
import shlex
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "run_stage2_training.sh"
REQUIRED_ENV = {
    "STAGE1_CHECKPOINT": "/models/stage 1",
    "DATA_PATH": "/datasets/train data.jsonl",
    "IMAGE_ROOT": "/datasets/images root",
    "OUTPUT_DIR": "/outputs/stage 2",
    "LOG_FILE": "/logs/stage 2.log",
}


def run_launcher(*args, env_overrides=None, cwd=None):
    env = {"PATH": os.environ["PATH"], "STAGE2_DRY_RUN": "true", **REQUIRED_ENV}
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def dry_run_tokens(result):
    assert result.returncode == 0, result.stderr
    return shlex.split(result.stdout.strip())


def test_stage2_launcher_resolves_repo_runtime_and_propagates_arguments(tmp_path):
    """Catches cwd-dependent paths, wrong Stage 2 bindings, or lost CLI arguments."""
    result = run_launcher(
        "--max_steps",
        "1",
        "--attention_implementation",
        "sdpa",
        cwd=tmp_path,
    )

    tokens = dry_run_tokens(result)
    expected_pythonpath = f"{REPO_ROOT / 'transformers/src'}:{REPO_ROOT / 'src'}"
    assert tokens[:3] == [
        f"PYTHONPATH={expected_pythonpath}",
        "TOKENIZERS_PARALLELISM=false",
        str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
    ]
    assert tokens[3:] == [
        str(REPO_ROOT / "src/main.py"),
        "--stage",
        "stage2",
        "--load_model_path",
        REQUIRED_ENV["STAGE1_CHECKPOINT"],
        "--data_path",
        REQUIRED_ENV["DATA_PATH"],
        "--image_root",
        REQUIRED_ENV["IMAGE_ROOT"],
        "--save_model_path",
        REQUIRED_ENV["OUTPUT_DIR"],
        "--log_file",
        REQUIRED_ENV["LOG_FILE"],
        "--latent_size",
        "8",
        "--supervise_latent_boundaries",
        "--learning_rate",
        "1e-5",
        "--gradient_accumulation_steps",
        "8",
        "--per_device_train_batch_size",
        "1",
        "--attention_implementation",
        "sdpa",
        "--no-gradient_checkpointing",
        "--max_seq_length_train",
        "8192",
        "--max_steps",
        "1",
        "--attention_implementation",
        "sdpa",
    ]


def test_stage2_launcher_maps_resume_without_reinterpreting_stage1_origin():
    """Catches resume replacing the explicit Stage 1 provenance argument."""
    result = run_launcher(env_overrides={"RESUME_CHECKPOINT": "/resume/checkpoint 10"})

    tokens = dry_run_tokens(result)
    load_index = tokens.index("--load_model_path")
    resume_index = tokens.index("--resume_from_checkpoint_path")
    assert tokens[load_index + 1] == REQUIRED_ENV["STAGE1_CHECKPOINT"]
    assert tokens[resume_index + 1] == "/resume/checkpoint 10"


def test_stage2_launcher_omits_resume_for_fresh_training():
    """Catches an empty resume value being forwarded as a real checkpoint."""
    tokens = dry_run_tokens(run_launcher())

    assert "--resume_from_checkpoint_path" not in tokens


@pytest.mark.parametrize("missing_name", REQUIRED_ENV)
def test_stage2_launcher_fails_when_required_environment_is_missing(missing_name):
    """Catches silently falling back to ambiguous training paths."""
    env = {"PATH": os.environ["PATH"], "STAGE2_DRY_RUN": "true", **REQUIRED_ENV}
    del env[missing_name]

    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, capture_output=True, text=True
    )

    assert result.returncode != 0
    assert f"set {missing_name}" in result.stderr


def test_stage2_launcher_has_valid_bash_syntax():
    """Catches shell syntax regressions before a training process is launched."""
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
