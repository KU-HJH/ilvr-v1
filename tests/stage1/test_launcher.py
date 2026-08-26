from pathlib import Path
import json
import os
import shlex
import subprocess

import pytest

from stage1.benchmark import config_digest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "run_stage1_training.sh"
LEGACY_LAUNCHER = REPO_ROOT / "run_training.sh"


def run_launcher(script, *args, env=None):
    return subprocess.run(
        [str(script), *map(str, args)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def base_args(output_dir, *, method="dora"):
    return (
        "--dry-run",
        "--method",
        method,
        "--model",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "--data-path",
        "/missing/CoMT train;literal/TRAIN.jsonl",
        "--image-root",
        "/missing/CoMT images & literal",
        "--output-dir",
        output_dir,
    )


def live_args(tmp_path, output_dir, *, method="dora"):
    data_path = tmp_path / "TRAIN data.jsonl"
    data_path.write_text("{}\n", encoding="utf-8")
    image_root = tmp_path / "image root"
    image_root.mkdir(exist_ok=True)
    return (
        "--method",
        method,
        "--model",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "--data-path",
        data_path,
        "--image-root",
        image_root,
        "--output-dir",
        output_dir,
    )


def fake_accelerate_environment(tmp_path):
    """Return a harmless executable boundary and a log proving dispatch."""
    bin_dir = tmp_path / "fake bin"
    bin_dir.mkdir()
    log_path = tmp_path / "accelerate argv.log"
    executable = bin_dir / "accelerate"
    executable.write_text(
        "#!/usr/bin/env bash\nprintf '%q ' \"$@\" > \"$FAKE_ACCELERATE_LOG\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_ACCELERATE_LOG"] = str(log_path)
    return env, log_path


def test_dora_dry_run_emits_exact_safe_two_process_command_without_mutation(tmp_path):
    """Catches wrong experiment defaults, shell evaluation, and dry-run writes."""
    output = tmp_path / "output with spaces;$(touch SHOULD_NOT_EXIST)"

    completed = run_launcher(LAUNCHER, *base_args(output))

    assert completed.returncode == 0, completed.stderr
    assert not output.exists()
    assert not (REPO_ROOT / "SHOULD_NOT_EXIST").exists()
    command = shlex.split(completed.stdout.strip())
    assert command[:7] == [
        "accelerate",
        "launch",
        "--num_processes",
        "2",
        "--main_process_port",
        "29501",
        "src/main.py",
    ]
    expected_pairs = {
        "--stage": "stage1",
        "--tuning_method": "dora",
        "--model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "--data_path": "/missing/CoMT train;literal/TRAIN.jsonl",
        "--image_root": "/missing/CoMT images & literal",
        "--save_model_path": str(output),
        "--per_device_train_batch_size": "8",
        "--epochs": "3",
        "--learning_rate": "1e-5",
        "--gradient_accumulation_steps": "2",
        "--seed": "42",
        "--attention_implementation": "flash_attention_2",
        "--save_strategy": "no",
        "--lora_r": "32",
        "--lora_alpha": "64",
        "--lora_dropout": "0.05",
        "--lora_target_modules": (
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        ),
    }
    for flag, value in expected_pairs.items():
        index = command.index(flag)
        assert command[index + 1] == value
    deepspeed = Path(command[command.index("--deepspeed_config") + 1])
    assert deepspeed == REPO_ROOT / "configs/config_stage3.json"
    assert "--final_save" in command
    assert "--gradient_checkpointing" in command


def test_full_dry_run_omits_every_peft_argument(tmp_path):
    """Catches full tuning accidentally inheriting adapter configuration."""
    completed = run_launcher(
        LAUNCHER, *base_args(tmp_path / "full-output", method="full")
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout.strip())
    for flag in (
        "--use_lora",
        "--lora_r",
        "--lora_alpha",
        "--lora_dropout",
        "--lora_target_modules",
    ):
        assert flag not in command


def test_probe_flags_forward_exact_max_steps_without_final_save(tmp_path):
    """Catches a capacity probe accidentally training fully or serializing weights."""
    completed = run_launcher(
        LAUNCHER,
        *base_args(tmp_path / "probe output"),
        "--max-steps",
        "3",
        "--no-final-save",
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout)
    assert command[command.index("--max_steps") + 1] == "3"
    assert "--no_final_save" in command
    assert "--final_save" not in command


def test_timed_manifest_is_loaded_without_shell_evaluating_paths(tmp_path):
    """Catches the Task 17 manifest interface losing quoting or config binding."""
    output = tmp_path / "timed output;$(touch MANIFEST_INJECTION)"
    manifest = {
        "schema_version": 1,
        "kind": "stage1-timed-launch-manifest",
        "method": "dora",
        "tuning_config": {
            "use_dora": True, "rank": 32, "alpha": 64, "dropout": 0.05,
            "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        },
        "model": "model path;literal",
        "source_data_path": "/missing/data path;literal/TRAIN.jsonl",
        "data_path": "/missing/data path;literal/TRAIN.jsonl",
        "data_stat": {"device": 1, "inode": 2, "size": 3, "mtime_ns": 4},
        "image_root": "/missing/image root & literal",
        "output_dir": str(output),
        "micro_batch": 16,
        "world_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_global_batch": 64,
        "epochs": 3,
        "learning_rate": 1e-5,
        "seed": 42,
        "max_steps": 0,
        "attention_implementation": "flash_attention_2",
        "deepspeed_config": str(REPO_ROOT / "configs/config_stage3.json"),
        "deepspeed_config_digest": "d" * 64,
        "save_strategy": "no",
        "final_save": True,
        "audit_digest": "a" * 64,
        "probe_rows": "/missing/audit.json",
        "probe_row_indices": [7, 3],
        "maximum_config_digest": "b" * 64,
        "prerequisite_chain": "/smoke/prerequisite-chain.json",
        "prerequisite_chain_digest": "f" * 64,
    }
    manifest["config_digest"] = config_digest(manifest)
    manifest_path = tmp_path / "launch manifest;literal.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    completed = run_launcher(
        LAUNCHER, "--manifest", manifest_path, "--dry-run"
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout)
    assert command[command.index("--model") + 1] == "model path;literal"
    assert command[command.index("--save_model_path") + 1] == str(output)
    assert command[command.index("--per_device_train_batch_size") + 1] == "16"
    assert "--final_save" in command
    assert not (REPO_ROOT / "MANIFEST_INJECTION").exists()


@pytest.mark.parametrize("value", ["0", "-1", "3.5", "nan", "word"])
def test_launcher_rejects_invalid_explicit_max_steps(tmp_path, value):
    """Catches malformed probe limits reaching Trainer."""
    completed = run_launcher(
        LAUNCHER,
        *base_args(tmp_path / "probe output"),
        "--max-steps",
        value,
    )

    assert completed.returncode == 2
    assert "max_steps" in completed.stderr


def test_launcher_requires_all_explicit_sources():
    """Catches placeholder model/data/image/output values reaching training."""
    completed = run_launcher(LAUNCHER, "--dry-run", "--method", "dora")

    assert completed.returncode != 0
    assert "--model is required" in completed.stderr


def test_live_launcher_refuses_nonempty_output_without_explicit_resume(tmp_path):
    """Catches accidental overwrite or implicit launcher resume."""
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "evidence.txt").write_text("owned", encoding="utf-8")
    env, accelerate_log = fake_accelerate_environment(tmp_path)

    refused = run_launcher(
        LAUNCHER, *live_args(tmp_path, output), env=env
    )

    assert refused.returncode == 2
    assert "nonempty" in refused.stderr
    assert not accelerate_log.exists()


def test_live_launcher_allows_nonempty_output_only_with_explicit_resume(tmp_path):
    """Catches safe explicit resume being rejected by output collision checks."""
    output = tmp_path / "occupied"
    checkpoint = output / "checkpoint-7"
    checkpoint.mkdir(parents=True)
    env, accelerate_log = fake_accelerate_environment(tmp_path)

    resumed = run_launcher(
        LAUNCHER,
        *live_args(tmp_path, output),
        "--resume",
        checkpoint,
        env=env,
    )

    assert resumed.returncode == 0, resumed.stderr
    assert accelerate_log.is_file()
    resumed_command = shlex.split(accelerate_log.read_text(encoding="utf-8"))
    index = resumed_command.index("--resume_from_checkpoint_path")
    assert resumed_command[index + 1] == str(checkpoint)


@pytest.mark.parametrize("target_location", ["inside", "outside"])
@pytest.mark.parametrize("nonempty", [False, True])
def test_live_launcher_rejects_symlink_output_before_dispatch(
    tmp_path, target_location, nonempty
):
    """Catches output symlinks escaping the explicit experiment directory."""
    experiment_root = tmp_path / "experiment root"
    experiment_root.mkdir()
    target = (
        experiment_root / "inside target"
        if target_location == "inside"
        else tmp_path / "outside target"
    )
    target.mkdir()
    if nonempty:
        (target / "owned.txt").write_text("owned", encoding="utf-8")
    output = experiment_root / "linked output"
    output.symlink_to(target, target_is_directory=True)
    env, accelerate_log = fake_accelerate_environment(tmp_path)

    completed = run_launcher(
        LAUNCHER, *live_args(tmp_path, output), env=env
    )

    assert completed.returncode == 2
    assert "symlink" in completed.stderr
    assert not accelerate_log.exists()


@pytest.mark.parametrize("output_kind", ["nonempty", "symlink", "dangling"])
def test_dry_run_skips_all_filesystem_checks_and_creates_nothing(
    tmp_path, output_kind
):
    """Catches dry-run consulting or mutating model/data/output paths."""
    if output_kind == "nonempty":
        output = tmp_path / "occupied"
        output.mkdir()
        (output / "owned.txt").write_text("owned", encoding="utf-8")
    else:
        target = (
            tmp_path / "target"
            if output_kind == "symlink"
            else tmp_path / "missing-target"
        )
        if output_kind == "symlink":
            target.mkdir()
        output = tmp_path / "linked output"
        output.symlink_to(target, target_is_directory=True)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    completed = run_launcher(LAUNCHER, *base_args(output))

    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert completed.returncode == 0, completed.stderr
    assert after == before
    assert str(output) in shlex.split(completed.stdout)


@pytest.mark.parametrize("flag", ["--epochs", "--learning-rate"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "garbage"])
def test_launcher_rejects_invalid_positive_finite_scalars(tmp_path, flag, value):
    """Catches malformed/non-finite scalar values reaching Accelerate."""
    completed = run_launcher(
        LAUNCHER,
        *base_args(tmp_path / "output"),
        flag,
        value,
    )

    assert completed.returncode == 2
    assert "positive finite" in completed.stderr


@pytest.mark.parametrize(
    "environment_name", ["STAGE1_EPOCHS", "STAGE1_LEARNING_RATE"]
)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "garbage"])
def test_launcher_rejects_invalid_scalar_environment_overrides(
    tmp_path, environment_name, value
):
    """Catches environment overrides bypassing launcher scalar validation."""
    env = os.environ.copy()
    env[environment_name] = value

    completed = run_launcher(
        LAUNCHER, *base_args(tmp_path / "output"), env=env
    )

    assert completed.returncode == 2
    assert "positive finite" in completed.stderr


@pytest.mark.parametrize(
    ("epochs", "learning_rate"),
    [(".5", ".00001"), ("3e0", "1e-5"), ("2.5E+0", "2E-5")],
)
def test_launcher_preserves_valid_decimal_and_scientific_scalars(
    tmp_path, epochs, learning_rate
):
    """Catches validation normalizing or rejecting valid numeric spellings."""
    completed = run_launcher(
        LAUNCHER,
        *base_args(tmp_path / "output"),
        "--epochs",
        epochs,
        "--learning-rate",
        learning_rate,
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout)
    assert command[command.index("--epochs") + 1] == epochs
    assert command[command.index("--learning_rate") + 1] == learning_rate


def test_launcher_preserves_valid_scalar_environment_overrides(tmp_path):
    """Catches environment validation rewriting accepted numeric spellings."""
    env = os.environ.copy()
    env["STAGE1_EPOCHS"] = ".5"
    env["STAGE1_LEARNING_RATE"] = "2E-5"

    completed = run_launcher(
        LAUNCHER, *base_args(tmp_path / "output"), env=env
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout)
    assert command[command.index("--epochs") + 1] == ".5"
    assert command[command.index("--learning_rate") + 1] == "2E-5"


def test_legacy_launcher_forwards_use_lora_without_promoting_it_to_dora(tmp_path):
    """Catches the legacy --use_lora selector being dropped or changed to DoRA."""
    completed = run_launcher(
        LEGACY_LAUNCHER,
        "--dry-run",
        "--use_lora",
        "--model",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "--data-path",
        "/missing/train.jsonl",
        "--image-root",
        "/missing/images",
        "--output-dir",
        tmp_path / "legacy",
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout.strip())
    assert "--use_lora" in command
    assert "--tuning_method" not in command
