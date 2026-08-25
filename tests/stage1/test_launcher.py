from pathlib import Path
import shlex
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "run_stage1_training.sh"
LEGACY_LAUNCHER = REPO_ROOT / "run_training.sh"


def run_launcher(script, *args):
    return subprocess.run(
        [str(script), *map(str, args)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
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


def test_launcher_requires_all_explicit_sources():
    """Catches placeholder model/data/image/output values reaching training."""
    completed = run_launcher(LAUNCHER, "--dry-run", "--method", "dora")

    assert completed.returncode != 0
    assert "--model is required" in completed.stderr


def test_launcher_refuses_nonempty_output_without_explicit_resume(tmp_path):
    """Catches accidental overwrite or implicit launcher resume."""
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "evidence.txt").write_text("owned", encoding="utf-8")

    refused = run_launcher(LAUNCHER, *base_args(output))
    resumed = run_launcher(
        LAUNCHER,
        *base_args(output),
        "--resume",
        output / "checkpoint-7",
    )

    assert refused.returncode != 0
    assert "nonempty" in refused.stderr
    assert resumed.returncode == 0, resumed.stderr
    resumed_command = shlex.split(resumed.stdout.strip())
    index = resumed_command.index("--resume_from_checkpoint_path")
    assert resumed_command[index + 1] == str(output / "checkpoint-7")


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
