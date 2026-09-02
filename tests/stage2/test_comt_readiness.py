import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest
import torch
from PIL import Image

import stage2.data as data_module


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "run_stage2_training.sh"
REQUIRED_ENV = {
    "STAGE1_CHECKPOINT": "/models/stage 1",
    "DATA_PATH": "/datasets/train data.jsonl",
    "IMAGE_ROOT": "/datasets/images root",
    "OUTPUT_DIR": "/outputs/stage 2",
    "LOG_FILE": "/logs/stage 2.log",
}


def run_launcher(*args, env_overrides=None):
    env = {"PATH": os.environ["PATH"], "STAGE2_DRY_RUN": "true", **REQUIRED_ENV}
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args], env=env, capture_output=True, text=True
    )


def resolved_launcher_arguments(result):
    assert result.returncode == 0, result.stderr
    tokens = shlex.split(result.stdout.strip())
    return tokens[tokens.index(str(REPO_ROOT / "src/main.py")) + 1 :]


def test_stage2_launcher_dry_run_exposes_reproduction_defaults():
    """Catches a reproduction-critical default being omitted from the real command."""
    arguments = resolved_launcher_arguments(run_launcher())

    assert arguments == [
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
        "--stage2_diagnostics_interval",
        "0",
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
        "--tuning_method",
        "full",
        "--lora_r",
        "32",
        "--lora_alpha",
        "64",
        "--lora_dropout",
        "0.05",
        "--lora_target_modules",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    ]


def test_stage2_launcher_dry_run_resolves_explicit_reproduction_settings():
    """Catches environment overrides being ignored or mapped to the wrong CLI flags."""
    arguments = resolved_launcher_arguments(
        run_launcher(
            env_overrides={
                "LATENT_SIZE": "12",
                "STAGE2_DIAGNOSTICS_INTERVAL": "4",
                "SUPERVISE_LATENT_BOUNDARIES": "false",
                "LEARNING_RATE": "2e-5",
                "GRADIENT_ACCUMULATION_STEPS": "16",
                "ATTENTION_IMPLEMENTATION": "flash_attention_2",
                "GRADIENT_CHECKPOINTING": "true",
                "MAX_SEQUENCE_LENGTH": "4096",
                "DEEPSPEED_CONFIG": "/configs/zero 3.json",
                "RESUME_CHECKPOINT": "/outputs/checkpoint 20",
            }
        )
    )

    assert arguments[-20:] == [
        "--latent_size",
        "12",
        "--stage2_diagnostics_interval",
        "4",
        "--no-supervise_latent_boundaries",
        "--learning_rate",
        "2e-5",
        "--gradient_accumulation_steps",
        "16",
        "--per_device_train_batch_size",
        "1",
        "--attention_implementation",
        "flash_attention_2",
        "--gradient_checkpointing",
        "--max_seq_length_train",
        "4096",
        "--resume_from_checkpoint_path",
        "/outputs/checkpoint 20",
        "--deepspeed_config",
        "/configs/zero 3.json",
    ]


def test_stage2_launcher_rejects_invalid_paired_boundary_setting():
    """Catches malformed boundary policy being forwarded ambiguously to training."""
    result = run_launcher(env_overrides={"SUPERVISE_LATENT_BOUNDARIES": "start-only"})

    assert result.returncode == 2
    assert "SUPERVISE_LATENT_BOUNDARIES must be true or false" in result.stderr


def test_stage2_launcher_rejects_invalid_checkpointing_setting():
    """Catches a non-Boolean checkpointing value selecting an unintended backend."""
    result = run_launcher(env_overrides={"GRADIENT_CHECKPOINTING": "sometimes"})

    assert result.returncode == 2
    assert "GRADIENT_CHECKPOINTING must be true or false" in result.stderr


def test_stage2_launcher_rejects_unknown_attention_backend():
    """Catches an unsupported attention backend reaching an expensive model startup."""
    result = run_launcher(env_overrides={"ATTENTION_IMPLEMENTATION": "magic"})

    assert result.returncode == 2
    assert "ATTENTION_IMPLEMENTATION must be eager, sdpa, or flash_attention_2" in result.stderr


def test_stage2_launcher_rejects_non_positive_integer_settings():
    """Catches invalid rollout, accumulation, or sequence sizes before training starts."""
    for name in ("LATENT_SIZE", "GRADIENT_ACCUMULATION_STEPS", "MAX_SEQUENCE_LENGTH"):
        result = run_launcher(env_overrides={name: "0"})

        assert result.returncode == 2
        assert f"{name} must be a positive integer" in result.stderr


@pytest.mark.parametrize("value", ("-1", "1.5", "often"))
def test_stage2_launcher_rejects_invalid_diagnostics_interval(value):
    """Catches malformed diagnostic cadences reaching Python startup."""
    result = run_launcher(
        env_overrides={"STAGE2_DIAGNOSTICS_INTERVAL": value}
    )

    assert result.returncode == 2
    assert (
        "STAGE2_DIAGNOSTICS_INTERVAL must be a non-negative integer"
        in result.stderr
    )


def test_stage2_launcher_rejects_invalid_learning_rate():
    """Catches a malformed or zero learning rate before training starts."""
    for value in ("fast", "0", "-1", "1e999", "1e-9999", "nan", "inf", "-inf"):
        result = run_launcher(env_overrides={"LEARNING_RATE": value})

        assert result.returncode == 2
        assert "LEARNING_RATE must be a positive number" in result.stderr


MANAGED_OPTIONS = (
    "--stage",
    "--load_model_path",
    "--data_path",
    "--image_root",
    "--save_model_path",
    "--log_file",
    "--latent_size",
    "--stage2_diagnostics_interval",
    "--supervise_latent_boundaries",
    "--no-supervise_latent_boundaries",
    "--learning_rate",
    "--gradient_accumulation_steps",
    "--per_device_train_batch_size",
    "--attention_implementation",
    "--gradient_checkpointing",
    "--no-gradient_checkpointing",
    "--max_seq_length_train",
    "--deepspeed_config",
    "--resume_from_checkpoint_path",
    "--tuning_method",
    "--lora_r",
    "--lora_alpha",
    "--lora_dropout",
    "--lora_target_modules",
)


@pytest.mark.parametrize("option", MANAGED_OPTIONS)
@pytest.mark.parametrize("syntax", ("separate", "equals"))
def test_stage2_launcher_rejects_trailing_managed_options(option, syntax):
    """Catches trailing argparse values replacing launcher-owned invariants."""
    arguments = [option, "override"] if syntax == "separate" else [f"{option}=override"]

    result = run_launcher(*arguments)

    assert result.returncode == 2
    assert f"launcher-owned option cannot be passed directly: {option}" in result.stderr


@pytest.mark.parametrize(
    "arguments, abbreviated_option",
    [
        (["--sta", "stage1"], "--sta"),
        (["--load_model_p=/wrong/model"], "--load_model_p"),
        (["--per_device_train_b", "2"], "--per_device_train_b"),
    ],
)
def test_stage2_launcher_rejects_argparse_abbreviations_of_managed_options(
    arguments, abbreviated_option
):
    """Catches argparse long-option abbreviation bypassing invariant ownership."""
    result = run_launcher(*arguments)

    assert result.returncode == 2
    assert (
        f"launcher-owned option cannot be passed directly: {abbreviated_option}"
        in result.stderr
    )


def test_stage2_launcher_preserves_unmanaged_passthrough_and_quoting():
    """Catches invariant protection dropping or splitting genuine passthrough options."""
    result = run_launcher(
        "--max_steps",
        "7",
        "--evaluation_output_path",
        "/reports/stage 2.json",
    )

    arguments = resolved_launcher_arguments(result)
    assert arguments[-4:] == [
        "--max_steps",
        "7",
        "--evaluation_output_path",
        "/reports/stage 2.json",
    ]


class FakeProcessor:
    def __init__(self):
        self.lengths = iter((4, 12))
        self.calls = []

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        return conversation[1]["content"][0]["text"]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        length = next(self.lengths)
        return {"input_ids": torch.zeros((1, length), dtype=torch.long)}


def test_comt_audit_cli_reports_untruncated_lengths_and_block_distribution(
    tmp_path, monkeypatch, capsys
):
    """Catches audit token truncation, helper-image loading, or unstable summary fields."""
    for name in ("first.png", "second.png"):
        Image.new("RGB", (4, 4)).save(tmp_path / name)
    samples = [
        {
            "image_input": "first.png",
            "text_input": "first question",
            "sequence_plan": [
                {"type": "latent", "helper_image": "unused-1.png"},
                {"type": "text", "content": "answer"},
            ],
        },
        {
            "image_input": "second.png",
            "text_input": "second question",
            "sequence_plan": [
                {"type": "latent", "helper_image": f"unused-{index}.png"}
                for index in range(6)
            ],
        },
    ]
    data_path = tmp_path / "TRAIN.jsonl"
    data_path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8"
    )
    processor = FakeProcessor()

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, path):
            assert path == "/checkpoints/stage 1"
            return processor

    monkeypatch.setattr(data_module, "AutoProcessor", FakeAutoProcessor, raising=False)

    result = data_module.main(
        [
            "--data-path",
            str(data_path),
            "--image-root",
            str(tmp_path),
            "--processor-path",
            "/checkpoints/stage 1",
            "--latent-size",
            "8",
            "--max-sequence-length",
            "8",
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {
        "filtered": 1,
        "filtered_percentage": 50.0,
        "latent_block_distribution": {"1": 1, "6": 1},
        "maximum": 12,
        "median": 8.0,
        "minimum": 4,
        "p90": 11.2,
        "p95": 11.6,
        "p99": 11.92,
        "retained": 1,
        "total": 2,
    }
    assert len(processor.calls) == 2
    assert all(call["truncation"] is False for call in processor.calls)
    assert all(
        isinstance(call["images"][0][0], Image.Image) for call in processor.calls
    )
