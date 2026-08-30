from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from stage1 import Stage1RuntimeConfig, resolve_stage1_tuning_config
from utils_deepseed import parse_args


def test_stage1_dora_defaults_match_approved_experiment():
    """Catches the approved Stage 1 DoRA configuration drifting at the CLI."""
    args = parse_args(
        [
            "--stage",
            "stage1",
            "--tuning_method",
            "dora",
            "--model",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "--data_path",
            "/data/comt/TRAIN.jsonl",
            "--image_root",
            "/data/comt",
        ]
    )

    tuning = resolve_stage1_tuning_config(args)

    assert tuning.method == "dora"
    assert (tuning.rank, tuning.alpha, tuning.dropout) == (32, 64, 0.05)
    assert tuning.target_suffixes == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def test_stage1_legacy_use_lora_resolves_only_to_lora():
    """Catches legacy Stage 1 LoRA selection being silently promoted to DoRA."""
    tuning = resolve_stage1_tuning_config(
        parse_args(["--stage", "stage1", "--use_lora"])
    )

    assert tuning.method == "lora"


def test_textual_states_parser_defaults_accepts_all_and_rejects_invalid_values():
    """Catches textual history mode being dropped or accepted outside its contract."""
    assert parse_args(["--stage", "stage1"]).textual_states == "previous"
    assert (
        parse_args(
            ["--stage", "stage1", "--textual_states", "all"]
        ).textual_states
        == "all"
    )
    with pytest.raises(SystemExit):
        parse_args(["--stage", "stage1", "--textual_states", "invalid"])


def test_stage1_explicit_target_order_is_preserved_after_normalization():
    """Catches an explicit Stage 1 PEFT target order being replaced by defaults."""
    tuning = resolve_stage1_tuning_config(
        parse_args(
            [
                "--stage",
                "stage1",
                "--tuning_method",
                "dora",
                "--lora_target_modules",
                "v_proj,q_proj",
            ]
        )
    )

    assert tuning.target_suffixes == ("v_proj", "q_proj")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--stage", "stage1", "--use_lora", "--tuning_method", "dora"],
            "cannot be combined",
        ),
        (
            [
                "--stage",
                "stage2",
                "--stage1_adapter_path",
                "/a",
                "--resume_from_checkpoint_path",
                "/b",
            ],
            "mutually exclusive",
        ),
        (["--stage", "stage1", "--learning_rate", "nan"], "finite"),
        (
            ["--stage", "stage1", "--per_device_train_batch_size", "0"],
            "at least 1",
        ),
    ],
)
def test_cli_rejects_contradictory_or_nonfinite_values(argv, message):
    """Catches invalid Stage 1/2 launch settings reaching composition."""
    with pytest.raises(ValueError, match=message):
        parse_args(argv)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--stage", "stage1", "--lora_r", "0"], "rank and alpha must be positive"),
        (["--stage", "stage1", "--lora_alpha", "0"], "rank and alpha must be positive"),
        (["--stage", "stage1", "--lora_dropout", "nan"], "dropout must be finite"),
        (["--stage", "stage1", "--lora_dropout", "1.0"], "dropout must be finite"),
        (
            ["--stage", "stage1", "--gradient_accumulation_steps", "0"],
            "at least 1",
        ),
        (
            ["--stage", "stage1", "--tuning_method", "lora"],
            "accepts only full or dora",
        ),
    ],
)
def test_stage1_parser_rejects_invalid_tuning_or_runtime_values(argv, message):
    """Catches invalid Stage 1 settings passing parser validation."""
    with pytest.raises(ValueError, match=message):
        parse_args(argv)


def test_stage2_implicit_target_order_preserves_legacy_metadata_contract():
    """Catches Stage 1 defaults changing Stage 2 adapter metadata ordering."""
    args = parse_args(["--stage", "stage2"])

    assert args.lora_target_modules == (
        "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj"
    )


def test_stage1_save_contract_allows_probe_runs_to_skip_final_serialization():
    """Catches a no-save probe still requesting a final Stage 1 save."""
    args = parse_args(
        ["--stage", "stage1", "--save_strategy", "no", "--no_final_save"]
    )

    assert args.save_strategy == "no"
    assert args.final_save is False


def test_stage1_runtime_config_is_immutable():
    """Catches runtime launch inputs becoming mutable after validation."""
    runtime = Stage1RuntimeConfig(
        data_path=Path("TRAIN.jsonl"),
        image_root=Path("images"),
        output_dir=Path("checkpoints"),
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        epochs=2.0,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        seed=42,
        attention_implementation="sdpa",
        deepspeed_config=None,
    )

    with pytest.raises(FrozenInstanceError):
        runtime.seed = 7
