from dataclasses import FrozenInstanceError

import pytest

from step_preference.config import StepPreferenceConfig, split_stage2_budget
from utils_deepseed import parse_args


def test_split_stage2_budget_uses_floor_and_preserves_positive_total():
    """Catches rounding drift or a zero-length phase in a valid small split."""
    assert split_stage2_budget(5, 0.5) == (2, 3)
    assert split_stage2_budget(3, 0.34) == (1, 2)


@pytest.mark.parametrize(
    ("total", "ratio"),
    [(1, 0.5), (2, 0.0), (2, 1.0), (2, -0.1), (2, 1.1), (2, float("nan"))],
)
def test_split_stage2_budget_rejects_invalid_or_empty_phase(total, ratio):
    """Catches invalid split settings reaching a phase with no optimizer step."""
    with pytest.raises(ValueError):
        split_stage2_budget(total, ratio)


def test_step_preference_config_is_immutable_and_exposes_normative_defaults():
    """Catches mutable run semantics or silent changes to the opt-in defaults."""
    config = StepPreferenceConfig()

    assert config.enabled is False
    assert config.mode == "stage2a"
    assert config.beta == 0.1
    assert config.lambda_ce == 1.0
    assert config.lambda_dpo == 1.0
    assert config.lambda_entropy == 0.1
    assert config.entropy_margin == 0.1
    assert config.stage2a_ratio == 0.5
    assert config.total_steps is None
    assert config.stage2a_checkpoint_path is None
    assert config.negative_path is None
    assert config.negative_manifest is None
    assert config.packing_enabled is False
    with pytest.raises(FrozenInstanceError):
        config.mode = "stage2b"


def test_training_cli_exposes_only_disabled_stage2a_defaults():
    """Catches Stage 2B becoming implicit or changing ordinary Stage 2 parsing."""
    args = parse_args(["--stage", "stage2"])

    assert args.step_preference_enabled is False
    assert args.step_preference_mode == "stage2a"
    assert args.step_negative_path is None
    assert args.step_negative_manifest is None
    assert args.step_dpo_beta == 0.1
    assert args.ce_weight == 1.0
    assert args.step_dpo_weight == 1.0
    assert args.step_entropy_weight == 0.1
    assert args.step_entropy_margin == 0.1
    assert args.stage2a_ratio == 0.5
    assert args.stage2_total_steps is None
    assert args.stage2a_checkpoint_path is None
    assert args.step_preference_packing is False


def test_training_cli_maps_zero_ce_weight_into_step_preference_config():
    """Catches the public CE coefficient being ignored by Stage 2B."""
    args = parse_args(["--stage", "stage2", "--ce_weight", "0"])

    assert args.step_preference_config.lambda_ce == 0.0


def test_packed_stage2b_configuration():
    """Catches packed execution leaking into disabled or Stage 2A training."""
    required = [
        "--stage", "stage2", "--step_preference_enabled",
        "--step_preference_mode", "stage2b",
        "--stage2a_checkpoint_path", "checkpoint",
        "--step_negative_path", "negatives.jsonl",
        "--step_negative_manifest", "manifest.json",
        "--step_preference_packing",
    ]

    args = parse_args(required)
    assert args.step_preference_config.packing_enabled is True

    with pytest.raises(ValueError, match="packing.*Stage 2B"):
        StepPreferenceConfig(packing_enabled=True)

    checkpointed_args = parse_args(required + ["--gradient_checkpointing"])
    assert checkpointed_args.step_preference_config.packing_enabled is True
    assert checkpointed_args.gradient_checkpointing is True


def test_stage2b_requires_checkpoint_and_both_negative_paths():
    """Catches a Stage 2B run starting without all frozen initialization inputs."""
    base = ["--stage", "stage2", "--step_preference_enabled", "--step_preference_mode", "stage2b"]
    with pytest.raises(ValueError, match="requires.*stage2a_checkpoint_path"):
        parse_args(base)
    with pytest.raises(ValueError, match="requires.*step_negative_path"):
        parse_args(base + ["--stage2a_checkpoint_path", "checkpoint"])
    with pytest.raises(ValueError, match="requires.*step_negative_manifest"):
        parse_args(
            base
            + [
                "--stage2a_checkpoint_path",
                "checkpoint",
                "--step_negative_path",
                "negatives.jsonl",
            ]
        )


@pytest.mark.parametrize("backend", ["flash_attention_2"])
def test_stage2b_rejects_unsupported_attention_or_stage2_resume(backend):
    """Catches ignored edge masks or Stage 2A optimizer state entering Stage 2B."""
    required = [
        "--stage",
        "stage2",
        "--step_preference_enabled",
        "--step_preference_mode",
        "stage2b",
        "--stage2a_checkpoint_path",
        "checkpoint",
        "--step_negative_path",
        "negatives.jsonl",
        "--step_negative_manifest",
        "manifest.json",
    ]
    with pytest.raises(ValueError, match="eager or sdpa"):
        parse_args(required + ["--attention_implementation", backend])
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_args(required + ["--resume_from_checkpoint_path", "checkpoint-2b"])


@pytest.mark.parametrize(
    "updates",
    [
        {"beta": 0.0},
        {"beta": float("inf")},
        {"lambda_dpo": float("nan")},
        {"lambda_entropy": float("inf")},
        {"entropy_margin": -0.1},
        {"mode": "other"},
    ],
)
def test_step_preference_config_rejects_invalid_objective_settings(updates):
    """Catches a non-finite objective or unknown lifecycle mode at construction."""
    with pytest.raises(ValueError):
        StepPreferenceConfig(**updates)


@pytest.mark.parametrize("field", ("lambda_ce", "lambda_dpo", "lambda_entropy"))
@pytest.mark.parametrize("value", (-0.1, float("nan"), float("inf")))
def test_step_preference_config_rejects_invalid_objective_coefficients(field, value):
    """Catches negative or non-finite coefficients reaching objective reduction."""
    with pytest.raises(ValueError):
        StepPreferenceConfig(**{field: value})


def test_optional_total_budget_selects_the_current_phase_steps():
    """Catches max_steps using the whole budget in either explicit phase."""
    common = dict(enabled=True, total_steps=9, stage2a_ratio=0.4)
    assert StepPreferenceConfig(mode="stage2a", **common).phase_steps == 3
    assert StepPreferenceConfig(
        mode="stage2b",
        stage2a_checkpoint_path="checkpoint",
        negative_path="negatives.jsonl",
        negative_manifest="manifest.json",
        **common,
    ).phase_steps == 6


def test_disabled_total_budget_does_not_change_existing_stage2_steps():
    """Catches an inactive preference option altering ordinary Stage 2 training."""
    assert StepPreferenceConfig(enabled=False, total_steps=9).phase_steps is None
