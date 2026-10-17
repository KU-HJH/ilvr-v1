"""DIL-BCE entry contracts through the real configuration parser."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from step_preference.config import StepPreferenceConfig
from step_preference.dil_config import (
    uses_entropy_selection,
    validate_dil_runtime,
)
from utils_deepseed import parse_args


def _bce_config(**updates):
    return StepPreferenceConfig(**(dict(
        enabled=True,
        mode="stage2b",
        objective="context_preference",
        direct_from_stage1=True,
        shared_prefix_enabled=True,
        visual_only_context_preference=True,
        visual_off_suppress_latent_boundaries=False,
        lambda_ce=1.0,
        lambda_entropy=0.0,
        preference_loss_type="dil_bce_rf_topk",
        preference_topk=8,
        lambda_topk_mass=1.0,
    ) | updates))


def test_real_parser_resolves_all_dil_fields():
    """A dropped CLI option or wrong beta destination would launch a different loss."""
    args = parse_args([
        "--stage", "stage2", "--step_preference_enabled",
        "--step_preference_mode", "stage2b", "--stage2b_direct_from_stage1",
        "--stage1_adapter_path", "checkpoint",
        "--step_shared_prefix_enabled", "--no-step_visual_off_suppress_latent_boundaries",
        "--step_entropy_weight", "0", "--step_preference_beta", ".5",
        "--step_preference_weight", "1", "--step_preference_loss_type", "dil_bce_rf_topk",
        "--step_preference_topk", "8", "--step_preference_lambda_topk_mass", "1",
        "--step_preference_entropy_top_percent", "20",
        "--step_preference_entropy_source", "voff",
    ])
    config = StepPreferenceConfig.from_args(args)
    assert args.step_dpo_beta == .5
    assert config.beta == .5 and config.lambda_preference == 1.
    assert (config.preference_topk, config.lambda_topk_mass) == (8, 1.)
    assert (config.preference_entropy_top_percent, config.preference_entropy_source) == (20., "voff")
    with pytest.raises(ValueError, match="vocab|topk"):
        validate_dil_runtime(config, vocab_size=8, supervise_latent_boundaries=True)


def test_omitted_and_explicit_dpo_preserve_existing_fields():
    """Selecting the default explicitly must leave historic DPO configuration intact."""
    args = ["--stage", "stage2", "--step_preference_enabled", "--step_preference_mode", "stage2b",
            "--stage2a_checkpoint_path", "checkpoint"]
    omitted = parse_args(args).step_preference_config
    explicit = parse_args(args + ["--step_preference_loss_type", "dpo"]).step_preference_config
    assert omitted == explicit
    assert omitted.preference_loss_type == "dpo"


@pytest.mark.parametrize("percent,source,selected", [
    (None, "clean", False), (100., "clean", False), (20., "clean", True),
    (20., "voff", True), (100., "voff", False),
])
@pytest.mark.parametrize("scope", ["immediate_text", "until_next_latent"])
@pytest.mark.parametrize("mass,preference", [(0., 1.), (1., 1.), (1., 0.)])
def test_supported_topk_matrix(percent, source, selected, scope, mass, preference):
    """Valid selection, scope and coefficient variants must keep their semantics."""
    config = _bce_config(preference_entropy_top_percent=percent,
                         preference_entropy_source=source, target_scope=scope,
                         lambda_topk_mass=mass, lambda_preference=preference)
    assert uses_entropy_selection(config) is selected
    validate_dil_runtime(config, vocab_size=16, supervise_latent_boundaries=True)


def test_basic_bce_accepts_no_topk_options():
    """The reference-free BCE selector has no top-k dependency."""
    config = _bce_config(preference_loss_type="dil_bce_rf", preference_topk=None,
                         lambda_topk_mass=0.)
    assert not uses_entropy_selection(config)
    validate_dil_runtime(config, vocab_size=16, supervise_latent_boundaries=True)


@pytest.mark.parametrize("updates", [
    {"enabled": False}, {"mode": "stage2a", "direct_from_stage1": False},
    {"objective": "legacy_dpo", "negative_path": "n", "negative_manifest": "m"},
    {"visual_only_context_preference": False},
    {"prefix_mode": "v_off_prefix"}, {"visual_off_suppress_latent_boundaries": True},
    {"lambda_ce": -0.5}, {"lambda_entropy": 0.1}, {"preference_margin": 0.1},
    {"shared_prefix_enabled": False},
])
def test_bce_rejects_unsupported_training_contract(updates):
    """Each unsupported variant must fail during config construction."""
    with pytest.raises(ValueError):
        _bce_config(**updates)


@pytest.mark.parametrize("updates", [
    {"preference_topk": None}, {"preference_topk": 1}, {"preference_topk": True},
    {"lambda_topk_mass": -1.}, {"lambda_topk_mass": float("nan")},
    {"lambda_topk_mass": float("inf")}, {"lambda_topk_mass": True},
    {"preference_entropy_top_percent": 0.}, {"preference_entropy_top_percent": 101.},
    {"preference_entropy_top_percent": float("nan")},
    {"preference_entropy_top_percent": True},
    {"preference_entropy_source": "bad"},
    {"preference_entropy_source": "voff", "preference_entropy_top_percent": None},
    {"beta": True}, {"lambda_preference": True},
])
def test_bce_rejects_bad_knobs(updates):
    """Malformed DIL settings must never be silently coerced or ignored."""
    with pytest.raises(ValueError):
        _bce_config(**updates)


@pytest.mark.parametrize("selector,updates", [
    ("dpo", {"preference_topk": 8}),
    ("dpo", {"lambda_topk_mass": 1.}),
    ("dpo", {"preference_entropy_top_percent": 20.}),
    ("dil_bce_rf", {"preference_topk": 8}),
    ("dil_bce_rf", {"lambda_topk_mass": 1.}),
    ("dil_bce_rf", {"preference_entropy_top_percent": 20.}),
])
def test_non_topk_selectors_reject_stale_topk_settings(selector, updates):
    """Switching selectors must not leave an apparently effective stale knob."""
    base = _bce_config() if selector != "dpo" else StepPreferenceConfig()
    with pytest.raises(ValueError):
        replace(base, **(dict(preference_loss_type=selector, preference_topk=None,
                              lambda_topk_mass=0.) | updates))


def test_runtime_checks_actual_output_vocab_and_boundary_supervision():
    """The resized output head, rather than tokenizer size, bounds k before forward."""
    config = _bce_config()
    with pytest.raises(ValueError, match="vocab|topk"):
        validate_dil_runtime(config, vocab_size=8, supervise_latent_boundaries=True)
    with pytest.raises(ValueError, match="boundar"):
        validate_dil_runtime(config, vocab_size=16, supervise_latent_boundaries=False)
    validate_dil_runtime(config, vocab_size=16, supervise_latent_boundaries=True)


def test_main_model_hook_checks_resized_output_head():
    """Model setup must validate its real output width before trainer creation."""
    from main import validate_stage2_dil_model

    class TinyModel:
        def get_output_embeddings(self):
            return SimpleNamespace(weight=torch.empty(8, 2))

    with pytest.raises(ValueError, match="vocab|topk"):
        validate_stage2_dil_model(_bce_config(), TinyModel(),
                                  supervise_latent_boundaries=True)
