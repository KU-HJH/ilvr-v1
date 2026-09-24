"""Explicit CE-off (ce_weight=0) context preference, with CE=1 bitwise unchanged."""

import math

import pytest
import torch

from step_preference.config import StepPreferenceConfig
from step_preference.equivalence import _trainer
from utils_deepseed import parse_args


def context_config(**kwargs):
    return StepPreferenceConfig(enabled=True, mode="stage2b", stage2a_checkpoint_path="tiny-stage2a",
        sidecar_attention_backend="dense", visual_only_context_preference=True, beta=1.0,
        lambda_preference=1.0, lambda_entropy=0.0, **kwargs)


def reduce(lambda_ce):
    trainer = _trainer(context_config(lambda_ce=lambda_ce))
    clean_ce = torch.tensor(2.0, requires_grad=True)
    clean_score = torch.tensor(-1.0, requires_grad=True)
    visual_off = torch.tensor(-2.0, requires_grad=True)
    total, metrics = trainer._reduce_context_objective(
        clean_ce, [clean_score], [torch.tensor([0.2])],
        {(0, "visual"): {"mean_logp": visual_off, "entropies": torch.tensor([0.5])}})
    return total, metrics, clean_ce


def test_ce_weight_zero_trains_preference_only_but_still_reports_ce():
    # Fails if CE is still added (the trainer previously hard-coded weight 1).
    total, metrics, clean_ce = reduce(0.0)
    visual_loss = math.log1p(math.exp(-1.0))
    assert float(total.detach()) == pytest.approx(visual_loss)
    assert float(metrics["loss/ce"]) == pytest.approx(2.0)
    total.backward()
    assert clean_ce.grad is None or float(clean_ce.grad) == 0.0


def test_ce_weight_one_keeps_existing_total_exactly():
    total, metrics, clean_ce = reduce(1.0)
    # Bitwise equal to the former hard-coded `clean_ce + 1.0 * preference + 0.0 * entropy`.
    former = clean_ce + 1.0 * metrics["loss/context_preference"] + 0.0 * metrics["loss/entropy"]
    assert torch.equal(total.detach(), former.detach())
    total.backward()
    assert float(clean_ce.grad) == 1.0


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_context_preference_accepts_only_ce_weight_zero_or_one(weight):
    assert context_config(lambda_ce=weight).lambda_ce == weight


@pytest.mark.parametrize("weight", [0.5, 2.0])
def test_context_preference_rejects_other_ce_weights(weight):
    with pytest.raises(ValueError, match="ce_weight"):
        context_config(lambda_ce=weight)


def test_external_profile_keeps_explicit_ce_weight_and_defaults_to_one():
    base = ["--stage", "stage2", "--stage2_execution_profile", "latest_external_until"]
    assert parse_args(base).ce_weight == 1.0
    assert parse_args(base).step_preference_config.lambda_ce == 1.0
    off = parse_args(base + ["--ce_weight", "0"])
    assert off.ce_weight == 0.0
    assert off.step_preference_config.lambda_ce == 0.0
    assert off.step_preference_config.lambda_preference == 1.0
    assert off.step_preference_config.beta == 0.5


def test_ce_weight_zero_is_rejected_outside_clean_prefix():
    # evidence_replacement reduces CE at a fixed weight; CE-off there would be silent.
    with pytest.raises(ValueError, match="ce_weight 0"):
        StepPreferenceConfig(enabled=True, mode="stage2b", stage2a_checkpoint_path="tiny-stage2a",
            sidecar_attention_backend="dense", visual_only_context_preference=True,
            prefix_mode="evidence_replacement", lambda_entropy=0.0, lambda_ce=0.0)


@pytest.mark.parametrize("ce_weight, accepted", [("0", True), ("1", True), ("0.5", False)])
def test_launcher_accepts_context_ce_weight_zero_or_one(ce_weight, accepted):
    from tests.stage2.test_launcher import option_value, run_launcher

    result = run_launcher(env_overrides={
        "STEP_PREFERENCE_ENABLED": "true", "STEP_PREFERENCE_MODE": "stage2b",
        "STEP_PREFERENCE_OBJECTIVE": "context_preference",
        "STAGE2A_CHECKPOINT_PATH": "/models/stage2a", "CE_WEIGHT": ce_weight,
    })
    if accepted:
        assert result.returncode == 0, result.stderr
        import shlex
        assert option_value(shlex.split(result.stdout.strip()), "--ce_weight") == ce_weight
    else:
        assert result.returncode != 0 and "CE_WEIGHT" in result.stderr
