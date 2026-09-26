"""The diagnostic Vision-gain probe must match the objective's full-replay scores."""

import math

import pytest
import torch

from stage2.objective import Stage2ObjectiveConfig
from stage2.objective_runtime import Stage2ObjectiveRuntime

from .packed_m_helpers import TEXT_VOCABULARY, TOKENS, make_packed_inputs, make_packed_model


def _probe_api():
    from stage2 import vision_gain_probe
    return vision_gain_probe


def test_probe_scores_equal_eval_mode_full_replay_objective():
    """Catches the probe scoring other positions, dropout, or a different M."""
    api = _probe_api()
    model = make_packed_model(tuning="dora").eval()
    batches = [make_packed_inputs(blocks=2), make_packed_inputs(blocks=1)]
    runtime = Stage2ObjectiveRuntime(
        Stage2ObjectiveConfig(True, 0, 1, post_window=1, stage2_m_execution="full_replay"),
        TOKENS, 2, TEXT_VOCABULARY, None,
    )

    report = api.probe_batches(model, batches, token_ids=TOKENS, latent_size=2,
                               text_vocabulary=TEXT_VOCABULARY)

    expected = []
    with torch.no_grad():
        for inputs in batches:
            result = runtime.forward(model, inputs, runtime.select(inputs))
            for block in runtime.select(inputs).blocks:
                expected.append(block)
            assert result.metric_counts.get("S_F")
    assert report["blocks"] == len(expected) == 3
    per_block = report["per_block"]
    assert len(per_block) == 3
    for row in per_block:
        assert math.isfinite(row["S_F"]) and math.isfinite(row["S_M"])
        assert row["G"] == pytest.approx(row["S_F"] - row["S_M"], abs=1e-6)
    with torch.no_grad():
        sums = {"S_F": 0.0, "S_M": 0.0}
        for inputs in batches:
            result = runtime.forward(model, inputs, runtime.select(inputs))
            sums["S_F"] += float(result.metric_sums["S_F"])
            sums["S_M"] += float(result.metric_sums["S_M"])
    assert report["S_F"] == pytest.approx(sums["S_F"] / 3, abs=1e-5)
    assert report["S_M"] == pytest.approx(sums["S_M"] / 3, abs=1e-5)
    assert report["G_mean"] == pytest.approx(report["S_F"] - report["S_M"], abs=1e-5)
    assert report["G_gt_0"] == pytest.approx(sum(r["G"] > 0 for r in per_block) / 3)
    assert report["s_m_mode"] == "full_replay" and report["model_mode"] == "eval"


def test_probe_rejects_training_mode_models():
    """Catches dropout leaking into a diagnostic that must be deterministic."""
    api = _probe_api()
    model = make_packed_model(tuning="dora").train()
    with pytest.raises(ValueError, match="eval mode"):
        api.probe_batches(model, [make_packed_inputs(blocks=1)], token_ids=TOKENS,
                          latent_size=2, text_vocabulary=TEXT_VOCABULARY)
