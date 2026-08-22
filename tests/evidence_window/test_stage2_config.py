import pytest

from evidence_window.stage2_config import (
    Stage2EvidenceWindowConfig,
    read_deepspeed_zero_stage,
)


def test_reference_free_ew_and_schedule():
    cfg = Stage2EvidenceWindowConfig.from_dict(
        {
            "version": 2,
            "preference": {"enabled": True, "lambda_ew": 2.0, "beta": 0.2},
            "schedule": {"ew_warmup_ratio": 0.1, "ew_ramp_ratio": 0.2},
        }
    )

    assert cfg.reference.mode == "none"
    assert cfg.required_outputs == frozenset({"token_logps"})
    assert cfg.effective_weight("ew", 0.1) == pytest.approx(0.0)
    assert cfg.effective_weight("ew", 0.2) == pytest.approx(1.0)
    assert cfg.effective_weight("ew", 0.3) == pytest.approx(2.0)


def test_relation_requires_reference():
    with pytest.raises(ValueError, match="relation.*reference"):
        Stage2EvidenceWindowConfig.from_dict(
            {"version": 2, "relation": {"enabled": True, "lambda_rel": 1.0}}
        )


def test_online_frozen_rejected_for_zero3(tmp_path):
    ds = tmp_path / "ds.json"
    ds.write_text('{"zero_optimization":{"stage":3}}')
    cfg = Stage2EvidenceWindowConfig.from_dict(
        {
            "version": 2,
            "reference": {"mode": "online_frozen", "checkpoint": "/fixed/stage1"},
            "preference": {"enabled": True, "lambda_ew": 1.0},
        }
    )

    with pytest.raises(ValueError, match="ZeRO-3"):
        cfg.validate_runtime(read_deepspeed_zero_stage(ds))


def test_residual_precomputed_reference_requests_hidden_states():
    cfg = Stage2EvidenceWindowConfig.from_dict(
        {
            "version": 2,
            "reference": {"mode": "precomputed", "artifact": "/fixed/targets"},
            "residual": {"enabled": True, "lambda_res": 3.0, "huber_delta": 0.5},
            "schedule": {"residual_warmup_ratio": 0.25, "residual_ramp_ratio": 0.0},
        }
    )

    assert cfg.required_outputs == frozenset({"token_logps", "hidden_states"})
    assert cfg.effective_weight("residual", 0.249) == pytest.approx(0.0)
    assert cfg.effective_weight("residual", 0.25) == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"version": 1}, "version"),
        ({"version": 2, "unknown": True}, "unknown"),
        (
            {"version": 2, "reference": {"mode": "precomputed"}},
            "artifact",
        ),
        (
            {"version": 2, "preference": {"contribution_reduction": "median"}},
            "contribution_reduction",
        ),
        ({"version": 2, "loo": {"max_evidence_edges_per_window": -1}}, "edge"),
    ],
)
def test_invalid_stage2_configuration_is_rejected(raw, message):
    with pytest.raises(ValueError, match=message):
        Stage2EvidenceWindowConfig.from_dict(raw)
