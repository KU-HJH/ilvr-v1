import json

import pytest

from evidence_window.config import EvidenceWindowConfig
from utils_deepseed import get_args


def test_disabled_requests_nothing():
    cfg = EvidenceWindowConfig.disabled()
    assert cfg.is_disabled
    assert cfg.required_conditions("policy") == frozenset()
    assert cfg.required_conditions("reference") == frozenset()


def test_unknown_key_fails(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 1, "unknown": True}))
    with pytest.raises(ValueError, match="unknown"):
        EvidenceWindowConfig.from_json(path)


def test_preference_requires_reference(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "reference": {"mode": "none"},
                "preference": {"lambda_ew": 1.0},
            }
        )
    )
    with pytest.raises(ValueError, match="reference"):
        EvidenceWindowConfig.from_json(path)


def test_relation_only_needs_policy_and_ema_loo(tmp_path):
    path = tmp_path / "relation.json"
    path.write_text(json.dumps({"version": 1, "relation": {"lambda_jsd": 1.0}}))
    cfg = EvidenceWindowConfig.from_json(path)
    assert cfg.required_conditions("policy") == frozenset({"full", "loo"})
    assert cfg.required_conditions("ema") == frozenset({"full", "loo"})
    assert cfg.required_conditions("reference") == frozenset()


def test_global_diagnostic_needs_fixed_reference_and_all_null(tmp_path):
    path = tmp_path / "diagnostic.json"
    path.write_text(json.dumps({"version": 1, "diagnostics": {"global_saturation": True}}))
    with pytest.raises(ValueError, match="reference"):
        EvidenceWindowConfig.from_json(path)

    path.write_text(
        json.dumps(
            {
                "version": 1,
                "reference": {"mode": "adapter_disabled"},
                "diagnostics": {"global_saturation": True},
            }
        )
    )
    cfg = EvidenceWindowConfig.from_json(path)
    assert cfg.required_conditions("policy") == frozenset({"full", "all_null"})
    assert cfg.required_conditions("reference") == frozenset({"full", "all_null"})


@pytest.mark.parametrize(
    ("section", "value"),
    [
        ("reference", {"mode": "mutable"}),
        ("preference", {"beta": 0}),
        ("relation", {"temperature": 0}),
        ("residual", {"vocab_chunk_size": 0}),
        ("loo", {"intervention": "drop"}),
        ("diagnostics", {"matrix_dump_interval": -1}),
    ],
)
def test_invalid_values_fail_at_startup(tmp_path, section, value):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"version": 1, section: value}))
    with pytest.raises(ValueError):
        EvidenceWindowConfig.from_json(path)


def test_nested_unknown_key_fails(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 1, "relation": {"unknown": True}}))
    with pytest.raises(ValueError, match="relation.*unknown"):
        EvidenceWindowConfig.from_json(path)


def test_evidence_window_cli_has_disabled_and_selectable_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train"])
    defaults = get_args()
    assert defaults.ew_config is None
    assert defaults.deepspeed_config == "configs/config_stage3.json"

    monkeypatch.setattr(
        "sys.argv",
        ["train", "--ew_config", "experiment.json", "--deepspeed_config", "zero2.json"],
    )
    selected = get_args()
    assert selected.ew_config == "experiment.json"
    assert selected.deepspeed_config == "zero2.json"
