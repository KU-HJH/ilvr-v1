import dataclasses
import importlib
import json

import pytest

from utils_deepseed import parse_args


def _config_type():
    module = importlib.import_module("stage2.objective")
    config_type = getattr(module, "Stage2ObjectiveConfig", None)
    assert config_type is not None, "Stage2ObjectiveConfig is required"
    return config_type


def test_objective_config_api_exists():
    try:
        module = importlib.import_module("stage2.objective")
    except ModuleNotFoundError:
        module = None
    assert module is not None, "stage2.objective must provide objective configuration"


def test_objective_cli_defaults_are_opt_in_and_json_serializable():
    args = parse_args([])
    config = _config_type().from_args(args)

    assert config.to_dict() == {
        "enable_stage2_objective": False,
        "lambda_dyn": 0.0,
        "lambda_vis": 0.0,
        "reference_model_path": None,
        "gamma_H": 0.0,
        "m_V": 0.0,
        "entropy_eps": 1e-6,
        "pre_window": 1,
        "post_window": 1,
        "stage2_m_execution": "full_replay",
    }
    assert config.active is False
    json.dumps(config.to_dict())
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.lambda_dyn = 1.0


def test_objective_cli_parses_all_explicit_values():
    args = parse_args(
        [
            "--stage",
            "stage2",
            "--enable_stage2_objective",
            "--lambda_dyn",
            "0.4",
            "--lambda_vis",
            "0.6",
            "--reference_model_path",
            "./checkpoints/reference",
            "--gamma_H",
            "-0.2",
            "--m_V",
            "0.3",
            "--entropy_eps",
            "0.0001",
            "--pre_window",
            "2",
            "--post_window",
            "3",
            "--stage2_m_execution",
            "packed_suffix",
        ]
    )
    config = _config_type().from_args(args)

    assert config.active is True
    assert config.to_dict() == {
        "enable_stage2_objective": True,
        "lambda_dyn": 0.4,
        "lambda_vis": 0.6,
        "reference_model_path": "./checkpoints/reference",
        "gamma_H": -0.2,
        "m_V": 0.3,
        "entropy_eps": 0.0001,
        "pre_window": 2,
        "post_window": 3,
        "stage2_m_execution": "packed_suffix",
    }


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"lambda_dyn": -0.1}, "lambda_dyn"),
        ({"lambda_vis": float("nan")}, "lambda_vis"),
        ({"gamma_H": float("inf")}, "gamma_H"),
        ({"m_V": float("-inf")}, "m_V"),
        ({"entropy_eps": 0.0}, "entropy_eps"),
        ({"pre_window": 0}, "pre_window"),
        ({"post_window": 1.5}, "post_window"),
        ({"stage2_m_execution": "unknown"}, "stage2_m_execution"),
    ],
)
def test_objective_config_rejects_invalid_numeric_values(updates, message):
    fields = _config_type()().to_dict()
    fields.update(updates)
    config = _config_type()(**fields)

    with pytest.raises(ValueError, match=message):
        config.validate(stage="stage2", supervise_latent_boundaries=True)


def test_objective_config_enforces_stage_reference_and_marker_requirements():
    config_type = _config_type()
    enabled = config_type(enable_stage2_objective=True)
    with pytest.raises(ValueError, match="Stage 2"):
        enabled.validate(stage="stage1", supervise_latent_boundaries=True)
    with pytest.raises(ValueError, match="boundar"):
        enabled.validate(stage="stage2", supervise_latent_boundaries=False)

    dynamic = config_type(enable_stage2_objective=True, lambda_dyn=1.0)
    with pytest.raises(ValueError, match="reference_model_path"):
        dynamic.validate(stage="stage2", supervise_latent_boundaries=True)

    config_type(enable_stage2_objective=False, lambda_dyn=1.0).validate(
        stage="stage1", supervise_latent_boundaries=False
    )
