"""Checkpoint objective identity is strict before optimizer restoration."""

import dataclasses
import json

import pytest

from step_preference.config import StepPreferenceConfig
from step_preference.dil_identity import (
    attach_dil_identity,
    build_dil_identity,
    read_dil_identity,
    validate_dil_resume,
)


def bce_config(**changes):
    values = dict(enabled=True, mode="stage2b", objective="context_preference",
                  preference_loss_type="dil_bce_rf_topk", preference_topk=4,
                  lambda_topk_mass=0.2, preference_entropy_top_percent=50.0,
                  beta=0.5, lambda_entropy=0.0, shared_prefix_enabled=True,
                  visual_off_suppress_latent_boundaries=False,
                  direct_from_stage1=True)
    values.update(changes)
    return StepPreferenceConfig(**values)


def checkpoint(tmp_path, identity):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"ilvr_context_preference": identity}))
    return tmp_path


def test_identity_roundtrip_and_mutation_rejection(tmp_path):
    config = bce_config()
    identity = build_dil_identity(config)
    assert identity == {
        "schema_version": 1, "loss_type": "dil_bce_rf_topk",
        "formula_version": "dil_bce_rf_topk_v1", "beta": 0.5,
        "lambda_preference": 1.0, "lambda_ce": 1.0, "lambda_entropy": 0.0,
        "lambda_topk_mass": 0.2, "preference_margin": 0.0, "topk": 4,
        "entropy_top_percent": 50.0, "entropy_source": "clean",
        "target_scope": "immediate_text", "prefix_mode": "clean_prefix",
        "branch_mode": "independent", "visual_only": True,
        "mask": "latent_body_attention", "boundaries": "visible_supervised",
        "block_original_image": False, "shared_prefix": True,
        "sidecar_attention_backend": "flex", "sidecar_query_budget": 1024,
        "sidecar_checkpoint": True,
        "reduction": "token_mean_then_sample_step_mean_v1",
        "entropy_selection": "sample_full_vocab_stable_ceil_original_step_denominator_v1",
    }
    path = checkpoint(tmp_path / "checkpoint-1", identity)
    assert read_dil_identity(json.loads((path / "config.json").read_text())) == identity
    validate_dil_resume(path, config)
    mutations = {
        "schema_version": 2, "loss_type": "dil_bce_rf",
        "formula_version": "dil_bce_rf_v1", "beta": 0.75,
        "lambda_preference": 0.5, "lambda_ce": 0.5,
        "lambda_entropy": 0.25, "lambda_topk_mass": 0.3,
        "preference_margin": 0.25, "topk": 5,
        "entropy_top_percent": 75.0, "entropy_source": "voff",
        "target_scope": "until_next_latent", "prefix_mode": "v_off_prefix",
        "branch_mode": "cumulative", "visual_only": False,
        "mask": "none", "boundaries": "hidden",
        "block_original_image": True, "shared_prefix": False,
        "sidecar_attention_backend": "dense", "sidecar_query_budget": 512,
        "sidecar_checkpoint": False, "reduction": "sum",
        "entropy_selection": "unstable",
    }
    assert set(mutations) == set(identity)
    for key, changed in mutations.items():
        altered = dict(identity)
        altered[key] = changed
        with pytest.raises(ValueError, match="identity"):
            validate_dil_resume(checkpoint(tmp_path / key, altered), config)
    for field, value in (("beta", 0.75), ("lambda_preference", 0.5),
                         ("target_scope", "until_next_latent"),
                         ("sidecar_query_budget", 512), ("preference_topk", 5)):
        with pytest.raises(ValueError, match="identity"):
            validate_dil_resume(path, dataclasses.replace(config, **{field: value}))


def test_bce_resume_missing_or_unknown_identity(tmp_path):
    config = bce_config()
    identity = build_dil_identity(config)
    malformed = [None, {**identity, "future": 1},
                 {key: value for key, value in identity.items() if key != "beta"},
                 {**identity, "schema_version": 2}, {**identity, "beta": True},
                 {**identity, "beta": 1},
                 {**identity, "beta": -0.5},
                 {**identity, "lambda_ce": 0.5},
                 {**identity, "lambda_entropy": 0.1},
                 {**identity, "preference_margin": 0.1},
                 {**identity, "lambda_preference": -1.0},
                 {**identity, "topk": 4.0}, {**identity, "shared_prefix": 1}]
    for index, value in enumerate(malformed):
        payload = {} if value is None else {"ilvr_context_preference": value}
        path = tmp_path / str(index)
        path.mkdir()
        (path / "config.json").write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="identity"):
            validate_dil_resume(path, config)
        if value is not None:
            with pytest.raises(ValueError, match="identity"):
                read_dil_identity(payload)
    assert build_dil_identity(StepPreferenceConfig()) is None
    assert read_dil_identity({}) is None


def test_bce_to_dpo_resume_rejected(tmp_path):
    path = checkpoint(tmp_path, build_dil_identity(bce_config()))
    with pytest.raises(ValueError, match="identity"):
        validate_dil_resume(path, StepPreferenceConfig())


def test_attach_replaces_stale_namespace():
    class Model:
        pass

    from transformers import Qwen2_5_VLConfig
    model = Model()
    model.config = Qwen2_5_VLConfig()
    model.model = Model()
    model.lm_head = Model()
    model.config.ilvr_context_preference = {"stale": True}
    attach_dil_identity(model, bce_config())
    assert model.config.ilvr_context_preference == build_dil_identity(bce_config())
    attach_dil_identity(model, StepPreferenceConfig())
    assert not hasattr(model.config, "ilvr_context_preference")
