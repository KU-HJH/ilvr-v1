"""Resume never invents fresh CE, and beta aliases survive worker objectives."""
import json

import pytest

from step_preference.config import StepPreferenceConfig
from utils_deepseed import parse_args
from tests.multinode.test_dil_forwarding import worker_commands


@pytest.mark.parametrize("stage,enabled", [("stage1", False), ("stage2", False), ("stage2", True)])
def test_unrecorded_resume_requires_explicit_saved_ce_before_training(stage, enabled):
    # Break: an omitted coefficient silently becomes fresh0.1 on FULL/CE-only/2A resume.
    args = ["--stage", stage, "--resume_from_checkpoint_path", "historical-full-checkpoint"]
    if enabled:
        args += ["--step_preference_enabled", "--step_preference_mode", "stage2a"]
    with pytest.raises(ValueError, match="resume.*ce_weight"):
        parse_args(args)


@pytest.mark.parametrize("enabled", [False, True])
def test_direct_plain_resume_cannot_invent_ce_default(enabled):
    with pytest.raises(ValueError, match="resume.*ce_weight"):
        StepPreferenceConfig(enabled=enabled, mode="stage2a", resume_from_checkpoint_path="historical")


@pytest.mark.parametrize("stage,enabled", [("stage1", False), ("stage2", False), ("stage2", True)])
def test_explicit_historical_ce_remains_supported_on_unrecorded_resume(stage, enabled):
    args = ["--stage", stage, "--resume_from_checkpoint_path", "historical-full-checkpoint", "--ce_weight", "1"]
    if enabled:
        args += ["--step_preference_enabled", "--step_preference_mode", "stage2a"]
    assert parse_args(args).ce_weight == 1.


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_both_plain_stage2_workers_preserve_resume_ce_omission(enabled, explicit):
    settings = {"STEP_PREFERENCE_ENABLED": "true" if enabled else "false", "STEP_PREFERENCE_MODE": "stage2a",
        "STAGE2B_DIRECT_FROM_STAGE1": "false", "STAGE1_ADAPTER_PATH": "", "STEP_SHARED_PREFIX_ENABLED": "false",
        "STEP_VISUAL_ONLY_CONTEXT_PREFERENCE": "false", "RESUME_CHECKPOINT": "/historical/full-checkpoint",
        "CE_WEIGHT": "1" if explicit else ""}
    rows = list(worker_commands(settings))
    assert len(rows) == 2
    for rank, _, result in rows:
        if not explicit:
            assert result.returncode != 0
            assert "resume" in result.stderr and "ce_weight" in result.stderr
        else:
            assert result.returncode == 0, result.stderr
            argv = json.loads(result.stdout)
            assert argv[argv.index("--ce_weight")+1] == "1"
            assert argv[argv.index("--machine_rank")+1] == str(rank)


@pytest.mark.parametrize("objective,modern,legacy,want", [
    ("legacy_dpo", ".37", "", ".37"),
    ("legacy_dpo", "", ".73", ".73"),
    ("legacy_dpo", "", "", "10.0"),
    ("legacy_dpo", ".37", ".73", ".73"),
    ("context_preference", ".37", ".73", ".37"),
])
def test_both_workers_preserve_beta_aliases_and_existing_dual_alias_precedence(objective, modern, legacy, want):
    # Break: default10 swallows explicit modern beta when legacy objective is chosen.
    settings = {"STEP_PREFERENCE_OBJECTIVE": objective, "STEP_PREFERENCE_BETA": modern, "STEP_DPO_BETA": legacy,
        "STAGE2B_DIRECT_FROM_STAGE1": "false", "STAGE1_ADAPTER_PATH": "", "STAGE2A_CHECKPOINT_PATH": "/stage2a"}
    if objective == "legacy_dpo":
        settings.update(STEP_NEGATIVE_PATH="/negative.jsonl", STEP_NEGATIVE_MANIFEST="/negative-manifest.json",
                        STEP_VISUAL_ONLY_CONTEXT_PREFERENCE="false")
    rows = list(worker_commands(settings))
    assert len(rows) == 2
    option = "--step_dpo_beta" if objective == "legacy_dpo" else "--step_preference_beta"
    for rank, _, result in rows:
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        assert argv[argv.index(option)+1] == want
        assert argv[argv.index("--machine_rank")+1] == str(rank)
