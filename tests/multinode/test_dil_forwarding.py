"""DIL settings must survive the coordinator and the actual worker shell."""

import json
import subprocess
from pathlib import Path

import pytest

import src.multinode as multinode
from src.multinode_runtime import build_owned_tmux_commands, validated_stage2_environment

REPO = Path(__file__).resolve().parents[2]


def worker_commands(settings, stage_arguments=()):
    plan = multinode.build_launch_plan(
        stage="stage2", tuning_method="dora", zero_stage=2,
        run_id="dil-forwarding-test", main_process_port=29631,
    )
    source = {
        "STAGE1_CHECKPOINT": "/models/base with spaces",
        "STAGE1_ADAPTER_PATH": "/models/stage1 with spaces",
        "DATA_PATH": "/data/train with spaces.jsonl",
        "IMAGE_ROOT": "/data/images with spaces",
        "TUNING_METHOD": "dora",
        "GRADIENT_ACCUMULATION_STEPS": "4",
        "GRADIENT_CHECKPOINTING": "false",
        "STAGE2_LATENT_ACTIVATION_CHECKPOINTING": "true",
        "CE_WEIGHT": "1",
        "STEP_PREFERENCE_ENABLED": "true",
        "STEP_PREFERENCE_MODE": "stage2b",
        "STEP_PREFERENCE_OBJECTIVE": "context_preference",
        "STEP_PREFERENCE_TARGET_SCOPE": "immediate_text",
        "STEP_PREFERENCE_BETA": "0.5",
        "STEP_PREFERENCE_WEIGHT": "1.0",
        "STEP_ENTROPY_WEIGHT": "0",
        "STEP_VISUAL_ONLY_CONTEXT_PREFERENCE": "true",
        "STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES": "false",
        "STEP_SHARED_PREFIX_ENABLED": "true",
        "STAGE2B_DIRECT_FROM_STAGE1": "true",
        "DATALOADER_NUM_WORKERS": "0",
        **settings,
    }
    environment = validated_stage2_environment(
        source, output_dir=str(REPO / "checkpoints" / plan.run_id),
        log_file=str(REPO / "logs" / plan.run_id / "train.log"),
        deepspeed_config=str(plan.deepspeed_config),
    )
    commands = build_owned_tmux_commands(plan, commit="a" * 40, stage_environment=environment, stage_arguments=stage_arguments)
    for rank, spec in enumerate(commands.node_specs):
        worker_env = dict(spec.environment)
        result = subprocess.run(
            spec.command[spec.command.index("--") + 1:], cwd=REPO,
            capture_output=True, text=True,
            env={**worker_env, "STAGE2_DRY_RUN": "true", "STAGE2_DRY_RUN_FORMAT": "json",
                 "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                 "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        yield rank, worker_env, result


@pytest.mark.parametrize("selector,topk,mass,percent,source", [
    ("dil_bce_rf", None, "0", None, "clean"),
    ("dil_bce_rf_topk", "8", "1", "20", "voff"),
    ("dil_bce_rf_topk_reverse", "1516", "1", "20", "clean"),
    ("dil_bce_rf_topk_gt", "1516", "1", "20", "clean"),
])
def test_dil_settings_reach_both_real_worker_commands(selector, topk, mass, percent, source):
    settings = {
        "STEP_PREFERENCE_LOSS_TYPE": selector,
        "STEP_PREFERENCE_TOPK": topk or "",
        "STEP_PREFERENCE_LAMBDA_TOPK_MASS": mass,
        "STEP_PREFERENCE_ENTROPY_TOP_PERCENT": percent or "",
        "STEP_PREFERENCE_ENTROPY_SOURCE": source,
    }
    for rank, worker_env, result in worker_commands(settings):
        assert {key: worker_env[key] for key in settings} == settings
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        expected = {
            "--num_machines": "2", "--num_processes": "4", "--machine_rank": str(rank),
            "--gradient_accumulation_steps": "4", "--step_preference_beta": "0.5",
            "--step_preference_weight": "1.0", "--step_preference_loss_type": selector,
            "--step_preference_lambda_topk_mass": mass,
            "--step_preference_entropy_source": source,
        }
        if topk:
            expected["--step_preference_topk"] = topk
        if percent:
            expected["--step_preference_entropy_top_percent"] = percent
        for option, value in expected.items():
            assert argv[argv.index(option) + 1] == value
        if not topk:
            assert "--step_preference_topk" not in argv
        if not percent:
            assert "--step_preference_entropy_top_percent" not in argv


@pytest.mark.parametrize("settings", [
    {"STEP_PREFERENCE_LOSS_TYPE": "dil_bce_rf", "STEP_PREFERENCE_TOPK": "8"},
    {"STEP_PREFERENCE_LOSS_TYPE": "dil_bce_rf_topk", "STEP_PREFERENCE_TOPK": "8",
     "STEP_PREFERENCE_ENTROPY_TOP_PERCENT": "0"},
    {"STEP_PREFERENCE_LOSS_TYPE": "dil_bce_rf_topk", "STEP_PREFERENCE_TOPK": "8",
     "STEP_PREFERENCE_ENTROPY_SOURCE": "voff"},
    {"STEP_PREFERENCE_LOSS_TYPE": "dil_bce_rf_topk", "STEP_PREFERENCE_TOPK": "8",
     "STEP_PREFERENCE_LAMBDA_TOPK_MASS": "nan"},
    {"STEP_PREFERENCE_LOSS_TYPE": "dil_bce_rf", "STEP_PREFERENCE_OBJECTIVE": "legacy_dpo",
     "STEP_NEGATIVE_PATH": "/data/negative.jsonl", "STEP_NEGATIVE_MANIFEST": "/data/negative.json",
     "STAGE2B_DIRECT_FROM_STAGE1": "false", "STAGE2A_CHECKPOINT_PATH": "/models/stage2a",
     "STEP_VISUAL_ONLY_CONTEXT_PREFERENCE": "false"},
])
def test_invalid_dil_combination_fails_in_worker_dry_run(settings):
    for _, _, result in worker_commands(settings):
        assert result.returncode != 0
        assert "top-k" in result.stderr.lower() or "preference" in result.stderr.lower()


@pytest.mark.parametrize("selector,topk", [("dil_bce_rf", ""), ("dil_bce_rf_topk", "8"),
    ("dil_bce_rf_topk_reverse", "1516"), ("dil_bce_rf_topk_gt", "1516")])
def test_dil_resume_reaches_both_worker_argv_without_fresh_initialization(selector, topk):
    checkpoint = str(REPO / "checkpoints" / "dil-forwarding-test" / "checkpoint-1")
    settings = {"STEP_PREFERENCE_LOSS_TYPE": selector, "STEP_PREFERENCE_TOPK": topk,
                "STAGE2B_DIRECT_FROM_STAGE1": "false", "STAGE1_ADAPTER_PATH": "",
                "RESUME_CHECKPOINT": checkpoint}
    if selector == "dil_bce_rf_topk_gt":
        settings.update(STEP_PREFERENCE_LAMBDA_TOPK_MASS="1",
                        STEP_PREFERENCE_ENTROPY_TOP_PERCENT="20", STEP_PREFERENCE_ENTROPY_SOURCE="clean")
    for rank, worker_env, result in worker_commands(settings, ("--max_steps", "212")):
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        assert argv[argv.index("--resume_from_checkpoint_path") + 1] == checkpoint
        assert argv[argv.index("--max_steps") + 1] == "212"
        assert argv[argv.index("--machine_rank") + 1] == str(rank)
        assert argv[argv.index("--step_preference_loss_type") + 1] == selector
        assert argv[argv.index("--step_preference_beta") + 1] == "0.5"
        assert argv[argv.index("--step_preference_weight") + 1] == "1.0"
        if selector == "dil_bce_rf_topk_gt":
            for option, value in (("--step_preference_topk", "1516"),
                                  ("--step_preference_lambda_topk_mass", "1"),
                                  ("--step_preference_entropy_top_percent", "20"),
                                  ("--step_preference_entropy_source", "clean")):
                assert argv[argv.index(option) + 1] == value
        assert "--stage2b_direct_from_stage1" not in argv
        assert "--stage1_adapter_path" not in argv
        assert "--stage2a_checkpoint_path" not in argv


@pytest.mark.parametrize("mixed", [
    {"STAGE2B_DIRECT_FROM_STAGE1": "true"}, {"STAGE1_ADAPTER_PATH": "/fresh"},
    {"STAGE2A_CHECKPOINT_PATH": "/fresh"}, {"STAGE2_TOTAL_STEPS": "212"},
])
def test_worker_rejects_mixed_dil_resume_initialization(mixed):
    settings = {"STEP_PREFERENCE_LOSS_TYPE": "dil_bce_rf",
                "STAGE2B_DIRECT_FROM_STAGE1": "false", "STAGE1_ADAPTER_PATH": "",
                "RESUME_CHECKPOINT": "/saved/checkpoint-1", **mixed}
    for _, _, result in worker_commands(settings):
        assert result.returncode != 0
        assert "RESUME_CHECKPOINT" in result.stderr


def test_planned_stop_reaches_both_workers_with_unchanged_total_budget():
    for rank, _, result in worker_commands(
        {'STEP_PREFERENCE_LOSS_TYPE': 'dil_bce_rf'},
        ('--max_steps', '3', '--stage2_stop_after_checkpoint', '2'),
    ):
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        assert argv[argv.index('--stage2_stop_after_checkpoint') + 1] == '2'
        assert argv[argv.index('--max_steps') + 1] == '3'
        assert argv[argv.index('--machine_rank') + 1] == str(rank)


def test_gt_omitted_beta_fresh_resolves_dil_defaults_in_both_workers():
    # Break: shell defaults ordinary beta/margin for GT or loses independent mass.
    settings = {"STEP_PREFERENCE_LOSS_TYPE": "dil_bce_rf_topk_gt",
                "STEP_PREFERENCE_TOPK": "1516", "STEP_PREFERENCE_LAMBDA_TOPK_MASS": "1",
                "STEP_PREFERENCE_BETA": "", "CE_WEIGHT": ""}
    for rank, _, result in worker_commands(settings):
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        for option, value in (("--step_preference_beta", "1.0"), ("--step_preference_margin", "0.0"),
                              ("--ce_weight", "0.1"), ("--step_preference_lambda_topk_mass", "1")):
            assert argv[argv.index(option)+1] == value
        assert argv[argv.index("--machine_rank")+1] == str(rank)
