"""Regression for Stage 2B settings lost between coordinator and workers."""

import json
import subprocess
from pathlib import Path

import pytest

import src.multinode as multinode
from src.multinode_runtime import build_owned_tmux_commands, validated_stage2_environment


REPO = Path(__file__).resolve().parents[2]


def stage2b_environment(dpo="1", entropy="0.1"):
    return {
        "STAGE1_CHECKPOINT": "/models/stage2a",
        "DATA_PATH": "/data/train.jsonl",
        "IMAGE_ROOT": "/data/images",
        "TUNING_METHOD": "dora",
        "GRADIENT_ACCUMULATION_STEPS": "2",
        "GRADIENT_CHECKPOINTING": "true",
        "CE_WEIGHT": "1",
        "STEP_PREFERENCE_ENABLED": "true",
        "STEP_PREFERENCE_MODE": "stage2b",
        "STEP_PREFERENCE_OBJECTIVE": "legacy_dpo",
        "STAGE2A_CHECKPOINT_PATH": "/models/stage2a",
        "STEP_NEGATIVE_PATH": "/data/negative cache.jsonl",
        "STEP_NEGATIVE_MANIFEST": "/data/negative manifest.json",
        "STEP_DPO_WEIGHT": dpo,
        "STEP_ENTROPY_WEIGHT": entropy,
        "STEP_DPO_BETA": "0.2",
        "STEP_ENTROPY_MARGIN": "0.3",
        "STEP_SHARED_PREFIX_ENABLED": "true",
        "STEP_SIDECAR_ATTENTION_BACKEND": "flex",
        "STEP_SIDECAR_QUERY_BUDGET": "37",
        "VISION_ATTENTION_IMPLEMENTATION": "flash_attention_2",
        "STAGE2_TOTAL_STEPS": "10",
        "STAGE2A_RATIO": "0.4",
        "DATALOADER_NUM_WORKERS": "0",
        "SEED": "17",
    }


@pytest.mark.parametrize("dpo,entropy", [("0", "0"), ("1", "0"), ("0", "0.1"), ("1", "0.1")])
def test_stage2b_variants_reach_both_worker_training_commands(dpo, entropy):
    """Dropping a variant setting must fail at the real shell/CLI boundary."""
    plan = multinode.build_launch_plan(
        stage="stage2", tuning_method="dora", zero_stage=2,
        run_id="stage2b-forwarding-test", main_process_port=29631,
    )
    source = stage2b_environment(dpo, entropy)
    environment = validated_stage2_environment(
        source, output_dir=str(REPO / "checkpoints" / plan.run_id),
        log_file=str(REPO / "logs" / plan.run_id / "train.log"),
        deepspeed_config=str(plan.deepspeed_config),
    )
    commands = build_owned_tmux_commands(plan, commit="a" * 40, stage_environment=environment)
    for rank, spec in enumerate(commands.node_specs):
        worker_env = dict(spec.environment)
        assert {name: worker_env.get(name) for name in source} == source
        # Only the final shell command is dry-run; the real coordinator builders
        # and the shell's argument parser still run. No GPU/SSH or model writes.
        stage_command = spec.command[spec.command.index("--") + 1:]
        result = subprocess.run(
            stage_command, cwd=REPO, capture_output=True, text=True,
            env={**worker_env, "STAGE2_DRY_RUN": "true", "STAGE2_DRY_RUN_FORMAT": "json"},
        )
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        for option, value in {
            "--num_machines": "2", "--num_processes": "4", "--machine_rank": str(rank),
            "--per_device_train_batch_size": "1", "--gradient_accumulation_steps": "2",
            "--tuning_method": "dora", "--deepspeed_config": str(plan.deepspeed_config),
            "--step_preference_mode": "stage2b", "--step_preference_objective": "legacy_dpo",
            "--ce_weight": "1",
            "--step_dpo_weight": dpo, "--step_entropy_weight": entropy,
            "--step_negative_path": "/data/negative cache.jsonl",
            "--step_negative_manifest": "/data/negative manifest.json",
            "--stage2a_checkpoint_path": "/models/stage2a",
            "--step_dpo_beta": "0.2", "--step_entropy_margin": "0.3",
            "--step_sidecar_attention_backend": "flex", "--step_sidecar_query_budget": "37",
            "--vision_attention_implementation": "flash_attention_2",
            "--stage2_total_steps": "10", "--stage2a_ratio": "0.4", "--seed": "17",
        }.items():
            assert argv[argv.index(option) + 1] == value
        assert "--step_preference_enabled" in argv
        assert "--step_shared_prefix_enabled" in argv


@pytest.mark.parametrize("target_scope", ["immediate_text", "until_next_latent"])
def test_context_preference_settings_reach_both_owned_worker_commands(target_scope):
    """Dropping context objective settings must fail at the real worker boundary."""
    plan = multinode.build_launch_plan(
        stage="stage2", tuning_method="dora", zero_stage=2,
        run_id="context-forwarding-test", main_process_port=29631,
    )
    source = stage2b_environment()
    source.pop("STEP_NEGATIVE_PATH")
    source.pop("STEP_NEGATIVE_MANIFEST")
    source.update(
        {
            "STEP_PREFERENCE_OBJECTIVE": "context_preference",
            "STEP_PREFERENCE_BETA": "0.37",
            "STEP_PREFERENCE_TARGET_SCOPE": target_scope,
            "STEP_PREFERENCE_WEIGHT": "1.7",
            "STEP_PREFERENCE_MARGIN": "-0.25",
            "STEP_VISUAL_ONLY_CONTEXT_PREFERENCE": "true",
            "STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES": "true",
        }
    )
    environment = validated_stage2_environment(
        source, output_dir=str(REPO / "checkpoints" / plan.run_id),
        log_file=str(REPO / "logs" / plan.run_id / "train.log"),
        deepspeed_config=str(plan.deepspeed_config),
    )
    commands = build_owned_tmux_commands(
        plan, commit="a" * 40, stage_environment=environment
    )

    for spec in commands.node_specs:
        worker_env = dict(spec.environment)
        assert {
            name: worker_env.get(name)
            for name in (
                "STEP_PREFERENCE_OBJECTIVE",
                "STEP_PREFERENCE_BETA",
                "STEP_PREFERENCE_TARGET_SCOPE",
                "STEP_PREFERENCE_WEIGHT",
                "STEP_PREFERENCE_MARGIN",
                "STEP_VISUAL_ONLY_CONTEXT_PREFERENCE",
                "STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES",
            )
        } == {
            "STEP_PREFERENCE_OBJECTIVE": "context_preference",
            "STEP_PREFERENCE_BETA": "0.37",
            "STEP_PREFERENCE_TARGET_SCOPE": target_scope,
            "STEP_PREFERENCE_WEIGHT": "1.7",
            "STEP_PREFERENCE_MARGIN": "-0.25",
            "STEP_VISUAL_ONLY_CONTEXT_PREFERENCE": "true",
            "STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES": "true",
        }
        stage_command = spec.command[spec.command.index("--") + 1:]
        result = subprocess.run(
            stage_command, cwd=REPO, capture_output=True, text=True,
            env={**worker_env, "STAGE2_DRY_RUN": "true", "STAGE2_DRY_RUN_FORMAT": "json"},
        )
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        for option, value in {
            "--step_preference_objective": "context_preference",
            "--step_preference_beta": "0.37",
            "--step_preference_target_scope": target_scope,
            "--step_preference_weight": "1.7",
            "--step_preference_margin": "-0.25",
        }.items():
            assert argv[argv.index(option) + 1] == value
        assert "--step_visual_only_context_preference" in argv
        assert "--step_visual_off_suppress_latent_boundaries" in argv


def test_direct_stage1_setting_reaches_both_owned_worker_commands():
    """Catches the coordinator dropping direct Stage 1 admission on either worker."""
    plan = multinode.build_launch_plan(
        stage="stage2", tuning_method="dora", zero_stage=2,
        run_id="direct-stage1-forwarding-test", main_process_port=29631,
    )
    source = stage2b_environment()
    source.pop("STAGE2A_CHECKPOINT_PATH")
    source.pop("STEP_NEGATIVE_PATH")
    source.pop("STEP_NEGATIVE_MANIFEST")
    source.update(
        {
            "STEP_PREFERENCE_OBJECTIVE": "context_preference",
            "STEP_VISUAL_ONLY_CONTEXT_PREFERENCE": "true",
            "STAGE2B_DIRECT_FROM_STAGE1": "true",
            "STAGE1_ADAPTER_PATH": "/models/stage1-adapter",
        }
    )
    environment = validated_stage2_environment(
        source, output_dir=str(REPO / "checkpoints" / plan.run_id),
        log_file=str(REPO / "logs" / plan.run_id / "train.log"),
        deepspeed_config=str(plan.deepspeed_config),
    )
    commands = build_owned_tmux_commands(
        plan, commit="a" * 40, stage_environment=environment
    )

    for spec in commands.node_specs:
        worker_env = dict(spec.environment)
        assert worker_env["STAGE2B_DIRECT_FROM_STAGE1"] == "true"
        stage_command = spec.command[spec.command.index("--") + 1:]
        result = subprocess.run(
            stage_command, cwd=REPO, capture_output=True, text=True,
            env={**worker_env, "STAGE2_DRY_RUN": "true", "STAGE2_DRY_RUN_FORMAT": "json"},
        )
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        assert "--stage2b_direct_from_stage1" in argv
        assert argv[argv.index("--stage1_adapter_path") + 1] == "/models/stage1-adapter"
        assert "--stage2a_checkpoint_path" not in argv


def test_stage2b_preflight_includes_checkpoint_and_negative_inputs(monkeypatch):
    """Missing Stage 2B inputs must be checked before either worker launches."""
    for name, value in stage2b_environment().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("STAGE1_CHECKPOINT", "/models/stage1")

    class PreflightReached(Exception):
        pass

    def capture_snapshot(plan, **kwargs):
        paths = set(kwargs["input_paths"])
        assert Path("/models/stage2a") in paths
        assert Path("/data/negative cache.jsonl") in paths
        assert Path("/data/negative manifest.json") in paths
        raise PreflightReached

    # Stop at the first external node inspection, before SSH or filesystem writes.
    monkeypatch.setattr(multinode, "collect_node_snapshot", capture_snapshot)
    with pytest.raises(PreflightReached):
        multinode._training_main("stage2", [
            "--run-id", "stage2b-forwarding-test", "--zero-stage", "2",
            "--main-process-port", "29631", "--max_steps", "6", "--save_steps", "6",
        ])
