"""Runnable DIL training examples and the fresh-run profile."""

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "scripts/stage2/dil_bce_profile.sh"


def source_profile(extra=None):
    environment = {**os.environ, **(extra or {})}
    result = subprocess.run(
        ["bash", "-c", 'set -e; source "$1"; env -0', "bash", str(PROFILE)],
        cwd=REPO, env=environment, text=True, capture_output=True,
    )
    values = dict(entry.split("=", 1) for entry in result.stdout.split("\0") if "=" in entry)
    return result, values


def test_fresh_profile_overrides_stale_source():
    """A prior run.env must not leak beta, accumulation, or DIL selectors."""
    result, values = source_profile({
        "GRADIENT_ACCUMULATION_STEPS": "2", "STEP_PREFERENCE_BETA": "0.1",
        "STEP_PREFERENCE_TOPK": "8", "STEP_PREFERENCE_ENTROPY_TOP_PERCENT": "20",
        "STEP_PREFERENCE_LAMBDA_TOPK_MASS": "1", "STEP_PREFERENCE_ENTROPY_SOURCE": "voff",
        "STEP_ENTROPY_WEIGHT": "0.1", "DATALOADER_NUM_WORKERS": "4",
        "RESUME_CHECKPOINT": "",
    })
    assert result.returncode == 0, result.stderr
    expected = {
        "GRADIENT_ACCUMULATION_STEPS": "4", "STEP_PREFERENCE_BETA": "1.0",
        "STEP_PREFERENCE_WEIGHT": "1.0", "STEP_PREFERENCE_TOPK": "",
        "STEP_PREFERENCE_ENTROPY_TOP_PERCENT": "",
        "STEP_PREFERENCE_LAMBDA_TOPK_MASS": "0",
        "STEP_PREFERENCE_ENTROPY_SOURCE": "clean", "STEP_ENTROPY_WEIGHT": "0",
        "DATALOADER_NUM_WORKERS": "0", "STAGE2_TOTAL_STEPS": "212",
        "LEARNING_RATE": "5e-5", "SEED": "42",
    }
    assert {key: values.get(key) for key in expected} == expected
    assert "RESUME_CHECKPOINT" not in values


def test_fresh_profile_rejects_resume():
    result, _ = source_profile({"RESUME_CHECKPOINT": "/checkpoint/old"})
    assert result.returncode != 0
    assert "RESUME_CHECKPOINT" in result.stderr


def test_empty_exported_resume_keeps_fresh_coordinator_mode():
    """An empty stale resume variable must not request a coordinator resume."""
    result = subprocess.run(
        ["bash", "-c", 'set -euo pipefail; source "$1"; "$2" --dry-run '
         '--run-id dil-empty-resume-profile-test --zero-stage 2 '
         '--main-process-port 29631 --max_steps "$STAGE2_TOTAL_STEPS"',
         "bash", str(PROFILE), str(REPO / "run_multinode_stage2_training.sh")],
        cwd=REPO,
        env={**os.environ, "RESUME_CHECKPOINT": "", "CUDA_VISIBLE_DEVICES": "",
             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["plan"]["attempt_id"] == "initial"


def bash_blocks():
    text = (REPO / "docs/dil-bce.md").read_text()
    common = re.search(r"```bash\n(# common setup:.*?)\n```", text, re.S).group(1)
    examples = {}
    for name, block in re.findall(r"### ([\w-]+)\n\n```bash\n(.*?)\n```", text, re.S):
        examples[name] = block
    voff = re.search(r"For the explicit visual-off.*?```bash\n(.*?)\n```", text, re.S).group(1)
    examples["topk-pct1-mass-entropy20-voff"] = voff
    return common, examples


def test_common_setup_loads_stage1_metadata_without_inherited_pythonpath():
    """The published setup must import Stage 1 metadata in a clean shell."""
    common, _ = bash_blocks()
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + common], cwd=REPO,
        env={**environment, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
             "TRANSFORMERS_OFFLINE": "1"},
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Stage1 resized_vocab_size=151668; floor(1%)=1516; floor(10%)=15166" in result.stderr


VARIANTS = {
    "bce": ("dil_bce_rf", None, "0", None, "clean"),
    "topk-k8": ("dil_bce_rf_topk", "8", "0", None, "clean"),
    "topk-k5-mass": ("dil_bce_rf_topk", "5", "1", None, "clean"),
    "topk-k8-mass": ("dil_bce_rf_topk", "8", "1", None, "clean"),
    "topk-k50-mass": ("dil_bce_rf_topk", "50", "1", None, "clean"),
    "topk-pct1-mass": ("dil_bce_rf_topk", "1516", "1", None, "clean"),
    "topk-pct10-mass": ("dil_bce_rf_topk", "15166", "1", None, "clean"),
    "topk-pct1-mass-entropy20": ("dil_bce_rf_topk", "1516", "1", "20", "clean"),
    "topk-pct1-mass-entropy20-voff": ("dil_bce_rf_topk", "1516", "1", "20", "voff"),
}


def stage1_fixture_path():
    # The real reproduced metadata provides the complete strict schema; the
    # test copies its bytes into a path with spaces to exercise shell quoting.
    source = REPO / "checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431/stage1_peft_metadata.json"
    payload = json.loads(source.read_text())
    assert (payload["resized_vocab_size"], payload["pretrained_model_vocab_size"], payload["base_vocab_size"]) == (151668, 151936, 151665)
    return payload


@pytest.mark.parametrize("name", VARIANTS)
def test_documented_variant_runs_coordinator_and_both_worker_dry_runs(name, tmp_path):
    """A wrong example option, ratio field, or one-node forwarding breaks this."""
    fixture = tmp_path / "Stage 1 with spaces"
    fixture.mkdir()
    (fixture / "stage1_peft_metadata.json").write_text(json.dumps(stage1_fixture_path()))
    common, examples = bash_blocks()
    assert len(examples) == 9
    common = common.replace(
        "source ./scripts/stage2/dil_bce_profile.sh",
        'source ./scripts/stage2/dil_bce_profile.sh\nexport STAGE1_ADAPTER_PATH="$1"',
    )
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + common + "\nPS4='+ '\nset -x\n" + examples[name], "bash", str(fixture)],
        cwd=REPO, env={**environment, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Stage1 resized_vocab_size=151668; floor(1%)=1516; floor(10%)=15166" in result.stderr
    coordinator_line = next(
        line for line in result.stderr.splitlines()
        if line.startswith("+ env ") and "./run_multinode_stage2_training.sh" in line
    )
    coordinator_argv = shlex.split(coordinator_line[2:])
    coordinator_budget = coordinator_argv[coordinator_argv.index("--max_steps") + 1]
    command_plan = json.loads(result.stdout)
    assert command_plan["plan"]["world_size"] == 4
    remote = shlex.split(shlex.split(command_plan["remote"][-1])[-1])
    for rank, owned in ((1, command_plan["local"]), (0, remote)):
        env_start = owned.index("-i") + 1
        worker_env = {}
        for field in owned[env_start:]:
            if "=" not in field or field.startswith("--"):
                break
            key, value = field.split("=", 1)
            worker_env[key] = value
        stage_command = owned[max(i for i, field in enumerate(owned) if field == "--") + 1:]
        worker = subprocess.run(
            stage_command, cwd=REPO,
            env={**worker_env, "STAGE2_DRY_RUN": "true", "STAGE2_DRY_RUN_FORMAT": "json",
                 "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            text=True, capture_output=True,
        )
        assert worker.returncode == 0, worker.stderr
        argv = json.loads(worker.stdout)
        selector, k, mass, percent, source = VARIANTS[name]
        expected = {
            "--num_machines": "2", "--num_processes": "4", "--machine_rank": str(rank),
            "--gradient_accumulation_steps": "4", "--per_device_train_batch_size": "1",
            "--learning_rate": "5e-5", "--seed": "42", "--warm_up_steps": "10",
            "--step_preference_beta": "1.0", "--step_preference_weight": "1.0",
            "--step_preference_loss_type": selector,
            "--step_preference_lambda_topk_mass": mass,
            "--step_preference_entropy_source": source,
            "--step_entropy_weight": "0", "--stage2_total_steps": "212",
            "--stage2_sampler_policy": "reference_random",
            "--stage2_assistant_plan_rendering": "concatenated-v1",
            "--stage1_adapter_path": str(fixture),
        }
        if k is not None:
            expected["--step_preference_topk"] = k
        if percent is not None:
            expected["--step_preference_entropy_top_percent"] = percent
        for option, value in expected.items():
            assert argv[argv.index(option) + 1] == value
        assert coordinator_budget == argv[argv.index("--stage2_total_steps") + 1]
        if k is None:
            assert "--step_preference_topk" not in argv
        if percent is None:
            assert "--step_preference_entropy_top_percent" not in argv
        assert "--no-gradient_checkpointing" in argv
        assert "--stage2_latent_activation_checkpointing" in argv
        assert "--step_shared_prefix_enabled" in argv
        assert "--step_visual_only_context_preference" in argv
        assert "--no-step_visual_off_suppress_latent_boundaries" in argv
        assert "--supervise_latent_boundaries" in argv
        assert not (REPO / "checkpoints" / command_plan["plan"]["run_id"]).exists()
        assert not (REPO / "results" / command_plan["plan"]["run_id"]).exists()
