import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from src.deepspeed_runtime import canonical_deepspeed_config_path
from src.multinode import (
    ClusterSpec,
    LaunchPlan,
    build_launch_plan,
    canonical_deepspeed_config,
    validate_node_local_deepspeed_config,
)


def test_default_cluster_is_fixed_two_node_four_process_topology():
    """Catches a launch silently drifting from the approved four-GPU topology."""
    spec = ClusterSpec.default()

    assert spec.remote_host == "korea@10.1.102.41"
    assert spec.remote_container == "llm_kd"
    assert spec.remote_repo_root == Path("/root/workspace/ILVR")
    assert spec.remote_host_repo_root == Path(
        "/home/korea/Workspace/hjh/workspace/ILVR"
    )
    assert (spec.num_machines, spec.gpus_per_node, spec.num_processes) == (2, 2, 4)
    assert (spec.remote_rank, spec.local_rank) == (0, 1)
    assert spec.master_addr == "10.1.102.41"
    assert spec.socket_ifname == "bond1.102"


@pytest.mark.parametrize("zero_stage", (1, 2, 3))
def test_launch_plan_selects_exact_zero_config(zero_stage):
    """Catches a supported ZeRO choice selecting a noncanonical config path."""
    plan = build_launch_plan(
        stage="stage1",
        tuning_method="lora",
        zero_stage=zero_stage,
        run_id=f"stage1-lora-zero{zero_stage}-four-gpu",
        main_process_port=29611 + zero_stage,
    )

    assert plan.world_size == 4
    assert plan.deepspeed_config == Path(
        f"/root/workspace/ILVR/configs/deepspeed_zero{zero_stage}.json"
    )
    assert plan.required_free_bytes == 100 * 1024**3


@pytest.mark.parametrize("zero_stage", (1, 2, 3))
def test_canonical_config_is_validated_for_the_four_process_world(zero_stage):
    """Catches topology planning bypassing canonical DeepSpeed policy validation."""
    plan = build_launch_plan(
        stage="stage2",
        tuning_method="full",
        zero_stage=zero_stage,
        run_id=f"stage2-full-zero{zero_stage}-four-gpu",
        main_process_port=29621 + zero_stage,
    )

    assert plan.zero_stage == zero_stage
    assert plan.world_size == 4
    assert canonical_deepspeed_config(zero_stage) == plan.deepspeed_config


def test_node_local_config_must_match_the_launch_plan_digest(tmp_path):
    """Catches a node launching a policy-valid config with different bytes."""
    plan = build_launch_plan(
        stage="stage1",
        tuning_method="lora",
        zero_stage=2,
        run_id="stage1-lora-zero2-four-gpu",
        main_process_port=29613,
    )
    node_config = tmp_path / "deepspeed_zero2.json"
    payload = json.loads(
        Path("configs/deepspeed_zero2.json").read_text(encoding="utf-8")
    )
    payload["wall_clock_breakdown"] = True
    node_config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        validate_node_local_deepspeed_config(plan, node_config)


def test_node_local_config_accepts_the_exact_validated_bytes():
    """Catches node-local validation rejecting the config that built the plan."""
    plan = build_launch_plan(
        stage="stage1",
        tuning_method="lora",
        zero_stage=2,
        run_id="stage1-lora-zero2-four-gpu",
        main_process_port=29613,
    )

    assert validate_node_local_deepspeed_config(
        plan, canonical_deepspeed_config_path(2)
    ) is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"zero_optimization": {"stage": 1}}, "stage"),
        ({"train_batch_size": 4}, "train_batch_size"),
    ],
)
def test_node_local_config_rejects_stage_or_policy_mismatch(
    tmp_path, changes, message
):
    """Catches a node bypassing the launch plan's ZeRO or policy contract."""
    plan = build_launch_plan(
        stage="stage1",
        tuning_method="lora",
        zero_stage=2,
        run_id="stage1-lora-zero2-four-gpu",
        main_process_port=29613,
    )
    node_config = tmp_path / "deepspeed_zero2.json"
    payload = json.loads(
        canonical_deepspeed_config_path(2).read_text(encoding="utf-8")
    )
    payload.update(changes)
    node_config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_node_local_deepspeed_config(plan, node_config)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"run_id": "Nope"}, "run_id"),
        ({"attempt_id": "x"}, "attempt_id"),
        ({"main_process_port": 1023}, "main_process_port"),
        ({"main_process_port": 65536}, "main_process_port"),
        ({"stage": "evaluation"}, "stage"),
        ({"tuning_method": "adapter"}, "tuning_method"),
        ({"zero_stage": 4}, "zero_stage"),
        ({"zero_stage": True}, "zero_stage"),
    ],
)
def test_launch_plan_rejects_invalid_identity_or_contract(kwargs, message):
    """Catches invalid coordinator-controlled launch inputs reaching a worker."""
    values = {
        "stage": "stage1",
        "tuning_method": "lora",
        "zero_stage": 2,
        "run_id": "stage1-lora-zero2-four-gpu",
        "main_process_port": 29613,
        "attempt_id": "initial",
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_launch_plan(**values)


def test_launch_plan_is_immutable_and_resume_only_changes_attempt_identity():
    """Catches resume attempts overwriting the checkpoint run identity or plan."""
    initial = build_launch_plan(
        stage="stage1",
        tuning_method="dora",
        zero_stage=3,
        run_id="stage1-dora-zero3-four-gpu",
        main_process_port=29614,
    )
    resumed = replace(initial, attempt_id="resume-step10-to11")

    assert isinstance(initial, LaunchPlan)
    assert initial.attempt_id == "initial"
    assert resumed.run_id == initial.run_id
    assert resumed.attempt_id == "resume-step10-to11"
    with pytest.raises(FrozenInstanceError):
        initial.run_id = "different-run"  # type: ignore[misc]


def test_plan_cli_prints_sorted_json_without_creating_files(tmp_path):
    """Catches the dry plan command performing coordinator side effects."""
    command = [
        sys.executable,
        "-m",
        "src.multinode",
        "plan",
        "--stage",
        "stage2",
        "--tuning-method",
        "lora",
        "--zero-stage",
        "2",
        "--run-id",
        "stage2-lora-zero2-four-gpu",
        "--main-process-port",
        "29615",
    ]
    before = set(tmp_path.iterdir())
    environment = os.environ | {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert completed.stdout == json.dumps(payload, sort_keys=True) + "\n"
    assert payload["attempt_id"] == "initial"
    assert payload["cluster"]["num_processes"] == 4
    assert payload["deepspeed_config"] == "/root/workspace/ILVR/configs/deepspeed_zero2.json"
    assert len(payload["deepspeed_config_digest"]) == 64
    assert set(tmp_path.iterdir()) == before


def test_launch_plan_rejects_runtime_profile_with_wrong_zero_stage(monkeypatch, tmp_path):
    """Catches a canonical path being paired with a config for another ZeRO stage."""
    config = tmp_path / "deepspeed_zero2.json"
    config.write_text(
        json.dumps(
            {
                "bf16": {"enabled": True},
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": "auto",
                "train_batch_size": "auto",
                "zero_optimization": {"stage": 1},
            }
        )
    )
    monkeypatch.setattr(
        "src.multinode.canonical_deepspeed_config_path", lambda _: config
    )

    with pytest.raises(ValueError, match="zero_optimization.stage"):
        build_launch_plan(
            stage="stage1",
            tuning_method="full",
            zero_stage=2,
            run_id="stage1-full-zero2-four-gpu",
            main_process_port=29616,
        )


def test_launch_plan_rejects_runtime_profile_with_fixed_global_batch(monkeypatch, tmp_path):
    """Catches a copied config replacing the world-size-derived batch contract."""
    config = tmp_path / "deepspeed_zero2.json"
    config.write_text(
        json.dumps(
            {
                "bf16": {"enabled": True},
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": "auto",
                "train_batch_size": 4,
                "zero_optimization": {"stage": 2},
            }
        )
    )
    monkeypatch.setattr(
        "src.multinode.canonical_deepspeed_config_path", lambda _: config
    )

    with pytest.raises(ValueError, match="train_batch_size"):
        build_launch_plan(
            stage="stage1",
            tuning_method="full",
            zero_stage=2,
            run_id="stage1-full-zero2-four-gpu",
            main_process_port=29617,
        )
