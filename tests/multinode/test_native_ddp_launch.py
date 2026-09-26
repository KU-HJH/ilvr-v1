import json
import os
import builtins
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

import pytest

from src.multinode import build_launch_plan
import src.multinode_runtime as runtime
import src.multinode as multinode
from src.stage2.objective import Stage2ObjectiveConfig


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def checkpoint_directory():
    root = Path(REPO_ROOT) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="task1-preflight-", dir=root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def run_stage2_coordinator(
    *arguments, environment=None, run_id="ddp-contract-test", attempt_id="initial"
):
    env = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": f"{REPO_ROOT}/transformers/src:{REPO_ROOT}/src",
        "CUDA_VISIBLE_DEVICES": "",
        "STAGE1_CHECKPOINT": "/models/stage1",
        "DATA_PATH": "/data/train.jsonl",
        "IMAGE_ROOT": "/data/images",
        "TUNING_METHOD": "dora",
        **(environment or {}),
    }
    return subprocess.run(
        [
            f"{REPO_ROOT}/venv/ilvr-venv/bin/python",
            "-m", "src.multinode", "train-stage2",
            "--run-id", run_id,
            "--attempt-id", attempt_id,
            "--main-process-port", "29871",
            "--dry-run",
            *arguments,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def coordinator_dry_run_destinations():
    run_id = f"task1-coordinator-dry-{uuid.uuid4().hex}"
    paths = (
        Path("/root/workspace/ILVR/checkpoints") / run_id,
        Path("/root/workspace/ILVR/results") / run_id,
        Path("/root/workspace/ILVR/logs/multinode") / run_id,
        Path("/root/workspace/ILVR/logs/zero-stage-compatibility") / run_id,
    )
    assert all(not path.exists() for path in paths)
    try:
        yield run_id, paths
    finally:
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()


def test_stage2_native_ddp_plan_has_no_deepspeed_identity():
    """Catches DDP plans retaining a ZeRO stage or DeepSpeed config identity."""
    plan = build_launch_plan(
        stage="stage2",
        tuning_method="dora",
        backend="ddp",
        run_id="ddp-contract-test",
        main_process_port=29871,
    )

    assert plan.backend == "ddp"
    assert plan.zero_stage is None
    assert plan.deepspeed_config is None
    assert plan.deepspeed_config_digest is None
    assert plan.world_size == 4


def test_stage2_native_ddp_plan_rejects_a_zero_stage():
    """Catches mixed native-DDP and ZeRO launch identities."""
    with pytest.raises(ValueError, match="native DDP requires Stage2 without a ZeRO stage"):
        build_launch_plan(
            stage="stage2",
            tuning_method="dora",
            backend="ddp",
            zero_stage=1,
            run_id="ddp-contract-test",
            main_process_port=29871,
        )


def test_launch_plan_rejects_an_unknown_backend_explicitly():
    """Catches an unknown backend being misclassified as DeepSpeed."""
    with pytest.raises(ValueError, match="backend"):
        build_launch_plan(
            stage="stage2",
            tuning_method="dora",
            backend="foreign",
            run_id="ddp-contract-test",
            main_process_port=29871,
        )


def test_resolve_stage2_objective_config_uses_training_parser_defaults_and_values():
    """Catches coordinator defaults diverging from the Stage 2 training parser."""
    config = runtime.resolve_stage2_objective_config(
        (
            "--enable_stage2_objective",
            "--lambda_dyn", "0.75",
            "--lambda_vis=0.25",
            "--reference_model_path", "/models/reference",
            "--gamma_H", "0.4",
            "--m_V", "0.2",
            "--entropy_eps", "0.00001",
            "--pre_window", "2",
            "--post_window=3",
            "--stage2_m_execution", "packed_suffix",
            "--vision_token_reduction", "token_loss_mean",
        ),
        supervise_latent_boundaries=True,
    )

    assert config.to_dict() == {
        "enable_stage2_objective": True,
        "lambda_dyn": 0.75,
        "lambda_vis": 0.25,
        "reference_model_path": "/models/reference",
        "gamma_H": 0.4,
        "m_V": 0.2,
        "entropy_eps": 0.00001,
        "pre_window": 2,
        "post_window": 3,
        "stage2_m_execution": "packed_suffix",
        "vision_token_reduction": "token_loss_mean",
    }


def test_resolve_stage2_objective_config_defaults_m_execution_to_full_replay():
    """Catches an absent saved/CLI mode changing legacy coordinator behavior."""
    config = runtime.resolve_stage2_objective_config(
        ("--enable_stage2_objective", "--lambda_vis", "0.25"),
        supervise_latent_boundaries=True,
    )

    assert config.to_dict()["stage2_m_execution"] == "full_replay"


def test_stage2_objective_arguments_are_forwarded_by_the_coordinator_allowlist():
    """Catches objective values being dropped before the worker launcher."""
    arguments = (
        "--no-enable_stage2_objective",
        "--lambda_dyn", "0",
        "--lambda_vis=0",
        "--reference_model_path", "/models/reference",
        "--gamma_H", "0.4",
        "--m_V", "0.2",
        "--entropy_eps", "0.00001",
        "--pre_window", "2",
        "--post_window=3",
        "--stage2_m_execution", "packed_suffix",
        "--vision_token_reduction", "token_loss_mean",
    )

    assert runtime.validate_stage2_arguments(arguments) == arguments


def _stage2_environment(**overrides):
    return {
        "STAGE1_CHECKPOINT": "/models/stage1",
        "DATA_PATH": "/data/train.jsonl",
        "IMAGE_ROOT": "/data/images",
        **overrides,
    }


def test_native_ddp_environment_sets_backend_without_a_deepspeed_config():
    """Catches native DDP workers receiving an active DeepSpeed config."""
    environment = runtime.validated_stage2_environment(
        _stage2_environment(),
        output_dir="/root/workspace/ILVR/checkpoints/ddp-contract-test",
        log_file="/root/workspace/ILVR/logs/stage2-ddp-dev/ddp-contract-test/train.log",
        deepspeed_config=None,
        backend="ddp",
    )

    assert environment["ILVR_STAGE2_BACKEND"] == "ddp"
    assert "DEEPSPEED_CONFIG" not in environment


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("ACCELERATE_USE_DEEPSPEED", "yes"),
        ("ACCELERATE_USE_FSDP", "T"),
        ("ACCELERATE_USE_MEGATRON_LM", "on"),
        ("ACCELERATE_USE_CPU", "1"),
        ("ACCELERATE_USE_DEEPSPEED", ""),
        ("ACCELERATE_USE_FSDP", "maybe"),
        ("ACCELERATE_DYNAMO_BACKEND", "inductor"),
    ),
)
def test_native_ddp_environment_rejects_conflicting_accelerate_mode(name, value):
    """Catches inherited Accelerate modes changing the native GPU launcher."""
    with pytest.raises(ValueError, match=name):
        runtime.validated_stage2_environment(
            _stage2_environment(**{name: value}),
            output_dir="/root/workspace/ILVR/checkpoints/ddp-contract-test",
            log_file="/root/workspace/ILVR/logs/stage2-ddp-dev/ddp-contract-test/train.log",
            deepspeed_config=None,
            backend="ddp",
        )


@pytest.mark.parametrize("value", ("n", "NO", "f", "False", "off", "0"))
def test_native_ddp_environment_accepts_disabled_accelerate_plugins(value):
    """Catches valid explicit false spellings being rejected as conflicts."""
    environment = runtime.validated_stage2_environment(
        _stage2_environment(
            ACCELERATE_USE_DEEPSPEED=value,
            ACCELERATE_USE_FSDP=value,
            ACCELERATE_USE_MEGATRON_LM=value,
            ACCELERATE_USE_CPU=value,
            ACCELERATE_DYNAMO_BACKEND="NO",
            ACCELERATE_DEEPSPEED_ZERO_STAGE="3",
        ),
        output_dir="/root/workspace/ILVR/checkpoints/ddp-contract-test",
        log_file="/root/workspace/ILVR/logs/stage2-ddp-dev/ddp-contract-test/train.log",
        deepspeed_config=None,
        backend="ddp",
    )

    assert environment["ILVR_STAGE2_BACKEND"] == "ddp"


def test_real_stage2_coordinator_dry_run_freezes_ddp_objective_and_worker_inputs(
    coordinator_dry_run_destinations,
):
    """Catches coordinator backend/topology/objective values being lost in real argv."""
    run_id, destinations = coordinator_dry_run_destinations
    result = run_stage2_coordinator(
        "--backend", "ddp",
        "--enable_stage2_objective",
        "--lambda_dyn", "0.75",
        "--lambda_vis", "0.25",
        "--reference_model_path", "/models/reference",
        "--stage2_m_execution", "packed_suffix",
        run_id=run_id,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["plan"]["backend"] == "ddp"
    assert payload["plan"]["zero_stage"] is None
    assert payload["plan"]["deepspeed_config"] is None
    assert payload["plan"]["stage2_objective_config"]["lambda_dyn"] == 0.75
    assert payload["plan"]["stage2_objective_config"]["stage2_m_execution"] == "packed_suffix"
    local = payload["local"]
    assert "ILVR_STAGE2_BACKEND=ddp" in local
    assert "DEEPSPEED_CONFIG=" not in " ".join(local)
    assert "--lambda_dyn" in local
    assert local[local.index("--lambda_dyn") + 1] == "0.75"
    assert "--stage2_m_execution" in local
    assert local[local.index("--stage2_m_execution") + 1] == "packed_suffix"
    assert f"OUTPUT_DIR=/root/workspace/ILVR/checkpoints/{run_id}" in local
    assert all(not path.exists() for path in destinations)


def test_owned_ddp_worker_commands_require_the_coordinator_backend_binding():
    """Catches direct command construction falling back to DeepSpeed at the worker."""
    plan = build_launch_plan(
        stage="stage2",
        tuning_method="dora",
        backend="ddp",
        run_id="ddp-contract-test",
        main_process_port=29871,
    )
    environment = runtime.validated_stage2_environment(
        _stage2_environment(),
        output_dir="/root/workspace/ILVR/checkpoints/ddp-contract-test",
        log_file="/root/workspace/ILVR/logs/stage2-ddp-dev/ddp-contract-test/train.log",
        deepspeed_config=None,
        backend="ddp",
    )
    environment.pop("ILVR_STAGE2_BACKEND")

    with pytest.raises(ValueError, match="ILVR_STAGE2_BACKEND"):
        runtime.build_owned_tmux_commands(
            plan,
            commit="a" * 40,
            stage_environment=environment,
        )


def test_stage2_ddp_coordinator_rejects_forwarded_cpu_training():
    """Catches a native DDP request bypassing the mandated GPU topology."""
    result = run_stage2_coordinator("--backend", "ddp", "--use_cpu")

    assert result.returncode == 2
    assert "--use_cpu" in result.stderr


def test_stage2_deepspeed_coordinator_rejects_active_auxiliary_objective():
    """Catches active auxiliary graphs entering an unsupported ZeRO launch."""
    result = run_stage2_coordinator(
        "--zero-stage", "3",
        "--enable_stage2_objective",
        "--lambda_vis", "0.25",
    )

    assert result.returncode == 2
    assert "native DDP" in result.stderr


def test_native_ddp_node_info_argv_omits_deepspeed_fields():
    """Catches DDP preflight inspecting a nonexistent DeepSpeed policy."""
    plan = build_launch_plan(
        stage="stage2",
        tuning_method="dora",
        backend="ddp",
        run_id="ddp-contract-test",
        main_process_port=29871,
    )
    commands = runtime.build_preflight_commands(
        plan,
        input_paths=("/models/stage1",),
        artifact_roots=("/root/workspace/ILVR/checkpoints",),
        remote_run_paths=("/root/workspace/ILVR/checkpoints/remote",),
        local_run_paths=("/root/workspace/ILVR/checkpoints/local",),
        remote_tmux_sessions=("ilvr-remote",),
        local_tmux_sessions=("ilvr-local",),
    )

    local = commands.local_node_info
    assert local[local.index("--backend") + 1] == "ddp"
    assert "--zero-stage" not in local
    assert "--deepspeed-config" not in local
    assert "--deepspeed-config-digest" not in local


def test_prepare_inference_rejects_native_ddp_until_artifact_support_exists(capsys):
    """Catches a shared plan parser silently enabling unsupported DDP inference."""
    with pytest.raises(SystemExit) as error:
        multinode._prepare_inference_main(
            [
                "--stage", "stage2",
                "--tuning-method", "dora",
                "--backend", "ddp",
                "--run-id", "ddp-contract-test",
                "--main-process-port", "29871",
                "--checkpoint-step", "1",
                "--destination",
                "/root/workspace/ILVR/checkpoints/evaluation-staging/ddp-contract-test/model",
            ]
        )

    assert error.value.code == 2
    assert "unsupported" in capsys.readouterr().err


def _write_stage1_dora_metadata(reference: Path, base: Path):
    from src.stage1.checkpointing import save_stage1_peft_metadata
    from tests.stage1.test_checkpointing import sample_metadata

    reference.mkdir()
    (reference / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    save_stage1_peft_metadata(
        reference,
        replace(sample_metadata(), base_model_reference=str(base)),
    )


def test_entropy_reference_preflight_includes_reference_and_original_dora_base_on_both_nodes(
    checkpoint_directory,
):
    """Catches either node omitting the entropy source or its original DoRA base."""
    base = checkpoint_directory / "original-base"
    reference = checkpoint_directory / "stage1-dora"
    base.mkdir()
    _write_stage1_dora_metadata(reference, base)
    config = Stage2ObjectiveConfig(
        enable_stage2_objective=True,
        lambda_dyn=1.0,
        reference_model_path=str(reference),
    )

    required = runtime.stage2_reference_preflight_paths(config)
    plan = build_launch_plan(
        stage="stage2",
        tuning_method="dora",
        backend="ddp",
        stage2_objective_config=config,
        run_id="ddp-contract-test",
        main_process_port=29871,
    )
    commands = runtime.build_preflight_commands(
        plan,
        input_paths=required,
        artifact_roots=("/root/workspace/ILVR/checkpoints",),
        remote_run_paths=("/root/workspace/ILVR/checkpoints/remote",),
        local_run_paths=("/root/workspace/ILVR/checkpoints/local",),
        remote_tmux_sessions=("ilvr-remote",),
        local_tmux_sessions=("ilvr-local",),
    )

    assert required == (reference.resolve(), base.resolve())
    local_values = [
        commands.local_node_info[index + 1]
        for index, value in enumerate(commands.local_node_info)
        if value == "--input-path"
    ]
    remote_argv = subprocess.list2cmdline(commands.remote_node_info)
    assert local_values == [str(reference.resolve()), str(base.resolve())]
    assert str(reference.resolve()) in remote_argv
    assert str(base.resolve()) in remote_argv


def test_entropy_reference_preflight_rejects_missing_reference(checkpoint_directory):
    """Catches a missing entropy reference surviving until model loading."""
    config = Stage2ObjectiveConfig(
        enable_stage2_objective=True,
        lambda_dyn=1.0,
        reference_model_path=str(checkpoint_directory / "missing"),
    )

    with pytest.raises(ValueError, match="reference_model_path.*local"):
        runtime.stage2_reference_preflight_paths(config)


def test_entropy_reference_preflight_rejects_nonlocal_dora_base(checkpoint_directory):
    """Catches a Hub-style original DoRA base reaching the download boundary."""
    reference = checkpoint_directory / "stage1-dora"
    _write_stage1_dora_metadata(reference, Path("forbidden/remote-model"))
    config = Stage2ObjectiveConfig(
        enable_stage2_objective=True,
        lambda_dyn=1.0,
        reference_model_path=str(reference),
    )

    with pytest.raises(ValueError, match="base_model_reference.*local"):
        runtime.stage2_reference_preflight_paths(config)


def test_zero_entropy_weight_skips_reference_preflight(checkpoint_directory):
    """Catches CE-only/visual-only runs unnecessarily requiring a reference."""
    config = Stage2ObjectiveConfig(
        enable_stage2_objective=True,
        lambda_dyn=0.0,
        reference_model_path=str(checkpoint_directory / "missing"),
    )

    assert runtime.stage2_reference_preflight_paths(config) == ()


@pytest.mark.parametrize("enabled,weight", [(True, 0.0), (False, 1.0)])
def test_zero_entropy_weight_does_not_import_stage1_metadata_code(monkeypatch, enabled, weight):
    """Catches the disabled entropy path warming the PEFT metadata boundary."""
    original_import = builtins.__import__

    def reject_stage1_checkpointing(name, *args, **kwargs):
        if name == "src.stage1.checkpointing":
            raise AssertionError("Stage 1 metadata import reached")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_stage1_checkpointing)
    config = Stage2ObjectiveConfig(enable_stage2_objective=enabled, lambda_dyn=weight)

    assert runtime.stage2_reference_preflight_paths(config) == ()
