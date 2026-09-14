import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "run_stage2_training.sh"
REQUIRED_ENV = {
    "STAGE1_CHECKPOINT": "/models/stage 1",
    "DATA_PATH": "/datasets/train data.jsonl",
    "IMAGE_ROOT": "/datasets/images root",
    "OUTPUT_DIR": "/outputs/stage 2",
    "LOG_FILE": "/logs/stage 2.log",
}


def run_launcher(*args, env_overrides=None, cwd=None):
    env = {"PATH": os.environ["PATH"], "STAGE2_DRY_RUN": "true", **REQUIRED_ENV}
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def dry_run_tokens(result):
    assert result.returncode == 0, result.stderr
    return shlex.split(result.stdout.strip())


def option_value(tokens: list[str], option: str) -> str:
    return tokens[tokens.index(option) + 1]


def assert_four_rank_prefix(tokens, rank, port):
    """Catches Accelerate receiving a partial or single-node topology."""
    expected = {
        "--num_machines": "2",
        "--num_processes": "4",
        "--machine_rank": str(rank),
        "--main_process_ip": "10.1.102.41",
        "--main_process_port": str(port),
    }
    for flag, value in expected.items():
        assert tokens[tokens.index(flag) + 1] == value


def native_ddp_environment(rank="0"):
    return {
        "ILVR_STAGE2_BACKEND": "ddp",
        "ILVR_NUM_MACHINES": "2",
        "ILVR_NUM_PROCESSES": "4",
        "ILVR_MACHINE_RANK": rank,
        "ILVR_MAIN_PROCESS_IP": "10.1.102.41",
        "ILVR_MAIN_PROCESS_PORT": "29871",
        "ILVR_SOCKET_IFNAME": "bond1.102",
    }


@pytest.fixture
def native_ddp_worker_dry_run_destinations():
    run_id = f"task1-worker-dry-{uuid.uuid4().hex}"
    output_dir = REPO_ROOT / "checkpoints" / run_id
    log_dir = REPO_ROOT / "logs" / "stage2-ddp-dev" / run_id
    log_file = log_dir / "train.log"
    assert not output_dir.exists()
    assert not log_dir.exists()
    try:
        yield output_dir, log_file
    finally:
        for path in (output_dir, log_dir):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()


def test_stage2_native_ddp_worker_dry_run_uses_multi_gpu_without_deepspeed(
    native_ddp_worker_dry_run_destinations,
):
    """Catches the DDP worker selecting the legacy DeepSpeed launcher backend."""
    output_dir, log_file = native_ddp_worker_dry_run_destinations
    tokens = dry_run_tokens(
        run_launcher(
            "--enable_stage2_objective",
            "--lambda_dyn", "0.75",
            "--lambda_vis", "0.25",
            "--reference_model_path", "/models/reference",
            env_overrides={
                **native_ddp_environment(),
                "OUTPUT_DIR": str(output_dir),
                "LOG_FILE": str(log_file),
            },
        )
    )

    assert_four_rank_prefix(tokens, rank=0, port=29871)
    assert "--multi_gpu" in tokens
    assert "--use_deepspeed" not in tokens
    assert "--deepspeed_config_file" not in tokens
    assert "--deepspeed_config" not in tokens
    assert option_value(tokens, "--lambda_dyn") == "0.75"
    assert option_value(tokens, "--lambda_vis") == "0.25"
    assert option_value(tokens, "--reference_model_path") == "/models/reference"
    assert option_value(tokens, "--save_model_path") == str(output_dir)
    assert option_value(tokens, "--log_file") == str(log_file)
    assert not output_dir.exists()
    assert not log_file.exists()
    assert not log_file.parent.exists()


def test_stage2_native_ddp_worker_rejects_incomplete_topology():
    """Catches native DDP entering the launcher's single-node fallback."""
    environment = native_ddp_environment()
    environment.pop("ILVR_SOCKET_IFNAME")

    result = run_launcher(env_overrides=environment)

    assert result.returncode == 2
    assert "all distributed variables" in result.stderr


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("ACCELERATE_USE_DEEPSPEED", "true"),
        ("ACCELERATE_USE_FSDP", "1"),
        ("ACCELERATE_USE_MEGATRON_LM", "yes"),
        ("ACCELERATE_USE_CPU", "on"),
        ("ACCELERATE_DYNAMO_BACKEND", "inductor"),
    ),
)
def test_stage2_native_ddp_worker_rejects_inherited_accelerate_mode(name, value):
    """Catches a remote worker honoring an inherited conflicting backend mode."""
    result = run_launcher(
        env_overrides={**native_ddp_environment(), name: value}
    )

    assert result.returncode == 2
    assert name in result.stderr


def test_stage2_native_ddp_worker_rejects_a_deepspeed_training_config():
    """Catches an invalid native-DDP/DeepSpeed mixed worker launch."""
    result = run_launcher(
        env_overrides={
            **native_ddp_environment(),
            "DEEPSPEED_CONFIG": "/configs/zero3.json",
        }
    )

    assert result.returncode == 2
    assert "DEEPSPEED_CONFIG" in result.stderr


def test_stage2_multinode_dry_run_emits_rank_zero_four_process_command():
    """Catches Stage 2 omitting the rank-zero four-GPU Accelerate contract."""
    tokens = dry_run_tokens(
        run_launcher(
            env_overrides={
                "ILVR_NUM_MACHINES": "2",
                "ILVR_NUM_PROCESSES": "4",
                "ILVR_MACHINE_RANK": "0",
                "ILVR_MAIN_PROCESS_IP": "10.1.102.41",
                "ILVR_MAIN_PROCESS_PORT": "29622",
                "ILVR_SOCKET_IFNAME": "bond1.102",
                "DEEPSPEED_CONFIG": str(REPO_ROOT / "configs/deepspeed_zero2.json"),
            }
        )
    )

    assert_four_rank_prefix(tokens, rank=0, port=29622)
    assert tokens[tokens.index("--deepspeed_multinode_launcher") + 1] == "standard"
    assert tokens[tokens.index("--deepspeed_config_file") + 1] == str(
        REPO_ROOT / "configs/deepspeed_zero2.json"
    )


def test_production_entrypoint_imports_with_build_marker_then_restores_platform():
    """Catches datasets rejecting 3.11.15+ or normalization leaking after imports."""
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{REPO_ROOT / 'transformers/src'}:{REPO_ROOT / 'src'}",
    }
    completed = subprocess.run(
        [
            str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
            "-c",
            (
                "import platform; "
                "original = platform.python_version; "
                "assert original().endswith('+'), original(); "
                "import main; "
                "assert platform.python_version is original; "
                "assert platform.python_version().endswith('+')"
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_stage2_launcher_resolves_repo_runtime_and_propagates_arguments(tmp_path):
    """Catches cwd-dependent paths, wrong Stage 2 bindings, or lost CLI arguments."""
    result = run_launcher(
        "--max_steps",
        "1",
        "--evaluation_output_path",
        "/reports/stage 2.json",
        cwd=tmp_path,
    )

    tokens = dry_run_tokens(result)
    expected_pythonpath = f"{REPO_ROOT / 'transformers/src'}:{REPO_ROOT / 'src'}"
    assert tokens[:3] == [
        f"PYTHONPATH={expected_pythonpath}",
        "TOKENIZERS_PARALLELISM=false",
        str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
    ]
    assert tokens[3:] == [
        str(REPO_ROOT / "src/main.py"),
        "--stage",
        "stage2",
        "--load_model_path",
        REQUIRED_ENV["STAGE1_CHECKPOINT"],
        "--data_path",
        REQUIRED_ENV["DATA_PATH"],
        "--image_root",
        REQUIRED_ENV["IMAGE_ROOT"],
        "--save_model_path",
        REQUIRED_ENV["OUTPUT_DIR"],
        "--log_file",
        REQUIRED_ENV["LOG_FILE"],
        "--latent_size",
        "8",
        "--supervise_latent_boundaries",
        "--learning_rate",
        "1e-5",
        "--gradient_accumulation_steps",
        "8",
        "--per_device_train_batch_size",
        "1",
        "--attention_implementation",
        "sdpa",
        "--no-gradient_checkpointing",
        "--max_seq_length_train",
        "8192",
        "--tuning_method",
        "full",
        "--lora_r",
        "32",
        "--lora_alpha",
        "64",
        "--lora_dropout",
        "0.05",
        "--lora_target_modules",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--max_steps",
        "1",
        "--evaluation_output_path",
        "/reports/stage 2.json",
    ]


def test_stage2_launcher_propagates_explicit_dora_settings():
    """Catches PEFT environment settings being defaulted or reordered incorrectly."""
    tokens = dry_run_tokens(
        run_launcher(
            env_overrides={
                "TUNING_METHOD": "dora",
                "LORA_R": "16",
                "LORA_ALPHA": "32",
                "LORA_DROPOUT": "0.1",
                "LORA_TARGET_MODULES": "q_proj,v_proj,down_proj",
            }
        )
    )

    assert option_value(tokens, "--tuning_method") == "dora"
    assert option_value(tokens, "--lora_r") == "16"
    assert option_value(tokens, "--lora_alpha") == "32"
    assert option_value(tokens, "--lora_dropout") == "0.1"
    assert option_value(tokens, "--lora_target_modules") == (
        "q_proj,v_proj,down_proj"
    )


def test_stage2_launcher_maps_resume_without_reinterpreting_stage1_origin():
    """Catches resume replacing the explicit Stage 1 provenance argument."""
    result = run_launcher(env_overrides={"RESUME_CHECKPOINT": "/resume/checkpoint 10"})

    tokens = dry_run_tokens(result)
    load_index = tokens.index("--load_model_path")
    resume_index = tokens.index("--resume_from_checkpoint_path")
    assert tokens[load_index + 1] == REQUIRED_ENV["STAGE1_CHECKPOINT"]
    assert tokens[resume_index + 1] == "/resume/checkpoint 10"


def test_stage2_launcher_resume_without_explicit_method_omits_all_tuning_options():
    """Catches environment defaults or stray LoRA values overriding the sidecar."""
    tokens = dry_run_tokens(
        run_launcher(
            env_overrides={
                "RESUME_CHECKPOINT": "/resume/checkpoint 10",
                "LORA_R": "16",
                "LORA_ALPHA": "32",
                "LORA_DROPOUT": "0.1",
                "LORA_TARGET_MODULES": "q_proj,v_proj",
            }
        )
    )

    for option in (
        "--tuning_method",
        "--lora_r",
        "--lora_alpha",
        "--lora_dropout",
        "--lora_target_modules",
    ):
        assert option not in tokens


def test_stage2_launcher_resume_with_explicit_method_forwards_complete_settings():
    """Catches an intentional resume override bypassing compatibility validation."""
    tokens = dry_run_tokens(
        run_launcher(
            env_overrides={
                "RESUME_CHECKPOINT": "/resume/checkpoint 10",
                "TUNING_METHOD": "lora",
                "LORA_R": "16",
                "LORA_ALPHA": "32",
                "LORA_DROPOUT": "0.1",
                "LORA_TARGET_MODULES": "q_proj,v_proj",
            }
        )
    )

    assert option_value(tokens, "--tuning_method") == "lora"
    assert option_value(tokens, "--lora_r") == "16"
    assert option_value(tokens, "--lora_alpha") == "32"
    assert option_value(tokens, "--lora_dropout") == "0.1"
    assert option_value(tokens, "--lora_target_modules") == "q_proj,v_proj"


@pytest.mark.parametrize("method", ("qlora", "adapter", "FULL", ""))
def test_stage2_launcher_rejects_unknown_tuning_method(method):
    """Catches unsupported tuning methods reaching model construction."""
    result = run_launcher(env_overrides={"TUNING_METHOD": method})

    assert result.returncode == 2
    assert "TUNING_METHOD must be full, lora, or dora" in result.stderr


@pytest.mark.parametrize("name", ("LORA_R", "LORA_ALPHA"))
@pytest.mark.parametrize("value", ("0", "-1", "1.5", "rank"))
def test_stage2_launcher_rejects_invalid_positive_lora_integers(name, value):
    """Catches malformed adapter dimensions before model startup."""
    result = run_launcher(env_overrides={name: value})

    assert result.returncode == 2
    assert f"{name} must be a positive integer" in result.stderr


@pytest.mark.parametrize(
    "value", ("nan", "inf", "-inf", "-0.1", "1", "1.1", "dropout")
)
def test_stage2_launcher_rejects_invalid_lora_dropout(value):
    """Catches non-finite or out-of-range adapter dropout before model startup."""
    result = run_launcher(env_overrides={"LORA_DROPOUT": value})

    assert result.returncode == 2
    assert "LORA_DROPOUT must be a finite number in [0, 1)" in result.stderr


@pytest.mark.parametrize(
    "value", ("", ",", "visual.q_proj", "rotary_emb", "q_proj,embed_tokens")
)
def test_stage2_launcher_rejects_invalid_lora_targets(value):
    """Catches empty, visual, or out-of-scope adapter targets before startup."""
    result = run_launcher(env_overrides={"LORA_TARGET_MODULES": value})

    assert result.returncode == 2
    assert "LORA_TARGET_MODULES must contain only approved decoder targets" in result.stderr


@pytest.mark.parametrize(
    "env_overrides",
    (
        {
            "TUNING_METHOD": "lora",
            "LORA_TARGET_MODULES": "q_proj,q_proj",
        },
        {
            "RESUME_CHECKPOINT": "/resume/checkpoint 10",
            "LORA_TARGET_MODULES": "q_proj,q_proj",
        },
    ),
    ids=("fresh-explicit", "resume-inherited"),
)
def test_stage2_launcher_rejects_duplicate_lora_targets(env_overrides):
    """Catches duplicate targets reaching Python or being ignored on resume."""
    result = run_launcher(env_overrides=env_overrides)

    assert result.returncode == 2
    assert "LORA_TARGET_MODULES must not contain duplicate targets" in result.stderr


def test_stage2_launcher_omits_resume_for_fresh_training():
    """Catches an empty resume value being forwarded as a real checkpoint."""
    tokens = dry_run_tokens(run_launcher())

    assert "--resume_from_checkpoint_path" not in tokens


@pytest.mark.parametrize(
    "missing_name", ("STAGE1_CHECKPOINT", "DATA_PATH", "IMAGE_ROOT", "OUTPUT_DIR")
)
def test_stage2_launcher_fails_when_required_environment_is_missing(missing_name):
    """Catches silently falling back to ambiguous training paths."""
    env = {"PATH": os.environ["PATH"], "STAGE2_DRY_RUN": "true", **REQUIRED_ENV}
    del env[missing_name]

    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, capture_output=True, text=True
    )

    assert result.returncode != 0
    assert f"set {missing_name}" in result.stderr


def test_stage2_launcher_has_valid_bash_syntax():
    """Catches shell syntax regressions before a training process is launched."""
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
