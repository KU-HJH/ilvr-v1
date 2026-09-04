import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "run_stage2_training.sh"
REQUIRED_ENV = {
    "STAGE1_CHECKPOINT": "/models/stage 1",
    "DATA_PATH": "/datasets/train data.jsonl",
    "IMAGE_ROOT": "/datasets/images root",
    "OUTPUT_DIR": "/outputs/stage 2",
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


CANONICAL_DEFAULT_LOG = (
    REPO_ROOT / "logs/zero-stage-compatibility/stage 2/train.log"
)


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
                "platform.python_version = lambda: original() + '+'; "
                "marked = platform.python_version; "
                "import main; "
                "assert platform.python_version is marked; "
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
        "ASSISTANT_PLAN_RENDERING=concatenated-v1",
    ]
    assert tokens[3:5] == [
        str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
        str(REPO_ROOT / "src/main.py"),
    ]
    assert tokens[5:] == [
        "--stage",
        "stage2",
        "--load_model_path",
        REQUIRED_ENV["STAGE1_CHECKPOINT"],
        "--data_path",
        REQUIRED_ENV["DATA_PATH"],
        "--image_root",
        REQUIRED_ENV["IMAGE_ROOT"],
        "--dataloader_num_workers",
        "0",
        "--save_model_path",
        REQUIRED_ENV["OUTPUT_DIR"],
        "--log_file",
        str(CANONICAL_DEFAULT_LOG),
        "--latent_size",
        "8",
        "--stage2_diagnostics_interval",
        "0",
        "--ce_weight",
        "1.0",
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
        "--stage2_length_manifest",
        str(REPO_ROOT / "results/stage 2/stage2-length-manifest.json"),
        "--stage2_length_manifest_sha256",
        "0" * 64,
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


def test_stage2_launcher_propagates_opt_in_stage2b_environment_without_resume_state():
    """Catches dropped objective settings or Stage 2A initialization becoming resume."""
    tokens = dry_run_tokens(
        run_launcher(
            env_overrides={
                "STEP_PREFERENCE_ENABLED": "true",
                "STEP_PREFERENCE_MODE": "stage2b",
                "STEP_NEGATIVE_PATH": "/results/negatives.jsonl",
                "STEP_NEGATIVE_MANIFEST": "/results/manifest.json",
                "STEP_DPO_BETA": "0.2",
                "CE_WEIGHT": "0.7",
                "STEP_DPO_WEIGHT": "1.5",
                "STEP_ENTROPY_WEIGHT": "0.4",
                "STEP_ENTROPY_MARGIN": "0.3",
                "STAGE2A_RATIO": "0.6",
                "STAGE2_TOTAL_STEPS": "10",
                "STAGE2A_CHECKPOINT_PATH": "/models/stage 2a",
            }
        )
    )

    assert "--step_preference_enabled" in tokens
    assert option_value(tokens, "--step_preference_mode") == "stage2b"
    assert option_value(tokens, "--step_negative_path") == "/results/negatives.jsonl"
    assert option_value(tokens, "--step_negative_manifest") == "/results/manifest.json"
    assert option_value(tokens, "--step_dpo_beta") == "0.2"
    assert option_value(tokens, "--ce_weight") == "0.7"
    assert option_value(tokens, "--step_dpo_weight") == "1.5"
    assert option_value(tokens, "--step_entropy_weight") == "0.4"
    assert option_value(tokens, "--step_entropy_margin") == "0.3"
    assert option_value(tokens, "--stage2a_ratio") == "0.6"
    assert option_value(tokens, "--stage2_total_steps") == "10"
    assert option_value(tokens, "--stage2a_checkpoint_path") == "/models/stage 2a"
    assert "--resume_from_checkpoint_path" not in tokens
    assert "--tuning_method" not in tokens


@pytest.mark.parametrize(
    ("ce_weight", "dpo_weight", "entropy_weight"),
    [("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1")],
)
def test_stage2b_launcher_forwards_isolated_objective_coefficient_triplets(
    ce_weight, dpo_weight, entropy_weight
):
    """Catches launcher ownership changing an isolated ablation request."""
    tokens = dry_run_tokens(
        run_launcher(
            env_overrides={
                "STEP_PREFERENCE_ENABLED": "true",
                "STEP_PREFERENCE_MODE": "stage2b",
                "STEP_NEGATIVE_PATH": "/results/negatives.jsonl",
                "STEP_NEGATIVE_MANIFEST": "/results/manifest.json",
                "STAGE2A_CHECKPOINT_PATH": "/models/stage2a",
                "CE_WEIGHT": ce_weight,
                "STEP_DPO_WEIGHT": dpo_weight,
                "STEP_ENTROPY_WEIGHT": entropy_weight,
            }
        )
    )

    assert option_value(tokens, "--ce_weight") == ce_weight
    assert option_value(tokens, "--step_dpo_weight") == dpo_weight
    assert option_value(tokens, "--step_entropy_weight") == entropy_weight


@pytest.mark.parametrize(
    "name", ("CE_WEIGHT", "STEP_DPO_WEIGHT", "STEP_ENTROPY_WEIGHT")
)
@pytest.mark.parametrize("value", ("-0.1", "nan", "inf"))
def test_stage2b_launcher_rejects_invalid_objective_coefficients(name, value):
    """Catches invalid objective coefficients surviving launcher preflight."""
    result = run_launcher(
        env_overrides={
            "STEP_PREFERENCE_ENABLED": "true",
            "STEP_PREFERENCE_MODE": "stage2b",
            "STEP_NEGATIVE_PATH": "/results/negatives.jsonl",
            "STEP_NEGATIVE_MANIFEST": "/results/manifest.json",
            "STAGE2A_CHECKPOINT_PATH": "/models/stage2a",
            name: value,
        }
    )

    assert result.returncode == 2
    assert f"{name} must be a finite non-negative number" in result.stderr


def test_stage2b_launcher_dry_run_identifies_space_lf_length_manifest_rendering():
    """Catches Stage 2B auditing lengths from concatenated assistant bytes."""
    tokens = dry_run_tokens(
        run_launcher(
            env_overrides={
                "STEP_PREFERENCE_ENABLED": "true",
                "STEP_PREFERENCE_MODE": "stage2b",
                "STEP_NEGATIVE_PATH": "/results/negatives.jsonl",
                "STEP_NEGATIVE_MANIFEST": "/results/manifest.json",
                "STAGE2A_CHECKPOINT_PATH": "/models/stage2a",
            }
        )
    )

    assert "ASSISTANT_PLAN_RENDERING=adjacent-text-space-lf-v1" in tokens


def test_ordinary_stage2_launcher_dry_run_identifies_concatenated_length_rendering():
    """Catches disabled step preference changing the legacy length identity."""
    tokens = dry_run_tokens(run_launcher())

    assert "ASSISTANT_PLAN_RENDERING=concatenated-v1" in tokens


@pytest.mark.parametrize("value", ("1", "true", "True"))
def test_stage2_launcher_propagates_explicit_packing_opt_in(value):
    """Catches a valid packed Stage 2B request being dropped by the launcher."""
    tokens = dry_run_tokens(
        run_launcher(
            env_overrides={
                "STEP_PREFERENCE_ENABLED": "true",
                "STEP_PREFERENCE_MODE": "stage2b",
                "STEP_NEGATIVE_PATH": "/results/negatives.jsonl",
                "STEP_NEGATIVE_MANIFEST": "/results/manifest.json",
                "STAGE2A_CHECKPOINT_PATH": "/models/stage2a",
                "STEP_PREFERENCE_PACKING": value,
            }
        )
    )
    assert "--step_preference_packing" in tokens


@pytest.mark.parametrize("value", ("", "0", "false", "False"))
def test_stage2_launcher_omits_disabled_packing(value):
    """Catches false-like packing values accidentally enabling the packed path."""
    tokens = dry_run_tokens(
        run_launcher(
            env_overrides={
                "STEP_PREFERENCE_ENABLED": "true",
                "STEP_PREFERENCE_MODE": "stage2b",
                "STEP_NEGATIVE_PATH": "/results/negatives.jsonl",
                "STEP_NEGATIVE_MANIFEST": "/results/manifest.json",
                "STAGE2A_CHECKPOINT_PATH": "/models/stage2a",
                "STEP_PREFERENCE_PACKING": value,
            }
        )
    )
    assert "--step_preference_packing" not in tokens


def test_stage2_launcher_rejects_unknown_packing_value():
    """Catches ambiguous packing environment values reaching Python."""
    result = run_launcher(env_overrides={"STEP_PREFERENCE_PACKING": "yes"})
    assert result.returncode == 2
    assert "STEP_PREFERENCE_PACKING" in result.stderr


def test_stage2_launcher_does_not_silently_ignore_packing_without_stage2b():
    """Catches a true packing request disappearing when Stage 2B is not enabled."""
    tokens = dry_run_tokens(
        run_launcher(env_overrides={"STEP_PREFERENCE_PACKING": "true"})
    )
    assert "--step_preference_packing" in tokens


def test_stage2_launcher_false_step_preference_keeps_disabled_tokens_identical():
    """Catches an explicitly false environment value adding inactive CLI tokens."""
    assert dry_run_tokens(
        run_launcher(env_overrides={"STEP_PREFERENCE_ENABLED": "false"})
    ) == dry_run_tokens(run_launcher())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"STEP_PREFERENCE_ENABLED": "maybe"}, "must be true or false"),
        (
            {"STEP_PREFERENCE_ENABLED": "true", "STEP_PREFERENCE_MODE": "stage2b"},
            "requires STAGE2A_CHECKPOINT_PATH",
        ),
        (
            {
                "STEP_PREFERENCE_ENABLED": "true",
                "STEP_PREFERENCE_MODE": "stage2b",
                "STAGE2A_CHECKPOINT_PATH": "/models/a",
            },
            "requires STEP_NEGATIVE_PATH",
        ),
    ],
)
def test_stage2_launcher_rejects_invalid_opt_in_environment(updates, message):
    """Catches malformed or incomplete Stage 2B settings before launch."""
    result = run_launcher(env_overrides=updates)

    assert result.returncode == 2
    assert message in result.stderr


def test_stage2b_launcher_preflights_stale_artifact_before_creating_run_paths(
    tmp_path,
):
    """Catches output/log/result writes preceding Stage 2B provenance validation."""
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "sample-a",
                "image_input": ["image.bin"],
                "text_input": "question",
                "sequence_plan": [
                    {"type": "text", "content": "before"},
                    {"type": "latent", "helper_image": "helper.bin"},
                    {"type": "text", "content": "after"},
                ],
                "original_final_answer": "answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "image.bin").write_bytes(b"image")
    stage2a = tmp_path / "stage2a"
    stage2a.mkdir()
    manifest_path = tmp_path / "manifest.json"
    negative_path = tmp_path / "negatives.jsonl"
    negative_path.write_bytes(b"")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "source_sha256": "0" * 64,
                "selected_row_indices": [0],
                "stage2a_checkpoint_path": str(stage2a.resolve()),
                "checkpoint_files": [],
                "tokenizer_config_identity": "stale-tokenizer",
                "latent_size": 8,
                "assistant_plan_rendering": "adjacent-text-space-lf-v1",
                "generation_settings": {
                    "min_tokens": 2,
                    "max_new_tokens": 32,
                },
                "negative_sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    experiment_id = f"t8-preflight-{abs(hash(str(tmp_path)))}"
    output_dir = REPO_ROOT / "checkpoints" / experiment_id
    log_dir = REPO_ROOT / "logs/zero-stage-compatibility" / experiment_id
    result_dir = REPO_ROOT / "results" / experiment_id
    assert not output_dir.exists()
    assert not log_dir.exists()
    assert not result_dir.exists()

    try:
        completed = run_launcher(
            env_overrides={
                "STAGE2_DRY_RUN": "false",
                "STAGE1_CHECKPOINT": str(stage2a),
                "DATA_PATH": str(source),
                "IMAGE_ROOT": str(image_root),
                "OUTPUT_DIR": str(output_dir),
                "LOG_FILE": str(log_dir / "train.log"),
                "STEP_PREFERENCE_ENABLED": "true",
                "STEP_PREFERENCE_MODE": "stage2b",
                "STAGE2A_CHECKPOINT_PATH": str(stage2a),
                "STEP_NEGATIVE_PATH": str(negative_path),
                "STEP_NEGATIVE_MANIFEST": str(manifest_path),
            }
        )

        assert completed.returncode != 0
        assert "source JSONL SHA-256 does not match" in completed.stderr
        assert not output_dir.exists()
        assert not log_dir.exists()
        assert not result_dir.exists()
    finally:
        for path in (output_dir, log_dir, result_dir):
            shutil.rmtree(path, ignore_errors=True)


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
