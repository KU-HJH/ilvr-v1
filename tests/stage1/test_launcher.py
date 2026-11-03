from pathlib import Path
import json
import os
import re
import shlex
import subprocess

import pytest

from stage1.benchmark import config_digest
from src.artifact_paths import resolve_run_artifact_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "run_stage1_training.sh"
LEGACY_LAUNCHER = REPO_ROOT / "run_training.sh"
ZERO_MATRIX_LAUNCHER = REPO_ROOT / "run_stage1_zero_matrix.sh"


def run_launcher(script, *args, env=None):
    return subprocess.run(
        [str(script), *map(str, args)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_readme_active_operator_examples_keep_generated_artifacts_canonical():
    """Catches active README examples sending new artifacts to legacy paths."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    historical_block = re.compile(
        r"<!-- BEGIN HISTORICAL NONCANONICAL PERFORMANCE WORKFLOW -->.*?"
        r"<!-- END HISTORICAL NONCANONICAL PERFORMANCE WORKFLOW -->",
        re.DOTALL,
    )
    active_readme, substitutions = historical_block.subn("", readme)

    assert substitutions == 1
    assert "outputs/" not in active_readme
    assert not re.search(r"configs/config_stage[0-9]+\\.json", active_readme)
    assert "zero_checkpoint_root=" in active_readme
    assert "zero_result_root=" in active_readme
    assert "zero_log_root=" in active_readme
    assert 'OUTPUT_DIR="$zero_checkpoint_root/$experiment_id"' in active_readme
    assert 'LOG_FILE="$zero_log_root/$experiment_id/' in active_readme
    assert '--output_json_path "$zero_result_root/$experiment_id/' in active_readme
    assert not re.search(
        r"^(?:SAVE_MODEL_PATH|OUTPUT_DIR|LOG_FILE)=/(?!NHNHOME/WORKSPACE/)",
        active_readme,
        re.MULTILINE,
    )
    assert not re.search(
        r"--(?:output-dir|log-file|save_model_path|output_json_path)\s+/path/to/",
        active_readme,
    )
    assert "STAGE1_CHECKPOINT=/path/to/" in active_readme


def test_zero_matrix_dry_run_orders_stages_methods_and_candidates_without_writes(
    tmp_path,
):
    """Catches the controller reordering rows or creating artifacts during planning."""
    completed = run_launcher(
        ZERO_MATRIX_LAUNCHER,
        "--dry-run",
        "--model",
        "model",
        "--source-data-path",
        tmp_path / "source.jsonl",
        "--data-path",
        tmp_path / "probe.jsonl",
        "--image-root",
        tmp_path / "images",
        "--sample-audit",
        tmp_path / "sample-audit.json",
    )

    assert completed.returncode == 0, completed.stderr
    plans = [json.loads(line) for line in completed.stdout.splitlines()]
    candidates = [
        (row["zero_stage"], row["method"], row["micro_batch"], row["gradient_accumulation"])
        for row in plans
        if row["run_kind"] == "candidate"
    ]
    assert candidates == [
        (stage, method, micro, accumulation)
        for stage in (3, 2, 1)
        for method in ("full", "dora")
        for micro, accumulation in ((2, 4), (4, 2), (8, 1), (1, 8))
    ]
    candidate_plans = [row for row in plans if row["run_kind"] == "candidate"]
    assert all(row["max_steps"] == 3 for row in candidate_plans)
    assert all(row["final_save"] is False for row in candidate_plans)
    for stage in (3, 2, 1):
        for method in ("full", "dora"):
            pair = [
                row for row in plans
                if row["zero_stage"] == stage and row["method"] == method
            ]
            assert [row["run_kind"] for row in pair[-2:]] == [
                "selected-save", "selected-resume"
            ]
            assert pair[-2]["max_steps"] == 3
            assert pair[-2]["save_strategy"] == "steps"
            assert pair[-2]["save_steps"] == 3
            assert pair[-2]["final_save"] is False
            assert pair[-2]["checkpoint_path"] == (
                f'{pair[-2]["checkpoint_dir"]}/checkpoint-3'
            )
            assert pair[-1]["max_steps"] == 4
            assert pair[-1]["save_strategy"] == "steps"
            assert pair[-1]["save_steps"] == 1
            assert pair[-1]["final_save"] is False
            assert pair[-1]["resume_from_kind"] == "selected-save"
            assert pair[-1]["resume_from"] == pair[-2]["checkpoint_path"]
            assert pair[-1]["checkpoint_path"] == (
                f'{pair[-1]["checkpoint_dir"]}/checkpoint-4'
            )
            assert pair[-2]["experiment_id"] != pair[-1]["experiment_id"]
            for row in pair:
                assert row["checkpoint_dir"].split("/")[-1] == row["experiment_id"]
                assert row["result_dir"].split("/")[-1] == row["experiment_id"]
                assert row["log_dir"].split("/")[-1] == row["experiment_id"]
    assert list(tmp_path.iterdir()) == []


def test_batch_probe_keeps_attempt_record_below_its_per_run_result_directory(tmp_path):
    """Catches recreation of a mixed-root attempts/evidence bundle."""
    from tests.stage1.test_benchmark import _run_probe

    completed, records, output_root, _ = _run_probe(tmp_path, fail_batches="8")

    assert completed.returncode == 42
    parts = next(records.glob("*.argv0")).read_bytes().split(b"\0")[:-1]
    run_config = Path(parts[parts.index(b"--benchmark-run-config") + 1].decode())
    attempt_records = list(run_config.parent.glob("attempt-record/evidence/attempt.json"))

    assert len(attempt_records) == 1
    assert not (output_root / "attempts").exists()


def base_args(output_dir, *, method="dora"):
    return (
        "--dry-run",
        "--method",
        method,
        "--model",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "--data-path",
        "/missing/CoMT train;literal/TRAIN.jsonl",
        "--image-root",
        "/missing/CoMT images & literal",
        "--output-dir",
        output_dir,
    )


def assert_four_rank_prefix(tokens, rank, port):
    """Catches Accelerate receiving a partial or single-node topology."""
    expected = {
        "--num_machines": "2",
        "--num_processes": "4",
        "--machine_rank": str(rank),
        "--main_process_ip": "172.16.11.41",
        "--main_process_port": str(port),
    }
    for flag, value in expected.items():
        assert tokens[tokens.index(flag) + 1] == value


def test_stage1_multinode_dry_run_emits_rank_one_four_process_command(tmp_path):
    """Catches Stage 1 omitting the rank-one four-GPU Accelerate contract."""
    env = os.environ.copy()
    env.update(
        {
            "ILVR_NUM_MACHINES": "2",
            "ILVR_NUM_PROCESSES": "4",
            "ILVR_MACHINE_RANK": "1",
            "ILVR_MAIN_PROCESS_IP": "172.16.11.41",
            "ILVR_MAIN_PROCESS_PORT": "29621",
            "ILVR_SOCKET_IFNAME": "bond0.11",
        }
    )

    completed = run_launcher(LAUNCHER, *base_args(tmp_path / "out"), env=env)

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout)
    assert_four_rank_prefix(command, rank=1, port=29621)
    assert command[command.index("--deepspeed_multinode_launcher") + 1] == "standard"
    assert command[command.index("--deepspeed_config_file") + 1] == str(
        REPO_ROOT / "configs/deepspeed_zero3.json"
    )


def live_args(tmp_path, output_dir, *, method="dora"):
    data_path = tmp_path / "TRAIN data.jsonl"
    data_path.write_text("{}\n", encoding="utf-8")
    image_root = tmp_path / "image root"
    image_root.mkdir(exist_ok=True)
    return (
        "--method",
        method,
        "--model",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "--data-path",
        data_path,
        "--image-root",
        image_root,
        "--output-dir",
        output_dir,
    )


def fake_accelerate_environment(tmp_path):
    """Return a copied wrapper with harmless pinned and poisoned boundaries."""
    copied_repo = tmp_path / "copied repository"
    copied_repo.mkdir()
    copied_launcher = copied_repo / LAUNCHER.name
    copied_launcher.write_bytes(LAUNCHER.read_bytes())
    copied_launcher.chmod(0o755)
    config_dir = copied_repo / "configs"
    config_dir.mkdir()
    (config_dir / "deepspeed_zero3.json").write_bytes(
        (REPO_ROOT / "configs/deepspeed_zero3.json").read_bytes()
    )
    source_dir = copied_repo / "src"
    source_dir.mkdir()
    for name in ("artifact_paths.py", "deepspeed_runtime.py"):
        (source_dir / name).write_bytes((REPO_ROOT / "src" / name).read_bytes())
    bin_dir = copied_repo / "venv/ilvr-venv/bin"
    bin_dir.mkdir(parents=True)
    log_path = tmp_path / "accelerate argv.log"
    executable = bin_dir / "accelerate"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%q ' \"$@\" > \"$FAKE_ACCELERATE_LOG\"\n"
        "printf '%s' \"${PYTHON_EXEC-}\" > \"$FAKE_PYTHON_EXEC_LOG\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    repository_python = bin_dir / "python"
    repository_python.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(str(REPO_ROOT / 'venv/ilvr-venv/bin/python'))} \"$@\"\n",
        encoding="utf-8",
    )
    repository_python.chmod(0o755)
    wrong_bin = tmp_path / "wrong PATH bin"
    wrong_bin.mkdir()
    wrong_log = tmp_path / "wrong accelerate.log"
    wrong_accelerate = wrong_bin / "accelerate"
    wrong_accelerate.write_text(
        "#!/usr/bin/env bash\ntouch \"$WRONG_ACCELERATE_LOG\"\nexit 98\n",
        encoding="utf-8",
    )
    wrong_accelerate.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{wrong_bin}{os.pathsep}{env['PATH']}"
    env["FAKE_ACCELERATE_LOG"] = str(log_path)
    env["FAKE_PYTHON_EXEC_LOG"] = str(tmp_path / "python exec.log")
    env["WRONG_ACCELERATE_LOG"] = str(wrong_log)
    return env, log_path, wrong_log, copied_launcher


def live_import_worker_environment(tmp_path):
    """Build a harmless live launcher boundary around the repository Python."""

    copied_repo = tmp_path / "copied repository"
    copied_repo.mkdir()
    copied_launcher = copied_repo / LAUNCHER.name
    copied_launcher.write_bytes(LAUNCHER.read_bytes())
    copied_launcher.chmod(0o755)
    config_dir = copied_repo / "configs"
    config_dir.mkdir()
    (config_dir / "deepspeed_zero3.json").write_bytes(
        (REPO_ROOT / "configs/deepspeed_zero3.json").read_bytes()
    )
    (copied_repo / "transformers").symlink_to(
        REPO_ROOT / "transformers", target_is_directory=True
    )

    source_dir = copied_repo / "src"
    source_dir.mkdir()
    for name in ("artifact_paths.py", "deepspeed_runtime.py"):
        (source_dir / name).write_bytes((REPO_ROOT / "src" / name).read_bytes())
    (source_dir / "main.py").write_text(
        """import json
import os
from pathlib import Path

import qwen_vl_utils
import transformers
import transformers.models.qwen2_5_vl.stage2_latent_rollout as rollout

Path(os.environ["WORKER_IMPORT_LOG"]).write_text(json.dumps({
    "pythonpath": os.environ.get("PYTHONPATH"),
    "transformers": transformers.__file__,
    "rollout": rollout.__file__,
    "qwen_vl_utils": qwen_vl_utils.__file__,
}))
""",
        encoding="utf-8",
    )

    bin_dir = copied_repo / "venv/ilvr-venv/bin"
    bin_dir.mkdir(parents=True)
    repository_python = bin_dir / "python"
    repository_python.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(str(REPO_ROOT / 'venv/ilvr-venv/bin/python'))} \"$@\"\n",
        encoding="utf-8",
    )
    repository_python.chmod(0o755)
    repository_accelerate = bin_dir / "accelerate"
    repository_accelerate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "\"$PYTHON_EXEC\" src/main.py\n",
        encoding="utf-8",
    )
    repository_accelerate.chmod(0o755)

    hostile = tmp_path / "hostile pythonpath"
    hostile_transformers = hostile / "transformers"
    hostile_transformers.mkdir(parents=True)
    (hostile_transformers / "__init__.py").write_text(
        "raise RuntimeError('hostile inherited transformers imported')\n",
        encoding="utf-8",
    )
    worker_log = tmp_path / "worker imports.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(hostile)
    environment["WORKER_IMPORT_LOG"] = str(worker_log)
    return environment, worker_log, copied_launcher, copied_repo


def test_dora_dry_run_emits_exact_safe_two_process_command_without_mutation(tmp_path):
    """Catches wrong experiment defaults, shell evaluation, and dry-run writes."""
    output = tmp_path / "output with spaces;$(touch SHOULD_NOT_EXIST)"

    completed = run_launcher(LAUNCHER, *base_args(output))

    assert completed.returncode == 0, completed.stderr
    assert not output.exists()
    assert not (REPO_ROOT / "SHOULD_NOT_EXIST").exists()
    command = shlex.split(completed.stdout.strip())
    assert command[:7] == [
        str(REPO_ROOT / "venv/ilvr-venv/bin/accelerate"),
        "launch",
        "--num_processes",
        "2",
        "--main_process_port",
        "29501",
        "src/main.py",
    ]
    expected_pairs = {
        "--stage": "stage1",
        "--tuning_method": "dora",
        "--model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "--data_path": "/missing/CoMT train;literal/TRAIN.jsonl",
        "--image_root": "/missing/CoMT images & literal",
        "--save_model_path": str(output),
        "--per_device_train_batch_size": "8",
        "--epochs": "3",
        "--learning_rate": "1e-5",
        "--gradient_accumulation_steps": "2",
        "--seed": "42",
        "--attention_implementation": "flash_attention_2",
        "--save_strategy": "no",
        "--lora_r": "32",
        "--lora_alpha": "64",
        "--lora_dropout": "0.05",
        "--lora_target_modules": (
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        ),
    }
    for flag, value in expected_pairs.items():
        index = command.index(flag)
        assert command[index + 1] == value
    deepspeed = Path(command[command.index("--deepspeed_config") + 1])
    assert deepspeed == REPO_ROOT / "configs/deepspeed_zero3.json"
    assert "--final_save" in command
    assert "--gradient_checkpointing" in command


def test_stage1_launcher_routes_canonical_lora(tmp_path):
    """Catches canonical LoRA being translated back into the legacy CLI flag."""
    completed = run_launcher(
        LAUNCHER,
        *base_args(tmp_path / "lora-output", method="lora"),
        "--deepspeed-config",
        REPO_ROOT / "configs/deepspeed_zero2.json",
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout.strip())
    assert command[command.index("--tuning_method") + 1] == "lora"
    assert "--use_lora" not in command
    assert command[command.index("--deepspeed_config") + 1].endswith(
        "configs/deepspeed_zero2.json"
    )


def test_dry_run_derives_canonical_log_from_checkpoint_experiment_id():
    """Catches omitted logs falling back below a checkpoint or the worktree."""
    paths = resolve_run_artifact_paths("stage1-zero2-dora-mb4-ga2-candidate")

    completed = run_launcher(LAUNCHER, *base_args(paths.checkpoint_dir))

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout)
    assert command[command.index("--log_file") + 1] == str(
        paths.log_dir / "train.log"
    )


def test_live_launcher_rejects_mismatched_canonical_artifact_ids_before_setup(tmp_path):
    """Catches preflight accepting canonical paths that belong to different runs."""
    checkpoint = resolve_run_artifact_paths("stage1-zero2-dora-mb4-ga2-candidate")
    other = resolve_run_artifact_paths("stage1-zero3-dora-mb4-ga2-candidate")
    completed = run_launcher(
        LAUNCHER,
        *live_args(tmp_path, checkpoint.checkpoint_dir),
        "--log-file",
        other.log_dir / "train.log",
        "--deepspeed-config",
        tmp_path / "not-reached.json",
    )

    assert completed.returncode == 2
    assert "share one experiment ID" in completed.stderr


def test_live_launcher_rejects_bare_preclaimed_artifact_bypass(tmp_path):
    """Catches an environment boolean reusing unclaimed artifact directories."""
    paths = resolve_run_artifact_paths("stage1-zero2-dora-mb4-ga2-preclaim")
    env = os.environ.copy()
    env["STAGE1_ARTIFACTS_PRECLAIMED"] = "true"
    completed = run_launcher(
        LAUNCHER,
        *live_args(tmp_path, paths.checkpoint_dir),
        "--log-file",
        paths.log_dir / "train.log",
        "--deepspeed-config",
        REPO_ROOT / "configs/deepspeed_zero3.json",
        env=env,
    )

    assert completed.returncode == 2
    assert "claim token" in completed.stderr


def test_full_dry_run_omits_every_peft_argument(tmp_path):
    """Catches full tuning accidentally inheriting adapter configuration."""
    completed = run_launcher(
        LAUNCHER, *base_args(tmp_path / "full-output", method="full")
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout.strip())
    for flag in (
        "--use_lora",
        "--lora_r",
        "--lora_alpha",
        "--lora_dropout",
        "--lora_target_modules",
    ):
        assert flag not in command


def test_probe_flags_forward_exact_max_steps_without_final_save(tmp_path):
    """Catches a capacity probe accidentally training fully or serializing weights."""
    completed = run_launcher(
        LAUNCHER,
        *base_args(tmp_path / "probe output"),
        "--max-steps",
        "3",
        "--no-final-save",
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout)
    assert command[command.index("--max_steps") + 1] == "3"
    assert "--no_final_save" in command
    assert "--final_save" not in command


def test_timed_manifest_is_loaded_without_shell_evaluating_paths(tmp_path):
    """Catches the Task 17 manifest interface losing quoting or config binding."""
    output = tmp_path / "timed output;$(touch MANIFEST_INJECTION)"
    manifest = {
        "schema_version": 1,
        "kind": "stage1-timed-launch-manifest",
        "method": "dora",
        "tuning_config": {
            "use_dora": True, "rank": 32, "alpha": 64, "dropout": 0.05,
            "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        },
        "model": "model path;literal",
        "source_data_path": "/missing/data path;literal/TRAIN.jsonl",
        "data_path": "/missing/data path;literal/TRAIN.jsonl",
        "data_stat": {"device": 1, "inode": 2, "size": 3, "mtime_ns": 4},
        "image_root": "/missing/image root & literal",
        "output_dir": str(output),
        "micro_batch": 16,
        "world_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_global_batch": 64,
        "epochs": 3,
        "learning_rate": 1e-5,
        "seed": 42,
        "max_steps": 0,
        "attention_implementation": "flash_attention_2",
        "deepspeed_config": str(REPO_ROOT / "configs/deepspeed_zero3.json"),
        "deepspeed_config_digest": "d" * 64,
        "save_strategy": "no",
        "final_save": True,
        "audit_digest": "a" * 64,
        "probe_rows": "/missing/audit.json",
        "probe_row_indices": [7, 3],
        "maximum_config_digest": "b" * 64,
        "prerequisite_chain": "/smoke/prerequisite-chain.json",
        "prerequisite_chain_digest": "f" * 64,
    }
    manifest["config_digest"] = config_digest(manifest)
    manifest_path = tmp_path / "launch manifest;literal.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    original_manifest_bytes = manifest_path.read_bytes()

    poison_bin = tmp_path / "wrong path bin"
    poison_bin.mkdir()
    poison_marker = tmp_path / "wrong interpreter used"
    for name in ("python", "accelerate"):
        executable = poison_bin / name
        executable.write_text(
            "#!/usr/bin/env bash\n"
            "touch \"$POISON_MARKER\"\n"
            "exit 97\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{poison_bin}{os.pathsep}{env['PATH']}"
    env["STAGE1_BENCHMARK_PYTHON"] = str(poison_bin / "python")
    env["POISON_MARKER"] = str(poison_marker)

    completed = run_launcher(
        LAUNCHER, "--manifest", manifest_path, "--dry-run", env=env
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout)
    assert command[0] == str(REPO_ROOT / "venv/ilvr-venv/bin/accelerate")
    assert command[command.index("--model") + 1] == "model path;literal"
    assert command[command.index("--save_model_path") + 1] == str(output)
    assert command[command.index("--per_device_train_batch_size") + 1] == "16"
    assert "--final_save" in command
    assert manifest_path.read_bytes() == original_manifest_bytes
    assert not poison_marker.exists()
    assert not (REPO_ROOT / "MANIFEST_INJECTION").exists()


@pytest.mark.parametrize("value", ["0", "-1", "3.5", "nan", "word"])
def test_launcher_rejects_invalid_explicit_max_steps(tmp_path, value):
    """Catches malformed probe limits reaching Trainer."""
    completed = run_launcher(
        LAUNCHER,
        *base_args(tmp_path / "probe output"),
        "--max-steps",
        value,
    )

    assert completed.returncode == 2
    assert "max_steps" in completed.stderr


def test_launcher_requires_all_explicit_sources():
    """Catches placeholder model/data/image/output values reaching training."""
    completed = run_launcher(LAUNCHER, "--dry-run", "--method", "dora")

    assert completed.returncode != 0
    assert "--model is required" in completed.stderr


def test_live_launcher_refuses_nonempty_output_without_explicit_resume(tmp_path):
    """Catches accidental overwrite or implicit launcher resume."""
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "evidence.txt").write_text("owned", encoding="utf-8")
    env, accelerate_log, wrong_log, launcher = fake_accelerate_environment(tmp_path)

    refused = run_launcher(
        launcher, *live_args(tmp_path, output), env=env
    )

    assert refused.returncode == 2
    assert "canonical checkpoint root" in refused.stderr
    assert not accelerate_log.exists()
    assert not wrong_log.exists()


def test_live_launcher_rejects_noncanonical_resume_output(tmp_path):
    """Catches resume bypassing canonical checkpoint ownership."""
    output = tmp_path / "occupied"
    checkpoint = output / "checkpoint-7"
    checkpoint.mkdir(parents=True)
    env, accelerate_log, wrong_log, launcher = fake_accelerate_environment(tmp_path)
    env["PYTHON_EXEC"] = "/wrong/hjh-venv/bin/python"

    resumed = run_launcher(
        launcher,
        *live_args(tmp_path, output),
        "--resume",
        checkpoint,
        env=env,
    )

    assert resumed.returncode == 2
    assert "canonical checkpoint root" in resumed.stderr
    assert not accelerate_log.exists()
    assert not wrong_log.exists()


def test_live_worker_rejects_noncanonical_output_before_dispatch(tmp_path):
    """Catches an inherited worker environment bypassing artifact preflight."""

    environment, worker_log, launcher, copied_repo = live_import_worker_environment(
        tmp_path
    )
    completed = run_launcher(
        launcher,
        *live_args(tmp_path, tmp_path / "worker output", method="full"),
        env=environment,
    )

    assert completed.returncode == 2
    assert "canonical checkpoint root" in completed.stderr
    assert not worker_log.exists()


@pytest.mark.parametrize("target_location", ["inside", "outside"])
@pytest.mark.parametrize("nonempty", [False, True])
def test_live_launcher_rejects_symlink_output_before_dispatch(
    tmp_path, target_location, nonempty
):
    """Catches output symlinks escaping the explicit experiment directory."""
    experiment_root = tmp_path / "experiment root"
    experiment_root.mkdir()
    target = (
        experiment_root / "inside target"
        if target_location == "inside"
        else tmp_path / "outside target"
    )
    target.mkdir()
    if nonempty:
        (target / "owned.txt").write_text("owned", encoding="utf-8")
    output = experiment_root / "linked output"
    output.symlink_to(target, target_is_directory=True)
    env, accelerate_log, wrong_log, launcher = fake_accelerate_environment(tmp_path)

    completed = run_launcher(
        launcher, *live_args(tmp_path, output), env=env
    )

    assert completed.returncode == 2
    assert "canonical checkpoint root" in completed.stderr
    assert not accelerate_log.exists()
    assert not wrong_log.exists()


@pytest.mark.parametrize("output_kind", ["nonempty", "symlink", "dangling"])
def test_dry_run_skips_all_filesystem_checks_and_creates_nothing(
    tmp_path, output_kind
):
    """Catches dry-run consulting or mutating model/data/output paths."""
    if output_kind == "nonempty":
        output = tmp_path / "occupied"
        output.mkdir()
        (output / "owned.txt").write_text("owned", encoding="utf-8")
    else:
        target = (
            tmp_path / "target"
            if output_kind == "symlink"
            else tmp_path / "missing-target"
        )
        if output_kind == "symlink":
            target.mkdir()
        output = tmp_path / "linked output"
        output.symlink_to(target, target_is_directory=True)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    completed = run_launcher(LAUNCHER, *base_args(output))

    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert completed.returncode == 0, completed.stderr
    assert after == before
    assert str(output) in shlex.split(completed.stdout)


@pytest.mark.parametrize("flag", ["--epochs", "--learning-rate"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "garbage"])
def test_launcher_rejects_invalid_positive_finite_scalars(tmp_path, flag, value):
    """Catches malformed/non-finite scalar values reaching Accelerate."""
    completed = run_launcher(
        LAUNCHER,
        *base_args(tmp_path / "output"),
        flag,
        value,
    )

    assert completed.returncode == 2
    assert "positive finite" in completed.stderr


@pytest.mark.parametrize(
    "environment_name", ["STAGE1_EPOCHS", "STAGE1_LEARNING_RATE"]
)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "garbage"])
def test_launcher_rejects_invalid_scalar_environment_overrides(
    tmp_path, environment_name, value
):
    """Catches environment overrides bypassing launcher scalar validation."""
    env = os.environ.copy()
    env[environment_name] = value

    completed = run_launcher(
        LAUNCHER, *base_args(tmp_path / "output"), env=env
    )

    assert completed.returncode == 2
    assert "positive finite" in completed.stderr


@pytest.mark.parametrize(
    ("epochs", "learning_rate"),
    [(".5", ".00001"), ("3e0", "1e-5"), ("2.5E+0", "2E-5")],
)
def test_launcher_preserves_valid_decimal_and_scientific_scalars(
    tmp_path, epochs, learning_rate
):
    """Catches validation normalizing or rejecting valid numeric spellings."""
    completed = run_launcher(
        LAUNCHER,
        *base_args(tmp_path / "output"),
        "--epochs",
        epochs,
        "--learning-rate",
        learning_rate,
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout)
    assert command[command.index("--epochs") + 1] == epochs
    assert command[command.index("--learning_rate") + 1] == learning_rate


def test_launcher_preserves_valid_scalar_environment_overrides(tmp_path):
    """Catches environment validation rewriting accepted numeric spellings."""
    env = os.environ.copy()
    env["STAGE1_EPOCHS"] = ".5"
    env["STAGE1_LEARNING_RATE"] = "2E-5"

    completed = run_launcher(
        LAUNCHER, *base_args(tmp_path / "output"), env=env
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout)
    assert command[command.index("--epochs") + 1] == ".5"
    assert command[command.index("--learning_rate") + 1] == "2E-5"


def test_legacy_launcher_normalizes_use_lora_to_canonical_lora(tmp_path):
    """Catches the legacy selector being dropped or translated to the wrong method."""
    completed = run_launcher(
        LEGACY_LAUNCHER,
        "--dry-run",
        "--use_lora",
        "--model",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "--data-path",
        "/missing/train.jsonl",
        "--image-root",
        "/missing/images",
        "--output-dir",
        tmp_path / "legacy",
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout.strip())
    assert command[command.index("--tuning_method") + 1] == "lora"
    assert "--use_lora" not in command


def test_smoke_launcher_forwards_exact_numbered_save_and_resume_witness(tmp_path):
    """Catches a 3->4 smoke silently losing its checkpoint or sealed witness."""
    marker = tmp_path / "checkpoint-topology.json"
    checkpoint = tmp_path / "run" / "checkpoint-3"
    completed = run_launcher(
        LAUNCHER,
        *base_args(tmp_path / "run", method="full"),
        "--save-strategy",
        "steps",
        "--save-steps",
        "3",
        "--max-steps",
        "3",
        "--no-final-save",
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split(completed.stdout)
    assert command[command.index("--save_strategy") + 1] == "steps"
    assert command[command.index("--save_steps") + 1] == "3"

    resumed = run_launcher(
        LAUNCHER,
        *base_args(tmp_path / "run", method="full"),
        "--resume",
        checkpoint,
        "--max-steps",
        "4",
        "--source-data-path",
        "/missing/CoMT train;literal/TRAIN.jsonl",
        "--benchmark-run-config",
        tmp_path / "run-config.json",
        "--producer-evidence",
        tmp_path / "evidence.json",
        "--smoke-resume-witness",
        marker,
        "--no-final-save",
    )

    assert resumed.returncode == 0, resumed.stderr
    resumed_command = shlex.split(resumed.stdout)
    assert resumed_command[resumed_command.index("--save_steps") + 1] == "612"
    assert resumed_command[
        resumed_command.index("--benchmark_smoke_resume_witness") + 1
    ] == str(marker)


@pytest.mark.parametrize("value", ["0", "-1", "3.5", "true"])
def test_launcher_rejects_invalid_save_steps(tmp_path, value):
    """Catches malformed checkpoint cadence reaching Trainer."""
    completed = run_launcher(
        LAUNCHER,
        *base_args(tmp_path / "output"),
        "--save-steps",
        value,
    )

    assert completed.returncode == 2
    assert "save_steps" in completed.stderr
