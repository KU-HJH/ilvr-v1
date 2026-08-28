import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

_platform_python_version = platform.python_version
if _platform_python_version().endswith("+"):
    platform.python_version = lambda: _platform_python_version().rstrip("+")

from main import (
    build_stage2_training_arguments,
    validate_stage2_distributed_settings,
)

platform.python_version = _platform_python_version
from stage2.data import Stage2DataCollator
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentTokenIds,
    parse_latent_spans,
)
from src.artifact_paths import resolve_run_artifact_paths
from stage2_rank_skew_probe import (
    _observe_decoder_forwards,
    build_rank_skew_parser,
    select_probe_examples,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_PATH = REPO_ROOT / "tests/stage2/integration/tiny_stage2_train.py"
TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)


def four_gpu_env(rank):
    return {
        "ILVR_NUM_MACHINES": "2",
        "ILVR_NUM_PROCESSES": "4",
        "ILVR_MACHINE_RANK": str(rank),
        "ILVR_MAIN_PROCESS_IP": "10.1.102.41",
        "ILVR_MAIN_PROCESS_PORT": "29623",
        "ILVR_SOCKET_IFNAME": "bond1.102",
    }


def run_stage1_dry(updates):
    environment = os.environ.copy()
    environment.update(updates)
    return subprocess.run(
        [
            str(REPO_ROOT / "run_stage1_training.sh"),
            "--dry-run",
            "--method",
            "full",
            "--model",
            "model",
            "--data-path",
            "/missing/train.jsonl",
            "--image-root",
            "/missing/images",
            "--output-dir",
            "/missing/output",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def run_stage2_dry(updates):
    environment = os.environ.copy()
    environment.update(
        {
            "STAGE1_CHECKPOINT": "/missing/stage1",
            "DATA_PATH": "/missing/train.jsonl",
            "IMAGE_ROOT": "/missing/images",
            "OUTPUT_DIR": "/missing/output",
            "STAGE2_DRY_RUN": "true",
        }
    )
    environment.update(updates)
    return subprocess.run(
        [str(REPO_ROOT / "run_stage2_training.sh")],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"ILVR_NUM_MACHINES": "2"}, "all distributed variables"),
        (four_gpu_env(0) | {"ILVR_NUM_PROCESSES": "3"}, "exactly 4"),
        (four_gpu_env(2), "machine rank"),
        (four_gpu_env(0) | {"ILVR_SOCKET_IFNAME": ""}, "interface"),
        (four_gpu_env(0) | {"ILVR_MAIN_PROCESS_PORT": "not-a-port"}, "1024"),
        (four_gpu_env(0) | {"ILVR_MAIN_PROCESS_PORT": "1023"}, "1024"),
        (four_gpu_env(0) | {"ILVR_MAIN_PROCESS_PORT": "65536"}, "65535"),
    ],
)
@pytest.mark.parametrize("run", (run_stage1_dry, run_stage2_dry))
def test_launcher_rejects_invalid_multinode_contract(run, updates, message):
    """Catches malformed topology reaching artifact setup or Accelerate."""
    completed = run(updates)

    assert completed.returncode == 2
    assert message in completed.stderr


@pytest.mark.parametrize(
    ("run", "updates"),
    [
        (run_stage1_dry, four_gpu_env(0) | {"STAGE1_NUM_PROCESSES": "2"}),
        (run_stage2_dry, four_gpu_env(0) | {"STAGE2_NUM_PROCESSES": "2"}),
    ],
)
def test_launcher_rejects_stage_process_count_that_contradicts_topology(run, updates):
    """Catches legacy process controls overriding the fixed four-GPU world size."""
    completed = run(updates)

    assert completed.returncode == 2
    assert "must match" in completed.stderr


def _load_harness_module():
    spec = importlib.util.spec_from_file_location("tiny_stage2_train", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tiny_stage2_harness_owns_checkpoint_and_witness_by_one_id():
    """Catches the save/resume harness accepting split generated destinations."""
    parser = _load_harness_module().build_tiny_stage2_parser()
    args = parser.parse_args(
        ["--experiment-id", "stage2-zero2-dora-tiny-lifecycle"]
    )
    assert args.experiment_id == "stage2-zero2-dora-tiny-lifecycle"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--experiment-id",
                "stage2-zero2-dora-tiny-lifecycle",
                "--witness-path",
                "/tmp/unowned.json",
            ]
        )


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
def test_rank_skew_harness_supports_each_canonical_zero_stage(zero_stage):
    """Catches a canonical ZeRO profile becoming unlaunchable for the probe."""
    args = build_rank_skew_parser().parse_args(
        [
            "--zero-stage",
            str(zero_stage),
            "--deepspeed",
            f"configs/deepspeed_zero{zero_stage}.json",
            "--model",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "--stage1-checkpoint",
            "/tmp/stage1",
            "--data-path",
            "/tmp/train.jsonl",
            "--image-root",
            "/tmp/images",
            "--experiment-id",
            f"stage2-zero{zero_stage}-dora-mb1-ga8-rank-skew",
        ]
    )

    assert args.zero_stage == zero_stage
    assert args.method == "dora"
    assert args.max_steps == 1
    assert args.micro_batch == 1
    assert args.gradient_accumulation_steps == 8


def test_rank_skew_observes_extra_decoder_invocation_directly():
    """Catches decoder-forward evidence reverting to a block-count formula."""

    class FakeStage2Qwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(2, 2)
            self.lm_head = torch.nn.Linear(2, 2)
            self.config = SimpleNamespace(stage="stage2", latent_size=2)
            self.stage2_rollout_engine = object()

        def prepare_stage2_base_inputs_embeds(self):
            raise AssertionError("not used by this boundary test")

        def get_rope_index(self):
            raise AssertionError("not used by this boundary test")

    model = FakeStage2Qwen()
    inputs = torch.ones(1, 2)

    _, ordinary_count = _observe_decoder_forwards(
        model, lambda: (model.model(inputs), model.model(inputs))
    )
    _, extra_count = _observe_decoder_forwards(
        model,
        lambda: (model.model(inputs), model.model(inputs), model.model(inputs)),
    )

    assert ordinary_count == 2
    assert extra_count == 3


def test_rank_skew_harness_selects_eight_examples_from_two_buckets():
    """Catches the probe selecting insufficient or unbalanced latent work."""
    examples = [
        SimpleNamespace(latent_block_count=count, marker=marker)
        for marker, count in enumerate((1, 2, 1, 3))
    ]

    selected = select_probe_examples(examples)

    assert len(selected) == 16
    assert [row.latent_block_count for row in selected] == [1] * 8 + [2] * 8


def test_rank_skew_harness_has_no_arbitrary_generated_path_arguments():
    """Catches the probe accepting an unowned witness destination."""
    parser = build_rank_skew_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--deepspeed",
                "configs/deepspeed_zero3.json",
                "--zero-stage",
                "3",
                "--model",
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "--stage1-checkpoint",
                "/tmp/stage1",
                "--data-path",
                "/tmp/train.jsonl",
                "--image-root",
                "/tmp/images",
                "--experiment-id",
                "stage2-zero3-dora-mb1-ga8-rank-skew",
                "--witness-path",
                "/tmp/unowned.json",
            ]
        )


def _tiny_rank_record(**overrides):
    record = {
        "first_global_step": 1,
        "resumed_global_step": 2,
        "first_loss": 0.25,
        "resumed_loss": 0.20,
        "first_batches": [1],
        "resumed_batches": [1],
        "optimizer_step_at_resume": 1,
        "scheduler_step_at_resume": 1,
        "adapter_parameters_changed": True,
        "visual_trainable_parameters": 0,
        "base_trainable_parameters": 0,
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    ("second_override", "message"),
    [
        ({"resumed_global_step": 3}, "global steps"),
        ({"first_loss": float("nan")}, "finite losses"),
    ],
)
def test_tiny_stage2_rejects_rank_lifecycle_disagreement(second_override, message):
    """Catches one rank's bad step or loss being hidden by rank-zero scalars."""
    harness = _load_harness_module()
    records = [_tiny_rank_record(), _tiny_rank_record(**second_override)]

    with pytest.raises(AssertionError, match=message):
        harness._validate_rank_lifecycle_records(records)


def test_tiny_stage2_writer_failure_reduces_before_barrier(tmp_path):
    """Catches rank-zero publication failure hanging peers at a barrier."""
    harness = _load_harness_module()
    events = []

    def failing_writer(path, witness):
        assert path == tmp_path / "witness.json"
        assert witness == {"schema_version": 1}
        raise OSError("disk full")

    def failure_counter(local_failure):
        events.append(("reduce", local_failure))
        return local_failure

    with pytest.raises(OSError, match="disk full"):
        harness._publish_rank_agreed_witness(
            tmp_path / "witness.json",
            {"schema_version": 1},
            is_writer=True,
            writer=failing_writer,
            failure_counter=failure_counter,
            synchronize=lambda: events.append(("barrier", None)),
        )

    assert events == [("reduce", 1)]


def test_tiny_stage2_nonwriter_propagates_remote_writer_failure(tmp_path):
    """Catches a non-writer rank reporting success after rank-zero failure."""
    harness = _load_harness_module()
    events = []

    with pytest.raises(RuntimeError, match="rank zero"):
        harness._publish_rank_agreed_witness(
            tmp_path / "witness.json",
            {"schema_version": 1},
            is_writer=False,
            writer=lambda *_: pytest.fail("non-writer attempted publication"),
            failure_counter=lambda local: events.append(("reduce", local)) or 1,
            synchronize=lambda: events.append(("barrier", None)),
        )

    assert events == [("reduce", 0)]


def test_tiny_stage2_rank_agreed_publication_is_atomic(tmp_path):
    """Catches successful publication leaving a partial temporary witness."""
    harness = _load_harness_module()
    witness_path = tmp_path / "result" / "witness.json"
    events = []

    harness._publish_rank_agreed_witness(
        witness_path,
        {"schema_version": 1},
        is_writer=True,
        failure_counter=lambda local: events.append(("reduce", local)) or local,
        synchronize=lambda: events.append(("barrier", None)),
    )

    assert json.loads(witness_path.read_text()) == {"schema_version": 1}
    assert list(witness_path.parent.iterdir()) == [witness_path]
    assert events == [("reduce", 0), ("barrier", None)]


def _stage2_args(**overrides):
    values = {
        "save_model_path": "out",
        "epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "warm_up_steps": 0,
        "learning_rate": 1e-4,
        "save_steps": 1,
        "max_seq_length_train": 32,
        "max_steps": 2,
        "gradient_checkpointing": False,
        "deepspeed_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("local_batch_size", [0, 2])
def test_stage2_rejects_non_unit_rank_local_batch(local_batch_size):
    """Catches a rank silently receiving zero or multiple Stage 2 samples."""
    with pytest.raises(
        ValueError,
        match="train_micro_batch_size_per_gpu must resolve to 1",
    ):
        validate_stage2_distributed_settings(local_batch_size)


def test_stage2_accepts_unit_rank_local_batch_with_accumulation():
    """Catches validation confusing effective global batch with rank-local batch."""
    validate_stage2_distributed_settings(1)

    args = build_stage2_training_arguments(
        _stage2_args(gradient_accumulation_steps=4)
    )

    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 4
    assert args.deepspeed is None


def test_stage2_training_arguments_enable_deepspeed_only_when_selected():
    """Catches Stage 2 forcing ZeRO-3 or selecting a non-reloadable ZeRO config."""
    args = build_stage2_training_arguments(
        _stage2_args(deepspeed_config=str(REPO_ROOT / "configs/deepspeed_zero3.json"))
    )

    assert args.deepspeed == str(REPO_ROOT / "configs/deepspeed_zero3.json")
    assert (
        args.hf_deepspeed_config.get_value(
            "zero_optimization.stage3_gather_16bit_weights_on_model_save"
        )
        is True
    )


def _dry_run_launcher(*, deepspeed_config=None):
    paths = resolve_run_artifact_paths("stage2-zero2-full-mb1-ga8-candidate")
    env = os.environ.copy()
    env.update(
        {
            "STAGE1_CHECKPOINT": "/tmp/stage1",
            "DATA_PATH": "/tmp/train.jsonl",
            "IMAGE_ROOT": "/tmp/images",
            "OUTPUT_DIR": str(paths.checkpoint_dir),
            "STAGE2_DRY_RUN": "true",
        }
    )
    if deepspeed_config is not None:
        env["DEEPSPEED_CONFIG"] = deepspeed_config
    else:
        env.pop("DEEPSPEED_CONFIG", None)
    return subprocess.run(
        [str(REPO_ROOT / "run_stage2_training.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_stage2_launcher_passes_optional_deepspeed_selection_by_behavior():
    """Catches the launcher forcing or ignoring the optional DeepSpeed config."""
    without_deepspeed = _dry_run_launcher()
    assert without_deepspeed.returncode == 0, without_deepspeed.stderr
    assert "--deepspeed_config" not in without_deepspeed.stdout

    config = str(REPO_ROOT / "configs/deepspeed_zero2.json")
    with_deepspeed = _dry_run_launcher(deepspeed_config=config)
    assert with_deepspeed.returncode == 0, with_deepspeed.stderr
    assert f"--deepspeed_config {config}" in with_deepspeed.stdout
    assert "accelerate launch --num_processes 2" in with_deepspeed.stdout
    paths = resolve_run_artifact_paths("stage2-zero2-full-mb1-ga8-candidate")
    assert f"--log_file {paths.log_dir / 'train.log'}" in with_deepspeed.stdout


def test_recording_collator_delegates_production_stage2_batch_contract(tmp_path):
    """Catches the distributed harness replacing production Stage 2 collation."""
    harness = _load_harness_module()
    prepared = harness.build_tiny_prepared_example(tmp_path)
    processor = harness.TinyStage2Processor()
    production_collator = Stage2DataCollator(
        processor=processor,
        token_ids=TOKENS,
        latent_size=2,
        supervise_latent_boundaries=True,
    )
    recorder = harness.RecordingStage2Collator(production_collator)

    batch = recorder([prepared])

    assert recorder.delegate is production_collator
    assert recorder.local_batch_sizes == [1]
    assert processor.calls
    assert set(batch) == {
        "input_ids",
        "attention_mask",
        "labels",
        "pixel_values",
        "image_grid_thw",
    }
    assert batch["input_ids"].shape == batch["attention_mask"].shape == (1, 11)
    assert batch["labels"].shape == (1, 11)
    assert batch["pixel_values"].shape == (1, 12)
    assert batch["image_grid_thw"].shape == (1, 3)
    assert len(
        parse_latent_spans(batch["input_ids"][0], token_ids=TOKENS, latent_size=2)
    ) == 1
    assert batch["labels"][0, 6:8].tolist() == [-100, -100]
    assert batch["labels"][0, [5, 8]].tolist() == [91, 92]
    images = processor.calls[-1]["images"]
    assert len(images) == 1 and len(images[0]) == 1
    assert isinstance(images[0][0], Image.Image)
    assert "pixel_values_latent" not in batch
    assert "image_grid_thw_latent" not in batch


def test_recording_collator_uses_production_b_greater_than_one_failure(tmp_path):
    """Catches the harness recording B=1 without delegating B>1 validation."""
    harness = _load_harness_module()
    prepared = harness.build_tiny_prepared_example(tmp_path)
    recorder = harness.RecordingStage2Collator(
        Stage2DataCollator(
            processor=harness.TinyStage2Processor(),
            token_ids=TOKENS,
            latent_size=2,
            supervise_latent_boundaries=True,
        )
    )

    with pytest.raises(NotImplementedError, match="batch_size=1"):
        recorder([prepared, prepared])

    assert recorder.local_batch_sizes == [2]


def test_rank_local_fixtures_have_distinct_valid_production_batches(tmp_path):
    """Catches launched ranks concurrently truncating the same original image."""
    harness = _load_harness_module()
    examples = [
        harness.build_rank_local_tiny_prepared_example(
            tmp_path, environ={"RANK": str(rank)}
        )
        for rank in (0, 1)
    ]

    image_paths = [Path(example.original_image_paths[0]) for example in examples]
    assert image_paths == [
        tmp_path / "tiny-data" / "rank-0" / "input.png",
        tmp_path / "tiny-data" / "rank-1" / "input.png",
    ]
    assert image_paths[0] != image_paths[1]

    batches = []
    for image_path, example in zip(image_paths, examples, strict=True):
        with Image.open(image_path) as image:
            image.verify()
        batch = Stage2DataCollator(
            processor=harness.TinyStage2Processor(),
            token_ids=TOKENS,
            latent_size=2,
            supervise_latent_boundaries=True,
        )([example])
        assert batch["input_ids"].shape == batch["attention_mask"].shape == (1, 11)
        assert batch["labels"].shape == (1, 11)
        assert batch["pixel_values"].shape == (1, 12)
        assert batch["image_grid_thw"].shape == (1, 3)
        assert len(
            parse_latent_spans(
                batch["input_ids"][0], token_ids=TOKENS, latent_size=2
            )
        ) == 1
        batches.append(batch)

    for key in batches[0]:
        assert torch.equal(batches[0][key], batches[1][key])


@pytest.mark.parametrize("raw_rank", ["", "worker-1", "-1"])
def test_rank_local_fixture_rejects_malformed_launcher_rank(tmp_path, raw_rank):
    """Catches malformed launcher state silently collapsing ranks to one path."""
    harness = _load_harness_module()

    with pytest.raises(ValueError, match="RANK must be a non-negative integer"):
        harness.build_rank_local_tiny_prepared_example(
            tmp_path, environ={"RANK": raw_rank}
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_harness_rehomes_buffers_for_partially_zero_initialized_resume():
    """Catches ZeRO-3 skipping device placement for nonpersistent RoPE buffers."""
    harness = _load_harness_module()
    model = harness._tiny_model()
    next(model.parameters()).ds_id = 0
    qwen = model
    target = torch.device("cuda", 0)

    assert qwen.model.rotary_emb.inv_freq.device.type == "cpu"
    assert next(model.parameters()).device.type == "cpu"

    harness._move_registered_buffers_to_device(model, target)

    assert qwen.model.rotary_emb.inv_freq.device == target
    assert next(model.parameters()).device.type == "cpu"


def test_full_harness_reload_refreezes_visual_parameters(tmp_path, monkeypatch):
    """Catches full-checkpoint reload restoring visual parameters as trainable."""
    harness = _load_harness_module()
    tuning = harness._tuning_config("full")
    source = harness._build_tuned_model(tuning)
    source.save_pretrained(tmp_path)
    loader_result = harness._tiny_model()
    loader_result.visual.requires_grad_(True)

    def load_trainable_visual(checkpoint):
        assert Path(checkpoint) == tmp_path
        return loader_result

    monkeypatch.setattr(
        harness.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        load_trainable_visual,
    )

    resumed = harness._build_tuned_model(tuning, adapter_path=tmp_path)

    assert resumed is loader_result
    assert not any(parameter.requires_grad for parameter in resumed.visual.parameters())


def test_boundary_counts_zero_sized_zero3_parameter_names():
    """Catches ZeRO-empty tensors hiding trainable visual or base violations."""
    harness = _load_harness_module()
    model = harness._build_tuned_model(harness._tuning_config("lora"))
    qwen = harness.resolve_stage2_qwen_model(model)
    visual = next(qwen.visual.parameters())
    base = next(qwen.model.embed_tokens.parameters())
    for parameter in (visual, base):
        parameter.requires_grad_(True)
        parameter.data = torch.empty(0, dtype=parameter.dtype)
        parameter.ds_numel = 0

    visual_violations, base_violations = harness._parameter_boundary(model)

    assert visual_violations == 1
    assert base_violations == 1


@pytest.mark.skipif(
    os.environ.get("RUN_STAGE2_TINY_TRAIN") != "1",
    reason="set RUN_STAGE2_TINY_TRAIN=1 for the save/resume witness",
)
@pytest.mark.parametrize("method", ["full", "lora", "dora"])
def test_tiny_stage2_harness_restores_full_training_state(tmp_path, method):
    """Catches resume losing model artifacts, frozen vision, or Trainer state."""
    experiment_id = f"stage2-zero-disabled-{method}-tiny-lifecycle"
    paths = resolve_run_artifact_paths(experiment_id)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tests/stage2/integration/tiny_stage2_train.py"),
            "--experiment-id",
            experiment_id,
            "--tuning-method",
            method,
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    witness = json.loads((paths.result_dir / "witness.json").read_text())
    assert witness["tuning_method"] == method
    assert witness["first_global_step"] == 1
    assert witness["resumed_global_step"] == 2
    assert witness["optimizer_state_restored"] is True
    assert witness["scheduler_state_restored"] is True
    assert witness["optimizer_step_at_resume"] == 1
    assert witness["scheduler_step_at_resume"] == 1
    assert witness["processor_present"] is True
    assert witness["visual_trainable_parameters"] == 0
    if method == "full":
        assert witness["adapter_parameters_changed"] is False
        assert witness["adapter_checkpoint_present"] is False
        assert witness["sidecar_present"] is False
        assert witness["full_model_weights_absent"] is False
        assert witness["base_trainable_parameters"] > 0
    else:
        assert witness["adapter_parameters_changed"] is True
        assert witness["adapter_checkpoint_present"] is True
        assert witness["sidecar_present"] is True
        assert witness["full_model_weights_absent"] is True
        assert witness["base_trainable_parameters"] == 0
    assert witness["observed_local_batch_sizes"]
    assert set(witness["observed_local_batch_sizes"]) == {1}
    assert math.isfinite(witness["first_loss"])
    assert math.isfinite(witness["resumed_loss"])
