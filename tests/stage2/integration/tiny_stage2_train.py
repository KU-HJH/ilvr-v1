"""Tiny Stage 2 save/resume harness for single-process and ZeRO-3 checks."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import peft
import torch
import torch.distributed as dist
import transformers
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    TrainerCallback,
)

from artifact_paths import resolve_run_artifact_paths
from main import build_stage2_training_arguments
from deepspeed_runtime import load_deepspeed_runtime_profile
from stage2.checkpointing import build_stage2_peft_metadata
from stage2.data import Stage2DataCollator, prepare_stage2_example
from stage2.peft import (
    ALLOWED_STAGE2_TARGET_SUFFIXES,
    Stage2TuningConfig,
    build_stage2_tuned_model,
    resolve_stage2_qwen_model,
)
from stage2.trainer import Stage2Trainer
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


class TinyStage2Dataset(Dataset):
    def __init__(self, prepared_example, size: int = 8):
        self.sample = prepared_example
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        del index
        return self.sample


class TinyStage2Tokenizer:
    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert text == "<|im_start|>assistant"
        assert add_special_tokens is False
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[31, 32]])}


class TinyStage2Processor:
    """Deterministic processor boundary used by the production collator."""

    def __init__(self):
        self.tokenizer = TinyStage2Tokenizer()
        self.calls = []

    model_input_names = ["input_ids"]

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tiny_stage2_processor.json").write_text(
            json.dumps(
                {
                    "latent_pad_token_id": 90,
                    "latent_start_token_id": 91,
                    "latent_end_token_id": 92,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        assistant_text = conversation[1]["content"][0]["text"]
        assert assistant_text == (
            "A<|latent_start|><|latent_pad|><|latent_pad|><|latent_end|>B"
        )
        return "rendered tiny Stage 2 conversation"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["text"] == ["rendered tiny Stage 2 conversation"]
        assert kwargs["padding"] is True
        assert kwargs["truncation"] is False
        assert kwargs["return_tensors"] == "pt"
        images = kwargs["images"]
        assert len(images) == 1 and len(images[0]) == 1
        assert isinstance(images[0][0], Image.Image)
        input_ids = torch.tensor(
            [[81, 80, 1, 31, 32, 91, 90, 90, 92, 2, 3]]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.arange(12, dtype=torch.float32).reshape(1, 12)
            / 12.0,
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }


def build_tiny_prepared_example(image_root: Path):
    image_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(17, 31, 47)).save(image_root / "input.png")
    return prepare_stage2_example(
        {
            "image_input": "input.png",
            "text_input": "question",
            "sequence_plan": [
                {"type": "text", "content": "A"},
                {"type": "latent", "helper_image": "helper-not-loaded.png"},
                {"type": "text", "content": "B"},
            ],
        },
        image_root=image_root,
        latent_size=2,
    )


def resolve_launcher_rank(environ=None) -> int:
    environment = os.environ if environ is None else environ
    raw_rank = environment.get("RANK", "0")
    try:
        rank = int(raw_rank)
    except (TypeError, ValueError) as error:
        raise ValueError("RANK must be a non-negative integer") from error
    if rank < 0:
        raise ValueError("RANK must be a non-negative integer")
    return rank


def build_rank_local_tiny_prepared_example(output_dir: Path, *, environ=None):
    rank = resolve_launcher_rank(environ)
    return build_tiny_prepared_example(
        output_dir / "tiny-data" / f"rank-{rank}"
    )


class RecordingStage2Collator:
    """Record local size while delegating the production Stage 2 collator."""

    def __init__(self, delegate: Stage2DataCollator):
        self.delegate = delegate
        self.local_batch_sizes: list[int] = []

    def __call__(self, features):
        self.local_batch_sizes.append(len(features))
        return self.delegate(features)


def _recording_production_collator(
    processor: TinyStage2Processor | None = None,
) -> RecordingStage2Collator:
    return RecordingStage2Collator(
        Stage2DataCollator(
            processor=processor or TinyStage2Processor(),
            token_ids=LatentTokenIds(
                pad_token_id=90,
                start_token_id=91,
                end_token_id=92,
            ),
            latent_size=2,
            supervise_latent_boundaries=True,
        )
    )


def _scalar_step(value) -> int | None:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return int(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _find_optimizer_step(value) -> int | None:
    steps: list[int] = []

    def visit(node):
        if isinstance(node, dict):
            for key, child in node.items():
                if key == "step":
                    step = _scalar_step(child)
                    if step is not None:
                        steps.append(step)
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    return max(steps) if steps else None


class ResumeStateRecorder(TrainerCallback):
    def __init__(self):
        self.global_step: int | None = None
        self.optimizer_step: int | None = None
        self.scheduler_step: int | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        del args, control
        self.global_step = int(state.global_step)
        optimizer = kwargs["optimizer"]
        self.optimizer_step = _find_optimizer_step(optimizer.state_dict())
        scheduler = kwargs["lr_scheduler"]
        self.scheduler_step = int(scheduler.state_dict()["last_epoch"])


def _tiny_model() -> Qwen2_5_VLForConditionalGeneration:
    config = Qwen2_5_VLConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 2,
            "out_hidden_size": 8,
            "fullatt_block_indexes": [0],
        },
        image_token_id=80,
        vision_start_token_id=81,
        video_token_id=82,
        stage="stage2",
        latent_size=2,
        latent_pad_token_id=90,
        latent_start_token_id=91,
        latent_end_token_id=92,
        supervise_latent_boundaries=True,
        use_cache=False,
        attn_implementation="eager",
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 0]},
    )
    return Qwen2_5_VLForConditionalGeneration(config)


def _training_args(output_dir: Path, *, max_steps: int, deepspeed: str | None):
    values = SimpleNamespace(
        save_model_path=str(output_dir),
        epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warm_up_steps=0,
        learning_rate=1e-4,
        save_steps=1,
        max_seq_length_train=32,
        max_steps=max_steps,
        gradient_checkpointing=False,
        deepspeed_config=deepspeed,
    )
    return build_stage2_training_arguments(values)


def _collect_rank_records(record):
    if not dist.is_available() or not dist.is_initialized():
        return [record]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, record)
    return gathered


def _validate_rank_lifecycle_records(
    records, *, initial_steps: int = 1, resumed_steps: int = 2
) -> None:
    if not records:
        raise AssertionError("Stage 2 lifecycle gathered no rank records")
    first_step_set = {record["first_global_step"] for record in records}
    resumed_step_set = {record["resumed_global_step"] for record in records}
    if first_step_set != {initial_steps} or resumed_step_set != {resumed_steps}:
        raise AssertionError(
            "Stage 2 rank global steps disagree with the requested lifecycle: "
            f"{records}"
        )
    if {record["optimizer_step"] for record in records} != {resumed_steps}:
        raise AssertionError(f"Stage 2 optimizer steps disagree: {records}")
    losses = [
        float(record[field])
        for record in records
        for field in ("first_loss", "resumed_loss")
    ]
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise AssertionError(f"Stage 2 ranks did not report finite losses: {records}")
    gradients = [record["gradient_norm"] for record in records]
    if not all(type(value) in (int, float) and math.isfinite(value) for value in gradients):
        raise AssertionError(f"Stage 2 ranks did not report finite gradients: {records}")
    if [record["rank"] for record in records] != list(range(len(records))):
        raise AssertionError(f"Stage 2 process ranks are incomplete: {records}")
    if any(record["worker_error"] is not None for record in records):
        raise AssertionError(f"Stage 2 rank reported a worker error: {records}")


def _is_world_process_zero() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _is_machine_local_leader() -> bool:
    return (
        not dist.is_available()
        or not dist.is_initialized()
        or int(os.environ["LOCAL_RANK"]) == 0
    )


def _claim_owned_paths(experiment_id: str):
    paths = resolve_run_artifact_paths(experiment_id)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    collision = paths.checkpoint_dir.exists() or paths.result_dir.exists()
    if dist.is_initialized():
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        observed = torch.tensor(int(collision), dtype=torch.int32, device=device)
        dist.all_reduce(observed, op=dist.ReduceOp.MAX)
        collision = bool(observed.item())
    if collision:
        raise FileExistsError(f"refusing to reuse experiment ID: {experiment_id}")
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    return paths


def _phase_paths(experiment_id: str):
    validated = resolve_run_artifact_paths(experiment_id)
    return SimpleNamespace(
        experiment_id=validated.experiment_id,
        checkpoint_dir=Path("/root/workspace/ILVR/checkpoints") / experiment_id,
        result_dir=Path("/root/workspace/ILVR/results") / experiment_id,
    )


def _claim_initial_phase_paths(experiment_id: str):
    paths = _phase_paths(experiment_id)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    collision = paths.checkpoint_dir.exists() or paths.result_dir.exists()
    if dist.is_initialized():
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        observed = torch.tensor(int(collision), dtype=torch.int32, device=device)
        dist.all_reduce(observed, op=dist.ReduceOp.MAX)
        collision = bool(observed.item())
    if collision:
        raise FileExistsError(f"refusing to reuse experiment ID: {experiment_id}")
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    return paths


def _write_witness_atomically(path: Path, witness: dict[str, object]) -> None:
    from multinode_runtime import _canonical_json_bytes, _publish_bytes_no_replace

    _publish_bytes_no_replace(path, _canonical_json_bytes(witness) + b"\n")


def _distributed_failure_count(local_failure: int) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return local_failure
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    failed = torch.tensor(local_failure, dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    return int(failed.item())


def _synchronize_ranks() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _publish_rank_agreed_witness(
    path: Path,
    witness: dict[str, object],
    *,
    is_writer: bool | None = None,
    writer=None,
    failure_counter=None,
    synchronize=None,
) -> None:
    writer = _write_witness_atomically if writer is None else writer
    failure_counter = (
        _distributed_failure_count if failure_counter is None else failure_counter
    )
    synchronize = _synchronize_ranks if synchronize is None else synchronize
    is_writer = _is_machine_local_leader() if is_writer is None else is_writer
    local_error: Exception | None = None
    if is_writer:
        try:
            writer(path, witness)
        except Exception as error:  # propagate after the mandatory reduction
            local_error = error
    failed_ranks = failure_counter(int(local_error is not None))
    if failed_ranks:
        if local_error is not None:
            raise local_error
        raise RuntimeError(
            "Stage 2 witness publication failed on a machine-local leader"
        )
    synchronize()


def _tuning_config(method: str) -> Stage2TuningConfig:
    return Stage2TuningConfig(
        method=method,
        r=2,
        alpha=4,
        dropout=0.0,
        target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
    )


def _build_tuned_model(
    tuning: Stage2TuningConfig, *, adapter_path: Path | None = None
):
    # Re-seeding makes fresh base B byte-identical to fresh base A.
    torch.manual_seed(23)
    base = _tiny_model()
    if tuning.method == "full":
        if adapter_path is None:
            base.visual.requires_grad_(False)
            return base
        loaded = Qwen2_5_VLForConditionalGeneration.from_pretrained(adapter_path)
        loaded.visual.requires_grad_(False)
        return loaded
    return build_stage2_tuned_model(
        base,
        tuning,
        adapter_path=adapter_path,
        is_trainable=True,
    )


def _gathered_parameters(parameters):
    if not any(hasattr(parameter, "ds_id") for parameter in parameters):
        return nullcontext()
    from deepspeed import zero

    return zero.GatheredParameters(parameters, modifier_rank=None)


def _move_registered_buffers_to_device(model, device: torch.device) -> None:
    """Place buffers without moving parameters already owned by ZeRO-3."""
    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None and buffer.device != device:
                setattr(module, name, buffer.to(device))


def _snapshot_trainable_parameters(model) -> dict[str, torch.Tensor]:
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    with _gathered_parameters([parameter for _, parameter in named]):
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in named
        }


def _parameter_boundary(model) -> tuple[int, int]:
    qwen = resolve_stage2_qwen_model(model)
    visual_ids = {id(parameter) for parameter in qwen.visual.parameters()}
    visual_trainable_names = tuple(
        name
        for name, parameter in qwen.visual.named_parameters()
        if parameter.requires_grad
    )
    base_trainable_names = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and id(parameter) not in visual_ids
        and "lora_" not in name
    )
    return len(visual_trainable_names), len(base_trainable_names)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
    ).strip()


def _deepspeed_identity(deepspeed: str | None) -> tuple[str | None, int | None]:
    if deepspeed is None:
        return None, None
    path = Path(deepspeed).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    return str(path), int(config["zero_optimization"]["stage"])


def _assert_contract(witness, *, initial_steps: int = 1, resumed_steps: int = 2):
    if (
        witness["first_global_step"] != initial_steps
        or witness["resumed_global_step"] != resumed_steps
    ):
        raise AssertionError(f"global step was not restored: {witness}")
    if witness["optimizer_step_at_resume"] != initial_steps:
        raise AssertionError(f"optimizer state was not restored: {witness}")
    if witness["scheduler_step_at_resume"] != initial_steps:
        raise AssertionError(f"scheduler state was not restored: {witness}")
    if not witness["observed_local_batch_sizes"] or set(
        witness["observed_local_batch_sizes"]
    ) != {1}:
        raise AssertionError(f"rank-local batch contract failed: {witness}")
    if not math.isfinite(witness["first_loss"]) or not math.isfinite(
        witness["resumed_loss"]
    ):
        raise AssertionError(f"non-finite training loss: {witness}")
    if not witness["processor_present"]:
        raise AssertionError(f"processor artifact was not saved: {witness}")
    if witness["visual_trainable_parameters"] != 0:
        raise AssertionError(f"visual parameters became trainable: {witness}")
    if witness["tuning_method"] == "full":
        if (
            witness["adapter_checkpoint_present"]
            or witness["sidecar_present"]
            or witness["full_model_weights_absent"]
        ):
            raise AssertionError(f"full checkpoint contract failed: {witness}")
        if witness["base_trainable_parameters"] <= 0:
            raise AssertionError(f"full language model is frozen: {witness}")
    if witness["tuning_method"] in ("lora", "dora"):
        required_true = (
            "adapter_parameters_changed",
            "adapter_checkpoint_present",
            "sidecar_present",
            "processor_present",
            "full_model_weights_absent",
        )
        if any(not witness[field] for field in required_true):
            raise AssertionError(f"PEFT checkpoint contract failed: {witness}")
        if witness["base_trainable_parameters"] != 0:
            raise AssertionError(f"base parameters became trainable: {witness}")


def _phase_metadata(tuning, model):
    if not tuning.is_peft:
        return None
    qwen = resolve_stage2_qwen_model(model)
    return build_stage2_peft_metadata(
        tuning=tuning,
        base_model_reference="tiny-stage1-base",
        model_config=qwen.config,
        token_ids=LatentTokenIds(90, 91, 92),
        resolved_target_names=tuple(model.peft_config["default"].target_modules),
    )


def _run_initial_phase(
    *, experiment_id: str, deepspeed: str | None, tuning_method: str,
    zero_stage: int | None, initial_steps: int,
):
    from multinode_evidence import build_lifecycle_phase_evidence
    from stage2.evidence import observe_stage2_checkpoint_artifact

    paths = _claim_initial_phase_paths(experiment_id)
    output_dir = paths.checkpoint_dir
    runtime_profile = load_deepspeed_runtime_profile(
        deepspeed, world_size=int(os.environ.get("WORLD_SIZE", "1"))
    )
    if zero_stage is not None and runtime_profile.zero_stage != zero_stage:
        raise ValueError("--zero-stage must match the canonical DeepSpeed config")
    dataset = TinyStage2Dataset(build_rank_local_tiny_prepared_example(output_dir))
    tuning = _tuning_config(tuning_method)
    model = _build_tuned_model(tuning)
    processor = TinyStage2Processor()
    trainer = Stage2Trainer(
        model=model,
        args=_training_args(output_dir, max_steps=initial_steps, deepspeed=deepspeed),
        train_dataset=dataset,
        data_collator=_recording_production_collator(processor),
        processing_class=processor,
        stage2_peft_metadata=_phase_metadata(tuning, model),
        deepspeed_runtime_profile=runtime_profile,
    )
    result = trainer.train()
    checkpoint = output_dir / f"checkpoint-{initial_steps}"
    if not checkpoint.is_dir() or int(trainer.state.global_step) != initial_steps:
        raise AssertionError("Stage 2 initial phase did not produce checkpoint-2")
    optimizer_step = trainer._observed_optimizer_step
    gradient = trainer._representative_gradient_norm
    if type(optimizer_step) is not int or optimizer_step != initial_steps:
        raise AssertionError("Stage 2 initial optimizer step is invalid")
    if type(gradient) not in (int, float) or not math.isfinite(gradient):
        raise AssertionError("Stage 2 initial gradient is invalid")
    artifact_kind = observe_stage2_checkpoint_artifact(
        checkpoint, method=tuning.method
    )
    records = _collect_rank_records({
        "rank": resolve_launcher_rank(),
        "global_step": int(trainer.state.global_step),
        "optimizer_step": optimizer_step,
        "loss": float(result.training_loss),
        "gradient_norm": float(gradient),
        "worker_error": None,
    })
    witness = build_lifecycle_phase_evidence(
        phase="initial", stage="stage2", method=tuning.method,
        zero_stage=runtime_profile.zero_stage,
        config_digest=runtime_profile.config_digest,
        experiment_id=paths.experiment_id, checkpoint_path=checkpoint,
        artifact_kind=artifact_kind, rank_records=records,
    )
    _publish_rank_agreed_witness(
        paths.result_dir / "initial-witness.json", witness
    )
    if _is_world_process_zero():
        print(json.dumps(witness, sort_keys=True), flush=True)
    return witness


def _run_resume_phase(
    *, experiment_id: str, deepspeed: str | None, tuning_method: str,
    zero_stage: int | None, initial_steps: int, resumed_steps: int,
):
    from multinode_evidence import build_lifecycle_phase_evidence
    from stage2.evidence import observe_stage2_checkpoint_artifact

    paths = _phase_paths(experiment_id)
    output_dir = paths.checkpoint_dir
    checkpoint = output_dir / f"checkpoint-{initial_steps}"
    if not checkpoint.is_dir():
        raise FileNotFoundError("Stage 2 resume checkpoint-2 is not node-local")
    runtime_profile = load_deepspeed_runtime_profile(
        deepspeed, world_size=int(os.environ.get("WORLD_SIZE", "1"))
    )
    if zero_stage is not None and runtime_profile.zero_stage != zero_stage:
        raise ValueError("--zero-stage must match the canonical DeepSpeed config")
    tuning = _tuning_config(tuning_method)
    artifact_kind = observe_stage2_checkpoint_artifact(
        checkpoint, method=tuning.method
    )
    dataset = TinyStage2Dataset(build_rank_local_tiny_prepared_example(output_dir))
    model = _build_tuned_model(tuning, adapter_path=checkpoint)
    if deepspeed is not None:
        # Registration exists before Trainer/DeepSpeed wraps the model.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        _move_registered_buffers_to_device(model, torch.device("cuda", local_rank))
    processor = TinyStage2Processor()
    trainer = Stage2Trainer(
        model=model,
        args=_training_args(output_dir, max_steps=resumed_steps, deepspeed=deepspeed),
        train_dataset=dataset,
        data_collator=_recording_production_collator(processor),
        processing_class=processor,
        stage2_peft_metadata=_phase_metadata(tuning, model),
        deepspeed_runtime_profile=runtime_profile,
    )
    result = trainer.train(resume_from_checkpoint=str(checkpoint))
    optimizer_step = trainer._observed_optimizer_step
    gradient = trainer._representative_gradient_norm
    if (
        int(trainer.state.global_step) != resumed_steps
        or type(optimizer_step) is not int
        or optimizer_step != resumed_steps
    ):
        raise AssertionError("Stage 2 resume phase did not advance 2-to-3")
    if type(gradient) not in (int, float) or not math.isfinite(gradient):
        raise AssertionError("Stage 2 resume gradient is invalid")
    records = _collect_rank_records({
        "rank": resolve_launcher_rank(),
        "global_step": int(trainer.state.global_step),
        "optimizer_step": optimizer_step,
        "loss": float(result.training_loss),
        "gradient_norm": float(gradient),
        "worker_error": None,
    })
    witness = build_lifecycle_phase_evidence(
        phase="resume", stage="stage2", method=tuning.method,
        zero_stage=runtime_profile.zero_stage,
        config_digest=runtime_profile.config_digest,
        experiment_id=paths.experiment_id, checkpoint_path=checkpoint,
        artifact_kind=artifact_kind, rank_records=records,
    )
    _publish_rank_agreed_witness(
        paths.result_dir / "resume-witness.json", witness
    )
    if _is_world_process_zero():
        print(json.dumps(witness, sort_keys=True), flush=True)
    return witness


def run_harness(
    *,
    experiment_id: str,
    deepspeed: str | None,
    tuning_method: str,
    zero_stage: int | None = None,
    initial_steps: int = 1,
    resumed_steps: int = 2,
    phase: str = "lifecycle",
):
    if phase not in ("lifecycle", "initial", "resume"):
        raise ValueError("phase must be lifecycle, initial, or resume")
    if phase == "initial":
        return _run_initial_phase(
            experiment_id=experiment_id, deepspeed=deepspeed,
            tuning_method=tuning_method, zero_stage=zero_stage,
            initial_steps=initial_steps,
        )
    if phase == "resume":
        return _run_resume_phase(
            experiment_id=experiment_id, deepspeed=deepspeed,
            tuning_method=tuning_method, zero_stage=zero_stage,
            initial_steps=initial_steps, resumed_steps=resumed_steps,
        )
    paths = _claim_owned_paths(experiment_id)
    output_dir = paths.checkpoint_dir
    witness_path = paths.result_dir / "witness.json"
    runtime_profile = load_deepspeed_runtime_profile(
        deepspeed, world_size=int(os.environ.get("WORLD_SIZE", "1"))
    )
    if zero_stage is not None and runtime_profile.zero_stage != zero_stage:
        raise ValueError("--zero-stage must match the canonical DeepSpeed config")
    dataset = TinyStage2Dataset(build_rank_local_tiny_prepared_example(output_dir))
    tuning = _tuning_config(tuning_method)
    first_model = _build_tuned_model(tuning)
    initial_trainable = _snapshot_trainable_parameters(first_model)
    metadata = None
    if tuning.is_peft:
        qwen = resolve_stage2_qwen_model(first_model)
        metadata = build_stage2_peft_metadata(
            tuning=tuning,
            base_model_reference="tiny-stage1-base",
            model_config=qwen.config,
            token_ids=LatentTokenIds(90, 91, 92),
            resolved_target_names=tuple(
                first_model.peft_config["default"].target_modules
            ),
        )

    first_processor = TinyStage2Processor()
    first_collator = _recording_production_collator(first_processor)
    first_recorder = ResumeStateRecorder()
    first_trainer = Stage2Trainer(
        model=first_model,
        args=_training_args(output_dir, max_steps=initial_steps, deepspeed=deepspeed),
        train_dataset=dataset,
        data_collator=first_collator,
        callbacks=[first_recorder],
        processing_class=first_processor,
        stage2_peft_metadata=metadata,
        deepspeed_runtime_profile=runtime_profile,
    )
    first_result = first_trainer.train()
    first_global_step = int(first_trainer.state.global_step)
    checkpoint = output_dir / f"checkpoint-{initial_steps}"
    if not checkpoint.is_dir():
        raise AssertionError(f"first Stage 2 checkpoint was not saved: {checkpoint}")
    adapter_checkpoint_present = all(
        (checkpoint / filename).is_file()
        for filename in ("adapter_config.json", "adapter_model.safetensors")
    )
    sidecar_present = (checkpoint / "stage2_peft_metadata.json").is_file()
    processor_present = (checkpoint / "tiny_stage2_processor.json").is_file()
    full_model_weights_absent = not any(
        (checkpoint / filename).exists()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )

    resumed_model = _build_tuned_model(tuning, adapter_path=checkpoint)
    if deepspeed is not None:
        _move_registered_buffers_to_device(
            resumed_model, first_trainer.accelerator.device
        )
    resumed_processor = TinyStage2Processor()
    resumed_collator = _recording_production_collator(resumed_processor)
    resumed_recorder = ResumeStateRecorder()
    resumed_trainer = Stage2Trainer(
        model=resumed_model,
        args=_training_args(output_dir, max_steps=resumed_steps, deepspeed=deepspeed),
        train_dataset=dataset,
        data_collator=resumed_collator,
        callbacks=[resumed_recorder],
        processing_class=resumed_processor,
        stage2_peft_metadata=metadata,
        deepspeed_runtime_profile=runtime_profile,
    )
    resumed_result = resumed_trainer.train(resume_from_checkpoint=str(checkpoint))
    resumed_optimizer_step = _find_optimizer_step(
        resumed_trainer.optimizer.state_dict()
    )

    final_trainable = _snapshot_trainable_parameters(resumed_model)
    adapter_parameters_changed = (
        tuning.is_peft
        and final_trainable.keys() == initial_trainable.keys()
        and any(
            not torch.equal(initial_trainable[name], final_trainable[name])
            for name in initial_trainable
        )
    )
    visual_trainable, base_trainable = _parameter_boundary(resumed_model)

    rank_records = _collect_rank_records(
        {
            "rank": resolve_launcher_rank(),
            "first_global_step": first_global_step,
            "resumed_global_step": int(resumed_trainer.state.global_step),
            "optimizer_step": resumed_optimizer_step,
            "first_loss": float(first_result.training_loss),
            "resumed_loss": float(resumed_result.training_loss),
            "gradient_norm": resumed_trainer._representative_gradient_norm,
            "worker_error": None,
            "first_batches": first_collator.local_batch_sizes,
            "resumed_batches": resumed_collator.local_batch_sizes,
            "optimizer_step_at_resume": resumed_recorder.optimizer_step,
            "scheduler_step_at_resume": resumed_recorder.scheduler_step,
            "adapter_parameters_changed": adapter_parameters_changed,
            "visual_trainable_parameters": visual_trainable,
            "base_trainable_parameters": base_trainable,
        }
    )
    _validate_rank_lifecycle_records(
        rank_records, initial_steps=initial_steps, resumed_steps=resumed_steps
    )
    observed_batch_sizes = [
        batch_size
        for record in rank_records
        for key in ("first_batches", "resumed_batches")
        for batch_size in record[key]
    ]
    optimizer_steps = {record["optimizer_step_at_resume"] for record in rank_records}
    scheduler_steps = {record["scheduler_step_at_resume"] for record in rank_records}
    adapter_changes = {record["adapter_parameters_changed"] for record in rank_records}
    visual_trainable_counts = {
        record["visual_trainable_parameters"] for record in rank_records
    }
    base_trainable_counts = {
        record["base_trainable_parameters"] for record in rank_records
    }
    deepspeed_config, deepspeed_zero_stage = _deepspeed_identity(deepspeed)
    from stage2.evidence import observe_stage2_checkpoint_artifact

    artifact_kind = observe_stage2_checkpoint_artifact(
        checkpoint, method=tuning.method
    )
    witness = {
        "schema_version": 1,
        "stage": "stage2",
        "method": tuning.method,
        "zero_stage": runtime_profile.zero_stage,
        "config_digest": runtime_profile.config_digest,
        "artifact_kind": artifact_kind,
        "tuning_method": tuning.method,
        "first_global_step": first_global_step,
        "resumed_global_step": int(resumed_trainer.state.global_step),
        "rank_first_global_steps": [
            record["first_global_step"] for record in rank_records
        ],
        "rank_resumed_global_steps": [
            record["resumed_global_step"] for record in rank_records
        ],
        "rank_first_losses": [record["first_loss"] for record in rank_records],
        "rank_resumed_losses": [record["resumed_loss"] for record in rank_records],
        "rank_records": rank_records,
        "process_ranks": [record["rank"] for record in rank_records],
        "finite_loss": True,
        "finite_gradient": True,
        "worker_errors": [],
        "optimizer_state_restored": optimizer_steps == {initial_steps},
        "scheduler_state_restored": scheduler_steps == {initial_steps},
        "optimizer_step_at_resume": next(iter(optimizer_steps))
        if len(optimizer_steps) == 1
        else None,
        "scheduler_step_at_resume": next(iter(scheduler_steps))
        if len(scheduler_steps) == 1
        else None,
        "observed_local_batch_sizes": observed_batch_sizes,
        "world_size": len(rank_records),
        "first_loss": float(first_result.training_loss),
        "resumed_loss": float(resumed_result.training_loss),
        "deepspeed": deepspeed is not None,
        "deepspeed_config": deepspeed_config,
        "deepspeed_zero_stage": deepspeed_zero_stage,
        "adapter_parameters_changed": adapter_changes == {True},
        "adapter_checkpoint_present": adapter_checkpoint_present,
        "sidecar_present": sidecar_present,
        "processor_present": processor_present,
        "full_model_weights_absent": full_model_weights_absent,
        "visual_trainable_parameters": next(iter(visual_trainable_counts))
        if len(visual_trainable_counts) == 1
        else None,
        "base_trainable_parameters": next(iter(base_trainable_counts))
        if len(base_trainable_counts) == 1
        else None,
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "harness_path": str(Path(__file__).resolve()),
        "checkpoint": str(checkpoint),
        "experiment_id": paths.experiment_id,
    }
    _assert_contract(
        witness, initial_steps=initial_steps, resumed_steps=resumed_steps
    )
    _publish_rank_agreed_witness(witness_path, witness)
    if _is_world_process_zero():
        print(json.dumps(witness, sort_keys=True), flush=True)


def _positive_step(value: str) -> int:
    step = int(value)
    if step <= 0:
        raise argparse.ArgumentTypeError("lifecycle steps must be positive")
    return step


class _LifecycleParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.resumed_steps != parsed.initial_steps + 1:
            self.error("--resumed-steps must equal --initial-steps + 1")
        return parsed


def build_tiny_stage2_parser() -> argparse.ArgumentParser:
    parser = _LifecycleParser()
    parser.add_argument(
        "--phase", choices=("lifecycle", "initial", "resume"), default="lifecycle"
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--deepspeed")
    parser.add_argument("--zero-stage", type=int, choices=(1, 2, 3))
    parser.add_argument(
        "--tuning-method", choices=("full", "lora", "dora"), default="full"
    )
    parser.add_argument("--initial-steps", type=_positive_step, default=1)
    parser.add_argument("--resumed-steps", type=_positive_step, default=2)
    return parser


def parse_args(argv: list[str] | None = None):
    return build_tiny_stage2_parser().parse_args(argv)


def main():
    args = parse_args()
    run_harness(
        phase=args.phase,
        experiment_id=args.experiment_id,
        deepspeed=args.deepspeed,
        tuning_method=args.tuning_method,
        zero_stage=args.zero_stage,
        initial_steps=args.initial_steps,
        resumed_steps=args.resumed_steps,
    )


if __name__ == "__main__":
    main()
