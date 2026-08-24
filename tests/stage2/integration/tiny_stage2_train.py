"""Tiny Stage 2 save/resume harness for single-process and ZeRO-3 checks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    TrainerCallback,
)

from main import build_stage2_training_arguments
from stage2.data import Stage2DataCollator, prepare_stage2_example
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


class RecordingStage2Collator:
    """Record local size while delegating the production Stage 2 collator."""

    def __init__(self, delegate: Stage2DataCollator):
        self.delegate = delegate
        self.local_batch_sizes: list[int] = []

    def __call__(self, features):
        self.local_batch_sizes.append(len(features))
        return self.delegate(features)


def _recording_production_collator() -> RecordingStage2Collator:
    return RecordingStage2Collator(
        Stage2DataCollator(
            processor=TinyStage2Processor(),
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


def _is_world_process_zero() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _assert_contract(witness):
    if witness["first_global_step"] != 1 or witness["resumed_global_step"] != 2:
        raise AssertionError(f"global step was not restored: {witness}")
    if witness["optimizer_step_at_resume"] != 1:
        raise AssertionError(f"optimizer state was not restored: {witness}")
    if witness["scheduler_step_at_resume"] != 1:
        raise AssertionError(f"scheduler state was not restored: {witness}")
    if not witness["observed_local_batch_sizes"] or set(
        witness["observed_local_batch_sizes"]
    ) != {1}:
        raise AssertionError(f"rank-local batch contract failed: {witness}")
    if not math.isfinite(witness["first_loss"]) or not math.isfinite(
        witness["resumed_loss"]
    ):
        raise AssertionError(f"non-finite training loss: {witness}")


def run_harness(*, output_dir: Path, witness_path: Path, deepspeed: str | None):
    torch.manual_seed(23)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = TinyStage2Dataset(
        build_tiny_prepared_example(output_dir / "tiny-data")
    )

    first_collator = _recording_production_collator()
    first_recorder = ResumeStateRecorder()
    first_trainer = Stage2Trainer(
        model=_tiny_model(),
        args=_training_args(output_dir, max_steps=1, deepspeed=deepspeed),
        train_dataset=dataset,
        data_collator=first_collator,
        callbacks=[first_recorder],
    )
    first_result = first_trainer.train()
    first_global_step = int(first_trainer.state.global_step)
    checkpoint = output_dir / "checkpoint-1"
    if not checkpoint.is_dir():
        raise AssertionError(f"first Stage 2 checkpoint was not saved: {checkpoint}")

    resumed_collator = _recording_production_collator()
    resumed_recorder = ResumeStateRecorder()
    resumed_trainer = Stage2Trainer(
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint),
        args=_training_args(output_dir, max_steps=2, deepspeed=deepspeed),
        train_dataset=dataset,
        data_collator=resumed_collator,
        callbacks=[resumed_recorder],
    )
    resumed_result = resumed_trainer.train(resume_from_checkpoint=str(checkpoint))

    rank_records = _collect_rank_records(
        {
            "first_batches": first_collator.local_batch_sizes,
            "resumed_batches": resumed_collator.local_batch_sizes,
            "optimizer_step_at_resume": resumed_recorder.optimizer_step,
            "scheduler_step_at_resume": resumed_recorder.scheduler_step,
        }
    )
    observed_batch_sizes = [
        batch_size
        for record in rank_records
        for key in ("first_batches", "resumed_batches")
        for batch_size in record[key]
    ]
    optimizer_steps = {record["optimizer_step_at_resume"] for record in rank_records}
    scheduler_steps = {record["scheduler_step_at_resume"] for record in rank_records}
    witness = {
        "first_global_step": first_global_step,
        "resumed_global_step": int(resumed_trainer.state.global_step),
        "optimizer_state_restored": optimizer_steps == {1},
        "scheduler_state_restored": scheduler_steps == {1},
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
        "checkpoint": str(checkpoint),
    }
    _assert_contract(witness)
    if _is_world_process_zero():
        witness_path.parent.mkdir(parents=True, exist_ok=True)
        witness_path.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n")
        print(json.dumps(witness, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--witness-path", type=Path)
    parser.add_argument("--deepspeed")
    return parser.parse_args()


def main():
    args = parse_args()
    witness_path = args.witness_path or args.output_dir / "witness.json"
    run_harness(
        output_dir=args.output_dir.resolve(),
        witness_path=witness_path.resolve(),
        deepspeed=args.deepspeed,
    )


if __name__ == "__main__":
    main()
