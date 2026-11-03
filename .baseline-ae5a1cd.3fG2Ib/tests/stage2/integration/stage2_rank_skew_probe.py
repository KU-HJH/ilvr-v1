"""One-step two-rank Stage 2 latent-work alignment and memory probe."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist

from artifact_paths import resolve_run_artifact_paths
from deepspeed_runtime import (
    canonical_deepspeed_config_path,
    load_deepspeed_runtime_profile,
)
from main import build_stage2_training_arguments, build_stage2_training_components
from stage2.peft import resolve_stage2_qwen_model
from stage2.trainer import Stage2Trainer
from utils_deepseed import parse_args as parse_training_args


def build_rank_skew_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--deepspeed", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.set_defaults(
        method="dora",
        max_steps=1,
        micro_batch=1,
        gradient_accumulation_steps=8,
    )
    return parser


def select_probe_examples(examples: Sequence):
    """Retain eight deterministic rows from each of the first two buckets."""
    buckets: dict[int, list] = defaultdict(list)
    for example in examples:
        buckets[example.latent_block_count].append(example)
    available = [count for count, rows in sorted(buckets.items()) if rows]
    if len(available) < 2:
        raise ValueError("rank-skew probe requires at least two latent-count buckets")
    selected = []
    for count in available[:2]:
        rows = buckets[count]
        selected.extend(rows[index % len(rows)] for index in range(8))
    return selected


class RecordingRankSkewCollator:
    """Record rank-local latent work while preserving production collation."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.latent_block_counts: list[int] = []

    def __call__(self, examples):
        if len(examples) != 1:
            raise AssertionError("rank-skew probe requires micro-batch size 1")
        self.latent_block_counts.append(int(examples[0].latent_block_count))
        return self.delegate(examples)


def _observe_decoder_forwards(model, operation):
    """Run one micro-step operation while counting actual decoder forwards."""
    qwen = resolve_stage2_qwen_model(model)
    forward_count = 0

    def record_forward(_module, _args, _output):
        nonlocal forward_count
        forward_count += 1

    handle = qwen.model.register_forward_hook(record_forward)
    try:
        result = operation()
    finally:
        handle.remove()
    return result, forward_count


class RankSkewProbeTrainer(Stage2Trainer):
    """Production Stage2Trainer with a read-only decoder-work recorder."""

    def __init__(self, *args, **kwargs):
        self.decoder_forward_counts: list[int] = []
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        parent_compute_loss = super().compute_loss
        result, forward_count = _observe_decoder_forwards(
            model,
            lambda: parent_compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            ),
        )
        diagnostics = self._latest_stage2_diagnostics
        if diagnostics is None:
            raise AssertionError("Stage 2 forward diagnostics were not collected")
        if forward_count < 1:
            raise AssertionError("Stage 2 decoder did not execute")
        self.decoder_forward_counts.append(forward_count)
        return result


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _initialize_distributed() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise ValueError(f"rank-skew probe requires world size 2, got {world_size}")
    if not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")


def _require_fresh_run(paths) -> None:
    collision = paths.checkpoint_dir.exists() or paths.result_dir.exists()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    observed = torch.tensor(int(collision), dtype=torch.int32, device=device)
    dist.all_reduce(observed, op=dist.ReduceOp.MAX)
    if bool(observed.item()):
        raise FileExistsError(
            f"refusing to reuse Stage 2 experiment ID: {paths.experiment_id}"
        )


def _production_args(args, paths):
    stage1_checkpoint = args.stage1_checkpoint.resolve(strict=True)
    argv = [
        "--stage",
        "stage2",
        "--model",
        args.model,
        "--load_model_path",
        (
            args.model
            if (stage1_checkpoint / "adapter_config.json").is_file()
            else str(stage1_checkpoint)
        ),
        "--data_path",
        str(args.data_path.resolve(strict=True)),
        "--image_root",
        str(args.image_root.resolve(strict=True)),
        "--save_model_path",
        str(paths.checkpoint_dir),
        "--log_file",
        str(paths.log_dir / "rank-skew.log"),
        "--tuning_method",
        "dora",
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "8",
        "--max_steps",
        "1",
        "--save_steps",
        "1",
        "--deepspeed_config",
        str(args.deepspeed.resolve(strict=True)),
        "--attention_implementation",
        "flash_attention_2",
        "--no-save_final_model",
        "--no_final_save",
    ]
    if (stage1_checkpoint / "adapter_config.json").is_file():
        argv.extend(("--stage1_adapter_path", str(stage1_checkpoint)))
    return parse_training_args(argv)


def _gather(record: dict[str, object]) -> list[dict[str, object]]:
    gathered: list[dict[str, object] | None] = [None, None]
    dist.all_gather_object(gathered, record)
    if any(item is None for item in gathered):
        raise RuntimeError("rank-skew collective returned an empty record")
    return [item for item in gathered if item is not None]


def _aligned(sequences: list[list[int]]) -> bool:
    return bool(sequences) and all(
        len(set(values)) == 1 for values in zip(*sequences, strict=True)
    )


def _validate_records(records: list[dict[str, object]]) -> None:
    if len(records) != 2:
        raise AssertionError(f"rank-skew probe expected two records: {records}")
    for record in records:
        if record["optimizer_steps"] != 1:
            raise AssertionError(f"rank did not apply one optimizer step: {records}")
        if len(record["latent_block_counts"]) != 8:
            raise AssertionError(f"rank did not execute eight micro-steps: {records}")
        if len(record["decoder_forward_counts"]) != 8:
            raise AssertionError(f"rank decoder sequence is incomplete: {records}")
        if record["peak_allocated_bytes"] <= 0 or record["peak_reserved_bytes"] <= 0:
            raise AssertionError(f"rank memory metric is invalid: {records}")
    block_sequences = [record["latent_block_counts"] for record in records]
    forward_sequences = [record["decoder_forward_counts"] for record in records]
    if not _aligned(block_sequences) or not _aligned(forward_sequences):
        raise AssertionError(f"rank-skew latent work is not aligned: {records}")


def _publish(paths, witness: dict[str, object]) -> None:
    local_error: Exception | None = None
    if _rank() == 0:
        try:
            paths.result_dir.mkdir(parents=True, exist_ok=False)
            (paths.result_dir / "rank-skew.json").write_text(
                json.dumps(witness, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as error:  # propagated collectively below
            local_error = error
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    failed = torch.tensor(int(local_error is not None), dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    if int(failed.item()):
        if local_error is not None:
            raise local_error
        raise RuntimeError("rank-skew witness publication failed on rank zero")
    dist.barrier()


def run_rank_skew_probe(args: argparse.Namespace) -> dict[str, object]:
    _initialize_distributed()
    paths = resolve_run_artifact_paths(args.experiment_id)
    _require_fresh_run(paths)
    training_args = _production_args(args, paths)
    profile = load_deepspeed_runtime_profile(
        training_args.deepspeed_config, world_size=2
    )
    expected_config = canonical_deepspeed_config_path(args.zero_stage).resolve()
    if profile.zero_stage != args.zero_stage or profile.config_path != expected_config:
        raise ValueError(
            f"rank-skew probe ZeRO-{args.zero_stage} requires {expected_config}"
        )
    training_args.deepspeed_runtime_profile = profile

    components = build_stage2_training_components(training_args)
    selected = select_probe_examples(components.train_dataset)
    collator = RecordingRankSkewCollator(components.data_collator)
    trainer = RankSkewProbeTrainer(
        model=components.model,
        args=build_stage2_training_arguments(training_args),
        train_dataset=selected,
        data_collator=collator,
        processing_class=components.processor,
        stage2_peft_metadata=components.peft_metadata,
        deepspeed_runtime_profile=profile,
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    checkpoint = paths.checkpoint_dir / "checkpoint-1"
    if not checkpoint.is_dir():
        raise AssertionError("rank-skew probe did not save checkpoint-1")
    identity = trainer.stage2_sampler_identity
    if identity is None:
        raise AssertionError("production Stage 2 balanced sampler was not selected")
    local_record = {
        "optimizer_steps": int(trainer.state.global_step),
        "latent_block_counts": collator.latent_block_counts,
        "decoder_forward_counts": trainer.decoder_forward_counts,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    records = _gather(local_record)
    _validate_records(records)

    block_sequences = [record["latent_block_counts"] for record in records]
    forward_sequences = [record["decoder_forward_counts"] for record in records]
    witness = {
        "schema_version": 1,
        "stage": "stage2",
        "zero_stage": profile.zero_stage,
        "config_digest": profile.config_digest,
        "world_size": profile.world_size,
        "method": "dora",
        "optimizer_steps": 1,
        "rank_block_counts": block_sequences,
        "rank_decoder_forward_counts": forward_sequences,
        "aligned_at_every_micro_step": _aligned(block_sequences)
        and _aligned(forward_sequences),
        "padded_samples": identity.padded_samples,
        "sampler_identity": identity.to_dict(),
        "rank_peak_allocated_bytes": [
            record["peak_allocated_bytes"] for record in records
        ],
        "rank_peak_reserved_bytes": [
            record["peak_reserved_bytes"] for record in records
        ],
        "experiment_id": paths.experiment_id,
        "checkpoint": str(checkpoint),
        "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
    }
    _publish(paths, witness)
    return witness


def main(argv: list[str] | None = None) -> int:
    args = build_rank_skew_parser().parse_args(argv)
    witness = run_rank_skew_probe(args)
    if _rank() == 0:
        print(json.dumps(witness, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
