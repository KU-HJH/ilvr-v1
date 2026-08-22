#!/usr/bin/env python3
"""One-optimizer-step witness for ILVR Stage 1 and Stage-2 EW variants.

The script deliberately performs no checkpoint hashing. Precomputed targets
are authenticated by their already-published manifest, and the resulting JSON
contains only small scalar/structural witnesses.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
STAGE2_CONFIG_ROOT = REPOSITORY_ROOT / "configs" / "evidence_window" / "stage2"
CASE_CONFIGS = {
    "baseline": "baseline.json",
    "ew-reference-free": "ew_reference_free.json",
    "ew-precomputed-dpo": "ew_precomputed_dpo.json",
    "relation-kd": "relation_kd.json",
    "hidden-residual-kd": "hidden_residual_kd.json",
    "online-frozen-zero1": "online_frozen_zero1.json",
    "online-frozen-zero2": "online_frozen_zero2.json",
}
SINGLE_GPU_CASES = (
    "stage1-baseline",
    "baseline",
    "ew-reference-free",
    "ew-precomputed-dpo",
    "relation-kd",
    "hidden-residual-kd",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real ILVR optimizer step and write a compact JSON witness."
    )
    parser.add_argument(
        "--case",
        required=True,
        choices=("all-single-gpu", "stage1-baseline", *CASE_CONFIGS),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--sample-index", required=True, type=int)
    parser.add_argument("--ew-config")
    parser.add_argument("--deepspeed-config")
    parser.add_argument("--reference-artifact-dir")
    parser.add_argument("--output-dir", required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.sample_index < 0:
        raise ValueError("sample index must be nonnegative")
    return args


def _require_inputs(args: argparse.Namespace) -> None:
    for name in ("stage1_checkpoint", "data", "image_root"):
        value = Path(getattr(args, name)).expanduser()
        if not value.exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} does not exist: {value}")
    if args.deepspeed_config is not None and not Path(args.deepspeed_config).is_file():
        raise FileNotFoundError(
            f"--deepspeed-config does not exist: {args.deepspeed_config}"
        )


def _read_source_row(path: str, sample_index: int) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == sample_index:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("selected dataset row must be a JSON object")
                return value
    raise IndexError(f"sample index {sample_index} is outside {path}")


def _load_processor_and_model(checkpoint: str, *, stage: str, device_dtype):
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    processor.tokenizer.add_tokens(
        ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"],
        special_tokens=True,
    )
    config = Qwen2_5_VLConfig.from_pretrained(checkpoint, trust_remote_code=True)
    config.stage = stage
    config.compress_strategy = "average"
    config.latent_size = 8
    marker_ids = {
        name: int(processor.tokenizer(text, return_tensors="pt")["input_ids"].item())
        for name, text in {
            "pad": "<|latent_pad|>",
            "start": "<|latent_start|>",
            "end": "<|latent_end|>",
        }.items()
    }
    config.latent_token_id = marker_ids["pad"]
    config.latent_start_id = marker_ids["start"]
    config.latent_end_id = marker_ids["end"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=device_dtype,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    return processor, model, marker_ids


def _apply_smoke_lora(model):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )


def _token_ids(processor, marker_ids):
    from evidence_window.layout import TokenIds

    padding = int(
        processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"].item()
    )
    termination = int(
        processor.tokenizer("<|im_end|>", return_tensors="pt")["input_ids"].item()
    )
    return TokenIds(
        latent_start=marker_ids["start"],
        latent_end=marker_ids["end"],
        latent_pad=marker_ids["pad"],
        padding=padding,
        special=frozenset(int(value) for value in processor.tokenizer.all_special_ids),
        assistant_termination=frozenset({termination}),
    )


def _read_manifest_identity(artifact_dir: str):
    from evidence_window.artifacts import ArtifactIdentity

    with (Path(artifact_dir) / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return ArtifactIdentity.from_dict(manifest["identity"])


def _resolve_stage2_config(args: argparse.Namespace, case: str):
    from evidence_window.stage2_config import (
        Stage2EvidenceWindowConfig,
        Stage2ScheduleConfig,
    )

    path = Path(args.ew_config) if args.ew_config else STAGE2_CONFIG_ROOT / CASE_CONFIGS[case]
    config = Stage2EvidenceWindowConfig.from_json(path)
    # A one-step smoke must exercise the named objective instead of remaining
    # inside a production warmup interval.
    config = replace(config, schedule=Stage2ScheduleConfig())
    if config.reference.mode == "precomputed":
        if not args.reference_artifact_dir:
            raise ValueError("precomputed smoke requires --reference-artifact-dir")
        config = replace(
            config,
            reference=replace(config.reference, artifact=args.reference_artifact_dir),
        )
    if config.reference.mode == "online_frozen":
        config = replace(
            config,
            reference=replace(config.reference, checkpoint=args.stage1_checkpoint),
        )
    return config


def _training_arguments(output_dir: Path, deepspeed_config: str | None):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(output_dir),
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=8192,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
    )


def _gradient_witness(model) -> dict[str, Any]:
    witness = {"seen": False, "finite": True, "nonzero": False}

    def observe(gradient):
        witness["seen"] = True
        witness["finite"] = witness["finite"] and bool(gradient.isfinite().all().item())
        witness["nonzero"] = witness["nonzero"] or bool(gradient.count_nonzero().item())
        return gradient

    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.register_hook(observe)
    return witness


def _forward_witness(model) -> tuple[dict[str, int], Any]:
    witness = {"total": 0, "blocked": 0, "hidden": 0}

    def observe(_module, _args, kwargs):
        witness["total"] += 1
        witness["blocked"] += int(kwargs.get("attention_block_mask") is not None)
        witness["hidden"] += int(bool(kwargs.get("output_hidden_states", False)))

    return witness, model.register_forward_pre_hook(observe, with_kwargs=True)


def _loss_witness(log_history) -> dict[str, float]:
    result: dict[str, float] = {}
    for record in log_history:
        for key, value in record.items():
            if key == "loss" or key.startswith("loss/") or key.startswith("train_loss"):
                if isinstance(value, (int, float)):
                    result[key] = float(value)
    if not result:
        raise RuntimeError("trainer did not publish a finite loss witness")
    if not all(value == value and abs(value) != float("inf") for value in result.values()):
        raise RuntimeError("optimizer smoke produced a non-finite loss")
    return result


def _run_stage2(args: argparse.Namespace, case: str, case_dir: Path) -> dict[str, Any]:
    import torch
    from evidence_window.artifacts import PrecomputedReferenceProvider
    from evidence_window.stage2_config import read_deepspeed_zero_stage
    from evidence_window.stage2_layout import extract_stage2_layout, select_stage2_edges
    from evidence_window.stage2_reference import OnlineFrozenStage2Reference
    from stage2_data import build_stage2_record, collate_fn_stage2
    import task_deepseed
    from trainer_stage2 import CustomTrainerStage2

    task_deepseed.TRAIN_IMAGE_ROOT = args.image_root
    zero_stage = (
        read_deepspeed_zero_stage(args.deepspeed_config)
        if args.deepspeed_config
        else 0
    )
    config = _resolve_stage2_config(args, case)
    config.validate_runtime(zero_stage)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor, model, marker_ids = _load_processor_and_model(
        args.stage1_checkpoint, stage="stage2", device_dtype=dtype
    )
    model = _apply_smoke_lora(model)
    for parameter in model.visual.parameters():
        parameter.requires_grad_(False)

    row = _read_source_row(args.data, args.sample_index)
    record = build_stage2_record(row, args.sample_index)
    collator_args = SimpleNamespace(latent_size=8, latent_ce_ratio=0.0)
    batch = collate_fn_stage2([record], processor, collator_args)
    token_ids = _token_ids(processor, marker_ids)
    layout = extract_stage2_layout(
        batch["input_ids"], batch["attention_mask"], batch["labels"], token_ids
    )
    edge_count = len(
        select_stage2_edges(layout, config.loo.max_evidence_edges_per_window)
    )

    reference_provider = None
    reference_identity = None
    reference_model = None
    if config.reference.mode == "precomputed":
        assert args.reference_artifact_dir is not None
        reference_identity = _read_manifest_identity(args.reference_artifact_dir)
        reference_provider = PrecomputedReferenceProvider(
            args.reference_artifact_dir, reference_identity
        )
    elif config.reference.mode == "online_frozen":
        _, reference_model, _ = _load_processor_and_model(
            args.stage1_checkpoint, stage="stage2", device_dtype=dtype
        )

    training_args = _training_arguments(case_dir, args.deepspeed_config)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=[record],
        data_collator=lambda records: collate_fn_stage2(records, processor, collator_args),
        processing_class=processor.tokenizer,
        evidence_window_config=config,
        latent_token_ids={
            "start": token_ids.latent_start,
            "end": token_ids.latent_end,
            "pad": token_ids.latent_pad,
            "padding": token_ids.padding,
            "special": tuple(token_ids.special),
            "assistant_termination": tuple(token_ids.assistant_termination),
        },
    )
    if reference_provider is not None:
        trainer_kwargs.update(
            reference_provider=reference_provider,
            reference_identity=reference_identity,
        )
    if reference_model is not None:
        trainer_kwargs.update(
            reference_model_factory=lambda model=reference_model: model,
            deepspeed_zero_stage=zero_stage,
            policy_initialization_checkpoint=args.stage1_checkpoint,
        )
    trainer = CustomTrainerStage2(**trainer_kwargs)
    policy_grad = _gradient_witness(trainer.model)
    policy_forwards, policy_hook = _forward_witness(trainer.model)
    fixed = getattr(trainer, "_stage2_reference_provider", None)
    reference_forwards = {"total": 0, "blocked": 0, "hidden": 0}
    reference_hook = None
    if isinstance(fixed, OnlineFrozenStage2Reference):
        reference_forwards, reference_hook = _forward_witness(fixed.model)

    trainer.train()
    policy_hook.remove()
    if reference_hook is not None:
        reference_hook.remove()
    reference_has_grad = bool(
        fixed is not None
        and hasattr(fixed, "parameters")
        and any(parameter.grad is not None for parameter in fixed.parameters())
    )
    expected_policy_calls = 1 if config.is_disabled else 2 + edge_count
    if policy_forwards["total"] != expected_policy_calls:
        raise RuntimeError(
            f"policy forward count mismatch: expected {expected_policy_calls}, "
            f"got {policy_forwards['total']}"
        )
    if config.residual.enabled and config.residual.lambda_res > 0:
        if policy_forwards["hidden"] != 1 + edge_count:
            raise RuntimeError("hidden-state residual smoke did not collect exact auxiliary states")
    elif policy_forwards["hidden"] != 0:
        raise RuntimeError("non-residual smoke unexpectedly collected hidden states")
    if reference_has_grad:
        raise RuntimeError("fixed Stage-2 reference received gradients")

    result = {
        "case": case,
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "device": str(next(trainer.model.parameters()).device),
        "dtype": str(next(trainer.model.parameters()).dtype),
        "zero_stage": zero_stage,
        "losses": _loss_witness(trainer.state.log_history),
        "edge_count": edge_count,
        "policy_forwards": policy_forwards,
        "reference_forwards": reference_forwards,
        "policy_gradient": policy_grad,
        "reference_has_gradient": reference_has_grad,
        "disabled_auxiliary_forward_count": (
            max(policy_forwards["total"] - 1, 0) if config.is_disabled else None
        ),
    }
    del trainer, model, reference_model, processor, batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _run_stage1(args: argparse.Namespace, case_dir: Path) -> dict[str, Any]:
    import torch
    from evidence_window.config import EvidenceWindowConfig
    from main import collate_fn_stage1
    import task_deepseed
    from task_deepseed import interleaved_latent_cot_preprocess_function
    from trainer import CustomTrainerStage1

    task_deepseed.TRAIN_IMAGE_ROOT = args.image_root
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor, model, marker_ids = _load_processor_and_model(
        args.model, stage="stage1", device_dtype=dtype
    )
    model = _apply_smoke_lora(model)
    for parameter in model.visual.parameters():
        parameter.requires_grad_(False)
    row = _read_source_row(args.data, args.sample_index)
    record = interleaved_latent_cot_preprocess_function(row)
    collator_args = SimpleNamespace(latent_size=8)
    trainer = CustomTrainerStage1(
        model=model,
        args=_training_arguments(case_dir, args.deepspeed_config),
        train_dataset=[record],
        data_collator=lambda records: collate_fn_stage1(records, processor, collator_args),
        tokenizer=processor.tokenizer,
        evidence_window_config=EvidenceWindowConfig.disabled(),
    )
    policy_grad = _gradient_witness(trainer.model)
    policy_forwards, policy_hook = _forward_witness(trainer.model)
    teacher_forwards, teacher_hook = _forward_witness(trainer._ema.teacher)
    teacher_grad = any(parameter.grad is not None for parameter in trainer._ema.teacher.parameters())
    trainer.train()
    policy_hook.remove()
    teacher_hook.remove()
    teacher_grad = teacher_grad or any(
        parameter.grad is not None for parameter in trainer._ema.teacher.parameters()
    )
    if teacher_grad:
        raise RuntimeError("Stage-1 EMA teacher received gradients")
    if trainer._ema_update_count != 1:
        raise RuntimeError(
            f"Stage-1 EMA update count mismatch: expected 1, got {trainer._ema_update_count}"
        )
    result = {
        "case": "stage1-baseline",
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "device": str(next(trainer.model.parameters()).device),
        "dtype": str(next(trainer.model.parameters()).dtype),
        "losses": _loss_witness(trainer.state.log_history),
        "edge_count": 0,
        "policy_forwards": policy_forwards,
        "teacher_forwards": teacher_forwards,
        "policy_gradient": policy_grad,
        "teacher_has_gradient": teacher_grad,
        "ema_update_count": trainer._ema_update_count,
        "disabled_auxiliary_forward_count": 0,
    }
    del trainer, model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _write_result(output_dir: Path, result: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{result['case']}-rank-{result['rank']:05d}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _require_inputs(args)
    output_dir = Path(args.output_dir)
    cases = SINGLE_GPU_CASES if args.case == "all-single-gpu" else (args.case,)
    for case in cases:
        case_dir = output_dir / case
        result = (
            _run_stage1(args, case_dir)
            if case == "stage1-baseline"
            else _run_stage2(args, case, case_dir)
        )
        path = _write_result(output_dir, result)
        if int(os.environ.get("RANK", "0")) == 0:
            print(path)


if __name__ == "__main__":
    main()
