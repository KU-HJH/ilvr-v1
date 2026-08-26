"""Tiny two-rank witness for Stage 1 PEFT gradient checkpointing with ZeRO-3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import deepspeed
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from stage1.peft import build_stage1_tuned_model
from stage1.types import Stage1TuningConfig


LATENT_TOKEN_IDS = (125, 126, 127)


def _model(*, use_reentrant: bool):
    config = Qwen2_5_VLConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
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
        stage="stage1",
        latent_size=2,
        latent_token_id=LATENT_TOKEN_IDS[0],
        latent_start_id=LATENT_TOKEN_IDS[1],
        latent_end_id=LATENT_TOKEN_IDS[2],
        attn_implementation="eager",
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 0]},
    )
    model = Qwen2_5_VLForConditionalGeneration(config)
    model = build_stage1_tuned_model(
        model,
        Stage1TuningConfig(
            method="dora",
            rank=4,
            alpha=8,
            dropout=0.0,
            target_suffixes=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
        ),
        LATENT_TOKEN_IDS,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
    )
    model.train()
    return model


def _deepspeed_config() -> dict:
    return {
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
            "stage3_param_persistence_threshold": 0,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
        "steps_per_print": 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-reentrant", action=argparse.BooleanOptionalAction)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    deepspeed.init_distributed(dist_backend="nccl")
    torch.manual_seed(19)
    engine, _, _, _ = deepspeed.initialize(
        model=_model(use_reentrant=args.use_reentrant),
        model_parameters=None,
        config=_deepspeed_config(),
    )
    device = engine.device
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
    labels = input_ids.clone()
    outputs = engine(input_ids=input_ids, labels=labels, use_cache=False)
    engine.backward(outputs.loss)
    engine.step()

    if int(os.environ["RANK"]) == 0:
        args.result.write_text(
            json.dumps(
                {
                    "use_reentrant": args.use_reentrant,
                    "loss": float(outputs.loss.detach().cpu()),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
