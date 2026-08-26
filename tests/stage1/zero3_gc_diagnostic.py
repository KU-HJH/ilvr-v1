"""Tiny two-rank Stage 1 DoRA train/save/resume witness under ZeRO-3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import deepspeed
import torch
from deepspeed.utils import safe_get_full_grad, safe_get_full_optimizer_state
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from stage1.peft import build_stage1_tuned_model
from stage1.peft import validate_stage1_trainable_boundary
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
            "stage3_max_live_parameters": 0,
            "stage3_max_reuse_distance": 0,
            "stage3_prefetch_bucket_size": 0,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
        "steps_per_print": 1000,
    }


def _gradient_witness_weight(model) -> torch.nn.Parameter:
    # PEFT initializes LoRA B at zero, so B (not A) has a nonzero first-step
    # gradient while still proving adapter optimizer state survives resume.
    suffix = "model.layers.0.self_attn.q_proj.lora_B.default.weight"
    matches = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one {suffix}, found {len(matches)}")
    return matches[0]


def _backward_step(engine) -> tuple[float, float]:
    device = engine.device
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
    labels = input_ids.clone()
    outputs = engine(input_ids=input_ids, labels=labels, use_cache=False)
    engine.backward(outputs.loss)
    gradient = safe_get_full_grad(_gradient_witness_weight(engine.module))
    if gradient is None or not bool(torch.isfinite(gradient).all().item()):
        raise AssertionError("DoRA resume diagnostic requires a finite LoRA gradient")
    gradient_norm = float(gradient.float().norm().detach().cpu())
    if gradient_norm <= 0:
        raise AssertionError("DoRA resume diagnostic requires a nonzero LoRA gradient")
    engine.step()
    return float(outputs.loss.detach().cpu()), gradient_norm


def _optimizer_state(model) -> torch.Tensor:
    state = safe_get_full_optimizer_state(
        _gradient_witness_weight(model), "exp_avg"
    )
    if state is None or not bool(torch.isfinite(state).all().item()):
        raise AssertionError("DoRA resume diagnostic requires finite Adam exp_avg state")
    return state.detach().clone()


def _install_dora_trace(model) -> list[dict[str, object]]:
    target = next(
        module
        for name, module in model.named_modules()
        if name.endswith("model.layers.0.self_attn.q_proj")
    )
    magnitude = target.lora_magnitude_vector["default"]
    observations: list[dict[str, object]] = []

    def observe(module, _args, kwargs):
        lora_a = kwargs["lora_A"].weight
        lora_b = kwargs["lora_B"].weight
        base = kwargs["base_layer"]
        external = set(getattr(module, "_external_params", {}))
        observations.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "lora_a_shape": list(lora_a.shape),
                "lora_a_external": id(lora_a) in external,
                "lora_b_shape": list(lora_b.shape),
                "lora_b_external": id(lora_b) in external,
                "base_shape": list(base.weight.shape),
                "base_external": id(base.weight) in external,
            }
        )

    magnitude.register_forward_pre_hook(observe, with_kwargs=True)
    return observations


def _install_scaled_resume_partition(model) -> None:
    """Emulate 3B release pressure immediately before DoRA recomputation.

    This hook is installed before DeepSpeed, so its recompute-only partition
    runs before ZeRO's magnitude-module pre-hook. Correct external registration
    lets that normal pre-hook gather LoRA A again for the DoRA consumer.
    """

    target = next(
        module
        for name, module in model.named_modules()
        if name.endswith("model.layers.0.self_attn.q_proj")
    )
    magnitude = target.lora_magnitude_vector["default"]

    def partition_before_deepspeed(_module, _args, kwargs):
        if not torch.is_grad_enabled():
            return
        lora_a = kwargs["lora_A"].weight
        # The 3B resume reaches this consumer with a normal ZeRO partition in
        # NOT_AVAILABLE/[0] state. This tiny graph's outer trace eagerly fetches
        # A, so restore the observed state before the consumer's own ZeRO hook.
        lora_a.data = torch.empty(
            0, device=lora_a.device, dtype=lora_a.dtype
        )
        lora_a.ds_status = type(lora_a.ds_status).NOT_AVAILABLE
        if tuple(lora_a.shape) != (0,):
            raise AssertionError(
                "scaled resume diagnostic failed to partition the DoRA LoRA A weight"
            )

    magnitude.register_forward_pre_hook(
        partition_before_deepspeed, with_kwargs=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "resume"), required=True)
    parser.add_argument("--use-reentrant", action=argparse.BooleanOptionalAction)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    deepspeed.init_distributed(dist_backend="nccl")
    torch.manual_seed(19)
    model = _model(use_reentrant=args.use_reentrant)
    if args.phase == "resume":
        _install_scaled_resume_partition(model)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=None,
        config=_deepspeed_config(),
    )
    validate_stage1_trainable_boundary(
        engine.module,
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
    dora_trace = _install_dora_trace(engine.module)
    if args.phase == "train":
        loss, gradient_norm = _backward_step(engine)
        optimizer_state_norm = float(
            _optimizer_state(engine.module).float().norm().detach().cpu()
        )
        if optimizer_state_norm <= 0:
            raise AssertionError("initial Adam exp_avg state must be nonzero")
        if engine.global_steps != 1:
            raise AssertionError(
                f"initial global step is {engine.global_steps}, expected 1"
            )
        if not engine.save_checkpoint(
            args.checkpoint,
            tag="step-1",
            client_state={
                "diagnostic": "stage1-dora-zero3-resume",
                "optimizer_state_norm": optimizer_state_norm,
            },
            save_latest=False,
        ):
            raise AssertionError("DeepSpeed checkpoint save failed")
    else:
        load_path, client_state = engine.load_checkpoint(
            args.checkpoint,
            tag="step-1",
            load_module_strict=True,
            load_optimizer_states=True,
        )
        if (
            load_path is None
            or client_state.get("diagnostic") != "stage1-dora-zero3-resume"
        ):
            raise AssertionError("DeepSpeed checkpoint load did not restore client state")
        if engine.global_steps != 1:
            raise AssertionError(
                f"restored global step is {engine.global_steps}, expected 1"
            )
        optimizer_state_norm = float(
            _optimizer_state(engine.module).float().norm().detach().cpu()
        )
        if optimizer_state_norm != client_state["optimizer_state_norm"]:
            raise AssertionError(
                "DeepSpeed checkpoint did not restore Adam exp_avg exactly"
            )
        loss, gradient_norm = _backward_step(engine)
        if engine.global_steps != 2:
            raise AssertionError(
                f"resumed global step is {engine.global_steps}, expected 2"
            )

    if int(os.environ["RANK"]) == 0:
        args.result.write_text(
            json.dumps(
                {
                    "phase": args.phase,
                    "use_reentrant": args.use_reentrant,
                    "loss": loss,
                    "gradient_norm": gradient_norm,
                    "global_steps": engine.global_steps,
                    "optimizer_state_norm": optimizer_state_norm,
                    "dora_trace": dora_trace,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
