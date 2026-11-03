"""Validated lifecycle settings for step-wise Stage 2 preference training."""

import argparse
from dataclasses import dataclass
import math
from typing import Literal


Stage2Phase = Literal["standard", "stage2a", "stage2b"]
StepPreferenceExecutionBackend = Literal[
    "shared_prefix_3pass", "v_off_insertion", "single_image_latent_gate"
]
DEFAULT_STEP_PREFERENCE_EXECUTION_BACKEND: StepPreferenceExecutionBackend = (
    "shared_prefix_3pass"
)
STAGE2B_SIDECAR_PREFIX_SOURCE = "final_clean_forward"


@dataclass(frozen=True)
class Stage2Budget:
    total_steps: int
    stage2a_steps: int
    stage2b_steps: int


@dataclass(frozen=True)
class StepPreferenceConfig:
    enabled: bool
    beta: float = 0.1
    lambda_preference: float = 1.0
    lambda_entropy: float = 0.1
    entropy_margin: float = 0.1
    ablation: str = "attention_mask"
    modalities: str = "visual_text"
    preference_margin: float = 0.0
    execution_backend: StepPreferenceExecutionBackend = (
        DEFAULT_STEP_PREFERENCE_EXECUTION_BACKEND
    )
    target_scope: str = "until_next_latent"
    prefix_mode: str = "clean_prefix"
    branch_mode: str = "independent"
    on_layer_checkpoint_stride: int = 1  # Historical insertion only; gated keeps an inactive schema default.


def normalize_step_preference_execution_backend(
    value: object,
) -> StepPreferenceExecutionBackend:
    if value not in {
        DEFAULT_STEP_PREFERENCE_EXECUTION_BACKEND,
        "v_off_insertion",
        "single_image_latent_gate",
    }:
        raise ValueError(
            "step preference execution backend must be shared_prefix_3pass or "
            "v_off_insertion, or single_image_latent_gate"
        )
    return value


def preference_prefix_source(prefix_mode: str) -> str:
    if prefix_mode == "clean_prefix":
        return "final_clean_forward"
    if prefix_mode == "v_off_insertion":
        return "off_forward"
    if prefix_mode == "single_image_latent_gate":
        return "on_trajectory"
    raise ValueError("unsupported preference prefix mode")


def validate_insertion_runtime(
    *,
    preference: StepPreferenceConfig,
    microbatch_size: int,
    tuning_method: str,
    zero_stage: int | None,
) -> None:
    if preference.prefix_mode != "v_off_insertion":
        return
    if type(microbatch_size) is not int or microbatch_size != 1:
        raise ValueError("v_off_insertion requires microbatch 1")
    if zero_stage == 3 and tuning_method in {"lora", "dora"}:
        raise ValueError(
            "insertion branch checkpointing with PEFT and ZeRO-3 is unsupported"
        )


def validate_preference_rollout(
    *,
    preference: StepPreferenceConfig,
    rollout_backend: str,
    native_checkpointing: bool,
    latent_checkpointing: bool,
    cache_checkpointing: bool,
    microbatch_size: int,
    tuning_method: str,
    zero_stage: int | None,
    on_layer_checkpoint_stride: int,
) -> None:
    """Validate the effective preference backend after source inheritance."""

    if rollout_backend not in {"no_cache", "differentiable_cache"}:
        raise ValueError("unsupported Stage 2 rollout backend")
    if preference.prefix_mode == "single_image_latent_gate" and not preference.enabled:
        raise ValueError("single_image_latent_gate requires enabled Stage 2B preference")
    if preference.prefix_mode == "single_image_latent_gate" and native_checkpointing:
        raise ValueError("single_image_latent_gate requires native checkpointing disabled")
    if (
        preference.enabled
        and preference.prefix_mode != "single_image_latent_gate"
        and rollout_backend == "differentiable_cache"
    ):
        raise ValueError(
            f"differentiable cache is not integrated with {preference.prefix_mode} preference"
        )
    if rollout_backend == "differentiable_cache":
        if native_checkpointing:
            raise ValueError(
                "differentiable cache requires native checkpointing disabled"
            )
        if latent_checkpointing:
            raise ValueError(
                "differentiable cache requires latent checkpointing disabled"
            )
    else:
        if cache_checkpointing:
            raise ValueError("no_cache rejects cache checkpointing")
        if native_checkpointing and latent_checkpointing:
            raise ValueError(
                "no_cache may use native or latent checkpointing, never both"
            )

    if not preference.enabled:
        return
    if preference.prefix_mode == "single_image_latent_gate":
        if preference.modalities != "visual" or preference.ablation not in {
            "attention_mask",
            "remove_full_visual_block",
            "remove_full_latent_block_keep_image",
        }:
            raise ValueError(
                "single_image_latent_gate requires visual and a supported ablation"
            )
        if preference.execution_backend != "single_image_latent_gate":
            raise ValueError(
                "single_image_latent_gate requires its matching execution backend"
            )
        if (
            on_layer_checkpoint_stride != 1
            or preference.on_layer_checkpoint_stride != 1
        ):
            raise ValueError("ON layer checkpoint stride is only supported for v_off_insertion")
        if type(microbatch_size) is not int or microbatch_size != 1:
            raise ValueError("single_image_latent_gate requires microbatch 1")
        if zero_stage == 3 and tuning_method in {"lora", "dora"}:
            raise ValueError(
                "gated branch checkpointing with PEFT and ZeRO-3 is unsupported"
            )
        return
    validate_insertion_runtime(
        preference=preference,
        microbatch_size=microbatch_size,
        tuning_method=tuning_method,
        zero_stage=zero_stage,
    )


def split_stage2_optimizer_budget(total_steps: int, stage2a_ratio: float) -> Stage2Budget:
    if type(total_steps) is not int or total_steps <= 1:
        raise ValueError("stage2_total_optimizer_steps must be an integer greater than one")
    if not math.isfinite(stage2a_ratio) or not 0.0 < stage2a_ratio < 1.0:
        raise ValueError("stage2a_ratio must be finite and strictly between zero and one")
    stage2a_steps = math.floor(total_steps * stage2a_ratio)
    stage2b_steps = total_steps - stage2a_steps
    if stage2a_steps <= 0 or stage2b_steps <= 0:
        raise ValueError("both Stage 2A and Stage 2B optimizer budgets must be positive")
    return Stage2Budget(total_steps, stage2a_steps, stage2b_steps)


def selected_phase_steps(
    phase: Stage2Phase,
    total_steps: int | None,
    stage2a_ratio: float | None,
) -> int | None:
    if phase == "standard":
        return None
    if phase == "stage2b" and type(stage2a_ratio) in (int, float) and stage2a_ratio == 0:
        if type(total_steps) is not int or total_steps <= 1:
            raise ValueError("direct preference total steps must be greater than one")
        return total_steps
    if total_steps is None or stage2a_ratio is None:
        raise ValueError("Stage 2A/2B require total steps and stage2a ratio")
    budget = split_stage2_optimizer_budget(total_steps, stage2a_ratio)
    return budget.stage2a_steps if phase == "stage2a" else budget.stage2b_steps


def resolve_step_preference_config(
    *,
    phase: str,
    attention_implementation: str,
    stage2a_checkpoint_path: str | None,
    step_negatives_path: str | None,
    beta: float,
    lambda_preference: float,
    lambda_entropy: float,
    entropy_margin: float,
    preference_margin: float = 0.0,
    ablation: str = "attention_mask",
    modalities: str = "visual_text",
    target_scope: str = "until_next_latent",
    prefix_mode: str = "clean_prefix",
    branch_mode: str = "independent",
    on_layer_checkpoint_stride: int = 1,
) -> StepPreferenceConfig:
    if target_scope not in {"until_next_latent", "immediate_text"}:
        raise ValueError("target_scope must be until_next_latent or immediate_text")
    if modalities not in {"visual", "visual_text"}:
        raise ValueError("modalities must be visual or visual_text")
    if ablation not in {
        "attention_mask",
        "remove_full_visual_block",
        "remove_full_latent_block_keep_image",
    }:
        raise ValueError("unsupported step preference ablation")
    if phase not in {"standard", "stage2a", "stage2b"}:
        raise ValueError("stage2_phase must be standard, stage2a, or stage2b")
    if prefix_mode not in {
        "clean_prefix",
        "v_off_insertion",
        "single_image_latent_gate",
    }:
        raise ValueError(
            "prefix_mode must be clean_prefix, v_off_insertion, or "
            "single_image_latent_gate"
        )
    if branch_mode not in {"independent", "cumulative"}:
        raise ValueError("branch_mode must be independent or cumulative")
    if type(on_layer_checkpoint_stride) is not int or on_layer_checkpoint_stride not in {1, 2, 4}:
        raise ValueError("on_layer_checkpoint_stride must be 1, 2, or 4")
    if prefix_mode == "v_off_insertion":
        if phase != "stage2b" or modalities != "visual" or ablation != "attention_mask":
            raise ValueError("v_off_insertion requires stage2b, visual and attention_mask")
        execution_backend = "v_off_insertion"
    elif prefix_mode == "single_image_latent_gate":
        if (
            phase != "stage2b"
            or modalities != "visual"
            or ablation not in {
                "attention_mask",
                "remove_full_visual_block",
                "remove_full_latent_block_keep_image",
            }
        ):
            raise ValueError(
                "single_image_latent_gate requires stage2b, visual and a supported ablation"
            )
        if on_layer_checkpoint_stride != 1:
            raise ValueError(
                "ON layer checkpoint stride is only supported for v_off_insertion"
            )
        execution_backend = "single_image_latent_gate"
    elif branch_mode != "independent" or on_layer_checkpoint_stride != 1:
        raise ValueError("branch mode and ON stride require v_off_insertion")
    else:
        execution_backend = DEFAULT_STEP_PREFERENCE_EXECUTION_BACKEND
    values = {
        "beta": beta,
        "lambda_preference": lambda_preference,
        "lambda_entropy": lambda_entropy,
        "entropy_margin": entropy_margin,
        "preference_margin": preference_margin,
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("step preference scalars must be finite")
    if beta <= 0.0:
        raise ValueError("step_preference_beta must be positive")
    if min(lambda_preference, lambda_entropy, entropy_margin, preference_margin) < 0.0:
        raise ValueError("step preference weights and margin must be non-negative")
    if phase != "stage2b":
        if stage2a_checkpoint_path is not None:
            raise ValueError(
                "stage2a_checkpoint_path is only supported for stage2_phase=stage2b"
            )
        return StepPreferenceConfig(
            enabled=False,
            beta=beta,
            lambda_preference=lambda_preference,
            lambda_entropy=lambda_entropy,
            entropy_margin=entropy_margin,
            preference_margin=preference_margin,
            ablation=ablation,
            modalities=modalities,
            target_scope=target_scope,
            execution_backend=execution_backend,
            prefix_mode=prefix_mode,
            branch_mode=branch_mode,
            on_layer_checkpoint_stride=on_layer_checkpoint_stride,
        )
    if attention_implementation not in {"eager", "sdpa", "flex_attention"}:
        raise ValueError("Stage 2B requires eager, sdpa, or flex_attention")
    if stage2a_checkpoint_path is not None and (
        not isinstance(stage2a_checkpoint_path, str)
        or not stage2a_checkpoint_path.strip()
    ):
        raise ValueError("Stage 2B requires Stage 2A checkpoint path")
    if step_negatives_path is not None:
        raise ValueError("context preference Stage 2B does not accept step_negatives_path")
    return StepPreferenceConfig(
        enabled=True,
        beta=beta,
        lambda_preference=lambda_preference,
        lambda_entropy=lambda_entropy,
        entropy_margin=entropy_margin,
        preference_margin=preference_margin,
        ablation=ablation,
        modalities=modalities,
        target_scope=target_scope,
        execution_backend=execution_backend,
        prefix_mode=prefix_mode,
        branch_mode=branch_mode,
        on_layer_checkpoint_stride=on_layer_checkpoint_stride,
    )


def _parse_phase_steps_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    phase_steps = subparsers.add_parser("phase-steps")
    phase_steps.add_argument("--phase", choices=("standard", "stage2a", "stage2b"), required=True)
    phase_steps.add_argument("--total-steps", type=int, required=True)
    phase_steps.add_argument("--stage2a-ratio", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_phase_steps_args()
    try:
        selected_steps = selected_phase_steps(
            args.phase, args.total_steps, args.stage2a_ratio
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if selected_steps is None:
        raise SystemExit("phase-steps only accepts Stage 2A or Stage 2B")
    print(selected_steps)


if __name__ == "__main__":
    main()
