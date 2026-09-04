"""Numerical witnesses for naive and tree-packed Stage 2B training."""

from __future__ import annotations

import copy
import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import torch

from stage2.trainer import Stage2Trainer
from step_preference.config import StepPreferenceConfig
from step_preference.negatives import StepNegative
from step_preference.spans import ReasoningStepSpan
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


_LOSS_NAMES = (
    "loss/total",
    "loss/ce",
    "loss/dpo_visual",
    "loss/dpo_text",
    "loss/entropy_visual",
    "loss/entropy_text",
)


@dataclass(frozen=True)
class Comparison:
    passed: bool
    errors: Mapping[str, str]
    absolute_errors: Mapping[str, float]
    relative_l2: Mapping[str, float]


@dataclass(frozen=True)
class EquivalenceResult:
    status: str
    loss_errors: Mapping[str, float]
    gradients: Mapping[str, float]
    trainable_parameters: tuple[str, ...]
    epsilon_loss: float
    epsilon_grad: float
    errors: tuple[str, ...]

    @property
    def max_loss_error(self) -> float:
        return max(self.loss_errors.values(), default=0.0)

    @property
    def max_gradient_relative_l2(self) -> float:
        return max(self.gradients.values(), default=0.0)


@dataclass(frozen=True)
class ProductionContext:
    model: object
    batch: Mapping
    naive_config: StepPreferenceConfig
    packed_config: StepPreferenceConfig
    identity: Mapping[str, object]


def relative_l2_error(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    epsilon: float = 1e-12,
) -> float:
    """Return ``||candidate-reference|| / (||reference|| + epsilon)`` in FP64."""
    ref = reference.detach().double().cpu()
    cand = candidate.detach().double().cpu()
    return float(
        torch.linalg.vector_norm(cand - ref)
        / (torch.linalg.vector_norm(ref) + epsilon)
    )


def compare_scalar_metrics(
    reference: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    epsilon: float,
) -> Comparison:
    errors = {}
    absolute_errors = {}
    for name in sorted(set(reference) | set(candidate)):
        if name not in reference or name not in candidate:
            errors[name] = "missing scalar"
            absolute_errors[name] = math.inf
            continue
        left = float(reference[name])
        right = float(candidate[name])
        if not math.isfinite(left) or not math.isfinite(right):
            errors[name] = "nonfinite scalar"
            absolute_errors[name] = math.inf
            continue
        difference = abs(right - left)
        absolute_errors[name] = difference
        if difference > epsilon:
            errors[name] = f"absolute error {difference} exceeds {epsilon}"
    return Comparison(not errors, errors, absolute_errors, {})


def compare_gradients(
    reference: Mapping[str, torch.Tensor | None],
    candidate: Mapping[str, torch.Tensor | None],
    *,
    epsilon: float,
) -> Comparison:
    errors = {}
    relative = {}
    for name in sorted(set(reference) | set(candidate)):
        left = reference.get(name)
        right = candidate.get(name)
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            errors[name] = "missing gradient"
            relative[name] = math.inf
            continue
        if left.shape != right.shape:
            errors[name] = "gradient shape mismatch"
            relative[name] = math.inf
            continue
        left64 = left.detach().double().cpu()
        right64 = right.detach().double().cpu()
        if not bool(torch.isfinite(left64).all()) or not bool(
            torch.isfinite(right64).all()
        ):
            errors[name] = "nonfinite gradient"
            relative[name] = math.inf
            continue
        reference_norm = float(torch.linalg.vector_norm(left64))
        difference_norm = float(torch.linalg.vector_norm(right64 - left64))
        if reference_norm < 1e-12:
            value = 0.0 if difference_norm <= epsilon else math.inf
        else:
            value = relative_l2_error(left64, right64)
        relative[name] = value
        if not math.isfinite(value) or value > epsilon:
            errors[name] = f"gradient error {value} exceeds {epsilon}"
    return Comparison(not errors, errors, {}, relative)


def write_evidence_exclusive(path: str | Path, payload: Mapping) -> None:
    """Write one canonical JSON witness without overwriting prior evidence."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise ValueError(f"evidence path already exists: {target}") from error


def _trainer(config: StepPreferenceConfig) -> Stage2Trainer:
    trainer = object.__new__(Stage2Trainer)
    trainer.stage2_diagnostics_interval = 0
    trainer._stage2_microbatch_index = 0
    trainer.step_preference_config = config
    trainer._step_preference_metric_totals = {}
    trainer._step_preference_metric_batches = 0
    return trainer


def _config(*, packing: bool) -> StepPreferenceConfig:
    return StepPreferenceConfig(
        enabled=True,
        mode="stage2b",
        negative_path="tiny-negatives.jsonl",
        negative_manifest="tiny-manifest.json",
        stage2a_checkpoint_path="tiny-stage2a",
        attention_implementation="eager",
        packing_enabled=packing,
    )


def _tiny_model() -> Qwen2_5_VLForConditionalGeneration:
    config = Qwen2_5_VLConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 2,
            "out_hidden_size": 64,
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
        attn_implementation="eager",
        attention_dropout=0.0,
        rope_scaling={"type": "mrope", "mrope_section": [4, 2, 2]},
    )
    model = Qwen2_5_VLForConditionalGeneration(config)
    model.visual.requires_grad_(False)
    return model


def _tiny_batch(device: torch.device) -> dict:
    clean_ids = torch.tensor(
        [[11, 91, 90, 90, 92, 12, 13, 14, 15]], device=device
    )
    rejected_ids = torch.tensor(
        [[11, 91, 90, 90, 92, 12, 23, 24, 15]], device=device
    )
    span = ReasoningStepSpan(0, 0, 1, 2, 4, 6, 8, None)
    negative = StepNegative(
        sample_id="tiny-equivalence",
        step_id=0,
        chosen_text="chosen",
        rejected_visual_text="rejected",
        rejected_textual_text=None,
        valid_visual_negative=True,
        valid_textual_negative=False,
        reference_chosen_score=-0.1,
        reference_visual_rejected_score=-0.2,
        reference_textual_rejected_score=None,
    )
    return {
        "input_ids": clean_ids,
        "attention_mask": torch.ones_like(clean_ids),
        "labels": clean_ids.clone(),
        "sample_ids": (negative.sample_id,),
        "reasoning_step_spans": (span,),
        "step_preference": {
            "steps": (
                {
                    "span": span,
                    "negative": negative,
                    "visual_rejected": {
                        "model_inputs": {
                            "input_ids": rejected_ids,
                            "attention_mask": torch.ones_like(rejected_ids),
                        },
                        "completion_token_positions": (6, 7),
                        "completion_token_ids": (23, 24),
                        "rendered_text": "tiny rejected",
                    },
                    "textual_rejected": None,
                },
            )
        },
    }


def _tiny_multistep_batch(device: torch.device) -> dict:
    sample_id = "tiny-multistep-equivalence"
    clean_ids = torch.tensor(
        [[11, 91, 90, 90, 92, 12, 13, 14, 91, 90, 90, 92, 15, 16, 17]],
        device=device,
    )
    spans = (
        ReasoningStepSpan(0, 0, 1, 2, 4, 6, 8, None),
        ReasoningStepSpan(1, 6, 8, 9, 11, 12, 15, None),
    )

    def negative(step_id: int, *, textual: bool) -> StepNegative:
        return StepNegative(
            sample_id=sample_id,
            step_id=step_id,
            chosen_text="chosen",
            rejected_visual_text="visual rejected",
            rejected_textual_text="text rejected" if textual else None,
            valid_visual_negative=True,
            valid_textual_negative=textual,
            reference_chosen_score=-0.1,
            reference_visual_rejected_score=-0.2,
            reference_textual_rejected_score=-0.3 if textual else None,
        )

    def encoded(ids: torch.Tensor, positions: tuple[int, ...]) -> dict:
        return {
            "model_inputs": {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
            },
            "completion_token_positions": positions,
            "completion_token_ids": tuple(int(ids[0, position]) for position in positions),
            "rendered_text": "tiny rejected",
        }

    step0_visual = clean_ids.clone()
    step0_visual[0, 6:8] = torch.tensor([23, 24], device=device)
    step0_textual = clean_ids.clone()
    step0_textual[0, 6:8] = torch.tensor([25, 26], device=device)
    step1_visual = clean_ids.clone()
    step1_visual[0, 12:15] = torch.tensor([27, 28, 29], device=device)
    negatives = (negative(0, textual=True), negative(1, textual=False))
    return {
        "input_ids": clean_ids,
        "attention_mask": torch.ones_like(clean_ids),
        "labels": clean_ids.clone(),
        "sample_ids": (sample_id,),
        "reasoning_step_spans": spans,
        "step_preference": {
            "steps": (
                {
                    "span": spans[0],
                    "negative": negatives[0],
                    "visual_rejected": encoded(step0_visual, (6, 7)),
                    "textual_rejected": encoded(step0_textual, (6, 7)),
                },
                {
                    "span": spans[1],
                    "negative": negatives[1],
                    "visual_rejected": encoded(step1_visual, (12, 13, 14)),
                    "textual_rejected": None,
                },
            )
        },
    }


def _copy_gradients(model, names: tuple[str, ...]) -> dict[str, torch.Tensor | None]:
    parameters = dict(model.named_parameters())
    return {
        name: (
            None
            if parameters[name].grad is None
            else parameters[name].grad.detach().cpu().clone()
        )
        for name in names
    }


def _loss_metrics(trainer: Stage2Trainer) -> dict[str, float]:
    return {name: trainer._step_preference_metric_totals[name] for name in _LOSS_NAMES}


def _result(
    *,
    scalar_comparison: Comparison,
    gradient_comparison: Comparison,
    trainable_parameters: tuple[str, ...],
    epsilon_loss: float,
    epsilon_grad: float,
) -> EquivalenceResult:
    errors = tuple(
        [f"{name}: {message}" for name, message in scalar_comparison.errors.items()]
        + [f"{name}: {message}" for name, message in gradient_comparison.errors.items()]
    )
    return EquivalenceResult(
        status="pass" if not errors else "fail",
        loss_errors=scalar_comparison.absolute_errors,
        gradients=gradient_comparison.relative_l2,
        trainable_parameters=trainable_parameters,
        epsilon_loss=epsilon_loss,
        epsilon_grad=epsilon_grad,
        errors=errors,
    )


def _run_tiny_fp32_equivalence(
    batch_factory,
    *,
    device: str | torch.device,
    model_seed: int,
    path_seed: int,
) -> EquivalenceResult:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("tiny packed equivalence requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(model_seed)
    base = _tiny_model().to(device=device, dtype=torch.float32).eval()
    naive_model = copy.deepcopy(base)
    packed_model = copy.deepcopy(base)
    names = tuple(
        name for name, parameter in naive_model.named_parameters() if parameter.requires_grad
    )
    batch = batch_factory(device)

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(path_seed)
        naive_trainer = _trainer(_config(packing=False))
        naive_loss = naive_trainer.compute_loss(naive_model, copy.deepcopy(batch))
        naive_loss.backward()
        naive_gradients = _copy_gradients(naive_model, names)

        torch.manual_seed(path_seed)
        packed_trainer = _trainer(_config(packing=True))
        packed_loss = packed_trainer.compute_loss(packed_model, copy.deepcopy(batch))
        packed_loss.backward()
        packed_gradients = _copy_gradients(packed_model, names)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.use_deterministic_algorithms(previous_deterministic)

    scalars = compare_scalar_metrics(
        _loss_metrics(naive_trainer),
        _loss_metrics(packed_trainer),
        epsilon=1e-5,
    )
    gradients = compare_gradients(naive_gradients, packed_gradients, epsilon=1e-4)
    return _result(
        scalar_comparison=scalars,
        gradient_comparison=gradients,
        trainable_parameters=names,
        epsilon_loss=1e-5,
        epsilon_grad=1e-4,
    )


def run_tiny_fp32_equivalence(device: str | torch.device = "cuda:0") -> EquivalenceResult:
    """Compare every trainable gradient on the fixed one-step tiny tree."""
    return _run_tiny_fp32_equivalence(
        _tiny_batch, device=device, model_seed=101, path_seed=211
    )


def run_tiny_multistep_fp32_equivalence(
    device: str | torch.device = "cuda:0",
) -> EquivalenceResult:
    """Compare every trainable gradient on a ragged two-step tree."""
    return _run_tiny_fp32_equivalence(
        _tiny_multistep_batch, device=device, model_seed=103, path_seed=223
    )


def _production_parameter_names(model) -> tuple[str, ...]:
    named = sorted(model.named_parameters())
    selected = []
    for suffix in ("q_proj.weight", "o_proj.weight", "down_proj.weight"):
        matches = [
            name
            for name, parameter in named
            if parameter.requires_grad and ".layers." in name and name.endswith(suffix)
        ]
        if not matches:
            raise ValueError(f"missing trainable production parameter: {suffix}")
        selected.append(matches[0])
    return tuple(selected)


def _run_production_comparison(
    model,
    batch: Mapping,
    *,
    naive_config: StepPreferenceConfig,
    packed_config: StepPreferenceConfig,
    autocast_bf16: bool,
    epsilon_loss: float,
    epsilon_grad: float,
) -> EquivalenceResult:
    names = _production_parameter_names(model)
    rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_deterministic = torch.are_deterministic_algorithms_enabled()

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        model.zero_grad(set_to_none=True)
        naive_trainer = _trainer(naive_config)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if autocast_bf16
            else nullcontext()
        )
        with context:
            naive_loss = naive_trainer.compute_loss(model, copy.deepcopy(batch))
        naive_loss.backward()
        naive_gradients = _copy_gradients(model, names)

        model.zero_grad(set_to_none=True)
        torch.random.set_rng_state(rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        packed_trainer = _trainer(packed_config)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if autocast_bf16
            else nullcontext()
        )
        with context:
            packed_loss = packed_trainer.compute_loss(model, copy.deepcopy(batch))
        packed_loss.backward()
        packed_gradients = _copy_gradients(model, names)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.use_deterministic_algorithms(previous_deterministic)

    scalars = compare_scalar_metrics(
        _loss_metrics(naive_trainer),
        _loss_metrics(packed_trainer),
        epsilon=epsilon_loss,
    )
    gradients = compare_gradients(
        naive_gradients, packed_gradients, epsilon=epsilon_grad
    )
    return _result(
        scalar_comparison=scalars,
        gradient_comparison=gradients,
        trainable_parameters=names,
        epsilon_loss=epsilon_loss,
        epsilon_grad=epsilon_grad,
    )


def run_production_fp32_equivalence(
    model,
    batch: Mapping,
    *,
    naive_config: StepPreferenceConfig,
    packed_config: StepPreferenceConfig,
) -> EquivalenceResult:
    """Strictly compare the authenticated production graph in FP32."""
    return _run_production_comparison(
        model,
        batch,
        naive_config=naive_config,
        packed_config=packed_config,
        autocast_bf16=False,
        epsilon_loss=1e-5,
        epsilon_grad=1e-4,
    )


def run_production_bf16_diagnostic(
    model,
    batch: Mapping,
    *,
    naive_config: StepPreferenceConfig,
    packed_config: StepPreferenceConfig,
) -> EquivalenceResult:
    """Record finite SDPA/Flex BF16 drift without fitting an acceptance bound."""
    return _run_production_comparison(
        model,
        batch,
        naive_config=naive_config,
        packed_config=packed_config,
        autocast_bf16=True,
        epsilon_loss=1e100,
        epsilon_grad=1e100,
    )


def run_production_bf16_equivalence(
    model,
    batch: Mapping,
    *,
    naive_config: StepPreferenceConfig,
    packed_config: StepPreferenceConfig,
) -> EquivalenceResult:
    """Backward-compatible name for the observational BF16 diagnostic."""
    return run_production_bf16_diagnostic(
        model,
        batch,
        naive_config=naive_config,
        packed_config=packed_config,
    )


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def load_production_context(
    *,
    checkpoint_path: str | Path,
    data_path: str | Path,
    image_root: str | Path,
    negative_path: str | Path,
    negative_manifest: str | Path,
    device: str | torch.device,
    tuning_method: str = "full",
    gradient_checkpointing: bool = False,
) -> ProductionContext:
    """Load one authenticated valid-pair sample through production assembly."""
    from main import build_stage2_training_components, preflight_stage2b_artifacts
    from step_preference.negatives import read_negative_manifest
    from utils_deepseed import parse_args

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("production packed equivalence requires CUDA")
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    source = Path(data_path).resolve(strict=True)
    images = Path(image_root).resolve(strict=True)
    negatives = Path(negative_path).resolve(strict=True)
    manifest_path = Path(negative_manifest).resolve(strict=True)
    manifest = read_negative_manifest(manifest_path)
    selected = manifest.selected_row_indices
    if not selected or selected != tuple(range(len(selected))):
        raise ValueError("production verifier requires contiguous selected rows from zero")
    args = parse_args(
        [
            "--stage",
            "stage2",
            "--load_model_path",
            str(checkpoint),
            "--data_path",
            str(source),
            "--image_root",
            str(images),
            "--step_preference_enabled",
            "--step_preference_mode",
            "stage2b",
            "--stage2a_checkpoint_path",
            str(checkpoint),
            "--step_negative_path",
            str(negatives),
            "--step_negative_manifest",
            str(manifest_path),
            "--max_train_samples",
            str(len(selected)),
            "--attention_implementation",
            "sdpa",
            "--tuning_method",
            tuning_method,
            (
                "--gradient_checkpointing"
                if gradient_checkpointing
                else "--no-gradient_checkpointing"
            ),
        ]
    )
    preflight = preflight_stage2b_artifacts(args)
    args._stage2b_preflight = preflight
    valid_sample_id = None
    for example in preflight.prepared_examples:
        rows = [
            preflight.negative_store[(example.sample_id, step.step_id)]
            for step in example.reasoning_steps
        ]
        if any(row.valid_visual_negative or row.valid_textual_negative for row in rows):
            valid_sample_id = example.sample_id
            break
    if valid_sample_id is None:
        raise ValueError("authenticated selected rows contain no valid preference pair")

    components = build_stage2_training_components(args)
    matches = [
        example
        for example in components.train_dataset
        if example.sample_id == valid_sample_id
    ]
    if len(matches) != 1:
        raise ValueError("lowest valid-pair sample was not retained exactly once")
    batch = components.data_collator(matches)
    if batch.get("sample_ids") != (valid_sample_id,):
        raise ValueError("collated sample identity does not match authenticated negatives")
    joined = batch.get("step_preference", {}).get("steps", ())
    if not joined or any(
        (valid_sample_id, item["span"].step_id) not in preflight.negative_store
        for item in joined
    ):
        raise ValueError("collated sample does not join the authenticated negative store")

    model = components.model
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    if hasattr(model.config, "attention_dropout"):
        model.config.attention_dropout = 0.0
    model.to(device)
    model.train()
    if getattr(model, "is_gradient_checkpointing", None) is not gradient_checkpointing:
        raise ValueError(
            "loaded model gradient checkpointing state does not match the request"
        )
    batch = _to_device(batch, device)
    naive_config = replace(
        components.step_preference_config,
        attention_implementation="sdpa",
        packing_enabled=False,
    )
    packed_config = replace(naive_config, packing_enabled=True)
    return ProductionContext(
        model=model,
        batch=batch,
        naive_config=naive_config,
        packed_config=packed_config,
        identity={
            "checkpoint_path": str(checkpoint),
            "sample_id": valid_sample_id,
            "negative_sha256": manifest.negative_sha256,
            "device": str(device),
            "dtype": "bfloat16",
            "tuning_method": tuning_method,
            "gradient_checkpointing": gradient_checkpointing,
        },
    )
