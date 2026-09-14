"""Local F/reference/M execution; distributed reduction belongs to the Trainer."""

from dataclasses import dataclass, fields
from functools import partial
import os

import torch
from torch.utils.checkpoint import checkpoint

from peft_common import resolve_qwen2_5_vl_model
from .objective import Stage2ObjectiveConfig, entropy_transition, text_entropy, target_log_likelihood, vision_gain
from .objective_windows import select_objective_windows, WindowSelection
from .rng import capture_rng_state, replay_rng_state


def objective_config_from_args(args):
    defaults = Stage2ObjectiveConfig()
    config = Stage2ObjectiveConfig(**{
        field.name: getattr(args, field.name, getattr(defaults, field.name))
        for field in fields(defaults)
    })
    config.validate(getattr(args, "stage", "stage2"), getattr(args, "supervise_latent_boundaries", True))
    if config.active:
        validate_objective_backend(args)
    return config


def validate_objective_backend(settings):
    """Fail closed for execution combinations without a paired-forward witness."""
    if (getattr(settings, "fp16", False)
            or os.environ.get("ACCELERATE_MIXED_PRECISION", "").lower() == "fp16"):
        raise ValueError("active Stage 2 objective does not support FP16; use FP32 or BF16")
    backend = getattr(settings, "attention_implementation", getattr(settings, "_attn_implementation", "eager"))
    if backend not in ("eager", "sdpa"):
        raise ValueError("Stage 2 objective backend must be eager or sdpa")
    if any(getattr(settings, key, None) not in (None, False) for key in ("deepspeed", "deepspeed_config")) or os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() in ("true", "1", "yes"):
        raise ValueError("active Stage 2 objective does not support DeepSpeed")
    if getattr(settings, "use_sliding_window", False):
        raise ValueError("active Stage 2 objective does not support sliding window")
    if getattr(settings, "attention_dropout", 0.0) != 0.0:
        raise ValueError("active Stage 2 objective requires zero native attention dropout")
    if getattr(settings, "gradient_checkpointing", False):
        checkpointing = getattr(settings, "gradient_checkpointing_kwargs", {"use_reentrant": False})
        if checkpointing != {"use_reentrant": False}:
            raise ValueError("active Stage 2 objective requires native non-reentrant gradient checkpointing")
    accelerator = getattr(settings, "accelerator_config", None)
    if accelerator is not None and getattr(accelerator, "gradient_accumulation_kwargs", None):
        raise ValueError("active Stage 2 objective does not support accumulation plugins")
    if (getattr(settings, "fsdp", None) or getattr(settings, "torch_compile", False)
            or os.environ.get("ACCELERATE_USE_FSDP", "").lower() in ("true", "1", "yes")):
        raise ValueError("active Stage 2 objective does not support FSDP/compiled plugins")


def model_inputs(inputs):
    return {key: value for key, value in inputs.items()
            if key != "stage2_text_segments" and not key.startswith("_stage2_objective_")}


def validate_objective_student(model):
    base = resolve_qwen2_5_vl_model(model)
    validate_objective_backend(base.config)
    if any(parameter.dtype == torch.float16 for parameter in model.parameters()):
        raise ValueError("active Stage 2 objective does not support FP16 parameters; use FP32 or BF16")
    if base.model.gradient_checkpointing:
        function = getattr(base.model, "_gradient_checkpointing_func", None)
        if not (isinstance(function, partial) and function.func is checkpoint
                and function.keywords == {"use_reentrant": False}):
            raise ValueError("active Stage 2 objective requires the validated native non-reentrant checkpoint function")


def counted_forward(model, inputs, *, student=None, **kwargs):
    """Observe decoder invocations only during this forward, never recomputation."""
    decoder = resolve_qwen2_5_vl_model(student if student is not None else model).model
    calls = 0

    def observed(module, args):
        nonlocal calls
        calls += 1

    handle = decoder.register_forward_pre_hook(observed)
    try:
        output = model(**inputs, **kwargs)
    finally:
        handle.remove()
    return output, calls


@dataclass
class ObjectiveBatchResult:
    outputs: object
    ce: torch.Tensor
    dyn_sum: torch.Tensor
    vis_sum: torch.Tensor
    count: int
    metric_sums: dict[str, torch.Tensor | float]
    metric_counts: dict[str, int]


class Stage2ObjectiveRuntime:
    def __init__(self, config, token_ids, latent_size, text_vocabulary, reference):
        self.config = config
        self.token_ids = token_ids
        self.latent_size = latent_size
        self.text_vocabulary = text_vocabulary
        self.reference = reference
        if config.lambda_dyn and reference is None:
            raise ValueError("entropy objective requires a frozen reference")

    def select(self, inputs) -> WindowSelection:
        return select_objective_windows(
            inputs["input_ids"], inputs["labels"], inputs["attention_mask"],
            inputs["stage2_text_segments"], token_ids=self.token_ids,
            latent_size=self.latent_size, text_vocabulary=self.text_vocabulary,
            pre_window=self.config.pre_window, post_window=self.config.post_window,
        )

    def forward(self, model, inputs, selection, *, counterfactual_model=None):
        current_student = counterfactual_model if counterfactual_model is not None else model
        validate_objective_student(current_student)
        clean = model_inputs(inputs)
        rng = capture_rng_state()
        outputs, decoder_calls = counted_forward(model, clean, student=current_student,
                                                 collect_stage2_diagnostics=True, stage2_finite_ce=True)
        zero = outputs.loss * 0.0
        result = ObjectiveBatchResult(outputs, outputs.loss, zero, zero, len(selection.blocks), {}, {})

        def add(name, value, count=1):
            detached = value.detach() if isinstance(value, torch.Tensor) else value
            result.metric_sums[name] = result.metric_sums.get(name, 0.0) + detached
            result.metric_counts[name] = result.metric_counts.get(name, 0) + count

        add("loss_ce", outputs.loss)
        add("model_forwards_F", 1, 0)
        add("decoder_forwards_F", decoder_calls, 0)
        add("valid_blocks", result.count, 0)
        add("excluded_blocks", int(outputs.stage2_diagnostics["block_count"].item()) - result.count, 0)
        for reason, count in selection.excluded.items():
            add("excluded_" + reason, count, 0)
        if not selection.blocks:
            return result

        auxiliary_inputs = {key: value for key, value in clean.items() if key != "labels"}
        ref_outputs = None
        if self.config.lambda_dyn:
            with torch.no_grad():
                ref_outputs, calls = counted_forward(self.reference.model, auxiliary_inputs)
            add("model_forwards_reference", 1, 0)
            add("decoder_forwards_reference", calls, 0)

        for block in selection.blocks:
            # Select sequence rows BEFORE FP32 entropy / likelihood operations.
            post_logits = outputs.logits[block.batch_index, list(block.post_positions)]
            try:
                if self.config.lambda_dyn:
                    pre = text_entropy(outputs.logits[block.batch_index, list(block.pre_positions)], self.text_vocabulary.token_ids).mean()
                    post = text_entropy(post_logits, self.text_vocabulary.token_ids).mean()
                    ref_pre_tokens = text_entropy(ref_outputs.logits[block.batch_index, list(block.pre_positions)], self.text_vocabulary.token_ids)
                    ref_post_tokens = text_entropy(ref_outputs.logits[block.batch_index, list(block.post_positions)], self.text_vocabulary.token_ids)
                    ref_pre, ref_post = ref_pre_tokens.mean(), ref_post_tokens.mean()
                    metrics = entropy_transition(pre, post, ref_pre, ref_post, eps=self.config.entropy_eps, gamma_H=self.config.gamma_H)
                    result.dyn_sum = result.dyn_sum + metrics["loss_dyn"]
                    for name, value in dict(H_pre=pre, H_post=post, H_ref_pre=ref_pre, H_ref_post=ref_post, **metrics).items():
                        add(name, value)
                    add("r_pre_gt_1", (metrics["r_pre"] > 1).float())
                    add("r_post_lt_1", (metrics["r_post"] < 1).float())
                    add("low_reference_entropy", (ref_pre_tokens < self.config.entropy_eps).sum() + (ref_post_tokens < self.config.entropy_eps).sum(), 0)
                if self.config.lambda_vis:
                    targets = torch.tensor(block.target_ids, device=post_logits.device)
                    score_f = target_log_likelihood(post_logits, targets).mean()
                    with torch.no_grad(), replay_rng_state(rng):
                        masked, calls = counted_forward(current_student, auxiliary_inputs,
                                                         stage2_ablation=(block.start_position, block.end_position))
                        score_m = target_log_likelihood(masked.logits[block.batch_index, list(block.post_positions)], targets).mean()
                    del masked
                    metrics = vision_gain(score_f, score_m.detach(), m_V=self.config.m_V)
                    result.vis_sum = result.vis_sum + metrics["loss_vis"]
                    for name, value in metrics.items():
                        add(name, value)
                    add("G_gt_0", (metrics["G"] > 0).float())
                    add("model_forwards_M", 1, 0)
                    add("decoder_forwards_M", calls, 0)
            except ValueError as error:
                raise ValueError(
                    f"Stage 2 objective batch_index={block.batch_index}, block_index={block.block_index}, "
                    f"start_position={block.start_position}, end_position={block.end_position}, "
                    f"pre_positions={block.pre_positions}, post_positions={block.post_positions}: {error}"
                ) from error
        return result
