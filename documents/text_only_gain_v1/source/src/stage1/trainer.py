"""True-batched Stage 1 loss and EMA lifecycle on top of ``SFTTrainer``."""

from __future__ import annotations

from contextlib import nullcontext
import math
import logging
import os
from importlib import metadata as importlib_metadata
from pathlib import Path
import tempfile
import weakref
from typing import Callable, Mapping

import torch
from torch import nn
import torch.nn.functional as F
from transformers import TrainerCallback
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from trl import SFTTrainer

from deepspeed_runtime import (
    DeepSpeedRuntimeProfile,
    validate_deepspeed_resume_identity,
    write_deepspeed_runtime_sidecar,
)
from .ema import (
    STAGE1_EMA_TEACHER_STATE_FILENAME,
    Stage1EmaTeacher,
    _decode_ema_state,
)
from .targets import (
    STAGE1_TEACHER_TARGET_MODE,
    STAGE1_TRAINING_IMPLEMENTATION_VERSION,
    Stage1TeacherTargetBuilder,
)
from .types import Stage1BatchMetadata, Stage1TeacherTargets
from .iier import Stage1PreferenceConfig, build_iier_latents, iier_preference_loss
from .clean_kl import CleanKLConfig, CleanKLAccumulator
from .text_kl import TextKLConfig, TextKLAccumulator


_HELPER_INPUTS = frozenset(
    {"pixel_values_latent", "image_grid_thw_latent"}
)
_SUPPORTED_ACCELERATE_VERSION = "1.10.1"
_SUPPORTED_DEEPSPEED_VERSION = "0.18.7"
_STAGE1_EMA_TEACHER_STATE_SCHEMA_VERSION = 2
_STAGE1_EMA_TEACHER_STATE_KIND = "stage1_ema_teacher"


def collect_deepspeed_pre_step_norm(
    optimizer: object, *, zero_stage: int
) -> torch.Tensor:
    """Collect the pinned optimizer's unscaled norm before its step mutates state."""

    if zero_stage == 3:
        get_norm_groups = getattr(optimizer, "_get_norm_groups", None)
        if not callable(get_norm_groups):
            raise RuntimeError("ZeRO-3 gradient norm groups are unavailable")
        raw_groups = get_norm_groups()
        if not isinstance(raw_groups, (tuple, list)) or not raw_groups:
            raise FloatingPointError("ZeRO-3 gradient norm groups are missing")
        scalar_groups: list[torch.Tensor] = []
        for value in raw_groups:
            group = torch.as_tensor(value).detach().float()
            if (
                group.numel() != 1
                or not bool(torch.isfinite(group).all().item())
                or bool((group < 0).item())
            ):
                raise FloatingPointError(
                    "DeepSpeed pre-step gradient norm must be finite"
                )
            scalar_groups.append(group.reshape(()))
        reference_device = scalar_groups[0].device
        scaled_norm = torch.linalg.vector_norm(
            torch.stack(
                [group.to(device=reference_device) for group in scalar_groups]
            )
        )
    elif zero_stage in (1, 2):
        expected_partitioning = zero_stage == 2
        if bool(getattr(optimizer, "partition_gradients", None)) != (
            expected_partitioning
        ):
            raise RuntimeError(
                f"ZeRO-{zero_stage} optimizer partition contract mismatch"
            )
        scaled_global_norm = getattr(optimizer, "scaled_global_norm", None)
        if not callable(scaled_global_norm):
            raise RuntimeError(
                f"ZeRO-{zero_stage} scaled global norm is unavailable"
            )
        scaled_norm = torch.as_tensor(scaled_global_norm()).detach().float()
        if (
            scaled_norm.numel() != 1
            or not bool(torch.isfinite(scaled_norm).all().item())
            or bool((scaled_norm < 0).item())
        ):
            raise FloatingPointError(
                "DeepSpeed pre-step gradient norm must be finite"
            )
        scaled_norm = scaled_norm.reshape(())
        reference_device = scaled_norm.device
    else:
        raise RuntimeError(f"Unsupported DeepSpeed ZeRO stage: {zero_stage}")

    raw_loss_scale = getattr(optimizer, "loss_scale", None)
    if raw_loss_scale is None:
        raise FloatingPointError("DeepSpeed gradient loss scale is missing")
    loss_scale = torch.as_tensor(
        raw_loss_scale, dtype=torch.float32, device=reference_device
    )
    if (
        loss_scale.numel() != 1
        or not bool(torch.isfinite(loss_scale).all().item())
        or not bool((loss_scale > 0).item())
    ):
        raise FloatingPointError("DeepSpeed gradient loss scale is invalid")
    norm = torch.as_tensor(
        scaled_norm / loss_scale.reshape(()), dtype=torch.float32
    )
    if norm.numel() != 1 or not bool(torch.isfinite(norm).all().item()):
        raise FloatingPointError(
            "DeepSpeed pre-step gradient norm must be finite"
        )
    return norm.reshape(())


class _DeepSpeedStepAdapter:
    """Guard the real DeepSpeed mutation boundary and finish EMA there."""

    def __init__(
        self,
        trainer: "CustomTrainerStage1",
        engine,
        original_step: Callable,
    ) -> None:
        self._trainer = weakref.ref(trainer)
        self.engine = engine
        self.original_step = original_step

    def __call__(self, *args, **kwargs):
        trainer = self._trainer()
        if trainer is None:
            raise RuntimeError("Stage 1 trainer disappeared before DeepSpeed step")

        boundary = self.engine.is_gradient_accumulation_boundary()
        if type(boundary) is not bool:
            raise RuntimeError(
                "unsupported DeepSpeed Stage 1 lifecycle: accumulation "
                "boundary witness is not boolean"
            )
        if not boundary:
            return self.original_step(*args, **kwargs)

        trainer._validate_deepspeed_gradient_norm(self.engine)
        trainer._prepare_gain_step()
        result = self.original_step(*args, **kwargs)
        applied = self.engine.was_step_applied()
        if type(applied) is not bool:
            raise RuntimeError(
                "unsupported DeepSpeed Stage 1 lifecycle: applied-step "
                "witness is not boolean"
            )
        trainer._finish_optimizer_step(self.engine, successful=applied)
        return result


class _Stage1OptimizerCallback(TrainerCallback):
    """Delegate Trainer's real optimizer boundary back to Stage 1."""

    def __init__(self, trainer: "CustomTrainerStage1") -> None:
        self._trainer = weakref.ref(trainer)

    def on_train_begin(self, args, state, control, **kwargs):
        del args, state, kwargs
        trainer = self._trainer()
        if trainer is not None:
            # Trainer invokes this only after Accelerator/DeepSpeed prepare.
            trainer._prepare_ema_teacher_for_training()
            trainer._restore_pending_stage1_ema_checkpoint()
            trainer._install_deepspeed_step_adapter()
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        del args, state
        trainer = self._trainer()
        if trainer is not None and getattr(
            trainer, "_deepspeed_step_adapter", None
        ) is None:
            trainer._validate_gradient_norm(kwargs.get("model"))
            trainer._prepare_gain_step()
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        del args, state
        trainer = self._trainer()
        if trainer is not None and getattr(
            trainer, "_deepspeed_step_adapter", None
        ) is None:
            skipped = bool(
                getattr(trainer.accelerator, "optimizer_step_was_skipped", False)
            )
            trainer._finish_optimizer_step(
                kwargs.get("model"), successful=not skipped
            )
        return control


class CustomTrainerStage1(SFTTrainer):
    """Build EMA targets once and train one true student batch per loss call."""

    stage1_preference = Stage1PreferenceConfig()
    clean_kl_config = CleanKLConfig()
    text_kl_config = TextKLConfig()

    def __init__(
        self,
        *args,
        deepspeed_runtime_profile: DeepSpeedRuntimeProfile,
        sim_weight: float = 1.0,
        ema_tau: float = 0.999,
        ema_setting: str = "moving",
        coverage_p: float = 0.9,
        image_pool_k: int = 8,
        ce_weight: float = 1.0,
        helper_group_L: int | None = 256,
        ema_method: str | None = None,
        target_builder: Stage1TeacherTargetBuilder | None = None,
        ema_teacher: Stage1EmaTeacher | None = None,
        stage1_peft_metadata=None,
        collect_stage1_diagnostics: bool = False,
        vision_gain_config=None,
        prefix_conditioning_config: "PrefixConditioningConfig | None" = None,
        stage1_preference: Stage1PreferenceConfig | None = None,
        clean_kl_config: CleanKLConfig | None = None,
        text_kl_config: TextKLConfig | None = None,
        **kwargs,
    ) -> None:
        if not isinstance(deepspeed_runtime_profile, DeepSpeedRuntimeProfile):
            raise TypeError("Stage 1 trainer requires a DeepSpeed runtime profile")
        from .prefix_mask_config import PrefixConditioningConfig, validate_prefix_runtime, validate_prefix_model
        from .vision_gain import Stage1VisionGainConfig
        self.prefix_conditioning_config = prefix_conditioning_config or PrefixConditioningConfig()
        self.vision_gain_config = vision_gain_config or Stage1VisionGainConfig()
        self.clean_kl_config = clean_kl_config or CleanKLConfig()
        self.clean_kl_config.validate_runtime(self.prefix_conditioning_config, self.vision_gain_config)
        self.text_kl_config = text_kl_config or TextKLConfig()
        self.text_kl_config.validate_runtime(self.prefix_conditioning_config, self.vision_gain_config)
        initial_model = kwargs.get('model', args[0] if args else None)
        validate_prefix_runtime(self.prefix_conditioning_config, self.vision_gain_config,
            getattr(getattr(initial_model, 'config', None), '_attn_implementation', None))
        if self.prefix_conditioning_config.enabled and self.vision_gain_config.alpha:
            validate_prefix_model(initial_model)
        self._prefix_parameter_version = 0
        self._prefix_teacher_version = 0
        self._latest_prefix_report = None
        self._latest_prefix_ledger = None
        self._prefix_identity = None
        self._prefix_pending_scalars = {}
        self._prefix_update_region = None
        self.deepspeed_runtime_profile = deepspeed_runtime_profile
        self.stage1_peft_metadata = stage1_peft_metadata
        self.stage1_preference = stage1_preference or Stage1PreferenceConfig()
        if not isinstance(self.stage1_preference, Stage1PreferenceConfig):
            raise TypeError("Stage 1 preference requires Stage1PreferenceConfig")
        # Preserve the repository's legacy constructor spelling across the
        # tokenizer -> processing_class Trainer API migration.
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)

        self.sim_weight = self._finite_weight(sim_weight, "sim_weight")
        self.ce_weight = self._finite_weight(ce_weight, "ce_weight")
        if ema_setting not in {"moving", "frozen"}:
            raise ValueError("ema_setting must be moving or frozen")
        # Experiment naming contract: moving=existing implementation;
        # frozen=new fixed-teacher implementation.
        self.ema_setting = ema_setting
        self.coverage_p = float(coverage_p)
        self.image_pool_k = int(image_pool_k)
        self.helper_group_L = helper_group_L
        self.collect_stage1_diagnostics = bool(collect_stage1_diagnostics)
        self._latest_stage1_diagnostics = None
        self._representative_gradient_norm: float | None = None
        self._successful_optimizer_steps = 0
        self._ema_execution_device: torch.device | None = None
        self._deepspeed_step_adapter: _DeepSpeedStepAdapter | None = None
        self._stage1_pending_ema_checkpoint: Path | None = None
        self._defer_stage1_checkpoint_rotation = False
        self._stage1_deferred_checkpoint_rotation = None

        if target_builder is None:
            processing_class = getattr(self, "processing_class", None)
            if processing_class is None:
                processing_class = getattr(self, "tokenizer", None)
            if processing_class is None:
                raise ValueError(
                    "Stage 1 trainer requires a tokenizer/processing_class "
                    "or an explicit target_builder"
                )
            tokenizer = getattr(processing_class, "tokenizer", processing_class)
            if not callable(tokenizer):
                raise ValueError(
                    "Stage 1 processing_class must be a tokenizer or expose "
                    "a callable .tokenizer for text-only teacher targets"
                )
            target_builder = Stage1TeacherTargetBuilder(
                tokenizer,
                coverage_p=self.coverage_p,
                image_pool_k=self.image_pool_k,
                helper_group_L=self.helper_group_L,
            )
        self.target_builder = target_builder

        # SFTTrainer may apply PEFT in super().__init__.  The EMA snapshot must
        # therefore be constructed only now, from self.model's final topology.
        method = ema_method or self._infer_ema_method(self.model)
        self.ema_teacher = ema_teacher or Stage1EmaTeacher(
            self.model, decay=float(ema_tau), method=method
        )
        if self.prefix_conditioning_config.enabled and self.vision_gain_config.alpha:
            validate_prefix_model(self.model)
            validate_prefix_model(self.ema_teacher.model)
        if self.prefix_conditioning_config.enabled:
            import logging
            from .prefix_mask_accounting import prefix_source_identity, CANONICAL_RESULTS_ROOT
            from artifact_paths import CANONICAL_REPOSITORY_ROOT
            output = Path(self.args.output_dir).resolve()
            checkpoint_root = CANONICAL_REPOSITORY_ROOT / 'checkpoints'
            if not output.is_relative_to(checkpoint_root):
                raise ValueError('prefix run requires a canonical checkpoint experiment path')
            relative = output.relative_to(checkpoint_root)
            self._prefix_diagnostics_path = CANONICAL_RESULTS_ROOT / relative / f'prefix-rank-{self.args.process_index}.jsonl'
            from dataclasses import asdict
            from .prefix_mask_accounting import canonical_digest
            self._prefix_identity = prefix_source_identity(Path(__file__).resolve().parents[2], self.prefix_conditioning_config)
            self._prefix_identity['config_hash'] = canonical_digest({
                'prefix': self.prefix_conditioning_config.to_metadata(),
                'gain': asdict(self.vision_gain_config), 'preference': asdict(self.stage1_preference),
                'attention': self.model.config._attn_implementation,
                'ema_tau': ema_tau, 'ema_setting': ema_setting,
                'coverage_p': self.coverage_p, 'image_pool_k': self.image_pool_k,
                'helper_group_L': self.helper_group_L, 'ce_weight': self.ce_weight, 'sim_weight': self.sim_weight})
            if self.clean_kl_config.enabled:
                self._prefix_identity['config_hash'] = canonical_digest({
                    'prefix_runtime_config_hash': self._prefix_identity['config_hash'],
                    'clean_kl': self.clean_kl_config.to_dict()})
            if self.text_kl_config.enabled:
                self._prefix_identity['config_hash'] = canonical_digest({
                    'prefix_runtime_config_hash': self._prefix_identity['config_hash'],
                    'text_kl': self.text_kl_config.to_dict()})
            logging.info('Stage 1 prefix runtime identity/settings: %s %s',
                         self._prefix_identity, self.prefix_conditioning_config.to_metadata())
        self._install_trainable_gradient_coverage_hooks()
        self.add_callback(_Stage1OptimizerCallback(self))
        from .vision_gain import Stage1VisionGainConfig
        from .vision_gain_update import Stage1GainController, validate_gain_runtime
        self.vision_gain_config = vision_gain_config or Stage1VisionGainConfig()
        attention = getattr(self.model.config, "_attn_implementation", None)
        validate_gain_runtime(self.vision_gain_config, attention, self.deepspeed_runtime_profile,
                              fp16=self.args.fp16)
        self._vision_gain = (Stage1GainController(
            tuple(p for p in self.model.parameters() if p.requires_grad), self.vision_gain_config
        ) if self.vision_gain_config.alpha else None)


    def _install_trainable_gradient_coverage_hooks(self) -> None:
        """Record whether backward reaches every trainable tensor."""

        handles = getattr(self, "_gradient_coverage_handles", None)
        if handles:
            raise RuntimeError("Stage 1 gradient coverage hooks are already installed")
        self._gradient_coverage_handles = []
        self._gradient_coverage_expected = {}
        self._gradient_coverage_seen = set()
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            identity = id(parameter)
            self._gradient_coverage_expected[identity] = name

            def mark_gradient(gradient, *, parameter_id=identity):
                self._gradient_coverage_seen.add(parameter_id)
                return gradient

            self._gradient_coverage_handles.append(
                parameter.register_hook(mark_gradient)
            )
        if not self._gradient_coverage_expected:
            raise RuntimeError("Stage 1 model has no trainable parameters")

    def _validate_trainable_gradient_coverage(self) -> None:
        expected = getattr(self, "_gradient_coverage_expected", None)
        if not expected:
            return
        seen = getattr(self, "_gradient_coverage_seen", set())
        missing = sorted(name for identity, name in expected.items() if identity not in seen)
        if missing:
            preview = ", ".join(missing[:8])
            if len(missing) > 8:
                preview += f", ... ({len(missing)} total)"
            raise RuntimeError(
                "Stage 1 missing trainable gradients: " + preview
            )

    def _reset_trainable_gradient_coverage(self) -> None:
        seen = getattr(self, "_gradient_coverage_seen", None)
        if seen is not None:
            seen.clear()

    def _publish_deepspeed_runtime_identity(
        self, directory: Path, *, label: str
    ) -> None:
        """Publish on rank zero and propagate failure before synchronization."""

        local_error: Exception | None = None
        if self.args.should_save:
            try:
                write_deepspeed_runtime_sidecar(
                    directory, self.deepspeed_runtime_profile
                )
            except Exception as error:
                local_error = error
        local_failure = torch.tensor(
            int(local_error is not None),
            dtype=torch.int32,
            device=self.accelerator.device,
        )
        failed_processes = self.accelerator.reduce(
            local_failure, reduction="sum"
        )
        if int(failed_processes.item()) != 0:
            if local_error is not None:
                raise local_error
            raise RuntimeError(f"{label} failed on another process")
        self.accelerator.wait_for_everyone()

    @staticmethod
    def _finite_weight(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @staticmethod
    def _infer_ema_method(model: nn.Module) -> str:
        peft_config = getattr(model, "peft_config", None)
        if not peft_config:
            return "full"
        configs = (
            peft_config.values()
            if isinstance(peft_config, Mapping)
            else (peft_config,)
        )
        return (
            "dora"
            if any(bool(getattr(item, "use_dora", False)) for item in configs)
            else "lora"
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        config = getattr(self, "prefix_conditioning_config", None)
        if config is None or not config.enabled:
            return self._compute_loss(model, inputs, return_outputs, num_items_in_batch)
        from .prefix_mask_accounting import DecoderCallLedger, instrument_prefix_execution
        for name in ("_prefix_parameter_version", "_prefix_teacher_version"):
            value = getattr(self, name, None)
            if type(value) is not int or value < 0:
                raise ValueError(f"conditioning version not initialized: {name}")
        if self.vision_gain_config.execution != "packed":
            raise ValueError("prefix conditioning requires packed gain execution")
        if self.vision_gain_config.alpha:
            from .prefix_mask_config import validate_prefix_model
            lengths = [value.shape[1] for name in ('input_ids', 'inputs_embeds')
                       if isinstance((value := inputs.get(name)), torch.Tensor) and value.ndim >= 2]
            length = max(lengths) if lengths else None
            validate_prefix_model(model, sequence_length=length)
            validate_prefix_model(self.ema_teacher.model, sequence_length=length)
        self._flush_prefix_report()
        ledger = DecoderCallLedger()
        self._prefix_microbatch_index = getattr(self, "_prefix_microbatch_index", 0) + 1
        gain = getattr(self, "_vision_gain", None)
        report = {"parameter_version": self._prefix_parameter_version,
                  "teacher_version": self._prefix_teacher_version,
                  "microbatch_index": self._prefix_microbatch_index,
                  "accumulation_window_index": gain.completed_windows + 1 if gain is not None else "unmeasured"}
        self._latest_prefix_report = report
        self._latest_prefix_ledger = ledger
        try:
            with instrument_prefix_execution(model, self.ema_teacher.model, ledger):
                return self._compute_loss(model, inputs, return_outputs, num_items_in_batch,
                                          conditioning_ledger=ledger, conditioning_report=report)
        finally:
            report["execution"] = ledger.snapshot()

    def _compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None, *,
        conditioning_ledger=None, conditioning_report=None,
    ):
        role = conditioning_ledger.role if conditioning_ledger is not None else lambda _: nullcontext()
        # ``num_items_in_batch`` remains in the public Trainer signature.  The
        # local Qwen loss has no compatible forward argument for this hint.
        del num_items_in_batch
        working_inputs = dict(inputs)
        metadata = working_inputs.pop("stage1_batch_metadata", None)
        if not isinstance(metadata, Stage1BatchMetadata):
            raise TypeError(
                "stage1_batch_metadata must be a Stage1BatchMetadata instance"
            )

        self._validate_ema_teacher_device(working_inputs)
        with role("helper"):
            targets = self.target_builder.build(
                self.ema_teacher.model, working_inputs, metadata
            )
        self._validate_targets(targets, metadata, working_inputs)
        model_inputs = {
            key: value
            for key, value in working_inputs.items()
            if key not in _HELPER_INPUTS
        }
        model_inputs["latent_hidden_states"] = targets.latent_values.detach()
        model_inputs["image_out_mask"] = targets.latent_mask
        if self.stage1_preference.enabled:
            model_inputs["use_cache"] = False

        labels = model_inputs.get("labels")
        if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
            raise TypeError("Stage 1 labels must be a rank-2 tensor")
        student_inputs = model_inputs
        if labels.shape[0] > 1:
            student_inputs = dict(model_inputs)
            student_inputs.pop("labels")

        gain = getattr(self, "_vision_gain", None)
        gain_plan = None
        if gain is not None and model.training and torch.is_grad_enabled():
            from .vision_gain import build_stage1_gain_plan
            from .vision_gain_execution import capture_gain_rng
            gain_plan = build_stage1_gain_plan(metadata, model_inputs["input_ids"], labels,
                                               model_inputs["attention_mask"])
            gain.accumulator.microbatch_count += 1
            rng_before = capture_gain_rng(self.model) if gain_plan.blocks else None
        conditioning_context = None
        if conditioning_ledger is not None and gain_plan is not None:
            from .prefix_mask_metadata import build_helper_snapshot
            from .vision_gain_packed_execution import PrefixConditioningContext
            snapshot = build_helper_snapshot(targets, metadata, gain_plan,
                config=self.prefix_conditioning_config,
                teacher_version=self._prefix_teacher_version,
                parameter_version=self._prefix_parameter_version)
            conditioning_context = PrefixConditioningContext(self.prefix_conditioning_config,
                snapshot, self.ema_teacher.model, conditioning_ledger, conditioning_report)
        if gain is not None:
            gain.conditioned = conditioning_ledger is not None
        with role("base"):
            outputs = model(**student_inputs)
        clean_kl = None
        kl_config = getattr(self, 'clean_kl_config', CleanKLConfig())
        kl_weight = 0.
        if kl_config.enabled and model.training and torch.is_grad_enabled():
            if conditioning_context is None:
                raise ValueError('clean KL requires the existing prefix reference path')
            kl_weight = kl_config.weight(self.state.global_step, self.state.max_steps)
            clean_kl = CleanKLAccumulator(getattr(outputs, 'logits', None),
                conditioning_context.snapshot.gt_queries,
                temperature=self.prefix_conditioning_config.temperature)
            conditioning_context.clean_kl = clean_kl
        text_kl = None
        text_kl_config = getattr(self, 'text_kl_config', TextKLConfig())
        text_kl_weight = 0.
        if text_kl_config.enabled and model.training and torch.is_grad_enabled():
            if conditioning_context is None:
                raise ValueError('text KL requires the existing conditioned gain path')
            text_kl_weight = text_kl_config.weight(self.state.global_step, self.state.max_steps)
            text_kl = TextKLAccumulator(getattr(outputs, 'logits', None),
                conditioning_context.snapshot.gt_queries,
                temperature=self.prefix_conditioning_config.temperature)
            conditioning_context.text_kl = text_kl
            conditioning_context.text_kl_weight = text_kl_weight
        if conditioning_report is not None and gain_plan is not None:
            conditioning_report["local_K"] = len(gain_plan.blocks)
            logits = getattr(outputs, "logits", None)
            if isinstance(logits, torch.Tensor):
                from .vision_gain import score_stage1_gain_block
                with torch.no_grad():
                    conditioning_report["S_F_if_already_available"] = [
                        float(score_stage1_gain_block(logits.detach(), block))
                        for block in gain_plan.blocks]
        if gain_plan is not None and gain_plan.blocks:
            from .vision_gain_execution import execute_stage1_gain
            rng_after = capture_gain_rng(self.model)
            try:
                execute_stage1_gain(self.model, model_inputs, gain_plan, outputs,
                                    rng_before, rng_after, accumulator=gain.accumulator,
                                    execution=self.vision_gain_config.execution,
                                    branch_mode=self.vision_gain_config.branch_mode,
                                    conditioning_context=conditioning_context)
            except (ValueError, RuntimeError, FloatingPointError) as error:
                # Keep the original site (e.g. OOM) before the collective failure.
                logging.exception("Stage 1 gain score execution failed for sources %s",
                                  [sample.source_index for sample in metadata.samples])
                gain.accumulator.fail(error)
                if conditioning_ledger is not None:
                    conditioning_ledger.record_gain_failure(error)

        model_ce_loss = getattr(outputs, "loss", None)
        if labels.shape[0] == 1 and (
            not isinstance(model_ce_loss, torch.Tensor)
            or model_ce_loss.numel() != 1
        ):
            raise RuntimeError("Stage 1 student forward must return a scalar loss")
        ce_loss = self._sample_balanced_ce_loss(
            outputs, labels, model_ce_loss
        )
        outputs.loss = ce_loss
        alignment_loss = self._alignment_loss(outputs, targets)
        loss = self.ce_weight * ce_loss + self.sim_weight * alignment_loss
        kl_metrics = {}
        if clean_kl is not None and not gain.accumulator.failed:
            # A failed gain/reference call already schedules a collective stop.
            # Do not raise on partial KL coverage before peers reach that gate.
            kl_loss, kl_metrics = clean_kl.finish()
            weighted_kl = kl_weight * kl_loss
            loss = loss + weighted_kl
            kl_metrics.update(clean_kl_beta=loss.detach().new_tensor(kl_weight),
                              clean_kl_weighted=weighted_kl.detach())
        if text_kl is not None and not gain.accumulator.failed:
            if not text_kl.keys:
                text_kl.optimizer_loss, _ = text_kl.finish()
            if text_kl.optimizer_loss is None or text_kl.metrics is None:
                raise RuntimeError('text KL optimizer gradient was not collected')
            weighted_text_kl = text_kl_weight * text_kl.optimizer_loss
            loss = loss + weighted_text_kl
            kl_metrics.update(text_kl.metrics,
                text_kl_beta=loss.detach().new_tensor(text_kl_weight),
                text_kl_weighted=weighted_text_kl.detach())
        preference = None
        if self.stage1_preference.enabled:
            rejected_inputs = dict(model_inputs)
            rejected_inputs.pop("labels", None)
            replacement = build_iier_latents(
                outputs.inputs_embeds, model_inputs["input_ids"], targets.latent_mask,
                metadata, image_token_id=self.model.config.image_token_id,
            )
            # Reuse precisely the clean pre-LM input (including text embeddings),
            # replacing only selected bodies. The decoder recomputes its own K/V.
            rejected_inputs["inputs_embeds"] = outputs.inputs_embeds.masked_scatter(
                targets.latent_mask.to(outputs.inputs_embeds.device).unsqueeze(-1),
                replacement.to(outputs.inputs_embeds),
            )
            rejected_inputs.pop("latent_hidden_states")
            rejected_inputs.pop("image_out_mask")
            rejected_inputs.pop("pixel_values", None)
            with role("iier"):
                rejected_outputs = model(**rejected_inputs)
            processor = getattr(self, "processing_class", None)
            tokenizer = getattr(processor, "tokenizer", processor)
            excluded = tuple(getattr(tokenizer, "all_special_ids", ()))
            eos_ids = getattr(self.model.config, "eos_token_id", None)
            if eos_ids is not None:
                excluded += tuple(eos_ids) if isinstance(eos_ids, (tuple, list)) else (eos_ids,)
            preference = iier_preference_loss(
                outputs.logits, rejected_outputs.logits, labels, targets.latent_mask,
                metadata, beta=self.stage1_preference.beta, excluded_token_ids=excluded,
            )
            loss = loss + self.stage1_preference.weight * preference.loss
        if not bool(torch.isfinite(loss.detach()).all().item()):
            raise FloatingPointError("Stage 1 loss is nonfinite")

        if self.collect_stage1_diagnostics:
            diagnostics = {
                name: value.detach()
                for name, value in targets.diagnostics.items()
            }
            diagnostics.update(
                {
                    "ce_loss": ce_loss.detach(),
                    "alignment_loss": alignment_loss.detach(),
                    "total_loss": loss.detach(),
                }
            )
            self._latest_stage1_diagnostics = diagnostics
            diagnostics.update(kl_metrics)
            if preference is not None:
                diagnostics.update({
                    "iier_loss": preference.loss.detach(),
                    "iier_margin": (preference.policy_margins.detach().mean()
                                    if preference.policy_margins.numel() else loss.detach() * 0),
                    "iier_pair_count": loss.new_tensor(preference.pair_losses.numel()).detach(),
                })
        else:
            self._latest_stage1_diagnostics = None
        if conditioning_ledger is not None:
            try:
                conditioning_ledger.validate_initial(gain_active=gain_plan is not None,
                    local_blocks=len(gain_plan.blocks) if gain_plan is not None else 0,
                    iier_enabled=self.stage1_preference.enabled)
            except (ValueError, RuntimeError) as error:
                if gain is None:
                    raise
                # Keep the original gain error while rejecting the partial
                # budget at the existing collective boundary on every rank.
                if not gain.accumulator.failed:
                    gain.accumulator.fail(error)
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _sample_balanced_ce_loss(
        outputs,
        labels: torch.Tensor,
        model_ce_loss: torch.Tensor | None,
    ) -> torch.Tensor:
        if labels.shape[0] == 1:
            assert isinstance(model_ce_loss, torch.Tensor)
            return model_ce_loss

        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError(
                "batched Stage 1 CE requires rank-3 student logits"
            )
        if tuple(logits.shape[:2]) != tuple(labels.shape):
            raise ValueError("Stage 1 logits and labels must share shape [B, T]")
        if logits.shape[1] <= 1:
            raise ValueError("Stage 1 CE requires at least two sequence positions")

        shift_labels = labels[:, 1:].contiguous().to(logits.device)
        per_sample = []
        for row in range(labels.shape[0]):
            valid_mask = shift_labels[row].ne(-100)
            if not bool(valid_mask.any().item()):
                raise ValueError(
                    "every Stage 1 sample must contain at least one supervised token"
                )
            supervised_logits = logits[row, :-1, :][valid_mask].float()
            supervised_labels = shift_labels[row][valid_mask]
            per_sample.append(
                F.cross_entropy(supervised_logits, supervised_labels)
            )
        return torch.stack(per_sample).mean()

    @staticmethod
    def _batch_mean_of_sample_means(
        position_losses: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        require_nonempty: bool,
        value_name: str,
    ) -> torch.Tensor:
        if position_losses.ndim != 2 or tuple(position_losses.shape) != tuple(
            valid_mask.shape
        ):
            raise ValueError("per-position losses and mask must share shape [B, T]")
        mask = valid_mask.to(device=position_losses.device, dtype=torch.bool)
        counts = mask.sum(dim=1)
        if require_nonempty and bool((counts == 0).any().item()):
            raise ValueError(
                f"every Stage 1 sample must contain at least one {value_name}"
            )
        sums = position_losses.masked_fill(~mask, 0).sum(dim=1)
        per_sample = sums / counts.clamp_min(1).to(dtype=sums.dtype)
        return per_sample.mean()

    @staticmethod
    def _validate_targets(
        targets: Stage1TeacherTargets,
        metadata: Stage1BatchMetadata,
        model_inputs: Mapping[str, object],
    ) -> None:
        if not isinstance(targets, Stage1TeacherTargets):
            raise TypeError("Stage 1 target builder must return Stage1TeacherTargets")
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise TypeError("Stage 1 input_ids must be a rank-2 tensor")
        expected_shape = (metadata.batch_size, metadata.sequence_length)
        if tuple(input_ids.shape) != expected_shape:
            raise ValueError("Stage 1 input_ids must match metadata B,T")

        latent_mask = targets.latent_mask
        latent_values = targets.latent_values
        if not isinstance(latent_mask, torch.Tensor):
            raise TypeError("Stage 1 latent_mask must be a tensor")
        if latent_mask.dtype != torch.bool:
            raise TypeError("Stage 1 latent_mask must be a BoolTensor")
        if tuple(latent_mask.shape) != expected_shape:
            raise ValueError("Stage 1 latent_mask must have shape [B, T]")
        if not isinstance(latent_values, torch.Tensor) or latent_values.ndim != 2:
            raise ValueError("Stage 1 latent_values must have shape [N, H]")

        if len(targets.selected_counts) != metadata.batch_size:
            raise RuntimeError(
                "Stage 1 selected-count rows must match metadata batch size"
            )
        expected_mask = torch.zeros_like(latent_mask)
        selected_total = 0
        for sample_index, (sample, block_counts) in enumerate(
            zip(metadata.samples, targets.selected_counts)
        ):
            if len(block_counts) != len(sample.latent_spans):
                raise RuntimeError(
                    "Stage 1 selected-count blocks must match metadata latent blocks"
                )
            for span, raw_count in zip(sample.latent_spans, block_counts):
                count = int(raw_count)
                if count <= 0 and span.latent_positions:
                    raise RuntimeError(
                        "valid latent block has no matching teacher target"
                    )
                if count < 0 or count > len(span.latent_positions):
                    raise RuntimeError(
                        "Stage 1 selected count exceeds its latent block"
                    )
                selected_total += count
                if count:
                    expected_mask[
                        sample_index, list(span.latent_positions[:count])
                    ] = True

        mask_total = int(latent_mask.sum().item())
        value_rows = int(latent_values.shape[0])
        if value_rows != mask_total or selected_total != mask_total:
            raise RuntimeError(
                "valid latent block has no matching teacher target: packed rows "
                "must equal true latent-mask entries"
            )
        if not torch.equal(latent_mask, expected_mask):
            raise RuntimeError(
                "Stage 1 latent mask must match row-major selected targets"
            )

    def _alignment_loss(
        self, outputs, targets: Stage1TeacherTargets
    ) -> torch.Tensor:
        ce_loss = outputs.loss
        if self.sim_weight == 0.0:
            return ce_loss.new_zeros(())

        predicted = getattr(outputs, "hidden_states", None)
        if isinstance(predicted, (tuple, list)):
            predicted = predicted[-1] if predicted else None
        inputs_embeds = getattr(outputs, "inputs_embeds", None)
        if not isinstance(predicted, torch.Tensor) or not isinstance(
            inputs_embeds, torch.Tensor
        ):
            raise RuntimeError(
                "Stage 1 alignment requires hidden_states and inputs_embeds"
            )
        if predicted.ndim != 3 or predicted.shape != inputs_embeds.shape:
            raise ValueError(
                "Stage 1 hidden_states and inputs_embeds must share shape [B, T, H]"
            )
        if tuple(predicted.shape[:2]) != tuple(targets.latent_mask.shape):
            raise ValueError("Stage 1 alignment mask must match student B,T")
        if predicted.shape[1] <= 1:
            return ce_loss.new_zeros(())

        mask = targets.latent_mask[:, 1:].to(
            device=predicted.device, dtype=torch.bool
        )
        sample_losses = []
        for row in range(mask.shape[0]):
            row_mask = mask[row]
            if not bool(row_mask.any().item()):
                sample_losses.append(ce_loss.new_zeros(()))
                continue
            predicted_latents = predicted[row, :-1, :][row_mask].float()
            target_latents = (
                inputs_embeds[row, 1:, :][row_mask].detach().float()
            )
            # Draw once per row so B>1 consumes RNG exactly like sequential B=1.
            target_latents = target_latents + 0.01 * torch.randn_like(
                target_latents
            )
            sample_losses.append(
                (
                    1.0
                    - F.cosine_similarity(
                        target_latents, predicted_latents, dim=-1
                    )
                ).mean()
            )
        return torch.stack(sample_losses).mean()

    def _validate_gradient_norm(self, model: nn.Module | None = None) -> torch.Tensor:
        self._validate_trainable_gradient_coverage()
        candidate = model if isinstance(model, nn.Module) else self.model
        wrapped = getattr(self, "model_wrapped", None)
        norm_source = (
            candidate
            if callable(getattr(candidate, "get_global_grad_norm", None))
            else wrapped
        )
        get_global_norm = getattr(norm_source, "get_global_grad_norm", None)
        if callable(get_global_norm):
            norm = get_global_norm()
            if norm is not None:
                norm = torch.as_tensor(norm, dtype=torch.float32)
                if norm.numel() != 1 or not bool(torch.isfinite(norm).all().item()):
                    raise FloatingPointError("Stage 1 gradient norm is nonfinite")
                if bool((norm < 0).item()):
                    raise FloatingPointError("Stage 1 gradient norm is invalid")
                self._record_gradient_diagnostic(norm)
                return norm

        squared_norm = None
        for parameter in candidate.parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if not bool(torch.isfinite(gradient).all().item()):
                raise FloatingPointError("Stage 1 gradient is nonfinite")
            contribution = gradient.float().square().sum()
            squared_norm = (
                contribution
                if squared_norm is None
                else squared_norm.to(contribution.device) + contribution
            )
        if squared_norm is None:
            raise FloatingPointError(
                "Stage 1 representative trainable gradient is missing"
            )
        norm = squared_norm.sqrt()
        if not bool(torch.isfinite(norm).all().item()):
            raise FloatingPointError("Stage 1 gradient norm is nonfinite")
        self._record_gradient_diagnostic(norm)
        return norm

    def _record_gradient_diagnostic(self, norm: torch.Tensor) -> None:
        if self.collect_stage1_diagnostics:
            self._representative_gradient_norm = float(norm.detach().item())
            if self._latest_stage1_diagnostics is None:
                self._latest_stage1_diagnostics = {}
            self._latest_stage1_diagnostics["gradient_norm"] = norm.detach()

    @staticmethod
    def _module_devices(module: nn.Module) -> set[torch.device]:
        return {
            tensor.device
            for tensor in (*tuple(module.parameters()), *tuple(module.buffers()))
        }

    def _execution_device(self) -> torch.device:
        wrapped_device = getattr(getattr(self, "model_wrapped", None), "device", None)
        accelerator_device = getattr(getattr(self, "accelerator", None), "device", None)
        if wrapped_device is not None:
            wrapped_device = self._canonical_execution_device(wrapped_device)
        if accelerator_device is not None:
            accelerator_device = self._canonical_execution_device(accelerator_device)
        if (
            wrapped_device is not None
            and accelerator_device is not None
            and wrapped_device != accelerator_device
        ):
            raise RuntimeError(
                "Stage 1 prepared model and accelerator execution devices disagree"
            )
        device = wrapped_device or accelerator_device
        if device is None:
            raise RuntimeError("Stage 1 prepared execution device is unavailable")
        return device

    @staticmethod
    def _canonical_execution_device(device) -> torch.device:
        result = torch.device(device)
        if result.type == "cuda" and result.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return result

    def _prepare_ema_teacher_for_training(self) -> None:
        device = self._execution_device()
        teacher = self.ema_teacher.model
        if not isinstance(teacher, nn.Module):
            raise TypeError("Stage 1 EMA teacher model must be a torch.nn.Module")
        teacher.to(device=device)
        teacher.eval()
        devices = self._module_devices(teacher)
        if devices and devices != {device}:
            actual = ", ".join(sorted(str(item) for item in devices))
            raise RuntimeError(
                "Stage 1 EMA teacher placement failed: expected device "
                f"{device}, found {actual}"
            )
        self._ema_execution_device = device

    def _validate_ema_teacher_device(
        self, model_inputs: Mapping[str, object]
    ) -> None:
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("Stage 1 input_ids must be a tensor")
        teacher = self.ema_teacher.model
        if not isinstance(teacher, nn.Module):
            raise TypeError("Stage 1 EMA teacher model must be a torch.nn.Module")
        devices = self._module_devices(teacher)
        if len(devices) > 1:
            actual = ", ".join(sorted(str(item) for item in devices))
            raise RuntimeError(
                f"Stage 1 EMA teacher spans multiple devices: {actual}"
            )
        teacher_device = next(iter(devices), self._ema_execution_device)
        if (
            teacher_device is not None
            and self._ema_execution_device is not None
            and teacher_device != self._ema_execution_device
        ):
            raise RuntimeError(
                "Stage 1 EMA teacher left its prepared execution device: "
                f"expected {self._ema_execution_device}, found {teacher_device}"
            )
        if teacher_device is not None and teacher_device != input_ids.device:
            raise RuntimeError(
                "Stage 1 EMA teacher and batch must share a device before target "
                f"construction: teacher={teacher_device}, batch={input_ids.device}"
            )

    def _restore_pending_stage1_ema_checkpoint(self) -> None:
        checkpoint = self._stage1_pending_ema_checkpoint
        if checkpoint is None:
            return
        if self._ema_execution_device is None:
            raise RuntimeError(
                "Stage 1 EMA teacher restore requires prepared device placement"
            )
        devices = self._module_devices(self.ema_teacher.model)
        if devices and devices != {self._ema_execution_device}:
            raise RuntimeError(
                "Stage 1 EMA teacher restore device mismatch after preparation"
            )
        self._load_stage1_ema_checkpoint(checkpoint)
        self._stage1_pending_ema_checkpoint = None

    def _validate_deepspeed_gradient_norm(self, engine) -> torch.Tensor:
        self._validate_trainable_gradient_coverage()
        zero_stage = self.deepspeed_runtime_profile.zero_stage
        if zero_stage is None:
            raise RuntimeError(
                "DeepSpeed norm collection requires an enabled profile"
            )
        norm = collect_deepspeed_pre_step_norm(
            engine.optimizer, zero_stage=zero_stage
        )
        self._record_gradient_diagnostic(norm)
        return norm

    def _attach_deepspeed_step_adapter(self, engine) -> None:
        existing = self._deepspeed_step_adapter
        if existing is not None:
            if existing.engine is not engine:
                raise RuntimeError(
                    "unsupported DeepSpeed Stage 1 lifecycle: engine changed "
                    "after step adapter installation"
                )
            return
        required = (
            "step",
            "is_gradient_accumulation_boundary",
            "was_step_applied",
        )
        if any(not callable(getattr(engine, name, None)) for name in required):
            raise RuntimeError(
                "unsupported DeepSpeed Stage 1 lifecycle: required engine "
                "step witnesses are unavailable"
            )
        adapter = _DeepSpeedStepAdapter(self, engine, engine.step)
        engine.step = adapter
        self._deepspeed_step_adapter = adapter

    def _install_deepspeed_step_adapter(self) -> None:
        from accelerate.utils import DeepSpeedEngineWrapper, DistributedType

        if getattr(self.accelerator, "distributed_type", None) != DistributedType.DEEPSPEED:
            return
        accelerate_version = importlib_metadata.version("accelerate")
        deepspeed_version = importlib_metadata.version("deepspeed")
        if (
            accelerate_version != _SUPPORTED_ACCELERATE_VERSION
            or deepspeed_version != _SUPPORTED_DEEPSPEED_VERSION
        ):
            raise RuntimeError(
                "unsupported DeepSpeed Stage 1 lifecycle: expected "
                f"accelerate {_SUPPORTED_ACCELERATE_VERSION} and deepspeed "
                f"{_SUPPORTED_DEEPSPEED_VERSION}, found accelerate "
                f"{accelerate_version} and deepspeed {deepspeed_version}"
            )

        wrapper = getattr(self.accelerator, "deepspeed_engine_wrapped", None)
        if not isinstance(wrapper, DeepSpeedEngineWrapper):
            raise RuntimeError(
                "unsupported DeepSpeed Stage 1 lifecycle: Accelerate engine "
                "wrapper is unavailable after prepare"
            )
        from deepspeed.runtime.engine import DeepSpeedEngine
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

        engine = wrapper.engine
        if not isinstance(engine, DeepSpeedEngine):
            raise RuntimeError(
                "unsupported DeepSpeed Stage 1 lifecycle: prepared engine type"
            )
        if getattr(self, "model_wrapped", None) is not engine:
            raise RuntimeError(
                "unsupported DeepSpeed Stage 1 lifecycle: Trainer and "
                "Accelerate prepared different engines"
            )
        profile = self.deepspeed_runtime_profile
        zero_stage = profile.zero_stage if profile.enabled else None
        optimizer = getattr(engine, "optimizer", None)
        if zero_stage == 3:
            optimizer_matches = isinstance(
                optimizer, DeepSpeedZeroOptimizer_Stage3
            )
        elif zero_stage in (1, 2):
            optimizer_matches = isinstance(optimizer, DeepSpeedZeroOptimizer)
            if optimizer_matches:
                optimizer_matches = bool(optimizer.partition_gradients) == (
                    zero_stage == 2
                )
        else:
            raise RuntimeError(
                "unsupported DeepSpeed Stage 1 lifecycle: enabled profile "
                "must select ZeRO stage 1, 2, or 3"
            )
        if not optimizer_matches:
            raise RuntimeError(
                "unsupported DeepSpeed Stage 1 lifecycle: prepared optimizer "
                f"does not match the bound ZeRO-{zero_stage} optimizer"
            )
        self._attach_deepspeed_step_adapter(engine)

    def set_initial_training_values(self, args, dataloader, total_train_batch_size):
        values = list(super().set_initial_training_values(args, dataloader, total_train_batch_size))
        if getattr(self, "_vision_gain", None) is not None and values[5] is not None:
            # The pinned Trainer floors this count, stopping before the final partial group.
            updates = max(math.ceil(values[5] / args.gradient_accumulation_steps), 1)
            values[1] = updates
            if values[4]:
                values[6] = math.ceil(args.num_train_epochs * updates)
            else:
                values[0] = math.ceil(values[6] / updates)
        return tuple(values)

    def _is_accumulation_boundary(self, *, step, steps_in_epoch, batch_index, batch_count):
        if getattr(self, "_vision_gain", None) is not None:
            # Prefetch groups include the actual final partial group, also after
            # resume skips previously consumed batches in the original epoch.
            return batch_index + 1 == batch_count
        return super()._is_accumulation_boundary(
            step=step, steps_in_epoch=steps_in_epoch,
            batch_index=batch_index, batch_count=batch_count,
        )

    def _prepare_gain_step(self):
        gain = getattr(self, "_vision_gain", None)
        if gain is not None:
            gain.prepare()
            if gain.conditioned:
                gain.last_metrics.update(parameter_version=self._prefix_parameter_version,
                                         teacher_version=self._prefix_teacher_version)
        ledger = getattr(self, '_latest_prefix_ledger', None)
        if ledger is not None:
            pending = getattr(gain, 'pending', None)
            if pending is not None:
                self._latest_prefix_report['projection_outputs'] = {
                    name: getattr(pending, name) for name in
                    ('projected', 'skipped', 'zero_direction', 'h_dot_g', 'h_norm_sq')}
            device = getattr(getattr(self, 'args', None), 'device', None)
            self._prefix_update_region = ledger.begin_region('optimizer_gain_ema_update', device)

    def _finish_gain_step(self, model, successful):
        gain = getattr(self, "_vision_gain", None)
        if gain is not None:
            optimizer = (model.optimizer if self.deepspeed_runtime_profile.enabled
                         else getattr(self, "optimizer", None))
            gain.finish(model, optimizer, self.deepspeed_runtime_profile, successful=successful)

    def _finish_optimizer_step(
        self, model: nn.Module | None = None, *, successful: bool
    ) -> None:
        student = model if isinstance(model, nn.Module) else self.model
        self._finish_gain_step(student, successful)
        if not successful:
            ledger = getattr(self, '_latest_prefix_ledger', None)
            region = getattr(self, '_prefix_update_region', None)
            if ledger is not None and region is not None:
                ledger.end_region(region)
                self._prefix_update_region = None
            self._reset_trainable_gradient_coverage()
            return
        config = getattr(self, "prefix_conditioning_config", None)
        if config is not None and config.enabled:
            self._prefix_parameter_version += 1
        if self.ema_setting == "moving":
            self.ema_teacher.update(student)
            if config is not None and config.enabled:
                self._prefix_teacher_version += 1
        self._successful_optimizer_steps += 1
        ledger = getattr(self, '_latest_prefix_ledger', None)
        region = getattr(self, '_prefix_update_region', None)
        if ledger is not None and region is not None:
            ledger.end_region(region)
            self._prefix_update_region = None
        self._reset_trainable_gradient_coverage()

    def optimizer_step(self, *args, optimizer=None, **kwargs):
        """Compatibility entry point mirroring the callback-managed boundary."""
        remaining = args
        if optimizer is None and remaining and hasattr(remaining[0], "step"):
            optimizer, remaining = remaining[0], remaining[1:]
        if optimizer is None:
            optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            raise RuntimeError("Stage 1 optimizer_step requires an optimizer")

        self._validate_gradient_norm(self.model)
        self._prepare_gain_step()
        result = optimizer.step(*remaining, **kwargs)
        skipped = bool(
            getattr(self.accelerator, "optimizer_step_was_skipped", False)
        )
        self._finish_optimizer_step(self.model, successful=not skipped)
        return result

    def _flush_prefix_report(self):
        report = getattr(self, '_latest_prefix_report', None)
        if report is None or getattr(self, '_prefix_identity', None) is None:
            return {}
        from .prefix_mask_accounting import prefix_diagnostics_report, write_prefix_diagnostics
        ledger = self._latest_prefix_ledger
        ledger.resolve_timings()
        report['execution'] = ledger.snapshot()
        gain = getattr(self, '_vision_gain', None)
        report['optimizer_update_count'] = self._prefix_parameter_version
        report['EMA_update_count'] = self._prefix_teacher_version
        norm = getattr(self, '_representative_gradient_norm', None)
        report['base_grad_norm'] = norm if norm is not None else 'unmeasured'
        report['base_grad_norm_scope'] = 'representative_trainable_parameter'
        device = getattr(self.args, 'device', None)
        if getattr(device, 'type', None) == 'cuda':
            report['peak_memory'] = torch.cuda.max_memory_allocated(device)
        output = prefix_diagnostics_report(report, config=self.prefix_conditioning_config,
            identity=self._prefix_identity, metrics=getattr(gain, 'completed_metrics', {}),
            counts=getattr(gain, 'counts', {}), parameter_version=report['parameter_version'],
            teacher_version=report['teacher_version'])
        write_prefix_diagnostics(self._prefix_diagnostics_path, output)
        self._latest_prefix_report = None
        self._latest_prefix_ledger = None
        self._prefix_pending_scalars = {'stage1_prefix/parameter_version': report['parameter_version'],
                'stage1_prefix/teacher_version': report['teacher_version'],
                'stage1_prefix/local_blocks': report.get('local_K', 0)}
        return self._prefix_pending_scalars

    def log(self, logs, start_time=None):
        ledger = getattr(self, "_latest_prefix_ledger", None)
        if ledger is not None:
            # Base/IIER checkpoint closures can run after compute_loss returns.
            self._latest_prefix_report["execution"] = ledger.snapshot()
        self._flush_prefix_report()
        logs = dict(logs, **getattr(self, '_prefix_pending_scalars', {}))
        self._prefix_pending_scalars = {}
        diagnostics = self._latest_stage1_diagnostics
        self._latest_stage1_diagnostics = None
        if self.collect_stage1_diagnostics and diagnostics:
            logs = dict(logs)
            for name, value in diagnostics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    logs[f"stage1/{name}"] = float(value.item())
        gain = getattr(self, "_vision_gain", None)
        if gain is not None:
            logs = dict(logs, **{f"stage1_gain/{k}": v for k, v in gain.last_metrics.items()})
        super().log(logs, start_time=start_time)

    def _save_stage1_ema_checkpoint(self, checkpoint_dir: str | Path) -> None:
        target = Path(checkpoint_dir)
        target.mkdir(parents=True, exist_ok=True)
        state = self.ema_teacher.state_dict()
        payload = {
            "schema_version": _STAGE1_EMA_TEACHER_STATE_SCHEMA_VERSION,
            "kind": _STAGE1_EMA_TEACHER_STATE_KIND,
            "ema_setting": getattr(self, "ema_setting", "moving"),
            "state": state,
        }
        final_path = target / STAGE1_EMA_TEACHER_STATE_FILENAME
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target,
                prefix=f".{STAGE1_EMA_TEACHER_STATE_FILENAME}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, final_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _stage1_ema_checkpoint_path(checkpoint_dir: str | Path) -> Path:
        checkpoint_path = Path(checkpoint_dir) / STAGE1_EMA_TEACHER_STATE_FILENAME
        if checkpoint_path.is_symlink():
            raise RuntimeError(
                "resumable Stage 1 EMA teacher state must not be a symlink: "
                f"{checkpoint_path}"
            )
        if not checkpoint_path.is_file():
            raise RuntimeError(
                "resumable Stage 1 checkpoint is missing EMA teacher state: "
                f"{checkpoint_path}"
            )
        return checkpoint_path

    def _load_stage1_ema_checkpoint(self, checkpoint_dir: str | Path) -> None:
        checkpoint_path = self._stage1_ema_checkpoint_path(checkpoint_dir)
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"invalid Stage 1 EMA teacher state: {checkpoint_path}"
            ) from error

        if type(payload) is not dict:
            raise RuntimeError("invalid Stage 1 EMA teacher state schema")
        schema_version = payload.get("schema_version")
        if type(schema_version) is not int or schema_version not in (1, 2):
            raise RuntimeError("unsupported Stage 1 EMA teacher state schema version")
        required = {"schema_version", "kind", "state"}
        if schema_version == 2:
            required.add("ema_setting")
        if set(payload) != required:
            raise RuntimeError("invalid Stage 1 EMA teacher state schema")
        if (
            type(payload["kind"]) is not str
            or payload["kind"] != _STAGE1_EMA_TEACHER_STATE_KIND
        ):
            raise RuntimeError("invalid Stage 1 EMA teacher state kind")
        checkpoint_ema_setting = (
            "moving" if schema_version == 1 else payload["ema_setting"]
        )
        if checkpoint_ema_setting not in ("moving", "frozen"):
            raise RuntimeError("invalid Stage 1 EMA setting")
        runtime_ema_setting = getattr(self, "ema_setting", "moving")
        if checkpoint_ema_setting != runtime_ema_setting:
            raise RuntimeError(
                "Stage 1 EMA setting mismatch: "
                f"checkpoint={checkpoint_ema_setting}, "
                f"runtime={runtime_ema_setting}"
            )

        state = payload["state"]
        try:
            (
                state_method,
                state_decay,
                state_names,
                checkpoint_shapes,
                parameters,
            ) = _decode_ema_state(state)
        except RuntimeError as error:
            raise RuntimeError(
                f"invalid Stage 1 EMA teacher {error}"
            ) from error
        if state_method != self.ema_teacher.method:
            raise RuntimeError(
                "Stage 1 EMA teacher method mismatch: "
                f"checkpoint={state_method}, runtime={self.ema_teacher.method}"
            )
        if state_decay != self.ema_teacher.decay:
            raise RuntimeError(
                "Stage 1 EMA teacher decay mismatch: "
                f"checkpoint={state_decay}, runtime={self.ema_teacher.decay}"
            )

        if state_names != self.ema_teacher.parameter_names:
            raise RuntimeError("Stage 1 EMA teacher parameter names mismatch")
        teacher_parameters = dict(
            self.ema_teacher.model.named_parameters(remove_duplicate=False)
        )
        expected_shapes = tuple(
            (
                name,
                tuple(int(dimension) for dimension in teacher_parameters[name].shape),
            )
            for name in self.ema_teacher.parameter_names
        )
        if checkpoint_shapes != expected_shapes:
            raise RuntimeError("Stage 1 EMA teacher parameter shapes mismatch")
        if tuple(parameters) != state_names:
            raise RuntimeError("invalid Stage 1 EMA teacher parameters")
        for name, expected_shape in expected_shapes:
            tensor = parameters.get(name)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"Stage 1 EMA teacher tensor shape mismatch for {name}"
                )
        self.ema_teacher.load_state_dict(state)

    def _save_checkpoint(self, model, trial):
        """Append replicated EMA state after Trainer/DeepSpeed checkpointing."""

        self._defer_stage1_checkpoint_rotation = True
        self._stage1_deferred_checkpoint_rotation = None
        try:
            super()._save_checkpoint(model, trial)
        except Exception:
            self._stage1_deferred_checkpoint_rotation = None
            raise
        finally:
            self._defer_stage1_checkpoint_rotation = False
        run_dir = Path(self._get_output_dir(trial=trial))
        checkpoint_dir = run_dir / (
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        )
        ema_save_error: Exception | None = None
        try:
            if self.args.should_save:
                try:
                    self._save_stage1_ema_checkpoint(checkpoint_dir)
                    self._save_prefix_runtime(checkpoint_dir)
                    write_deepspeed_runtime_sidecar(
                        checkpoint_dir, self.deepspeed_runtime_profile
                    )
                except Exception as error:
                    ema_save_error = error
            local_failure = torch.tensor(
                int(ema_save_error is not None),
                dtype=torch.int32,
                device=self.accelerator.device,
            )
            failed_processes = self.accelerator.reduce(
                local_failure, reduction="sum"
            )
            if int(failed_processes.item()) != 0:
                if ema_save_error is not None:
                    raise ema_save_error
                raise RuntimeError(
                    "Stage 1 EMA checkpoint save failed on another process"
                )
            self.accelerator.wait_for_everyone()
            deferred = self._stage1_deferred_checkpoint_rotation
            if deferred is not None:
                use_mtime, output_dir = deferred
                super()._rotate_checkpoints(
                    use_mtime=use_mtime, output_dir=output_dir
                )
        finally:
            self._stage1_deferred_checkpoint_rotation = None

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if getattr(self, "_defer_stage1_checkpoint_rotation", False):
            if self._stage1_deferred_checkpoint_rotation is not None:
                raise RuntimeError("Stage 1 checkpoint rotation was requested twice")
            self._stage1_deferred_checkpoint_rotation = (use_mtime, output_dir)
            return
        super()._rotate_checkpoints(use_mtime=use_mtime, output_dir=output_dir)

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Require and defer compatible EMA state for Stage 1 resume."""

        self._publish_deepspeed_runtime_identity(
            Path(self.args.output_dir), label="Stage 1 runtime identity publication"
        )

        if resume_from_checkpoint is False:
            resume_from_checkpoint = None
        requested_resume = (
            kwargs["model_path"]
            if "model_path" in kwargs
            else resume_from_checkpoint
        )

        if requested_resume is True:
            checkpoint = get_last_checkpoint(self.args.output_dir)
            if checkpoint is None:
                raise ValueError(
                    "No valid checkpoint found in output directory "
                    f"({self.args.output_dir})"
                )
            pending_checkpoint = Path(checkpoint)
        elif requested_resume is None:
            pending_checkpoint = None
        elif isinstance(requested_resume, (str, os.PathLike)):
            pending_checkpoint = Path(requested_resume)
        else:
            raise TypeError(
                "Stage 1 resume_from_checkpoint must be a path, True, or None"
            )

        if pending_checkpoint is not None:
            self.stage1_preference.validate_resume(pending_checkpoint)
            self.clean_kl_config.validate_resume(pending_checkpoint)
            self.text_kl_config.validate_resume(pending_checkpoint)
            validate_deepspeed_resume_identity(
                pending_checkpoint, self.deepspeed_runtime_profile
            )
            self._stage1_ema_checkpoint_path(pending_checkpoint)
        from .vision_gain import Stage1VisionGainConfig
        from .checkpointing import validate_stage1_gain_resume
        gain_config = getattr(self, "vision_gain_config", Stage1VisionGainConfig())
        if pending_checkpoint is not None:
            attention = getattr(getattr(getattr(self, "model", None), "config", None), "_attn_implementation", None)
            prefix = getattr(self, 'prefix_conditioning_config', None)
            counts = validate_stage1_gain_resume(pending_checkpoint, gain_config, attention, prefix_config=prefix)
            if prefix is not None and prefix.enabled:
                from .checkpointing import read_prefix_runtime_state
                self._prefix_parameter_version, self._prefix_teacher_version = read_prefix_runtime_state(pending_checkpoint)
            if getattr(self, "_vision_gain", None) is not None and counts is not None:
                self._vision_gain.counts = counts
        self._stage1_pending_ema_checkpoint = pending_checkpoint
        resolved_resume = (
            None if pending_checkpoint is None else str(pending_checkpoint)
        )
        if "model_path" in kwargs:
            kwargs["model_path"] = resolved_resume
        result = super().train(
            resume_from_checkpoint=resolved_resume,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs,
        )
        self._flush_prefix_report()
        return result

    def _save(self, output_dir=None, state_dict=None):
        """Keep Trainer's gathered state while writing canonical PEFT weights."""

        if self.stage1_peft_metadata is None:
            return super()._save(output_dir, state_dict=state_dict)

        from stage1.checkpointing import _save_stage1_peft_weights

        target = Path(output_dir or self.args.output_dir)
        target.mkdir(parents=True, exist_ok=True)
        _save_stage1_peft_weights(
            self.model,
            target,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(target)
        torch.save(self.args, target / TRAINING_ARGS_NAME)

    def _save_prefix_runtime(self, directory):
        config = getattr(self, 'prefix_conditioning_config', None)
        if config is not None and config.enabled:
            from .checkpointing import write_prefix_runtime_state
            write_prefix_runtime_state(Path(directory), parameter_version=self._prefix_parameter_version,
                                       teacher_version=self._prefix_teacher_version)

    def save_model(self, output_dir=None, _internal_call=False):
        self._flush_prefix_report()
        result = self._save_stage1_model(output_dir, _internal_call=_internal_call)
        if self.args.should_save:
            self.stage1_preference.save(output_dir or self.args.output_dir)
            self.clean_kl_config.save(output_dir or self.args.output_dir)
            self.text_kl_config.save(output_dir or self.args.output_dir)
            self._save_prefix_runtime(output_dir or self.args.output_dir)
        gain = getattr(self, "_vision_gain", None)
        prefix = getattr(self, 'prefix_conditioning_config', None)
        if (gain is not None or (prefix is not None and prefix.enabled)) and self.args.should_save:
            from .checkpointing import write_stage1_gain_metadata
            write_stage1_gain_metadata(output_dir or self.args.output_dir, self.vision_gain_config,
                self.model.config._attn_implementation, gain.counts if gain is not None else None,
                prefix_config=prefix)
        return result

    def _save_stage1_model(self, output_dir=None, _internal_call=False):
        """Extend ordinary Trainer saves with strict Stage 1 PEFT sidecars."""

        metadata = self.stage1_peft_metadata
        if metadata is None:
            return super().save_model(output_dir, _internal_call=_internal_call)

        from stage1.checkpointing import (
            _save_stage1_peft_supplements,
            _validate_stage1_peft_runtime,
            classify_stage1_artifact,
        )

        if not self.args.save_safetensors:
            raise ValueError("Stage 1 PEFT checkpoints require safetensors")
        processor = getattr(self, "processing_class", None)
        if processor is None:
            raise ValueError("Stage 1 PEFT checkpoint requires a processor")
        _validate_stage1_peft_runtime(self.model, processor, metadata)
        for field in (
            "ce_weight",
            "sim_weight",
            "coverage_p",
            "image_pool_k",
            "ema_setting",
        ):
            runtime_value = getattr(self, field, None)
            metadata_value = getattr(metadata, field)
            if runtime_value != metadata_value:
                raise ValueError(
                    f"Stage 1 {field} metadata mismatch: "
                    f"runtime={runtime_value}, metadata={metadata_value}"
                )
        if (
            metadata.teacher_target_mode != STAGE1_TEACHER_TARGET_MODE
            or metadata.training_implementation_version
            != STAGE1_TRAINING_IMPLEMENTATION_VERSION
        ):
            raise ValueError(
                "Stage 1 teacher implementation metadata mismatch: "
                f"metadata=({metadata.teacher_target_mode!r}, "
                f"{metadata.training_implementation_version}), "
                f"runtime=({STAGE1_TEACHER_TARGET_MODE!r}, "
                f"{STAGE1_TRAINING_IMPLEMENTATION_VERSION})"
            )

        super().save_model(output_dir, _internal_call=_internal_call)
        target = Path(output_dir or self.args.output_dir)
        if self.is_deepspeed_enabled:
            # The sparse ZeRO gather is collective. Non-writing ranks use a
            # private temporary destination while rank zero publishes.
            if self.args.should_save:
                _save_stage1_peft_supplements(self.model, target, metadata)
            else:
                with tempfile.TemporaryDirectory(
                    prefix="stage1-token-rows-"
                ) as temporary:
                    _save_stage1_peft_supplements(
                        self.model, Path(temporary), metadata
                    )
        elif self.args.should_save:
            _save_stage1_peft_supplements(self.model, target, metadata)
        if self.args.should_save:
            classify_stage1_artifact(target)


__all__ = [
    "CustomTrainerStage1",
    "STAGE1_EMA_TEACHER_STATE_FILENAME",
    "collect_deepspeed_pre_step_norm",
]
