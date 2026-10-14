"""True-batched Stage 1 loss and EMA lifecycle on top of ``SFTTrainer``."""

from __future__ import annotations

import math
import os
from importlib import metadata as importlib_metadata
from pathlib import Path
import tempfile
import weakref
from typing import Callable, Mapping

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
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
from .targets import Stage1TeacherTargetBuilder
from .types import Stage1BatchMetadata, Stage1TeacherTargets


_HELPER_INPUTS = frozenset(
    {"pixel_values_latent", "image_grid_thw_latent"}
)
_SUPPORTED_ACCELERATE_VERSION = "1.10.1"
_SUPPORTED_DEEPSPEED_VERSION = "0.18.0"
_STAGE1_EMA_TEACHER_STATE_SCHEMA_VERSION = 1
_STAGE1_EMA_TEACHER_STATE_KIND = "stage1_ema_teacher"
_STAGE1_CE_TOKEN_CHUNK_SIZE = 1024


def _checkpointed_stage1_ce_sum(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    return F.cross_entropy(
        logits.float(),
        labels,
        reduction="sum",
        ignore_index=-100,
    )


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

    def __init__(
        self,
        *args,
        deepspeed_runtime_profile: DeepSpeedRuntimeProfile,
        sim_weight: float = 1.0,
        ema_tau: float = 0.999,
        coverage_p: float = 0.9,
        image_pool_k: int = 8,
        ce_weight: float = 1.0,
        helper_group_L: int | None = 256,
        textual_states: str = "previous",
        ema_method: str | None = None,
        target_builder: Stage1TeacherTargetBuilder | None = None,
        ema_teacher: Stage1EmaTeacher | None = None,
        stage1_peft_metadata=None,
        collect_stage1_diagnostics: bool = False,
        **kwargs,
    ) -> None:
        if not isinstance(deepspeed_runtime_profile, DeepSpeedRuntimeProfile):
            raise TypeError("Stage 1 trainer requires a DeepSpeed runtime profile")
        self.deepspeed_runtime_profile = deepspeed_runtime_profile
        self.stage1_peft_metadata = stage1_peft_metadata
        # Preserve the repository's legacy constructor spelling across the
        # tokenizer -> processing_class Trainer API migration.
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)

        self.sim_weight = self._finite_weight(sim_weight, "sim_weight")
        self.ce_weight = self._finite_weight(ce_weight, "ce_weight")
        self.coverage_p = float(coverage_p)
        self.image_pool_k = int(image_pool_k)
        self.helper_group_L = helper_group_L
        self.textual_states = textual_states
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
                textual_states=self.textual_states,
            )
        self.target_builder = target_builder

        # SFTTrainer may apply PEFT in super().__init__.  The EMA snapshot must
        # therefore be constructed only now, from self.model's final topology.
        method = ema_method or self._infer_ema_method(self.model)
        self.ema_teacher = ema_teacher or Stage1EmaTeacher(
            self.model, decay=float(ema_tau), method=method
        )
        self.add_callback(_Stage1OptimizerCallback(self))

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
        # ``num_items_in_batch`` remains in the public Trainer signature.  The
        # local Qwen loss has no compatible forward argument for this hint.
        del num_items_in_batch
        model_inputs, metadata, targets = self._prepare_stage1_loss_inputs(inputs)
        labels = model_inputs.get("labels")
        if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
            raise TypeError("Stage 1 labels must be a rank-2 tensor")
        forward_inputs = model_inputs
        if labels.shape[0] > 1:
            forward_inputs = dict(model_inputs)
            forward_inputs.pop("labels")

        outputs = model(**forward_inputs)
        ce_loss, alignment_loss = self._stage1_native_losses(outputs, labels, targets)
        loss = self.ce_weight * ce_loss + self.sim_weight * alignment_loss
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
        else:
            self._latest_stage1_diagnostics = None
        return (loss, outputs) if return_outputs else loss

    def _prepare_stage1_loss_inputs(self, inputs) -> tuple[dict, Stage1BatchMetadata, Stage1TeacherTargets]:
        """Build helper targets once, preserving the original model tensors."""
        working_inputs = dict(inputs)
        metadata = working_inputs.pop("stage1_batch_metadata", None)
        if not isinstance(metadata, Stage1BatchMetadata):
            raise TypeError(
                "stage1_batch_metadata must be a Stage1BatchMetadata instance"
            )

        self._validate_ema_teacher_device(working_inputs)
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

        return model_inputs, metadata, targets

    def _stage1_native_losses(self, outputs, labels: torch.Tensor, targets: Stage1TeacherTargets) -> tuple[torch.Tensor, torch.Tensor]:
        """Unchanged native per-sample CE and latent alignment reductions."""
        model_ce_loss = getattr(outputs, "loss", None)
        if labels.shape[0] == 1 and (
            not isinstance(model_ce_loss, torch.Tensor)
            or model_ce_loss.numel() != 1
        ):
            raise RuntimeError("Stage 1 student forward must return a scalar loss")
        ce_loss = self._equal_sample_ce_loss(outputs, labels, model_ce_loss)
        outputs.loss = ce_loss
        alignment_loss = self._alignment_loss(outputs, targets)
        return ce_loss, alignment_loss

    @staticmethod
    def _equal_sample_ce_loss(outputs, labels, model_ce_loss) -> torch.Tensor:
        if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
            raise TypeError("Stage 1 labels must be a rank-2 tensor")
        if labels.shape[0] == 1:
            return model_ce_loss

        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError("Stage 1 student forward must return rank-3 logits")
        if tuple(logits.shape[:2]) != tuple(labels.shape):
            raise ValueError("Stage 1 logits and labels must share shape [B, T]")

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:].to(shift_logits.device)
        valid_tokens = shift_labels.ne(-100)
        valid_counts = valid_tokens.sum(dim=1)
        if bool(valid_counts.eq(0).any().item()):
            raise RuntimeError(
                "each Stage 1 sample must contain a supervised target token"
            )
        sample_losses = []
        for sample_index in range(labels.shape[0]):
            chunk_sums = []
            for start in range(0, shift_logits.shape[1], _STAGE1_CE_TOKEN_CHUNK_SIZE):
                stop = min(start + _STAGE1_CE_TOKEN_CHUNK_SIZE, shift_logits.shape[1])
                chunk_sums.append(
                    activation_checkpoint(
                        _checkpointed_stage1_ce_sum,
                        shift_logits[sample_index, start:stop],
                        shift_labels[sample_index, start:stop],
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                )
            sample_losses.append(
                torch.stack(chunk_sums).sum() / valid_counts[sample_index]
            )
        return torch.stack(sample_losses).mean()

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
        if not bool(mask.any().item()):
            return ce_loss.new_zeros(())
        predicted_latents = predicted[:, :-1, :][mask].contiguous().float()
        target_latents = (
            inputs_embeds[:, 1:, :][mask].contiguous().detach().float()
        )
        # Preserve the original B=1 objective's small target perturbation.
        target_latents = target_latents + 0.01 * torch.randn_like(target_latents)
        cosine = F.cosine_similarity(
            target_latents, predicted_latents, dim=-1
        ).mean()
        return 1.0 - cosine

    def _validate_gradient_norm(self, model: nn.Module | None = None) -> torch.Tensor:
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

    def _finish_optimizer_step(
        self, model: nn.Module | None = None, *, successful: bool
    ) -> None:
        if not successful:
            return
        student = model if isinstance(model, nn.Module) else self.model
        if self.ema_teacher.decay != 1.0:
            self.ema_teacher.update(student)
        self._successful_optimizer_steps += 1

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
        result = optimizer.step(*remaining, **kwargs)
        skipped = bool(
            getattr(self.accelerator, "optimizer_step_was_skipped", False)
        )
        self._finish_optimizer_step(self.model, successful=not skipped)
        return result

    def log(self, logs, start_time=None):
        diagnostics = self._latest_stage1_diagnostics
        self._latest_stage1_diagnostics = None
        if self.collect_stage1_diagnostics and diagnostics:
            logs = dict(logs)
            for name, value in diagnostics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    logs[f"stage1/{name}"] = float(value.item())
        super().log(logs, start_time=start_time)

    def _save_stage1_ema_checkpoint(self, checkpoint_dir: str | Path) -> None:
        target = Path(checkpoint_dir)
        target.mkdir(parents=True, exist_ok=True)
        state = self.ema_teacher.state_dict()
        payload = {
            "schema_version": _STAGE1_EMA_TEACHER_STATE_SCHEMA_VERSION,
            "kind": _STAGE1_EMA_TEACHER_STATE_KIND,
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

        required = {"schema_version", "kind", "state"}
        if type(payload) is not dict or set(payload) != required:
            raise RuntimeError("invalid Stage 1 EMA teacher state schema")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"]
            != _STAGE1_EMA_TEACHER_STATE_SCHEMA_VERSION
        ):
            raise RuntimeError("unsupported Stage 1 EMA teacher state schema version")
        if (
            type(payload["kind"]) is not str
            or payload["kind"] != _STAGE1_EMA_TEACHER_STATE_KIND
        ):
            raise RuntimeError("invalid Stage 1 EMA teacher state kind")

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
            validate_deepspeed_resume_identity(
                pending_checkpoint, self.deepspeed_runtime_profile
            )
            self._stage1_ema_checkpoint_path(pending_checkpoint)
        self._stage1_pending_ema_checkpoint = pending_checkpoint
        resolved_resume = (
            None if pending_checkpoint is None else str(pending_checkpoint)
        )
        if "model_path" in kwargs:
            kwargs["model_path"] = resolved_resume
        return super().train(
            resume_from_checkpoint=resolved_resume,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs,
        )

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

    def save_model(self, output_dir=None, _internal_call=False):
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
        for field in ("ce_weight", "sim_weight", "coverage_p", "image_pool_k"):
            runtime_value = getattr(self, field, None)
            metadata_value = getattr(metadata, field)
            if runtime_value != metadata_value:
                raise ValueError(
                    f"Stage 1 {field} metadata mismatch: "
                    f"runtime={runtime_value}, metadata={metadata_value}"
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
