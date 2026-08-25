"""True-batched Stage 1 loss and EMA lifecycle on top of ``SFTTrainer``."""

from __future__ import annotations

import math
import weakref
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F
from transformers import TrainerCallback
from trl import SFTTrainer

from .ema import Stage1EmaTeacher
from .targets import Stage1TeacherTargetBuilder
from .types import Stage1BatchMetadata, Stage1TeacherTargets


_HELPER_INPUTS = frozenset(
    {"pixel_values_latent", "image_grid_thw_latent"}
)


class _Stage1OptimizerCallback(TrainerCallback):
    """Delegate Trainer's real optimizer boundary back to Stage 1."""

    def __init__(self, trainer: "CustomTrainerStage1") -> None:
        self._trainer = weakref.ref(trainer)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        del args, state
        trainer = self._trainer()
        if trainer is not None:
            trainer._validate_gradient_norm(kwargs.get("model"))
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        del args, state
        trainer = self._trainer()
        if trainer is not None:
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
        sim_weight: float = 1.0,
        ema_tau: float = 0.999,
        coverage_p: float = 0.9,
        image_pool_k: int = 8,
        ce_weight: float = 1.0,
        helper_group_L: int | None = 256,
        ema_method: str | None = None,
        target_builder: Stage1TeacherTargetBuilder | None = None,
        ema_teacher: Stage1EmaTeacher | None = None,
        collect_stage1_diagnostics: bool = False,
        **kwargs,
    ) -> None:
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
        self.collect_stage1_diagnostics = bool(collect_stage1_diagnostics)
        self._latest_stage1_diagnostics = None
        self._successful_optimizer_steps = 0

        if target_builder is None:
            tokenizer = getattr(self, "processing_class", None)
            if tokenizer is None:
                tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                raise ValueError(
                    "Stage 1 trainer requires a tokenizer/processing_class "
                    "or an explicit target_builder"
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
        self.add_callback(_Stage1OptimizerCallback(self))

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
        working_inputs = dict(inputs)
        metadata = working_inputs.pop("stage1_batch_metadata", None)
        if not isinstance(metadata, Stage1BatchMetadata):
            raise TypeError(
                "stage1_batch_metadata must be a Stage1BatchMetadata instance"
            )

        targets = self.target_builder.build(
            self.ema_teacher.model, working_inputs, metadata
        )
        self._validate_targets(targets, metadata, working_inputs)
        model_inputs = {
            key: value
            for key, value in working_inputs.items()
            if key not in _HELPER_INPUTS
        }
        model_inputs["latent_hidden_states"] = targets.latent_values
        model_inputs["image_out_mask"] = targets.latent_mask

        outputs = model(**model_inputs)
        ce_loss = getattr(outputs, "loss", None)
        if not isinstance(ce_loss, torch.Tensor) or ce_loss.numel() != 1:
            raise RuntimeError("Stage 1 student forward must return a scalar loss")
        alignment_loss = self._alignment_loss(outputs, targets)
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
            if self._latest_stage1_diagnostics is None:
                self._latest_stage1_diagnostics = {}
            self._latest_stage1_diagnostics["gradient_norm"] = norm.detach()

    def _finish_optimizer_step(
        self, model: nn.Module | None = None, *, successful: bool
    ) -> None:
        if not successful:
            return
        student = model if isinstance(model, nn.Module) else self.model
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


__all__ = ["CustomTrainerStage1"]
