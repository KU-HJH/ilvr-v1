"""Thin Hugging Face Trainer adapter for Stage 2 model-native loss."""

import os
import math
import tempfile
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import Trainer, TrainerCallback
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

from deepspeed_runtime import (
    DeepSpeedRuntimeProfile,
    load_deepspeed_runtime_profile,
    validate_deepspeed_resume_identity,
    write_deepspeed_runtime_sidecar,
)
from peft_common import resolve_qwen2_5_vl_model
from stage1.trainer import collect_deepspeed_pre_step_norm
from .checkpointing import save_stage2_peft_metadata
from .sampler import (
    LatentBlockBalancedSampler,
    Stage2SamplerIdentity,
    validate_stage2_sampler_resume_identity,
    write_stage2_sampler_sidecar,
)
from stage1.token_rows import (
    save_sparse_token_rows,
    sparse_token_rows_are_partitioned,
)
from step_preference.ablation import (
    StepAblationSpec,
    build_additive_step_attention_mask,
)
from step_preference.config import StepPreferenceConfig
from step_preference.dpo_loss import aggregate_step_dpo, context_preference_loss
from step_preference.negatives import StepNegative
from step_preference.packing import build_packed_tree_plan
from step_preference.sidecar import build_context_sidecar_plan, build_sidecar_plan
from step_preference.spans import ReasoningStepSpan


_STEP_PREFERENCE_LOG_NAMES = (
    "loss/total",
    "loss/ce",
    "loss/dpo",
    "loss/entropy",
    "loss/dpo_visual",
    "loss/dpo_text",
    "loss/entropy_visual",
    "loss/entropy_text",
    "stats/dpo_margin_visual",
    "stats/dpo_margin_text",
    "stats/entropy_clean",
    "stats/entropy_visual_off",
    "stats/entropy_text_off",
    "stats/valid_visual_pairs",
    "stats/valid_textual_pairs",
    "stats/branch_forwards",
    "stats/scored_tokens",
)

_CONTEXT_PREFERENCE_LOG_NAMES = (
    "loss/total",
    "loss/ce",
    "loss/context_preference",
    "loss/entropy",
    "loss/context_preference_visual",
    "loss/context_preference_text",
    "loss/entropy_visual",
    "loss/entropy_text",
    "stats/context_margin_visual",
    "stats/context_margin_text",
    "stats/logp_clean",
    "stats/logp_visual_off",
    "stats/logp_text_off",
    "stats/entropy_clean",
    "stats/entropy_visual_off",
    "stats/entropy_text_off",
    "stats/scored_tokens",
    "stats/logical_scoring_groups",
)


def _model_compute_dtype(model) -> torch.dtype:
    dtype = getattr(model, "dtype", None)
    if isinstance(dtype, torch.dtype) and dtype.is_floating_point:
        return dtype
    parameter = next(model.parameters(), None)
    if parameter is None or not parameter.dtype.is_floating_point:
        raise ValueError("Stage 2B model must expose a floating compute dtype")
    return parameter.dtype


def _selected_logits(outputs, expected_rows: int) -> torch.Tensor:
    logits = getattr(outputs, "stage2_selected_logits", None)
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[1] != expected_rows
        or logits.shape[2] == 0
        or not logits.is_floating_point()
    ):
        raise ValueError("Stage 2B model returned invalid selected logits")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Stage 2B selected logits must be finite")
    return logits


def _mean_selected_logp(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    if (
        target_ids.ndim != 2
        or target_ids.shape[0] != 1
        or target_ids.shape[1] != logits.shape[1]
    ):
        raise ValueError("Stage 2B selected target IDs do not match logits")
    target_ids = target_ids.to(device=logits.device)
    if target_ids.is_floating_point() or target_ids.dtype == torch.bool:
        raise ValueError("Stage 2B selected target IDs must be integers")
    if bool((target_ids < 0).any()) or bool((target_ids >= logits.shape[-1]).any()):
        raise ValueError("Stage 2B selected target IDs exceed the vocabulary")
    values = torch.log_softmax(logits.float(), dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    )
    score = values.mean()
    if not bool(torch.isfinite(score)):
        raise ValueError("Stage 2B policy score must be finite")
    return score


def _selected_entropies(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    values = -(log_probs.exp() * log_probs).sum(dim=-1).reshape(-1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Stage 2B target entropies must be finite")
    return values


class _ObservedOptimizerStep(TrainerCallback):
    def __init__(self, owner):
        self.owner = owner

    def on_train_begin(self, args, state, control, **kwargs):
        del args, state, kwargs
        self.owner._install_deepspeed_gradient_adapter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        del args, control, kwargs
        if not getattr(self.owner, "_stage2_deepspeed_gradient_adapter", None):
            self.owner._observed_optimizer_step = int(state.global_step)


class _Stage2DeepSpeedGradientAdapter:
    """Observe the pinned ZeRO norm immediately before its real step."""

    def __init__(self, trainer, engine, original_step):
        self._trainer = weakref.ref(trainer)
        self.engine = engine
        self.original_step = original_step

    def __call__(self, *args, **kwargs):
        trainer = self._trainer()
        if trainer is None:
            raise RuntimeError("Stage 2 trainer disappeared before DeepSpeed step")
        boundary = self.engine.is_gradient_accumulation_boundary()
        if type(boundary) is not bool:
            raise RuntimeError("Stage 2 DeepSpeed accumulation boundary is invalid")
        if boundary:
            zero_stage = trainer.deepspeed_runtime_profile.zero_stage
            if zero_stage is None:
                raise RuntimeError("Stage 2 DeepSpeed profile has no ZeRO stage")
            norm = collect_deepspeed_pre_step_norm(
                self.engine.optimizer, zero_stage=zero_stage
            )
            trainer._representative_gradient_norm = float(norm.cpu().item())
            value = trainer._representative_gradient_norm
            if not math.isfinite(value):
                trainer._nonfinite_pre_step_gradient_count = getattr(trainer, "_nonfinite_pre_step_gradient_count", 0) + 1
            elif value > 0:
                trainer._positive_pre_step_gradient_count = getattr(trainer, "_positive_pre_step_gradient_count", 0) + 1
            if not hasattr(trainer, "_observed_pre_step_gradient_norms"):
                trainer._observed_pre_step_gradient_norms = []
            if math.isfinite(value) and value > 0 and len(trainer._observed_pre_step_gradient_norms) < 2:
                trainer._observed_pre_step_gradient_norms.append(value)
        skipped_before = getattr(self.engine, "skipped_steps", 0)
        result = self.original_step(*args, **kwargs)
        if boundary:
            if getattr(self.engine, "skipped_steps", 0) != skipped_before:
                raise RuntimeError("Stage 2 DeepSpeed skipped an optimizer step")
            observed_step = getattr(trainer, "_observed_optimizer_step", None)
            if observed_step is None:
                observed_step = getattr(getattr(trainer, "state", None), "global_step", 0)
            trainer._observed_optimizer_step = observed_step + 1
        return result


class Stage2Trainer(Trainer):
    """Add Stage 2 observability and PEFT sidecars to ordinary Trainer flows."""

    def __init__(
        self,
        *args,
        deepspeed_runtime_profile: DeepSpeedRuntimeProfile | None = None,
        stage2_peft_metadata=None,
        stage2_diagnostics_interval: int = 0,
        step_preference_config: StepPreferenceConfig | None = None,
        **kwargs,
    ):
        if deepspeed_runtime_profile is None:
            training_args = kwargs.get("args")
            if training_args is None and len(args) > 1:
                training_args = args[1]
            configured_deepspeed = getattr(training_args, "deepspeed", None)
            accelerate_deepspeed = os.environ.get(
                "ACCELERATE_USE_DEEPSPEED", ""
            ).strip().lower() in {"1", "true", "yes"}
            if (
                configured_deepspeed is not None
                and configured_deepspeed is not False
            ) or accelerate_deepspeed:
                raise TypeError(
                    "Stage 2 DeepSpeed construction requires an explicit "
                    "deepspeed_runtime_profile"
                )
            deepspeed_runtime_profile = load_deepspeed_runtime_profile(
                None, world_size=int(os.environ.get("WORLD_SIZE", "1"))
            )
        if not isinstance(deepspeed_runtime_profile, DeepSpeedRuntimeProfile):
            raise TypeError("Stage 2 trainer requires a DeepSpeed runtime profile")
        self.deepspeed_runtime_profile = deepspeed_runtime_profile
        self.stage2_peft_metadata = stage2_peft_metadata
        self.stage2_sampler_identity: Stage2SamplerIdentity | None = None
        self._stage2_train_sampler: LatentBlockBalancedSampler | None = None
        self.stage2_diagnostics_interval = stage2_diagnostics_interval
        if step_preference_config is None:
            step_preference_config = StepPreferenceConfig()
        if not isinstance(step_preference_config, StepPreferenceConfig):
            raise TypeError("step_preference_config must be a StepPreferenceConfig")
        self.step_preference_config = step_preference_config
        self._step_preference_metric_totals: dict[str, float] = {}
        self._step_preference_metric_batches = 0
        self._stage2_microbatch_index = 0
        self._representative_gradient_norm: float | None = None
        self._observed_optimizer_step: int | None = None
        self._observed_pre_step_gradient_norms = []
        self._positive_pre_step_gradient_count = 0
        self._nonfinite_pre_step_gradient_count = 0
        self._runtime_objective_observation_counts = {}
        self._stage2_deepspeed_gradient_adapter = None
        callbacks = list(kwargs.pop("callbacks", ()) or ())
        callbacks.append(_ObservedOptimizerStep(self))
        kwargs["callbacks"] = callbacks
        super().__init__(*args, **kwargs)

    def _install_deepspeed_gradient_adapter(self) -> None:
        from accelerate.utils import DistributedType

        if getattr(self.accelerator, "distributed_type", None) != DistributedType.DEEPSPEED:
            return
        wrapper = getattr(self.accelerator, "deepspeed_engine_wrapped", None)
        engine = getattr(wrapper, "engine", None)
        if engine is None or getattr(self, "model_wrapped", None) is not engine:
            raise RuntimeError("Stage 2 prepared DeepSpeed engine is unavailable")
        existing = self._stage2_deepspeed_gradient_adapter
        if existing is not None:
            if existing.engine is not engine:
                raise RuntimeError("Stage 2 DeepSpeed engine changed after preparation")
            return
        if any(
            not callable(getattr(engine, name, None))
            for name in ("step", "is_gradient_accumulation_boundary")
        ):
            raise RuntimeError("Stage 2 DeepSpeed step boundary is unavailable")
        adapter = _Stage2DeepSpeedGradientAdapter(self, engine, engine.step)
        engine.step = adapter
        self._stage2_deepspeed_gradient_adapter = adapter

    def _record_representative_gradient_norm(self, model) -> None:
        squared_norm = 0.0
        observed = False
        for parameter in model.parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            local_norm = float(parameter.grad.detach().float().norm(2).cpu().item())
            if not math.isfinite(local_norm):
                self._representative_gradient_norm = float("nan")
                return
            squared_norm += local_norm * local_norm
            observed = True
        if observed:
            self._representative_gradient_norm = math.sqrt(squared_norm)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(
            model, inputs, num_items_in_batch=num_items_in_batch
        )
        self._record_representative_gradient_norm(model)
        return loss

    def _publish_deepspeed_runtime_identity(
        self, directory: Path, *, label: str
    ) -> None:
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

    def _ensure_stage2_train_sampler(
        self,
    ) -> LatentBlockBalancedSampler | None:
        from step_preference.capacity_evidence import CapacitySampleRecorder
        if isinstance(getattr(self,"data_collator",None), CapacitySampleRecorder):
            self.stage2_sampler_identity = None
            self._stage2_train_sampler = None
            return None
        if self.args.world_size <= 1:
            self.stage2_sampler_identity = None
            self._stage2_train_sampler = None
            return None
        sampler = getattr(self, "_stage2_train_sampler", None)
        if (
            sampler is None
            or sampler.dataset is not self.train_dataset
            or sampler.num_replicas != self.args.world_size
            or sampler.seed != self.args.seed
            or sampler.identity.dataset_size != len(self.train_dataset)
        ):
            sampler = LatentBlockBalancedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                seed=self.args.seed,
            )
            self._stage2_train_sampler = sampler
        self.stage2_sampler_identity = sampler.identity
        return sampler

    def _ensure_stage2_sampler_identity(self) -> Stage2SamplerIdentity | None:
        sampler = self._ensure_stage2_train_sampler()
        return None if sampler is None else sampler.identity

    def _publish_stage2_sampler_identity(
        self, directory: Path, *, global_step: int, label: str
    ) -> None:
        identity = self._ensure_stage2_sampler_identity()
        if identity is None:
            return
        local_error: Exception | None = None
        if self.args.should_save:
            try:
                write_stage2_sampler_sidecar(
                    directory, identity, global_step=global_step
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

    def _get_train_sampler(self):
        from step_preference.capacity_evidence import CapacitySampleRecorder, require
        if isinstance(getattr(self,"data_collator",None), CapacitySampleRecorder):
            require(len(self.train_dataset) == 1 and self.train_dataset[0].sample_id == self.data_collator.sample_id
                    and self.args.world_size == 2 and self.args.max_steps == 2
                    and self.args.gradient_accumulation_steps == 2 and self.args.per_device_train_batch_size == 1,
                    "capacity sampler requires the authenticated singleton and exact two-step protocol")
            self.stage2_sampler_identity = None
            # Accelerate shards eight global draws into four per rank. One epoch
            # therefore contains both real accumulation-2 optimizer boundaries.
            return torch.utils.data.SubsetRandomSampler([0] * 8, generator=torch.Generator().manual_seed(self.args.seed))
        if self.args.world_size <= 1:
            self.stage2_sampler_identity = None
            self._stage2_train_sampler = None
            return super()._get_train_sampler()
        sampler = self._ensure_stage2_train_sampler()
        assert sampler is not None
        return sampler

    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        if self.stage2_sampler_identity is not None:
            self._publish_stage2_sampler_identity(
                Path(self.args.output_dir),
                global_step=0,
                label="Stage 2 sampler identity publication",
            )
        return dataloader

    def train(self, *args, resume_from_checkpoint=None, **kwargs):
        self._publish_deepspeed_runtime_identity(
            Path(self.args.output_dir), label="Stage 2 runtime identity publication"
        )
        if resume_from_checkpoint is True:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    "No valid checkpoint found in output directory "
                    f"({self.args.output_dir})"
                )
        elif resume_from_checkpoint is False:
            resume_from_checkpoint = None
        if resume_from_checkpoint is not None:
            if not isinstance(resume_from_checkpoint, (str, Path)):
                raise TypeError(
                    "Stage 2 resume_from_checkpoint must be a path, True, or None"
                )
            validate_deepspeed_resume_identity(
                Path(resume_from_checkpoint), self.deepspeed_runtime_profile
            )
            sampler_identity = self._ensure_stage2_sampler_identity()
            if sampler_identity is not None:
                validate_stage2_sampler_resume_identity(
                    Path(resume_from_checkpoint), sampler_identity
                )
        return super().train(
            *args, resume_from_checkpoint=resume_from_checkpoint, **kwargs
        )

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        checkpoint = Path(self._get_output_dir(trial=trial)) / (
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        )
        self._publish_deepspeed_runtime_identity(
            checkpoint, label="Stage 2 checkpoint runtime identity publication"
        )
        self._publish_stage2_sampler_identity(
            checkpoint,
            global_step=self.state.global_step,
            label="Stage 2 checkpoint sampler identity publication",
        )

    def _accumulate_step_preference_metrics(self, values) -> None:
        totals = getattr(self, "_step_preference_metric_totals", None)
        if totals is None:
            totals = {}
            self._step_preference_metric_totals = totals
            self._step_preference_metric_batches = 0
        config = getattr(self, "step_preference_config", None)
        required = (
            _CONTEXT_PREFERENCE_LOG_NAMES
            if isinstance(config, StepPreferenceConfig) and config.is_context_preference
            else _STEP_PREFERENCE_LOG_NAMES
        )
        missing = set(required) - set(values)
        if missing:
            raise ValueError(f"Stage 2B metrics are missing {sorted(missing)}")
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(f"Stage 2B metric {name} must be scalar")
                value = float(value.detach().float().cpu().item())
            else:
                value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Stage 2B metric {name} must be finite")
            totals[name] = totals.get(name, 0.0) + value
            counts = getattr(self, "_runtime_objective_observation_counts", None)
            if counts is None:
                counts = self._runtime_objective_observation_counts = {}
            counts[name] = counts.get(name, 0) + 1
        self._step_preference_metric_batches += 1

    def _reduce_context_objective(
        self, clean_ce, clean_scores, clean_entropies, results
    ):
        config = self.step_preference_config
        zero = clean_ce * 0.0
        pref, ent, margins, off_logps, off_entropies = {}, {}, {}, {}, {}
        for modality in ("visual", "textual"):
            if clean_scores:
                clean = torch.stack(clean_scores)
                off = torch.stack(
                    [
                        results[(step_id, modality)]["mean_logp"]
                        for step_id in range(len(clean_scores))
                    ]
                )
                pref[modality] = context_preference_loss(
                    clean, off, config.beta, config.preference_margin
                )
                sidecar_entropies = [
                    results[(step_id, modality)]["entropies"]
                    for step_id in range(len(clean_scores))
                ]
                if any(
                    sidecar.shape != clean_entropy.shape
                    for sidecar, clean_entropy in zip(
                        sidecar_entropies, clean_entropies
                    )
                ):
                    raise ValueError("sidecar entropy rows must match clean target rows")
                ent[modality] = torch.stack(
                    [
                        F.relu(
                            config.entropy_margin - (sidecar - clean_entropy)
                        ).mean()
                        for clean_entropy, sidecar in zip(
                            clean_entropies, sidecar_entropies
                        )
                    ]
                ).mean()
                margins[modality] = (clean - off).mean()
                off_logps[modality] = off.mean()
                off_entropies[modality] = torch.stack(
                    [entropy.mean() for entropy in sidecar_entropies]
                ).mean()
            else:
                pref[modality] = ent[modality] = margins[modality] = zero
                off_logps[modality] = off_entropies[modality] = zero
        preference = (pref["visual"] + pref["textual"]) / 2
        entropy = (ent["visual"] + ent["textual"]) / 2
        total = (
            clean_ce
            + config.lambda_preference * preference
            + config.lambda_entropy * entropy
        )
        metrics = {
            "loss/total": total,
            "loss/ce": clean_ce,
            "loss/context_preference": preference,
            "loss/entropy": entropy,
            "stats/logp_clean": torch.stack(clean_scores).mean()
            if clean_scores
            else zero,
            "stats/entropy_clean": torch.stack(
                [entropy.mean() for entropy in clean_entropies]
            ).mean()
            if clean_entropies
            else zero,
        }
        for modality, label in (("visual", "visual"), ("textual", "text")):
            metrics[f"loss/context_preference_{label}"] = pref[modality]
            metrics[f"loss/entropy_{label}"] = ent[modality]
            metrics[f"stats/context_margin_{label}"] = margins[modality]
            metrics[f"stats/logp_{label}_off"] = off_logps[modality]
            metrics[f"stats/entropy_{label}_off"] = off_entropies[modality]
        if not bool(torch.isfinite(total)):
            raise ValueError("context preference total loss must be finite")
        return total, metrics

    def _compute_context_preference_loss(
        self, model, inputs, *, collect_diagnostics: bool, return_outputs: bool
    ):
        config = self.step_preference_config
        clean_inputs = dict(inputs)
        sample_ids = clean_inputs.pop("sample_ids", None)
        spans = clean_inputs.pop("reasoning_step_spans", None)
        metadata = clean_inputs.pop("step_preference", None)
        if (
            not isinstance(sample_ids, (tuple, list))
            or len(sample_ids) != 1
            or not isinstance(sample_ids[0], str)
            or not sample_ids[0]
        ):
            raise ValueError("context preference requires one sample_id")
        if not isinstance(spans, (tuple, list)) or not isinstance(metadata, dict):
            raise ValueError("context preference requires reasoning step metadata")
        steps = metadata.get("steps")
        if not isinstance(steps, (tuple, list)) or len(steps) != len(spans):
            raise ValueError("context preference step cardinality mismatch")
        if any(
            not isinstance(item, dict)
            or set(item) != {"span"}
            or item["span"] != span
            for item, span in zip(steps, spans)
        ):
            raise ValueError("context preference requires clean span-only metadata")
        plan = build_context_sidecar_plan(
            clean_input_ids=clean_inputs["input_ids"],
            attention_mask=clean_inputs["attention_mask"],
            spans=spans,
        )
        call = dict(
            clean_inputs,
            use_cache=False,
            stage2_loss_only=not return_outputs,
            collect_stage2_diagnostics=collect_diagnostics,
            stage2_shared_prefix_inputs=dict(
                plan=plan,
                backend=config.sidecar_attention_backend,
                query_budget=config.sidecar_query_budget,
            ),
        )
        if plan.root_selected_positions:
            call["stage2_logit_positions"] = torch.tensor(
                plan.root_selected_positions,
                device=clean_inputs["input_ids"].device,
                dtype=torch.long,
            )
        outputs = model(**call)
        clean_ce = outputs.loss
        if (
            not isinstance(clean_ce, torch.Tensor)
            or clean_ce.numel() != 1
            or not bool(torch.isfinite(clean_ce))
        ):
            raise ValueError("clean CE must be a finite scalar")
        diagnostics = getattr(outputs, "stage2_diagnostics", None)
        self._latest_stage2_diagnostics = (
            None
            if diagnostics is None
            else {key: value.detach() for key, value in diagnostics.items()}
        )
        rows = (
            _selected_logits(outputs, len(plan.root_selected_positions))
            if spans
            else None
        )
        results = {}
        for result in outputs.stage2_sidecar_results:
            modality = {"visual_off": "visual", "textual_off": "textual"}.get(
                result["kind"]
            )
            key = (result["step_id"], modality)
            if modality is None or key in results:
                raise ValueError("unexpected or duplicate context sidecar")
            results[key] = result
        if set(results) != {
            (span.step_id, modality)
            for span in spans
            for modality in ("visual", "textual")
        }:
            raise ValueError("context sidecar layout mismatch")
        clean_scores, clean_entropies, offset = [], [], 0
        for span in spans:
            count = span.target_text_end - span.target_text_start
            logits = rows[:, offset : offset + count]
            targets = clean_inputs["input_ids"][
                :, span.target_text_start : span.target_text_end
            ]
            clean_scores.append(_mean_selected_logp(logits, targets))
            clean_entropies.append(_selected_entropies(logits))
            offset += count
        total, metrics = self._reduce_context_objective(
            clean_ce, clean_scores, clean_entropies, results
        )
        metrics.update(
            {
                "stats/scored_tokens": 3 * len(plan.root_selected_positions),
                "stats/logical_scoring_groups": 1
                + sum(bool(group) for group in plan.groups),
            }
        )
        self._accumulate_step_preference_metrics(metrics)
        return (total, outputs) if return_outputs else total

    def _reduce_stage2b_objective(
        self,
        *,
        clean_ce,
        chosen_scores,
        visual_rejected_scores,
        textual_rejected_scores,
        reference_chosen,
        reference_visual_rejected,
        reference_textual_rejected,
        valid_visual,
        valid_textual,
        visual_entropy_losses,
        textual_entropy_losses,
        clean_entropy_values,
        visual_entropy_values,
        textual_entropy_values,
        branch_forwards,
        scored_tokens,
        extra_metrics=None,
    ):
        """Reduce pre-scored branches with the normative Stage 2B objective."""
        config = self.step_preference_config
        zero = clean_ce * 0.0
        if chosen_scores:
            dpo = aggregate_step_dpo(
                clean_ce=clean_ce,
                policy_chosen=torch.stack(chosen_scores),
                policy_visual_rejected=torch.stack(visual_rejected_scores),
                policy_textual_rejected=torch.stack(textual_rejected_scores),
                reference_chosen=torch.stack(reference_chosen),
                reference_visual_rejected=torch.stack(reference_visual_rejected),
                reference_textual_rejected=torch.stack(reference_textual_rejected),
                valid_visual=torch.tensor(
                    valid_visual, dtype=torch.bool, device=clean_ce.device
                ),
                valid_textual=torch.tensor(
                    valid_textual, dtype=torch.bool, device=clean_ce.device
                ),
                beta=config.beta,
            )
            visual_entropy_tokens = torch.cat(visual_entropy_losses)
            textual_entropy_tokens = torch.cat(textual_entropy_losses)
            entropy_loss = torch.cat(
                (visual_entropy_tokens, textual_entropy_tokens)
            ).mean()
            visual_entropy_loss = visual_entropy_tokens.mean()
            textual_entropy_loss = textual_entropy_tokens.mean()
            clean_entropy = torch.cat(clean_entropy_values).mean()
            visual_off_entropy = torch.cat(visual_entropy_values).mean()
            textual_off_entropy = torch.cat(textual_entropy_values).mean()
        else:
            dpo = SimpleNamespace(
                loss=zero,
                visual_loss=zero,
                textual_loss=zero,
                visual_margin=zero,
                textual_margin=zero,
                visual_pair_count=0,
                textual_pair_count=0,
            )
            entropy_loss = zero
            visual_entropy_loss = zero
            textual_entropy_loss = zero
            clean_entropy = zero
            visual_off_entropy = zero
            textual_off_entropy = zero

        total_loss = (
            config.lambda_ce * clean_ce
            + config.lambda_dpo * dpo.loss
            + config.lambda_entropy * entropy_loss
        )
        if not bool(torch.isfinite(total_loss)):
            raise ValueError("Stage 2B total loss must be finite")
        metrics = {
            "loss/total": total_loss,
            "loss/ce": clean_ce,
            "loss/dpo": dpo.loss,
            "loss/entropy": entropy_loss,
            "loss/dpo_visual": dpo.visual_loss,
            "loss/dpo_text": dpo.textual_loss,
            "loss/entropy_visual": visual_entropy_loss,
            "loss/entropy_text": textual_entropy_loss,
            "stats/dpo_margin_visual": dpo.visual_margin,
            "stats/dpo_margin_text": dpo.textual_margin,
            "stats/entropy_clean": clean_entropy,
            "stats/entropy_visual_off": visual_off_entropy,
            "stats/entropy_text_off": textual_off_entropy,
            "stats/valid_visual_pairs": dpo.visual_pair_count,
            "stats/valid_textual_pairs": dpo.textual_pair_count,
            "stats/branch_forwards": branch_forwards,
            "stats/scored_tokens": scored_tokens,
        }
        if extra_metrics:
            metrics.update(extra_metrics)
        self._accumulate_step_preference_metrics(metrics)
        return total_loss

    def _compute_stage2b_shared_prefix_loss(
        self, model, *, clean_inputs, reasoning_spans, steps, validated_steps,
        collect_diagnostics, return_outputs,
    ):
        config = self.step_preference_config
        input_ids = clean_inputs["input_ids"]
        plan = build_sidecar_plan(clean_input_ids=input_ids,
            attention_mask=clean_inputs["attention_mask"], spans=reasoning_spans, steps=steps)
        consumed = 0
        missing = 0
        for _, negative, _ in validated_steps:
            scores = [negative.reference_chosen_score]
            for modality in ("visual", "textual"):
                if getattr(negative, f"valid_{modality}_negative"):
                    scores.append(getattr(negative, f"reference_{modality}_rejected_score"))
            for value in scores:
                if value is None or not math.isfinite(value):
                    missing += 1
                else:
                    consumed += 1
        self._latest_stage2_reference_cache_stats = dict(consumed=consumed, missing=missing)
        if missing:
            raise ValueError(f"Stage 2B shared prefix is missing {missing} valid cached reference scores")
        call = dict(clean_inputs, use_cache=False, stage2_loss_only=not return_outputs,
            collect_stage2_diagnostics=collect_diagnostics,
            stage2_shared_prefix_inputs=dict(plan=plan, backend=config.sidecar_attention_backend,
                                            query_budget=config.sidecar_query_budget))
        if plan.root_selected_positions:
            call["stage2_logit_positions"] = torch.tensor(plan.root_selected_positions,
                dtype=torch.long, device=input_ids.device)
        outputs = model(**call)
        clean_ce = outputs.loss
        if not isinstance(clean_ce, torch.Tensor) or clean_ce.numel() != 1 or not bool(torch.isfinite(clean_ce)):
            raise ValueError("Stage 2B clean CE must be a finite scalar")
        diagnostics = getattr(outputs, "stage2_diagnostics", None)
        self._latest_stage2_diagnostics = None if diagnostics is None else {
            name: value.detach() for name, value in diagnostics.items()}
        selected = _selected_logits(outputs, len(plan.root_selected_positions)) if validated_steps else None
        results = {result["name"]: result for result in outputs.stage2_sidecar_results}
        expected_names = {branch.name for group in plan.groups for branch in group}
        if results.keys() != expected_names or len(results) != len(outputs.stage2_sidecar_results):
            raise ValueError("shared-prefix sidecar score layout is inconsistent")
        values = {name: [] for name in (
            "chosen_scores", "visual_rejected_scores", "textual_rejected_scores",
            "reference_chosen", "reference_visual_rejected", "reference_textual_rejected",
            "valid_visual", "valid_textual", "visual_entropy_losses", "textual_entropy_losses",
            "clean_entropy_values", "visual_entropy_values", "textual_entropy_values")}
        offset = 0
        zero = clean_ce * 0.0
        for span, negative, _ in validated_steps:
            count = span.target_text_end - span.target_text_start
            logits = selected[:, offset:offset+count]
            offset += count
            values["chosen_scores"].append(_mean_selected_logp(logits,
                input_ids[:, span.target_text_start:span.target_text_end]))
            clean_entropy = _selected_entropies(logits)
            values["clean_entropy_values"].extend((clean_entropy, clean_entropy))
            values["reference_chosen"].append(clean_ce.new_tensor(negative.reference_chosen_score))
            for modality in ("visual", "textual"):
                entropy = results[f"step-{span.step_id}-{modality}-off"]["entropies"]
                if entropy.shape != clean_entropy.shape or not bool(torch.isfinite(entropy).all()):
                    raise ValueError("shared-prefix entropy must match clean target rows and be finite")
                values[f"{modality}_entropy_values"].append(entropy)
                values[f"{modality}_entropy_losses"].append(F.relu(clean_entropy + config.entropy_margin - entropy))
                valid = getattr(negative, f"valid_{modality}_negative")
                values[f"valid_{modality}"].append(valid)
                score = results[f"step-{span.step_id}-{modality}-rejected"]["mean_logp"] if valid else zero
                if score.numel() != 1 or not bool(torch.isfinite(score)):
                    raise ValueError("shared-prefix rejected score must be a finite scalar")
                values[f"{modality}_rejected_scores"].append(score)
                reference = getattr(negative, f"reference_{modality}_rejected_score") if valid else 0.0
                values[f"reference_{modality}_rejected"].append(clean_ce.new_tensor(reference))
        stats = outputs.stage2_sidecar_stats
        total = self._reduce_stage2b_objective(clean_ce=clean_ce, **values,
            branch_forwards=1,
            scored_tokens=len(plan.root_selected_positions) + stats["entropy_scored_rows"] + stats["rejected_scored_rows"],
            extra_metrics={**{f"stats/sidecar_{key}": value for key, value in stats.items()},
                "stats/reference_cache_entries_consumed": consumed,
                "stats/reference_cache_entries_missing": missing})
        if return_outputs:
            outputs.loss = total
            return total, outputs
        return total

    def _compute_stage2b_packed_loss(
        self,
        model,
        *,
        clean_inputs,
        reasoning_spans,
        steps,
        validated_steps,
        collect_diagnostics,
        return_outputs,
    ):
        input_ids = clean_inputs["input_ids"]
        attention_mask = clean_inputs["attention_mask"]
        plan = build_packed_tree_plan(
            clean_input_ids=input_ids,
            attention_mask=attention_mask,
            reasoning_spans=reasoning_spans,
            joined_steps=steps,
        )
        packed_inputs = plan.to_model_inputs(input_ids.device)
        selected_positions = plan.selected_positions().to(input_ids.device)
        call = dict(clean_inputs)
        call.update(
            use_cache=False,
            stage2_loss_only=not return_outputs,
            collect_stage2_diagnostics=collect_diagnostics,
            stage2_packed_inputs=packed_inputs,
            stage2_logit_positions=selected_positions,
        )
        outputs = model(**call)
        clean_ce = getattr(outputs, "loss", None)
        if (
            not isinstance(clean_ce, torch.Tensor)
            or clean_ce.numel() != 1
            or not bool(torch.isfinite(clean_ce))
        ):
            raise ValueError("Stage 2B clean CE must be a finite scalar")
        diagnostics = getattr(outputs, "stage2_diagnostics", None)
        self._latest_stage2_diagnostics = (
            None
            if diagnostics is None
            else {name: value.detach() for name, value in diagnostics.items()}
        )
        selected = _selected_logits(outputs, selected_positions.numel())
        root_count = len(plan.root_selected_positions)
        branch_logits = {}
        offset = root_count
        for branch in plan.branches:
            count = len(branch.selected_positions)
            branch_logits[branch.name] = selected[:, offset : offset + count]
            offset += count
        if offset != selected.shape[1]:
            raise ValueError("packed Stage 2B selected-logit layout is inconsistent")

        chosen_scores = []
        visual_rejected_scores = []
        textual_rejected_scores = []
        reference_chosen = []
        reference_visual_rejected = []
        reference_textual_rejected = []
        valid_visual = []
        valid_textual = []
        visual_entropy_losses = []
        textual_entropy_losses = []
        clean_entropy_values = []
        visual_entropy_values = []
        textual_entropy_values = []
        root_offset = 0
        zero = clean_ce * 0.0
        for span, negative, _item in validated_steps:
            token_count = span.target_text_end - span.target_text_start
            clean_logits = selected[:, root_offset : root_offset + token_count]
            root_offset += token_count
            target_ids = input_ids[:, span.target_text_start : span.target_text_end]
            chosen_scores.append(_mean_selected_logp(clean_logits, target_ids))
            clean_entropy = _selected_entropies(clean_logits)
            modality_entropies = {}
            for modality in ("visual", "textual"):
                branch = plan.branch(f"step-{span.step_id}-{modality}-off")
                modality_entropies[modality] = _selected_entropies(
                    branch_logits[branch.name]
                )
            visual_entropy_losses.append(
                F.relu(
                    clean_entropy
                    + self.step_preference_config.entropy_margin
                    - modality_entropies["visual"]
                )
            )
            textual_entropy_losses.append(
                F.relu(
                    clean_entropy
                    + self.step_preference_config.entropy_margin
                    - modality_entropies["textual"]
                )
            )
            clean_entropy_values.extend((clean_entropy, clean_entropy))
            visual_entropy_values.append(modality_entropies["visual"])
            textual_entropy_values.append(modality_entropies["textual"])
            valid_visual.append(negative.valid_visual_negative)
            valid_textual.append(negative.valid_textual_negative)
            for modality, destination in (
                ("visual", visual_rejected_scores),
                ("textual", textual_rejected_scores),
            ):
                if getattr(negative, f"valid_{modality}_negative"):
                    branch = plan.branch(
                        f"step-{span.step_id}-{modality}-rejected"
                    )
                    logits = branch_logits[branch.name]
                    targets = torch.tensor(
                        branch.target_token_ids,
                        dtype=torch.long,
                        device=logits.device,
                    ).unsqueeze(0)
                    destination.append(_mean_selected_logp(logits, targets))
                else:
                    destination.append(zero)
            reference_chosen.append(
                clean_ce.new_tensor(negative.reference_chosen_score).detach()
            )
            reference_visual_rejected.append(
                clean_ce.new_tensor(
                    negative.reference_visual_rejected_score
                    if negative.reference_visual_rejected_score is not None
                    else 0.0
                ).detach()
            )
            reference_textual_rejected.append(
                clean_ce.new_tensor(
                    negative.reference_textual_rejected_score
                    if negative.reference_textual_rejected_score is not None
                    else 0.0
                ).detach()
            )

        block_mask = packed_inputs["block_mask"]
        sparsity = getattr(block_mask, "sparsity", None)
        sparsity_value = float(sparsity()) if callable(sparsity) else 0.0
        total_loss = self._reduce_stage2b_objective(
            clean_ce=clean_ce,
            chosen_scores=chosen_scores,
            visual_rejected_scores=visual_rejected_scores,
            textual_rejected_scores=textual_rejected_scores,
            reference_chosen=reference_chosen,
            reference_visual_rejected=reference_visual_rejected,
            reference_textual_rejected=reference_textual_rejected,
            valid_visual=valid_visual,
            valid_textual=valid_textual,
            visual_entropy_losses=visual_entropy_losses,
            textual_entropy_losses=textual_entropy_losses,
            clean_entropy_values=clean_entropy_values,
            visual_entropy_values=visual_entropy_values,
            textual_entropy_values=textual_entropy_values,
            branch_forwards=1,
            scored_tokens=selected_positions.numel(),
            extra_metrics={
                "stats/packed_tokens": plan.padded_length,
                "stats/packed_mask_sparsity": sparsity_value,
            },
        )
        if return_outputs:
            outputs.loss = total_loss
            return total_loss, outputs
        return total_loss

    def _compute_stage2b_loss(
        self, model, inputs, *, collect_diagnostics: bool, return_outputs: bool
    ):
        config = self.step_preference_config
        clean_inputs = dict(inputs)
        sample_ids = clean_inputs.pop("sample_ids", None)
        reasoning_spans = clean_inputs.pop("reasoning_step_spans", None)
        preference = clean_inputs.pop("step_preference", None)
        if (
            not isinstance(sample_ids, (tuple, list))
            or len(sample_ids) != 1
            or not isinstance(sample_ids[0], str)
            or not sample_ids[0]
        ):
            raise ValueError("Stage 2B requires one sample_id")
        if not isinstance(reasoning_spans, (tuple, list)):
            raise ValueError("Stage 2B requires reasoning_step_spans")
        if not isinstance(preference, dict):
            raise ValueError("Stage 2B requires private step_preference metadata")
        steps = preference.get("steps")
        if not isinstance(steps, (tuple, list)) or len(steps) != len(reasoning_spans):
            raise ValueError("Stage 2B reasoning step metadata cardinality mismatch")
        if "input_ids" not in clean_inputs or "attention_mask" not in clean_inputs:
            raise ValueError("Stage 2B requires input_ids and attention_mask")
        input_ids = clean_inputs["input_ids"]
        attention_mask = clean_inputs["attention_mask"]
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or not isinstance(attention_mask, torch.Tensor)
            or attention_mask.shape != input_ids.shape
        ):
            raise ValueError("Stage 2B clean tensors must have shape [1, sequence]")

        validated_steps = []
        all_query_positions: list[int] = []
        sequence_length = input_ids.shape[1]
        for index, (span, item) in enumerate(zip(reasoning_spans, steps)):
            if not isinstance(span, ReasoningStepSpan) or span.step_id != index:
                raise ValueError("Stage 2B reasoning step IDs must be ordered and contiguous")
            if not isinstance(item, dict) or item.get("span") != span:
                raise ValueError("Stage 2B private reasoning step span does not match")
            negative = item.get("negative")
            if (
                not isinstance(negative, StepNegative)
                or negative.sample_id != sample_ids[0]
                or negative.step_id != span.step_id
            ):
                raise ValueError("Stage 2B negative join identity does not match")
            if (
                span.target_text_start < 1
                or span.target_text_end <= span.target_text_start
                or span.target_text_end > sequence_length
            ):
                raise ValueError("Stage 2B target span has no scoreable causal rows")
            all_query_positions.extend(
                range(span.target_text_start - 1, span.target_text_end - 1)
            )
            for modality in ("visual", "textual"):
                valid = getattr(negative, f"valid_{modality}_negative")
                encoded = item.get(f"{modality}_rejected")
                if valid != (encoded is not None):
                    raise ValueError(
                        f"Stage 2B {modality} rejected branch validity does not match"
                    )
            validated_steps.append((span, negative, item))

        if config.shared_prefix_enabled:
            return self._compute_stage2b_shared_prefix_loss(
                model, clean_inputs=clean_inputs, reasoning_spans=reasoning_spans,
                steps=steps, validated_steps=validated_steps,
                collect_diagnostics=collect_diagnostics, return_outputs=return_outputs,
            )
        if config.packing_enabled:
            return self._compute_stage2b_packed_loss(
                model,
                clean_inputs=clean_inputs,
                reasoning_spans=reasoning_spans,
                steps=steps,
                validated_steps=validated_steps,
                collect_diagnostics=collect_diagnostics,
                return_outputs=return_outputs,
            )

        clean_call = dict(clean_inputs)
        clean_call.update(
            stage2_loss_only=not return_outputs,
            collect_stage2_diagnostics=collect_diagnostics,
        )
        if all_query_positions:
            clean_call["stage2_logit_positions"] = torch.tensor(
                all_query_positions, dtype=torch.long, device=input_ids.device
            )
        clean_outputs = model(**clean_call)
        clean_ce = getattr(clean_outputs, "loss", None)
        if (
            not isinstance(clean_ce, torch.Tensor)
            or clean_ce.numel() != 1
            or not bool(torch.isfinite(clean_ce))
        ):
            raise ValueError("Stage 2B clean CE must be a finite scalar")
        diagnostics = getattr(clean_outputs, "stage2_diagnostics", None)
        self._latest_stage2_diagnostics = (
            None
            if diagnostics is None
            else {name: value.detach() for name, value in diagnostics.items()}
        )

        zero = clean_ce * 0.0
        if all_query_positions:
            clean_selected = _selected_logits(
                clean_outputs, len(all_query_positions)
            )
        else:
            clean_selected = None
        clean_offset = 0
        chosen_scores = []
        visual_rejected_scores = []
        textual_rejected_scores = []
        reference_chosen = []
        reference_visual_rejected = []
        reference_textual_rejected = []
        valid_visual = []
        valid_textual = []
        visual_entropy_losses = []
        textual_entropy_losses = []
        clean_entropy_values = []
        visual_entropy_values = []
        textual_entropy_values = []
        branch_forwards = 1
        scored_tokens = len(all_query_positions)
        model_dtype = _model_compute_dtype(model)

        def rejected_score(encoded) -> torch.Tensor:
            nonlocal branch_forwards, scored_tokens
            if not isinstance(encoded, dict):
                raise ValueError("Stage 2B rejected encoding must be a mapping")
            model_inputs = encoded.get("model_inputs")
            completion_positions = encoded.get("completion_token_positions")
            if not isinstance(model_inputs, dict) or not isinstance(
                completion_positions, (tuple, list)
            ):
                raise ValueError("Stage 2B rejected encoding is incomplete")
            completion_positions = tuple(completion_positions)
            rejected_ids = model_inputs.get("input_ids")
            if (
                not completion_positions
                or tuple(sorted(set(completion_positions))) != completion_positions
                or not isinstance(rejected_ids, torch.Tensor)
                or rejected_ids.ndim != 2
                or rejected_ids.shape[0] != 1
                or completion_positions[0] < 1
                or completion_positions[-1] >= rejected_ids.shape[1]
                or "labels" in model_inputs
            ):
                raise ValueError("Stage 2B rejected completion positions are invalid")
            query_positions = torch.tensor(
                [position - 1 for position in completion_positions],
                dtype=torch.long,
                device=rejected_ids.device,
            )
            call = dict(model_inputs)
            call.update(
                use_cache=False,
                stage2_final_attention_mask=None,
                stage2_logit_positions=query_positions,
                stage2_selective_logits_only=True,
            )
            outputs = model(**call)
            logits = _selected_logits(outputs, len(completion_positions))
            branch_forwards += 1
            scored_tokens += len(completion_positions)
            return _mean_selected_logp(
                logits, rejected_ids[:, list(completion_positions)]
            )

        for span, negative, item in validated_steps:
            token_count = span.target_text_end - span.target_text_start
            assert clean_selected is not None
            step_clean_logits = clean_selected[
                :, clean_offset : clean_offset + token_count
            ]
            clean_offset += token_count
            target_ids = input_ids[:, span.target_text_start : span.target_text_end]
            chosen_score = _mean_selected_logp(step_clean_logits, target_ids)
            chosen_scores.append(chosen_score)
            step_clean_entropy = _selected_entropies(step_clean_logits)

            modality_entropy = {}
            for modality in ("visual", "textual"):
                source_start, source_end = (
                    (span.latent_start, span.latent_end)
                    if modality == "visual"
                    else (span.prev_text_start, span.prev_text_end)
                )
                spec = StepAblationSpec(
                    modality=modality,
                    target_start=span.target_text_start,
                    target_end=span.target_text_end,
                    source_start=source_start,
                    source_end=source_end,
                )
                final_mask = build_additive_step_attention_mask(
                    attention_mask,
                    spec,
                    dtype=model_dtype,
                    device=input_ids.device,
                    attn_implementation=config.attention_implementation,
                )
                branch_inputs = {
                    name: value for name, value in clean_inputs.items() if name != "labels"
                }
                branch_inputs.update(
                    use_cache=False,
                    stage2_final_attention_mask=final_mask,
                    stage2_logit_positions=torch.arange(
                        span.target_text_start - 1,
                        span.target_text_end - 1,
                        dtype=torch.long,
                        device=input_ids.device,
                    ),
                    stage2_selective_logits_only=True,
                )
                outputs = model(**branch_inputs)
                ablated_logits = _selected_logits(outputs, token_count)
                ablated_entropy = _selected_entropies(ablated_logits)
                modality_entropy[modality] = ablated_entropy
                branch_forwards += 1
                scored_tokens += token_count

            visual_entropy_losses.append(
                F.relu(
                    step_clean_entropy
                    + config.entropy_margin
                    - modality_entropy["visual"]
                )
            )
            textual_entropy_losses.append(
                F.relu(
                    step_clean_entropy
                    + config.entropy_margin
                    - modality_entropy["textual"]
                )
            )
            # Clean target entropy participates once per modality in the global
            # token denominator, matching the two independent ablation branches.
            clean_entropy_values.extend((step_clean_entropy, step_clean_entropy))
            visual_entropy_values.append(modality_entropy["visual"])
            textual_entropy_values.append(modality_entropy["textual"])

            valid_visual.append(negative.valid_visual_negative)
            valid_textual.append(negative.valid_textual_negative)
            visual_rejected_scores.append(
                rejected_score(item["visual_rejected"])
                if negative.valid_visual_negative
                else zero
            )
            textual_rejected_scores.append(
                rejected_score(item["textual_rejected"])
                if negative.valid_textual_negative
                else zero
            )
            reference_chosen.append(
                clean_ce.new_tensor(negative.reference_chosen_score).detach()
            )
            reference_visual_rejected.append(
                clean_ce.new_tensor(
                    negative.reference_visual_rejected_score
                    if negative.reference_visual_rejected_score is not None
                    else 0.0
                ).detach()
            )
            reference_textual_rejected.append(
                clean_ce.new_tensor(
                    negative.reference_textual_rejected_score
                    if negative.reference_textual_rejected_score is not None
                    else 0.0
                ).detach()
            )

        total_loss = self._reduce_stage2b_objective(
            clean_ce=clean_ce,
            chosen_scores=chosen_scores,
            visual_rejected_scores=visual_rejected_scores,
            textual_rejected_scores=textual_rejected_scores,
            reference_chosen=reference_chosen,
            reference_visual_rejected=reference_visual_rejected,
            reference_textual_rejected=reference_textual_rejected,
            valid_visual=valid_visual,
            valid_textual=valid_textual,
            visual_entropy_losses=visual_entropy_losses,
            textual_entropy_losses=textual_entropy_losses,
            clean_entropy_values=clean_entropy_values,
            visual_entropy_values=visual_entropy_values,
            textual_entropy_values=textual_entropy_values,
            branch_forwards=branch_forwards,
            scored_tokens=scored_tokens,
        )
        if return_outputs:
            clean_outputs.loss = total_loss
            return total_loss, clean_outputs
        return total_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        interval = self.stage2_diagnostics_interval
        collect_diagnostics = (
            interval > 0 and self._stage2_microbatch_index % interval == 0
        )
        self._stage2_microbatch_index += 1
        forbidden = {"pixel_values_latent", "image_grid_thw_latent"} & inputs.keys()
        if forbidden:
            raise ValueError(
                f"helper-image tensors leaked into Stage 2: {sorted(forbidden)}"
            )
        config = getattr(self, "step_preference_config", None)
        if isinstance(config, StepPreferenceConfig) and config.is_context_preference:
            return self._compute_context_preference_loss(
                model,
                inputs,
                collect_diagnostics=collect_diagnostics,
                return_outputs=return_outputs,
            )
        if isinstance(config, StepPreferenceConfig) and config.is_stage2b:
            return self._compute_stage2b_loss(
                model,
                inputs,
                collect_diagnostics=collect_diagnostics,
                return_outputs=return_outputs,
            )
        outputs = model(
            **inputs,
            stage2_loss_only=not return_outputs,
            collect_stage2_diagnostics=collect_diagnostics,
        )
        diagnostics = getattr(outputs, "stage2_diagnostics", None)
        self._latest_stage2_diagnostics = (
            None
            if diagnostics is None
            else {name: value.detach() for name, value in diagnostics.items()}
        )
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def log(self, logs, start_time=None):
        diagnostics = getattr(self, "_latest_stage2_diagnostics", None)
        self._latest_stage2_diagnostics = None
        if diagnostics is not None:
            logs = dict(logs)
            logs["stage2/latent_block_count"] = float(
                diagnostics["block_count"].item()
            )
            norms = diagnostics["latent_l2_norms"]
            if norms.numel():
                logs.update(
                    {
                        "stage2/latent_l2_norm_mean": float(norms.mean().item()),
                        "stage2/latent_l2_norm_min": float(norms.min().item()),
                        "stage2/latent_l2_norm_max": float(norms.max().item()),
                    }
                )
            adjacent = diagnostics["adjacent_cosine_similarities"]
            if adjacent.numel():
                logs["stage2/adjacent_cosine_mean"] = float(adjacent.mean().item())
        metric_batches = getattr(self, "_step_preference_metric_batches", 0)
        if metric_batches:
            logs = dict(logs)
            totals = self._step_preference_metric_totals
            logs.update(
                {
                    name: totals[name] / metric_batches
                    for name in totals
                }
            )
            self._step_preference_metric_totals = {}
            self._step_preference_metric_batches = 0
        super().log(logs, start_time=start_time)

    def _save(self, output_dir=None, state_dict=None):
        """Save Stage 2 adapters without PEFT's automatic embedding export."""

        model = self.accelerator.unwrap_model(self.model)
        if not isinstance(model, PeftModel):
            return super()._save(output_dir, state_dict=state_dict)

        target = Path(output_dir or self.args.output_dir)
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(
            target,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
            save_embedding_layers=False,
        )
        resolve_qwen2_5_vl_model(model).config.save_pretrained(target)
        from step_preference.capacity_evidence import CapacitySampleRecorder
        defer_processor = (
            isinstance(getattr(self, "data_collator", None), CapacitySampleRecorder)
            and target.resolve() == Path(self.args.output_dir).resolve()
        )
        if self.processing_class is not None and not defer_processor:
            self.processing_class.save_pretrained(target)
        torch.save(self.args, target / TRAINING_ARGS_NAME)

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)
        metadata = self.stage2_peft_metadata
        if metadata is None:
            return
        target = Path(output_dir or self.args.output_dir)
        if metadata.token_row_source == "sparse_adapter":
            if sparse_token_rows_are_partitioned(self.model):
                # The sparse ZeRO gather is collective. Non-writing ranks use a
                # private temporary destination while rank zero publishes.
                if self.args.should_save:
                    save_sparse_token_rows(self.model, target)
                else:
                    with tempfile.TemporaryDirectory(
                        prefix="stage2-token-rows-"
                    ) as temporary:
                        save_sparse_token_rows(self.model, Path(temporary))
            elif self.args.should_save:
                save_sparse_token_rows(self.model, target)
        if self.args.should_save:
            save_stage2_peft_metadata(target, metadata)
