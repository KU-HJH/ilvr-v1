"""Thin Hugging Face Trainer adapter for Stage 2 model-native loss."""

from dataclasses import replace
import os
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from transformers.models.qwen2_5_vl.stage2_sidecar import (
    Stage2ComputeStats,
    Stage2SidecarLogits,
)

from deepspeed_runtime import (
    DeepSpeedRuntimeProfile,
    load_deepspeed_runtime_profile,
    validate_deepspeed_resume_identity,
    write_deepspeed_runtime_sidecar,
)
from .checkpointing import (
    STEP_PREFERENCE_METADATA_FILENAME,
    StepPreferenceMetadata,
    load_step_preference_metadata,
    save_stage2_peft_metadata,
    save_step_preference_metadata,
    validate_step_preference_resume_identity,
)
from .peft import resolve_stage2_qwen_model, validate_gated_execution_peft_identity
from .dora import dora_microbatch_scope
from .data import StepPreferenceBatchItem, immediate_text_target_indices
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
from step_preference.config import StepPreferenceConfig
from step_preference.insertion import InsertionRequest
from step_preference.gated import GatedRequest
from transformers.models.qwen2_5_vl.stage2_gated_execution import GatedScores
from transformers.models.qwen2_5_vl.stage2_insertion import Stage2InsertionOutput
from step_preference.dpo_loss import differentiable_zero, mean_token_logps, context_preference_loss
from step_preference.entropy_loss import (
    entropy_margin_loss,
    selected_token_entropies,
)
from step_preference.negatives import StepNegativeIndex
from step_preference.packing import build_context_sidecar_execution_plan


_STAGE2_PYTHON_METADATA_KEYS = frozenset(
    {"reasoning_step_spans", "sample_ids", "step_preference_items"}
)
_STEP_PREFERENCE_DIAGNOSTIC_KEYS = (
    "loss/ce",
    "loss/context_preference_all",
    "loss/entropy_all",
    "loss/total",
    "loss/context_preference_visual",
    "loss/context_preference_text",
    "loss/entropy_visual",
    "loss/entropy_text",
    "stats/context_margin_visual",
    "stats/context_margin_text",
    "stats/entropy_clean",
    "stats/entropy_visual_off",
    "stats/entropy_text_off",
)

_INSERTION_DIAGNOSTIC_KEYS = (
    "loss/ce", "loss/context_preference_all", "loss/entropy_all", "loss/total",
    "loss/context_preference_visual", "loss/entropy_visual",
    "stats/context_margin_visual", "stats/logp_evidence_on", "stats/logp_evidence_off",
    "stats/entropy_evidence_on", "stats/entropy_evidence_off",
)


class Stage2Trainer(Trainer):
    """Add Stage 2 observability and PEFT sidecars to ordinary Trainer flows."""

    def __init__(
        self,
        *args,
        deepspeed_runtime_profile: DeepSpeedRuntimeProfile | None = None,
        stage2_peft_metadata=None,
        step_preference_config: StepPreferenceConfig | None = None,
        step_preference_metadata: StepPreferenceMetadata | None = None,
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
        if step_preference_config is None:
            step_preference_config = StepPreferenceConfig(enabled=False)
        if not isinstance(step_preference_config, StepPreferenceConfig):
            raise TypeError("step_preference_config must be StepPreferenceConfig")
        if step_preference_metadata is not None and not isinstance(
            step_preference_metadata, StepPreferenceMetadata
        ):
            raise TypeError("step_preference_metadata must be StepPreferenceMetadata")
        self.step_preference_config = step_preference_config
        self.step_preference_metadata = step_preference_metadata
        validate_gated_execution_peft_identity(
            self.step_preference_metadata, self.stage2_peft_metadata
        )
        self._latest_step_preference_diagnostics = None
        self._latest_step_preference_compute_stats = None
        self.stage2_sampler_identity: Stage2SamplerIdentity | None = None
        self._stage2_train_sampler: LatentBlockBalancedSampler | None = None
        super().__init__(*args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        model = self.accelerator.unwrap_model(self.model)
        if not isinstance(model, PeftModel):
            return super()._save(output_dir, state_dict=state_dict)

        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
            save_embedding_layers=False,
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

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
        if self.args.world_size <= 1:
            self.stage2_sampler_identity = None
            self._stage2_train_sampler = None
            return None
        preference = getattr(self, "step_preference_config", None)
        preference_signatures = None
        if preference is not None and preference.enabled and preference.target_scope == "immediate_text":
            include_text = preference.modalities == "visual_text"
            preference_signatures = tuple(
                tuple(
                    (step_id in targets, include_text and step_id in targets)
                    for step_id in range(example.latent_block_count)
                )
                for example in self.train_dataset
                for targets in (immediate_text_target_indices(example),)
            )
        step_negative_index = self._stage2_sampler_negative_index()
        sampler = getattr(self, "_stage2_train_sampler", None)
        if (
            sampler is None
            or sampler.dataset is not self.train_dataset
            or sampler.num_replicas != self.args.world_size
            or sampler.seed != self.args.seed
            or sampler.step_negative_index is not step_negative_index
            or sampler.preference_signatures != preference_signatures
            or sampler.identity.dataset_size != len(self.train_dataset)
        ):
            sampler = LatentBlockBalancedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                seed=self.args.seed,
                step_negative_index=step_negative_index,
                preference_signatures=preference_signatures,
            )
            self._stage2_train_sampler = sampler
        self.stage2_sampler_identity = sampler.identity
        return sampler

    def _stage2_sampler_negative_index(self) -> StepNegativeIndex | None:
        # Every valid step now contributes both modalities. Existing block-count
        # bucketing gives ranks equal step denominators without negative validity.
        return None

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
            from stage2_objective.metadata import validate_objective_resume
            validate_objective_resume(Path(resume_from_checkpoint),
                getattr(self, "objective_metadata", None))
            validate_deepspeed_resume_identity(
                Path(resume_from_checkpoint), self.deepspeed_runtime_profile
            )
            sampler_identity = self._ensure_stage2_sampler_identity()
            if sampler_identity is not None:
                validate_stage2_sampler_resume_identity(
                    Path(resume_from_checkpoint), sampler_identity
                )
            step_preference_metadata = getattr(
                self, "step_preference_metadata", None
            )
            preference_sidecar = (
                Path(resume_from_checkpoint) / STEP_PREFERENCE_METADATA_FILENAME
            )
            checkpoint_has_preference_sidecar = (
                preference_sidecar.exists() or preference_sidecar.is_symlink()
            )
            if step_preference_metadata is None:
                if checkpoint_has_preference_sidecar:
                    raise ValueError(
                        "unexpected step preference metadata for standard resume"
                    )
            else:
                checkpoint_metadata = load_step_preference_metadata(
                    Path(resume_from_checkpoint)
                )
                validate_step_preference_resume_identity(
                    checkpoint_metadata, step_preference_metadata
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
        self._publish_step_preference_identity(
            checkpoint,
            label="Stage 2 checkpoint preference identity publication",
        )

    def _publish_step_preference_identity(
        self, directory: Path, *, label: str
    ) -> None:
        metadata = getattr(self, "step_preference_metadata", None)
        if metadata is None:
            return
        local_error: Exception | None = None
        if self.args.should_save:
            try:
                save_step_preference_metadata(directory, metadata)
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

    def training_step(self, model, inputs, num_items_in_batch=None):
        # The scope must include checkpoint replay in backward, and end before
        # another microbatch can observe new optimizer weights.
        enabled = (not self.is_fsdp_enabled
                   and self.deepspeed_runtime_profile.zero_stage != 3)
        with dora_microbatch_scope(model, enabled=enabled):
            return super().training_step(model, inputs, num_items_in_batch)

    def _should_collect_stage2_diagnostics(self) -> bool:
        if not self.accelerator.sync_gradients:
            return False
        next_global_step = self.state.global_step + 1
        return (self.args.logging_first_step and next_global_step == 1) or (
            next_global_step % self.args.logging_steps == 0
        )

    @property
    def stage2_debugging(self) -> bool:
        return getattr(self.args, "stage2_debugging", False)

    def _needs_preference_entropy(self) -> bool:
        return (getattr(self.step_preference_config, "prefix_mode", "clean_prefix") == "v_off_insertion"
                or self.step_preference_config.lambda_entropy != 0 or self.stage2_debugging)

    def _preference_diagnostic_keys(self) -> tuple[str, ...]:
        if getattr(self.step_preference_config, "prefix_mode", "clean_prefix") in {"v_off_insertion", "single_image_latent_gate"}:
            return _INSERTION_DIAGNOSTIC_KEYS
        return _STEP_PREFERENCE_DIAGNOSTIC_KEYS

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forbidden = {"pixel_values_latent", "image_grid_thw_latent"} & inputs.keys()
        if forbidden:
            raise ValueError(
                f"helper-image tensors leaked into Stage 2: {sorted(forbidden)}"
            )
        preference = getattr(self, "step_preference_config", None)
        if preference is None:
            preference = StepPreferenceConfig(enabled=False)
        if preference.enabled and getattr(preference, "prefix_mode", "clean_prefix") == "single_image_latent_gate":
            # Empty-target batches bypass the gated model request, but still obey
            # the same execution policy as batches that contribute preference loss.
            if resolve_stage2_qwen_model(model).model.gradient_checkpointing:
                raise ValueError("single_image_latent_gate requires native checkpointing disabled")
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in _STAGE2_PYTHON_METADATA_KEYS
        }
        collect_diagnostics = self._should_collect_stage2_diagnostics()
        self._latest_step_preference_compute_stats = None
        if not preference.enabled:
            outputs = model(
                **model_inputs,
                stage2_loss_only=True,
                collect_stage2_diagnostics=collect_diagnostics and self.stage2_debugging,
            )
            self._store_stage2_diagnostics(outputs)
            self._latest_step_preference_diagnostics = None
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        items = self._step_preference_items(inputs)
        if not items:
            # A final-only/empty first item contributes CE, never a fabricated target.
            outputs = model(**model_inputs, stage2_loss_only=True,
                            collect_stage2_diagnostics=collect_diagnostics and self.stage2_debugging)
            if getattr(preference, "prefix_mode", "clean_prefix") in {"v_off_insertion", "single_image_latent_gate"}:
                total_loss = outputs.loss
                diagnostics = {key: outputs.loss.detach().new_zeros(())
                               for key in self._preference_diagnostic_keys()}
                diagnostics.update({"loss/ce": outputs.loss, "loss/total": outputs.loss})
            else:
                total_loss, diagnostics, _ = self._reduce_step_preference_components(
                    outputs.loss, {}, {}, (), {})
            self._store_stage2_diagnostics(outputs)
            self._latest_step_preference_diagnostics = (
                self._synchronize_step_preference_diagnostics(diagnostics)
                if collect_diagnostics else None)
            return (total_loss, outputs) if return_outputs else total_loss
        items = self._prepare_shared_preference_items(items, model_inputs)
        if getattr(preference, "prefix_mode", "clean_prefix") == "single_image_latent_gate":
            request = GatedRequest(
                items=items, branch_mode=preference.branch_mode,
                ablation=preference.ablation,
                stage2_debugging=self.stage2_debugging,
                compute_entropy=self._needs_preference_entropy(),
            )
            outputs = model(
                **model_inputs, stage2_loss_only=True,
                collect_stage2_diagnostics=collect_diagnostics and self.stage2_debugging,
                stage2_gated_request=request,
            )
            result = getattr(outputs, "stage2_gated_scores", None)
            if not isinstance(result, GatedScores):
                raise ValueError("Stage 2 gated preference requires gated scores")
            total_loss, diagnostics = self._reduce_evidence_preference_scores(outputs, items, result)
        elif getattr(preference, "prefix_mode", "clean_prefix") == "v_off_insertion":
            request = InsertionRequest(
                items=items, branch_mode=preference.branch_mode,
                on_layer_checkpoint_stride=preference.on_layer_checkpoint_stride,
                stage2_debugging=self.stage2_debugging,
            )
            outputs = model(
                **model_inputs, stage2_loss_only=True,
                collect_stage2_diagnostics=collect_diagnostics and self.stage2_debugging,
                stage2_insertion_request=request,
            )
            total_loss, diagnostics = self._compute_step_preference_insertion(outputs, items)
        else:
            score_mask = self._chosen_score_mask(
                model_inputs, items, stage2_debugging=self.stage2_debugging)
            plan = build_context_sidecar_execution_plan(
                clean_input_ids=model_inputs["input_ids"], items=items,
                ablation=preference.ablation, modalities=preference.modalities,
                stage2_debugging=self.stage2_debugging,
            )
            outputs = model(
                **model_inputs, stage2_loss_only=True,
                collect_stage2_diagnostics=collect_diagnostics and self.stage2_debugging,
                stage2_score_token_mask=score_mask, stage2_sidecar_plan=plan,
            )
            total_loss, diagnostics = self._compute_step_preference_shared(
                outputs, score_mask, items, plan)
        self._store_stage2_diagnostics(outputs)
        self._latest_step_preference_diagnostics = (
            self._synchronize_step_preference_diagnostics(diagnostics)
            if collect_diagnostics
            else None
        )
        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_step_preference_insertion(self, outputs, items):
        """Reduce same-GT evidence ON/OFF scores while retaining original clean CE."""
        preference = self.step_preference_config
        result = getattr(outputs, "stage2_insertion_scores", None)
        if not isinstance(result, Stage2InsertionOutput):
            raise ValueError("Stage 2 insertion requires insertion scores")
        stats = result.stats
        expected_on = len(items) if preference.branch_mode == "independent" else 1
        if (stats.clean_prediction_calls != 1 or stats.off_prediction_calls != 1
                or stats.on_prediction_calls != expected_on or stats.reference_calls != 0
                or stats.reference_score_hits != 0 or stats.rejected_dpo_tokens != 0):
            raise RuntimeError(f"Stage 2 insertion compute contract violated: {stats.counters()}")
        return self._reduce_evidence_preference_scores(outputs, items, result)

    def _reduce_evidence_preference_scores(self, outputs, items, result):
        """Shared same-GT loss math, independent of each executor's call budget."""
        preference = self.step_preference_config
        keys = [(item.sample_id, item.span.step_id) for item in items]
        if not keys or len(set(keys)) != len(keys) or set(result.by_step) != set(keys):
            raise ValueError("evidence score identities do not match preference items")
        stats = result.stats
        pairs, entropy_terms, margins = [], [], []
        on_logps, off_logps, on_entropies, off_entropies = [], [], [], []
        for key in keys:
            pair = result.by_step[key]
            comparison = context_preference_loss(
                pair.on.mean_logp, pair.off.mean_logp, preference.beta,
                preference.preference_margin, stage2_debugging=self.stage2_debugging)
            pairs.append(comparison.loss)
            margins.append(comparison.policy_margins)
            if self._needs_preference_entropy():
                if pair.on.entropies is None or pair.off.entropies is None:
                    raise ValueError("requested evidence entropy is missing")
                entropy_terms.append(entropy_margin_loss(
                    pair.on.entropies, pair.off.entropies, preference.entropy_margin,
                    stage2_debugging=self.stage2_debugging).loss)
                on_entropies.append(pair.on.entropies.mean())
                off_entropies.append(pair.off.entropies.mean())
            on_logps.append(pair.on.mean_logp)
            off_logps.append(pair.off.mean_logp)
        pref = torch.stack(pairs).mean()
        entropy = torch.stack(entropy_terms).mean() if entropy_terms else pref.detach().new_zeros(())
        entropy_for_loss = entropy if preference.lambda_entropy else entropy.detach()
        total = (outputs.loss + preference.lambda_preference * pref
                 + preference.lambda_entropy * entropy_for_loss)
        if not torch.isfinite(total.detach()).all():
            raise ValueError("Stage 2 evidence preference total loss must be finite")
        diagnostics = {
            "loss/ce": outputs.loss, "loss/context_preference_all": pref,
            "loss/entropy_all": entropy, "loss/total": total,
            "loss/context_preference_visual": pref, "loss/entropy_visual": entropy,
            "stats/context_margin_visual": torch.stack(margins).mean(),
            "stats/logp_evidence_on": torch.stack(on_logps).mean(),
            "stats/logp_evidence_off": torch.stack(off_logps).mean(),
            "stats/entropy_evidence_on": torch.stack(on_entropies).mean() if on_entropies else entropy,
            "stats/entropy_evidence_off": torch.stack(off_entropies).mean() if off_entropies else entropy,
        }
        self._latest_step_preference_compute_stats = stats
        return total, diagnostics

    def _compute_step_preference_shared(self, outputs, score_mask, items, plan):
        """Pair the model's selected sidecar logits using stable branch identities."""
        ce_loss, clean_scores, clean_entropies = self._clean_preference_components(
            outputs, score_mask, items
        )
        sidecars = getattr(outputs, "stage2_sidecar_logits", None)
        stats = getattr(outputs, "stage2_compute_stats", None)
        if not isinstance(sidecars, Stage2SidecarLogits) or not isinstance(
            stats, Stage2ComputeStats
        ):
            raise ValueError(
                "Stage 2B shared forward requires sidecar logits and compute stats"
            )
        requests = (*plan.visual, *plan.text)
        if (
            stats.clean_decoder_calls < 1
            or stats.prediction_forward_calls > 3
            or stats.reference_calls != 0
            or stats.reference_score_hits != 0
            or stats.rejected_dpo_tokens != 0
        ):
            raise RuntimeError(
                f"Stage 2B context preference compute contract violated: {stats.counters()}"
            )
        expected_keys = {branch.branch_key for branch in requests}
        if set(sidecars.by_branch_key) != expected_keys:
            raise ValueError("Stage 2B sidecar branch keys are inconsistent")
        branch_logits = {}
        for branch in requests:
            logits = sidecars.by_branch_key[branch.branch_key]
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 2
                or logits.shape[0] != branch.target_token_ids.numel()
            ):
                raise ValueError("Stage 2B sidecar selected-logit count is inconsistent")
            branch_logits[branch.branch_key] = logits.unsqueeze(0)
        total_loss, diagnostics, reference_hits = (
            self._reduce_step_preference_components(
                ce_loss, clean_scores, clean_entropies, items, branch_logits
            )
        )
        # Context preference has no reference score source.
        outputs.stage2_compute_stats = replace(
            stats, reference_score_hits=reference_hits
        )
        self._latest_step_preference_compute_stats = outputs.stage2_compute_stats
        return total_loss, diagnostics

    def _clean_preference_components(self, outputs, score_mask, items):
        ce_loss = outputs.loss
        if not isinstance(ce_loss, torch.Tensor) or ce_loss.numel() != 1:
            raise ValueError("Stage 2B clean chosen forward must return scalar CE loss")
        clean_logits = self._require_scored_logits(
            outputs, expected_count=sum(item.chosen_target_ids.numel() for item in items)
        )
        clean_scores, clean_entropies = self._reduce_clean_logits(
            clean_logits, score_mask, items
        )
        return ce_loss, clean_scores, clean_entropies

    @staticmethod
    def _preference_branch_key(item, mode, role):
        return f"{item.sample_id}:{item.span.step_id}:{mode}:{role}"

    def _prepare_shared_preference_items(self, items, model_inputs):
        """Move collator dataclass tensors to Trainer's prepared clean device."""
        device = model_inputs["input_ids"].device
        prepared_items = []
        for item in items:
            prepared = {
                "chosen_predictor_positions": item.chosen_predictor_positions.to(device),
                "chosen_target_ids": item.chosen_target_ids.to(device),
            }
            prepared_items.append(replace(item, **prepared))
        return tuple(prepared_items)

    def _reduce_step_preference_components(
        self, ce_loss, clean_scores, clean_entropies, items, branch_logits
    ):
        """Reuse exactly clean/V-off/T-off logits for both auxiliary losses."""
        preference = self.step_preference_config
        modes = ("visual",) if preference.modalities == "visual" else ("visual", "text")
        preference_pair_losses: dict[str, list[torch.Tensor]] = {"visual": [], "text": []}
        context_margins: dict[str, list[torch.Tensor]] = {"visual": [], "text": []}
        reference_hits = 0
        for item in items:
            item_key = (item.sample_id, item.span.step_id)
            for mode in modes:
                branch_key = self._preference_branch_key(item, mode, "chosen_entropy")
                ablated_score = self._mean_logp(branch_logits[branch_key], item.chosen_target_ids)
                comparison = context_preference_loss(
                    clean_scores[item_key],
                    ablated_score,
                    preference.beta,
                    preference.preference_margin,
                    stage2_debugging=self.stage2_debugging,
                )
                preference_pair_losses[mode].append(comparison.pair_losses.reshape(()))
                context_margins[mode].append(comparison.policy_margins.reshape(()))

        loss_anchor = (
            ce_loss if not clean_scores else next(iter(clean_scores.values()))
        )
        zero = differentiable_zero(loss_anchor)
        # The distributed sampler gives every rank in a synchronized batch
        # the same latent/step count. Both modalities include every step, so
        # each local denominator is equal and DDP's mean of local means is
        # exactly the corresponding global context-comparison mean.
        preference_visual = self._mean_or_zero(preference_pair_losses["visual"], zero)
        preference_text = self._mean_or_zero(preference_pair_losses["text"], zero)
        preference_all = self._mean_or_zero(
            preference_pair_losses["visual"] + preference_pair_losses["text"], zero
        )
        margin_visual = self._mean_or_zero(context_margins["visual"], zero)
        margin_text = self._mean_or_zero(context_margins["text"], zero)
        entropy_branch_losses: dict[str, list[torch.Tensor]] = {"visual": [], "text": []}
        entropy_clean_means: dict[str, list[torch.Tensor]] = {"visual": [], "text": []}
        ablated_entropy_means: dict[str, list[torch.Tensor]] = {"visual": [], "text": []}
        for item in (items if self._needs_preference_entropy() else ()):
            item_key = (item.sample_id, item.span.step_id)
            for mode in modes:
                branch_key = self._preference_branch_key(item, mode, "chosen_entropy")
                ablated_logits = branch_logits[branch_key]
                clean_token_entropies = clean_entropies[item_key]
                ablated_token_entropies = self._token_entropies(ablated_logits)
                entropy_result = entropy_margin_loss(
                    clean_token_entropies,
                    ablated_token_entropies,
                    preference.entropy_margin,
                    stage2_debugging=self.stage2_debugging,
                )
                entropy_branch_losses[mode].append(entropy_result.loss)
                entropy_clean_means[mode].append(
                    clean_token_entropies.mean(dtype=torch.float32)
                )
                ablated_entropy_means[mode].append(
                    ablated_token_entropies.mean(dtype=torch.float32)
                )

        entropy_visual = self._mean_or_zero(
            entropy_branch_losses["visual"], zero
        )
        entropy_text = self._mean_or_zero(entropy_branch_losses["text"], zero)
        entropy_all = self._mean_or_zero(
            entropy_branch_losses["visual"] + entropy_branch_losses["text"],
            zero,
        )
        all_clean_means = (
            entropy_clean_means["visual"] + entropy_clean_means["text"]
        )
        total_loss = ce_loss + preference.lambda_preference * preference_all
        if preference.lambda_entropy != 0:
            total_loss = total_loss + preference.lambda_entropy * entropy_all
        # One scalar guard per microbatch replaces repeated vocabulary-wide scans.
        if not torch.isfinite(total_loss.detach()).all():
            raise ValueError("Stage 2 total loss must be finite")
        diagnostics = {
            "loss/ce": ce_loss,
            "loss/context_preference_all": preference_all,
            "loss/entropy_all": entropy_all,
            "loss/total": total_loss,
            "loss/context_preference_visual": preference_visual,
            "loss/context_preference_text": preference_text,
            "loss/entropy_visual": entropy_visual,
            "loss/entropy_text": entropy_text,
            "stats/context_margin_visual": margin_visual,
            "stats/context_margin_text": margin_text,
            "stats/entropy_clean": self._mean_or_zero(
                all_clean_means, zero
            ),
            "stats/entropy_visual_off": self._mean_or_zero(
                ablated_entropy_means["visual"], zero
            ),
            "stats/entropy_text_off": self._mean_or_zero(
                ablated_entropy_means["text"], zero
            ),
        }
        return total_loss, diagnostics, reference_hits

    @staticmethod
    def _reject_helper_tensors(inputs) -> None:
        forbidden = {"pixel_values_latent", "image_grid_thw_latent"} & inputs.keys()
        if forbidden:
            raise ValueError(
                f"helper-image tensors leaked into Stage 2: {sorted(forbidden)}"
            )

    @staticmethod
    def _prepare_rejected_branch_inputs(branch_inputs, clean_model_inputs):
        """Reuse prepared clean tensors and move branch-only tensors likewise."""
        clean_ids = clean_model_inputs.get("input_ids")
        if not isinstance(clean_ids, torch.Tensor):
            raise ValueError("Stage 2B clean input_ids must be a tensor")
        prepared = {}
        for key, value in branch_inputs.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError("rejected preference branches must contain only tensors")
            if key not in {"input_ids", "attention_mask"} and key in clean_model_inputs:
                clean_value = clean_model_inputs[key]
                if not isinstance(clean_value, torch.Tensor):
                    raise ValueError("clean Stage 2 model inputs must contain only tensors")
                prepared[key] = clean_value
            else:
                prepared[key] = value.to(device=clean_ids.device)
        return prepared

    def _store_stage2_diagnostics(self, outputs) -> None:
        diagnostics = getattr(outputs, "stage2_diagnostics", None)
        self._latest_stage2_diagnostics = (
            None
            if diagnostics is None
            else {name: value.detach() for name, value in diagnostics.items()}
        )

    @staticmethod
    def _step_preference_items(inputs) -> tuple[StepPreferenceBatchItem, ...]:
        raw_items = inputs.get("step_preference_items")
        if not isinstance(raw_items, (list, tuple)):
            raise ValueError("Stage 2B requires collated step_preference_items")
        if len(raw_items) == 1 and isinstance(raw_items[0], (list, tuple)):
            raw_items = raw_items[0]
        items = tuple(raw_items)
        if any(not isinstance(item, StepPreferenceBatchItem) for item in items):
            raise ValueError("step_preference_items contains an invalid item")
        return items

    @staticmethod
    def _chosen_score_mask(model_inputs, items, *, stage2_debugging=True) -> torch.Tensor:
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("Stage 2B input_ids must be two-dimensional")
        if input_ids.shape[0] != 1:
            raise NotImplementedError("Stage 2B currently supports batch_size=1 only")
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        intervals = []
        for item in items:
            positions = item.chosen_predictor_positions
            targets = item.chosen_target_ids
            if (
                positions.ndim != 1
                or targets.ndim != 1
                or positions.numel() == 0
                or positions.numel() != targets.numel()
            ):
                raise ValueError("chosen preference positions and targets are inconsistent")
            start, end = item.span.target_text_start - 1, item.span.target_text_end - 1
            if not 0 <= start < end < input_ids.shape[1] or end - start != targets.numel():
                raise ValueError("chosen preference predictor position is out of bounds")
            if any(start < old_end and old_start < end for old_start, old_end in intervals):
                raise ValueError("chosen preference predictor positions overlap")
            intervals.append((start, end))
            if not stage2_debugging:
                mask[0, start:end] = True
                continue
            expected_positions = torch.arange(
                item.span.target_text_start - 1,
                item.span.target_text_end - 1,
                dtype=positions.dtype,
                device=positions.device,
            )
            if not torch.equal(positions, expected_positions):
                raise ValueError("chosen preference predictor positions are not target-shifted")
            expected_targets = input_ids[
                0, item.span.target_text_start : item.span.target_text_end
            ].to(targets.device)
            if not torch.equal(targets, expected_targets):
                raise ValueError("chosen preference target IDs disagree with clean input")
            device_positions = positions.to(mask.device)
            if int(device_positions.min().item()) < 0 or int(
                device_positions.max().item()
            ) >= input_ids.shape[1]:
                raise ValueError("chosen preference predictor position is out of bounds")
            if bool(mask[0, device_positions].any()):
                raise ValueError("chosen preference predictor positions overlap")
            mask[0, device_positions] = True
        return mask

    @staticmethod
    def _require_scored_logits(outputs, *, expected_count: int) -> torch.Tensor:
        logits = getattr(outputs, "stage2_scored_logits", None)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.shape[0] != 1
            or logits.shape[1] != expected_count
        ):
            raise ValueError("Stage 2B selected-logit count is inconsistent")
        return logits

    def _reduce_clean_logits(self, logits, score_mask, items):
        scores = {}
        entropies = {}
        cursor = 0
        for item in sorted(items, key=lambda item: item.span.target_text_start):
            item_key = (item.sample_id, item.span.step_id)
            if item_key in scores:
                raise ValueError("duplicate step preference item")
            count = item.chosen_target_ids.numel()
            item_logits = logits[:, cursor:cursor + count]
            cursor += count
            scores[item_key] = self._mean_logp(item_logits, item.chosen_target_ids)
            if self._needs_preference_entropy():
                entropies[item_key] = self._token_entropies(item_logits)
        if cursor != logits.shape[1]:
            raise ValueError("Stage 2B selected-logit count is inconsistent")
        return scores, entropies

    def _mean_logp(self, logits, target_ids) -> torch.Tensor:
        targets = target_ids.to(device=logits.device, dtype=torch.long)[None]
        token_mask = torch.ones_like(targets, dtype=torch.bool)
        return mean_token_logps(logits, targets, token_mask,
                                stage2_debugging=self.stage2_debugging).reshape(1)

    def _token_entropies(self, logits) -> torch.Tensor:
        token_mask = torch.ones(
            logits.shape[:-1], dtype=torch.bool, device=logits.device
        )
        per_block = selected_token_entropies(logits, token_mask,
                                             stage2_debugging=self.stage2_debugging)
        if len(per_block) != 1:
            raise ValueError("Stage 2B entropy branch must contain exactly one block")
        return per_block[0]

    @staticmethod
    def _rejected_branch(item, mode):
        negative = item.negative
        if mode == "visual":
            branch = item.rejected_visual_inputs
            targets = item.rejected_visual_target_ids
            valid = negative.valid_visual_negative
        else:
            branch = item.rejected_text_inputs
            targets = item.rejected_text_target_ids
            valid = negative.valid_textual_negative
        if not valid:
            if branch is not None or targets is not None:
                raise ValueError("invalid negative must not contain a rejected branch")
            return None, None
        if (
            branch is None
            or not isinstance(targets, torch.Tensor)
            or targets.ndim != 1
        ):
            raise ValueError("valid negative requires one rejected branch target")
        if targets.numel() == 0:
            raise ValueError("valid rejected target must not be empty")
        return branch, targets

    @staticmethod
    def _mean_or_zero(values, zero):
        return torch.stack(tuple(values)).mean(dtype=torch.float32) if values else zero

    @staticmethod
    def _stack_or_empty(values, reference):
        if values:
            return torch.stack(tuple(values))
        return reference.new_empty((0,), dtype=torch.float32)

    def _synchronize_step_preference_diagnostics(self, diagnostics):
        # Signature-aligned replica groups have equal overall and per-modality
        # valid counts, so reducing these local means yields global pair means.
        values = torch.stack(
            tuple(
                diagnostics[name].detach().float().reshape(())
                for name in self._preference_diagnostic_keys()
            )
        )
        reduce = getattr(self.accelerator, "reduce", None)
        if callable(reduce):
            values = reduce(values, reduction="mean")
        return {
            name: value
            for name, value in zip(
                self._preference_diagnostic_keys(), values.unbind(), strict=True
            )
        }

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
        preference_diagnostics = getattr(
            self, "_latest_step_preference_diagnostics", None
        )
        self._latest_step_preference_diagnostics = None
        compute_stats = getattr(self, "_latest_step_preference_compute_stats", None)
        self._latest_step_preference_compute_stats = None
        if preference_diagnostics is not None:
            logs = dict(logs)
            logs.update(
                {
                    name: float(preference_diagnostics[name].item())
                    for name in self._preference_diagnostic_keys()
                    if self._needs_preference_entropy() or "entropy" not in name
                }
            )
            logs["stage2/debugging"] = int(self.stage2_debugging)
            logs["stage2/entropy_computed"] = int(self._needs_preference_entropy())
            if compute_stats is not None:
                counters = compute_stats.counters()
                # Match loss diagnostics' per-rank mean and logging cadence.
                # Snapshot here: activation checkpoint replays occur in backward.
                values = next(iter(preference_diagnostics.values())).new_tensor(
                    tuple(counters.values()), dtype=torch.float64
                )
                reduce = getattr(self.accelerator, "reduce", None)
                if callable(reduce):
                    values = reduce(values, reduction="mean")
                logs.update({
                    "stats/stage2b_" + name: value
                    for name, value in zip(counters, values.tolist(), strict=True)
                })
        super().log(logs, start_time=start_time)

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)
        metadata = self.stage2_peft_metadata
        target = Path(output_dir or self.args.output_dir)
        if metadata is not None and metadata.token_row_source == "sparse_adapter":
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
        if metadata is not None and self.args.should_save:
            save_stage2_peft_metadata(target, metadata)
        step_preference_metadata = getattr(
            self, "step_preference_metadata", None
        )
        if step_preference_metadata is not None and self.args.should_save:
            save_step_preference_metadata(target, step_preference_metadata)
