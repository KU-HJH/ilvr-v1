"""Thin Hugging Face Trainer adapter for Stage 2 model-native loss."""

import os
import math
import tempfile
import weakref
from pathlib import Path

import torch
from accelerate.utils import get_mixed_precision_context_manager
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
from .objective import Stage2ObjectiveConfig
from .objective_runtime import counted_forward, model_inputs, validate_objective_backend, validate_objective_student
from .reference import STAGE2_OBJECTIVE_FILENAME, save_objective_identity, validate_objective_resume
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


# Every rank reduces the same schema, even when its valid-block set is empty.
_OBJECTIVE_METRICS = (
    "loss_ce", "loss_dyn", "loss_vis",
    "H_pre", "H_post", "H_ref_pre", "H_ref_post",
    "r_pre", "r_post", "D", "S_F", "S_M", "G",
    "r_pre_gt_1", "r_post_lt_1", "G_gt_0",
    "valid_blocks", "excluded_blocks",
    "excluded_empty_pre", "excluded_empty_post",
    "excluded_overlap_empty_pre", "excluded_overlap_empty_post",
    "low_reference_entropy",
    "m_branches", "m_query_rows",
    "model_forwards_F", "model_forwards_M", "model_forwards_reference",
    "decoder_forwards_F", "decoder_forwards_M", "decoder_forwards_reference",
)


class _ObservedOptimizerStep(TrainerCallback):
    def __init__(self, owner):
        self.owner = owner

    def on_train_begin(self, args, state, control, **kwargs):
        del args, state, kwargs
        self.owner._install_deepspeed_gradient_adapter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        del args, control, kwargs
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
        return self.original_step(*args, **kwargs)


class Stage2Trainer(Trainer):
    """Add Stage 2 observability and PEFT sidecars to ordinary Trainer flows."""

    stage2_objective_config = Stage2ObjectiveConfig()

    def __init__(
        self,
        *args,
        deepspeed_runtime_profile: DeepSpeedRuntimeProfile | None = None,
        stage2_peft_metadata=None,
        stage2_objective_config=None,
        stage2_objective_runtime=None,
        **kwargs,
    ):
        self.stage2_objective_config = stage2_objective_config or Stage2ObjectiveConfig()
        self.stage2_objective_runtime = stage2_objective_runtime
        self._objective_metrics = {}
        config = self.stage2_objective_config
        config.validate("stage2", True)
        training_args = kwargs.get("args", args[1] if len(args) > 1 else None)
        if config.active:
            if stage2_objective_runtime is None or stage2_objective_runtime.config != config:
                raise ValueError("active Stage 2 objective requires its matching runtime")
            validate_objective_backend(training_args)
            if kwargs.get("compute_loss_func") is not None:
                raise ValueError("active Stage 2 objective requires native loss scaling")
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
        if config.active and deepspeed_runtime_profile.enabled:
            raise ValueError("active Stage 2 objective does not support DeepSpeed")
        self.stage2_peft_metadata = stage2_peft_metadata
        self.stage2_sampler_identity: Stage2SamplerIdentity | None = None
        self._stage2_train_sampler: LatentBlockBalancedSampler | None = None
        self._representative_gradient_norm: float | None = None
        self._observed_optimizer_step: int | None = None
        self._stage2_deepspeed_gradient_adapter = None
        callbacks = list(kwargs.pop("callbacks", ()) or ())
        callbacks.append(_ObservedOptimizerStep(self))
        kwargs["callbacks"] = callbacks
        super().__init__(*args, **kwargs)
        if config.active:
            validate_objective_student(self.model)
            self.model_accepts_loss_kwargs = False
            if self.accelerator.gradient_accumulation_steps != 1:
                raise ValueError("active Stage 2 objective requires native Trainer accumulation")
            reference = stage2_objective_runtime.reference
            if reference is not None:
                # Trainer has now placed the student on this rank's actual device.
                reference.model.to(self.accelerator.device)
                reference.model.eval()

    def _objective_identity(self):
        runtime = self.stage2_objective_runtime
        return (
            self.stage2_objective_config.to_dict(),
            runtime.reference.identity if runtime is not None and runtime.reference is not None else None,
            runtime.text_vocabulary.identity if runtime is not None else None,
        )

    def _global_objective_count(self, local_count):
        count = torch.tensor(local_count, device=self.accelerator.device, dtype=torch.int64)
        return int(self.accelerator.reduce(count, reduction="sum").item())

    def get_batch_samples(self, epoch_iterator, num_batches):
        samples, num_items_in_batch = super().get_batch_samples(epoch_iterator, num_batches)
        if self.stage2_objective_config.active:
            selected = [self.stage2_objective_runtime.select(sample) for sample in samples]
            valid_count = self._global_objective_count(sum(len(window.blocks) for window in selected))
            for sample, window in zip(samples, selected):
                sample["_stage2_objective_selection"] = window
                sample["_stage2_objective_global_count"] = valid_count
        return samples, num_items_in_batch

    def _accumulate_objective_metrics(self, sums, counts):
        for name in sums:
            aggregate = self._objective_metrics.setdefault(name, [0.0, 0])
            aggregate[0] += float(sums[name])
            aggregate[1] += counts[name]

    def _drain_objective_logs(self):
        statistics = torch.zeros(
            (len(_OBJECTIVE_METRICS), 3), device=self.accelerator.device, dtype=torch.float64
        )
        for row, name in enumerate(_OBJECTIVE_METRICS):
            if name in self._objective_metrics:
                total, count = self._objective_metrics[name]
                statistics[row, 0] = total
                statistics[row, 1] = count
                statistics[row, 2] = 1
        statistics = self.accelerator.reduce(statistics, reduction="sum").tolist()
        self._objective_metrics.clear()
        means = {}
        for name, (total, count, present) in zip(_OBJECTIVE_METRICS, statistics):
            if present:
                means[name] = total / count if count else total
        if "loss_ce" in means:
            config = self.stage2_objective_config
            means["loss_total"] = means["loss_ce"]
            means["loss_total"] += config.lambda_dyn * means.get("loss_dyn", 0.0)
            means["loss_total"] += config.lambda_vis * means.get("loss_vis", 0.0)
            means["backend"] = resolve_qwen2_5_vl_model(self.model).config._attn_implementation
        return {"stage2/objective/" + name: value for name, value in means.items()}

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
            if self.stage2_objective_config.enable_stage2_objective or (Path(resume_from_checkpoint) / STAGE2_OBJECTIVE_FILENAME).exists():
                validate_objective_resume(resume_from_checkpoint, *self._objective_identity())
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

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forbidden = {"pixel_values_latent", "image_grid_thw_latent"} & inputs.keys()
        if forbidden:
            raise ValueError(
                f"helper-image tensors leaked into Stage 2: {sorted(forbidden)}"
            )
        config = self.stage2_objective_config
        if config.active:
            runtime = self.stage2_objective_runtime
            selection = inputs.get("_stage2_objective_selection")
            if selection is None:
                selection = runtime.select(inputs)
            # A prepared student's forward owns its own autocast wrapper. The
            # independent reference needs the same policy around the whole run.
            # Use Accelerate's underlying torch context so exceptions also exit it.
            with get_mixed_precision_context_manager(
                self.accelerator.native_amp, self.accelerator.autocast_handler
            ):
                result = runtime.forward(model, inputs, selection,
                                         counterfactual_model=self.accelerator.unwrap_model(model))
            outputs = result.outputs
            loss = result.ce
            count = inputs.get("_stage2_objective_global_count")
            if count is None:
                # A direct compute_loss call describes one global batch, with no
                # pending Trainer backward division to cancel.
                count = self._global_objective_count(result.count)
                native_divisor = 1
            else:
                # The pinned native Trainer divides by configured A, including
                # its final partial window. Accelerator contributes no extra /A.
                native_divisor = self.args.gradient_accumulation_steps
            if count > 0:
                local_sum = config.lambda_dyn * result.dyn_sum + config.lambda_vis * result.vis_sum
                loss = loss + local_sum * (self.accelerator.num_processes * native_divisor / count)
            else:
                loss = loss + result.ce * 0.0
            self._accumulate_objective_metrics(result.metric_sums, result.metric_counts)
        elif config.enable_stage2_objective:
            outputs, calls = counted_forward(model, model_inputs(inputs), student=self.accelerator.unwrap_model(model),
                                             collect_stage2_diagnostics=True, stage2_finite_ce=True)
            loss = outputs.loss
            self._accumulate_objective_metrics(
                {"loss_ce": loss.detach(), "model_forwards_F": 1, "decoder_forwards_F": calls},
                {"loss_ce": 1, "model_forwards_F": 0, "decoder_forwards_F": 0},
            )
        else:
            outputs = model(**inputs, collect_stage2_diagnostics=True)
            loss = outputs.loss
        if config.enable_stage2_objective and not bool(torch.isfinite(outputs.loss).all()):
            raise ValueError("nonfinite Stage 2 CE in normal F branch")
        diagnostics = getattr(outputs, "stage2_diagnostics", None)
        self._latest_stage2_diagnostics = (
            None
            if diagnostics is None
            else {name: value.detach() for name, value in diagnostics.items()}
        )
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if self.stage2_objective_config.enable_stage2_objective:
            logs = {**logs, **self._drain_objective_logs()}
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
        if self.processing_class is not None:
            self.processing_class.save_pretrained(target)
        torch.save(self.args, target / TRAINING_ARGS_NAME)

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)
        if self.stage2_objective_config.enable_stage2_objective and self.args.should_save:
            save_objective_identity(Path(output_dir or self.args.output_dir), *self._objective_identity())
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
