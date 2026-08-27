"""Thin Hugging Face Trainer adapter for Stage 2 model-native loss."""

import os
import tempfile
from pathlib import Path

import torch
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

from deepspeed_runtime import (
    DeepSpeedRuntimeProfile,
    load_deepspeed_runtime_profile,
    validate_deepspeed_resume_identity,
    write_deepspeed_runtime_sidecar,
)
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


class Stage2Trainer(Trainer):
    """Add Stage 2 observability and PEFT sidecars to ordinary Trainer flows."""

    def __init__(
        self,
        *args,
        deepspeed_runtime_profile: DeepSpeedRuntimeProfile | None = None,
        stage2_peft_metadata=None,
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
        super().__init__(*args, **kwargs)

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
        outputs = model(**inputs, collect_stage2_diagnostics=True)
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
        super().log(logs, start_time=start_time)

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
