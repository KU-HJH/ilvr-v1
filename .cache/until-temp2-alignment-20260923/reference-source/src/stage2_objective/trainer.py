"""Trainer integration with globally normalized auxiliary block losses."""
from dataclasses import replace
import math
from pathlib import Path

import torch
from accelerate.utils import get_mixed_precision_context_manager
from transformers.trainer_utils import get_last_checkpoint
from transformers import TrainerCallback

from stage2.trainer import Stage2Trainer
from .execution import ObjectiveRuntime, execute_objective, replay_metric_totals
from .reduction import (
    AccumulationContext, merge_metric_totals, reduce_metric_totals, scale_objective_loss,
)
from .windows import WindowPlan
from .metadata import (ObjectiveMetadata, make_objective_metadata, save_objective_metadata,
                       validate_objective_resume)
from .execution import apply_objective_dropout_policy, validate_objective_dropout_policy
from .config import validate_objective_zero2_matrix, validate_objective_ddp_matrix
from .reference import align_reference_precision


class ObjectiveTrainerConfigurationError(ValueError):
    """An execution setting would violate the objective's reduction contract."""


class _ObjectiveBackwardCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer._validate_execution_contract(engine=self.trainer.model_wrapped)

    def on_substep_end(self, args, state, control, **kwargs):
        self.trainer._complete_objective_backward()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.trainer._complete_objective_backward()


class Stage2ObjectiveTrainer(Stage2Trainer):
    def __init__(self, *args, objective_runtime: ObjectiveRuntime,
                 objective_metadata: ObjectiveMetadata | None = None, **kwargs):
        if not isinstance(objective_runtime, ObjectiveRuntime) or not objective_runtime.config.enabled:
            raise ObjectiveTrainerConfigurationError('an enabled ObjectiveRuntime is required')
        self.objective_runtime = objective_runtime
        self.objective_metadata = objective_metadata
        self._objective_metrics = {}
        self._objective_pending = None
        self._objective_seen_gradients = set()
        super().__init__(*args, **kwargs)
        if self.compute_loss_func is not None:
            raise ObjectiveTrainerConfigurationError('compute_loss_func must be unset')
        if self.step_preference_config.enabled:
            raise ObjectiveTrainerConfigurationError('old preference objective must be disabled')
        if self.args.per_device_train_batch_size != 1 or self.args.remove_unused_columns:
            raise ObjectiveTrainerConfigurationError('microbatch size 1 and remove_unused_columns=False required')
        if self.args.ddp_find_unused_parameters is True:
            raise ObjectiveTrainerConfigurationError('ddp_find_unused_parameters must be False')
        self.args.ddp_find_unused_parameters = False
        self.model_accepts_loss_kwargs = False
        self._validate_execution_contract()
        if objective_runtime.reference is not None:
            reference_ids = {id(p) for p in objective_runtime.reference.model.parameters()}
            if reference_ids.intersection(id(p) for p in self.model.parameters()):
                raise ObjectiveTrainerConfigurationError('reference must be independent of student')
        # Observe actual leaf participation, including all-ignore zero anchors.
        # Hooks also work for ZeRO-2, whose engine may clear .grad in backward.
        self._objective_trainable_names = set()
        self._objective_gradient_hooks = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and parameter.numel():
                self._objective_trainable_names.add(name)
                def observed(gradient, name=name):
                    self._objective_seen_gradients.add(name)
                self._objective_gradient_hooks.append(parameter.register_hook(observed))
        self.add_callback(_ObjectiveBackwardCallback(self))

    def _verify_objective_dropout(self, model):
        validate_objective_dropout_policy(model, self.objective_runtime.dropout_policy)

    def _require_objective_metadata(self):
        metadata = self.objective_metadata
        if not isinstance(metadata, ObjectiveMetadata):
            raise ValueError('save/resume requires non-null ObjectiveMetadata')
        current = make_objective_metadata(self.objective_runtime,
            backend=('sdpa_vision_flash_attention_2' if getattr(self.model, 'stage2_split_attention', False)
                     else self.model.config._attn_implementation),
            latent_size=self.model.config.latent_size,
            supervise_latent_boundaries=self.model.config.supervise_latent_boundaries)
        if current != metadata:
            raise ValueError('ObjectiveMetadata does not match current runtime')
        self._verify_objective_dropout(self.model)
        return metadata

    def _agree_objective_error(self, error, label):
        failed = self.accelerator.reduce(torch.tensor(int(error is not None),
            dtype=torch.int32, device=self.accelerator.device), reduction='sum')
        if int(failed.item()):
            if error is not None:
                raise error
            raise RuntimeError(f'{label} failed on another process')

    def _validate_objective_metadata_all_ranks(self):
        error = None
        try:
            self._require_objective_metadata()
        except Exception as caught:
            error = caught
        self._agree_objective_error(error, 'objective metadata validation')

    def _publish_objective_metadata(self, directory):
        error = None
        try:
            metadata = self._require_objective_metadata()
            if self.args.should_save:
                save_objective_metadata(Path(directory), metadata)
        except Exception as caught:
            error = caught
        self._agree_objective_error(error, 'objective metadata publication')
        self.accelerator.wait_for_everyone()

    def train(self, *args, resume_from_checkpoint=None, **kwargs):
        if resume_from_checkpoint is True:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError('No valid checkpoint found in output directory')
        if resume_from_checkpoint not in (None, False):
            error = None
            try:
                validate_objective_resume(Path(resume_from_checkpoint), self._require_objective_metadata())
            except Exception as caught:
                error = caught
            self._agree_objective_error(error, 'objective resume validation')
        return super().train(*args, resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        # Original adapter configuration was restored by composition. Loading
        # student weights must never replace the independently loaded reference.
        target = self.model if model is None else model
        apply_objective_dropout_policy(target)
        self._verify_objective_dropout(target)

    def _save_checkpoint(self, model, trial):
        self._validate_objective_metadata_all_ranks()
        # The inherited path calls our save_model and retains PEFT/runtime/sampler sidecars.
        return super()._save_checkpoint(model, trial)

    def save_model(self, output_dir=None, _internal_call=False):
        self._validate_objective_metadata_all_ranks()
        error = None
        try:
            super().save_model(output_dir, _internal_call=_internal_call)
        except Exception as caught:
            error = caught
        self._agree_objective_error(error, 'objective student save')
        self._publish_objective_metadata(output_dir or self.args.output_dir)

    def _validate_execution_contract(self, engine=None):
        kind = str(self.accelerator.distributed_type).split('.')[-1]
        if kind not in {'NO', 'MULTI_CPU', 'MULTI_GPU', 'DEEPSPEED'}:
            raise ObjectiveTrainerConfigurationError('objective requires replicated single/DDP/ZeRO-2 parameters')
        if kind == 'NO' and self.args.n_gpu > 1:
            raise ObjectiveTrainerConfigurationError(
                'objective does not support implicit DataParallel; select one CUDA device or launch DDP/ZeRO-2')
        validate_objective_ddp_matrix(self.objective_runtime.config, kind=kind,
            backend=self.model.config._attn_implementation,
            native_checkpointing=self.args.gradient_checkpointing,
            latent_checkpointing=getattr(self.model.config, 'stage2_latent_activation_checkpointing', False))
        plugin = self.accelerator.state.deepspeed_plugin
        if plugin is not None:
            config = plugin.deepspeed_config
            zero = config.get('zero_optimization', {})
            offload = any(zero.get(key, {}).get('device', 'none') != 'none'
                          for key in ('offload_optimizer', 'offload_param'))
            if zero.get('stage') != 2 or offload:
                raise ObjectiveTrainerConfigurationError('objective requires ZeRO-2 without offload')
            validate_objective_zero2_matrix(backend=self.model.config._attn_implementation,
                native_checkpointing=self.args.gradient_checkpointing,
                latent_checkpointing=getattr(self.model.config, 'stage2_latent_activation_checkpointing', False))
            if engine is not None:
                divisor = engine.gradient_accumulation_steps()
                if divisor != self.args.gradient_accumulation_steps:
                    raise ObjectiveTrainerConfigurationError('DeepSpeed accumulation divisor differs from configured G')
        elif self.accelerator.gradient_accumulation_steps != 1:
            raise ObjectiveTrainerConfigurationError('Accelerator accumulation divisor must be 1; Trainer divides by G')

    def _is_accumulation_boundary(self, *, step, steps_in_epoch, batch_index, batch_count):
        # Prefetch owns the actual group, including resumed and distributed tails.
        return batch_index + 1 == batch_count

    def get_batch_samples(self, epoch_iterator, num_batches):
        batches, num_items = super().get_batch_samples(
            epoch_iterator, self.args.gradient_accumulation_steps)
        length = torch.tensor([len(batches)], device=self.accelerator.device, dtype=torch.long)
        lengths = self.accelerator.gather(length)
        if not bool(lengths.eq(lengths[0]).all()):
            raise ObjectiveTrainerConfigurationError('actual accumulation group lengths differ across ranks')
        if not batches:
            return batches, num_items
        for batch in batches:
            if not isinstance(batch.get('stage2_objective_plan'), WindowPlan):
                raise ObjectiveTrainerConfigurationError('each batch requires a finalized WindowPlan')
        count = torch.tensor(sum(len(b['stage2_objective_plan'].blocks) for b in batches),
                             dtype=torch.long, device=self.accelerator.device)
        global_count = self.accelerator.reduce(count, reduction='sum')
        context = AccumulationContext(self.accelerator.num_processes, len(batches),
                                      self.args.gradient_accumulation_steps, int(global_count.item()))
        for batch in batches:
            batch['stage2_objective_accumulation'] = context
        return batches, num_items

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        plan = inputs.get('stage2_objective_plan')
        if not isinstance(plan, WindowPlan):
            raise ObjectiveTrainerConfigurationError('batch requires a finalized WindowPlan')
        training = self.is_in_train and torch.is_grad_enabled()
        context = inputs.get('stage2_objective_accumulation') if training else None
        if context is None:
            count = torch.tensor(len(plan.blocks), dtype=torch.long, device=self.accelerator.device)
            count = self.accelerator.reduce(count, reduction='sum')
            context = AccumulationContext(self.accelerator.num_processes, 1, 1, int(count.item()))
        self._validate_execution_contract()
        core = self.accelerator.unwrap_model(model)
        model_inputs = {key: value for key, value in inputs.items()
                        if key not in {'stage2_objective_accumulation', 'stage2_objective_plan',
                                       'sample_ids', 'reasoning_step_spans', 'step_preference_items'}}
        if training:
            if self._objective_pending is not None:
                raise RuntimeError('previous objective backward was not completed')
            self._objective_seen_gradients.clear()
        reference = self.objective_runtime.reference
        if reference is not None:
            # Preparation can uniformly cast ZeRO-2 tensors, while native AMP
            # preserves mixed BF16-base/FP32-adapter policies. Match all names.
            align_reference_precision(reference.model, core)
        # Accelerator wraps only the student forward. The outer shared context
        # also covers the raw reference; under ZeRO-2 native AMP is disabled and
        # the runtime parameter alignment above supplies BF16 for every branch.
        with get_mixed_precision_context_manager(self.accelerator.native_amp,
                                                 self.accelerator.autocast_handler):
            result = execute_objective(forward_model=model, student_core=core,
                                       model_inputs=model_inputs, plan=plan, runtime=self.objective_runtime)
        loss = scale_objective_loss(result.ce_loss, result.weighted_block_sum, context)
        if training:
            # Keep only detached scalar totals and graph-free live counters.
            self._objective_pending = replace(result, ce_loss=result.ce_loss.detach(),
                weighted_block_sum=result.weighted_block_sum.detach(), outputs=None)
        return (loss, result.outputs) if return_outputs else loss

    def _complete_objective_backward(self):
        result = self._objective_pending
        if result is None:
            return
        missing = self._objective_trainable_names - self._objective_seen_gradients
        failure = torch.tensor(int(bool(missing)), device=self.accelerator.device)
        failures = self.accelerator.reduce(failure, reduction='sum')
        if failures.item():
            raise ObjectiveTrainerConfigurationError(
                'trainable leaves did not participate in objective backward: ' + ', '.join(sorted(missing))
                if missing else 'trainable leaves did not participate on another rank')
        self._objective_metrics = merge_metric_totals([
            self._objective_metrics, {**result.metrics, **replay_metric_totals(result)}])
        self._objective_pending = None

    def log(self, logs, start_time=None):
        if self._objective_metrics:
            metrics = reduce_metric_totals(self._objective_metrics,
                reduce_sum=lambda tensor: self.accelerator.reduce(tensor, reduction='sum'),
                lambda_dyn=self.objective_runtime.config.lambda_dyn,
                lambda_vis=self.objective_runtime.config.lambda_vis)
            logs = {**logs, **metrics}
            self._objective_metrics = {}
        super().log(logs, start_time=start_time)

    def set_initial_training_values(self, args, dataloader, total_train_batch_size):
        values = super().set_initial_training_values(args, dataloader, total_train_batch_size)
        epochs, updates, examples, samples, epoch_based, length, max_steps = values
        if length:
            updates = max(math.ceil(length / args.gradient_accumulation_steps), 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                epochs = math.ceil(max_steps / updates)
                samples = max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * updates)
                epochs = math.ceil(args.num_train_epochs)
                samples = examples * args.num_train_epochs
        return epochs, updates, examples, samples, epoch_based, length, max_steps
