"""Stage1-v5 integration with native Trainer backward and optimizer ownership."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
from importlib import metadata as packages
import json
from pathlib import Path
import shutil

import torch
from transformers.trainer_utils import get_last_checkpoint
from peft_common import resolve_qwen2_5_vl_model
from .equivalence import Stage1StochasticContext
from .trainer import CustomTrainerStage1
from .v5_config import Stage1V5Config, validate_v5_training_config
from .v5_execution import V5Execution
from .v5_metadata import V5BatchMetadata
from .v5_objective import compute_v5_loss_sums
from .v5_reduction import V5Window, v5_auxiliary_scale, v5_dil_scale
from .v5_reference import V5ReferenceIdentity, V5_REFERENCE_FILENAME
from .v5_resume import (V5_CONTRACT_FILENAME, build_v5_contract, load_v5_contract,
                        validate_v5_resume)
from .v5_selection import select_v5_text_keys
from .query import Stage1QueryConfig, query_execution_config, resolve_query_layers
from .targets import Stage1TeacherTargetBuilder
from .history_attention import ReducedHistoryAttentionObserver
from .history_teacher import HistoryPrefixAdapter


class CustomTrainerStage1V5(CustomTrainerStage1):
    # Opt in only this class to actual epoch windows and absolute resume tails.
    stage1_v5_actual_window_schedule = True

    def __init__(self, *args, v5_config: Stage1V5Config,
                 v5_reference_identity: V5ReferenceIdentity,
                 v5_saved_contract=None, stage1_v5_debug=False, **kwargs):
        if not isinstance(v5_config, Stage1V5Config):
            raise TypeError('v5 requires Stage1V5Config')
        validate_v5_training_config(v5_config)
        if not isinstance(v5_reference_identity, V5ReferenceIdentity):
            raise TypeError('v5 requires immutable reference identity')
        self.stage1_v5_debug = bool(stage1_v5_debug)
        kwargs["collect_stage1_diagnostics"] = self.stage1_v5_debug
        self.v5_config = v5_config
        self.v5_reference_identity = v5_reference_identity
        self.v5_saved_contract = v5_saved_contract
        self._v5_contract_ready = False
        self._v5_step_stack = None
        super().__init__(*args, **kwargs)
        from transformers.models.qwen2_5_vl.stage1_v5_trace import validate_v5_decoder
        validate_v5_decoder(resolve_qwen2_5_vl_model(self.model).model)
        validate_v5_decoder(resolve_qwen2_5_vl_model(self.ema_teacher.model).model)
        profile = self.deepspeed_runtime_profile
        if (self.ema_teacher.decay != 1. or self.ema_teacher.method != 'dora'
                or not profile.enabled or profile.zero_stage != 1
                or profile.world_size != 4 or self.args.per_device_train_batch_size != 1
                or self.args.gradient_accumulation_steps != 4 or not self.args.bf16
                or self.model_accepts_loss_kwargs or self.compute_loss_func is not None
                or self.use_apex):
            raise ValueError('unsupported Stage1-v5 execution profile')
        if v5_saved_contract is not None:
            resolved = validate_v5_resume(v5_saved_contract, {
                **asdict(v5_config), 'ce_weight': self.ce_weight, 'sim_weight': self.sim_weight})
            if (resolved != v5_config or v5_saved_contract['reference'] !=
                    json.loads(json.dumps(asdict(v5_reference_identity)))):
                raise ValueError('v5 reference/objective resume identity conflict')

    def get_batch_samples(self, epoch_iterator, num_batches):
        batches, hint = super().get_batch_samples(epoch_iterator, num_batches)
        if not batches:
            return batches, hint
        for batch in batches:
            if not isinstance(batch.get('stage1_v5_metadata'), V5BatchMetadata):
                raise ValueError('missing v5 original-block metadata')
        counts = torch.tensor([
            sum(b['stage1_v5_metadata'].original_step_count for b in batches),
            sum(b['stage1_v5_metadata'].selected_token_count(self.v5_config) for b in batches),
        ], device=self.accelerator.device, dtype=torch.int64)
        totals = self.accelerator.reduce(counts, reduction='sum').tolist()
        window = V5Window(totals[0], len(batches), self.args.gradient_accumulation_steps,
                          self.accelerator.num_processes, totals[1])
        # Copy dictionaries only; original token/label/image tensors stay identical.
        return [dict(batch, stage1_v5_window=window) for batch in batches], hint

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not isinstance(inputs.get('stage1_v5_window'), V5Window):
            raise ValueError('missing Stage1-v5 accumulation window identity')
        if not self._v5_contract_ready:
            raise RuntimeError('v5 contract/reference must be validated before targets')
        if self._v5_step_stack is not None:
            raise RuntimeError('nested v5 training step')
        metadata = inputs['stage1_batch_metadata']
        rows = tuple((s.canonical_slot_id if s.canonical_slot_id is not None else s.source_index,
                      s.unpadded_length) for s in metadata.samples)
        sites = tuple(getattr(self.model, 'stage1_stochastic_sites', ()))
        spec = Stage1StochasticContext(self.args.seed, self.state.global_step, rows, sites)
        try:
            with ExitStack() as stack:
                self._v5_step_stack = stack
                self._v5_execution = stack.enter_context(V5Execution(spec, debug=self.stage1_v5_debug))
                # Preserve exact parent dispatch, loss/GAS scaling, DeepSpeed and
                # ordinary optimizer lifecycle. compute_loss opens backward only
                # after all original graph/observation scopes have exited.
                return super().training_step(model, inputs, num_items_in_batch)
        finally:
            self._v5_step_stack = None
            self._v5_execution = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        if self._v5_step_stack is None:
            raise RuntimeError('v5 loss requires the actual parent training_step context')
        window = inputs.get('stage1_v5_window')
        metadata = inputs.get('stage1_v5_metadata')
        if not isinstance(window, V5Window) or not isinstance(metadata, V5BatchMetadata):
            raise ValueError('missing v5 metadata/window')
        working = {k:v for k,v in inputs.items() if k not in ('stage1_v5_window','stage1_v5_metadata')}
        need_voff = (self.v5_config.lambda_topk_entropy > 0 or self.v5_config.lambda_tail_kl > 0
                     or self.v5_config.selection in ('voff','gap','intersection'))
        need_textoff = self.v5_config.text_kl_weight > 0
        replacing = need_voff and bool(metadata.query_keys) and self.v5_config.voff_mode != 'attention_mask'
        model_inputs, source_metadata, targets = self._prepare_stage1_loss_inputs(working,
            **({'voff_mode': self.v5_config.voff_mode} if replacing else {}))
        replacements = {}
        if replacing:
            spans = {(row, index): span for row, sample in enumerate(source_metadata.samples)
                     for index, span in enumerate(sample.latent_spans)}
            for block in targets.rejected_blocks:
                key = (block.batch_index, block.block_index)
                if key not in spans or key in replacements or block.positions != spans[key].latent_positions:
                    raise ValueError('replacement block ownership mismatch')
                replacements[key] = block.values
            if set(replacements) != set(spans):
                raise ValueError('replacement block ownership is incomplete')
            replacements = {(row, spans[(row, index)].start, spans[(row, index)].end): values
                            for (row, index), values in replacements.items()}
            if set(replacements) != {(b.batch_index, b.start, b.end) for b in metadata.blocks}:
                raise ValueError('replacement metadata ownership mismatch')
        labels = model_inputs['labels']
        forward_inputs = dict(model_inputs, use_cache=False, return_dict=True)
        if labels.shape[0] > 1:
            forward_inputs.pop('labels')
        qwen = resolve_qwen2_5_vl_model(self.model)
        from transformers.models.qwen2_5_vl.stage1_v5_trace import capture_v5_trace, bind_v5_execution
        from transformers.models.qwen2_5_vl.stage1_v5_replay import V5ReplayBlock, run_v5_packed_replay
        keys = metadata.query_keys
        candidates = tuple(b.candidate_positions for b in metadata.blocks for _ in b.query_positions)
        execution = self._v5_execution
        with execution.graph('F'), bind_v5_execution(qwen.model, execution, role='F', visual=qwen.visual), \
                capture_v5_trace(qwen, keys, candidates) as trace:
            outputs = model(**forward_inputs)
        ce, alignment = self._stage1_native_losses(outputs, labels, targets)
        clean = (torch.stack([outputs.logits[row,position] for row,position in keys])
                 if keys else outputs.logits.reshape(-1,outputs.logits.shape[-1])[:0])
        reference, voff, textoff = clean.detach(), None, None
        if keys:
            reference = self._stage1_reference_logits(model_inputs, keys, source_metadata)
            attention = {}
            for key, positions in zip(keys, candidates):
                row = clean.new_zeros(labels.shape[1], dtype=torch.float32)
                row[list(positions)] = trace.attention_rows[key]
                attention[key] = row
            selected = select_v5_text_keys(metadata, attention, clean, reference, self.v5_config, validate=self.stage1_v5_debug)
            blocks = tuple(V5ReplayBlock(b.batch_index,b.start,b.end,b.target_positions,b.target_ids,
                           tuple(selected[(b.batch_index,q)] for q in b.query_positions),
                           replacements.get((b.batch_index,b.start,b.end)))
                           for b in metadata.blocks)
            if need_voff or need_textoff:
                # These decoder calls intentionally bypass distributed model
                # wrappers. ZeRO1 leaves parameters replicated; OFF/reference
                # cardinality may differ between ranks without forward collectives.
                with execution.graph('packed'):
                    off = run_v5_packed_replay(qwen.model,qwen.lm_head,trace,blocks,
                        execution=execution,need_voff=need_voff,need_textoff=need_textoff,
                        chunk_positions=self.v5_config.loss_chunk_positions,
                        **({'voff_mode': self.v5_config.voff_mode}
                           if self.v5_config.voff_mode != 'attention_mask' else {}))
                voff, textoff = off.voff_logits, off.textoff_logits
        sums = compute_v5_loss_sums(clean, reference, voff, textoff, metadata, self.v5_config,
            validate=self.stage1_v5_debug, collect_diagnostics=self.stage1_v5_debug)
        dil_scale = v5_dil_scale(window)
        block_scale = v5_auxiliary_scale(window)
        config = self.v5_config
        loss = (self.ce_weight*ce + self.sim_weight*alignment + config.text_kl_weight*sums.text_kl +
                dil_scale*config.lambda_preference*sums.dil_sum +
                block_scale*(config.lambda_topk_entropy*sums.entropy_sum +
                       config.lambda_topk_mass*sums.mass_sum))
        if config.lambda_tail_kl > 0:
            loss = loss + block_scale*config.lambda_tail_kl*sums.tail_kl_sum
            if getattr(self, '_stage1_record_loss_terms', False):
                components = (
                    ('ce', ce, self.ce_weight), ('alignment', alignment, self.sim_weight),
                    ('dil', dil_scale*sums.dil_sum, config.lambda_preference),
                    ('topk_entropy', block_scale*sums.entropy_sum, config.lambda_topk_entropy),
                    ('topk_mass', block_scale*sums.mass_sum, config.lambda_topk_mass),
                    ('text_kl', sums.text_kl, config.text_kl_weight),
                    ('tail_kl', block_scale*sums.tail_kl_sum, config.lambda_tail_kl),
                )
                metrics = {f'{name}_loss': value.detach().float() for name, value, _ in components}
                metrics.update({f'weighted_{name}_loss': (weight*value).detach().float()
                                for name, value, weight in components})
                metrics['total_loss'] = loss.detach().float()
                self._stage1_loss_term_names = tuple(metrics)
                self._stage1_pending_loss_terms = torch.stack(tuple(metrics.values()))
        if self.stage1_v5_debug and not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError('Stage1-v5 loss is nonfinite')
        self._latest_stage1_diagnostics = {
            **sums.diagnostics,
            **{key:value for key,value in targets.diagnostics.items() if key.startswith('rejected_')},
            'ce_loss':ce.detach(), 'alignment_loss':alignment.detach(),
            'total_loss':loss.detach(), 'auxiliary_scale':loss.new_tensor(block_scale),
            'dil_scale':loss.new_tensor(dil_scale),
            'global_dil_tokens':loss.new_tensor(window.global_dil_tokens),
            'global_original_steps':loss.new_tensor(window.global_original_steps),
            'actual_microbatches':loss.new_tensor(window.actual_microbatches),
        } if self.collect_stage1_diagnostics else None
        self._v5_step_stack.enter_context(execution.backward())
        return (loss, outputs) if return_outputs else loss

    def _publish_v5_files(self, directory: Path):
        error = None
        if self.args.should_save:
            try:
                from artifact_paths import require_canonical_artifact_path
                directory = require_canonical_artifact_path(directory, kind='checkpoint')
                if (directory / V5_CONTRACT_FILENAME).is_symlink():
                    raise ValueError('v5 contract must be a regular file')
                directory.mkdir(parents=True, exist_ok=True)
                source = Path(self.args.output_dir) / V5_REFERENCE_FILENAME
                target = directory / V5_REFERENCE_FILENAME
                if source.is_symlink() or not source.is_file() or target.is_symlink():
                    raise ValueError('missing or unsafe immutable v5 initial reference')
                if source.resolve() != target.resolve():
                    shutil.copyfile(source, target)
                (directory / V5_CONTRACT_FILENAME).write_text(
                    json.dumps(self._v5_contract, sort_keys=True, indent=2, allow_nan=False)+'\n')
            except Exception as exc:
                error = exc
        failures = self.accelerator.reduce(torch.tensor(int(error is not None),
            dtype=torch.int32, device=self.accelerator.device), reduction='sum')
        if int(failures.item()):
            if error is not None:
                raise error
            raise RuntimeError('Stage1-v5 publication failed on another rank')
        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def _stage1_reference_logits(self, model_inputs, keys, source_metadata):
        cache = getattr(self, 'run_cache', None)
        enabled = cache is not None and cache.plan.reference_hidden
        if enabled and (self.ema_teacher.decay != 1. or cache.plan.helper != 'targets'):
            raise ValueError('reference cache requires fixed helper targets and invariant full inputs')
        if enabled and source_metadata.batch_size != 1:
            raise ValueError('reference cache requires the supported fixed microbatch-1 geometry')
        chunks = self.v5_config.loss_chunk_positions
        if enabled and cache.phase.value == 1:
            from .cache_payloads import restore_teacher_forward_state
            record = cache.read('reference', source_metadata.samples[0].source_index)
            if (record.get('kind') != 'reference' or record['positions'] != tuple(q for _, q in keys)
                    or record['sequence_length'] != source_metadata.sequence_length
                    or record['chunk_positions'] != chunks
                    or record['hidden'].shape[0] != len(keys)):
                raise ValueError('cached reference query/projection geometry differs')
            owner = resolve_qwen2_5_vl_model(self.ema_teacher.model)
            restore_teacher_forward_state(self.ema_teacher.model, record['forward_state'])
            hidden = record['hidden'].to(device=next(owner.parameters()).device)
            return owner.project_stage1_selected_hidden(hidden, chunks)
        # Original token/image preparation and reference owner; never F's KV or
        # embeddings. Capture only already selected final hidden rows.
        reference_inputs = dict(model_inputs, use_cache=False, return_dict=True,
            stage1_selected_positions=keys, stage1_selected_chunk_positions=chunks)
        reference_inputs.pop('labels', None)
        if enabled:
            reference_inputs['stage1_capture_selected_hidden'] = True
        output = self.ema_teacher.model(**reference_inputs)
        if enabled:
            from .cache_payloads import capture_teacher_forward_state
            cache.publish('reference', source_metadata.samples[0].source_index, {
                'kind': 'reference', 'positions': tuple(q for _, q in keys),
                'sequence_length': source_metadata.sequence_length, 'chunk_positions': chunks,
                'hidden': output.stage1_selected_hidden_states,
                'forward_state': capture_teacher_forward_state(self.ema_teacher.model),
            })
        return output.stage1_selected_logits

    def save_model(self, output_dir=None, _internal_call=False):
        if not self._v5_contract_ready:
            raise RuntimeError('v5 checkpoint requires validated initial contract')
        result = super().save_model(output_dir, _internal_call=_internal_call)
        self._publish_v5_files(Path(output_dir or self.args.output_dir))
        return result

    def _load_stage1_ema_checkpoint(self, checkpoint_dir):
        super()._load_stage1_ema_checkpoint(checkpoint_dir)
        from .v5_reference import _state_identity
        if _state_identity(self.ema_teacher, self.ema_teacher.state_dict()) != self.v5_reference_identity.effective_state_sha256:
            raise ValueError('saved EMA would replace immutable v5 initial reference')

    def _build_v5_runtime_contract(self, loader):
        """Describe the prepared loader and effective profile without changing it."""
        import hashlib
        import inspect
        from transformers import Trainer
        from .equivalence import read_stage1_backend_settings
        gas, microbatches = self.args.gradient_accumulation_steps, len(loader)
        qualified = lambda obj: type(obj).__module__+'.'+type(obj).__qualname__
        schedule = dict(dataset_samples=len(self.train_dataset),
            prepared_microbatches_per_rank=microbatches, prepared_loader=qualified(loader),
            sampler=qualified(loader.sampler), batch_sampler=qualified(loader.batch_sampler),
            dataloader_drop_last=self.args.dataloader_drop_last,
            world_size=self.accelerator.num_processes, gradient_accumulation_steps=gas,
            effective_global_batch=gas*self.accelerator.num_processes*self.args.per_device_train_batch_size,
            configured_max_steps=self.args.max_steps, num_train_epochs=self.args.num_train_epochs,
            one_exposure_window_updates=(microbatches+gas-1)//gas,
            parent_epoch_update_budget=self.set_initial_training_values(self.args,loader,
                gas*self.accelerator.num_processes*self.args.per_device_train_batch_size)[1],
            final_window_microbatches=(microbatches-1)%gas+1 if microbatches else 0,
            tail_native_denominator='configured_gas_world_size',
            dil_denominator=('actual_window_selected_dil_tokens' if self.v5_config.selection == 'clean'
                             else 'actual_window_eligible_dil_tokens'),
            auxiliary_denominator='actual_window_original_blocks_including_empty')
        execution = {key:getattr(self.args,key) for key in (
            'per_device_train_batch_size','gradient_accumulation_steps','bf16','seed',
            'learning_rate','warmup_steps','weight_decay','max_grad_norm',
            'gradient_checkpointing','dataloader_num_workers','optim','lr_scheduler_type')}
        qwen = resolve_qwen2_5_vl_model(self.model)
        adapter = self.model.peft_config['default']
        execution.update(stage1_v5_debug=self.stage1_v5_debug, tuning_method=self.ema_teacher.method,ema_tau=self.ema_teacher.decay,
            zero_stage=self.deepspeed_runtime_profile.zero_stage,
            deepspeed_profile=self.deepspeed_runtime_profile.to_dict(),
            attention_implementation=qwen.model.config._attn_implementation,
            vision_attention_implementation=qwen.visual.config._attn_implementation,
            latent_size=qwen.config.latent_size,compress_strategy=qwen.config.compress_strategy,
            max_sequence_length=getattr(self.args,'max_seq_length',None),
            lora_r=adapter.r,lora_alpha=adapter.lora_alpha,lora_dropout=adapter.lora_dropout,
            lora_target_modules=','.join(self.stage1_peft_metadata.requested_target_suffixes),
            resolved_target_modules=sorted(adapter.target_modules),
            model=self.stage1_peft_metadata.base_model_reference,
            coverage_p=self.coverage_p,image_pool_k=self.image_pool_k,
            helper_group_L=self.helper_group_L,textual_states=self.textual_states,
            loss_chunk_positions=self.v5_config.loss_chunk_positions,
            backend=read_stage1_backend_settings(),
            packages={key:packages.version(key) for key in ('torch','transformers','accelerate','deepspeed','peft','trl')})
        query_config = getattr(self, 'query_config', Stage1QueryConfig())
        layers = (resolve_query_layers(query_config.attention_layers, len(qwen.model.layers))
                  if query_config.mode == 'adaptive' else ())
        execution['query_config'] = query_execution_config(query_config, layers)
        from .v5_donors import select_reverse_topk_donors
        from transformers.models.qwen2_5_vl.stage1_v5_replay import run_v5_packed_replay
        execution['source_sha256'] = {
            key:hashlib.sha256(Path(inspect.getsourcefile(cls)).read_bytes()).hexdigest()
            for key,cls in (('v5_trainer',CustomTrainerStage1V5),
                            ('stage1_trainer',CustomTrainerStage1),('transformers_trainer',Trainer),
                            ('query',Stage1QueryConfig),('targets',Stage1TeacherTargetBuilder),
                            ('history_attention',ReducedHistoryAttentionObserver),
                            ('history_teacher',HistoryPrefixAdapter),
                            ('v5_donors',select_reverse_topk_donors),
                            ('v5_replay',run_v5_packed_replay))}
        if self.v5_config.lambda_tail_kl != 0 or self.v5_config.tail_kl_temperature != 1:
            from step_preference.topk_entropy_loss import entropy_support_tail_kl
            execution['source_sha256'].update({
                key: hashlib.sha256(Path(inspect.getsourcefile(function)).read_bytes()).hexdigest()
                for key, function in (('tail_kl', entropy_support_tail_kl),
                                      ('v5_objective', compute_v5_loss_sums))})
        return build_v5_contract(self.v5_config,self.v5_reference_identity,
            ce_weight=self.ce_weight,sim_weight=self.sim_weight,schedule=schedule,execution=execution)

    @staticmethod
    def _effective_v5_contract(contract):
        # Only approved location fields are replaced by their authenticated
        # content identity. Coefficients, packages, source and all other values
        # remain in the compared/saved semantics.
        effective = json.loads(json.dumps(contract, allow_nan=False))
        execution = effective['execution']
        execution.setdefault('query_config', query_execution_config(Stage1QueryConfig(), ()))
        execution.pop('stage1_v5_debug', None)  # Instrumentation is not resume semantics.
        if 'model' in execution:
            execution['model'] = effective['reference']['base_content_sha256']
        if 'deepspeed_profile' in execution:
            profile = execution['deepspeed_profile']
            profile['config_path'] = profile['config_digest']
        return effective

    def _agree_v5_contract(self, contract):
        import hashlib
        def digest(value):
            canonical = json.dumps(self._effective_v5_contract(value),
                                   sort_keys=True, separators=(',', ':'), allow_nan=False)
            return hashlib.sha256(canonical.encode()).digest()
        # Bind the saved contract too: no rank may independently proceed from a
        # different resume identity while the other ranks enter publication.
        local_bytes = digest(contract) + (digest(self.v5_saved_contract)
                                          if self.v5_saved_contract is not None else bytes(32))
        # Debug norm collection issues collectives: CURRENT ranks must agree,
        # while saved instrumentation stays neutral for resume compatibility.
        local_bytes += bytes((int(getattr(self, 'stage1_v5_debug', False)),))
        local = torch.tensor(list(local_bytes),dtype=torch.uint8,device=self.accelerator.device)
        gathered = self.accelerator.gather(local).reshape(-1,local.numel())
        if (gathered.shape[0] != self.accelerator.num_processes
                or not bool(gathered.eq(local.unsqueeze(0)).all())):
            raise ValueError('Stage1-v5 effective contract agreement failed across ranks')

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        self._v5_contract_ready = False
        requested = kwargs.get('model_path',resume_from_checkpoint)
        checkpoint = get_last_checkpoint(self.args.output_dir) if requested is True else (
            None if requested is None or requested is False else requested)
        if checkpoint is not None:
            saved = load_v5_contract(Path(checkpoint))
            validate_v5_resume(saved,{**asdict(self.v5_config),
                'ce_weight':self.ce_weight,'sim_weight':self.sim_weight})
            if self.v5_saved_contract != saved:
                raise ValueError('v5 resume requires composition-restored initial reference')
        contract = self._build_v5_runtime_contract(self.get_train_dataloader())
        self._agree_v5_contract(contract)
        if (self.v5_saved_contract is not None and self._effective_v5_contract(self.v5_saved_contract)
                != self._effective_v5_contract(contract)):
            raise ValueError('saved v5 schedule/execution/reference conflicts with runtime')
        self._v5_contract = contract
        self._publish_v5_files(Path(self.args.output_dir))
        self._v5_contract_ready = True
        return super().train(resume_from_checkpoint=resume_from_checkpoint,trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,**kwargs)
