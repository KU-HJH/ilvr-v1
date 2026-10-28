"""Explicit bounded verifier. Never imported or installed by production runs."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import TrainerCallback

from .equivalence import canonical_tensor_sha256


def fingerprint(value):
    if isinstance(value, torch.Tensor):
        return {'sha256': canonical_tensor_sha256(value), 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, np.ndarray):
        return fingerprint(torch.from_numpy(value.copy()))
    if is_dataclass(value):
        return fingerprint(asdict(value))
    if isinstance(value, dict):
        return {str(key): fingerprint(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [fingerprint(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'unsupported validation value: {type(value).__name__}')


def rng_state():
    return fingerprint({'python': random.getstate(), 'numpy': np.random.get_state(),
        'cpu': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None})


def assert_frozen_replay(live, replay):
    for record in (live, replay):
        if record['rng_before'] != record['rng_after']:
            raise ValueError('frozen producer consumes RNG; skipped-call replay is ineligible')
    if live['payload'] != replay['payload']:
        raise ValueError('frozen payload/projection/state mismatch')


def _comparison_fields(record):
    required = ('before', 'microbatches', 'optimizer_steps', 'after', 'workers', 'world_size',
                'profile', 'deterministic', 'kind', 'initial_step', 'final_step')
    def require(condition, label):
        if not condition:
            raise ValueError('incomplete acceptance evidence: ' + label)
    def tensor(value):
        if isinstance(value, dict):
            digest = value.get('sha256')
            if isinstance(digest, str) and len(digest) == 64 and isinstance(value.get('shape'), list) and isinstance(value.get('dtype'), str):
                return True
            return any(tensor(item) for item in value.values())
        return isinstance(value, list) and any(tensor(item) for item in value)
    def quantities(value, names, label):
        require(isinstance(value, dict), label)
        for name in names:
            require(tensor(value.get(name)), label + '.' + name)
    require(all(name in record for name in required), 'required fields')
    require(record['workers'] == record['world_size'] == 4, 'topology/workers')
    require(record['kind'] == 'correctness' and record['deterministic'] is True, 'deterministic execution')
    require(record['profile'] in ('v5', 'native-ema'), 'profile')
    require(record['initial_step'] == 0 and record['final_step'] == 2, 'complete two-update arm')
    quantities(record['before'], ('student', 'teacher', 'rng'), 'before')
    quantities(record['after'], ('student', 'teacher', 'rng', 'scheduler'), 'after')
    require(record['after'].get('successful_updates') == 2, 'successful updates')
    rows, steps = record['microbatches'], record['optimizer_steps']
    require(isinstance(rows, list) and len(rows) == 8, 'eight actual microbatches')
    require(isinstance(steps, list) and len(steps) == 2, 'two actual optimizer calls')
    for row in rows:
        quantities(row, ('inputs', 'targets', 'loss', 'rng_before', 'rng_after'), 'microbatch')
        require(type(row.get('query_count')) is int and row['query_count'] >= 0, 'query count')
        require(isinstance(row.get('projection'), list), 'projection observations')
        if record['profile'] == 'v5' and row['query_count']:
            quantities(row, ('reference',), 'microbatch')
            require(bool(row['projection']), 'reference projection chunks')
            for chunk in row['projection']:
                quantities(chunk, ('hidden',), 'projection')
                require(type(chunk.get('autocast')) is bool and bool(chunk.get('owner')), 'projection context')
            selection = row.get('selection')
            quantities(selection, ('orig_positions', 'entropy_support'), 'semantic selection')
            require(isinstance(selection.get('text_keys'), dict) and bool(selection['text_keys']), 'Text-off selection')
        else:
            require(not row['projection'], 'inapplicable reference projection')
    for step in steps:
        quantities(step, ('gradients', 'master_parameters_after', 'group_state_after', 'rng_before', 'rng_after'), 'optimizer')
        require(isinstance(step.get('learning_rates'), list) and bool(step['learning_rates'])
                and all(isinstance(rate, (int, float)) for rate in step['learning_rates']), 'learning rates')
    return {name: record[name] for name in required}


def compare_acceptance(a, repeated, cached):
    original = _comparison_fields(a)
    if original != _comparison_fields(repeated):
        raise ValueError('uncached deterministic A/A differs at ' + first_difference(original, _comparison_fields(repeated))
                         + '; cache A/B is not interpretable')
    if not any(rate > 0 for step in a['optimizer_steps'] for rate in step['learning_rates']):
        raise ValueError('nonzero learning-rate update evidence is required')
    if a['optimizer_steps'][0]['master_parameters_after'] == a['optimizer_steps'][1]['master_parameters_after']:
        raise ValueError('nonzero parameter update evidence is required')
    if a['profile'] == 'native-ema' and a['before']['teacher'] == a['after']['teacher']:
        raise ValueError('moving EMA update evidence is required')
    if original != _comparison_fields(cached):
        raise ValueError('deterministic cache A/B differs at ' + first_difference(original, _comparison_fields(cached)))
    if any(record.get('production_checks', {}).get('optional_calls') != 0 for record in (a, repeated, cached)):
        raise ValueError('production optional-check witness is absent or nonzero')
    projection = ('not_applicable:native_has_no_reference' if a['profile'] == 'native-ema'
                  else True if any(row['query_count'] for row in a['microbatches'])
                  else 'not_applicable:no_reference_queries')
    return dict(payload=True, projection=projection, rng_state=True, semantic_selection=True,
                loss=True, gradient=True, nonzero_update=True, ema=True, production_checks=True)


def first_difference(left, right, path='root'):
    if type(left) is type(right) and isinstance(left, dict) and left.keys() == right.keys():
        for key in left:
            if left[key] != right[key]:
                return first_difference(left[key], right[key], f'{path}.{key}')
    if type(left) is type(right) and isinstance(left, list) and len(left) == len(right):
        for index, (a, b) in enumerate(zip(left, right)):
            if a != b:
                return first_difference(a, b, f'{path}[{index}]')
    return path


def validate_validation_profile(args):
    mode = args.stage1_cache_validation
    stop = args.stage1_cache_validation_stop
    if mode not in ('payload', 'correctness', 'performance') or stop not in (1, 2):
        raise ValueError('cache verifier requires a bounded zero/two-update protocol')
    if mode == 'performance' and stop != 1:
        raise ValueError('performance verifier is bounded to one update')
    common = dict(max_steps=2160, epochs=10, learning_rate=5e-5, warm_up_steps=10, seed=42,
        per_device_train_batch_size=1, gradient_accumulation_steps=4, dataloader_num_workers=4,
        tuning_method='dora', ce_weight=.1, stage1_match_v5_execution=False,
        vision_attention_implementation='flash_attention_2')
    v5 = args.stage1_variant == 'v5'
    common.update(sim_weight=.1 if v5 else 1., ema_tau=1. if v5 else .999,
        attention_implementation='sdpa' if v5 else 'flash_attention_2',
        stage1_runtime_profile='default' if v5 else 'native-ema-0180')
    if any(getattr(args, key, None) != value for key, value in common.items()):
        raise ValueError('cache verifier must preserve its approved profile coefficients and execution')
    if v5:
        expected = dict(lambda_preference=1., lambda_topk_entropy=.1, lambda_topk_mass=.05,
            lambda_topk_chosen_mass=0., lambda_entropy=0., preference_topk_k=50,
            preference_topk_mode='independent', topk_include_gt=False, selection='clean',
            preference_entropy_top_percent=20., text_kl_weight=.01, voff_mode='attention_mask',
            loss_chunk_positions=128, beta=1., margin=0.)
        if any(getattr(args.stage1_v5_config, key) != value for key, value in expected.items()):
            raise ValueError('cache verifier must preserve its approved V5 objective')


def configure_validation(args):
    validate_validation_profile(args)
    args._stage1_cache_validation = True
    # No algorithm/backend substitution. Environment flags are set by the
    # coordinator before importing CUDA/FlashAttention in the child process.
    if args.stage1_cache_validation == 'correctness':
        if os.environ.get('FLASH_ATTENTION_DETERMINISTIC') != '1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('deterministic verifier environment was not propagated')
        torch.use_deterministic_algorithms(True)


def _forward_state(teacher):
    from .cache_payloads import capture_teacher_forward_state
    return fingerprint({'rope': capture_teacher_forward_state(teacher),
                        'buffers': dict(teacher.named_buffers())})


def _write_rank_record(output_dir, record):
    rank = torch.distributed.get_rank()
    output_dir.mkdir(parents=True, exist_ok=True)
    record.update(rank=rank, world_size=torch.distributed.get_world_size(),
        host=os.uname().nodename, pid=os.getpid(), local_rank=int(os.environ['LOCAL_RANK']),
        deterministic=torch.are_deterministic_algorithms_enabled())
    path = output_dir / f"{os.environ['ILVR_STAGE1_CACHE_ATTEMPT']}-cache-rank-{rank}.json"
    with path.open('x') as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(f'STAGE1_CACHE_VALIDATION rank={rank} world_size={record["world_size"]} record={path}', flush=True)


def payload_witness(trainer, args, output_dir, *, prepare_only=False):
    """Original fast frozen forwards, actual workers and coordinator boundary."""
    from .cache_lifecycle import Stage1CacheEpochCallback
    from .cache_setup import preserve_rng
    from .cache_payloads import capture_teacher_forward_state, restore_teacher_forward_state
    from types import SimpleNamespace
    cache = trainer.run_cache
    if cache is None or cache.phase.value != 0 or len(trainer.train_dataset) != 16:
        raise ValueError('payload witness requires a fresh, complete 16-row run cache')
    teacher = trainer.ema_teacher.model
    teacher.to(device=trainer.accelerator.device).eval()
    trainer._ema_execution_device = trainer.accelerator.device
    initial_forward = capture_teacher_forward_state(teacher)
    rows = {}
    vision_payloads = []
    original_visual = trainer.target_builder._helper_visual
    def helper_visual(*positional, **keywords):
        result = original_visual(*positional, **keywords)
        vision_payloads.append(fingerprint(result))
        return result
    trainer.target_builder._helper_visual = helper_visual
    started = time.monotonic()
    with preserve_rng():
        loader = trainer.get_train_dataloader()
        if loader.num_workers != 4 or len(loader) != 4:
            raise ValueError('payload verifier requires actual four-worker/four-microbatch loader')
        for phase in (0, 1):
            if phase == 1:
                callback = next(c for c in trainer.callback_handler.callbacks if isinstance(c, Stage1CacheEpochCallback))
                callback.on_epoch_end(trainer.args, SimpleNamespace(epoch=1.), None)
                gathered = [None] * 4
                torch.distributed.all_gather_object(gathered, rows)
                rows = {key: value for part in gathered for key, value in part.items()}
                if len(rows) != 16:
                    raise ValueError('payload witness did not observe all retained samples')
                if prepare_only:
                    break
            for batch in loader:
                inputs = trainer._prepare_inputs(batch)
                index = inputs['stage1_batch_metadata'].samples[0].source_index
                metadata = inputs.pop('stage1_v5_metadata', None)
                before = rng_state()
                vision_payloads.clear()
                forward_before = _forward_state(teacher)
                with torch.no_grad(), trainer.compute_loss_context_manager():
                    model_inputs, source, targets = trainer._prepare_stage1_loss_inputs(inputs)
                    projection, reference = [], None
                    if metadata is not None and metadata.query_keys:
                        from peft_common import resolve_qwen2_5_vl_model
                        head = resolve_qwen2_5_vl_model(teacher).lm_head
                        handle = head.register_forward_pre_hook(lambda module, values: projection.append(
                            {'hidden': fingerprint(values[0]), 'autocast': torch.is_autocast_enabled('cuda')}))
                        try:
                            reference = trainer._stage1_reference_logits(model_inputs, metadata.query_keys, source)
                        finally:
                            handle.remove()
                forward_after = _forward_state(teacher)
                no_forward = trainer.query_config.mode == 'adaptive' and not source.samples[0].latent_spans
                record = {'rng_before': before, 'rng_after': rng_state(), 'payload': fingerprint({
                    'inputs': {key: value for key, value in model_inputs.items() if key != 'stage1_cache_sources'},
                    'targets': targets, 'reference': reference, 'projection': projection,
                    'cached_vision': list(vision_payloads) if cache.plan.helper == 'vision' else None,
                    'forward_effect': ('unchanged' if forward_before == forward_after else 'unexpected mutation')
                                      if no_forward else forward_after})}
                if phase == 0:
                    rows[index] = record
                else:
                    assert_frozen_replay(rows[index], record)
    restore_teacher_forward_state(teacher, initial_forward)
    trainer.target_builder._helper_visual = original_visual
    if not prepare_only:
        _write_rank_record(output_dir, {'kind': 'payload', 'workers': 4, 'rows': len(rows),
            'payload': True, 'projection': True, 'rng_state': True, 'lifecycle': True,
            'seconds': time.monotonic() - started,
            'cache_bytes': sum(path.stat().st_size for path in cache.plan.root.glob('*/*.pt'))})


class Stage1CacheValidationObserver(TrainerCallback):
    def __init__(self, manifest, output_dir):
        self.manifest, self.output_dir = manifest, Path(output_dir)
        self.rows, self.optimizer_steps, self.handles = [], [], []
        self.current, self.optimizer = None, None
        self.workers = None
        self.optional_calls = 0
        self.in_reference = False
        self.restorations = []
        self.performance = manifest['mode'] == 'performance'

    def snapshot(self):
        if self.performance:
            return {}
        return {'student': fingerprint({name: value for name, value in self.trainer.model.named_parameters() if value.requires_grad}),
                'teacher': fingerprint(self.trainer.ema_teacher.state_dict()), 'rng': rng_state()}

    def install(self, trainer):
        self.trainer = trainer
        self.profile = 'v5' if hasattr(trainer, 'v5_config') else 'native-ema'
        if self.profile == 'v5' and not self.performance:
            self.observe_v5_semantics()
        original_loader = trainer.get_train_dataloader
        def loader(*args, **kwargs):
            result = original_loader(*args, **kwargs)
            self.workers = result.num_workers
            if self.workers != 4 or len(result) != 4:
                raise ValueError('bounded verifier requires actual workers4 and 16 retained samples')
            return result
        trainer.get_train_dataloader = loader
        for name in ('_validate_targets', '_validate_ema_teacher_device', '_validate_deepspeed_gradient_norm'):
            original = getattr(trainer, name)
            def counted(*args, _original=original, **kwargs):
                self.optional_calls += 1
                return _original(*args, **kwargs)
            setattr(trainer, name, counted)
        original_step = trainer.training_step
        def step(model, inputs, *args, **kwargs):
            if len(self.rows) >= 8 or self.current is not None:
                raise ValueError('cache verifier exceeded its bounded update budget')
            engine = trainer.accelerator.deepspeed_engine_wrapped.engine
            if self.optimizer is None:
                self.optimizer = engine.optimizer.optimizer
                self.handles.extend((self.optimizer.register_step_pre_hook(self.before_optimizer),
                                     self.optimizer.register_step_post_hook(self.after_optimizer)))
            self.current = {} if self.performance else {'inputs': fingerprint({key: value for key, value in inputs.items()
                if key not in ('pixel_values_latent', 'stage1_cache_sources')}),
                'rng_before': rng_state(), 'projection': [], 'selection': {},
                'query_count': len(inputs['stage1_v5_metadata'].query_keys) if self.profile == 'v5' else 0}
            torch.cuda.synchronize()
            started = time.monotonic()
            try:
                loss = original_step(model, inputs, *args, **kwargs)
                torch.cuda.synchronize()
                seconds = time.monotonic() - started
                if self.performance:
                    self.current['seconds'] = seconds
                else:
                    self.current.update(loss=fingerprint(loss.detach()), rng_after=rng_state())
                self.rows.append(self.current)
                return loss
            finally:
                self.current = None
        trainer.training_step = step
        original_prepare = trainer._prepare_stage1_loss_inputs
        def prepare(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            if self.current is not None and not self.performance:
                self.current['targets'] = fingerprint(result[2])
            return result
        trainer._prepare_stage1_loss_inputs = prepare
        if hasattr(trainer, '_stage1_reference_logits'):
            original_reference = trainer._stage1_reference_logits
            def reference(*args, **kwargs):
                self.in_reference = True
                try:
                    result = original_reference(*args, **kwargs)
                    if self.current is not None and not self.performance:
                        self.current['reference'] = fingerprint(result)
                    return result
                finally:
                    self.in_reference = False
            trainer._stage1_reference_logits = reference
            from peft_common import resolve_qwen2_5_vl_model
            owner = resolve_qwen2_5_vl_model(trainer.ema_teacher.model)
            def head(module, inputs):
                if self.current is not None and self.in_reference and not self.performance:
                    self.current['projection'].append({'hidden': fingerprint(inputs[0]),
                        'autocast': torch.is_autocast_enabled('cuda'), 'owner': type(module).__name__})
            self.handles.append(owner.lm_head.register_forward_pre_hook(head))
        trainer.add_callback(self)

    def observe_v5_semantics(self):
        from . import v5_trainer, v5_objective
        for module, name, field, multiple in (
            (v5_trainer, 'select_v5_text_keys', 'text_keys', False),
            (v5_objective, 'select_v5_orig_positions', 'orig_positions', False),
            (v5_objective, 'build_entropy_support', 'entropy_support', True),
            (v5_trainer, 'compute_v5_loss_sums', None, False),
        ):
            original = getattr(module, name)
            def observed(*args, _original=original, _field=field, _multiple=multiple, **kwargs):
                if self.current is not None:
                    self.optional_calls += int(kwargs.get('validate', True) or kwargs.get('collect_diagnostics', False))
                result = _original(*args, **kwargs)
                if self.current is not None and _field is not None:
                    value = fingerprint(result)
                    if _multiple:
                        self.current['selection'].setdefault(_field, []).append(value)
                    else:
                        self.current['selection'][_field] = value
                return result
            self.restorations.append((module, name, original))
            setattr(module, name, observed)

    def restore_observers(self):
        for module, name, original in reversed(self.restorations):
            setattr(module, name, original)
        self.restorations.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def before_optimizer(self, optimizer, args, kwargs):
        rates = [group['lr'] for group in optimizer.param_groups]
        record = {'learning_rates': rates}
        if not self.performance:
            gradients = [p.grad for group in optimizer.param_groups for p in group['params']]
            if not any(value is not None for value in gradients):
                raise ValueError('actual optimizer gradients were not observed')
            record.update(gradients=fingerprint(gradients), rng_before=rng_state())
        self.optimizer_steps.append(record)

    def after_optimizer(self, optimizer, args, kwargs):
        if not self.performance:
            self.optimizer_steps[-1].update(
                group_state_after=fingerprint([optimizer.state.get(p, {}) for group in optimizer.param_groups for p in group['params']]),
                master_parameters_after=fingerprint([p for group in optimizer.param_groups for p in group['params']]),
                rng_after=rng_state())

    def on_train_begin(self, args, state, control, **kwargs):
        self.before = self.snapshot()
        self.initial_step = state.global_step
        self.started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > self.manifest['stop']:
            raise ValueError('cache verifier update ceiling exceeded')
        self.after = self.snapshot()
        self.after.update(scheduler=fingerprint(self.trainer.lr_scheduler.state_dict()),
                          successful_updates=self.initial_step + self.trainer._successful_optimizer_steps)
        if state.global_step == self.manifest['stop']:
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if state.global_step != self.manifest['stop'] or not self.rows:
            raise ValueError('incomplete bounded optimizer witness')
        record = {'kind': self.manifest['mode'], 'profile': self.profile, 'before': self.before,
            'microbatches': self.rows, 'optimizer_steps': self.optimizer_steps, 'after': self.after,
            'workers': self.workers, 'initial_step': self.initial_step, 'final_step': state.global_step,
            'seconds': time.monotonic() - self.started,
            'production_checks': {'optional_calls': self.optional_calls}}
        _write_rank_record(self.output_dir, record)
        self.restore_observers()
