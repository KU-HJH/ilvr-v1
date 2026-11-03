"""Actual decoder/vision entry accounting for one conditioning loss call."""
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from time import perf_counter

import torch
from torch import Tensor, nn


class DecoderCallLedger:
    roles = frozenset({'base', 'gain_clean', 'reference', 'packed_pair', 'iier', 'helper', 'forbidden_extra'})

    def __init__(self):
        self._role = ContextVar('prefix_execution_role', default='forbidden_extra')
        self._entries = {'decoder': Counter(), 'vision': Counter()}
        self._tokens = {'decoder': Counter(), 'vision': Counter()}
        self._recompute = {'decoder': defaultdict(Counter), 'vision': defaultdict(Counter)}
        self._seconds = defaultdict(float)
        self._gpu_seconds = defaultdict(float)
        self._pending_events = []
        self._order = []
        self._gain_failure = None

    def begin_region(self, name, device=None):
        events = None
        if getattr(device, 'type', None) == 'cuda':
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        return name, perf_counter(), events

    def end_region(self, region):
        name, start, events = region
        self._seconds[name] += perf_counter() - start
        if events is not None:
            events[1].record()
            self._pending_events.append((name, *events))

    @contextmanager
    def time_region(self, name, *, device=None):
        region = self.begin_region(name, device)
        try:
            yield
        finally:
            self.end_region(region)

    def resolve_timings(self):
        # Called only at the trainer's normal report/update boundary.
        pending, self._pending_events = self._pending_events, []
        for name, start, end in pending:
            end.synchronize()
            self._gpu_seconds[name] += start.elapsed_time(end) / 1000.

    @property
    def current_role(self):
        return self._role.get()

    @contextmanager
    def role(self, name):
        if name not in self.roles:
            raise ValueError(f'unknown decoder role: {name}')
        token = self._role.set(name)
        try:
            yield self
        finally:
            self._role.reset(token)

    def record_entry(self, kind, *, query_tokens):
        if kind not in self._entries or type(query_tokens) is not int or query_tokens < 0:
            raise ValueError('invalid entry kind or query token count')
        role = self.current_role
        if role == 'forbidden_extra':
            raise RuntimeError('forbidden_extra model entry without an authorized role')
        if kind == 'decoder' and role != 'helper':
            expected_previous = {'base': (), 'gain_clean': ('base',),
                'reference': ('base', 'gain_clean'),
                'packed_pair': ('base', 'gain_clean', 'reference')}
            if role == 'iier':
                valid = self._order in (['base'], ['base', 'gain_clean', 'reference', 'packed_pair'])
                if self._gain_failure is not None:
                    valid = valid or self._order in (['base', 'gain_clean'],
                                                     ['base', 'gain_clean', 'reference'])
            else:
                valid = tuple(self._order) == expected_previous[role]
            if not valid:
                raise RuntimeError(f'decoder call order violation: {self._order} -> {role}')
            self._order.append(role)
        self._entries[kind][role] += 1
        self._tokens[kind][role] += query_tokens

    def record_gain_failure(self, error):
        """Allow the remaining IIER baseline call before collective rejection."""
        if self._gain_failure is None:
            self._gain_failure = str(error)

    def record_recompute(self, role, layer_index, *, kind='decoder'):
        if (kind not in self._recompute or role not in self.roles - {'forbidden_extra'}
                or type(layer_index) is not int or layer_index < 0):
            raise ValueError('invalid checkpoint recompute role/layer')
        self._recompute[kind][role][layer_index] += 1

    def validate_initial(self, *, gain_active, local_blocks, iier_enabled):
        expected = ['base']
        if gain_active and local_blocks:
            expected += ['gain_clean', 'reference', 'packed_pair']
        if iier_enabled:
            expected.append('iier')
        if self._order != expected:
            raise RuntimeError(f'decoder initial budget mismatch: expected {expected}, got {self._order}')

    def snapshot(self):
        return deepcopy({'decoder_initial': {k: v for k, v in self._entries['decoder'].items() if k != 'helper'},
            'helper_decoder_initial': self._entries['decoder']['helper'],
            'gain_failure': self._gain_failure,
            'vision_initial': dict(self._entries['vision']),
            'query_tokens': {k: dict(v) for k, v in self._tokens.items()},
            'checkpoint_recompute': {k: dict(v) for k, v in self._recompute['decoder'].items()},
            'vision_checkpoint_recompute': {k: dict(v) for k, v in self._recompute['vision'].items()},
            'host_elapsed_seconds': dict(self._seconds),
            'gpu_elapsed_seconds': dict(self._gpu_seconds)})


def _entry_modules(model):
    decoders = [m for m in model.modules() if type(m).__name__ == 'Qwen2_5_VLModel']
    visions = [m for m in model.modules() if type(m).__name__ == 'Qwen2_5_VisionTransformerPretrainedModel']
    if len(decoders) != 1 or len(visions) != 1:
        raise ValueError('conditioning accounting requires exactly one Qwen language decoder and vision module')
    return (('decoder', decoders[0]), ('vision', visions[0]))


@contextmanager
def instrument_prefix_execution(student: nn.Module, teacher: nn.Module, ledger: DecoderCallLedger):
    """Hook real entry modules, including nested PEFT/DDP wrappers; no CUDA sync."""
    handles, seen, checkpoint_bindings = [], set(), []
    try:
        for owner, model in (("student", student), ("teacher", teacher)):
            for kind, module in _entry_modules(model):
                if id(module) in seen:
                    raise ValueError('student and teacher execution modules must be distinct')
                seen.add(id(module))
                starts = []
                if hasattr(module, '_gradient_checkpointing_func'):
                    original = module._gradient_checkpointing_func
                    layers = module.layers if kind == 'decoder' else module.blocks
                    indices = {id(layer): i for i, layer in enumerate(layers)}
                    def checkpoint_entry(function, *args, original=original, indices=indices, kind=kind, **kwargs):
                        role = ledger.current_role
                        layer = getattr(function, '__self__', None)
                        index = indices[id(layer)]
                        calls = 0
                        def layer_entry(*inputs, **call_kwargs):
                            nonlocal calls
                            recompute = calls > 0
                            calls += 1
                            if recompute:
                                ledger.record_recompute(role, index, kind=kind)
                                device = next((v.device for v in inputs if isinstance(v, Tensor)), None)
                                with ledger.time_region(f'recompute/{kind}/{role}', device=device):
                                    return function(*inputs, **call_kwargs)
                            return function(*inputs, **call_kwargs)
                        return original(layer_entry, *args, **kwargs)
                    checkpoint_bindings.append((module, original))
                    module._gradient_checkpointing_func = checkpoint_entry
                def before(module, args, kwargs, kind=kind, starts=starts, owner=owner):
                    permitted = {'helper', 'reference'} if owner == 'teacher' else {'base', 'gain_clean', 'packed_pair', 'iier'}
                    if ledger.current_role not in permitted:
                        raise RuntimeError(f'forbidden {owner} model entry for role {ledger.current_role}')
                    value = kwargs.get('inputs_embeds')
                    if value is None:
                        value = kwargs.get('input_ids')
                    if value is None and args:
                        value = args[0]
                    count = 0
                    if isinstance(value, Tensor):
                        count = value.shape[0] * value.shape[1] if kind == 'decoder' else value.shape[0]
                    ledger.record_entry(kind, query_tokens=int(count))
                    starts.append(ledger.begin_region(f'{kind}/{ledger.current_role}',
                        getattr(value, 'device', None)))
                def after(module, args, kwargs, output, kind=kind, starts=starts):
                    if starts:
                        ledger.end_region(starts.pop())
                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(module.register_forward_hook(after, with_kwargs=True, always_call=True))
        yield
    finally:
        for module, original in reversed(checkpoint_bindings):
            module._gradient_checkpointing_func = original
        for handle in reversed(handles):
            handle.remove()


# Only source text is authenticated here; checkpoint provenance remains owned by
# the existing PEFT/full artifact witness. Never walk a weight/cache directory.
import hashlib
import json
from pathlib import Path
import subprocess
from artifact_paths import CANONICAL_REPOSITORY_ROOT

CANONICAL_RESULTS_ROOT = CANONICAL_REPOSITORY_ROOT / 'results'
PREFIX_SOURCE_FILES = tuple(f'src/stage1/{name}.py' for name in (
    'prefix_mask_config', 'prefix_mask_metadata', 'prefix_mask_plan', 'prefix_mask_reference',
    'prefix_mask_accounting', 'helper_attention_proxy', 'vision_gain', 'vision_gain_execution',
    'vision_gain_packed_execution', 'vision_gain_update', 'trainer', 'composition', 'checkpointing',
    'targets', 'data', 'types', 'clean_kl', 'text_kl')) + ('src/utils_deepseed.py', 'run_stage1_training.sh') + tuple(
    f'transformers/src/transformers/models/qwen2_5_vl/{name}.py' for name in (
        'stage1_prefix_conditioning', 'stage1_prefix_trace', 'stage1_vision_gain_packed',
        'modeling_qwen2_5_vl', 'modular_qwen2_5_vl'))


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def prefix_source_identity(root, config, *, sources=PREFIX_SOURCE_FILES):
    root = Path(root)
    digest = hashlib.sha256()
    total_bytes = 0
    for name in sources:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.suffix not in {'.py', '.sh'}:
            raise ValueError('source identity accepts bounded source files only')
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root.resolve()) or resolved.stat().st_size > 2_000_000:
            raise ValueError('source file escapes root or exceeds source limit')
        total_bytes += resolved.stat().st_size
        if total_bytes > 8_000_000:
            raise ValueError('source identity exceeds 8 MB limit')
        digest.update(name.encode() + b'\0' + resolved.read_bytes())
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
                          capture_output=True).stdout.decode().strip()
    # Path-limited diff excludes all weights, checkpoints, datasets and logs.
    diff = subprocess.run(['git', 'diff', '--no-ext-diff', '--no-textconv', 'HEAD', '--', *sources],
                          cwd=root, check=True, capture_output=True).stdout
    return {'worktree_HEAD': head, 'dirty_diff_hash': hashlib.sha256(diff).hexdigest(),
            'source_hash': digest.hexdigest(), 'source_files': list(sources),
            'config_hash': canonical_digest(config.to_metadata())}


def _compact_image_selections(selected):
    """Losslessly encode image positions; keep query-local text selections intact."""
    compact = []
    for row in selected:
        ranges = []
        for position in row['image_keys']:
            if ranges and ranges[-1][1] == position:
                ranges[-1][1] = position + 1
            else:
                ranges.append([position, position + 1])
        compact.append({**{key: value for key, value in row.items() if key != 'image_keys'},
                        'image_key_ranges': ranges})
    return compact


def prefix_diagnostics_report(report, *, config, identity, metrics, counts,
                              parameter_version, teacher_version):
    """Expose already observed quantities; unavailable values are explicit."""
    execution = report.get('execution', {})
    queries = report.get('queries', ())
    unmeasured = 'unmeasured'
    selected = report.get('selected_source_ids', unmeasured)
    compact_images = config.mask_all_original_image_keys and isinstance(selected, (list, tuple))
    if compact_images:
        selected = _compact_image_selections(selected)
    kl = [row['kl'] for row in queries]
    host = execution.get('host_elapsed_seconds', {})
    def elapsed(*names, prefix=None):
        values = [value for key, value in host.items() if key in names or (prefix and key.startswith(prefix))]
        return sum(values) if values else unmeasured
    durations = dict(helper_vision=elapsed('vision/helper'), base=elapsed('decoder/base'),
        gain_clean=elapsed('decoder/gain_clean'), reference=elapsed('decoder/reference'),
        observer=elapsed('observer'), selector=elapsed('selector'), packed=elapsed('decoder/packed_pair'),
        backward_checkpoint=elapsed(prefix='recompute/'), update=elapsed('optimizer_gain_ema_update'))
    window_identity = {key: metrics.get(key, unmeasured) for key in
        ('accumulation_window_index', 'parameter_version', 'teacher_version', 'local_microbatch_count')}
    return {
        'identity': {**identity, 'parameter_version': parameter_version,
            'alias_version': parameter_version, 'reference_teacher_version': teacher_version,
            'helper_snapshot_hash': report.get('snapshot_id', unmeasured),
            'latent_input_realization_id': report.get('snapshot_id', unmeasured),
            'plan_hash': report.get('plan_hash', unmeasured)},
        'selection': {'helper_trace_method': 'helper_prediction_attention_v1',
            'actual_selected_positions': report.get('actual_selected_positions', unmeasured),
            'latent_prediction_query_ids': report.get('latent_prediction_query_ids', unmeasured),
            'source_groups': report.get('source_groups', unmeasured),
            'selected_source_ids': selected,
            **({'selected_source_ids_encoding': 'image_key_half_open_ranges_v1'} if compact_images else {}),
            'query_summaries': queries, 'token_KL_distribution': kl,
            'flat_KL_fraction': report.get('flat_KL_fraction', unmeasured),
            'flat_KL_definition': 'fraction_of_queries_in_samples_with_KL_range_at_most_epsilon'},
        'scoring': {**{key: report.get(key, unmeasured) for key in ('U_i', 'e_i', 'm_i', 'local_K')},
            'S_F_if_already_available': report.get('S_F_if_already_available', unmeasured),
            'S_F_scope': 'local_microbatch_per_block',
            'microbatch_identity': {
                'microbatch_index': report.get('microbatch_index', unmeasured),
                'accumulation_window_index': report.get('accumulation_window_index', unmeasured),
                'parameter_version': parameter_version, 'teacher_version': teacher_version},
            'window_index_scope': 'session_local_since_controller_initialization',
            'global_window_identity': window_identity,
            'global_S_R': metrics.get('S_R', unmeasured),
            'global_S_RV': metrics.get('S_RV', unmeasured),
            'S_R_scope': 'local_microbatch_per_block' if 'S_R' in report else 'last_completed_accumulation_window_global_block_mean',
            'S_RV_scope': 'local_microbatch_per_block' if 'S_RV' in report else 'last_completed_accumulation_window_global_block_mean',
            'S_R': report.get('S_R', metrics.get('S_R', unmeasured)),
            'S_RV': report.get('S_RV', metrics.get('S_RV', unmeasured)),
            'conditional_gain': metrics.get('G_cond', unmeasured),
            'global_valid_block_count': metrics.get('block_count', unmeasured),
            'global_metrics_scope': 'last_completed_accumulation_window'},
        'execution': {**execution, 'execution_profile': config.to_metadata()['execution_profile'],
            'branch_mode': report.get('branch_mode', unmeasured), 'condition_count': 2,
            'online_ablation_calls': 0, 'online_anchor_calls': 0, 'extra_attention_trace_calls': 0,
            'selector_extra_decoder_calls': 0, 'online_full_oracle_calls': 0,
            **{key: report.get(key, unmeasured) for key in
               ('R_pack', 'clean_bank_size', 'mask_shape', 'reference_query_rows')}},
        'cost': {'host_elapsed_seconds': execution.get('host_elapsed_seconds', {}),
            'gpu_elapsed_seconds': execution.get('gpu_elapsed_seconds', {}),
            'timing_status': 'only_named_regions_measured',
            'peak_memory': report.get('peak_memory', unmeasured),
            'peak_memory_scope': 'process_allocator_high_water_without_reset',
            'reference_peak_memory': unmeasured,
            'trace_buffer_bytes': report.get('trace_buffer_bytes', unmeasured),
            **{f'elapsed_{key}': value for key, value in durations.items()}},
        'gradient_and_update': {'base_grad_norm': report.get('base_grad_norm', unmeasured),
            'gain_positive_grad_norm': metrics.get('gain_positive_grad_norm', unmeasured),
            'gain_negative_grad_norm': metrics.get('gain_negative_grad_norm', unmeasured),
            'gain_grad_norm_scope': 'last_completed_accumulation_window_global_block_mean',
            'global_window_identity': window_identity,
            'optimizer_step_successful': metrics.get('optimizer_step_successful', unmeasured),
            'base_grad_norm_scope': report.get('base_grad_norm_scope', unmeasured),
            'existing_projection_outputs': report.get('projection_outputs', dict(metrics)),
            'displacement_norm': metrics.get('auxiliary_norm', unmeasured),
            'optimizer_update_count': report.get('optimizer_update_count', parameter_version),
            'gain_update_count': counts.get('updates', 0),
            'gain_update_count_semantics': 'eligible_boundaries_including_zero_or_tiny', 'EMA_update_count': report.get('EMA_update_count', teacher_version)},
        'offline_evaluation_only': dict.fromkeys(('D_delta_when_measured', 'S_V',
            'matched_latent_swap_results', 'additional_decoder_calls', 'additional_time',
            'evaluation_subset_id'), unmeasured),
    }


def write_prefix_diagnostics(path: Path, report: Mapping[str, object]) -> None:
    import os
    import stat
    required = {'identity', 'selection', 'scoring', 'execution', 'cost',
                'gradient_and_update', 'offline_evaluation_only'}
    if set(report) != required:
        raise ValueError('invalid prefix diagnostic schema')
    path = Path(path)
    root = CANONICAL_RESULTS_ROOT.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or len(resolved.relative_to(root).parts) < 2:
        raise ValueError('prefix diagnostics require a canonical results experiment path')
    line = json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    if len(line.encode()) > 4_000_000:
        raise ValueError('prefix diagnostic record exceeds bounded 4 MB limit')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError('prefix diagnostics require a regular file')
        handle.write(line)
