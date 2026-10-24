#!/usr/bin/env python
"""Passive, bounded Stage 1 OFF/cache trace; no generation implementation here.

Run with PYTHONPATH=.:src:transformers/src. Only the controller launches GPU work.
Tensor clones stay process-local; JSON contains summaries, never tensor dumps.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from evaluation.numerics import (NUMERICAL_POLICIES, numerical_policy_scope, numerical_execution_scope,
    observe_numerical_policy, validate_numerical_policy)
from evaluation.adapter_numerics import (ADAPTER_TOKEN_FP64_POLICIES,
    temporary_fp64_token_linear)


MODEL = 'checkpoints/evaluation-staging/stage1-v5-mn-g16-e3-s648-20261014-150317/prepared-model'
BASE = '/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct'
SOURCE = 'data/TEST-stage1-v5-static-cache-source12-20261015-r1.jsonl'
FP64_BASELINE_REPORT = Path('results/kv-mlp-isolation-20261024T081537Z/probe-r1/report.json')
PROFILE = dict(max_new_tokens=32, max_sequence_length=32768, repetition_penalty=1.05,
               temperature=0., top_p=1., generation_limit_policy='bounded')


def require_fidelity(actual, frozen, *, expected_steps=32):
    for key in ('generated_suffix_token_ids', 'prompt_identity', 'stop_reason',
                'latent_start_count', 'latent_block_count'):
        if actual.get(key) != frozen.get(key):
            raise ValueError(f'observer fidelity mismatch: {key}; interpretation is inconclusive')
    if expected_steps is not None and len(actual['generated_suffix_token_ids']) != expected_steps:
        raise ValueError(f'observer fidelity requires the full {expected_steps}-token suffix')


def comparison_steps(left, right):
    """Include the prediction that branches; exclude any unequal-history child."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index + 1
    return min(len(left), len(right))


def tensor_record(value):
    x = value.detach().contiguous().cpu()
    raw = x.reshape(-1).view(torch.uint8).numpy().tobytes()
    finite = x.float()[torch.isfinite(x)]
    return dict(shape=list(x.shape), dtype=str(x.dtype), sha256=hashlib.sha256(raw).hexdigest(),
                norm=float(finite.norm()), max_abs=float(finite.abs().max()) if finite.numel() else None,
                finite_count=int(finite.numel()), count=x.numel())


def tensor_difference(left, right):
    if left.shape != right.shape:
        return dict(value_shape_mismatch=True, equal=False, max_abs=None, unequal_count=None)
    same = (left == right) | (torch.isnan(left) & torch.isnan(right))
    count = int((~same).sum())
    delta = torch.where(same, 0., (left.float()-right.float()).abs())
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    return dict(value_shape_mismatch=False, equal=count == 0, unequal_count=count, finite=finite,
                max_abs=(float(delta.max()) if delta.numel() else 0.) if finite else None)


def last_query_visibility(mask, query_length, key_length, is_causal):
    """Normalize actual SDPA boolean/additive masking without changing its call.

    SDPA implicit causality is upper-left aligned: last query sees k <= qlen-1.
    The observed Qwen path uses is_causal=False for a single cached query.
    """
    visible = torch.arange(key_length) <= (query_length-1 if is_causal else key_length)
    if mask is not None:
        row = mask.detach()[..., -1, :key_length].reshape(-1, key_length)
        if row.shape[0] != 1 and not torch.equal(row, row[:1].expand_as(row)):
            raise ValueError('head-dependent mask cannot be normalized to one visibility row')
        row = row[0].cpu()
        visible &= row if row.dtype == torch.bool else (torch.isfinite(row) & (row > -1e20))
    return visible


def operand_layout(value):
    return dict(dtype=str(value.dtype), shape=list(value.shape), stride=list(value.stride()),
                storage_offset=value.storage_offset(), device=str(value.device))


class MLPComponents:
    """Only the requested up-projection; hooks observe, never replace results.

    Retain the actual outer operand for replay. Component values are private
    snapshots; JSON exposes summaries. No internal arithmetic is intercepted.
    """
    def __init__(self, module, *, enabled=lambda: True, retain_adapter_operands=False):
        self.module, self.enabled = module, enabled
        self.retain_adapter_operands = retain_adapter_operands
        self.stack = ExitStack()
        self.events = []
        self.operand = self.output = None
        self.capture_hash = None
        self.order = 0
        self.closed = False
        self.pending = {}
        self.last_a = {}

    def state(self, module):
        forward = module.forward
        function = getattr(forward, '__func__', forward)
        device_type = next(module.parameters()).device.type
        return dict(training=module.training,
            active_adapters=list(getattr(self.module, 'active_adapters', [])),
            disable_adapters=getattr(self.module, 'disable_adapters', None),
            merged=getattr(self.module, 'merged', None),
            autocast_enabled=torch.is_autocast_enabled(device_type),
            autocast_dtype=str(torch.get_autocast_dtype(device_type)),
            forward=dict(module=getattr(function, '__module__', None),
                qualname=getattr(function, '__qualname__', type(function).__qualname__),
                function_id=id(function), bound_to_module=getattr(forward, '__self__', None) is module),
            parameters={name: operand_layout(p) for name, p in module.named_parameters()})

    def before(self, component, adapter, module, args):
        if not self.enabled():
            return
        x = args[0]
        role = 'token'
        evidence = {}
        if component == 'outer':
            # More than one target invocation is preserved, then rejected as ambiguous.
            self.operand = x
            self.capture_hash = tensor_record(x)['sha256']
        elif component == 'lora_A':
            dora = self.pending.get(('dora', adapter), [])
            token = bool(dora and x is dora[-1]['_operand'])
            identity = (x.ndim == 2 and x.shape[0] == x.shape[1]
                        and torch.equal(x, torch.eye(x.shape[0], device=x.device, dtype=x.dtype)))
            evidence = dict(identity_operand=identity, same_as_dora_operand=token)
            role = ('ambiguous' if token == identity else
                    'identity_reconstruction' if identity else 'token')
        elif component == 'lora_B':
            a = self.last_a.get(adapter)
            role = a['role'] if a is not None and x is a.get('_output') else 'ambiguous'
            evidence = dict(same_as_preceding_a_output=a is not None and x is a.get('_output'),
                            preceding_a_start_order=None if a is None else a['start_order'])
        event = dict(component=component, adapter=adapter, role=role,
            role_evidence=evidence, start_order=self.order, operand=operand_layout(x),
            state=self.state(module), _operand=x)
        if self.retain_adapter_operands and role == 'token' and component in ('lora_A', 'lora_B'):
            event.update(_module=module, input_capture_sha256=tensor_record(x)['sha256'])
        self.order += 1
        self.events.append(event)
        self.pending.setdefault((component, adapter), []).append(event)

    def after(self, component, adapter, _module, _args, output):
        if not self.enabled():
            return
        event = self.pending[(component, adapter)].pop()
        event.update(output=operand_layout(output), end_order=self.order)
        self.order += 1
        value = output if event['role'] != 'token' else output[..., -1:, :]
        event['_value'] = value.detach().clone()
        if self.retain_adapter_operands and event['role'] == 'token' and component in ('lora_A', 'lora_B'):
            event.update(_native_output=output, output_capture_sha256=tensor_record(output)['sha256'])
        if component == 'lora_A':
            event['_output'] = output
            self.last_a[adapter] = event
        if component == 'outer':
            self.output = event['_value']

    def __enter__(self):
        try:
            modules = [('outer', None, self.module)]
            if hasattr(self.module, 'base_layer'):
                modules.append(('base', None, self.module.base_layer))
                for adapter in self.module.active_adapters:
                    for label, attr in [('lora_A', 'lora_A'), ('lora_B', 'lora_B'),
                                        ('dora', 'lora_magnitude_vector')]:
                        collection = getattr(self.module, attr, {})
                        if adapter in collection:
                            modules.append((label, adapter, collection[adapter]))
            for label, adapter, module in modules:
                def pre(m, args, label=label, adapter=adapter):
                    self.before(label, adapter, m, args)
                def post(m, args, output, label=label, adapter=adapter):
                    self.after(label, adapter, m, args, output)
                self.stack.callback(module.register_forward_pre_hook(pre).remove)
                self.stack.callback(module.register_forward_hook(post).remove)
            return self
        except BaseException:
            self.stack.close()
            self.closed = True
            raise

    def __exit__(self, *exception):
        self.closed = True
        result = self.stack.__exit__(*exception)
        # Keep actual DoRA/A/B operands only for the opt-in precheck; the
        # separate original outer reference remains available for outer replay.
        for event in self.events:
            if not (self.retain_adapter_operands and event['role'] == 'token'
                    and event['component'] in ('dora', 'lora_A', 'lora_B')):
                event.pop('_operand', None)
            event.pop('_output', None)
        self.last_a.clear()
        return result


def compare_mlp_components(off, on, *, numerical_policy):
    """Report boundaries only after role, operand integrity and two-sided replay gates."""
    report = dict(status='inconclusive', target=dict(step=1, layer=24, module='mlp.up_proj'),
        invocation_geometry_layout_sensitivity=False, observed={}, replay={},
        internal_accumulator_dtype='unobserved', cuda_kernel_identity='unobserved',
        internal_normalization_correction_add_cast='unobserved',
        limitation='Module boundaries do not isolate internal primitives, pure geometry, kernels, or final-token causality.')
    for side, trace in (('off', off), ('on', on)):
        report['observed'][side] = ([] if trace is None else [
            dict(**{k: v for k, v in event.items() if not k.startswith('_')},
                 value=tensor_record(event['_value']) if '_value' in event else None)
            for event in trace.events])
    if off is None or on is None:
        report['reason'] = 'missing target layer/component observation'
        return report
    expected = [('outer', 'token'), ('base', 'token'), ('dora', 'token'),
                ('lora_A', 'identity_reconstruction'), ('lora_B', 'identity_reconstruction'),
                ('lora_A', 'token'), ('lora_B', 'token')]
    for trace in (off, on):
        if (not trace.closed or [(e['component'], e['role']) for e in trace.events] != expected
                or any('_value' not in e or 'end_order' not in e for e in trace.events)
                or trace.operand is None or trace.output is None):
            report['reason'] = 'missing/ambiguous role, incomplete call, or observers still active'
            return report
    inputs = tensor_difference(off.operand[..., -1:, :], on.operand[..., -1:, :])
    report['observed']['newest_input_difference'] = inputs
    integrity = True
    for side, trace in (('off', off), ('on', on)):
        current = tensor_record(trace.operand)['sha256']
        original_layout = trace.events[0]['operand']
        unchanged = current == trace.capture_hash and operand_layout(trace.operand) == original_layout
        report['replay'][side] = dict(capture_sha256=trace.capture_hash, pre_replay_sha256=current,
            original_layout=original_layout, pre_replay_layout=operand_layout(trace.operand),
            operand_unchanged=unchanged)
        integrity &= unchanged
    if not integrity or not inputs['equal'] or not inputs.get('finite', False):
        report['reason'] = 'mutated retained operand or unequal/nonfinite actual newest inputs'
        return report
    for side, trace in (('off', off), ('on', on)):
        with numerical_execution_scope(numerical_policy, trace.module, require_adapters=False) as execution, torch.inference_mode():
            actual = trace.module(trace.operand)[..., -1:, :]
            observation = observe_numerical_policy(numerical_policy, execution=execution)
        difference = tensor_difference(actual, trace.output)
        report['replay'][side].update(numerical_observation=observation,
            output=tensor_record(actual), difference=difference,
            matches_observed=difference['equal'] and difference.get('finite', False))
    if not all(report['replay'][side]['matches_observed'] for side in ('off', 'on')):
        report['reason'] = 'one or both original-operand replays did not reproduce observed outputs'
        return report
    # End-order is the order in which actual outputs became observable.
    rows = []
    for a, b in zip(off.events, on.events):
        rows.append(dict(component=a['component'], role=a['role'],
            end_order=a['end_order'], difference=tensor_difference(a['_value'], b['_value'])))
    rows.sort(key=lambda row: row['end_order'])
    report['observed']['boundaries'] = rows
    report['observed']['earliest_differing_boundary'] = next(
        (row for row in rows if not row['difference']['equal']), None)
    output_difference = tensor_difference(off.output, on.output)
    report['status'] = 'observed'
    report['invocation_geometry_layout_sensitivity'] = (
        not output_difference['equal'] and off.events[0]['operand'] != on.events[0]['operand'])
    return report


def _fp64_operand_precheck(off, on, *, baseline_fidelity):
    """Replay captured original A/B operands before trying temporary arithmetic."""
    report = dict(status='inconclusive', baseline_fidelity=list(baseline_fidelity),
        input_difference=None, native_replays_match={'A': [False, False], 'B': [False, False]},
        protected_A_difference=None, A64_B32_difference=None, A64_B64_difference=None,
        temporary_operand_dtypes=None)
    if baseline_fidelity != [True, True]:
        report['reason'] = 'original baseline fidelity incomplete'
        return report
    selected = []
    expected = [('outer', 'token'), ('base', 'token'), ('dora', 'token'),
                ('lora_A', 'identity_reconstruction'), ('lora_B', 'identity_reconstruction'),
                ('lora_A', 'token'), ('lora_B', 'token')]
    for trace in (off, on):
        if (trace is None or not trace.closed or
                [(e['component'], e['role']) for e in trace.events] != expected):
            report['reason'] = 'missing/incomplete actual A/B roles'
            return report
        a, b = trace.events[5:]
        for label, event in (('A', a), ('B', b)):
            x, output, module = (event.get(key) for key in ('_operand', '_native_output', '_module'))
            if (not isinstance(x, torch.Tensor) or not isinstance(output, torch.Tensor)
                    or type(module) is not torch.nn.Linear or x.ndim != 3 or x.dtype != torch.float32
                    or module is not getattr(trace.module, 'lora_'+label)[event['adapter']]
                    or getattr(module.forward, '__func__', None) is not torch.nn.Linear.forward
                    or event.get('input_capture_sha256') != tensor_record(x)['sha256']
                    or event['operand'] != operand_layout(x)
                    or event.get('output_capture_sha256') != tensor_record(output)['sha256']
                    or not torch.isfinite(x).all() or not torch.isfinite(output).all()
                    or not torch.equal(output[..., -1:, :], event['_value'])):
                report['reason'] = 'mutated, nonfinite, unsupported, or missing original A/B operand/output'
                return report
        if (a['_operand'] is not trace.events[2].get('_operand')
                or b['_operand'] is not a['_native_output']):
            report['reason'] = 'actual token A/B linkage changed'
            return report
        selected.append((a, b))
    report['input_difference'] = tensor_difference(selected[0][0]['_operand'][..., -1:, :],
                                                  selected[1][0]['_operand'][..., -1:, :])
    if not report['input_difference']['equal'] or not report['input_difference'].get('finite'):
        report['reason'] = 'unequal/nonfinite actual newest A inputs'
        return report
    with torch.inference_mode():
        # All four FP32 positive controls precede every protected operation.
        for index, label in enumerate(('A', 'B')):
            for side, events in enumerate(selected):
                event = events[index]
                replay = event['_module'](event['_operand'])
                difference = tensor_difference(replay, event['_native_output'])
                report['native_replays_match'][label][side] = difference['equal'] and difference.get('finite', False)
        if not all(all(values) for values in report['native_replays_match'].values()):
            report['reason'] = 'original full-operand A/B replay did not reproduce native output'
            return report
        protected_a, ordinary_b, protected_b = [], [], []
        for a, b in selected:
            a64, a_dtype = temporary_fp64_token_linear(a['_module'], a['_operand'])
            b32 = b['_module'](a64)
            b64, b_dtype = temporary_fp64_token_linear(b['_module'], a64)
            protected_a.append(a64[..., -1:, :])
            ordinary_b.append(b32[..., -1:, :])
            protected_b.append(b64[..., -1:, :])
        report.update(protected_A_difference=tensor_difference(*protected_a),
            A64_B32_difference=tensor_difference(*ordinary_b),
            A64_B64_difference=tensor_difference(*protected_b),
            temporary_operand_dtypes={'A': a_dtype, 'B': b_dtype})
    required = (report['protected_A_difference'], report['A64_B64_difference'])
    if not all(d.get('finite') for d in required):
        report['reason'] = 'protected replay is nonfinite or shape-incompatible'
    elif not all(d['equal'] for d in required):
        report.update(status='fail', reason='finite witnessed protected A or A/B difference')
    else:
        report['status'] = 'pass'
    return report


def fp64_adapter_precheck(model, processor, sample, *, baseline_report: dict) -> dict:
    """Cheap two-prediction original-profile reproduction, then exact-operand shadows.

    This is a local premise witness only. It cannot establish full32 correction.
    """
    import eval as evaluation
    from peft_common import resolve_qwen2_5_vl_model
    report = dict(status='inconclusive', baseline_fidelity=[False, False], input_difference=None,
        native_replays_match={'A': [False, False], 'B': [False, False]}, protected_A_difference=None,
        A64_B32_difference=None, A64_B64_difference=None, temporary_operand_dtypes=None,
        profile=dict(PROFILE, max_new_tokens=2), controls=[], results=[])
    baseline = baseline_report if isinstance(baseline_report, dict) else {}
    if (any(baseline.get(k) != value for k, value in dict(model=MODEL, base=BASE, source=SOURCE,
            source_index=12, profile=PROFILE, numerical_policy='full_accumulation').items())
            or baseline.get('observer_fidelity') != [True, True]
            or not isinstance(baseline.get('results'), list) or len(baseline['results']) != 2
            or not isinstance(baseline.get('mlp_components'), dict)
            or baseline['mlp_components'].get('status') != 'observed'
            or not isinstance(baseline['mlp_components'].get('observed'), dict)):
        report['reason'] = 'missing/changed original source12 full_accumulation baseline identity'
        return report
    frozen = baseline['results']
    if any(not isinstance(row, dict) or len(row.get('generated_suffix_token_ids', [])) != 32
           for row in frozen):
        report['reason'] = 'baseline lacks the original full32 suffix'
        return report
    qwen = resolve_qwen2_5_vl_model(model)
    traces, fidelity = [], []
    with numerical_execution_scope('full_accumulation', qwen), torch.inference_mode():
        for cached in (False, True):
            report['controls'].append(evaluation.run_one_example(model, processor, sample, report['profile'],
                disable_latent_generation=True, use_kv_cache=cached,
                evaluation_protocol='continuous', numerical_policy='full_accumulation'))
        for side, cached in enumerate((False, True)):
            with PassiveTrace(qwen, max_steps=2, mlp_components=True, retain_adapter_operands=True) as trace:
                actual = evaluation.run_one_example(model, processor, sample, report['profile'],
                    disable_latent_generation=True, use_kv_cache=cached,
                    evaluation_protocol='continuous', numerical_policy='full_accumulation')
            report['results'].append(actual)
            traces.append(trace)
            matches = True
            try:
                require_fidelity(actual, report['controls'][side], expected_steps=2)
                require_fidelity(actual, dict(frozen[side],
                    generated_suffix_token_ids=frozen[side]['generated_suffix_token_ids'][:2]), expected_steps=2)
            except ValueError:
                matches = False
            observed = trace.mlp_components
            saved = baseline['mlp_components'].get('observed', {}).get(('off', 'on')[side], [])
            if observed is None or len(observed.events) != 7 or len(saved) != 7:
                matches = False
            else:
                for index in (5, 6):
                    event, original = observed.events[index], saved[index]
                    value = tensor_record(event['_value'])
                    matches &= (event['component'] == original.get('component')
                        and event['role'] == original.get('role') == 'token'
                        and event['operand'] == original.get('operand')
                        and all(value[k] == original.get('value', {}).get(k) for k in
                            ('shape', 'dtype', 'sha256', 'finite_count', 'count'))
                        and value['finite_count'] == value['count'])
            fidelity.append(bool(matches))
        result = _fp64_operand_precheck(*(trace.mlp_components for trace in traces), baseline_fidelity=fidelity)
    report.update(result)
    return report


def math_sdpa_eligible(report: dict) -> bool:
    """Derive the first unequal boundary from ordered actual comparisons."""
    rows = report.get('comparisons', [])
    if not rows or any(not isinstance(row.get('difference'), dict) for row in rows):
        return False
    if any(row['difference'].get('finite') is not True for row in rows):
        return False
    first_index = next((i for i, row in enumerate(rows)
                        if row['difference'].get('equal') is not True), None)
    if first_index is None:
        return False
    first = rows[first_index]
    if not first.get('operation', '').endswith('.sdpa.output'):
        return False
    step = first.get('step')
    prefix = first['operation'][:-len('sdpa.output')]
    if type(step) is not int or step < 0:
        return False
    layer_counts = report.get('decoder_layer_counts')
    if (not isinstance(layer_counts, list) or len(layer_counts) != 2
            or any(type(count) is not int or count < 1 for count in layer_counts)
            or layer_counts[0] != layer_counts[1]):
        return False
    layer_prefixes = [f'layer.{index}.' for index in range(layer_counts[0])]
    if prefix not in layer_prefixes:
        return False
    target_layer = layer_prefixes.index(prefix)
    # Require the native hook sequence from prediction zero, rather than infer
    # coverage from present rows. Both loaded decoders supply their layer count.
    layer_boundaries = ('input', 'cache_position', 'input_norm', 'q_proj', 'k_proj', 'v_proj',
        'sdpa.q', 'sdpa.k', 'sdpa.v', 'sdpa.visibility', 'sdpa.output', 'o_proj',
        'post_attention_norm.input_residual', 'post_attention_norm',
        'mlp_gate', 'mlp_up', 'mlp_down', 'mlp', 'output')
    expected = []
    for prediction in range(step+1):
        expected.extend((prediction, name) for name in
                        ('decoder.embedding', 'decoder.mask2d', 'decoder.mrope'))
        for layer in range(layer_counts[0] if prediction < step else target_layer+1):
            boundaries = layer_boundaries
            if prediction == step and layer == target_layer:
                boundaries = boundaries[:boundaries.index('sdpa.output')+1]
            expected.extend((prediction, layer_prefixes[layer]+name) for name in boundaries)
        if prediction < step:
            expected.extend((prediction, name) for name in
                            ('decoder.final_norm', 'logits.raw', 'logits.post_penalty'))
    observed = [(row.get('step'), row.get('operation')) for row in rows[:first_index+1]]
    if observed != expected:
        return False
    operands = {}
    for operation in ('input', 'input_norm', 'q_proj', 'k_proj', 'v_proj',
                      'sdpa.q', 'sdpa.k', 'sdpa.v', 'sdpa.visibility'):
        selected = [r for r in rows if r.get('step') == step and r.get('operation') == prefix+operation]
        if len(selected) != 1 or selected[0]['difference'].get('equal') is not True:
            return False
        operands[operation] = selected[0]
    off, on = report.get('decoder_off', []), report.get('decoder_on', [])
    if len(off) <= step or len(on) <= step:
        return False
    prompt = off[0].get('query_length')
    if type(prompt) is not int or prompt <= 0:
        return False
    for index in range(step+1):
        for cached, decoder in ((False, off[index]), (True, on[index])):
            wanted = (1 if cached and index else prompt+index,
                      prompt+index-1 if cached and index else 0, cached,
                      prompt+index if cached else None)
            if tuple(decoder.get(k) for k in ('query_length', 'past_length', 'use_cache', 'returned_cache_length')) != wanted:
                return False
    for operation in ('sdpa.k', 'sdpa.v'):
        for side in ('off', 'on'):
            row = operands[operation]
            value = row.get(side, {})
            if (value.get('shape') != row.get('physical_shape_'+side)
                    or len(value.get('shape', [])) != 4 or value['shape'][-2] != prompt+step
                    or value.get('finite_count') != value.get('count')):
                return False
    for side in ('off', 'on'):
        visibility = operands['sdpa.visibility'].get(side, {})
        if visibility.get('shape') != [prompt+step]:
            return False
    return True


class PassiveTrace:
    """Scoped hooks on the unchanged native path, with restoring call wrappers.

    Keep newest rows except complete effective K/V (older positions matter).
    Keep exact replay inputs only for layer 0 rowwise boundaries, bounding memory.
    No cache object is retained and wrappers return the original result object.
    """
    def __init__(self, model, *, max_steps, mlp_components=False, retain_adapter_operands=False):
        self.model = model
        self.max_steps = max_steps
        self.steps = []
        self.current = None
        self.active_layer = None
        self.stack = ExitStack()
        self.mlp_components_requested = mlp_components
        self.retain_adapter_operands = retain_adapter_operands
        self.mlp_components = None

    def record(self, name, value, *, full=False, row_axis=-2, module=None, operand=None):
        if self.current is None or not isinstance(value, torch.Tensor):
            return
        selected = value if full else value.narrow(row_axis, value.shape[row_axis]-1, 1)
        event = dict(shape=list(value.shape), value=selected.detach().clone(),
                     output_dtype=str(value.dtype), output_stride=list(value.stride()))
        if module is not None and operand is not None:
            event['module'] = module
            event['operand_shape'] = list(operand.shape)
            event['operand_dtype'] = str(operand.dtype)
            event['operand_stride'] = list(operand.stride())
            # Exact full inputs only in the earliest layer, for a bounded causal replay.
            if name.startswith('layer.0.'):
                event['operand'] = operand.detach().clone()
        self.current['events'][name] = event

    def decoder_pre(self, _module, args, kwargs):
        self.current = None
        if len(self.steps) >= self.max_steps:
            return
        embeds = kwargs['inputs_embeds']
        cache = kwargs.get('past_key_values')
        self.current = dict(events={}, history=None, decoder=dict(
            query_length=embeds.shape[1], past_length=cache.get_seq_length() if cache is not None else 0,
            input_ids=None if kwargs.get('input_ids') is None else kwargs['input_ids'].tolist(),
            input_provenance='native controller inputs_embeds; actual history observed at repetition call',
            use_cache=kwargs.get('use_cache'), generate_mode=kwargs.get('generate_mode'),
            stage2_capture_kv=kwargs.get('stage2_capture_kv', False)))
        self.steps.append(self.current)
        self.record('decoder.embedding', embeds)
        self.record('decoder.mask2d', kwargs['attention_mask'], full=True)
        self.record('decoder.mrope', kwargs['position_ids'], row_axis=-1)
        if kwargs.get('cache_position') is not None:
            self.record('decoder.cache_position', kwargs['cache_position'], row_axis=-1)
        else:
            # This is derived by native decoder when its argument is absent.
            self.current['decoder']['cache_position_argument'] = None

    def __enter__(self):
        try:
            self.stack.enter_context(torch.inference_mode())
            if self.mlp_components_requested and len(self.model.model.layers) > 24:
                self.mlp_components = self.stack.enter_context(MLPComponents(
                    self.model.model.layers[24].mlp.up_proj,
                    enabled=lambda: self.current is not None and len(self.steps) == 2,
                    retain_adapter_operands=self.retain_adapter_operands))
            h = self.model.model.register_forward_pre_hook(self.decoder_pre, with_kwargs=True)
            self.stack.callback(h.remove)
            def after_decoder(_m, _a, output):
                if self.current is not None:
                    cache = getattr(output, 'past_key_values', None)
                    self.current['decoder']['returned_cache_length'] = cache.get_seq_length() if cache is not None else None
            self.stack.callback(self.model.model.register_forward_hook(after_decoder).remove)
            for i, layer in enumerate(self.model.model.layers):
                prefix = f'layer.{i}'
                def before_layer(_m, args, kwargs, index=i, name=prefix):
                    self.active_layer = index
                    self.record(name+'.input', args[0])
                    self.record(name+'.cache_position', kwargs['cache_position'], row_axis=-1)
                self.stack.callback(layer.register_forward_pre_hook(before_layer, with_kwargs=True).remove)
                def after_layer(_m, _args, output, name=prefix):
                    self.record(name+'.output', output[0])
                    self.active_layer = None
                self.stack.callback(layer.register_forward_hook(after_layer).remove)
                modules = dict(input_norm=layer.input_layernorm,
                    q_proj=layer.self_attn.q_proj, k_proj=layer.self_attn.k_proj,
                    v_proj=layer.self_attn.v_proj, o_proj=layer.self_attn.o_proj,
                    post_attention_norm=layer.post_attention_layernorm,
                    mlp_gate=layer.mlp.gate_proj, mlp_up=layer.mlp.up_proj,
                    mlp_down=layer.mlp.down_proj, mlp=layer.mlp)
                for label, module in modules.items():
                    name = prefix+'.'+label
                    def observe(m, args, output, name=name):
                        # post_attention_norm input is the actual first residual sum.
                        if name.endswith('post_attention_norm'):
                            self.record(name+'.input_residual', args[0])
                        self.record(name, output, module=m, operand=args[0])
                    self.stack.callback(module.register_forward_hook(observe).remove)
            def final_norm(m, args, output):
                self.record('decoder.final_norm', output)
            self.stack.callback(self.model.model.norm.register_forward_hook(final_norm).remove)
            original_sdpa = F.scaled_dot_product_attention
            def sdpa(query, key, value, *args, **kwargs):
                relevant = self.current is not None and self.active_layer is not None
                if relevant:
                    label = f'layer.{self.active_layer}.sdpa'
                    self.record(label+'.q', query)
                    self.record(label+'.k', key, full=True)
                    self.record(label+'.v', value, full=True)
                    mask = kwargs.get('attn_mask', args[0] if args else None)
                    causal = kwargs.get('is_causal', args[2] if len(args) > 2 else False)
                    visibility = last_query_visibility(mask, query.shape[-2], key.shape[-2], causal)
                    self.record(label+'.visibility', visibility, full=True)
                    self.current['events'][label+'.q']['sdpa'] = dict(
                        is_causal=causal, mask_shape=None if mask is None else list(mask.shape),
                        callable='torch.nn.functional.scaled_dot_product_attention', kernel_identity='unobserved')
                result = original_sdpa(query, key, value, *args, **kwargs)
                if relevant:
                    self.record(label+'.output', result)
                return result
            F.scaled_dot_product_attention = sdpa
            self.stack.callback(setattr, F, 'scaled_dot_product_attention', original_sdpa)
            original_penalty = RepetitionPenaltyLogitsProcessor.__call__
            def penalty(processor, input_ids, scores):
                if self.current is not None:
                    self.current['history'] = input_ids[0].tolist()
                    self.record('logits.raw', scores, full=True)
                result = original_penalty(processor, input_ids, scores)
                self.record('logits.post_penalty', result, full=True)
                return result
            RepetitionPenaltyLogitsProcessor.__call__ = penalty
            self.stack.callback(setattr, RepetitionPenaltyLogitsProcessor, '__call__', original_penalty)
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exception):
        self.current = None
        self.active_layer = None
        return self.stack.__exit__(*exception)


def replay_row_operation(module, full_input, *, numerical_policy="historical"):
    """Identical actual operands, two original row geometries, no observer hooks."""
    with numerical_execution_scope(numerical_policy, module, require_adapters=False) as execution, torch.inference_mode():
        full = module(full_input)
        newest = module(full_input[..., -1:, :].contiguous())
        metadata = ({"numerical_observation": observe_numerical_policy(numerical_policy, execution=execution)}
                    if numerical_policy != "historical" else {})
    return dict(**metadata, kind='rowwise geometry replay', synthetic_preceding_rows=False,
                full_shape=list(full_input.shape), newest_shape=list(full_input[..., -1:, :].shape),
                difference=tensor_difference(full[..., -1:, :], newest),
                full_output=tensor_record(full[..., -1:, :]), newest_output=tensor_record(newest))


def compare_traces(off, on, *, numerical_policy="historical"):
    rows = []
    first = None
    replay = None
    compared = 0
    for step, (a, b) in enumerate(zip(off.steps, on.steps)):
        if a['history'] is None or b['history'] is None:
            raise ValueError('missing observed logical history')
        if a['history'] != b['history']:
            break
        compared += 1
        for name, left in a['events'].items():
            right = b['events'].get(name)
            # OFF omits the cache-position argument, but actual layer positions are observed.
            if right is None:
                rows.append(dict(step=step, operation=name, missing_in='on'))
                continue
            difference = tensor_difference(left['value'], right['value'])
            row = dict(step=step, operation=name, physical_shape_off=left['shape'],
                       physical_shape_on=right['shape'], physical_shape_differs=left['shape'] != right['shape'],
                       difference=difference, off=tensor_record(left['value']), on=tensor_record(right['value']))
            for side, event in (('off', left), ('on', right)):
                if 'operand_dtype' in event:
                    row['outer_operands_'+side] = dict(dtype=event['operand_dtype'],
                        shape=event['operand_shape'], stride=event['operand_stride'])
            if name.startswith('logits.'):
                row['top_off'] = top_logits(left['value'])
                row['top_on'] = top_logits(right['value'])
            if 'sdpa' in left:
                row['sdpa_off'] = left['sdpa']
                row['sdpa_on'] = right['sdpa']
            rows.append(row)
            if not difference['equal'] and first is None:
                first = dict(step=step, operation=name, difference=difference,
                             scope='first unequal observed boundary, not every internal arithmetic instruction')
                operand = left.get('operand')
                if operand is not None and right.get('operand') is not None:
                    input_difference = tensor_difference(operand[..., -1:, :], right['operand'][..., -1:, :])
                    first['newest_operand_difference'] = input_difference
                    if input_difference['equal'] and operand.shape[-2] != right['operand'].shape[-2]:
                        replay = replay_row_operation(left['module'], operand, numerical_policy=numerical_policy)
                        replay['observed_difference'] = difference
                        replay['scope'] = 'same actual OFF full input and identical newest operand; rowwise geometry sensitivity'
    report = dict(compared_steps=compared, first_unequal_observed=first, replay=replay,
                replay_limitation=('Replay clones may change original strides/layout; internal kernel identity remains unobserved.'
                                   if replay else 'No trustworthy exact-operand geometry replay at first unequal observed boundary.'),
                comparisons=rows, decoder_off=[s['decoder'] for s in off.steps[:compared]],
                decoder_on=[s['decoder'] for s in on.steps[:compared]])
    if numerical_policy in ADAPTER_TOKEN_FP64_POLICIES:
        report['decoder_layer_counts'] = [
            len(trace.model.model.layers) if trace.model is not None else None for trace in (off, on)]
    return report


def alignment_verdict(off_trace: PassiveTrace, on_trace: PassiveTrace,
                      off_result: dict, on_result: dict, *, expected_steps: int,
                      numerical_policy="historical", component_report=None) -> dict:
    """Accept only exact required tensors on equal logical histories and real KV geometry.

    Missing evidence is inconclusive. Equal token choices alone never establish
    numerical equality. An unequal-history child contributes no tensor comparison.
    """
    names = ('layer.0.q_proj', 'logits.raw', 'logits.post_penalty')
    missing = False
    finite = True
    equal = {name: True for name in names}
    seen = {name: 0 for name in names}
    histories = True
    geometry = True
    compared = 0
    tokens_off = off_result.get('generated_suffix_token_ids')
    tokens_on = on_result.get('generated_suffix_token_ids')
    tokens_equal = None if tokens_off is None or tokens_on is None else tokens_off == tokens_on
    prompt = off_result.get('prompt_token_length')
    if type(prompt) is not int or on_result.get('prompt_token_length') != prompt:
        missing = True
        prompt = None
    if off_result.get('prompt_identity') is None or on_result.get('prompt_identity') is None:
        missing = True
    elif off_result['prompt_identity'] != on_result['prompt_identity']:
        histories = False
    for i, (off, on) in enumerate(zip(off_trace.steps[:expected_steps], on_trace.steps[:expected_steps])):
        a, b = off.get('history'), on.get('history')
        if a is None or b is None:
            missing = True
            break
        if a != b:
            histories = False
            break
        if prompt is not None and (len(a) != prompt+i or
                (tokens_off is not None and a[prompt:] != tokens_off[:i]) or
                (tokens_on is not None and b[prompt:] != tokens_on[:i])):
            histories = False
            break
        compared += 1
        for cached, step in ((False, off), (True, on)):
            decoder = step.get('decoder', {})
            required = ('query_length', 'past_length', 'use_cache', 'returned_cache_length')
            if prompt is None or not all(key in decoder for key in required):
                missing = True
                continue
            wanted = (1 if cached and i else prompt+i,
                      prompt+i-1 if cached and i else 0, cached,
                      prompt+i if cached else None)
            if tuple(decoder[key] for key in required) != wanted:
                geometry = False
        for name in names:
            left = off.get('events', {}).get(name, {}).get('value')
            right = on.get('events', {}).get(name, {}).get('value')
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                missing = True
                continue
            seen[name] += 1
            finite &= bool(torch.isfinite(left).all() and torch.isfinite(right).all())
            equal[name] &= torch.equal(left, right)
    coverage = (not missing and histories and compared == expected_steps and expected_steps > 0
                and tokens_off is not None and tokens_on is not None
                and len(tokens_off) == len(tokens_on) == expected_steps
                and all(count == expected_steps for count in seen.values()))
    q_equal = equal[names[0]] if seen[names[0]] else None
    logits_equal = (all(equal[name] for name in names[1:])
                    if all(seen[name] for name in names[1:]) else None)
    failed = (not finite or tokens_equal is False or q_equal is False
              or logits_equal is False or not geometry)
    if numerical_policy in ADAPTER_TOKEN_FP64_POLICIES:
        boundaries = (component_report or {}).get('observed', {}).get('boundaries', [])
        selected = [row for row in boundaries if row.get('component') in ('lora_A', 'lora_B')
                    and row.get('role') == 'token']
        components_complete = ((component_report or {}).get('status') == 'observed'
                               and len(selected) == 2
                               and {r['component'] for r in selected} == {'lora_A', 'lora_B'})
        coverage &= components_complete and expected_steps == 32
        for row in selected:
            difference = row.get('difference', {})
            failed |= difference.get('equal') is False or difference.get('finite') is False
            coverage &= difference.get('equal') is True and difference.get('finite') is True
    status = 'fail' if failed else 'pass' if coverage else 'inconclusive'
    return dict(status=status, q_equal=q_equal, logits_equal=logits_equal,
        tokens_equal=tokens_equal, coverage_complete=coverage, finite=finite,
        history_equal=histories, cache_geometry_expected=geometry,
        compared_steps=compared, expected_steps=expected_steps,
        completed_lengths=[None if tokens is None else len(tokens) for tokens in (tokens_off, tokens_on)])


def top_logits(value):
    values, ids = value.float().reshape(-1).topk(5)
    return [dict(token_id=int(i), value=float(v) if torch.isfinite(v) else None)
            for i, v in zip(ids, values)]


def read_row(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    selected = [row for row in rows if not row.get('summary')]
    if len(selected) != 1:
        raise ValueError(f'expected exactly one sample: {path}')
    return selected[0]


def selected_sample(source_index=12):
    from multinode_evaluation import resolve_evaluation_image_paths
    if type(source_index) is not int or source_index not in (0, 3, 12):
        raise ValueError('source index must be 0, 3 or 12')
    sample = (read_row(SOURCE) if source_index == 12 else
              json.loads(Path('data/TEST.jsonl').read_text().splitlines()[source_index]))
    sample['image_input'] = resolve_evaluation_image_paths(sample.get('image_input', []), Path('data').resolve())
    return sample


def output_directory(path):
    path = Path(path)
    root = Path('results').resolve()
    if not path.resolve().is_relative_to(root):
        raise ValueError('output directory must be under current repository results/')
    path.mkdir(parents=True, exist_ok=False)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--frozen-off')
    parser.add_argument('--frozen-on')
    parser.add_argument('--numerical-policy', choices=NUMERICAL_POLICIES, default='historical')
    parser.add_argument('--source-index', type=int, choices=(0, 3, 12), default=12)
    parser.add_argument('--fresh-controls', action='store_true')
    parser.add_argument('--mlp-components', action='store_true')
    args = parser.parse_args(argv)
    if args.mlp_components and (not args.fresh_controls or args.source_index != 12
                                or args.numerical_policy not in ('full_accumulation',) + ADAPTER_TOKEN_FP64_POLICIES):
        parser.error('--mlp-components requires --fresh-controls --source-index 12 --numerical-policy full_accumulation')
    if (args.numerical_policy in ADAPTER_TOKEN_FP64_POLICIES and args.source_index == 12
            and not args.mlp_components):
        parser.error('source12 adapter profile requires --mlp-components')
    if not args.fresh_controls and (not args.frozen_off or not args.frozen_on):
        parser.error('provide both frozen controls or --fresh-controls')
    if args.numerical_policy != 'historical' and not args.fresh_controls:
        parser.error('nonhistorical numerical policy requires --fresh-controls')
    frozen = [] if args.fresh_controls else [read_row(args.frozen_off), read_row(args.frozen_on)]
    sample = selected_sample(args.source_index)
    steps = PROFILE['max_new_tokens'] if args.fresh_controls else comparison_steps(
        *(row['generated_suffix_token_ids'] for row in frozen))
    destination = output_directory(args.output_dir)
    # Import the proven loader/runner, preserving their configured dtype and backends.
    import eval as evaluation
    from peft_common import resolve_qwen2_5_vl_model
    traces, results, controls, fidelity = [], [], [], []
    precheck = None
    try:
        model, processor = evaluation.load_evaluation_artifacts(model_dir=MODEL, base_model_dir=BASE,
            cache_dir='data/hf-cache', evaluation_protocol='continuous')
        model.eval()
        qwen = resolve_qwen2_5_vl_model(model)
        validate_numerical_policy(args.numerical_policy, protocol='continuous', stage=qwen.config.stage,
                                  disable_latent_generation=True)
        if args.numerical_policy in ADAPTER_TOKEN_FP64_POLICIES and args.source_index == 12:
            try:
                baseline = json.loads(FP64_BASELINE_REPORT.read_text())
            except (OSError, ValueError):
                baseline = {}
            precheck = fp64_adapter_precheck(model, processor, sample, baseline_report=baseline)
            if precheck['status'] != 'pass':
                report = dict(status=precheck['status'], precheck=precheck, numerical_policy=args.numerical_policy,
                    results=[], fresh_controls=[], observer_fidelity=[], source_index=args.source_index,
                    alignment_verdict=dict(status='inconclusive', executed=False))
                with (destination/'report.json').open('x') as stream:
                    json.dump(report, stream, indent=2, allow_nan=False)
                return
        with numerical_execution_scope(args.numerical_policy, qwen) as execution:
            observation = observe_numerical_policy(args.numerical_policy, qwen, execution=execution)
            if args.fresh_controls:
                for cached in (False, True):
                    controls.append(evaluation.run_one_example(model, processor, sample, PROFILE,
                        disable_latent_generation=True, use_kv_cache=cached,
                        evaluation_protocol='continuous', numerical_policy=args.numerical_policy))
            references = controls if args.fresh_controls else frozen
            for cached, reference in zip((False, True), references):
                with PassiveTrace(qwen, max_steps=steps, mlp_components=args.mlp_components) as trace:
                    result = evaluation.run_one_example(model, processor, sample, PROFILE,
                        disable_latent_generation=True, use_kv_cache=cached,
                        evaluation_protocol='continuous', numerical_policy=args.numerical_policy)
                results.append(result)
                require_fidelity(result, reference,
                    expected_steps=32 if args.mlp_components or not args.fresh_controls else None)
                fidelity.append(True)
                traces.append(trace)
            report = compare_traces(*traces, numerical_policy=args.numerical_policy)
            if args.mlp_components:
                report['mlp_components'] = compare_mlp_components(
                    *(trace.mlp_components for trace in traces), numerical_policy=args.numerical_policy)
            expected = 32 if args.source_index == 12 else max(
                len(row['generated_suffix_token_ids']) for row in results)
            verdict = alignment_verdict(*traces, *results, expected_steps=expected,
                numerical_policy=args.numerical_policy if args.source_index == 12 else 'historical',
                component_report=report.get('mlp_components'))
            observation = observe_numerical_policy(args.numerical_policy, qwen, execution=execution)
        if precheck is not None:
            report['precheck'] = precheck
        if args.numerical_policy in ADAPTER_TOKEN_FP64_POLICIES:
            report['math_sdpa_eligible'] = math_sdpa_eligible(report)
        report.update(status=verdict['status'] if args.fresh_controls else 'faithful',
            alignment_verdict=verdict, observer_fidelity=fidelity, fresh_controls=controls,
            numerical_policy=args.numerical_policy, numerical_observation=observation,
            results=results, profile=PROFILE, max_new_tokens=32, source_index=args.source_index,
            source=SOURCE if args.source_index == 12 else 'data/TEST.jsonl', model=MODEL, base=BASE,
            interpretation='Actual outer boundary tensors and SDPA operands observed. Kernel identity and internal GEMM operands unobserved. '
                           'Parameter dtypes describe loaded tensors; operand/output dtypes describe outer module calls. '
                           'Residual sums observed at post-attention norm input and layer output; internal arithmetic not traced. '
                           'Penalty logits precede native latent-control token suppression. '
                           'Histories observed at native repetition call; decoder input_ids are normally absent. '
                           'Replay clones may not preserve original strides/layout; replay cannot establish kernel identity.')
    except Exception as error:
        report = dict(status='inconclusive', error=f'{type(error).__name__}: {error}', results=results,
                      fresh_controls=controls, observer_fidelity=fidelity, numerical_policy=args.numerical_policy)
        with (destination/'report.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        raise
    with (destination/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
