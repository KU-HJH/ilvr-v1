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


MODEL = 'checkpoints/evaluation-staging/stage1-v5-mn-g16-e3-s648-20261014-150317/prepared-model'
BASE = '/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct'
SOURCE = 'data/TEST-stage1-v5-static-cache-source12-20261015-r1.jsonl'
PROFILE = dict(max_new_tokens=32, max_sequence_length=32768, repetition_penalty=1.05,
               temperature=0., top_p=1., generation_limit_policy='bounded')


def require_fidelity(actual, frozen):
    for key in ('generated_suffix_token_ids', 'prompt_identity'):
        if actual.get(key) != frozen.get(key):
            raise ValueError(f'observer fidelity mismatch: {key}; interpretation is inconclusive')
    if len(actual['generated_suffix_token_ids']) != 32:
        raise ValueError('observer fidelity requires the full 32-token suffix')


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
    return dict(value_shape_mismatch=False, equal=count == 0, unequal_count=count,
                max_abs=float(delta.max()) if delta.numel() else 0.)


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


class PassiveTrace:
    """Scoped hooks on the unchanged native path, with restoring call wrappers.

    Keep newest rows except complete effective K/V (older positions matter).
    Keep exact replay inputs only for layer 0 rowwise boundaries, bounding memory.
    No cache object is retained and wrappers return the original result object.
    """
    def __init__(self, model, *, max_steps):
        self.model = model
        self.max_steps = max_steps
        self.steps = []
        self.current = None
        self.active_layer = None
        self.stack = ExitStack()

    def record(self, name, value, *, full=False, row_axis=-2, module=None, operand=None):
        if self.current is None or not isinstance(value, torch.Tensor):
            return
        selected = value if full else value.narrow(row_axis, value.shape[row_axis]-1, 1)
        event = dict(shape=list(value.shape), value=selected.detach().clone())
        if module is not None and operand is not None:
            event['module'] = module
            event['operand_shape'] = list(operand.shape)
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


def replay_row_operation(module, full_input):
    """Identical actual operands, two original row geometries, no observer hooks."""
    with torch.inference_mode():
        full = module(full_input)
        newest = module(full_input[..., -1:, :].contiguous())
    return dict(kind='rowwise geometry replay', synthetic_preceding_rows=False,
                full_shape=list(full_input.shape), newest_shape=list(full_input[..., -1:, :].shape),
                difference=tensor_difference(full[..., -1:, :], newest),
                full_output=tensor_record(full[..., -1:, :]), newest_output=tensor_record(newest))


def compare_traces(off, on):
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
                        replay = replay_row_operation(left['module'], operand)
                        replay['observed_difference'] = difference
                        replay['scope'] = 'same actual OFF full input and identical newest operand; rowwise geometry sensitivity'
    return dict(compared_steps=compared, first_unequal_observed=first, replay=replay,
                replay_limitation=None if replay else 'No trustworthy exact-operand geometry replay at first unequal observed boundary.',
                comparisons=rows, decoder_off=[s['decoder'] for s in off.steps[:compared]],
                decoder_on=[s['decoder'] for s in on.steps[:compared]])


def top_logits(value):
    values, ids = value.float().reshape(-1).topk(5)
    return [dict(token_id=int(i), value=float(v)) for i, v in zip(ids, values)]


def read_row(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    selected = [row for row in rows if not row.get('summary')]
    if len(selected) != 1:
        raise ValueError(f'expected exactly one sample: {path}')
    return selected[0]


def selected_sample():
    from multinode_evaluation import resolve_evaluation_image_paths
    sample = read_row(SOURCE)
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
    parser.add_argument('--frozen-off', required=True)
    parser.add_argument('--frozen-on', required=True)
    args = parser.parse_args(argv)
    frozen = [read_row(args.frozen_off), read_row(args.frozen_on)]
    sample = selected_sample()
    steps = comparison_steps(*(row['generated_suffix_token_ids'] for row in frozen))
    destination = output_directory(args.output_dir)
    # Import the proven loader/runner, preserving their configured dtype and backends.
    import eval as evaluation
    from peft_common import resolve_qwen2_5_vl_model
    traces, results = [], []
    try:
        model, processor = evaluation.load_evaluation_artifacts(model_dir=MODEL, base_model_dir=BASE,
            cache_dir='data/hf-cache', evaluation_protocol='continuous')
        model.eval()
        qwen = resolve_qwen2_5_vl_model(model)
        for cached, reference in zip((False, True), frozen):
            with PassiveTrace(qwen, max_steps=steps) as trace:
                result = evaluation.run_one_example(model, processor, sample, PROFILE,
                    disable_latent_generation=True, use_kv_cache=cached, evaluation_protocol='continuous')
            results.append(result)
            require_fidelity(result, reference)
            traces.append(trace)
        report = compare_traces(*traces)
        report.update(status='faithful', results=results, profile=PROFILE,
            source=SOURCE, model=MODEL, base=BASE,
            interpretation='Actual boundary tensors and SDPA operands observed. Kernel identity unobserved. '
                           'Residual sums observed at post-attention norm input and layer output; internal arithmetic not traced. '
                           'Penalty logits precede native latent-control token suppression. '
                           'Histories observed at native repetition call; decoder input_ids are normally absent.')
    except Exception as error:
        report = dict(status='inconclusive', error=f'{type(error).__name__}: {error}', results=results)
        with (destination/'report.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        raise
    with (destination/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
