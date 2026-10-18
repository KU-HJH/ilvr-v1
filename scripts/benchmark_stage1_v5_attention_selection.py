#!/usr/bin/env python
"""Bounded four-operator measurement; no model loading, training, or state writes.

Run with the repository venv and PYTHONPATH=transformers/src:src:.
Independent before implementations are byte-identical source65643f0 copies.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
import tracemalloc

import torch

from stage1.v5_config import Stage1V5Config
from stage1.v5_execution import V5Execution
from stage1.v5_selection import select_v5_text_keys
from transformers.models.qwen2_5_vl.stage1_v5_replay import run_v5_packed_replay
from transformers.models.qwen2_5_vl.stage1_v5_trace import capture_v5_trace
from tests.stage1.v5.gpu_utilization_fixtures import (
    BASELINE_COMMIT, SOURCE_PATHS, OperatorCounts, dropout_fixture, original_module,
    replay_fixture, run_dropout, run_replay_mask,
)
from tests.stage1.v5.throughput_fixtures import observation_fixture, selector_fixture


ATTENTION_ATOL, ATTENTION_RTOL = 1e-7, 1e-5


@contextmanager
def count_host_lists():
    original, sizes = torch.Tensor.tolist, []
    def observed(value):
        sizes.append(value.numel())
        return original(value)
    torch.Tensor.tolist = observed
    try:
        yield sizes
    finally:
        torch.Tensor.tolist = original


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure_memory(function, device):
    gc.collect()
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        function()
        synchronize(device)
        return dict(baseline_allocated_bytes=baseline,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            extra_peak_allocated_bytes=torch.cuda.max_memory_allocated(device)-baseline,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    tracemalloc.start()
    try:
        function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return dict(peak_python_bytes=peak, note='Python allocations only; excludes CPU tensor storage')


def measure_pair(before, after, device, warmup, repetitions):
    functions = dict(before=before, after=after)
    for _ in range(warmup):
        before()
        after()
    synchronize(device)
    times = dict(before=[], after=[])
    orders = []
    for repetition in range(repetitions):
        order = ['before', 'after'] if repetition % 2 == 0 else ['after', 'before']
        orders.append(order)
        for name in order:
            synchronize(device)
            start = time.perf_counter_ns()
            functions[name]()
            synchronize(device)
            times[name].append((time.perf_counter_ns() - start) / 1e6)
    results = [dict(milliseconds=times[name], median_ms=statistics.median(times[name]),
                    memory=measure_memory(functions[name], device)) for name in ('before', 'after')]
    return *results, orders


def source_identity():
    root = Path(__file__).resolve().parents[1]
    candidate = {}
    paths = {}
    for name in SOURCE_PATHS:
        module = importlib.import_module('stage1.' + name if name.startswith('v5_') else
                                        'transformers.models.qwen2_5_vl.' + name)
        path = Path(module.__file__).resolve()
        paths[name] = str(path)
        candidate[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    return dict(baseline_commit=BASELINE_COMMIT, candidate_commit=revision,
                candidate_sha256=candidate, candidate_paths=paths,
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                fixtures_sha256=hashlib.sha256(Path(__file__).resolve().parents[1].joinpath(
                    'tests/stage1/v5/gpu_utilization_fixtures.py').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions < 1:
        parser.error('warmup must be nonnegative and repetitions must be positive')
    device = torch.device(args.device)
    torch.set_num_threads(1)
    decoder, inputs, keys, candidates = observation_fixture(
        length=512, queries=32, heads=8, dim=32, layers=2)
    decoder.to(device)
    inputs = {key:value.to(device) for key,value in inputs.items()}
    original_capture = original_module('stage1_v5_trace').capture_v5_trace

    @torch.no_grad()
    def observe(capture):
        with capture(decoder, keys, candidates) as trace:
            decoder(**inputs)
        return trace.attention_rows

    before_rows, after_rows = observe(original_capture), observe(capture_v5_trace)
    attention_equal = all(torch.equal(before_rows[key], after_rows[key]) for key in keys)
    attention_close = all(torch.allclose(before_rows[key], after_rows[key], atol=ATTENTION_ATOL,
                                        rtol=ATTENTION_RTOL) for key in keys)
    attention_error = max(float((before_rows[key] - after_rows[key]).abs().max()) for key in keys)
    del before_rows, after_rows
    metadata_, attention, clean, reference = selector_fixture(count=8, candidates=128, length=4096)
    attention = {key:(value*.25).to(device) for key,value in attention.items()}
    clean, reference = clean.to(device), reference.to(device)
    compact = {(b.batch_index,q):attention[(b.batch_index,q)][list(b.candidate_positions)]
               for b in metadata_.blocks for q in b.query_positions}
    config = Stage1V5Config()
    original_select = original_module('v5_selection').select_v5_text_keys
    # Both selectors receive the same prebuilt compact candidate values.
    def select_before():
        return original_select(metadata_,compact,clean,reference,config,
                               validate=False,compact_attention=True)
    def select_after():
        return select_v5_text_keys(metadata_,compact,clean,reference,config,
                                  validate=False,compact_attention=True)
    # A separate full-row oracle prevents an accidentally wrong compact gather
    # from making both timed paths agree on the same incorrect candidates.
    selection_equal = original_select(metadata_,attention,clean,reference,config,
                                     validate=False) == select_before() == select_after()
    mask = replay_fixture(device)
    original_replay = original_module('stage1_v5_replay').run_v5_packed_replay
    def mask_before():
        return run_replay_mask(original_replay, mask)
    def mask_after():
        return run_replay_mask(run_v5_packed_replay, mask)
    before_mask, after_mask = mask_before(), mask_after()
    mask_equal = torch.equal(before_mask, after_mask)
    mask_geometry = list(before_mask.shape)
    del before_mask, after_mask
    dropout = dropout_fixture(device)
    original_execution = original_module('v5_execution').V5Execution
    def dropout_before():
        return run_dropout(original_execution, dropout)
    def dropout_after():
        return run_dropout(V5Execution, dropout)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    before_dropout = run_dropout(original_execution, dropout, capture=True)
    after_dropout = run_dropout(V5Execution, dropout, capture=True)
    dropout_equal = all(torch.equal(before_dropout[name], after_dropout[name]) for name in
        ('native_output', 'packed_output', 'native_gradient', 'packed_gradient')) and all(
        torch.equal(a,b) for a,b in zip(before_dropout['raw_masks'], after_dropout['raw_masks']))
    dropout_equal = dropout_equal and len(before_dropout['raw_masks']) == len(after_dropout['raw_masks']) == 4
    rng_unchanged = torch.equal(cpu_rng, torch.random.get_rng_state()) and (
        cuda_rng is None or torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))
    del before_dropout, after_dropout
    packages = {}
    for name in ('torch', 'transformers', 'peft', 'flash-attn', 'deepspeed'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = dict(device=str(device), torch=torch.__version__, python=platform.python_version(),
        packages=packages, cuda_runtime=torch.version.cuda, transformers_runtime=importlib.import_module('transformers').__version__,
        seed_attention=211, seed_selection=311, seed_replay=419, seed_dropout_fixture=521,
        dropout_base_seed=42, dropout_optimizer_step=17,
        threads=1, warmup=args.warmup, repetitions=args.repetitions,
        attention_fixture=dict(batch=2,length=512,queries=64,heads=8,head_dim=32,layers=2,
                               dtype='bfloat16 projections, float32 observation',
                               scope='projection/hook fixture with fixed decoder callbacks; no model load'),
        selection_fixture=dict(queries=32,candidates_per_query=128,full_row_length=4096,vocab=17,
                               dtype='float32',validation=False,input='identical prebuilt compact candidate rows'),
        replay_mask_fixture=dict(batch=2,length=512,blocks=4,rows=8,mask_shape=mask_geometry,
            suffix_lengths=[24,19,13,7],dtype='float32',validation=False,
            scope='actual packed replay bias, prefix concatenation, binding, dummy decoder callback and identity head; no attention/model forward'),
        dropout_fixture=dict(source_lengths=[512,499],hidden_size=64,native_shape=list(dropout.native.shape),
            packed_shape=list(dropout.packed.shape),probability=dropout.p,sites=1,
            checkpoint='non-reentrant early-stop',dtype='float32',validation=False,
            scope='actual private-generator F/packed dropout, checkpoint recomputation and backward; includes owner/checkpoint/sin/reduction/output-detach overhead; no train step'),
        attention_exact_equal=attention_equal,attention_close=attention_close,
        attention_atol=ATTENTION_ATOL,attention_rtol=ATTENTION_RTOL,attention_max_absolute_error=attention_error,
        selected_keys_exact_equal=selection_equal,mask_exact_equal=mask_equal,
        dropout_exact_equal=dropout_equal,dropout_global_rng_unchanged=rng_unchanged,
        source_identity=source_identity(),
        limitations='Local bounded operator fixtures only; CPU memory is Python-only. No production BF16/four-rank equivalence, stage bottleneck attribution or utilization fix established.',
        oracle_sha256={name:hashlib.sha256(Path(original_module(name).__file__).read_bytes()).hexdigest()
                       for name in SOURCE_PATHS})
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        report['gpu'] = props.name
        report['gpu_identity'] = dict(name=props.name,total_memory_bytes=props.total_memory,
                                     capability=list(torch.cuda.get_device_capability(device)))
    for name, before, after in [('attention',lambda:observe(original_capture),lambda:observe(capture_v5_trace)),
                                ('selection',select_before,select_after),
                                ('replay_mask',mask_before,mask_after),('dropout',dropout_before,dropout_after)]:
        before_result, after_result, order = measure_pair(before,after,device,args.warmup,args.repetitions)
        report[name + '_paired_order'] = order
        for suffix, function, result in [('before',before,before_result),('after',after,after_result)]:
            with OperatorCounts((8,512,32)) as counts, count_host_lists() as lists:
                function()
            result['operations'] = {key:getattr(counts,key) for key in (
                'key_float_copies','arange_calls','scalar_reads','candidate_index_tensors',
                'random_draw_calls','random_draw_elements','select_calls','stack_calls','index_put_calls')}
            result['operations'].update(tolist_calls=len(lists),tolist_elements=sum(lists))
            report[name + '_' + suffix] = result
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert attention_close and selection_equal and mask_equal and dropout_equal and rng_unchanged, 'local oracle mismatch'


if __name__ == '__main__':
    main()
