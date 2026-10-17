#!/usr/bin/env python
"""Bounded operator measurement; no model loading, training, or state writes.

Run with the repository venv and PYTHONPATH=transformers/src:src:.
The independent before implementations are byte-identical sources from
bb942b7323495cda716b7c5b889d6a8dc699aa23, shipped with the tests.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import tracemalloc

import torch

from stage1.v5_config import Stage1V5Config
from stage1.v5_selection import select_v5_text_keys
from transformers.models.qwen2_5_vl.stage1_v5_trace import capture_v5_trace
from tests.stage1.v5.throughput_fixtures import (
    OperatorCounts, observation_fixture, original_module, selector_fixture,
)


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


def measure(function, device, warmup, repetitions):
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    for _ in range(warmup):
        function()
    synchronize()
    times = []
    for _ in range(repetitions):
        synchronize()
        start = time.perf_counter_ns()
        function()
        synchronize()
        times.append((time.perf_counter_ns() - start) / 1e6)
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        function()
        synchronize()
        memory = dict(baseline_allocated_bytes=baseline,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            extra_peak_allocated_bytes=torch.cuda.max_memory_allocated(device)-baseline,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    else:
        tracemalloc.start()
        function()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory = dict(peak_python_bytes=peak, note='Python allocations only; excludes CPU tensor storage')
    return dict(milliseconds=times, median_ms=statistics.median(times), memory=memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
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
    del before_rows, after_rows
    metadata, attention, clean, reference = selector_fixture(count=8, candidates=128, length=4096)
    attention = {key:(value*.25).to(device) for key,value in attention.items()}
    clean, reference = clean.to(device), reference.to(device)
    compact = {(b.batch_index,q):attention[(b.batch_index,q)][list(b.candidate_positions)]
               for b in metadata.blocks for q in b.query_positions}
    config = Stage1V5Config()
    original_select = original_module('v5_selection').select_v5_text_keys
    def select_before():
        return original_select(metadata,attention,clean,reference,config,validate=False)
    def select_after():
        return select_v5_text_keys(metadata,compact,clean,reference,config,
                                   validate=False,compact_attention=True)
    selection_equal = select_before() == select_after()
    report = dict(device=str(device), torch=torch.__version__, seed_attention=211, seed_selection=311,
        threads=1, warmup=args.warmup, repetitions=args.repetitions,
        attention_fixture=dict(batch=2,length=512,queries=64,heads=8,head_dim=32,layers=2,
                               dtype='bfloat16 projections, float32 observation'),
        selection_fixture=dict(queries=32,candidates_per_query=128,full_row_length=4096,vocab=17,
                               dtype='float32',validation=False),
        attention_exact_equal=attention_equal,selected_keys_exact_equal=selection_equal,
        limitations='Local operator fixtures only; no production BF16/four-rank equivalence or utilization fix established.',
        oracle_sha256={name:hashlib.sha256(Path(original_module(name).__file__).read_bytes()).hexdigest()
                       for name in ('stage1_v5_trace','v5_selection')})
    if device.type == 'cuda':
        report['gpu'] = torch.cuda.get_device_name(device)
    for name, function in [('attention_before',lambda:observe(original_capture)),
                           ('attention_after',lambda:observe(capture_v5_trace)),
                           ('selection_before',select_before),('selection_after',select_after)]:
        result = measure(function,device,args.warmup,args.repetitions)
        with OperatorCounts((8,512,32)) as counts, count_host_lists() as lists:
            function()
        result['operations'] = dict(key_float_copies=counts.key_float_copies,
            arange_calls=counts.arange_calls,scalar_reads=counts.scalar_reads,
            candidate_index_tensors=counts.candidate_index_tensors,
            tolist_calls=len(lists),tolist_elements=sum(lists))
        report[name] = result
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert attention_equal and selection_equal, 'local oracle mismatch'


if __name__ == '__main__':
    main()
