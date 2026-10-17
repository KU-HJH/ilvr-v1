"""Offline full Qwen witness; no weights are loaded and nothing is downloaded.

Run with the repository Python and PYTHONPATH=transformers/src:src:.
Example: python -m tests.stage1.v5.benchmark_collation --processor-path
checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct --output results/.../collation.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc
from unittest.mock import patch

from PIL import Image
import torch
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds

from tests.stage1.v5.collation_fixtures import (
    CountingProcessor, assert_batches_equal, examples_at, make_collator, with_occurrences,
)


def processor_at(path):
    # Match the production loader's default selection: fast tokenizer (offsets)
    # and the locally saved image processor; do not override both via use_fast.
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    processor.tokenizer.add_special_tokens({'additional_special_tokens': [
        '<|latent_pad|>', '<|latent_start|>', '<|latent_end|>']})
    ids = LatentTokenIds(*(processor.tokenizer.convert_tokens_to_ids(token) for token in (
        '<|latent_pad|>', '<|latent_start|>', '<|latent_end|>')))
    return processor, ids


def digest_batch(batch):
    return {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                      sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
            if isinstance(value, torch.Tensor) else repr(value)
            for key, value in batch.items()}


def worker(args):
    processor, token_ids = processor_at(args.processor_path)
    processor = CountingProcessor(processor)
    examples = examples_at(Path(args.fixture_root), scale=args.scale)
    collator = make_collator(processor, original=args.worker == 'original', token_ids=token_ids)
    with patch.object(Image, 'open', wraps=Image.open) as opened:
        batch = collator(examples)
    counts = dict(processor_calls=processor.calls, image_opens=opened.call_count,
                  processor_image_counts=list(processor.image_counts))
    tensors = digest_batch(batch)
    del batch
    for _ in range(args.warmups):
        collator(examples)
    times = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        result = collator(examples)
        times.append(time.perf_counter() - start)
        del result
    tracemalloc.start()
    result = collator(examples)
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(mode=args.worker, counts=counts, outputs=tensors,
                median_seconds=statistics.median(times), seconds=times,
                tracemalloc_peak_bytes=python_peak,
                process_peak_rss_kib=int(next(line.split()[1] for line in
                    Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processor-path', required=True)
    parser.add_argument('--fixture-root', default='data/stage1-v5-throughput-20261015/collation-witness')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=15)
    parser.add_argument('--output')
    parser.add_argument('--worker', choices=['original', 'optimized'])
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
        return
    # Complete direct equality, separately from timing/memory workers. Include
    # singleton no-helper shape, batching, and all occurrence/source metadata.
    processor, token_ids = processor_at(args.processor_path)
    examples = examples_at(Path(args.fixture_root), scale=args.scale)
    comparisons = 0
    for rows in (examples, with_occurrences(examples)):
        for subset in ((0,), (1,), (2,), (0, 1, 2)):
            batch = [rows[i] for i in subset]
            expected = make_collator(processor, original=True, token_ids=token_ids)(batch)
            actual = make_collator(processor, token_ids=token_ids)(batch)
            assert_batches_equal(actual, expected)
            comparisons += 1
    modes = {}
    for mode in ('original', 'optimized'):
        command = [sys.executable, '-m', __spec__.name,
                   '--processor-path', args.processor_path, '--fixture-root', args.fixture_root,
                   '--scale', str(args.scale), '--warmups', str(args.warmups),
                   '--repetitions', str(args.repetitions), '--worker', mode]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        sys.stderr.write(completed.stderr)
        modes[mode] = json.loads(completed.stdout)
    assert modes['original']['outputs'] == modes['optimized']['outputs']
    report = dict(processor_path=args.processor_path, fixture_root=args.fixture_root,
                  processor_type=type(processor).__name__,
                  tokenizer_type=type(processor.tokenizer).__name__,
                  image_processor_type=type(processor.image_processor).__name__,
                  scale=args.scale, examples=3, originals=4, helpers=3,
                  warmups=args.warmups, repetitions=args.repetitions,
                  exact_equal=True, complete_comparisons=comparisons, modes=modes,
                  speedup=modes['original']['median_seconds'] / modes['optimized']['median_seconds'],
                  memory_note='Each mode in a fresh process; RSS includes imports/processor. '
                  'tracemalloc peak is one complete collator call and excludes native torch allocations.')
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
