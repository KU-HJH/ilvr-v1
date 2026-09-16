"""Instrumentation for the isolated ours1-zero-speed-* comparison, not a training default."""
import json
import os
import time
from pathlib import Path

import torch
from transformers import TrainerCallback
from .paired_probe import create_production_observer


class InsertionSpeedProbe(TrainerCallback):
    def __init__(self, owner):
        self.owner = owner
        self.decoder_calls = None
        self.handles = []
        self.pair_observer = None

    def on_train_begin(self, args, state, control, **kwargs):
        if args is not None and Path(args.output_dir).name.startswith('ours1-zero-speed-pair-'):
            self.pair_observer = create_production_observer(self.owner, args)
            self.pair_observer.install()
        decoder = next(module for module in self.owner.model.modules()
                       if type(module).__name__ == 'Qwen2_5_VLModel')
        self.decoder_calls = {kind: [0] * len(decoder.layers) for kind in ('on', 'clean_or_off')}
        for index, layer in enumerate(decoder.layers):
            def observe(module, inputs, keywords, index=index):
                prefix = inputs[-1] if len(inputs) > 10 else keywords.get('stage2_prefix_key_value')
                kind = 'on' if prefix is not None else 'clean_or_off'
                self.decoder_calls[kind][index] += 1
            self.handles.append(layer.register_forward_pre_hook(observe, with_kwargs=True))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.pair_observer is not None:
            self.pair_observer.close()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        if self.decoder_calls is not None:
            for values in self.decoder_calls.values():
                values[:] = [0] * len(values)
        self.previous = dict(self.owner._step_preference_metric_totals)
        self.previous_batches = self.owner._step_preference_metric_batches
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        count = self.owner._step_preference_metric_batches - self.previous_batches
        metrics = {key: (value - self.previous.get(key, 0.0)) / count
                   for key, value in self.owner._step_preference_metric_totals.items()} if count else {}
        if self.pair_observer is not None and state.global_step == 1:
            self.pair_observer.finish_first_step(metrics)
        row = dict(
            rank=int(os.environ['RANK']), step=state.global_step, seconds=elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gradient_norm=self.owner._representative_gradient_norm,
            metrics=metrics)
        if self.decoder_calls is not None:
            row['decoder_calls'] = self.decoder_calls
        if self.pair_observer is not None:
            row['timing_scope'] = 'instrumented-not-valid-for-speed-comparison'
        print('INSERTION_SPEED_PROBE ' + json.dumps(row, allow_nan=False), flush=True)
        return control
