"""Instrumentation for the isolated ours1-zero-speed-* comparison, not a training default."""
import json
import os
import time

import torch
from transformers import TrainerCallback


class InsertionSpeedProbe(TrainerCallback):
    def __init__(self, owner):
        self.owner = owner

    def on_step_begin(self, args, state, control, **kwargs):
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
        print('INSERTION_SPEED_PROBE ' + json.dumps(dict(
            rank=int(os.environ['RANK']), step=state.global_step, seconds=elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gradient_norm=self.owner._representative_gradient_norm,
            metrics=metrics), allow_nan=False), flush=True)
        return control
