"""Publish the read phase before Trainer constructs its next epoch iterator."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch.distributed as dist
from transformers import TrainerCallback

from .cache_transport import local_manifest, validate_manifest


def publish_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.publish-', mode='w', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, sort_keys=True)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_release(control_dir, manifest):
    publish_json(Path(control_dir) / 'release.json', manifest)


class Stage1CacheEpochCallback(TrainerCallback):
    def __init__(self, cache, *, expected, control_dir):
        self.cache = cache
        self.expected = expected
        self.control_dir = Path(control_dir)

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.cache.phase.value == 0 and (state.epoch or 0) >= 1:
            raise RuntimeError('later-epoch cache requires completed resume preparation')

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.cache.phase.value == 1 or not math.isclose(state.epoch or 0, 1., abs_tol=1e-8):
            return
        # All publications finish before leaders hash. Never hold a GPU
        # collective open across the potentially large disk/network exchange.
        if dist.is_initialized():
            dist.barrier()
        if args.local_process_index == 0:
            manifest = local_manifest(self.cache, self.expected)
            publish_json(self.control_dir / 'ready.json', {
                'manifest': manifest, 'expected': self.expected,
                'run_id': self.cache.plan.run_id,
            })
        release = self.control_dir / 'release.json'
        while not release.exists():
            # Existing coordinator owns timeout and peer failure termination.
            time.sleep(.1)
        manifest = json.loads(release.read_text())
        validate_manifest(self.cache, manifest)
        required = {f'{layer}/{index}.pt' for layer, indices in self.expected.items() for index in indices}
        if set(manifest['records']) != required:
            raise ValueError('cache release lacks complete coverage')
        self.cache.phase.value = 1
