"""Bounded four-operator fixtures; independent sources frozen at source65643f0.

No weights are loaded or saved. The older frozen_throughput oracles stay intact.
"""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from stage1.equivalence import Stage1StochasticContext
from tests.stage1.v5.throughput_fixtures import OperatorCounts as ThroughputCounts
from transformers.models.qwen2_5_vl.stage1_v5_replay import V5ReplayBlock
from transformers.models.qwen2_5_vl.stage1_v5_trace import V5Trace


BASELINE_COMMIT = '65643f08cb6e61fbbb200de825d57086f5c7a127'
SOURCE_PATHS = {
    'v5_execution': 'src/stage1/v5_execution.py',
    'v5_selection': 'src/stage1/v5_selection.py',
    'stage1_v5_trace': 'transformers/src/transformers/models/qwen2_5_vl/stage1_v5_trace.py',
    'stage1_v5_replay': 'transformers/src/transformers/models/qwen2_5_vl/stage1_v5_replay.py',
}


def original_module(name):
    """Use unique names in original packages, retaining byte-identical imports."""
    package = 'stage1' if name.startswith('v5_') else 'transformers.models.qwen2_5_vl'
    module_name = package + '._frozen_gpu_utilization_' + name
    if module_name not in sys.modules:
        path = Path(__file__).with_name('frozen_gpu_utilization') / (name + '.py')
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


class OperatorCounts(ThroughputCounts):
    """Count real dispatches, including mask assembly and private RNG draws."""
    def __init__(self, key_shape=None):
        super().__init__(key_shape)
        self.random_draw_calls = self.random_draw_elements = 0
        self.select_calls = self.stack_calls = self.index_put_calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.rand.generator:
            self.random_draw_calls += 1
            self.random_draw_elements += math.prod(args[0])
        if func == torch.ops.aten.select.int:
            self.select_calls += 1
        if func == torch.ops.aten.stack.default:
            self.stack_calls += 1
        if func == torch.ops.aten.index_put_.default:
            self.index_put_calls += 1
        return super().__torch_dispatch__(func, types, args, kwargs)


class ReplayMaskDecoder(nn.Module):
    """Capture the actual replay bias; no model/attention computation."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.bias = None

    def forward(self, *, inputs_embeds, attention_mask, **kwargs):
        self.bias = attention_mask
        return SimpleNamespace(last_hidden_state=inputs_embeds)


def replay_fixture(device, length=512):
    generator = torch.Generator().manual_seed(419)
    x = torch.randn(2, length, 32, generator=generator).to(device)
    padding = torch.ones(2, length, device=device, dtype=torch.long)
    padding[0, 1:4], padding[1, 20:23] = 0, 0
    positions = torch.arange(length, device=device)[None, None].expand(3, 2, -1)
    kv = tuple(tuple(torch.randn(2, 2, length, 8, generator=generator).to(device)
                     for _ in range(2)) for _ in range(2))
    trace = V5Trace(inputs_embeds=x, position_ids=positions, padding=padding, key_values=kv)
    blocks = []
    for sample, end, suffix in ((1, 128, 24), (0, 192, 19), (1, 256, 13), (0, 320, 7)):
        targets = tuple(range(end + 1, end + suffix + 1))
        blocked = tuple((q % 16, 32 + q % 16) for q in range(suffix))
        blocks.append(V5ReplayBlock(sample, end - 9, end, targets, (0,) * suffix, blocked))
    def unused_dropout(*args, **kwargs):
        raise AssertionError('dummy replay decoder must not execute dropout')
    return SimpleNamespace(decoder=ReplayMaskDecoder().to(device), head=nn.Identity(), trace=trace,
                           blocks=tuple(blocks), execution=SimpleNamespace(debug=False, dropout_mask=unused_dropout))


@torch.no_grad()
def run_replay_mask(replay, fixture):
    replay(fixture.decoder, fixture.head, fixture.trace, fixture.blocks, execution=fixture.execution,
           need_voff=True, need_textoff=True)
    bias, fixture.decoder.bias = fixture.decoder.bias, None
    return bias


def dropout_fixture(device):
    generator = torch.Generator().manual_seed(521)
    spec = Stage1StochasticContext(42, 17, ((7, 512), (19, 499)), ('operator.dropout',))
    samples = torch.arange(2, device=device).repeat_interleave(96)
    positions = torch.arange(96, device=device).flip(0).repeat(2)
    samples, positions = torch.cat((samples, samples[:8])), torch.cat((positions, positions[:8]))
    return SimpleNamespace(spec=spec, p=.05, site='operator.dropout', packed_samples=samples,
                           packed_positions=positions,
                           native=torch.randn(2, 512, 64, generator=generator).to(device),
                           packed=torch.randn(samples.numel(), 64, generator=generator).to(device))


def run_dropout(execution_type, fixture, *, capture=False):
    """Real F/packed checkpoint owners and backward, without a training step.

    Includes fixed owner/checkpoint/sin/reduction/output-detach overhead. Raw-mask
    retention is enabled only for correctness, outside timing/memory measurement.
    """
    native, packed = fixture.native.detach().requires_grad_(), fixture.packed.detach().requires_grad_()
    raw_masks = []
    with execution_type(fixture.spec, debug=capture) as owner:
        if capture:
            original = owner.dropout_mask
            def observed(*args, **kwargs):
                raw = original(*args, **kwargs)
                raw_masks.append(raw.detach())
                return raw
            owner.dropout_mask = observed
        def native_call(value):
            return owner.native_dropout(fixture.site, fixture.p, value).sin()
        def packed_call(value):
            raw = owner.dropout_mask(fixture.site, fixture.p, sample_indices=fixture.packed_samples,
                token_positions=fixture.packed_positions, hidden_size=value.shape[-1], device=value.device)
            return (value * (raw.to(value.dtype) * (1 / (1 - fixture.p)))).sin()
        outputs = []
        for role, function, value in (('F', native_call, native), ('packed', packed_call, packed)):
            with owner.graph(role):
                outputs.append(checkpoint(function, value, use_reentrant=False,
                    context_fn=lambda r=role: owner.checkpoint_contexts(r, ('decoder', 0))))
        with owner.backward():
            (outputs[0].sum() + outputs[1].sum()).backward()
    return dict(native_output=outputs[0].detach(), packed_output=outputs[1].detach(),
                native_gradient=native.grad.detach(), packed_gradient=packed.grad.detach(),
                raw_masks=tuple(raw_masks), state=owner.state, events=owner.events)
