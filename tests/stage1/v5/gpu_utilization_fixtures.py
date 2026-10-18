"""Bounded four-operator fixtures; independent sources frozen at source65643f0.

No weights are loaded or saved. The older frozen_throughput oracles stay intact.
"""
import importlib.util
from dataclasses import replace
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from stage1.equivalence import Stage1StochasticContext
from stage1.v5_config import Stage1V5Config
from stage1.v5_metadata import V5BatchMetadata, V5Block
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
        self.copy_calls = self.masked_fill_calls = 0
        self.integer_float_casts = self.stable_sort_calls = 0
        self.matmul_shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten._to_copy.default and not args[0].is_floating_point():
            if kwargs.get('dtype') in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                self.integer_float_casts += 1
        if func == torch.ops.aten.sort.stable and kwargs.get('stable') and kwargs.get('descending'):
            self.stable_sort_calls += 1
        if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
            self.matmul_shapes.append((tuple(args[0].shape), tuple(args[1].shape)))
        if func == torch.ops.aten.rand.generator:
            self.random_draw_calls += 1
            self.random_draw_elements += math.prod(args[0])
        if func == torch.ops.aten.select.int:
            self.select_calls += 1
        if func == torch.ops.aten.stack.default:
            self.stack_calls += 1
        if func == torch.ops.aten.index_put_.default:
            self.index_put_calls += 1
        if func == torch.ops.aten.copy_.default:
            self.copy_calls += 1
        if func == torch.ops.aten.masked_fill_.Scalar:
            self.masked_fill_calls += 1
        return super().__torch_dispatch__(func, types, args, kwargs)


def selector_boundary_fixture(case='normal', *, attention_device='cpu', logits_device='cpu'):
    """Interleaved samples and exact FP32 neighbors of both decision boundaries."""
    blocks, attention = [], {}
    candidates = tuple(range(2, 42, 2))
    config = Stage1V5Config()
    if case == 'zero_cap':
        config = replace(config, text_cap=0.)
    if case.startswith('cumulative_'):
        config = replace(config, rho_min=.5, rho_max=.5)
    if case.startswith('lowmass_'):
        config = replace(config, low_attention_mass=float(torch.tensor(.1)))
    for index, sample in enumerate((1, 0, 1, 0)):
        start, end = 43 + 4 * index, 45 + 4 * index
        positions = () if case == 'empty' else candidates
        block = V5Block(sample, index, start, end, (end + 1, end + 2), (0, 1), positions)
        blocks.append(block)
        for query in block.query_positions:
            row = torch.zeros(64)
            values = torch.linspace(.005, .025, len(candidates))
            if case == 'ties':
                values.fill_(.02)
            elif case == 'low_mass':
                values.fill_(.001)
            elif case.startswith('cumulative_'):
                values.zero_()
                values[:3] = torch.tensor([.25, .125, .125])
                if case != 'cumulative_equal':
                    values[0] = torch.nextafter(values[0], torch.tensor(
                        0. if case.endswith('below') else float('inf')))
            elif case.startswith('lowmass_'):
                values.zero_()
                values[0] = .1
                if case != 'lowmass_equal':
                    values[0] = torch.nextafter(values[0], torch.tensor(
                        0. if case.endswith('below') else float('inf')))
            if positions:
                row[list(positions)] = values
            attention[(sample, query)] = row.to(attention_device)
    metadata = V5BatchMetadata(tuple(blocks))
    generator = torch.Generator().manual_seed(733)
    clean = torch.randn(len(metadata.query_keys), 11, generator=generator).to(logits_device)
    reference = torch.randn(clean.shape, generator=generator).to(logits_device)
    return metadata, attention, clean, reference, config


ATTENTION_ATOL, ATTENTION_RTOL = 1e-7, 1e-5


class JointObservationAttention(nn.Module):
    """Real projections with GQA; the observer owns the native mRoPE/GQA math."""
    def __init__(self, index, heads, kv_heads, dim, dtype):
        super().__init__()
        self.layer_idx, self.head_dim = index, dim
        self.num_key_value_groups, self.attention_dropout = heads // kv_heads, 0.
        self.rope_scaling, self.scaling = {'mrope_section': [dim // 6] * 3}, dim ** -.5
        self.q_proj = nn.Linear(heads * dim, heads * dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(heads * dim, kv_heads * dim, bias=False, dtype=dtype)

    def forward(self, x, *, position_embeddings):
        q, k = self.q_proj(x), self.k_proj(x)
        return x + q * .125, (k, q)


class JointObservationDecoder(nn.Module):
    def __init__(self, heads, kv_heads, dim, layers, dtype):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation='eager', use_sliding_window=False)
        self.layers = nn.ModuleList([nn.Module() for _ in range(layers)])
        for index, layer in enumerate(self.layers):
            layer.self_attn = JointObservationAttention(index, heads, kv_heads, dim, dtype)

    def forward(self, *, inputs_embeds, position_ids, **kwargs):
        x, kv = inputs_embeds, []
        dim = self.layers[0].self_attn.head_dim
        frequencies = 10000. ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
        angles = position_ids[..., None].float() * frequencies
        angles = torch.cat((angles, angles), -1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        for layer in self.layers:
            x, pair = layer.self_attn(x, position_embeddings=(cos, sin))
            kv.append(pair)
        return SimpleNamespace(stage2_key_values=kv, last_hidden_state=x)


def joint_observation_fixture(device, *, queries=35, length=128, case='normal'):
    """More than 32 queries, source holes, grouped heads and three-axis RoPE."""
    heads, kv_heads, dim, layers = 4, 2, 12, 2
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(797)
        decoder = JointObservationDecoder(heads, kv_heads, dim, layers, torch.bfloat16)
        x = torch.randn(2, length, heads * dim, dtype=torch.bfloat16)
        if case in ('ties', 'near_ties'):
            for layer in decoder.layers:
                layer.self_attn.q_proj.weight.data.mul_(0. if case == 'ties' else 1e-4)
    decoder.to(device)
    positions = torch.arange(length)
    position_ids = torch.stack((positions * 2 + 1, positions // 3 + 5, positions % 7 + 2))
    position_ids = position_ids[:, None].expand(-1, 2, -1).to(device)
    padding = torch.ones(2, length, dtype=torch.long, device=device)
    padding[0, [1, 5, 9]], padding[1, [2, 8, 12]] = 0, 0
    keys = tuple((sample, length - queries + offset) for offset in range(queries) for sample in (1, 0))
    candidates = tuple(tuple(range(sample, min(48, length - queries - 2), 2)) for sample, _ in keys)
    metadata = joint_metadata(keys, candidates)
    generator = torch.Generator().manual_seed(809)
    clean = torch.randn(len(keys), 17, generator=generator).to(device)
    reference = torch.randn(clean.shape, generator=generator).to(device)
    return SimpleNamespace(decoder=decoder, inputs=dict(inputs_embeds=x.to(device),
        attention_mask=padding, position_ids=position_ids), keys=keys, candidates=candidates,
        metadata=metadata, clean=clean, reference=reference, heads=heads, dim=dim,
        length=length, layers=layers, queries_per_sample=queries)


def joint_metadata(keys, candidates):
    return V5BatchMetadata(tuple(V5Block(sample, index, query - 1, query, (query + 1,), (0,), positions)
        for index, ((sample, query), positions) in enumerate(zip(keys, candidates))))


def joint_selection_comparison(metadata, before_rows, after_rows, clean, reference):
    """Cross both trace and selector changes, including measured FP32 neighbors.

    Boundaries come from the frozen row, never the candidate. They exercise the
    same projections/logits through both paths; finite cases do not prove future
    input choices invariant under a different GEMM reduction shape.
    """
    from stage1.v5_selection import select_v5_text_keys
    original = original_module('v5_selection').select_v5_text_keys
    first = before_rows[metadata.query_keys[0]]
    mass = first.sum()
    prefix = sorted(first.tolist(), reverse=True)[:2]
    ratio = torch.tensor(sum(prefix) / float(mass), dtype=torch.float32)
    configs = {'default': Stage1V5Config()}
    for suffix, target in (('below', torch.tensor(0.)), ('equal', None), ('above', torch.tensor(float('inf')))):
        rho = ratio if target is None else torch.nextafter(ratio, target)
        threshold = mass.cpu() if target is None else torch.nextafter(mass.cpu(), target)
        configs['cumulative_' + suffix] = Stage1V5Config(rho_min=float(rho), rho_max=float(rho))
        configs['lowmass_' + suffix] = Stage1V5Config(low_attention_mass=float(threshold))
    results = {}
    for case, config in configs.items():
        expected = original(metadata, before_rows, clean, reference, config, compact_attention=True)
        actual = select_v5_text_keys(metadata, after_rows, clean, reference, config, compact_attention=True)
        results[case] = dict(exact=list(actual.items()) == list(expected.items()),
            keys=[dict(query=list(key), before=list(expected[key]), after=list(actual[key]))
                  for key in metadata.query_keys],
            rho_min=config.rho_min, rho_max=config.rho_max, low_attention_mass=config.low_attention_mass)
    return results


@torch.no_grad()
def observe_joint(capture, fixture):
    projections = []
    handles = [getattr(layer.self_attn, name + '_proj').register_forward_hook(
        lambda _module, _args, output: projections.append(output.detach().clone()))
        for layer in fixture.decoder.layers for name in ('q', 'k')]
    try:
        with capture(fixture.decoder, fixture.keys, fixture.candidates) as trace:
            fixture.decoder(**fixture.inputs)
    finally:
        for handle in handles:
            handle.remove()
    return trace.attention_rows, projections


class ReplayMaskDecoder(nn.Module):
    """Capture the actual replay bias; no model/attention computation."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.bias = None
        self.arguments = None

    def forward(self, *, inputs_embeds, attention_mask, **kwargs):
        self.bias = attention_mask
        self.arguments = dict(inputs_embeds=inputs_embeds, **kwargs)
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
def run_replay_mask(replay, fixture, *, need_voff=True, need_textoff=True):
    replay(fixture.decoder, fixture.head, fixture.trace, fixture.blocks, execution=fixture.execution,
           need_voff=need_voff, need_textoff=need_textoff)
    bias, fixture.decoder.bias = fixture.decoder.bias, None
    fixture.decoder.arguments = None
    return bias


def replay_edge_fixture(device, dtype=torch.float32):
    """Unequal source/suffix lengths, sparse targets, padding and empty body/block.

    Values and coordinates are inputs only: expected masks come from the pinned
    original replay, never from a candidate visibility formula.
    """
    generator = torch.Generator().manual_seed(619)
    length = 40
    x = torch.randn(3, length, 8, generator=generator).to(device=device, dtype=dtype)
    padding = (torch.arange(length, device=device)[None] <
               torch.tensor([37, 31, 23], device=device)[:, None]).long()
    padding[0, [1, 5]], padding[1, [0, 2]], padding[2, [3, 6]] = 0, 0, 0
    positions = torch.stack((torch.arange(length) * 2, torch.arange(length) * 3 + 1,
                             torch.arange(length) + 5))[:, None].expand(-1, 3, -1).to(device)
    kv = tuple(tuple(torch.randn(3, 2, length, 4, generator=generator).to(device=device, dtype=dtype)
                     for _ in range(2)) for _ in range(2))
    trace = V5Trace(inputs_embeds=x, position_ids=positions, padding=padding, key_values=kv)
    blocks = (
        V5ReplayBlock(2, 10, 11, (12, 14, 17), (0, 1, 2), ((5, 1, 5), (), (0, 4))),
        V5ReplayBlock(0, 4, 7, (), (), ()),
        V5ReplayBlock(0, 15, 21, (22, 25, 29), (3, 4, 5), ((0, 8), (9,), (8, 2))),
        V5ReplayBlock(1, 4, 9, (10, 13), (6, 7), ((), (3, 1))),
    )
    return SimpleNamespace(decoder=ReplayMaskDecoder().to(device), head=nn.Identity(), trace=trace,
                           blocks=blocks, execution=SimpleNamespace(debug=False, dropout_mask=None))


def run_raw_dropout(execution_type, spec, samples, positions, *, p=.05, hidden_size=5, debug=False):
    """Exercise real graph/unit ownership around each stochastic site."""
    results = []
    with execution_type(spec, debug=debug) as owner:
        with owner.graph('F'):
            with owner.checkpoint_contexts('F', ('decoder', 0))[0]:
                for site in spec.expected_sites:
                    results.append(owner.dropout_mask(site, p, sample_indices=samples,
                        token_positions=positions, hidden_size=hidden_size, device=samples.device))
        with owner.backward():
            pass
    return tuple(results), owner


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
