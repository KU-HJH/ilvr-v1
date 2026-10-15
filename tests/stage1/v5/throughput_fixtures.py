"""Small operator fixtures and byte-for-byte pinned, independently loaded oracles.

The two frozen sources are from bb942b7323495cda716b7c5b889d6a8dc699aa23;
load them under their original package so relative imports remain unchanged.
"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from stage1.v5_metadata import V5BatchMetadata, V5Block


def original_module(name):
    package = 'stage1' if name == 'v5_selection' else 'transformers.models.qwen2_5_vl'
    module_name = package + '._frozen_throughput_' + name
    if module_name not in sys.modules:
        path = Path(__file__).with_name('frozen_throughput') / (name + '.py')
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


class OperatorCounts(TorchDispatchMode):
    """Observe real dispatched work without replacing tensor computations."""
    def __init__(self, key_shape=None):
        super().__init__()
        self.key_shape = key_shape
        self.key_float_copies = self.arange_calls = self.scalar_reads = 0
        self.candidate_index_tensors = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten._to_copy.default and tuple(args[0].shape) == self.key_shape:
            if kwargs.get('dtype') == torch.float32:
                self.key_float_copies += 1
        if func in (torch.ops.aten.arange.default, torch.ops.aten.arange.start):
            self.arange_calls += 1
        if func == torch.ops.aten._local_scalar_dense.default:
            self.scalar_reads += 1
        if func == torch.ops.aten.lift_fresh.default and args[0].ndim == 1 and args[0].dtype == torch.long:
            self.candidate_index_tensors += 1
        return func(*args, **kwargs)


class ObservationAttention(nn.Module):
    """Projection/hook operator fixture; native Qwen integration is tested separately."""
    def __init__(self, index, heads, dim, dtype):
        super().__init__()
        self.layer_idx, self.head_dim = index, dim
        self.num_key_value_groups, self.attention_dropout = 1, 0.
        self.rope_scaling, self.scaling = {'mrope_section': [dim // 2, 0, 0]}, dim ** -.5
        self.q_proj = nn.Linear(heads * dim, heads * dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(heads * dim, heads * dim, bias=False, dtype=dtype)

    def forward(self, x, *, position_embeddings):
        q, k = self.q_proj(x), self.k_proj(x)
        return q + k, (k, q)


class ObservationDecoder(nn.Module):
    def __init__(self, heads=4, dim=8, layers=2, dtype=torch.bfloat16):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation='eager', use_sliding_window=False)
        self.layers = nn.ModuleList([nn.Module() for _ in range(layers)])
        for i, layer in enumerate(self.layers):
            layer.self_attn = ObservationAttention(i, heads, dim, dtype)

    def forward(self, *, inputs_embeds, attention_mask=None, position_ids=None, **kwargs):
        x, kv = inputs_embeds, []
        b, t, _ = x.shape
        dim = self.layers[0].self_attn.head_dim
        cos = x.new_ones(3, b, t, dim)
        sin = x.new_zeros(3, b, t, dim)
        for layer in self.layers:
            x, pair = layer.self_attn(x, position_embeddings=(cos, sin))
            kv.append(pair)
        return SimpleNamespace(stage2_key_values=kv, last_hidden_state=x)


def observation_fixture(length=64, queries=6, heads=4, dim=8, layers=2):
    torch.manual_seed(211)
    decoder = ObservationDecoder(heads, dim, layers)
    x = torch.randn(2, length, heads * dim, dtype=torch.bfloat16, requires_grad=True)
    # Interleave samples deliberately: grouping may not reorder per-key layer sums.
    keys = tuple((row, length - queries + p) for p in range(queries) for row in (1, 0))
    candidates = tuple(tuple(range(row, length // 2, 2)) for row, _ in keys)
    padding = torch.ones(2, length, dtype=torch.long)
    padding[0, 1:4] = 0
    return decoder, dict(inputs_embeds=x, attention_mask=padding), keys, candidates


def selector_fixture(count=4, candidates=12, length=None):
    length = length or candidates * count + 8
    blocks, attention = [], {}
    for sample in (1, 0):
        for index in range(count):
            positions = tuple(range(index * candidates, (index + 1) * candidates))
            start = count * candidates + 1
            # Two queries per block and disjoint candidates across blocks.
            end = start + 2 + index * 3
            block = V5Block(sample, index, start, end, (end + 1, end + 2), (0, 1), positions)
            blocks.append(block)
            for query in block.query_positions:
                row = torch.zeros(max(length, end + 3))
                row[list(positions)] = torch.linspace(.01, .02, candidates)
                attention[(sample, query)] = row
    metadata = V5BatchMetadata(tuple(blocks))
    generator = torch.Generator().manual_seed(311)
    logits = torch.randn(len(metadata.query_keys), 17, generator=generator)
    reference = torch.randn(logits.shape, generator=generator)
    return metadata, attention, logits, reference
