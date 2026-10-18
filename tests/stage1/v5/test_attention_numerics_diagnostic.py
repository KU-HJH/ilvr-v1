"""Narrow CPU diagnosis of the retained Task4 long-geometry failure.

The original failure and pre-repair numerical diagnosis are retained in results.
This regression compares actual frozen/current dispatches and the assessed
bounded operator probe; it makes no CUDA or performance claim.
"""
from collections import defaultdict
import json

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from stage1.v5_config import Stage1V5Config
from stage1.v5_selection import select_v5_text_keys
from tests.stage1.v5.gpu_utilization_fixtures import (
    joint_metadata, joint_selection_comparison, original_module,
)
from tests.stage1.v5.throughput_fixtures import observation_fixture
from transformers.models.qwen2_5_vl.stage1_v5_trace import capture_v5_trace


BAD_KEYS = {(1, 8125), (0, 8136), (1, 8148), (1, 8179)}


class AttentionStages(TorchDispatchMode):
    def __init__(self, keys, *, score_stack=False):
        super().__init__()
        grouped = defaultdict(list)
        for key in keys:
            grouped[key[0]].append(key)
        self.keys = [key for group in grouped.values() for key in group]
        self.offset, self.current, self.score_stack = 0, None, score_stack
        self.stages = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        original_scores = (not self.score_stack and func == torch.ops.aten.bmm.default
                           and tuple(args[1].shape) == (8, 32, 8192))
        repaired_scores = (self.score_stack and func == torch.ops.aten.stack.default
                           and output.ndim == 3 and output.shape[0] == 8 and output.shape[-1] == 8192)
        if original_scores or repaired_scores:
            width = output.shape[1]
            layer = self.offset // len(self.keys)
            start = self.offset % len(self.keys)
            self.current = (layer, self.keys[start:start + width])
            self.offset += width
            self.save('scores', output)
        elif func == torch.ops.aten._softmax.default and args[0].shape[-1] == 8192:
            self.save('masked_scaled_scores', args[0])
            self.save('probabilities', output)
        elif (func == torch.ops.aten.mean.dim and self.current is not None
              and args[0].shape[0] == 8 and args[0].shape[-1] == 8192):
            layer, keys = self.current
            means = output[None] if output.ndim == 1 else output
            for index, key in enumerate(keys):
                if key in BAD_KEYS:
                    self.stages[layer, key, 'head_mean'] = means[index].clone()
        return output

    def save(self, name, values):
        layer, keys = self.current
        values = values[:, None] if values.ndim == 2 else values
        for index, key in enumerate(keys):
            if key in BAD_KEYS:
                self.stages[layer, key, name] = values[:, index].clone()


def difference(before, after):
    finite = before.isfinite() & after.isfinite()
    error = (before[finite] - after[finite]).abs()
    return dict(exact=torch.equal(before, after), different=int((before != after).sum()),
                max_abs=float(error.max()) if error.numel() else 0.)


@torch.no_grad()
def probe_bounded_matrix_vectors(decoder, inputs, keys, candidates):
    """Only assessed repair: zero-stride K views, original M=1 dot shape.

    Batch independent queries within each head without expanding/copying K's
    storage. Scores retain the approved [heads, bounded_chunk, keys] workspace.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb, repeat_kv,
    )
    projections, rows, stages, handles = {}, {}, {}, []
    grouped = defaultdict(list)
    for key, positions in zip(keys, candidates):
        grouped[key[0]].append((key, positions))

    def observe(attn, args, kwargs, output):
        q, k = projections.pop((attn.layer_idx, 'q')), projections.pop((attn.layer_idx, 'k'))
        b, t, _ = q.shape
        q, k = q.view(b, t, -1, attn.head_dim).transpose(1, 2), k.view(b, t, -1, attn.head_dim).transpose(1, 2)
        q, k = apply_multimodal_rotary_pos_emb(q, k, *kwargs['position_embeddings'], attn.rope_scaling['mrope_section'])
        k = repeat_kv(k, attn.num_key_value_groups)
        coordinates = torch.arange(t)
        for sample, queries in grouped.items():
            sample_k = k[sample].float().transpose(-2, -1)
            chunk = max(1, min(32, 33554432 // (2 * q.shape[1] * t * 4)))
            for start in range(0, len(queries), chunk):
                block = queries[start:start + chunk]
                positions = torch.tensor([key[1] for key, _ in block])
                query = q[sample].index_select(1, positions).float()
                scores = torch.stack([torch.bmm(query[head, :, None],
                    sample_k[head][None].expand(len(block), -1, -1)).squeeze(1)
                    for head in range(query.shape[0])])
                visible = (coordinates[None] <= positions[:, None]) & inputs['attention_mask'][sample].bool()[None]
                for index, (key, _) in enumerate(block):
                    if key in BAD_KEYS:
                        stages[attn.layer_idx, key, 'scores'] = scores[:, index].clone()
                weights = scores.mul_(attn.scaling).masked_fill_(~visible[None], -float('inf')).softmax(-1).mean(0)
                for index, (key, positions) in enumerate(block):
                    values = weights[index][list(positions)] / len(decoder.layers)
                    rows[key] = rows.get(key, 0) + values
                    if key in BAD_KEYS:
                        stages[attn.layer_idx, key, 'head_mean'] = weights[index].clone()
    for layer in decoder.layers:
        attn = layer.self_attn
        for name in ('q', 'k'):
            def save(_module, _args, output, identity=(attn.layer_idx, name)):
                projections[identity] = output.detach()
            handles.append(getattr(attn, name + '_proj').register_forward_hook(save))
        handles.append(attn.register_forward_hook(observe, with_kwargs=True))
    try:
        decoder(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return rows, stages


@torch.no_grad()
def test_long_attention_repair_preserves_original_matrix_vector_arithmetic():
    decoder, inputs, keys, candidates = observation_fixture(length=8192, queries=67, heads=8, dim=32, layers=2)
    captured = []
    for index, capture in enumerate((original_module('stage1_v5_trace').capture_v5_trace, capture_v5_trace)):
        projections = []
        handles = [getattr(layer.self_attn, name + '_proj').register_forward_hook(
            lambda _module, _args, output: projections.append(output.detach().clone()))
            for layer in decoder.layers for name in ('q', 'k')]
        try:
            with AttentionStages(keys, score_stack=bool(index)) as dispatched, capture(decoder, keys, candidates) as trace:
                decoder(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        assert dispatched.offset == 2 * len(keys)
        captured.append((trace.attention_rows, dispatched, projections))
    (before, old, old_projections), (after, new, new_projections) = captured
    assert len(old_projections) == len(new_projections) == 4
    assert all(torch.equal(a, b) for a, b in zip(old_projections, new_projections))
    generator = torch.Generator().manual_seed(809)
    clean = torch.randn(len(keys), 17, generator=generator)
    reference = torch.randn(clean.shape, generator=generator)
    metadata = joint_metadata(keys, candidates)
    joint = joint_selection_comparison(metadata, before, after, clean, reference)
    original_select = original_module('v5_selection').select_v5_text_keys
    identical_rows = {}
    for name, case in joint.items():
        config = Stage1V5Config(rho_min=case['rho_min'], rho_max=case['rho_max'], low_attention_mass=case['low_attention_mass'])
        identical_rows[name] = original_select(metadata, after, clean, reference, config, compact_attention=True) == select_v5_text_keys(metadata, after, clean, reference, config, compact_attention=True)
    repair, repair_stages = probe_bounded_matrix_vectors(decoder, inputs, keys, candidates)
    repair_joint = joint_selection_comparison(metadata, before, repair, clean, reference)
    comparisons = []
    for layer in range(2):
        for key in sorted(BAD_KEYS):
            comparisons.append(dict(layer=layer, query=key,
                stages={stage: difference(old.stages[layer, key, stage], new.stages[layer, key, stage])
                        for stage in ('scores', 'masked_scaled_scores', 'probabilities', 'head_mean')},
                matrix_vector_probe={stage: difference(old.stages[layer, key, stage], repair_stages[layer, key, stage])
                                     for stage in ('scores', 'head_mean')}))
    changed = {name: [entry for entry in case['keys'] if entry['before'] != entry['after']]
               for name, case in joint.items() if not case['exact']}
    result = dict(geometry=dict(batch=2, length=8192, heads=8, head_dim=32, layers=2, queries_per_sample=67, candidates_per_query=2048),
        input_projections_q_k_exact=True, stage_comparisons=comparisons,
        original_selector_on_identical_current_rows_exact=identical_rows,
        current_changed_keys=changed,
        current_rows_exact=all(torch.equal(before[key], after[key]) for key in keys),
        matrix_vector_probe=dict(selected_keys_exact=all(case['exact'] for case in repair_joint.values()),
            full_rows_exact=all(torch.equal(before[key], repair[key]) for key in keys),
            max_row_absolute_error=max(float((before[key] - repair[key]).abs().max()) for key in keys),
            changed={name: [entry for entry in case['keys'] if entry['before'] != entry['after']]
                     for name, case in repair_joint.items() if not case['exact']}))
    print(json.dumps(result, indent=2))
    assert not changed, 'the retained long-profile discrete-selection failure recurred'
    assert result['current_rows_exact']
    assert all(stage['exact'] for comparison in comparisons for stage in comparison['stages'].values())
    assert result['matrix_vector_probe']['full_rows_exact']
    assert result['matrix_vector_probe']['selected_keys_exact']
    assert all(identical_rows.values()), 'selector change independently contributes to the failure'
