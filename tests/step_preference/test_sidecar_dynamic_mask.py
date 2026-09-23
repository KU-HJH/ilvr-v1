"""PyTorch 2.7 Flex captured-buffer contract and independent visibility oracles."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from step_preference.sidecar import SidecarBranch, build_sidecar_flex_block_mask


def branches(lengths):
    return tuple(SidecarBranch(
        name=f'visual-{i}', kind='visual_off', step_id=i, token_ids=(12,)*length,
        clean_embedding_sources=(), logical_positions=(), clean_ancestor_end=ancestor,
        blocked_start=left, blocked_end=right, selected_positions=(),
        target_token_ids=(), rejected_model_inputs=None,
    ) for i, (length, ancestor, left, right) in enumerate(zip(lengths, (4,10), (2,8), (4,10))))


def literal_visibility(root, lengths, padded_q, padded_k):
    """Two independently specified branches: roots {0,1} and {0..7}."""
    expected = torch.zeros((padded_q, padded_k), dtype=torch.bool)
    first, second = lengths
    for q in range(first):
        expected[q, :2] = True
        expected[q, root:root+q+1] = True
    for q in range(first, first+second):
        expected[q, :8] = True
        expected[q, root+first:root+q+1] = True
    return expected


@pytest.mark.parametrize('root,lengths', [(13,(2,3)), (213,(99,7)), (257,(129,5))])
def test_dynamic_flex_mask_uses_tensor_captures_and_matches_dense_attention(root, lengths):
    """Captured SymInt bounds/2-D metadata break Flex's tensor-only lowering path."""
    from torch.nn.attention.flex_attention import flex_attention
    mask = build_sidecar_flex_block_mask(root, branches(lengths), 'cpu')
    captured = []
    def tensor_buffer_backend(graph, example_inputs):
        for node in graph.graph.nodes:
            if node.op == 'call_function' and node.target is torch.ops.higher_order.flex_attention:
                buffers = node.args[-1]
                predicate = getattr(graph, node.args[4][-1].target)
                inputs = [n for n in predicate.graph.nodes if n.op == 'placeholder'][4:]
                # Dynamo also adds unused tensor-shape symbols; AOT drops those.
                # Arithmetic on a used captured SymInt reaches the failing lowering.
                used = [item for item, parameter in zip(buffers, inputs) if parameter.users]
                invalid = [str(item) for item in used if not isinstance(item, torch.fx.Node)
                           or not isinstance(item.meta.get('example_value'), torch.Tensor)]
                assert not invalid, f'Flex mask uses non-tensor buffers: {invalid}'
                captured.append(len(used))
        return graph.forward
    compiled = torch.compile(flex_attention, backend=tensor_buffer_backend, dynamic=True, fullgraph=True)
    q = torch.randn(1,2,sum(lengths),8)
    k, v = (torch.randn(1,2,root+sum(lengths),8) for _ in range(2))
    actual = compiled(q,k,v,block_mask=mask)
    visible = literal_visibility(root,lengths,sum(lengths),root+sum(lengths))
    expected = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=visible)
    assert captured and all(captured)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('root,lengths', [(13,(2,3)), (213,(99,7)), (257,(129,5))])
def test_padded_predicate_matches_literal_edges_and_frozen_source(root,lengths):
    """Padded query/key lanes and other branches must remain invisible."""
    source_root = Path('/root/workspace/ILVR/.cache/until-temp2-alignment-20260923/reference-source')
    if not source_root.is_dir():
        pytest.skip('frozen source unavailable')
    from scripts.verify_until_native import ExternalSource
    qlen = sum(lengths)
    pq, pk = ((qlen+127)//128)*128, ((root+qlen+127)//128)*128
    q, k = torch.arange(pq)[:,None], torch.arange(pk)[None,:]
    mask = build_sidecar_flex_block_mask(root,branches(lengths),'cpu')
    actual = mask.mask_mod(torch.tensor(0),torch.tensor(0),q,k)
    expected = literal_visibility(root,lengths,pq,pk)
    assert torch.equal(actual,expected)
    layout = SimpleNamespace(input_token_ids=torch.zeros((1,qlen),dtype=torch.long),
        clean_length=root, clean_attention_mask=torch.ones((1,root),dtype=torch.long),
        branches=(SimpleNamespace(query_start=0,query_end=lengths[0],clean_visible_end=4,blocked_clean_interval=(2,4)),
                  SimpleNamespace(query_start=lengths[0],query_end=qlen,clean_visible_end=10,blocked_clean_interval=(8,10))))
    with ExternalSource(source_root) as source:
        native = source.module('transformers.models.qwen2_5_vl.stage2_sidecar')
        native_mask = native._sidecar_visibility(layout,torch.device('cpu'))
        assert torch.equal(actual,native_mask(0,0,q,k))
