"""Independent rectangular visibility and actual CUDA Flex replay witnesses."""

import copy

import pytest
import torch

import step_preference.sidecar as sidecar
from test_shared_prefix_model import fixture


def test_rectangular_block_mask_matches_literal_edges_and_padding():
    """Catches root leakage, other-branch edges, or padded create_block_mask probes."""
    _, _, _, _, plan = fixture()
    block = sidecar.build_sidecar_flex_block_mask(13, plan.groups[0], "cpu")
    q = torch.arange(128)[:, None]
    k = torch.arange(128)[None, :]
    actual = block.mask_mod(torch.tensor(0), torch.tensor(0), q, k)
    expected = torch.zeros((128, 128), dtype=torch.bool)
    # Four branches: chosen0 [4,6), rejected0 [4,8), chosen1 [10,12), rejected1 [10,14).
    for start, stop, ancestor, blocked in [(0,2,4,(2,4)), (2,6,4,(0,0)),
                                            (6,8,10,(8,10)), (8,12,10,(0,0))]:
        for row in range(start, stop):
            expected[row, :ancestor] = True
            expected[row, blocked[0]:blocked[1]] = False
            expected[row, 13+start:13+row+1] = True
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Flex")
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_cuda_flex_rectangular_scores_and_backward_match_dense(monkeypatch, backend):
    """Catches fallback to square/dense attention, expanded GQA, or missing suffix gradients."""
    import transformers.integrations.flex_attention as integration
    model, ids, _, _, plan = fixture(backend)
    model = model.to("cuda:0")
    ids = ids.to("cuda:0")
    reference = copy.deepcopy(model)
    calls = []
    original = integration.compile_friendly_flex_attention
    def witness(q, k, v, **kwargs):
        assert kwargs.get("kernel_options") == {"FORCE_USE_FLEX_ATTENTION": True}
        calls.append((q.shape, k.shape, kwargs["enable_gqa"]))
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(integration, "compile_friendly_flex_attention", witness)
    outputs = []
    for current, attention in ((model, "flex"), (reference, "dense")):
        out = current(input_ids=ids, use_cache=False,
            stage2_shared_prefix_inputs=dict(plan=plan, backend=attention, query_budget=1024))
        total = sum(r["mean_logp"] + .13*r["entropies"].mean() for r in out.stage2_sidecar_results)
        total.backward()
        outputs.append(out)
    assert calls and all(k[-2] == 13 + q[-2] and gqa for q,k,gqa in calls)
    assert all(q[1] > k[1] for q,k,_ in calls)
    for got, want in zip(outputs[0].stage2_sidecar_results, outputs[1].stage2_sidecar_results):
        torch.testing.assert_close(got["mean_logp"], want["mean_logp"], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(got["entropies"], want["entropies"], atol=1e-5, rtol=1e-4)
    for (name, got), (_, want) in zip(model.named_parameters(), reference.named_parameters()):
        if got.requires_grad:
            torch.testing.assert_close(got.grad, want.grad, atol=1e-5, rtol=1e-4, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual rectangular CUDA Flex geometry")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_short_query_gqa_geometry_uses_normal_flex_with_dense_gradient_oracle(monkeypatch, dtype):
    """Catches dropped kernel options and the observed Q99/GQA8 decoding resource failure."""
    import transformers.integrations.flex_attention as integration
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.manual_seed(419)
    branch = sidecar.SidecarBranch(name="actual-shape", kind="visual_off", step_id=0,
        token_ids=(12,)*99, clean_embedding_sources=tuple(range(114, 213)),
        logical_positions=tuple(range(114, 213)), clean_ancestor_end=114,
        blocked_start=100, blocked_end=108, selected_positions=tuple(range(99)),
        target_token_ids=(12,)*99, rejected_model_inputs=None)
    mask = sidecar.build_sidecar_flex_block_mask(213, (branch,), "cuda:0")
    q = torch.randn((1, 16, 99, 128), device="cuda:0", dtype=dtype, requires_grad=True)
    k = torch.randn((1, 2, 312, 128), device="cuda:0", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    reference_inputs = [t.detach().clone().requires_grad_() for t in (q,k,v)]
    original = integration.compile_friendly_flex_attention
    def observed(q, k, v, **kwargs):
        # Fail before recompiling the already observed invalid default decoding
        # specialization. GREEN delegates to actual compiled normal Flex.
        assert kwargs.get("kernel_options") == {"FORCE_USE_FLEX_ATTENTION": True}
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(integration, "compile_friendly_flex_attention", observed)
    got, _ = integration.flex_attention_forward(torch.nn.Module(), q, k, v, mask,
        scaling=128**-0.5, kernel_options={"FORCE_USE_FLEX_ATTENTION": True})
    visible = torch.zeros((99, 312), device="cuda:0", dtype=torch.bool)
    visible[:, :114] = True
    visible[:, 100:108] = False
    visible[:, 213:] = torch.ones((99,99), device="cuda:0", dtype=torch.bool).tril()
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(*reference_inputs,
            attn_mask=visible[None,None], enable_gqa=True).transpose(1,2)
    upstream = torch.randn_like(got)
    score = (got.float()*upstream.float()).sum()
    reference_score = (expected.float()*upstream.float()).sum()
    score.backward()
    reference_score.backward()
    assert all(torch.isfinite(t.grad).all() and t.grad.norm() > 0 for t in (q,k,v))
    if dtype == torch.float32:
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(score, reference_score, atol=1e-5, rtol=1e-4)
        for actual, reference in zip((q,k,v), reference_inputs):
            torch.testing.assert_close(actual.grad, reference.grad, atol=1e-5, rtol=1e-4)
    print(dict(dtype=str(dtype), q_heads=16, kv_heads=2, head_dim=128, query_rows=99,
        root_rows=213, max_output_abs=float((got-expected).abs().max()),
        score_abs=float((score-reference_score).abs()),
        max_gradient_abs=max(float((a.grad-r.grad).abs().max()) for a,r in zip((q,k,v),reference_inputs))))
