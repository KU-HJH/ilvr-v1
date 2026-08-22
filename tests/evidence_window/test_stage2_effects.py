import torch

from evidence_window.stage2_effects import (
    Stage2ConditionOutput,
    compute_stage2_effects,
)
from evidence_window.scores import window_scores
from evidence_window.stage2_layout import (
    EvidenceWindowEdge,
    Stage2InterleavedLayout,
    select_stage2_edges,
)


def _layout() -> Stage2InterleavedLayout:
    evidence = torch.tensor(
        [
            [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]],
            [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=torch.bool,
    )
    windows = torch.tensor(
        [
            [[0, 1, 0, 0, 0], [0, 0, 0, 1, 1]],
            [[0, 1, 1, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=torch.bool,
    )
    causal = torch.tensor(
        [
            [[1, 1], [0, 1]],
            [[1, 0], [0, 0]],
        ],
        dtype=torch.bool,
    )
    return Stage2InterleavedLayout(
        evidence_token_mask=evidence,
        window_token_mask=windows,
        evidence_valid=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        window_valid=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        causal_pair_mask=causal,
    )


def test_window_scores_exclude_other_tokens_and_support_both_reductions():
    """Including the large out-of-window score or omitting mean pooling biases windows."""
    logps = torch.tensor([[1.0, 2.0, 100.0, 4.0]])
    masks = torch.tensor([[[1, 1, 0, 0], [0, 0, 0, 1]]], dtype=torch.bool)

    assert torch.equal(window_scores(logps, masks, "sum"), torch.tensor([[3.0, 4.0]]))
    assert torch.equal(window_scores(logps, masks, "mean"), torch.tensor([[1.5, 4.0]]))


def test_empty_window_mean_is_finite_zero():
    """Dividing an empty padded window by zero would inject NaN into later reductions."""
    scores = window_scores(
        torch.tensor([[1.0, 2.0]]),
        torch.zeros((1, 1, 2), dtype=torch.bool),
        "mean",
    )

    assert torch.equal(scores, torch.zeros((1, 1)))
    assert torch.isfinite(scores).all()


def test_one_matched_full_forward_plus_one_exact_forward_per_selected_edge():
    """Reusing a global evidence ablation would not realize pair-local LOO semantics."""
    layout = _layout()
    edges = select_stage2_edges(layout, 0)
    calls = []

    def recording_forward(block_mask, need_hidden):
        calls.append((block_mask, need_hidden))
        return Stage2ConditionOutput(torch.zeros((2, 5)), None)

    effects = compute_stage2_effects(layout, edges, recording_forward, "mean", False)

    assert calls[0] == (None, False)
    assert len(calls) == 1 + effects.pair_mask.sum().item()
    assert effects.forward_count == len(calls)
    assert torch.equal(effects.pair_mask, layout.causal_pair_mask)
    for (block_mask, need_hidden), edge in zip(calls[1:], edges):
        assert need_hidden is False
        predictor_queries = torch.zeros_like(
            layout.window_token_mask[edge.batch_index, edge.window_index]
        )
        predictor_queries[:-1] = layout.window_token_mask[
            edge.batch_index, edge.window_index, 1:
        ]
        expected = (
            predictor_queries[:, None]
            & layout.evidence_token_mask[edge.batch_index, edge.evidence_index][None, :]
        )
        assert torch.equal(block_mask[edge.batch_index], expected)
        assert not block_mask[torch.arange(block_mask.shape[0]) != edge.batch_index].any()


def test_one_token_window_blocks_its_predictor_query_and_gets_nonzero_effect():
    """Blocking target row t instead of predictor row t-1 makes a one-token edge score zero."""
    layout = Stage2InterleavedLayout(
        evidence_token_mask=torch.tensor([[[1, 1, 1, 0]]], dtype=torch.bool),
        window_token_mask=torch.tensor([[[0, 0, 0, 1]]], dtype=torch.bool),
        evidence_valid=torch.tensor([[1]], dtype=torch.bool),
        window_valid=torch.tensor([[1]], dtype=torch.bool),
        causal_pair_mask=torch.tensor([[[1]]], dtype=torch.bool),
    )
    calls = []

    def forward(block_mask, need_hidden):
        del need_hidden
        calls.append(block_mask)
        token_logps = torch.tensor([[0.0, 0.0, 0.0, 5.0]])
        if block_mask is not None and block_mask[0, 2, :3].all():
            token_logps[0, 3] = 2.0
        return Stage2ConditionOutput(token_logps, None)

    effects = compute_stage2_effects(
        layout,
        [EvidenceWindowEdge(0, 0, 0)],
        forward,
        "mean",
        False,
    )

    expected_block = torch.zeros((1, 4, 4), dtype=torch.bool)
    expected_block[0, 2, :3] = True
    assert torch.equal(calls[1], expected_block)
    assert effects.contributions[0, 0, 0].item() == 3.0


def test_contributions_are_exact_full_minus_pair_loo_for_heterogeneous_batch():
    """Wrong sign, window, or padded-edge writes corrupt the relation matrix."""
    layout = _layout()
    edges = select_stage2_edges(layout, 0)
    full = torch.tensor([[10.0, 2.0, 99.0, 4.0, 6.0], [8.0, 5.0, 7.0, 0.0, 0.0]])

    loo_by_edge = {
        (0, 0, 0): torch.tensor([[10.0, 1.0, 99.0, 4.0, 6.0], [8.0, 5.0, 7.0, 0.0, 0.0]]),
        (0, 2, 0): torch.tensor([[10.0, 2.0, 99.0, 1.0, 3.0], [8.0, 5.0, 7.0, 0.0, 0.0]]),
        (0, 2, 2): torch.tensor([[10.0, 2.0, 99.0, 3.0, 5.0], [8.0, 5.0, 7.0, 0.0, 0.0]]),
        (1, 0, 0): torch.tensor([[10.0, 2.0, 99.0, 4.0, 6.0], [8.0, 1.0, 3.0, 0.0, 0.0]]),
    }

    def forward(block_mask, need_hidden):
        del need_hidden
        if block_mask is None:
            return Stage2ConditionOutput(full, None)
        batch_index, query, key = block_mask.nonzero(as_tuple=False)[0].tolist()
        return Stage2ConditionOutput(loo_by_edge[(batch_index, query, key)], None)

    effects = compute_stage2_effects(layout, edges, forward, "mean", False)
    expected = torch.tensor(
        [
            [[1.0, 3.0], [0.0, 1.0]],
            [[4.0, 0.0], [0.0, 0.0]],
        ]
    )

    assert torch.equal(effects.contributions, expected)
    assert torch.isfinite(effects.contributions).all()


def test_sum_reduction_is_used_for_a_real_selected_edge():
    """Routing real edges through mean despite explicit sum changes length-dependent credit."""
    layout = Stage2InterleavedLayout(
        evidence_token_mask=torch.tensor([[[1, 0, 0, 0, 0]]], dtype=torch.bool),
        window_token_mask=torch.tensor([[[0, 0, 0, 1, 1]]], dtype=torch.bool),
        evidence_valid=torch.tensor([[1]], dtype=torch.bool),
        window_valid=torch.tensor([[1]], dtype=torch.bool),
        causal_pair_mask=torch.tensor([[[1]]], dtype=torch.bool),
    )

    def forward(block_mask, need_hidden):
        del need_hidden
        token_logps = (
            torch.tensor([[0.0, 0.0, 0.0, 4.0, 6.0]])
            if block_mask is None
            else torch.tensor([[0.0, 0.0, 0.0, 1.0, 3.0]])
        )
        return Stage2ConditionOutput(token_logps, None)

    effects = compute_stage2_effects(
        layout,
        [EvidenceWindowEdge(0, 0, 0)],
        forward,
        "sum",
        False,
    )

    assert effects.contributions[0, 0, 0].item() == 6.0


def test_full_and_loo_policy_branches_both_receive_gradients():
    """Detaching either branch defeats learned full-versus-counterfactual credit."""
    layout = _layout()
    edge = EvidenceWindowEdge(0, 0, 0)
    full_scale = torch.tensor(2.0, requires_grad=True)
    loo_scale = torch.tensor(3.0, requires_grad=True)

    def forward(block_mask, need_hidden):
        del need_hidden
        scale = full_scale if block_mask is None else loo_scale
        return Stage2ConditionOutput(scale * torch.ones((2, 5)), None)

    effects = compute_stage2_effects(layout, [edge], forward, "mean", False)
    effects.contributions[effects.pair_mask].sum().backward()

    assert full_scale.grad is not None and full_scale.grad.item() == 1.0
    assert loo_scale.grad is not None and loo_scale.grad.item() == -1.0


def test_hidden_residual_is_window_mean_full_minus_loo():
    """Pooling target rows instead of causal predictor rows misaligns residuals and logits."""
    layout = Stage2InterleavedLayout(
        evidence_token_mask=torch.tensor([[[1, 1, 0, 0]]], dtype=torch.bool),
        window_token_mask=torch.tensor([[[0, 0, 1, 1]]], dtype=torch.bool),
        evidence_valid=torch.tensor([[1]], dtype=torch.bool),
        window_valid=torch.tensor([[1]], dtype=torch.bool),
        causal_pair_mask=torch.tensor([[[1]]], dtype=torch.bool),
    )
    full_hidden = torch.tensor([[[50.0, 50.0], [4.0, 8.0], [6.0, 10.0], [100.0, 100.0]]])
    loo_hidden = torch.tensor([[[1.0, 1.0], [1.0, 5.0], [3.0, 7.0], [0.0, 0.0]]])
    hidden_requests = []

    def forward(block_mask, need_hidden):
        hidden_requests.append(need_hidden)
        hidden = full_hidden if block_mask is None else loo_hidden
        return Stage2ConditionOutput(torch.zeros((1, 4)), hidden)

    effects = compute_stage2_effects(
        layout,
        [EvidenceWindowEdge(0, 0, 0)],
        forward,
        "mean",
        True,
    )

    assert hidden_requests == [True, True]
    assert effects.hidden_residuals is not None
    assert effects.hidden_residuals.shape == (1, 1, 1, 2)
    assert torch.equal(effects.hidden_residuals[0, 0, 0], torch.tensor([3.0, 3.0]))


def test_no_edge_batch_returns_finite_empty_effects_after_only_matched_full():
    """Empty edge reductions must fall back safely instead of creating NaN or fake pairs."""
    layout = Stage2InterleavedLayout(
        evidence_token_mask=torch.zeros((1, 0, 3), dtype=torch.bool),
        window_token_mask=torch.tensor([[[1, 1, 0]]], dtype=torch.bool),
        evidence_valid=torch.zeros((1, 0), dtype=torch.bool),
        window_valid=torch.tensor([[1]], dtype=torch.bool),
        causal_pair_mask=torch.zeros((1, 0, 1), dtype=torch.bool),
    )
    calls = []

    def forward(block_mask, need_hidden):
        calls.append((block_mask, need_hidden))
        return Stage2ConditionOutput(torch.tensor([[1.0, 2.0, 3.0]]), None)

    effects = compute_stage2_effects(layout, [], forward, "sum", False)

    assert calls == [(None, False)]
    assert effects.forward_count == 1
    assert effects.contributions.shape == (1, 0, 1)
    assert effects.pair_mask.shape == (1, 0, 1)
    assert torch.isfinite(effects.contributions).all()
    assert effects.hidden_residuals is None
