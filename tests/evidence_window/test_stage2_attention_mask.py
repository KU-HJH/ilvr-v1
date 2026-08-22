import pytest
import torch

from evidence_window.stage2_layout import EvidenceWindowEdge, Stage2InterleavedLayout


@pytest.fixture
def layout() -> Stage2InterleavedLayout:
    evidence = torch.tensor(
        [
            [[False, True, True, False, False, False, False, False],
             [False, False, False, False, False, True, True, False]],
            [[True, True, False, False, False, False, False, False],
             [False, False, False, True, True, False, False, False]],
        ],
        dtype=torch.bool,
    )
    windows = torch.tensor(
        [
            [[True, False, False, False, False, False, False, False],
             [False, False, False, True, True, False, False, False]],
            [[False, False, True, False, False, False, False, False],
             [False, False, False, False, False, True, False, False]],
        ],
        dtype=torch.bool,
    )
    causal = torch.tensor(
        [
            [[False, True], [False, False]],
            [[True, False], [False, True]],
        ],
        dtype=torch.bool,
    )
    return Stage2InterleavedLayout(
        evidence_token_mask=evidence,
        window_token_mask=windows,
        evidence_valid=torch.ones((2, 2), dtype=torch.bool),
        window_valid=torch.ones((2, 2), dtype=torch.bool),
        causal_pair_mask=causal,
    )


def test_exact_cartesian_product_only_for_selected_causal_edge(layout):
    """Blocking another batch row or token pair leaks an intervention."""
    from evidence_window.attention import build_window_local_block_mask

    edge = EvidenceWindowEdge(0, 0, 1)
    mask = build_window_local_block_mask(layout, edge)

    expected = torch.zeros((2, 8, 8), dtype=torch.bool)
    expected[0, 3, 1:3] = True
    expected[0, 4, 1:3] = True
    assert torch.equal(mask, expected)
    assert mask.shape == (2, 8, 8)
    assert mask.dtype is torch.bool
    assert mask.device == layout.window_token_mask.device


def test_prefix_slices_query_and_key_axes(layout):
    """Leaving key columns unsliced changes prefix-attention semantics."""
    from evidence_window.attention import build_window_local_block_mask, slice_block_mask

    full = build_window_local_block_mask(layout, EvidenceWindowEdge(0, 0, 1))

    assert torch.equal(slice_block_mask(full, 6), full[:, :6, :6])


@pytest.mark.parametrize(
    "edge",
    [
        EvidenceWindowEdge(0, 0, 0),
        EvidenceWindowEdge(-1, 0, 1),
        EvidenceWindowEdge(2, 0, 1),
        EvidenceWindowEdge(0, 2, 1),
        EvidenceWindowEdge(0, 0, 2),
    ],
)
def test_rejects_noncausal_or_out_of_range_edges(layout, edge):
    """Accepting an invalid edge could mask unrelated or future evidence."""
    from evidence_window.attention import build_window_local_block_mask

    with pytest.raises(ValueError):
        build_window_local_block_mask(layout, edge)
