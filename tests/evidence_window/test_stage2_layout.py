import pytest
import torch

from evidence_window.layout import TokenIds
from evidence_window.stage2_layout import (
    extract_stage2_layout,
    select_stage2_edges,
    validate_plan_types,
)


TOKENS = TokenIds(90, 91, 92, 0, frozenset({0, 90, 91, 92, 99}))


def _layout(ids, labels=None):
    ids = torch.tensor([ids])
    if labels is None:
        labels = ids.clone()
    else:
        labels = torch.tensor([labels])
    return extract_stage2_layout(ids, ids.ne(0), labels, TOKENS)


def _three_evidence_layout():
    return _layout([10, 90, 92, 91, 20, 90, 92, 91, 30, 90, 92, 91, 40])


def test_marker_inclusive_evidence_and_all_causal_edges():
    """Dropping boundary markers or an eligible earlier edge breaks exact LOO blocks."""
    layout = _layout(
        [11, 90, 92, 92, 91, 21, 22, 90, 92, 91, 31, 0],
        [11, -100, -100, -100, -100, 21, 22, -100, -100, -100, 31, -100],
    )

    assert layout.evidence_token_mask[0, 0].nonzero().flatten().tolist() == [1, 2, 3, 4]
    assert layout.window_token_mask[0, 1].nonzero().flatten().tolist() == [5, 6]
    assert layout.causal_pair_mask[0].nonzero().tolist() == [[0, 1], [0, 2], [1, 2]]


def test_cap_selects_nearest_predecessors_per_window_in_stable_order():
    """Selecting the oldest candidates under a cap loses the nearest evidence intervention."""
    edges = select_stage2_edges(_three_evidence_layout(), 2)

    assert [(edge.evidence_index, edge.window_index) for edge in edges] == [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
    ]


@pytest.mark.parametrize(
    ("ids", "error"),
    [
        ([90, 92, 7], "unmatched"),
        ([90, 92, 90, 92, 91, 91], "nested"),
    ],
)
def test_malformed_marker_boundaries_fail(ids, error):
    """Accepting malformed delimiter order makes evidence ownership ambiguous."""
    with pytest.raises(ValueError, match=error):
        _layout(ids)


def test_adjacent_spans_create_no_empty_window():
    """An empty inter-span run must not become a target window."""
    layout = _layout([90, 92, 91, 90, 92, 91, 7])

    assert layout.window_valid[0].tolist() == [True]
    assert layout.window_token_mask[0, 0].nonzero().flatten().tolist() == [6]


def test_evidence_without_a_latent_pad_is_structural_but_not_eligible():
    """A marker pair with no latent-pad interior cannot form an intervention edge."""
    layout = _layout([90, 7, 91, 10])

    assert layout.evidence_token_mask[0, 0].nonzero().flatten().tolist() == [0, 1, 2]
    assert layout.evidence_valid[0].tolist() == [False]
    assert not layout.causal_pair_mask.any()


def test_initial_window_without_predecessor_has_no_edges():
    """Crediting evidence to R0 would violate causal ordering."""
    layout = _layout([10, 90, 92, 91])

    assert layout.window_valid[0].tolist() == [True]
    assert not layout.causal_pair_mask.any()
    assert select_stage2_edges(layout) == []


def test_text_without_any_latent_span_has_no_edges():
    """Inventing evidence for a text-only sequence would create a nonexistent intervention."""
    layout = _layout([10, 11, 0], [10, 11, -100])

    assert layout.evidence_valid.shape == (1, 0)
    assert layout.window_token_mask[0, 0].nonzero().flatten().tolist() == [0, 1]
    assert select_stage2_edges(layout) == []


def test_unsupervised_text_run_is_not_a_window():
    """Treating label-masked text as a target window would score non-assistant tokens."""
    layout = _layout([90, 92, 91, 10], [-100, -100, -100, -100])

    assert layout.window_valid.shape == (1, 0)
    assert select_stage2_edges(layout) == []


def test_padding_and_special_tokens_are_excluded_from_windows():
    """Padding or special tokens in the score mask corrupts the window likelihood."""
    layout = _layout(
        [90, 92, 91, 20, 99, 21, 0],
        [-100, -100, -100, 20, 99, -100, -100],
    )

    assert layout.window_token_mask[0, 0].nonzero().flatten().tolist() == [3]


def test_heterogeneous_batch_pads_without_fake_edges():
    """Batch padding must not turn another row's absent evidence into a candidate."""
    ids = torch.tensor(
        [
            [90, 92, 91, 10, 90, 92, 91, 11, 0],
            [10, 90, 92, 91, 0, 0, 0, 0, 0],
        ]
    )
    labels = ids.clone()
    labels[0, -1] = -100
    labels[1, 4:] = -100
    layout = extract_stage2_layout(ids, ids.ne(0), labels, TOKENS)

    assert layout.evidence_valid.tolist() == [[True, True], [True, False]]
    assert layout.window_valid.tolist() == [[True, True], [True, False]]
    assert layout.causal_pair_mask[1].nonzero().tolist() == []


def test_plan_mismatch_reports_source_row_index():
    """A plan disagreement must identify the original sample, not only its packed row."""
    layout = _layout([10, 90, 92, 91, 20])

    with pytest.raises(ValueError, match="row 17"):
        validate_plan_types(layout, (("text", "latent", "text", "latent"),), [17])
