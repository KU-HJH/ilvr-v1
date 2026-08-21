import pytest
import torch

from evidence_window.layout import TokenIds, extract_interleaved_layout


TOKENS = TokenIds(90, 91, 92, 0, frozenset({0, 90, 91, 99}))


def test_extracts_r0_v1_r1_v2_r2_v3_r3():
    """Removing active evidence or its native predecessor breaks these masks."""
    ids = torch.tensor([[10, 90, 92, 92, 91, 20, 21, 90, 92, 91, 30, 90, 92, 91, 40, 0]])
    labels = ids.clone()
    labels[:, -1] = -100
    active = torch.tensor(
        [[0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.bool
    )
    layout = extract_interleaved_layout(ids, ids.ne(0), labels, active, TOKENS)

    assert layout.evidence_token_mask[0].nonzero().tolist() == [[0, 2], [0, 3], [1, 8], [2, 12]]
    assert layout.window_token_mask[0].nonzero().tolist() == [[0, 0], [1, 5], [1, 6], [2, 10], [3, 14]]
    assert layout.native_predecessor[0].tolist() == [-1, 0, 1, 2]
    assert layout.training_pair_mask[0].nonzero().tolist() == [[0, 1], [1, 2], [2, 3]]
    assert layout.causal_pair_mask[0].nonzero().tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]


def test_padding_ignored_and_special_tokens_are_excluded():
    """Changing supervision, padding, or special exclusion must change the window."""
    ids = torch.tensor([[90, 92, 91, 20, 99, 21, 0]])
    labels = torch.tensor([[-100, -100, -100, 20, 99, -100, -100]])
    layout = extract_interleaved_layout(ids, ids.ne(0), labels, ids.eq(92), TOKENS)

    assert layout.window_token_mask[0, 0].nonzero().flatten().tolist() == [3]


def test_adjacent_spans_create_no_empty_window():
    """Creating a window for delimiter-adjacent spans is an invalid layout."""
    ids = torch.tensor([[90, 92, 91, 90, 92, 91, 7]])
    layout = extract_interleaved_layout(ids, torch.ones_like(ids), ids, ids.eq(92), TOKENS)

    assert layout.window_valid.sum().item() == 1
    assert layout.native_predecessor[0, 0].item() == 1


def test_unmatched_delimiter_fails():
    """Accepting malformed latent boundaries would assign evidence ambiguously."""
    ids = torch.tensor([[90, 92, 7]])
    with pytest.raises(ValueError, match="unmatched"):
        extract_interleaved_layout(ids, torch.ones_like(ids), ids, ids.eq(92), TOKENS)


def test_first_and_last_span_leave_only_eligible_outer_windows():
    """An outer text run must be retained only when it contains supervised reasoning."""
    ids = torch.tensor([[10, 90, 92, 91, 20, 90, 92, 91]])
    labels = torch.tensor([[-100, -100, -100, -100, 20, -100, -100, -100]])
    layout = extract_interleaved_layout(ids, torch.ones_like(ids), labels, ids.eq(92), TOKENS)

    assert layout.window_token_mask[0].nonzero().tolist() == [[0, 4]]
    assert layout.native_predecessor[0].tolist() == [0]


def test_inactive_span_stays_indexed_but_cannot_form_pairs():
    """Making inactive spans pairable would credit an unselected latent span."""
    ids = torch.tensor([[90, 92, 91, 10, 90, 92, 91, 11]])
    active = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    layout = extract_interleaved_layout(ids, torch.ones_like(ids), ids, active, TOKENS)

    assert layout.evidence_valid[0].tolist() == [True, False]
    assert layout.native_predecessor[0].tolist() == [0, 1]
    assert layout.causal_pair_mask[0].nonzero().tolist() == [[0, 0], [0, 1]]
    assert layout.training_pair_mask[0].nonzero().tolist() == [[0, 0]]


def test_r0_has_no_predecessor_and_no_pairs():
    """Assigning a predecessor to R0 would create a non-native training edge."""
    ids = torch.tensor([[10, 90, 92, 91]])
    layout = extract_interleaved_layout(ids, torch.ones_like(ids), ids, ids.eq(92), TOKENS)

    assert layout.window_valid[0].tolist() == [True]
    assert layout.native_predecessor[0].tolist() == [-1]
    assert not layout.causal_pair_mask.any()
    assert not layout.training_pair_mask.any()


def test_two_examples_with_different_counts_are_padded_without_fake_pairs():
    """Packing a heterogeneous batch must not turn padded entries into valid pairs."""
    ids = torch.tensor(
        [
            [90, 92, 91, 10, 90, 92, 91, 11, 0],
            [10, 90, 92, 91, 0, 0, 0, 0, 0],
        ]
    )
    labels = ids.clone()
    labels[0, -1] = -100
    labels[1, 4:] = -100
    layout = extract_interleaved_layout(ids, ids.ne(0), labels, ids.eq(92), TOKENS)

    assert layout.evidence_valid.tolist() == [[True, True], [True, False]]
    assert layout.window_valid.tolist() == [[True, True], [True, False]]
    assert layout.native_predecessor.tolist() == [[0, 1], [-1, -1]]
    assert layout.causal_pair_mask[1].nonzero().tolist() == []
    assert layout.training_pair_mask[1].nonzero().tolist() == []


def test_nested_delimiter_fails():
    """Nested starts must fail rather than merge two latent spans."""
    ids = torch.tensor([[90, 92, 90, 92, 91, 91]])
    with pytest.raises(ValueError, match="nested"):
        extract_interleaved_layout(ids, torch.ones_like(ids), ids, ids.eq(92), TOKENS)


def test_layout_extraction_avoids_scalar_tensor_reads(monkeypatch):
    """Scalar reads would synchronize every inspected CUDA token to the host."""
    ids = torch.tensor([[10, 90, 92, 91, 20, 21, 90, 92, 91, 30]])

    def scalar_read_is_forbidden(*_args, **_kwargs):
        raise AssertionError("layout extraction performed a scalar tensor read")

    monkeypatch.setattr(torch.Tensor, "item", scalar_read_is_forbidden)
    layout = extract_interleaved_layout(ids, torch.ones_like(ids), ids, ids.eq(92), TOKENS)

    assert layout.evidence_token_mask.nonzero().tolist() == [[0, 0, 2], [0, 1, 7]]
    assert layout.window_token_mask.nonzero().tolist() == [[0, 0, 0], [0, 1, 4], [0, 1, 5], [0, 2, 9]]
