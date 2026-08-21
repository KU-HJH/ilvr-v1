import json

import pytest
import torch

from evidence_window.metrics import (
    MatrixJsonlWriter,
    contribution_summaries,
    global_outer_coefficient,
)


def test_positive_mass_summaries_ignore_negative_relations():
    """A negative relation must not reduce descriptive positive-mass shares."""
    relations = torch.tensor([[[3.0, 0.0], [1.0, -2.0]]])
    pairs = torch.ones_like(relations, dtype=torch.bool)

    out = contribution_summaries(relations, pairs, threshold=0.5)

    assert torch.allclose(out.top1_ratio[0, 0], torch.tensor(0.75))
    assert out.above_threshold_count[0, 0] == 2
    assert out.entropy_valid[0, 0]
    assert not out.entropy_valid[0, 1]


def test_entropy_requires_positive_mass_and_multiple_valid_candidates():
    """Empty positive mass and one-candidate windows cannot yield entropy."""
    relations = torch.tensor([[[0.0, 3.0], [-1.0, 0.0]]])
    pairs = torch.tensor([[[True, True], [True, False]]])

    out = contribution_summaries(relations, pairs, threshold=0.0)

    assert not out.entropy_valid[0, 0]
    assert not out.entropy_valid[0, 1]
    assert out.normalized_entropy[0, 0] == 0
    assert out.normalized_entropy[0, 1] == 0


def test_global_outer_coefficient_uses_exact_sigmoid_formula():
    """The saturation diagnostic must retain the beta multiplier."""
    margin = torch.tensor([0.0, 2.0])

    out = global_outer_coefficient(margin, beta=0.5)

    assert torch.allclose(out, 0.5 * torch.sigmoid(-0.5 * margin))


def test_rank_local_jsonl_contains_no_sample_payload(tmp_path):
    """Matrix dumps are limited to numeric matrices and an opaque identifier."""
    writer = MatrixJsonlWriter(tmp_path, rank=2)
    writer.write(7, 3, torch.tensor([[1.0]]), torch.tensor([[True]]))

    record = json.loads((tmp_path / "rank-00002.jsonl").read_text())
    assert record["observation_id"] == "r2-s7-o3"
    assert set(record) == {"observation_id", "rank", "step", "relation", "pair_mask"}
    assert record["relation"] == [[1.0]]
    assert record["pair_mask"] == [[True]]


def test_matrix_writer_rejects_non_numeric_or_misaligned_matrix_inputs(tmp_path):
    """Writer input validation blocks arbitrary payloads from reaching JSONL."""
    writer = MatrixJsonlWriter(tmp_path, rank=0)

    with pytest.raises(ValueError):
        writer.write(0, 0, torch.tensor([[1.0]]), torch.tensor([True]))
    with pytest.raises(ValueError):
        writer.write(0, 0, torch.tensor([[float("nan")]]), torch.tensor([[True]]))
