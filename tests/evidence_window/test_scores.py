import pytest
import torch

from evidence_window.scores import (
    causal_token_logps,
    evidence_window_margins,
    masked_token_sums,
    reference_adjusted_effect,
)


def test_t_minus_one_predicts_t():
    logits = torch.tensor([[[0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]])
    labels = torch.tensor([[-100, 1, 0]])
    scores = causal_token_logps(logits, labels)
    logp = logits.log_softmax(-1)
    assert torch.allclose(scores[0, 1], logp[0, 0, 1])
    assert torch.allclose(scores[0, 2], logp[0, 1, 0])
    assert scores[0, 0] == 0


def test_window_sum_excludes_every_outside_token():
    scores = torch.tensor([[1.0, 2.0, 4.0, 8.0, 16.0]])
    masks = torch.tensor(
        [[[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]]], dtype=torch.bool
    )
    assert masked_token_sums(scores, masks).tolist() == [[6.0, 8.0]]


def test_reference_adjustment_is_full_minus_loo_policy_delta_minus_reference_delta():
    policy_full = torch.tensor([[4.0, 9.0]])
    policy_cf = torch.tensor([[1.0, 2.0]])
    ref_full = torch.tensor([[2.0, 7.0]])
    ref_cf = torch.tensor([[1.0, 3.0]])
    assert torch.equal(
        reference_adjusted_effect(policy_full, policy_cf, ref_full, ref_cf),
        torch.tensor([[2.0, 3.0]]),
    )


def test_evidence_window_margins_directly_sum_window_tokens():
    delta = torch.tensor([[[0.0, 3.0, 5.0, 7.0, 11.0]]])
    masks = torch.tensor(
        [[[[0, 1, 1, 0, 0], [0, 0, 0, 1, 1]]]], dtype=torch.bool
    )
    assert evidence_window_margins(delta, masks).tolist() == [[[8.0, 18.0]]]


def test_invalid_score_shapes_raise_value_error():
    with pytest.raises(ValueError):
        causal_token_logps(torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.long))
    with pytest.raises(ValueError):
        masked_token_sums(torch.zeros(2, 3), torch.zeros(1, 2, 3, dtype=torch.bool))


def test_empty_causal_sequence_preserves_sequence_shape():
    logits = torch.empty((2, 0, 4))
    labels = torch.empty((2, 0), dtype=torch.long)
    assert causal_token_logps(logits, labels).shape == (2, 0)
