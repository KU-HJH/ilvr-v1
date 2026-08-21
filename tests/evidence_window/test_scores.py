import math

import pytest
import torch
import torch.nn.functional as F

import evidence_window.scores as score_primitives

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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_chunked_fp32_target_logps_match_full_fp32_values_and_gradients(dtype):
    logits = torch.linspace(-2.0, 3.0, steps=2 * 6 * 7).reshape(2, 6, 7).to(dtype)
    labels = torch.tensor(
        [
            [-100, 1, -100, 3, 0, 6],
            [-100, 2, 4, -100, 5, 1],
        ]
    )
    reference_logits = logits.clone().requires_grad_()
    chunked_logits = logits.clone().requires_grad_()

    expected = causal_token_logps(reference_logits.float(), labels)[:, 1:]
    actual = score_primitives.chunked_causal_target_logps(
        chunked_logits,
        labels,
        token_chunk_size=2,
    )

    assert actual.shape == (2, 5)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert torch.equal(actual[labels[:, 1:] == -100], torch.zeros(2))

    weights = torch.linspace(0.25, 2.5, steps=actual.numel()).reshape_as(actual)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    torch.testing.assert_close(
        chunked_logits.grad,
        reference_logits.grad,
        atol=0.0,
        rtol=0.0,
    )
    ignored_prediction_rows = labels[:, 1:] == -100
    assert torch.equal(
        chunked_logits.grad[:, :-1][ignored_prediction_rows],
        torch.zeros_like(chunked_logits.grad[:, :-1][ignored_prediction_rows]),
    )


def test_chunked_fp32_target_logps_bounds_casts_and_recomputes_backward(monkeypatch):
    batch_size, sequence_length, vocab_size = 2, 9, 11
    token_chunk_size = 3
    logits = torch.linspace(
        -3.0,
        4.0,
        steps=batch_size * sequence_length * vocab_size,
        dtype=torch.bfloat16,
    ).reshape(batch_size, sequence_length, vocab_size)
    logits.requires_grad_()
    labels = torch.tensor(
        [
            [-100, 1, 2, 3, -100, 5, 6, 7, 8],
            [-100, 9, 8, -100, 6, 5, 4, 3, 2],
        ]
    )
    cast_shapes = []
    log_softmax_inputs = []
    original_float = torch.Tensor.float
    original_log_softmax = F.log_softmax

    def traced_float(tensor, *args, **kwargs):
        result = original_float(tensor, *args, **kwargs)
        if tensor.ndim == 3:
            cast_shapes.append((tuple(tensor.shape), result.dtype))
        return result

    def traced_log_softmax(tensor, *args, **kwargs):
        if tensor.ndim == 3:
            log_softmax_inputs.append((tuple(tensor.shape), tensor.dtype))
        return original_log_softmax(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "float", traced_float)
    monkeypatch.setattr(F, "log_softmax", traced_log_softmax)

    scores = score_primitives.chunked_causal_target_logps(
        logits,
        labels,
        token_chunk_size=token_chunk_size,
    )
    expected_chunks = math.ceil((sequence_length - 1) / token_chunk_size)

    assert scores.shape == (batch_size, sequence_length - 1)
    assert scores.dtype == torch.float32
    assert len(log_softmax_inputs) == expected_chunks
    assert len(cast_shapes) == expected_chunks
    for shape, dtype in (*cast_shapes, *log_softmax_inputs):
        assert shape[0] == batch_size
        assert shape[1] <= token_chunk_size
        assert shape[-1] == vocab_size
        assert dtype == torch.float32

    scores.sum().backward()

    assert len(log_softmax_inputs) == 2 * expected_chunks
    assert len(cast_shapes) == 2 * expected_chunks
    for shape, dtype in (*cast_shapes, *log_softmax_inputs):
        assert shape[1] <= token_chunk_size
        assert dtype == torch.float32
