"""Independent arithmetic and boundary checks for external history relevance."""
import importlib
from dataclasses import FrozenInstanceError

import pytest
import torch


def api():
    return importlib.import_module('stage1.history_metric')


def test_literal_cosine_logits_weight_raw_values_without_gradients():
    text = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], requires_grad=True)
    values = torch.tensor([[4., 0.], [0., 1.]], requires_grad=True)
    expected_weights = torch.softmax(
        torch.tensor([[1., 0.], [1., 0.], [0., 1.]]), dim=1).mean(0)
    expected_summary = (expected_weights[:, None] * values.detach()).sum(0, keepdim=True)

    history = api().content_history(text, values, query_chunk_size=1, key_chunk_size=1)

    torch.testing.assert_close(history.weights, expected_weights)
    torch.testing.assert_close(history.summary, expected_summary)
    assert history.weights.dtype == history.summary.dtype == torch.float32
    assert history.present is True
    assert not history.weights.requires_grad and not history.summary.requires_grad
    assert text.grad is None and values.grad is None


@pytest.mark.parametrize(('query_chunk_size', 'key_chunk_size'), [(1, 1), (2, 2), (64, 2048)])
def test_chunked_result_matches_independent_full_matrix_softmax(query_chunk_size, key_chunk_size):
    text = torch.tensor([[1., 2.], [-3., 1.], [2., -1.]])
    values = torch.tensor([[3., 1.], [-1., 2.], [4., -1.], [2., 5.], [-2., -3.]])
    normalized_text = torch.nn.functional.normalize(text, dim=1)
    normalized_values = torch.nn.functional.normalize(values, dim=1)
    expected_weights = torch.softmax(normalized_text @ normalized_values.T, dim=1).mean(0)
    expected_summary = expected_weights.unsqueeze(0) @ values

    history = api().content_history(
        text, values, query_chunk_size=query_chunk_size, key_chunk_size=key_chunk_size)

    torch.testing.assert_close(history.weights, expected_weights)
    torch.testing.assert_close(history.summary, expected_summary)
    torch.testing.assert_close(history.weights.sum(), torch.tensor(1.))


@pytest.mark.parametrize(('text_rows', 'history_rows'), [(0, 2), (2, 0), (0, 0)])
def test_empty_inputs_have_zero_absent_history_and_keep_prior_text(text_rows, history_rows):
    metric = api()
    history = metric.content_history(torch.zeros(text_rows, 2), torch.zeros(history_rows, 2))

    assert history.present is False
    torch.testing.assert_close(history.weights, torch.zeros(history_rows))
    torch.testing.assert_close(history.summary, torch.zeros(1, 2))
    assert history.weights.dtype == history.summary.dtype == torch.float32
    intent, prior_text = torch.tensor([[1., 0.]]), torch.tensor([[0., 2.]])
    torch.testing.assert_close(
        metric.compose_content_query(intent, prior_text, history), (intent + prior_text) / 2)


@pytest.mark.parametrize('side', ['text', 'values'])
@pytest.mark.parametrize('bad_row', [
    [0., 0.], [1e-12, 0.], [float('nan'), 0.], [float('inf'), 0.], [3e38, 3e38],
])
def test_nonfinite_or_degenerate_scoring_rows_fail_explicitly(side, bad_row):
    inputs = {'text': torch.tensor([[1., 0.]]), 'values': torch.tensor([[0., 1.]])}
    inputs[side] = torch.tensor([bad_row])
    with pytest.raises(FloatingPointError, match='norm|finite|FP32'):
        api().content_history(**inputs)


@pytest.mark.parametrize('side', ['text', 'values'])
@pytest.mark.parametrize('bad_matrix', [torch.ones(2), torch.ones(1, 1, 2), torch.empty(1, 0)])
def test_malformed_matrices_are_rejected(side, bad_matrix):
    inputs = {'text': torch.ones(1, 2), 'values': torch.ones(1, 2)}
    inputs[side] = bad_matrix
    with pytest.raises(ValueError, match='shape|width|matrix'):
        api().content_history(**inputs)


def test_scoring_hidden_widths_must_match():
    with pytest.raises(ValueError, match='width|shape'):
        api().content_history(torch.ones(1, 2), torch.ones(1, 3))


@pytest.mark.parametrize('side', ['text', 'values'])
@pytest.mark.parametrize('bad_matrix', [[[1., 0.]], torch.ones(1, 2, dtype=torch.int64)])
def test_scoring_requires_floating_tensors(side, bad_matrix):
    inputs = {'text': torch.ones(1, 2), 'values': torch.ones(1, 2)}
    inputs[side] = bad_matrix
    with pytest.raises(TypeError, match='Tensor|floating'):
        api().content_history(**inputs)


@pytest.mark.parametrize('chunk_name', ['query_chunk_size', 'key_chunk_size'])
@pytest.mark.parametrize('bad_chunk', [0, True, 1.5])
def test_chunks_require_exact_positive_integers(chunk_name, bad_chunk):
    with pytest.raises(ValueError, match='positive integer'):
        api().content_history(torch.ones(1, 2), torch.ones(1, 2), **{chunk_name: bad_chunk})


@pytest.mark.parametrize('side', ['text', 'values'])
def test_nonfinite_fp32_conversion_is_rejected_before_empty_shortcut(side):
    inputs = {'text': torch.empty(0, 2), 'values': torch.empty(0, 2)}
    inputs[side] = torch.tensor([[1e100, 0.]], dtype=torch.float64)
    with pytest.raises(FloatingPointError, match='finite|FP32'):
        api().content_history(**inputs)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float64])
def test_floating_inputs_produce_fp32_outputs(dtype):
    history = api().content_history(torch.tensor([[1., 0.]], dtype=dtype), torch.eye(2, dtype=dtype))
    assert history.weights.dtype == history.summary.dtype == torch.float32


def test_scoring_remains_fp32_inside_callers_autocast():
    text = torch.tensor([[1., 2.], [-3., 1.], [2., -1.]])
    values = torch.tensor([[3., 1.], [-1., 2.], [4., -1.], [2., 5.], [-2., -3.]])
    logits = torch.nn.functional.normalize(text, dim=1) @ torch.nn.functional.normalize(values, dim=1).T
    expected_weights = torch.softmax(logits, dim=1).mean(0)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        history = api().content_history(text, values, query_chunk_size=2, key_chunk_size=2)
    torch.testing.assert_close(history.weights, expected_weights)
    torch.testing.assert_close(history.summary, expected_weights.unsqueeze(0) @ values)


def test_composition_means_available_components_and_detaches_them():
    metric = api()
    intent = torch.tensor([[1., 2.]], requires_grad=True)
    text = torch.tensor([[2., 3.]], requires_grad=True)
    summary = torch.tensor([[3., 7.]], requires_grad=True)
    history = metric.ContentHistory(torch.tensor([1.]), summary, True)

    query = metric.compose_content_query(intent, text, history)

    torch.testing.assert_close(query, torch.tensor([[2., 4.]]))
    torch.testing.assert_close(
        metric.compose_content_query(intent, None, history), torch.tensor([[2., 4.5]]))
    absent = metric.ContentHistory(torch.zeros(0), torch.zeros(1, 2), False)
    torch.testing.assert_close(metric.compose_content_query(intent, None, absent), intent.detach())
    assert query.dtype == torch.float32 and not query.requires_grad
    assert intent.grad is None and text.grad is None and summary.grad is None
    with pytest.raises(FrozenInstanceError):
        history.present = False


def test_present_history_requires_a_tensor_summary():
    metric = api()
    malformed = metric.ContentHistory(torch.ones(1), None, True)
    with pytest.raises(TypeError, match='Tensor'):
        metric.compose_content_query(torch.ones(1, 2), None, malformed)


@pytest.mark.parametrize('component', ['intent', 'text', 'history'])
@pytest.mark.parametrize('bad_matrix', [torch.ones(2), torch.ones(2, 2), torch.ones(1, 3), torch.empty(1, 0)])
def test_composition_requires_matching_single_rows(component, bad_matrix):
    metric = api()
    parts = {'intent': torch.ones(1, 2), 'text': torch.ones(1, 2), 'history': torch.ones(1, 2)}
    parts[component] = bad_matrix
    history = metric.ContentHistory(torch.ones(1), parts['history'], True)
    with pytest.raises(ValueError, match='shape|width|matrix'):
        metric.compose_content_query(parts['intent'], parts['text'], history)


@pytest.mark.parametrize('component', ['intent', 'text', 'history'])
@pytest.mark.parametrize('bad_matrix', [[[1., 0.]], torch.ones(1, 2, dtype=torch.int64)])
def test_composition_requires_floating_tensors(component, bad_matrix):
    metric = api()
    parts = {'intent': torch.ones(1, 2), 'text': torch.ones(1, 2), 'history': torch.ones(1, 2)}
    parts[component] = bad_matrix
    history = metric.ContentHistory(torch.ones(1), parts['history'], True)
    with pytest.raises(TypeError, match='Tensor|floating'):
        metric.compose_content_query(parts['intent'], parts['text'], history)


@pytest.mark.parametrize('component', ['intent', 'text', 'history'])
@pytest.mark.parametrize('bad_value', [float('nan'), float('inf'), 1e100])
def test_composition_rejects_nonfinite_fp32_components(component, bad_value):
    metric = api()
    parts = {'intent': torch.ones(1, 2), 'text': torch.ones(1, 2), 'history': torch.ones(1, 2)}
    parts[component] = torch.tensor([[bad_value, 0.]], dtype=torch.float64)
    history = metric.ContentHistory(torch.ones(1), parts['history'], True)
    with pytest.raises(FloatingPointError, match='finite|FP32'):
        metric.compose_content_query(parts['intent'], parts['text'], history)


@pytest.mark.parametrize('intent', [torch.zeros(1, 2), torch.tensor([[1e-12, 0.]]), torch.tensor([[3e38, 3e38]])])
def test_composition_rejects_degenerate_or_overflowed_final_norm(intent):
    metric = api()
    absent = metric.ContentHistory(torch.zeros(0), torch.zeros(1, 2), False)
    with pytest.raises(FloatingPointError, match='norm'):
        metric.compose_content_query(intent, None, absent)


def test_composition_rejects_zero_from_cancelling_components():
    metric = api()
    history = metric.ContentHistory(torch.ones(1), torch.tensor([[-1., 0.]]), True)
    with pytest.raises(FloatingPointError, match='norm'):
        metric.compose_content_query(torch.tensor([[1., 0.]]), None, history)


def test_content_relevance_changes_actual_selector_with_fixed_other_inputs():
    from stage1.targets import Stage1TeacherTargetBuilder
    from tests.stage1.history_fixtures import make_history_case

    metric = api()
    builder = Stage1TeacherTargetBuilder(
        make_history_case().tokenizer, coverage_p=.1, helper_group_L=0, collect_diagnostics=True)
    text = torch.tensor([[1., 0.]])
    values = torch.tensor([[4., 0.], [0., 4.]])
    intent = torch.tensor([[0., 1.5]])
    candidates = torch.eye(2)
    history = metric.content_history(text, values)
    uniform = metric.ContentHistory(torch.full((2,), .5), values.mean(0, keepdim=True), True)
    metric_query = metric.compose_content_query(intent, text, history)
    uniform_query = metric.compose_content_query(intent, text, uniform)

    _, metric_indices, metric_margin = builder._select(candidates, metric_query, max_count=1)
    _, uniform_indices, uniform_margin = builder._select(candidates, uniform_query, max_count=1)

    assert metric_indices == (0,)
    assert uniform_indices == (1,)
    assert metric_margin > 0 and uniform_margin > 0
