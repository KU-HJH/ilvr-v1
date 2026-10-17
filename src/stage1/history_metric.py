"""External, parameter-free cosine relevance over Stage1 history values."""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ContentHistory:
    weights: Tensor  # FP32 [M]
    summary: Tensor  # FP32 [1,H]
    present: bool


def _matrix(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f'{name} must be a Tensor')
    if not value.is_floating_point():
        raise TypeError(f'{name} must be floating point')
    if value.ndim != 2 or value.shape[1] == 0:
        raise ValueError(f'{name} must have matrix shape [N,H] with positive hidden width')
    value = value.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f'{name} must be finite in FP32')
    return value


@torch.no_grad()
def content_history(
    text: Tensor, values: Tensor, *, query_chunk_size: int = 64,
    key_chunk_size: int = 2048,
) -> ContentHistory:
    """Average text-row cosine probabilities and aggregate unnormalized values."""
    text = _matrix('text', text)
    values = _matrix('values', values)
    if text.shape[1] != values.shape[1] or text.device != values.device:
        raise ValueError('text and values must share hidden width and device')
    for name, size in (('query_chunk_size', query_chunk_size), ('key_chunk_size', key_chunk_size)):
        if type(size) is not int or size <= 0:
            raise ValueError(f'{name} must be a positive integer')
    weights = values.new_zeros(values.shape[0])
    summary = values.new_zeros((1, values.shape[1]))
    if text.shape[0] == 0 or values.shape[0] == 0:
        return ContentHistory(weights=weights, summary=summary, present=False)

    normalized = []
    for name, value in (('text', text), ('values', values)):
        norms = torch.linalg.vector_norm(value, dim=1, keepdim=True)
        if not bool(torch.isfinite(norms).all()) or bool((norms <= 1e-8).any()):
            raise FloatingPointError(f'{name} row norms must be finite and greater than 1e-8')
        normalized.append(value / norms)
    queries, keys = normalized

    # Cosines use temperature 1 and a uniform history-row prior. Each denominator
    # spans all key chunks; a separate chunk softmax would change that prior.
    with torch.autocast(device_type=text.device.type, enabled=False):
        for query_start in range(0, text.shape[0], query_chunk_size):
            query = queries[query_start:query_start + query_chunk_size]
            denominator = query.new_zeros((query.shape[0], 1))
            for key_start in range(0, values.shape[0], key_chunk_size):
                key = keys[key_start:key_start + key_chunk_size]
                denominator += torch.exp(query @ key.T).sum(dim=1, keepdim=True)
            for key_start in range(0, values.shape[0], key_chunk_size):
                key = keys[key_start:key_start + key_chunk_size]
                probabilities = torch.exp(query @ key.T) / denominator
                weights[key_start:key_start + key_chunk_size] += probabilities.sum(dim=0)
        weights /= text.shape[0]
        summary = (weights[:, None] * values).sum(dim=0, keepdim=True)
    if not bool(torch.isfinite(summary).all()):
        raise FloatingPointError('history summary must be finite in FP32')
    return ContentHistory(weights=weights, summary=summary, present=True)


@torch.no_grad()
def compose_content_query(
    global_intent: Tensor, text_summary: Tensor | None, history: ContentHistory,
) -> Tensor:
    """Mean intent, optional accumulated text, and present content history."""
    intent = _matrix('global intent', global_intent)
    if intent.shape[0] != 1:
        raise ValueError('global intent must have shape [1,H]')
    parts = [intent]
    components = []
    if text_summary is not None:
        components.append(('text summary', text_summary))
    if history.present:
        components.append(('history summary', history.summary))
    for name, value in components:
        value = _matrix(name, value)
        if value.shape != intent.shape or value.device != intent.device:
            raise ValueError(f'{name} must share global intent shape and device')
        parts.append(value)
    query = torch.stack(parts).mean(dim=0)
    norm = torch.linalg.vector_norm(query)
    if not bool(torch.isfinite(norm)) or bool(norm <= 1e-8):
        raise FloatingPointError('content query norm must be finite and greater than 1e-8')
    return query
