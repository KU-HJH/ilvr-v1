"""Requests and observed attention rows; observation uses an existing forward."""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ContextManager
import math

import torch
from torch import Tensor, nn
from .stage1_prefix_conditioning import QueryKey


@dataclass(frozen=True)
class TraceRequest:
    query_keys: tuple[QueryKey, ...]
    key_positions: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class AttentionRows:
    request: TraceRequest
    values: tuple[Tensor, ...]


class PrefixTraceObserver:
    """Collect compact, detached FP32 rows from the first gain-clean pass.

    Inputs use [batch, repeated heads, sequence, head_dim]. Query positions
    index this full clean sequence; callers must not use an incremental decode.
    Each chunk normalizes over the complete key axis before candidate gathering.
    """

    def __init__(self, request: TraceRequest, *, layer_count: int, chunk_rows: int):
        if layer_count <= 0 or chunk_rows <= 0:
            raise ValueError('layer_count and chunk_rows must be positive')
        if len(request.query_keys) != len(request.key_positions):
            raise ValueError('trace queries and key positions must align')
        self.request = request
        self.layer_count = layer_count
        self.chunk_rows = chunk_rows
        self._seen = set()
        self._values = None
        self._closed = False
        self._bound = False

    @torch.no_grad()
    def observe(
        self, layer_idx: int, query: Tensor, key: Tensor, attention_mask: Tensor | None,
        *, is_causal: bool, scale: float,
    ) -> None:
        if self._closed:
            return
        if layer_idx in self._seen:
            raise ValueError('attention layer already observed')
        if layer_idx not in range(self.layer_count):
            raise ValueError('attention layer index out of range')
        if (query.ndim != 4 or key.ndim != 4 or query.shape[:2] != key.shape[:2]
                or query.shape[-1] != key.shape[-1]):
            raise ValueError('expected Q/K with matching repeated heads and head dimensions')
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('attention scale must be finite and positive')
        for (batch, position), keys in zip(self.request.query_keys, self.request.key_positions):
            if not (0 <= batch < query.shape[0] and 0 <= position < query.shape[2]):
                raise ValueError('trace query outside clean sequence')
            if any(k < 0 or k >= key.shape[2] for k in keys):
                raise ValueError('trace key outside clean sequence')
        if attention_mask is not None:
            if attention_mask.ndim != 4 or not attention_mask.is_floating_point():
                raise ValueError('expected a four-dimensional additive attention mask')
            if (attention_mask.shape[0] not in (1, query.shape[0])
                    or attention_mask.shape[1] not in (1, query.shape[1])
                    or attention_mask.shape[2] not in (1, query.shape[2])
                    or attention_mask.shape[3] < key.shape[2]):
                raise ValueError('attention mask does not match Q/K')
        # Explicit FP32 operands alone do not prevent autocast matmul downcasting.
        with torch.autocast(device_type=query.device.type, enabled=False):
            values = []
            for start in range(0, len(self.request.query_keys), self.chunk_rows):
                values.extend(self._chunk(query.detach(), key.detach(), attention_mask,
                                          start, is_causal, scale))
            if self._values is None:
                self._values = values
            else:
                for total, value in zip(self._values, values):
                    total.add_(value)
        self._seen.add(layer_idx)

    def _chunk(self, query, key, mask, start, is_causal, scale):
        stop = min(start + self.chunk_rows, len(self.request.query_keys))
        rows = []
        for batch, position in self.request.query_keys[start:stop]:
            scores = torch.matmul(query[batch, :, position:position + 1].float(),
                                  key[batch].float().transpose(-2, -1)).squeeze(-2) * scale
            if mask is not None:
                bias = mask[batch if mask.shape[0] > 1 else 0, :,
                            position if mask.shape[2] > 1 else 0, :key.shape[2]].detach()
                # Both -inf and the dtype minimum are standard additive mask sentinels.
                blocked = torch.isneginf(bias) | (bias == torch.finfo(bias.dtype).min)
                scores = (scores + bias.float()).masked_fill(blocked, float('-inf'))
            if is_causal:
                scores.masked_fill_(torch.arange(key.shape[2], device=key.device) > position,
                                    float('-inf'))
            if not torch.isfinite(scores).any(-1).all():
                raise ValueError('attention row has no finite visible key')
            if torch.isnan(scores).any() or torch.isposinf(scores).any():
                raise ValueError('attention row contains non-finite scores')
            rows.append(scores)
        probabilities = torch.softmax(torch.stack(rows), dim=-1)
        means = probabilities.mean(dim=1)
        return [means[i, list(keys)].clone() for i, keys in
                enumerate(self.request.key_positions[start:stop])]

    def finish(self) -> AttentionRows:
        if len(self._seen) != self.layer_count:
            raise ValueError('incomplete attention layer collection')
        self.close()
        return AttentionRows(self.request, tuple(value / self.layer_count for value in self._values))

    def close(self) -> None:
        """Make stale callbacks inert, including checkpoint recomputation."""
        self._closed = True


@contextmanager
def bind_prefix_observer(decoder: nn.Module, observer: PrefixTraceObserver) -> ContextManager[None]:
    """Bind decoder.layers[*].self_attn only; always remove and close on exit.

    The caller scopes this around gain-clean only, before reference or replay.
    Nested binding is rejected so one clean pass cannot feed two collectors.
    """
    if observer._bound:
        raise ValueError('prefix observer already bound')
    bound = []
    missing = object()
    try:
        attentions = [layer.self_attn for layer in decoder.layers]
        if len(attentions) != observer.layer_count:
            raise ValueError('decoder layer count does not match observer')
        for idx, attention in enumerate(attentions):
            if attention.config._attn_implementation not in ('eager', 'sdpa'):
                raise ValueError('prefix observer requires eager or sdpa backend')
            if attention.layer_idx != idx:
                raise ValueError('decoder attention layer index mismatch')
            if getattr(attention, '_prefix_trace_observer', None) is not None:
                raise ValueError('prefix observer already bound')
        if observer._closed:
            raise ValueError('prefix observer already closed')
        for attention in attentions:
            previous = getattr(attention, '_prefix_trace_observer', missing)
            bound.append((attention, previous))
            attention._prefix_trace_observer = observer
        observer._bound = True
        yield
    finally:
        for attention, previous in reversed(bound):
            if previous is missing:
                del attention._prefix_trace_observer
            else:
                attention._prefix_trace_observer = previous
        observer._bound = False
        observer.close()
