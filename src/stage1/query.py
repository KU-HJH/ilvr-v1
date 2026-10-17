"""Configuration, text layout and mass-preserving Stage1 history queries."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor

from .types import Stage1LatentSpan


_CLI_FIELDS = {
    'query_mode': 'mode',
    'history_text_gamma': 'text_gamma',
    'history_attention_layers': 'attention_layers',
}
_CHUNKS = ('query_chunk_size', 'key_chunk_size', 'head_chunk_size')
_LEGACY_FORMULA = 'stage1_legacy_query_v1'
_HISTORICAL_ADAPTIVE_FORMULA = 'stage1_injected_prefix_discounted_segments_attention_mass_v1'
_ADAPTIVE_FORMULA = 'stage1_placeholder_discounted_segments_cosine_relative_v2'


def _gamma(value: object) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('text gamma must be finite and in [0, 1]')
    return float(value)


def _positive_int(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class Stage1QueryConfig:
    mode: str = 'legacy'
    text_gamma: float = .9
    attention_layers: str = 'last'
    query_chunk_size: int = 64
    key_chunk_size: int = 2048
    head_chunk_size: int = 4

    def __post_init__(self) -> None:
        if self.mode not in ('legacy', 'adaptive'):
            raise ValueError('query mode must be legacy or adaptive')
        _gamma(self.text_gamma)
        if self.attention_layers not in ('last', 'last4', 'all'):
            raise ValueError('attention layers must be last, last4 or all')
        for name in _CHUNKS:
            _positive_int(name, getattr(self, name))


def _validate_layers(indices: object) -> None:
    if (not isinstance(indices, (tuple, list)) or not indices
            or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('resolved layer indices must be nonempty unique nonnegative integers')


def resolve_query_layers(selector: str, layer_count: int) -> tuple[int, ...]:
    """Return zero-based decoder layers; last4 requires at least four layers."""
    _positive_int('layer_count', layer_count)
    if selector == 'last':
        return (layer_count - 1,)
    if selector == 'last4':
        if layer_count < 4:
            raise ValueError('last4 requires at least four actual decoder layers')
        return tuple(range(layer_count - 4, layer_count))
    if selector == 'all':
        return tuple(range(layer_count))
    raise ValueError('attention layers must be last, last4 or all')


def _historical_query_execution_config(
    config: Stage1QueryConfig, resolved_layers: tuple[int, ...],
) -> dict[str, object]:
    """Serialize the formula and execution semantics used by this query."""
    result = {'schema_version': 1, 'mode': config.mode,
              'formula_id': _LEGACY_FORMULA if config.mode == 'legacy' else _HISTORICAL_ADAPTIVE_FORMULA}
    if config.mode == 'adaptive':
        _validate_layers(resolved_layers)
        if config.attention_layers == 'last4' and len(resolved_layers) != 4:
            raise ValueError('last4 requires exactly four resolved decoder layers')
        result.update({
            'text_mode': 'discounted_segments', 'latent_mode': 'attention_mass',
            'text_gamma': config.text_gamma, 'attention_layers': config.attention_layers,
            'resolved_layer_indices': list(resolved_layers),
            **{name: getattr(config, name) for name in _CHUNKS},
            'mask_mode': 'full_causal_2d_v1',
        })
    return result


def validate_query_training_config(config: Stage1QueryConfig) -> None:
    """Reject utility-only historical selectors before any training model work."""
    if not isinstance(config, Stage1QueryConfig):
        raise ValueError('query training config must be a Stage1QueryConfig')
    if config.mode == 'adaptive':
        if config.attention_layers != 'last':
            raise ValueError('adaptive training uses final-normalized hidden states (last only)')
        if config.head_chunk_size != 4:
            raise ValueError('adaptive training does not support a head_chunk_size control')


def query_execution_config(
    config: Stage1QueryConfig, resolved_layers: tuple[int, ...],
) -> dict[str, object]:
    """Serialize current training semantics; layers remain a compatibility argument."""
    validate_query_training_config(config)
    if config.mode == 'legacy':
        return _historical_query_execution_config(config, ())
    return {
        'schema_version': 2, 'mode': 'adaptive', 'formula_id': _ADAPTIVE_FORMULA,
        'teacher_context': 'ilvr_placeholder_full_sequence_v1',
        'mask_mode': 'native_legacy_v1', 'text_mode': 'discounted_segments',
        'text_gamma': config.text_gamma, 'latent_mode': 'external_cosine_relative',
        'hidden_state': 'final_normalized', 'temperature': 1.0,
        'query_chunk_size': config.query_chunk_size, 'key_chunk_size': config.key_chunk_size,
    }


def validate_query_execution_config(saved: Mapping[str, object]) -> Stage1QueryConfig:
    """Strictly read current and historical artifacts without authorizing training."""
    if not isinstance(saved, Mapping):
        raise ValueError('saved query config must be an execution mapping')
    version, mode = saved.get('schema_version'), saved.get('mode')
    if type(version) is not int:
        raise ValueError('saved query config has invalid schema')
    if version == 1 and mode == 'legacy':
        inherited = Stage1QueryConfig()
        expected = _historical_query_execution_config(inherited, ())
    elif version == 1 and mode == 'adaptive':
        names = ('text_gamma', 'attention_layers', *_CHUNKS)
        if any(name not in saved for name in (*names, 'resolved_layer_indices')):
            raise ValueError('saved adaptive query config has missing required fields')
        indices = saved['resolved_layer_indices']
        if type(indices) is not list:
            raise ValueError('saved resolved layer indices must be a JSON list')
        _validate_layers(indices)
        inherited = Stage1QueryConfig(mode='adaptive', **{name: saved[name] for name in names})
        expected = _historical_query_execution_config(inherited, tuple(indices))
    elif version == 2 and mode == 'adaptive':
        names = ('text_gamma', 'query_chunk_size', 'key_chunk_size')
        if any(name not in saved for name in names):
            raise ValueError('saved adaptive query config has missing required fields')
        if type(saved.get('temperature')) is not float:
            raise ValueError('saved adaptive temperature must be the float 1.0')
        inherited = Stage1QueryConfig(mode='adaptive', **{name: saved[name] for name in names})
        expected = query_execution_config(inherited, ())
    else:
        raise ValueError('saved query config has invalid or unsupported mode/schema')
    if dict(saved) != expected:
        raise ValueError('saved query config has invalid schema or semantic fields')
    return inherited


def resolve_query_config(
    explicit: Mapping[str, object], *, saved: Mapping[str, object] | None = None,
) -> Stage1QueryConfig:
    """Resolve training fields without reinterpreting historical native adaptive."""
    unknown = set(explicit) - _CLI_FIELDS.keys()
    if unknown:
        raise ValueError(f'unknown query option: {sorted(unknown)}')
    values = {_CLI_FIELDS[name]: value for name, value in explicit.items()}
    explicit_config = Stage1QueryConfig(**values)
    if saved is None:
        validate_query_training_config(explicit_config)
        return explicit_config
    inherited = validate_query_execution_config(saved)
    if inherited.mode == 'adaptive' and saved['schema_version'] == 1:
        raise ValueError('historical native-adaptive training requires its original reviewed code; '
                         'external content adaptive requires a fresh run')
    for name, value in values.items():
        if value != getattr(inherited, name):
            raise ValueError(f'explicit {name} conflicts with saved query resume config')
    validate_query_training_config(inherited)
    return inherited


@dataclass(frozen=True)
class DiscountedTextState:
    total: Tensor | None = None
    weight: float = 0.


@dataclass(frozen=True)
class HistoryQueryResult:
    query: Tensor
    latent_mass: Tensor
    text_present: bool


def history_text_positions(
    ids: Tensor, attention_mask: Tensor, spans: tuple[Stage1LatentSpan, ...],
    assistant_content_start: int, special_ids: frozenset[int],
) -> tuple[tuple[int, ...], ...]:
    """Return assistant text preceding each START, excluding whole latent blocks."""
    if ids.ndim != 1 or attention_mask.shape != ids.shape:
        raise ValueError('ids and attention_mask must have the same sample-row [S] shape')
    if ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('layout token ids must be integers')
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError('layout attention_mask must contain only 0 or 1')
    if (type(assistant_content_start) is not int
            or not 0 <= assistant_content_start <= ids.numel()):
        raise ValueError('assistant content start is outside the sample row')
    previous = assistant_content_start
    result = []
    tokens, mask = ids.tolist(), attention_mask.tolist()
    for span in spans:
        if (type(span.start) is not int or type(span.end) is not int
                or not previous <= span.start < span.end < len(tokens)
                or not span.latent_positions
                or any(type(index) is not int for index in span.latent_positions)
                or span.latent_positions != tuple(range(span.start + 1, span.end))):
            raise ValueError('latent layout requires ordered complete nonoverlapping spans')
        if not all(mask[span.start:span.end + 1]):
            raise ValueError('latent span overlaps masked padding')
        result.append(tuple(index for index in range(previous, span.start)
                            if mask[index] and tokens[index] not in special_ids))
        previous = span.end + 1
    return tuple(result)


def _finite_tensor(name: str, tensor: Tensor) -> Tensor:
    value = tensor.float()
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f'{name} must be finite in FP32')
    return value


def _row(name: str, tensor: Tensor) -> Tensor:
    if tensor.ndim != 2 or tensor.shape[0] != 1 or tensor.shape[1] == 0:
        raise ValueError(f'{name} must have shape [1,H] with H positive')
    return _finite_tensor(name, tensor)


def _compatible(name: str, value: Tensor, other: Tensor) -> None:
    if value.shape != other.shape or value.device != other.device:
        raise ValueError(f'{name} must share query shape and device')


def update_text_history(
    state: DiscountedTextState, current_mean: Tensor | None, *, gamma: float,
) -> tuple[DiscountedTextState, Tensor | None]:
    """Age every step and add one vote per nonempty segment, regardless of length."""
    gamma = _gamma(gamma)
    if (isinstance(state.weight, bool) or not isinstance(state.weight, (int, float))
            or not math.isfinite(state.weight) or state.weight < 0
            or (state.total is None) != (state.weight == 0)):
        raise ValueError('text state requires finite nonnegative weight and a matching total')
    total = None if state.total is None else _row('text total', state.total) * gamma
    weight = state.weight * gamma
    if current_mean is not None:
        current = _row('current text mean', current_mean)
        if total is not None:
            _compatible('text mean', current, total)
        total = current if total is None else total + current
        weight += 1.
    if weight == 0:
        return DiscountedTextState(), None
    total = _finite_tensor('text total', total)
    summary = _finite_tensor('text summary', total / weight)
    return DiscountedTextState(total=total, weight=weight), summary


def aggregate_history_latents(hidden: Tensor, scores: Tensor) -> tuple[Tensor, Tensor]:
    """Sum raw attention-weighted latents without renormalizing their mass."""
    if (hidden.ndim != 2 or hidden.shape[1] == 0 or scores.ndim != 1
            or hidden.shape[0] != scores.shape[0] or hidden.device != scores.device):
        raise ValueError('latent hidden [M,H] and scores [M] must match on one device')
    hidden = _finite_tensor('latent hidden', hidden)
    scores = _finite_tensor('latent scores', scores)
    if bool((scores < 0).any()):
        raise ValueError('latent attention scores must be nonnegative')
    latent_sum = (hidden * scores[:, None]).sum(dim=0, keepdim=True)
    mass = scores.sum()
    return _finite_tensor('latent sum', latent_sum), _finite_tensor('latent mass', mass)


def compose_history_query(
    global_intent: Tensor, text_summary: Tensor | None, latent_sum: Tensor,
    latent_mass: Tensor,
) -> HistoryQueryResult:
    """Mix intent, optional segment mean and unnormalized latent attention mass."""
    intent = _row('global intent', global_intent)
    body = _row('latent sum', latent_sum)
    _compatible('latent sum', body, intent)
    mass = _finite_tensor('latent mass', latent_mass)
    if mass.ndim != 0 or mass.device != intent.device or bool(mass < 0):
        raise ValueError('latent mass must be a nonnegative scalar on the query device')
    numerator = intent + body
    present = text_summary is not None
    if present:
        text = _row('text summary', text_summary)
        _compatible('text summary', text, intent)
        numerator = numerator + text
    query = _finite_tensor('history query', numerator / (1 + int(present) + mass))
    norm = torch.linalg.vector_norm(query)
    if not bool(torch.isfinite(norm)) or bool(norm <= 1e-8):
        raise FloatingPointError('history query norm must be finite and greater than 1e-8')
    return HistoryQueryResult(query=query, latent_mass=mass, text_present=present)
