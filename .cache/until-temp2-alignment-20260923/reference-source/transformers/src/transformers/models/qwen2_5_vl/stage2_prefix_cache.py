"""Neutral differentiable prefix retention for Stage 2 sidecars."""

from dataclasses import dataclass
from typing import Iterator, overload

import torch

from ...cache_utils import DynamicCache


@dataclass(frozen=True)
class CacheTensorLayout:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout


@dataclass(frozen=True)
class OrderedCacheTensors:
    """K/V tensors and their producer layouts in deterministic layer order."""

    tensors: tuple[torch.Tensor, ...]
    layouts: tuple[CacheTensorLayout, ...]

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.tensors)

    def __len__(self) -> int:
        return len(self.tensors)

    @overload
    def __getitem__(self, index: int) -> torch.Tensor: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[torch.Tensor, ...]: ...

    def __getitem__(self, index: int | slice):
        return self.tensors[index]


@dataclass(frozen=True)
class DifferentiablePrefixState:
    ordered_tensors: tuple[torch.Tensor, ...]
    layouts: tuple[CacheTensorLayout, ...]
    sequence_length: int
    producer_tensor_ids: tuple[int, ...]
    producer_differentiability: tuple[bool, ...] = ()


def _tensor_layout(tensor: torch.Tensor) -> CacheTensorLayout:
    return CacheTensorLayout(
        tuple(tensor.shape), tensor.dtype, tensor.device, tensor.layout
    )


def _tensor_is_differentiable(tensor: torch.Tensor) -> bool:
    return tensor.grad_fn is not None or tensor.requires_grad


def validate_stage2_cache_sequence_lengths(
    layouts: tuple[CacheTensorLayout, ...],
    *,
    expected_sequence_length: int,
) -> None:
    if type(expected_sequence_length) is not int or expected_sequence_length < 0:
        raise ValueError("expected sequence length must be a non-negative integer")
    if not isinstance(layouts, tuple) or any(
        not isinstance(layout, CacheTensorLayout) for layout in layouts
    ):
        raise ValueError("cache tensor layouts must be an ordered tuple")
    if any(len(layout.shape) < 2 for layout in layouts):
        raise ValueError("cache tensors must expose a sequence dimension")
    if any(
        layout.shape[-2] != expected_sequence_length for layout in layouts
    ):
        raise ValueError("cache does not match expected sequence length")


def flatten_stage2_dynamic_cache(
    past_key_values: DynamicCache,
) -> OrderedCacheTensors:
    """Flatten one supported cache without copying its producer tensors."""
    if not isinstance(past_key_values, DynamicCache):
        raise ValueError("Stage 2 prefix cache must be a DynamicCache")
    legacy = past_key_values.to_legacy_cache()
    if not isinstance(legacy, tuple) or any(
        not isinstance(layer, tuple) or len(layer) != 2 for layer in legacy
    ):
        raise ValueError("cache must contain ordered key/value pairs")
    flattened = tuple(tensor for layer in legacy for tensor in layer)
    if any(not isinstance(tensor, torch.Tensor) for tensor in flattened):
        raise ValueError("cache key/value entries must be tensors")
    return OrderedCacheTensors(
        flattened,
        tuple(_tensor_layout(tensor) for tensor in flattened),
    )


def _flatten_dynamic_cache(
    cache: DynamicCache | None,
    *,
    expected_sequence_length: int,
) -> OrderedCacheTensors:
    """Compatibility boundary used by the differentiable rollout engine."""
    if cache is None:
        if expected_sequence_length != 0:
            raise ValueError("missing cache for non-empty prefix")
        return OrderedCacheTensors((), ())
    ordered = flatten_stage2_dynamic_cache(cache)
    if cache.get_seq_length() != expected_sequence_length:
        raise ValueError("cache does not match expected sequence length")
    validate_stage2_cache_sequence_lengths(
        ordered.layouts,
        expected_sequence_length=expected_sequence_length,
    )
    return ordered


def _rebuild_dynamic_cache(
    cache_tensors: OrderedCacheTensors | tuple[torch.Tensor, ...],
    *,
    expected_layer_count: int | None,
    expected_sequence_length: int,
) -> DynamicCache:
    """Rebuild a fresh wrapper around witnessed ordered K/V tensors."""
    raw_tensors = (
        cache_tensors.tensors
        if isinstance(cache_tensors, OrderedCacheTensors)
        else cache_tensors
    )
    if not isinstance(raw_tensors, tuple):
        raise ValueError("cache tensors must be an ordered tuple")
    if len(raw_tensors) % 2:
        raise ValueError("cache must contain ordered key/value pairs")
    if any(not isinstance(tensor, torch.Tensor) for tensor in raw_tensors):
        raise ValueError("cache key/value entries must be tensors")
    layer_count = len(raw_tensors) // 2
    if expected_layer_count is not None and layer_count != expected_layer_count:
        raise ValueError("cache does not match expected layer count")
    if not raw_tensors:
        if expected_sequence_length != 0:
            raise ValueError("cache does not match expected sequence length")
        return DynamicCache()
    if not isinstance(cache_tensors, OrderedCacheTensors):
        raise ValueError("cache must preserve ordered key/value pairs")
    if len(cache_tensors.layouts) != len(raw_tensors) or any(
        not isinstance(layout, CacheTensorLayout)
        for layout in cache_tensors.layouts
    ):
        raise ValueError("cache tensor layout count must match ordered tensors")
    if any(
        _tensor_layout(tensor) != expected_layout
        for tensor, expected_layout in zip(raw_tensors, cache_tensors.layouts)
    ):
        raise ValueError("cache must preserve ordered key/value pairs and tensor layout")
    validate_stage2_cache_sequence_lengths(
        cache_tensors.layouts,
        expected_sequence_length=expected_sequence_length,
    )

    legacy = tuple(
        (raw_tensors[index], raw_tensors[index + 1])
        for index in range(0, len(raw_tensors), 2)
    )
    cache = DynamicCache.from_legacy_cache(legacy)
    if len(cache) != layer_count:
        raise ValueError("cache does not match expected layer count")
    if cache.get_seq_length() != expected_sequence_length:
        raise ValueError("cache does not match expected sequence length")
    return cache


def retain_stage2_prefix_state(
    past_key_values: DynamicCache,
    *,
    expected_sequence_length: int,
) -> DifferentiablePrefixState:
    """Retain the exact differentiable K/V tensors emitted by a clean pass."""
    ordered = flatten_stage2_dynamic_cache(past_key_values)
    validate_stage2_cache_sequence_lengths(
        ordered.layouts,
        expected_sequence_length=expected_sequence_length,
    )
    if past_key_values.get_seq_length() != expected_sequence_length:
        raise ValueError("cache does not match expected sequence length")
    producer_differentiability = tuple(
        _tensor_is_differentiable(tensor) for tensor in ordered.tensors
    )
    if not producer_differentiability or not all(producer_differentiability):
        raise ValueError("Stage 2 sidecar prefix must remain differentiable")
    return DifferentiablePrefixState(
        ordered_tensors=ordered.tensors,
        layouts=ordered.layouts,
        sequence_length=expected_sequence_length,
        producer_tensor_ids=tuple(id(tensor) for tensor in ordered.tensors),
        producer_differentiability=producer_differentiability,
    )


def rebuild_stage2_dynamic_cache(
    prefix_state: DifferentiablePrefixState,
) -> DynamicCache:
    """Build an isolated cache wrapper over one validated differentiable prefix."""
    if not isinstance(prefix_state, DifferentiablePrefixState):
        raise TypeError("prefix_state must be a DifferentiablePrefixState")
    if not isinstance(prefix_state.ordered_tensors, tuple):
        raise ValueError("prefix cache tensors must be an ordered tuple")
    if any(
        not isinstance(tensor, torch.Tensor)
        for tensor in prefix_state.ordered_tensors
    ):
        raise ValueError("prefix cache key/value entries must be tensors")
    if (
        not isinstance(prefix_state.layouts, tuple)
        or len(prefix_state.layouts) != len(prefix_state.ordered_tensors)
        or any(
            not isinstance(layout, CacheTensorLayout)
            for layout in prefix_state.layouts
        )
    ):
        raise ValueError("prefix cache tensor layout count must match ordered tensors")
    if (
        not isinstance(prefix_state.producer_tensor_ids, tuple)
        or len(prefix_state.producer_tensor_ids) != len(prefix_state.ordered_tensors)
        or any(type(identity) is not int for identity in prefix_state.producer_tensor_ids)
    ):
        raise ValueError("prefix cache must retain producer tensor identities")
    if any(
        id(tensor) != producer_id or _tensor_layout(tensor) != layout
        for tensor, layout, producer_id in zip(
            prefix_state.ordered_tensors,
            prefix_state.layouts,
            prefix_state.producer_tensor_ids,
        )
    ):
        raise ValueError("prefix cache tensors must match the producer witness")
    if (
        not isinstance(prefix_state.producer_differentiability, tuple)
        or len(prefix_state.producer_differentiability)
        != len(prefix_state.ordered_tensors)
        or any(
            type(differentiable) is not bool
            for differentiable in prefix_state.producer_differentiability
        )
    ):
        raise ValueError("prefix cache must retain producer differentiability witnesses")
    if tuple(
        _tensor_is_differentiable(tensor)
        for tensor in prefix_state.ordered_tensors
    ) != prefix_state.producer_differentiability:
        raise ValueError("prefix cache tensors must match the producer differentiable-tensor witness")
    if prefix_state.ordered_tensors and not any(
        prefix_state.producer_differentiability
    ):
        raise ValueError("prefix cache must preserve a trainable autograd path")
    ordered = OrderedCacheTensors(
        prefix_state.ordered_tensors,
        prefix_state.layouts,
    )
    return _rebuild_dynamic_cache(
        ordered,
        expected_layer_count=len(prefix_state.ordered_tensors) // 2,
        expected_sequence_length=prefix_state.sequence_length,
    )
