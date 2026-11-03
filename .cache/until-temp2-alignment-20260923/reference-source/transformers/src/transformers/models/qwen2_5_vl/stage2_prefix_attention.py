"""Functional attention state for opt-in OFF/ON insertion decoder calls."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from ...modeling_outputs import BaseModelOutputWithPast


LayerKV = tuple[torch.Tensor, torch.Tensor]
LayerKVs = tuple[LayerKV, ...]


@dataclass
class InsertionDecoderOutput(BaseModelOutputWithPast):
    stage2_layer_key_values: LayerKVs | None = None


def join_insertion_kv(key: torch.Tensor, value: torch.Tensor, prefix: LayerKV | None) -> LayerKV:
    """Join post-RoPE keys and unrotated values without mutating or detaching."""
    if prefix is None:
        return key, value
    pk, pv = prefix
    for previous, current in ((pk, key), (pv, value)):
        if (previous.ndim != 4 or previous.shape[:2] != current.shape[:2]
                or previous.shape[-1] != current.shape[-1]
                or previous.device != current.device or previous.dtype != current.dtype):
            raise ValueError("insertion prefix KV layout mismatch")
    if pk.shape != pv.shape:
        raise ValueError("insertion prefix K/V shapes differ")
    return torch.cat((pk, key), dim=-2), torch.cat((pv, value), dim=-2)


def insertion_prefix_length(prefix: LayerKVs | None, layer_count: int) -> int:
    """Validate common physical length before constructing attention masks."""
    if prefix is None:
        return 0
    if not isinstance(prefix, tuple) or len(prefix) != layer_count:
        raise ValueError("insertion prefix layer count mismatch")
    lengths = []
    for pair in prefix:
        if (not isinstance(pair, tuple) or len(pair) != 2
                or any(not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 for tensor in pair)):
            raise ValueError("insertion prefix KV layout mismatch")
        if pair[0].shape != pair[1].shape:
            raise ValueError("insertion prefix K/V shapes differ")
        lengths.append(pair[0].shape[-2])
    if len(set(lengths)) != 1:
        raise ValueError("insertion prefix lengths differ across layers")
    return lengths[0]


def insertion_visibility(prefix_length: int, query_length: int, window: int | None, device: torch.device):
    """Causal/window predicate shared by dense and Flex masks (physical offsets)."""
    prefix, queries, keys = torch.tensor(
        [prefix_length, query_length, prefix_length + query_length], device=device
    )
    width = torch.tensor(window if window is not None else prefix_length + query_length, device=device)

    def visible(batch, head, query, key):
        absolute_query = prefix + query
        return ((query < queries) & (key < keys) & (key <= absolute_query)
                & (key > absolute_query - width))

    return visible


def build_insertion_attention_mask(*, prefix_length: int, query_length: int, window: int | None,
                                   backend: str, dtype: torch.dtype, device: torch.device):
    visible = insertion_visibility(prefix_length, query_length, window, device)
    if backend == "flex_attention":
        from torch.nn.attention.flex_attention import create_block_mask

        builder = torch.compile(create_block_mask, dynamic=True) if device.type == "cuda" else create_block_mask
        return builder(visible, B=1, H=None, Q_LEN=query_length,
                       KV_LEN=prefix_length + query_length, device=device)
    if backend not in ("eager", "sdpa"):
        raise ValueError(f"unsupported insertion attention backend: {backend}")
    allowed = visible(0, 0, torch.arange(query_length, device=device)[:, None],
                      torch.arange(prefix_length + query_length, device=device)[None, :])
    return torch.zeros((1, 1, query_length, prefix_length + query_length), dtype=dtype, device=device).masked_fill_(
        ~allowed, torch.finfo(dtype).min
    )


def run_insertion_layer(layer, hidden, cos, sin, *flat_prefix, **kwargs):
    """Keep all differentiable state explicit, including post-RoPE prefix KV."""
    prefix = (flat_prefix[0], flat_prefix[1]) if flat_prefix else None
    return layer(hidden, position_embeddings=(cos, sin),
                 stage2_prefix_key_value=prefix, **kwargs)


def checkpoint_insertion_layer(layer, hidden, position_embeddings, prefix, *, observer=None, **kwargs):
    """Bind a layer's metadata once; replay never captures a changing loop cell."""
    @contextmanager
    def recomputing():
        if observer is not None:
            observer("layer_recompute")
        yield

    def contexts():
        return nullcontext(), recomputing()

    call = partial(run_insertion_layer, layer, **kwargs)
    args = (hidden, *position_embeddings) if prefix is None else (hidden, *position_embeddings, *prefix)
    return checkpoint(call, *args, use_reentrant=False, preserve_rng_state=True, context_fn=contexts)
