"""Immutable coordinate policy for training-only latent-gated attention."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GatedVisibility:
    """Image and producer coordinates for one transformed ON trajectory."""

    image_positions: tuple[int, ...]
    producer_positions: tuple[int, ...]
    active_start_positions: tuple[int, ...]
    sequence_length: int


def _visibility_predicate(
    policy: GatedVisibility,
    *,
    query_start: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    window: int | None,
):
    """Build the physical-position predicate shared by dense and Flex masks."""
    if query_start < 0 or query_length < 1:
        raise ValueError("gated attention query start/length are out of bounds")
    if query_start + query_length > policy.sequence_length:
        raise ValueError("gated attention query range exceeds the visibility policy")
    if key_length != query_start + query_length or key_length > policy.sequence_length:
        raise ValueError("gated attention key length is inconsistent with the query range")
    if window is not None and window < 1:
        raise ValueError("gated attention window must preserve causal self visibility")

    lookups = []
    for positions in (
        policy.image_positions,
        policy.producer_positions,
        policy.active_start_positions,
    ):
        lookup = torch.zeros(policy.sequence_length + 1, dtype=torch.bool, device=device)
        if positions:
            lookup[torch.tensor(positions, dtype=torch.long, device=device)] = True
        lookups.append(lookup)
    is_image, is_producer, is_active_start = lookups
    # Capture only tensors in the compiled predicate. Capturing ``policy``
    # specializes Dynamo on every image/producer tuple, exhausting its cache
    # and falling back to dense eager FlexAttention during real training.
    # Chunk bounds vary at every cached latent step, so keep those as data too.
    bounds = torch.tensor(
        [query_start, query_length, key_length,
         policy.sequence_length if window is None else window],
        dtype=torch.long, device=device,
    )

    def visible(batch, head, local_query, key):
        query = bounds[0] + local_query
        safe_query = query.clamp(0, is_image.shape[0] - 1)
        safe_key = key.clamp(0, is_image.shape[0] - 1)
        in_bounds = (local_query >= 0) & (local_query < bounds[1]) & (key >= 0) & (key < bounds[2])
        causal = key <= query
        image_access = ~is_image[safe_key] | is_image[safe_query] | is_producer[safe_query]
        start_access = ~is_active_start[safe_key] | (query == key)
        allowed = in_bounds & causal & image_access & start_access
        return allowed & (key > query - bounds[3])

    local_rows = torch.arange(query_length, device=device)
    if not bool(visible(0, 0, local_rows, query_start + local_rows).all()):
        raise ValueError("gated attention policy must preserve causal self visibility")
    return visible


def build_gated_attention_mask(
    policy: GatedVisibility,
    *,
    query_start: int,
    query_length: int,
    key_length: int,
    backend: str,
    dtype: torch.dtype,
    device: torch.device,
    window: int | None = None,
):
    """Build the additive or Flex mask for one full or cached ON call."""
    visible = _visibility_predicate(
        policy,
        query_start=query_start,
        query_length=query_length,
        key_length=key_length,
        device=device,
        window=window,
    )
    if backend == "flex_attention":
        from torch.nn.attention.flex_attention import create_block_mask

        builder = torch.compile(create_block_mask, dynamic=True) if device.type == "cuda" else create_block_mask
        return builder(visible, B=1, H=None, Q_LEN=query_length, KV_LEN=key_length, device=device)
    if backend not in ("eager", "sdpa"):
        raise ValueError(f"unsupported gated attention backend: {backend}")
    if not dtype.is_floating_point:
        raise ValueError("gated additive attention mask requires a floating-point dtype")
    allowed = visible(
        0,
        0,
        torch.arange(query_length, device=device)[:, None],
        torch.arange(key_length, device=device)[None, :],
    )
    return torch.zeros((1, 1, query_length, key_length), dtype=dtype, device=device).masked_fill_(
        ~allowed, torch.finfo(dtype).min
    )
