"""Helper-indexed raw image attention; consumes observations, never models."""
from __future__ import annotations

import torch
from torch import Tensor
from transformers.models.qwen2_5_vl.stage1_prefix_trace import AttentionRows

from .prefix_mask_metadata import BlockSources


def _attention_index(rows: AttentionRows) -> dict:
    request = rows.request
    if len(request.query_keys) != len(request.key_positions) or len(rows.values) != len(request.query_keys):
        raise ValueError('attention row count mismatch')
    indexed = {}
    for query, keys, value in zip(request.query_keys, request.key_positions, rows.values):
        if query in indexed or len(set(keys)) != len(keys):
            raise ValueError('duplicate attention query or key')
        if value.ndim != 1 or value.numel() != len(keys) or not value.is_floating_point():
            raise ValueError('attention row shape/dtype mismatch')
        value = value.detach().float()
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError('attention must be finite and nonnegative')
        indexed[query] = ({key: i for i, key in enumerate(keys)}, value)
    return indexed


def _gather_attention(indexed: dict, query: tuple[int, int], keys: tuple[int, ...]) -> Tensor:
    if query not in indexed:
        raise ValueError(f'missing attention query {query}')
    positions, value = indexed[query]
    if any(key not in positions for key in keys):
        raise ValueError(f'missing attention keys for {query}')
    return value[[positions[key] for key in keys]]


def _proxy(indexed: dict, sources: BlockSources) -> Tensor:
    device = next(iter(indexed.values()))[1].device if indexed else torch.device('cpu')
    if not sources.selected_latent_positions or not sources.image_groups:
        return torch.zeros(len(sources.image_groups), dtype=torch.float32, device=device)
    if len(set(sources.selected_latent_positions)) != len(sources.selected_latent_positions):
        raise ValueError('duplicate selected helper position')
    masses = [torch.stack([_gather_attention(indexed, (sources.key[0], position - 1), group.positions).sum()
                          for group in sources.image_groups])
              for position in sources.selected_latent_positions]
    result = torch.stack(masses).mean(0)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('nonfinite helper proxy')
    return result


def helper_attention_proxy(rows: AttentionRows, sources: BlockSources) -> Tensor:
    """Mean group mass over actual selected latent prediction positions.

    Target feature contents intentionally do not enter this attention proxy.
    Missing selections produce zeros; missing requested observations are errors.
    """
    return _proxy(_attention_index(rows), sources)
