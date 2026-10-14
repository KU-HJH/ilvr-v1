"""Detached v5 selectors in aligned query order and original key coordinates."""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import Tensor

from step_preference.entropy_selection import select_entropy_masks
from .v5_config import Stage1V5Config
from .v5_metadata import V5BatchMetadata

QueryKey = tuple[int, int]


def _validate_selected_logits(logits: Tensor, metadata: V5BatchMetadata, name: str,
                              *, like: Tensor | None = None) -> None:
    if (logits.ndim != 2 or logits.shape[0] != len(metadata.query_keys)
            or logits.shape[1] == 0 or not logits.is_floating_point()):
        raise ValueError(f'{name} logits must have aligned [N,V] shape')
    if like is not None and (logits.shape != like.shape or logits.device != like.device):
        raise ValueError(f'{name} logits must match clean shape and device')


def _chunk_logp(logits: Tensor, temperature: float = 1.) -> Tensor:
    logp = (logits.detach().float() / temperature).log_softmax(-1)
    if not bool(torch.isfinite(logp).all()):
        raise ValueError('nonfinite normalized selection logits')
    return logp


@torch.no_grad()
def select_v5_text_keys(
    metadata: V5BatchMetadata, attention: Mapping[QueryKey, Tensor],
    clean_logits: Tensor, reference_logits: Tensor, config: Stage1V5Config,
) -> dict[QueryKey, tuple[int, ...]]:
    """Port the pinned v3 KL selector, indexing full original attention rows."""
    _validate_selected_logits(clean_logits, metadata, 'clean')
    _validate_selected_logits(reference_logits, metadata, 'reference', like=clean_logits)
    keys = metadata.query_keys
    if not keys:
        return {}
    parts = []
    for start in range(0, len(keys), config.loss_chunk_positions):
        stop = start + config.loss_chunk_positions
        p = _chunk_logp(clean_logits[start:stop], config.temperature)
        r = _chunk_logp(reference_logits[start:stop], config.temperature)
        parts.append(config.temperature ** 2 * (p.exp() * (p-r)).sum(-1))
    d = torch.cat(parts)
    if not bool(torch.isfinite(d).all()):
        raise ValueError('nonfinite selector KL')
    u = -(d.clamp_min(0) + 1e-6).log()
    rhos = {}
    for sample in sorted({k[0] for k in keys}):
        indices = [i for i, key in enumerate(keys) if key[0] == sample]
        rho = config.rho_min + (config.rho_max-config.rho_min) * (
            u[indices] - u[indices].mean()).sigmoid()
        rhos.update((keys[i], value) for i, value in zip(indices, rho.tolist()))
    result = {}
    for block in metadata.blocks:
        candidates = block.candidate_positions
        cap = math.floor(config.text_cap * len(candidates))
        for query in block.query_positions:
            key = (block.batch_index, query)
            row = attention[key].detach().float()
            if (row.ndim != 1 or row.numel() <= query
                    or not bool(torch.isfinite(row).all()) or bool((row < 0).any())):
                raise ValueError('invalid original-sequence attention row')
            weights = row[list(candidates)]
            mass = weights.sum().item()
            if mass > 1. + 1e-5:
                raise ValueError('candidate attention mass exceeds causal total')
            ranked = sorted(zip(candidates, weights.tolist()), key=lambda pair: (-pair[1], pair[0]))
            if mass < config.low_attention_mass:
                chosen = ranked[:min(cap, math.floor(config.low_attention_fraction * len(candidates)))]
            else:
                chosen, total = [], 0.
                for pair in ranked[:cap]:
                    chosen.append(pair)
                    total += pair[1]
                    if total >= rhos[key] * mass:
                        break
            result[key] = tuple(position for position, _ in chosen)
    return result


@torch.no_grad()
def select_v5_orig_positions(
    metadata: V5BatchMetadata, clean_logits: Tensor, voff_logits: Tensor | None,
    config: Stage1V5Config,
) -> Tensor:
    """Sample-local full-vocabulary entropy ranks, with no intersection refill."""
    _validate_selected_logits(clean_logits, metadata, 'clean')
    count = len(metadata.query_keys)
    if config.selection == 'all':
        return torch.ones(count, dtype=torch.bool, device=clean_logits.device)
    if not count:
        return torch.empty(0, dtype=torch.bool, device=clean_logits.device)
    needs_off = config.selection in ('voff', 'gap', 'intersection')
    if needs_off:
        if voff_logits is None:
            raise ValueError('V-off logits required by orig position selection')
        _validate_selected_logits(voff_logits, metadata, 'V-off', like=clean_logits)
    clean_parts, off_parts = [], []
    for start in range(0, count, config.loss_chunk_positions):
        stop = start + config.loss_chunk_positions
        lp = _chunk_logp(clean_logits[start:stop])
        clean_parts.append(-(lp.exp() * lp).sum(-1))
        if needs_off:
            lo = _chunk_logp(voff_logits[start:stop])
            off_parts.append(-(lo.exp() * lo).sum(-1))
    clean = torch.cat(clean_parts)
    off = torch.cat(off_parts) if needs_off else None
    values = {'clean': clean, 'voff': off}
    if config.selection == 'gap':
        values['gap'] = (clean-off).abs()
    result = torch.zeros(count, dtype=torch.bool, device=clean_logits.device)
    keys = metadata.query_keys
    for sample in sorted({key[0] for key in keys}):
        # Stable ranking must break ties by original predictor position, not block order.
        indices = sorted((i for i, key in enumerate(keys) if key[0] == sample), key=lambda i: keys[i][1])
        def ranked_mask(entropies: Tensor) -> Tensor:
            return select_entropy_masks((entropies[indices],), config.preference_entropy_top_percent)[0]
        if config.selection == 'intersection':
            mask = ranked_mask(clean) & ranked_mask(off)
        else:
            mask = ranked_mask(values[config.selection])
        result[indices] = mask
    return result
