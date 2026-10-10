"""Detached text-key selection on full-causal attention mass."""
from __future__ import annotations
from collections.abc import Mapping
import math
import torch
from torch import Tensor
from .v3_config import Stage1V3Config
from .v3_metadata import V3Block
from .v3_objective import QueryKey


@torch.no_grad()
def select_text_keys(blocks: tuple[V3Block, ...], attention: Mapping[QueryKey, Tensor],
                     student_logits: Tensor, reference_logits: Tensor,
                     config: Stage1V3Config) -> dict[QueryKey, tuple[int, ...]]:
    if student_logits.ndim != 3 or student_logits.shape != reference_logits.shape:
        raise ValueError('selector logits must have matching [B,T,V] shape')
    keys = tuple((b.batch_index,q) for b in blocks for q in b.query_positions)
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate selector query ownership')
    if not keys:
        return {}
    rows, queries = zip(*keys)
    p = (student_logits[rows,queries].float()/config.temperature).log_softmax(-1)
    r = (reference_logits[rows,queries].float()/config.temperature).log_softmax(-1)
    d = config.temperature**2 * (p.exp()*(p-r)).sum(-1)
    if not torch.isfinite(d).all():
        raise ValueError('nonfinite selector KL')
    u = -(d.clamp_min(0)+1e-6).log()
    rhos = {}
    for sample in sorted(set(rows)):
        indices = [i for i,k in enumerate(keys) if k[0] == sample]
        rho = config.rho_min+(config.rho_max-config.rho_min)*(u[indices]-u[indices].mean()).sigmoid()
        rhos.update((keys[i],value) for i,value in zip(indices,rho.tolist()))
    result = {}
    for block in blocks:
        candidates = block.candidate_positions
        cap = math.floor(config.text_cap*len(candidates))
        for query in block.query_positions:
            key = (block.batch_index,query)
            weights = attention[key].detach().float()
            if weights.shape != (len(candidates),) or not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError('invalid candidate attention values')
            # Do not renormalize: low-mass fallback uses original causal mass.
            mass = weights.sum().item()
            if mass > 1.+1e-5:
                raise ValueError('candidate attention mass exceeds causal total')
            ranked = sorted(zip(candidates, weights.tolist()), key=lambda pair:(-pair[1],pair[0]))
            if mass < config.low_mass_threshold:
                chosen = ranked[:min(cap,math.floor(config.low_mass_ratio*len(candidates)))]
            else:
                chosen, total = [], 0.
                for pair in ranked[:cap]:
                    chosen.append(pair)
                    total += pair[1]
                    if total >= rhos[key]*mass:
                        break
            result[key] = tuple(position for position,_ in chosen)
    return result
