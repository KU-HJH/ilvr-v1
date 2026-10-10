"""Distinct block-score, per-sample KL and global parameter projection reductions."""
from __future__ import annotations
import math
import torch
from torch import Tensor
from .v3_config import Stage1V3Config
from .v3_metadata import V3Block

QueryKey = tuple[int, int]


def block_score_sum(logits: Tensor, blocks: tuple[V3Block, ...]) -> Tensor:
    if logits.ndim != 2 or logits.shape[0] != sum(len(b.target_ids) for b in blocks):
        raise ValueError('block logits must be [Q,V] in block/query order')
    logp = logits.float().log_softmax(-1)
    result = logp.sum() * 0
    offset = 0
    for block in blocks:
        n = len(block.target_ids)
        ids = torch.tensor(block.target_ids, device=logits.device)
        result = result + logp[offset:offset+n].gather(1, ids[:,None]).mean()
        offset += n
    return result


def text_kl_loss(clean_logits: Tensor, restricted_logits: Tensor, query_keys: tuple[QueryKey, ...],
                 *, temperature: float = 2.) -> Tensor:
    if (clean_logits.shape != restricted_logits.shape or restricted_logits.ndim != 2
            or restricted_logits.shape[0] != len(query_keys) or temperature <= 0 or not math.isfinite(temperature)):
        raise ValueError('invalid KL query shape or temperature')
    if not query_keys:
        return restricted_logits.float().sum() * 0
    f = (clean_logits.detach().float()/temperature).log_softmax(-1)
    r = (restricted_logits.float()/temperature).log_softmax(-1)
    per_query = temperature**2 * (f.exp() * (f-r)).sum(-1)
    return torch.stack([per_query[[i for i,k in enumerate(query_keys) if k[0]==sample]].mean()
                        for sample in sorted({k[0] for k in query_keys})]).mean()


def text_kl_weight(completed_steps: int, planned_steps: int, config: Stage1V3Config) -> float:
    if type(completed_steps) is not int or type(planned_steps) is not int or completed_steps < 0 or planned_steps <= 0:
        raise ValueError('KL schedule requires nonnegative completed and positive planned integer steps')
    warmup = math.ceil(config.text_kl_warmup_ratio * planned_steps)
    return config.text_kl_beta * (min(1., completed_steps/warmup) if warmup else 1.)


def project_gain(g: tuple[Tensor, ...], h: tuple[Tensor, ...], *, min_h_norm_sq: float = 1e-30) -> tuple[Tensor, ...]:
    if len(g) != len(h) or not math.isfinite(min_h_norm_sq) or min_h_norm_sq <= 0:
        raise ValueError('invalid Gain projection identity/threshold')
    for a,b in zip(g,h):
        if a.shape != b.shape or a.device != b.device or a.dtype != torch.float32 or b.dtype != torch.float32:
            raise ValueError('Gain requires matching FP32 gradient tensors')
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('nonfinite Gain gradient')
    d = sum((a.double()*b.double()).sum().item() for a,b in zip(g,h))
    n = sum(b.double().square().sum().item() for b in h)
    if not math.isfinite(d) or not math.isfinite(n):
        raise ValueError('nonfinite Gain dot/norm')
    if 0 < n < min_h_norm_sq:
        return tuple(torch.zeros_like(a) for a in g)
    result = tuple((a.double()-(d/n)*b.double()).float().detach() if n and d < 0 else a.detach().clone()
                   for a,b in zip(g,h))
    if any(not torch.isfinite(a).all() for a in result):
        raise ValueError('nonfinite Gain projection')
    return result
