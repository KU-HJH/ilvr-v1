"""Detached full-vocabulary KL and deterministic query-local prefix selection."""
from __future__ import annotations

import math

import torch
from torch import Tensor
from transformers.models.qwen2_5_vl.stage1_prefix_conditioning import PrefixMaskPlan, QueryMask
from transformers.models.qwen2_5_vl.stage1_prefix_trace import AttentionRows

from .helper_attention_proxy import _attention_index, _gather_attention, _proxy
from .prefix_mask_config import PrefixConditioningConfig
from .prefix_mask_metadata import HelperSnapshot, ImageGroup


def _finite_float(value: Tensor, name: str) -> Tensor:
    if not value.is_floating_point():
        raise ValueError(f'{name} must be floating point')
    result = value.detach().float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f'{name} must be finite')
    return result


def reverse_kl(student_logits: Tensor, reference_logits: Tensor, *, temperature: float) -> Tensor:
    """Return tau² KL(student || reference), without retaining either graph."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if student_logits.ndim != 2 or student_logits.shape != reference_logits.shape or student_logits.shape[1] == 0:
        raise ValueError('logits must have matching [Q,V] shapes with nonempty vocabulary')
    student = _finite_float(student_logits, 'student logits') / temperature
    reference = _finite_float(reference_logits, 'reference logits') / temperature
    logp, logr = student.log_softmax(-1), reference.log_softmax(-1)
    result = temperature ** 2 * (logp.exp() * (logp - logr)).sum(-1)
    return _finite_float(result, 'KL')


def query_budgets(kl: Tensor, sample_ids: Tensor, *, rho_min: float, rho_max: float,
                  epsilon: float, sigmoid_scale: float) -> Tensor:
    if not all(math.isfinite(x) for x in (rho_min, rho_max, epsilon, sigmoid_scale)) or not 0 <= rho_min <= rho_max <= 1 or epsilon <= 0 or sigmoid_scale <= 0:
        raise ValueError('invalid budget parameters')
    if kl.ndim != 1 or kl.shape != sample_ids.shape or sample_ids.dtype not in (torch.int32, torch.int64) or bool((sample_ids < 0).any()):
        raise ValueError('KL/sample IDs must be aligned vectors with nonnegative integer IDs')
    uncertainty = -torch.log(_finite_float(kl, 'KL').clamp_min(0) + epsilon)
    sample_ids = sample_ids.to(uncertainty.device)
    centered = torch.empty_like(uncertainty)
    for sample in sample_ids.unique():
        mask = sample_ids == sample
        centered[mask] = uncertainty[mask] - uncertainty[mask].mean()
    return _finite_float(rho_min + (rho_max - rho_min) * torch.sigmoid(centered / sigmoid_scale), 'budget')


def _scores(scores: Tensor, count: int, rho: float) -> Tensor:
    if not math.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError('rho must be in [0,1]')
    if scores.ndim != 1 or scores.numel() != count:
        raise ValueError('score/candidate shape mismatch')
    result = _finite_float(scores, 'scores')
    if bool((result < 0).any()) or not bool(torch.isfinite(result.sum())):
        raise ValueError('scores must be nonnegative with finite mass')
    return result


def _rank(scores: Tensor, source_positions: tuple[int, ...]) -> list[int]:
    values = scores.tolist()
    return sorted(range(len(values)), key=lambda i: (-values[i], source_positions[i]))


def select_text_keys(keys: tuple[int, ...], attention: Tensor, *, rho: float,
                     config: PrefixConditioningConfig) -> tuple[int, ...]:
    scores = _scores(attention, len(keys), rho)
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate text key')
    cap = math.floor(config.text_max_mask_ratio * len(keys))
    order = _rank(scores, keys)
    mass = scores.sum()
    if bool(mass < config.text_low_attention_threshold):
        count = min(cap, math.floor(config.text_fallback_mask_ratio * len(keys)))
        return tuple(keys[i] for i in order[:count])
    selected, achieved = [], scores.new_zeros(())
    for index in order:
        if len(selected) >= cap or bool(achieved >= rho * mass):
            break
        selected.append(keys[index])
        achieved = achieved + scores[index]
    return tuple(selected)


def select_image_groups(groups: tuple[ImageGroup, ...], scores: Tensor, *, rho: float,
                        config: PrefixConditioningConfig) -> tuple[int, ...]:
    scores = _scores(scores, len(groups), rho)
    positions = tuple(p for group in groups for p in group.positions)
    if any(not group.positions for group in groups) or len(set(positions)) != len(positions):
        raise ValueError('empty or overlapping image groups')
    cap = math.floor(config.image_max_mask_ratio * len(positions))
    mass = scores.sum()
    if bool(mass <= config.image_score_epsilon):
        return ()
    selected, used, achieved = [], 0, scores.new_zeros(())
    for index in _rank(scores, tuple(min(group.positions) for group in groups)):
        if bool(achieved >= rho * mass):
            break
        size = len(groups[index].positions)
        if used + size > cap:
            continue
        selected.append(index)
        used += size
        achieved = achieved + scores[index]
    return tuple(selected)


def build_prefix_mask_plan(snapshot: HelperSnapshot, rows: AttentionRows, kl: Tensor, *,
                           config: PrefixConditioningConfig) -> tuple[PrefixMaskPlan, dict[str, object]]:
    queries = snapshot.gt_queries
    owned = [query for source in snapshot.sources for query in source.gt_queries]
    if tuple(sorted(set(queries))) != queries or len(set(owned)) != len(owned) or tuple(sorted(owned)) != queries:
        raise ValueError('duplicate or inconsistent GT query ownership')
    if any(query[0] != source.key[0] for source in snapshot.sources for query in source.gt_queries):
        raise ValueError('GT query sample ownership mismatch')
    if kl.shape != (len(queries),):
        raise ValueError('KL must align with unique GT queries')
    kl = _finite_float(kl, 'KL')
    sample_ids = torch.tensor([query[0] for query in queries], dtype=torch.long, device=kl.device)
    budgets = {kind: query_budgets(kl, sample_ids, rho_min=getattr(config, f'{kind}_rho_min'),
        rho_max=getattr(config, f'{kind}_rho_max'), epsilon=config.epsilon, sigmoid_scale=config.sigmoid_scale)
        for kind in ('text', 'image')}
    indexed = _attention_index(rows)
    owners = {query: source for source in snapshot.sources for query in source.gt_queries}
    proxies = {source.key: _proxy(indexed, source) for source in snapshot.sources}
    if len(proxies) != len(snapshot.sources):
        raise ValueError('duplicate block ownership')
    masks, records = [], []
    for i, query in enumerate(queries):
        source = owners[query]
        text = _gather_attention(indexed, query, source.text_keys)
        text_mass = text.sum()
        rho_text, rho_image = float(budgets['text'][i]), float(budgets['image'][i])
        chosen_text = select_text_keys(source.text_keys, text, rho=rho_text, config=config)
        proxy = proxies[source.key]
        group_attention = torch.stack([_gather_attention(indexed, query, group.positions).sum()
            for group in source.image_groups]) if source.image_groups else text.new_empty(0)
        image_scores = _scores(proxy * group_attention, len(source.image_groups), rho_image)
        proxy_mass, image_mass = proxy.sum(), image_scores.sum()
        if not source.image_groups:
            skip = 'no_image_candidates'
        elif not source.selected_latent_positions:
            skip = 'no_selected_helper'
        elif bool(proxy_mass < config.helper_proxy_mass_floor):
            skip = 'low_proxy_mass'
        elif bool(image_mass <= config.image_score_epsilon):
            skip = 'low_image_score'
        else:
            skip = None
        if config.mask_all_original_image_keys:
            # The explicit full-image intervention bypasses budget/proxy gates.
            # The existing common plan applies it to both R and RV scored queries.
            selected = tuple(range(len(source.image_groups)))
            skip = None if selected else 'no_image_candidates'
        else:
            selected = () if skip else select_image_groups(source.image_groups, image_scores, rho=rho_image, config=config)
        image_keys = tuple(p for index in selected for p in source.image_groups[index].positions)
        if skip is None and not selected:
            skip = 'token_cap' if rho_image > 0 else 'zero_budget'
        masks.append(QueryMask(query, chosen_text, image_keys))
        selected_text_mass = float(_gather_attention(indexed, query, chosen_text).sum())
        records.append({
            'query': query, 'block': source.key, 'kl': float(kl[i]),
            'rho_text': rho_text, 'rho_image': rho_image,
            'text_raw_mass': float(text_mass), 'text_candidate_count': len(source.text_keys),
            'text_cap': math.floor(config.text_max_mask_ratio * len(source.text_keys)),
            'text_selected_count': len(chosen_text),
            'text_achieved_mass': selected_text_mass / float(text_mass) if text_mass > 0 else 0.,
            'text_fallback_reason': 'low_text_attention' if text_mass < config.text_low_attention_threshold else None,
            'image_proxy': tuple(proxy.tolist()), 'image_proxy_raw_mass': float(proxy_mass),
            'image_raw_mass': float(image_mass),
            'image_candidate_count': sum(len(group.positions) for group in source.image_groups),
            'image_cap': (len(image_keys) if config.mask_all_original_image_keys else
                          math.floor(config.image_max_mask_ratio * sum(len(group.positions) for group in source.image_groups))),
            'image_selected_count': len(image_keys),
            'image_achieved_mass': float(image_scores[list(selected)].sum() / image_mass) if image_mass > 0 else 0.,
            'image_skip_reason': skip,
            'image_group_ids': tuple((source.image_groups[j].image_index, source.image_groups[j].group_index) for j in selected),
            'image_groups': tuple((group.image_index, group.group_index, len(group.positions)) for group in source.image_groups),
        })
    plan = PrefixMaskPlan(tuple(masks), snapshot.snapshot_id, snapshot.parameter_version, snapshot.teacher_version)
    return plan, {'snapshot_id': snapshot.snapshot_id, 'queries': tuple(records)}
