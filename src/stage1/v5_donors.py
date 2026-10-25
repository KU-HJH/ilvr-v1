"""Sample-owned, detached reverse-cosine vision donors; no model execution."""
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class V5DonorSelection:
    values: Tensor
    donor_kind: str
    image_indices: tuple[int, ...]
    candidate_indices: tuple[int, ...]


def select_reverse_topk_donors(
    helper_candidates: tuple[Tensor, ...], helper_image_ids: tuple[str, ...],
    helper_to_block: tuple[int, ...], block_index: int,
    original_candidates: tuple[Tensor, ...], query: Tensor, *, count: int,
    source_index: int,
) -> V5DonorSelection:
    prefix = f'source_index={source_index} block={block_index}: '
    def fail(message):
        raise ValueError(prefix + message)

    if (type(count) is not int or count <= 0 or type(block_index) is not int or block_index < 0
            or len(helper_candidates) != len(helper_image_ids)
            or len(helper_candidates) != len(helper_to_block)
            or any(not isinstance(i, str) or not i for i in helper_image_ids)
            or any(type(i) is not int or i < 0 for i in helper_to_block)
            or block_index not in helper_to_block):
        fail('invalid count or helper ownership/identity')
    if (not isinstance(query, Tensor) or query.ndim != 2 or query.shape[0] != 1
            or query.shape[1] <= 0 or not query.is_floating_point()
            or not bool(torch.isfinite(query).all())):
        fail('query must be finite floating [1,H]')
    excluded = {identity for identity, owner in zip(helper_image_ids, helper_to_block)
                if owner == block_index}
    seen, eligible = set(), []
    for index, identity in enumerate(helper_image_ids):
        if identity not in excluded and identity not in seen:
            eligible.append(index)
            seen.add(identity)
    source, kind = (helper_candidates, 'helper') if eligible else (original_candidates, 'original')
    indices = eligible if eligible else list(range(len(source)))
    # Validate the supplied helper feature ownership even if an identity is excluded.
    for values in (*helper_candidates, *(original_candidates if kind == 'original' else ())):
        if (not isinstance(values, Tensor) or values.ndim != 2
                or values.shape[1] != query.shape[1] or not values.is_floating_point()
                or values.device != query.device or not bool(torch.isfinite(values).all())):
            fail('candidate features must be finite, sample-owned floating [N,H]')
    if not indices:
        fail('no projected original fallback candidates')
    values = [source[i].detach() for i in indices]
    if len({v.dtype for v in values}) != 1:
        fail('candidate feature dtypes disagree')
    pool = torch.cat(values, dim=0)
    if pool.shape[0] < count:
        fail(f'insufficient {kind} candidates: {pool.shape[0]} < {count}')
    scores = F.cosine_similarity(pool.float(), query.detach().float(), dim=-1, eps=1e-8)
    if not bool(torch.isfinite(scores).all()):
        fail('candidate cosine scores must be finite')
    selected = torch.argsort(scores, descending=False, stable=True)[:count]
    provenance = [(image, candidate) for image in indices for candidate in range(source[image].shape[0])]
    owned = [provenance[i] for i in selected.tolist()]
    return V5DonorSelection(pool.index_select(0, selected).detach(), kind,
                           tuple(i for i, _ in owned), tuple(j for _, j in owned))
