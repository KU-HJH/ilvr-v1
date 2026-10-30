"""Detached candidate selection, live OFF entropy and clean-to-OFF mass KL."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .dil_bce_orig_loss import _validate_logits, _validate_pair
from .dil_topk_loss import _group_mass_terms


@dataclass(frozen=True)
class CandidateSupport:
    ids: Tensor
    valid: Tensor
    # Membership in the initial top-k, before the optional GT union.
    gt_coverage: Tensor | None


@dataclass(frozen=True)
class TopkEntropyComponents:
    kl: Tensor
    entropy: Tensor
    normalized_entropy: Tensor | None
    support_size: Tensor


@dataclass(frozen=True)
class MassTokenComponents:
    loss: Tensor
    clean_mass: Tensor | None
    off_mass: Tensor | None
    mass_abs_gap: Tensor | None


def build_entropy_support(
    clean_logits: Tensor, off_logits: Tensor, target_ids: Tensor,
    *, k: int, mode: str, include_gt: bool, validate: bool = True,
    collect_diagnostics: bool = True,
) -> CandidateSupport:
    """Choose top-k with the existing torch tie behavior; append missing GT."""
    if validate:
        _validate_pair(clean_logits, off_logits, target_ids)
    if type(k) is not int or not 2 <= k < clean_logits.shape[1]:
        raise ValueError("k must be an integer with 2 <= k < vocabulary size")
    if mode not in ("independent", "sync_topk"):
        raise ValueError("mode must be independent or sync_topk")
    if type(include_gt) is not bool:
        raise ValueError("include_gt must be boolean")
    with torch.no_grad():
        source = off_logits if mode == "independent" else clean_logits
        ids = source.detach().float().topk(k, dim=-1).indices
        coverage = ((ids == target_ids[:, None]).any(-1)
                    if include_gt or collect_diagnostics or validate else None)
        valid = torch.ones_like(ids, dtype=torch.bool)
        if include_gt:
            ids = torch.cat((ids, target_ids.long()[:, None]), -1)
            valid = torch.cat((valid, ~coverage[:, None]), -1)
    return CandidateSupport(ids, valid, coverage)


def _validate_support(logits: Tensor, support: CandidateSupport) -> None:
    ids, valid, coverage = support.ids, support.valid, support.gt_coverage
    if any(t.device != logits.device for t in (ids, valid, coverage)):
        raise ValueError("candidate and logit devices must match")
    _validate_logits(logits, "OFF logits")
    if ids.ndim != 2 or ids.shape[0] != logits.shape[0] or valid.shape != ids.shape:
        raise ValueError("candidate IDs and mask must have aligned [P,K] shape")
    if ids.dtype != torch.long or valid.dtype != torch.bool:
        raise ValueError("candidate IDs must be long and valid mask boolean")
    if coverage.shape != (logits.shape[0],) or coverage.dtype != torch.bool:
        raise ValueError("GT coverage must be boolean [P]")
    if bool((valid.sum(-1) < 2).any()):
        raise ValueError("each support must contain at least two valid candidates")
    if bool((((ids < 0) | (ids >= logits.shape[1])) & valid).any()):
        raise ValueError("valid candidate IDs must be in vocabulary range")
    sorted_ids = ids.masked_fill(~valid, logits.shape[1]).sort(-1).values
    duplicates = (sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (sorted_ids[:, 1:] < logits.shape[1])
    if bool(duplicates.any()):
        raise ValueError("valid candidate IDs must be unique")


def topk_entropy_components(off_logits: Tensor, support: CandidateSupport, *,
                            validate: bool = True, collect_diagnostics: bool = True) -> TopkEntropyComponents:
    """KL to uniform in nats on exactly the valid support, with q live."""
    if validate:
        _validate_support(off_logits, support)
    selected = off_logits.float().gather(-1, support.ids.masked_fill(~support.valid, 0))
    selected = selected.masked_fill(~support.valid, -torch.inf)
    logq = selected.log_softmax(-1)
    q = logq.exp()
    # Padded entries and saturated valid entries both have q=0. Mask before
    # multiplication to implement the entropy limit 0*log(0)=0, without floors.
    safe_logq = torch.where(support.valid & (q > 0), logq, 0.0)
    entropy = -(q * safe_logq).sum(-1)
    size = support.valid.sum(-1)
    logn = size.float().log()
    return TopkEntropyComponents(logn - entropy, entropy, entropy / logn if collect_diagnostics else None, size)


def entropy_support_mass_components(
    clean_logits: Tensor, off_logits: Tensor, support: CandidateSupport, *,
    validate: bool = True, collect_diagnostics: bool = True,
) -> MassTokenComponents:
    """Binary KL with detached clean mass and live support/complement OFF."""
    if clean_logits.shape != off_logits.shape or clean_logits.device != off_logits.device:
        raise ValueError("clean and OFF logits must have matching shapes/devices")
    if validate:
        _validate_logits(clean_logits, "clean logits")
        _validate_support(off_logits, support)
    # scatter_add avoids an invalid padded zero overwriting a valid ID zero.
    counts = torch.zeros_like(off_logits, dtype=torch.long).scatter_add(
        -1, support.ids.masked_fill(~support.valid, 0), support.valid.long(),
    )
    mask = counts > 0
    return MassTokenComponents(*_group_mass_terms(clean_logits.detach().float(), off_logits.float(), mask,
                                                 collect_diagnostics=collect_diagnostics))


def entropy_support_tail_kl(
    clean_logits: Tensor, off_logits: Tensor, support: CandidateSupport, *,
    temperature: float = 1.0, validate: bool = True,
) -> Tensor:
    """Decoupled KL(clean.detach || OFF), normalized only outside the support."""
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('tail KL temperature must be finite and positive')
    if clean_logits.shape != off_logits.shape or clean_logits.device != off_logits.device:
        raise ValueError('clean and OFF logits must have matching shapes/devices')
    if validate:
        _validate_logits(clean_logits, 'clean logits')
        _validate_support(off_logits, support)
    counts = torch.zeros_like(off_logits, dtype=torch.long).scatter_add(
        -1, support.ids.masked_fill(~support.valid, 0), support.valid.long())
    tail = counts == 0
    empty = ~tail.any(-1, keepdim=True)
    # Empty supports get a benign temporary normalizer; masking below makes
    # their loss and gradients zero. No host read or full-softmax subtraction.
    target = (clean_logits.detach().float() / temperature).masked_fill(~tail, -torch.inf)
    prediction = (off_logits.float() / temperature).masked_fill(~tail, -torch.inf)
    logp = target.masked_fill(empty, 0.).log_softmax(-1)
    logq = prediction.masked_fill(empty, 0.).log_softmax(-1)
    p = logp.exp().masked_fill(~tail, 0.)
    difference = torch.where(tail & (p > 0), logp - logq, 0.)
    return temperature**2 * (p * difference).sum(-1)
