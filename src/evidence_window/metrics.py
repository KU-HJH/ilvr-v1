"""Descriptive evidence-window metrics and safe rank-local matrix dumps."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
import torch.distributed as distributed


@dataclass(frozen=True)
class MetricBatch:
    """A metric represented by sufficient statistics instead of a local mean."""

    numerator: torch.Tensor
    count: torch.Tensor

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.numerator) or not torch.is_tensor(self.count):
            raise ValueError("metric numerator and count must be tensors")
        if self.numerator.shape != self.count.shape:
            raise ValueError("metric numerator and count must have matching shapes")
        if self.numerator.device != self.count.device:
            raise ValueError("metric numerator and count must be on the same device")

    @property
    def mean(self) -> torch.Tensor:
        """Return a zero-valued mean for statistics with no valid observations."""
        count = self.count.to(dtype=self.numerator.dtype)
        return torch.where(
            count > 0,
            self.numerator / count.clamp_min(1),
            torch.zeros_like(self.numerator),
        )


@dataclass(frozen=True)
class ContributionSummaries:
    """Positive-mass-only descriptive summaries for each ``[B, K]`` window."""

    top1_ratio: torch.Tensor
    normalized_entropy: torch.Tensor
    entropy_valid: torch.Tensor
    above_threshold_count: torch.Tensor


def contribution_summaries(
    relations: torch.Tensor,
    pair_mask: torch.Tensor,
    threshold: float,
) -> ContributionSummaries:
    """Summarize relation mass without altering the signed relation matrix.

    ``relations`` is ``[B, J, K]``.  Negative entries are excluded only from
    these descriptive mass summaries; callers retain the original signed
    matrix for losses and other calculations.
    """
    _validate_relation_inputs(relations, pair_mask)
    _validate_threshold(threshold)

    valid = pair_mask.to(dtype=torch.bool)
    positive = relations.clamp_min(0) * valid.to(dtype=relations.dtype)
    mass = positive.sum(dim=1)
    eps = torch.finfo(relations.dtype).eps
    probability = positive / mass.unsqueeze(1).clamp_min(eps)
    candidate_count = valid.sum(dim=1)
    top1_ratio = probability.max(dim=1).values
    entropy = -(probability.clamp_min(eps).log() * probability).sum(dim=1)
    entropy_valid = mass.gt(0) & candidate_count.gt(1)
    normalized_entropy = entropy / candidate_count.clamp_min(2).to(relations.dtype).log()
    normalized_entropy = torch.where(
        entropy_valid, normalized_entropy, torch.zeros_like(normalized_entropy)
    )
    above_threshold_count = ((relations > threshold) & valid).sum(dim=1)
    return ContributionSummaries(
        top1_ratio=top1_ratio,
        normalized_entropy=normalized_entropy,
        entropy_valid=entropy_valid,
        above_threshold_count=above_threshold_count,
    )


def global_outer_coefficient(global_margin: torch.Tensor, beta: float) -> torch.Tensor:
    """Return the exact outer derivative coefficient of a global DPO margin."""
    if not torch.is_tensor(global_margin) or not global_margin.is_floating_point():
        raise ValueError("global_margin must be a floating-point tensor")
    beta_value = _validate_positive_finite(beta, "beta")
    return beta_value * torch.sigmoid(-beta_value * global_margin)


def reduce_sufficient_statistics(
    numerator: torch.Tensor | MetricBatch,
    count: torch.Tensor | None = None,
) -> MetricBatch:
    """SUM-reduce metric numerators and counts before exposing their mean."""
    if isinstance(numerator, MetricBatch):
        if count is not None:
            raise ValueError("count must be omitted when reducing a MetricBatch")
        batch = numerator
    else:
        if count is None:
            raise ValueError("count is required when numerator is a tensor")
        batch = MetricBatch(numerator=numerator, count=count)

    reduced_numerator = batch.numerator.clone()
    reduced_count = batch.count.clone()
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(reduced_numerator, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(reduced_count, op=distributed.ReduceOp.SUM)
    return MetricBatch(numerator=reduced_numerator, count=reduced_count)


class MatrixJsonlWriter:
    """Write rank-local numeric relation matrices with opaque observation IDs."""

    def __init__(self, directory: str | Path, rank: int) -> None:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        self._directory = Path(directory)
        self._rank = rank

    def write(
        self,
        step: int,
        observation_counter: int,
        relation: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> None:
        """Append numeric matrices for one opaque rank/step/counter observation."""
        _validate_opaque_identifier(step, "step")
        _validate_opaque_identifier(observation_counter, "observation_counter")
        _validate_dump_tensors(relation, pair_mask)

        self._directory.mkdir(parents=True, exist_ok=True)
        record = {
            "observation_id": f"r{self._rank}-s{step}-o{observation_counter}",
            "rank": self._rank,
            "step": step,
            "relation": relation.detach().cpu().tolist(),
            "pair_mask": pair_mask.detach().cpu().to(dtype=torch.bool).tolist(),
        }
        with (self._directory / f"rank-{self._rank:05d}.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record, allow_nan=False, separators=(",", ":")))
            handle.write("\n")


def _validate_relation_inputs(relations: torch.Tensor, pair_mask: torch.Tensor) -> None:
    if not torch.is_tensor(relations) or relations.ndim != 3:
        raise ValueError("relations must have shape [B, J, K]")
    if not relations.is_floating_point():
        raise ValueError("relations must be floating point")
    if not torch.is_tensor(pair_mask) or pair_mask.shape != relations.shape:
        raise ValueError("pair_mask must have the same shape as relations")
    if pair_mask.device != relations.device:
        raise ValueError("pair_mask and relations must be on the same device")


def _validate_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a finite number")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be a finite number")


def _validate_positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite positive number")
    if value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _validate_opaque_identifier(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _validate_dump_tensors(relation: torch.Tensor, pair_mask: torch.Tensor) -> None:
    if not torch.is_tensor(relation) or not torch.is_tensor(pair_mask):
        raise ValueError("relation and pair_mask must be tensors")
    if relation.shape != pair_mask.shape:
        raise ValueError("relation and pair_mask must have matching shapes")
    if relation.device != pair_mask.device:
        raise ValueError("relation and pair_mask must be on the same device")
    if relation.is_complex() or not (relation.is_floating_point() or relation.dtype in _INTEGER_DTYPES):
        raise ValueError("relation must be numeric and real")
    if pair_mask.is_complex() or not (
        pair_mask.dtype == torch.bool
        or pair_mask.is_floating_point()
        or pair_mask.dtype in _INTEGER_DTYPES
    ):
        raise ValueError("pair_mask must be numeric and real")
    if not torch.isfinite(relation).all() or not torch.isfinite(pair_mask).all():
        raise ValueError("relation and pair_mask must be finite")
    if not torch.logical_or(pair_mask == 0, pair_mask == 1).all():
        raise ValueError("pair_mask must contain only zero/one values")


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
