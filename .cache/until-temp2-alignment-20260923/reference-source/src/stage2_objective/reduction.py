"""Global-normalization and reporting primitives for the Stage 2 objective."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor


@dataclass(frozen=True)
class AccumulationContext:
    world_size: int
    actual_microbatches: int
    configured_divisor: int
    global_blocks: int


@dataclass(frozen=True)
class MetricTotal:
    total: Tensor
    count: int


MEAN_METRIC_KEYS = (
    "loss_ce",
    "loss_dyn",
    "loss_vis",
    "entropy_pre",
    "entropy_post",
    "entropy_ref_pre",
    "entropy_ref_post",
    "ratio_pre",
    "ratio_post",
    "entropy_delta",
    "score_clean",
    "score_masked",
    "vision_gain",
    "ratio_pre_gt_one",
    "ratio_post_lt_one",
    "vision_gain_gt_zero",
)

EVENT_SUM_METRIC_KEYS = (
    "low_reference_entropy_pre",
    "low_reference_entropy_post",
    "valid_blocks",
    "excluded_empty_pre",
    "excluded_empty_post",
    "excluded_overlap_empty_pre",
    "excluded_overlap_empty_post",
    "invocations_f",
    "invocations_reference",
    "invocations_m",
    "decoder_forward_calls_f",
    "decoder_forward_calls_reference",
    "decoder_forward_calls_m",
    "prefix_checkpoint_replay_calls_f",
    "prefix_checkpoint_replay_calls_reference",
    "prefix_checkpoint_replay_calls_m",
    "native_checkpoint_recompute_calls_f",
    "native_checkpoint_recompute_calls_reference",
    "native_checkpoint_recompute_calls_m",
)

# ``loss_total`` has a transport slot so every rank always reduces one identical
# schema. Its slot is deliberately zero: the reported value is reconstructed
# from the independently reduced component means.
METRIC_KEYS = (
    "loss_ce",
    "loss_dyn",
    "loss_vis",
    "loss_total",
    *MEAN_METRIC_KEYS[3:],
    *EVENT_SUM_METRIC_KEYS,
)

_KNOWN_METRIC_KEYS = frozenset(METRIC_KEYS)
_EVENT_SUM_KEYS = frozenset(EVENT_SUM_METRIC_KEYS)


def _require_plain_int(value: object, name: str, *, positive: bool) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _require_loss_scalar(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value)):
        raise ValueError(f"{name} must be finite")


def _validate_context(context: AccumulationContext) -> None:
    if not isinstance(context, AccumulationContext):
        raise TypeError("context must be an AccumulationContext")
    _require_plain_int(context.world_size, "world_size", positive=True)
    actual = _require_plain_int(
        context.actual_microbatches, "actual_microbatches", positive=True
    )
    configured = _require_plain_int(
        context.configured_divisor, "configured_divisor", positive=True
    )
    _require_plain_int(context.global_blocks, "global_blocks", positive=False)
    if actual > configured:
        raise ValueError("actual_microbatches must not exceed configured_divisor")


def scale_objective_loss(
    ce: Tensor, weighted_block_sum: Tensor, context: AccumulationContext
) -> Tensor:
    """Undo the trainer's configured-G division and DDP's gradient mean."""
    _validate_context(context)
    _require_loss_scalar(ce, "ce")
    _require_loss_scalar(weighted_block_sum, "weighted_block_sum")
    if ce.device != weighted_block_sum.device:
        raise ValueError("ce and weighted_block_sum must be on the same device")

    loss = ce * (context.configured_divisor / context.actual_microbatches)
    if context.global_blocks:
        loss = loss + weighted_block_sum * (
            context.configured_divisor
            * context.world_size
            / context.global_blocks
        )
    else:
        loss = loss + weighted_block_sum * 0.0
    if not bool(torch.isfinite(loss.detach())):
        raise ValueError("scaled objective loss is nonfinite")
    return loss


def _validate_metric(name: str, metric: MetricTotal) -> None:
    if name not in _KNOWN_METRIC_KEYS:
        raise KeyError(f"unknown metric key {name!r}")
    if not isinstance(metric, MetricTotal):
        raise TypeError(f"metric {name!r} must be a MetricTotal")
    _require_loss_scalar(metric.total, f"metric {name!r} total")
    _require_plain_int(metric.count, f"metric {name!r} count", positive=False)
    if metric.count == 0 and float(metric.total.detach()) != 0.0:
        raise ValueError(f"metric {name!r} with zero count must have zero total")


def merge_metric_totals(
    records: list[dict[str, MetricTotal]],
) -> dict[str, MetricTotal]:
    """Detach and combine local numerator/count records without averaging."""
    merged: dict[str, MetricTotal] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("metric records must be dictionaries")
        for name, metric in record.items():
            _validate_metric(name, metric)
            total = metric.total.detach().float()
            count = metric.count
            previous = merged.get(name)
            if previous is not None:
                if previous.total.device != total.device:
                    raise ValueError(f"metric {name!r} totals must share a device")
                total = previous.total + total
                count += previous.count
            merged[name] = MetricTotal(total, count)
    return merged


def _coefficient(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-boolean scalar")
    converted = float(value)
    if not math_isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def math_isfinite(value: float) -> bool:
    # Avoid constructing a device tensor merely to validate a host coefficient.
    return value not in (float("inf"), float("-inf")) and value == value


def reduce_metric_totals(
    totals: dict[str, MetricTotal],
    *,
    reduce_sum: Callable[[Tensor], Tensor | None],
    lambda_dyn: float = 0.0,
    lambda_vis: float = 0.0,
) -> dict[str, float]:
    """SUM one fixed payload, then apply each metric's declared denominator."""
    dyn_weight = _coefficient(lambda_dyn, "lambda_dyn")
    vis_weight = _coefficient(lambda_vis, "lambda_vis")
    if not callable(reduce_sum):
        raise TypeError("reduce_sum must be callable")
    if not isinstance(totals, dict):
        raise TypeError("totals must be a dictionary")
    for name, metric in totals.items():
        _validate_metric(name, metric)

    devices = {metric.total.device for metric in totals.values()}
    if len(devices) > 1:
        raise ValueError("all metric totals must share a device")
    device = next(iter(devices), torch.device("cpu"))
    width = len(METRIC_KEYS)
    packed = torch.zeros(2 * width, dtype=torch.float64, device=device)
    for index, name in enumerate(METRIC_KEYS):
        if name == "loss_total":
            continue
        metric = totals.get(name)
        if metric is not None:
            packed[index] = metric.total.detach().double()
            packed[width + index] = metric.count

    reduced = reduce_sum(packed)
    if reduced is None:
        reduced = packed
    if not isinstance(reduced, Tensor) or reduced.shape != packed.shape:
        raise ValueError("reduce_sum must return or mutate the fixed metric tensor")
    if not reduced.dtype.is_floating_point or not bool(torch.isfinite(reduced).all()):
        raise ValueError("reduced metric totals must be finite floating point values")

    values: dict[str, float] = {}
    for index, name in enumerate(METRIC_KEYS):
        if name == "loss_total":
            continue
        count_value = float(reduced[width + index].item())
        if count_value < 0 or not count_value.is_integer():
            raise ValueError(f"reduced metric {name!r} count must be nonnegative integer")
        if count_value == 0:
            continue
        total_value = float(reduced[index].item())
        values[name] = (
            total_value if name in _EVENT_SUM_KEYS else total_value / count_value
        )

    if "loss_ce" in values:
        values["loss_total"] = (
            values["loss_ce"]
            + dyn_weight * values.get("loss_dyn", 0.0)
            + vis_weight * values.get("loss_vis", 0.0)
        )
    return {name: values[name] for name in METRIC_KEYS if name in values}
