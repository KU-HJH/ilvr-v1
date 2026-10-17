"""Sample-local stable ranking of detached full-vocabulary token entropies."""

from fractions import Fraction
import math
from collections.abc import Sequence

import torch
from torch import Tensor


@torch.no_grad()
def select_entropy_masks(step_entropies: Sequence[Tensor], percent: float) -> tuple[Tensor, ...]:
    """Select the highest-entropy tokens across a single sample's steps."""
    if type(percent) not in (int, float) or not math.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError("entropy selection percent must be finite and in (0, 100]")
    if any(not isinstance(value, Tensor) or value.ndim != 1 or not value.is_floating_point()
           for value in step_entropies):
        raise ValueError("step entropies must be floating [T] tensors")
    if not step_entropies:
        return ()
    device = step_entropies[0].device
    if any(value.device != device for value in step_entropies):
        raise ValueError("step entropies must be on the same device")
    if any(not bool(torch.isfinite(value).all()) for value in step_entropies):
        raise ValueError("step entropies must be finite")
    values = torch.cat(tuple(value.detach() for value in step_entropies))
    count = values.numel()
    if not count:
        return tuple(torch.empty_like(value, dtype=torch.bool) for value in step_entropies)
    fraction = Fraction(str(percent)) / 100
    keep = math.ceil(count * fraction)
    mask = torch.zeros(count, dtype=torch.bool, device=device)
    mask[torch.argsort(values, descending=True, stable=True)[:keep]] = True
    masks = []
    cursor = 0
    for value in step_entropies:
        masks.append(mask[cursor:cursor + value.numel()])
        cursor += value.numel()
    return tuple(masks)
