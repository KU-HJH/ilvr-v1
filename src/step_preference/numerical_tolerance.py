"""Fixed numerical gate; callers must supply matched weights, inputs and RNG.

Passing checks losses and full gradients only. It does not establish matching
tokens/masks/targets, GPU equivalence, or final training accuracy.
"""

import math
from collections.abc import Mapping

import torch


LOSS_ABSOLUTE_TOLERANCE = 1e-4
LOSS_RELATIVE_TOLERANCE = 0.01
GRADIENT_RELATIVE_L2_TOLERANCE = 0.01
REQUIRED_LOSSES = ("loss/ce", "loss/context_preference")


def _scaled_l2(value: torch.Tensor) -> float:
    """Norm of a nonempty CPU FP64 tensor without squaring its raw scale."""
    scale = value.abs().max().item()
    if scale == 0. or not math.isfinite(scale):
        return scale
    return scale * math.sqrt((value / scale).square().sum().item())


def compare_numerical_tolerance(
    reference_losses: Mapping[str, float],
    candidate_losses: Mapping[str, float],
    reference_gradients: Mapping[str, torch.Tensor | None],
    candidate_gradients: Mapping[str, torch.Tensor | None],
) -> dict:
    """Compare separate objective terms and all supplied named gradients.

    Gradient values are detached and converted to CPU FP64 before arithmetic.
    Callers must include every trainable parameter, including inactive None
    entries; identical omissions cannot be detected from two mappings alone.
    """
    errors, losses = [], {}
    for key in REQUIRED_LOSSES:
        values = {}
        for side, source in (("reference", reference_losses), ("candidate", candidate_losses)):
            if key not in source:
                errors.append(f"{side} missing required loss {key}")
                continue
            value = source[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            try:
                value = float(value)
            except (TypeError, ValueError, RuntimeError, OverflowError):
                errors.append(f"{side} {key} must be a finite scalar")
                continue
            if not math.isfinite(value):
                errors.append(f"{side} {key} must be finite")
                continue
            values[side] = value
        if len(values) != 2:
            losses[key] = {**values, "absolute_error": None, "limit": None, "passed": False}
            continue
        error = abs(values["candidate"] - values["reference"])
        limit = LOSS_ABSOLUTE_TOLERANCE + LOSS_RELATIVE_TOLERANCE * abs(values["reference"])
        passed = error <= limit
        losses[key] = {**values, "absolute_error": error, "limit": limit, "passed": passed}
        if not passed:
            errors.append(f"{key} absolute error {error} exceeds {limit}")

    gradient_errors = []
    reference_names, candidate_names = set(reference_gradients), set(candidate_gradients)
    if reference_names != candidate_names:
        gradient_errors.append(
            f"gradient keys differ: missing={sorted(reference_names - candidate_names)}, "
            f"extra={sorted(candidate_names - reference_names)}"
        )
    reference_l2 = difference_l2 = 0.
    difference_is_zero = True
    active_tensors, active_elements, inactive_names = 0, 0, []
    for name in sorted(reference_names & candidate_names):
        reference, candidate = reference_gradients[name], candidate_gradients[name]
        if reference is None and candidate is None:
            inactive_names.append(name)
            continue
        if reference is None or candidate is None:
            gradient_errors.append(f"{name}: one-sided None gradient")
            continue
        if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
            gradient_errors.append(f"{name}: gradients must be tensors")
            continue
        if reference.shape != candidate.shape:
            gradient_errors.append(f"{name}: gradient shape mismatch")
            continue
        if (reference.layout != torch.strided or candidate.layout != torch.strided
                or reference.is_complex() or candidate.is_complex()):
            gradient_errors.append(f"{name}: gradients must be dense real tensors")
            continue
        reference = reference.detach().to(device="cpu", dtype=torch.float64)
        candidate = candidate.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(reference).all() & torch.isfinite(candidate).all()):
            gradient_errors.append(f"{name}: gradients must be finite")
            continue
        if reference.numel() == 0:
            continue
        active_tensors += 1
        active_elements += reference.numel()
        difference = candidate - reference
        difference_is_zero = difference_is_zero and torch.equal(candidate, reference)
        # hypot combines per-tensor norms into the full-vector L2 without
        # under/overflow from directly squaring small/large FP64 values.
        reference_l2 = math.hypot(reference_l2, _scaled_l2(reference))
        difference_l2 = math.hypot(difference_l2, _scaled_l2(difference))

    if not active_tensors:
        gradient_errors.append("no active gradient tensors")
    if not math.isfinite(reference_l2) or not math.isfinite(difference_l2):
        gradient_errors.append("gradient FP64 accumulation must be finite")
    relative_l2 = (
        difference_l2 / reference_l2 if reference_l2 > 0.
        else (0. if difference_is_zero else math.inf)
    )
    if not relative_l2 <= GRADIENT_RELATIVE_L2_TOLERANCE:
        gradient_errors.append(
            f"gradient relative L2 {relative_l2} exceeds {GRADIENT_RELATIVE_L2_TOLERANCE}"
        )
    gradients = dict(
        reference_l2=reference_l2, difference_l2=difference_l2,
        relative_l2=relative_l2, limit=GRADIENT_RELATIVE_L2_TOLERANCE,
        active_tensors=active_tensors, active_elements=active_elements,
        inactive_names=tuple(inactive_names), passed=not gradient_errors,
    )
    errors.extend(gradient_errors)
    return dict(passed=not errors, errors=tuple(errors), losses=losses, gradients=gradients)
