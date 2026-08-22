"""Matched full and pair-local LOO effects for explicit ILVR Stage 2."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch

from .attention import build_window_local_block_mask
from .scores import causal_prediction_query_mask, window_scores
from .stage2_layout import EvidenceWindowEdge, Stage2InterleavedLayout


@dataclass(frozen=True)
class Stage2ConditionOutput:
    """Reduced output with token log-probs aligned to their target positions."""

    token_logps: torch.Tensor
    final_hidden_state: torch.Tensor | None


@dataclass(frozen=True)
class Stage2EvidenceEffects:
    """Selected evidence-to-window conditional effects."""

    contributions: torch.Tensor
    pair_mask: torch.Tensor
    hidden_residuals: torch.Tensor | None
    forward_count: int


ConditionForward = Callable[[torch.Tensor | None, bool], Stage2ConditionOutput]


def compute_stage2_effects(
    layout: Stage2InterleavedLayout,
    edges: Sequence[EvidenceWindowEdge],
    forward: ConditionForward,
    reduction: str = "mean",
    need_hidden: bool = False,
) -> Stage2EvidenceEffects:
    """Compute one matched full forward and one exact LOO forward per edge.

    Token log-probabilities are aggregated at target positions ``t``.  The
    attention intervention and hidden residual instead use causal decoder query
    rows ``t - 1``, because those representations produce the logits that score
    each target.  This shift preserves sequence length and positions.  Both
    current-policy branches remain attached to autograd.
    """
    _validate_layout(layout)
    if not callable(forward):
        raise TypeError("forward must be callable")
    if not isinstance(need_hidden, bool):
        raise ValueError("need_hidden must be boolean")
    edges = tuple(edges)
    if len(set(edges)) != len(edges):
        raise ValueError("edges must be unique")

    prediction_query_mask = causal_prediction_query_mask(layout.window_token_mask)
    intervention_layout = replace(layout, window_token_mask=prediction_query_mask)

    full = forward(None, need_hidden)
    _validate_output(full, layout, need_hidden=need_hidden)

    batch_size, evidence_count, _ = layout.evidence_token_mask.shape
    window_count = layout.window_token_mask.shape[1]
    effect_shape = (batch_size, evidence_count, window_count)
    contributions = full.token_logps.new_zeros(effect_shape)
    pair_mask = torch.zeros(effect_shape, dtype=torch.bool, device=full.token_logps.device)

    hidden_residuals = None
    if need_hidden:
        assert full.final_hidden_state is not None
        hidden_residuals = full.final_hidden_state.new_zeros(
            (*effect_shape, full.final_hidden_state.shape[-1])
        )

    for edge in edges:
        block_mask = build_window_local_block_mask(intervention_layout, edge)
        loo = forward(block_mask, need_hidden)
        _validate_output(loo, layout, need_hidden=need_hidden, matched_to=full)

        batch_index = edge.batch_index
        window_index = edge.window_index
        window_mask = layout.window_token_mask[
            batch_index : batch_index + 1,
            window_index : window_index + 1,
        ]
        full_score = window_scores(
            full.token_logps[batch_index : batch_index + 1], window_mask, reduction
        )[0, 0]
        loo_score = window_scores(
            loo.token_logps[batch_index : batch_index + 1], window_mask, reduction
        )[0, 0]

        selector = torch.zeros(effect_shape, dtype=torch.bool, device=pair_mask.device)
        selector[batch_index, edge.evidence_index, window_index] = True
        pair_mask |= selector
        contributions = contributions + selector.to(contributions.dtype) * (
            full_score - loo_score
        )

        if hidden_residuals is not None:
            assert full.final_hidden_state is not None
            assert loo.final_hidden_state is not None
            full_hidden = _mean_window_hidden(
                full.final_hidden_state[batch_index : batch_index + 1],
                prediction_query_mask[
                    batch_index : batch_index + 1,
                    window_index : window_index + 1,
                ],
            )[0, 0]
            loo_hidden = _mean_window_hidden(
                loo.final_hidden_state[batch_index : batch_index + 1],
                prediction_query_mask[
                    batch_index : batch_index + 1,
                    window_index : window_index + 1,
                ],
            )[0, 0]
            hidden_residuals = hidden_residuals + selector.unsqueeze(-1).to(
                hidden_residuals.dtype
            ) * (full_hidden - loo_hidden)

    return Stage2EvidenceEffects(
        contributions=contributions,
        pair_mask=pair_mask,
        hidden_residuals=hidden_residuals,
        forward_count=1 + len(edges),
    )


def _mean_window_hidden(
    hidden_state: torch.Tensor,
    window_token_mask: torch.Tensor,
) -> torch.Tensor:
    mask = window_token_mask.to(dtype=hidden_state.dtype).unsqueeze(-1)
    sums = (hidden_state.unsqueeze(1) * mask).sum(dim=2)
    counts = window_token_mask.sum(dim=-1).clamp_min(1).to(dtype=hidden_state.dtype)
    return sums / counts.unsqueeze(-1)


def _validate_layout(layout: Stage2InterleavedLayout) -> None:
    if not isinstance(layout, Stage2InterleavedLayout):
        raise TypeError("layout must be a Stage2InterleavedLayout")
    evidence = layout.evidence_token_mask
    windows = layout.window_token_mask
    batch_size, evidence_count, token_count = evidence.shape
    expected_shapes = {
        "window_token_mask": (batch_size, layout.window_token_mask.shape[1], token_count),
        "evidence_valid": (batch_size, evidence_count),
        "window_valid": (batch_size, layout.window_token_mask.shape[1]),
        "causal_pair_mask": (
            batch_size,
            evidence_count,
            layout.window_token_mask.shape[1],
        ),
    }
    tensors = (evidence, windows, layout.evidence_valid, layout.window_valid, layout.causal_pair_mask)
    if any(not torch.is_tensor(value) or value.dtype is not torch.bool for value in tensors):
        raise ValueError("layout masks must be boolean tensors")
    for name, shape in expected_shapes.items():
        value = getattr(layout, name)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value.device != evidence.device:
            raise ValueError("layout masks must share a device")


def _validate_output(
    output: Stage2ConditionOutput,
    layout: Stage2InterleavedLayout,
    *,
    need_hidden: bool,
    matched_to: Stage2ConditionOutput | None = None,
) -> None:
    if not isinstance(output, Stage2ConditionOutput):
        raise TypeError("forward must return Stage2ConditionOutput")
    expected = (
        layout.window_token_mask.shape[0],
        layout.window_token_mask.shape[-1],
    )
    if not torch.is_tensor(output.token_logps) or output.token_logps.shape != expected:
        raise ValueError(f"token_logps must have shape {expected}")
    if not output.token_logps.is_floating_point():
        raise ValueError("token_logps must be floating point")
    if output.token_logps.device != layout.window_token_mask.device:
        raise ValueError("condition output and layout must share a device")

    hidden = output.final_hidden_state
    if need_hidden:
        if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape[:2] != expected:
            raise ValueError("final_hidden_state must have shape [B, T, H] when requested")
        if not hidden.is_floating_point():
            raise ValueError("final_hidden_state must be floating point")
        if hidden.device != output.token_logps.device:
            raise ValueError("final_hidden_state and token_logps must share a device")

    if matched_to is None:
        return
    if output.token_logps.dtype != matched_to.token_logps.dtype:
        raise ValueError("matched token_logps must share a dtype")
    if need_hidden:
        assert hidden is not None
        assert matched_to.final_hidden_state is not None
        if hidden.shape != matched_to.final_hidden_state.shape:
            raise ValueError("matched final_hidden_state tensors must share a shape")
        if hidden.dtype != matched_to.final_hidden_state.dtype:
            raise ValueError("matched final_hidden_state tensors must share a dtype")
