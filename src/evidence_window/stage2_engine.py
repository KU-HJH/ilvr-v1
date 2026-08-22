"""Reference-free edge-local credit objective for explicit ILVR Stage 2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .losses import stage2_ew_preference_loss
from .metrics import global_outer_coefficient, softmax_contribution_summaries
from .stage2_config import Stage2EvidenceWindowConfig
from .stage2_effects import (
    ConditionForward,
    Stage2EvidenceEffects,
    compute_stage2_effects,
)
from .stage2_layout import Stage2InterleavedLayout, select_stage2_edges


MetricSums = dict[str, tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class Stage2EvidenceWindowResult:
    """One batch of edge effects, losses, and reducible diagnostics."""

    auxiliary_loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    metric_sums: MetricSums
    effects: Stage2EvidenceEffects


class Stage2EvidenceWindowEngine:
    """Compute the exact matched-SDPA reference-free Stage-2 objective."""

    def __init__(self, config: Stage2EvidenceWindowConfig) -> None:
        if not isinstance(config, Stage2EvidenceWindowConfig):
            raise TypeError("config must be a Stage2EvidenceWindowConfig")
        config.validate()
        self.config = config

    def compute(
        self,
        layout: Stage2InterleavedLayout,
        policy_forward: ConditionForward,
        progress: float,
        reference_targets=None,
    ) -> Stage2EvidenceWindowResult:
        """Compute independent evidence-edge preference for the current policy."""
        if not isinstance(layout, Stage2InterleavedLayout):
            raise TypeError("layout must be a Stage2InterleavedLayout")
        if not callable(policy_forward):
            raise TypeError("policy_forward must be callable")
        if self.config.reference.mode != "none":
            raise NotImplementedError(
                "fixed-reference Stage-2 modes require the reference integration"
            )
        if reference_targets is not None:
            raise ValueError("reference-free Stage-2 does not accept reference targets")

        effective_weight = self.config.effective_weight("ew", progress)
        edges = select_stage2_edges(
            layout,
            self.config.loo.max_evidence_edges_per_window,
        )
        if effective_weight == 0.0 or not edges:
            return self._empty_result(layout)

        effects = compute_stage2_effects(
            layout,
            edges,
            policy_forward,
            reduction=self.config.preference.contribution_reduction,
            need_hidden="hidden_states" in self.config.required_outputs,
        )
        contributions = effects.contributions
        valid = effects.pair_mask
        beta = self.config.preference.beta
        raw_loss = stage2_ew_preference_loss(contributions, valid, beta)
        auxiliary_loss = raw_loss * effective_weight
        metric_sums = _reference_free_metrics(
            contributions,
            valid,
            beta=beta,
            temperature=self.config.relation.temperature,
            threshold=self.config.diagnostics.contribution_threshold,
        )
        edge_losses = -F.logsigmoid(beta * contributions).masked_select(valid)
        metric_sums["loss/ew_pref"] = (
            edge_losses.detach().sum(),
            _count_tensor(edge_losses.numel(), contributions),
        )
        metric_sums["schedule/ew_weight"] = (
            contributions.new_tensor(effective_weight),
            contributions.new_tensor(1.0),
        )
        return Stage2EvidenceWindowResult(
            auxiliary_loss=auxiliary_loss,
            losses={"ew_pref": raw_loss},
            metric_sums=metric_sums,
            effects=effects,
        )

    @staticmethod
    def _empty_result(layout: Stage2InterleavedLayout) -> Stage2EvidenceWindowResult:
        batch_size, evidence_count, _ = layout.evidence_token_mask.shape
        window_count = layout.window_token_mask.shape[1]
        shape = (batch_size, evidence_count, window_count)
        contributions = torch.zeros(
            shape,
            dtype=torch.float32,
            device=layout.evidence_token_mask.device,
        )
        effects = Stage2EvidenceEffects(
            contributions=contributions,
            pair_mask=torch.zeros(shape, dtype=torch.bool, device=contributions.device),
            hidden_residuals=None,
            forward_count=0,
        )
        zero = contributions.sum()
        return Stage2EvidenceWindowResult(
            auxiliary_loss=zero,
            losses={"ew_pref": zero},
            metric_sums={},
            effects=effects,
        )


def _reference_free_metrics(
    contributions: torch.Tensor,
    valid: torch.Tensor,
    *,
    beta: float,
    temperature: float,
    threshold: float,
) -> MetricSums:
    detached = contributions.detach()
    valid = valid.detach().to(dtype=torch.bool)
    edge_values = detached.masked_select(valid)
    edge_count = _count_tensor(edge_values.numel(), detached)
    one = detached.new_tensor(1.0)
    metrics: MetricSums = {
        "ew/mean_contribution": (edge_values.sum(), edge_count),
        "ew/positive_edge_ratio": ((edge_values > 0).sum().to(detached.dtype), edge_count),
        "ew/num_edges": (edge_count, one),
    }

    row_valid = valid.flatten(start_dim=1).any(dim=1)
    global_margin = (
        detached * valid.to(dtype=detached.dtype)
    ).flatten(start_dim=1).sum(dim=1)
    selected_global = global_margin.masked_select(row_valid)
    global_count = _count_tensor(selected_global.numel(), detached)
    metrics["ew/global_margin"] = (selected_global.sum(), global_count)
    metrics["ew/global_outer_coefficient"] = (
        global_outer_coefficient(selected_global, beta).sum(),
        global_count,
    )

    summaries = softmax_contribution_summaries(
        detached,
        valid,
        temperature=temperature,
        threshold=threshold,
    )
    window_has_edge = valid.any(dim=1)
    window_count = window_has_edge.sum().to(dtype=detached.dtype)
    metrics["ew/top1_evidence_mass"] = (
        summaries.top1_ratio.masked_select(window_has_edge).sum(),
        window_count,
    )
    entropy_values = summaries.normalized_entropy.masked_select(summaries.entropy_valid)
    entropy_count = summaries.entropy_valid.sum().to(dtype=detached.dtype)
    metrics["ew/normalized_contribution_entropy"] = (
        entropy_values.sum(),
        entropy_count,
    )
    metrics["ew/entropy_valid_count"] = (entropy_count, one)
    metrics["ew/evidence_above_threshold_count"] = (
        summaries.above_threshold_count.masked_select(window_has_edge).sum().to(detached.dtype),
        window_count,
    )

    for window_index in range(detached.shape[2]):
        window_valid = valid[:, :, window_index]
        window_values = detached[:, :, window_index].masked_select(window_valid)
        metrics[f"ew/window/{window_index}/mean_contribution"] = (
            window_values.sum(),
            _count_tensor(window_values.numel(), detached),
        )
    return metrics


def _count_tensor(value: int, like: torch.Tensor) -> torch.Tensor:
    return like.new_tensor(float(value))


__all__ = ["Stage2EvidenceWindowEngine", "Stage2EvidenceWindowResult"]
