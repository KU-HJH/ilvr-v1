"""Reference-free or fixed-reference edge-local credit for ILVR Stage 2."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import torch
import torch.distributed as distributed
import torch.nn.functional as F

from .artifacts import ReferenceBatchTargets
from .losses import (
    stage2_ew_dpo_loss,
    stage2_ew_preference_loss,
    stage2_hidden_residual_loss,
    stage2_relation_jsd,
)
from .metrics import global_outer_coefficient, softmax_contribution_summaries
from .stage2_config import Stage2EvidenceWindowConfig
from .stage2_effects import (
    ConditionForward,
    Stage2EvidenceEffects,
    compute_stage2_effects,
)
from .stage2_layout import Stage2InterleavedLayout, select_stage2_edges


MetricSums = dict[str, tuple[torch.Tensor, torch.Tensor]]
EdgeCountCoordinator = Callable[[int, torch.device], tuple[int, int]]


@dataclass(frozen=True)
class Stage2EvidenceWindowResult:
    """One batch of edge effects, losses, and reducible diagnostics."""

    auxiliary_loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    metric_sums: MetricSums
    effects: Stage2EvidenceEffects
    weighted_objectives: dict[str, torch.Tensor] = field(default_factory=dict)


class Stage2EvidenceWindowEngine:
    """Compute the exact matched-SDPA Stage-2 edge objective."""

    def __init__(
        self,
        config: Stage2EvidenceWindowConfig,
        *,
        edge_count_coordinator: EdgeCountCoordinator | None = None,
    ) -> None:
        if not isinstance(config, Stage2EvidenceWindowConfig):
            raise TypeError("config must be a Stage2EvidenceWindowConfig")
        config.validate()
        if edge_count_coordinator is not None and not callable(edge_count_coordinator):
            raise TypeError("edge_count_coordinator must be callable")
        self.config = config
        self._edge_count_coordinator = (
            _distributed_edge_counts
            if edge_count_coordinator is None
            else edge_count_coordinator
        )

    def compute(
        self,
        layout: Stage2InterleavedLayout,
        policy_forward: ConditionForward,
        progress: float,
        reference_targets: ReferenceBatchTargets | None = None,
    ) -> Stage2EvidenceWindowResult:
        """Compute independent evidence-edge preference for the current policy."""
        if not isinstance(layout, Stage2InterleavedLayout):
            raise TypeError("layout must be a Stage2InterleavedLayout")
        if not callable(policy_forward):
            raise TypeError("policy_forward must be callable")
        if self.config.reference.mode == "none" and reference_targets is not None:
            raise ValueError("reference-free Stage-2 does not accept reference targets")
        if self.config.reference.mode != "none" and reference_targets is None:
            raise ValueError("fixed-reference Stage-2 requires reference targets")

        ew_weight = self.config.effective_weight("ew", progress)
        relation_weight = self.config.effective_weight("relation", progress)
        residual_weight = self.config.effective_weight("residual", progress)
        active_weights = {
            name: weight
            for name, weight in (
                ("ew_pref", ew_weight),
                ("relation", relation_weight),
                ("residual", residual_weight),
            )
            if weight > 0.0
        }
        if not active_weights:
            return self._empty_result(layout)
        edges = select_stage2_edges(
            layout,
            self.config.loo.max_evidence_edges_per_window,
        )
        global_edge_count, global_max_edges = self._edge_count_coordinator(
            len(edges), layout.evidence_token_mask.device
        )
        _validate_coordinated_counts(
            local_count=len(edges),
            global_count=global_edge_count,
            global_max=global_max_edges,
        )
        if global_edge_count == 0:
            return self._empty_result(layout, active_objectives=active_weights)
        if residual_weight > 0.0 and edges:
            _prevalidate_hidden_reference(reference_targets, layout)

        # ZeRO parameter gathers and DDP buffer broadcasts happen inside model
        # forwards, so every rank must execute the same number before any
        # backward. Reusing a prior output would not enter those collectives.
        graph_zeros: list[torch.Tensor] = []
        need_hidden = residual_weight > 0.0

        def coordinated_forward(block_mask, request_hidden):
            output = policy_forward(block_mask, request_hidden)
            graph_zero = output.token_logps.sum() * 0.0
            if request_hidden and output.final_hidden_state is not None:
                graph_zero = graph_zero + output.final_hidden_state.sum() * 0.0
            graph_zeros.append(graph_zero)
            return output

        effects = compute_stage2_effects(
            layout,
            edges,
            coordinated_forward,
            reduction=self.config.preference.contribution_reduction,
            need_hidden=need_hidden,
        )
        dummy_mask = torch.zeros(
            (
                layout.evidence_token_mask.shape[0],
                layout.evidence_token_mask.shape[-1],
                layout.evidence_token_mask.shape[-1],
            ),
            dtype=torch.bool,
            device=layout.evidence_token_mask.device,
        )
        for _ in range(global_max_edges - len(edges)):
            # An actual same-policy callback is required. Its graph-zero keeps
            # backward hook/collective participation without creating an edge.
            coordinated_forward(dummy_mask, need_hidden)
        effects = replace(effects, forward_count=len(graph_zeros))
        graph_zero = torch.stack(graph_zeros).sum()

        if not edges:
            objective_zeros = {
                name: effects.contributions.sum() * 0.0 for name in active_weights
            }
            return Stage2EvidenceWindowResult(
                auxiliary_loss=graph_zero,
                losses={},
                metric_sums={},
                effects=effects,
                weighted_objectives=objective_zeros,
            )

        contributions = effects.contributions
        valid = effects.pair_mask
        beta = self.config.preference.beta
        objective_values = contributions
        reference = None
        if reference_targets is not None:
            _validate_reference_targets(
                reference_targets,
                effects,
                require_hidden_residuals=residual_weight > 0.0,
            )
            reference = reference_targets.contributions.to(
                device=contributions.device,
                dtype=contributions.dtype,
            ).detach()
            objective_values = contributions - reference.detach()

        losses: dict[str, torch.Tensor] = {}
        weighted_objectives: dict[str, torch.Tensor] = {}
        auxiliary_loss = graph_zero
        if ew_weight > 0.0:
            if reference is None:
                ew_loss = stage2_ew_preference_loss(contributions, valid, beta)
            else:
                ew_loss = stage2_ew_dpo_loss(contributions, reference, valid, beta)
            losses["ew_pref"] = ew_loss
            weighted_objectives["ew_pref"] = ew_loss * ew_weight
            auxiliary_loss = auxiliary_loss + weighted_objectives["ew_pref"]
        if relation_weight > 0.0:
            if reference is None:
                raise ValueError("relation objective requires fixed reference targets")
            relation_loss = stage2_relation_jsd(
                contributions,
                reference,
                valid,
                self.config.relation.temperature,
            )
            losses["relation"] = relation_loss
            weighted_objectives["relation"] = relation_loss * relation_weight
            auxiliary_loss = auxiliary_loss + weighted_objectives["relation"]
        if residual_weight > 0.0:
            assert reference_targets is not None
            assert effects.hidden_residuals is not None
            assert reference_targets.hidden_residuals is not None
            reference_hidden = reference_targets.hidden_residuals.to(
                device=effects.hidden_residuals.device,
            ).detach()
            residual_loss = stage2_hidden_residual_loss(
                effects.hidden_residuals,
                reference_hidden,
                valid,
                self.config.residual.huber_delta,
            )
            losses["residual"] = residual_loss
            weighted_objectives["residual"] = residual_loss * residual_weight
            auxiliary_loss = auxiliary_loss + weighted_objectives["residual"]

        metric_sums = _edge_metrics(
            contributions,
            valid,
            beta=beta,
            temperature=self.config.relation.temperature,
            threshold=self.config.diagnostics.contribution_threshold,
            preference_margins=(
                None if reference_targets is None else objective_values
            ),
        )
        if ew_weight > 0.0:
            edge_losses = -F.logsigmoid(beta * objective_values).masked_select(valid)
            metric_sums["loss/ew_pref"] = (
                edge_losses.detach().sum(),
                _count_tensor(edge_losses.numel(), contributions),
            )
            metric_sums["schedule/ew_weight"] = (
                contributions.new_tensor(ew_weight),
                contributions.new_tensor(1.0),
            )
        if relation_weight > 0.0:
            relation_count = (valid.sum(dim=1) >= 2).sum().to(dtype=contributions.dtype)
            metric_sums["loss/relation"] = (
                losses["relation"].detach() * relation_count,
                relation_count,
            )
            metric_sums["schedule/relation_weight"] = (
                contributions.new_tensor(relation_weight),
                contributions.new_tensor(1.0),
            )
        if residual_weight > 0.0:
            assert effects.hidden_residuals is not None
            residual_count = valid.sum().to(dtype=contributions.dtype)
            policy_norms = effects.hidden_residuals.detach().to(dtype=torch.float32).norm(
                dim=-1
            )
            metric_sums["loss/residual"] = (
                losses["residual"].detach() * residual_count,
                residual_count,
            )
            metric_sums["residual/valid_edge_count"] = (
                residual_count,
                contributions.new_tensor(1.0),
            )
            metric_sums["residual/policy_effect_norm"] = (
                policy_norms.masked_select(valid).sum(),
                residual_count,
            )
            metric_sums["schedule/residual_weight"] = (
                contributions.new_tensor(residual_weight),
                contributions.new_tensor(1.0),
            )
        return Stage2EvidenceWindowResult(
            auxiliary_loss=auxiliary_loss,
            losses=losses,
            metric_sums=metric_sums,
            effects=effects,
            weighted_objectives=weighted_objectives,
        )

    @staticmethod
    def _empty_result(
        layout: Stage2InterleavedLayout,
        active_objectives: dict[str, float] | None = None,
    ) -> Stage2EvidenceWindowResult:
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
            losses={},
            metric_sums={},
            effects=effects,
            weighted_objectives=(
                {}
                if active_objectives is None
                else {name: zero for name in active_objectives}
            ),
        )


def _edge_metrics(
    contributions: torch.Tensor,
    valid: torch.Tensor,
    *,
    beta: float,
    temperature: float,
    threshold: float,
    preference_margins: torch.Tensor | None = None,
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

    diagnostic_margins = (
        detached
        if preference_margins is None
        else preference_margins.detach().to(device=detached.device, dtype=detached.dtype)
    )
    if diagnostic_margins.shape != detached.shape:
        raise ValueError("preference margins must match contribution shape")
    if preference_margins is not None:
        margin_values = diagnostic_margins.masked_select(valid)
        metrics["ew/mean_margin"] = (margin_values.sum(), edge_count)

    row_valid = valid.flatten(start_dim=1).any(dim=1)
    global_margin = (
        diagnostic_margins * valid.to(dtype=detached.dtype)
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


def _validate_reference_targets(
    targets: ReferenceBatchTargets,
    effects: Stage2EvidenceEffects,
    *,
    require_hidden_residuals: bool = False,
) -> None:
    if not isinstance(targets, ReferenceBatchTargets):
        raise TypeError("reference_targets must be ReferenceBatchTargets")
    contributions = targets.contributions
    if not torch.is_tensor(contributions) or contributions.shape != effects.contributions.shape:
        raise ValueError("reference contributions must match policy contribution layout")
    if not contributions.is_floating_point() or not torch.isfinite(contributions).all():
        raise ValueError("reference contributions must be finite floating-point values")
    pair_mask = targets.pair_mask
    if (
        not torch.is_tensor(pair_mask)
        or pair_mask.dtype is not torch.bool
        or pair_mask.shape != effects.pair_mask.shape
    ):
        raise ValueError("reference pair_mask must match policy pair layout")
    if pair_mask.device != effects.pair_mask.device:
        pair_mask = pair_mask.to(device=effects.pair_mask.device)
    if not torch.equal(pair_mask, effects.pair_mask):
        raise ValueError("reference pair_mask does not match selected policy edges")
    if require_hidden_residuals:
        hidden = targets.hidden_residuals
        policy_hidden = effects.hidden_residuals
        if policy_hidden is None:
            raise ValueError("policy hidden residuals are required")
        if (
            not torch.is_tensor(hidden)
            or hidden.shape != policy_hidden.shape
            or not hidden.is_floating_point()
            or not torch.isfinite(hidden).all()
        ):
            raise ValueError(
                "reference hidden residuals must be finite and match policy hidden width"
            )


def _prevalidate_hidden_reference(
    targets: ReferenceBatchTargets | None,
    layout: Stage2InterleavedLayout,
) -> None:
    """Reject an absent/malformed residual artifact before policy replay begins."""
    if not isinstance(targets, ReferenceBatchTargets):
        raise ValueError("fixed-reference residual objective requires reference targets")
    hidden = targets.hidden_residuals
    expected_prefix = layout.causal_pair_mask.shape
    if (
        not torch.is_tensor(hidden)
        or hidden.ndim != 4
        or hidden.shape[:3] != expected_prefix
        or hidden.shape[-1] <= 0
        or not hidden.is_floating_point()
        or not torch.isfinite(hidden).all()
    ):
        raise ValueError(
            "hidden residual reference targets are required with shape [B, J, K, H]"
        )


def _distributed_edge_counts(
    local_count: int,
    device: torch.device,
) -> tuple[int, int]:
    if not (distributed.is_available() and distributed.is_initialized()):
        return local_count, local_count
    global_count = torch.tensor(local_count, dtype=torch.long, device=device)
    global_max = global_count.clone()
    distributed.all_reduce(global_count, op=distributed.ReduceOp.SUM)
    distributed.all_reduce(global_max, op=distributed.ReduceOp.MAX)
    return int(global_count.item()), int(global_max.item())


def _validate_coordinated_counts(
    *,
    local_count: int,
    global_count: int,
    global_max: int,
) -> None:
    values = {
        "global edge count": global_count,
        "global maximum edge count": global_max,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} coordinator result must be a nonnegative integer")
    if global_count < local_count or global_max < local_count:
        raise ValueError("coordinated edge counts cannot be smaller than the local count")
    if global_count < global_max:
        raise ValueError("coordinated total edge count cannot be smaller than its maximum")
    if (global_count == 0) != (global_max == 0):
        raise ValueError("coordinated total and maximum edge counts disagree on emptiness")


__all__ = ["Stage2EvidenceWindowEngine", "Stage2EvidenceWindowResult"]
