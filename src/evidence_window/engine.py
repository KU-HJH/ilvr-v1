"""Condition orchestration for evidence-window preference objectives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import operator
from typing import Literal

import torch

from .config import EvidenceWindowConfig
from .interventions import apply_all_null, apply_zero_latent_loo, select_loo_spans
from .layout import InterleavedLayout
from .losses import (
    evidence_window_preference_loss,
    global_preference_loss,
    masked_relation_huber,
    masked_relation_jsd,
    relation_matrix,
    window_preference_loss,
)
from .metrics import contribution_summaries, global_outer_coefficient
from .scores import (
    evidence_window_margins,
    masked_token_sums,
    reference_adjusted_effect,
)


@dataclass(frozen=True)
class Condition:
    """One fixed-token-history latent condition."""

    kind: Literal["full", "all_null", "loo"]
    evidence_index: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"full", "all_null", "loo"}:
            raise ValueError(f"unsupported condition kind: {self.kind}")
        if self.kind != "loo":
            if self.evidence_index is not None:
                raise ValueError(f"{self.kind} condition cannot name an evidence index")
            return
        try:
            evidence_index = operator.index(self.evidence_index)
        except TypeError as exc:
            raise ValueError("loo condition requires an evidence index") from exc
        if evidence_index < 0:
            raise ValueError("loo evidence index must be nonnegative")
        object.__setattr__(self, "evidence_index", evidence_index)


@dataclass
class ModelConditionOutput:
    """Reduced model output retained by the condition cache."""

    token_logps: torch.Tensor
    hidden_states: torch.Tensor | None = None


@dataclass
class EvidenceWindowResult:
    """Auxiliary objective and sufficient statistics returned to the trainer."""

    total_loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ew_margins: torch.Tensor | None
    relations: dict[str, torch.Tensor]
    preference_pair_mask: torch.Tensor
    relation_pair_mask: torch.Tensor


ConditionForward = Callable[
    [str, Condition, torch.Tensor, bool],
    ModelConditionOutput,
]


class EvidenceWindowEngine:
    """Execute each required condition once and compose preference losses."""

    def __init__(self, config: EvidenceWindowConfig) -> None:
        if not isinstance(config, EvidenceWindowConfig):
            raise TypeError("config must be an EvidenceWindowConfig")
        config.validate()
        self.config = config

    def compute(
        self,
        layout: InterleavedLayout,
        latent_values: torch.Tensor,
        policy_full: ModelConditionOutput,
        forward: ConditionForward,
    ) -> EvidenceWindowResult:
        """Compute enabled preference objectives from a reused policy/full output."""
        self._validate_inputs(layout, latent_values, policy_full, forward)
        policy_full = self._prepare_output(
            policy_full,
            role="policy",
            need_hidden=self._needs_hidden("policy", Condition("full")),
            layout=layout,
            condition=Condition("full"),
        )
        active_mask = layout.evidence_token_mask.any(dim=1)
        cache: dict[tuple[str, str, int | None], ModelConditionOutput] = {
            ("policy", "full", None): policy_full
        }
        zero = policy_full.token_logps.sum() * 0.0
        total_loss = zero
        losses: dict[str, torch.Tensor] = {}
        metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        need_all_null = (
            self.config.preference.lambda_global > 0
            or self.config.preference.lambda_window > 0
            or self.config.diagnostics.global_saturation
        )
        need_relation = (
            self.config.relation.lambda_jsd > 0
            or self.config.relation.lambda_magnitude > 0
        )
        need_loo = self.config.preference.lambda_ew > 0 or need_relation
        loo_selected = self._selected_evidence(layout, need_loo)
        preference_pair_mask = layout.training_pair_mask & loo_selected.unsqueeze(-1)
        relation_pair_mask = layout.causal_pair_mask & loo_selected.unsqueeze(-1)

        def get(role: str, condition: Condition) -> ModelConditionOutput:
            key = (role, condition.kind, condition.evidence_index)
            cached = cache.get(key)
            if cached is not None:
                return cached
            if condition.kind == "full":
                values = latent_values
            elif condition.kind == "all_null":
                values = apply_all_null(latent_values)
            else:
                assert condition.evidence_index is not None
                values = apply_zero_latent_loo(
                    latent_values,
                    active_mask,
                    layout.evidence_token_mask,
                    condition.evidence_index,
                )
            need_hidden = self._needs_hidden(role, condition)
            if role == "policy":
                output = forward(role, condition, values, need_hidden)
            else:
                with torch.no_grad():
                    output = forward(role, condition, values, need_hidden)
            output = self._prepare_output(
                output,
                role=role,
                need_hidden=need_hidden,
                layout=layout,
                condition=condition,
            )
            cache[key] = output
            return output

        if need_all_null:
            visual_window_mask = layout.causal_pair_mask.any(dim=1)
            if visual_window_mask.any():
                policy_null = get("policy", Condition("all_null"))
                reference_full = get("reference", Condition("full"))
                reference_null = get("reference", Condition("all_null"))
                all_null_effect = reference_adjusted_effect(
                    policy_full.token_logps,
                    policy_null.token_logps,
                    reference_full.token_logps,
                    reference_null.token_logps,
                )
                window_margins = masked_token_sums(
                    all_null_effect,
                    layout.window_token_mask,
                )
                window_margins = torch.where(
                    visual_window_mask,
                    window_margins,
                    torch.zeros_like(window_margins),
                )
            else:
                window_margins = zero + policy_full.token_logps.new_zeros(
                    layout.window_valid.shape
                )
            self._require_finite("all-null window margins", window_margins, layout)
            self._add_all_null_results(
                window_margins,
                visual_window_mask,
                losses,
                metric_sums,
                layout,
            )
            if self.config.preference.lambda_global > 0:
                total_loss = total_loss + (
                    self.config.preference.lambda_global
                    * losses["global_preference"]
                )
            if self.config.preference.lambda_window > 0:
                total_loss = total_loss + (
                    self.config.preference.lambda_window
                    * losses["window_preference"]
                )

        ew_margins: torch.Tensor | None = None
        if self.config.preference.lambda_ew > 0:
            evidence_count = layout.evidence_token_mask.shape[1]
            token_effects: list[torch.Tensor] = []
            reference_full: ModelConditionOutput | None = None
            zero_effect = policy_full.token_logps * 0.0
            for evidence_index in range(evidence_count):
                if not loo_selected[:, evidence_index].any():
                    token_effects.append(zero_effect)
                    continue
                policy_loo = get("policy", Condition("loo", evidence_index))
                if reference_full is None:
                    reference_full = get("reference", Condition("full"))
                reference_loo = get("reference", Condition("loo", evidence_index))
                token_effects.append(
                    reference_adjusted_effect(
                        policy_full.token_logps,
                        policy_loo.token_logps,
                        reference_full.token_logps,
                        reference_loo.token_logps,
                    )
                )
            if token_effects:
                delta_by_evidence = torch.stack(token_effects, dim=1)
            else:
                delta_by_evidence = policy_full.token_logps.unsqueeze(1)[:, :0] * 0.0
            direct_window_masks = layout.window_token_mask.unsqueeze(1).expand(
                -1,
                evidence_count,
                -1,
                -1,
            )
            ew_margins = evidence_window_margins(
                delta_by_evidence,
                direct_window_masks,
            )
            ew_margins = torch.where(
                relation_pair_mask,
                ew_margins,
                torch.zeros_like(ew_margins),
            )
            self._require_finite("evidence-window margins", ew_margins, layout)
            ew_loss = evidence_window_preference_loss(
                ew_margins,
                preference_pair_mask,
                self.config.preference.beta,
            )
            self._require_finite("evidence-window preference loss", ew_loss, layout)
            losses["ew_dpo"] = ew_loss
            total_loss = total_loss + self.config.preference.lambda_ew * ew_loss
            metric_sums["preference/ew_margin_mean"] = self._masked_sum_and_count(
                ew_margins,
                preference_pair_mask,
            )

        relations: dict[str, torch.Tensor] = {}
        if need_relation:
            if relation_pair_mask.any():
                evidence_count = layout.evidence_token_mask.shape[1]
                window_lengths = layout.window_token_mask.sum(dim=-1)
                policy_full_sums = masked_token_sums(
                    policy_full.token_logps,
                    layout.window_token_mask,
                )
                ema_full = get("ema", Condition("full"))
                ema_full_sums = masked_token_sums(
                    ema_full.token_logps,
                    layout.window_token_mask,
                )
                policy_loo_sums: list[torch.Tensor] = []
                ema_loo_sums: list[torch.Tensor] = []
                for evidence_index in range(evidence_count):
                    if not loo_selected[:, evidence_index].any():
                        policy_loo_sums.append(policy_full_sums)
                        ema_loo_sums.append(ema_full_sums)
                        continue
                    policy_loo = get("policy", Condition("loo", evidence_index))
                    ema_loo = get("ema", Condition("loo", evidence_index))
                    policy_loo_sums.append(
                        masked_token_sums(
                            policy_loo.token_logps,
                            layout.window_token_mask,
                        )
                    )
                    ema_loo_sums.append(
                        masked_token_sums(
                            ema_loo.token_logps,
                            layout.window_token_mask,
                        )
                    )
                policy_loo_by_evidence = torch.stack(policy_loo_sums, dim=1)
                ema_loo_by_evidence = torch.stack(ema_loo_sums, dim=1)
                student_relations = relation_matrix(
                    policy_full_sums,
                    policy_loo_by_evidence,
                    window_lengths,
                    relation_pair_mask,
                )
                teacher_relations = relation_matrix(
                    ema_full_sums,
                    ema_loo_by_evidence,
                    window_lengths,
                    relation_pair_mask,
                ).detach()
            else:
                student_relations = policy_full.token_logps.sum() * 0.0 + torch.zeros(
                    relation_pair_mask.shape,
                    dtype=policy_full.token_logps.dtype,
                    device=policy_full.token_logps.device,
                )
                teacher_relations = torch.zeros_like(student_relations).detach()

            self._require_finite("student relation matrix", student_relations, layout)
            self._require_finite("teacher relation matrix", teacher_relations, layout)
            relations = {
                "student": student_relations,
                "teacher": teacher_relations,
            }
            self._add_relation_results(
                student_relations,
                teacher_relations,
                relation_pair_mask,
                losses,
                metric_sums,
                layout,
            )
            if self.config.relation.lambda_jsd > 0:
                total_loss = total_loss + (
                    self.config.relation.lambda_jsd * losses["relation_jsd"]
                )
            if self.config.relation.lambda_magnitude > 0:
                total_loss = total_loss + (
                    self.config.relation.lambda_magnitude
                    * losses["relation_magnitude"]
                )

        if need_loo:
            selected_per_row = loo_selected.sum(dim=1)
            metric_sums["loo/selected_span_count"] = (
                selected_per_row.detach().sum(),
                selected_per_row.new_tensor(selected_per_row.shape[0]).detach(),
            )

        self._require_finite("total evidence-window loss", total_loss, layout)
        return EvidenceWindowResult(
            total_loss=total_loss,
            losses=losses,
            metric_sums=metric_sums,
            ew_margins=ew_margins,
            relations=relations,
            preference_pair_mask=preference_pair_mask,
            relation_pair_mask=relation_pair_mask,
        )

    def _selected_evidence(
        self,
        layout: InterleavedLayout,
        need_loo: bool,
    ) -> torch.Tensor:
        if not need_loo:
            return layout.evidence_valid.to(dtype=torch.bool)
        eligible = layout.evidence_valid & layout.causal_pair_mask.any(dim=-1)
        selections = select_loo_spans(eligible, self.config.loo.max_spans_per_example)
        selected = torch.zeros_like(eligible)
        for batch_index, indices in enumerate(selections):
            if indices:
                selected[batch_index, indices] = True
        return selected

    def _needs_hidden(self, role: str, condition: Condition) -> bool:
        return (
            self.config.residual.lambda_logit > 0
            and role in {"policy", "ema"}
            and condition.kind in {"full", "loo"}
        )

    def _add_all_null_results(
        self,
        window_margins: torch.Tensor,
        valid: torch.Tensor,
        losses: dict[str, torch.Tensor],
        metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        layout: InterleavedLayout,
    ) -> None:
        beta = self.config.preference.beta
        if self.config.preference.lambda_global > 0:
            loss = global_preference_loss(window_margins, valid, beta)
            self._require_finite("global preference loss", loss, layout)
            losses["global_preference"] = loss
        if self.config.preference.lambda_window > 0:
            loss = window_preference_loss(window_margins, valid, beta)
            self._require_finite("window preference loss", loss, layout)
            losses["window_preference"] = loss

        global_margins = (window_margins * valid.to(window_margins.dtype)).sum(dim=1)
        row_valid = valid.any(dim=1)
        metric_sums["preference/global_margin"] = self._masked_sum_and_count(
            global_margins,
            row_valid,
        )
        outer = global_outer_coefficient(global_margins, beta)
        metric_sums["preference/global_outer_weight"] = self._masked_sum_and_count(
            outer,
            row_valid,
        )
        metric_sums["preference/window_margin_mean"] = self._masked_sum_and_count(
            window_margins,
            valid,
        )

    def _add_relation_results(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        valid: torch.Tensor,
        losses: dict[str, torch.Tensor],
        metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        layout: InterleavedLayout,
    ) -> None:
        if self.config.relation.lambda_jsd > 0:
            loss, _ = masked_relation_jsd(
                student,
                teacher,
                valid,
                self.config.relation.temperature,
            )
            self._require_finite("relation JSD loss", loss, layout)
            losses["relation_jsd"] = loss
        if self.config.relation.lambda_magnitude > 0:
            loss, _ = masked_relation_huber(
                student,
                teacher,
                valid,
                self.config.relation.huber_delta,
            )
            self._require_finite("relation magnitude loss", loss, layout)
            losses["relation_magnitude"] = loss

        window_valid = valid.any(dim=1)
        for role, values in (("student", student), ("teacher", teacher)):
            if values.shape[1] == 0:
                zeros = values.new_zeros(window_valid.shape)
                metric_sums[f"relation/{role}/top1_ratio"] = self._masked_sum_and_count(
                    zeros,
                    window_valid,
                )
                metric_sums[
                    f"relation/{role}/normalized_entropy"
                ] = self._masked_sum_and_count(zeros, window_valid)
                metric_sums[
                    f"relation/{role}/above_threshold_count"
                ] = self._masked_sum_and_count(zeros, window_valid)
                continue
            summaries = contribution_summaries(
                values,
                valid,
                self.config.diagnostics.contribution_threshold,
            )
            metric_sums[f"relation/{role}/top1_ratio"] = self._masked_sum_and_count(
                summaries.top1_ratio,
                window_valid,
            )
            metric_sums[
                f"relation/{role}/normalized_entropy"
            ] = self._masked_sum_and_count(
                summaries.normalized_entropy,
                summaries.entropy_valid,
            )
            metric_sums[
                f"relation/{role}/above_threshold_count"
            ] = self._masked_sum_and_count(
                summaries.above_threshold_count,
                window_valid,
            )

    @staticmethod
    def _masked_sum_and_count(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        detached_values = values.detach()
        detached_mask = mask.detach().to(dtype=detached_values.dtype)
        return (
            (detached_values * detached_mask).sum(),
            detached_mask.sum(),
        )

    def _prepare_output(
        self,
        output: ModelConditionOutput,
        *,
        role: str,
        need_hidden: bool,
        layout: InterleavedLayout,
        condition: Condition,
    ) -> ModelConditionOutput:
        if not isinstance(output, ModelConditionOutput):
            raise TypeError("condition forward must return ModelConditionOutput")
        self._validate_model_output(output, layout)
        self._require_finite(
            f"{role}/{condition.kind} condition output",
            output.token_logps,
            layout,
        )
        token_logps = output.token_logps if role == "policy" else output.token_logps.detach()
        hidden_states = output.hidden_states if need_hidden else None
        if hidden_states is not None and role != "policy":
            hidden_states = hidden_states.detach()
        return ModelConditionOutput(token_logps=token_logps, hidden_states=hidden_states)

    @staticmethod
    def _validate_model_output(
        output: ModelConditionOutput,
        layout: InterleavedLayout,
    ) -> None:
        token_logps = output.token_logps
        expected = (
            layout.window_token_mask.shape[0],
            layout.window_token_mask.shape[-1],
        )
        if not torch.is_tensor(token_logps) or token_logps.shape != expected:
            raise ValueError(f"token_logps must have shape {expected}")
        if not token_logps.is_floating_point():
            raise ValueError("token_logps must be floating point")
        if token_logps.device != layout.window_token_mask.device:
            raise ValueError("token_logps and layout must be on the same device")
        if output.hidden_states is not None:
            if not torch.is_tensor(output.hidden_states):
                raise ValueError("hidden_states must be a tensor when retained")
            if output.hidden_states.device != token_logps.device:
                raise ValueError("hidden_states and token_logps must be on the same device")

    @staticmethod
    def _validate_inputs(
        layout: InterleavedLayout,
        latent_values: torch.Tensor,
        policy_full: ModelConditionOutput,
        forward: ConditionForward,
    ) -> None:
        if not isinstance(layout, InterleavedLayout):
            raise TypeError("layout must be an InterleavedLayout")
        if not torch.is_tensor(latent_values) or latent_values.ndim != 3:
            raise ValueError("latent_values must have shape [B, N, D]")
        batch_size, evidence_count, token_count = layout.evidence_token_mask.shape
        window_count = layout.window_token_mask.shape[1]
        expected_shapes = {
            "window_token_mask": (batch_size, window_count, token_count),
            "evidence_valid": (batch_size, evidence_count),
            "window_valid": (batch_size, window_count),
            "causal_pair_mask": (batch_size, evidence_count, window_count),
            "training_pair_mask": (batch_size, evidence_count, window_count),
            "native_predecessor": (batch_size, window_count),
        }
        for name, expected in expected_shapes.items():
            value = getattr(layout, name)
            if not torch.is_tensor(value) or value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if value.device != layout.evidence_token_mask.device:
                raise ValueError("layout tensors must be on the same device")
        if latent_values.shape[0] != batch_size:
            raise ValueError("latent_values batch dimension must match layout")
        if latent_values.device != layout.evidence_token_mask.device:
            raise ValueError("latent_values and layout must be on the same device")
        if not callable(forward):
            raise TypeError("forward must be callable")
        EvidenceWindowEngine._validate_model_output(policy_full, layout)

    def _require_finite(
        self,
        name: str,
        value: torch.Tensor,
        layout: InterleavedLayout,
    ) -> None:
        if torch.isfinite(value).all():
            return
        batch_size, evidence_count, _ = layout.evidence_token_mask.shape
        window_count = layout.window_token_mask.shape[1]
        preference_pairs = int(layout.training_pair_mask.sum().item())
        relation_pairs = int(layout.causal_pair_mask.sum().item())
        raise FloatingPointError(
            f"non-finite {name}; layout counts: batch={batch_size}, "
            f"evidence={evidence_count}, windows={window_count}, "
            f"preference_pairs={preference_pairs}, relation_pairs={relation_pairs}"
        )

__all__ = [
    "Condition",
    "ConditionForward",
    "EvidenceWindowEngine",
    "EvidenceWindowResult",
    "ModelConditionOutput",
]
