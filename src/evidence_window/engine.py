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
    chunked_logit_residual_loss,
    evidence_window_preference_loss,
    global_preference_loss,
    masked_relation_huber,
    masked_relation_jsd,
    relation_matrix,
    select_residual_tokens,
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
LMHeadWeight = Callable[[Literal["policy", "ema"]], torch.Tensor]


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
        *,
        lm_head_weight: LMHeadWeight | None = None,
    ) -> EvidenceWindowResult:
        """Compute enabled preference objectives from a reused policy/full output."""
        self._validate_inputs(layout, latent_values, policy_full, forward)
        need_residual = self.config.residual.lambda_logit > 0
        active_mask = layout.evidence_token_mask.any(dim=1)
        need_all_null = (
            self.config.preference.lambda_global > 0
            or self.config.preference.lambda_window > 0
            or self.config.diagnostics.global_saturation
        )
        need_relation = (
            self.config.relation.lambda_jsd > 0
            or self.config.relation.lambda_magnitude > 0
            or self.config.diagnostics.matrix_dump_interval > 0
        )
        need_loo = self.config.preference.lambda_ew > 0 or need_relation or need_residual
        loo_selected = self._selected_evidence(layout, need_loo)
        preference_pair_mask = layout.training_pair_mask & loo_selected.unsqueeze(-1)
        relation_pair_mask = layout.causal_pair_mask & loo_selected.unsqueeze(-1)
        residual_selections: dict[int, list[tuple[int, torch.Tensor]]] = {}
        residual_tokens_per_row = torch.zeros(
            preference_pair_mask.shape[0],
            dtype=torch.long,
            device=preference_pair_mask.device,
        )
        if need_residual:
            residual_selections, residual_tokens_per_row = (
                self._selected_residual_predictors(layout, preference_pair_mask)
            )
        if residual_selections and not callable(lm_head_weight):
            raise ValueError("active logit residual requires an LM-head weight provider")

        def needs_hidden(role: str, condition: Condition) -> bool:
            if role not in {"policy", "ema"} or not residual_selections:
                return False
            if condition.kind == "full":
                return True
            return (
                condition.kind == "loo"
                and condition.evidence_index in residual_selections
            )

        policy_full = self._prepare_output(
            policy_full,
            role="policy",
            need_hidden=needs_hidden("policy", Condition("full")),
            layout=layout,
            condition=Condition("full"),
        )
        cache: dict[tuple[str, str, int | None], ModelConditionOutput] = {
            ("policy", "full", None): policy_full
        }
        zero = policy_full.token_logps.sum() * 0.0
        total_loss = zero
        losses: dict[str, torch.Tensor] = {}
        metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

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
            need_hidden = needs_hidden(role, condition)
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

        if need_residual:
            if residual_selections:
                assert lm_head_weight is not None
                policy_full_hidden = self._required_hidden_states(
                    policy_full,
                    role="policy",
                    condition=Condition("full"),
                )
                ema_full = get("ema", Condition("full"))
                ema_full_hidden = self._required_hidden_states(
                    ema_full,
                    role="ema",
                    condition=Condition("full"),
                )
                student_residuals: list[torch.Tensor] = []
                teacher_residuals: list[torch.Tensor] = []
                for evidence_index, selections in residual_selections.items():
                    policy_loo = get("policy", Condition("loo", evidence_index))
                    ema_loo = get("ema", Condition("loo", evidence_index))
                    policy_loo_hidden = self._required_hidden_states(
                        policy_loo,
                        role="policy",
                        condition=Condition("loo", evidence_index),
                    )
                    ema_loo_hidden = self._required_hidden_states(
                        ema_loo,
                        role="ema",
                        condition=Condition("loo", evidence_index),
                    )
                    self._validate_hidden_width(
                        policy_loo_hidden,
                        full_hidden_states=policy_full_hidden,
                        role="policy",
                        condition=Condition("loo", evidence_index),
                    )
                    self._validate_hidden_width(
                        ema_loo_hidden,
                        full_hidden_states=ema_full_hidden,
                        role="ema",
                        condition=Condition("loo", evidence_index),
                    )
                    for batch_index, predictor_positions in selections:
                        student_residuals.append(
                            policy_full_hidden[batch_index].index_select(
                                0, predictor_positions
                            )
                            - policy_loo_hidden[batch_index].index_select(
                                0, predictor_positions
                            )
                        )
                        teacher_residuals.append(
                            ema_full_hidden[batch_index].index_select(
                                0, predictor_positions
                            )
                            - ema_loo_hidden[batch_index].index_select(
                                0, predictor_positions
                            )
                        )
                    policy_loo.hidden_states = None
                    ema_loo.hidden_states = None
                student_dh = torch.cat(student_residuals, dim=0)
                teacher_dh = torch.cat(teacher_residuals, dim=0)
                policy_full.hidden_states = None
                ema_full.hidden_states = None
                residual_loss = chunked_logit_residual_loss(
                    student_dh,
                    teacher_dh,
                    lm_head_weight("policy"),
                    lm_head_weight("ema"),
                    self.config.residual.metric,
                    self.config.residual.vocab_chunk_size,
                    self.config.residual.huber_delta,
                )
            else:
                residual_loss = zero
            self._require_finite("logit residual loss", residual_loss, layout)
            losses["logit_residual"] = residual_loss
            total_loss = total_loss + self.config.residual.lambda_logit * residual_loss
            residual_count = residual_tokens_per_row.detach().sum()
            self._add_loss_metric(
                metric_sums,
                "logit_residual",
                residual_loss,
                residual_count,
            )
            metric_sums["residual/selected_token_count"] = (
                residual_count,
                residual_tokens_per_row.new_tensor(
                    residual_tokens_per_row.shape[0]
                ).detach(),
            )

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
            self._add_loss_metric(
                metric_sums,
                "ew_dpo",
                ew_loss,
                preference_pair_mask.detach().sum(),
            )
            self._add_ordinal_metrics(
                metric_sums,
                "preference/ew_margin",
                ew_margins,
                preference_pair_mask,
                layout.native_predecessor,
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

    def _selected_residual_predictors(
        self,
        layout: InterleavedLayout,
        pair_mask: torch.Tensor,
    ) -> tuple[dict[int, list[tuple[int, torch.Tensor]]], torch.Tensor]:
        """Map adjacent pairs to capped causal predictor positions ``t - 1``."""
        selected: dict[int, list[tuple[int, torch.Tensor]]] = {}
        selected_per_row = torch.zeros(
            pair_mask.shape[0],
            dtype=torch.long,
            device=pair_mask.device,
        )
        predictor_eligible = torch.ones(
            layout.window_token_mask.shape[-1],
            dtype=torch.bool,
            device=pair_mask.device,
        )
        if predictor_eligible.numel():
            predictor_eligible[0] = False
        for batch_index, evidence_index, window_index in pair_mask.nonzero().tolist():
            target_mask = (
                layout.window_token_mask[batch_index, window_index]
                & predictor_eligible
            )
            target_positions = select_residual_tokens(
                target_mask,
                self.config.residual.max_tokens_per_window,
            )
            selected_per_row[batch_index] += target_positions.numel()
            if target_positions.numel():
                selected.setdefault(evidence_index, []).append(
                    (batch_index, target_positions - 1)
                )
        return selected, selected_per_row

    @staticmethod
    def _required_hidden_states(
        output: ModelConditionOutput,
        *,
        role: str,
        condition: Condition,
    ) -> torch.Tensor:
        if output.hidden_states is None:
            raise ValueError(
                f"{role}/{condition.kind} must retain final normalized hidden states"
            )
        return output.hidden_states

    @staticmethod
    def _validate_hidden_width(
        hidden_states: torch.Tensor,
        *,
        full_hidden_states: torch.Tensor,
        role: str,
        condition: Condition,
    ) -> None:
        if hidden_states.shape[-1] != full_hidden_states.shape[-1]:
            assert condition.kind == "loo"
            raise ValueError(
                f"{role}/loo({condition.evidence_index}) hidden width "
                f"{hidden_states.shape[-1]} must match {role}/full width "
                f"{full_hidden_states.shape[-1]}"
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
            self._add_loss_metric(
                metric_sums,
                "global_preference",
                loss,
                valid.any(dim=1).detach().sum(),
            )
        if self.config.preference.lambda_window > 0:
            loss = window_preference_loss(window_margins, valid, beta)
            self._require_finite("window preference loss", loss, layout)
            losses["window_preference"] = loss
            self._add_loss_metric(
                metric_sums,
                "window_preference",
                loss,
                valid.detach().sum(),
            )

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
        self._add_ordinal_metrics(
            metric_sums,
            "preference/window_margin",
            window_margins,
            valid,
            layout.native_predecessor,
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
            loss, count = masked_relation_jsd(
                student,
                teacher,
                valid,
                self.config.relation.temperature,
            )
            self._require_finite("relation JSD loss", loss, layout)
            losses["relation_jsd"] = loss
            self._add_loss_metric(
                metric_sums,
                "relation_jsd",
                loss,
                count.detach(),
            )
        if self.config.relation.lambda_magnitude > 0:
            loss, count = masked_relation_huber(
                student,
                teacher,
                valid,
                self.config.relation.huber_delta,
            )
            self._require_finite("relation magnitude loss", loss, layout)
            losses["relation_magnitude"] = loss
            self._add_loss_metric(
                metric_sums,
                "relation_magnitude",
                loss,
                count.detach(),
            )

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

    @staticmethod
    def _add_loss_metric(
        metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        name: str,
        loss: torch.Tensor,
        count: torch.Tensor,
    ) -> None:
        detached_count = count.detach()
        metric_sums[f"loss/{name}"] = (
            loss.detach() * detached_count.to(dtype=loss.dtype),
            detached_count,
        )

    @classmethod
    def _add_ordinal_metrics(
        cls,
        metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        prefix: str,
        values: torch.Tensor,
        mask: torch.Tensor,
        native_predecessor: torch.Tensor,
    ) -> None:
        ordinals = sorted(
            ordinal
            for ordinal in native_predecessor.detach().unique().tolist()
            if ordinal >= 0
        )
        for ordinal in ordinals:
            ordinal_mask = native_predecessor.eq(ordinal)
            if mask.ndim == 3:
                ordinal_mask = ordinal_mask.unsqueeze(1).expand_as(mask)
            ordinal_mask = mask & ordinal_mask
            if not ordinal_mask.any().item():
                continue
            metric_sums[f"{prefix}/{ordinal}"] = cls._masked_sum_and_count(
                values,
                ordinal_mask,
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
        if need_hidden:
            if hidden_states is None:
                raise ValueError(
                    f"{role}/{condition.kind} must retain final normalized hidden states"
                )
            expected_hidden_prefix = (
                layout.window_token_mask.shape[0],
                layout.window_token_mask.shape[-1],
            )
            if (
                hidden_states.ndim != 3
                or hidden_states.shape[:2] != expected_hidden_prefix
                or not hidden_states.is_floating_point()
            ):
                raise ValueError(
                    "hidden_states must be floating point with shape "
                    f"{expected_hidden_prefix + ('H',)}"
                )
            self._require_finite(
                f"{role}/{condition.kind} hidden states",
                hidden_states,
                layout,
            )
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
    "LMHeadWeight",
    "ModelConditionOutput",
]
