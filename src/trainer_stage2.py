"""Explicit CE plus optional evidence-edge trainer for ILVR Stage 2."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any

import torch
import torch.distributed as distributed
from trl import SFTTrainer

from evidence_window.layout import TokenIds
from evidence_window.artifacts import ArtifactIdentity, ReferenceBatchTargets
from evidence_window.metrics import MatrixJsonlWriter, reduce_sufficient_statistics
from evidence_window.scores import chunked_causal_target_logps
from evidence_window.stage2_config import Stage2EvidenceWindowConfig
from evidence_window.stage2_effects import Stage2ConditionOutput
from evidence_window.stage2_engine import Stage2EvidenceWindowEngine, Stage2EvidenceWindowResult
from evidence_window.stage2_layout import extract_stage2_layout, validate_plan_types
from evidence_window.stage2_reference import (
    OnlineFrozenStage2Reference,
    build_stage2_reference_provider,
)


class CustomTrainerStage2(SFTTrainer):
    """Train self-generated Stage-2 latents with CE and optional edge credit."""

    def __init__(
        self,
        *args,
        evidence_window_config: Stage2EvidenceWindowConfig | None = None,
        latent_token_ids: Mapping[str, Any] | TokenIds | None = None,
        reference_provider=None,
        reference_identity: ArtifactIdentity | None = None,
        reference_model_factory=None,
        deepspeed_zero_stage: int | None = None,
        policy_initialization_checkpoint: str | None = None,
        **kwargs,
    ):
        config = (
            Stage2EvidenceWindowConfig.disabled()
            if evidence_window_config is None
            else evidence_window_config
        )
        if not isinstance(config, Stage2EvidenceWindowConfig):
            raise TypeError("evidence_window_config must be a Stage2EvidenceWindowConfig")
        config.validate()
        online_enabled = (
            not config.is_disabled and config.reference.mode == "online_frozen"
        )
        if online_enabled:
            if deepspeed_zero_stage is None:
                raise ValueError("online frozen Stage-2 reference requires a ZeRO stage")
            config.validate_runtime(deepspeed_zero_stage)
            if not callable(reference_model_factory):
                raise TypeError("online frozen Stage-2 reference requires a model factory")
            if not policy_initialization_checkpoint:
                raise ValueError(
                    "online frozen Stage-2 reference requires the policy initialization "
                    "checkpoint identity"
                )
            if reference_provider is not None or reference_identity is not None:
                raise ValueError(
                    "online frozen Stage-2 reference does not accept an artifact "
                    "provider or identity"
                )
        elif any(
            value is not None
            for value in (
                reference_model_factory,
                deepspeed_zero_stage,
                policy_initialization_checkpoint,
            )
        ):
            raise ValueError(
                "online reference construction inputs require enabled online_frozen mode"
            )
        self.evidence_window_config = config
        super().__init__(*args, **kwargs)

        # Disabled Stage 2 deliberately has no auxiliary runtime objects. This
        # keeps its hot path identical to the established CE-only baseline.
        if config.is_disabled:
            if reference_provider is not None or reference_identity is not None:
                raise ValueError(
                    "disabled Stage 2 does not accept a reference provider or identity"
                )
            return
        self._token_ids = _resolve_token_ids(latent_token_ids)
        self._ew_engine = Stage2EvidenceWindowEngine(config)
        if config.reference.mode == "none":
            if reference_provider is not None or reference_identity is not None:
                raise ValueError(
                    "reference-free Stage 2 does not accept a reference provider or identity"
                )
            self._stage2_reference_provider = None
        elif config.reference.mode == "online_frozen":
            self._stage2_reference_provider = build_stage2_reference_provider(
                config,
                model_factory=reference_model_factory,
                accelerator=self.accelerator,
                zero_stage=deepspeed_zero_stage,
                policy_checkpoint=policy_initialization_checkpoint,
            )
        else:
            self._stage2_reference_provider = (
                build_stage2_reference_provider(
                    config, expected_identity=reference_identity
                )
                if reference_provider is None
                else reference_provider
            )
            if not callable(getattr(self._stage2_reference_provider, "get_batch", None)):
                raise TypeError("reference_provider must provide get_batch")
        self._ew_pending_metrics: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {
            "train": {},
            "eval": {},
        }
        self._ew_matrix_writer: MatrixJsonlWriter | None = None
        self._ew_matrix_observation_counter = 0
        self._ew_last_matrix_dump_step: int | None = None
        logging.info(
            "Stage-2 evidence-window enabled: reference=%s, intervention=%s, "
            "reduction=%s, lambda_ew=%s, beta=%s",
            config.reference.mode,
            config.loo.intervention,
            config.preference.contribution_reduction,
            config.preference.lambda_ew,
            config.preference.beta,
        )
        if config.reference.mode == "online_frozen":
            logging.info(
                "Stage-2 online reference owns one additional frozen full model per "
                "rank under ZeRO-%s; precomputed targets use less training memory",
                deepspeed_zero_stage,
            )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        model_inputs = dict(inputs)
        sample_indices = model_inputs.pop("ew_sample_indices", None)
        plan_types = model_inputs.pop("ew_plan_types", None)

        # Establish ordinary Stage-2 CE first. The disabled branch returns
        # here without parsing or auxiliary replay.
        outputs = model(**model_inputs)
        ce_loss = outputs.loss
        if self.evidence_window_config.is_disabled:
            return (ce_loss, outputs) if return_outputs else ce_loss

        progress = self._training_progress()
        active_objectives = self._active_auxiliary_objectives(progress)
        if not active_objectives:
            self._record_ce_metric(ce_loss, model_inputs, model.training)
            return (ce_loss, outputs) if return_outputs else ce_loss

        required = {"input_ids", "attention_mask", "labels"}
        missing = required - set(model_inputs)
        if missing:
            raise ValueError(
                "enabled Stage-2 evidence-window loss requires model input(s): "
                + ", ".join(sorted(missing))
            )
        if (sample_indices is None) != (plan_types is None):
            raise ValueError("Stage-2 plan types and sample indices must be provided together")

        layout = extract_stage2_layout(
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs["labels"],
            self._token_ids,
        )
        if plan_types is not None:
            validate_plan_types(layout, plan_types, sample_indices)

        labels = model_inputs["labels"]
        auxiliary_inputs = dict(model_inputs)
        auxiliary_inputs.pop("labels", None)

        def policy_forward(
            attention_block_mask: torch.Tensor | None,
            need_hidden: bool,
        ) -> Stage2ConditionOutput:
            condition_outputs = model(
                **auxiliary_inputs,
                attention_block_mask=attention_block_mask,
                attention_backend_override="sdpa",
                output_hidden_states=need_hidden,
                return_dict=True,
                use_cache=False,
            )
            shifted_logps = chunked_causal_target_logps(condition_outputs.logits, labels)
            token_logps = torch.cat(
                (shifted_logps.new_zeros((shifted_logps.shape[0], 1)), shifted_logps),
                dim=1,
            )
            hidden = _final_hidden_state(condition_outputs) if need_hidden else None
            return Stage2ConditionOutput(token_logps, hidden)

        reference_targets = None
        active_reference_objectives = (
            active_objectives & self._reference_dependent_objective_names()
        )
        if (
            self._stage2_reference_provider is not None
            and active_reference_objectives
        ):
            if sample_indices is None:
                raise ValueError(
                    "fixed-reference Stage-2 requires deterministic ew_sample_indices"
                )
            reference_targets = _coordinated_reference_lookup(
                self._stage2_reference_provider,
                sample_indices,
                layout,
                require_hidden_residuals=self.evidence_window_config._residual_enabled(),
                model_inputs=(
                    auxiliary_inputs
                    if isinstance(
                        self._stage2_reference_provider,
                        OnlineFrozenStage2Reference,
                    )
                    else None
                ),
                labels=labels,
                contribution_reduction=(
                    self.evidence_window_config.preference.contribution_reduction
                ),
                max_evidence_edges_per_window=(
                    self.evidence_window_config.loo.max_evidence_edges_per_window
                ),
            )
        auxiliary = self._ew_engine.compute(
            layout,
            policy_forward,
            progress=progress,
            reference_targets=reference_targets,
        )
        split = "train" if model.training else "eval"
        batch_count = ce_loss.new_tensor(float(model_inputs["input_ids"].shape[0]))
        metrics = dict(auxiliary.metric_sums)
        metrics["loss/ce"] = (ce_loss.detach() * batch_count, batch_count)
        self._accumulate_evidence_window_metrics(metrics, split=split)
        self._maybe_dump_evidence_window_matrices(auxiliary)
        loss = ce_loss + self._globally_scaled_auxiliary(ce_loss, auxiliary)
        return (loss, outputs) if return_outputs else loss

    def _active_auxiliary_objectives(self, progress: float) -> frozenset[str]:
        """Return scheduled objectives that require work at this step.

        Tasks 11/12 can attach relation and residual computation to the same
        scheduling gate without reintroducing reference replay during CE-only
        warmup.
        """
        return frozenset(
            name
            for name in ("ew", "relation", "residual")
            if self.evidence_window_config.effective_weight(name, progress) > 0.0
        )

    def _reference_dependent_objective_names(self) -> frozenset[str]:
        names = {"relation", "residual"}
        if self.evidence_window_config.reference.mode != "none":
            names.add("ew")
        return frozenset(names)

    def _record_ce_metric(
        self,
        ce_loss: torch.Tensor,
        model_inputs: Mapping[str, Any],
        training: bool,
    ) -> None:
        input_ids = model_inputs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim < 1:
            raise ValueError("enabled Stage-2 requires batched input_ids")
        batch_count = ce_loss.new_tensor(float(input_ids.shape[0]))
        self._accumulate_evidence_window_metrics(
            {"loss/ce": (ce_loss.detach() * batch_count, batch_count)},
            split="train" if training else "eval",
        )

    def _training_progress(self) -> float:
        state = getattr(self, "state", None)
        step = getattr(state, "global_step", 0)
        max_steps = getattr(state, "max_steps", 0)
        if isinstance(step, bool) or not isinstance(step, int):
            step = 0
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            return 0.0
        return min(max(float(step) / float(max_steps), 0.0), 1.0)

    @staticmethod
    def _globally_scaled_auxiliary(
        ce_loss: torch.Tensor,
        auxiliary: Stage2EvidenceWindowResult,
    ) -> torch.Tensor:
        statistics = auxiliary.metric_sums.get("loss/ew_pref")
        local_count = (
            torch.zeros((), device=ce_loss.device, dtype=torch.float64)
            if statistics is None
            else statistics[1].detach().to(device=ce_loss.device, dtype=torch.float64)
        )
        global_count = local_count.clone()
        world_size = 1
        if distributed.is_available() and distributed.is_initialized():
            world_size = distributed.get_world_size()
            distributed.all_reduce(global_count, op=distributed.ReduceOp.SUM)
        if global_count.item() <= 0:
            return auxiliary.auxiliary_loss * 0.0
        scale = (world_size * local_count / global_count).to(
            dtype=auxiliary.auxiliary_loss.dtype
        )
        return auxiliary.auxiliary_loss * scale

    def _accumulate_evidence_window_metrics(
        self,
        metric_sums: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        split: str,
    ) -> None:
        if split not in {"train", "eval"}:
            raise ValueError("metric split must be 'train' or 'eval'")
        pending = self._ew_pending_metrics[split]
        for key, (numerator, count) in metric_sums.items():
            numerator = numerator.detach().to(dtype=torch.float64)
            count = count.detach().to(device=numerator.device, dtype=torch.float64)
            previous = pending.get(key)
            pending[key] = (
                numerator if previous is None else previous[0] + numerator,
                count if previous is None else previous[1] + count,
            )

    def _reduce_evidence_window_metrics(self, split: str) -> dict[str, float]:
        pending = self._ew_pending_metrics[split]
        distributed_active = distributed.is_available() and distributed.is_initialized()
        if not pending and not distributed_active:
            return {}
        keys = set(pending)
        if distributed_active:
            gathered: list[list[str] | None] = [None] * distributed.get_world_size()
            distributed.all_gather_object(gathered, sorted(keys))
            keys = {key for rank_keys in gathered if rank_keys for key in rank_keys}
        if not keys:
            return {}
        device = (
            next(iter(pending.values()))[0].device
            if pending
            else next(self.model.parameters()).device
        )
        zero = torch.zeros((), dtype=torch.float64, device=device)
        reduced = {}
        for key in sorted(keys):
            numerator, count = pending.get(key, (zero, zero))
            statistic = reduce_sufficient_statistics(numerator, count)
            if statistic.count.item() > 0:
                reduced[key] = statistic.mean.item()
        return reduced

    def _maybe_dump_evidence_window_matrices(
        self,
        auxiliary: Stage2EvidenceWindowResult,
    ) -> None:
        diagnostics = self.evidence_window_config.diagnostics
        interval = diagnostics.matrix_dump_interval
        state = getattr(self, "state", None)
        step = getattr(state, "global_step", None)
        if (
            interval <= 0
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or step % interval != 0
            or step == self._ew_last_matrix_dump_step
            or not auxiliary.effects.pair_mask.any()
        ):
            return
        if self._ew_matrix_writer is None:
            rank = getattr(getattr(self, "accelerator", None), "process_index", None)
            if rank is None:
                rank = distributed.get_rank() if distributed.is_initialized() else 0
            self._ew_matrix_writer = MatrixJsonlWriter(
                diagnostics.matrix_dump_dir,
                rank=int(rank),
            )
        for batch_index in range(auxiliary.effects.contributions.shape[0]):
            self._ew_matrix_writer.write(
                step,
                self._ew_matrix_observation_counter,
                auxiliary.effects.contributions[batch_index].detach(),
                auxiliary.effects.pair_mask[batch_index].detach(),
            )
            self._ew_matrix_observation_counter += 1
        self._ew_last_matrix_dump_step = step

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self.evidence_window_config.is_disabled:
            super().log(logs, start_time)
            return
        is_evaluation = any(key.startswith("eval_") for key in logs)
        split = "eval" if is_evaluation else "train"
        combined = dict(logs)
        for key, value in self._reduce_evidence_window_metrics(split).items():
            combined[f"eval_{key}" if is_evaluation else key] = value
        super().log(combined, start_time)
        self._ew_pending_metrics[split].clear()


def _final_hidden_state(outputs) -> torch.Tensor:
    hidden = getattr(outputs, "hidden_states", None)
    if isinstance(hidden, (tuple, list)):
        hidden = hidden[-1] if hidden else None
    if not torch.is_tensor(hidden):
        raise ValueError("Stage-2 residual output requires a final hidden-state tensor")
    return hidden


def _coordinated_reference_lookup(
    provider,
    sample_indices,
    layout: Stage2InterleavedLayout,
    *,
    require_hidden_residuals: bool,
    model_inputs=None,
    labels=None,
    contribution_reduction: str = "mean",
    max_evidence_edges_per_window: int = 0,
):
    """Make sample-specific artifact failures collective before model replays."""
    targets = None
    local_error = None
    try:
        if isinstance(provider, OnlineFrozenStage2Reference):
            targets = provider.get_batch(
                sample_indices,
                layout,
                model_inputs=model_inputs,
                labels=labels,
                contribution_reduction=contribution_reduction,
                max_evidence_edges_per_window=max_evidence_edges_per_window,
                require_hidden_residuals=require_hidden_residuals,
            )
        else:
            targets = provider.get_batch(
                sample_indices,
                layout,
                require_hidden_residuals=require_hidden_residuals,
            )
        if not isinstance(targets, ReferenceBatchTargets):
            raise TypeError("reference provider must return ReferenceBatchTargets")
        requested = (
            tuple(sample_indices.detach().cpu().tolist())
            if torch.is_tensor(sample_indices)
            else tuple(sample_indices)
        )
        if targets.sample_indices != requested:
            raise ValueError(
                "reference sample order mismatch: "
                f"requested {requested}, received {targets.sample_indices}"
            )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    if distributed.is_available() and distributed.is_initialized():
        failed = torch.tensor(
            int(local_error is not None),
            dtype=torch.long,
            device=layout.causal_pair_mask.device,
        )
        distributed.all_reduce(failed, op=distributed.ReduceOp.MAX)
        if failed.item():
            messages: list[str | None] = [None] * distributed.get_world_size()
            distributed.all_gather_object(messages, local_error)
            detail = "; ".join(message for message in messages if message)
            raise RuntimeError(f"distributed reference target lookup failed: {detail}")
    elif local_error is not None:
        raise ValueError(f"reference target lookup failed: {local_error}")

    assert targets is not None
    return targets


def _resolve_token_ids(configured: Mapping[str, Any] | TokenIds | None) -> TokenIds:
    if isinstance(configured, TokenIds):
        return configured
    if configured is None or not isinstance(configured, Mapping):
        raise ValueError("enabled Stage-2 evidence-window training requires latent_token_ids")
    required = {"start", "end", "pad", "padding", "special"}
    missing = required - set(configured)
    if missing:
        raise ValueError("latent_token_ids missing required key(s): " + ", ".join(sorted(missing)))
    scalars = {}
    for source, target in (
        ("start", "latent_start"),
        ("end", "latent_end"),
        ("pad", "latent_pad"),
        ("padding", "padding"),
    ):
        value = configured[source]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"latent_token_ids.{source} must be an integer")
        scalars[target] = value
    special = _integer_set(configured["special"], "latent_token_ids.special")
    termination = _integer_set(
        configured.get("assistant_termination", ()),
        "latent_token_ids.assistant_termination",
    )
    return TokenIds(special=special, assistant_termination=termination, **scalars)


def _integer_set(values: Any, name: str) -> frozenset[int]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of integers")
    try:
        result = frozenset(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of integers") from exc
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise ValueError(f"{name} must contain only integers")
    return frozenset(int(value) for value in result)


__all__ = ["CustomTrainerStage2"]
