"""Fixed reference providers for explicit ILVR Stage 2.

The online provider owns one independently loaded model.  Under ZeRO-1/2 this
costs one additional full parameter set (plus transient no-grad activations),
so precomputed targets remain the lower-memory option.  ZeRO-3 is rejected
before the model factory is called because an unmanaged second full copy is
not supported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Iterator, Protocol, Sequence

import torch
import torch.distributed as distributed

from .artifacts import (
    ArtifactIdentity,
    PrecomputedReferenceProvider,
    ReferenceBatchTargets,
)
from .reference import ParameterWitness, parameter_witness
from .scores import chunked_causal_target_logps
from .stage2_config import Stage2EvidenceWindowConfig
from .stage2_effects import Stage2ConditionOutput, compute_stage2_effects
from .stage2_layout import Stage2InterleavedLayout, select_stage2_edges


EdgeCountCoordinator = Callable[[int, torch.device], tuple[int, int]]


class Stage2ReferenceProvider(Protocol):
    def get_batch(
        self,
        sample_indices: Sequence[int] | torch.Tensor,
        layout: Stage2InterleavedLayout,
        *,
        require_hidden_residuals: bool = False,
    ) -> ReferenceBatchTargets: ...


class OnlineFrozenStage2Reference:
    """Evaluation-only Stage-1 initialization replayed with Stage-2 semantics."""

    def __init__(self, model: torch.nn.Module, witness: ParameterWitness):
        self.model = model
        self.witness = witness

    @classmethod
    def create(
        cls,
        model_factory: Callable[[], torch.nn.Module],
        accelerator: Any,
        zero_stage: int,
    ) -> "OnlineFrozenStage2Reference":
        """Build and prepare an independently owned frozen model.

        Runtime compatibility is checked before invoking ``model_factory`` so
        ZeRO-3 cannot allocate the unsupported second full model.
        """
        if isinstance(zero_stage, bool) or not isinstance(zero_stage, int):
            raise ValueError("ZeRO stage must be 1 or 2 for online frozen reference")
        if zero_stage == 3:
            raise ValueError("online frozen Stage-2 reference is incompatible with ZeRO-3")
        if zero_stage not in {1, 2}:
            raise ValueError("online frozen Stage-2 reference requires ZeRO-1 or ZeRO-2")
        if not callable(model_factory):
            raise TypeError("online reference model_factory must be callable")
        prepare_model = getattr(accelerator, "prepare_model", None)
        if not callable(prepare_model):
            raise TypeError("accelerator must provide prepare_model")

        model = model_factory()
        if not isinstance(model, torch.nn.Module):
            raise TypeError("online reference model_factory must return a torch module")
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("online reference model must expose config.stage")
        config.stage = "stage2"
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model = prepare_model(model, evaluation_mode=True)
        model.eval()
        witness = parameter_witness(model, adapters=True)
        if witness.trainable_parameter_count != 0:
            raise RuntimeError("prepared online reference contains trainable parameters")
        return cls(model, witness)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.model.parameters()

    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        return self.model.named_parameters()

    def get_batch(
        self,
        sample_indices: Sequence[int] | torch.Tensor,
        layout: Stage2InterleavedLayout,
        *,
        model_inputs: Mapping[str, Any],
        labels: torch.Tensor,
        contribution_reduction: str,
        max_evidence_edges_per_window: int,
        require_hidden_residuals: bool = False,
        edge_count_coordinator: EdgeCountCoordinator | None = None,
    ) -> ReferenceBatchTargets:
        """Replay matched full/LOO conditions with the exact policy layout.

        Every distributed rank runs one full slot and the globally maximum
        number of edge slots.  Ranks with fewer local edges fill the remainder
        with all-false masks, matching the policy engine's collective order.
        """
        if not isinstance(layout, Stage2InterleavedLayout):
            raise TypeError("layout must be a Stage2InterleavedLayout")
        indices = _sample_index_tuple(sample_indices)
        if len(indices) != layout.causal_pair_mask.shape[0]:
            raise ValueError("sample indices must have one entry per batch row")
        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        if "labels" in model_inputs:
            raise ValueError("online reference model_inputs must not include labels")
        expected_labels = (
            layout.window_token_mask.shape[0],
            layout.window_token_mask.shape[-1],
        )
        if not torch.is_tensor(labels) or labels.shape != expected_labels:
            raise ValueError("online reference labels must have shape [B, T]")
        if contribution_reduction not in {"mean", "sum"}:
            raise ValueError("contribution reduction must be mean or sum")
        if (
            isinstance(max_evidence_edges_per_window, bool)
            or not isinstance(max_evidence_edges_per_window, int)
            or max_evidence_edges_per_window < 0
        ):
            raise ValueError("max evidence edges per window must be nonnegative")
        if not isinstance(require_hidden_residuals, bool):
            raise ValueError("require_hidden_residuals must be boolean")

        coordinator = (
            _distributed_edge_counts
            if edge_count_coordinator is None
            else edge_count_coordinator
        )
        if not callable(coordinator):
            raise TypeError("edge_count_coordinator must be callable")
        edges = select_stage2_edges(layout, max_evidence_edges_per_window)
        global_count, global_max = coordinator(
            len(edges), layout.causal_pair_mask.device
        )
        _validate_coordinated_counts(len(edges), global_count, global_max)
        shape = layout.causal_pair_mask.shape
        if global_count == 0:
            return ReferenceBatchTargets(
                sample_indices=indices,
                contributions=torch.zeros(
                    shape, dtype=torch.float32, device=layout.causal_pair_mask.device
                ),
                pair_mask=torch.zeros_like(layout.causal_pair_mask),
                hidden_residuals=None,
            )

        reserved = {
            "attention_block_mask",
            "attention_backend_override",
            "output_hidden_states",
            "return_dict",
            "use_cache",
        }
        conflict = reserved & set(model_inputs)
        if conflict:
            raise ValueError(
                "online reference model_inputs contain reserved key(s): "
                + ", ".join(sorted(conflict))
            )
        fixed_inputs = dict(model_inputs)

        def fixed_forward(block_mask, need_hidden):
            output = self.model(
                **fixed_inputs,
                attention_block_mask=block_mask,
                attention_backend_override="sdpa",
                output_hidden_states=need_hidden,
                return_dict=True,
                use_cache=False,
            )
            logits = getattr(output, "logits", None)
            if not torch.is_tensor(logits):
                raise ValueError("online reference forward must return logits")
            shifted = chunked_causal_target_logps(logits, labels)
            token_logps = torch.cat(
                (shifted.new_zeros((shifted.shape[0], 1)), shifted), dim=1
            )
            hidden = _final_hidden_state(output) if need_hidden else None
            return Stage2ConditionOutput(token_logps, hidden)

        with torch.no_grad():
            effects = compute_stage2_effects(
                layout,
                edges,
                fixed_forward,
                reduction=contribution_reduction,
                need_hidden=require_hidden_residuals,
            )
            dummy_mask = torch.zeros(
                (
                    layout.causal_pair_mask.shape[0],
                    layout.window_token_mask.shape[-1],
                    layout.window_token_mask.shape[-1],
                ),
                dtype=torch.bool,
                device=layout.causal_pair_mask.device,
            )
            for _ in range(global_max - len(edges)):
                fixed_forward(dummy_mask, require_hidden_residuals)

        return ReferenceBatchTargets(
            sample_indices=indices,
            contributions=effects.contributions.detach(),
            pair_mask=effects.pair_mask.detach(),
            hidden_residuals=(
                None
                if effects.hidden_residuals is None
                else effects.hidden_residuals.detach()
            ),
        )


def build_stage2_reference_provider(
    config: Stage2EvidenceWindowConfig,
    *,
    expected_identity: ArtifactIdentity | None = None,
    model_factory: Callable[[], torch.nn.Module] | None = None,
    accelerator: Any = None,
    zero_stage: int | None = None,
) -> Stage2ReferenceProvider | None:
    """Build only the configured immutable reference source."""
    if not isinstance(config, Stage2EvidenceWindowConfig):
        raise TypeError("config must be a Stage2EvidenceWindowConfig")
    if config.reference.mode == "none":
        construction_values = (
            expected_identity,
            model_factory,
            accelerator,
            zero_stage,
        )
        if any(value is not None for value in construction_values):
            raise ValueError(
                "reference-free mode does not accept reference construction inputs"
            )
        return None
    if config.reference.mode == "online_frozen":
        if expected_identity is not None:
            raise ValueError("online frozen reference does not accept an artifact identity")
        if zero_stage is None:
            raise ValueError("online frozen reference requires an explicit ZeRO stage")
        return OnlineFrozenStage2Reference.create(model_factory, accelerator, zero_stage)
    if any(value is not None for value in (model_factory, accelerator, zero_stage)):
        raise ValueError("precomputed reference does not accept online model inputs")
    assert config.reference.artifact is not None
    if expected_identity is None:
        raise ValueError("precomputed reference requires an independent expected identity")
    return PrecomputedReferenceProvider(config.reference.artifact, expected_identity)


def _sample_index_tuple(values: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    if torch.is_tensor(values):
        if values.ndim != 1:
            raise ValueError("sample indices must have shape [B]")
        result = tuple(values.detach().cpu().tolist())
    else:
        result = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in result
    ):
        raise ValueError("sample indices must be nonnegative integers")
    return result


def _final_hidden_state(output: Any) -> torch.Tensor:
    hidden = getattr(output, "hidden_states", None)
    if isinstance(hidden, (tuple, list)):
        hidden = hidden[-1] if hidden else None
    if not torch.is_tensor(hidden):
        raise ValueError("online reference residual output requires final hidden states")
    return hidden


def _distributed_edge_counts(local_count: int, device: torch.device) -> tuple[int, int]:
    if not (distributed.is_available() and distributed.is_initialized()):
        return local_count, local_count
    total = torch.tensor(local_count, dtype=torch.long, device=device)
    maximum = total.clone()
    distributed.all_reduce(total, op=distributed.ReduceOp.SUM)
    distributed.all_reduce(maximum, op=distributed.ReduceOp.MAX)
    return int(total.item()), int(maximum.item())


def _validate_coordinated_counts(local: int, total: int, maximum: int) -> None:
    for name, value in (("total", total), ("maximum", maximum)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"coordinated {name} edge count must be nonnegative")
    if total < local or maximum < local or total < maximum:
        raise ValueError("coordinated edge counts are inconsistent with local edges")
    if (total == 0) != (maximum == 0):
        raise ValueError("coordinated edge counts disagree on emptiness")


__all__ = [
    "OnlineFrozenStage2Reference",
    "Stage2ReferenceProvider",
    "build_stage2_reference_provider",
]
