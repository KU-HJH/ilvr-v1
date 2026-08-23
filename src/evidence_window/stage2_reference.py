"""Fixed reference providers for explicit ILVR Stage 2.

The online provider owns one independently loaded model.  Under ZeRO-1/2 this
costs one additional full parameter set (plus transient no-grad activations),
so precomputed targets remain the lower-memory option.  ZeRO-3 is rejected
before the model factory is called because an unmanaged second full copy is
not supported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
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
        *,
        policy_checkpoint: str,
        reference_checkpoint: str,
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
        validate_online_reference_checkpoint_identity(
            policy_checkpoint,
            reference_checkpoint,
        )
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
        first_parameter = next(model.parameters(), None)
        fallback_device = (
            torch.device("cpu")
            if first_parameter is None
            else first_parameter.device
        )
        execution_device = torch.device(
            getattr(accelerator, "device", fallback_device)
        )
        execution_dtype = (
            torch.bfloat16
            if execution_device.type == "cuda"
            else torch.float32
        )
        model.to(device=execution_device, dtype=execution_dtype)
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
        validated: dict[str, Any] = {}

        def validate_local_inputs() -> None:
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
            validated.update(
                indices=indices,
                coordinator=coordinator,
                edges=select_stage2_edges(
                    layout, max_evidence_edges_per_window
                ),
                fixed_inputs=dict(model_inputs),
            )

        _collective_online_preflight(
            validate_local_inputs,
            device=_online_preflight_device(layout),
        )
        indices = validated["indices"]
        coordinator = validated["coordinator"]
        edges = validated["edges"]
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

        fixed_inputs = validated["fixed_inputs"]

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
    policy_checkpoint: str | None = None,
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
            policy_checkpoint,
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
        if policy_checkpoint is None:
            raise ValueError(
                "online frozen reference requires the policy initialization checkpoint"
            )
        assert config.reference.checkpoint is not None
        return OnlineFrozenStage2Reference.create(
            model_factory,
            accelerator,
            zero_stage,
            policy_checkpoint=policy_checkpoint,
            reference_checkpoint=config.reference.checkpoint,
        )
    if any(
        value is not None
        for value in (model_factory, accelerator, zero_stage, policy_checkpoint)
    ):
        raise ValueError("precomputed reference does not accept online model inputs")
    assert config.reference.artifact is not None
    if expected_identity is None:
        raise ValueError("precomputed reference requires an independent expected identity")
    return PrecomputedReferenceProvider(config.reference.artifact, expected_identity)


def validate_online_reference_checkpoint_identity(
    policy_checkpoint: str,
    reference_checkpoint: str,
) -> str:
    """Require the reference to be the exact Stage-1 policy initialization.

    Existing local paths compare by canonical resolved path so symlink aliases
    are accepted. Non-local model identifiers compare as exact explicit
    strings. This deliberately does not read or hash checkpoint weights.
    The returned identifier is canonical for local paths and unchanged for
    remote identifiers, so tokenizer and model configuration are loaded from
    the same source that initialized the policy.
    """
    values = {
        "policy initialization checkpoint": policy_checkpoint,
        "online reference checkpoint": reference_checkpoint,
    }
    for name, value in values.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty explicit identifier")

    policy_path = Path(policy_checkpoint).expanduser()
    reference_path = Path(reference_checkpoint).expanduser()
    policy_local = policy_path.exists()
    reference_local = reference_path.exists()
    if policy_local or reference_local:
        if not (policy_local and reference_local):
            raise ValueError(
                "online frozen reference must use the exact Stage-1 checkpoint "
                "that initialized the policy"
            )
        policy_identity = policy_path.resolve(strict=True)
        reference_identity = reference_path.resolve(strict=True)
        if policy_identity != reference_identity:
            raise ValueError(
                "online frozen reference must use the exact Stage-1 checkpoint "
                "that initialized the policy"
            )
        return str(policy_identity)

    if policy_checkpoint != reference_checkpoint:
        raise ValueError(
            "online frozen reference must use the exact Stage-1 checkpoint "
            "identifier that initialized the policy"
        )
    return policy_checkpoint


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


def _online_preflight_device(layout: Any) -> torch.device:
    pair_mask = getattr(layout, "causal_pair_mask", None)
    if torch.is_tensor(pair_mask):
        return pair_mask.device
    if (
        distributed.is_available()
        and distributed.is_initialized()
        and distributed.get_backend() == "nccl"
    ):
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _collective_online_preflight(
    operation: Callable[[], None],
    *,
    device: torch.device,
) -> None:
    """Reach error consensus before any rank enters reference collectives."""
    local_exception: Exception | None = None
    local_error: str | None = None
    try:
        operation()
    except Exception as exc:
        local_exception = exc
        local_error = f"{type(exc).__name__}: {exc}"

    if distributed.is_available() and distributed.is_initialized():
        failed = torch.tensor(
            int(local_error is not None), dtype=torch.long, device=device
        )
        distributed.all_reduce(failed, op=distributed.ReduceOp.MAX)
        if failed.item():
            errors: list[str | None] = [None] * distributed.get_world_size()
            distributed.all_gather_object(errors, local_error)
            details = "; ".join(
                f"rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error is not None
            )
            raise RuntimeError(
                f"distributed online reference preflight failed: {details}"
            )
    elif local_exception is not None:
        raise local_exception


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
    "validate_online_reference_checkpoint_identity",
]
