"""Fixed reference providers for evidence-window preference objectives."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable

import torch

from .config import ReferenceConfig


@dataclass(frozen=True)
class ParameterWitness:
    """Cheap parameter metadata recorded when a fixed reference is created."""

    tensor_count: int
    parameter_count: int
    parameter_bytes: int
    trainable_parameter_count: int
    dtypes: tuple[str, ...]
    includes_adapters: bool


@runtime_checkable
class ReferenceProvider(Protocol):
    """The reference boundary consumed by the condition engine and trainer."""

    witness: ParameterWitness

    def forward(
        self,
        wrapped_policy: torch.nn.Module,
        inputs: Mapping[str, Any],
    ) -> Any: ...

    def parameters(self) -> Iterator[torch.nn.Parameter]: ...

    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]: ...


def _is_adapter_parameter(name: str) -> bool:
    parts = name.lower().replace("-", "_").split(".")
    return any(part == "adapter" or part.startswith("lora_") for part in parts)


def locate_disable_adapter_owner(policy: torch.nn.Module) -> torch.nn.Module:
    """Find the PEFT owner through common model-wrapper chains."""

    pending = deque([policy])
    visited: set[int] = set()
    while pending:
        candidate = pending.popleft()
        identity = id(candidate)
        if identity in visited:
            continue
        visited.add(identity)

        if callable(getattr(candidate, "disable_adapter", None)):
            return candidate
        for attribute in ("module", "model"):
            child = getattr(candidate, attribute, None)
            if child is not None:
                pending.append(child)

    raise ValueError("adapter-disabled reference requires a disable_adapter context owner")


def validate_only_adapter_parameters_trainable(model: torch.nn.Module) -> None:
    """Reject shared-base references whose non-adapter weights can change."""

    invalid = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not _is_adapter_parameter(name)
    ]
    if invalid:
        raise ValueError(
            "adapter-disabled reference has trainable non-adapter parameter(s): "
            + ", ".join(invalid)
        )


def parameter_witness(
    model: torch.nn.Module,
    *,
    adapters: bool,
) -> ParameterWitness:
    """Summarize parameter ownership without reading or hashing tensor contents."""

    selected = [
        parameter
        for name, parameter in model.named_parameters()
        if adapters or not _is_adapter_parameter(name)
    ]
    return ParameterWitness(
        tensor_count=len(selected),
        parameter_count=sum(parameter.numel() for parameter in selected),
        parameter_bytes=sum(
            parameter.numel() * parameter.element_size() for parameter in selected
        ),
        trainable_parameter_count=sum(
            parameter.numel() for parameter in selected if parameter.requires_grad
        ),
        dtypes=tuple(sorted({str(parameter.dtype) for parameter in selected})),
        includes_adapters=adapters,
    )


class AdapterDisabledReference:
    """A fixed view of a PEFT policy with its adapters temporarily disabled."""

    def __init__(self, unwrapped_policy: torch.nn.Module):
        self.peft_model = locate_disable_adapter_owner(unwrapped_policy)
        validate_only_adapter_parameters_trainable(self.peft_model)
        self.base_witness = parameter_witness(self.peft_model, adapters=False)
        self.witness = self.base_witness

    def forward(
        self,
        wrapped_policy: torch.nn.Module,
        inputs: Mapping[str, Any],
    ) -> Any:
        with torch.no_grad(), self.peft_model.disable_adapter():
            return wrapped_policy(**inputs)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.peft_model.parameters()

    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        return self.peft_model.named_parameters()


class FrozenCopyReference:
    """A separately owned, evaluation-only snapshot of the initialized policy."""

    def __init__(self, model: torch.nn.Module, witness: ParameterWitness):
        self.model = model
        self.witness = witness

    @classmethod
    def create(
        cls,
        initialized_policy: torch.nn.Module,
        accelerator: Any,
    ) -> FrozenCopyReference:
        model = copy.deepcopy(initialized_policy)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model = accelerator.prepare_model(model, evaluation_mode=True)
        model.eval()
        return cls(model, parameter_witness(model, adapters=True))

    def forward(
        self,
        wrapped_policy: torch.nn.Module,
        inputs: Mapping[str, Any],
    ) -> Any:
        del wrapped_policy
        with torch.no_grad():
            return self.model(**inputs)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.model.parameters()

    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        return self.model.named_parameters()


def build_reference_provider(
    reference_config: ReferenceConfig,
    initialized_policy: torch.nn.Module,
    accelerator: Any,
) -> ReferenceProvider | None:
    """Construct the configured fixed reference without changing ZeRO stages."""

    mode = reference_config.mode
    if mode == "none":
        return None
    if mode == "adapter_disabled":
        return AdapterDisabledReference(initialized_policy)
    if mode == "frozen_copy":
        return FrozenCopyReference.create(initialized_policy, accelerator)
    raise ValueError(f"unsupported reference mode: {mode}")
