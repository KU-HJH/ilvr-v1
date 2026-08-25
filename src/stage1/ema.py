"""Topology-safe exponential moving average teachers for Stage 1."""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Mapping

import torch
from torch import nn


Stage1EmaMethod = Literal["full", "lora", "dora"]
_PEFT_PARAMETER_MARKERS = (
    "lora_",
    "trainable_tokens_delta",
    "output_delta",
)


@dataclass(frozen=True)
class EmaParameterPair:
    """One immutable name-and-shape correspondence in an EMA topology."""

    name: str
    shape: torch.Size


def _unwrap_model(model: nn.Module) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise TypeError("EMA student must be a torch.nn.Module")

    current = model
    visited: set[int] = set()
    while isinstance(getattr(current, "module", None), nn.Module):
        if id(current) in visited:
            raise RuntimeError("EMA model wrapper contains a module cycle")
        visited.add(id(current))
        current = current.module
    return current


def _named_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    try:
        entries = list(model.named_parameters(remove_duplicate=False))
    except TypeError:
        entries = list(model.named_parameters())

    named: dict[str, nn.Parameter] = {}
    for name, parameter in entries:
        if name in named:
            raise RuntimeError(f"EMA topology has duplicate parameter name: {name}")
        named[name] = parameter
    return named


def _is_visual_parameter(name: str) -> bool:
    return "visual" in name.split(".")


def _is_peft_parameter(name: str) -> bool:
    return any(marker in name for marker in _PEFT_PARAMETER_MARKERS)


def _selected_parameters(
    model: nn.Module, method: Stage1EmaMethod
) -> dict[str, nn.Parameter]:
    named = _named_parameters(model)
    trainable_visual = sorted(
        name
        for name, parameter in named.items()
        if parameter.requires_grad and _is_visual_parameter(name)
    )
    if trainable_visual:
        raise RuntimeError(
            "EMA cannot include trainable visual parameters: "
            + ", ".join(trainable_visual)
        )

    trainable_language = {
        name: parameter
        for name, parameter in named.items()
        if parameter.requires_grad and not _is_visual_parameter(name)
    }
    if method == "full":
        selected = trainable_language
    else:
        unsupported = sorted(
            name for name in trainable_language if not _is_peft_parameter(name)
        )
        if unsupported:
            raise RuntimeError(
                f"EMA {method} topology has unsupported trainable parameters: "
                + ", ".join(unsupported)
            )
        selected = {
            name: parameter
            for name, parameter in trainable_language.items()
            if _is_peft_parameter(name)
        }

    if not selected:
        raise RuntimeError(f"EMA parameter set is empty for method {method}")

    parameter_owners: dict[int, str] = {}
    for name in sorted(selected):
        identity = id(selected[name])
        previous = parameter_owners.get(identity)
        if previous is not None:
            raise RuntimeError(
                "EMA topology has duplicate parameter aliases: "
                f"{previous} and {name}"
            )
        parameter_owners[identity] = name
    return selected


def _is_partitioned(parameter: nn.Parameter) -> bool:
    return getattr(parameter, "ds_id", None) is not None


def _parameter_shape(parameter: nn.Parameter) -> torch.Size:
    if _is_partitioned(parameter) and hasattr(parameter, "ds_shape"):
        return torch.Size(parameter.ds_shape)
    return torch.Size(parameter.shape)


def _validate_method(method: str) -> Stage1EmaMethod:
    if method not in ("full", "lora", "dora"):
        raise ValueError(f"unsupported EMA method: {method}")
    return method


def _validate_decay(decay: float) -> float:
    value = float(decay)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("EMA decay must be finite and in [0, 1)")
    return value


def build_ema_parameter_map(
    student: nn.Module,
    teacher: nn.Module,
    method: Stage1EmaMethod,
) -> tuple[EmaParameterPair, ...]:
    """Build an exact, deterministic EMA correspondence by name and shape."""

    validated_method = _validate_method(method)
    student = _unwrap_model(student)
    teacher = _unwrap_model(teacher)
    selected = _selected_parameters(student, validated_method)
    teacher_named = _named_parameters(teacher)

    pairs: list[EmaParameterPair] = []
    teacher_owners: dict[int, str] = {}
    for name in sorted(selected):
        source = selected[name]
        target = teacher_named.get(name)
        if target is None:
            raise RuntimeError(f"EMA topology mismatch: teacher is missing {name}")

        source_shape = _parameter_shape(source)
        target_shape = _parameter_shape(target)
        if source_shape != target_shape:
            raise RuntimeError(
                f"EMA topology mismatch: shape for {name} is "
                f"student={tuple(source_shape)}, teacher={tuple(target_shape)}"
            )

        target_identity = id(target)
        previous = teacher_owners.get(target_identity)
        if previous is not None:
            raise RuntimeError(
                "EMA topology has duplicate teacher parameter aliases: "
                f"{previous} and {name}"
            )
        teacher_owners[target_identity] = name
        pairs.append(EmaParameterPair(name=name, shape=source_shape))
    return tuple(pairs)


@contextmanager
def _gathered_parameter_pair(
    source: nn.Parameter, target: nn.Parameter
) -> Iterator[None]:
    source_partitioned = _is_partitioned(source)
    target_partitioned = _is_partitioned(target)
    if not source_partitioned and not target_partitioned:
        with nullcontext():
            yield
        return

    import deepspeed

    parameters: list[nn.Parameter] = []
    if source_partitioned:
        parameters.append(source)
    if target_partitioned:
        parameters.append(target)
    modifier_rank = 0 if target_partitioned else None
    with deepspeed.zero.GatheredParameters(
        parameters, modifier_rank=modifier_rank
    ):
        yield


def _topology_mismatch(
    expected_names: tuple[str, ...], actual_names: tuple[str, ...]
) -> RuntimeError:
    expected = set(expected_names)
    actual = set(actual_names)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append("missing required " + ", ".join(missing))
    if extra:
        details.append("extra required " + ", ".join(extra))
    return RuntimeError("EMA topology mismatch: " + "; ".join(details))


class Stage1EmaTeacher:
    """A frozen EMA copy updated through an exact named-parameter contract."""

    def __init__(
        self,
        student: nn.Module,
        decay: float,
        method: Stage1EmaMethod,
    ) -> None:
        self.decay = _validate_decay(decay)
        self.method = _validate_method(method)
        unwrapped_student = _unwrap_model(student)
        self.model = copy.deepcopy(unwrapped_student).eval()
        self.model.requires_grad_(False)
        self._pairs = build_ema_parameter_map(
            unwrapped_student, self.model, self.method
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(pair.name for pair in self._pairs)

    def _validated_parameters(
        self, student: nn.Module
    ) -> tuple[
        dict[str, nn.Parameter],
        dict[str, nn.Parameter],
    ]:
        student = _unwrap_model(student)
        selected = _selected_parameters(student, self.method)
        selected_names = tuple(sorted(selected))
        if selected_names != self.parameter_names:
            raise _topology_mismatch(self.parameter_names, selected_names)

        teacher_named = _named_parameters(self.model)
        target_owners: dict[int, str] = {}
        for pair in self._pairs:
            source = selected.get(pair.name)
            target = teacher_named.get(pair.name)
            if source is None or target is None:
                absent = "student" if source is None else "teacher"
                raise RuntimeError(
                    f"EMA topology mismatch: {absent} is missing {pair.name}"
                )
            source_shape = _parameter_shape(source)
            target_shape = _parameter_shape(target)
            if source_shape != pair.shape or target_shape != pair.shape:
                raise RuntimeError(
                    f"EMA topology mismatch: shape for {pair.name} is "
                    f"expected={tuple(pair.shape)}, student={tuple(source_shape)}, "
                    f"teacher={tuple(target_shape)}"
                )
            target_identity = id(target)
            previous = target_owners.get(target_identity)
            if previous is not None:
                raise RuntimeError(
                    "EMA topology has duplicate teacher parameter aliases: "
                    f"{previous} and {pair.name}"
                )
            target_owners[target_identity] = pair.name
        return selected, teacher_named

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        student_named, teacher_named = self._validated_parameters(student)
        for pair in self._pairs:
            source = student_named[pair.name]
            target = teacher_named[pair.name]
            with _gathered_parameter_pair(source, target):
                if torch.Size(source.shape) != pair.shape:
                    raise RuntimeError(
                        f"EMA topology mismatch: gathered source shape for "
                        f"{pair.name} is {tuple(source.shape)}, expected "
                        f"{tuple(pair.shape)}"
                    )
                if torch.Size(target.shape) != pair.shape:
                    raise RuntimeError(
                        f"EMA topology mismatch: gathered teacher shape for "
                        f"{pair.name} is {tuple(target.shape)}, expected "
                        f"{tuple(pair.shape)}"
                    )
                source_value = source.detach().to(
                    device=target.device, dtype=target.dtype
                )
                target.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "decay": self.decay,
            "method": self.method,
            "parameter_names": self.parameter_names,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {"model", "decay", "method", "parameter_names"}
        missing = sorted(required - set(state))
        extra = sorted(set(state) - required)
        if missing or extra:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if extra:
                details.append("unexpected " + ", ".join(extra))
            raise RuntimeError("invalid EMA state: " + "; ".join(details))

        state_method = _validate_method(state["method"])
        if state_method != self.method:
            raise RuntimeError(
                f"EMA state method mismatch: state={state_method}, "
                f"teacher={self.method}"
            )
        state_names = tuple(state["parameter_names"])
        if state_names != self.parameter_names:
            raise _topology_mismatch(self.parameter_names, state_names)
        state_decay = _validate_decay(state["decay"])

        self.model.load_state_dict(state["model"], strict=True)
        self.decay = state_decay
        self.model.eval()
        self.model.requires_grad_(False)


__all__ = [
    "EmaParameterPair",
    "Stage1EmaTeacher",
    "build_ema_parameter_map",
]
