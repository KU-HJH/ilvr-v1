"""Topology-safe exponential moving average teachers for Stage 1."""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Mapping

import torch
from torch import nn

from peft_common import resolve_qwen2_5_vl_model


Stage1EmaMethod = Literal["full", "lora", "dora"]
STAGE1_EMA_TEACHER_STATE_FILENAME = "stage1_ema_teacher.pt"
_PEFT_PARAMETER_COMPONENTS = frozenset(
    {
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
        "trainable_tokens_delta",
        "output_delta",
    }
)
_SUPPORTED_QWEN_PARAMETER_PREFIXES = frozenset(("", "base_model.model."))
_STAGE1_EMA_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EmaParameterPair:
    """One immutable name-and-shape correspondence in an EMA topology."""

    name: str
    shape: torch.Size


@dataclass(frozen=True)
class _TiedEmbeddingAliasTopology:
    canonical_name: str
    alias_names: tuple[str, str]


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
    return any(
        component in _PEFT_PARAMETER_COMPONENTS
        for component in name.split(".")
    )


def _declared_qwen_tied_embedding_topology(
    model: nn.Module,
    named: Mapping[str, nn.Parameter],
) -> _TiedEmbeddingAliasTopology | None:
    config = getattr(model, "config", None)
    if getattr(config, "tie_word_embeddings", None) is not True:
        return None
    try:
        qwen = resolve_qwen2_5_vl_model(model)
        input_weight = qwen.model.embed_tokens.weight
        output_weight = qwen.lm_head.weight
    except (AttributeError, TypeError) as error:
        raise RuntimeError(
            "EMA declared tied embedding alias topology does not satisfy the "
            "Qwen2.5-VL input/output interface"
        ) from error
    if not isinstance(input_weight, nn.Parameter) or not isinstance(
        output_weight, nn.Parameter
    ):
        raise RuntimeError(
            "EMA declared tied embedding alias topology requires Parameters"
        )
    if input_weight is not output_weight:
        raise RuntimeError(
            "EMA declared tied embedding alias topology must use the same "
            "Parameter and storage"
        )
    if _parameter_shape(input_weight) != _parameter_shape(output_weight):
        raise RuntimeError(
            "EMA declared tied embedding alias topology has mismatched shapes"
        )

    aliases = tuple(
        sorted(name for name, parameter in named.items() if parameter is input_weight)
    )
    input_suffix = "model.embed_tokens.weight"
    output_suffix = "lm_head.weight"
    input_names = tuple(name for name in aliases if name.endswith(input_suffix))
    output_names = tuple(name for name in aliases if name.endswith(output_suffix))
    if len(aliases) != 2 or len(input_names) != 1 or len(output_names) != 1:
        raise RuntimeError(
            "EMA declared tied embedding alias topology must contain exactly "
            "the decoder input embedding weight and lm_head.weight"
        )
    input_name = input_names[0]
    output_name = output_names[0]
    input_prefix = input_name[: -len(input_suffix)]
    output_prefix = output_name[: -len(output_suffix)]
    if input_prefix != output_prefix:
        raise RuntimeError(
            "EMA declared tied embedding alias topology has inconsistent "
            "supported wrapper prefixes"
        )
    if input_prefix not in _SUPPORTED_QWEN_PARAMETER_PREFIXES:
        raise RuntimeError(
            "EMA declared tied embedding alias topology has an unsupported "
            f"wrapper prefix: {input_prefix!r}"
        )
    return _TiedEmbeddingAliasTopology(
        canonical_name=input_name,
        alias_names=(input_name, output_name),
    )


def _validate_full_tied_embedding_parity(
    student: nn.Module,
    teacher: nn.Module,
) -> None:
    student_topology = _declared_qwen_tied_embedding_topology(
        student, _named_parameters(student)
    )
    try:
        teacher_topology = _declared_qwen_tied_embedding_topology(
            teacher, _named_parameters(teacher)
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"EMA student/teacher alias topology mismatch: {error}"
        ) from error
    if student_topology != teacher_topology:
        raise RuntimeError(
            "EMA student/teacher alias topology mismatch for declared tied "
            "embeddings"
        )


def _selected_parameters(
    model: nn.Module, method: Stage1EmaMethod
) -> dict[str, nn.Parameter]:
    named = _named_parameters(model)
    tied_topology = (
        _declared_qwen_tied_embedding_topology(model, named)
        if method == "full"
        else None
    )
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
    canonical_selected: dict[str, nn.Parameter] = {}
    for name in sorted(selected):
        if (
            tied_topology is not None
            and name in tied_topology.alias_names
            and name != tied_topology.canonical_name
        ):
            continue
        identity = id(selected[name])
        previous = parameter_owners.get(identity)
        if previous is not None:
            raise RuntimeError(
                "EMA topology has duplicate parameter aliases: "
                f"{previous} and {name}"
            )
        parameter_owners[identity] = name
        canonical_selected[name] = selected[name]
    return canonical_selected


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
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("EMA decay must be finite and in [0, 1]")
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
    if validated_method == "full":
        _validate_full_tied_embedding_parity(student, teacher)
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


def _decode_ema_state(
    state: Mapping[str, Any],
) -> tuple[
    Stage1EmaMethod,
    float,
    tuple[str, ...],
    tuple[tuple[str, tuple[int, ...]], ...],
    dict[str, torch.Tensor],
]:
    required = {
        "schema_version",
        "decay",
        "method",
        "parameter_names",
        "parameter_shapes",
        "parameters",
    }
    if type(state) is not dict:
        raise RuntimeError("invalid EMA state mapping")
    missing = sorted(required - set(state))
    extra = sorted(set(state) - required)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise RuntimeError("invalid EMA state: " + "; ".join(details))
    version = state["schema_version"]
    if type(version) is not int or version != _STAGE1_EMA_STATE_SCHEMA_VERSION:
        raise RuntimeError("unsupported EMA state schema version")

    raw_method = state["method"]
    if type(raw_method) is not str:
        raise RuntimeError("invalid EMA state method")
    try:
        method = _validate_method(raw_method)
    except ValueError as error:
        raise RuntimeError("invalid EMA state method") from error
    decay = state["decay"]
    if type(decay) is not float or not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
        raise RuntimeError("invalid EMA state decay")

    names = state["parameter_names"]
    if (
        type(names) is not tuple
        or not names
        or any(type(name) is not str or not name for name in names)
    ):
        raise RuntimeError("invalid EMA state parameter names")
    if len(set(names)) != len(names):
        raise RuntimeError("invalid EMA state parameter names")

    shapes = state["parameter_shapes"]
    if type(shapes) is not tuple:
        raise RuntimeError("invalid EMA state parameter shapes")
    decoded_shapes: list[tuple[str, tuple[int, ...]]] = []
    for entry in shapes:
        if type(entry) is not tuple or len(entry) != 2:
            raise RuntimeError("invalid EMA state parameter shapes")
        name, shape = entry
        if (
            type(name) is not str
            or not name
            or type(shape) is not tuple
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        ):
            raise RuntimeError("invalid EMA state parameter shapes")
        decoded_shapes.append((name, shape))
    if tuple(name for name, _ in decoded_shapes) != names:
        raise RuntimeError("invalid EMA state parameter shapes")

    parameters = state["parameters"]
    if type(parameters) is not dict:
        raise RuntimeError("invalid EMA state parameters")
    return method, decay, names, tuple(decoded_shapes), parameters


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
        if self.method == "full":
            _validate_full_tied_embedding_parity(student, self.model)
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
        if self.decay == 1.0:
            return
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

        for pair in self._pairs:
            source = student_named[pair.name]
            target = teacher_named[pair.name]
            with _gathered_parameter_pair(source, target):
                source_value = source.detach().to(
                    device=target.device, dtype=target.dtype
                )
                target.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        teacher_named = _named_parameters(self.model)
        parameters: dict[str, torch.Tensor] = {}
        parameter_shapes: list[tuple[str, tuple[int, ...]]] = []
        for pair in self._pairs:
            target = teacher_named.get(pair.name)
            if target is None:
                raise RuntimeError(
                    f"EMA topology mismatch: teacher is missing {pair.name}"
                )
            if _is_partitioned(target):
                raise RuntimeError(
                    "EMA teacher checkpoint state must be replicated, found "
                    f"partitioned parameter {pair.name}"
                )
            shape = tuple(int(dimension) for dimension in target.shape)
            if torch.Size(shape) != pair.shape:
                raise RuntimeError(
                    f"EMA topology mismatch: shape for {pair.name} is "
                    f"expected={tuple(pair.shape)}, teacher={shape}"
                )
            parameter_shapes.append((pair.name, shape))
            parameters[pair.name] = target.detach().to(device="cpu").clone()
        return {
            "schema_version": _STAGE1_EMA_STATE_SCHEMA_VERSION,
            "decay": self.decay,
            "method": self.method,
            "parameter_names": self.parameter_names,
            "parameter_shapes": tuple(parameter_shapes),
            "parameters": parameters,
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        (
            state_method,
            state_decay,
            state_names,
            state_shapes,
            parameters,
        ) = _decode_ema_state(state)
        if state_method != self.method:
            raise RuntimeError(
                f"EMA state method mismatch: state={state_method}, "
                f"teacher={self.method}"
            )
        if state_names != self.parameter_names:
            raise _topology_mismatch(self.parameter_names, state_names)
        expected_shapes = tuple(
            (pair.name, tuple(int(dimension) for dimension in pair.shape))
            for pair in self._pairs
        )
        if state_shapes != expected_shapes:
            raise RuntimeError("EMA state parameter shapes mismatch")

        parameter_keys = tuple(parameters)
        if parameter_keys != self.parameter_names:
            raise _topology_mismatch(self.parameter_names, parameter_keys)
        teacher_named = _named_parameters(self.model)
        copies: list[tuple[nn.Parameter, torch.Tensor]] = []
        for pair in self._pairs:
            target = teacher_named.get(pair.name)
            source = parameters.get(pair.name)
            if target is None or not isinstance(source, torch.Tensor):
                raise RuntimeError(
                    f"invalid EMA state tensor for {pair.name}"
                )
            expected_shape = tuple(int(dimension) for dimension in pair.shape)
            if tuple(source.shape) != expected_shape:
                raise RuntimeError(
                    f"EMA state tensor shape mismatch for {pair.name}: "
                    f"expected={expected_shape}, state={tuple(source.shape)}"
                )
            if source.device.type != "cpu" or source.layout != torch.strided:
                raise RuntimeError(
                    f"invalid EMA state tensor storage for {pair.name}"
                )
            if source.dtype != target.dtype:
                raise RuntimeError(
                    f"EMA state tensor dtype mismatch for {pair.name}: "
                    f"state={source.dtype}, teacher={target.dtype}"
                )
            if tuple(target.shape) != expected_shape:
                raise RuntimeError(
                    f"EMA teacher tensor shape mismatch for {pair.name}: "
                    f"expected={expected_shape}, teacher={tuple(target.shape)}"
                )
            copies.append((target, source))
        for target, source in copies:
            target.copy_(source)
        self.decay = state_decay
        self.model.eval()
        self.model.requires_grad_(False)


__all__ = [
    "EmaParameterPair",
    "STAGE1_EMA_TEACHER_STATE_FILENAME",
    "Stage1EmaTeacher",
    "build_ema_parameter_map",
]
