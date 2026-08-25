from __future__ import annotations

import sys
from contextlib import AbstractContextManager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stage1.ema import Stage1EmaTeacher, build_ema_parameter_map


class ToyAttention(nn.Module):
    def __init__(self, *, peft: bool) -> None:
        super().__init__()
        self.q_proj = nn.Linear(2, 2, bias=False)
        if peft:
            self.q_proj.weight.requires_grad_(False)
            self.lora_A = nn.Parameter(torch.full((1, 2), 2.0))
            self.lora_B = nn.Parameter(torch.full((2, 1), 3.0))


class ToyLayer(nn.Module):
    def __init__(self, *, peft: bool) -> None:
        super().__init__()
        self.self_attn = ToyAttention(peft=peft)
        self.mlp = nn.Linear(2, 2, bias=False)
        if peft:
            self.mlp.weight.requires_grad_(False)


class ToyLanguageModel(nn.Module):
    def __init__(self, *, peft: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(peft=peft)])
        if peft:
            self.trainable_tokens_delta = nn.Parameter(torch.full((2, 2), 4.0))
            self.output_delta = nn.Parameter(torch.full((2, 2), 5.0))


class ToyStudent(nn.Module):
    def __init__(self, *, peft: bool = False) -> None:
        super().__init__()
        self.visual = nn.Linear(2, 2, bias=False)
        self.visual.requires_grad_(False)
        self.model = ToyLanguageModel(peft=peft)


class ModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module


class SingleParameterStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(2, 2, bias=False)


@pytest.fixture
def full_student() -> ToyStudent:
    torch.manual_seed(7)
    return ToyStudent()


@pytest.fixture
def dora_student() -> ToyStudent:
    torch.manual_seed(11)
    return ToyStudent(peft=True)


def clone_named_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def assert_named_parameters_equal(
    module: nn.Module, expected: dict[str, torch.Tensor]
) -> None:
    actual = dict(module.named_parameters())
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value)


def test_ema_teacher_deep_copies_unwrapped_topology_and_freezes_it(full_student):
    wrapped = ModuleWrapper(ModuleWrapper(full_student))

    ema = Stage1EmaTeacher(wrapped, decay=0.9, method="full")

    assert ema.model is not full_student
    assert ema.model.training is False
    assert all(parameter.requires_grad is False for parameter in ema.model.parameters())
    assert all(not name.startswith("module.") for name in ema.parameter_names)


def test_ema_rejects_name_mismatch(full_student):
    ema = Stage1EmaTeacher(full_student, decay=0.9, method="full")
    delattr(ema.model.model.layers[0].self_attn, "q_proj")

    with pytest.raises(RuntimeError, match=r"EMA topology mismatch.*q_proj"):
        ema.update(full_student)


def test_ema_rejects_shape_mismatch(full_student):
    ema = Stage1EmaTeacher(full_student, decay=0.9, method="full")
    ema.model.model.layers[0].self_attn.q_proj.weight = nn.Parameter(
        torch.zeros(3, 2), requires_grad=False
    )

    with pytest.raises(RuntimeError, match=r"EMA topology mismatch.*shape.*q_proj"):
        ema.update(full_student)


def test_ema_rejects_new_required_student_parameter(full_student):
    ema = Stage1EmaTeacher(full_student, decay=0.9, method="full")
    full_student.model.register_parameter("late_weight", nn.Parameter(torch.ones(2)))

    with pytest.raises(RuntimeError, match=r"EMA topology mismatch.*late_weight"):
        ema.update(full_student)


def test_full_ema_updates_trainable_language_but_not_visual(full_student):
    ema = Stage1EmaTeacher(full_student, decay=0.5, method="full")
    before_language = clone_named_parameters(ema.model.model)
    before_visual = clone_named_parameters(ema.model.visual)
    with torch.no_grad():
        for parameter in full_student.model.parameters():
            if parameter.requires_grad:
                parameter.add_(2.0)

    ema.update(full_student)

    teacher_language = dict(ema.model.model.named_parameters())
    for name, before in before_language.items():
        torch.testing.assert_close(teacher_language[name], before + 1.0)
    assert_named_parameters_equal(ema.model.visual, before_visual)


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_peft_ema_updates_only_adapter_and_token_rows(dora_student, method):
    ema = Stage1EmaTeacher(dora_student, decay=0.5, method=method)
    expected = (
        "model.layers.0.self_attn.lora_A",
        "model.layers.0.self_attn.lora_B",
        "model.output_delta",
        "model.trainable_tokens_delta",
    )
    before = clone_named_parameters(ema.model)
    with torch.no_grad():
        for parameter in dora_student.parameters():
            parameter.add_(2.0)

    ema.update(dora_student)

    assert ema.parameter_names == expected
    teacher = dict(ema.model.named_parameters())
    for name, initial in before.items():
        increment = 1.0 if name in expected else 0.0
        torch.testing.assert_close(teacher[name], initial + increment)


@pytest.mark.parametrize("decay", [-0.1, 1.0, float("inf"), float("nan")])
def test_ema_rejects_invalid_decay(full_student, decay):
    with pytest.raises(ValueError, match="EMA decay"):
        Stage1EmaTeacher(full_student, decay=decay, method="full")


def test_ema_rejects_unsupported_method(full_student):
    with pytest.raises(ValueError, match="EMA method"):
        Stage1EmaTeacher(full_student, decay=0.9, method="prefix")


def test_ema_rejects_empty_selected_parameter_set(full_student):
    full_student.requires_grad_(False)

    with pytest.raises(RuntimeError, match="EMA parameter set is empty"):
        Stage1EmaTeacher(full_student, decay=0.9, method="full")


def test_ema_rejects_trainable_visual_parameter(full_student):
    full_student.visual.weight.requires_grad_(True)

    with pytest.raises(RuntimeError, match=r"trainable visual.*visual.weight"):
        Stage1EmaTeacher(full_student, decay=0.9, method="full")


def test_peft_ema_rejects_unclassified_trainable_parameter(dora_student):
    dora_student.model.register_parameter(
        "unexpected_trainable", nn.Parameter(torch.ones(2))
    )

    with pytest.raises(RuntimeError, match=r"unsupported trainable.*unexpected_trainable"):
        Stage1EmaTeacher(dora_student, decay=0.9, method="dora")


def test_parameter_map_rejects_duplicate_parameter_aliases():
    student = SingleParameterStudent()
    student.alias = student.model.weight
    teacher = SingleParameterStudent()
    teacher.alias = teacher.model.weight

    with pytest.raises(RuntimeError, match="duplicate.*alias.*model.weight"):
        build_ema_parameter_map(student, teacher, method="full")


class RecordingGather(AbstractContextManager):
    calls: list[tuple[tuple[nn.Parameter, ...], int | None]] = []
    replacement_by_id: dict[int, torch.Tensor] = {}

    def __init__(self, parameters, modifier_rank=None) -> None:
        if isinstance(parameters, nn.Parameter):
            parameters = [parameters]
        self.parameters = tuple(parameters)
        self.modifier_rank = modifier_rank
        type(self).calls.append((self.parameters, modifier_rank))

    def __enter__(self):
        for parameter in self.parameters:
            replacement = type(self).replacement_by_id.get(id(parameter))
            if replacement is not None:
                parameter.data = replacement.clone()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


@pytest.mark.parametrize("partition_teacher", [False, True])
def test_zero3_gathers_only_current_source_and_partitioned_target(
    monkeypatch, partition_teacher
):
    student = SingleParameterStudent()
    ema = Stage1EmaTeacher(student, decay=0.5, method="full")
    source = student.model.weight
    target = ema.model.model.weight
    initial_target = target.detach().clone()
    gathered_source = source.detach().clone() + 2.0
    source.ds_id = 17
    source.ds_status = "NOT_AVAILABLE"
    source.ds_shape = torch.Size((2, 2))
    source.data = torch.empty(0)
    if partition_teacher:
        target.ds_id = 23
        target.ds_status = "AVAILABLE"
        target.ds_shape = torch.Size((2, 2))

    RecordingGather.calls = []
    RecordingGather.replacement_by_id = {id(source): gathered_source}
    fake_deepspeed = SimpleNamespace(
        zero=SimpleNamespace(GatheredParameters=RecordingGather)
    )
    monkeypatch.setitem(sys.modules, "deepspeed", fake_deepspeed)

    ema.update(ModuleWrapper(student))

    assert len(RecordingGather.calls) == 1
    gathered, modifier_rank = RecordingGather.calls[0]
    expected_ids = [id(source)]
    if partition_teacher:
        expected_ids.append(id(target))
    assert [id(parameter) for parameter in gathered] == expected_ids
    assert modifier_rank == (0 if partition_teacher else None)
    torch.testing.assert_close(target, initial_target + 1.0)


def test_zero_sized_unpartitioned_source_is_rejected():
    student = SingleParameterStudent()
    ema = Stage1EmaTeacher(student, decay=0.5, method="full")
    student.model.weight.data = torch.empty(0)

    with pytest.raises(RuntimeError, match=r"EMA topology mismatch.*shape.*model.weight"):
        ema.update(student)


def test_ema_state_dict_round_trip_restores_model_and_decay(full_student):
    ema = Stage1EmaTeacher(full_student, decay=0.25, method="full")
    state = ema.state_dict()
    restored = Stage1EmaTeacher(full_student, decay=0.8, method="full")
    with torch.no_grad():
        restored.model.model.layers[0].mlp.weight.zero_()

    restored.load_state_dict(state)

    assert restored.decay == 0.25
    assert restored.method == "full"
    assert restored.parameter_names == ema.parameter_names
    assert_named_parameters_equal(restored.model, clone_named_parameters(ema.model))
    assert restored.model.training is False
    assert all(parameter.requires_grad is False for parameter in restored.model.parameters())


def test_ema_state_dict_rejects_incompatible_method(dora_student):
    dora = Stage1EmaTeacher(dora_student, decay=0.5, method="dora")
    lora = Stage1EmaTeacher(dora_student, decay=0.5, method="lora")
    state = dora.state_dict()
    state["method"] = "full"

    with pytest.raises(RuntimeError, match="EMA state method"):
        lora.load_state_dict(state)
