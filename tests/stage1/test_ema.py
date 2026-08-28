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


class TiedEmbeddingStudent(nn.Module):
    def __init__(self, *, declared: bool = True) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=declared)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(5, 2)
        self.lm_head = nn.Linear(2, 5, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


class TwoParameterStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.first = nn.Parameter(torch.tensor([2.0]))
        self.model.second = nn.Parameter(torch.tensor([3.0]))


class CanonicalPeftStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.proj = nn.Module()
        self.model.proj.lora_A = nn.ModuleDict(
            {"default": nn.Linear(2, 1, bias=False)}
        )
        self.model.proj.lora_B = nn.ModuleDict(
            {"default": nn.Linear(1, 2, bias=False)}
        )
        self.model.proj.lora_magnitude_vector = nn.ParameterDict(
            {"default": nn.Parameter(torch.ones(2))}
        )


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


def test_peft_ema_accepts_canonical_direction_and_magnitude_components():
    ema = Stage1EmaTeacher(
        CanonicalPeftStudent(), decay=0.9, method="dora"
    )

    assert ema.parameter_names == (
        "model.proj.lora_A.default.weight",
        "model.proj.lora_B.default.weight",
        "model.proj.lora_magnitude_vector.default",
    )


@pytest.mark.parametrize(
    "deceptive_name",
    [
        "not_an_adapter_lora_backup",
        "lora_A_backup",
        "trainable_tokens_delta_backup",
        "my_output_delta_shadow",
    ],
)
def test_peft_ema_rejects_deceptive_embedded_component_names(deceptive_name):
    student = nn.Module()
    student.model = nn.Module()
    student.model.register_parameter(deceptive_name, nn.Parameter(torch.ones(1)))

    with pytest.raises(RuntimeError, match=r"unsupported trainable"):
        Stage1EmaTeacher(student, decay=0.9, method="dora")


def test_parameter_map_rejects_duplicate_parameter_aliases():
    student = SingleParameterStudent()
    student.alias = student.model.weight
    teacher = SingleParameterStudent()
    teacher.alias = teacher.model.weight

    with pytest.raises(RuntimeError, match="duplicate.*alias.*model.weight"):
        build_ema_parameter_map(student, teacher, method="full")


def test_full_ema_updates_declared_qwen_tied_embeddings_exactly_once():
    student = TiedEmbeddingStudent()
    ema = Stage1EmaTeacher(student, decay=0.5, method="full")
    before = ema.model.model.embed_tokens.weight.detach().clone()

    assert student.lm_head.weight is student.model.embed_tokens.weight
    assert ema.model.lm_head.weight is ema.model.model.embed_tokens.weight
    assert "model.embed_tokens.weight" in ema.parameter_names
    assert "lm_head.weight" not in ema.parameter_names

    with torch.no_grad():
        student.model.embed_tokens.weight.add_(2.0)
    ema.update(student)

    torch.testing.assert_close(
        ema.model.model.embed_tokens.weight,
        before + 1.0,
    )
    assert ema.model.lm_head.weight is ema.model.model.embed_tokens.weight


def test_full_ema_rejects_undeclared_qwen_embedding_alias():
    student = TiedEmbeddingStudent(declared=False)
    teacher = TiedEmbeddingStudent(declared=False)

    with pytest.raises(RuntimeError, match="duplicate.*alias.*lm_head.weight"):
        build_ema_parameter_map(student, teacher, method="full")


def test_full_ema_rejects_mismatched_teacher_tied_embedding_topology():
    student = TiedEmbeddingStudent()
    teacher = TiedEmbeddingStudent()
    teacher.lm_head.weight = nn.Parameter(
        teacher.lm_head.weight.detach().clone()
    )

    with pytest.raises(RuntimeError, match="alias topology mismatch"):
        build_ema_parameter_map(student, teacher, method="full")


def test_full_ema_rejects_declared_but_untied_embedding_topology():
    student = TiedEmbeddingStudent()
    student.lm_head.weight = nn.Parameter(
        student.lm_head.weight.detach().clone()
    )

    with pytest.raises(RuntimeError, match="declared tied.*alias topology"):
        Stage1EmaTeacher(student, decay=0.5, method="full")


def test_full_ema_rejects_shared_storage_with_distinct_embedding_parameters():
    student = TiedEmbeddingStudent()
    student.lm_head.weight = nn.Parameter(student.model.embed_tokens.weight.data)

    assert student.lm_head.weight is not student.model.embed_tokens.weight
    assert (
        student.lm_head.weight.untyped_storage().data_ptr()
        == student.model.embed_tokens.weight.untyped_storage().data_ptr()
    )
    with pytest.raises(RuntimeError, match="declared tied.*same Parameter"):
        Stage1EmaTeacher(student, decay=0.5, method="full")


def test_full_ema_rejects_extra_third_alias_on_declared_tied_embedding():
    student = TiedEmbeddingStudent()
    student.extra_alias = student.model.embed_tokens.weight

    with pytest.raises(RuntimeError, match="declared tied.*exactly"):
        Stage1EmaTeacher(student, decay=0.5, method="full")


def test_full_ema_accepts_engine_wrapped_declared_tied_embedding():
    student = TiedEmbeddingStudent()
    ema = Stage1EmaTeacher(ModuleWrapper(student), decay=0.5, method="full")

    assert ema.parameter_names == ("model.embed_tokens.weight",)


def test_declared_tied_embedding_normalizes_supported_peft_wrapper_prefix():
    import stage1.ema as ema_module

    student = TiedEmbeddingStudent()
    weight = student.model.embed_tokens.weight
    topology = ema_module._declared_qwen_tied_embedding_topology(
        student,
        {
            "base_model.model.model.embed_tokens.weight": weight,
            "base_model.model.lm_head.weight": weight,
        },
    )

    assert topology.canonical_name == (
        "base_model.model.model.embed_tokens.weight"
    )


def test_declared_tied_embedding_rejects_unsupported_wrapper_prefix():
    import stage1.ema as ema_module

    student = TiedEmbeddingStudent()
    weight = student.model.embed_tokens.weight
    with pytest.raises(RuntimeError, match="unsupported wrapper prefix"):
        ema_module._declared_qwen_tied_embedding_topology(
            student,
            {
                "untrusted.model.embed_tokens.weight": weight,
                "untrusted.lm_head.weight": weight,
            },
        )


def test_full_ema_accepts_real_local_qwen_declared_tied_embeddings():
    from tests.stage1.test_checkpointing import tiny_original_qwen

    student = tiny_original_qwen()
    student.config.tie_word_embeddings = True
    student.tie_weights()
    student.visual.requires_grad_(False)

    ema = Stage1EmaTeacher(student, decay=0.5, method="full")

    assert student.lm_head.weight is student.model.embed_tokens.weight
    assert ema.model.lm_head.weight is ema.model.model.embed_tokens.weight
    assert "model.embed_tokens.weight" in ema.parameter_names
    assert "lm_head.weight" not in ema.parameter_names


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

    assert len(RecordingGather.calls) == 2
    expected_ids = [id(source)]
    if partition_teacher:
        expected_ids.append(id(target))
    for gathered, modifier_rank in RecordingGather.calls:
        assert [id(parameter) for parameter in gathered] == expected_ids
        assert modifier_rank == (0 if partition_teacher else None)
    torch.testing.assert_close(target, initial_target + 1.0)


def test_replicated_ema_parameters_update_without_deepspeed_gather(monkeypatch):
    student = SingleParameterStudent()
    ema = Stage1EmaTeacher(student, decay=0.5, method="full")
    target = ema.model.model.weight
    initial_target = target.detach().clone()
    with torch.no_grad():
        student.model.weight.add_(2.0)
    fake_deepspeed = SimpleNamespace(
        zero=SimpleNamespace(
            GatheredParameters=lambda *args, **kwargs: pytest.fail(
                "replicated EMA tensors entered partition gathering"
            )
        )
    )
    monkeypatch.setitem(sys.modules, "deepspeed", fake_deepspeed)

    ema.update(student)

    torch.testing.assert_close(target, initial_target + 1.0)


def test_gathered_shape_preflight_prevents_partial_ema_update(monkeypatch):
    student = TwoParameterStudent()
    ema = Stage1EmaTeacher(student, decay=0.5, method="full")
    before = clone_named_parameters(ema.model)
    first = student.model.first
    second = student.model.second
    with torch.no_grad():
        first.add_(2.0)
    for ds_id, parameter in enumerate((first, second), start=1):
        parameter.ds_id = ds_id
        parameter.ds_status = "NOT_AVAILABLE"
        parameter.ds_shape = torch.Size((1,))

    RecordingGather.calls = []
    RecordingGather.replacement_by_id = {id(second): torch.zeros(2)}
    fake_deepspeed = SimpleNamespace(
        zero=SimpleNamespace(GatheredParameters=RecordingGather)
    )
    monkeypatch.setitem(sys.modules, "deepspeed", fake_deepspeed)

    with pytest.raises(
        RuntimeError, match=r"gathered source shape.*model.second"
    ):
        ema.update(student)

    assert_named_parameters_equal(ema.model, before)


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


@pytest.mark.parametrize(
    ("fixture_name", "method"),
    [("full_student", "full"), ("dora_student", "dora")],
)
def test_ema_state_dict_serializes_only_selected_parameter_snapshots(
    request, fixture_name, method
):
    """Catches duplicating frozen visual/base tensors in numbered checkpoints."""

    ema = Stage1EmaTeacher(
        request.getfixturevalue(fixture_name), decay=0.25, method=method
    )

    state = ema.state_dict()

    assert set(state) == {
        "schema_version",
        "decay",
        "method",
        "parameter_names",
        "parameter_shapes",
        "parameters",
    }
    assert tuple(state["parameters"]) == ema.parameter_names
    assert state["parameter_shapes"] == tuple(
        (name, tuple(state["parameters"][name].shape))
        for name in ema.parameter_names
    )
    assert all(tensor.device.type == "cpu" for tensor in state["parameters"].values())
    assert not any("visual" in name.split(".") for name in state["parameters"])
    if method == "dora":
        assert not any(
            name.endswith("base_layer.weight") or name.endswith("base_head.weight")
            for name in state["parameters"]
        )


def test_tied_embedding_ema_state_dict_round_trip_preserves_alias_topology():
    student = TiedEmbeddingStudent()
    ema = Stage1EmaTeacher(student, decay=0.25, method="full")
    state = ema.state_dict()
    restored = Stage1EmaTeacher(student, decay=0.8, method="full")

    restored.load_state_dict(state)

    assert restored.parameter_names == ("model.embed_tokens.weight",)
    assert restored.model.lm_head.weight is restored.model.model.embed_tokens.weight
    torch.testing.assert_close(
        restored.model.model.embed_tokens.weight,
        ema.model.model.embed_tokens.weight,
    )


def test_ema_state_dict_rejects_incompatible_method(dora_student):
    dora = Stage1EmaTeacher(dora_student, decay=0.5, method="dora")
    lora = Stage1EmaTeacher(dora_student, decay=0.5, method="lora")
    state = dora.state_dict()
    state["method"] = "full"

    with pytest.raises(RuntimeError, match="EMA state method"):
        lora.load_state_dict(state)
