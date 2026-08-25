from __future__ import annotations

import sys
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from stage1.token_rows import (
    SparseOutputTokenRows,
    attach_sparse_output_token_rows,
    find_sparse_output_token_rows,
    load_sparse_token_rows,
    save_sparse_token_rows,
)


ARTIFACT_NAME = "ilvr_token_rows.safetensors"


class TinyQwen(nn.Module):
    def __init__(
        self, *, hidden_size: int = 4, vocab_size: int = 9, dtype=torch.float32
    ) -> None:
        super().__init__()
        self.model = nn.Identity()
        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=False, dtype=dtype
        )


class ModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module


def wrapped_tiny_head(*, dtype=torch.float32, token_ids=(4, 5, 6)) -> TinyQwen:
    model = TinyQwen(dtype=dtype)
    wrapper = attach_sparse_output_token_rows(model, token_ids)
    with torch.no_grad():
        wrapper.output_delta.copy_(
            torch.tensor(
                [
                    [0.125, -0.25, 0.5, -1.0],
                    [1.5, 0.75, -0.5, 0.25],
                    [-2.0, 0.0, 1.25, 0.5],
                ],
                dtype=dtype,
            )
        )
    return model


def write_artifact(
    directory: Path,
    *,
    token_ids: torch.Tensor | None = None,
    output_delta: torch.Tensor | None = None,
    metadata: dict[str, str] | None = None,
) -> Path:
    path = directory / ARTIFACT_NAME
    tensors = {}
    if token_ids is not None:
        tensors["token_ids"] = token_ids
    if output_delta is not None:
        tensors["output_delta"] = output_delta
    save_file(
        tensors,
        path,
        metadata=metadata
        if metadata is not None
        else {"format_version": "1", "training_dtype": "float32"},
    )
    return path


def test_output_delta_has_hand_derived_logits_and_changes_only_three_rows():
    base = nn.Linear(2, 5, bias=False)
    with torch.no_grad():
        base.weight.copy_(
            torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0], [2.0, -1.0]]
            )
        )
    wrapper = SparseOutputTokenRows(base, token_ids=(0, 2, 4))
    with torch.no_grad():
        wrapper.output_delta.copy_(
            torch.tensor([[1.0, 1.0], [2.0, 0.0], [0.0, -3.0]])
        )
    hidden = torch.tensor([[2.0, -1.0]])
    base_logits = base(hidden)

    logits = wrapper(hidden)

    torch.testing.assert_close(
        logits, torch.tensor([[3.0, -1.0, 5.0, -4.0, 8.0]]), rtol=0, atol=0
    )
    assert torch.equal(logits[..., (1, 3)], base_logits[..., (1, 3)])
    torch.testing.assert_close(
        logits[..., (0, 2, 4)] - base_logits[..., (0, 2, 4)],
        F.linear(hidden, wrapper.output_delta),
        rtol=0,
        atol=0,
    )


def test_zero_initialized_attachment_preserves_all_logits_bit_exactly():
    torch.manual_seed(7)
    model = TinyQwen()
    hidden = torch.randn(2, 3, 4)
    before = model.lm_head(hidden)
    base_weight = model.lm_head.weight.detach().clone()

    wrapper = attach_sparse_output_token_rows(model, (1, 3, 8))

    assert torch.equal(wrapper(hidden), before)
    assert torch.equal(wrapper.base_head.weight, base_weight)
    assert torch.count_nonzero(wrapper.output_delta) == 0


def test_only_output_delta_receives_gradient():
    wrapper = SparseOutputTokenRows(
        nn.Linear(4, 9, bias=False), token_ids=(1, 3, 8)
    )

    wrapper(torch.randn(2, 4)).sum().backward()

    assert wrapper.output_delta.grad is not None
    assert torch.isfinite(wrapper.output_delta.grad).all()
    assert wrapper.base_head.weight.grad is None
    trainable = [
        name
        for name, value in wrapper.named_parameters()
        if value.requires_grad
    ]
    assert trainable == ["output_delta"]


def test_wrapper_preserves_linear_head_interface_without_copying_base():
    base = nn.Linear(4, 9, bias=False)
    wrapper = SparseOutputTokenRows(base, (1, 3, 8))

    assert wrapper.weight is base.weight
    assert wrapper.in_features == 4
    assert wrapper.out_features == 9
    assert wrapper.output_delta.shape == (3, 4)
    assert wrapper.base_head is base


@pytest.mark.parametrize(
    ("token_ids", "message"),
    [
        ((1, 2), "exactly three"),
        ((1, 2, 3, 4), "exactly three"),
        ((1, 1, 2), "unique"),
        ((-1, 1, 2), "in-vocabulary"),
        ((1, 2, 9), "in-vocabulary"),
        ((1, 2, 3.0), "integers"),
        ((1, 2, True), "integers"),
    ],
)
def test_constructor_rejects_malformed_token_ids(token_ids, message):
    with pytest.raises(ValueError, match=message):
        SparseOutputTokenRows(nn.Linear(4, 9, bias=False), token_ids)


def test_attach_and_find_resolve_direct_and_engine_wrapped_models():
    direct = TinyQwen()
    direct_rows = attach_sparse_output_token_rows(direct, (7, 2, 5))
    wrapped = ModuleWrapper(TinyQwen())
    wrapped_rows = attach_sparse_output_token_rows(wrapped, (7, 2, 5))

    assert direct.lm_head is direct_rows
    assert wrapped.module.lm_head is wrapped_rows
    assert find_sparse_output_token_rows(direct) is direct_rows
    assert find_sparse_output_token_rows(wrapped) is wrapped_rows
    assert direct_rows.token_ids.tolist() == [7, 2, 5]
    assert wrapped_rows.token_ids.tolist() == [7, 2, 5]


def test_repeat_attachment_is_idempotent_only_for_exact_token_ids():
    model = TinyQwen()
    first = attach_sparse_output_token_rows(model, (1, 3, 8))

    assert attach_sparse_output_token_rows(model, (1, 3, 8)) is first
    with pytest.raises(ValueError, match="already attached.*token IDs"):
        attach_sparse_output_token_rows(model, (1, 8, 3))


def test_attach_and_find_reject_non_qwen_or_unattached_heads():
    with pytest.raises(TypeError, match="Qwen2.5-VL"):
        attach_sparse_output_token_rows(nn.Linear(2, 2), (0, 1, 2))
    with pytest.raises(ValueError, match="not attached"):
        find_sparse_output_token_rows(TinyQwen())


def test_sparse_rows_round_trip_through_fp32_container_from_bfloat16(tmp_path):
    original = wrapped_tiny_head(dtype=torch.bfloat16, token_ids=(4, 5, 6))
    original_base = (
        find_sparse_output_token_rows(original).base_head.weight.detach().clone()
    )

    artifact = save_sparse_token_rows(original, tmp_path)

    assert artifact == tmp_path / ARTIFACT_NAME
    assert sorted(path.name for path in tmp_path.iterdir()) == [ARTIFACT_NAME]
    with safe_open(artifact, framework="pt", device="cpu") as stored:
        assert list(stored.keys()) == ["output_delta", "token_ids"]
        assert stored.get_tensor("token_ids").dtype == torch.int64
        assert stored.get_tensor("output_delta").dtype == torch.float32
        assert stored.metadata() == {
            "format_version": "1",
            "training_dtype": "bfloat16",
        }

    restored = TinyQwen(dtype=torch.bfloat16)
    restored_base = restored.lm_head.weight.detach().clone()
    attach_sparse_output_token_rows(restored, (4, 5, 6))
    load_sparse_token_rows(restored, tmp_path, expected_token_ids=(4, 5, 6))

    torch.testing.assert_close(
        find_sparse_output_token_rows(original).output_delta,
        find_sparse_output_token_rows(restored).output_delta,
        rtol=0,
        atol=0,
    )
    assert find_sparse_output_token_rows(restored).output_delta.dtype == torch.bfloat16
    assert torch.equal(
        find_sparse_output_token_rows(original).base_head.weight, original_base
    )
    assert torch.equal(
        find_sparse_output_token_rows(restored).base_head.weight, restored_base
    )


def test_save_creates_output_directory(tmp_path):
    output_dir = tmp_path / "new" / "artifact"

    path = save_sparse_token_rows(wrapped_tiny_head(), output_dir)

    assert path.is_file()


def test_load_rejects_missing_or_invalid_safetensors_file(tmp_path):
    model = wrapped_tiny_head()
    with pytest.raises(FileNotFoundError, match=ARTIFACT_NAME):
        load_sparse_token_rows(model, tmp_path, (4, 5, 6))

    (tmp_path / ARTIFACT_NAME).write_bytes(b"not safetensors")
    with pytest.raises(ValueError, match="invalid.*token-row artifact"):
        load_sparse_token_rows(model, tmp_path, (4, 5, 6))


@pytest.mark.parametrize(
    "tensors",
    [
        {"token_ids": torch.tensor([4, 5, 6], dtype=torch.int64)},
        {"output_delta": torch.zeros(3, 4, dtype=torch.float32)},
        {
            "token_ids": torch.tensor([4, 5, 6], dtype=torch.int64),
            "output_delta": torch.zeros(3, 4, dtype=torch.float32),
            "unexpected": torch.zeros(1),
        },
    ],
)
def test_load_rejects_malformed_tensor_set(tmp_path, tensors):
    save_file(
        tensors,
        tmp_path / ARTIFACT_NAME,
        metadata={"format_version": "1", "training_dtype": "float32"},
    )

    with pytest.raises(ValueError, match="tensor fields"):
        load_sparse_token_rows(wrapped_tiny_head(), tmp_path, (4, 5, 6))


@pytest.mark.parametrize(
    "metadata",
    [
        {"training_dtype": "float32"},
        {"format_version": "2", "training_dtype": "float32"},
        {"format_version": "1"},
        {"format_version": "1", "training_dtype": "int64"},
        {
            "format_version": "1",
            "training_dtype": "float32",
            "unexpected": "value",
        },
    ],
)
def test_load_rejects_malformed_metadata(tmp_path, metadata):
    write_artifact(
        tmp_path,
        token_ids=torch.tensor([4, 5, 6], dtype=torch.int64),
        output_delta=torch.zeros(3, 4, dtype=torch.float32),
        metadata=metadata,
    )

    with pytest.raises(ValueError, match="metadata"):
        load_sparse_token_rows(wrapped_tiny_head(), tmp_path, (4, 5, 6))


@pytest.mark.parametrize(
    ("token_ids", "output_delta", "message"),
    [
        (torch.tensor([4, 5, 6], dtype=torch.int32), torch.zeros(3, 4), "int64"),
        (torch.tensor([[4, 5, 6]], dtype=torch.int64), torch.zeros(3, 4), "shape"),
        (
            torch.tensor([4, 5, 6], dtype=torch.int64),
            torch.zeros(3, 4, dtype=torch.bfloat16),
            "float32",
        ),
        (torch.tensor([4, 5, 6], dtype=torch.int64), torch.zeros(3, 5), "shape"),
    ],
)
def test_load_rejects_tensor_dtype_and_shape_mismatches(
    tmp_path, token_ids, output_delta, message
):
    write_artifact(tmp_path, token_ids=token_ids, output_delta=output_delta)

    with pytest.raises(ValueError, match=message):
        load_sparse_token_rows(wrapped_tiny_head(), tmp_path, (4, 5, 6))


def test_load_rejects_stored_or_expected_token_id_mismatch(tmp_path):
    write_artifact(
        tmp_path,
        token_ids=torch.tensor([4, 6, 5], dtype=torch.int64),
        output_delta=torch.zeros(3, 4, dtype=torch.float32),
    )
    model = wrapped_tiny_head()

    with pytest.raises(ValueError, match="stored token IDs.*expected"):
        load_sparse_token_rows(model, tmp_path, (4, 5, 6))
    with pytest.raises(ValueError, match="attached token IDs.*expected"):
        load_sparse_token_rows(model, tmp_path, (4, 6, 5))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_load_rejects_nonfinite_output_delta_without_mutation(tmp_path, bad_value):
    delta = torch.zeros(3, 4, dtype=torch.float32)
    delta[1, 2] = bad_value
    write_artifact(
        tmp_path,
        token_ids=torch.tensor([4, 5, 6], dtype=torch.int64),
        output_delta=delta,
    )
    model = wrapped_tiny_head()
    rows = find_sparse_output_token_rows(model)
    before = rows.output_delta.detach().clone()

    with pytest.raises(ValueError, match="finite"):
        load_sparse_token_rows(model, tmp_path, (4, 5, 6))

    assert torch.equal(rows.output_delta, before)


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


def test_save_gathers_only_zero_partitioned_output_delta(monkeypatch, tmp_path):
    model = wrapped_tiny_head()
    rows = find_sparse_output_token_rows(model)
    delta = rows.output_delta
    gathered = delta.detach().clone() + 3.0
    delta.ds_id = 17
    delta.ds_status = "NOT_AVAILABLE"
    delta.ds_shape = torch.Size((3, 4))
    delta.data = torch.empty(0)
    RecordingGather.calls = []
    RecordingGather.replacement_by_id = {id(delta): gathered}
    monkeypatch.setitem(
        sys.modules,
        "deepspeed",
        SimpleNamespace(zero=SimpleNamespace(GatheredParameters=RecordingGather)),
    )

    artifact = save_sparse_token_rows(ModuleWrapper(model), tmp_path)

    assert len(RecordingGather.calls) == 1
    parameters, modifier_rank = RecordingGather.calls[0]
    assert [id(parameter) for parameter in parameters] == [id(delta)]
    assert modifier_rank is None
    with safe_open(artifact, framework="pt", device="cpu") as stored:
        torch.testing.assert_close(stored.get_tensor("output_delta"), gathered)
