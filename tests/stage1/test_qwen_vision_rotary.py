from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
from flash_attn.layers.rotary import apply_rotary_emb_torch

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_rotary_pos_emb_vision,
)


QWEN_MODULES = (
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    "transformers.models.qwen2_5_vl.modular_qwen2_5_vl",
)


def _duplicated_position_embeddings(
    seq_length: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    angles = torch.linspace(-1.5, 1.5, seq_length * (head_dim // 2)).reshape(
        seq_length, head_dim // 2
    )
    cos_half = angles.cos()
    sin_half = angles.sin()
    return (
        torch.cat((cos_half, cos_half), dim=-1),
        torch.cat((sin_half, sin_half), dim=-1),
    )


def test_native_vision_rotary_executes_past_flash_kernel_grid_y_limit():
    # flash-attn 2.8.3 uses BLOCK_M=8 here, so this is the first sequence
    # length whose legacy dense rotary launch would require grid_y=65_536.
    seq_length = 524_281
    q = torch.arange(seq_length * 2, dtype=torch.float32).reshape(
        seq_length, 1, 2
    )
    k = -q
    cos = torch.ones((seq_length, 2))
    sin = torch.zeros_like(cos)

    q_rotated, k_rotated = apply_rotary_pos_emb_vision(q, k, cos, sin)

    torch.testing.assert_close(q_rotated, q)
    torch.testing.assert_close(k_rotated, k)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_native_vision_rotary_matches_legacy_torch_reference_and_gradients(dtype):
    torch.manual_seed(0)
    seq_length, num_heads, head_dim = 13, 3, 8
    cos, sin = _duplicated_position_embeddings(seq_length, head_dim)
    q_native = torch.randn(
        seq_length, num_heads, head_dim, dtype=dtype, requires_grad=True
    )
    k_native = torch.randn(
        seq_length, num_heads, head_dim, dtype=dtype, requires_grad=True
    )
    q_reference = q_native.detach().clone().requires_grad_(True)
    k_reference = k_native.detach().clone().requires_grad_(True)
    q_weight = torch.randn_like(q_native)
    k_weight = torch.randn_like(k_native)

    q_actual, k_actual = apply_rotary_pos_emb_vision(
        q_native, k_native, cos, sin
    )
    q_expected = apply_rotary_emb_torch(
        q_reference.unsqueeze(0).float(),
        cos[:, : head_dim // 2],
        sin[:, : head_dim // 2],
    ).type_as(q_reference).squeeze(0)
    k_expected = apply_rotary_emb_torch(
        k_reference.unsqueeze(0).float(),
        cos[:, : head_dim // 2],
        sin[:, : head_dim // 2],
    ).type_as(k_reference).squeeze(0)

    actual_grads = torch.autograd.grad(
        (q_actual * q_weight).sum() + (k_actual * k_weight).sum(),
        (q_native, k_native),
    )
    expected_grads = torch.autograd.grad(
        (q_expected * q_weight).sum() + (k_expected * k_weight).sum(),
        (q_reference, k_reference),
    )

    torch.testing.assert_close(q_actual, q_expected)
    torch.testing.assert_close(k_actual, k_expected)
    torch.testing.assert_close(actual_grads[0], expected_grads[0])
    torch.testing.assert_close(actual_grads[1], expected_grads[1])


@pytest.mark.parametrize("module_name", QWEN_MODULES)
def test_vision_flash_attention_uses_native_rotary_then_varlen_flash(
    monkeypatch, module_name
):
    module = importlib.import_module(module_name)
    calls = []

    def fake_flash_attention(q, k, v, cu_q, cu_k, max_q, max_k):
        calls.append((cu_q.clone(), cu_k.clone(), max_q, max_k))
        return q + 2 * k + 3 * v

    monkeypatch.setattr(module, "flash_attn_varlen_func", fake_flash_attention)
    attention = module.Qwen2_5_VLVisionFlashAttention2(dim=8, num_heads=2)
    hidden_actual = torch.randn(5, 8, requires_grad=True)
    hidden_reference = hidden_actual.detach().clone().requires_grad_(True)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    cos, sin = _duplicated_position_embeddings(5, 4)

    actual = attention(
        hidden_actual,
        cu_seqlens,
        position_embeddings=(cos, sin),
    )

    q, k, v = (
        attention.qkv(hidden_reference)
        .reshape(5, 3, 2, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    expected = attention.proj((q + 2 * k + 3 * v).reshape(5, -1))

    actual_grad = torch.autograd.grad(actual.square().sum(), hidden_actual)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), hidden_reference)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], cu_seqlens)
    torch.testing.assert_close(calls[0][1], cu_seqlens)
    assert calls[0][2:] == (3, 3)


def test_generated_and_modular_vision_flash_paths_stay_native_and_synchronized():
    repository = Path(__file__).resolve().parents[2]
    model_dir = repository / "transformers/src/transformers/models/qwen2_5_vl"

    for filename in ("modeling_qwen2_5_vl.py", "modular_qwen2_5_vl.py"):
        source = (model_dir / filename).read_text()
        flash_class = source.split(
            "class Qwen2_5_VLVisionFlashAttention2", maxsplit=1
        )[1].split("\nclass ", maxsplit=1)[0]
        assert "q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)" in flash_class
        assert "apply_rotary_pos_emb_flashatt" not in source
        assert "from flash_attn.layers.rotary import apply_rotary_emb" not in source
