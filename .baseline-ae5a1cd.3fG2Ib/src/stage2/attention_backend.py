"""Authenticated mixed-attention construction for vendored Qwen2.5-VL."""

from __future__ import annotations

from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLSdpaAttention,
    Qwen2_5_VLVisionFlashAttention2,
)


_LANGUAGE_ATTENTION = "sdpa"
_VISION_ATTENTION = "flash_attention_2"


def _require_supported_mixed_contract(
    attention_implementation: str,
    vision_attention_implementation: str,
) -> None:
    if (
        attention_implementation != _LANGUAGE_ATTENTION
        or vision_attention_implementation != _VISION_ATTENTION
    ):
        raise ValueError(
            "supported mixed attention contract requires "
            "attention_implementation='sdpa' and "
            "vision_attention_implementation='flash_attention_2'"
        )


def load_qwen2_5_vl_with_attention(
    source,
    *,
    attention_implementation,
    vision_attention_implementation,
    config_load_kwargs=None,
    model_load_kwargs=None,
    load_config=None,
    load_model=None,
) -> object:
    """Construct Qwen with SDPA language attention and FlashAttention2 vision."""
    _require_supported_mixed_contract(
        attention_implementation, vision_attention_implementation
    )
    config_kwargs = dict(config_load_kwargs or {})
    model_kwargs = dict(model_load_kwargs or {})
    if "config" in model_kwargs:
        raise ValueError("model_load_kwargs must not provide config")
    if "attn_implementation" in model_kwargs:
        raise ValueError("model_load_kwargs must not provide attn_implementation")

    load_config = load_config or AutoConfig.from_pretrained
    load_model = load_model or Qwen2_5_VLForConditionalGeneration.from_pretrained
    config = load_config(source, **config_kwargs)
    Qwen2_5_VLForConditionalGeneration._check_and_enable_flash_attn_2(
        config,
        torch_dtype=model_kwargs.get("torch_dtype"),
        device_map=model_kwargs.get("device_map"),
        hard_check_only=True,
    )

    config._attn_implementation_internal = _LANGUAGE_ATTENTION
    config._attn_implementation = _LANGUAGE_ATTENTION
    config._attn_implementation_autoset = True
    config.vision_config._attn_implementation_internal = _VISION_ATTENTION
    config.vision_config._attn_implementation = _VISION_ATTENTION
    config.vision_config._attn_implementation_autoset = True
    return load_model(source, config=config, **model_kwargs)


def validate_qwen2_5_vl_attention_backends(
    model,
    *,
    attention_implementation,
    vision_attention_implementation,
) -> None:
    """Authenticate the concrete attention modules of one resolved Qwen model."""
    _require_supported_mixed_contract(
        attention_implementation, vision_attention_implementation
    )

    visual = getattr(model, "visual", None)
    blocks = tuple(getattr(visual, "blocks", ()))
    if not blocks:
        raise ValueError("mixed Qwen requires nonempty vision attention blocks")
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        if type(attention) is not Qwen2_5_VLVisionFlashAttention2:
            raise ValueError(
                "vision attention backend mismatch at block "
                f"{index}: expected Qwen2_5_VLVisionFlashAttention2, "
                f"got {type(attention).__name__}"
            )

    decoder = getattr(model, "model", None)
    layers = tuple(getattr(decoder, "layers", ()))
    if not layers:
        raise ValueError("mixed Qwen requires nonempty language attention layers")
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if type(attention) is not Qwen2_5_VLSdpaAttention:
            raise ValueError(
                "language attention backend mismatch at layer "
                f"{index}: expected Qwen2_5_VLSdpaAttention, "
                f"got {type(attention).__name__}"
            )
