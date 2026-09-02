"""Prompt preparation and lineage helpers shared by evaluation entry points."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sys
from typing import Any, Mapping

import torch


_PROMPT_IDENTITY_FIELDS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
)


@dataclass(frozen=True)
class PreparedEvaluationInput:
    """The rendered prompt, its processor tensors, and stable lineage identity."""

    prompt: str
    tensors: Mapping[str, torch.Tensor]
    prompt_token_length: int
    prompt_identity: str


def _prompt_identity(tensors: Mapping[str, torch.Tensor]) -> str:
    ordered_digests = [
        [field_name, _canonical_tensor_sha256(tensors[field_name])]
        for field_name in _PROMPT_IDENTITY_FIELDS
        if field_name in tensors
    ]
    return hashlib.sha256(_canonical_json_bytes(ordered_digests)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash prompt tensor metadata and contiguous little-endian CPU bytes."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("canonical tensor hash requires a torch.Tensor")
    if tensor.layout is not torch.strided:
        raise ValueError("canonical tensor hash requires a strided tensor")
    if tensor.is_quantized:
        raise ValueError("canonical tensor hash does not support quantized tensors")
    if tensor.device.type == "meta":
        raise ValueError("canonical tensor hash requires materialized tensor values")
    cpu_tensor = tensor.detach().cpu().contiguous()
    if (cpu_tensor.is_floating_point() or cpu_tensor.is_complex()) and not bool(
        torch.isfinite(cpu_tensor).all()
    ):
        raise ValueError("canonical tensor hash requires finite tensor values")
    payload = cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    if sys.byteorder == "big" and cpu_tensor.element_size() > 1:
        component_size = (
            cpu_tensor.element_size() // 2
            if cpu_tensor.is_complex()
            else cpu_tensor.element_size()
        )
        payload = b"".join(
            payload[offset : offset + component_size][::-1]
            for offset in range(0, len(payload), component_size)
        )
    metadata = _canonical_json_bytes(
        {"dtype": str(cpu_tensor.dtype), "shape": list(cpu_tensor.shape)}
    )
    return hashlib.sha256(metadata + payload).hexdigest()


def prepare_evaluation_input(
    processor: Any, sample: Mapping[str, Any], images: Any
) -> PreparedEvaluationInput:
    """Render a user-only prompt without ever consulting its gold annotation."""
    content = []
    if images:
        content.extend({"type": "image"} for _ in images)
    content.append({"type": "text", "text": sample.get("text_input", "")})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    processor_tensors = processor(
        text=[prompt], images=images, return_tensors="pt", padding=True
    )
    tensors = {
        name: value
        for name, value in processor_tensors.items()
        if isinstance(value, torch.Tensor)
    }
    input_ids = tensors.get("input_ids")
    if input_ids is None:
        raise ValueError("processor output must contain input_ids")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError("evaluation currently supports batch_size=1 only")
    return PreparedEvaluationInput(
        prompt=prompt,
        tensors=tensors,
        prompt_token_length=int(input_ids.shape[1]),
        prompt_identity=_prompt_identity(tensors),
    )
