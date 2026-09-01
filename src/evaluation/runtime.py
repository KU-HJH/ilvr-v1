"""Prompt preparation and lineage helpers shared by evaluation entry points."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import torch

from stage1.equivalence import canonical_json_bytes, canonical_tensor_sha256


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
        [field_name, canonical_tensor_sha256(tensors[field_name])]
        for field_name in _PROMPT_IDENTITY_FIELDS
        if field_name in tensors
    ]
    return hashlib.sha256(canonical_json_bytes(ordered_digests)).hexdigest()


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
