"""Sparse trainable output rows for the three Stage 1 latent tokens."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import struct
from typing import Iterator

import torch
import torch.nn.functional as F
from safetensors import SafetensorError, safe_open
from safetensors.torch import load as load_safetensors, save_file
from torch import nn

from peft_common import resolve_qwen2_5_vl_model
from stage1.ema import _is_partitioned, _parameter_shape


SPARSE_TOKEN_ROWS_FILENAME = "ilvr_token_rows.safetensors"
_FORMAT_VERSION = "1"
_METADATA_KEYS = frozenset({"format_version", "training_dtype"})
_SUPPORTED_TRAINING_DTYPES = frozenset(
    {"float16", "bfloat16", "float32", "float64"}
)
_TENSOR_KEYS = frozenset({"token_ids", "output_delta"})


def _validate_exact_latent_token_ids(
    token_ids: Sequence[int], vocab_size: int
) -> tuple[int, int, int]:
    try:
        values = tuple(token_ids)
    except TypeError as error:
        raise ValueError("latent token IDs must be a sequence") from error
    if len(values) != 3:
        raise ValueError("latent token IDs must contain exactly three values")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("latent token IDs must be integers")
    if len(set(values)) != 3:
        raise ValueError("latent token IDs must be unique")
    if any(value < 0 or value >= vocab_size for value in values):
        raise ValueError(
            f"latent token IDs must be in-vocabulary for vocab size {vocab_size}"
        )
    return values


class SparseOutputTokenRows(nn.Module):
    """Add three trainable output rows without copying or merging the LM head."""

    def __init__(self, base_head: nn.Linear, token_ids: Sequence[int]) -> None:
        super().__init__()
        if not isinstance(base_head, nn.Linear):
            raise TypeError("base_head must be a torch.nn.Linear")
        ids = _validate_exact_latent_token_ids(token_ids, base_head.out_features)
        self.base_head = base_head
        self.base_head.requires_grad_(False)
        self.register_buffer(
            "token_ids", torch.tensor(ids, dtype=torch.long), persistent=True
        )
        self.output_delta = nn.Parameter(
            torch.zeros(
                3,
                base_head.in_features,
                dtype=base_head.weight.dtype,
                device=base_head.weight.device,
            )
        )

    @property
    def weight(self) -> nn.Parameter:
        return self.base_head.weight

    @property
    def in_features(self) -> int:
        return self.base_head.in_features

    @property
    def out_features(self) -> int:
        return self.base_head.out_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_logits = self.base_head(hidden_states)
        delta_logits = F.linear(hidden_states, self.output_delta)
        return torch.index_add(
            base_logits,
            dim=-1,
            index=self.token_ids.to(device=base_logits.device),
            source=delta_logits.to(dtype=base_logits.dtype),
        )


def _attached_token_ids(rows: SparseOutputTokenRows) -> tuple[int, int, int]:
    values = tuple(rows.token_ids.detach().cpu().tolist())
    return _validate_exact_latent_token_ids(values, rows.out_features)


def _shares_parameter_storage(left: nn.Parameter, right: nn.Parameter) -> bool:
    if left is right:
        return True
    if left.device.type == "meta" or right.device.type == "meta":
        return False
    return (
        left.untyped_storage().data_ptr()
        == right.untyped_storage().data_ptr()
    )


def prepare_sparse_output_head(model: nn.Module) -> nn.Linear:
    """Untie the canonical Qwen output head for sparse-row PEFT exactly once."""

    qwen = resolve_qwen2_5_vl_model(model)
    head = qwen.lm_head
    embedding = qwen.model.embed_tokens
    if not isinstance(head, nn.Linear):
        raise TypeError("Qwen2.5-VL lm_head must be a torch.nn.Linear")
    if not isinstance(embedding, nn.Embedding):
        raise TypeError("Qwen2.5-VL input embedding must be a torch.nn.Embedding")
    if embedding.num_embeddings != head.out_features:
        raise ValueError("input and output vocabulary sizes must match")
    if embedding.weight.shape != head.weight.shape:
        raise ValueError("input and output embedding shapes must match")

    declared_tied = getattr(qwen.config, "tie_word_embeddings", None)
    if type(declared_tied) is not bool:
        raise ValueError("config.tie_word_embeddings must be exactly bool")
    shares_parameter = head.weight is embedding.weight
    shares_storage = _shares_parameter_storage(head.weight, embedding.weight)
    if declared_tied:
        if not shares_parameter:
            raise ValueError(
                "declared tied Qwen input/output embeddings have invalid tie topology"
            )
        source = head.weight
        head.weight = nn.Parameter(
            source.detach().clone(memory_format=torch.preserve_format),
            requires_grad=source.requires_grad,
        )
        qwen.config.tie_word_embeddings = False
    elif shares_storage:
        raise ValueError(
            "declared untied Qwen input/output embeddings have invalid tie topology"
        )

    if qwen.config.tie_word_embeddings is not False or _shares_parameter_storage(
        head.weight, embedding.weight
    ):
        raise ValueError("failed to establish untied sparse-output topology")
    return head


def initialize_frozen_output_token_rows(
    model: nn.Module, token_ids: Sequence[int]
) -> None:
    """Deterministically initialize appended frozen output rows for adapters."""

    qwen = resolve_qwen2_5_vl_model(model)
    head = qwen.lm_head
    embedding = qwen.model.embed_tokens
    if not isinstance(head, nn.Linear):
        raise TypeError("Qwen2.5-VL lm_head must be a torch.nn.Linear")
    if not isinstance(embedding, nn.Embedding):
        raise TypeError("Qwen2.5-VL input embedding must be a torch.nn.Embedding")
    ids = _validate_exact_latent_token_ids(token_ids, head.out_features)
    base_vocab_size = head.out_features - len(ids)
    if base_vocab_size <= 0 or ids != tuple(
        range(base_vocab_size, head.out_features)
    ):
        raise ValueError(
            "latent token IDs must be the exact appended output rows"
        )
    head = prepare_sparse_output_head(qwen)

    with torch.no_grad():
        ordinary_mean = head.weight[:base_vocab_size].float().mean(
            dim=0, keepdim=True
        )
        head.weight.index_copy_(
            0,
            torch.tensor(ids, dtype=torch.long, device=head.weight.device),
            ordinary_mean.to(dtype=head.weight.dtype).expand(len(ids), -1),
        )


def attach_sparse_output_token_rows(
    model: nn.Module, token_ids: Sequence[int]
) -> SparseOutputTokenRows:
    """Attach or return the exact sparse-row wrapper on the active Qwen head."""

    qwen = resolve_qwen2_5_vl_model(model)
    head = qwen.lm_head
    vocab_size = getattr(head, "out_features", None)
    if not isinstance(vocab_size, int):
        raise TypeError("Qwen2.5-VL lm_head does not expose out_features")
    ids = _validate_exact_latent_token_ids(token_ids, vocab_size)
    if isinstance(head, SparseOutputTokenRows):
        attached = _attached_token_ids(head)
        if attached != ids:
            raise ValueError(
                "sparse output token rows are already attached with different "
                f"token IDs: attached={attached}, requested={ids}"
            )
        return head
    if not isinstance(head, nn.Linear):
        raise TypeError("Qwen2.5-VL lm_head must be a torch.nn.Linear")

    rows = SparseOutputTokenRows(head, ids)
    qwen.lm_head = rows
    return rows


def find_sparse_output_token_rows(model: nn.Module) -> SparseOutputTokenRows:
    """Return the sparse rows attached to the active direct or wrapped Qwen."""

    qwen = resolve_qwen2_5_vl_model(model)
    rows = qwen.lm_head
    if not isinstance(rows, SparseOutputTokenRows):
        raise ValueError("sparse output token rows are not attached")
    _attached_token_ids(rows)
    return rows


@contextmanager
def _gathered_output_delta(parameter: nn.Parameter) -> Iterator[None]:
    """Gather only a partitioned sparse delta, following the Stage 1 EMA gate."""

    if not _is_partitioned(parameter):
        with nullcontext():
            yield
        return

    import deepspeed

    with deepspeed.zero.GatheredParameters([parameter], modifier_rank=None):
        yield


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def save_sparse_token_rows(model: nn.Module, output_dir: Path) -> Path:
    """Save only the three output deltas and their exact token identities."""

    rows = find_sparse_output_token_rows(model)
    ids = _attached_token_ids(rows)
    expected_shape = torch.Size((3, rows.in_features))
    if _parameter_shape(rows.output_delta) != expected_shape:
        raise ValueError(
            "sparse output_delta shape mismatch: "
            f"expected {tuple(expected_shape)}, got "
            f"{tuple(_parameter_shape(rows.output_delta))}"
        )

    with _gathered_output_delta(rows.output_delta):
        if torch.Size(rows.output_delta.shape) != expected_shape:
            raise ValueError(
                "gathered sparse output_delta shape mismatch: "
                f"expected {tuple(expected_shape)}, got "
                f"{tuple(rows.output_delta.shape)}"
            )
        output_delta = rows.output_delta.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        if not torch.isfinite(output_delta).all():
            raise ValueError("sparse output_delta must contain only finite values")
        training_dtype = _dtype_name(rows.output_delta.dtype)

    if training_dtype not in _SUPPORTED_TRAINING_DTYPES:
        raise ValueError(f"unsupported sparse output_delta dtype: {training_dtype}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / SPARSE_TOKEN_ROWS_FILENAME
    save_file(
        {
            "token_ids": torch.tensor(ids, dtype=torch.int64),
            "output_delta": output_delta,
        },
        artifact,
        metadata={
            "format_version": _FORMAT_VERSION,
            "training_dtype": training_dtype,
        },
    )
    return artifact


def _validate_metadata(metadata: dict[str, str] | None) -> None:
    if metadata is None or set(metadata) != _METADATA_KEYS:
        raise ValueError("invalid sparse token-row metadata fields")
    if metadata["format_version"] != _FORMAT_VERSION:
        raise ValueError("invalid sparse token-row metadata format_version")
    if metadata["training_dtype"] not in _SUPPORTED_TRAINING_DTYPES:
        raise ValueError("invalid sparse token-row metadata training_dtype")


def _safetensors_metadata(payload: bytes) -> dict[str, str] | None:
    try:
        if len(payload) < 8:
            raise ValueError("safetensors payload has no header")
        header_size = struct.unpack("<Q", payload[:8])[0]
        header_end = 8 + header_size
        if header_end > len(payload):
            raise ValueError("safetensors header exceeds payload")
        header = json.loads(payload[8:header_end])
    except (UnicodeError, json.JSONDecodeError, struct.error) as error:
        raise ValueError("invalid safetensors metadata header") from error
    metadata = header.get("__metadata__")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        raise ValueError("invalid safetensors metadata values")
    return metadata


def decode_sparse_token_rows_state(
    payload: bytes,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode and validate one sparse-row safetensors byte payload."""

    if type(payload) is not bytes:
        raise TypeError("sparse token-row payload must be bytes")
    try:
        tensors = load_safetensors(payload)
    except SafetensorError as error:
        raise ValueError("invalid sparse token-row artifact bytes") from error
    keys = set(tensors)
    if keys != _TENSOR_KEYS:
        raise ValueError(
            "invalid sparse token-row tensor fields: "
            f"expected={sorted(_TENSOR_KEYS)}, actual={sorted(keys)}"
        )
    _validate_metadata(_safetensors_metadata(payload))
    stored_ids = tensors["token_ids"]
    output_delta = tensors["output_delta"]
    if stored_ids.dtype != torch.int64:
        raise ValueError("sparse token_ids tensor must have int64 dtype")
    if stored_ids.shape != torch.Size((3,)):
        raise ValueError(
            "sparse token_ids shape mismatch: expected (3,), "
            f"got {tuple(stored_ids.shape)}"
        )
    if output_delta.dtype != torch.float32:
        raise ValueError("sparse output_delta tensor must have float32 dtype")
    if not torch.isfinite(output_delta).all():
        raise ValueError("sparse output_delta must contain only finite values")
    return stored_ids, output_delta


def apply_sparse_token_rows_state(
    model: nn.Module,
    payload: bytes,
    expected_token_ids: Sequence[int],
) -> None:
    """Restore validated sparse rows directly from immutable artifact bytes."""

    rows = find_sparse_output_token_rows(model)
    expected = _validate_exact_latent_token_ids(
        expected_token_ids, rows.out_features
    )
    attached = _attached_token_ids(rows)
    if attached != expected:
        raise ValueError(
            "attached token IDs do not match expected token IDs: "
            f"attached={attached}, expected={expected}"
        )

    stored_ids, output_delta = decode_sparse_token_rows_state(payload)
    stored_tuple = tuple(stored_ids.tolist())
    if stored_tuple != expected:
        raise ValueError(
            "stored token IDs do not match expected token IDs: "
            f"stored={stored_tuple}, expected={expected}"
        )
    expected_shape = torch.Size((3, rows.in_features))
    if output_delta.shape != expected_shape:
        raise ValueError(
            "sparse output_delta shape mismatch: "
            f"expected {tuple(expected_shape)}, got {tuple(output_delta.shape)}"
        )
    with torch.no_grad():
        rows.output_delta.copy_(
            output_delta.to(
                device=rows.output_delta.device, dtype=rows.output_delta.dtype
            )
        )


def load_sparse_token_rows(
    model: nn.Module,
    artifact_dir: Path,
    expected_token_ids: Sequence[int],
) -> None:
    """Validate and restore a sparse-row artifact without touching the base head."""

    artifact = Path(artifact_dir) / SPARSE_TOKEN_ROWS_FILENAME
    if not artifact.is_file():
        raise FileNotFoundError(f"missing sparse token-row artifact: {artifact}")
    try:
        payload = artifact.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid sparse token-row artifact: {artifact}") from error
    try:
        apply_sparse_token_rows_state(model, payload, expected_token_ids)
    except ValueError as error:
        if "artifact bytes" in str(error):
            raise ValueError(
                f"invalid sparse token-row artifact: {artifact}"
            ) from error
        raise


__all__ = [
    "SparseOutputTokenRows",
    "apply_sparse_token_rows_state",
    "attach_sparse_output_token_rows",
    "find_sparse_output_token_rows",
    "decode_sparse_token_rows_state",
    "initialize_frozen_output_token_rows",
    "prepare_sparse_output_head",
    "load_sparse_token_rows",
    "save_sparse_token_rows",
]
