"""Sparse trainable output rows for the three Stage 1 latent tokens."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file
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


def load_sparse_token_rows(
    model: nn.Module,
    artifact_dir: Path,
    expected_token_ids: Sequence[int],
) -> None:
    """Validate and restore a sparse-row artifact without touching the base head."""

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

    artifact = Path(artifact_dir) / SPARSE_TOKEN_ROWS_FILENAME
    if not artifact.is_file():
        raise FileNotFoundError(f"missing sparse token-row artifact: {artifact}")
    try:
        with safe_open(artifact, framework="pt", device="cpu") as stored:
            keys = set(stored.keys())
            if keys != _TENSOR_KEYS:
                raise ValueError(
                    "invalid sparse token-row tensor fields: "
                    f"expected={sorted(_TENSOR_KEYS)}, actual={sorted(keys)}"
                )
            _validate_metadata(stored.metadata())
            stored_ids = stored.get_tensor("token_ids")
            output_delta = stored.get_tensor("output_delta")
    except SafetensorError as error:
        raise ValueError(f"invalid sparse token-row artifact: {artifact}") from error

    if stored_ids.dtype != torch.int64:
        raise ValueError("sparse token_ids tensor must have int64 dtype")
    if stored_ids.shape != torch.Size((3,)):
        raise ValueError(
            "sparse token_ids shape mismatch: expected (3,), "
            f"got {tuple(stored_ids.shape)}"
        )
    stored_tuple = tuple(stored_ids.tolist())
    if stored_tuple != expected:
        raise ValueError(
            "stored token IDs do not match expected token IDs: "
            f"stored={stored_tuple}, expected={expected}"
        )
    if output_delta.dtype != torch.float32:
        raise ValueError("sparse output_delta tensor must have float32 dtype")
    expected_shape = torch.Size((3, rows.in_features))
    if output_delta.shape != expected_shape:
        raise ValueError(
            "sparse output_delta shape mismatch: "
            f"expected {tuple(expected_shape)}, got {tuple(output_delta.shape)}"
        )
    if not torch.isfinite(output_delta).all():
        raise ValueError("sparse output_delta must contain only finite values")

    with torch.no_grad():
        rows.output_delta.copy_(
            output_delta.to(
                device=rows.output_delta.device, dtype=rows.output_delta.dtype
            )
        )


__all__ = [
    "SparseOutputTokenRows",
    "attach_sparse_output_token_rows",
    "find_sparse_output_token_rows",
    "load_sparse_token_rows",
    "save_sparse_token_rows",
]
