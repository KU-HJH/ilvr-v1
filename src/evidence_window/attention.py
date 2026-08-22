"""Pair-specific boolean attention blocks for Stage-2 interventions."""

import torch

from .stage2_layout import EvidenceWindowEdge, Stage2InterleavedLayout


def build_window_local_block_mask(
    layout: Stage2InterleavedLayout,
    edge: EvidenceWindowEdge,
) -> torch.Tensor:
    """Block one causal window's attention to one earlier evidence span.

    The returned boolean mask has shape ``[B, T, T]``.  ``True`` denotes a
    blocked query/key pair, with query positions on the second axis.
    """
    _validate_layout(layout)
    if not isinstance(edge, EvidenceWindowEdge):
        raise ValueError("edge must be an EvidenceWindowEdge")

    batch_size, evidence_count, sequence_length = layout.evidence_token_mask.shape
    window_count = layout.window_token_mask.shape[1]
    if not 0 <= edge.batch_index < batch_size:
        raise ValueError("edge batch_index is out of range")
    if not 0 <= edge.evidence_index < evidence_count:
        raise ValueError("edge evidence_index is out of range")
    if not 0 <= edge.window_index < window_count:
        raise ValueError("edge window_index is out of range")
    if not bool(
        layout.causal_pair_mask[edge.batch_index, edge.evidence_index, edge.window_index]
    ):
        raise ValueError("edge must be causal")

    block_mask = torch.zeros(
        (batch_size, sequence_length, sequence_length),
        dtype=torch.bool,
        device=layout.window_token_mask.device,
    )
    block_mask[edge.batch_index] = (
        layout.window_token_mask[edge.batch_index, edge.window_index][:, None]
        & layout.evidence_token_mask[edge.batch_index, edge.evidence_index][None, :]
    )
    return block_mask


def slice_block_mask(mask: torch.Tensor, prefix_length: int) -> torch.Tensor:
    """Return the square query/key prefix of a boolean block mask."""
    if mask.ndim != 3 or mask.shape[1] != mask.shape[2]:
        raise ValueError("mask must have shape [B, T, T]")
    if mask.dtype is not torch.bool:
        raise ValueError("mask must be boolean")
    if isinstance(prefix_length, bool) or not isinstance(prefix_length, int):
        raise ValueError("prefix_length must be an integer")
    if not 0 <= prefix_length <= mask.shape[1]:
        raise ValueError("prefix_length is out of range")
    return mask[:, :prefix_length, :prefix_length]


def _validate_layout(layout: Stage2InterleavedLayout) -> None:
    if not isinstance(layout, Stage2InterleavedLayout):
        raise ValueError("layout must be a Stage2InterleavedLayout")
    evidence = layout.evidence_token_mask
    windows = layout.window_token_mask
    causal = layout.causal_pair_mask
    if evidence.ndim != 3 or windows.ndim != 3 or causal.ndim != 3:
        raise ValueError("layout masks must be rank three")
    batch_size, evidence_count, sequence_length = evidence.shape
    if windows.shape[0] != batch_size or windows.shape[2] != sequence_length:
        raise ValueError("evidence and window masks must share batch and sequence dimensions")
    if causal.shape != (batch_size, evidence_count, windows.shape[1]):
        raise ValueError("causal_pair_mask must match evidence and window dimensions")
    tensors = (
        evidence,
        windows,
        layout.evidence_valid,
        layout.window_valid,
        causal,
    )
    if any(value.dtype is not torch.bool for value in tensors):
        raise ValueError("layout masks must be boolean")
    if any(value.device != evidence.device for value in tensors):
        raise ValueError("layout masks must share a device")
    if layout.evidence_valid.shape != (batch_size, evidence_count):
        raise ValueError("evidence_valid must match evidence dimensions")
    if layout.window_valid.shape != (batch_size, windows.shape[1]):
        raise ValueError("window_valid must match window dimensions")
