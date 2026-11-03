"""Immutable, model-local controls for Stage 2 objective forwards."""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .stage2_latent_rollout import Stage2DecoderCallStats


@dataclass(frozen=True)
class LocalLatentIntervention:
    start_marker: int
    end_marker: int
    block_markers: bool = False

    def __post_init__(self):
        if type(self.block_markers) is not bool:
            raise ValueError("block_markers must be boolean")


@dataclass(frozen=True)
class ObjectiveForwardRequest:
    intervention: LocalLatentIntervention | None = None
    allow_empty_ce: bool = True

    def __post_init__(self):
        if self.intervention is not None and not isinstance(self.intervention, LocalLatentIntervention):
            raise ValueError("intervention must be a LocalLatentIntervention")
        if type(self.allow_empty_ce) is not bool:
            raise ValueError("allow_empty_ce must be boolean")


def build_objective_attention_mask(
    padding: torch.Tensor, *, length: int,
    intervention: LocalLatentIntervention | None, dtype: torch.dtype,
) -> torch.Tensor:
    """Causal/key-padding visibility with a local latent-body key restriction.

    Coordinates always refer to the original sequence, even for short prefixes.
    The end marker itself cannot read the selected body, avoiding marker leakage.
    """
    if padding.ndim != 2 or not bool(((padding == 0) | (padding == 1)).all()):
        raise ValueError("padding must be a binary two-dimensional tensor")
    if type(length) is not int or not 0 < length <= padding.shape[1]:
        raise ValueError("length must be positive and within padding extent")
    if not dtype.is_floating_point:
        raise ValueError("objective attention mask requires floating dtype")
    if intervention is not None:
        if not isinstance(intervention, LocalLatentIntervention):
            raise ValueError("intervention must be a LocalLatentIntervention")
        start, end = intervention.start_marker, intervention.end_marker
        if (type(start) is not int or type(end) is not int
                or not (0 <= start and start + 1 < end < padding.shape[1])):
            raise ValueError("intervention requires ordered in-range markers and at least one slot")
    positions = torch.arange(length, device=padding.device)
    q, k = positions[:, None], positions[None, :]
    allowed = (k <= q)[None, None] & padding[:, None, None, :length].bool()
    if intervention is not None:
        if intervention.block_markers:
            blocked_keys = ((k >= intervention.start_marker)
                            & (k <= intervention.end_marker))
        else:
            blocked_keys = ((k > intervention.start_marker)
                            & (k < intervention.end_marker))
        blocked = (q >= intervention.end_marker) & blocked_keys
        allowed = allowed & ~blocked[None, None]
    mask = torch.zeros(allowed.shape, device=padding.device, dtype=dtype)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)


@dataclass
class ObjectiveReplayStats:
    """One invocation's live counters; checkpoint callbacks retain this object.

    ``rollout_stats`` is the engine's live Stage2DecoderCallStats, not a copy.
    Keeping the carrier model-local avoids importing training application code.
    """
    rollout_stats: "Stage2DecoderCallStats | None" = None
    final_decoder_forward_calls: int = 0
    native_checkpoint_recompute_calls: int = 0

    def observe_checkpoint(self, event: str) -> None:
        if event == "layer_recompute":
            self.native_checkpoint_recompute_calls += 1

    @property
    def decoder_forward_calls(self) -> int:
        return self.final_decoder_forward_calls + (
            self.rollout_stats.clean_forward_calls if self.rollout_stats is not None else 0
        )

    @property
    def prefix_checkpoint_replay_calls(self) -> int:
        return self.rollout_stats.checkpoint_recompute_calls if self.rollout_stats is not None else 0
