"""Reference-provider boundary for explicit ILVR Stage 2."""

from __future__ import annotations

from typing import Protocol, Sequence

import torch

from .artifacts import (
    ArtifactIdentity,
    PrecomputedReferenceProvider,
    ReferenceBatchTargets,
)
from .stage2_config import Stage2EvidenceWindowConfig
from .stage2_layout import Stage2InterleavedLayout


class Stage2ReferenceProvider(Protocol):
    def get_batch(
        self,
        sample_indices: Sequence[int] | torch.Tensor,
        layout: Stage2InterleavedLayout,
        *,
        require_hidden_residuals: bool = False,
    ) -> ReferenceBatchTargets: ...


def build_stage2_reference_provider(
    config: Stage2EvidenceWindowConfig,
    *,
    expected_identity: ArtifactIdentity | None = None,
) -> Stage2ReferenceProvider | None:
    """Build only the configured immutable reference source."""
    if not isinstance(config, Stage2EvidenceWindowConfig):
        raise TypeError("config must be a Stage2EvidenceWindowConfig")
    if config.reference.mode == "none":
        if expected_identity is not None:
            raise ValueError("reference-free mode does not accept an artifact identity")
        return None
    if config.reference.mode == "online_frozen":
        raise NotImplementedError("online frozen Stage-2 reference is implemented separately")
    assert config.reference.artifact is not None
    if expected_identity is None:
        raise ValueError("precomputed reference requires an independent expected identity")
    return PrecomputedReferenceProvider(config.reference.artifact, expected_identity)


__all__ = ["Stage2ReferenceProvider", "build_stage2_reference_provider"]
