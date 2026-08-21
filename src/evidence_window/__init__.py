"""Configuration and runtime helpers for evidence-window training."""

from .config import EvidenceWindowConfig
from .layout import InterleavedLayout, TokenIds, extract_interleaved_layout
from .interventions import apply_all_null, apply_zero_latent_loo, select_loo_spans
from .losses import (
    evidence_window_preference_loss,
    global_preference_loss,
    window_preference_loss,
)
from .metrics import (
    ContributionSummaries,
    MatrixJsonlWriter,
    MetricBatch,
    contribution_summaries,
    global_outer_coefficient,
    reduce_sufficient_statistics,
)
from .scores import (
    causal_token_logps,
    evidence_window_margins,
    masked_token_sums,
    reference_adjusted_effect,
)
from .reference import (
    AdapterDisabledReference,
    FrozenCopyReference,
    ParameterWitness,
    ReferenceProvider,
    build_reference_provider,
)

__all__ = [
    "EvidenceWindowConfig",
    "InterleavedLayout",
    "TokenIds",
    "extract_interleaved_layout",
    "apply_zero_latent_loo",
    "apply_all_null",
    "select_loo_spans",
    "causal_token_logps",
    "masked_token_sums",
    "reference_adjusted_effect",
    "evidence_window_margins",
    "global_preference_loss",
    "window_preference_loss",
    "evidence_window_preference_loss",
    "MetricBatch",
    "ContributionSummaries",
    "contribution_summaries",
    "global_outer_coefficient",
    "reduce_sufficient_statistics",
    "MatrixJsonlWriter",
    "AdapterDisabledReference",
    "FrozenCopyReference",
    "ParameterWitness",
    "ReferenceProvider",
    "build_reference_provider",
]
