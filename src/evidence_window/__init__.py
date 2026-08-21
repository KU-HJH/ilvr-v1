"""Configuration and runtime helpers for evidence-window training."""

from .config import EvidenceWindowConfig
from .layout import InterleavedLayout, TokenIds, extract_interleaved_layout
from .losses import (
    evidence_window_preference_loss,
    global_preference_loss,
    window_preference_loss,
)
from .scores import (
    causal_token_logps,
    evidence_window_margins,
    masked_token_sums,
    reference_adjusted_effect,
)

__all__ = [
    "EvidenceWindowConfig",
    "InterleavedLayout",
    "TokenIds",
    "extract_interleaved_layout",
    "causal_token_logps",
    "masked_token_sums",
    "reference_adjusted_effect",
    "evidence_window_margins",
    "global_preference_loss",
    "window_preference_loss",
    "evidence_window_preference_loss",
]
