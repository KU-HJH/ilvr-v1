"""Configuration and runtime helpers for evidence-window training."""

from .config import EvidenceWindowConfig
from .layout import InterleavedLayout, TokenIds, extract_interleaved_layout

__all__ = [
    "EvidenceWindowConfig",
    "InterleavedLayout",
    "TokenIds",
    "extract_interleaved_layout",
]
