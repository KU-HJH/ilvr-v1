"""Exact primitives for the opt-in Stage 2 entropy/vision objective."""

from .config import ObjectiveConfig, config_from_args, validate_objective_config, validate_objective_runtime
from .losses import DynTerms, VisTerms, entropy_transition, target_logps, text_entropies, vision_gain
from .vocabulary import TextVocabulary, build_text_vocabulary

__all__ = [
    "DynTerms", "ObjectiveConfig", "TextVocabulary", "VisTerms",
    "build_text_vocabulary", "config_from_args", "entropy_transition", "target_logps",
    "text_entropies", "validate_objective_config", "validate_objective_runtime", "vision_gain",
]
