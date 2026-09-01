"""Compatibility exports for the Stage 2 continuous-latent controller."""

from evaluation.continuous_latent import (
    ContinuousLatentController,
    ContinuousLatentGenerationConfig,
    ContinuousLatentGenerationResult,
    build_continuous_generation_config,
)


Stage2GenerationConfig = ContinuousLatentGenerationConfig
Stage2GenerationResult = ContinuousLatentGenerationResult
build_stage2_generation_config = build_continuous_generation_config


class Stage2InferenceController(ContinuousLatentController):
    """Stage 2 compatibility wrapper around neutral latent generation."""
