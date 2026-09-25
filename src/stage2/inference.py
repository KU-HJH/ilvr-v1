"""Compatibility exports for the Stage 2 continuous-latent controller."""

import torch

from evaluation.continuous_latent import (
    ContinuousLatentController,
    ContinuousLatentGenerationConfig,
    ContinuousLatentGenerationResult,
    build_continuous_generation_config,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


Stage2GenerationConfig = ContinuousLatentGenerationConfig
Stage2GenerationResult = ContinuousLatentGenerationResult
build_stage2_generation_config = build_continuous_generation_config


class Stage2InferenceController(ContinuousLatentController):
    """Stage 2 compatibility wrapper around neutral latent generation."""

    @torch.no_grad()
    def generate(
        self,
        *,
        model,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        token_ids: LatentTokenIds,
        generation_config: Stage2GenerationConfig,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        return_latent_states: bool = False,
        disable_latent_generation: bool = False,
        intervention=None,
    ) -> Stage2GenerationResult:
        return super().generate(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_ids=token_ids,
            generation_config=generation_config,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_latent_states=return_latent_states,
            disable_latent_generation=disable_latent_generation,
            intervention=intervention,
        )
