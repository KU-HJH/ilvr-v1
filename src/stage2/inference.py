"""Model-specific, no-cache Stage 2 inference controller."""

from dataclasses import dataclass
from functools import partial

import torch

from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentSpan,
    LatentTokenIds,
)


@dataclass(frozen=True)
class Stage2GenerationConfig:
    """Sampling and sequence limits for Stage 2 generation."""

    max_new_tokens: int = 64
    max_sequence_length: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.max_sequence_length is not None and self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")


@dataclass(frozen=True)
class Stage2GenerationResult:
    """Generated token scaffold and optional continuous latent states."""

    sequences: torch.LongTensor
    latent_states_by_block: tuple[tuple[torch.Tensor, ...], ...]


class Stage2InferenceController:
    """Generate Stage 2 text and continuous latent blocks without a KV cache."""

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
    ) -> Stage2GenerationResult:
        self._validate_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

        sequences = input_ids
        mask = attention_mask
        inputs_embeds = model.prepare_stage2_base_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        latent_blocks: list[tuple[torch.Tensor, ...]] = []
        generated = 0

        while generated < generation_config.max_new_tokens:
            position_ids, _ = model.get_rope_index(
                sequences,
                image_grid_thw=image_grid_thw,
                attention_mask=mask,
            )
            outputs = model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
                generate_mode=True,
            )
            logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
            next_id = _select_text_token(logits, generation_config)
            next_token_id = int(next_id.item())

            if next_token_id == token_ids.start_token_id:
                completed_length = sequences.shape[1] + model.config.latent_size + 2
                if (
                    generation_config.max_sequence_length is not None
                    and completed_length > generation_config.max_sequence_length
                ):
                    raise ValueError("latent block would exceed max_sequence_length")
            elif (
                generation_config.max_sequence_length is not None
                and sequences.shape[1] + 1 > generation_config.max_sequence_length
            ):
                raise ValueError("generated token would exceed max_sequence_length")

            next_ids = next_id[:, None]
            sequences = torch.cat((sequences, next_ids), dim=1)
            mask = torch.cat((mask, torch.ones_like(next_ids)), dim=1)
            inputs_embeds = torch.cat(
                (inputs_embeds, model.get_input_embeddings()(next_id).unsqueeze(1)),
                dim=1,
            )
            generated += 1

            if next_token_id == token_ids.start_token_id:
                block_start = sequences.shape[1] - 1
                suffix = torch.tensor(
                    [
                        [token_ids.pad_token_id] * model.config.latent_size
                        + [token_ids.end_token_id]
                    ],
                    dtype=sequences.dtype,
                    device=sequences.device,
                )
                sequences = torch.cat((sequences, suffix), dim=1)
                mask = torch.cat((mask, torch.ones_like(suffix)), dim=1)
                inputs_embeds = torch.cat(
                    (inputs_embeds, model.get_input_embeddings()(suffix)),
                    dim=1,
                )
                full_position_ids, _ = model.get_rope_index(
                    sequences,
                    image_grid_thw=image_grid_thw,
                    attention_mask=mask,
                )
                span = LatentSpan(
                    start_position=block_start,
                    latent_positions=tuple(
                        range(
                            block_start + 1,
                            block_start + 1 + model.config.latent_size,
                        )
                    ),
                    end_position=block_start + model.config.latent_size + 1,
                )
                rollout = model.stage2_rollout_engine.rollout(
                    language_model=partial(model.model, generate_mode=True),
                    base_inputs_embeds=inputs_embeds,
                    attention_mask=mask,
                    full_position_ids=full_position_ids,
                    latent_spans=(span,),
                )
                inputs_embeds = rollout.final_inputs_embeds
                latent_blocks.append(rollout.latent_states)
                generated += model.config.latent_size + 1
                continue

            if (
                generation_config.eos_token_id is not None
                and next_token_id == generation_config.eos_token_id
            ):
                break

        return Stage2GenerationResult(
            sequences=sequences,
            latent_states_by_block=tuple(latent_blocks) if return_latent_states else (),
        )

    @staticmethod
    def _validate_inputs(*, model, input_ids, attention_mask, generation_config) -> None:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise NotImplementedError("Stage 2 inference currently supports batch_size=1 only")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if model.config.latent_size <= 0:
            raise ValueError("latent_size must be positive")
        if (
            generation_config.max_sequence_length is not None
            and input_ids.shape[1] > generation_config.max_sequence_length
        ):
            raise ValueError("input_ids exceed max_sequence_length")


def _select_text_token(
    logits: torch.Tensor,
    config: Stage2GenerationConfig,
) -> torch.LongTensor:
    if config.temperature == 0:
        return logits.argmax(dim=-1)

    probabilities = torch.softmax(logits / config.temperature, dim=-1)
    if config.top_p < 1:
        sorted_probabilities, sorted_indices = probabilities.sort(descending=True)
        remove = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities > config.top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        probabilities = torch.zeros_like(probabilities).scatter(
            -1,
            sorted_indices,
            sorted_probabilities,
        )
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)
