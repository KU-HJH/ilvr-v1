"""Stage-neutral continuous latent generation with opt-in Stage 1 OFF caching."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Protocol, runtime_checkable

import torch
from transformers import RepetitionPenaltyLogitsProcessor

from evaluation.static_inference import (
    IncrementalMRoPE,
    static_inference_cache as static_cache_scope,
)

from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentSpan,
    LatentTokenIds,
)


@runtime_checkable
class ContinuousLatentModelProtocol(Protocol):
    """The model operations required by continuous-latent generation."""

    config: object
    model: object
    lm_head: object
    stage2_rollout_engine: object

    def prepare_stage2_base_inputs_embeds(self, *, input_ids, pixel_values, image_grid_thw): ...

    def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None): ...

    def get_input_embeddings(self): ...


class GenerationLimitError(ValueError):
    """Raised before generation would exceed its configured sequence limit."""


@dataclass(frozen=True)
class ContinuousLatentGenerationConfig:
    """Sampling and sequence limits for continuous-latent generation."""

    max_new_tokens: int = 4096
    max_sequence_length: int | None = None
    repetition_penalty: float = 1.05
    temperature: float = 0.0
    top_p: float = 1.0
    eos_token_id: int | list[int] | tuple[int, ...] | None = None
    generation_limit_policy: str = "bounded"

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.max_sequence_length is not None and (
            isinstance(self.max_sequence_length, bool)
            or not isinstance(self.max_sequence_length, int)
            or self.max_sequence_length <= 0
        ):
            raise ValueError("max_sequence_length must be a positive integer")
        if (
            isinstance(self.repetition_penalty, bool)
            or not isinstance(self.repetition_penalty, (int, float))
            or not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty <= 0
        ):
            raise ValueError("repetition_penalty must be finite and positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if self.generation_limit_policy not in (
            "bounded",
            "external_full_latent_block",
        ):
            raise ValueError(
                "generation_limit_policy must be bounded or "
                "external_full_latent_block"
            )
        if self.eos_token_id is None:
            return
        if isinstance(self.eos_token_id, bool):
            raise ValueError("eos_token_id must be a non-negative integer or list")
        if isinstance(self.eos_token_id, int):
            if self.eos_token_id < 0:
                raise ValueError("eos_token_id must be non-negative")
            return
        if not isinstance(self.eos_token_id, (list, tuple)) or not self.eos_token_id:
            raise ValueError("eos_token_id must be a non-empty integer list")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in self.eos_token_id
        ):
            raise ValueError("eos_token_id entries must be non-negative integers")
        object.__setattr__(self, "eos_token_id", tuple(self.eos_token_id))

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        if self.eos_token_id is None:
            return ()
        if isinstance(self.eos_token_id, int):
            return (self.eos_token_id,)
        return self.eos_token_id


def build_continuous_generation_config(
    *, model_config, model_generation_config, tokenizer, generation_kwargs
) -> ContinuousLatentGenerationConfig:
    """Translate evaluation generation settings into the neutral contract."""
    do_sample = bool(
        generation_kwargs.get(
            "do_sample", generation_kwargs.get("temperature", 0) > 0
        )
    )
    if "eos_token_id" in generation_kwargs:
        eos_token_id = generation_kwargs["eos_token_id"]
    else:
        eos_token_id = getattr(model_generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
    max_sequence_length = generation_kwargs.get("max_sequence_length")
    if max_sequence_length is None:
        max_sequence_length = getattr(model_config, "max_position_embeddings", None)
    return ContinuousLatentGenerationConfig(
        max_new_tokens=int(generation_kwargs.get("max_new_tokens", 4096)),
        max_sequence_length=max_sequence_length,
        repetition_penalty=float(
            generation_kwargs.get(
                "repetition_penalty",
                getattr(model_generation_config, "repetition_penalty", 1.05),
            )
        ),
        temperature=(
            float(generation_kwargs.get("temperature", 0.0)) if do_sample else 0.0
        ),
        top_p=float(generation_kwargs.get("top_p", 1.0)),
        eos_token_id=eos_token_id,
        generation_limit_policy=generation_kwargs.get(
            "generation_limit_policy", "bounded"
        ),
    )


@dataclass(frozen=True)
class ContinuousLatentGenerationResult:
    """Generated scaffold, continuous states, and generation evidence."""

    sequences: torch.LongTensor
    latent_states_by_block: tuple[tuple[torch.Tensor, ...], ...]
    generated_suffix_ids: tuple[int, ...] = ()
    latent_start_count: int = 0
    latent_block_count: int = 0
    stop_reason: str = "max_new_tokens"
    post_end_logits_by_block: tuple[torch.Tensor, ...] = ()


class ContinuousLatentController:
    """Generate continuous latent blocks, or optionally cache Stage 1 text only."""

    @torch.no_grad()
    def generate(
        self,
        *,
        model: ContinuousLatentModelProtocol,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        token_ids: LatentTokenIds,
        generation_config: ContinuousLatentGenerationConfig,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        return_latent_states: bool = False,
        return_post_end_logits: bool = False,
        disable_latent_generation: bool = False,
        use_kv_cache: bool = False,
        static_inference_cache: bool = False,
    ) -> ContinuousLatentGenerationResult:
        self._validate_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        if type(disable_latent_generation) is not bool:
            raise TypeError("disable_latent_generation must be a boolean")
        if type(use_kv_cache) is not bool:
            raise TypeError("use_kv_cache must be a boolean")
        if use_kv_cache and (
            getattr(model.config, "stage", None) != "stage1"
            or not disable_latent_generation
        ):
            raise ValueError("KV cache requires Stage 1 with latent generation disabled")
        if type(static_inference_cache) is not bool:
            raise TypeError("static_inference_cache must be a boolean")
        if static_inference_cache and not use_kv_cache:
            raise ValueError("static inference cache requires Stage 1 OFF with KV cache")
        with static_cache_scope(model) if static_inference_cache else nullcontext():
            return self._generate(
                model=model, input_ids=input_ids, attention_mask=attention_mask,
                token_ids=token_ids, generation_config=generation_config,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                return_latent_states=return_latent_states,
                return_post_end_logits=return_post_end_logits,
                disable_latent_generation=disable_latent_generation,
                use_kv_cache=use_kv_cache, static_inference_cache=static_inference_cache,
            )

    def _generate(
        self, *, model, input_ids, attention_mask, token_ids, generation_config,
        pixel_values, image_grid_thw, return_latent_states, return_post_end_logits,
        disable_latent_generation, use_kv_cache, static_inference_cache,
    ):
        sequences = input_ids
        mask = attention_mask
        inputs_embeds = model.prepare_stage2_base_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        rope = None
        latent_blocks: list[tuple[torch.Tensor, ...]] = []
        post_end_logits: list[torch.Tensor] = []
        latent_start_count = 0
        generated = 0
        stop_reason = "max_new_tokens"
        past_key_values = None
        cached_length = 0
        repetition_processor = RepetitionPenaltyLogitsProcessor(
            generation_config.repetition_penalty
        )

        while True:
            external_full_block = (
                generation_config.generation_limit_policy
                == "external_full_latent_block"
            )
            if external_full_block:
                remaining_capacity = generation_config.max_new_tokens - generated
                boundary_stop_reason = "max_new_tokens"
            else:
                remaining_capacity, boundary_stop_reason = _remaining_capacity(
                    sequence_length=sequences.shape[1],
                    generated=generated,
                    config=generation_config,
                )
            if remaining_capacity <= 0:
                stop_reason = boundary_stop_reason
                break
            if static_inference_cache:
                if rope is None:
                    rope = IncrementalMRoPE(model, input_ids, attention_mask, image_grid_thw)
                position_ids = rope.get(sequences, mask, cached_length)
            else:
                position_ids, _ = model.get_rope_index(
                    sequences, image_grid_thw=image_grid_thw, attention_mask=mask
                )
            cache_kwargs = {}
            if use_kv_cache:
                # Physical cache offsets include prompt padding; MRoPE positions
                # retain the original image/text layout instead of using offsets.
                if rope is None:
                    position_ids = position_ids[..., cached_length:]
                cache_kwargs = {
                    "past_key_values": past_key_values,
                    "cache_position": torch.arange(
                        cached_length, sequences.shape[1], device=sequences.device
                    ),
                }
            outputs = model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=use_kv_cache,
                return_dict=True,
                generate_mode=True,
                **cache_kwargs,
            )
            if use_kv_cache:
                past_key_values = getattr(outputs, "past_key_values", None)
                if past_key_values is None:
                    raise RuntimeError("requested KV cache was not returned by the decoder")
                cached_length = sequences.shape[1]
            logits = model.lm_head(outputs.last_hidden_state[:, -1, :]).clone()
            logits = repetition_processor(sequences, logits)
            if disable_latent_generation:
                logits[..., token_ids.start_token_id] = -torch.inf
            logits[..., token_ids.pad_token_id] = -torch.inf
            logits[..., token_ids.end_token_id] = -torch.inf
            next_id = _select_text_token(logits, generation_config)
            next_token_id = int(next_id.item())
            if rope is not None:
                rope.appended(next_token_id)
            if disable_latent_generation and next_token_id in {
                token_ids.start_token_id,
                token_ids.pad_token_id,
                token_ids.end_token_id,
            }:
                raise RuntimeError("disabled latent generation selected a control token")

            if (
                external_full_block
                and generation_config.max_sequence_length is not None
            ):
                completed_length = sequences.shape[1] + 1
                if next_token_id == token_ids.start_token_id:
                    completed_length += model.config.latent_size + 1
                if completed_length > generation_config.max_sequence_length:
                    description = (
                        "latent block"
                        if next_token_id == token_ids.start_token_id
                        else "generated token"
                    )
                    raise GenerationLimitError(
                        f"{description} would exceed max_sequence_length"
                    )

            next_ids = next_id[:, None]
            sequences = torch.cat((sequences, next_ids), dim=1)
            mask = torch.cat((mask, torch.ones_like(next_ids)), dim=1)
            next_embeds = model.get_input_embeddings()(next_id).unsqueeze(1)
            inputs_embeds = (
                next_embeds if use_kv_cache
                else torch.cat((inputs_embeds, next_embeds), dim=1)
            )
            generated += 1

            if next_token_id == token_ids.start_token_id:
                latent_start_count += 1
                block_start = sequences.shape[1] - 1
                suffix = torch.tensor(
                    [[token_ids.pad_token_id] * model.config.latent_size + [token_ids.end_token_id]],
                    dtype=sequences.dtype,
                    device=sequences.device,
                )
                suffix_capacity = (
                    suffix.shape[1] if external_full_block else remaining_capacity - 1
                )
                suffix = suffix[:, :suffix_capacity]
                if suffix.shape[1] > 0:
                    sequences = torch.cat((sequences, suffix), dim=1)
                    mask = torch.cat((mask, torch.ones_like(suffix)), dim=1)
                    inputs_embeds = torch.cat(
                        (inputs_embeds, model.get_input_embeddings()(suffix)), dim=1
                    )
                    generated += suffix.shape[1]
                if suffix.shape[1] < model.config.latent_size + 1:
                    stop_reason = boundary_stop_reason
                    break
                full_position_ids, _ = model.get_rope_index(
                    sequences, image_grid_thw=image_grid_thw, attention_mask=mask
                )
                span = LatentSpan(
                    start_position=block_start,
                    latent_positions=tuple(
                        range(block_start + 1, block_start + 1 + model.config.latent_size)
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
                if return_post_end_logits:
                    post_end_outputs = model.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=mask,
                        position_ids=full_position_ids,
                        use_cache=False,
                        return_dict=True,
                        generate_mode=True,
                    )
                    post_end_logits.append(
                        model.lm_head(post_end_outputs.last_hidden_state[:, -1, :])
                    )
                latent_blocks.append(rollout.latent_states)
                if generated == generation_config.max_new_tokens or (
                    generation_config.max_sequence_length is not None
                    and sequences.shape[1] == generation_config.max_sequence_length
                ):
                    stop_reason = boundary_stop_reason
                    break
                continue

            if next_token_id in generation_config.eos_token_ids:
                stop_reason = "eos"
                break
            if remaining_capacity == 1:
                stop_reason = boundary_stop_reason
                break

        result = ContinuousLatentGenerationResult(
            sequences=sequences,
            latent_states_by_block=tuple(latent_blocks) if return_latent_states else (),
            generated_suffix_ids=tuple(sequences[0, input_ids.shape[1] :].tolist()),
            latent_start_count=latent_start_count,
            latent_block_count=len(latent_blocks),
            stop_reason=stop_reason,
            post_end_logits_by_block=tuple(post_end_logits) if return_post_end_logits else (),
        )
        if disable_latent_generation and (
            result.latent_start_count
            or result.latent_block_count
            or set(result.generated_suffix_ids)
            & {token_ids.start_token_id, token_ids.pad_token_id, token_ids.end_token_id}
        ):
            raise RuntimeError("disabled latent generation produced a latent control")
        return result

    @staticmethod
    def _validate_inputs(*, model, input_ids, attention_mask, generation_config) -> None:
        required = (
            "config",
            "prepare_stage2_base_inputs_embeds",
            "get_rope_index",
            "get_input_embeddings",
            "model",
            "lm_head",
            "stage2_rollout_engine",
        )
        if any(not hasattr(model, attribute) for attribute in required):
            raise TypeError("model does not satisfy the continuous latent interface")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise NotImplementedError("continuous latent inference currently supports batch_size=1 only")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if model.config.latent_size <= 0:
            raise ValueError("latent_size must be positive")
        if (
            generation_config.max_sequence_length is not None
            and input_ids.shape[1] > generation_config.max_sequence_length
        ):
            raise GenerationLimitError("input_ids exceed max_sequence_length")


def _select_text_token(
    logits: torch.Tensor, config: ContinuousLatentGenerationConfig
) -> torch.LongTensor:
    if config.temperature == 0:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits / config.temperature, dim=-1)
    if config.top_p < 1:
        sorted_probabilities, sorted_indices = probabilities.sort(descending=True)
        remove = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities > config.top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        probabilities = torch.zeros_like(probabilities).scatter(
            -1, sorted_indices, sorted_probabilities
        )
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


def _remaining_capacity(
    *,
    sequence_length: int,
    generated: int,
    config: ContinuousLatentGenerationConfig,
) -> tuple[int, str]:
    """Return the strict token budget and the cap that will end generation."""
    remaining_new_tokens = config.max_new_tokens - generated
    if config.max_sequence_length is None:
        return remaining_new_tokens, "max_new_tokens"
    remaining_sequence_tokens = config.max_sequence_length - sequence_length
    if remaining_new_tokens <= remaining_sequence_tokens:
        return remaining_new_tokens, "max_new_tokens"
    return remaining_sequence_tokens, "max_sequence_length"
