"""Structural types and parser for Stage 2 latent token blocks."""

from dataclasses import dataclass

import torch


class LatentStructureError(ValueError):
    """Raised when a token scaffold does not contain valid latent blocks."""


@dataclass(frozen=True)
class LatentTokenIds:
    pad_token_id: int
    start_token_id: int
    end_token_id: int


@dataclass(frozen=True)
class LatentSpan:
    start_position: int
    latent_positions: tuple[int, ...]
    end_position: int


def parse_latent_spans(
    input_ids: torch.LongTensor,
    *,
    token_ids: LatentTokenIds,
    latent_size: int,
) -> tuple[LatentSpan, ...]:
    """Parse all ordered latent blocks from one-dimensional token IDs."""
    if input_ids.ndim != 1:
        raise LatentStructureError("parse_latent_spans expects a 1D token sequence")
    if latent_size <= 0:
        raise LatentStructureError("latent_size must be positive")

    spans: list[LatentSpan] = []
    open_start: int | None = None
    latent_positions: list[int] = []
    for position, raw_token_id in enumerate(input_ids.tolist()):
        token_id = int(raw_token_id)
        if open_start is None:
            if token_id == token_ids.start_token_id:
                open_start = position
                latent_positions = []
            elif token_id == token_ids.end_token_id:
                raise LatentStructureError("latent_end without latent_start")
            elif token_id == token_ids.pad_token_id:
                raise LatentStructureError("latent_pad outside latent block")
            continue

        if token_id == token_ids.start_token_id:
            raise LatentStructureError("nested latent_start")
        if token_id == token_ids.end_token_id:
            if len(latent_positions) != latent_size:
                raise LatentStructureError(f"expected {latent_size} latent_pad tokens")
            spans.append(LatentSpan(open_start, tuple(latent_positions), position))
            open_start = None
            latent_positions = []
            continue
        if token_id != token_ids.pad_token_id:
            raise LatentStructureError("non-latent_pad token inside latent block")
        latent_positions.append(position)
        if len(latent_positions) > latent_size:
            raise LatentStructureError(f"expected {latent_size} latent_pad tokens")

    if open_start is not None:
        raise LatentStructureError("missing latent_end")
    return tuple(spans)


@dataclass(frozen=True)
class LatentRolloutResult:
    final_inputs_embeds: torch.Tensor
    latent_states: tuple[torch.Tensor, ...]
    prefix_lengths: tuple[int, ...]
    diagnostics: "LatentRolloutDiagnostics | None" = None


def _replace_embedding(embeds: torch.Tensor, position: int, value: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (embeds[:, :position], value.unsqueeze(1), embeds[:, position + 1 :]),
        dim=1,
    )


class LatentRolloutEngine:
    """Run the differentiable, no-cache continuous latent recurrence."""

    def rollout(
        self,
        *,
        language_model,
        base_inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        full_position_ids: torch.Tensor,
        latent_spans: tuple[LatentSpan, ...],
        retain_latent_grads: bool = False,
        collect_diagnostics: bool = False,
    ) -> LatentRolloutResult:
        if base_inputs_embeds.ndim != 3:
            raise ValueError("base_inputs_embeds must have shape [batch, sequence, hidden]")
        if base_inputs_embeds.shape[0] != 1:
            raise NotImplementedError("Stage 2 rollout currently supports batch_size=1 only")
        sequence_shape = base_inputs_embeds.shape[:2]
        if attention_mask.shape != sequence_shape:
            raise ValueError("attention_mask must match the embedding batch and sequence dimensions")
        if full_position_ids.ndim != 3 or full_position_ids.shape[-2:] != sequence_shape:
            raise ValueError("full_position_ids must match the embedding batch and sequence dimensions")

        current = base_inputs_embeds
        states = []
        prefix_lengths = []
        for span in latent_spans:
            for position in span.latent_positions:
                outputs = language_model(
                    inputs_embeds=current[:, :position],
                    attention_mask=attention_mask[:, :position],
                    position_ids=full_position_ids[..., :position],
                    use_cache=False,
                    return_dict=True,
                )
                latent = outputs.last_hidden_state[:, -1, :]
                if retain_latent_grads:
                    latent.retain_grad()
                states.append(latent)
                prefix_lengths.append(position)
                current = _replace_embedding(current, position, latent)
        return LatentRolloutResult(current, tuple(states), tuple(prefix_lengths))
