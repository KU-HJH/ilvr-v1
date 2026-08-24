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
