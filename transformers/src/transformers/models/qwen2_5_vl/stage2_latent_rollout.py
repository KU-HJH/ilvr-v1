"""Structural types and parser for Stage 2 latent token blocks."""

from dataclasses import dataclass

import torch
import torch.utils.checkpoint


class LatentStructureError(ValueError):
    """Raised when a token scaffold does not contain valid latent blocks."""


@dataclass(frozen=True)
class LatentTokenIds:
    pad_token_id: int
    start_token_id: int
    end_token_id: int


_LATENT_CONFIG_FIELDS = {
    "pad_token_id": ("latent_pad_token_id", "latent_token_id"),
    "start_token_id": ("latent_start_token_id", "latent_start_id"),
    "end_token_id": ("latent_end_token_id", "latent_end_id"),
}


def validate_stage2_config(
    config,
    *,
    fallback_token_ids: LatentTokenIds | None = None,
    require_canonical_ids: bool = True,
    require_latent_size: bool = True,
) -> tuple[LatentTokenIds, int | None]:
    """Validate Stage 2 structural metadata without requiring a tokenizer."""
    resolved = {}
    for result_field, (canonical_field, legacy_field) in _LATENT_CONFIG_FIELDS.items():
        canonical_value = getattr(config, canonical_field, None)
        legacy_value = getattr(config, legacy_field, None)
        if require_canonical_ids and (
            not isinstance(canonical_value, int) or isinstance(canonical_value, bool)
        ):
            raise ValueError(f"{canonical_field} must be present as an integer")
        for field, value in (
            (canonical_field, canonical_value),
            (legacy_field, legacy_value),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError(f"{field} must be an integer")
        if canonical_value is not None and legacy_value is not None:
            if canonical_value != legacy_value:
                raise ValueError(
                    f"{canonical_field} and {legacy_field} disagree: "
                    f"{canonical_value} != {legacy_value}"
                )
        value = canonical_value if canonical_value is not None else legacy_value
        if value is None and fallback_token_ids is not None:
            value = getattr(fallback_token_ids, result_field)
        if value is None:
            raise ValueError(f"{canonical_field} must be present as an integer")
        resolved[result_field] = value

    token_ids = LatentTokenIds(**resolved)
    if len(set(resolved.values())) != 3:
        raise ValueError("latent pad/start/end token IDs must be distinct")

    latent_size = getattr(config, "latent_size", None)
    if require_latent_size and (
        not isinstance(latent_size, int)
        or isinstance(latent_size, bool)
        or latent_size <= 0
    ):
        raise ValueError("latent_size must be a positive integer")
    return token_ids, latent_size


@dataclass(frozen=True)
class LatentSpan:
    start_position: int
    latent_positions: tuple[int, ...]
    end_position: int


@dataclass(frozen=True)
class LatentRolloutDiagnostics:
    block_count: int
    latent_l2_norms: torch.Tensor
    adjacent_cosine_similarities: torch.Tensor


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


def normalize_stage2_ablation(value) -> tuple[tuple[int, int], ...]:
    """Return the hidden latent spans as ordered (start, end) pairs.

    Accepts one ``(start, end)`` pair or a tuple of such pairs. For every pair,
    queries at or after ``end`` cannot attend the block's latent slots
    ``start < key < end``; the markers stay visible.
    """
    if value is None:
        return ()
    if not isinstance(value, tuple):
        raise ValueError("stage2_ablation must be an integer (start, end) tuple or a tuple of them")
    pairs = (value,) if len(value) == 2 and all(isinstance(item, int) for item in value) else value
    if not pairs:
        raise ValueError("stage2_ablation must contain at least one span")
    for pair in pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(not isinstance(position, int) or isinstance(position, bool) for position in pair)
            or not 0 <= pair[0] < pair[1]
        ):
            raise ValueError("stage2_ablation must be an integer (start, end) tuple or a tuple of them")
    ordered = tuple(sorted(pairs))
    if len(set(ordered)) != len(ordered) or any(
        later[0] <= earlier[1] for earlier, later in zip(ordered, ordered[1:])
    ):
        raise ValueError("stage2_ablation spans must be distinct and non-overlapping")
    return ordered


@dataclass(frozen=True)
class LatentRolloutResult:
    final_inputs_embeds: torch.Tensor
    latent_states: tuple[torch.Tensor, ...]
    prefix_lengths: tuple[int, ...]
    diagnostics: LatentRolloutDiagnostics | None = None


def _replace_embedding(embeds: torch.Tensor, position: int, value: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (embeds[:, :position], value.unsqueeze(1), embeds[:, position + 1 :]),
        dim=1,
    )


def _build_diagnostics(
    states: list[torch.Tensor],
    block_count: int,
    *,
    batch_size: int,
    device: torch.device,
) -> LatentRolloutDiagnostics:
    if not states:
        empty = torch.empty((batch_size, 0), dtype=torch.float32, device=device)
        return LatentRolloutDiagnostics(block_count, empty, empty.clone())

    stacked = torch.stack(states, dim=1)
    norms = torch.linalg.vector_norm(stacked.float(), dim=-1).detach()
    adjacent = torch.nn.functional.cosine_similarity(
        stacked[:, :-1].float(), stacked[:, 1:].float(), dim=-1
    ).detach()
    return LatentRolloutDiagnostics(block_count, norms, adjacent)


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
        stage2_ablation: tuple | None = None,
        retain_latent_grads: bool = False,
        collect_diagnostics: bool = False,
        latent_activation_checkpointing: bool = False,
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
        spans_to_hide = normalize_stage2_ablation(stage2_ablation)
        for start, end in spans_to_hide:
            matching_spans = tuple(
                span
                for span in latent_spans
                if (span.start_position, span.end_position) == (start, end)
            )
            if len(matching_spans) != 1 or not matching_spans[0].latent_positions:
                raise ValueError("stage2_ablation must match one parsed nonempty latent span")
            if end >= sequence_shape[1]:
                raise ValueError("stage2_ablation end marker must be inside the sequence")
            if not torch.all(attention_mask[0, start : end + 1] != 0):
                raise ValueError("stage2_ablation latent span must not be padding")
        first_hidden_end = min((end for _, end in spans_to_hide), default=None)

        current = base_inputs_embeds
        states = []
        prefix_lengths = []
        for span in latent_spans:
            for position in span.latent_positions:
                # This call contains queries [0, position). Before END enters
                # that prefix, ablation has no edges to block. Keep the normal
                # decoder path (including SDPA dispatch) until it is needed.
                intervention_kwargs = (
                    {"stage2_ablation": stage2_ablation}
                    if first_hidden_end is not None and position > first_hidden_end
                    else {}
                )

                def predict_last_hidden(
                    bound_inputs, bound_mask, bound_positions, kwargs=intervention_kwargs
                ):
                    outputs = language_model(
                        inputs_embeds=bound_inputs,
                        attention_mask=bound_mask,
                        position_ids=bound_positions,
                        use_cache=False,
                        return_dict=True,
                        **kwargs,
                    )
                    return outputs.last_hidden_state[:, -1, :]

                prefix = (
                    current[:, :position],
                    attention_mask[:, :position],
                    full_position_ids[..., :position],
                )
                if latent_activation_checkpointing and torch.is_grad_enabled():
                    # Keep only this prefix call's inputs; replay it in backward
                    # with the saved RNG state so dropout masks are identical.
                    with torch.utils.checkpoint.set_checkpoint_early_stop(False):
                        latent = torch.utils.checkpoint.checkpoint(
                            predict_last_hidden,
                            *prefix,
                            use_reentrant=False,
                            preserve_rng_state=True,
                        )
                else:
                    latent = predict_last_hidden(*prefix)
                if retain_latent_grads:
                    latent.retain_grad()
                states.append(latent)
                prefix_lengths.append(position)
                current = _replace_embedding(current, position, latent)
        diagnostics = None
        if collect_diagnostics:
            diagnostics = _build_diagnostics(
                states,
                len(latent_spans),
                batch_size=base_inputs_embeds.shape[0],
                device=base_inputs_embeds.device,
            )
        return LatentRolloutResult(current, tuple(states), tuple(prefix_lengths), diagnostics)
