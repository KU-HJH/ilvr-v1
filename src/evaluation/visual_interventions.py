"""Generation-local controls for visual causal evaluation."""

from __future__ import annotations

from functools import wraps

import torch


_MODE_ALIASES = {
    "baseline": "noop",
    "image_replacement": "noop",
    "no_input_image": "noop",
    "no_latent_execution": "suppress_start",
    "latent_image_attention_block": "latent_mask",
    "global_image_attention_block": "global_mask",
}


def _canonical_mode(mode: str) -> str:
    try:
        return _MODE_ALIASES[mode]
    except (KeyError, TypeError) as error:
        choices = ", ".join(_MODE_ALIASES)
        raise ValueError(
            f"unknown visual intervention mode {mode!r}; expected one of {choices}"
        ) from error


def identify_image_positions(input_ids: torch.Tensor, image_token_id: int) -> tuple[int, ...]:
    """Return physical image-token indices from a single initial prompt."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError("visual interventions currently support batch_size=1 only")
    return tuple((input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten().tolist())


def build_image_attention_mask(
    validity_mask: torch.Tensor,
    image_positions: tuple[int, ...],
    latent_query_positions: tuple[int, ...],
    mode: str,
    dtype: torch.dtype,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Build a finite additive causal mask with optional visual edge removals."""
    canonical_mode = _canonical_mode(mode)
    if validity_mask.ndim != 2 or validity_mask.shape[0] != 1:
        raise NotImplementedError("visual attention masks currently support batch_size=1 only")
    if not dtype.is_floating_point:
        raise TypeError("attention mask dtype must be floating point")
    if sliding_window is not None and (
        isinstance(sliding_window, bool)
        or not isinstance(sliding_window, int)
        or sliding_window <= 0
    ):
        raise ValueError("sliding_window must be a positive integer")

    length = validity_mask.shape[1]
    device = validity_mask.device
    query = torch.arange(length, device=device)[:, None]
    key = torch.arange(length, device=device)[None, :]
    allowed = (key <= query) & validity_mask[0].to(torch.bool)[None, :]
    if sliding_window is not None:
        allowed = allowed & (key > query - sliding_window)

    valid_images = tuple(position for position in image_positions if 0 <= position < length)
    if canonical_mode == "latent_mask" and valid_images:
        valid_queries = tuple(
            position for position in latent_query_positions if 0 <= position < length
        )
        if valid_queries:
            allowed[
                torch.tensor(valid_queries, device=device)[:, None],
                torch.tensor(valid_images, device=device)[None, :],
            ] = False
    elif canonical_mode == "global_mask" and valid_images:
        image_set = set(valid_images)
        nonimage_queries = tuple(position for position in range(length) if position not in image_set)
        allowed[
            torch.tensor(nonimage_queries, device=device)[:, None],
            torch.tensor(valid_images, device=device)[None, :],
        ] = False

    minimum = torch.finfo(dtype).min
    additive = torch.full((length, length), minimum, dtype=dtype, device=device)
    additive.masked_fill_(allowed, 0)
    return additive[None, None]


class VisualIntervention:
    """Apply one visual intervention without retaining state across generations."""

    def __init__(
        self, mode: str, image_token_id: int, sliding_window: int | None = None
    ) -> None:
        self.mode = mode
        self._effect = _canonical_mode(mode)
        self.image_token_id = image_token_id
        self.sliding_window = sliding_window
        self.events: list[dict[str, object]] = []
        self.forward_records: list[dict[str, object]] = []
        self._image_positions: tuple[int, ...] = ()
        self._latent_query_positions: list[int] = []
        self._initial_length: int | None = None

    def begin(self, input_ids: torch.Tensor, token_ids) -> None:
        """Start a generation and discard all evidence from the prior one."""
        self.events = []
        self.forward_records = []
        self._latent_query_positions = []
        self._image_positions = identify_image_positions(input_ids, self.image_token_id)
        self._initial_length = input_ids.shape[1]

    def register_latent_span(self, span) -> None:
        """Register original START and PAD rows that produce a latent block."""
        positions = (span.start_position, *span.latent_positions)
        self._latent_query_positions.extend(
            position for position in positions if position not in self._latent_query_positions
        )

    def wrap_decoder(self, language_model):
        """Return a no-cache decoder callable that composes E3/E4 masks locally."""

        @wraps(language_model)
        def wrapped(**kwargs):
            if kwargs.get("past_key_values") is not None:
                raise ValueError("past_key_values are incompatible with visual interventions")
            if kwargs.get("use_cache") is not False:
                raise ValueError("visual intervention decoder calls require use_cache=False")
            inputs_embeds = kwargs.get("inputs_embeds")
            attention_mask = kwargs.get("attention_mask")
            if inputs_embeds is None or inputs_embeds.ndim != 3:
                raise ValueError("inputs_embeds must have shape [batch, sequence, hidden]")
            if attention_mask is None or attention_mask.ndim != 2:
                raise ValueError("attention_mask must be a 2-D validity mask before intervention")
            prefix_length = inputs_embeds.shape[1]
            if attention_mask.shape != inputs_embeds.shape[:2]:
                raise ValueError("attention_mask must match the received embedding prefix")

            self.forward_records.append(
                {
                    "prefix_length": prefix_length,
                    "inputs_embeds_shape": tuple(inputs_embeds.shape),
                    "attention_mask_shape": tuple(attention_mask.shape),
                    "position_ids_shape": (
                        tuple(kwargs["position_ids"].shape)
                        if kwargs.get("position_ids") is not None
                        else None
                    ),
                    "role": self._forward_role(prefix_length),
                    "latent_query_positions": tuple(
                        position
                        for position in self._latent_query_positions
                        if position < prefix_length
                    ),
                    "image_positions": tuple(
                        position for position in self._image_positions if position < prefix_length
                    ),
                }
            )
            if self._effect in {"latent_mask", "global_mask"}:
                kwargs["attention_mask"] = build_image_attention_mask(
                    attention_mask,
                    self._image_positions,
                    tuple(self._latent_query_positions),
                    self.mode,
                    inputs_embeds.dtype,
                    self.sliding_window,
                )
            return language_model(**kwargs)

        return wrapped

    def process_logits(
        self,
        sequences: torch.Tensor,
        raw_logits: torch.Tensor,
        decision_logits: torch.Tensor,
        token_ids,
    ) -> torch.Tensor:
        """Record START diagnostics and apply E2 only to the START decision logit."""
        if self._initial_length is None:
            raise RuntimeError("begin must be called before processing logits")
        if raw_logits.ndim != 2 or raw_logits.shape[0] != 1:
            raise ValueError("raw_logits must have shape [1, vocabulary]")
        if decision_logits.shape != raw_logits.shape:
            raise ValueError("decision_logits must match raw_logits")

        adjusted = (
            decision_logits
            if self._effect != "suppress_start"
            else decision_logits.clone()
        )
        start_id = token_ids.start_token_id
        before = int(decision_logits.argmax(dim=-1).item())
        probabilities = torch.softmax(decision_logits.float(), dim=-1)
        start_score = decision_logits[0, start_id]
        rank = int((decision_logits[0] > start_score).sum().item()) + 1
        excluded = {
            token_ids.start_token_id,
            token_ids.pad_token_id,
            token_ids.end_token_id,
        }
        noncontrol = decision_logits[0].clone()
        for token_id in excluded:
            noncontrol[token_id] = -torch.inf
        best_noncontrol_id = int(noncontrol.argmax().item())
        if self._effect == "suppress_start":
            adjusted[..., start_id] = -torch.inf
        after = int(adjusted.argmax(dim=-1).item())
        self.events.append(
            {
                "decision_position": sequences.shape[1] - self._initial_length,
                "original_start_logit": float(raw_logits[0, start_id].item()),
                "repetition_adjusted_start_logit": float(start_score.item()),
                "start_probability": float(probabilities[0, start_id].item()),
                "start_rank": rank,
                "best_non_control_token_id": best_noncontrol_id,
                "best_non_control_logit": float(noncontrol[best_noncontrol_id].item()),
                "pre_argmax_token_id": before,
                "post_argmax_token_id": after,
                "changed": before != after,
            }
        )
        return adjusted

    def _forward_role(self, prefix_length: int) -> str:
        producer = prefix_length - 1
        if producer in self._latent_query_positions:
            return "latent_rollout"
        if self._latent_query_positions:
            return "text_or_post_end"
        return "text"
