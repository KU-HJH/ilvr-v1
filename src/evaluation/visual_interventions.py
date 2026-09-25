"""Generation-local image masks for latent-only image access tests.

Ported subset of ours1-visual-causal src/evaluation/visual_interventions.py
(E3 latent mask and its same-format causal control only).
"""
from __future__ import annotations

from functools import wraps

import torch

_MODES = ("causal_mask", "latent_mask")


def _mode(mode):
    if mode not in _MODES:
        raise ValueError(f"unknown visual intervention mode {mode!r}; expected one of {', '.join(_MODES)}")
    return mode


def identify_image_positions(input_ids, image_token_id):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError("visual interventions currently support batch_size=1 only")
    return tuple((input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten().tolist())


def build_image_attention_mask(validity_mask, image_positions, latent_query_positions, mode, dtype):
    """Finite additive causal mask; latent_mask removes image keys from latent query rows."""
    mode = _mode(mode)
    if validity_mask.ndim != 2 or validity_mask.shape[0] != 1:
        raise NotImplementedError("visual attention masks currently support batch_size=1 only")
    if not dtype.is_floating_point:
        raise TypeError("attention mask dtype must be floating point")
    length, device = validity_mask.shape[1], validity_mask.device
    query = torch.arange(length, device=device)[:, None]
    key = torch.arange(length, device=device)[None, :]
    allowed = (key <= query) & validity_mask[0].to(torch.bool)[None, :]
    images = tuple(p for p in image_positions if 0 <= p < length)
    queries = tuple(p for p in latent_query_positions if 0 <= p < length)
    if mode == "latent_mask" and images and queries:
        allowed[torch.tensor(queries, device=device)[:, None], torch.tensor(images, device=device)[None, :]] = False
    additive = torch.full((length, length), torch.finfo(dtype).min, dtype=dtype, device=device)
    additive.masked_fill_(allowed, 0)
    return additive[None, None]


class VisualIntervention:
    """Apply one image mask mode to every no-cache decoder call of one generation."""

    def __init__(self, mode, image_token_id):
        self.mode = _mode(mode)
        self.image_token_id = image_token_id
        self.forward_records = []
        self._image_positions = ()
        self._latent_query_positions = []

    def begin(self, input_ids, token_ids):
        self.forward_records = []
        self._latent_query_positions = []
        self._image_positions = identify_image_positions(input_ids, self.image_token_id)

    def register_latent_span(self, span):
        for position in (span.start_position, *span.latent_positions):
            if position not in self._latent_query_positions:
                self._latent_query_positions.append(position)

    def wrap_decoder(self, language_model):
        @wraps(language_model)
        def wrapped(**kwargs):
            if kwargs.get("past_key_values") is not None or kwargs.get("use_cache") is not False:
                raise ValueError("visual intervention decoder calls require use_cache=False and no cache")
            embeds, mask = kwargs.get("inputs_embeds"), kwargs.get("attention_mask")
            if embeds is None or embeds.ndim != 3 or mask is None or mask.ndim != 2 or mask.shape != embeds.shape[:2]:
                raise ValueError("expected [batch, sequence, hidden] embeddings with a matching 2-D validity mask")
            prefix = embeds.shape[1]
            self.forward_records.append({
                "prefix_length": prefix,
                "role": "latent_rollout" if prefix - 1 in self._latent_query_positions else "text",
                "latent_query_positions": tuple(p for p in self._latent_query_positions if p < prefix),
            })
            kwargs["attention_mask"] = build_image_attention_mask(
                mask, self._image_positions, tuple(self._latent_query_positions), self.mode, embeds.dtype)
            return language_model(**kwargs)
        return wrapped
