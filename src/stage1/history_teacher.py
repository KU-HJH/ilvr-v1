"""Frozen effective-DoRA language prefixes with original-image MRoPE."""
from dataclasses import dataclass

import torch
from torch import Tensor

from peft_common import resolve_qwen2_5_vl_model


@dataclass(frozen=True)
class HistorySelection:
    block_index: int
    positions: tuple[int, ...]
    values: Tensor
    candidate_indices: tuple[int, ...]


@dataclass(frozen=True)
class PreparedHistoryPrefix:
    input_ids: Tensor
    inputs_embeds: Tensor
    attention_mask: Tensor
    position_ids: Tensor


@dataclass(frozen=True)
class HistoryPrefixObservation:
    last_hidden_state: Tensor


class HistoryPrefixAdapter:
    """Reuse the teacher's effective modules; helper images never enter this API.

    The metadata/tokenizer-aware caller validates the single assistant trajectory
    and original-image placement before the assistant. Here, coordinates are
    checked against the model's image and latent token layout without retokenizing.
    """

    @torch.no_grad()
    def __init__(self, teacher, *, input_ids: Tensor, attention_mask: Tensor,
                 original_pixels: Tensor, original_grid: Tensor):
        self._teacher = teacher
        self._require_frozen()
        self._qwen = resolve_qwen2_5_vl_model(teacher)
        self.backbone = self._qwen.model
        if (input_ids.ndim != 2 or input_ids.shape[0] != 1
                or input_ids.shape[1] == 0 or input_ids.dtype != torch.long):
            raise ValueError('history input_ids must have integer [1,S] shape')
        if (attention_mask.shape != input_ids.shape
                or not bool(((attention_mask == 0) | (attention_mask == 1)).all())):
            raise ValueError('history requires a binary 2D [1,S] padding mask')
        valid = attention_mask[0].nonzero().flatten().tolist()
        if not valid or valid != list(range(valid[0], valid[-1] + 1)):
            raise ValueError('history requires contiguous valid tokens, without mask holes')
        self._valid_start, self._valid_end = valid[0], valid[-1] + 1
        self._ids = input_ids.detach().clone()
        self._mask = attention_mask.detach().clone()
        self._grid = original_grid.detach().clone()
        self._spans = self._latent_spans()
        self._validate_images(original_pixels)
        self._embeddings = self.backbone.embed_tokens(self._ids).detach().clone()
        if self._grid.shape[0]:
            features = self._qwen.visual(
                original_pixels.to(dtype=self._qwen.visual.dtype), grid_thw=self._grid)
            image_mask = self._ids == self._qwen.config.image_token_id
            if (features.ndim != 2 or features.shape != (
                    int(image_mask.sum()), self._embeddings.shape[-1])):
                raise ValueError('original image token/feature count or hidden width mismatch')
            self._embeddings = self._embeddings.masked_scatter(
                image_mask.unsqueeze(-1), features.to(self._embeddings))

    def _require_frozen(self):
        if any(module.training for module in self._teacher.modules()) or any(
                p.requires_grad or p.grad is not None for p in self._teacher.parameters()):
            raise ValueError('history teacher must be frozen, gradient-free and in eval mode')

    def _latent_spans(self):
        config = self._qwen.config
        pad, start, end = (getattr(config, name, None) for name in
                           ('latent_token_id', 'latent_start_id', 'latent_end_id'))
        if any(type(value) is not int for value in (pad, start, end)) or len(
                {pad, start, end}) != 3:
            raise ValueError('history requires configured distinct latent token IDs')
        spans, opened = [], None
        tokens = self._ids[0].tolist()
        for index in range(self._valid_start, self._valid_end):
            token = tokens[index]
            if token == start:
                if opened is not None:
                    raise ValueError('history requires complete nonnested latent blocks')
                opened = index
            elif token == end:
                if opened is None or index == opened + 1:
                    raise ValueError('history requires complete nonempty latent blocks')
                spans.append((opened, index, tuple(range(opened + 1, index))))
                opened = None
            elif token == pad:
                if opened is None:
                    raise ValueError('latent pad outside a complete block')
            elif opened is not None:
                raise ValueError('latent block body must contain only pad tokens')
        if opened is not None:
            raise ValueError('history requires complete latent blocks')
        return tuple(spans)

    def _validate_images(self, pixels):
        grid = self._grid
        if (grid.ndim != 2 or grid.shape[1] != 3 or grid.dtype != torch.long
                or bool((grid <= 0).any())):
            raise ValueError('original image grid must be positive integer [N,3]')
        merge = self._qwen.visual.spatial_merge_size
        if bool((grid[:, 1:] % merge != 0).any()):
            raise ValueError('original image grid must align with spatial merge size')
        if pixels.ndim != 2 or pixels.shape[0] != int(grid.prod(dim=1).sum()):
            raise ValueError('original pixel count must match original grid')
        config = self._qwen.config
        tokens = self._ids[0].tolist()
        if config.video_token_id in tokens:
            raise ValueError('history video trajectories are unsupported')
        image_positions = [i for i, token in enumerate(tokens) if token == config.image_token_id]
        first_start = self._spans[0][0] if self._spans else self._valid_end
        if any(not self._valid_start <= i < first_start for i in image_positions):
            raise ValueError('original image placement must precede latent trajectory')
        expected_positions, image_ends = [], []
        image_index = 0
        for index, token in enumerate(tokens):
            if token != config.vision_start_token_id:
                continue
            if (index + 1 >= len(tokens) or tokens[index + 1] != config.image_token_id
                    or image_index >= grid.shape[0]):
                raise ValueError('unsupported vision marker or original grid ownership')
            t, h, w = grid[image_index].tolist()
            count = t * (h // merge) * (w // merge)
            expected_positions.extend(range(index + 1, index + 1 + count))
            image_ends.append(index + 1 + count)
            image_index += 1
        if image_index != grid.shape[0] or expected_positions != image_positions:
            raise ValueError('original image token count/placement does not match grid')
        self._image_end = max(image_ends, default=self._valid_start)

    @torch.no_grad()
    def prepare(self, stop: int, selections: tuple[HistorySelection, ...]
                ) -> PreparedHistoryPrefix:
        self._require_frozen()
        if (type(stop) is not int or not max(self._image_end, self._valid_start + 1)
                <= stop <= self._valid_end
                or any(start < stop <= end for start, end, _ in self._spans)):
            raise ValueError('history stop must end a nonempty complete original-image prefix')
        embeddings = self._embeddings[:, :stop].clone()
        previous = -1
        for selection in selections:
            index = selection.block_index
            if (type(index) is not int or not previous < index < len(self._spans)):
                raise ValueError('history selections require unique ordered past block indices')
            _, end, pads = self._spans[index]
            positions, values = selection.positions, selection.values
            if (end >= stop or not isinstance(positions, tuple)
                    or any(type(p) is not int for p in positions)
                    or positions != pads[:len(positions)]):
                raise ValueError('history selections must be first Kstar pads of a past block')
            if (values.ndim != 2 or values.shape != (len(positions), embeddings.shape[-1])
                    or not values.is_floating_point()):
                raise ValueError('selected raw targets must have floating [Kstar,H] shape')
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError('selected raw targets must be finite')
            candidates = selection.candidate_indices
            if (not isinstance(candidates, tuple)
                    or (candidates and len(candidates) != len(positions))
                    or any(type(i) is not int or i < 0 for i in candidates)
                    or len(set(candidates)) != len(candidates)):
                raise ValueError('candidate indices must uniquely match selected target rows')
            embeddings[0, list(positions)] = values.detach().to(embeddings)
            previous = index
        ids, mask = self._ids[:, :stop].clone(), self._mask[:, :stop].clone()
        # Unlike wrapper.forward, get_rope_index is pure: do not write rope_deltas
        # or supply a cache. Original image coordinates survive latent injection.
        positions, _ = self._qwen.get_rope_index(
            input_ids=ids, image_grid_thw=self._grid, attention_mask=mask)
        return PreparedHistoryPrefix(ids, embeddings, mask, positions)

    @torch.no_grad()
    def forward(self, prefix: PreparedHistoryPrefix) -> HistoryPrefixObservation:
        self._require_frozen()
        outputs = self.backbone(
            input_ids=None, inputs_embeds=prefix.inputs_embeds,
            attention_mask=prefix.attention_mask, position_ids=prefix.position_ids,
            past_key_values=None, use_cache=False, output_hidden_states=False,
            output_attentions=False, return_dict=True)
        return HistoryPrefixObservation(outputs.last_hidden_state)
