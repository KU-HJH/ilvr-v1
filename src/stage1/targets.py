"""Batch-safe EMA-teacher target selection for Stage 1 training."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import MutableMapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from peft_common import resolve_qwen2_5_vl_model
from stage1.history_attention import ReducedHistoryAttentionObserver
from stage1.history_teacher import HistoryPrefixAdapter, HistorySelection
from stage1.query import (
    DiscountedTextState, Stage1QueryConfig, aggregate_history_latents,
    compose_history_query, history_text_positions, resolve_query_layers,
    update_text_history,
)
from stage1.types import (
    Stage1BatchMetadata,
    Stage1SampleMetadata,
    Stage1TeacherTargets,
    Stage1RejectedLatentBlock,
)


class _UnsupportedAttentionInterface(Exception):
    """Signal that a teacher cannot provide the optional attention query."""


class Stage1TeacherTargetBuilder:
    """Build sample-isolated helper-image targets from row-owned teacher passes."""

    def __init__(
        self,
        tokenizer,
        *,
        coverage_p: float = 0.9,
        image_pool_k: int = 8,
        helper_group_L: int | None = 256,
        textual_states: str = "previous",
        validate: bool = True,
        collect_diagnostics: bool = True,
        query_config: Stage1QueryConfig | None = None,
        run_cache=None,
    ) -> None:
        if not math.isfinite(coverage_p) or not 0.0 < coverage_p <= 1.0:
            raise ValueError("coverage_p must be finite and in (0, 1]")
        if image_pool_k <= 0:
            raise ValueError("image_pool_k must be positive")
        if textual_states not in ("all", "previous"):
            raise ValueError("textual_states must be 'all' or 'previous'")
        if query_config is not None and not isinstance(query_config, Stage1QueryConfig):
            raise TypeError("query_config must be a Stage1QueryConfig or None")
        self.query_config = query_config or Stage1QueryConfig()
        self.validate = bool(validate)
        self.collect_diagnostics = bool(collect_diagnostics)
        self.run_cache = run_cache
        self.tokenizer = tokenizer
        self.coverage_p = float(coverage_p)
        self.image_pool_k = int(image_pool_k)
        self.helper_group_L = (
            self.image_pool_k if helper_group_L is None else int(helper_group_L)
        )
        self.textual_states = textual_states
        self._assistant_patterns = self._token_patterns(
            ("<|im_start|>assistant", "<|im_start|>assistant\n")
        )
        self._special_ids = {
            self._single_token_id("<|latent_pad|>"),
            self._single_token_id("<|latent_start|>"),
            self._single_token_id("<|latent_end|>"),
        }
        for token in ("<|vision_start|>", "<|vision_end|>"):
            try:
                self._special_ids.add(self._single_token_id(token))
            except (KeyError, TypeError, ValueError):
                pass

    def _token_patterns(self, texts: Sequence[str]) -> tuple[torch.Tensor, ...]:
        patterns = []
        for text in texts:
            ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
            if ids.numel() and not any(torch.equal(ids, item) for item in patterns):
                patterns.append(ids.detach().to(dtype=torch.long, device="cpu"))
        if not patterns:
            raise ValueError("assistant marker must tokenize to at least one token")
        return tuple(patterns)

    def _single_token_id(self, text: str) -> int:
        ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
        if ids.numel() == 0:
            raise ValueError(f"{text} must tokenize to at least one token")
        return int(ids[0].item())

    @torch.no_grad()
    def build(
        self,
        teacher: nn.Module,
        batch: MutableMapping[str, object],
        metadata: Stage1BatchMetadata,
        *, voff_mode: str = 'attention_mask',
    ) -> Stage1TeacherTargets:
        """Consume helper tensors and return row-major targets for the batch."""
        from .v5_config import V5_VOFF_MODES
        if voff_mode not in V5_VOFF_MODES:
            raise ValueError('unsupported Stage1-v5 voff_mode')
        cache = self.run_cache
        sources = batch.pop('stage1_cache_sources', None)
        if cache is not None:
            # PyTorch's pinned-memory collator preserves values but converts
            # ordinary tuples to lists; ownership is the ordered IDs, not type.
            sources = tuple(sources) if isinstance(sources, (tuple, list)) else None
            if sources != tuple(sample.source_index for sample in metadata.samples):
                raise ValueError('cache source descriptors must match current samples')
            if cache.phase.value == 1 and cache.plan.helper == 'targets':
                from .cache_payloads import decode_target_batch, restore_teacher_forward_state
                records = tuple(cache.read('targets', index) for index in sources)
                batch.pop('pixel_values_latent', None)
                batch.pop('image_grid_thw_latent')
                targets = decode_target_batch(records, metadata,
                    device=self._teacher_device(teacher, batch['input_ids'].device))
                executed = [record['forward_state'] for record in records if record['forward_state'] is not None]
                if executed:
                    restore_teacher_forward_state(teacher, executed[-1])
                if self.collect_diagnostics:
                    diagnostics = self._diagnostics(targets.latent_values)
                    if targets.rejected_blocks:
                        for kind in ('helper', 'original'):
                            diagnostics[f'rejected_{kind}_blocks'] = targets.latent_values.new_tensor(
                                sum(b.donor_kind == kind for b in targets.rejected_blocks), dtype=torch.long)
                    targets = replace(targets, diagnostics=diagnostics)
                return targets
        elif sources is not None:
            raise ValueError('cached preprocessing requires its bound target cache')
        vision_replay = cache is not None and cache.phase.value == 1 and cache.plan.helper == 'vision'
        rejected_blocks = []
        missing_helpers = [
            key
            for key in ("pixel_values_latent", "image_grid_thw_latent")
            if key not in batch and not (vision_replay and key == 'pixel_values_latent')
        ]
        if missing_helpers:
            raise KeyError(f"missing Stage 1 helper tensors: {missing_helpers}")
        helper_pixels = batch.pop("pixel_values_latent", None)
        helper_grid = batch.pop("image_grid_thw_latent")

        input_ids = self._require_tensor(batch, "input_ids")
        attention_mask = self._require_tensor(batch, "attention_mask")
        original_pixels = self._require_tensor(batch, "pixel_values")
        original_grid = self._require_tensor(batch, "image_grid_thw")
        if not vision_replay:
            helper_pixels = self._require_tensor_value(helper_pixels, 'pixel_values_latent')
        helper_grid = self._require_tensor_value(
            helper_grid, "image_grid_thw_latent"
        )
        if self.validate or self.query_config.mode == "adaptive":
            self._validate_batch_shape(input_ids, attention_mask, metadata)

        if self.query_config.mode == "adaptive":
            if metadata.batch_size != 1:
                raise ValueError("adaptive history requires batch size 1")
            if any(module.training for module in teacher.modules()) or any(
                p.requires_grad or p.grad is not None for p in teacher.parameters()
            ):
                raise ValueError("adaptive history teacher must be frozen and in eval mode")
            sample_length = metadata.samples[0].unpadded_length
            if (type(sample_length) is not int
                    or not 0 < sample_length <= metadata.sequence_length
                    or not bool((attention_mask[:, :sample_length] == 1).all())
                    or not bool((attention_mask[:, sample_length:] == 0).all())):
                raise ValueError("adaptive attention mask must match unpadded metadata length")
        else:
            teacher.eval()
        device = self._teacher_device(teacher, input_ids.device)
        teacher_ids = input_ids.to(device)
        teacher_attention = attention_mask.to(device)
        original_pixels = original_pixels.to(device)
        original_grid = original_grid.to(device)
        if helper_pixels is not None:
            helper_pixels = helper_pixels.to(
                device=device, dtype=getattr(teacher.visual, 'dtype', helper_pixels.dtype))
        helper_grid = helper_grid.to(device)

        image_token_id = self._teacher_image_token_id(teacher)

        latent_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        packed: list[torch.Tensor] = []
        selected_counts: list[tuple[int, ...]] = []
        selected_candidate_indices: list[tuple[tuple[int, ...], ...]] = []
        selection_margins: list[tuple[float | str, ...]] = []
        forward_states = []
        for sample_index, sample in enumerate(metadata.samples):
            (
                sample_values,
                block_counts,
                block_indices,
                block_margins,
                sample_empty_values,
            ) = (
                self._build_sample(
                    teacher=teacher,
                    sample=sample,
                    sample_ids=teacher_ids[
                        sample_index : sample_index + 1, : sample.unpadded_length
                    ],
                    sample_attention=teacher_attention[
                        sample_index : sample_index + 1, : sample.unpadded_length
                    ],
                    original_pixels=self._owned_slice(
                        original_pixels,
                        sample.original_pixel_range,
                        sample,
                        "original pixel",
                    ),
                    original_grid=self._owned_slice(
                        original_grid,
                        sample.original_grid_range,
                        sample,
                        "original grid",
                    ),
                    helper_pixels=None if helper_pixels is None else self._owned_slice(
                        helper_pixels,
                        sample.helper_pixel_range,
                        sample,
                        "helper pixel",
                    ),
                    helper_grid=self._owned_slice(
                        helper_grid,
                        sample.helper_grid_range,
                        sample,
                        "helper grid",
                    ),
                    image_token_id=image_token_id,
                    **({'rejected_blocks': rejected_blocks, 'batch_index': sample_index}
                       if voff_mode != 'attention_mask' else {}),
                )
            )
            packed.extend(sample_values)
            packed.append(sample_empty_values)
            selected_counts.append(block_counts)
            selected_candidate_indices.append(block_indices)
            selection_margins.append(block_margins)
            if cache is not None and cache.plan.helper == 'targets':
                from .cache_payloads import capture_teacher_forward_state
                forward_states.append(None if self.query_config.mode == 'adaptive' and not sample.latent_spans
                                      else capture_teacher_forward_state(teacher))
            for span, selected_count in zip(sample.latent_spans, block_counts):
                if selected_count:
                    latent_mask[
                        sample_index,
                        list(span.latent_positions[:selected_count]),
                    ] = True

        latent_values = torch.cat(packed, dim=0)
        if ((self.validate or self.query_config.mode == "adaptive")
                and latent_values.shape[0] != int(latent_mask.sum().item())):
            raise RuntimeError(
                "row-major latent target count does not match latent mask"
            )
        diagnostics = self._diagnostics(latent_values) if self.collect_diagnostics else {}
        if rejected_blocks and self.collect_diagnostics:
            for kind in ('helper', 'original'):
                diagnostics[f'rejected_{kind}_blocks'] = latent_values.new_tensor(
                    sum(b.donor_kind == kind for b in rejected_blocks), dtype=torch.long)
        targets = Stage1TeacherTargets(
            latent_values=latent_values,
            latent_mask=latent_mask,
            selected_counts=tuple(selected_counts),
            diagnostics=diagnostics,
            selected_candidate_indices=tuple(selected_candidate_indices),
            selection_margins=tuple(selection_margins),
            rejected_blocks=tuple(rejected_blocks),
        )
        if cache is not None and cache.plan.helper == 'targets':
            from .cache_payloads import encode_target_sample
            for row, sample in enumerate(metadata.samples):
                cache.publish('targets', sample.source_index, encode_target_sample(
                    targets, metadata, row, forward_state=forward_states[row]))
        return targets

    def _helper_visual(self, teacher, sample, pixels, grid):
        cache = self.run_cache
        if cache is not None and cache.plan.helper == 'vision' and cache.phase.value == 1:
            record = cache.read('vision', sample.source_index)
            if record.get('kind') != 'vision' or len(record['counts']) != grid.shape[0]:
                raise ValueError('cached vision partitions do not match helper geometry')
            return record['features'].to(device=grid.device), record['counts']
        features = teacher.visual(pixels, grid_thw=grid)
        counts = self._visual_token_counts(grid, teacher)
        if cache is not None and cache.plan.helper == 'vision':
            cache.publish('vision', sample.source_index,
                {'kind': 'vision', 'features': features, 'counts': counts})
        return features, counts

    def _build_sample(
        self,
        *,
        teacher,
        sample,
        sample_ids,
        sample_attention,
        original_pixels,
        original_grid,
        helper_pixels,
        helper_grid,
        image_token_id,
        rejected_blocks=None, batch_index=0,
    ):
        build_sample = (self._build_sample_adaptive
                        if self.query_config.mode == "adaptive"
                        else self._build_sample_legacy)
        return build_sample(
            teacher=teacher, sample=sample, sample_ids=sample_ids,
            sample_attention=sample_attention, original_pixels=original_pixels,
            original_grid=original_grid, helper_pixels=helper_pixels,
            helper_grid=helper_grid, image_token_id=image_token_id,
            **({'rejected_blocks': rejected_blocks, 'batch_index': batch_index}
               if rejected_blocks is not None else {}),
        )

    def _build_sample_legacy(
        self,
        *,
        teacher,
        sample,
        sample_ids,
        sample_attention,
        original_pixels,
        original_grid,
        helper_pixels,
        helper_grid,
        image_token_id,
        rejected_blocks=None, batch_index=0,
    ):
        text_outputs = teacher(
            input_ids=sample_ids,
            attention_mask=sample_attention,
            pixel_values=original_pixels,
            image_grid_thw=original_grid,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_pre, hidden_last = self._hidden_states(
            text_outputs, (1, sample.unpadded_length), validate=self.validate
        )
        input_embeddings = self._language_input_embeddings(
            teacher, text_outputs, sample_ids
        )
        if helper_grid.shape[0]:
            helper_features, helper_counts = self._helper_visual(teacher, sample, helper_pixels, helper_grid)
            helper_images = self._split_visual_features(
                helper_features, helper_grid, teacher, counts=helper_counts
            )
        else:
            helper_images = ()

        row_ids = sample_ids[0]
        row_hidden_pre = hidden_pre[0]
        row_hidden_last = hidden_last[0]
        if self.textual_states == "all":
            assistant_start, assistant_end = self._assistant_bounds(
                row_ids, sample
            )
        else:
            assistant_start = self._assistant_start(row_ids, sample)
            assistant_end = assistant_start
        user_query = self._user_query(
            teacher=teacher,
            sample=sample,
            sample_ids=row_ids,
            hidden_pre=row_hidden_pre,
            hidden_last=row_hidden_last,
            input_embeddings=input_embeddings[0],
            assistant_start=assistant_start,
            image_token_id=image_token_id,
            original_grid=original_grid,
            original_counts=self._visual_token_counts(original_grid, teacher),
        )
        text_spans = self._assistant_text_spans(
            row_ids,
            sample,
            assistant_end if self.textual_states == "all" else assistant_start,
        )
        candidates_by_block = self._sample_helper_candidates(helper_images, sample)

        packed = []
        block_counts = []
        block_indices = []
        block_margins = []
        text_history_sum = None
        text_history_count = 0
        latent_history_sum = None
        latent_history_count = 0
        previous_selected_mean = None
        for block_index, span in enumerate(sample.latent_spans):
            if text_spans[block_index]:
                text_indices = torch.tensor(
                    text_spans[block_index],
                    device=row_hidden_last.device,
                    dtype=torch.long,
                )
                text_values = row_hidden_last.index_select(0, text_indices)
                if self.textual_states == "all":
                    current_text_sum = text_values.sum(dim=0, keepdim=True)
                    text_history_sum = (
                        current_text_sum
                        if text_history_sum is None
                        else text_history_sum + current_text_sum
                    )
                    text_history_count += int(text_values.shape[0])
            if self.textual_states == "all":
                textual_history = (
                    text_history_sum / text_history_count
                    if text_history_count
                    else None
                )
                latent_history = (
                    latent_history_sum / latent_history_count
                    if latent_history_count
                    else None
                )
            else:
                textual_history = (
                    text_values.mean(dim=0, keepdim=True)
                    if text_spans[block_index]
                    else None
                )
                latent_history = previous_selected_mean
            query = self._reasoning_query(
                user_query, textual_history, latent_history
            )

            selected, indices, margin = self._select(
                candidates_by_block[block_index],
                query,
                max_count=len(span.latent_positions),
            )
            if rejected_blocks is not None:
                rejected_blocks.append(self._rejected_block(
                    teacher, sample, helper_images, query, block_index, batch_index,
                    getattr(text_outputs, 'inputs_embeds', None), sample_ids,
                    original_grid, image_token_id))
            packed.append(selected)
            block_counts.append(int(selected.shape[0]))
            block_indices.append(indices)
            block_margins.append(margin)
            if selected.shape[0]:
                previous_selected_mean = selected.mean(dim=0, keepdim=True)
                if self.textual_states == "all":
                    current_latent_sum = selected.sum(dim=0, keepdim=True)
                    latent_history_sum = (
                        current_latent_sum
                        if latent_history_sum is None
                        else latent_history_sum + current_latent_sum
                    )
                    latent_history_count += int(selected.shape[0])
        return (
            packed,
            tuple(block_counts),
            tuple(block_indices),
            tuple(block_margins),
            row_hidden_last[:0],
        )

    def _build_sample_adaptive(
        self,
        *,
        teacher,
        sample,
        sample_ids,
        sample_attention,
        original_pixels,
        original_grid,
        helper_pixels,
        helper_grid,
        image_token_id,
        rejected_blocks=None, batch_index=0,
    ):
        qwen = resolve_qwen2_5_vl_model(teacher)
        owner = getattr(teacher, "module", teacher)
        configs = getattr(owner, "peft_config", {})
        if (getattr(qwen.config, "model_type", None) != "qwen2_5_vl"
                or getattr(qwen.config, "stage", None) != "stage1"
                or not configs or not all(getattr(c, "use_dora", False)
                                           for c in configs.values())
                or any(layer.attn_type != "sdpa" for layer in qwen.model.layers)):
            raise ValueError("adaptive history requires Stage1 Qwen2.5-VL DoRA with SDPA")
        row_ids = sample_ids[0]
        assistant_start, assistant_end = self._assistant_bounds(row_ids, sample)
        # Count unique starts, not overlapping short/long spellings of one header.
        assistant_starts = set()
        for pattern in self._assistant_patterns:
            pattern = pattern.to(row_ids.device)
            assistant_starts.update(
                i for i in range(row_ids.numel() - pattern.numel() + 1)
                if torch.equal(row_ids[i:i + pattern.numel()], pattern))
        if assistant_starts != {assistant_start}:
            raise ValueError("adaptive history requires a single assistant trajectory")
        special_ids = frozenset(self._special_ids | set(
            getattr(self.tokenizer, "all_special_ids", ())) | {image_token_id})
        text_spans = history_text_positions(
            row_ids, sample_attention[0], sample.latent_spans,
            assistant_end, special_ids)
        pad, start, end = (self._single_token_id(token) for token in
                          ("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"))
        if ((pad, start, end) != (qwen.config.latent_token_id,
                                  qwen.config.latent_start_id, qwen.config.latent_end_id)
                or len({pad, start, end}) != 3):
            raise ValueError("adaptive tokenizer/model latent token IDs must agree")
        expected_markers = {}
        for span in sample.latent_spans:
            expected_markers[span.start] = start
            expected_markers[span.end] = end
            expected_markers.update((p, pad) for p in span.latent_positions)
        actual_markers = {i: token for i, token in enumerate(row_ids.tolist())
                          if token in (pad, start, end)}
        if actual_markers != expected_markers:
            raise ValueError("adaptive latent markers must match complete metadata spans")
        image_positions = (row_ids == image_token_id).nonzero().flatten().tolist()
        if any(i >= assistant_start for i in image_positions):
            raise ValueError("original images must precede the assistant trajectory")
        counts = self._visual_token_counts(original_grid, teacher)
        if (sum(counts) != len(image_positions)
                or original_grid.dtype != torch.long
                or bool((original_grid <= 0).any())
                or bool((original_grid[:, 1:] % teacher.visual.spatial_merge_size != 0).any())
                or original_pixels.ndim != 2
                or original_pixels.shape[0] != int(original_grid.prod(dim=1).sum())):
            raise ValueError("adaptive original image ownership must match pixels and tokens")
        # Validate this even for a no-latent sample, without encoding its images.
        tokens = row_ids.tolist()
        if qwen.config.video_token_id in tokens:
            raise ValueError("adaptive history video trajectories are unsupported")
        vision_starts = [i for i, token in enumerate(tokens)
                         if token == qwen.config.vision_start_token_id]
        expected_images = tuple(p for i, count in zip(vision_starts, counts)
                                for p in range(i + 1, i + 1 + count))
        if (len(vision_starts) != len(counts)
                or expected_images != tuple(image_positions)):
            raise ValueError("adaptive original image markers must match owned grids")
        if (len(sample.helper_to_block) != helper_grid.shape[0]
                or helper_grid.ndim != 2 or helper_grid.shape[1] != 3
                or bool((helper_grid <= 0).any())
                or (helper_pixels is not None and (helper_pixels.ndim != 2
                    or helper_pixels.shape[0] != int(helper_grid.prod(dim=1).sum())))
                or any(type(i) is not int or not 0 <= i < len(sample.latent_spans)
                       for i in sample.helper_to_block)
                or set(sample.helper_to_block) != set(range(len(sample.latent_spans)))):
            raise ValueError("adaptive history requires owned helper candidates for every block")
        # This check covers prompt and first/empty-text blocks, where the observer
        # deliberately has no hooks. Never change the teacher's native mask config.
        stops = (assistant_end, *(span.start for span in sample.latent_spans))
        window = qwen.model.config.sliding_window
        if sample.latent_spans and window is not None and max(stops) > window:
            raise ValueError("adaptive history requires full-causal prefixes within native "
                             f"sliding_window={window}; required length={max(stops)}")
        if not sample.latent_spans:
            empty = qwen.model.embed_tokens.weight.new_empty(
                (0, qwen.model.embed_tokens.weight.shape[1]))
            return [], (), (), (), empty

        adapter = HistoryPrefixAdapter(
            teacher, input_ids=sample_ids, attention_mask=sample_attention,
            original_pixels=original_pixels, original_grid=original_grid)
        helper_features, helper_counts = self._helper_visual(teacher, sample, helper_pixels, helper_grid)
        if self.validate and not bool(torch.isfinite(helper_features).all()):
            raise FloatingPointError("teacher helper visual features must be finite")
        helper_images = self._split_visual_features(helper_features, helper_grid, teacher, counts=helper_counts)
        candidates = self._sample_helper_candidates(helper_images, sample)
        if any(values.shape[0] == 0 for values in candidates):
            raise ValueError("adaptive history requires nonempty candidates for every block")
        layers = resolve_query_layers(self.query_config.attention_layers,
                                      len(adapter.backbone.layers))
        # Native wrapper exposes final-normalized hidden_states as one tensor;
        # preserve that existing global-intent contract outside observer scope.
        prompt = adapter.prepare(assistant_end, ())
        prompt_hidden = adapter.forward(prompt).last_hidden_state
        self._hidden_states(SimpleNamespace(hidden_states=prompt_hidden),
                            (1, assistant_end), validate=self.validate)
        user_query = self._user_query(
            teacher=teacher, sample=sample, sample_ids=prompt.input_ids[0],
            hidden_pre=prompt_hidden[0], hidden_last=prompt_hidden[0],
            input_embeddings=prompt.inputs_embeds[0], assistant_start=assistant_start,
            image_token_id=image_token_id, original_grid=original_grid,
            original_counts=counts)
        packed, block_counts, block_indices, block_margins = [], [], [], []
        ledger = []
        state = DiscountedTextState()
        for index, span in enumerate(sample.latent_spans):
            prefix = adapter.prepare(span.start, tuple(ledger))
            active_positions = tuple(p for selection in ledger for p in selection.positions)
            with ReducedHistoryAttentionObserver(
                adapter.backbone, layer_indices=layers,
                query_positions=text_spans[index], latent_positions=active_positions,
                attention_mask=prefix.attention_mask, config=self.query_config,
            ) as observer:
                hidden = adapter.forward(prefix).last_hidden_state
            if (hidden.shape != prefix.inputs_embeds.shape
                    or (self.validate and not bool(torch.isfinite(hidden).all()))):
                raise ValueError("adaptive prefix hidden states must be finite [1,S,H]")
            current_mean = (hidden[0, list(text_spans[index])].float().mean(0, keepdim=True)
                            if text_spans[index] else None)
            state, text_summary = update_text_history(
                state, current_mean, gamma=self.query_config.text_gamma)
            latent_sum, mass = aggregate_history_latents(
                hidden[0, list(active_positions)], observer.scores())
            query = compose_history_query(user_query, text_summary, latent_sum, mass).query
            selected, indices, margin = self._select(
                candidates[index], query, max_count=len(span.latent_positions))
            if rejected_blocks is not None:
                rejected_blocks.append(self._rejected_block(
                    teacher, sample, helper_images, query, index, batch_index,
                    prompt.inputs_embeds, prompt.input_ids, original_grid, image_token_id))
            if (selected.ndim != 2 or not 0 < selected.shape[0] <= len(span.latent_positions)
                    or selected.shape[1] != hidden.shape[-1]
                    or (self.validate and not bool(torch.isfinite(selected).all()))):
                raise ValueError("adaptive selection must contain finite raw first-Kstar targets")
            count = int(selected.shape[0])
            ledger.append(HistorySelection(index, span.latent_positions[:count],
                                           selected, indices))
            packed.append(selected)
            block_counts.append(count)
            block_indices.append(indices)
            block_margins.append(margin)
        return (packed, tuple(block_counts), tuple(block_indices), tuple(block_margins),
                prompt_hidden[0, :0])

    def _rejected_block(self, teacher, sample, helper_images, query, block_index,
                        batch_index, original_embeddings, original_ids, original_grid,
                        image_token_id):
        from .v5_donors import select_reverse_topk_donors
        prefix = f'source_index={sample.source_index} block={block_index}: '
        span = sample.latent_spans[block_index]
        if (not span.latent_positions
                or span.latent_positions != tuple(range(span.start + 1, span.end))
                or len(sample.helper_image_ids) != len(helper_images)):
            raise ValueError(prefix + 'replacement requires complete body and helper image identities')
        excluded = {identity for identity, owner in zip(sample.helper_image_ids, sample.helper_to_block)
                    if owner == block_index}
        original_candidates = ()
        if not any(identity not in excluded for identity in sample.helper_image_ids):
            # This must be the prepared teacher feature tensor, never the generic
            # special-token embedding fallback used by the historical query path.
            if (not isinstance(original_embeddings, torch.Tensor)
                    or original_embeddings.ndim != 3
                    or original_embeddings.shape[:2] != original_ids.shape
                    or original_embeddings.shape[0] != 1):
                raise ValueError(prefix + 'missing projected original image embeddings')
            merge = int(teacher.visual.spatial_merge_size)
            if (original_grid.dtype != torch.long or original_grid.ndim != 2
                    or original_grid.shape[-1] != 3 or merge <= 0
                    or bool((original_grid <= 0).any())
                    or bool((original_grid[:, 1:] % merge != 0).any())):
                raise ValueError(prefix + 'invalid projected original image grid')
            counts = self._visual_token_counts(original_grid, teacher)
            values = original_embeddings[0, original_ids[0] == image_token_id]
            if (not counts or sum(counts) != values.shape[0]
                    or len(sample.original_image_ids) != len(counts)
                    or any(not isinstance(i, str) or not i for i in sample.original_image_ids)):
                raise ValueError(prefix + 'projected original image feature ownership mismatch')
            original_candidates = tuple(self._maybe_group_for_helper(v, self.helper_group_L)
                                        for v in values.split(counts))
        grouped = tuple(self._maybe_group_for_helper(v, self.helper_group_L) for v in helper_images)
        selected = select_reverse_topk_donors(grouped, sample.helper_image_ids,
            sample.helper_to_block, block_index, original_candidates, query,
            count=len(span.latent_positions), source_index=sample.source_index)
        return Stage1RejectedLatentBlock(batch_index, block_index, span.latent_positions,
            selected.values, selected.donor_kind, selected.image_indices,
            selected.candidate_indices, f'clean_teacher_{self.query_config.mode}')

    @staticmethod
    def _owned_slice(tensor, owned_range, sample, role):
        start, end = owned_range
        if not 0 <= start <= end <= tensor.shape[0]:
            raise ValueError(
                f"source_index={sample.source_index}: invalid {role} range"
            )
        return tensor[start:end]

    @staticmethod
    def _require_tensor(batch: MutableMapping[str, object], key: str) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"missing Stage 1 tensor: {key}")
        return Stage1TeacherTargetBuilder._require_tensor_value(batch[key], key)

    @staticmethod
    def _require_tensor_value(value: object, key: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Stage 1 {key} must be a tensor")
        return value

    @staticmethod
    def _teacher_device(teacher: nn.Module, fallback: torch.device) -> torch.device:
        try:
            return next(teacher.parameters()).device
        except StopIteration:
            return fallback

    @staticmethod
    def _teacher_image_token_id(teacher: nn.Module) -> int:
        config = getattr(teacher, "config", None)
        image_token_id = getattr(config, "image_token_id", None)
        if (
            isinstance(image_token_id, bool)
            or not isinstance(image_token_id, int)
            or image_token_id < 0
        ):
            raise ValueError(
                "teacher.config.image_token_id must be a non-negative integer"
            )
        return image_token_id

    @staticmethod
    def _validate_batch_shape(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Stage1BatchMetadata,
    ) -> None:
        expected = (metadata.batch_size, metadata.sequence_length)
        if tuple(input_ids.shape) != expected or tuple(attention_mask.shape) != expected:
            raise ValueError(
                "Stage 1 teacher input_ids and attention_mask must match metadata B,T"
            )
        if len(metadata.samples) != metadata.batch_size:
            raise ValueError("Stage 1 metadata sample count must match batch_size")

    @staticmethod
    def _hidden_states(text_outputs, expected_prefix, *, validate: bool = True):
        hidden_states = getattr(text_outputs, "hidden_states", None)
        if isinstance(hidden_states, (tuple, list)):
            if not hidden_states:
                raise ValueError("teacher language forward returned no hidden states")
            hidden_last = hidden_states[-1]
            hidden_pre = hidden_states[-2] if len(hidden_states) > 1 else hidden_last
        elif isinstance(hidden_states, torch.Tensor):
            hidden_last = hidden_states
            hidden_pre = hidden_states
        else:
            raise TypeError("teacher language forward must return hidden states")
        if hidden_last.ndim != 3 or tuple(hidden_last.shape[:2]) != expected_prefix:
            raise ValueError("teacher hidden states must have shape [B, T, H]")
        if hidden_pre.shape != hidden_last.shape:
            raise ValueError("teacher pre-final and final hidden states must share shape")
        if validate and (not bool(torch.isfinite(hidden_pre).all().item()) or not bool(
            torch.isfinite(hidden_last).all().item()
        )):
            raise FloatingPointError("teacher hidden states must be finite")
        return hidden_pre, hidden_last

    @staticmethod
    def _language_input_embeddings(teacher, text_outputs, input_ids):
        inputs_embeds = getattr(text_outputs, "inputs_embeds", None)
        if isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.ndim == 3:
            return inputs_embeds
        return teacher.get_input_embeddings()(input_ids)

    @staticmethod
    def _visual_token_counts(grid_thw: torch.Tensor, teacher) -> tuple[int, ...]:
        if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
            raise ValueError("image_grid_thw must have shape [N, 3]")
        merge = int(getattr(teacher.visual, "spatial_merge_size", 2))
        if merge <= 0:
            raise ValueError("teacher visual spatial_merge_size must be positive")
        grid = grid_thw.to(dtype=torch.long)
        counts = grid[:, 0] * (grid[:, 1] // merge) * (grid[:, 2] // merge)
        return tuple(int(count.item()) for count in counts)

    def _split_visual_features(self, features, grid_thw, teacher, *, counts=None):
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("teacher helper visual features must have shape [N, H]")
        if self.validate and not bool(torch.isfinite(features).all().item()):
            raise FloatingPointError("teacher helper visual features must be finite")
        counts = self._visual_token_counts(grid_thw, teacher) if counts is None else counts
        if sum(counts) != features.shape[0]:
            raise ValueError(
                "teacher helper feature rows must match helper image grids"
            )
        images = []
        offset = 0
        for count in counts:
            images.append(features[offset : offset + count])
            offset += count
        return tuple(images)

    def _assistant_bounds(
        self, sample_ids: torch.Tensor, sample: Stage1SampleMetadata
    ) -> tuple[int, int]:
        best_match = None
        for cpu_pattern in self._assistant_patterns:
            pattern = cpu_pattern.to(sample_ids.device)
            for start in range(sample_ids.numel() - pattern.numel() + 1):
                if torch.equal(sample_ids[start : start + pattern.numel()], pattern):
                    if start > 0:
                        candidate = (start, start + int(pattern.numel()))
                        if (
                            best_match is None
                            or candidate[0] < best_match[0]
                            or (
                                candidate[0] == best_match[0]
                                and candidate[1] > best_match[1]
                            )
                        ):
                            best_match = candidate
                    break
        if best_match is not None:
            return best_match
        raise ValueError(
            f"source_index={sample.source_index}: assistant marker must follow the user prefix"
        )

    def _assistant_start(
        self, sample_ids: torch.Tensor, sample: Stage1SampleMetadata
    ) -> int:
        for cpu_pattern in self._assistant_patterns:
            pattern = cpu_pattern.to(sample_ids.device)
            for start in range(sample_ids.numel() - pattern.numel() + 1):
                if torch.equal(sample_ids[start : start + pattern.numel()], pattern):
                    if start > 0:
                        return start
                    break
        raise ValueError(
            f"source_index={sample.source_index}: assistant marker must follow the user prefix"
        )

    def _user_query(
        self,
        *,
        teacher,
        sample,
        sample_ids,
        hidden_pre,
        hidden_last,
        input_embeddings,
        assistant_start,
        image_token_id,
        original_grid,
        original_counts,
    ):
        prefix_ids = sample_ids[:assistant_start]
        prompt_indices = [
            index
            for index, token in enumerate(prefix_ids.tolist())
            if token not in self._special_ids and token != image_token_id
        ]
        image_indices = (
            (prefix_ids == image_token_id).nonzero(as_tuple=False).view(-1).tolist()
        )
        if prompt_indices and image_indices:
            try:
                return self._attention_user_query(
                    teacher,
                    hidden_pre,
                    hidden_last,
                    prompt_indices,
                    image_indices,
                )
            except _UnsupportedAttentionInterface:
                return self._cosine_user_query(
                    hidden_last, prompt_indices, image_indices
                )

        token_embeddings = teacher.get_input_embeddings()(sample_ids.unsqueeze(0))
        text_mean = self._prefix_text_mean(
            token_embeddings, sample_ids.unsqueeze(0), assistant_start
        )
        image_mean = self._original_image_mean(
            input_embeddings,
            sample_ids,
            sample,
            image_token_id,
            original_grid,
            original_counts,
        )
        parts = [text_mean]
        if image_mean is not None:
            parts.append(image_mean)
        return torch.stack([part.squeeze(0) for part in parts], dim=0).mean(
            dim=0, keepdim=True
        )

    @staticmethod
    def _last_self_attention(teacher):
        model = getattr(teacher, "model", None)
        layers = getattr(model, "layers", None)
        if layers is None:
            get_base_model = getattr(teacher, "get_base_model", None)
            if not callable(get_base_model):
                raise _UnsupportedAttentionInterface
            base = teacher.get_base_model()
            layers = getattr(getattr(base, "model", None), "layers", None)
        if layers is None:
            raise _UnsupportedAttentionInterface
        try:
            attention = getattr(layers[-1], "self_attn", None)
        except (IndexError, TypeError) as error:
            raise _UnsupportedAttentionInterface from error
        if attention is None:
            raise _UnsupportedAttentionInterface
        if not callable(getattr(attention, "q_proj", None)) or not callable(
            getattr(attention, "k_proj", None)
        ):
            raise _UnsupportedAttentionInterface
        if getattr(attention, "num_heads", None) is None or getattr(
            attention, "num_key_value_heads", None
        ) is None:
            raise _UnsupportedAttentionInterface
        return attention

    def _attention_user_query(
        self, teacher, hidden_pre, hidden_last, prompt_indices, image_indices
    ):
        attention = self._last_self_attention(teacher)
        q = attention.q_proj(hidden_pre.unsqueeze(0))
        k = attention.k_proj(hidden_pre.unsqueeze(0))
        num_heads = int(attention.num_heads)
        head_dim = q.shape[-1] // num_heads
        q = q.view(1, hidden_pre.shape[0], num_heads, head_dim).transpose(1, 2)
        key_value_heads = int(attention.num_key_value_heads)
        k = k.view(
            1, hidden_pre.shape[0], key_value_heads, head_dim
        ).transpose(1, 2)
        if key_value_heads != num_heads:
            repetitions = num_heads // key_value_heads
            k = (
                k[:, :, None, :, :]
                .expand(1, key_value_heads, repetitions, hidden_pre.shape[0], head_dim)
                .reshape(1, num_heads, hidden_pre.shape[0], head_dim)
            )
        q_selected = q[:, :, prompt_indices, :]
        k_selected = k[:, :, image_indices, :]
        logits = torch.einsum(
            "bhpd,bhqd->bhpq", q_selected, k_selected
        ) / math.sqrt(head_dim)
        weights = torch.softmax(logits, dim=-1).mean(dim=2).mean(dim=1).squeeze(0)
        weights = weights / (weights.sum() + 1e-9)
        image_values = hidden_last[image_indices]
        prompt_values = hidden_last[prompt_indices]
        image_summary = (weights.unsqueeze(0) @ image_values).squeeze(0)
        text_summary = prompt_values.mean(dim=0)
        return torch.stack((image_summary, text_summary), dim=0).mean(
            dim=0, keepdim=True
        )

    @staticmethod
    def _cosine_user_query(hidden_last, prompt_indices, image_indices):
        image_values = hidden_last[image_indices]
        prompt_values = hidden_last[prompt_indices]
        similarities = (prompt_values @ image_values.T) / math.sqrt(
            image_values.shape[-1]
        )
        weights = torch.softmax(similarities, dim=-1).mean(dim=0)
        weights = weights / (weights.sum() + 1e-9)
        image_summary = (weights.unsqueeze(0) @ image_values).squeeze(0)
        text_summary = prompt_values.mean(dim=0)
        return torch.stack((image_summary, text_summary), dim=0).mean(
            dim=0, keepdim=True
        )

    def _prefix_text_mean(self, token_embeddings, input_ids, upto_index):
        text_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if upto_index > 0:
            text_mask[:, :upto_index] = True
        for token_id in self._special_ids:
            text_mask &= input_ids != token_id
        features = token_embeddings[0, text_mask[0]]
        if features.numel() == 0:
            features = token_embeddings[0, : max(1, upto_index)]
        return features.mean(dim=0, keepdim=True)

    def _original_image_mean(
        self,
        input_embeddings,
        sample_ids,
        sample,
        image_token_id,
        original_grid,
        original_counts,
    ):
        image_values = input_embeddings[sample_ids == image_token_id]
        counts = original_counts
        if sum(counts) != image_values.shape[0] or not counts:
            return None
        grouped = []
        offset = 0
        for count in counts:
            grouped.append(
                self._grouped_mean(
                    image_values[offset : offset + count], self.image_pool_k
                )
            )
            offset += count
        return torch.cat(grouped, dim=0).mean(dim=0, keepdim=True)

    def _assistant_text_spans(self, sample_ids, sample, assistant_start):
        spans = []
        previous = assistant_start
        for span in sample.latent_spans:
            indices = [
                index
                for index in range(previous, span.start)
                if int(sample_ids[index].item()) not in self._special_ids
            ]
            spans.append(indices)
            previous = span.end + 1
        return tuple(spans)

    @staticmethod
    def _reasoning_query(global_intent, textual_history=None, latent_history=None):
        parts = [global_intent]
        if textual_history is not None:
            parts.append(textual_history)
        if latent_history is not None:
            parts.append(latent_history)
        return torch.cat(parts, dim=0).mean(dim=0, keepdim=True)

    def _sample_helper_candidates(self, helper_images, sample):
        if len(helper_images) != len(sample.helper_to_block):
            raise ValueError(
                f"source_index={sample.source_index}: helper ownership count mismatch"
            )
        by_block: list[list[torch.Tensor]] = [
            [] for _ in sample.latent_spans
        ]
        for image, block_index in zip(helper_images, sample.helper_to_block):
            if not 0 <= block_index < len(by_block):
                raise ValueError(
                    f"source_index={sample.source_index}: invalid helper-to-block mapping"
                )
            by_block[block_index].append(
                self._maybe_group_for_helper(image, self.helper_group_L)
            )
        hidden_size = helper_images[0].shape[-1] if helper_images else 0
        candidates = []
        for block_images in by_block:
            if block_images:
                candidates.append(torch.cat(block_images, dim=0))
            else:
                device = helper_images[0].device if helper_images else torch.device("cpu")
                dtype = helper_images[0].dtype if helper_images else torch.float32
                candidates.append(
                    torch.empty((0, hidden_size), device=device, dtype=dtype)
                )
        return tuple(candidates)

    @staticmethod
    def _grouped_mean(features: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return features.new_zeros((0, features.shape[-1]))
        patch_count = features.shape[0]
        if patch_count >= count and patch_count % count:
            usable = patch_count - patch_count % count
            if usable <= 0:
                return features.new_zeros((count, features.shape[-1]))
            features = features[:usable]
        if features.shape[0] >= count and features.shape[0] % count == 0:
            return features.view(
                count, features.shape[0] // count, features.shape[-1]
            ).mean(dim=1)
        if features.shape[0] == 0:
            return features.new_zeros((count, features.shape[-1]))
        repeats = math.ceil(count / features.shape[0])
        return features.repeat(repeats, 1)[:count]

    def _maybe_group_for_helper(self, features, group_count):
        if group_count is None or group_count <= 0:
            return features
        if features.shape[0] < group_count:
            return features
        return self._grouped_mean(features, group_count)

    def _select(self, candidates, query, max_count):
        if candidates.shape[0] == 0 or max_count <= 0:
            return candidates[:0], (), "all_selected"
        similarities = F.cosine_similarity(
            candidates, query.expand_as(candidates), dim=-1
        )
        if self.validate and not bool(torch.isfinite(similarities).all().item()):
            raise FloatingPointError(
                "teacher candidate similarities must be finite"
            )
        probabilities = torch.softmax(similarities, dim=0)
        if self.validate and not bool(torch.isfinite(probabilities).all().item()):
            raise FloatingPointError(
                "teacher selection probabilities must be finite"
            )
        sorted_indices = torch.argsort(
            probabilities, descending=True, stable=True
        )
        sorted_probabilities = probabilities.index_select(0, sorted_indices)
        cumulative = torch.cumsum(sorted_probabilities, dim=0)
        positions = torch.nonzero(
            cumulative >= self.coverage_p, as_tuple=False
        )
        coverage_count = (
            int(positions[0].item()) + 1
            if positions.numel()
            else int(probabilities.numel())
        )
        selected_count = min(coverage_count, int(max_count))
        selected_indices = sorted_indices[:selected_count]
        if self.collect_diagnostics and selected_count < int(probabilities.numel()):
            margin = float(
                (
                    sorted_probabilities[selected_count - 1].to(torch.float64)
                    - sorted_probabilities[selected_count].to(torch.float64)
                ).item()
            )
            if not math.isfinite(margin):
                raise FloatingPointError("teacher selection margin must be finite")
        else:
            margin = "all_selected"
        return (
            candidates.index_select(0, selected_indices),
            tuple(int(index) for index in selected_indices.tolist()) if self.collect_diagnostics else (),
            margin,
        )

    @staticmethod
    def _diagnostics(latent_values):
        if latent_values.shape[0]:
            norms = latent_values.float().norm(dim=-1)
            mean_norm = norms.mean()
            max_norm = norms.max()
        else:
            mean_norm = latent_values.new_zeros((), dtype=torch.float32)
            max_norm = latent_values.new_zeros((), dtype=torch.float32)
        return {
            "latent_count": torch.tensor(
                latent_values.shape[0],
                device=latent_values.device,
                dtype=torch.long,
            ),
            "latent_norm_mean": mean_norm.detach(),
            "latent_norm_max": max_norm.detach(),
        }
