"""Batch-safe EMA-teacher target selection for Stage 1 training."""

from __future__ import annotations

import math
from typing import MutableMapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from stage1.types import (
    Stage1BatchMetadata,
    Stage1SampleMetadata,
    Stage1TeacherTargets,
)


class Stage1TeacherTargetBuilder:
    """Build sample-isolated helper-image targets from one batched teacher pass."""

    def __init__(
        self,
        tokenizer,
        *,
        coverage_p: float = 0.9,
        image_pool_k: int = 8,
        helper_group_L: int | None = 256,
    ) -> None:
        if not math.isfinite(coverage_p) or not 0.0 < coverage_p <= 1.0:
            raise ValueError("coverage_p must be finite and in (0, 1]")
        if image_pool_k <= 0:
            raise ValueError("image_pool_k must be positive")
        self.tokenizer = tokenizer
        self.coverage_p = float(coverage_p)
        self.image_pool_k = int(image_pool_k)
        self.helper_group_L = (
            None if helper_group_L is None else int(helper_group_L)
        )
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
    ) -> Stage1TeacherTargets:
        """Consume helper tensors and return row-major targets for the batch."""
        missing_helpers = [
            key
            for key in ("pixel_values_latent", "image_grid_thw_latent")
            if key not in batch
        ]
        if missing_helpers:
            raise KeyError(f"missing Stage 1 helper tensors: {missing_helpers}")
        helper_pixels = batch.pop("pixel_values_latent")
        helper_grid = batch.pop("image_grid_thw_latent")

        input_ids = self._require_tensor(batch, "input_ids")
        attention_mask = self._require_tensor(batch, "attention_mask")
        original_pixels = self._require_tensor(batch, "pixel_values")
        original_grid = self._require_tensor(batch, "image_grid_thw")
        helper_pixels = self._require_tensor_value(
            helper_pixels, "pixel_values_latent"
        )
        helper_grid = self._require_tensor_value(
            helper_grid, "image_grid_thw_latent"
        )
        self._validate_batch_shape(input_ids, attention_mask, metadata)

        teacher.eval()
        device = self._teacher_device(teacher, input_ids.device)
        teacher_ids = input_ids.to(device)
        teacher_attention = attention_mask.to(device)
        original_pixels = original_pixels.to(device)
        original_grid = original_grid.to(device)
        helper_pixels = helper_pixels.to(
            device=device, dtype=getattr(teacher.visual, "dtype", helper_pixels.dtype)
        )
        helper_grid = helper_grid.to(device)

        text_outputs = teacher(
            input_ids=teacher_ids,
            attention_mask=teacher_attention,
            pixel_values=original_pixels,
            image_grid_thw=original_grid,
            output_hidden_states=True,
            return_dict=True,
        )
        helper_features = teacher.visual(helper_pixels, grid_thw=helper_grid)
        hidden_pre, hidden_last = self._hidden_states(text_outputs, metadata)
        input_embeddings = self._language_input_embeddings(
            teacher, text_outputs, teacher_ids
        )
        helper_images = self._split_visual_features(
            helper_features, helper_grid, teacher
        )
        original_counts = self._visual_token_counts(original_grid, teacher)

        latent_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        packed: list[torch.Tensor] = []
        selected_counts: list[tuple[int, ...]] = []
        for sample_index, sample in enumerate(metadata.samples):
            sample_ids = teacher_ids[sample_index, : sample.unpadded_length]
            sample_hidden_pre = hidden_pre[
                sample_index, : sample.unpadded_length
            ]
            sample_hidden_last = hidden_last[
                sample_index, : sample.unpadded_length
            ]
            sample_input_embeddings = input_embeddings[
                sample_index, : sample.unpadded_length
            ]
            assistant_start = self._assistant_start(sample_ids, sample)
            image_token_id = int(
                getattr(teacher.config, "image_token_id", 151655)
            )
            user_query = self._user_query(
                teacher=teacher,
                sample=sample,
                sample_ids=sample_ids,
                hidden_pre=sample_hidden_pre,
                hidden_last=sample_hidden_last,
                input_embeddings=sample_input_embeddings,
                assistant_start=assistant_start,
                image_token_id=image_token_id,
                original_grid=original_grid,
                original_counts=original_counts,
            )
            text_spans = self._assistant_text_spans(
                sample_ids, sample, assistant_start
            )
            candidates_by_block = self._sample_helper_candidates(
                helper_images, sample
            )

            previous_selected_mean = None
            block_counts = []
            for block_index, span in enumerate(sample.latent_spans):
                query_parts = [user_query]
                if text_spans[block_index]:
                    text_indices = torch.tensor(
                        text_spans[block_index],
                        device=sample_hidden_last.device,
                        dtype=torch.long,
                    )
                    query_parts.append(
                        sample_hidden_last.index_select(0, text_indices).mean(
                            dim=0, keepdim=True
                        )
                    )
                if previous_selected_mean is not None:
                    query_parts.append(previous_selected_mean)
                query = torch.stack(
                    [part.squeeze(0) for part in query_parts], dim=0
                ).mean(dim=0, keepdim=True)

                candidates = candidates_by_block[block_index]
                selected = self._select(
                    candidates,
                    query,
                    max_count=len(span.latent_positions),
                )
                packed.append(selected)
                selected_count = int(selected.shape[0])
                block_counts.append(selected_count)
                if selected_count:
                    latent_mask[
                        sample_index,
                        list(span.latent_positions[:selected_count]),
                    ] = True
                    previous_selected_mean = selected.mean(dim=0, keepdim=True)
            selected_counts.append(tuple(block_counts))

        if packed:
            latent_values = torch.cat(packed, dim=0)
        else:
            latent_values = helper_features.new_empty((0, hidden_last.shape[-1]))
        expected_count = int(latent_mask.sum().item())
        if latent_values.shape[0] != expected_count:
            raise RuntimeError(
                "row-major latent target count does not match latent mask"
            )
        return Stage1TeacherTargets(
            latent_values=latent_values,
            latent_mask=latent_mask,
            selected_counts=tuple(selected_counts),
            diagnostics=self._diagnostics(latent_values),
        )

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
    def _hidden_states(text_outputs, metadata):
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
        expected_prefix = (metadata.batch_size, metadata.sequence_length)
        if hidden_last.ndim != 3 or tuple(hidden_last.shape[:2]) != expected_prefix:
            raise ValueError("teacher hidden states must have shape [B, T, H]")
        if hidden_pre.shape != hidden_last.shape:
            raise ValueError("teacher pre-final and final hidden states must share shape")
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

    def _split_visual_features(self, features, grid_thw, teacher):
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("teacher helper visual features must have shape [N, H]")
        counts = self._visual_token_counts(grid_thw, teacher)
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
            except (AttributeError, IndexError, RuntimeError, ValueError):
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
        try:
            return teacher.model.layers[-1].self_attn
        except AttributeError:
            base = teacher.get_base_model()
            return base.model.layers[-1].self_attn

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
        grid_start, grid_end = sample.original_grid_range
        if not 0 <= grid_start <= grid_end <= original_grid.shape[0]:
            raise ValueError(
                f"source_index={sample.source_index}: invalid original grid range"
            )
        counts = original_counts[grid_start:grid_end]
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

    def _sample_helper_candidates(self, helper_images, sample):
        grid_start, grid_end = sample.helper_grid_range
        if not 0 <= grid_start <= grid_end <= len(helper_images):
            raise ValueError(
                f"source_index={sample.source_index}: invalid helper grid range"
            )
        owned_images = helper_images[grid_start:grid_end]
        if len(owned_images) != len(sample.helper_to_block):
            raise ValueError(
                f"source_index={sample.source_index}: helper ownership count mismatch"
            )
        by_block: list[list[torch.Tensor]] = [
            [] for _ in sample.latent_spans
        ]
        for image, block_index in zip(owned_images, sample.helper_to_block):
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
            return candidates[:0]
        similarities = F.cosine_similarity(
            candidates, query.expand_as(candidates), dim=-1
        )
        probabilities = torch.softmax(similarities, dim=0)
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities, descending=True
        )
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
        return candidates.index_select(0, sorted_indices[:selected_count])

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
