"""Single-image ON/OFF coordinate plans for latent-gated preference scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from transformers.models.qwen2_5_vl.stage2_gated_attention import GatedVisibility
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentSpan,
    normalize_stage2_rollout_backend,
    stage2_prefix_dropout_semantics,
)

from step_preference.insertion import (
    InsertionSequence,
    InsertionTarget,
    InsertionTokenIds,
    validate_insertion_inputs,
)

if TYPE_CHECKING:
    from stage2.data import StepPreferenceBatchItem


@dataclass(frozen=True)
class GatedExecutionIdentity:
    """Training-only semantics that distinguish one gated execution run."""

    mask_policy_version: int
    visibility_semantics: str
    image_placement: str
    off_policy: str
    clean_backend: str
    on_backend: str
    prefix_dropout_semantics: str
    native_checkpointing: bool
    latent_checkpointing: bool
    cache_checkpointing: bool
    training_only: bool


def gated_execution_identity(
    *,
    rollout_backend: str,
    native_checkpointing: bool,
    latent_checkpointing: bool,
    cache_checkpointing: bool,
    ablation: str = "attention_mask",
) -> GatedExecutionIdentity:
    """Build the validated semantic identity for gated clean/ON execution."""
    backend = normalize_stage2_rollout_backend(rollout_backend)
    checkpointing = {
        "native_checkpointing": native_checkpointing,
        "latent_checkpointing": latent_checkpointing,
        "cache_checkpointing": cache_checkpointing,
    }
    for name, value in checkpointing.items():
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if native_checkpointing:
        raise ValueError("single_image_latent_gate requires native checkpointing disabled")
    if backend == "differentiable_cache":
        if latent_checkpointing:
            raise ValueError(
                "differentiable_cache requires latent checkpointing disabled"
            )
    else:
        if cache_checkpointing:
            raise ValueError("cache checkpointing requires differentiable_cache")
    return GatedExecutionIdentity(
        mask_policy_version=1,
        visibility_semantics="latent_positions_A",
        image_placement="original_once",
        off_policy=gated_off_policy(ablation),
        clean_backend=backend,
        on_backend=backend,
        prefix_dropout_semantics=stage2_prefix_dropout_semantics(backend),
        native_checkpointing=native_checkpointing,
        latent_checkpointing=latent_checkpointing,
        cache_checkpointing=cache_checkpointing,
        training_only=True,
    )


@dataclass(frozen=True)
class GatedRequest:
    items: tuple[StepPreferenceBatchItem, ...]
    branch_mode: str = "independent"
    ablation: str = "attention_mask"
    stage2_debugging: bool = False
    compute_entropy: bool = True


@dataclass(frozen=True)
class GatedSequence:
    sequence: InsertionSequence
    latent_spans: tuple[LatentSpan, ...]
    visibility: GatedVisibility


@dataclass(frozen=True)
class GatedPlan:
    off: InsertionSequence
    on: tuple[GatedSequence, ...]
    branch_mode: Literal["independent", "cumulative"]


def build_gated_plan(
    *, clean_input_ids, clean_attention_mask, items, token_ids: InsertionTokenIds, image_grid_thw,
    spatial_merge_size, latent_size, branch_mode, ablation="attention_mask",
) -> GatedPlan:
    """Build configured OFF and single-copy-image ON trajectories from clean IDs."""
    (
        clean_input_ids,
        ids,
        images,
        latents,
        selected_items,
    ) = validate_insertion_inputs(
        clean_input_ids=clean_input_ids,
        clean_attention_mask=clean_attention_mask,
        items=items,
        token_ids=token_ids,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=spatial_merge_size,
        latent_size=latent_size,
        branch_mode=branch_mode,
    )
    device = clean_input_ids.device
    image_sources = [position for start, end in images for position in range(start, end)]
    image_set = set(image_sources)
    latent_bodies = {start: range(start + 1, end - 1) for start, end in latents}
    all_body = {position for body in latent_bodies.values() for position in body}
    all_blocks = {position for start, end in latents for position in range(start, end)}
    off_policy = gated_off_policy(ablation)
    off_latents = all_body if ablation == "attention_mask" else all_blocks
    keep_off_images = off_policy == "keep_images_delete_full_latent_blocks"

    def make_targets(scored, inverse):
        targets = tuple(
            InsertionTarget(
                item.sample_id,
                item.span.step_id,
                item.chosen_target_ids.clone(),
                torch.tensor(
                    [inverse[position] - 1 for position in range(
                        item.span.target_text_start, item.span.target_text_end,
                    )],
                    device=device,
                    dtype=torch.long,
                ),
            )
            for item in scored
        )
        return targets

    def make_off():
        sources = [
            position for position in range(len(ids))
            if (keep_off_images or position not in image_set)
            and position not in off_latents
        ]
        inverse = {old: new for new, old in enumerate(sources)}
        source_tensor = torch.tensor(sources, device=device, dtype=torch.long)
        sequence = InsertionSequence(
            input_ids=clean_input_ids[:, source_tensor],
            source_positions=source_tensor,
            targets=make_targets(selected_items, inverse),
            latent_positions=(),
            prefix_length=0,
            image_grid_indices=(
                torch.arange(len(images), device=device, dtype=torch.long)
                if keep_off_images
                else torch.empty(0, device=device, dtype=torch.long)
            ),
        )
        _check_target_alignment(sequence)
        return sequence

    def make_on(selected, scored):
        selected_starts = {item.span.latent_start - 1 for item in selected}
        restored = {position for start in selected_starts for position in latent_bodies[start]}
        sources = [
            position for position in range(len(ids))
            if position not in all_body or position in restored
        ]
        inverse = {old: new for new, old in enumerate(sources)}
        active_starts = tuple(sorted(inverse[start] for start in selected_starts))
        spans = tuple(
            LatentSpan(
                start,
                tuple(range(start + 1, start + 1 + latent_size)),
                start + 1 + latent_size,
            )
            for start in active_starts
        )
        producers = tuple(
            position
            for span in spans
            for position in (span.start_position, *span.latent_positions[:-1])
        )
        source_tensor = torch.tensor(sources, device=device, dtype=torch.long)
        sequence = InsertionSequence(
            input_ids=clean_input_ids[:, source_tensor],
            source_positions=source_tensor,
            targets=make_targets(scored, inverse),
            latent_positions=tuple(sorted(inverse[position] for position in restored)),
            prefix_length=0,
            image_grid_indices=torch.arange(len(images), device=device, dtype=torch.long),
        )
        _check_target_alignment(sequence)
        return GatedSequence(
            sequence=sequence,
            latent_spans=spans,
            visibility=GatedVisibility(
                image_positions=tuple(inverse[position] for position in image_sources),
                producer_positions=producers,
                active_start_positions=active_starts,
                sequence_length=len(sources),
            ),
        )

    off = make_off()
    on = (
        tuple(make_on((item,), (item,)) for item in selected_items)
        if branch_mode == "independent"
        else ((make_on(selected_items, selected_items),) if selected_items else ())
    )
    return GatedPlan(off=off, on=on, branch_mode=branch_mode)


def gated_off_policy(ablation: str) -> str:
    policies = {
        "attention_mask": "delete_images_and_latent_bodies_keep_boundaries",
        "remove_full_visual_block": "delete_images_and_full_latent_blocks",
        "remove_full_latent_block_keep_image": "keep_images_delete_full_latent_blocks",
    }
    try:
        return policies[ablation]
    except KeyError as error:
        raise ValueError(
            "unsupported gated ablation"
        ) from error


def _check_target_alignment(sequence: InsertionSequence) -> None:
    for target in sequence.targets:
        if not torch.equal(
            sequence.input_ids[0, target.predictor_positions + 1], target.target_ids,
        ):
            raise ValueError("transformed target ids disagree with target predictors")
