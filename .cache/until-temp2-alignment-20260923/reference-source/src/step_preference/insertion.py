"""Token-coordinate plans for optional visual OFF/ON insertion scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

import torch


if TYPE_CHECKING:
    from stage2.data import StepPreferenceBatchItem


@dataclass(frozen=True)
class InsertionTokenIds:
    image: int
    vision_start: int
    vision_end: int
    video: int
    latent_pad: int
    latent_start: int
    latent_end: int


@dataclass(frozen=True)
class InsertionTarget:
    sample_id: str
    step_id: int
    target_ids: torch.Tensor
    predictor_positions: torch.Tensor


@dataclass(frozen=True)
class InsertionSequence:
    input_ids: torch.Tensor
    source_positions: torch.Tensor
    targets: tuple[InsertionTarget, ...]
    latent_positions: tuple[int, ...]
    prefix_length: int
    image_grid_indices: torch.Tensor


@dataclass(frozen=True)
class InsertionPlan:
    off: InsertionSequence
    on: tuple[InsertionSequence, ...]
    branch_mode: Literal["independent", "cumulative"]


@dataclass(frozen=True)
class InsertionRequest:
    items: tuple[StepPreferenceBatchItem, ...]
    branch_mode: Literal["independent", "cumulative"] = "independent"
    on_layer_checkpoint_stride: int = 1
    stage2_debugging: bool = True


def scan_insertion_blocks(
    ids: list[int], token_ids: InsertionTokenIds, latent_size: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return half-open image and latent intervals, including their boundaries."""
    images, latents = [], []
    i = 0
    while i < len(ids):
        token = ids[i]
        if token == token_ids.video:
            raise ValueError("insertion does not support video blocks")
        if token == token_ids.vision_start:
            end = i + 1
            while end < len(ids) and ids[end] == token_ids.image:
                end += 1
            if end == i + 1 or end >= len(ids) or ids[end] != token_ids.vision_end:
                raise ValueError("insertion requires canonical image blocks")
            images.append((i, end + 1))
            i = end + 1
        elif token == token_ids.latent_start:
            end = i + latent_size + 1
            if (end >= len(ids) or ids[end] != token_ids.latent_end
                    or ids[i + 1:end] != [token_ids.latent_pad] * latent_size):
                raise ValueError("insertion requires canonical latent blocks")
            latents.append((i, end + 1))
            i = end + 1
        elif token in (token_ids.image, token_ids.vision_end,
                       token_ids.latent_pad, token_ids.latent_end):
            raise ValueError("insertion found an image or latent token outside its canonical block")
        else:
            i += 1
    return images, latents


def _require_positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_token_ids(token_ids: object) -> InsertionTokenIds:
    if not isinstance(token_ids, InsertionTokenIds):
        raise ValueError("token_ids must be InsertionTokenIds")
    values = tuple(vars(token_ids).values())
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError("insertion token IDs must be integers")
    if len(set(values)) != len(values):
        raise ValueError("insertion special token IDs must be unique")
    return token_ids


def _trim_clean_inputs(
    clean_input_ids: object, clean_attention_mask: object,
) -> torch.LongTensor:
    if not isinstance(clean_input_ids, torch.Tensor) or clean_input_ids.ndim != 2:
        raise ValueError("clean_input_ids must be a two-dimensional tensor")
    if clean_input_ids.dtype != torch.long:
        raise ValueError("clean_input_ids must use torch.long")
    if clean_input_ids.shape[0] != 1 or clean_input_ids.shape[1] == 0:
        raise ValueError("insertion requires one non-empty clean input row")
    if not isinstance(clean_attention_mask, torch.Tensor) or clean_attention_mask.ndim != 2:
        raise ValueError("clean_attention_mask must be a two-dimensional tensor")
    if clean_attention_mask.shape != clean_input_ids.shape:
        raise ValueError("clean attention mask must match clean input shape")
    if clean_attention_mask.device != clean_input_ids.device:
        raise ValueError("clean attention mask must share the clean input device")
    if clean_attention_mask.dtype not in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("clean attention mask must be boolean or integral")
    mask = clean_attention_mask[0]
    if not bool(torch.all((mask == 0) | (mask == 1))):
        raise ValueError("clean attention mask must contain only zero and one")
    length = int(mask.sum().item())
    if length == 0 or not bool(torch.all(mask[:length] == 1)) or not bool(torch.all(mask[length:] == 0)):
        raise ValueError("clean input padding must be a right-padded suffix")
    return clean_input_ids[:, :length]


def _validate_grid(
    image_grid_thw: object, *, images: Sequence[tuple[int, int]], token_ids: InsertionTokenIds,
    ids: list[int], spatial_merge_size: int, device: torch.device,
) -> None:
    if not images:
        raise ValueError("insertion requires at least one canonical image block")
    if not isinstance(image_grid_thw, torch.Tensor) or image_grid_thw.ndim != 2:
        raise ValueError("image grid must be a two-dimensional tensor")
    if image_grid_thw.dtype != torch.long:
        raise ValueError("image grid must use torch.long")
    if image_grid_thw.device != device or image_grid_thw.shape != (len(images), 3):
        raise ValueError("image grid must have one [t, h, w] row per image")
    if not bool(torch.all(image_grid_thw > 0)):
        raise ValueError("image grid values must be positive")
    for image_index, (start, end) in enumerate(images):
        t, h, w = (int(value) for value in image_grid_thw[image_index].tolist())
        if h % spatial_merge_size or w % spatial_merge_size:
            raise ValueError("image grid spatial dimensions must divide by spatial_merge_size")
        image_tokens = end - start - 2
        expected_tokens = t * (h // spatial_merge_size) * (w // spatial_merge_size)
        if image_tokens != expected_tokens:
            raise ValueError("image block token count disagrees with image grid")
        if any(ids[position] != token_ids.image for position in range(start + 1, end - 1)):
            raise ValueError("image block token count disagrees with image grid")


def _validate_items(
    items: Sequence[StepPreferenceBatchItem], *, ids: torch.LongTensor,
    latents: Sequence[tuple[int, int]], token_ids: InsertionTokenIds,
) -> tuple[StepPreferenceBatchItem, ...]:
    selected = tuple(items)
    step_ids = [item.span.step_id for item in selected]
    if step_ids != sorted(step_ids) or len(set(step_ids)) != len(step_ids):
        raise ValueError("selected step IDs must be unique and sorted")
    blocks_by_start = {start: end for start, end in latents}
    special_ids = set(vars(token_ids).values())
    clean_length = ids.shape[1]
    for item in selected:
        if not isinstance(item.sample_id, str) or not item.sample_id:
            raise ValueError("insertion target sample_id must be a non-empty string")
        span = item.span
        if not (
            0 <= span.prev_text_start < span.prev_text_end <= span.latent_start
            < span.latent_end < span.target_text_start < span.target_text_end <= clean_length
        ):
            raise ValueError("reasoning step spans must be ordered within clean input")
        if span.prev_text_end != span.latent_start - 1:
            raise ValueError("reasoning step previous text must end at the latent start marker")
        block_start = span.latent_start - 1
        if blocks_by_start.get(block_start) != span.latent_end + 1:
            raise ValueError("reasoning step latent span must match a canonical latent block")
        if span.target_text_start != span.latent_end + 1:
            raise ValueError("reasoning step target span must follow its latent block")
        targets = item.chosen_target_ids
        predictors = item.chosen_predictor_positions
        if not isinstance(targets, torch.Tensor) or targets.ndim != 1 or targets.dtype != torch.long:
            raise ValueError("chosen target ids must be a one-dimensional torch.long tensor")
        if not isinstance(predictors, torch.Tensor) or predictors.ndim != 1 or predictors.dtype != torch.long:
            raise ValueError("chosen predictor positions must be a one-dimensional torch.long tensor")
        if targets.device != ids.device or predictors.device != ids.device:
            raise ValueError("chosen targets and predictors must share the clean input device")
        if targets.numel() == 0 or predictors.numel() != targets.numel():
            raise ValueError("chosen predictor and target lengths must match and be non-empty")
        expected_predictors = torch.arange(
            span.target_text_start - 1, span.target_text_end - 1, device=ids.device, dtype=torch.long,
        )
        if not torch.equal(predictors, expected_predictors):
            raise ValueError("chosen predictor positions must be contiguous and match the target span")
        expected_targets = ids[0, span.target_text_start:span.target_text_end]
        if not torch.equal(targets, expected_targets):
            raise ValueError("chosen target ids disagree with the clean trajectory")
        if any(int(token) in special_ids for token in targets.tolist()):
            raise ValueError("preference targets cannot contain image or control tokens")
    return selected


def validate_insertion_inputs(
    *, clean_input_ids, clean_attention_mask, items, token_ids, image_grid_thw,
    spatial_merge_size, latent_size, branch_mode,
) -> tuple[
    torch.LongTensor,
    list[int],
    list[tuple[int, int]],
    list[tuple[int, int]],
    tuple[StepPreferenceBatchItem, ...],
]:
    """Validate and normalize the shared insertion-planner input contract."""
    if branch_mode not in {"independent", "cumulative"}:
        raise ValueError("branch_mode must be independent or cumulative")
    spatial_merge_size = _require_positive_int(spatial_merge_size, "spatial_merge_size")
    latent_size = _require_positive_int(latent_size, "latent_size")
    token_ids = _validate_token_ids(token_ids)
    clean_input_ids = _trim_clean_inputs(clean_input_ids, clean_attention_mask)
    device = clean_input_ids.device
    ids = clean_input_ids[0].tolist()
    images, latents = scan_insertion_blocks(ids, token_ids, latent_size)
    _validate_grid(
        image_grid_thw, images=images, token_ids=token_ids, ids=ids,
        spatial_merge_size=spatial_merge_size, device=device,
    )
    if not isinstance(items, Sequence):
        raise ValueError("items must be a sequence of StepPreferenceBatchItem objects")
    selected_items = _validate_items(
        items, ids=clean_input_ids, latents=latents, token_ids=token_ids,
    )
    return clean_input_ids, ids, images, latents, selected_items


def build_insertion_plan(
    *, clean_input_ids, clean_attention_mask, items, token_ids, image_grid_thw,
    spatial_merge_size, latent_size, branch_mode, stage2_debugging=True,
) -> InsertionPlan:
    """Build OFF and selected ON token plans from one right-padded clean row."""
    clean_input_ids, ids, images, latents, selected_items = validate_insertion_inputs(
        clean_input_ids=clean_input_ids, clean_attention_mask=clean_attention_mask,
        items=items, token_ids=token_ids, image_grid_thw=image_grid_thw,
        spatial_merge_size=spatial_merge_size, latent_size=latent_size,
        branch_mode=branch_mode,
    )
    device = clean_input_ids.device

    image_sources = [position for start, end in images for position in range(start, end)]
    image_set = set(image_sources)
    latent_bodies = {start: range(start + 1, end - 1) for start, end in latents}
    all_body = {position for body in latent_bodies.values() for position in body}
    off_sources = [position for position in range(len(ids)) if position not in image_set and position not in all_body]
    off_inverse = {old: new for new, old in enumerate(off_sources)}

    def make_sequence(selected, scored):
        selected_starts = {item.span.latent_start - 1 for item in selected}
        restored = {position for start in selected_starts for position in latent_bodies[start]}
        sources, grid_indices = [], []
        for old in range(len(ids)):
            if old in selected_starts:
                sources.extend(image_sources)
                grid_indices.extend(range(len(images)))
            if old not in image_set and (old not in all_body or old in restored):
                sources.append(old)
        inverse = {old: new for new, old in enumerate(sources) if old not in image_set}
        source_tensor = torch.tensor(sources, device=device, dtype=torch.long)
        targets = tuple(InsertionTarget(
            item.sample_id, item.span.step_id, item.chosen_target_ids.clone(),
            torch.tensor(
                [inverse[position] - 1 for position in range(
                    item.span.target_text_start, item.span.target_text_end,
                )],
                device=device, dtype=torch.long,
            ),
        ) for item in scored)
        input_ids = clean_input_ids[:, source_tensor]
        for target in targets:
            if not torch.equal(input_ids[0, target.predictor_positions + 1], target.target_ids):
                raise ValueError("transformed target ids disagree with target predictors")
        prefix = min((off_inverse[start] for start in selected_starts), default=0)
        return InsertionSequence(
            input_ids=input_ids,
            source_positions=source_tensor,
            targets=targets,
            latent_positions=tuple(inverse[position] for position in sorted(restored)),
            prefix_length=prefix,
            image_grid_indices=torch.tensor(grid_indices, device=device, dtype=torch.long),
        )

    off = make_sequence((), selected_items)
    on = (
        tuple(make_sequence((item,), (item,)) for item in selected_items)
        if branch_mode == "independent"
        else ((make_sequence(selected_items, selected_items),) if selected_items else ())
    )
    return InsertionPlan(off=off, on=on, branch_mode=branch_mode)
