"""Immutable token windows for the opt-in Stage 2 objective."""

from dataclasses import dataclass

import torch

from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds

from .config import ObjectiveConfig
from .vocabulary import TextVocabulary


@dataclass(frozen=True)
class SegmentBlock:
    sample_id: str
    batch_index: int
    step_id: int
    assistant_start: int
    assistant_end: int
    pre_start: int
    pre_end: int
    start_marker: int
    end_marker: int
    post_start: int
    post_end: int
    # Original tokens crossing adjacent text items; never silently skip/refill.
    ambiguous_positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class WindowBlock:
    segment: SegmentBlock
    pre_positions: tuple[int, ...]
    post_positions: tuple[int, ...]
    post_ids: tuple[int, ...]


@dataclass(frozen=True)
class ExcludedBlock:
    sample_id: str
    batch_index: int
    step_id: int
    reason: str


@dataclass(frozen=True)
class WindowPlan:
    blocks: tuple[WindowBlock, ...]
    excluded: tuple[ExcludedBlock, ...]


def align_segments(
    *,
    input_ids: torch.Tensor,
    offsets: torch.Tensor,
    sequence_plan: tuple[dict, ...],
    assistant_bounds: tuple[int, int],
    token_ids: LatentTokenIds,
    sample_id: str,
) -> tuple[SegmentBlock, ...]:
    """Align sequence-plan text items to an unchanged tokenized trajectory."""
    if input_ids.ndim == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids[0]
    if offsets.ndim == 3 and offsets.shape[0] == 1:
        offsets = offsets[0]
    if input_ids.ndim != 1:
        raise ValueError("align_segments requires one tokenized sample")
    if offsets.shape != (input_ids.numel(), 2):
        raise ValueError("offsets must have shape [sequence, 2]")
    if (
        not isinstance(assistant_bounds, tuple)
        or len(assistant_bounds) != 2
        or any(type(value) is not int for value in assistant_bounds)
    ):
        raise TypeError("assistant_bounds must be an integer pair")
    assistant_start, assistant_end = assistant_bounds
    if not 0 <= assistant_start <= assistant_end <= input_ids.numel():
        raise ValueError(f"sample {sample_id!r}: assistant bounds are out of range")

    latent_plan_indices: list[int] = []
    for plan_index, item in enumerate(sequence_plan):
        if not isinstance(item, dict):
            raise ValueError(f"sample {sample_id!r}: sequence-plan items must be objects")
        item_type = item.get("type")
        if item_type == "text":
            if not isinstance(item.get("content"), str):
                raise ValueError(f"sample {sample_id!r}: text content must be a string")
        elif item_type == "latent":
            latent_plan_indices.append(plan_index)
        else:
            raise ValueError(
                f"sample {sample_id!r}: unsupported sequence-plan type {item_type!r}"
            )
    if not latent_plan_indices:
        return ()

    def malformed(step_id: int, detail: str) -> ValueError:
        return ValueError(f"sample {sample_id!r} step {step_id}: {detail}")

    token_values = [int(value) for value in input_ids.tolist()]
    parsed_spans: list[tuple[int, tuple[int, ...], int]] = []
    open_start: int | None = None
    pads: list[int] = []
    for position, token_id in enumerate(token_values):
        step_id = len(parsed_spans)
        if open_start is None:
            if token_id == token_ids.start_token_id:
                open_start = position
                pads = []
            elif token_id == token_ids.end_token_id:
                raise malformed(step_id, "unmatched latent end marker")
            elif token_id == token_ids.pad_token_id:
                raise malformed(step_id, "latent slot occurs outside markers")
            continue
        if token_id == token_ids.start_token_id:
            raise malformed(step_id, "nested latent start marker")
        if token_id == token_ids.pad_token_id:
            pads.append(position)
            continue
        if token_id == token_ids.end_token_id:
            if not pads:
                raise malformed(step_id, "latent block has zero slots")
            parsed_spans.append((open_start, tuple(pads), position))
            open_start = None
            pads = []
            continue
        raise malformed(step_id, "latent block contains a non-slot token")
    if open_start is not None:
        raise malformed(len(parsed_spans), "unmatched latent start marker")
    if len(parsed_spans) != len(latent_plan_indices):
        step_id = min(len(parsed_spans), len(latent_plan_indices))
        raise malformed(step_id, "token and sequence-plan latent counts disagree")
    for step_id, (start, _, end) in enumerate(parsed_spans):
        if not assistant_start <= start < end < assistant_end:
            raise malformed(step_id, "latent markers lie outside assistant bounds")

    latent_start_text = "<|latent_start|>"
    latent_pad_text = "<|latent_pad|>"
    latent_end_text = "<|latent_end|>"
    relative_items: list[tuple[int, int]] = []
    relative_markers: list[tuple[tuple[int, int], ...]] = []
    cursor = 0
    latent_cursor = 0
    for item in sequence_plan:
        item_start = cursor
        if item["type"] == "text":
            cursor += len(item["content"])
            relative_markers.append(())
        else:
            _, pad_positions, _ = parsed_spans[latent_cursor]
            marker_bounds: list[tuple[int, int]] = []
            for rendered in (
                latent_start_text,
                *((latent_pad_text,) * len(pad_positions)),
                latent_end_text,
            ):
                marker_bounds.append((cursor, cursor + len(rendered)))
                cursor += len(rendered)
            relative_markers.append(tuple(marker_bounds))
            latent_cursor += 1
        relative_items.append((item_start, cursor))

    first_plan_index = latent_plan_indices[0]
    first_start_position = parsed_spans[0][0]
    first_offset = tuple(int(value) for value in offsets[first_start_position].tolist())
    if first_offset[1] <= first_offset[0]:
        raise malformed(0, "fast-tokenizer marker offsets are unavailable")
    translation = first_offset[0] - relative_items[first_plan_index][0]

    for step_id, (plan_index, parsed_span) in enumerate(
        zip(latent_plan_indices, parsed_spans)
    ):
        start_position, pad_positions, end_position = parsed_span
        token_positions = (start_position, *pad_positions, end_position)
        expected_bounds = relative_markers[plan_index]
        for position, relative_bounds in zip(token_positions, expected_bounds):
            actual = tuple(int(value) for value in offsets[position].tolist())
            expected = tuple(value + translation for value in relative_bounds)
            if actual != expected:
                raise malformed(
                    step_id,
                    f"marker offset {actual} does not match translated bound {expected}",
                )

    translated_items = tuple(
        (start + translation, end + translation) for start, end in relative_items
    )
    required_boundaries = {
        boundary
        for bounds in translated_items
        for boundary in bounds
    }
    text_join_boundaries = {
        translated_items[index][0]
        for index in range(1, len(sequence_plan))
        if sequence_plan[index - 1]["type"] == sequence_plan[index]["type"] == "text"
    }
    # Latent and assistant boundaries remain strict. Only joins between original
    # text items can be deferred until the actual scoring windows are known.
    strict_boundaries = (required_boundaries - text_join_boundaries) | {
        boundary for index in latent_plan_indices for boundary in translated_items[index]
    } | {translated_items[0][0], translated_items[-1][1]}
    ambiguous_positions: list[int] = []
    for position in range(assistant_start, assistant_end):
        offset_start, offset_end = (
            int(value) for value in offsets[position].tolist()
        )
        if offset_end < offset_start:
            raise ValueError(f"sample {sample_id!r}: invalid tokenizer offset ordering")
        if offset_end == offset_start:
            continue
        if any(offset_start < boundary < offset_end for boundary in strict_boundaries):
            raise ValueError(
                f"sample {sample_id!r}: token {position} straddles a required text-item boundary"
            )

        if any(offset_start < boundary < offset_end for boundary in text_join_boundaries):
            ambiguous_positions.append(position)

    text_bounds: dict[int, tuple[int, int]] = {}
    structural_positions = {
        position
        for start, pads, end in parsed_spans
        for position in (start, *pads, end)
    }
    for plan_index, item in enumerate(sequence_plan):
        if item["type"] != "text":
            continue
        char_start, char_end = translated_items[plan_index]
        positions = tuple(
            position
            for position in range(assistant_start, assistant_end)
            if position not in structural_positions
            and int(offsets[position, 1]) > int(offsets[position, 0])
            and char_start < char_end
            and int(offsets[position, 0]) < char_end
            and int(offsets[position, 1]) > char_start
        )
        if item["content"] and not positions:
            raise ValueError(
                f"sample {sample_id!r}: non-empty text item {plan_index} has no usable offsets"
            )
        if positions and positions != tuple(range(positions[0], positions[-1] + 1)):
            raise ValueError(
                f"sample {sample_id!r}: text item {plan_index} is not token-contiguous"
            )
        text_bounds[plan_index] = (
            (positions[0], positions[-1] + 1)
            if positions
            else (0, 0)
        )

    segments: list[SegmentBlock] = []
    for step_id, (plan_index, parsed_span) in enumerate(
        zip(latent_plan_indices, parsed_spans)
    ):
        start_marker, _, end_marker = parsed_span
        pre_start = pre_end = start_marker
        if plan_index > 0 and sequence_plan[plan_index - 1]["type"] == "text":
            pre_start, pre_end = text_bounds[plan_index - 1]
            if pre_start == pre_end:
                pre_start = pre_end = start_marker
            elif pre_end != start_marker:
                raise malformed(step_id, "pre text is not adjacent to latent start")
        post_start = post_end = end_marker + 1
        if (
            plan_index + 1 < len(sequence_plan)
            and sequence_plan[plan_index + 1]["type"] == "text"
        ):
            post_start, post_end = text_bounds[plan_index + 1]
            if post_start == post_end:
                post_start = post_end = end_marker + 1
            elif post_start != end_marker + 1:
                raise malformed(step_id, "post text is not adjacent to latent end")
        segments.append(
            SegmentBlock(
                sample_id=sample_id,
                batch_index=0,
                step_id=step_id,
                assistant_start=assistant_start,
                assistant_end=assistant_end,
                pre_start=pre_start,
                pre_end=pre_end,
                start_marker=start_marker,
                end_marker=end_marker,
                post_start=post_start,
                post_end=post_end,
                ambiguous_positions=tuple(ambiguous_positions),
            )
        )
    return tuple(segments)


def _validate_window_inputs(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    padding: torch.Tensor,
) -> tuple[int, int]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if labels.shape != input_ids.shape or padding.shape != input_ids.shape:
        raise ValueError("labels and padding must match input_ids")
    return input_ids.shape


def _validate_segment(
    segment: SegmentBlock,
    *,
    batch_size: int,
    sequence_length: int,
    input_ids: torch.Tensor,
    text_id_set: set[int],
) -> None:
    where = f"sample {segment.sample_id!r} step {segment.step_id}"
    if not 0 <= segment.batch_index < batch_size:
        raise ValueError(f"{where}: batch index is out of bounds")
    positions = (
        segment.assistant_start,
        segment.pre_start,
        segment.pre_end,
        segment.start_marker,
        segment.end_marker,
        segment.post_start,
        segment.post_end,
        segment.assistant_end,
    )
    if any(type(position) is not int for position in positions):
        raise TypeError(f"{where}: segment positions must be integers")
    if not (
        0 <= segment.assistant_start
        <= segment.pre_start
        <= segment.pre_end
        == segment.start_marker
        < segment.end_marker
        and segment.post_start == segment.end_marker + 1
        <= segment.post_end
        <= segment.assistant_end
        <= sequence_length
    ):
        raise ValueError(f"{where}: invalid segment ordering or assistant bounds")
    row = input_ids[segment.batch_index]
    structural = [
        int(row[position])
        for position in range(segment.start_marker, segment.end_marker + 1)
    ]
    interior = structural[1:-1]
    if (
        structural[0] == structural[-1]
        or not interior
        or len(set(interior)) != 1
        or interior[0] in {structural[0], structural[-1]}
        or any(token_id in text_id_set for token_id in structural)
    ):
        raise ValueError(f"{where}: malformed latent markers")


def build_windows(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    padding: torch.Tensor,
    segments: tuple[SegmentBlock, ...],
    vocabulary: TextVocabulary,
    config: ObjectiveConfig,
) -> WindowPlan:
    """Select fixed pre/post candidates, then remove cross-role overlap."""
    batch_size, sequence_length = _validate_window_inputs(input_ids, labels, padding)
    text_id_set = set(vocabulary.text_ids)
    candidates: list[tuple[SegmentBlock, tuple[int, ...], tuple[tuple[int, int], ...]]] = []
    initial_excluded: list[ExcludedBlock] = []
    for segment in segments:
        _validate_segment(
            segment,
            batch_size=batch_size,
            sequence_length=sequence_length,
            input_ids=input_ids,
            text_id_set=text_id_set,
        )
        b = segment.batch_index
        pre = tuple(
            p
            for p in range(segment.pre_start, segment.pre_end)
            if bool(padding[b, p]) and int(input_ids[b, p]) in text_id_set
        )[-config.pre_window :]
        targets = tuple(
            i
            for i in range(segment.post_start, segment.post_end)
            if bool(padding[b, i])
            and int(labels[b, i]) != -100
            and int(input_ids[b, i]) in text_id_set
        )[: config.post_window]
        # Check fixed candidates before overlap removal; ambiguity must never
        # shift the window or replace a target with a later token.
        selected = set(pre) | set(targets) | {i - 1 for i in targets}
        ambiguous = selected.intersection(segment.ambiguous_positions)
        if ambiguous:
            raise ValueError(
                f"sample {segment.sample_id!r} step {segment.step_id}: token "
                f"{min(ambiguous)} straddles a required text-item boundary in a selected window"
            )
        post_pairs = tuple((i - 1, int(input_ids[b, i])) for i in targets)
        if not pre:
            initial_excluded.append(
                ExcludedBlock(segment.sample_id, b, segment.step_id, "empty_pre")
            )
        if not post_pairs:
            initial_excluded.append(
                ExcludedBlock(segment.sample_id, b, segment.step_id, "empty_post")
            )
        candidates.append((segment, pre, post_pairs))

    pre_by_batch: dict[int, set[int]] = {}
    post_by_batch: dict[int, set[int]] = {}
    for segment, pre, post_pairs in candidates:
        pre_by_batch.setdefault(segment.batch_index, set()).update(pre)
        post_by_batch.setdefault(segment.batch_index, set()).update(
            position for position, _ in post_pairs
        )
    overlap_by_batch = {
        batch_index: positions & post_by_batch.get(batch_index, set())
        for batch_index, positions in pre_by_batch.items()
    }

    blocks: list[WindowBlock] = []
    excluded = list(initial_excluded)
    for segment, initial_pre, initial_post_pairs in candidates:
        overlap = overlap_by_batch.get(segment.batch_index, set())
        pre = tuple(position for position in initial_pre if position not in overlap)
        post_pairs = tuple(
            pair for pair in initial_post_pairs if pair[0] not in overlap
        )
        if initial_pre and not pre:
            excluded.append(
                ExcludedBlock(
                    segment.sample_id,
                    segment.batch_index,
                    segment.step_id,
                    "overlap_empty_pre",
                )
            )
        if initial_post_pairs and not post_pairs:
            excluded.append(
                ExcludedBlock(
                    segment.sample_id,
                    segment.batch_index,
                    segment.step_id,
                    "overlap_empty_post",
                )
            )
        if pre and post_pairs:
            blocks.append(
                WindowBlock(
                    segment=segment,
                    pre_positions=pre,
                    post_positions=tuple(position for position, _ in post_pairs),
                    post_ids=tuple(token_id for _, token_id in post_pairs),
                )
            )
    return WindowPlan(tuple(blocks), tuple(excluded))


def score_mask(
    plan: WindowPlan,
    input_ids: torch.Tensor,
    *,
    include_pre: bool,
    include_post: bool,
) -> torch.Tensor:
    """Return the union of requested predictor positions as a boolean mask."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    result = torch.zeros_like(input_ids, dtype=torch.bool)
    for block in plan.blocks:
        positions: tuple[int, ...] = ()
        if include_pre:
            positions += block.pre_positions
        if include_post:
            positions += block.post_positions
        if positions:
            result[block.segment.batch_index, list(positions)] = True
    return result
