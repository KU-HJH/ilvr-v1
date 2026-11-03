"""Source ownership and detached helper snapshots, without model execution."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from transformers.models.qwen2_5_vl.stage1_prefix_conditioning import BlockKey, QueryKey
from transformers.models.qwen2_5_vl.stage1_prefix_trace import TraceRequest
from .prefix_mask_config import PrefixConditioningConfig
from .types import Stage1BatchMetadata, Stage1LatentSpan, Stage1TeacherTargets
from .vision_gain import Stage1GainPlan

if TYPE_CHECKING:
    from .data import PreparedStage1Example


@dataclass(frozen=True)
class ImageGroup:
    image_index: int
    group_index: int
    positions: tuple[int, ...]


@dataclass(frozen=True)
class BlockSources:
    key: BlockKey
    gt_queries: tuple[QueryKey, ...]
    text_keys: tuple[int, ...]
    image_groups: tuple[ImageGroup, ...]
    selected_latent_positions: tuple[int, ...]


@dataclass(frozen=True)
class HelperSnapshot:
    latent_values: Tensor
    latent_mask: Tensor
    sources: tuple[BlockSources, ...]
    gt_queries: tuple[QueryKey, ...]
    teacher_version: int
    parameter_version: int
    snapshot_id: str


def align_stage1_sources(tokenizer: object, row: Tensor, example: PreparedStage1Example,
                         spans: tuple[Stage1LatentSpan, ...], original_grid: Tensor,
                         spatial_merge_size: int) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Map complete assistant items and per-image runs using lossless offsets."""
    from .data import align_stage1_text_items

    # Preserve and reuse the established strict latent/item boundary contract.
    align_stage1_text_items(tokenizer, row, example, spans)
    text = tokenizer.decode(row.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors='pt')
    if not torch.equal(encoded['input_ids'][0].to(row.device), row):
        raise ValueError('source offset roundtrip changed token IDs')
    offsets = encoded['offset_mapping'][0].tolist()
    assistants = [turn for turn in example.conversation if turn.get('role') == 'assistant']
    if len(assistants) != 1:
        raise ValueError('source mapping requires one assistant trajectory')
    content = assistants[0]['content']
    bounds, cursor, latent_index = [], 0, 0
    first_latent_offset = None
    for item in content:
        start = cursor
        if item['type'] == 'text':
            cursor += len(item['text'])
            bounds.append((start, cursor))
        elif item['type'] == 'image':
            if latent_index >= len(spans):
                raise ValueError('source latent count mismatch')
            if first_latent_offset is None:
                first_latent_offset = start
            cursor += len('<|latent_start|>') + len(spans[latent_index].latent_positions) * len('<|latent_pad|>') + len('<|latent_end|>')
            latent_index += 1
        else:
            raise ValueError('unsupported assistant source item')
    if latent_index != len(spans):
        raise ValueError('source latent count mismatch')
    if spans:
        translation = offsets[spans[0].start][0] - first_latent_offset
    else:
        marker = '<|im_start|>assistant\n'
        if text.count(marker) != 1:
            raise ValueError('source mapping requires an unambiguous assistant boundary')
        translation = text.index(marker) + len(marker)
    special = set(tokenizer.all_special_ids)
    reasoning = tuple(p for p, (a, b) in enumerate(offsets)
        if row[p].item() not in special and a < b
        and any(translation + start <= a < b <= translation + end for start, end in bounds))

    if type(spatial_merge_size) is not int or spatial_merge_size <= 0:
        raise ValueError('image spatial_merge_size must be positive')
    if original_grid.shape != (len(example.original_image_paths), 3):
        raise ValueError('original image grid count/shape mismatch')
    image_id = int(tokenizer.convert_tokens_to_ids('<|image_pad|>'))
    positions = [p for p, value in enumerate(row.tolist()) if value == image_id]
    images, offset = [], 0
    for grid in original_grid.tolist():
        if any(type(v) is not int or v <= 0 for v in grid) or grid[1] % spatial_merge_size or grid[2] % spatial_merge_size:
            raise ValueError('invalid original image merged grid')
        count = grid[0] * grid[1] * grid[2] // spatial_merge_size ** 2
        owned = tuple(positions[offset:offset + count])
        if len(owned) != count or owned != tuple(range(owned[0], owned[0] + count)):
            raise ValueError('original image token run does not match merged grid')
        images.append(owned)
        offset += count
    if offset != len(positions):
        raise ValueError('original image token count does not match merged grid')
    return reasoning, tuple(images)


def _positions(positions: tuple[int, ...], length: int, name: str) -> None:
    if any(type(p) is not int or not 0 <= p < length for p in positions) or tuple(sorted(set(positions))) != positions:
        raise ValueError(f'invalid {name} positions')


def build_helper_snapshot(targets: Stage1TeacherTargets, metadata: Stage1BatchMetadata,
                          gain_plan: Stage1GainPlan, *, config: PrefixConditioningConfig,
                          teacher_version: int, parameter_version: int) -> HelperSnapshot:
    """Own one compact tensor copy and witness; never regenerate helper targets."""
    for version in (teacher_version, parameter_version):
        if type(version) is not int or version < 0:
            raise ValueError('snapshot versions must be nonnegative integers')
    mask, values = targets.latent_mask, targets.latent_values
    if len(metadata.samples) != metadata.batch_size or mask.shape != (metadata.batch_size, metadata.sequence_length):
        raise ValueError('snapshot batch/mask shape mismatch')
    if mask.dtype != torch.bool or values.ndim != 2 or not values.is_floating_point() or values.shape[0] != int(mask.sum()):
        raise ValueError('snapshot latent values/mask shape or dtype mismatch')
    if not bool(torch.isfinite(values).all()):
        raise ValueError('snapshot latent values must be finite')
    if len(targets.selected_counts) != metadata.batch_size:
        raise ValueError('selected_counts batch mismatch')
    selected_by_block, groups_by_row = {}, {}
    for row, sample in enumerate(metadata.samples):
        if sample.reasoning_positions is None or sample.original_image_positions is None:
            raise ValueError('missing prefix source metadata')
        if not 0 <= sample.unpadded_length <= metadata.sequence_length:
            raise ValueError('source sample length out of bounds')
        _positions(sample.reasoning_positions, sample.unpadded_length, 'reasoning')
        if len(sample.original_image_positions) != sample.original_grid_range[1] - sample.original_grid_range[0]:
            raise ValueError('source image ownership count mismatch')
        groups, image_owned, latent_owned = [], set(), set()
        for image_index, positions in enumerate(sample.original_image_positions):
            _positions(positions, sample.unpadded_length, 'image')
            if not positions or image_owned.intersection(positions):
                raise ValueError('invalid image ownership')
            image_owned.update(positions)
            count = min(config.groups_per_initial_image, len(positions))
            quotient, remainder = divmod(len(positions), count)
            offset = 0
            for group_index in range(count):
                size = quotient + (group_index < remainder)
                groups.append(ImageGroup(image_index, group_index, positions[offset:offset + size]))
                offset += size
        if len(targets.selected_counts[row]) != len(sample.latent_spans):
            raise ValueError('selected_counts block mismatch')
        for block_index, span in enumerate(sample.latent_spans):
            _positions(span.latent_positions, sample.unpadded_length, 'latent')
            if not 0 <= span.start < span.end < sample.unpadded_length or any(not span.start < p < span.end for p in span.latent_positions):
                raise ValueError('invalid latent boundaries')
            body = set(range(span.start, span.end + 1))
            if latent_owned.intersection(body):
                raise ValueError('overlapping latent ownership')
            latent_owned.update(body)
            selected = tuple(p for p in span.latent_positions if bool(mask[row, p]))
            count = targets.selected_counts[row][block_index]
            if type(count) is not int or count != len(selected):
                raise ValueError('selected_counts disagrees with latent mask')
            selected_by_block[row, block_index] = selected
        actual = set(mask[row].nonzero(as_tuple=False).flatten().tolist())
        expected = {p for (r, _), positions in selected_by_block.items() if r == row for p in positions}
        if actual != expected:
            raise ValueError('latent mask selects positions outside latent ownership')
        if image_owned.intersection(latent_owned) or set(sample.reasoning_positions).intersection(image_owned | latent_owned | set(sample.non_text_positions)):
            raise ValueError('overlapping source ownership')
        groups_by_row[row] = tuple(groups)
    sources, queries, block_keys = [], set(), set()
    for block in gain_plan.blocks:
        key = (block.batch_index, block.block_index)
        if key not in selected_by_block or key in block_keys:
            raise ValueError('invalid or overlapping block ownership')
        block_keys.add(key)
        sample = metadata.samples[block.batch_index]
        span = sample.latent_spans[block.block_index]
        if (block.start_marker, block.end_marker) != (span.start, span.end) or block.source_index != sample.source_index:
            raise ValueError('gain block source ownership mismatch')
        _positions(block.target_positions, sample.unpadded_length, 'GT target')
        if not block.target_positions or any(p <= span.end for p in block.target_positions):
            raise ValueError('invalid GT target ownership')
        gt = tuple((block.batch_index, p - 1) for p in block.target_positions)
        if queries.intersection(gt):
            raise ValueError('overlapping GT query ownership')
        queries.update(gt)
        sources.append(BlockSources(key, gt,
            tuple(p for p in sample.reasoning_positions if p < span.start),
            tuple(group for group in groups_by_row[block.batch_index] if all(p < span.start for p in group.positions)),
            selected_by_block[key]))
    owned_values, owned_mask = values.detach().clone(), mask.detach().clone()
    # Hash only compact helper tensors and small metadata, never model weights.
    witness = hashlib.sha256(repr((tuple(sources), teacher_version, parameter_version,
        tuple(values.shape), str(values.dtype), tuple(mask.shape))).encode())
    for tensor in (owned_values, owned_mask):
        witness.update(tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return HelperSnapshot(owned_values, owned_mask, tuple(sources), tuple(sorted(queries)),
                          teacher_version, parameter_version, witness.hexdigest())


def build_trace_request(snapshot: HelperSnapshot) -> TraceRequest:
    requested: dict[QueryKey, set[int]] = {}
    owners = set()
    for source in snapshot.sources:
        images = {p for group in source.image_groups for p in group.positions}
        for query in source.gt_queries:
            if query in owners:
                raise ValueError('overlapping GT query ownership')
            owners.add(query)
            requested.setdefault(query, set()).update(images | set(source.text_keys))
        for position in source.selected_latent_positions:
            requested.setdefault((source.key[0], position - 1), set()).update(images)
    if tuple(sorted(owners)) != snapshot.gt_queries:
        raise ValueError('snapshot GT ownership mismatch')
    queries = tuple(sorted(requested))
    return TraceRequest(queries, tuple(tuple(sorted(requested[q])) for q in queries))
