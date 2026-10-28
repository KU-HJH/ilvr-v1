"""Static record codecs; live occurrence and batch coordinates are never saved."""
from __future__ import annotations

from dataclasses import asdict, fields

import torch

from peft_common import resolve_qwen2_5_vl_model

from .types import (Stage1LatentSpan, Stage1SampleMetadata, Stage1BatchMetadata,
                    Stage1TeacherTargets, Stage1RejectedLatentBlock)


STATIC_ROW_KEYS = ('input_ids', 'attention_mask', 'labels', 'image_out_mask',
                   'pixel_values', 'image_grid_thw', 'image_grid_thw_latent',
                   'pad_token_id')


def encode_static_row(row: dict, *, sample: Stage1SampleMetadata,
                      v5_blocks: tuple) -> dict:
    return {
        'kind': 'static_row', 'row': {key: row[key] for key in STATIC_ROW_KEYS},
        'latent_spans': tuple(asdict(span) for span in sample.latent_spans),
        'unpadded_length': sample.unpadded_length,
        'original_pixel_count': sample.original_pixel_range[1] - sample.original_pixel_range[0],
        'helper_pixel_count': sample.helper_pixel_range[1] - sample.helper_pixel_range[0],
        'original_grid_count': sample.original_grid_range[1] - sample.original_grid_range[0],
        'helper_grid_count': sample.helper_grid_range[1] - sample.helper_grid_range[0],
        'v5_blocks': tuple({key: value for key, value in asdict(block).items()
                            if key != 'batch_index'} for block in v5_blocks),
    }


def decode_static_row(record: dict) -> dict:
    if record.get('kind') != 'static_row':
        raise ValueError('expected a cached static row')
    row = record['row']
    length = record['unpadded_length']
    if (type(length) is not int or length <= 0 or any(
            tuple(row[key].shape) != (length,)
            for key in ('input_ids', 'attention_mask', 'labels', 'image_out_mask'))):
        raise ValueError('cached row token geometry is invalid')
    return {**record, 'row': {key: row[key] for key in STATIC_ROW_KEYS},
            'latent_spans': tuple(Stage1LatentSpan(**span) for span in record['latent_spans'])}


def capture_teacher_forward_state(teacher) -> dict:
    owner = resolve_qwen2_5_vl_model(teacher)
    rotary = getattr(owner.model, 'rotary_emb', None)
    if 'dynamic' in getattr(rotary, 'rope_type', ''):
        raise ValueError('cache replay requires non-dynamic RoPE')
    value = getattr(owner, 'rope_deltas', None)
    return {'rope_deltas': None if value is None else value.detach().clone()}


def restore_teacher_forward_state(teacher, state: dict) -> None:
    owner = resolve_qwen2_5_vl_model(teacher)
    if set(state) != {'rope_deltas'}:
        raise ValueError('unsupported cached teacher forward state')
    value = state['rope_deltas']
    owner.rope_deltas = (None if value is None else
                         value.to(device=next(owner.parameters()).device).clone())


def encode_target_sample(targets: Stage1TeacherTargets, metadata: Stage1BatchMetadata,
                         row: int, *, forward_state: dict) -> dict:
    start = sum(sum(counts) for counts in targets.selected_counts[:row])
    count = sum(targets.selected_counts[row])
    return {
        'kind': 'targets', 'latent_values': targets.latent_values[start:start + count],
        'selected_counts': targets.selected_counts[row],
        'selected_candidate_indices': (targets.selected_candidate_indices[row]
                                       if targets.selected_candidate_indices else None),
        'selection_margins': targets.selection_margins[row] if targets.selection_margins else None,
        'rejected_blocks': tuple({field.name: getattr(block, field.name)
                                 for field in fields(block) if field.name != 'batch_index'}
                                for block in targets.rejected_blocks if block.batch_index == row),
        'forward_state': forward_state,
    }


def decode_target_batch(records: tuple[dict, ...], metadata: Stage1BatchMetadata,
                         *, device: torch.device) -> Stage1TeacherTargets:
    if not records or len(records) != metadata.batch_size:
        raise ValueError('cached target rows must match batch ownership')
    mask = torch.zeros((metadata.batch_size, metadata.sequence_length), dtype=torch.bool, device=device)
    values, counts, rejected = [], [], []
    for row, (record, sample) in enumerate(zip(records, metadata.samples, strict=True)):
        row_counts = record['selected_counts']
        row_values = record['latent_values']
        if (record.get('kind') != 'targets' or len(row_counts) != len(sample.latent_spans)
                or row_values.ndim != 2 or row_values.shape[0] != sum(row_counts)):
            raise ValueError('cached target geometry does not match the sample')
        for span, count in zip(sample.latent_spans, row_counts, strict=True):
            if type(count) is not int or not 0 <= count <= len(span.latent_positions):
                raise ValueError('cached target count exceeds its latent span')
            if count:
                mask[row, list(span.latent_positions[:count])] = True
        values.append(row_values.to(device=device))
        counts.append(row_counts)
        for block in record['rejected_blocks']:
            index = block['block_index']
            if not 0 <= index < len(sample.latent_spans) or block['positions'] != sample.latent_spans[index].latent_positions:
                raise ValueError('cached reverse donor block does not match its owner')
            rejected.append(Stage1RejectedLatentBlock(**{**block, 'batch_index': row,
                                                         'values': block['values'].to(device=device)}))
    indices = tuple(record['selected_candidate_indices'] for record in records)
    margins = tuple(record['selection_margins'] for record in records)
    if any(item is None for item in indices) and not all(item is None for item in indices):
        raise ValueError('mixed target diagnostic policy')
    if any(item is None for item in margins) and not all(item is None for item in margins):
        raise ValueError('mixed target diagnostic policy')
    return Stage1TeacherTargets(torch.cat(values, dim=0), mask, tuple(counts), {},
        indices if indices[0] is not None else (), margins if margins[0] is not None else (),
        tuple(rejected))
