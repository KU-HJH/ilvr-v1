"""Static record codecs; live occurrence and batch coordinates are never saved."""
from __future__ import annotations

from dataclasses import asdict

from .types import Stage1LatentSpan, Stage1SampleMetadata


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
