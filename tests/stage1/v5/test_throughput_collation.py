"""Catch repeated processing, raw-patch mis-splits, and lost validation/identity."""
from dataclasses import replace
from unittest.mock import patch

from PIL import Image
import pytest
import torch

from tests.stage1.v5.collation_fixtures import (
    CountingProcessor, PixelProcessor, assert_batches_equal, examples_at,
    make_collator, with_occurrences,
)


def test_v5_materializes_each_image_once_and_processes_each_sample_once(tmp_path):
    examples = examples_at(tmp_path)
    processor = CountingProcessor(PixelProcessor())
    with patch.object(Image, 'open', wraps=Image.open) as opened:
        batch = make_collator(processor)(examples)
    assert processor.calls == 3
    assert processor.image_counts == [4, 1, 2]
    assert opened.call_count == 7
    assert batch['image_grid_thw'].tolist() == [[1, 6, 4], [1, 4, 8], [1, 4, 6], [1, 8, 4]]
    assert batch['pixel_values'].shape == (112, 1176)
    assert batch['pixel_values_latent'].shape == (136, 1176)


@pytest.mark.parametrize('occurrences', [False, True])
@pytest.mark.parametrize('subset', [(0,), (1,), (2,), (0, 1, 2)])
def test_complete_outputs_match_frozen_three_call_oracle(tmp_path, occurrences, subset):
    examples = examples_at(tmp_path)
    if occurrences:
        examples = with_occurrences(examples)
    examples = [examples[i] for i in subset]
    expected = make_collator(PixelProcessor(), original=True)(examples)
    actual = make_collator(PixelProcessor())(examples)
    assert_batches_equal(actual, expected)
    if subset == (1,):
        assert actual['pixel_values_latent'].shape == (0, 0)
        assert actual['pixel_values_latent'].dtype == torch.float32


def test_standard_keeps_three_calls_and_image_materialization(tmp_path):
    examples = examples_at(tmp_path)
    processor = CountingProcessor(PixelProcessor())
    with patch.object(Image, 'open', wraps=Image.open) as opened:
        actual = make_collator(processor, variant='standard')(examples)
    assert_batches_equal(actual, make_collator(PixelProcessor(), original=True, variant='standard')(examples))
    assert processor.calls == 9
    assert opened.call_count == 14


@pytest.mark.parametrize('damage', ['grid_count', 'grid_shape', 'pixels_short', 'pixels_scalar', 'missing_pixels', 'token_mask'])
def test_v5_rejects_malformed_processor_outputs(tmp_path, damage):
    class DamagedProcessor(PixelProcessor):
        def __call__(self, **kwargs):
            batch = super().__call__(**kwargs)
            if damage == 'grid_count':
                batch['image_grid_thw'] = batch['image_grid_thw'][:-1]
            elif damage == 'grid_shape':
                batch['image_grid_thw'] = batch['image_grid_thw'][:, :2]
            elif damage == 'pixels_short':
                batch['pixel_values'] = batch['pixel_values'][:-1]
            elif damage == 'pixels_scalar':
                batch['pixel_values'] = torch.tensor(1.)
            elif damage == 'missing_pixels':
                del batch['pixel_values']
            else:
                batch['attention_mask'] = batch['attention_mask'][:, :-1]
            return batch
    with pytest.raises(ValueError, match='source_index=0.*observed.*expected'):
        make_collator(DamagedProcessor())([examples_at(tmp_path)[0]])


def test_v5_rejects_duplicate_source_and_occurrence_ownership(tmp_path):
    examples = examples_at(tmp_path)
    collator = make_collator(PixelProcessor())
    with pytest.raises(ValueError, match='duplicate source_index'):
        collator([examples[0], examples[0]])
    repeated = with_occurrences(examples)[0]
    with pytest.raises(ValueError, match='duplicate canonical_slot_id'):
        collator([repeated, repeated])
    with pytest.raises(ValueError, match='mixed ordinary/equivalence'):
        collator([repeated, examples[1]])
    with pytest.raises(ValueError, match='matching source provenance'):
        collator([replace(repeated, source_index=2)])


def test_v5_unreadable_helper_keeps_source_provenance(tmp_path):
    example = examples_at(tmp_path)[0]
    example = replace(example, helper_image_paths=(str(tmp_path / 'missing.png'), *example.helper_image_paths[1:]))
    with pytest.raises(ValueError, match='source_index=0.*unreadable helper image'):
        make_collator(PixelProcessor())([example])
