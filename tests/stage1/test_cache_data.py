"""Replay bypasses preprocessing but keeps current sampler/batch ownership."""
from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader

import stage1.data as data_module
from stage1.data import Stage1DataCollator
from stage1.types import Stage1SampleOccurrence
from tests.stage1.helpers import TOKENS
from tests.stage1.test_run_cache import make_cache
from tests.stage1.v5.test_metadata import SourceProcessor, source_example


def collator(cache):
    return Stage1DataCollator(SourceProcessor(), TOKENS, 3, stage1_variant='v5', run_cache=cache)


def assert_same_static(live, replay):
    for name in ('input_ids', 'labels', 'attention_mask', 'image_out_mask',
                 'pixel_values', 'image_grid_thw', 'image_grid_thw_latent'):
        assert torch.equal(live[name], replay[name]), name
    assert live['stage1_v5_metadata'] == replay['stage1_v5_metadata']
    assert 'pixel_values_latent' not in replay


def test_replay_skips_all_decode_processor_and_tokenizer_calls(tmp_path, monkeypatch):
    cache = make_cache(tmp_path)
    build = collator(cache)
    example = source_example(['reason', None, 'answer'])
    live = build([example])
    cache.phase.value = 1
    record = cache.read('preprocess', 17)
    assert 'pixel_values_latent' not in record['row']
    def forbidden(*args, **kwargs):
        pytest.fail('replay entered image/token preprocessing')
    monkeypatch.setattr(data_module, '_materialize_image', forbidden)
    monkeypatch.setattr(SourceProcessor, '__call__', forbidden)
    monkeypatch.setattr(type(build.processor.tokenizer), '__call__', forbidden)
    monkeypatch.setattr(build.processor, 'apply_chat_template', forbidden)
    replay = build([example])
    assert_same_static(live, replay)
    assert replay['stage1_cache_sources'] == (17,)


def test_padding_and_occurrence_rebuilt_after_shuffle(tmp_path):
    cache = make_cache(tmp_path)
    build = collator(cache)
    a = source_example(['reason', None, 'longer answer'], source_index=17)
    b = source_example(['short', None, 'a'], source_index=18)
    build([a, b])
    expected = build([b, a])
    cache.phase.value = 1
    replay = build([b, a])
    assert_same_static(expected, replay)
    assert expected['stage1_batch_metadata'] == replay['stage1_batch_metadata']
    occurrence = Stage1SampleOccurrence(1, 17, 0, 3456 + 17, 17, 0, 0, 0, 0)
    current = build([replace(a, occurrence=occurrence)])['stage1_batch_metadata'].samples[0]
    assert current.epoch == 1 and current.canonical_slot_id == 3473
    assert current.original_pixel_range == (0, 4)
    assert current.helper_grid_range == (0, 1)


def test_duplicates_compute_then_publish_once(tmp_path):
    cache = make_cache(tmp_path)
    build = collator(cache)
    example = source_example(['reason', None, 'answer'])
    first = build([example])
    path = cache.plan.root / 'preprocess' / '17.pt'
    original = path.read_bytes()
    # If the duplicate tried to read, phase=populate would reject it.
    second = build([example])
    assert torch.equal(first['input_ids'], second['input_ids'])
    assert second['pixel_values_latent'].numel() > 0
    assert original == path.read_bytes()


@pytest.mark.parametrize('items', [['text only'], ['reason', None, '']])
def test_empty_helper_and_empty_v5_query_rows(tmp_path, items):
    cache = make_cache(tmp_path)
    build = collator(cache)
    examples = [source_example(items)]
    live = build(examples)
    cache.phase.value = 1
    replay = build(examples)
    assert_same_static(live, replay)
    assert replay['stage1_v5_metadata'].query_keys == ()


def test_four_worker_phase_transition(tmp_path, monkeypatch):
    cache = make_cache(tmp_path)
    build = collator(cache)
    examples = [source_example(['reason', None, f'answer {i}'], source_index=i) for i in range(8)]
    original = data_module._materialize_image
    def guard(*args, **kwargs):
        if cache.phase.value == 1:
            raise AssertionError('worker used stale populate mode')
        return original(*args, **kwargs)
    monkeypatch.setattr(data_module, '_materialize_image', guard)
    loader = DataLoader(examples, batch_size=1, collate_fn=build, num_workers=4,
                        persistent_workers=True)
    try:
        first = list(loader)
        cache.validate_coverage({'preprocess': tuple(range(8))})
        cache.phase.value = 1
        second = list(loader)
        assert len(first) == len(second) == 8
        for live, replay in zip(first, second):
            assert_same_static(live, replay)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
