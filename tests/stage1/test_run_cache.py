"""Lossless records must survive replay without aliasing mutable training data."""
import importlib
import multiprocessing

import pytest
import torch


def cache_api():
    assert importlib.util.find_spec('stage1.run_cache'), 'run-scoped cache is missing'
    return importlib.import_module('stage1.run_cache')


def make_cache(tmp_path, *, helper='targets', reference=True, identity=None):
    api = cache_api()
    plan = api.Stage1CachePlan('trial', tmp_path / 'trial', helper, reference,
                              identity or {'source': 'fixed-example', 'schema': 1})
    return api.Stage1RunCache(plan, multiprocessing.Value('i', 0))


def test_lossless_roundtrip_and_no_backing_mutation(tmp_path):
    cache = make_cache(tmp_path)
    record = {'fp32': torch.tensor([1.125, -2.5]),
              'bf16': torch.tensor([[3.25]], dtype=torch.bfloat16),
              'ids': torch.tensor([9, 5]), 'positions': (2, 4), 'missing': ()}
    assert cache.publish('preprocess', 7, record)
    record['fp32'].zero_()
    cache.phase.value = 1
    loaded = cache.read('preprocess', 7)
    assert loaded['fp32'].tolist() == [1.125, -2.5]
    assert loaded['bf16'].dtype == torch.bfloat16
    assert loaded['bf16'].tolist() == [[3.25]]
    assert loaded['ids'].dtype == torch.int64
    assert loaded['positions'] == (2, 4) and loaded['missing'] == ()
    loaded['fp32'].zero_()  # Even accidental mutation cannot write through to disk.
    assert cache.read('preprocess', 7)['fp32'].tolist() == [1.125, -2.5]
    cache.close()
    with pytest.raises(RuntimeError, match='closed'):
        cache.read('preprocess', 7)


def test_duplicate_publication_never_overwrites(tmp_path):
    cache = make_cache(tmp_path)
    assert cache.publish('targets', 3, {'values': torch.tensor([1.])})
    assert not cache.publish('targets', 3, {'values': torch.tensor([99.])})
    cache.phase.value = 1
    assert cache.read('targets', 3)['values'].item() == 1.
    with pytest.raises(RuntimeError, match='replay'):
        cache.publish('targets', 4, {'values': torch.tensor([2.])})


def test_incomplete_or_wrong_identity_is_rejected(tmp_path):
    cache = make_cache(tmp_path)
    cache.publish('preprocess', 0, {'ids': torch.tensor([1, 2])})
    with pytest.raises((ValueError, FileNotFoundError), match='missing|coverage'):
        cache.validate_coverage({'preprocess': (0, 1)})
    with pytest.raises(ValueError, match='identity'):
        make_cache(tmp_path, identity={'source': 'different-image'})
    manifest = cache.validate_coverage({'preprocess': (0,)})
    assert len(manifest['records']) == 1
    entry = manifest['records']['preprocess/0.pt']
    assert len(entry['sha256']) == 64 and entry['bytes'] > 0
    cache.phase.value = 1
    with pytest.raises(FileNotFoundError):
        cache.read('preprocess', 1)


def test_record_paths_cannot_escape_run_or_select_ineligible_layer(tmp_path):
    cache = make_cache(tmp_path, helper='vision', reference=False)
    for layer, index in [('../foreign', 1), ('vision', -1), ('vision', True),
                         ('targets', 1), ('reference', 1)]:
        with pytest.raises(ValueError):
            cache.publish(layer, index, {'x': torch.ones(1)})
    assert not (tmp_path / 'foreign').exists()


def test_cache_off_has_no_cache_io(tmp_path, monkeypatch):
    from utils_deepseed import parse_args
    monkeypatch.chdir(tmp_path)
    args = parse_args(['--stage', 'stage1'])
    assert args.stage1_cache == 'off'
    assert not list(tmp_path.iterdir())
