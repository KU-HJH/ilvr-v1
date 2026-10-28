import io
import tarfile

import pytest
import torch

from tests.stage1.test_run_cache import make_cache


def test_duplicate_records_have_canonical_bytes(tmp_path):
    a, b = (make_cache(tmp_path / node) for node in ('a', 'b'))
    for cache in (a, b):
        cache.publish('preprocess', 0, {'value': torch.arange(19)})
    assert (a.plan.root / 'preprocess/0.pt').read_bytes() == (b.plan.root / 'preprocess/0.pt').read_bytes()


def test_transfer_is_bounded_and_rejects_foreign_paths(tmp_path):
    from stage1.cache_transport import export_records, import_records
    a, b = (make_cache(tmp_path / node) for node in ('a', 'b'))
    a.publish('preprocess', 0, {'value': torch.arange(300000)})
    manifest = a.validate_coverage({'preprocess': (0,)})
    archive = io.BytesIO()
    export_records(a, manifest, ('preprocess/0.pt',), archive)
    class Bounded(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            return super().read(size)
    import_records(b, manifest, ('preprocess/0.pt',), Bounded(archive.getvalue()))
    assert b.validate_coverage({'preprocess': (0,)}) == manifest
    bad = io.BytesIO()
    with tarfile.open(fileobj=bad, mode='w') as tar:
        info = tarfile.TarInfo('../foreign.pt')
        info.size = 3
        tar.addfile(info, io.BytesIO(b'bad'))
    bad.seek(0)
    with pytest.raises(ValueError, match='record|path'):
        import_records(b, manifest, ('preprocess/0.pt',), bad)
    assert not (b.plan.root.parent / 'foreign.pt').exists()


def test_exchange_requires_both_nodes_and_complete_coverage(tmp_path):
    from stage1.cache_transport import exchange_stage1_cache, LocalCacheExchange
    a, b = (make_cache(tmp_path / node) for node in ('a', 'b'))
    a.publish('preprocess', 0, {'value': torch.tensor(0)})
    b.publish('preprocess', 1, {'value': torch.tensor(1)})
    runner = LocalCacheExchange({0: a, 1: b})
    with pytest.raises(FileNotFoundError, match='coverage'):
        exchange_stage1_cache(None, runner, expected={'preprocess': (0, 1, 2)})
    result = exchange_stage1_cache(None, runner, expected={'preprocess': (0, 1)})
    assert result['records'] == 2
    assert a.validate_coverage({'preprocess': (0, 1)}) == b.validate_coverage({'preprocess': (0, 1)})


def test_conflicting_duplicate_stops_exchange(tmp_path):
    from stage1.cache_transport import exchange_stage1_cache, LocalCacheExchange
    a, b = (make_cache(tmp_path / node) for node in ('a', 'b'))
    a.publish('preprocess', 0, {'value': torch.tensor(0)})
    b.publish('preprocess', 0, {'value': torch.tensor(1)})
    with pytest.raises(ValueError, match='conflict'):
        exchange_stage1_cache(None, LocalCacheExchange({0: a, 1: b}), expected={'preprocess': (0,)})


def test_cache_worker_stream_crosses_process_boundary(tmp_path):
    import json
    import subprocess
    import sys
    a, b = (make_cache(tmp_path / node) for node in ('a', 'b'))
    a.publish('preprocess', 5, {'value': torch.arange(100)})
    manifest = a.validate_coverage({'preprocess': (5,)})
    header = (json.dumps({'manifest': manifest, 'names': ['preprocess/5.pt']}) + '\n').encode()
    exported = subprocess.run([sys.executable, '-m', 'stage1.cache_transport', 'export',
        str(a.plan.root), str(tmp_path / 'control-a')], input=header, capture_output=True)
    assert exported.returncode == 0, exported.stderr
    imported = subprocess.run([sys.executable, '-m', 'stage1.cache_transport', 'import',
        str(b.plan.root), str(tmp_path / 'control-b')], input=header + exported.stdout, capture_output=True)
    assert imported.returncode == 0, imported.stderr
    assert b.validate_coverage({'preprocess': (5,)}) == manifest
