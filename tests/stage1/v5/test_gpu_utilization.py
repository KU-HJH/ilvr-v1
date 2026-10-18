"""Four-operator measurements use independent, source-pinned baselines."""
import hashlib
import json
from pathlib import Path

import pytest
import torch


BASELINE_SHA256 = {
    'v5_execution': '0abd7e9fb51f327916b735b74d7cad53dd0d4b7242d3dc74725dc43257c19a69',
    'v5_selection': '1b285303a6da390c4de2635aff46c57582272f952d75efc72b52a67eddce2850',
    'stage1_v5_trace': '33f55a0f6480bbfc50b9256cceb71c6ab61a968cbf4b6815c5dee8f80f5f7efb',
    'stage1_v5_replay': 'bf15d87ea7d6f9a2dd7775c6e1ab4d4b4edb3750bc4096213b8b3e310135fa5e',
}


def test_benchmark_reports_four_paired_operators(tmp_path, monkeypatch):
    from scripts.benchmark_stage1_v5_attention_selection import main
    output = tmp_path / 'operators.json'
    monkeypatch.setattr('sys.argv', ['benchmark', '--device', 'cpu', '--warmup', '0',
                                  '--repetitions', '1', '--output', str(output)])
    main()
    report = json.loads(output.read_text())
    for name in ('attention', 'selection', 'replay_mask', 'dropout'):
        for suffix in ('before', 'after'):
            result = report[name + '_' + suffix]
            assert len(result['milliseconds']) == 1
            assert result['milliseconds'][0] > 0
            assert result['memory']['peak_python_bytes'] > 0
            assert 'Python allocations only' in result['memory']['note']
            assert result['operations']
    for name in ('attention', 'selection', 'replay_mask', 'dropout'):
        assert report[name + '_paired_order'] == [['before', 'after']]
    assert report['mask_exact_equal'] is True
    assert report['dropout_exact_equal'] is True
    assert report['attention_exact_equal'] is True
    assert report['attention_close'] is True
    assert report['attention_max_absolute_error'] == 0
    assert report['selected_keys_exact_equal'] is True
    assert report['oracle_sha256'] == BASELINE_SHA256
    assert report['source_identity']['baseline_commit'] == '65643f08cb6e61fbbb200de825d57086f5c7a127'
    assert set(report['source_identity']['candidate_sha256']) == set(BASELINE_SHA256)
    assert report['replay_mask_fixture']['mask_shape'] == [8, 1, 24, 344]
    assert report['dropout_fixture']['source_lengths'] == [512, 499]
    assert report['dropout_fixture']['checkpoint'] == 'non-reentrant early-stop'
    assert 'dummy decoder' in report['replay_mask_fixture']['scope']
    assert 'backward' in report['dropout_fixture']['scope']


@pytest.mark.parametrize('name,expected', BASELINE_SHA256.items())
def test_frozen_gpu_utilization_sources_have_literal_hashes(name, expected):
    path = Path(__file__).with_name('frozen_gpu_utilization') / (name + '.py')
    assert path.is_file(), f'missing independent frozen source: {name}'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_paired_measurement_alternates_actual_call_order():
    from scripts import benchmark_stage1_v5_attention_selection as benchmark
    measure = getattr(benchmark, 'measure_pair', None)
    assert callable(measure), 'missing paired timing execution'
    calls = []
    before, after, order = measure(lambda: calls.append('before'), lambda: calls.append('after'),
                                  torch.device('cpu'), 0, 3)
    assert calls[:6] == ['before', 'after', 'after', 'before', 'before', 'after']
    assert order == [['before', 'after'], ['after', 'before'], ['before', 'after']]
    assert len(before['milliseconds']) == len(after['milliseconds']) == 3


def test_dropout_fixture_uses_private_rng_and_checkpoint_backward():
    import importlib.util
    assert importlib.util.find_spec('tests.stage1.v5.gpu_utilization_fixtures') is not None, 'missing dropout execution fixture'
    from tests.stage1.v5.gpu_utilization_fixtures import dropout_fixture, original_module, run_dropout
    from stage1.v5_execution import V5Execution
    fixture = dropout_fixture(torch.device('cpu'))
    rng = torch.random.get_rng_state().clone()
    expected = run_dropout(original_module('v5_execution').V5Execution, fixture, capture=True)
    actual = run_dropout(V5Execution, fixture, capture=True)
    assert torch.equal(torch.random.get_rng_state(), rng)
    for name in ('native_output', 'packed_output', 'native_gradient', 'packed_gradient'):
        assert torch.equal(actual[name], expected[name])
    assert len(actual['raw_masks']) == len(expected['raw_masks']) == 4
    assert all(torch.equal(a, b) for a, b in zip(actual['raw_masks'], expected['raw_masks']))
    assert actual['state'] == expected['state'] == 'CLOSED'
    assert {(event['role'], event['phase']) for event in actual['events'] if event['site']} == {
        ('F', 'original'), ('packed', 'original'), ('F', 'recompute'), ('packed', 'recompute')}
    assert fixture.packed_samples.numel() > torch.unique(torch.stack((fixture.packed_samples, fixture.packed_positions)), dim=1).shape[1]
