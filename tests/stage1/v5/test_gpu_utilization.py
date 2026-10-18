"""Four-operator measurements use independent, source-pinned baselines."""
import hashlib
import json
from pathlib import Path
from dataclasses import replace

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


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA operator comparison requires an available device'))]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize('branches', [(True, False), (False, True), (True, True), (False, False)])
def test_replay_bias_exactly_matches_frozen_source_with_padding_and_sparse_targets(device, dtype, branches):
    from tests.stage1.v5.gpu_utilization_fixtures import original_module, replay_edge_fixture
    from transformers.models.qwen2_5_vl.stage1_v5_replay import run_v5_packed_replay
    fixture = replay_edge_fixture(torch.device(device), dtype)
    snapshots = []
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device == 'cuda' else None
    for replay in (original_module('stage1_v5_replay').run_v5_packed_replay, run_v5_packed_replay):
        output = replay(fixture.decoder, fixture.head, fixture.trace, fixture.blocks,
                        execution=fixture.execution, need_voff=branches[0], need_textoff=branches[1])
        snapshots.append((output, fixture.decoder.bias, fixture.decoder.arguments))
        fixture.decoder.bias = fixture.decoder.arguments = None
    expected, actual = snapshots
    assert actual[0].query_keys == expected[0].query_keys
    assert actual[0].query_keys == ((2, 11), (2, 13), (2, 16), (0, 21), (0, 24), (0, 28), (1, 9), (1, 12))
    for name in ('voff_logits', 'textoff_logits'):
        want, got = getattr(expected[0], name), getattr(actual[0], name)
        assert got is None if want is None else torch.equal(got, want)
    if any(branches):
        assert actual[1].dtype == dtype
        assert torch.equal(actual[1], expected[1])
        assert ((actual[1] == 0) | actual[1].isneginf()).all()
        for name in ('inputs_embeds', 'position_ids'):
            assert torch.equal(actual[2][name], expected[2][name])
        for got_pair, want_pair in zip(actual[2]['stage2_prefix_key_values'], expected[2]['stage2_prefix_key_values']):
            assert all(torch.equal(got, want) for got, want in zip(got_pair, want_pair))
        # Sample 1 has padding at key zero, yet its padded queries need the
        # original key-zero escape. Valid queries must keep that key masked.
        row = 2
        assert actual[1][row, 0, :4, 0].isneginf().all()
        assert (actual[1][row, 0, 4:, 0] == 0).all()
        assert actual[1][row, 0, 4:, 1:].isneginf().all()
    else:
        assert actual[1:] == expected[1:] == (None, None)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)


@pytest.mark.parametrize('empty', [False, True])
def test_replay_without_active_blocks_never_dispatches_decoder(empty):
    from tests.stage1.v5.gpu_utilization_fixtures import original_module, replay_edge_fixture
    from transformers.models.qwen2_5_vl.stage1_v5_replay import run_v5_packed_replay
    fixture = replay_edge_fixture(torch.device('cpu'))
    fixture.blocks = () if empty else (fixture.blocks[1],)
    # No attached trace is needed on the inactive path.
    fixture.trace = None
    for replay in (original_module('stage1_v5_replay').run_v5_packed_replay, run_v5_packed_replay):
        output = replay(fixture.decoder, fixture.head, fixture.trace, fixture.blocks,
                        execution=fixture.execution, need_voff=True, need_textoff=True)
        assert output.query_keys == ()
        assert output.voff_logits is output.textoff_logits is None
        assert fixture.decoder.arguments is None


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('debug', [False, True])
@pytest.mark.parametrize('p', [0., .05, .4])
@pytest.mark.parametrize('coordinates', ['permuted_duplicates', 'one_source', 'empty'])
def test_raw_dropout_exactly_matches_frozen_source_without_global_rng(device, debug, p, coordinates):
    from stage1.equivalence import Stage1StochasticContext
    from stage1.v5_execution import V5Execution
    from tests.stage1.v5.gpu_utilization_fixtures import OperatorCounts, original_module, run_raw_dropout
    spec = Stage1StochasticContext(77, 11, ((31, 11), (2, 4), (6, 8)), ('operator.first', 'operator.second'))
    samples, positions = {
        'permuted_duplicates': ([2, 0, 2, 1, 0, 2, 1], [7, 10, 3, 0, 10, 7, 3]),
        'one_source': ([1, 1, 1], [3, 0, 3]),
        'empty': ([], []),
    }[coordinates]
    samples = torch.tensor(samples, device=device, dtype=torch.long)
    positions = torch.tensor(positions, device=device, dtype=torch.long)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device == 'cuda' else None
    snapshots = []
    for execution_type in (original_module('v5_execution').V5Execution, V5Execution):
        with OperatorCounts() as counts:
            masks, owner = run_raw_dropout(execution_type, spec, samples, positions, p=p, debug=debug)
        snapshots.append((masks, owner, counts))
    expected, actual = snapshots
    assert all(torch.equal(got, want) for got, want in zip(actual[0], expected[0]))
    assert all(got.shape == (samples.numel(), 5) and got.dtype == torch.float32 for got in actual[0])
    assert actual[1].state == expected[1].state == 'CLOSED'
    assert actual[1].events == expected[1].events
    referenced = set(samples.tolist())
    for _, _, counts in snapshots:
        assert counts.random_draw_calls == 2 * len(referenced)
        assert counts.random_draw_elements == 2 * 5 * sum(spec.rows[row][1] for row in referenced)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)


@pytest.mark.parametrize('samples,positions', [([3], [0]), ([-1], [0]), ([1], [4]), ([0], [-1])])
def test_dropout_diagnostic_coordinate_rejection_matches_frozen_source(samples, positions):
    from stage1.equivalence import Stage1StochasticContext
    from stage1.v5_execution import V5Execution, current_v5_execution
    from tests.stage1.v5.gpu_utilization_fixtures import original_module, run_raw_dropout
    spec = Stage1StochasticContext(77, 11, ((31, 11), (2, 4)), ('operator.first',))
    for execution_type in (original_module('v5_execution').V5Execution, V5Execution):
        with pytest.raises(ValueError, match='outside source valid length'):
            run_raw_dropout(execution_type, spec, torch.tensor(samples), torch.tensor(positions), debug=True)
    assert current_v5_execution() is None


def test_replay_mask_dispatch_count_is_bounded_by_rows_not_suffix_queries():
    from tests.stage1.v5.gpu_utilization_fixtures import OperatorCounts, original_module, replay_fixture, run_replay_mask
    from transformers.models.qwen2_5_vl.stage1_v5_replay import run_v5_packed_replay
    counts_by_source = []
    for replay in (original_module('stage1_v5_replay').run_v5_packed_replay, run_v5_packed_replay):
        sizes = []
        for suffix_length in (4, 32):
            fixture = replay_fixture(torch.device('cpu'))
            fixture.blocks = (replace(fixture.blocks[0], target_positions=(128 + suffix_length,),
                                      target_ids=(0,), blocked_text_keys=((),)),)
            with OperatorCounts() as counts:
                bias = run_replay_mask(replay, fixture, need_textoff=False)
            assert bias.shape == (1, 1, suffix_length, 128 + suffix_length)
            sizes.append((counts.select_calls, counts.copy_calls))
        counts_by_source.append(sizes)
    frozen, candidate = counts_by_source
    print(f'REPLAY_ASSEMBLY_DISPATCH frozen={frozen} candidate={candidate}')
    assert frozen[1][0] > frozen[0][0] and frozen[1][1] > frozen[0][1]
    assert candidate[0] == candidate[1], 'replay mask still dispatches work for each suffix query'
    assert candidate[1][0] < frozen[0][0]
    assert candidate[1][1] <= 1, 'replay bias still uses per-query copy assignments'


def test_dropout_dispatch_uses_source_gathers_without_token_select_stack(monkeypatch):
    from stage1.equivalence import Stage1StochasticContext
    from stage1.v5_execution import V5Execution
    from tests.stage1.v5.gpu_utilization_fixtures import OperatorCounts, original_module, run_raw_dropout
    spec = Stage1StochasticContext(77, 11, ((31, 512), (2, 499), (6, 267)), ('operator.first',))
    counts_by_source = []
    original_tolist = torch.Tensor.tolist
    lists = []
    def observed_tolist(value):
        lists.append(value.numel())
        return original_tolist(value)
    monkeypatch.setattr(torch.Tensor, 'tolist', observed_tolist)
    for execution_type in (original_module('v5_execution').V5Execution, V5Execution):
        sizes = []
        for count in (16, 256):
            samples = torch.arange(count) % 3
            positions = torch.arange(count).flip(0)
            lists.clear()
            with OperatorCounts() as dispatches:
                run_raw_dropout(execution_type, spec, samples, positions)
            sizes.append((dispatches.select_calls, dispatches.stack_calls, tuple(lists)))
        counts_by_source.append(sizes)
    frozen, candidate = counts_by_source
    print(f'DROPOUT_ASSEMBLY_DISPATCH frozen={frozen} candidate={candidate}')
    assert frozen[1][0] > frozen[0][0]
    assert frozen[1][1] == frozen[0][1] == 1
    assert candidate[0] == candidate[1], 'dropout assembly still scales dispatch/host reads with requested tokens'
    assert candidate[1][0] <= 3 and candidate[1][1] == 0, 'dropout must gather per source without token select/stack'
    assert candidate[1][2] == (3,), 'ordinary path may read compact source IDs only'
