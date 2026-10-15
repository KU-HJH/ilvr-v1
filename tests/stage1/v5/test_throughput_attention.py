"""Catch repeated FP32 K allocation/scalar synchronization without relaxing math."""
from dataclasses import replace

import pytest
import torch

from stage1.v5_config import Stage1V5Config
from stage1.v5_metadata import V5BatchMetadata, V5Block
from stage1.v5_selection import select_v5_text_keys
from transformers.models.qwen2_5_vl.stage1_v5_trace import capture_v5_trace
from tests.stage1.v5.throughput_fixtures import (
    OperatorCounts, observation_fixture, original_module, selector_fixture,
)


def test_trace_converts_keys_once_per_sample_layer_and_shares_coordinates():
    # Reintroducing K.float()/arange in the query loop must fail these counts.
    decoder, kwargs, keys, candidates = observation_fixture()
    with OperatorCounts((4, 64, 8)) as counts:
        with capture_v5_trace(decoder, keys, candidates) as trace:
            output = decoder(**kwargs)
    assert counts.key_float_copies == 2 * 2
    assert counts.arange_calls <= 2  # source positions and one shared causal coordinate
    assert all(not value.requires_grad for value in trace.attention_rows.values())
    assert all(value.requires_grad for pair in trace.key_values for value in pair)
    output.last_hidden_state.sum().backward()
    assert kwargs['inputs_embeds'].grad is not None


def test_observed_rows_match_frozen_original_exactly():
    decoder, kwargs, keys, candidates = observation_fixture()
    with original_module('stage1_v5_trace').capture_v5_trace(decoder, keys, candidates) as original:
        decoder(**kwargs)
    with capture_v5_trace(decoder, keys, candidates) as actual:
        decoder(**kwargs)
    assert actual.attention_rows.keys() == original.attention_rows.keys()
    for key in keys:
        torch.testing.assert_close(actual.attention_rows[key], original.attention_rows[key], rtol=0, atol=0)


def test_trace_materializes_candidate_coordinates_once_per_distinct_set():
    # Restoring Python-list indexing per query repeats tensor creation/host copies.
    decoder, kwargs, keys, candidates = observation_fixture()
    with OperatorCounts() as counts:
        with capture_v5_trace(decoder, keys, candidates):
            decoder(**kwargs)
    assert counts.candidate_index_tensors == len(set(candidates))


def test_selector_bulk_transfers_without_per_query_scalar_reads(monkeypatch):
    # Restoring mass.item() or per-query weights.tolist() must fail this witness.
    args = selector_fixture()
    transfers = []
    original = torch.Tensor.tolist
    def tolist(value):
        transfers.append(value.numel())
        return original(value)
    monkeypatch.setattr(torch.Tensor, 'tolist', tolist)
    with OperatorCounts() as counts:
        result = select_v5_text_keys(*args, Stage1V5Config(), validate=False)
    assert len(result) == len(args[0].query_keys)
    assert counts.scalar_reads == 0
    assert len(transfers) <= 3


@pytest.mark.parametrize('case', ['normal', 'empty', 'ties', 'zero_cap', 'low_mass', 'threshold'])
@pytest.mark.parametrize('compact', [False, True])
def test_selected_keys_match_frozen_original_for_branch_boundaries(case, compact):
    metadata, attention, clean, reference = selector_fixture()
    config = Stage1V5Config()
    if case == 'empty':
        metadata = V5BatchMetadata((V5Block(0, 0, 1, 2, (3,), (0,), ()),))
        attention, clean, reference = {(0, 2): torch.zeros(4)}, clean[:1], reference[:1]
    elif case == 'zero_cap':
        config = replace(config, text_cap=0.)
    elif case in ('ties', 'low_mass', 'threshold'):
        for block in metadata.blocks:
            for q in block.query_positions:
                row = attention[(block.batch_index, q)]
                row.zero_()
                if case == 'ties':
                    row[list(block.candidate_positions)] = .03
                elif case == 'low_mass':
                    row[list(block.candidate_positions)] = .001
                else:
                    # rho=.4 exactly; values straddle its cumulative boundary.
                    row[list(block.candidate_positions)] = .3 / 11
                    row[block.candidate_positions[0]] = torch.nextafter(torch.tensor(.2), torch.tensor(0.))
        if case == 'threshold':
            reference = clean.clone()
    expected = original_module('v5_selection').select_v5_text_keys(
        metadata, attention, clean, reference, config)
    if compact:
        attention = {(b.batch_index, q): attention[(b.batch_index, q)][list(b.candidate_positions)]
                     for b in metadata.blocks for q in b.query_positions}
    actual = select_v5_text_keys(metadata, attention, clean, reference, config,
                                **({'compact_attention': True} if compact else {}))
    assert actual == expected
    assert clean.grad is None


def test_empty_queries_remain_valid_in_compact_mode():
    assert select_v5_text_keys(V5BatchMetadata(()), {}, torch.empty(0, 3), torch.empty(0, 3),
                               Stage1V5Config(), compact_attention=True) == {}


def test_trainer_hands_off_compact_candidates_without_full_sequence_expansion(tmp_path, monkeypatch):
    from tests.stage1.v5.test_trainer import integrated_fixture
    from stage1 import v5_trainer
    trainer, batch, *_ = integrated_fixture(tmp_path, debug=False)
    observed = []
    def select(metadata, attention, *args, **kwargs):
        assert kwargs.get('compact_attention') is True
        for block in metadata.blocks:
            for query in block.query_positions:
                assert attention[(block.batch_index, query)].shape == (len(block.candidate_positions),)
        observed.append(len(attention))
        return select_v5_text_keys(metadata, attention, *args, **kwargs)
    monkeypatch.setattr(v5_trainer, 'select_v5_text_keys', select)
    batches, _ = trainer.get_batch_samples(iter([batch]), 4)
    assert torch.isfinite(trainer.training_step(trainer.model, batches[0]))
    assert observed == [8]


def test_benchmark_compares_identical_candidate_inputs(tmp_path, monkeypatch):
    # Passing full rows to the compact benchmark path must fail its exact oracle.
    from scripts.benchmark_stage1_v5_attention_selection import main
    output = tmp_path / 'operators.json'
    monkeypatch.setattr('sys.argv', ['benchmark', '--device', 'cpu', '--warmup', '0',
                                  '--repetitions', '1', '--output', str(output)])
    main()
    import json
    report = json.loads(output.read_text())
    assert report['selected_keys_exact_equal'] and report['attention_exact_equal']
    assert report['selection_after']['operations']['tolist_elements'] == 4096 + 32 + 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason='mixed CPU/CUDA direct selector contract')
@pytest.mark.parametrize('mixed', [False, True])
def test_direct_selector_preserves_attention_devices_independent_of_logits(mixed):
    # Co-locating host rows with CUDA rho must not become a new public API requirement.
    metadata, attention, clean, reference = selector_fixture()
    clean, reference = clean.cuda(), reference.cuda()
    if mixed:
        attention = {key:(value.cuda() if i % 2 else value)
                     for i,(key,value) in enumerate(attention.items())}
    config = Stage1V5Config()
    expected = original_module('v5_selection').select_v5_text_keys(
        metadata, attention, clean, reference, config)
    assert select_v5_text_keys(metadata, attention, clean, reference, config) == expected
