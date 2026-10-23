"""Clean masks and metadata-prefetched cardinalities must describe one objective."""
from types import SimpleNamespace

import pytest
import torch

from stage1.v5_config import resolve_v5_config
from stage1.v5_metadata import V5BatchMetadata, V5Block
from stage1.v5_selection import select_v5_orig_positions


@pytest.mark.parametrize('percent,positions', [
    (20, [3, 4, 6, 7]),
    (20.000000000000004, [3, 4, 5, 6, 7]),
    (33.333333333333336, [3, 4, 5, 6, 7, 8]),
    (100, list(range(12))),
])
def test_prefetch_count_matches_sample_local_rational_ceiling_and_stable_tied_mask(percent, positions):
    from stage1.v5_trainer import CustomTrainerStage1V5
    config = resolve_v5_config({'selection': 'clean', 'preference_entropy_top_percent': percent})
    # Unsorted blocks, ragged samples, a sample with no text, and empty metadata.
    meta = V5BatchMetadata((
        V5Block(0, 3, 7, 8, (10, 11, 12), (0, 0, 0), ()),
        V5Block(1, 1, 0, 1, (2,), (0,), ()),
        V5Block(0, 1, 0, 1, (2, 3), (0, 0), ()),
        V5Block(2, 1, 0, 1, (2, 3, 4, 5, 6, 7), (0,) * 6, ()),
        V5Block(3, 1, 0, 1, (), (), ())))
    logits = torch.zeros(12, 4, requires_grad=True)
    mask = select_v5_orig_positions(meta, logits, None, config)
    assert mask.nonzero().flatten().tolist() == positions
    assert not mask.requires_grad
    trainer = CustomTrainerStage1V5.__new__(CustomTrainerStage1V5)
    trainer.v5_config = config
    trainer.args = SimpleNamespace(average_tokens_across_devices=False, gradient_accumulation_steps=4)
    seen = []
    def reduce(value, reduction):
        assert reduction == 'sum'
        seen.append(value.tolist())
        # Represents three other empty ranks; their original blocks still count.
        return value + torch.tensor([3, 0])
    trainer.accelerator = SimpleNamespace(device='cpu', num_processes=4, reduce=reduce)
    original = [{'stage1_v5_metadata': meta}, {'stage1_v5_metadata': V5BatchMetadata(())}]
    batches, _ = trainer.get_batch_samples(iter(original), 4)
    assert seen == [[5, len(positions)]]
    assert batches[0]['stage1_v5_window'].global_dil_tokens == len(positions)
    assert batches[0]['stage1_v5_window'].global_original_steps == 8
    assert batches[0]['stage1_v5_window'].actual_microbatches == 2
    assert all('stage1_v5_window' not in batch for batch in original)


@pytest.mark.parametrize('requested,want', [(None, 1), (10, 10)])
def test_stage1_retention_reaches_training_arguments_without_changing_default(tmp_path, requested, want):
    from stage1.composition import build_stage1_training_arguments
    from tests.stage1.test_composition import stage1_args
    extra = [] if requested is None else ['--save_total_limit', str(requested)]
    args = stage1_args(tmp_path, '--save_strategy', 'steps', '--save_steps', '216', *extra)
    args.use_cpu = True
    args.deepspeed_config = None
    actual = build_stage1_training_arguments(args)
    assert actual.save_total_limit == want
    assert actual.save_steps == 216
    assert actual.save_strategy.value == 'steps'


@pytest.mark.parametrize('value', ['0', '-1', '1.5', 'true'])
def test_stage1_retention_rejects_invalid_worker_values(value):
    from utils_deepseed import parse_args
    with pytest.raises((ValueError, SystemExit)):
        parse_args(['--save_total_limit', value])
