"""Loss logging must survive production policy without changing training."""
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Trainer, TrainingArguments

from stage1.matched_execution import enable_matched_standard_execution
from tests.stage1.test_trainer import CountingTargetBuilder, make_batch, make_trainer


class ScalarStudent(nn.Module):
    """Known CE and opposite one-dimensional latent targets (alignment=2)."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    @property
    def device(self):
        return self.scale.device

    def forward(self, input_ids, latent_hidden_states, image_out_mask, **kwargs):
        self.calls += 1
        embeds = torch.ones((*input_ids.shape, 1), device=self.device)
        if hasattr(self, 'dropout_site'):
            embeds = self.dropout_site(embeds)
        return SimpleNamespace(
            loss=self.scale.square() * input_ids[0, 0].float(),
            logits=embeds.expand(-1, -1, 4),
            hidden_states=-embeds * self.scale,
            inputs_embeds=embeds,
        )


def batch(value):
    result = make_batch(torch.tensor([[False, True, False]]))
    result['input_ids'][0, 0] = value
    return result


def logging_trainer(path, *, gas=4, logging_steps=2):
    model = ScalarStudent()
    builder = CountingTargetBuilder(torch.ones(1, 1),
        torch.tensor([[False, True, False]]), ((1,),))
    trainer = make_trainer(model, builder, ce_weight=.2, sim_weight=.5)
    # Exercise native loss/backward/log dispatch with a real CPU Trainer; avoid
    # SFT tokenization/model loading, which is unrelated to the logging defect.
    Trainer.__init__(trainer, model=model, args=TrainingArguments(
        output_dir=str(path), use_cpu=True, report_to=[], disable_tqdm=True,
        per_device_train_batch_size=1, gradient_accumulation_steps=gas,
        max_steps=3, logging_steps=logging_steps, save_strategy='no',
        remove_unused_columns=False, learning_rate=0., max_grad_norm=0.),
        train_dataset=[batch(v) for v in [2, 4, 6, 8, 10]],
        data_collator=lambda rows: rows[0])
    trainer.model_accepts_loss_kwargs = False
    trainer._metrics = defaultdict(list)
    trainer.create_optimizer_and_scheduler(3)
    trainer.stage1_check_policy = 'production'
    trainer._globalstep_last_logged = 0
    trainer._total_loss_scalar = 0.
    trainer.current_flos = 0.
    return trainer


def log_update(trainer, values):
    summed = sum(trainer.training_step(trainer.model, batch(v)) for v in values)
    trainer.state.global_step += 1
    trainer.control.should_log = True
    trainer._maybe_log_save_evaluate(summed, None, trainer.model, None, 0, None, time.time())
    return trainer.state.log_history[-1]


def assert_terms(row, ce, alignment, total):
    assert row['stage1/ce_loss'] == pytest.approx(ce)
    assert row['stage1/alignment_loss'] == pytest.approx(alignment)
    assert row['stage1/weighted_ce_loss'] == pytest.approx(.2 * ce)
    assert row['stage1/weighted_alignment_loss'] == pytest.approx(.5 * alignment)
    assert row['stage1/total_loss'] == pytest.approx(total)
    assert (row['stage1/weighted_ce_loss'] + row['stage1/weighted_alignment_loss']
            == pytest.approx(row['stage1/total_loss']))


def test_production_logs_interval_means_instead_of_last_microbatch(tmp_path):
    """Catches diagnostic gating or using the last CE (8) instead of mean (5)."""
    trainer = logging_trainer(tmp_path)
    row = log_update(trainer, [2, 4, 6, 8])
    assert_terms(row, 5., 2., 2.)
    assert row['loss'] == pytest.approx(row['stage1/total_loss'])
    assert not trainer.collect_stage1_diagnostics
    assert trainer.model.calls == trainer.target_builder.teacher_language_calls == 4
    assert trainer._latest_stage1_diagnostics is None


def test_resumed_tail_and_eval_do_not_corrupt_interval(tmp_path):
    """Catches dividing resumed metrics by lifetime steps or counting eval CE."""
    trainer = logging_trainer(tmp_path)
    trainer.state.global_step = 216
    trainer.training_step(trainer.model, batch(2))
    with torch.no_grad():
        trainer.compute_loss(trainer.model, batch(100))
    trainer.log({'eval_loss': 21.})
    assert 'stage1/ce_loss' not in trainer.state.log_history[-1]
    trainer.training_step(trainer.model, batch(4))
    trainer.training_step(trainer.model, batch(6))
    trainer.state.global_step = 217
    trainer.log({'train_loss': 999.})
    row = trainer.state.log_history[-1]
    assert_terms(row, 3., 1.5, 1.35)
    assert row['train_loss'] == 999.  # Whole-run summary must not be relabeled.
    assert row['stage1/loss_interval_start_step'] == 216
    assert row['stage1/loss_interval_end_step'] == 217
    assert row['stage1/loss_interval_updates'] == 1
    trainer.log({'train_loss': 999.})
    assert 'stage1/ce_loss' not in trainer.state.log_history[-1]
    trainer._globalstep_last_logged = 217
    assert_terms(log_update(trainer, [10, 10, 10, 10]), 10., 2., 3.)


def test_last_microbatch_diagnostics_cannot_overwrite_interval_losses(tmp_path):
    """Catches collision between debug fields and averaged public loss keys."""
    trainer = logging_trainer(tmp_path)
    trainer.collect_stage1_diagnostics = True
    row = log_update(trainer, [2, 4, 6, 8])
    assert_terms(row, 5., 2., 2.)
    assert row['stage1/latent_count'] == 1


def test_logging_keeps_native_gradients_and_forward_count(tmp_path):
    """Catches a live metric graph, extra forward, or changed loss scaling."""
    trainer = logging_trainer(tmp_path / 'logged')
    reference = logging_trainer(tmp_path / 'reference')
    torch.manual_seed(42)
    (reference.compute_loss(reference.model, batch(6)) / 4).backward()
    torch.manual_seed(42)
    actual = trainer.training_step(trainer.model, batch(6))
    torch.testing.assert_close(trainer.model.scale.grad, reference.model.scale.grad,
                               rtol=0, atol=0)
    assert actual.item() == pytest.approx(.55)
    assert trainer.model.calls == trainer.target_builder.teacher_language_calls == 1
    trainer.state.global_step = 1
    trainer.log({'train_loss': .55})
    assert_terms(trainer.state.log_history[-1], 1.5, .5, .55)


def test_real_trainer_loop_logs_final_partial_interval(tmp_path):
    """Catches wiring that only works with manual log calls, not Trainer.train."""
    trainer = logging_trainer(tmp_path)
    enable_matched_standard_execution(trainer)
    from stage1.equivalence import SampleStableStage1Dropout
    trainer.model.dropout_site = SampleStableStage1Dropout('unit', 0.)
    Trainer.train(trainer)
    rows = trainer.state.log_history
    ordinary = next(row for row in rows if 'loss' in row)
    # First epoch has five microbatches, hence two updates with fixed GAS=4.
    assert_terms(ordinary, 3.75, 1.25, 1.375)
    assert ordinary['loss'] == pytest.approx(1.375)
    final = rows[-1]
    assert final['stage1/loss_interval_start_step'] == 2
    assert final['stage1/loss_interval_end_step'] == 3
    assert final['stage1/total_loss'] == pytest.approx(
        final['stage1/weighted_ce_loss'] + final['stage1/weighted_alignment_loss'])


def distributed_logging_worker(rank, root):
    import torch.distributed as dist
    trainer = logging_trainer(Path(root) / f'rank{rank}')
    dist.init_process_group('gloo', init_method=f'file://{root}/rendezvous',
                            rank=rank, world_size=4)
    try:
        # Use Trainer's real distributed_concat path on CPU, without GPU training.
        trainer.args.distributed_state = SimpleNamespace(distributed_type='MULTI_CPU')
        row = log_update(trainer, [2 + 2 * rank] * 4)
        (Path(root) / f'metrics{rank}.json').write_text(json.dumps(row))
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_four_rank_logging_averages_terms_not_only_total(tmp_path):
    """Catches rank-local term metrics beside a globally averaged Trainer loss."""
    torch.multiprocessing.start_processes(distributed_logging_worker,
        args=(str(tmp_path),), nprocs=4, join=True, start_method='spawn')
    for rank in range(4):
        row = json.loads((tmp_path / f'metrics{rank}.json').read_text())
        assert_terms(row, 5., 2., 2.)
        assert row['loss'] == pytest.approx(2.)
