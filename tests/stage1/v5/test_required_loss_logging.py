"""Production V5 must log all components even when Tail-KL is disabled."""
from collections import defaultdict
from dataclasses import replace

import pytest
import torch
from transformers import Trainer

from stage1.trainer import CustomTrainerStage1
from tests.stage1.v5.test_trainer import integrated_fixture


TERMS = ('ce', 'alignment', 'dil', 'topk_entropy', 'topk_mass', 'text_kl', 'tail_kl')


def setup_trainer(path, *, debug=False, tail=0., empty=False):
    torch.manual_seed(42)
    trainer, inputs, calls, _, backwards, _ = integrated_fixture(path, debug=debug, empty=empty)
    trainer.v5_config = replace(trainer.v5_config, lambda_topk_entropy=1.,
        lambda_topk_mass=.5, lambda_tail_kl=tail, text_kl_weight=.1,
        preference_topk_mode='sync_topk', topk_include_gt=True)
    trainer.ce_weight = trainer.sim_weight = 1.
    trainer._metrics = defaultdict(list)
    trainer._latest_stage1_diagnostics = None
    trainer.state.global_step = 216  # Resume intervals must not use lifetime steps.
    return trainer, inputs, calls, backwards


@pytest.mark.parametrize('debug', [False, True])
@pytest.mark.parametrize('tail', [0., .7])
def test_actual_v5_logs_all_terms_over_partial_intervals(tmp_path, monkeypatch, debug, tail):
    trainer, inputs, calls, backwards = setup_trainer(tmp_path, debug=debug, tail=tail)
    rows = []
    # Only intercept the final sink; retain V5, parent Trainer and SFT log dispatch.
    monkeypatch.setattr(Trainer, 'log', lambda self, logs, start_time=None: rows.append(dict(logs)))
    coefficients = (1., 1., 1., 1., .5, .1, tail)
    for step, count in [(217, 3), (218, 2)]:
        batches, _ = trainer.get_batch_samples(iter([inputs] * count), 4)
        losses = [trainer.training_step(trainer.model, batch) for batch in batches]
        expected = sum(loss.item() for loss in losses)
        before_log = len(calls)
        trainer.state.global_step = step
        trainer.log({'loss': expected})
        row = rows[-1]
        for name, weight in zip(TERMS, coefficients):
            assert row[f'stage1/weighted_{name}_loss'] == pytest.approx(
                weight * row[f'stage1/{name}_loss'], rel=1e-5, abs=1e-7)
        assert sum(row[f'stage1/weighted_{name}_loss'] for name in TERMS) == pytest.approx(expected)
        assert row['stage1/total_loss'] == pytest.approx(expected)
        assert row['stage1/loss_interval_start_step'] == step - 1
        assert row['stage1/loss_interval_updates'] == 1
        assert len(calls) == before_log
        if not tail:
            assert row['stage1/tail_kl_loss'] == 0.
        if not debug:
            assert trainer._latest_stage1_diagnostics is None
    assert calls == ['helper', 'F', 'reference', 'packed'] * 5
    assert len(backwards) == 5
    trainer.log({'train_loss': 99.})
    assert 'stage1/total_loss' not in rows[-1]  # No stale interval gets logged twice.


def test_logging_preserves_exact_v5_loss_gradient_and_forward_count(tmp_path, monkeypatch):
    results = []
    for enabled in [False, True]:
        trainer, inputs, calls, backwards = setup_trainer(tmp_path / str(enabled))
        batches, _ = trainer.get_batch_samples(iter([inputs]), 4)
        with monkeypatch.context() as context:
            if not enabled:
                # The old tail-disabled parent path bypassed only metric accumulation.
                context.setattr(CustomTrainerStage1, 'training_step',
                                CustomTrainerStage1._stage1_training_step)
            torch.manual_seed(123)
            loss = trainer.training_step(trainer.model, batches[0])
        grads = {n: p.grad.clone() for n, p in trainer.model.named_parameters() if p.grad is not None}
        results.append((loss, grads, calls, len(backwards)))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert results[0][1].keys() == results[1][1].keys()
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=0, atol=0)
    assert results[0][2:] == results[1][2:]


def test_empty_query_v5_still_reports_zero_auxiliary_terms(tmp_path, monkeypatch):
    trainer, inputs, calls, _ = setup_trainer(tmp_path, empty=True)
    rows = []
    monkeypatch.setattr(Trainer, 'log', lambda self, logs, start_time=None: rows.append(dict(logs)))
    batches, _ = trainer.get_batch_samples(iter([inputs]), 4)
    loss = trainer.training_step(trainer.model, batches[0])
    trainer.state.global_step += 1
    trainer.log({'loss': loss.item()})
    for name in TERMS[2:]:
        assert rows[-1][f'stage1/{name}_loss'] == 0.
    assert rows[-1]['stage1/total_loss'] == pytest.approx(loss.item())
    assert calls == ['helper', 'F']
