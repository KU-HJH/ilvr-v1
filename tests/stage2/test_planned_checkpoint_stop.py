"""CPU save/stop/resume lifecycle; no four-rank or DIL numerics claim."""
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import TrainerCallback, TrainingArguments

from stage2.trainer import Stage2Trainer
from tests.stage2.helpers import tiny_stage2_qwen
from tests.stage2.test_dil_peft_roundtrip import bce_config
from tests.stage2.test_dil_resume_entry import resume_argv
from utils_deepseed import parse_args


def fresh_argv():
    argv = resume_argv('saved', 'base', 'data')
    index = argv.index('--resume_from_checkpoint_path')
    del argv[index:index + 2]
    argv[-1] = '3'
    return argv + ['--stage2b_direct_from_stage1', '--stage1_adapter_path', 'stage1']


@pytest.mark.parametrize('kind', ['dil_bce_rf', 'dil_bce_rf_topk_reverse'])
def test_parser_accepts_stop_only_before_resolved_budget(kind):
    argv = fresh_argv()
    argv[argv.index('--step_preference_loss_type') + 1] = kind
    if kind == 'dil_bce_rf_topk_reverse':
        argv += ['--step_preference_topk', '8']
    args = parse_args(argv + ['--stage2_stop_after_checkpoint', '2'])
    assert args.stage2_stop_after_checkpoint == 2
    assert args.max_steps == 3
    # Explicit phase allocation wins over raw max_steps.
    with pytest.raises(ValueError, match='stop_after_checkpoint'):
        parse_args(argv + ['--stage2_total_steps', '2', '--stage2_stop_after_checkpoint', '2'])


@pytest.mark.parametrize('value', ['0', '-1', '3', '4', 'true', '2.5'])
def test_parser_rejects_invalid_stop(value):
    with pytest.raises((ValueError, SystemExit)):
        parse_args(fresh_argv() + ['--stage2_stop_after_checkpoint', value])


def test_stop_rejects_resume():
    with pytest.raises(ValueError, match='fresh'):
        parse_args(resume_argv('saved', 'base', 'data') + ['--stage2_stop_after_checkpoint', '2'])


def test_ordinary_preference_stop_preserves_the_full_baseline_budget():
    """R1: the real argument parser must admit the approved ordinary smoke."""
    argv = fresh_argv()
    argv[argv.index('--step_preference_loss_type') + 1] = 'dpo'
    argv[argv.index('--step_preference_beta') + 1] = '10'
    argv[argv.index('--max_steps') + 1] = '212'
    args = parse_args(argv + [
        '--stage2_total_steps', '212', '--ce_weight', '0.1',
        '--step_preference_margin', '0.5', '--stage2_stop_after_checkpoint', '2',
    ])
    assert args.stage2_stop_after_checkpoint == 2
    assert args.max_steps == args.step_preference_config.phase_steps == 212
    assert args.ce_weight == 0.1
    assert args.step_preference_beta == 10
    assert args.step_preference_margin == 0.5


def tiny_trainer(output, stop=None, kind='dil_bce_rf'):
    torch.manual_seed(42)
    config = bce_config()
    if kind == 'dpo':
        config = replace(config, preference_loss_type='dpo', beta=10.,
                         preference_margin=.5, lambda_ce=.1)
    trainer = Stage2Trainer(
        model=tiny_stage2_qwen(),
        args=TrainingArguments(output_dir=str(output), use_cpu=True, report_to=[],
            max_steps=3, per_device_train_batch_size=1, learning_rate=.01,
            save_steps=3, save_total_limit=1, disable_tqdm=True, logging_steps=1),
        train_dataset=[{'input_ids': torch.tensor([index])} for index in range(6)],
        step_preference_config=config,
        **({'stage2_stop_after_checkpoint': stop} if stop is not None else {}),
    )
    trainer.model.config.supervise_latent_boundaries = True
    trainer.model_accepts_loss_kwargs = False
    seen = []
    def loss(model, inputs, **kwargs):
        seen.append(int(inputs['input_ids'].item()))
        # Tiny differentiable CPU objective isolates lifecycle from image/sidecar work.
        return model.get_input_embeddings().weight.square().mean() * (1 + torch.rand(()))
    trainer.compute_loss = loss
    return trainer, seen


@pytest.mark.parametrize('kind', ['dil_bce_rf', 'dpo'])
def test_actual_trainer_saves_before_normal_stop_then_restores_and_updates(tmp_path, kind):
    # Missing callback, stop-before-save, rewritten max_steps, or skipped restore
    # each breaks this witness. All fixture model state is under checkpoints/.
    trainer, seen = tiny_trainer(tmp_path / 'split', 2, kind)
    saved_events = []
    class Persisted(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            payload = json.loads((checkpoint / 'trainer_state.json').read_text())
            for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'deepspeed-runtime.json'):
                assert (checkpoint / name).is_file(), name
            saved_events.append((state.global_step, payload['max_steps'], control.should_training_stop))
    trainer.add_callback(Persisted())
    result = trainer.train()
    assert result.global_step == trainer.state.global_step == 2
    assert trainer.args.max_steps == trainer.state.max_steps == 3
    assert saved_events == [(2, 3, True)]
    checkpoint = tmp_path / 'split/checkpoint-2'
    saved_rng = torch.load(checkpoint / 'rng_state.pth', weights_only=False)['cpu']
    saved_optimizer = torch.load(checkpoint / 'optimizer.pt', weights_only=False)
    saved_scheduler = torch.load(checkpoint / 'scheduler.pt', weights_only=False)
    before = trainer.model.get_input_embeddings().weight.detach().clone()

    resumed, continued = tiny_trainer(tmp_path / 'split', kind=kind)
    restored = []
    load_optimizer = resumed._load_optimizer_and_scheduler
    def observe_optimizer(path):
        load_optimizer(path)
        assert resumed.lr_scheduler.state_dict() == saved_scheduler
        state = resumed.optimizer.state_dict()
        assert state['param_groups'] == saved_optimizer['param_groups']
        for key, expected in saved_optimizer['state'].items():
            for field, value in expected.items():
                torch.testing.assert_close(state['state'][key][field], value)
        restored.append('optimizer+scheduler')
    resumed._load_optimizer_and_scheduler = observe_optimizer
    load_rng = resumed._load_rng_state
    def observe_rng(path):
        load_rng(path)
        assert torch.equal(torch.get_rng_state(), saved_rng)
        restored.append('rng')
    resumed._load_rng_state = observe_rng
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert restored == ['optimizer+scheduler', 'rng']
    assert resumed.state.global_step == resumed.state.max_steps == 3
    assert len(continued) == 1
    assert not torch.equal(resumed.model.get_input_embeddings().weight, before)
    full, full_seen = tiny_trainer(tmp_path / 'full', kind=kind)
    full.train()
    assert seen + continued == full_seen
    torch.testing.assert_close(resumed.model.get_input_embeddings().weight, full.model.get_input_embeddings().weight, rtol=0, atol=0)
    assert json.loads((tmp_path / 'split/checkpoint-3/trainer_state.json').read_text())['max_steps'] == 3
    print({'save_events': saved_events, 'restore_order': restored, 'split_sample_order': seen + continued,
           'uninterrupted_sample_order': full_seen, 'global_step': resumed.state.global_step})


def test_planned_stop_rejects_conflicting_coordinator_and_resolved_budgets():
    with pytest.raises(ValueError, match='max_steps'):
        parse_args(fresh_argv() + ['--stage2_total_steps', '4', '--stage2_stop_after_checkpoint', '2'])


@pytest.mark.parametrize('stop', [True, False, 0, 1.5, '2', 3])
def test_trainer_api_rejects_noninteger_or_out_of_budget_stop(tmp_path, stop):
    with pytest.raises(ValueError, match='stage2_stop_after_checkpoint'):
        tiny_trainer(tmp_path, stop)
