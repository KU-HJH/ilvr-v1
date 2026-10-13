"""Real CPU Stage1 save/stop/resume lifecycle; no ZeRO/Gain numerical claim."""
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig, Trainer, TrainerCallback, default_data_collator
from trl import SFTConfig

from deepspeed_runtime import DeepSpeedRuntimeProfile
from stage1.trainer import CustomTrainerStage1, STAGE1_EMA_TEACHER_STATE_FILENAME
from stage1.v3_config import Stage1V3Config
from stage1.v3_resume import V3_FRESH_PROFILE, resolve_coordinator_arguments
from src.multinode_runtime import validate_stage1_arguments
from tests.stage1.test_integration import TinyIntegrationProcessor
from tests.stage1.test_v3_optimizer_cli import saved
from utils_deepseed import parse_args

ROOT = Path(__file__).resolve().parents[2]
FLAG = '--stage1_v3_stop_after_checkpoint'
SHELL_FLAG = '--stage1-v3-stop-after-checkpoint'


def test_stop_is_invocation_only_and_preserves_full_profile():
    args = parse_args(['--stage1_variant', 'v3', '--no_final_save', FLAG, '2'])
    assert args.stage1_v3_stop_after_checkpoint == 2
    assert args.max_steps == -1 and args.epochs == 1
    assert 'stage1_v3_stop_after_checkpoint' not in V3_FRESH_PROFILE
    assert 'stop_after_checkpoint' not in vars(Stage1V3Config())
    assert parse_args(['--stage1_variant', 'v3']).stage1_v3_stop_after_checkpoint is None


@pytest.mark.parametrize('value', ['0', '-1', '2.5', 'true'])
def test_parser_rejects_invalid_stop(value):
    with pytest.raises((SystemExit, ValueError)):
        parse_args(['--stage1_variant', 'v3', '--no_final_save', FLAG, value])


@pytest.mark.parametrize('extra', [[], ['--stage1_variant', 'standard'], ['--stage', 'stage2']])
def test_stop_rejects_non_v3(extra):
    with pytest.raises(ValueError, match='Stage1-v3'):
        parse_args([*extra, '--no_final_save', FLAG, '2'])


def test_stop_requires_no_final_save():
    with pytest.raises(ValueError, match='no_final_save'):
        parse_args(['--stage1_variant', 'v3', FLAG, '2'])


@pytest.mark.parametrize('automatic', [False, True])
def test_parser_rejects_explicit_and_automatic_resume(tmp_path, automatic):
    checkpoint = saved(tmp_path)
    resume = (['--save_model_path', str(tmp_path)] if automatic else
              ['--resume_from_checkpoint_path', str(checkpoint)])
    with pytest.raises(ValueError, match='fresh'):
        parse_args([*resume, '--no_final_save', FLAG, '2'])


@pytest.mark.parametrize('inline', [False, True])
def test_worker_dry_run_transports_stop_and_keeps_derived_budget(tmp_path, inline):
    pairs = [('--stage1-variant', 'v3'), (SHELL_FLAG, '2')]
    argv = [f'{k}={v}' for k, v in pairs] if inline else [x for p in pairs for x in p]
    resolved = resolve_coordinator_arguments(validate_stage1_arguments((*argv, '--no-final-save')), tmp_path / 'out')
    assert resolved[resolved.index(SHELL_FLAG) + 1] == '2'
    assert '--max-steps' not in resolved
    env = dict(os.environ, ILVR_NUM_MACHINES='2', ILVR_NUM_PROCESSES='4', ILVR_MACHINE_RANK='1',
               ILVR_MAIN_PROCESS_IP='10.1.102.41', ILVR_MAIN_PROCESS_PORT='29579', ILVR_SOCKET_IFNAME='bond1.102')
    result = subprocess.run([str(ROOT / 'run_stage1_training.sh'), '--dry-run', *resolved,
        '--model', 'models/Qwen2.5-VL-3B-Instruct', '--data-path', str(ROOT / 'data/TRAIN.jsonl'),
        '--image-root', str(ROOT / 'data/images'), '--output-dir', str(tmp_path / 'out')],
        env=env, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[command.index(FLAG) + 1] == '2'
    assert command[command.index('--max_steps') + 1] == '-1'
    assert '--no_final_save' in command


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig(tie_word_embeddings=False)
        self.weight = nn.Parameter(torch.tensor(1.))

    def forward(self, input_ids, labels=None):
        return {'loss': self.weight.square() * input_ids.float().mean() * (1 + torch.rand(())),
                'logits': self.weight * input_ids.float()}


class CpuLifecycleTrainer(CustomTrainerStage1):
    """Replace only heavyweight objective; keep product constructor/save/resume/hooks."""
    def training_step(self, model, inputs, num_items_in_batch=None):
        self.occurrences.extend(inputs['input_ids'].flatten().tolist())
        return Trainer.training_step(self, model, inputs, num_items_in_batch)

    def compute_loss(self, *args, **kwargs):
        return Trainer.compute_loss(self, *args, **kwargs)


def tiny_trainer(output, stop=None, *, save_only_model=False, v3=True):
    trainer = CpuLifecycleTrainer(
        model=ScalarModel(), processing_class=TinyIntegrationProcessor(), target_builder=object(),
        data_collator=default_data_collator,
        deepspeed_runtime_profile=DeepSpeedRuntimeProfile(False, None, None, None, 1),
        ema_tau=1., v3_config=Stage1V3Config() if v3 else None,
        **({'stage1_v3_stop_after_checkpoint': stop} if stop is not None else {}),
        args=SFTConfig(output_dir=str(output), use_cpu=True, report_to=[], num_train_epochs=1,
            per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=.01,
            warmup_steps=10, save_steps=3, save_total_limit=1, save_only_model=save_only_model,
            disable_tqdm=True, logging_steps=1, dataset_kwargs={'skip_prepare_dataset': True}),
        train_dataset=[{'input_ids': torch.tensor([i + 1]), 'labels': torch.tensor([0])} for i in range(12)],
    )
    # CPU scalar objective isolates the new lifecycle from the separately validated
    # ZeRO-only Gain path. Constructor and save/callback/resume stay production.
    trainer.v3_gain_controller = None
    trainer.v3_resolved_profile = {'frozen_teacher_sha256': 'a' * 64}
    trainer.occurrences = []
    trainer.model_accepts_loss_kwargs = False
    return trainer


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_state_equal(left, right)
    else:
        assert a == b


def test_actual_save_finishes_before_stop_and_resume_matches_full_budget(tmp_path):
    split = tiny_trainer(tmp_path / 'split', 2)
    saves = []
    class ObserveSave(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                         STAGE1_EMA_TEACHER_STATE_FILENAME, 'stage1_v3.json', 'deepspeed-runtime.json'):
                assert (checkpoint / name).is_file(), name
            assert json.loads((checkpoint / 'stage1_v3.json').read_text())['planned_steps'] == 3
            saves.append((state.global_step, state.max_steps, control.should_training_stop))
    split.add_callback(ObserveSave())
    split.train()
    assert saves == [(2, 3, True)]
    assert split.state.global_step == 2 and split.v3_planned_steps == split.state.max_steps == 3
    checkpoint = tmp_path / 'split/checkpoint-2'
    assert split.model.weight.item() != 1., 'split must follow a nonzero-LR update'
    optimizer = torch.load(checkpoint / 'optimizer.pt', weights_only=False)
    scheduler = torch.load(checkpoint / 'scheduler.pt', weights_only=False)
    rng = torch.load(checkpoint / 'rng_state.pth', weights_only=False)['cpu']
    teacher = split.ema_teacher.model.state_dict()
    resumed = tiny_trainer(tmp_path / 'split')
    restored = []
    original_load = resumed._load_optimizer_and_scheduler
    def load(path):
        original_load(path)
        assert_state_equal(resumed.optimizer.state_dict(), optimizer)
        assert_state_equal(resumed.lr_scheduler.state_dict(), scheduler)
        restored.append('optimizer+scheduler')
    resumed._load_optimizer_and_scheduler = load
    original_rng = resumed._load_rng_state
    def load_rng(path):
        original_rng(path)
        assert torch.equal(torch.get_rng_state(), rng)
        restored.append('rng')
    resumed._load_rng_state = load_rng
    # Force a different teacher first; restoration must replace it.
    resumed.ema_teacher.model.weight.data.fill_(7.)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    full = tiny_trainer(tmp_path / 'full')
    full.train()
    assert restored == ['optimizer+scheduler', 'rng']
    assert resumed.state.global_step == resumed.state.max_steps == 3
    assert len(resumed.occurrences) == 4
    assert split.occurrences + resumed.occurrences == full.occurrences
    assert len(set(full.occurrences)) == 12
    assert_state_equal(resumed.model.state_dict(), full.model.state_dict())
    assert_state_equal(resumed.optimizer.state_dict(), full.optimizer.state_dict())
    assert_state_equal(resumed.lr_scheduler.state_dict(), full.lr_scheduler.state_dict())
    assert_state_equal(resumed.ema_teacher.model.state_dict(), teacher)
    assert not checkpoint.exists(), 'ordinary later checkpoint rotation remains active'


@pytest.mark.parametrize('stop', [True, False, 0, 1.5, '2', 3, 4])
def test_invalid_stop_fails_before_first_update(tmp_path, stop):
    with pytest.raises(ValueError, match='stop_after_checkpoint'):
        trainer = tiny_trainer(tmp_path, stop)
        before = trainer.model.weight.detach().clone()
        try:
            trainer.train()
        finally:
            assert trainer.state.global_step == 0
            assert torch.equal(before, trainer.model.weight)


def test_model_only_save_and_non_v3_api_rejected(tmp_path):
    with pytest.raises(ValueError, match='full checkpoint'):
        tiny_trainer(tmp_path / 'model-only', 2, save_only_model=True)
    with pytest.raises(ValueError, match='Stage1-v3'):
        tiny_trainer(tmp_path / 'standard', 2, v3=False)


def test_trainer_resume_alias_cannot_bypass_fresh_only(tmp_path):
    trainer = tiny_trainer(tmp_path, 2)
    for kwargs in ({'resume_from_checkpoint': True}, {'resume_from_checkpoint': 'missing'}, {'model_path': 'missing'}):
        with pytest.raises(ValueError, match='fresh'):
            trainer.train(**kwargs)
    assert trainer.state.global_step == 0


@pytest.mark.parametrize('failure', ['optimizer', 'ema', 'peer'])
def test_save_failure_never_becomes_successful_stop(tmp_path, monkeypatch, failure):
    trainer = tiny_trainer(tmp_path, 2)
    events = []
    class ObserveSave(TrainerCallback):
        def on_save(self, *args, **kwargs):
            events.append('save-finished')
    trainer.add_callback(ObserveSave())
    def fail(*args, **kwargs):
        raise OSError('injected save failure')
    if failure == 'optimizer':
        monkeypatch.setattr(trainer, '_save_optimizer_and_scheduler', fail)
    elif failure == 'ema':
        monkeypatch.setattr(trainer, '_save_stage1_ema_checkpoint', fail)
    else:
        original = trainer.accelerator.reduce
        def peer_failure(value, **kwargs):
            # Publication precedes training; inject at the checkpoint barrier only.
            return torch.tensor(1) if trainer.state.global_step == 2 else original(value, **kwargs)
        monkeypatch.setattr(trainer.accelerator, 'reduce', peer_failure)
    with pytest.raises((OSError, RuntimeError), match='save failed|save failure'):
        trainer.train()
    assert events == []
    assert not trainer.control.should_training_stop


@pytest.mark.parametrize('completed', [True, False])
def test_composition_emits_stop_only_after_observed_callback(tmp_path, completed):
    from types import SimpleNamespace
    from stage1.checkpointing import Stage1ResumeDecision
    from stage1.composition import run_stage1
    from tests.stage1.test_composition import RecordingTrainer, RecordingProcessor
    output = tmp_path / 'run'
    (output / 'checkpoint-2').mkdir(parents=True)
    producer = tmp_path / 'producer.json'
    trainer = RecordingTrainer(args=SimpleNamespace())
    trainer.state = SimpleNamespace(global_step=2, max_steps=3)
    trainer.v3_planned_steps = 3
    trainer.stage1_v3_planned_stop_callback = SimpleNamespace(completed_step=2 if completed else None)
    trainer.train = lambda **kw: SimpleNamespace(metrics={'train_loss': .125})
    profile = DeepSpeedRuntimeProfile(True, 1, ROOT / 'configs/deepspeed_zero1.json', 'a' * 64, 4)
    args = SimpleNamespace(final_save=False, stage1_v3_stop_after_checkpoint=2, stage1_variant='v3',
        save_model_path=str(output), training_evidence_path=str(producer), tuning_method='dora',
        per_device_train_batch_size=1, gradient_accumulation_steps=4, ema_tau=1., seed=42,
        benchmark_run_config=None, benchmark_evidence_path=None, deepspeed_runtime_profile=profile)
    bundle = SimpleNamespace(trainer=trainer, processor=RecordingProcessor(),
        resume_decision=Stage1ResumeDecision(None, None, False))
    def run():
        run_stage1(args, component_builder=lambda _: bundle, deepspeed_runtime_profile=profile,
            monotonic=iter((1., 2.)).__next__)
    if completed:
        run()
        assert json.loads(producer.read_text())['planned_stop'] == {
            'global_step': 2, 'max_steps': 3, 'budget_complete': False}
    else:
        with pytest.raises(ValueError, match='planned stop'):
            run()
        assert not producer.exists()


def test_coordinator_rejects_stop_in_benchmark_protocol_before_launch(tmp_path):
    with pytest.raises(ValueError, match='benchmark'):
        resolve_coordinator_arguments(('--stage1-variant', 'v3', '--no-final-save',
            SHELL_FLAG, '2', '--benchmark-run-config', 'benchmark.json'), tmp_path / 'out')


@pytest.mark.parametrize('flags,valid', [(['--final-save', '--no-final-save'], True),
                                        (['--no-final-save', '--final-save'], False)])
def test_coordinator_obeys_effective_final_save_flag(tmp_path, flags, valid):
    args = ('--stage1-variant', 'v3', SHELL_FLAG, '2', *flags)
    if valid:
        assert '--no-final-save' in resolve_coordinator_arguments(args, tmp_path / 'out')
    else:
        with pytest.raises(ValueError, match='no_final_save'):
            resolve_coordinator_arguments(args, tmp_path / 'out')
