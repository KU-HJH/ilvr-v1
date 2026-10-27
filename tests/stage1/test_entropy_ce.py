"""Causal CE selection and actual standard-trainer backward witnesses."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from stage1.trainer import CustomTrainerStage1
from utils_deepseed import parse_args


def select(*args, **kwargs):
    assert importlib.util.find_spec('stage1.entropy_ce'), 'entropy CE selector missing'
    from stage1.entropy_ce import select_entropy_ce_labels
    return select_entropy_ce_labels(*args, **kwargs)


def test_causal_top20_preserves_specials_and_ignored_targets():
    """Catches ranking t instead of t-1, wrong quota, or losing special CE."""
    logits = torch.zeros(1, 10, 12, requires_grad=True)
    with torch.no_grad():
        logits[0, :, 0] = 8
        logits[0, 2] = 0  # predictor of target3: unique maximal entropy
        logits[0, 4] = 0  # ignored target5 must not consume quota
    labels = torch.tensor([[-100, 1, 2, 3, 4, -100, 5, 10, 11, -100]])
    original = labels.clone()
    selected = select(logits, labels, percent=20, special_token_ids=(10, 11), chunk_positions=2)
    assert selected.tolist() == [[-100, -100, -100, 3, -100, -100, -100, 10, 11, -100]]
    assert torch.equal(labels, original) and not selected.requires_grad
    outputs = SimpleNamespace(logits=logits, loss=None)
    ce = CustomTrainerStage1._equal_sample_ce_loss(outputs, selected, None, recompute=True)
    manual = F.cross_entropy(logits[0, [2, 6, 7]], labels[0, [3, 7, 8]])
    torch.testing.assert_close(ce, manual)
    ce.backward()
    assert logits.grad[0, [2, 6, 7]].abs().sum().item() > 0
    assert logits.grad[0, [0, 1, 3, 4, 5, 8, 9]].abs().sum().item() == 0


def test_quota_is_per_sample_ceil_with_original_position_ties():
    """Catches batch-wide mining and unstable selection across entropy chunks."""
    logits = torch.zeros(2, 8, 12)
    labels = torch.tensor([[-100, 1, 2, 3, 4, 5, 6, 11], [-100, 1, 2, 3, 4, 5, -100, 11]])
    selected = select(logits, labels, percent=20, special_token_ids=(11,), chunk_positions=2)
    assert selected.tolist() == [[-100, 1, 2, -100, -100, -100, -100, 11],
                                 [-100, 1, -100, -100, -100, -100, -100, 11]]


def test_empty_text_pool_preserves_special_ce_and_all_ignored_fails():
    """Catches NaN reduction or discarding a special-only sample."""
    logits = torch.randn(1, 4, 12, requires_grad=True)
    labels = torch.tensor([[-100, 10, -100, 11]])
    selected = select(logits, labels, percent=20, special_token_ids=(10, 11))
    assert torch.equal(selected, labels)
    outputs = SimpleNamespace(logits=logits, loss=None)
    with pytest.raises(RuntimeError, match='supervised'):
        CustomTrainerStage1._equal_sample_ce_loss(outputs, labels.new_full(labels.shape, -100), None, recompute=True)


@pytest.mark.parametrize('batch', [1, 2])
def test_full_selection_matches_manual_equal_sample_ce_and_gradient(batch):
    """Catches normalization by sequence length or pooling unequal sample lengths."""
    torch.manual_seed(42)
    logits = torch.randn(batch, 8, 12, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3, 4, 5, 6, 11], [-100, 1, 2, -100, -100, -100, -100, 11]])[:batch]
    selected = select(logits, labels, percent=100, special_token_ids=(11,))
    assert torch.equal(selected, labels)
    outputs = SimpleNamespace(logits=logits, loss=None)
    actual = CustomTrainerStage1._equal_sample_ce_loss(outputs, selected, None, recompute=True)
    manual = torch.stack([F.cross_entropy(logits[i, :-1], labels[i, 1:]) for i in range(batch)]).mean()
    torch.testing.assert_close(actual, manual)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                               torch.autograd.grad(manual, logits)[0])


@pytest.mark.parametrize('percent', [0, -1, 101, float('nan'), float('inf')])
def test_worker_rejects_invalid_selection_percent(percent):
    """Catches silently coercing invalid objectives to an empty/all selection."""
    with pytest.raises(ValueError, match='percent'):
        parse_args(['--stage', 'stage1', '--stage1_match_v5_execution',
                    '--stage1_ce_entropy_top_percent', str(percent)])


@pytest.mark.parametrize('scope', [[], ['--stage', 'stage2'], ['--stage', 'stage1', '--stage1_variant', 'v5']])
def test_selection_requires_standard_matched_stage1(scope):
    """Catches using the new loss under a different algorithm/execution identity."""
    with pytest.raises(ValueError, match='matched standard Stage1'):
        parse_args([*scope, '--stage1_ce_entropy_top_percent', '20'])


def test_checkpoint_identity_records_loss_policy_and_detects_conflict(tmp_path):
    """Catches producing a clean20 checkpoint labelled as full-CE training."""
    from stage1.matched_execution import write_matched_execution_identity, MATCHED_EXECUTION_FILENAME
    write_matched_execution_identity(tmp_path, ce_entropy_top_percent=20)
    policy = json.loads((tmp_path / MATCHED_EXECUTION_FILENAME).read_text())['ce_entropy_selection']
    assert policy == dict(percent=20, source='clean_student_logits', temperature=1,
        candidate_pool='supervised_non_special_text', quota='sample_local_ceil',
        ties='original_position', special_targets='preserve', reduction='retained_label_sample_mean')
    with pytest.raises(ValueError, match='conflicting'):
        write_matched_execution_identity(tmp_path)


def test_constructor_uses_processor_special_ids_and_enforces_matched_mode(monkeypatch):
    """Catches losing real tokenizer specials or allowing selector in legacy trainer."""
    from trl import SFTTrainer
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    def parent_init(self, **kwargs):
        self.model = torch.nn.Linear(1, 1)
        self.processing_class = SimpleNamespace(tokenizer=SimpleNamespace(all_special_ids=[10, 11]))
    monkeypatch.setattr(SFTTrainer, '__init__', parent_init)
    monkeypatch.setattr(CustomTrainerStage1, 'add_callback', lambda *args: None)
    kwargs = dict(deepspeed_runtime_profile=DeepSpeedRuntimeProfile(True, 1, None, None, 4),
                  ema_teacher=object(), target_builder=object(), ce_entropy_top_percent=20)
    with pytest.raises(ValueError, match='matched standard Stage1'):
        CustomTrainerStage1(**kwargs)
    trainer = CustomTrainerStage1(**kwargs, match_v5_execution=True)
    assert trainer.ce_entropy_top_percent == 20
    assert trainer.ce_entropy_special_token_ids == (10, 11)


@pytest.fixture
def owned_accelerate_state():
    """Release the state created by this test without masking earlier leaks."""
    from accelerate.state import AcceleratorState, GradientState, PartialState
    assert not AcceleratorState._shared_state
    assert not PartialState._shared_state
    assert not GradientState._shared_state
    try:
        yield
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
        PartialState._reset_state()
        GradientState._reset_state()


def test_real_standard_parent_backward_reuses_one_student_forward(tmp_path, owned_accelerate_state):
    """Catches selector bypass, extra forwards, changed alignment/context or no backward."""
    from accelerate import Accelerator
    from stage1.equivalence import stage1_checkpoint_context_fn
    from tests.stage1.v5.test_trainer import integrated_fixture
    from tests.stage1.v5.test_replay import digest_state
    trainer, batch, calls, forwards, _, _ = integrated_fixture(tmp_path)
    trainer.__class__ = CustomTrainerStage1
    trainer.stage1_match_v5_execution = True
    trainer.ce_weight = trainer.sim_weight = 1
    trainer.ce_entropy_top_percent = 20
    trainer.ce_entropy_special_token_ids = (125, 126, 127)
    trainer.collect_stage1_diagnostics = True
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        'use_reentrant': False, 'context_fn': stage1_checkpoint_context_fn})
    trainer.accelerator.backward = Accelerator.backward.__get__(trainer.accelerator)
    batch.pop('stage1_v5_metadata')
    original_labels = batch['labels'].clone()
    captured = []
    trainer.model.register_forward_hook(lambda _m, _a, out: captured.append(out))
    before = digest_state(trainer.ema_teacher.model)
    rng_before = torch.random.get_rng_state()
    loss = trainer.training_step(trainer.model, batch)
    assert calls == ['helper', 'F']
    assert 'labels' not in forwards[1]  # no unused native full-token CE
    assert torch.equal(batch['labels'], original_labels)
    out = captured[0]
    logits = out.logits.detach()
    candidates = [i for i in range(1, 26) if original_labels[0, i].item() not in (125, 126, 127)]
    log_probs = logits[0].float().log_softmax(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    chosen = sorted(candidates, key=lambda i: (-entropy[i-1].item(), i))[:(len(candidates)+4)//5]
    active = sorted(chosen + [i for i in range(1,26) if original_labels[0,i].item() in (125,126,127)])
    manual = F.cross_entropy(logits[0, [i-1 for i in active]].float(), original_labels[0, active])
    torch.testing.assert_close(trainer._latest_stage1_diagnostics['ce_loss'], manual)
    diag = trainer._latest_stage1_diagnostics
    torch.testing.assert_close(loss, (diag['ce_loss'] + diag['alignment_loss']) / 4)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.model.parameters() if p.requires_grad)
    assert digest_state(trainer.ema_teacher.model) == before
    trainer.model.zero_grad(set_to_none=True)
    trainer.ce_entropy_top_percent = None
    torch.random.set_rng_state(rng_before)
    trainer.training_step(trainer.model, batch)
    assert calls == ['helper', 'F', 'helper', 'F']
    torch.testing.assert_close(captured[1].logits, logits)
    torch.testing.assert_close(trainer._latest_stage1_diagnostics['alignment_loss'], diag['alignment_loss'])


def test_coordinator_percent_reaches_both_worker_parsers():
    """Catches shell/coordinator acceptance that silently loses the objective."""
    import os
    import shlex
    import subprocess
    from tests.stage1.v5.test_launcher import ROOT, REMOTE, HOST
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''}
    run = subprocess.run([str(ROOT/'run_multinode_stage1_training.sh'),
        '--run-id', 'clean20-ce-command-test', '--zero-stage', '1', '--main-process-port', '29683',
        '--local-repo-root', str(ROOT), '--remote-repo-root', str(REMOTE), '--remote-host-repo-root', str(HOST),
        '--dry-run', '--stage1-match-v5-execution', '--stage1-ce-entropy-top-percent', '20',
        '--ce-weight','1','--sim-weight','1', '--method','dora', '--model','/original/base',
        '--data-path',str(ROOT/'data/TRAIN.jsonl'), '--image-root',str(ROOT/'data/images_comt')],
        cwd=ROOT, env=env, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    commands = json.loads(run.stdout)
    remote = shlex.split(shlex.split(commands['remote'][-1])[-1])
    for owned in (remote, commands['local']):
        cursor = owned.index('-i') + 2
        worker_env = {}
        while '=' in owned[cursor] and not owned[cursor].startswith('--'):
            name,value=owned[cursor].split('=',1);worker_env[name]=value;cursor+=1
        stage = owned[max(i for i,v in enumerate(owned) if v=='--')+1:]
        dry = subprocess.run(['bash', str(ROOT/'run_stage1_training.sh'), *stage[2:], '--dry-run'],
            cwd=ROOT, env={**env, **worker_env}, text=True, capture_output=True)
        assert dry.returncode == 0, dry.stderr
        argv = shlex.split(dry.stdout)
        parsed = parse_args(argv[argv.index('src/main.py')+1:])
        assert parsed.stage1_ce_entropy_top_percent == 20
        assert parsed.stage1_match_v5_execution and parsed.ce_weight == parsed.sim_weight == 1
