"""Independent arithmetic and fail-closed comparison contracts for reproduction."""
import importlib.util
from pathlib import Path

import pytest
import torch


def probe():
    path = Path(__file__).resolve().parents[2] / 'scripts/verify_voff_reproduction.py'
    assert path.is_file(), 'bounded reproduction witness is missing'
    spec = importlib.util.spec_from_file_location('voff_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_means_do_not_weight_longer_examples_or_more_steps():
    # p=2: first CE=4, preference=6; second CE=16, preference=2.
    # Objective=14, gradient=(4+3+16+1)/2=12, independent of token counts.
    p = torch.tensor(2.0, requires_grad=True)
    value = probe().reference_objective([
        (torch.stack([p*p]), torch.stack([p, 5*p])),
        (torch.stack([4*p*p]*3), torch.stack([p])),
    ])
    assert value.item() == 14
    assert torch.autograd.grad(value, p)[0].item() == 12


def test_partial_group_uses_actual_examples_and_rejects_wrong_divisors():
    p = torch.tensor(2.0, requires_grad=True)
    value = probe().reference_objective([(torch.stack([p*p]), torch.stack([3*p]))])
    assert value.item() == 10
    grad = torch.autograd.grad(value, p)[0]
    assert grad.item() == 7
    assert probe().gradient_comparison(grad, torch.tensor(7.0))['pass']
    for wrong in (torch.tensor(3.5), torch.tensor(14.0), torch.tensor(1.75)):
        assert not probe().gradient_comparison(grad, wrong)['pass']


def test_nonfinite_or_missing_gradients_cannot_pass():
    for actual in (None, torch.tensor(float('nan')), torch.tensor(float('inf'))):
        assert not probe().gradient_comparison(torch.tensor(1.), actual)['pass']


def test_empty_objective_cannot_silently_be_zero():
    with pytest.raises(ValueError):
        probe().reference_objective([])


def test_comparison_catches_tokens_targets_and_replay_mismatches():
    original = {'samples': [{'sample_id': 'a', 'input_ids': [1, 2], 'branches': [{'targets': [2], 'positions': [0]}]}]}
    changed = {'samples': [{'sample_id': 'a', 'input_ids': [1, 2], 'branches': [{'targets': [9], 'positions': [1]}]}]}
    assert probe().compare_records(original['samples'], original['samples']) == []
    assert probe().compare_records(original['samples'], changed['samples'])


def test_current_worker_collates_actual_train_with_supported_rendering():
    # Catches invalid rendering selection and broken normalization at real boundaries.
    import json
    import os
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=f'{repo}/src:{repo}/transformers/src')
    result = subprocess.run([sys.executable, str(repo/'scripts/verify_voff_reproduction.py'),
        '--worker', 'current', '--source', str(repo)], cwd=repo, env=env,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert len(payload['samples']) == 3
    assert payload['common_logits']['gradient_finite']
    assert payload['common_logits']['clean_gradient_nonzero']
    assert payload['common_logits']['ablated_gradient_nonzero']
    assert any(x['latent_blocks'] == 1 for x in payload['samples'])
    assert any(x['latent_blocks'] > 1 for x in payload['samples'])
    assert payload['parent_scaling']['groups'][0]['comparison']['pass']
    assert not payload['parent_scaling']['groups'][1]['comparison']['pass']


def test_external_partial_snapshot_worker_reports_real_comparison():
    import json
    import os
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    external = Path('/raid/workspace/ILVR/ILVR/.worktrees/ours1/source')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=f'{external}/src:{repo}/transformers/src')
    result = subprocess.run([sys.executable, str(repo/'scripts/verify_voff_reproduction.py'),
        '--worker', 'external', '--source', str(external)], cwd=repo, env=env,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert len(payload['samples']) == 3
    assert payload['common_logits']['gradient_finite']
    assert payload['common_logits']['clean_gradient_nonzero']
    assert payload['common_logits']['ablated_gradient_nonzero']
    assert payload['parent_scaling']['groups'][0]['observed_gradient'] == 10
    assert payload['parent_scaling']['groups'][1]['observed_gradient'] == 1
    assert str(external) in payload['parent_scaling']['source']
    assert payload['shared_dependency_hashes']


def test_external_worker_does_not_write_copied_source(tmp_path):
    # The RED fixture is a copy: never deliberately mutate external evidence.
    import os
    import shutil
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    copied = tmp_path/'source'
    shutil.copytree('/raid/workspace/ILVR/ILVR/.worktrees/ours1/source', copied,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    before = {str(p.relative_to(copied)): (p.stat().st_size,p.stat().st_mtime_ns)
              for p in copied.rglob('*') if p.is_file()}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=f'{copied}/src:{repo}/transformers/src')
    env.pop('PYTHONDONTWRITEBYTECODE',None)
    result = subprocess.run([sys.executable, str(repo/'scripts/verify_voff_reproduction.py'),
        '--worker','external','--source',str(copied)],cwd=repo,env=env,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    after = {str(p.relative_to(copied)): (p.stat().st_size,p.stat().st_mtime_ns)
             for p in copied.rglob('*') if p.is_file()}
    assert after == before
