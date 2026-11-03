"""Fresh runtime defaults must not reinterpret any saved objective."""
from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

import pytest
import torch

from stage1.v3_config import Stage1V3Config
from stage1.v3_metadata import V3Block
from stage1 import v3_objective as objective, v3_resume as resume
from utils_deepseed import parse_args

ROOT = Path(__file__).resolve().parents[2]


def resolved(args):
    return Stage1V3Config(**args.stage1_v3_resolved_config)


@pytest.fixture(autouse=True)
def clean_fresh_environment(monkeypatch):
    for key in ('STAGE1_VARIANT', 'STAGE1_V3_GAIN_UPDATE_MODE', 'STAGE1_V3_GAIN_OPTIMIZER_WEIGHT'):
        monkeypatch.delenv(key, raising=False)


def old_checkpoint(tmp_path, weighting='token_entropy', mode='adamw_projected', weight=.25,
                   ratio=.05, omit_modes=False):
    # Explicit historical values, independent of the new fresh-run defaults.
    config = Stage1V3Config(text_selector='entropy', gain_weighting=weighting,
        gain_update_mode=mode, gain_optimizer_weight=weight, text_kl_warmup_ratio=ratio)
    schedule = resume.build_v3_schedule_identity(retained_ids=('a', 'b'),
        sampler_identity={'world_size': 4}, microsteps_per_rank=6, accumulation_steps=4)
    contract = resume.build_v3_checkpoint_contract(config, schedule,
        dict(resume.V3_FRESH_PROFILE, sim_weight=1., frozen_teacher_sha256='a' * 64))
    contract['config'].pop('gain_entropy_mode', None)
    if omit_modes:
        for name in ('gain_weighting', 'gain_update_mode', 'gain_optimizer_weight', 'text_selector'):
            contract['config'].pop(name)
        contract['formula_ids']['selector'] = 'T2_KL_student_reference_sample_centered_full_causal_v1'
    path = tmp_path / 'checkpoint-1'
    path.mkdir()
    raw = json.dumps(contract, indent=3).encode()
    (path / resume.FILENAME).write_bytes(raw)
    return path, contract, raw


def test_new_runtime_defaults_apply_to_objective_and_first_step(tmp_path):
    args = parse_args(['--stage1_variant', 'v3', '--save_model_path', str(tmp_path / 'new')])
    assert args.sim_weight == .1
    assert args.stage1_v3_gain_weighting == 'token_entropy'
    assert args.stage1_v3_gain_entropy_mode == 'raw'
    assert args.stage1_v3_gain_update_mode == 'adamw_projected'
    assert args.stage1_v3_gain_optimizer_weight == 10.
    config = resolved(args)
    assert config.text_kl_warmup_ratio == 0.
    assert objective.text_kl_weight(0, 216, config) == .01
    assert objective.text_kl_weight(1, 216, config) == .01
    assert args.warm_up_steps == 10 and args.stage1_v3_selector == 'kl'


def test_raw_entropy_above_one_weights_gain_but_not_rv_protection():
    clean = torch.zeros(1, 2, 4, requires_grad=True)
    keys = ((0, 0), (0, 1))
    weights = objective.token_entropy(clean, keys, mode='raw', query_chunk_size=1)
    assert not weights.requires_grad
    torch.testing.assert_close(weights, torch.full((2,), math.log(4)), rtol=0, atol=0)
    assert (weights > 1).all()
    torch.testing.assert_close(objective.normalized_token_entropy(clean, keys), torch.ones(2), rtol=0, atol=0)
    r, rv = torch.zeros(2, 4, requires_grad=True), torch.zeros(2, 4, requires_grad=True)
    blocks = (V3Block(0, 1, 1, 3, (4, 5), (0, 1), ()),)
    gain, protection = objective.entropy_weighted_gain_and_rv_sums(r, rv, blocks, weights, entropy_mode='raw')
    g, rv_gain, f = torch.autograd.grad(gain, (r, rv, clean), allow_unused=True, retain_graph=True)
    h, = torch.autograd.grad(protection, (rv,))
    expected = torch.tensor([[.375, -.125, -.125, -.125], [-.125, .375, -.125, -.125]])
    torch.testing.assert_close(g, expected * math.log(4))
    torch.testing.assert_close(rv_gain, -expected * math.log(4))
    torch.testing.assert_close(h, expected)
    assert f is None


@pytest.mark.parametrize('mode,weight,weighting,ratio,omit', [
    ('post_adam', None, 'none', .05, True),
    ('post_adam', None, 'token_entropy', .05, False),
    ('adamw_projected', .25, 'token_entropy', .05, False),
    ('adamw_projected', 0., 'token_entropy', 0., False),
    ('adamw_projected', 2., 'none', .2, False),
])
@pytest.mark.parametrize('automatic', [False, True])
def test_legacy_resume_preserves_formula_schedule_and_bytes(tmp_path, monkeypatch, mode, weight, weighting, ratio, omit, automatic):
    checkpoint, old, raw = old_checkpoint(tmp_path, weighting, mode, weight, ratio, omit)
    monkeypatch.setenv('STAGE1_V3_GAIN_UPDATE_MODE', 'adamw_projected')
    monkeypatch.setenv('STAGE1_V3_GAIN_OPTIMIZER_WEIGHT', '999')
    argv = (['--save_model_path', str(tmp_path)] if automatic else
            ['--resume_from_checkpoint_path', str(checkpoint)])
    args = parse_args(argv)
    assert args.sim_weight == 1.
    assert args.stage1_v3_gain_weighting == weighting
    assert args.stage1_v3_gain_update_mode == mode and args.stage1_v3_gain_optimizer_weight == weight
    config = resolved(args)
    assert config.gain_entropy_mode == 'normalized'
    assert config.text_kl_warmup_ratio == ratio
    assert objective.text_kl_weight(0, 21, config) == (.01 if ratio == 0 else 0.)
    current = resume.build_v3_checkpoint_contract(config, resume.V3ScheduleIdentity(**old['schedule']), old['profile'])
    assert current['formula_ids'] == old['formula_ids']
    resume.validate_v3_resume(old, current)
    resume.write_v3_contract(checkpoint, current)
    assert (checkpoint / resume.FILENAME).read_bytes() == raw
    for changed in ('gain_entropy_mode', 'gain_optimizer_weight'):
        extra = ['--stage1_v3_' + changed, 'raw' if changed == 'gain_entropy_mode' else '10']
        with pytest.raises(ValueError, match='resume|mismatch'):
            parse_args([*argv, *extra])


def test_fresh_raw_contract_roundtrip_and_missing_mode_cannot_change_formula(tmp_path):
    args = parse_args(['--stage1_variant', 'v3', '--save_model_path', str(tmp_path / 'new')])
    config = resolved(args)
    schedule = resume.build_v3_schedule_identity(retained_ids=('a',), sampler_identity={},
        microsteps_per_rank=1, accumulation_steps=4)
    contract = resume.build_v3_checkpoint_contract(config, schedule,
        dict(resume.V3_FRESH_PROFILE, frozen_teacher_sha256='a' * 64))
    resume.write_v3_contract(tmp_path, contract)
    assert resume.read_v3_contract(tmp_path) == contract
    assert resume._formula_ids(replace(config, gain_entropy_mode='normalized'))['gain'] != contract['formula_ids']['gain']
    missing = deepcopy(contract)
    del missing['config']['gain_entropy_mode']
    with pytest.raises(ValueError, match='formula|inconsistent'):
        resume.validate_v3_resume(missing, missing)
    loaded = parse_args(['--resume_from_checkpoint_path', str(tmp_path)])
    assert resolved(loaded) == config and loaded.sim_weight == .1


@pytest.mark.parametrize('mode,expected', [('raw', math.log(4)), ('normalized', 1.)])
def test_worker_dry_run_and_actual_parser_agree(tmp_path, mode, expected):
    from src.multinode_runtime import validate_stage1_arguments
    flags = validate_stage1_arguments(('--stage1-variant=v3', '--stage1-v3-gain-entropy-mode=' + mode))
    flags = resume.resolve_coordinator_arguments(flags, tmp_path / 'out')
    env = dict(os.environ, ILVR_NUM_MACHINES='2', ILVR_NUM_PROCESSES='4', ILVR_MACHINE_RANK='1',
        ILVR_MAIN_PROCESS_IP='172.16.11.41', ILVR_MAIN_PROCESS_PORT='29579', ILVR_SOCKET_IFNAME='bond0.11')
    result = subprocess.run([str(ROOT / 'run_stage1_training.sh'), '--dry-run',
        '--data-path', str(ROOT / 'data/TRAIN.jsonl'), '--image-root', str(ROOT / 'data/images'),
        '--output-dir', str(tmp_path / 'out'), *flags], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    args = parse_args(argv[argv.index('src/main.py') + 1:])
    config = resolved(args)
    assert args.sim_weight == .1 and config.gain_optimizer_weight == 10.
    assert config.gain_entropy_mode == mode and objective.text_kl_weight(0, 216, config) == .01
    h = objective.token_entropy(torch.zeros(1, 1, 4), ((0, 0),), mode=config.gain_entropy_mode)
    torch.testing.assert_close(h, torch.tensor([expected]))


@pytest.mark.parametrize('extra', [[], ['--stage1_variant', 'standard'], ['--stage', 'stage2']])
def test_entropy_mode_rejected_outside_v3(extra):
    with pytest.raises(ValueError, match='Stage1-v3'):
        parse_args([*extra, '--stage1_v3_gain_entropy_mode', 'raw'])
