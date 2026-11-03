"""Effective optimizer mode/weight identity through CLI, environment and resume."""
import json
import os
import shlex
import subprocess
from pathlib import Path
import pytest
from utils_deepseed import parse_args
from stage1.v3_config import Stage1V3Config
from stage1.v3_resume import (V3_FRESH_PROFILE, build_v3_checkpoint_contract,
    build_v3_schedule_identity, resolve_coordinator_arguments)
from src.multinode_runtime import validate_stage1_arguments

ROOT = Path(__file__).resolve().parents[2]


def saved(tmp_path, *, legacy=False):
    config = Stage1V3Config() if legacy else Stage1V3Config(gain_update_mode='adamw_projected', gain_optimizer_weight=0.)
    schedule = build_v3_schedule_identity(retained_ids=('a',), sampler_identity={'world_size':4},
        microsteps_per_rank=1, accumulation_steps=4)
    contract = build_v3_checkpoint_contract(config, schedule, dict(V3_FRESH_PROFILE, frozen_teacher_sha256='a'*64))
    if legacy:
        contract['config'].pop('gain_update_mode')
        contract['config'].pop('gain_optimizer_weight')
    path = tmp_path/'checkpoint-1'
    path.mkdir()
    (path/'stage1_v3.json').write_text(json.dumps(contract, indent=2))
    return path


def test_cli_wins_over_fresh_environment_and_zero_is_not_missing(monkeypatch):
    monkeypatch.setenv('STAGE1_V3_GAIN_UPDATE_MODE','adamw_projected')
    monkeypatch.setenv('STAGE1_V3_GAIN_OPTIMIZER_WEIGHT','.75')
    fresh = parse_args(['--stage1_variant','v3'])
    assert fresh.stage1_v3_gain_update_mode == 'adamw_projected'
    assert fresh.stage1_v3_gain_optimizer_weight == .75
    explicit = parse_args(['--stage1_variant','v3','--stage1_v3_gain_optimizer_weight=0'])
    assert explicit.stage1_v3_gain_optimizer_weight == 0.


@pytest.mark.parametrize('legacy',[False,True])
def test_resume_ignores_stale_environment_and_keeps_original_bytes(tmp_path,monkeypatch,legacy):
    checkpoint = saved(tmp_path,legacy=legacy)
    before = (checkpoint/'stage1_v3.json').read_bytes()
    monkeypatch.setenv('STAGE1_V3_GAIN_UPDATE_MODE','invalid-stale-setting')
    monkeypatch.setenv('STAGE1_V3_GAIN_OPTIMIZER_WEIGHT','invalid-stale-weight')
    args = parse_args(['--resume_from_checkpoint_path',str(checkpoint)])
    assert args.stage1_v3_gain_update_mode == ('post_adam' if legacy else 'adamw_projected')
    assert args.stage1_v3_gain_optimizer_weight == (None if legacy else 0.)
    with pytest.raises(ValueError,match='resume|mismatch'):
        parse_args(['--resume_from_checkpoint_path',str(checkpoint),'--stage1_v3_gain_optimizer_weight','.25'])
    assert (checkpoint/'stage1_v3.json').read_bytes() == before


@pytest.mark.parametrize('extra',[[],['--stage1_variant','standard'],['--stage','stage2']])
def test_explicit_optimizer_options_rejected_outside_v3(extra):
    with pytest.raises(ValueError,match='Stage1-v3'):
        parse_args([*extra,'--stage1_v3_gain_update_mode','adamw_projected','--stage1_v3_gain_optimizer_weight','0'])


def test_fresh_default_and_explicit_legacy_mode():
    args = parse_args(['--stage1_variant','v3'])
    assert args.stage1_v3_gain_update_mode == 'adamw_projected'
    assert args.stage1_v3_gain_optimizer_weight == 10.
    explicit = parse_args(['--stage1_variant','v3','--stage1_v3_gain_update_mode','adamw_projected'])
    assert explicit.stage1_v3_gain_optimizer_weight == 10.
    legacy = parse_args(['--stage1_variant','v3','--stage1_v3_gain_update_mode','post_adam'])
    assert legacy.stage1_v3_gain_optimizer_weight is None
    with pytest.raises(ValueError,match='weight'):
        parse_args(['--stage1_variant','v3','--stage1_v3_gain_update_mode','post_adam',
                    '--stage1_v3_gain_optimizer_weight','10'])


@pytest.mark.parametrize('inline',[False,True])
def test_coordinator_resolves_mode_and_never_serializes_none(tmp_path,inline):
    pairs = [('--stage1-variant','v3'),('--stage1-v3-gain-update-mode','adamw_projected'),
             ('--stage1-v3-gain-optimizer-weight','0')]
    argv = tuple(f'{k}={v}' for k,v in pairs) if inline else tuple(x for pair in pairs for x in pair)
    resolved = resolve_coordinator_arguments(validate_stage1_arguments(argv),tmp_path/'out')
    assert resolved[resolved.index('--stage1-v3-gain-update-mode')+1] == 'adamw_projected'
    assert resolved[resolved.index('--stage1-v3-gain-optimizer-weight')+1] == '0.0'
    fresh = resolve_coordinator_arguments(('--stage1-variant','v3'),tmp_path/'out')
    assert fresh[fresh.index('--stage1-v3-gain-optimizer-weight')+1] == '10.0'
    legacy = resolve_coordinator_arguments(('--stage1-variant','v3','--stage1-v3-gain-update-mode','post_adam'),tmp_path/'out')
    assert '--stage1-v3-gain-optimizer-weight' not in legacy


@pytest.mark.parametrize('inline',[False,True])
def test_shell_dry_run_passes_optimizer_flags_without_launch(inline):
    pairs=[('--stage1-v3-gain-update-mode','adamw_projected'),('--stage1-v3-gain-optimizer-weight','.25')]
    extra=[f'{k}={v}' for k,v in pairs] if inline else [x for p in pairs for x in p]
    env=dict(os.environ,ILVR_NUM_MACHINES='2',ILVR_NUM_PROCESSES='4',ILVR_MACHINE_RANK='1',
        ILVR_MAIN_PROCESS_IP='172.16.11.41',ILVR_MAIN_PROCESS_PORT='29579',ILVR_SOCKET_IFNAME='bond0.11')
    result=subprocess.run([str(ROOT/'run_stage1_training.sh'),'--dry-run','--stage1-variant','v3','--method','dora',
        '--model','models/Qwen2.5-VL-3B-Instruct','--data-path',str(ROOT/'data/TRAIN.jsonl'),
        '--image-root',str(ROOT/'data/images'),'--output-dir',str(ROOT/'checkpoints/v3-optimizer-dry'),*extra],
        env=env,cwd=ROOT,text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
    argv=shlex.split(result.stdout)
    assert argv[argv.index('--stage1_v3_gain_update_mode')+1] == 'adamw_projected'
    assert argv[argv.index('--stage1_v3_gain_optimizer_weight')+1] == '0.25'
