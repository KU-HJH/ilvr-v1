"""Tail settings must reach both worker parsers through the real coordinator."""
import json
import os
import shlex
import subprocess

import pytest

from tests.stage1.v5.test_launcher import ROOT, REMOTE, HOST, PROFILE
from utils_deepseed import parse_args


def test_tail_only_coordinator_to_both_worker_parsers():
    profile = list(PROFILE)
    profile[profile.index('--lambda-topk-entropy')+1] = '0'
    profile += ['--lambda-tail-kl', '.7', '--tail-kl-temperature', '2']
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'CUDA_VISIBLE_DEVICES':''}
    result = subprocess.run([str(ROOT/'run_multinode_stage1_training.sh'),
        '--run-id','tail-kl-command-test','--zero-stage','1','--main-process-port','29683',
        '--local-repo-root',str(ROOT),'--remote-repo-root',str(REMOTE),
        '--remote-host-repo-root',str(HOST),'--dry-run','--method','dora',
        '--model','/original/base','--data-path',str(ROOT/'data/TRAIN.jsonl'),
        '--image-root',str(ROOT/'data/images_comt'),*profile],
        text=True,capture_output=True,env=environment)
    assert result.returncode == 0, result.stderr
    commands = json.loads(result.stdout)
    assert commands['plan']['world_size'] == 4
    remote = shlex.split(shlex.split(commands['remote'][-1])[-1])
    for rank, owned in ((0, remote), (1, commands['local'])):
        cursor = owned.index('-i') + 2
        worker_environment = {}
        while '=' in owned[cursor] and not owned[cursor].startswith('--'):
            name, value = owned[cursor].split('=', 1)
            worker_environment[name] = value
            cursor += 1
        stage = owned[max(i for i, value in enumerate(owned) if value == '--')+1:]
        worker = subprocess.run(['bash',str(ROOT/'run_stage1_training.sh'),*stage[2:],'--dry-run'],
            env={**environment,**worker_environment},text=True,capture_output=True)
        assert worker.returncode == 0, worker.stderr
        argv = shlex.split(worker.stdout)
        assert argv[argv.index('--machine_rank')+1] == str(rank)
        args = parse_args(argv[argv.index('src/main.py')+1:])
        assert args.stage1_v5_config.lambda_tail_kl == .7
        assert args.stage1_v5_config.tail_kl_temperature == 2.
        assert args.stage1_v5_config.lambda_topk_entropy == args.stage1_v5_config.lambda_topk_mass == 0


@pytest.mark.parametrize('stage', ['stage1', 'stage2'])
def test_tail_options_do_not_silently_apply_outside_v5(stage):
    with pytest.raises(ValueError, match='v5'):
        parse_args(['--stage',stage,'--lambda_tail_kl','0'])


def test_tail_resume_worker_inherits_settings_and_rejects_explicit_drift(tmp_path):
    from stage1.v5_config import Stage1V5Config
    from stage1.v5_reference import V5ReferenceIdentity
    from stage1.v5_resume import build_v5_contract, V5_CONTRACT_FILENAME
    saved = build_v5_contract(
        Stage1V5Config(lambda_tail_kl=.7, tail_kl_temperature=2., preference_topk_k=50),
        V5ReferenceIdentity(1, 'a'*64, 'b'*64, 'c'*64, ('weight',)),
        ce_weight=.1, sim_weight=.1,
        schedule={'dil_denominator': 'actual_window_eligible_dil_tokens'}, execution={})
    (tmp_path / V5_CONTRACT_FILENAME).write_text(json.dumps(saved))
    argv = ['--stage', 'stage1', '--resume_from_checkpoint_path', str(tmp_path)]
    args = parse_args(argv)
    assert args.stage1_variant == 'v5'
    assert args.lambda_tail_kl == args.stage1_v5_config.lambda_tail_kl == .7
    assert args.tail_kl_temperature == args.stage1_v5_config.tail_kl_temperature == 2.
    for option, value in [('--lambda_tail_kl', '0'), ('--tail_kl_temperature', '1')]:
        with pytest.raises(ValueError, match='conflict'):
            parse_args([*argv, option, value])
