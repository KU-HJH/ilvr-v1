"""The existing coordinator retains schedule and objective for a bounded resume."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_worker_accepts_a_same_budget_matched_resume_stop():
    from utils_deepseed import parse_args
    args = parse_args(['--stage','stage1','--stage1_match_v5_execution',
        '--ce_weight','1',
        '--stage1_ce_entropy_top_percent','20','--max_steps','2160',
        '--stage1_stop_after_checkpoint','217','--resume_from_checkpoint_path','/copied/checkpoint-216'])
    assert args.max_steps == 2160
    assert args.stage1_stop_after_checkpoint == 217


@pytest.mark.parametrize('flags', [[], ['--stage','stage2'],
    ['--stage','stage1','--stage1_match_v5_execution','--max_steps','217']])
def test_worker_refuses_an_unsupported_stop_scope(flags):
    from utils_deepseed import parse_args
    with pytest.raises(ValueError, match='matched|budget|stop'):
        parse_args([*flags,'--stage1_stop_after_checkpoint','217'])


def test_stop_requests_save_then_stops_after_state_publication():
    from transformers import TrainerControl
    from stage1 import matched_resume
    assert hasattr(matched_resume, 'MatchedCheckpointStop'), 'matched resume bounded stop is missing'
    stop = matched_resume.MatchedCheckpointStop(217)
    args = SimpleNamespace(max_steps=2160)
    state = SimpleNamespace(global_step=217)
    control = TrainerControl()
    stop.on_step_end(args, state, control)
    assert control.should_save and not control.should_training_stop
    stop.on_save(args, state, control)
    assert control.should_training_stop and args.max_steps == 2160


def test_ready_command_keeps_source_training_flags_and_moves_output_roots():
    from stage1 import resume_preparation
    assert hasattr(resume_preparation, 'resume_command'), 'usable resume command is missing'
    from src.multinode import build_launch_plan
    from tests.stage1.v5.test_launcher import ROOT, REMOTE, HOST
    plan = build_launch_plan(stage='stage1', tuning_method='dora', zero_stage=1,
        run_id='matched-resume-command', main_process_port=29931, attempt_id='resume1',
        local_repo_root=ROOT, remote_repo_root=REMOTE, remote_host_repo_root=HOST)
    source = Path('/root/old-runtime')
    command = [str(source/'run_multinode_stage1_training.sh'), '--run-id','original',
        '--zero-stage','1','--main-process-port','29927','--local-repo-root',str(source),
        '--remote-repo-root','/root/old-remote','--remote-host-repo-root','/old-host',
        '--method','dora','--model','/original/base','--data-path',str(source/'data/TRAIN.jsonl'),
        '--image-root',str(source/'data/images'),'--max-steps','2160','--save-steps','216',
        '--dataloader-num-workers','0','--stage1-match-v5-execution',
        '--stage1-ce-entropy-top-percent','20','--ce-weight','1','--sim-weight','1']
    checkpoint = ROOT/'checkpoints'/plan.run_id/'checkpoint-216'
    ready = resume_preparation.resume_command(command, source_root=source, plan=plan,
                                               checkpoint=checkpoint, stop=217)
    def value(flag): return ready[ready.index(flag)+1]
    assert value('--run-id') == plan.run_id
    assert value('--attempt-id') == 'resume1'
    assert value('--max-steps') == '2160'
    assert value('--dataloader-num-workers') == '0'
    assert value('--stage1-ce-entropy-top-percent') == '20'
    assert value('--data-path') == str(ROOT/'data/TRAIN.jsonl')
    assert value('--resume') == str(checkpoint)
    assert value('--stop-after-checkpoint') == '217'


def test_both_coordinator_worker_commands_keep_stop_and_budget():
    import os
    import shlex
    import subprocess
    from tests.stage1.v5.test_launcher import ROOT, REMOTE, HOST
    run = subprocess.run([str(ROOT/'run_multinode_stage1_training.sh'),
        '--run-id','matched-bounded-command','--attempt-id','resume1','--zero-stage','1',
        '--main-process-port','29931','--local-repo-root',str(ROOT),
        '--remote-repo-root',str(REMOTE),'--remote-host-repo-root',str(HOST),
        '--dry-run','--method','dora','--model','/original/base',
        '--data-path',str(ROOT/'data/TRAIN.jsonl'),'--image-root',str(ROOT/'data/images_comt'),
        '--stage1-match-v5-execution','--stage1-ce-entropy-top-percent','20',
        '--max-steps','2160','--resume',str(ROOT/'checkpoints/matched-bounded-command/checkpoint-216'),
        '--stop-after-checkpoint','217'], cwd=ROOT,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    payload = json.loads(run.stdout)
    assert payload['plan']['world_size'] == 4
    remote = shlex.split(shlex.split(payload['remote'][-1])[-1])
    for owned in (payload['local'], remote):
        assert '--stop-after-checkpoint' in owned
        assert owned[owned.index('--max-steps')+1] == '2160'


def test_prepare_resume_cli_is_available_without_starting_workers():
    import subprocess
    from tests.stage1.v5.test_launcher import ROOT
    run = subprocess.run([str(ROOT/'venv/ilvr-venv/bin/python'), '-m','src.multinode',
        'prepare-stage1-resume','--help'], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert '--source-log-dir' in run.stdout
    assert '--checkpoint-step' in run.stdout


def test_remote_destination_reservation_is_exclusive_and_canonical(tmp_path, monkeypatch):
    from src import multinode
    assert hasattr(multinode, '_prepare_stage1_resume_directory'), 'resume destination command missing'
    (tmp_path/'checkpoints').mkdir()
    monkeypatch.setattr(multinode, '_REPOSITORY_ROOT', tmp_path)
    argv = ['--run-id','matched-resume-fresh','--checkpoint-step','216']
    assert multinode._prepare_stage1_resume_directory(argv) == 0
    assert (tmp_path/'checkpoints/matched-resume-fresh/checkpoint-216').is_dir()
    with pytest.raises(FileExistsError):
        multinode._prepare_stage1_resume_directory(argv)
    with pytest.raises(ValueError):
        multinode._prepare_stage1_resume_directory(['--run-id','../escape','--checkpoint-step','216'])


def test_prepared_copy_publication_validates_exact_remote_union(tmp_path):
    from dataclasses import replace
    from src.multinode_runtime import SubprocessCoordinatorRunner
    from tests.multinode.test_artifacts import FakeProductionTransferRunner, identity
    assert hasattr(SubprocessCoordinatorRunner, 'publish_resume_copy'), 'prepared checkpoint publication is missing'
    canonical = FakeProductionTransferRunner().canonical
    local = replace(canonical, node_rank=1,
                    entries=tuple(replace(e, node_rank=1) for e in canonical.entries))
    events = []
    runner = SubprocessCoordinatorRunner.__new__(SubprocessCoordinatorRunner)
    runner.plan = SimpleNamespace(run_id=identity()['run_id'])
    runner._artifact_checkpoint_paths = {}
    runner.read_artifact_manifest = lambda *a, **kw: local
    runner._internal_command = lambda rank, command, args: (rank, command, args)
    runner._run = lambda command: events.append(command[1])
    runner.validate_push_destination = lambda *a: events.append('validate')
    runner.push_artifacts = lambda *a: events.append('push')
    runner.read_canonical_artifact_manifest = lambda *a: canonical
    runner.publish_resume_copy(tmp_path/identity()['run_id']/'checkpoint-2', identity())
    assert events == ['prepare-stage1-resume-directory','validate','push']


def test_transfer_reader_accepts_source_worktree_alias_of_same_checkpoint_storage(tmp_path, monkeypatch):
    from src import multinode_runtime
    physical = tmp_path/'storage'; physical.mkdir()
    current = tmp_path/'current'; current.mkdir()
    old = tmp_path/'source'; old.mkdir()
    (current/'checkpoints').symlink_to(physical, target_is_directory=True)
    (old/'checkpoints').symlink_to(physical, target_is_directory=True)
    monkeypatch.setattr(multinode_runtime, '_CHECKPOINT_ROOT', current/'checkpoints')
    plan = SimpleNamespace(cluster=SimpleNamespace(local_repo_root=old))
    reader = multinode_runtime.SubprocessCoordinatorRunner(plan, commands=None,
        commit='8'*40, checkpoint_root=old/'checkpoints')
    assert reader.checkpoint_root == old/'checkpoints'
