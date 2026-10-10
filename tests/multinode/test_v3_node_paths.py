"""Exercise real argv builders; no SSH or worker process is executed."""
import json
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from src import multinode as mn
from src import multinode_runtime as rt
from src.multinode_artifacts import build_pull_command, build_push_command
from tests.multinode.test_artifacts import strict_manifest
from tests.multinode.test_v3_roots import LOCAL, REMOTE, HOST, native_plan


def remote_inner(command):
    return shlex.split(shlex.split(command[-1])[-1])


def value(argv, flag):
    return argv[argv.index(flag)+1]


def test_node_commands_and_checkpoint_paths_use_declared_roots():
    plan = native_plan()
    commands = rt.build_owned_tmux_commands(plan, commit='a'*40,
        stage_arguments=('--method','dora','--data-path',str(LOCAL/'data/train.jsonl'),
                         '--resume',str(LOCAL/'checkpoints/old-run/checkpoint-1')))
    for rank, root in ((0,REMOTE),(1,LOCAL)):
        spec = commands.node_specs[rank]
        assert spec.command[0] == str(root/'venv/ilvr-venv/bin/python')
        assert spec.ownership_path == root/'logs/multinode'/plan.run_id/'initial'/f'ownership-rank{rank}.json'
        stage = spec.command[spec.command.index('--')+1:]
        assert stage[:2] == ('/usr/bin/bash',str(root/'run_stage1_training.sh'))
        assert value(stage,'--data-path') == str(root/'data/train.jsonl')
        assert value(stage,'--resume') == str(root/'checkpoints/old-run/checkpoint-1')
        assert value(stage,'--output-dir') == str(root/'checkpoints'/plan.run_id)
        assert value(stage,'--deepspeed-config') == str(root/'configs/deepspeed_zero1.json')
        env = dict(spec.environment)
        assert env['PYTHONPATH'] == f'{root}:{root / "src"}'
        assert env['PATH'].startswith(str(root/'venv/ilvr-venv/bin')+':')
    docker = shlex.split(commands.remote[-1])
    assert value(docker,'--workdir') == str(REMOTE)
    assert f'--chdir={LOCAL}' in commands.local


def test_preflight_uses_node_python_inputs_and_storage():
    p = native_plan()
    args = dict(input_paths=(LOCAL/'data/train.jsonl',),artifact_roots=(LOCAL/'checkpoints',),
        remote_run_paths=(LOCAL/'logs'/p.run_id,),local_run_paths=(LOCAL/'logs'/p.run_id,),
        remote_tmux_sessions=(),local_tmux_sessions=())
    cmds = rt.build_preflight_commands(p,**args)
    for argv,root in ((remote_inner(cmds.remote_node_info),REMOTE),(cmds.local_node_info,LOCAL)):
        assert str(root/'venv/ilvr-venv/bin/python') in argv
        assert value(argv,'--repository') == str(root)
        assert value(argv,'--input-path') == str(root/'data/train.jsonl')
        assert value(argv,'--artifact-root') == str(root/'checkpoints')
        assert value(argv,'--deepspeed-config') == str(root/'configs/deepspeed_zero1.json')


def test_checkpoint_rsync_uses_container_destination_and_local_source():
    p=native_plan()
    m = strict_manifest(1)
    push = build_push_command(m,cluster=p.cluster)
    remote = str(REMOTE/'checkpoints'/m.run_id/m.checkpoint_relative)+'/'
    local = str(LOCAL/'checkpoints'/m.run_id/m.checkpoint_relative)+'/'
    assert push[-2:] == (local,'korea@10.1.102.41:'+remote)
    pull=build_pull_command(strict_manifest(0),cluster=p.cluster)
    assert pull[-2] == 'korea@10.1.102.41:'+remote
    assert pull[-1] == str(LOCAL/'checkpoints'/m.run_id)+'/'
    assert str(HOST) not in ' '.join(push+pull)


def test_remote_exec_rejects_cwd_outside_declared_root():
    with pytest.raises(ValueError,match='working directory'):
        rt.build_remote_exec(('true',),working_directory='/tmp/foreign',cluster=native_plan().cluster)
    argv=rt.build_remote_exec(('printf','literal $(touch BAD)'),working_directory=REMOTE,cluster=native_plan().cluster)
    assert remote_inner(argv) == ['printf','literal $(touch BAD)']


def test_node_path_rejects_parent_traversal_and_does_not_resolve_symlinks():
    p=native_plan()
    assert rt.node_path(p,LOCAL/'checkpoints/run/checkpoint-1',machine_rank=0) == REMOTE/'checkpoints/run/checkpoint-1'
    with pytest.raises(ValueError,match='traversal'):
        rt.node_path(p,LOCAL/'checkpoints/../foreign',machine_rank=0)


def test_runner_internal_command_maps_only_path_arguments():
    p=native_plan()
    runner=rt.SubprocessCoordinatorRunner(p,commands=None,commit='a'*40)
    argv=remote_inner(runner._internal_command(0,'artifact-manifest',
        ('--run-directory',str(LOCAL/'checkpoints/run/checkpoint-1'),
         '--checkpoint-root',str(LOCAL/'checkpoints'),'--runtime-identity-json','{"path":"literal"}')))
    assert argv[2] == str(REMOTE/'venv/ilvr-venv/bin/python')
    assert value(argv,'--run-directory') == str(REMOTE/'checkpoints/run/checkpoint-1')
    assert value(argv,'--runtime-identity-json') == '{"path":"literal"}'


def test_coordinator_dry_run_carries_roots_to_both_workers(capsys):
    result=mn._main(['train-stage1','--run-id','stage1-v3-dry-roots','--zero-stage','1',
        '--main-process-port','29681','--dry-run','--method','dora',
        '--local-repo-root',str(LOCAL),'--remote-repo-root',str(REMOTE),
        '--remote-host-repo-root',str(HOST)])
    assert result == 0
    payload=json.loads(capsys.readouterr().out)
    assert str(REMOTE/'run_stage1_training.sh') in str(payload)
    assert str(LOCAL/'run_stage1_training.sh') in str(payload)


def test_preflight_cli_uses_local_storage_for_local_node(capsys):
    assert mn._main(['preflight','--stage','stage1','--tuning-method','dora',
        '--run-id','v3-preflight','--zero-stage','1','--main-process-port','29682',
        '--remote-repo-root',str(REMOTE),'--remote-host-repo-root',str(HOST),'--dry-run']) == 0
    record=json.loads(capsys.readouterr().out)
    assert value(record['local_node_info'],'--artifact-root') == str(LOCAL/'checkpoints')
    assert value(remote_inner(record['remote_node_info']),'--artifact-root') == str(REMOTE/'checkpoints')


def test_same_root_supervisor_preserves_legacy_environment_argv():
    p = native_plan()
    p = replace(p, cluster=replace(p.cluster, remote_repo_root=LOCAL))
    command = rt.build_owned_tmux_commands(p, commit='a'*40,
        stage_arguments=('--method', 'dora')).local
    # Existing documented consumers read environment assignments immediately after -i.
    start = command.index('/usr/bin/env')
    assert command[start:start + 3] == ('/usr/bin/env', '-i', 'HOME=/root')


def test_entropy_coordinator_dry_run_passes_mode_to_both_nodes(capsys):
    assert mn._main(['train-stage1','--run-id','stage1-v3-entropy-dry','--zero-stage','1',
        '--main-process-port','29681','--dry-run','--stage1-variant','v3',
        '--stage1-v3-selector','entropy','--local-repo-root',str(LOCAL),
        '--remote-repo-root',str(REMOTE),'--remote-host-repo-root',str(HOST)])==0
    payload=json.loads(capsys.readouterr().out)
    for worker in (payload['local'],remote_inner(payload['remote'])):
        assert value(worker,'--stage1-v3-selector')=='entropy'
        assert value(worker,'--stage1-variant')=='v3'
