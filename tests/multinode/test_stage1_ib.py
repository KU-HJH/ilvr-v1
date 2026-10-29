"""Stage1 IB must survive the real sanitized worker command boundary."""

import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from src.multinode import build_launch_plan
from src.multinode_runtime import (
    _remote_repository_module_stdin_command,
    build_owned_tmux_commands,
    build_preflight_commands,
)

ROOT = Path(__file__).resolve().parents[2]
REMOTE = Path('/root/workspace/ILVR/.worktrees/stage1-v5-ib-20261029')


def ib_commands():
    plan = build_launch_plan(stage='stage1', tuning_method='dora', zero_stage=1,
        run_id='stage1-v5-ib-contract', main_process_port=29943,
        local_repo_root=ROOT, remote_repo_root=REMOTE,
        remote_host_repo_root=Path('/home/korea/Workspace/hjh/workspace/ILVR/.worktrees/stage1-v5-ib-20261029'))
    commands = build_owned_tmux_commands(plan, commit='a' * 40,
        stage_arguments=('--stage1-variant', 'v5', '--stage1-cache', 'off',
            '--stage1-check-policy', 'production', '--micro-batch', '1',
            '--gradient-accumulation', '4', '--epochs', '10', '--max-steps', '2160'))
    return plan, commands


@pytest.mark.parametrize('machine_rank', (0, 1))
def test_stage1_ib_reaches_each_sanitized_worker_with_node_local_paths(machine_rank):
    # Missing any forced value here can silently choose Socket after env-i.
    _, commands = ib_commands()
    spec = commands.node_specs[machine_rank]
    observed = subprocess.run(
        ['/usr/bin/env', '-i', *[f'{k}={v}' for k, v in spec.environment],
         sys.executable, '-c', 'import json,os; print(json.dumps(dict(os.environ)))'],
        check=True, text=True, capture_output=True,
    )
    actual = json.loads(observed.stdout)
    assert {k: actual.get(k) for k in (
        'NCCL_NET', 'NCCL_IB_DISABLE', 'NCCL_IB_HCA', 'NCCL_SOCKET_IFNAME',
        'GLOO_SOCKET_IFNAME', 'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS',
    )} == {
        'NCCL_NET': 'IB', 'NCCL_IB_DISABLE': '0', 'NCCL_IB_HCA': 'mlx5_0:1',
        'NCCL_SOCKET_IFNAME': 'bond0.11', 'GLOO_SOCKET_IFNAME': 'bond0.11',
        'NCCL_DEBUG': 'INFO', 'NCCL_DEBUG_SUBSYS': 'INIT,NET',
    }
    node_root = REMOTE if machine_rank == 0 else ROOT
    assert actual['PATH'].split(':')[0] == str(node_root / 'venv/ilvr-venv/bin')
    assert actual['ILVR_MACHINE_RANK'] == str(machine_rank)
    assert actual['ILVR_STAGE1_RUNTIME_PROFILE'] == 'default'
    assert actual['ILVR_STAGE1_CACHE_ATTEMPT'] == 'initial'
    assert 'PYTORCH_CUDA_ALLOC_CONF' not in actual
    command = spec.command[spec.command.index('--') + 1:]
    assert command[:2] == ('/usr/bin/bash', str(node_root / 'run_stage1_training.sh'))
    assert command[command.index('--stage1-cache') + 1] == 'off'
    assert command[command.index('--stage1-check-policy') + 1] == 'production'


def test_stage1_ib_launch_preflight_and_stdin_target_same_rdma_container():
    # A stale remote container would mix Socket/IB runtimes or checkpoint storage.
    plan, commands = ib_commands()
    preflight = build_preflight_commands(plan, input_paths=(), artifact_roots=(),
        remote_run_paths=(), local_run_paths=(), remote_tmux_sessions=(),
        local_tmux_sessions=())
    stdin = _remote_repository_module_stdin_command('receive-launch-plan', cluster=plan.cluster)
    for command in (commands.remote, preflight.remote_node_info, stdin):
        assert command[-2] == 'korea@172.16.11.41'
        docker = shlex.split(command[-1])
        assert docker[docker.index('/usr/bin/bash') - 1] == 'llm_kd_rdma'
        if command == preflight.remote_node_info:
            assert str(REMOTE / 'venv/ilvr-venv/bin/python') in shlex.split(docker[-1])
        else:
            assert docker[docker.index('--workdir') + 1] == str(REMOTE)
    assert shlex.split(preflight.container_inspect[-1])[-1] == 'llm_kd_rdma'
