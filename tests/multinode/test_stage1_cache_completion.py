from dataclasses import replace

import pytest

from tests.multinode.test_preflight import snapshot


def test_native_profile_requires_matching_supported_versions_on_both_nodes():
    from src.multinode_runtime import validate_preflight
    remote = replace(snapshot('remote', 0), deepspeed_version='0.18.0')
    local = replace(snapshot('local', 1), deepspeed_version='0.18.0')
    assert validate_preflight(remote, local, stage1_runtime_profile='native-ema-0180').world_size == 4
    with pytest.raises(ValueError, match='environment'):
        validate_preflight(remote, snapshot('local', 1), stage1_runtime_profile='native-ema-0180')
    with pytest.raises(ValueError, match='environment'):
        validate_preflight(remote, local)


@pytest.mark.parametrize('event', ['interrupted', 'evaluation-failed', 'planned-stop'])
def test_interrupted_or_failed_evaluation_retains_cache(event):
    from stage1.cache_setup import finish_stage1_run_cache
    class Runner:
        def remove_cache(self, rank, run_id):
            pytest.fail('nonterminal event deleted resumable cache')
    finish_stage1_run_cache(run_id='trial', completion=event, runner=Runner())


def test_completion_deletes_only_closed_owned_run():
    from stage1.cache_setup import finish_stage1_run_cache
    calls = []
    class Runner:
        def remove_cache(self, rank, run_id):
            calls.append((rank, run_id))
    finish_stage1_run_cache(run_id='trial', completion='complete', runner=Runner())
    assert calls == [(0, 'trial'), (1, 'trial')]


def test_coordinator_propagates_cache_and_native_profile():
    from src.multinode import build_launch_plan
    from src.multinode_runtime import build_owned_tmux_commands
    plan = build_launch_plan(stage='stage1', tuning_method='dora', zero_stage=1,
                             run_id='cache-trial', main_process_port=29631)
    commands = build_owned_tmux_commands(plan, commit='a' * 40,
        stage_arguments=('--method', 'dora', '--stage1-cache', 'run',
                         '--stage1-runtime-profile', 'native-ema-0180'))
    for node in commands.node_specs:
        assert dict(node.environment)['ILVR_STAGE1_RUNTIME_PROFILE'] == 'native-ema-0180'
        assert dict(node.environment)['ILVR_STAGE1_CACHE_ATTEMPT'] == 'initial'


def test_native_preflight_uses_same_preserved_runtime_as_workers():
    from src.multinode import build_launch_plan
    from src.multinode_runtime import build_preflight_commands
    plan = build_launch_plan(stage='stage1', tuning_method='dora', zero_stage=1,
                             run_id='cache-native', main_process_port=29631)
    commands = build_preflight_commands(plan, input_paths=(), artifact_roots=(),
        remote_run_paths=(), local_run_paths=(), remote_tmux_sessions=(), local_tmux_sessions=(),
        stage1_runtime_profile='native-ema-0180')
    assert '/root/workspace/ILVR/.worktrees/stage1-visual-history-only-20261019/venv/ilvr-venv/bin/python' in commands.local_node_info
    assert 'stage1-visual-history-only-20261019/venv/ilvr-venv/bin/python' in commands.remote_node_info[-1]
