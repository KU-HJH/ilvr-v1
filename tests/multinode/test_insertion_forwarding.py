from tests.multinode.root_fixtures import REMOTE_ROOT_KWARGS, ROOT_ARGUMENTS
"""Exercise coordinator -> both worker shells -> actual argument validation."""
import json
import subprocess

import pytest

from tests.multinode.test_stage2b_forwarding import REPO, stage2b_environment
from src.multinode import build_launch_plan
from src.multinode_runtime import build_owned_tmux_commands, validated_stage2_environment
from utils_deepseed import parse_args


@pytest.mark.parametrize('mode', ['independent', 'cumulative'])
def test_insertion_mode_reaches_both_workers_and_config(mode):
    source = stage2b_environment()
    source.pop('STEP_NEGATIVE_PATH')
    source.pop('STEP_NEGATIVE_MANIFEST')
    source.update(STEP_PREFERENCE_OBJECTIVE='context_preference',
        STEP_VISUAL_ONLY_CONTEXT_PREFERENCE='true',
        STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES='false',
        STEP_PREFERENCE_PREFIX_MODE='v_off_insertion', STEP_PREFERENCE_BRANCH_MODE=mode)
    plan = build_launch_plan(**REMOTE_ROOT_KWARGS, stage='stage2', tuning_method='dora', zero_stage=2,
        run_id='insertion-forward-test', main_process_port=29873)
    environment = validated_stage2_environment(source,
        output_dir=str(REPO/'checkpoints'/plan.run_id),
        log_file=str(REPO/'logs'/plan.run_id/'train.log'),
        deepspeed_config=str(plan.deepspeed_config))
    commands = build_owned_tmux_commands(plan, commit='a'*40, stage_environment=environment)
    for spec in commands.node_specs:
        worker_env = dict(spec.environment)
        assert worker_env.get('STEP_PREFERENCE_BRANCH_MODE') == mode
        command = spec.command[spec.command.index('--')+1:]
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
            env={**worker_env, 'STAGE2_DRY_RUN': 'true', 'STAGE2_DRY_RUN_FORMAT': 'json'})
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        assert argv[argv.index('--step_preference_prefix_mode')+1] == 'v_off_insertion'
        assert argv[argv.index('--step_preference_branch_mode')+1] == mode
    args = parse_args(['--stage', 'stage2', '--step_preference_enabled',
        '--step_preference_mode', 'stage2b', '--stage2a_checkpoint_path', 'tiny-a',
        '--step_shared_prefix_enabled', '--step_preference_prefix_mode', 'v_off_insertion',
        '--step_preference_branch_mode', mode, '--no-step_visual_off_suppress_latent_boundaries'])
    assert args.step_preference_config.branch_mode == mode


@pytest.mark.parametrize('prefix', ['clean_prefix', 'v_off_prefix'])
def test_cumulative_is_rejected_for_old_cli_modes(prefix):
    with pytest.raises(ValueError):
        parse_args(['--stage', 'stage2', '--step_preference_enabled',
            '--step_preference_mode', 'stage2b', '--stage2a_checkpoint_path', 'tiny-a',
            '--step_preference_prefix_mode', prefix, '--step_preference_branch_mode', 'cumulative',
            '--no-step_visual_off_suppress_latent_boundaries'])
