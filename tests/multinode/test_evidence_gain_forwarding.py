from tests.multinode.root_fixtures import REMOTE_ROOT_KWARGS, ROOT_ARGUMENTS
"""New preference options must survive coordinator, shell and CLI validation."""
import json
import subprocess

import pytest

from tests.multinode.test_stage2b_forwarding import REPO, stage2b_environment
from src.multinode import build_launch_plan
from src.multinode_runtime import build_owned_tmux_commands, validated_stage2_environment


@pytest.mark.parametrize('mode,blocked', [('clean_prefix', 'false'),
    ('v_off_prefix', 'false'), ('v_off_prefix', 'true')])
def test_evidence_gain_options_reach_both_workers(mode, blocked):
    """Catches a silently dropped mode/image flag producing identical A/B runs."""
    source = stage2b_environment()
    source.pop('STEP_NEGATIVE_PATH')
    source.pop('STEP_NEGATIVE_MANIFEST')
    source.update(STEP_PREFERENCE_OBJECTIVE='context_preference',
        STEP_VISUAL_ONLY_CONTEXT_PREFERENCE='true',
        STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES='false',
        STEP_PREFERENCE_PREFIX_MODE=mode, STEP_PREFERENCE_BLOCK_ORIGINAL_IMAGE=blocked)
    plan = build_launch_plan(**REMOTE_ROOT_KWARGS, stage='stage2', tuning_method='dora', zero_stage=2,
        run_id='evidence-gain-forward-test', main_process_port=29873)
    environment = validated_stage2_environment(source,
        output_dir=str(REPO/'checkpoints'/plan.run_id),
        log_file=str(REPO/'logs'/plan.run_id/'train.log'),
        deepspeed_config=str(plan.deepspeed_config))
    commands = build_owned_tmux_commands(plan, commit='a'*40, stage_environment=environment)
    for spec in commands.node_specs:
        worker_env = dict(spec.environment)
        assert worker_env.get('STEP_PREFERENCE_PREFIX_MODE') == mode
        assert worker_env.get('STEP_PREFERENCE_BLOCK_ORIGINAL_IMAGE') == blocked
        command = spec.command[spec.command.index('--')+1:]
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
            env={**worker_env, 'STAGE2_DRY_RUN': 'true', 'STAGE2_DRY_RUN_FORMAT': 'json'})
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        assert argv[argv.index('--step_preference_prefix_mode')+1] == mode
        assert ('--step_preference_block_original_image' in argv) == (blocked == 'true')


def test_evidence_replacement_forwards_fixed_entropy_seed_and_boundaries_to_both_workers():
    """Catches rank-specific worker environments silently changing replacement sampling."""
    source = stage2b_environment()
    source.pop('STEP_NEGATIVE_PATH')
    source.pop('STEP_NEGATIVE_MANIFEST')
    source.update(
        STEP_PREFERENCE_OBJECTIVE='context_preference',
        STEP_PREFERENCE_PREFIX_MODE='evidence_replacement',
        STEP_ENTROPY_WEIGHT='0',
        STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES='false',
        SEED='917',
    )
    plan = build_launch_plan(**REMOTE_ROOT_KWARGS, stage='stage2', tuning_method='dora', zero_stage=2,
        run_id='replacement-forward-test', main_process_port=29874)
    environment = validated_stage2_environment(source,
        output_dir=str(REPO/'checkpoints'/plan.run_id),
        log_file=str(REPO/'logs'/plan.run_id/'train.log'),
        deepspeed_config=str(plan.deepspeed_config))
    commands = build_owned_tmux_commands(plan, commit='b'*40, stage_environment=environment)
    for spec in commands.node_specs:
        result = subprocess.run(spec.command[spec.command.index('--')+1:], cwd=REPO,
            capture_output=True, text=True, env={**dict(spec.environment),
            'STAGE2_DRY_RUN': 'true', 'STAGE2_DRY_RUN_FORMAT': 'json'})
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        assert argv[argv.index('--step_preference_prefix_mode')+1] == 'evidence_replacement'
        assert argv[argv.index('--step_entropy_weight')+1] == '0'
        assert argv[argv.index('--seed')+1] == '917'
        assert '--no-step_visual_off_suppress_latent_boundaries' in argv
