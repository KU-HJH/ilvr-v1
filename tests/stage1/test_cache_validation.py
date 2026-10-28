import copy
import importlib
import subprocess
import sys

import pytest
import torch


def witness():
    return {'before': {'weights': 'initial'}, 'microbatches': [{'inputs': 1, 'targets': 2,
        'reference': 3, 'loss': 4, 'rng_before': 5, 'rng_after': 6}],
        'optimizer_steps': [{'gradients': 7, 'learning_rates': [5e-6],
            'master_parameters_after': 8, 'group_state_after': 9}],
        'after': {'teacher': 10, 'scheduler': 11, 'successful_updates': 2},
        'workers': 4, 'world_size': 4, 'production_checks': {'optional_calls': 0}}


def test_verifier_rejects_unequal_aa_before_ab():
    from stage1.cache_validation import compare_acceptance
    a, repeated, cached = witness(), witness(), witness()
    repeated['optimizer_steps'][0]['gradients'] = 99
    with pytest.raises(ValueError, match='A/A'):
        compare_acceptance(a, repeated, cached)


def test_lr_zero_cannot_establish_update_equivalence():
    from stage1.cache_validation import compare_acceptance
    a = witness()
    a['optimizer_steps'][0]['learning_rates'] = [0.]
    with pytest.raises(ValueError, match='nonzero'):
        compare_acceptance(a, copy.deepcopy(a), copy.deepcopy(a))


def test_verifier_rejects_perturbed_cache_and_rng_consuming_producer():
    from stage1.cache_validation import compare_acceptance, assert_frozen_replay
    a, bad = witness(), witness()
    bad['microbatches'][0]['targets'] = -1
    with pytest.raises(ValueError, match='A/B'):
        compare_acceptance(a, copy.deepcopy(a), bad)
    with pytest.raises(ValueError, match='RNG'):
        assert_frozen_replay({'payload': 1, 'rng_before': 2, 'rng_after': 3},
                             {'payload': 1, 'rng_before': 2, 'rng_after': 2})


def test_production_does_not_import_or_invoke_observer():
    result = subprocess.run([sys.executable, '-c',
        "import sys; import stage1.composition; assert 'stage1.cache_validation' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_verifier_cannot_enable_full_experiment_or_change_coefficients():
    from stage1.cache_validation import validate_validation_profile
    from types import SimpleNamespace
    args = SimpleNamespace(stage1_cache_validation='correctness', stage1_cache_validation_stop=2,
        stage1_variant='standard', stage1_runtime_profile='native-ema-0180',
        stage1_match_v5_execution=False, ema_tau=.999, ce_weight=.1, sim_weight=1.,
        max_steps=2160, epochs=10, learning_rate=5e-5, warm_up_steps=10, seed=42,
        per_device_train_batch_size=1, gradient_accumulation_steps=4, dataloader_num_workers=4,
        attention_implementation='flash_attention_2', vision_attention_implementation='flash_attention_2',
        tuning_method='dora')
    validate_validation_profile(args)
    args.stage1_cache_validation_stop = 2160
    with pytest.raises(ValueError, match='bounded'):
        validate_validation_profile(args)
    args.stage1_cache_validation_stop = 2
    args.ce_weight = 1.
    with pytest.raises(ValueError, match='profile'):
        validate_validation_profile(args)


def test_driver_dry_run_keeps_full_schedule_and_bounds_optimizer_work():
    import json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(repo / 'scripts/verify_stage1_run_cache.py'),
        '--phase', 'dry-run', '--profile', 'v5', '--run-id', 'validation-test'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['optimizer_updates'] == 6 and plan['samples'] == 16
    for command in plan['commands']:
        assert command[0].endswith('run_multinode_stage1_training.sh')
        for key, value in [('--max-steps', '2160'), ('--warm-up-steps', '10'),
                           ('--dataloader-num-workers', '4'), ('--ce-weight', '.1')]:
            assert command[command.index(key) + 1] == value
