import copy
import importlib
import subprocess
import sys

import pytest
import torch


def witness(profile='v5'):
    from stage1.cache_validation import fingerprint
    def fp(value):
        return fingerprint(torch.tensor(float(value)))
    rng = {key: fp(1) for key in ('python', 'numpy', 'cpu', 'cuda')}
    row = {'inputs': fp(1), 'targets': fp(2), 'reference': fp(3), 'loss': fp(4),
        'rng_before': rng, 'rng_after': rng, 'query_count': 1,
        'projection': [{'hidden': fp(5), 'autocast': True, 'owner': 'Linear'}],
        'selection': {'text_keys': {'(0, 1)': [0]}, 'orig_positions': fp(1),
                      'entropy_support': [fp(2)]}}
    if profile == 'native-ema':
        row.update(query_count=0, projection=[], selection={})
        row.pop('reference')
    return {'kind': 'correctness', 'profile': profile, 'deterministic': True,
        'initial_step': 0, 'final_step': 2,
        'before': {'student': fp(1), 'teacher': fp(1), 'rng': rng},
        'microbatches': [copy.deepcopy(row) for _ in range(8)],
        'optimizer_steps': [{'gradients': fp(7), 'learning_rates': [rate],
            'master_parameters_after': fp(i + 1), 'group_state_after': fp(i + 9),
            'rng_before': rng, 'rng_after': rng} for i, rate in enumerate((0., 5e-6))],
        'after': {'student': fp(2), 'teacher': fp(1 if profile == 'v5' else 2),
                  'rng': rng, 'scheduler': fp(11), 'successful_updates': 2},
        'workers': 4, 'world_size': 4, 'production_checks': {'optional_calls': 0}}


def test_verifier_rejects_unequal_aa_before_ab():
    from stage1.cache_validation import compare_acceptance
    a, repeated, cached = witness(), witness(), witness()
    repeated['optimizer_steps'][0]['gradients']['sha256'] = '9' * 64
    with pytest.raises(ValueError, match='A/A'):
        compare_acceptance(a, repeated, cached)


def test_lr_zero_cannot_establish_update_equivalence():
    from stage1.cache_validation import compare_acceptance
    a = witness()
    for step in a['optimizer_steps']:
        step['learning_rates'] = [0.]
    with pytest.raises(ValueError, match='nonzero'):
        compare_acceptance(a, copy.deepcopy(a), copy.deepcopy(a))


def test_verifier_rejects_perturbed_cache_and_rng_consuming_producer():
    from stage1.cache_validation import compare_acceptance, assert_frozen_replay
    a, bad = witness(), witness()
    bad['microbatches'][0]['targets']['sha256'] = '9' * 64
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


@pytest.mark.parametrize('path', [
    ('microbatches',), ('optimizer_steps',), ('before', 'student'), ('before', 'teacher'),
    ('before', 'rng'), ('after', 'scheduler'), ('after', 'successful_updates'),
    ('microbatches', 0, 'targets'), ('microbatches', 0, 'loss'),
    ('microbatches', 0, 'rng_before'), ('microbatches', 0, 'reference'),
    ('microbatches', 0, 'projection'), ('microbatches', 0, 'selection'),
    ('optimizer_steps', 1, 'gradients'), ('optimizer_steps', 1, 'master_parameters_after'),
    ('optimizer_steps', 1, 'group_state_after'), ('deterministic',),
])
def test_acceptance_rejects_missing_claim_specific_evidence(path):
    from stage1.cache_validation import compare_acceptance
    record = witness()
    parent = record
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    with pytest.raises(ValueError, match='evidence'):
        compare_acceptance(record, copy.deepcopy(record), copy.deepcopy(record))


def test_acceptance_rejects_empty_equal_records():
    from stage1.cache_validation import compare_acceptance
    record = {'before': {}, 'microbatches': [], 'optimizer_steps': [{'learning_rates': [5e-6]}],
              'after': {}, 'workers': 4, 'world_size': 4, 'production_checks': {'optional_calls': 0}}
    with pytest.raises(ValueError, match='evidence'):
        compare_acceptance(record, copy.deepcopy(record), copy.deepcopy(record))


def test_reference_projection_is_not_claimed_for_native_ema():
    from stage1.cache_validation import compare_acceptance
    record = witness('native-ema')
    result = compare_acceptance(record, copy.deepcopy(record), copy.deepcopy(record))
    assert result['projection'] == 'not_applicable:native_has_no_reference'
    assert result['ema'] is True


def test_observer_records_actual_v5_selections_without_enabling_diagnostics(tmp_path):
    from stage1.cache_validation import Stage1CacheValidationObserver
    from tests.stage1.v5.test_trainer import integrated_fixture
    trainer, batch, *_ = integrated_fixture(tmp_path, debug=True)
    trainer.stage1_check_policy = 'production'
    observer = Stage1CacheValidationObserver({'mode': 'correctness'}, tmp_path)
    observer.current = {'selection': {}}
    observer.observe_v5_semantics()
    try:
        batches, _ = trainer.get_batch_samples(iter([batch]), 4)
        trainer.training_step(trainer.model, batches[0])
        selected = observer.current['selection']
        assert selected['text_keys']
        assert selected['orig_positions']['dtype'] == 'torch.bool'
        assert selected['entropy_support'][0]['ids']['dtype'] == 'torch.int64'
        assert observer.optional_calls == 0
    finally:
        observer.restore_observers()
