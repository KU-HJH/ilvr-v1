import json
import random

import numpy as np
import pytest
import torch

from tests.stage1.test_run_cache import make_cache


class Teacher:
    def __init__(self, value):
        self.value = value
    def state_dict(self):
        return {'parameters': {'weight': torch.tensor(self.value)}}
    def load_state_dict(self, state):
        self.value = float(state['parameters']['weight'])


def checkpoint(tmp_path, cache, epoch):
    from stage1.cache_setup import write_stage1_cache_contract
    path = tmp_path / 'checkpoints/trial/checkpoint-2'
    path.mkdir(parents=True)
    (path / 'trainer_state.json').write_text(json.dumps({'epoch': epoch, 'global_step': 2}))
    (path / 'objective.json').write_text('{"ce":1,"alignment":1}')
    torch.save(Teacher(3.).state_dict(), path / 'stage1_cache_initial_teacher.pt')
    write_stage1_cache_contract(path, cache.plan, phase=0, check_policy='production', completion='train')
    return path


def test_first_epoch_resume_continues_population(tmp_path):
    from stage1.cache_setup import prepare_stage1_cache_resume
    cache = make_cache(tmp_path)
    saved = checkpoint(tmp_path, cache, .5)
    prepare_stage1_cache_resume(cache=cache, checkpoint=saved, initial_teacher=Teacher(9.),
        expected={'preprocess': (0,)}, rebuild=lambda ids: pytest.fail('first epoch prepass'))
    assert cache.phase.value == 0


def test_later_resume_rebuilds_from_initial_teacher_not_student(tmp_path):
    from stage1.cache_setup import prepare_stage1_cache_resume
    cache = make_cache(tmp_path)
    saved = checkpoint(tmp_path, cache, 2.)
    teacher = Teacher(9.)
    seen = []
    def rebuild(ids):
        seen.append((ids, teacher.value))
        cache.publish('targets', 0, {'value': torch.tensor(teacher.value)})
    prepare_stage1_cache_resume(cache=cache, checkpoint=saved, initial_teacher=teacher,
        expected={'targets': (0,)}, rebuild=rebuild)
    assert seen == [((0,), 3.)]
    assert teacher.value == 9.
    assert cache.phase.value == 1
    assert float(cache.read('targets', 0)['value']) == 3.


def test_resume_rebuild_preserves_rng_and_saved_objective(tmp_path):
    from stage1.cache_setup import prepare_stage1_cache_resume
    cache = make_cache(tmp_path)
    saved = checkpoint(tmp_path, cache, 1.)
    torch_state = torch.random.get_rng_state()
    py_state = random.getstate()
    np_state = np.random.get_state()
    objective = (saved / 'objective.json').read_bytes()
    def rebuild(ids):
        torch.rand(9); random.random(); np.random.rand()
        cache.publish('targets', 0, {'value': torch.tensor(3.)})
    prepare_stage1_cache_resume(cache=cache, checkpoint=saved, initial_teacher=Teacher(9.),
        expected={'targets': (0,)}, rebuild=rebuild)
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert py_state == random.getstate()
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    assert objective == (saved / 'objective.json').read_bytes()


def test_old_resume_does_not_enable_cache(tmp_path):
    from stage1.cache_setup import load_stage1_cache_contract
    assert load_stage1_cache_contract(tmp_path, requested='off') == {'check_policy': 'legacy', 'mode': 'off'}
    with pytest.raises(ValueError, match='old|initial|contract'):
        load_stage1_cache_contract(tmp_path, requested='run')


def test_ssd_namespace_preserves_existing_dataset_cache(tmp_path):
    from stage1.cache_setup import expose_cache_on_ssd, remove_owned_cache
    from stage1.run_cache import Stage1CachePlan, Stage1RunCache
    import multiprocessing
    logical = tmp_path / 'data/cache'
    logical.mkdir(parents=True)
    (logical / 'foreign').write_text('keep')
    root = tmp_path / 'ssd/trial'
    cache = Stage1RunCache(Stage1CachePlan('trial', root, 'off', False, {}), multiprocessing.Value('i', 0))
    expose_cache_on_ssd(cache, logical_root=logical)
    assert (logical / 'trial').is_symlink()
    with pytest.raises(RuntimeError, match='closed'):
        remove_owned_cache(cache, logical_root=logical)
    cache.close()
    remove_owned_cache(cache, logical_root=logical)
    assert (logical / 'foreign').read_text() == 'keep'
    assert not root.exists()
    assert not (logical / 'trial').is_symlink()


def test_composition_binds_production_policy_and_opt_in_cache(tmp_path, monkeypatch, request):
    from tests.stage1.test_composition import fake_dependencies, stage1_args, _clear_training_process_state
    _clear_training_process_state()
    request.addfinalizer(_clear_training_process_state)
    from stage1.composition import build_stage1_training_components
    import stage1.cache_setup as setup
    deps, _, _, _ = fake_dependencies([], metadata=object())
    args = stage1_args(tmp_path)
    result = build_stage1_training_components(args, dependencies=deps)
    assert result.trainer.kwargs['stage1_check_policy'] == 'production'
    seen = []
    monkeypatch.setattr(setup, 'configure_stage1_run_cache', lambda *a, **k: seen.append((a, k)), raising=False)
    args.stage1_cache = 'run'
    deps, _, _, _ = fake_dependencies([], metadata=object())
    build_stage1_training_components(args, dependencies=deps)
    assert len(seen) == 1


def test_reference_coverage_follows_actual_nonempty_queries(tmp_path):
    from stage1.cache_setup import resolve_expected
    cache = make_cache(tmp_path)
    cache.publish('preprocess', 0, {'v5_blocks': ({'target_positions': ()},)})
    cache.publish('preprocess', 1, {'v5_blocks': ({'target_positions': (7,)},)})
    assert resolve_expected(cache, {'preprocess': (0, 1), 'reference': None}) == {
        'preprocess': (0, 1), 'reference': (1,)}
