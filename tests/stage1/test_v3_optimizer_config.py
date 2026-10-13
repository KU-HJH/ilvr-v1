"""Optimizer formula identity must not reinterpret historical checkpoints."""
import importlib
import json
from copy import deepcopy

import pytest
from stage1.v3_config import Stage1V3Config
from stage1 import v3_resume


def contract(config):
    schedule = v3_resume.build_v3_schedule_identity(retained_ids=('a', 'b'),
        sampler_identity={'world_size': 4}, microsteps_per_rank=2, accumulation_steps=4)
    return v3_resume.build_v3_checkpoint_contract(config, schedule, {'frozen_teacher_sha256': 'a'*64})


@pytest.mark.parametrize('weighting', ['none', 'token_entropy'])
def test_fresh_new_formula_and_zero_weight_are_explicit(weighting):
    config = Stage1V3Config(gain_update_mode='adamw_projected', gain_optimizer_weight=0., gain_weighting=weighting)
    saved = contract(config)
    assert saved['config']['gain_optimizer_weight'] == 0.
    want = 'global_block_mean_same_state_adamw_joint_moments_active_lr_project_fp64_fp32_master_v1'
    if weighting == 'token_entropy':
        want = 'F_T1_detached_H_logV_token_gap_' + want
    assert saved['formula_ids']['gain'] == want
    assert saved['formula_ids']['gain_alpha_role'] == 'inactive_legacy_displacement_v1'
    v3_resume.validate_v3_resume(saved, deepcopy(saved))


@pytest.mark.parametrize('kwargs', [
    {'gain_update_mode': 'other'}, {'gain_update_mode': 'adamw_projected'},
    {'gain_optimizer_weight': 0.},
    *[{'gain_update_mode': 'adamw_projected', 'gain_optimizer_weight': w}
      for w in [True, '0.2', -1., float('nan'), float('inf')]],
])
def test_invalid_optimizer_mode_and_coefficient_rejected(kwargs):
    with pytest.raises(ValueError):
        Stage1V3Config(**kwargs)


@pytest.mark.parametrize('weighting', ['none', 'token_entropy'])
def test_legacy_contract_missing_new_fields_keeps_bytes_and_formula(tmp_path, weighting):
    current = contract(Stage1V3Config(gain_weighting=weighting))
    assert current['config']['gain_update_mode'] == 'post_adam'
    assert current['config']['gain_optimizer_weight'] is None
    assert 'gain_alpha_role' not in current['formula_ids']
    old = deepcopy(current)
    old['config'].pop('gain_update_mode')
    old['config'].pop('gain_optimizer_weight')
    original = json.dumps(old, indent=3).encode()
    path = tmp_path / v3_resume.FILENAME
    path.write_bytes(original)
    loaded = v3_resume.read_v3_contract(tmp_path)
    assert loaded == old
    v3_resume.validate_v3_resume(loaded, current)
    assert path.read_bytes() == original
    # Omissions in other objective coefficients are not treated as legacy defaults.
    bad = deepcopy(old)
    bad['config'].pop('gain_alpha')
    with pytest.raises(ValueError):
        v3_resume.validate_v3_resume(bad, current)


def test_resolver_inherits_saved_mode_and_zero_but_rejects_changes():
    resolve = importlib.import_module('stage1.v3_resume')._resolve_gain_update
    saved = Stage1V3Config(gain_update_mode='adamw_projected', gain_optimizer_weight=0.)
    assert resolve(saved, None, None) == ('adamw_projected', 0.)
    assert resolve(saved, 'adamw_projected', 0.) == ('adamw_projected', 0.)
    assert resolve(None, None, None) == ('post_adam', None)
    for mode, weight in [('post_adam', None), (None, .1)]:
        with pytest.raises(ValueError, match='resume'):
            resolve(saved, mode, weight)
    with pytest.raises(ValueError):
        resolve(None, 'adamw_projected', None)
