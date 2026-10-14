"""Opt-in objectives must resolve explicitly without borrowing Stage2 defaults."""
from dataclasses import FrozenInstanceError, asdict
import importlib

import pytest

from utils_deepseed import parse_args


def api():
    return importlib.import_module('stage1.v5_config')


def test_orig_alias_defaults_and_formula_identity():
    config = api().resolve_v5_config({'loss_type': 'dil-bce-orig'})
    assert config.loss_type == 'dil_bce_orig'
    assert api().resolve_v5_config({}).lambda_topk_entropy == 0.0
    assert config.temperature == 2 and config.text_kl_weight == 0.01
    assert (config.beta, config.margin, config.lambda_preference) == (1., 0., 1.)
    assert (config.rho_min, config.rho_max, config.text_cap,
            config.low_attention_mass, config.low_attention_fraction) == (.3, .5, .3, .1, .1)
    assert api().v5_formula_id(config) == 'dil_bce_orig_token_ref_full_vocab_v1'
    assert api().v5_formula_id(api().resolve_v5_config({
        'lambda_topk_entropy': .1, 'preference_topk_k': 50})) == 'dil_bce_orig_token_ref_full_vocab_topk_entropy_v1'
    assert api().v5_formula_id(api().resolve_v5_config({
        'lambda_topk_entropy': .1, 'preference_topk_k': 50,
        'topk_include_gt': True})) == 'dil_bce_orig_token_ref_full_vocab_topk_gt_union_entropy_v1'
    with pytest.raises(FrozenInstanceError):
        config.beta = 2.


@pytest.mark.parametrize('values', [
    {'loss_type': 'dil_bce_rf'}, {'beta': 10}, {'margin': .5},
    {'lambda_entropy': .1}, {'lambda_topk_chosen_mass': .1},
    {'lambda_topk_entropy': .1},
    {'lambda_topk_entropy': .1, 'preference_topk_k': 0},
    {'lambda_topk_entropy': .1, 'preference_topk_k': True},
    {'lambda_topk_entropy': .1, 'preference_topk_k': 1.5},
    {'lambda_topk_mass': .1}, {'lambda_preference': 0},
    {'lambda_topk_entropy': float('nan')}, {'text_kl_weight': -1},
    {'temperature': 0}, {'loss_chunk_positions': 0},
    {'preference_topk_mode': 'shared'}, {'topk_include_gt': 1},
    {'selection': 'unknown'}, {'preference_entropy_top_percent': 101},
    {'rho_min': .6}, {'text_cap': 1.1}, {'unexpected': 1},
])
def test_unsupported_objectives_or_invalid_values_fail(values):
    with pytest.raises(ValueError):
        api().resolve_v5_config(values)


def test_alias_conflicts_and_resume_inheritance_are_explicit():
    config = api().resolve_v5_config({
        'step_preference_weight': 2., 'lambda_preference': 2.,
        'step_preference_beta': 1., 'step_dpo_beta': 1.,
        'step_preference_topk': 50, 'preference_topk_k': 50,
        'lambda_topk_entropy': .1})
    assert config.lambda_preference == 2 and config.preference_topk_k == 50
    for values in ({'step_preference_weight': 2., 'lambda_preference': 1.},
                   {'step_preference_topk': 50, 'preference_topk_k': 32},
                   {'step_preference_beta': 1., 'step_dpo_beta': 2.}):
        with pytest.raises(ValueError, match='conflict'):
            api().resolve_v5_config(values)
    saved = asdict(config)
    assert api().resolve_v5_config({}, saved=saved) == config
    assert api().resolve_v5_config({'loss_type': 'dil-bce-orig'}, saved=saved) == config
    with pytest.raises(ValueError, match='resume|saved|conflict'):
        api().resolve_v5_config({'lambda_topk_entropy': .2}, saved=saved)


def test_worker_v5_options_resolve_separately_and_preserve_explicit_values():
    args = parse_args(['--stage1_variant', 'v5', '--stage1_loss_type', 'dil-bce-orig',
        '--lambda_topk_entropy', '.1', '--preference_topk_k', '50',
        '--lambda_preference', '2', '--step_preference_weight', '2',
        '--step_preference_beta', '1', '--step_dpo_beta', '1',
        '--step_preference_margin', '0', '--stage1_v5_text_kl_weight', '.02'])
    assert args.stage1_v5_config.loss_type == 'dil_bce_orig'
    assert args.stage1_v5_config.lambda_topk_entropy == .1
    assert args.stage1_v5_config.preference_topk_k == 50
    assert args.stage1_v5_config.text_kl_weight == .02
    assert args.step_preference_beta == 1 and args.step_preference_margin == 0
    assert args.step_preference_config.enabled is False
    assert args.step_preference_config.lambda_preference == 1
    assert args.stage1_v5_explicit_values['lambda_preference'] == 2
    assert 'preference_topk_mode' not in args.stage1_v5_explicit_values


@pytest.mark.parametrize('argv', [
    ['--lambda_topk_entropy', '0'],
    ['--stage', 'stage2', '--stage1_variant', 'v5'],
    ['--stage', 'stage2', '--lambda_topk_mass', '0'],
    ['--stage1_loss_type', 'dil-bce-orig'],
    ['--stage1_variant', 'v5', '--preference_topk_k', '50', '--step_preference_topk', '32'],
    ['--stage1_variant', 'v5', '--step_preference_beta', '2', '--step_dpo_beta', '1'],
    ['--stage1_variant', 'v5', '--step_entropy_weight', '.1'],
])
def test_worker_rejects_wrong_path_or_conflicting_explicit_aliases(argv):
    with pytest.raises(ValueError):
        parse_args(argv)


def test_standard_and_stage2_keep_existing_defaults():
    standard = parse_args([])
    stage2 = parse_args(['--stage', 'stage2'])
    assert standard.stage1_variant == 'standard' and standard.stage1_v5_config is None
    assert stage2.stage1_v5_config is None
    assert (stage2.step_preference_beta, stage2.step_preference_margin) == (10., .5)


def test_percentage_only_uses_clean_but_explicit_and_saved_selection_are_authoritative():
    assert api().resolve_v5_config({'preference_entropy_top_percent': 50}).selection == 'clean'
    for values in ({'selection': 'all', 'preference_entropy_top_percent': 50},
                   {'selection': 'clean'}):
        with pytest.raises(ValueError):
            api().resolve_v5_config(values)
    saved = asdict(api().resolve_v5_config({}))
    with pytest.raises(ValueError, match='resume|saved|conflict'):
        api().resolve_v5_config({'preference_entropy_top_percent': 50}, saved=saved)


def test_argparse_abbreviations_cannot_bypass_v5_presence_or_fixed_coefficients():
    with pytest.raises(ValueError):
        parse_args(['--lambda_topk_ent=0'])
    with pytest.raises(ValueError):
        parse_args(['--stage1_variant=v5', '--step_preference_bet=2'])
    args = parse_args(['--stage1_variant=v5', '--lambda_topk_ent=.1', '--preference_topk_k=50'])
    assert args.stage1_v5_config.lambda_topk_entropy == .1
    assert args.stage1_v5_explicit_values['lambda_topk_entropy'] == .1
