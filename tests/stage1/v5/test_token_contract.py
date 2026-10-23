"""Historical inspection stays readable; token training has a distinct identity."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from stage1.v5_config import Stage1V5Config, resolve_v5_config
from stage1.v5_resume import load_v5_contract, validate_v5_resume
from tests.stage1.v5.test_resume import contract


def historical_contract():
    # Original schema-1 payload, independently spelled out rather than made by
    # the current contract builder (which must never emit block-mean training).
    return {
        'schema_version': 1, 'variant': 'v5',
        'formula_id': 'dil_bce_orig_token_ref_full_vocab_v1',
        'config': dict(loss_type='dil_bce_orig', beta=1., margin=0.,
            lambda_preference=1., lambda_topk_entropy=0., preference_topk_k=None,
            preference_topk_mode='independent', topk_include_gt=False,
            lambda_topk_mass=0., lambda_topk_chosen_mass=0., lambda_entropy=0.,
            selection='all', preference_entropy_top_percent=None,
            loss_chunk_positions=128, text_kl_weight=.01, temperature=2.,
            rho_min=.3, rho_max=.5, text_cap=.3, low_attention_mass=.1,
            low_attention_fraction=.1),
        'coefficients': dict(ce_weight=.1, sim_weight=.1),
        'semantics': dict(reference='immutable_initial_effective_teacher_v1',
            auxiliary_targets='immediate_original_text_exclude_crossing_special_padding_v1',
            interventions='independent_clean_prefix_voff_body_textoff_previous_text_v1',
            reduction='global_actual_window_original_blocks_empty_included_v1',
            ce_alignment_text_reduction='configured_stage1_accumulation_v1',
            text_target='detached_clean_full_vocab_temperature_squared_kl_v1',
            text_schedule='from_first_update_no_kl_warmup_v1'),
        'reference': dict(schema_version=1, base_content_sha256='a'*64,
            processor_sha256='b'*64, effective_state_sha256='c'*64,
            selected_parameter_names=['lora_A', 'output_delta'],
            formula_domain='teacher_forced_token'),
        'schedule': {'microsteps': 9, 'gradient_accumulation_steps': 4},
        'execution': {'tuning_method': 'dora', 'zero_stage': 1, 'ema_tau': 1.},
    }


@pytest.mark.parametrize('selection', ['all', 'clean', 'voff', 'gap', 'intersection'])
def test_original_schema_is_strictly_readable_but_cannot_resume_training(tmp_path, selection):
    saved = historical_contract()
    if selection != 'all':
        saved['config'].update(selection=selection, preference_entropy_top_percent=50)
    path = tmp_path/'stage1_v5.json'
    encoded = json.dumps(saved)
    path.write_text(encoded)
    assert load_v5_contract(tmp_path) == saved
    assert path.read_text() == encoded
    with pytest.raises(ValueError, match='block.mean.*fresh|fresh.*block.mean'):
        validate_v5_resume(saved, {})
    saved['semantics']['text_schedule'] = 'changed'
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError): load_v5_contract(tmp_path)


def test_new_schema_separates_token_dil_from_block_auxiliaries(tmp_path):
    saved = contract()
    assert saved['schema_version'] == 2
    assert saved['semantics']['reduction'] == 'global_actual_window_dil_tokens_aux_original_blocks_v2'
    (tmp_path/'stage1_v5.json').write_text(json.dumps(saved))
    assert load_v5_contract(tmp_path) == saved
    assert validate_v5_resume(saved, {}) == Stage1V5Config(**saved['config'])


@pytest.mark.parametrize('mutation', ['old_reduction', 'unknown', 'null', 'missing'])
def test_new_schema_rejects_malformed_or_conflicting_reduction(tmp_path, mutation):
    saved = contract()
    if mutation == 'old_reduction':
        saved['semantics']['reduction'] = historical_contract()['semantics']['reduction']
    elif mutation == 'unknown': saved['semantics']['reduction'] = 'unknown_token_reduction'
    elif mutation == 'null': saved['semantics']['reduction'] = None
    else: saved['semantics'].pop('reduction')
    (tmp_path/'stage1_v5.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError): load_v5_contract(tmp_path)
    with pytest.raises(ValueError): validate_v5_resume(saved, {})


@pytest.mark.parametrize('selector', ['voff', 'gap', 'intersection'])
def test_fresh_resolution_and_parser_reject_unsupported_token_selectors(selector):
    from utils_deepseed import parse_args
    with pytest.raises(ValueError, match='token.mean.*selection=all'):
        resolve_v5_config({'selection': selector, 'preference_entropy_top_percent': 50})
    with pytest.raises(ValueError, match='token.mean.*selection=all'):
        parse_args(['--stage1_variant', 'v5', '--stage1_v5_selection', selector,
                    '--preference_entropy_top_percent', '50'])


def test_parser_rejects_old_training_before_model_loading(tmp_path, monkeypatch):
    from utils_deepseed import parse_args
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    calls = []
    def forbidden(*a, **kw): calls.append('model'); raise AssertionError('model loaded')
    monkeypatch.setattr(AutoProcessor, 'from_pretrained', forbidden)
    monkeypatch.setattr(Qwen2_5_VLForConditionalGeneration, 'from_pretrained', forbidden)
    # The parser must reject the sidecar independently, before any composition
    # loader can be reached by the public entrypoint.
    checkpoint = tmp_path/'checkpoints/old/checkpoint-1'
    checkpoint.mkdir(parents=True)
    (checkpoint/'stage1_v5.json').write_text(json.dumps(historical_contract()))
    with pytest.raises(ValueError, match='block.mean.*fresh|fresh.*block.mean'):
        parse_args(['--resume_from_checkpoint_path', str(checkpoint)])
    assert calls == []


@pytest.mark.parametrize('case', ['historical', 'voff', 'gap', 'intersection'])
def test_direct_composition_rejects_before_any_model_or_processor_loader(tmp_path, monkeypatch, case):
    from deepspeed_runtime import bind_deepspeed_runtime_profile
    from stage1 import composition
    from tests.stage1.test_composition import stage1_args, fake_dependencies, REPO_ROOT
    args = stage1_args(tmp_path, '--gradient_checkpointing', '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '4', '--attention_implementation', 'sdpa',
        '--deepspeed_config', str(REPO_ROOT/'configs/deepspeed_zero1.json'),
        '--ema_tau', '1', '--vision_attention_implementation', 'flash_attention_2',
        '--ce_weight', '.1', '--sim_weight', '.1')
    args.stage1_variant = 'v5'
    args.stage1_v5_config = Stage1V5Config() if case == 'historical' else Stage1V5Config(
        selection=case, preference_entropy_top_percent=50)
    args.stage1_v5_explicit_values = {}
    profile = bind_deepspeed_runtime_profile(args, world_size=4)
    calls = []
    dependencies, *_ = fake_dependencies(calls)
    if case == 'historical':
        checkpoint = tmp_path/'old-checkpoint'
        checkpoint.mkdir()
        (checkpoint/'stage1_v5.json').write_text(json.dumps(historical_contract()))
        # The classification/tuning boundaries have separate tests. Supply a
        # classified old checkpoint here to isolate the objective boundary.
        decision = SimpleNamespace(checkpoint_path=checkpoint, peft_metadata_snapshot=None,
                                   artifact_kind=None)
        dependencies = replace(dependencies, resolve_resume=lambda *a: decision)
        monkeypatch.setattr(composition, '_validate_resume_tuning', lambda *a: None)
    def forbidden(*a, **kw): calls.append('LOADER'); raise AssertionError('loader called')
    dependencies = replace(dependencies, load_processor=forbidden, load_config=forbidden,
                           load_model=forbidden)
    pattern = 'block.mean.*fresh|fresh.*block.mean' if case == 'historical' else 'token.mean.*selection=all'
    with pytest.raises(ValueError, match=pattern):
        composition.build_stage1_training_components(args, dependencies=dependencies,
                                                     deepspeed_runtime_profile=profile)
    assert 'LOADER' not in calls


@pytest.mark.parametrize('mutation', ['selection', 'percent', 'reduction', 'units', 'schema', 'denominator'])
def test_clean_contract_roundtrips_and_rejects_changed_meaning(tmp_path, mutation):
    import copy
    from stage1.v5_resume import build_v5_contract
    from stage1.v5_reference import V5ReferenceIdentity
    config = resolve_v5_config({'selection': 'clean', 'preference_entropy_top_percent': 20,
        'lambda_topk_entropy': .1, 'preference_topk_k': 50})
    old = contract()
    saved = build_v5_contract(config, V5ReferenceIdentity(**old['reference']),
        **old['coefficients'], execution=old['execution'],
        schedule={'dil_denominator': 'actual_window_selected_dil_tokens'})
    assert saved['schema_version'] == 3
    assert saved['semantics']['reduction'] == 'global_actual_window_selected_dil_tokens_aux_original_blocks_v3'
    assert saved['semantics']['dil_sum_units'] == 'selected_token_loss_sum'
    (tmp_path/'stage1_v5.json').write_text(json.dumps(saved))
    assert load_v5_contract(tmp_path) == saved
    assert validate_v5_resume(saved, {}) == config
    if mutation in ('selection', 'percent'):
        values = ({'selection': 'all'} if mutation == 'selection'
                  else {'preference_entropy_top_percent': 50})
        with pytest.raises(ValueError, match='conflict'):
            validate_v5_resume(saved, values)
        return
    bad = copy.deepcopy(saved)
    if mutation == 'reduction': bad['semantics']['reduction'] = old['semantics']['reduction']
    elif mutation == 'units': bad['semantics']['dil_sum_units'] = 'eligible_token_loss_sum'
    elif mutation == 'schema': bad['schema_version'] = 2
    else: bad['schedule']['dil_denominator'] = 'actual_window_eligible_dil_tokens'
    (tmp_path/'stage1_v5.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError): load_v5_contract(tmp_path)
    with pytest.raises(ValueError): validate_v5_resume(bad, {})
