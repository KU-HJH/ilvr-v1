"""Saved objective resumes inherit omissions and reject semantic conflicts."""
import importlib
import json
from dataclasses import asdict

import pytest
from stage1.v5_config import Stage1V5Config


def api():
    assert importlib.util.find_spec('stage1.v5_resume'), 'saved objective implementation missing'
    return importlib.import_module('stage1.v5_resume')


def contract():
    resume = api()
    from stage1.v5_reference import V5ReferenceIdentity
    reference = V5ReferenceIdentity(1, 'a' * 64, 'b' * 64, 'c' * 64, ('lora_A', 'output_delta'))
    config = Stage1V5Config(lambda_topk_entropy=0.1, preference_topk_k=50)
    return resume.build_v5_contract(config, reference, ce_weight=0.1, sim_weight=0.1,
                                    schedule={'microsteps': 9, 'gradient_accumulation_steps': 4},
                                    execution={'tuning_method': 'dora', 'zero_stage': 1, 'ema_tau': 1.})


def test_saved_config_inherits_omissions_and_hyphen_alias(tmp_path):
    resume = api()
    saved = contract()
    (tmp_path / 'stage1_v5.json').write_text(json.dumps(saved))
    loaded = resume.load_v5_contract(tmp_path)
    result = resume.validate_v5_resume(loaded, {'stage1_loss_type': 'dil-bce-orig'})
    assert result.lambda_topk_entropy == 0.1 and result.preference_topk_k == 50
    assert result.text_kl_weight == 0.01
    assert resume.validate_v5_resume(saved, {'--stage1-loss-type': 'dil-bce-orig'}) == result


@pytest.mark.parametrize('explicit', [{'lambda_topk_entropy': 0.}, {'stage1_variant': 'v3'},
                                      {'ce_weight': 1.}, {'sim_weight': 0.5}, {'ema_tau': 0.9},
                                      {'step_preference_beta': 10.}, {'unknown': 1}])
def test_explicit_resume_conflicts_reject(explicit):
    with pytest.raises(ValueError):
        api().validate_v5_resume(contract(), explicit)


@pytest.mark.parametrize('key', ['variant', 'formula_id', 'reference', 'semantics', 'schema_version'])
def test_tampered_saved_contract_rejects(key):
    saved = contract()
    saved[key] = 'changed'
    with pytest.raises(ValueError):
        api().validate_v5_resume(saved, {})


def test_missing_sidecar_never_silently_relabels_legacy(tmp_path):
    with pytest.raises(ValueError):
        api().load_v5_contract(tmp_path)


@pytest.mark.parametrize('execution', [{'tuning_method': 'full'}, {'ema_tau': 0.9}, {'zero_stage': 3}])
def test_build_rejects_unsupported_runtime(execution):
    resume = api()
    from stage1.v5_reference import V5ReferenceIdentity
    identity = V5ReferenceIdentity(1, 'a' * 64, 'b' * 64, 'c' * 64, ('lora_A',))
    with pytest.raises(ValueError):
        resume.build_v5_contract(Stage1V5Config(), identity, ce_weight=0.1, sim_weight=0.1,
                                 schedule={}, execution=execution)



def adaptive_contract():
    from stage1.query import Stage1QueryConfig, query_execution_config
    saved = contract()
    saved['execution']['query_config'] = query_execution_config(
        Stage1QueryConfig('adaptive', .55, 'last'), ())
    return saved


def test_query_resume_inherits_saved_settings_without_changing_loss_formula():
    saved = adaptive_contract()
    original_formula = saved['formula_id']
    result = api().resolve_v5_query_resume(saved, {})
    assert (result.mode, result.text_gamma, result.attention_layers) == ('adaptive', .55, 'last')
    assert saved['formula_id'] == original_formula
    assert saved['execution']['query_config']['formula_id'] != original_formula
    assert api().resolve_v5_query_resume(contract(), {}).mode == 'legacy'
    with pytest.raises(ValueError, match='conflict'):
        api().resolve_v5_query_resume(contract(), {'query_mode': 'adaptive'})


@pytest.mark.parametrize('mutation', ['gamma', 'layers', 'short_last4', 'formula', 'missing', 'reference', 'loss'])
def test_query_resume_rejects_tampered_semantics_and_identity(mutation):
    saved = adaptive_contract()
    query = saved['execution']['query_config']
    if mutation == 'gamma': query['text_gamma'] = float('nan')
    elif mutation == 'layers': query['resolved_layer_indices'] = []
    elif mutation == 'short_last4': query['resolved_layer_indices'] = [0, 1]
    elif mutation == 'formula': query['formula_id'] = 'other'
    elif mutation == 'missing': query.pop('text_gamma')
    elif mutation == 'reference': saved['reference']['schema_version'] = 77
    else: saved['formula_id'] = 'other'
    with pytest.raises(ValueError): api().resolve_v5_query_resume(saved, {})


@pytest.mark.parametrize('explicit', [{'query_mode': 'legacy'}, {'history_text_gamma': .9},
                                    {'history_attention_layers': 'all'}])
def test_query_resume_rejects_explicit_semantic_changes(explicit):
    with pytest.raises(ValueError, match='conflict'):
        api().resolve_v5_query_resume(adaptive_contract(), explicit)


def test_query_parser_restores_saved_mode_before_fresh_conflict_validation(tmp_path):
    from utils_deepseed import parse_args
    saved = adaptive_contract()
    saved['execution'].update(tuning_method='dora', per_device_train_batch_size=1,
                              attention_implementation='sdpa')
    checkpoint = tmp_path/'checkpoints/query-resume/checkpoint-1'
    checkpoint.mkdir(parents=True)
    (checkpoint/'stage1_v5.json').write_text(json.dumps(saved))
    base = ['--resume_from_checkpoint_path', str(checkpoint)]
    args = parse_args([*base, '--history_text_gamma=.55'])
    assert args.stage1_query_config.mode == 'adaptive'
    assert args.stage1_query_config.attention_layers == 'last'
    with pytest.raises(ValueError, match='textual_states'):
        parse_args([*base, '--textual_sta=previous'])
    with pytest.raises(ValueError, match='conflict'):
        parse_args([*base, '--history_attention_layers=all'])



@pytest.mark.parametrize('key', ['query', 'targets', 'history_attention', 'history_teacher', 'history_metric'])
def test_current_source_binding_cannot_lose_its_query_contract(key):
    saved = contract()
    saved['execution']['source_sha256'] = {key: 'a'*64}
    with pytest.raises(ValueError, match='missing query_config'):
        api().resolve_v5_query_resume(saved, {})


@pytest.mark.parametrize(('mode', 'selector'), [
    ('legacy', 'last4'), ('adaptive', 'last'), ('adaptive', 'all'), ('adaptive', 'last4')])
def test_runtime_contract_records_effective_topology_and_independent_query_semantics(tmp_path, mode, selector):
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from peft_common import resolve_qwen2_5_vl_model
    from stage1.query import Stage1QueryConfig
    from stage1.v5_reference import V5ReferenceIdentity
    from stage1.types import DEFAULT_STAGE1_TARGET_SUFFIXES
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    from tests.stage1.v5.test_trainer import integrated_fixture
    trainer, *_ = integrated_fixture(tmp_path)
    trainer.v5_reference_identity = V5ReferenceIdentity(1, 'a'*64, 'b'*64, 'c'*64, ('weight',))
    trainer.stage1_peft_metadata = SimpleNamespace(base_model_reference='original/tiny',
        requested_target_suffixes=DEFAULT_STAGE1_TARGET_SUFFIXES)
    trainer.coverage_p = .9; trainer.image_pool_k = 2; trainer.helper_group_L = 2
    trainer.textual_states = 'previous'
    trainer.query_config = Stage1QueryConfig(mode, .55, selector)
    resolve_qwen2_5_vl_model(trainer.model).config.compress_strategy = 'average'
    trainer.args.bf16 = True; trainer.args.max_seq_length = 32768; trainer.args.max_steps = 3
    trainer.deepspeed_runtime_profile = DeepSpeedRuntimeProfile(True, 1, None, 'a'*64, 4)
    trainer.train_dataset = list(range(10))
    loader = DataLoader(trainer.train_dataset, batch_size=1)
    if mode == 'adaptive' and selector != 'last':
        with pytest.raises(ValueError, match='final|last.only'):
            trainer._build_v5_runtime_contract(loader)
        return
    saved = trainer._build_v5_runtime_contract(loader)
    assert saved['schedule']['dil_denominator'] == 'actual_window_eligible_dil_tokens'
    assert saved['schedule']['auxiliary_denominator'] == 'actual_window_original_blocks_including_empty'
    assert saved['semantics']['dil_sum_units'] == 'eligible_token_loss_sum'
    assert saved['semantics']['auxiliary_sum_units'] == 'original_block_token_mean_sum'
    query = saved['execution']['query_config']
    assert query['mode'] == mode
    assert query['formula_id'] != saved['formula_id']
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    assert saved['execution']['source_sha256'] == {
        key: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for key, path in RUNTIME_SOURCE_PATHS.items()}
    assert saved['execution']['operator_profile'] == {
        'id': 'stage1_v5_vectorized_ops_v1', 'attention_query_chunk_max': 32,
        'attention_score_probability_target_bytes': 33554432}
    if mode == 'adaptive':
        from tests.stage1.test_history_query import NEW_QUERY
        assert query == {**NEW_QUERY, 'text_gamma': .55}
    else:
        assert query == {'schema_version': 1, 'mode': 'legacy', 'formula_id': 'stage1_legacy_query_v1'}


def test_historical_query_normalization_preserves_source_and_reference_guards():
    import copy
    from stage1.query import Stage1QueryConfig, query_execution_config
    from stage1.v5_trainer import CustomTrainerStage1V5
    historical = contract()
    historical['execution']['source_sha256'] = {'v5_trainer': 'a'*64}
    current = copy.deepcopy(historical)
    current['execution']['query_config'] = query_execution_config(Stage1QueryConfig(), ())
    normalize = CustomTrainerStage1V5._effective_v5_contract
    assert normalize(historical) == normalize(current)
    current['execution']['source_sha256']['v5_trainer'] = 'b'*64
    assert normalize(historical) != normalize(current)
    current = copy.deepcopy(historical)
    current['reference']['base_content_sha256'] = 'd'*64
    assert normalize(historical) != normalize(current)



@pytest.mark.parametrize('entrypoint', ['build', 'load', 'loss_resume', 'query_resume'])
def test_explicit_null_query_metadata_rejects_before_resume_inheritance(tmp_path, entrypoint):
    saved = contract()
    saved['execution']['query_config'] = None
    resume = api()
    with pytest.raises(ValueError, match='query.*mapping'):
        if entrypoint == 'build':
            from stage1.v5_reference import V5ReferenceIdentity
            resume.build_v5_contract(Stage1V5Config(**saved['config']),
                V5ReferenceIdentity(**saved['reference']), **saved['coefficients'],
                schedule=saved['schedule'], execution=saved['execution'])
        elif entrypoint == 'load':
            checkpoint = tmp_path/'checkpoints/null-query/checkpoint-1'
            checkpoint.mkdir(parents=True)
            (checkpoint/'stage1_v5.json').write_text(json.dumps(saved))
            resume.load_v5_contract(checkpoint)
        elif entrypoint == 'loss_resume':
            resume.validate_v5_resume(saved, {})
        else:
            resume.resolve_v5_query_resume(saved, {})


RUNTIME_SOURCE_PATHS = {
    'v5_trainer': 'src/stage1/v5_trainer.py',
    'stage1_trainer': 'src/stage1/trainer.py',
    'transformers_trainer': 'transformers/src/transformers/trainer.py',
    'query': 'src/stage1/query.py', 'targets': 'src/stage1/targets.py',
    'history_attention': 'src/stage1/history_attention.py',
    'history_teacher': 'src/stage1/history_teacher.py',
    'history_metric': 'src/stage1/history_metric.py',
    'v5_execution': 'src/stage1/v5_execution.py',
    'v5_selection': 'src/stage1/v5_selection.py',
    'v5_trace': 'transformers/src/transformers/models/qwen2_5_vl/stage1_v5_trace.py',
    'v5_replay': 'transformers/src/transformers/models/qwen2_5_vl/stage1_v5_replay.py',
}


@pytest.mark.parametrize('selector,indices', [('last4', [0, 1, 2, 3]), ('all', [0, 1]), ('last', [35])])
def test_old_native_query_sidecar_roundtrips_for_inspection_but_cannot_resume(tmp_path, selector, indices):
    from stage1.query import validate_query_execution_config
    from tests.stage1.test_history_query import OLD_NATIVE_QUERY
    saved = contract()  # Current outer token-mean isolates the query guard.
    old_query = {**OLD_NATIVE_QUERY, 'attention_layers': selector, 'resolved_layer_indices': indices}
    saved['execution']['query_config'] = old_query
    (tmp_path / 'stage1_v5.json').write_text(json.dumps(saved))
    loaded = api().load_v5_contract(tmp_path)
    assert loaded == saved and loaded['execution']['query_config'] == old_query
    assert validate_query_execution_config(old_query).attention_layers == selector
    with pytest.raises(ValueError, match='original reviewed code.*fresh run'):
        api().validate_v5_resume(loaded, {})
    with pytest.raises(ValueError, match='original reviewed code.*fresh run'):
        api().resolve_v5_query_resume(loaded, {})


@pytest.mark.parametrize('historical', [False, True])
def test_query_parser_rejects_unsupported_training_before_any_loader(tmp_path, monkeypatch, historical):
    import transformers
    from utils_deepseed import parse_args
    from tests.stage1.test_history_query import OLD_NATIVE_QUERY
    calls = []
    def load(*args, **kwargs):
        calls.append(args)
        pytest.fail('model/processor loader reached before query validation')
    for cls in (transformers.AutoProcessor, transformers.AutoConfig,
                transformers.Qwen2_5_VLForConditionalGeneration):
        monkeypatch.setattr(cls, 'from_pretrained', load)
    if historical:
        saved = contract()
        saved['execution']['query_config'] = dict(OLD_NATIVE_QUERY)
        (tmp_path / 'stage1_v5.json').write_text(json.dumps(saved))
        flags = ['--resume_from_checkpoint_path', str(tmp_path)]
        message = 'original reviewed code.*fresh run'
    else:
        flags = ['--stage1_variant', 'v5', '--tuning_method', 'dora',
                 '--per_device_train_batch_size', '1', '--query_mode', 'adaptive',
                 '--history_attention_layers', 'all']
        message = 'final|last.only'
    with pytest.raises(ValueError, match=message):
        parse_args(flags)
    assert calls == []
