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
