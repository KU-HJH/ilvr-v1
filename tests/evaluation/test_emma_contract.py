from dataclasses import asdict
import copy
import sys
import pytest

from emma_fixtures import CATEGORIES, make_case, write_rows
from evaluation.schema import ContractError


def test_parse_preserves_raw_answer_and_emma_raw_identity():
    from evaluation.io import parse_config, parse_sample, read_manifest
    from evaluation.schema import EmmaTarget
    case = make_case(1)
    sample = parse_sample(case['rows'][0])
    assert isinstance(sample.target, EmmaTarget) and asdict(sample.target)['answer'] == 'b'
    write_rows(case['manifest'], case['rows'])
    assert read_manifest(case['manifest'], parse_config(case['config'], case['root']), case['root']).samples[0].id == sample.id


@pytest.mark.parametrize('change', ['type','bool_index','range','content','choices','question','category','task'])
def test_invalid_target_contract_rejected(change):
    from evaluation.io import parse_sample
    case = make_case(1); row = copy.deepcopy(case['rows'][0])
    if change == 'type': row['target']['index'] = None
    elif change == 'bool_index': row['target']['index'] = True
    elif change == 'range': row['target']['index'] = 8
    elif change == 'content': row['target']['content'] = 'first'
    elif change == 'choices': row['model_input']['choices'] = []
    elif change == 'question': row['target']['question_type'] = 'unknown'
    elif change == 'category': row['target']['category'] = 1
    else: row['target']['task'] = None
    with pytest.raises(ContractError): parse_sample(row)


def test_open_ended_target_requires_null_index_empty_choices_and_matching_content():
    from evaluation.io import parse_sample
    case = make_case(1); row = case['rows'][0]
    row['model_input']['choices'] = []
    row['target'].update(question_type='open_ended', answer='x', index=None, content='x')
    assert parse_sample(row).target.answer == 'x'
    row['target']['question_type'] = 'open-ended'
    with pytest.raises(ContractError): parse_sample(row)
    row['target']['question_type'] = 'open_ended'
    row['target']['content'] = 'y'
    with pytest.raises(ContractError): parse_sample(row)


def test_emma_config_subset_is_allowlist_and_max_steps_is_null_only():
    from evaluation.io import parse_config, read_manifest
    case = make_case(2); raw = copy.deepcopy(case['config']); raw['categories'] = ['Math']
    config = parse_config(raw, case['root'])
    with pytest.raises(ContractError): read_manifest(case['manifest'], config, case['root'])
    for categories in ([], ['Math','Math'], ['Other']):
        bad = copy.deepcopy(case['config']); bad['categories'] = categories
        with pytest.raises(ContractError): parse_config(bad, case['root'])
    bad = copy.deepcopy(case['config']); bad['max_steps'] = 1
    with pytest.raises(ContractError): parse_config(bad, case['root'])
    case['config']['max_steps'] = None
    assert parse_config(case['config'], case['root']).max_steps is None


def test_registry_is_lazy_and_unknown_profile_rejected():
    from evaluation.registry import get_scorer
    sys.modules.pop('sympy', None)
    assert callable(get_scorer('emma', 'emma_safe_rules_v1')) and 'sympy' not in sys.modules
    with pytest.raises(ContractError): get_scorer('emma', 'unknown')


def test_old_id_prefix_and_serialized_shapes_remain_unchanged():
    from evaluation.io import parse_config, read_manifest
    from evaluation.schema import ModelInput, McqTarget, Provenance, Sample, Score
    sample = Sample(1,'x','comt','creation',ModelInput('',(),()),McqTarget(0,'A','x'),Provenance('d','r','c','s','x',{}))
    score = Score('x','creation','comt','p','1','parsed','A','raw',True,None,None)
    assert tuple(asdict(sample)) == ('schema_version','id','benchmark','subtask','model_input','target','provenance')
    assert tuple(asdict(score)) == ('id','category','benchmark','profile','scorer_version','status','parsed_answer','extraction_method','correctness','simulation','error')
    case = make_case(1); case['rows'][0]['benchmark'] = 'comt'; case['rows'][0]['id'] = case['rows'][0]['provenance']['source_id']
    write_rows(case['manifest'], case['rows'])
    case['config']['benchmark']='comt'; case['config']['profile']='comt_reference_rules_v1'; case['config']['categories']=['creation']
    with pytest.raises(ContractError): read_manifest(case['manifest'], parse_config(case['config'],case['root']),case['root'])


@pytest.mark.parametrize('choices,answer,index', [
    (['only'], 'A', 0),
    (['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'], 'I', 8),
])
def test_manifest_rejects_mcq_choice_population_outside_two_to_eight(choices, answer, index):
    from evaluation.io import parse_sample
    row = make_case(1)['rows'][0]
    row['model_input']['choices'] = choices
    row['target'].update(answer=answer, index=index, content=choices[index])
    with pytest.raises(ContractError):
        parse_sample(row)


@pytest.mark.parametrize('choices,answer,index', [
    (['one', 'two'], 'b', 1),
    (['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'], 'h', 7),
])
def test_manifest_accepts_two_and_eight_choice_boundaries(choices, answer, index):
    from evaluation.io import parse_sample
    row = make_case(1)['rows'][0]
    row['model_input']['choices'] = choices
    row['target'].update(answer=answer, index=index, content=choices[index])
    sample = parse_sample(row)
    assert (sample.target.answer, sample.target.index) == (answer, index)
