"""Causal witness rejection tests: corrupted evidence must never pass."""
import copy
import importlib
import importlib.util

import pytest
import torch


def probe():
    assert importlib.util.find_spec('stage2.reference_sampler_probe') is not None, 'sampler witness is missing'
    return importlib.import_module('stage2.reference_sampler_probe')


def evidence():
    records = []
    for rank in range(4):
        records.append(dict(event='init', rank=rank, world_size=4, hostname=f'node{rank//2}',
                            gas=2, microbatch=1, fixed_denominator=8, zero_stage=2))
        for micro in range(3):
            index = micro*4+rank
            padding = index == 11
            records.append(dict(event='example', rank=rank, micro=micro, update=1 if micro<2 else 2,
                                sample_id='s0' if padding else f's{index}',
                                sample_index=-1 if padding else index, weight=0.0 if padding else 1.0,
                                latent_blocks=1+index%2, reference_gradient_max_abs=0.0 if padding else 1.0,
                                loss=0.0 if padding else 1.0, reference_zero_hook_calls=0))
            records.append(dict(event='collective', rank=rank, micro=micro, sequence=micro,
                                layout=[dict(name='p', shape=[2], dtype='torch.bfloat16')],
                                slices=[dict(name='p', offset=0, numel=2)],
                                tensor_shapes=[[2]], communication_dtype='torch.float32', reduce_rank=None))
            records.append(dict(event='backward', rank=rank, micro=micro, parameter_count=1,
                                gradient_max_abs=0.0 if padding else 0.5, finite=True))
        for update, micros, count in [(1, 2, 8), (2, 1, 3)]:
            records.append(dict(event='update', rank=rank, update=update, microsteps=micros,
                                actual_count=count, fixed_denominator=8,
                                fixed_gradient=dict(pass_=True), completed=True))
            records[-1]['fixed_gradient']={'pass':True,'relative_l2':0.0,'relative_l2_limit':.02,'coordinate':{'pass':True,'finite':True,'expected_l2':1.0,'l2_error':0.0,'max_abs_error':0.0,'rtol':0.02,'atol':1e-7}}
    return records


def test_complete_four_rank_full_and_tail_evidence_passes():
    assert probe().check_records(evidence(), [f's{i}' for i in range(11)])['pass']


@pytest.mark.parametrize('fault', ['missing_rank','nonzero_weight','padding_gradient','bucket_order','denominator','gradient','missing_completion','duplicate','one_host','missing_collective'])
def test_corrupted_witness_cannot_pass(fault):
    records=copy.deepcopy(evidence())
    if fault=='missing_rank': records=[r for r in records if r['rank']!=3]
    if fault=='nonzero_weight': next(r for r in records if r['event']=='example' and r['sample_index']==-1)['weight']=1.0
    if fault=='padding_gradient': next(r for r in records if r['event']=='backward' and r['rank']==3 and r['micro']==2)['gradient_max_abs']=0.01
    if fault=='bucket_order': next(r for r in records if r['event']=='collective' and r['rank']==3)['slices'][0]['name']='other'
    if fault=='denominator': next(r for r in records if r['event']=='update' and r['update']==2)['fixed_denominator']=3
    if fault=='gradient': next(r for r in records if r['event']=='update')['fixed_gradient']['pass']=False
    if fault=='missing_completion': records.pop()
    if fault=='duplicate': next(r for r in records if r['event']=='example' and r['sample_index']==10)['sample_id']='s0'
    if fault=='one_host':
        for r in records:
            if r['event']=='init': r['hostname']='same'
    if fault=='missing_collective': records=[r for r in records if not (r['event']=='collective' and r['micro']==2)]
    result=probe().check_records(records,[f's{i}' for i in range(11)])
    assert not result['pass'], fault
    assert result['failures']


def test_collective_slices_preserve_actual_payload_order_and_parameter_offsets():
    # Reordering the payload must change its semantic description even when shapes match.
    flat=torch.zeros(7)
    layout=[{'name':'a','shape':[3],'dtype':str(flat.dtype),'numel':3},
            {'name':'b','shape':[4],'dtype':str(flat.dtype),'numel':4}]
    slices=probe().describe_collective_slices(flat,layout,[flat[4:],flat[:4]])
    assert slices==[{'name':'b','offset':1,'numel':3},
                    {'name':'a','offset':0,'numel':3},
                    {'name':'b','offset':0,'numel':1}]
    with pytest.raises(RuntimeError,match='outside'):
        probe().describe_collective_slices(flat,layout,[torch.zeros(3)])


def test_arithmetic_acceptance_uses_preapproved_global_l2_but_preserves_coordinate_failure():
    expected=[torch.tensor([1.,1e-6])]
    actual=[torch.tensor([1.01,2e-6])]
    result=probe().compare_fixed_gradient(expected,actual)
    assert result['pass']
    assert not result['coordinate']['pass']
    assert result['relative_l2'] == pytest.approx(0.01,abs=1e-6)
    assert not probe().compare_fixed_gradient([torch.tensor([3./8])],[torch.tensor([1.])])['pass']
    assert not probe().compare_fixed_gradient(expected,[torch.tensor([float('nan'),0.])])['pass']


def test_reporter_rechecks_numeric_evidence_instead_of_trusting_pass_flag():
    records=evidence()
    first=next(r for r in records if r['event']=='update')
    first['fixed_gradient']={'pass':True,'relative_l2':0.7,'relative_l2_limit':.02,
                             'coordinate':{'finite':True,'expected_l2':1.0,'l2_error':.7}}
    assert not probe().check_records(records,[f's{i}' for i in range(11)])['pass']


@pytest.mark.parametrize('actual,accepted', [(0.,True),(1e-12,False)])
def test_zero_reference_requires_exact_zero_error_in_comparator_and_reporter(actual,accepted):
    comparison=probe().compare_fixed_gradient([torch.tensor([0.])],[torch.tensor([actual])])
    assert comparison['pass'] is accepted
    assert comparison['coordinate']['expected_l2']==0.
    records=evidence()
    next(r for r in records if r['event']=='update')['fixed_gradient']=comparison
    assert probe().check_records(records,[f's{i}' for i in range(11)])['pass'] is accepted
