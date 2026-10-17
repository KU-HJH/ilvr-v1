"""CPU negative controls for the fixed mixed20 runtime evidence contract."""
import copy
import json
from types import SimpleNamespace as N

import pytest
import torch

from stage2 import reproduction_probe as probe
from tests.stage2.test_historical_balanced_sampler import PROBE_ORDER


def comparison():
    return dict(pass_=True, finite=True, expected_l2=1., l2_error=0., max_abs_error=0., rtol=.02, atol=1e-7)


def evidence():
    records = []
    counts = {f's{i}':c for i,c in enumerate([1]*8+[3]*8+[6]*4)}
    order = [f's{i}' for i in PROBE_ORDER]
    for rank in range(4):
        records.append(dict(event='init',rank=rank,world_size=4,hostname=f'node{rank//2}',
            gas=4,microbatch=1,fixed_denominator=16,zero_stage=2,
            sampler_policy='historical_balanced_2rank',model_accepts_loss_kwargs=False,
            compute_loss_func_present=False,parameter_names=['p']))
        for micro in range(5):
            sid=order[micro*4+rank]
            records.append(dict(event='example',rank=rank,micro=micro,update=1 if micro<4 else 2,
                sample_id=sid,weight=1.,latent_blocks=counts[sid],rng_preserved=True,
                reference_zero_hook_calls=0))
            records.append(dict(event='collective',rank=rank,micro=micro,sequence=micro,
                layout=[dict(name='p',shape=[2],dtype='torch.bfloat16',numel=2)],
                slices=[dict(name='p',offset=0,numel=2)],tensor_shapes=[[2]],
                communication_dtype='torch.float32',reduce_rank=None))
            records.append(dict(event='backward',rank=rank,micro=micro,parameter_count=1,
                finite=True,gradient_max_abs=.5))
        for update,micros,count in [(1,4,16),(2,1,4)]:
            r=dict(event='update',rank=rank,update=update,microsteps=micros,actual_count=count,
                fixed_denominator=16,completed=True,clip=1.,clip_shared_global_factor=True,
                clip_once_per_partition_group=True,clip_calls_global_norm=[1.],clip_group_indices=[0],
                clip_divisors=[1.000001],options=[dict(lr=0. if update==1 else 5e-6)])
            for key in ('historical_fixed_gradient','historical_fixed_adam_delta','effective_gradient_adam_delta'):
                r[key]=comparison();r[key]['pass']=r[key].pop('pass_')
            records.append(r)
    return records,order,counts


def check(records,order,counts):
    assert hasattr(probe,'check_historical_records'), 'mixed20 checker is missing'
    return probe.check_historical_records(records,order,counts)


def test_complete_mixed20_two_update_evidence_passes():
    records,order,counts=evidence()
    result=check(records,order,counts)
    assert result['pass'] and result['fixed_denominator']==16


@pytest.mark.parametrize('fault', ['missing_rank','missing_backward','order','block_count','weight',
    'update_assignment','one_host','layout_order','slice_offset','dtype','collective_count',
    'sequence','tail_denominator','tail_count','independent_gradient','effective_adam',
    'missing_completion','clip','gradient_numeric','zero_gradient','loss_kwargs'])
def test_corrupted_runtime_evidence_fails_closed(fault):
    records,order,counts=evidence()
    row=lambda event:next(r for r in records if r['event']==event and r['rank']==3)
    if fault=='missing_rank':records=[r for r in records if r['rank']!=3]
    elif fault=='missing_backward':records.remove(row('backward'))
    elif fault=='order':row('example')['sample_id']='s0'
    elif fault=='block_count':row('example')['latent_blocks']=1
    elif fault=='weight':row('example')['weight']=0.
    elif fault=='update_assignment':row('example')['update']=2
    elif fault=='one_host':
        for r in records:
            if r['event']=='init':r['hostname']='same'
    elif fault=='layout_order':row('collective')['layout'][0]['name']='other'
    elif fault=='slice_offset':row('collective')['slices'][0]['offset']=1
    elif fault=='dtype':row('collective')['communication_dtype']='torch.bfloat16'
    elif fault=='collective_count':records.remove(row('collective'))
    elif fault=='sequence':row('collective')['sequence']=10
    elif fault=='tail_denominator':next(r for r in records if r['event']=='update' and r['update']==2)['fixed_denominator']=4
    elif fault=='tail_count':next(r for r in records if r['event']=='update' and r['update']==2)['actual_count']=3
    elif fault=='independent_gradient':row('update')['historical_fixed_gradient']['pass']=False
    elif fault=='effective_adam':row('update')['effective_gradient_adam_delta']['pass']=False
    elif fault=='gradient_numeric':row('update')['historical_fixed_gradient']['l2_error']=float('nan')
    elif fault=='zero_gradient':row('backward')['gradient_max_abs']=0.
    elif fault=='missing_completion':row('update')['completed']=False
    elif fault=='clip':row('update')['clip_once_per_partition_group']=False
    elif fault=='loss_kwargs':row('init')['model_accepts_loss_kwargs']=True
    result=check(records,order,counts)
    assert not result['pass'] and result['failures'],fault


def test_shared_capture_observes_real_views_before_original_hooks(tmp_path):
    # Transport stand-in only: captures actual tensors, without claiming DS/NCCL correctness.
    p=torch.nn.Parameter(torch.ones(3));q=torch.nn.Parameter(torch.ones(4))
    flat=torch.zeros(7)
    events=[]
    observer=N(named=[('p',p),('q',q)],micro=2,sequence=0,rank=0,hook_calls=0,actual_gradients={})
    observer.write=events.append
    calls=[]
    zero=N(reduce_scatter=True,contiguous_gradients=True,loss_scale=1.,bit16_groups=[[p,q]],
        ipg_buckets={torch.float32:N(params=[(0,0,0),(0,1,1)])},
        get_gradient_for_reduction=lambda parameter:parameter.grad,
        process_gradients=lambda parameter,group:calls.append(('process',parameter,group)))
    def reduce(bucket,dtype,**kwargs):
        calls.append(('reduce',[t.clone() for t in bucket],dtype,kwargs))
        assert events[-1]['event']=='collective'  # payload observed before original transport
        return 'original-result'
    zero.allreduce_bucket=reduce
    zero.average_tensor=lambda tensor,dtype:zero.allreduce_bucket([tensor[4:],tensor[:4]],dtype,rank=2)
    assert hasattr(probe,'install_collective_observation'), 'shared layout capture is missing'
    probe.install_collective_observation(observer,zero)
    p.grad=torch.arange(1.,4.)
    zero.process_gradients(p,0)
    assert observer.hook_calls==1 and observer.actual_gradients['p']==(True,3.)
    assert calls[0]==('process',p,0)
    assert zero.average_tensor(flat,torch.float32)=='original-result'
    assert events[0]['slices']==[dict(name='q',offset=1,numel=3),dict(name='p',offset=0,numel=3),dict(name='q',offset=0,numel=1)]
    assert events[0]['micro']==2 and events[0]['sequence']==0 and observer.sequence==1
    assert calls[1][2]==torch.float32 and calls[1][3]['rank']==2


def test_historical_observer_installed_engine_hooks_preserve_normal_backward(observer_factory):
    engine=capture_engine()
    p=engine.module.p

    observer=observer_factory(engine,'historical-hooks',historical=True)
    output=engine(1.)
    loss=output['logits'].sum()+output['nested'][0]
    observer.observe_loss(loss,dict(sample_ids=['real'],input_ids=torch.tensor([[7,1,7,1,7,1]])))
    engine.backward(loss,scale_wrt_gas=False)
    observer._finish_backward()
    assert engine.optimizer.events==['leaf','reduce']
    assert torch.equal(observer.total[0],p.grad)
    observer.stream.flush()
    records=[json.loads(line) for line in open(observer.stream.name)]
    example=next(r for r in records if r['event']=='example')
    assert example['latent_blocks']==3 and example['weight']==1. and example['micro']==0
    assert len([r for r in records if r['event']=='backward'])==1
    assert next(r for r in records if r['event']=='collective')['layout'][0]['name']=='p'
    # Intentional unscaled production backward above differs from raw reference/4.
    # This advisory must report the mismatch and still call the original leaf hook.
    advisory_rows=[r for r in records if r['event']=='selected_leaf_advisory']
    assert len(advisory_rows)==1, 'selected leaf advisory is missing'
    advisory=advisory_rows[0]
    assert advisory['comparison']['pass'] is False
    assert advisory['parameter']=='p' and advisory['hook_ordinal']==1
    assert advisory['sample_id']=='real' and advisory['micro']==0
    assert next(r for r in records if r['event']=='backward')['selected_leaf_hook_count']==1


def capture_engine():
    from tests.stage2.test_reproduction_engine_hooks import Engine
    engine=Engine()
    engine.module.config=N(latent_start_token_id=7)
    zero=engine.optimizer
    zero.reduce_scatter=True;zero.loss_scale=1.
    zero.get_gradient_for_reduction=lambda parameter:parameter.grad
    zero.allreduce_bucket=lambda *a,**kw:None
    zero.average_tensor=lambda tensor,dtype:zero.allreduce_bucket([tensor],dtype)
    original_reduce=zero.overlapping_partition_gradients_reduce_epilogue
    def reduce():
        zero.bucket.params=[(0,0,0)]
        zero.average_tensor(zero.bucket.buffer[0],torch.float32)
        return original_reduce()
    zero.overlapping_partition_gradients_reduce_epilogue=reduce
    return engine


def test_reference_observer_shared_capture_keeps_legacy_backward(tmp_path,monkeypatch):
    import torch.distributed as dist
    from stage2.reference_sampler_probe import ReferenceSamplerObserver
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dist,'get_rank',lambda:0)
    monkeypatch.setattr(dist,'get_world_size',lambda:4)
    engine=capture_engine()
    trainer=N(deepspeed_runtime_profile=N(zero_stage=2),state=N(global_step=0),
        args=N(per_device_train_batch_size=1,gradient_accumulation_steps=2),
        stage2_sampler_policy='reference_random',model_accepts_loss_kwargs=False,compute_loss_func=None)
    observer=ReferenceSamplerObserver(trainer,engine,'results/reference-hooks')
    output=engine(1.)
    loss=output['logits'].sum()+output['nested'][0]
    observer.observe_loss(loss,dict(sample_ids=['real'],stage2_sample_weight=1.,
        stage2_sample_index=0,stage2_sample_latent_block_count=3))
    engine.backward(loss/2,scale_wrt_gas=False)
    observer._finish_backward()
    assert engine.optimizer.events==['leaf','reduce']
    torch.testing.assert_close(engine.module.p.grad,observer.total[0]/2)
    observer.stream.close()


# Reuse real installed DS engine fixture; historical mode adds only capture APIs.
from tests.stage2.test_reproduction_engine_hooks import observer_factory
