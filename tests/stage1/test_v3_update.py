"""CPU boundary contracts; simulated collectives/master storage are not ZeRO acceptance."""
import importlib
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from stage1.v3_config import Stage1V3Config
from stage1.v3_gradients import V3Gradients
from stage1.trainer import _DeepSpeedStepAdapter


def api():return importlib.import_module('stage1.v3_update')


def result(r,rv,count):
    zero=torch.tensor(0.)
    return V3Gradients((torch.tensor(r,dtype=torch.float32),),(torch.tensor(rv,dtype=torch.float32),),count,zero,zero)


def controller():
    p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.bfloat16))
    return api().V3GainController((p,),Stage1V3Config()),p


def fake_engine(monkeypatch,c,p,*,skip=False,fail=False):
    # Actual AdamW on an FP32 master; only ZeRO storage/gather is simulated.
    module=api();events=[]
    master=torch.nn.Parameter(p.detach().float().clone())
    adam=torch.optim.AdamW([master],lr=.03,weight_decay=.01)
    p._hp_mapping=None
    p.get_full_hp_param=lambda:master.detach().clone()
    p.set_full_hp_param=lambda value:master.data.copy_(value)
    optimizer=SimpleNamespace(partition_gradients=False,cpu_offload=False,bit16_groups=[[p]],
        single_partition_of_fp32_groups=[master])
    def sync():
        events.append('bf16_sync');p.data.copy_(master.detach().to(p))
    optimizer.update_lp_params=sync
    def step():
        events.append('adamw')
        if fail:raise RuntimeError('base step failed')
        if not skip:
            master.grad=torch.tensor([.5,-.25]);adam.step();adam.zero_grad()
    engine=SimpleNamespace(optimizer=optimizer,zero_optimization_stage=lambda:1,
        step=step,is_gradient_accumulation_boundary=lambda:True,was_step_applied=lambda:not skip)
    monkeypatch.setattr(module,'safe_get_full_fp32_param',lambda param:master.detach().clone())
    def write(param,value):events.append('gain_master_apply');master.data.copy_(value)
    monkeypatch.setattr(module,'safe_set_full_fp32_param',write)
    return engine,master,adam,events


def test_all_four_ranks_reduce_sums_and_count_in_same_order_including_zero(monkeypatch):
    module=api();traces=[]
    for rank in range(4):
        c,p=controller()
        for count in (rank//2,rank-rank//2):c.accumulate(result([2*count,-count],[count,count],count))
        expected=[torch.tensor(0,dtype=torch.int32),torch.tensor(rank,dtype=torch.int64),
                  torch.tensor([2*rank,-rank],dtype=torch.float32),torch.tensor([rank,rank],dtype=torch.float32),
                  torch.tensor(0,dtype=torch.int32)]
        totals=[torch.tensor(0),torch.tensor(6),torch.tensor([12.,-6.]),torch.tensor([6.,6.]),torch.tensor(0)]
        trace=[]
        def reduce(tensor,op=None,group=None):
            i=len(trace);assert torch.equal(tensor,expected[i]);trace.append((tensor.dtype,tuple(tensor.shape)))
            tensor.copy_(totals[i])
        monkeypatch.setattr(module.dist,'is_initialized',lambda:True)
        monkeypatch.setattr(module.dist,'all_reduce',reduce)
        c.prepare()
        assert c.global_block_count==6
        torch.testing.assert_close(c.direction[0],torch.tensor([1.5,-1.5]))
        traces.append(trace)
    assert traces==[traces[0]]*4


@pytest.mark.parametrize('skip',[False,True])
@pytest.mark.parametrize('weighting',['none','token_entropy'])
def test_adapter_applies_gain_after_adam_only_on_success_and_preserves_moments(monkeypatch,skip,weighting):
    c,p=controller()
    if weighting=='token_entropy':
        c=api().V3GainController((p,),Stage1V3Config(gain_weighting=weighting))
        zero=torch.tensor(0.)
        c.accumulate(V3Gradients(None,(torch.tensor([1.,1.]),),1,zero,zero,gain_sum_grads=(torch.tensor([1.,-2.]),)))
    else:c.accumulate(result([2.,-1.],[1.,1.],1))
    engine,master,adam,events=fake_engine(monkeypatch,c,p,skip=skip)
    baseline=torch.nn.Parameter(master.detach().clone());base_adam=torch.optim.AdamW([baseline],lr=.03,weight_decay=.01)
    before=master.detach().clone()
    if not skip:baseline.grad=torch.tensor([.5,-.25]);base_adam.step()
    trainer=SimpleNamespace(v3_gain_controller=c,_validate_deepspeed_gradient_norm=lambda engine:None)
    # SimpleNamespace is not weak-referenceable; use an ordinary holder.
    class Holder:pass
    holder=Holder();holder.v3_gain_controller=c;holder._validate_deepspeed_gradient_norm=trainer._validate_deepspeed_gradient_norm
    def finish(engine,successful):
        c.finish(engine,successful=successful);events.append('completed')
    holder._finish_optimizer_step=finish
    original_prepare=c.prepare
    def prepare(*a,**kw):events.append('gain_prepare');return original_prepare(*a,**kw)
    monkeypatch.setattr(c,'prepare',prepare)
    _DeepSpeedStepAdapter(holder,engine,engine.step)()
    if skip:
        assert torch.equal(master,before) and adam.state=={}
        assert events==['gain_prepare','adamw','completed']
    else:
        torch.testing.assert_close(master,baseline+.001*torch.tensor([1.5,-1.5]))
        assert torch.equal(adam.state[master]['exp_avg'],base_adam.state[baseline]['exp_avg'])
        assert torch.equal(adam.state[master]['exp_avg_sq'],base_adam.state[baseline]['exp_avg_sq'])
        assert torch.equal(p,master.to(torch.bfloat16))
        assert events==['gain_prepare','adamw','gain_master_apply','bf16_sync','completed']
    c.assert_checkpointable()
    with pytest.raises(RuntimeError):c.finish(engine,successful=True)


@pytest.mark.parametrize('fault',['nan','remote','base','apply'])
def test_failure_prevents_base_mutation_or_checkpoint(monkeypatch,fault):
    module=api();c,p=controller();r=[float('nan'),1.] if fault=='nan' else [2.,-1.]
    c.accumulate(result(r,[1.,1.],1));engine,master,adam,events=fake_engine(monkeypatch,c,p,fail=fault=='base')
    if fault=='remote':
        monkeypatch.setattr(module.dist,'is_initialized',lambda:True)
        monkeypatch.setattr(module.dist,'all_reduce',lambda tensor,**kw:tensor.fill_(1))
    if fault=='apply':
        monkeypatch.setattr(module,'safe_set_full_fp32_param',lambda *a:(_ for _ in ()).throw(RuntimeError('apply failed')))
    class Holder:pass
    trainer=Holder();trainer.v3_gain_controller=c;trainer._validate_deepspeed_gradient_norm=lambda engine:None
    trainer._finish_optimizer_step=lambda engine,successful:c.finish(engine,successful=successful)
    with pytest.raises((RuntimeError,ValueError),match='nonfinite|invalid|failed'):
        _DeepSpeedStepAdapter(trainer,engine,engine.step)()
    if fault in ('nan','remote'):assert events==[] and adam.state=={}
    with pytest.raises(RuntimeError,match='checkpoint'):c.assert_checkpointable()
    with pytest.raises(RuntimeError):c.accumulate(result([0.,0.],[0.,0.],0))


def test_global_zero_tiny_h_and_duplicate_prepare(monkeypatch):
    for r,rv,count in (([0.,0.],[0.,0.],0),([1.,1.],[1e-16,1e-16],1)):
        c,p=controller();c.accumulate(result(r,rv,count));c.prepare()
        assert torch.count_nonzero(c.direction[0])==0
        with pytest.raises(RuntimeError):c.prepare()


@pytest.mark.parametrize('field,value',[('partition_gradients',True),('cpu_offload',True),('update_lp_params',None)])
def test_runtime_mapping_rejected_before_base(monkeypatch,field,value):
    c,p=controller();engine,master,adam,events=fake_engine(monkeypatch,c,p)
    setattr(engine.optimizer,field,value)
    with pytest.raises(RuntimeError,match='ZeRO|master|offload'):c.validate_engine(engine)
    assert events==[]


@pytest.mark.parametrize('checkpointed',[False,True])
@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('image',[False,True])
@pytest.mark.parametrize('weighting',['none','token_entropy'])
def test_real_parent_training_step_helper_once_and_configured_accumulation(tmp_path,monkeypatch,checkpointed,empty,image,weighting):
    from tests.stage1.v3_helpers import product_v3_trainer
    from stage1.v3_lifecycle import current_v3_lifecycle
    from peft_common import resolve_qwen2_5_vl_model
    from transformers import Trainer
    from trl import SFTTrainer
    trainer,batch=product_v3_trainer(tmp_path,checkpointed=checkpointed,empty=empty,image=image,
                                  config=Stage1V3Config(gain_weighting=weighting))
    assert 'training_step' not in SFTTrainer.__dict__
    calls=[];losses={};seen=[]
    qwen=resolve_qwen2_5_vl_model(trainer.model)
    original_loss=trainer.compute_loss
    def compute(*a,**kw):
        value=original_loss(*a,**kw);losses['raw']=value.detach().clone();return value
    monkeypatch.setattr(trainer,'compute_loss',compute)
    backward=trainer.accelerator.backward
    def observed(loss,*a,**kw):
        life=current_v3_lifecycle();seen.append(life)
        assert life.frame[3]=='ordinary'
        torch.testing.assert_close(loss.detach(),losses['raw']/4,rtol=0,atol=0)
        return backward(loss,*a,**kw)
    monkeypatch.setattr(trainer.accelerator,'backward',observed)
    hook=qwen.model.register_forward_pre_hook(lambda *a:calls.append(current_v3_lifecycle().frame[0]))
    try:loss=trainer.training_step(trainer.model,batch)
    finally:hook.remove()
    assert calls==(['F'] if empty else ['F','gain_clean','packed_replay'])
    assert trainer.target_builder.teacher_language_calls==1 and trainer.target_builder.helper_visual_calls==1
    assert seen[0].state=='CLOSED' and current_v3_lifecycle() is None
    assert all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)
    assert trainer.v3_gain_controller.state=='ACCUMULATING'
    with pytest.raises(RuntimeError,match='checkpoint'):trainer.v3_gain_controller.assert_checkpointable()


def test_real_parent_short_tail_retains_global_sixteen_denominator(tmp_path,monkeypatch):
    from tests.stage1.v3_helpers import product_v3_trainer
    # Four independent CPU rank witnesses; no distributed training is launched.
    rank_gradients=[];rank_unscaled=[]
    for rank in range(4):
        trainer,batch=product_v3_trainer(tmp_path/str(rank),empty=rank==0)
        params=tuple(p for p in trainer.model.parameters() if p.requires_grad)
        unscaled=[torch.zeros_like(p) for p in params]
        backward=trainer.accelerator.backward
        def observed(loss,*args,**kwargs):
            gradients=torch.autograd.grad(loss*4,params,retain_graph=True,allow_unused=True)
            for total,g in zip(unscaled,gradients):
                if g is not None:total.add_(g)
            return backward(loss,*args,**kwargs)
        monkeypatch.setattr(trainer.accelerator,'backward',observed)
        for _ in range(2):trainer.training_step(trainer.model,batch)
        rank_gradients.append(tuple(p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params))
        rank_unscaled.append(tuple(unscaled))
    for values,raw in zip(zip(*rank_gradients),zip(*rank_unscaled)):
        torch.testing.assert_close(sum(values)/4,sum(raw)/16,atol=1e-6,rtol=1e-4)
