"""Actual causal Qwen/DoRA replay vs independently constructed full sequences.

Breaks caught: stale or detached KV, cumulative reset, masked donors, context
collisions, changed dropout coordinates, target reordering and Text-off leakage.
"""
from contextlib import contextmanager
from dataclasses import replace
import sys
import pytest
import torch
from torch.utils.checkpoint import set_checkpoint_early_stop
from peft_common import resolve_qwen2_5_vl_model
from stage1.equivalence import SampleStableStage1Dropout
from stage1.v5_config import Stage1V5Config
from stage1.v5_metadata import V5BatchMetadata, V5Block
from stage1.v5_objective import compute_v5_loss_sums
from tests.stage1.v5.test_replay import (fixture, realize, NativeRunner, runtime_api,
    BLOCKS, IDS, POSITIONS, BODY, gradients, equal, ATOL, RTOL)

MODES=('cross_helper_reverse_topk_independent','cross_helper_reverse_topk_cumulative')


def replay_blocks(empty_first=False, first_change=0., future_change=0., one_block=False):
    _,_,_,_,Block,_=runtime_api()
    result=[]
    for i,(s,e,t,masked) in enumerate(BLOCKS[:1] if one_block else BLOCKS):
        if i==0 and empty_first:t=();masked=()
        values=(torch.arange((e-s-1)*8).reshape(e-s-1,8).float()*.035-.4+i*.6)
        values=values+(first_change if i==0 else future_change)
        result.append(Block(0,s,e,t,tuple(IDS[0,list(t)].tolist()),masked,
                            replacement_values=values.requires_grad_()))
    return tuple(result)


def metadata_for(blocks):
    return V5BatchMetadata(tuple(V5Block(b.batch_index,i,b.start,b.end,b.target_positions,
        b.target_ids,tuple(range(3,b.start))) for i,b in enumerate(blocks)))


@contextmanager
def execute(q,x,blocks,mode,checkpointed,branches=(True,True),padding=None):
    Execution,Spec,capture,bind,_,replay=runtime_api()
    sites=tuple(sorted(m.site_name for m in q.modules() if isinstance(m,SampleStableStage1Dropout)))
    owner=Execution(Spec(42,0,((7,26),),sites))
    q.model.gradient_checkpointing=checkpointed
    keys=tuple((0,t-1) for b in blocks for t in b.target_positions)
    candidates=tuple(tuple(range(3,b.start)) for b in blocks for _ in b.target_positions)
    calls=[]
    hook=q.model.register_forward_pre_hook(lambda *_:calls.append(1))
    try:
        with owner:
            with owner.graph('F'),bind(q.model,owner,role='F',visual=q.visual),capture(q,keys,candidates) as trace:
                clean=q.model(inputs_embeds=x,position_ids=POSITIONS,attention_mask=padding,
                              use_cache=False,return_dict=True)
            snapshot=x.detach().clone()
            with owner.graph('packed'):
                off=replay(q.model,q.lm_head,trace,blocks,execution=owner,need_voff=branches[0],
                    need_textoff=branches[1],chunk_positions=3,voff_mode=mode)
            assert off.query_keys==keys
            torch.testing.assert_close(trace.inputs_embeds,snapshot,rtol=0,atol=0)
            yield clean,off,owner,calls,trace
    finally:hook.remove()


def reference_full_sequences(q,rq,x,blocks,mode,runner,padding=None):
    # Full source sequence for every branch. No trace, prefix KV, selector or
    # production row-planning helper is consumed by this reference.
    visible=torch.ones(26,26,dtype=torch.bool).tril()
    if padding is not None:visible &= padding[0].bool()[None,:]
    base=torch.zeros(26,26).masked_fill(~visible,-torch.inf)[None,None]
    clean=runner(q.model,x,POSITIONS,mask=base,role='F')
    with torch.no_grad():
        reference=rq.lm_head(runner(rq.model,x.detach(),POSITIONS,mask=base,role='reference').last_hidden_state)
    v=[];text=[]
    for i,b in enumerate(blocks):
        if not b.target_positions:continue
        rejected=x.clone()
        for donor in (blocks[:i+1] if mode==MODES[1] else (b,)):
            rejected[0,donor.start+1:donor.end]=donor.replacement_values.detach()
        hidden=runner(q.model,rejected,POSITIONS,mask=base,role=f'v-{b.start}').last_hidden_state
        v.append(q.lm_head(hidden[0,torch.tensor(b.target_positions)-1]))
        mask=base.clone()
        for t,keys in zip(b.target_positions,b.blocked_text_keys):mask[0,0,t-1,list(keys)]=-torch.inf
        hidden=runner(q.model,x,POSITIONS,mask=mask,role=f'text-{b.start}').last_hidden_state
        text.append(q.lm_head(hidden[0,torch.tensor(b.target_positions)-1]))
    return clean,reference,torch.cat(v),torch.cat(text)


def objective(q,clean,reference,v,text,metadata):
    keys=metadata.query_keys
    full=q.lm_head(clean.last_hidden_state)
    c=torch.stack([full[r,p] for r,p in keys])
    ref=torch.stack([reference[r,p] for r,p in keys])
    cfg=Stage1V5Config(lambda_topk_entropy=.1,lambda_topk_mass=.05,preference_topk_k=50,
        selection='clean',preference_entropy_top_percent=20)
    sums=compute_v5_loss_sums(c,ref,v,text,metadata,cfg,collect_diagnostics=False)
    n=metadata.selected_token_count(cfg)
    total=(.1*full.square().mean()+sums.dil_sum/n+
           (.1*sums.entropy_sum+.05*sums.mass_sum)/metadata.original_step_count+.01*sums.text_kl)
    return total,sums


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('dropout',[0.,.05])
@pytest.mark.parametrize('checkpoint_mode',['off','early','full'])
@pytest.mark.parametrize('empty_first',[False,True])
def test_replacement_logits_and_gradients_match_full_sequence(mode,dropout,checkpoint_mode,empty_first):
    student,reference,os,oref,live,helper,_,_=fixture(dropout)
    q,x=realize(student,live,helper)
    oracle_live=live.detach().clone().requires_grad_()
    oq,ox=realize(os,oracle_live,helper)
    rq=resolve_qwen2_5_vl_model(oref)
    blocks=replay_blocks(empty_first)
    meta=metadata_for(blocks)
    runner=NativeRunner(oq,rq,checkpoint_mode!='off')
    try:
        with set_checkpoint_early_stop(checkpoint_mode!='full'):
            with execute(q,x,blocks,mode,checkpoint_mode!='off') as (clean,off,owner,calls,trace):
                want,ref,v,text=reference_full_sequences(oq,rq,ox,blocks,mode,runner)
                torch.testing.assert_close(off.voff_logits,v,atol=ATOL,rtol=RTOL)
                torch.testing.assert_close(off.textoff_logits,text,atol=ATOL,rtol=RTOL)
                torch.testing.assert_close(clean.last_hidden_state,want.last_hidden_state,atol=ATOL,rtol=RTOL)
                total,terms=objective(q,clean,ref,off.voff_logits,off.textoff_logits,meta)
                oracle_total,oracle_terms=objective(oq,want,ref,v,text,meta)
                for key in ('dil_sum','entropy_sum','mass_sum','text_kl'):
                    torch.testing.assert_close(getattr(terms,key),getattr(oracle_terms,key),atol=ATOL,rtol=RTOL)
                torch.testing.assert_close(total,oracle_total,atol=ATOL,rtol=RTOL)
                assert len(calls)==(3 if mode==MODES[1] else 2)
                with owner.backward():
                    # Compare the OFF gradient separately: a clean CE path cannot
                    # hide a missing gradient through reused prefix K/V.
                    equal(gradients(student,live,off.voff_logits.square().mean()),
                          gradients(os,oracle_live,v.square().mean()))
                    total.backward()
                oracle_total.backward()
                equal(gradients(student,live),gradients(os,oracle_live))
                assert all(b.replacement_values.grad is None for b in blocks)
                assert helper.grad is None
                assert any(k.requires_grad for k,_ in trace.key_values)
    finally:runner.close()


@pytest.mark.parametrize('mode',MODES)
def test_earlier_donors_distinguish_cumulative_and_independent_and_future_is_causal(mode):
    def run(first=0.,future=0.,one=False,padding=None):
        student,_,_,_,live,helper,_,_=fixture(0.)
        q,x=realize(student,live,helper)
        blocks=replay_blocks(first_change=first,future_change=future,one_block=one)
        with execute(q,x,blocks,mode,False,padding=padding) as (_,off,owner,_,_):
            result=(off.voff_logits.detach().clone(),off.textoff_logits.detach().clone())
            with owner.backward():off.voff_logits.sum().backward()
        return result
    base,text=run()
    first,first_text=run(first=2.)
    future,future_text=run(future=3.)
    assert not torch.allclose(base[:3],first[:3],atol=ATOL,rtol=RTOL)
    if mode==MODES[0]:torch.testing.assert_close(base[3:],first[3:],atol=0,rtol=0)
    else:assert not torch.allclose(base[3:],first[3:],atol=ATOL,rtol=RTOL)
    torch.testing.assert_close(base[:3],future[:3],atol=0,rtol=0)
    torch.testing.assert_close(text,first_text,atol=0,rtol=0)
    torch.testing.assert_close(text,future_text,atol=0,rtol=0)
    one,_=run(one=True)
    torch.testing.assert_close(base[:3],one,atol=ATOL,rtol=RTOL)


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('branches',[(True,False),(False,True),(False,False)])
def test_optional_replacement_branches_prune_original_decoder_calls(mode,branches):
    student,_,_,_,live,helper,_,_=fixture(.05)
    q,x=realize(student,live,helper)
    blocks=replay_blocks()
    if not branches[0]:blocks=tuple(replace(b,replacement_values=None) for b in blocks)
    with execute(q,x,blocks,mode,True,branches) as (clean,off,owner,calls,_):
        assert (off.voff_logits is not None)==branches[0]
        assert (off.textoff_logits is not None)==branches[1]
        assert len(calls)==1+int(any(branches))+int(all(branches) and mode==MODES[1])
        with owner.backward():clean.last_hidden_state.square().sum().backward()


@pytest.mark.parametrize('mode',MODES)
def test_no_target_queries_skip_all_off_calls(mode):
    student,_,_,_,live,helper,_,_=fixture(0.)
    q,x=realize(student,live,helper)
    blocks=tuple(replace(b,target_positions=(),target_ids=(),blocked_text_keys=()) for b in replay_blocks())
    with execute(q,x,blocks,mode,False) as (clean,off,owner,calls,_):
        assert off.voff_logits is None and off.textoff_logits is None and len(calls)==1
        with owner.backward():clean.last_hidden_state.square().sum().backward()


@pytest.mark.parametrize('mode',MODES)
def test_original_padding_and_one_block_match_full_sequence(mode):
    student,_,os,oref,live,helper,_,_=fixture(.05)
    q,x=realize(student,live,helper);oq,ox=realize(os,live.detach(),helper)
    rq=resolve_qwen2_5_vl_model(oref);runner=NativeRunner(oq,rq,False)
    blocks=replay_blocks(one_block=True)
    padding=torch.ones(1,26,dtype=torch.long);padding[0,:2]=0;padding[0,-1]=0
    try:
        with execute(q,x,blocks,mode,False,padding=padding) as (_,off,owner,_,_):
            _,_,v,text=reference_full_sequences(oq,rq,ox,blocks,mode,runner,padding)
            torch.testing.assert_close(off.voff_logits,v,atol=ATOL,rtol=RTOL)
            torch.testing.assert_close(off.textoff_logits,text,atol=ATOL,rtol=RTOL)
            with owner.backward():off.voff_logits.sum().backward()
    finally:runner.close()


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('bad',['missing','short','nonfinite'])
def test_bad_replacement_body_is_rejected_before_off_decoder(mode,bad):
    student,_,_,_,live,helper,_,_=fixture(0.)
    q,x=realize(student,live,helper)
    blocks=replay_blocks()
    value=None if bad=='missing' else blocks[0].replacement_values.detach().clone()
    if bad=='short':value=value[:1]
    elif bad=='nonfinite':value[0,0]=float('nan')
    blocks=(replace(blocks[0],replacement_values=value),blocks[1])
    # Deliberate expected failure must close the execution as a failed graph,
    # so exception restoration itself is part of this witness.
    with pytest.raises(ValueError,match='replacement'):
        with execute(q,x,blocks,mode,False):pass
