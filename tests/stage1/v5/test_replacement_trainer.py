"""Real native Trainer vs literal full-sequence replacement forwards/gradients."""
from dataclasses import replace
import pytest
import torch
from peft_common import resolve_qwen2_5_vl_model
from tests.stage1.v5.test_trainer import integrated_fixture
from tests.stage1.v5.test_replay import digest_state, equal
from stage1.query import Stage1QueryConfig
from transformers.models.qwen2_5_vl.stage1_v5_replay import V5ReplayOutput
from transformers.models.qwen2_5_vl.stage1_v5_trace import bind_v5_execution, _PACKED

MODES=('cross_helper_reverse_topk_independent','cross_helper_reverse_topk_cumulative')


def prepared(tmp_path,mode,query_mode='legacy',empty_first=False,disabled=False,empty_all=False):
    trainer,batch,calls,forwards,backs,_=integrated_fixture(tmp_path,
        history_config=Stage1QueryConfig(mode=query_mode),empty=empty_all)
    sample=batch['stage1_batch_metadata'].samples[0]
    sample=replace(sample,helper_image_ids=('helper0','helper1'),original_image_ids=('original',))
    batch['stage1_batch_metadata']=replace(batch['stage1_batch_metadata'],samples=(sample,))
    if empty_first:
        meta=batch['stage1_v5_metadata']
        batch['stage1_v5_metadata']=replace(meta,blocks=(replace(meta.blocks[0],
            target_positions=(),target_ids=()),meta.blocks[1]))
    trainer.v5_config=replace(trainer.v5_config,voff_mode=mode,selection='clean',
        preference_entropy_top_percent=20,lambda_topk_entropy=0. if disabled else .1,
        lambda_topk_mass=0. if disabled else .05)
    return trainer,batch,calls,forwards,backs


def full_sequence_replay(decoder,head,trace,blocks,*,execution,need_voff,need_textoff,
                         chunk_positions,voff_mode):
    x=trace.inputs_embeds;length=x.shape[1]
    keys=tuple((b.batch_index,t-1) for b in blocks for t in b.target_positions)
    v=[];text=[]
    samples=torch.arange(x.shape[0])[:,None].expand(x.shape[:2])
    positions=torch.arange(length)[None,:].expand(x.shape[:2])
    valid=trace.padding.bool()
    def forward(embeds,mask):
        token=_PACKED.set((execution.dropout_mask,samples,positions,valid,{}))
        try:
            with bind_v5_execution(decoder,execution,role='packed'):
                return decoder(inputs_embeds=embeds,position_ids=trace.position_ids,
                    attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
        finally:_PACKED.reset(token)
    mask=torch.ones(length,length,dtype=torch.bool).tril()[None,None]
    mask=mask & valid[:,None,None,:]
    causal=x.new_zeros(mask.shape).masked_fill(~mask,-torch.inf)
    for i,b in enumerate(blocks):
        if not b.target_positions:continue
        query=torch.tensor(b.target_positions)-1
        if need_voff:
            changed=x.clone()
            donors=blocks[:i+1] if voff_mode==MODES[1] else (b,)
            for d in donors:
                if d.batch_index==b.batch_index:changed[d.batch_index,d.start+1:d.end]=d.replacement_values.detach()
            v.append(head(forward(changed,causal)[b.batch_index,query]))
        if need_textoff:
            bias=causal.clone()
            for t,blocked in zip(b.target_positions,b.blocked_text_keys):bias[b.batch_index,0,t-1,list(blocked)]=-torch.inf
            text.append(head(forward(x,bias)[b.batch_index,query]))
    return V5ReplayOutput(torch.cat(v) if v else None,torch.cat(text) if text else None,keys)


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('query_mode',['legacy','adaptive'])
@pytest.mark.parametrize('empty_first',[False,True])
def test_replacement_uses_one_native_backward_and_saved_loss_profile(tmp_path,monkeypatch,mode,query_mode,empty_first):
    import transformers.models.qwen2_5_vl.stage1_v5_replay as replay
    native,batch,calls,forwards,backs=prepared(tmp_path/'native',mode,query_mode,empty_first)
    oracle,obatch,_,_,_=prepared(tmp_path/'oracle',mode,query_mode,empty_first)
    assert digest_state(native.model)==digest_state(oracle.model)
    torch.testing.assert_close(batch['pixel_values_latent'],obatch['pixel_values_latent'],rtol=0,atol=0)
    reference_before=digest_state(native.ema_teacher.model)
    clean=[];native.model.register_forward_hook(lambda m,a,o:clean.append(o))
    windows,_=native.get_batch_samples(iter([batch]),4)
    rng=torch.random.get_rng_state().clone()
    loss=native.training_step(native.model,windows[0])
    targets=native.target_builder.targets
    assert len(targets.rejected_blocks)==2
    assert len(backs)==1 and backs[0].state=='CLOSED'
    assert calls.count('packed')==(2 if mode==MODES[1] else 1)
    assert calls.count('F')==1 and calls.count('reference')==1
    assert calls.count('helper')==(3 if query_mode=='adaptive' else 1)
    assert digest_state(native.ema_teacher.model)==reference_before
    assert all(p.grad is None for p in native.ema_teacher.model.parameters())
    diagnostics=native._latest_stage1_diagnostics
    assert int(diagnostics['rejected_helper_blocks'])==2
    assert int(diagnostics['rejected_original_blocks'])==0
    assert diagnostics['selected_tokens']==1
    expected=(.1*diagnostics['ce_loss']+.1*diagnostics['alignment_loss']+
        diagnostics['dil_scale']*diagnostics['dil_raw_sum']+
        diagnostics['auxiliary_scale']*(.1*diagnostics['topk_entropy_raw_sum']+
        .05*diagnostics['topk_mass_raw_sum'])+.01*diagnostics['text_kl_raw'])/4
    torch.testing.assert_close(loss,expected)
    monkeypatch.setattr(replay,'run_v5_packed_replay',full_sequence_replay)
    owindows,_=oracle.get_batch_samples(iter([obatch]),4)
    torch.random.set_rng_state(rng)  # identical existing alignment-noise draw
    expected_loss=oracle.training_step(oracle.model,owindows[0])
    torch.testing.assert_close(loss,expected_loss,atol=1e-5,rtol=1e-4)
    equal({n:p.grad for n,p in native.model.named_parameters() if p.requires_grad},
          {n:p.grad for n,p in oracle.model.named_parameters() if p.requires_grad})
    torch.testing.assert_close(targets.latent_values,oracle.target_builder.targets.latent_values,rtol=0,atol=0)
    assert forwards[-1]['latent_hidden_states'].data_ptr()==targets.latent_values.data_ptr()


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('disabled,empty_all',[(True,False),(False,True)])
def test_unneeded_replacement_skips_donor_work(tmp_path,mode,disabled,empty_all):
    trainer,batch,calls,_,backs=prepared(tmp_path,mode,disabled=disabled,empty_all=empty_all)
    windows,_=trainer.get_batch_samples(iter([batch]),4)
    trainer.training_step(trainer.model,windows[0])
    assert trainer.target_builder.targets.rejected_blocks==()
    assert calls.count('packed')==(0 if empty_all else 1)
    assert len(backs)==1


def test_wrong_rejected_ownership_fails_before_replay(tmp_path,monkeypatch):
    trainer,batch,calls,_,_=prepared(tmp_path,MODES[0])
    build=trainer.target_builder.build
    def corrupted(*args,**kwargs):
        targets=build(*args,**kwargs)
        assert targets.rejected_blocks, 'replacement targets were not requested'
        bad=replace(targets.rejected_blocks[0],block_index=1)
        return replace(targets,rejected_blocks=(bad,targets.rejected_blocks[1]))
    monkeypatch.setattr(trainer.target_builder,'build',corrupted)
    windows,_=trainer.get_batch_samples(iter([batch]),4)
    with pytest.raises(ValueError,match='replacement.*ownership'):
        trainer.training_step(trainer.model,windows[0])
    assert calls.count('packed')==0
