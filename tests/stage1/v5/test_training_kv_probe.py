"""CPU contracts for the standalone historical trained-state diagnostic.

Each mutation-sensitive assertion targets a numerical/path or lifecycle error;
fixtures use native Qwen/DoRA/row, collation and reference restore code.
"""
from dataclasses import asdict, replace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import argparse
import importlib
import importlib.util
import json

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'scripts/probe_stage1_training_kv.py'


def api():
    assert SCRIPT.is_file(), 'standalone trained KV diagnostic is missing'
    return importlib.import_module('scripts.probe_stage1_training_kv')


def test_first_eligible_row_uses_original_bounds():
    probe = api()
    from tests.stage1.v5.test_metadata import source_example, source_collator
    examples = [source_example(['pre', None, 'z'], source_index=9),
                source_example(['pre', None, 'x' * 700], source_index=0),
                source_example(['pre', None, 'target'], source_index=3)]
    collator = source_collator(enabled=True)
    selected, batch, evidence = probe.select_example(examples, collator)
    expected = collator([examples[2]])
    assert selected is examples[2] and selected.source_index == 3
    assert evidence['original_blocks'] == 1 and evidence['scored_targets'] == 6
    assert evidence['length'] <= 512
    for name, value in expected.items():
        if isinstance(value, torch.Tensor): assert torch.equal(batch[name], value), name
    assert selected.original_image_paths == examples[2].original_image_paths
    assert selected.helper_image_paths == examples[2].helper_image_paths


def test_no_eligible_row_does_not_fallback():
    probe = api()
    from tests.stage1.v5.test_metadata import source_example, source_collator
    with pytest.raises(ValueError, match='eligible'):
        probe.select_example([source_example(['none']), source_example(['pre',None,'x'*700],source_index=18)], source_collator(enabled=True))


@pytest.mark.parametrize('blocks', [1,2])
def test_window_preserves_historical_block_normalization(blocks):
    probe = api()
    from stage1.v5_reduction import V5Window, v5_auxiliary_scale
    assert probe.diagnostic_window(blocks) == V5Window(16*blocks,4,4,4)
    assert v5_auxiliary_scale(probe.diagnostic_window(blocks)) == 1/blocks
    for invalid in (0,-1,True,1.5):
        with pytest.raises(ValueError): probe.diagnostic_window(invalid)


def test_runtime_rejects_foreign_native_import(monkeypatch):
    probe = api()
    import stage1.v5_objective as objective
    assert probe.authenticate_runtime()['modules']['stage1.v5_objective']['sha256']
    monkeypatch.setattr(objective, '__file__', '/tmp/foreign/stage1/v5_objective.py')
    with pytest.raises(ValueError, match='foreign'): probe.authenticate_runtime()


def saved_bundle(tmp_path):
    from tests.stage1.test_checkpointing import tiny_original_qwen, sample_metadata, TinyStage1Processor, stage1_tuning
    from stage1.composition import _configure_latent_model
    from stage1.peft import build_stage1_tuned_model
    from stage1.checkpointing import save_stage1_peft_artifact
    from stage1.ema import Stage1EmaTeacher
    from stage1.v5_reference import capture_v5_reference
    from stage1.v5_resume import build_v5_contract
    from stage1.v5_config import Stage1V5Config
    base = tmp_path/'original'; artifact=tmp_path/'artifact'
    original = tiny_original_qwen().to(torch.bfloat16)
    original.save_pretrained(base)
    from transformers import Qwen2_5_VLForConditionalGeneration
    original=Qwen2_5_VLForConditionalGeneration.from_pretrained(base,torch_dtype=torch.bfloat16,attn_implementation="eager")
    original.config._name_or_path=str(base)
    original.name_or_path=str(base)
    tuning=replace(stage1_tuning(),dropout=.05)
    metadata=replace(sample_metadata(original,tuning), base_model_reference=str(base),ce_weight=.1,sim_weight=.1)
    original.resize_token_embeddings(128)
    _configure_latent_model(original.config,(125,126,127),latent_size=2)
    original.config.compress_strategy="average"; original.config.use_cache=False
    student=build_stage1_tuned_model(original,tuning,(125,126,127))
    processor=TinyStage1Processor()
    teacher=Stage1EmaTeacher(student,1.,'dora')
    identity=capture_v5_reference(teacher,student,processor,base_path=base,directory=artifact)
    initial=deepcopy(teacher.state_dict())
    with torch.no_grad():
        for p in student.parameters():
            if p.requires_grad: p.add_(.03125)
    student.to(torch.bfloat16)
    save_stage1_peft_artifact(student,processor,artifact,metadata)
    contract=build_v5_contract(Stage1V5Config(lambda_topk_entropy=.1,preference_topk_k=50),identity,
        ce_weight=.1,sim_weight=.1,schedule={},execution=dict(bf16=True,tuning_method='dora',ema_tau=1.,
        zero_stage=1,lora_dropout=.05,gradient_checkpointing=True,compress_strategy='average',attention_implementation='eager',
        vision_attention_implementation='eager',seed=42,backend=dict(cudnn_deterministic=True,
        cudnn_benchmark=False,cuda_matmul_allow_tf32=False,cudnn_allow_tf32=True,float32_matmul_precision='highest')))
    (artifact/'stage1_v5.json').write_text(json.dumps(contract))
    return base,artifact,processor,contract,student,initial


def test_saved_contract_and_original_added_tokens_are_preserved(tmp_path):
    probe=api()
    base,artifact,processor,contract,_,_=saved_bundle(tmp_path)
    assert probe.read_contract(artifact) == contract
    (artifact/'stage1_v5.json').write_text(json.dumps(dict(contract,formula_id='other')))
    with pytest.raises(ValueError,match='formula'): probe.read_contract(artifact)
    from transformers import PreTrainedTokenizerFast, AddedToken
    from tokenizers import Tokenizer, models
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({'[UNK]':0},unk_token='[UNK]')),unk_token='[UNK]')
    from stage1.composition import _add_fresh_latent_tokens
    _add_fresh_latent_tokens(tokenizer)
    tokenizer.save_pretrained(base/'processor')
    restored=probe.restore_added_tokens(PreTrainedTokenizerFast.from_pretrained(base/'processor'),base/'processor')
    for i in (1,2,3):
        assert isinstance(restored.added_tokens_decoder[i],AddedToken)
        assert restored.added_tokens_decoder[i].__getstate__() == tokenizer.added_tokens_decoder[i].__getstate__()


def test_trainable_artifact_values_and_reference_dtype_are_preserved(tmp_path,monkeypatch):
    probe=api()
    base,artifact,processor,contract,student,initial=saved_bundle(tmp_path)
    from transformers import Qwen2_5_VLForConditionalGeneration
    import stage1.composition as composition
    monkeypatch.setattr(composition,"_load_stage1_model",lambda source,config,args,dependencies:
        Qwen2_5_VLForConditionalGeneration.from_pretrained(source,config=config,torch_dtype=torch.bfloat16,attn_implementation="eager"))
    restored,teacher=probe.restore_models(artifact,base,processor,contract,device='cpu')
    expected=dict(student.named_parameters())
    for name,p in restored.named_parameters():
        if p.requires_grad:
            assert p.dtype == torch.bfloat16
            assert torch.equal(p,expected[name]),name
    actual=teacher.state_dict()['parameters']
    for name,value in initial['parameters'].items():
        assert actual[name].dtype == value.dtype
        assert torch.equal(actual[name],value),name
    assert all(not p.requires_grad and p.grad is None for p in teacher.model.parameters())


def test_prepare_only_never_loads_model_or_initializes_cuda(tmp_path,monkeypatch):
    probe=api()
    from tests.stage1.v5.test_metadata import source_example, source_collator
    def forbidden(*a,**kw): raise AssertionError('model/CUDA touched by prepare-only')
    monkeypatch.setattr(probe,'restore_models',forbidden)
    monkeypatch.setattr(torch.cuda,'init',forbidden)
    monkeypatch.setattr(probe,'prepare_inputs',lambda args: (None,{}, {'source_index':3}, {'formula_id':'native'}, {}))
    monkeypatch.setattr(probe,'ROOT',tmp_path)
    args=argparse.Namespace(run_id='prepare',prepare_only=True,device='cpu')
    report=probe.run_probe(args)
    assert report['status']=='prepared'
    assert json.loads((tmp_path/'results/prepare/prepared-input.json').read_text())['input']=={'source_index':3}


def required(name):
    probe=api()
    assert hasattr(probe,name), f'missing diagnostic interface: {name}'
    return getattr(probe,name)


def stochastic_for(trainer,batch):
    from stage1.equivalence import Stage1StochasticContext
    from stage1.equivalence import SampleStableStage1Dropout
    sites=tuple(sorted(m.site_name for m in trainer.model.modules() if isinstance(m,SampleStableStage1Dropout)))
    return Stage1StochasticContext(42,0,((7,26),),sites)


@pytest.mark.parametrize('checkpointed',[False,True])
@pytest.mark.parametrize('real_helper',[False,True])
def test_carrier_matches_actual_parent_loss_and_all_gradients(tmp_path,checkpointed,real_helper):
    run=required('run_local_backward')
    from tests.stage1.v5.test_trainer import integrated_fixture
    from peft_common import resolve_qwen2_5_vl_model
    from stage1.v5_reduction import V5Window
    parent,batch,_,_,_,parent_values=integrated_fixture(tmp_path/'parent',real_helper=real_helper,debug=False)
    carrier,other,_,_,_,carrier_values=integrated_fixture(tmp_path/'carrier',real_helper=real_helper,debug=False)
    carrier.model.train()  # Match actual parent training_step mode before comparison.
    carrier_values.copy_(parent_values)
    for key,value in batch.items():
        if isinstance(value,torch.Tensor): other[key]=value.clone()
    resolve_qwen2_5_vl_model(parent.model).model.gradient_checkpointing=checkpointed
    resolve_qwen2_5_vl_model(carrier.model).model.gradient_checkpointing=checkpointed
    batch=dict(batch,stage1_v5_window=V5Window(32,4,4,4))
    other=dict(other,stage1_v5_window=V5Window(32,4,4,4))
    torch.manual_seed(42)  # Native alignment adds ambient torch.randn_like noise.
    loss=parent.training_step(parent.model,batch)
    actual=run(carrier,other,stochastic=stochastic_for(carrier,other),oracle=False,observe=False)
    assert actual.scaled_loss == pytest.approx(float(loss),abs=1e-5,rel=1e-4)
    assert actual.raw_loss == pytest.approx(float(loss)*4,abs=1e-5,rel=1e-4)
    expected={n:p.grad for n,p in parent.model.named_parameters() if p.requires_grad}
    assert actual.gradients.keys()==expected.keys()
    assert any('magnitude' in n for n in expected) and any('output_delta' in n for n in expected)
    for name,value in expected.items():
        if value is None: assert actual.gradients[name] is None
        else: torch.testing.assert_close(actual.gradients[name],value,atol=1e-5,rtol=1e-4,msg=name)


class ForbiddenTrace:
    def __getattribute__(self,name): raise AssertionError('forbidden native trace read: '+name)


def independent_case(*,checkpointed=False,mutate=None):
    from tests.stage1.v5.test_replay import fixture,realize,POSITIONS,BLOCKS,IDS
    from stage1.v5_execution import V5Execution
    from stage1.equivalence import Stage1StochasticContext
    from stage1.equivalence import SampleStableStage1Dropout
    from transformers.models.qwen2_5_vl.stage1_v5_trace import bind_v5_execution
    from transformers.models.qwen2_5_vl.stage1_v5_replay import V5ReplayBlock
    student,reference,_,_,live,helper,noise,labels=fixture(.05)
    q,x=realize(student,live,helper)
    q.model.gradient_checkpointing=checkpointed
    blocks=tuple(V5ReplayBlock(0,s,e,t,tuple(IDS[0,list(t)].tolist()),k) for s,e,t,k in BLOCKS)
    sites=tuple(sorted(m.site_name for m in student.modules() if isinstance(m,SampleStableStage1Dropout)))
    execution=V5Execution(Stage1StochasticContext(42,0,((7,26),),sites))
    calls=[]
    handle=q.model.register_forward_pre_hook(lambda m,a,k:calls.append(dict(k)),with_kwargs=True)
    try:
        with execution:
            with execution.graph('F'),bind_v5_execution(q.model,execution,role='F',visual=q.visual),required('capture_original_inputs')(q.model) as original:
                clean=q.model(inputs_embeds=x,position_ids=POSITIONS,use_cache=False,return_dict=True)
            channel=original[0]
            if mutate=='detach': channel=replace(channel,inputs_embeds=channel.inputs_embeds.detach())
            with execution.graph('packed'):
                output=required('full_prefix_replay')(q.model,q.lm_head,channel,blocks,execution=execution,need_voff=True,need_textoff=True,chunk_positions=3)
            objective=output.voff_logits.square().sum()+output.textoff_logits.square().sum()+clean.last_hidden_state[:0].sum()
            with execution.backward(): objective.backward()
        return student,helper,live,output,calls,execution,x,blocks,reference
    finally: handle.remove()


def test_oracle_never_reads_native_trace_or_supplies_prefix_state(tmp_path):
    required('run_local_backward'); required('full_prefix_replay')
    from tests.stage1.v5.test_trainer import integrated_fixture
    trainer,batch,*_=integrated_fixture(tmp_path,debug=False)
    # Adapter receives a real raising sentinel by an interposed native caller.
    # The production adapter itself is exercised separately, never handed trace data.
    with required('capture_original_inputs')(__import__('peft_common').resolve_qwen2_5_vl_model(trainer.model).model) as seen:
        pass
    student,helper,live,output,calls,*_=independent_case()
    assert len(calls)==2 and calls[1]['inputs_embeds'].shape[:2]==(4,26)
    assert all(call.get('past_key_values') is None and call.get('stage2_prefix_key_values') is None for call in calls)
    adapter=required('oracle_adapter')
    # An adapter must only use the frozen restricted original channel.
    captured=[]
    sentinel=ForbiddenTrace()
    fake=lambda *a,**kw: captured.append((a,kw)) or 'output'
    import scripts.probe_stage1_training_kv as module
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module,'full_prefix_replay',fake)
        bound=adapter([SimpleNamespace(inputs_embeds=None)],None)
        assert bound(None,None,sentinel,(),execution=None,need_voff=True,need_textoff=False)=='output'
    assert captured[0][0][2] is not sentinel


def test_two_overlapping_prefixes_preserve_live_rows_and_detached_helpers():
    required('full_prefix_replay')
    from tests.stage1.v5.test_replay import NativeRunner,realize,oracle,POSITIONS,PREDICTORS,fixture
    from peft_common import resolve_qwen2_5_vl_model
    student,helper,live,got,calls,execution,x,blocks,reference=independent_case()
    # Independently reconstruct exactly the tiny source graph from the factory.
    os,oref,_,_,ol,oh,noise,labels=fixture(.05)
    oq,ox=realize(os,ol,oh)
    runner=NativeRunner(oq,resolve_qwen2_5_vl_model(oref),False)
    try:
        expected=oracle(oq,resolve_qwen2_5_vl_model(oref),ox,runner)
        torch.testing.assert_close(got.voff_logits,expected['v'],atol=1e-5,rtol=1e-4)
        torch.testing.assert_close(got.textoff_logits,expected['text'],atol=1e-5,rtol=1e-4)
        (expected['v'].square().sum()+expected['text'].square().sum()).backward()
        for (n,p),(on,op) in zip(student.named_parameters(),os.named_parameters()):
            if p.requires_grad:
                if p.grad is None: assert op.grad is None
                else: torch.testing.assert_close(p.grad,op.grad,atol=1e-5,rtol=1e-4,msg=n)
    finally: runner.close()
    rows=[p for n,p in student.named_parameters() if 'trainable_tokens_delta' in n]
    assert rows and rows[0].grad.abs().sum()>0 and helper.grad is None
    detached,*_=independent_case(mutate='detach')
    detached_row=next(p for n,p in detached.named_parameters() if 'trainable_tokens_delta' in n)
    assert detached_row.grad is None or not torch.equal(rows[0].grad,detached_row.grad)


def test_all_row_latent_mask_changes_oracle_result(monkeypatch):
    required('full_prefix_replay')
    good=independent_case()[3]
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    original=Qwen2_5_VLModel.forward
    def wrong(self,*a,**kw):
        if kw.get('attention_mask') is not None and kw['attention_mask'].ndim==4:
            mask=kw['attention_mask'].clone()
            mask[:2,0,:,7:9]=-torch.inf; mask[:2,0,:,16:19]=-torch.inf
            # Leave fully hidden queries with at least the self edge.
            kw['attention_mask']=mask
        return original(self,*a,**kw)
    monkeypatch.setattr(Qwen2_5_VLModel,'forward',wrong)
    wrong_output=independent_case()[3]
    assert not torch.allclose(good.voff_logits,wrong_output.voff_logits,atol=1e-5,rtol=1e-4)


def test_detached_native_prefix_changes_gradient():
    required('full_prefix_replay')
    from tests.stage1.v5.test_replay import fixture,realize,native_execute,POSITIONS
    from peft_common import resolve_qwen2_5_vl_model
    gradients=[]
    for mutation in (None,'detach_prefix'):
        student,reference,_,_,live,helper,*_=fixture(.05)
        q,x=realize(student,live,helper)
        with native_execute(q,resolve_qwen2_5_vl_model(reference),x,False,mutation) as (output,owner,*_):
            with owner.backward(): (output['v'].square().sum()+output['text'].square().sum()).backward()
        gradients.append(next(p.grad for n,p in student.named_parameters() if 'trainable_tokens_delta' in n))
    assert not torch.allclose(gradients[0],gradients[1],atol=1e-5,rtol=1e-4)


def test_textoff_edges_do_not_contaminate_voff():
    required('full_prefix_replay')
    *_,calls,execution,x,blocks,reference=independent_case()
    bias=calls[1]['attention_mask']
    assert torch.isfinite(bias[0,0,9,6]) and torch.isfinite(bias[0,0,9,9])
    assert torch.isneginf(bias[0,0,9,7:9]).all()
    assert torch.isfinite(bias[1,0,10,7:9]).all()  # second prefix remains clean
    assert torch.isfinite(bias[0,0,8,7:9]).all()   # first body's prefix rows are clean
    assert torch.isfinite(bias[1,0,18,16:19]).all() # second body's prefix rows are clean
    assert torch.isfinite(bias[0,0,9,3:6]).all()   # text keys untouched in V-off
    for i,b in enumerate(blocks):
        for target,keys in zip(b.target_positions,b.blocked_text_keys):
            assert torch.isneginf(bias[2+i,0,target-1,list(keys)]).all()
            assert torch.isfinite(bias[2+i,0,target-1,b.start+1:b.end]).all()


def test_dropout_and_recompute_use_original_source_coordinates():
    required('full_prefix_replay')
    uncheckpointed=independent_case(checkpointed=False)
    checkpointed=independent_case(checkpointed=True)
    for key in ('voff_logits','textoff_logits'):
        assert torch.equal(getattr(uncheckpointed[3],key),getattr(checkpointed[3],key))
    for (n,p),(on,op) in zip(uncheckpointed[0].named_parameters(),checkpointed[0].named_parameters()):
        if p.requires_grad:
            assert (p.grad is None)==(op.grad is None)
            if p.grad is not None: assert torch.equal(p.grad,op.grad),n
    assert any(e['phase']=='recompute' and e['role']=='packed' and e['site'] for e in checkpointed[5].events)
    assert all('forward' not in layer.__dict__ for layer in __import__('peft_common').resolve_qwen2_5_vl_model(checkpointed[0]).model.layers)


def test_failed_replay_restores_execution_and_binding_presence(tmp_path,monkeypatch):
    run=required('run_local_backward')
    from tests.stage1.v5.test_trainer import integrated_fixture
    from stage1.v5_execution import current_v5_execution
    import transformers.models.qwen2_5_vl.stage1_v5_replay as replay
    from peft_common import resolve_qwen2_5_vl_model
    trainer,batch,*_=integrated_fixture(tmp_path,debug=False)
    before=replay.run_v5_packed_replay
    presence={name: (name in trainer.__dict__,getattr(trainer,name,None)) for name in ('_v5_execution','_v5_step_stack')}
    def fail(*a,**kw): raise RuntimeError('oracle failure')
    monkeypatch.setattr(api(),'full_prefix_replay',fail)
    with pytest.raises(RuntimeError,match='oracle failure'):
        run(trainer,batch,stochastic=stochastic_for(trainer,batch),oracle=True,observe=True)
    assert replay.run_v5_packed_replay is before and current_v5_execution() is None
    for name,(had,value) in presence.items():
        assert (name in trainer.__dict__)==had
        assert getattr(trainer,name,None) is value
    assert all('forward' not in layer.__dict__ for layer in resolve_qwen2_5_vl_model(trainer.model).model.layers)


def test_entropy_observer_returns_original_support_object(monkeypatch):
    observe=required('observe_supports')
    from stage1 import v5_objective as objective
    original=objective.build_entropy_support
    returned=[]
    def audited(*a,**kw):
        support=original(*a,**kw); returned.append(support); return support
    monkeypatch.setattr(objective,'build_entropy_support',audited)
    clean=torch.arange(128).float()[None]
    records=[]
    with observe(records):
        actual=objective.build_entropy_support(clean,clean,torch.tensor([4]),k=50,mode='independent',include_gt=False,validate=False,collect_diagnostics=False)
    assert actual is returned[0]
    assert records[0]['indices']==actual.ids.tolist()
    assert objective.build_entropy_support is audited


def test_crossing_rank_keeps_each_arms_native_independent_support():
    observe=required('observe_supports')
    from stage1.v5_objective import compute_v5_loss_sums
    from stage1.v5_metadata import V5BatchMetadata,V5Block
    from stage1.v5_config import Stage1V5Config
    metadata=V5BatchMetadata((V5Block(0,1,2,5,(6,),(4,),(0,)),))
    config=Stage1V5Config(lambda_topk_entropy=.1,preference_topk_k=50,text_kl_weight=0)
    clean=torch.arange(128,0,-1).float()[None].requires_grad_()
    native_supports=[]
    for swap in (False,True):
        off=clean.detach().clone()
        if swap: off[:,49],off[:,50]=off[:,50].clone(),off[:,49].clone()
        off.requires_grad_()
        records=[]
        with observe(records):
            actual=compute_v5_loss_sums(clean,clean.detach(),off,None,metadata,config,validate=False,collect_diagnostics=False)
        expected=off.detach().topk(50,dim=-1).indices.tolist()
        assert records[0]['indices']==expected
        actual.entropy_sum.backward()
        assert off.grad.abs().sum()>0
        native_supports.append(records[0]['indices'])
    assert native_supports[0]!=native_supports[1]
    assert 49 in native_supports[0][0] and 50 in native_supports[1][0]


def test_observer_preserves_native_loss_and_all_gradients(tmp_path):
    required('observe_supports')
    from tests.stage1.v5.test_trainer import integrated_fixture
    trainer,batch,*_=integrated_fixture(tmp_path,real_helper=True,debug=False)
    trainer.model.train()
    before=api().parameter_identity(trainer.model)
    rng=torch.get_rng_state().clone()
    u=api().run_local_backward(trainer,batch,stochastic=stochastic_for(trainer,batch),oracle=False,observe=False)
    a=api().run_local_backward(trainer,batch,stochastic=stochastic_for(trainer,batch),oracle=False,observe=True)
    assert torch.equal(rng,torch.get_rng_state())
    assert u.raw_loss==a.raw_loss and u.scaled_loss==a.scaled_loss
    assert u.evidence['clean_logits_sha256']==a.evidence['clean_logits_sha256']
    assert a.evidence['helper']['values_sha256'] and a.evidence['reference_logits_sha256']
    assert a.evidence['supports'] and a.components['dil_sum']>0
    assert api().parameter_identity(trainer.model)==before
    for name,value in u.gradients.items():
        assert (value is None)==(a.gradients[name] is None)
        if value is not None: assert torch.equal(value,a.gradients[name]),name


def test_observer_exception_restores_original_binding(tmp_path,monkeypatch):
    observe=required('observe_supports')
    import stage1.v5_objective as objective
    before=objective.build_entropy_support
    class FailingRecords(list):
        def append(self,value): raise RuntimeError('observer failure')
    with pytest.raises(RuntimeError,match='observer failure'):
        with observe(FailingRecords()):
            x=torch.arange(128).float()[None]
            objective.build_entropy_support(x,x,torch.tensor([3]),k=50,mode='independent',include_gt=False)
    assert objective.build_entropy_support is before
    from tests.stage1.v5.test_trainer import integrated_fixture
    trainer,batch,*_=integrated_fixture(tmp_path,debug=False)
    trainer.model.train()
    rng=torch.get_rng_state().clone()
    build=objective.build_entropy_support
    def fail(*a,**kw): raise RuntimeError('support failed')
    monkeypatch.setattr(objective,'build_entropy_support',fail)
    with pytest.raises(RuntimeError,match='support failed'):
        api().run_local_backward(trainer,batch,stochastic=stochastic_for(trainer,batch),oracle=False,observe=True)
    assert objective.build_entropy_support is fail
    assert torch.equal(rng,torch.get_rng_state())
    from stage1.v5_execution import current_v5_execution
    assert current_v5_execution() is None
    assert '_prepare_stage1_loss_inputs' not in vars(trainer)


def test_four_passes_use_fresh_graphs_and_unchanged_state(tmp_path):
    four=required('four_passes')
    from tests.stage1.v5.test_trainer import integrated_fixture
    trainer,batch,*_=integrated_fixture(tmp_path,real_helper=True,debug=False)
    trainer.model.train()
    results=four(trainer,batch,stochastic=stochastic_for(trainer,batch))
    assert list(results)==['U','A','A-repeat','B']
    assert len({id(result.gradients) for result in results.values()})==4
    assert all(result.evidence['parameters_unchanged'] for result in results.values())
    assert results['U'].evidence['observed'] is False
    a,repeat,b=results['A'],results['A-repeat'],results['B']
    assert a.raw_loss==repeat.raw_loss
    for key in ('helper','reference_logits_sha256','clean_logits_sha256','supports','selected_text_keys'):
        assert a.evidence[key]==repeat.evidence[key],key
    for key in ('helper','reference_logits_sha256','clean_logits_sha256','selected_text_keys'):
        assert a.evidence[key]==b.evidence[key],key
    for name,value in a.gradients.items():
        if value is None: assert repeat.gradients[name] is None
        else: assert torch.equal(value,repeat.gradients[name]),name


def test_gradient_report_distinguishes_unused_missing_and_zero():
    compare=required('compare_gradients')
    left=dict(unused=None,one_sided=None,zero=torch.zeros(2),signal=torch.tensor([3.,4.]),zero_left=torch.zeros(2))
    right=dict(unused=None,one_sided=torch.ones(2),zero=torch.zeros(2),signal=torch.tensor([6.,8.]),zero_left=torch.ones(2))
    report=compare(left,right)
    assert report['matched_unused']==['unused']
    assert report['presence_mismatches']==['one_sided']
    assert report['parameters']['zero']['both_zero'] is True
    assert report['parameters']['zero']['relative_l2']==0.
    assert report['parameters']['zero']['cosine'] is None
    assert report['parameters']['zero_left']['relative_l2'] is None
    assert report['parameters']['zero_left']['zero_denominator'] is True
    assert report['parameters']['zero_left']['cosine'] is None
    assert report['parameters']['signal']['relative_l2']==1.
    assert report['parameters']['signal']['cosine']==1.
    assert report['global']['relative_l2']==pytest.approx((27/25)**.5)
    assert report['exact'] is False
    assert compare({'a':None},{})['presence_mismatches']==['a']
    assert compare({'a':None},{'a':None})['exact'] is True


def test_nonfinite_report_cannot_be_measured():
    compare=required('compare_gradients'); summarize=required('summarize_results')
    metrics=compare({'x':torch.tensor([float('nan')])},{'x':torch.tensor([float('inf')])})
    assert metrics['finite'] is False and metrics['parameters']['x']['max_abs'] is None
    result=api().ArmResult(float('nan'),float('inf'),{}, {'x':torch.tensor([float('nan')])}, {})
    report=summarize(dict(U=result,A=result,**{'A-repeat':result},B=result))
    assert report['status'] in ('incomplete','inconclusive') and report['finite'] is False
    json.dumps(report,allow_nan=False)


def test_inactive_branch_stays_unexercised(tmp_path):
    summarize=required('summarize_results')
    from tests.stage1.v5.test_trainer import integrated_fixture
    trainer,batch,*_=integrated_fixture(tmp_path,empty=True,debug=False)
    trainer.model.train()
    arms=api().four_passes(trainer,batch,stochastic=stochastic_for(trainer,batch))
    report=summarize(arms)
    assert report['arms']['A']['evidence']['branches']['voff']=='unexercised'
    assert report['arms']['B']['evidence']['branches']['textoff']=='unexercised'
    assert all(not r.evidence['supports'] for k,r in arms.items() if k!='U')


def test_cli_failure_report_is_json_and_never_overwrites_evidence(tmp_path,monkeypatch):
    main=required('main')
    monkeypatch.setattr(api(),'ROOT',tmp_path)
    args=['--checkpoint',str(tmp_path/'missing'),'--base',str(tmp_path/'base'),
          '--data-path',str(tmp_path/'train'),'--image-root',str(tmp_path/'images'),
          '--run-id','failure','--device','cpu']
    assert main(args)!=0
    path=tmp_path/'results/failure/report.json'
    content=path.read_bytes(); report=json.loads(content)
    assert report['status']=='incomplete' and report['reason']
    assert set(('schema_version','scope','runtime','input','objective','arms','controls','comparisons','limitations'))<=report.keys()
    assert main(args)!=0 and path.read_bytes()==content


def test_call_witness_distinguishes_native_suffix_and_oracle_full_prefix(tmp_path):
    required('summarize_results')
    from tests.stage1.v5.test_trainer import integrated_fixture
    trainer,batch,*_=integrated_fixture(tmp_path,debug=False)
    trainer.model.train()
    arms=api().four_passes(trainer,batch,stochastic=stochastic_for(trainer,batch))
    native=arms['A'].evidence['decoder_calls']; oracle=arms['B'].evidence['decoder_calls']
    assert [c['length'] for c in native]==[26,5]
    assert [c['prefix_key_values'] for c in native]==[False,True]
    assert [c['length'] for c in oracle]==[26,26]
    assert all(not c['past_key_values'] and not c['prefix_key_values'] for c in oracle)
    report=api().summarize_results(arms)
    assert report['controls']['observer']['valid'] and report['controls']['repeat']['valid']
    assert report['status']=='measured' and report['finite']
    json.dumps(report,allow_nan=False)


def test_backward_failure_restores_rng_hooks_and_attributes(tmp_path):
    required('summarize_results')
    from tests.stage1.v5.test_trainer import integrated_fixture
    from stage1.v5_execution import current_v5_execution
    import stage1.v5_trainer as native
    import stage1.v5_objective as objective
    trainer,batch,*_=integrated_fixture(tmp_path,debug=False)
    trainer.model.train()
    before=(native.compute_v5_loss_sums,native.select_v5_text_keys,objective.build_entropy_support)
    rng=torch.get_rng_state().clone()
    p=next(p for p in trainer.model.parameters() if p.requires_grad)
    handle=p.register_hook(lambda grad: (_ for _ in ()).throw(RuntimeError('backward exploded')))
    try:
        with pytest.raises(RuntimeError,match='backward exploded'):
            api().run_local_backward(trainer,batch,stochastic=stochastic_for(trainer,batch),oracle=True,observe=True)
    finally: handle.remove()
    assert current_v5_execution() is None and torch.equal(rng,torch.get_rng_state())
    assert before==(native.compute_v5_loss_sums,native.select_v5_text_keys,objective.build_entropy_support)
    assert '_prepare_stage1_loss_inputs' not in vars(trainer) and '_v5_execution' not in vars(trainer)


def test_partial_four_pass_failure_retains_completed_arms(tmp_path,monkeypatch):
    from tests.stage1.v5.test_trainer import integrated_fixture
    trainer,batch,*_=integrated_fixture(tmp_path/'fixture',debug=False)
    trainer.model.train()
    results={}
    original=api().run_local_backward
    def fail_b(*a,**kw):
        if kw['oracle']: raise RuntimeError('B did not finish')
        return original(*a,**kw)
    monkeypatch.setattr(api(),'run_local_backward',fail_b)
    with pytest.raises(RuntimeError,match='B did not finish'):
        api().four_passes(trainer,batch,stochastic=stochastic_for(trainer,batch),results=results)
    assert list(results)==['U','A','A-repeat']
    report=api().summarize_results(results)
    assert report['status']=='incomplete' and list(report['arms'])==list(results)


def test_profile_readback_rejects_spoofed_attention_class():
    validate=required('validate_model_profile')
    from tests.stage1.v5.test_replay import fixture
    from peft_common import resolve_qwen2_5_vl_model
    student,reference,*_=fixture(.05)
    qwen=resolve_qwen2_5_vl_model(student)
    # Config strings alone must not authenticate the real eager vision modules.
    qwen.model.config._attn_implementation='sdpa'
    qwen.visual.config._attn_implementation='flash_attention_2'
    saved=dict(execution=dict(attention_implementation='sdpa',vision_attention_implementation='flash_attention_2',
        gradient_checkpointing=False,lora_dropout=.05))
    with pytest.raises(ValueError,match='backend mismatch'):
        validate(student,saved)


def test_cli_composes_real_four_arms_from_frozen_preparation(tmp_path,monkeypatch):
    from tests.stage1.v5.test_trainer import integrated_fixture
    from stage1.v5_reference import V5ReferenceIdentity
    from stage1.v5_resume import build_v5_contract
    from stage1.equivalence import read_stage1_backend_settings
    from stage1.checkpointing import load_stage1_peft_metadata
    trainer,batch,*_=integrated_fixture(tmp_path/'fixture',real_helper=True,debug=False)
    trainer.model.train()
    # The CLI-only external artifact boundary is substituted; all four native
    # loss/helper/replay/backward paths below remain real tiny Qwen computations.
    saved=build_v5_contract(trainer.v5_config,V5ReferenceIdentity(1,'a'*64,'b'*64,'c'*64,('selected',)),
        ce_weight=.1,sim_weight=.1,schedule={},execution=dict(backend=read_stage1_backend_settings()))
    artifact=tmp_path/'artifact'; artifact.mkdir(); (artifact/'trainer_state.json').write_text('{"global_step":648}')
    monkeypatch.setattr(api(),'ROOT',tmp_path)
    monkeypatch.setattr(api(),'prepare_inputs',lambda args:(None,batch,{'source_index':7},deepcopy(saved),{}))
    monkeypatch.setattr(api(),'restore_models',lambda *a,**kw:(trainer.model,trainer.ema_teacher))
    monkeypatch.setattr(api(),'make_carrier',lambda *a:trainer)
    args=argparse.Namespace(run_id='composed',prepare_only=True,device='cpu',checkpoint=str(artifact),base='substituted')
    assert api().run_probe(args)['status']=='prepared'
    frozen=(tmp_path/'results/composed/prepared-input.json').read_bytes()
    args.prepare_only=False
    report=api().run_probe(args)
    assert report['status']=='measured'
    assert (tmp_path/'results/composed/prepared-input.json').read_bytes()==frozen
    payload=json.loads((tmp_path/'results/composed/report.json').read_text())
    assert list(payload['arms'])==['A','A-repeat','B','U']  # JSON is sorted, execution order is independently witnessed.
    assert payload['objective']['diagnostic_window']['global_original_steps']==32
    assert payload['runtime']['constructed']['decoder_attention_class'].endswith('Qwen2_5_VLSdpaAttention')


def test_empty_textoff_edges_are_reported_unexercised(tmp_path,monkeypatch):
    from tests.stage1.v5.test_trainer import integrated_fixture
    import stage1.v5_trainer as native
    trainer,batch,*_=integrated_fixture(tmp_path,debug=False)
    trainer.model.train()
    original=native.select_v5_text_keys
    def empty(*a,**kw): return {key:() for key in original(*a,**kw)}
    monkeypatch.setattr(native,'select_v5_text_keys',empty)
    arm=api().run_local_backward(trainer,batch,stochastic=stochastic_for(trainer,batch),oracle=False,observe=True)
    assert arm.evidence['branches']['textoff']=='unexercised'
    assert arm.evidence['textoff_edges']==0
    assert arm.evidence['voff_edges']>0


def test_runtime_rejects_recorded_trainer_hash_mismatch():
    with pytest.raises(ValueError,match='Trainer hash'):
        api().authenticate_runtime({'execution':{'source_sha256':{'stage1_trainer':'0'*64}}})
