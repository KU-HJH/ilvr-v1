import importlib
import math
from copy import deepcopy
import pytest
import torch
from torch.utils.checkpoint import set_checkpoint_early_stop
from peft_common import resolve_qwen2_5_vl_model
from stage1.peft import build_stage1_tuned_model,install_stage1_sample_stable_dropout
from stage1.equivalence import Stage1StochasticContext,activate_stage1_stochastic_context
from stage1.v3_lifecycle import V3Lifecycle
from stage1.v3_config import Stage1V3Config
from stage1.v3_metadata import V3Block,V3BatchMetadata
from transformers.models.qwen2_5_vl.stage1_v3_trace import bind_v3_execution
from tests.stage1.test_peft import tiny_qwen,tuning
from tests.stage1.test_v3_replay import full_mask


def api(): return importlib.import_module('stage1.v3_gradients')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('p',[0.,.05])
@pytest.mark.parametrize('mode',['off','early','full'])
@pytest.mark.parametrize('selector',['kl','entropy'])
def test_joint_vjp_preserves_optimizer_grads_and_call_budget(dtype,p,mode,monkeypatch,selector,tied=False,fault=False,gain_weighting='none',gain_entropy_mode='normalized'):
    module = api()
    model = build_stage1_tuned_model(tiny_qwen.__wrapped__(),tuning('dora',dropout=.05),(125,126,127)).to(dtype)
    qwen = resolve_qwen2_5_vl_model(model)
    qwen.model.gradient_checkpointing = mode != 'off'
    sites = install_stage1_sample_stable_dropout(model)
    # Keep the declared site inventory when testing deterministic p=0 algebra.
    from stage1.equivalence import SampleStableStage1Dropout
    for module_dropout in model.modules():
        if isinstance(module_dropout,SampleStableStage1Dropout): module_dropout.p=p
    if tied:
        tied_parameter=next(p for p in model.parameters() if p.requires_grad)
        model.register_parameter('v3_tied_alias',tied_parameter)
        model.register_parameter('v3_unused',torch.nn.Parameter(torch.ones(2,dtype=dtype)))
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    names = [n for n,p in model.named_parameters(remove_duplicate=False) if p.requires_grad]
    assert any('magnitude' in n for n in names) and any('trainable_tokens' in n for n in names)
    with torch.no_grad():
        for name,param in model.named_parameters():
            if 'lora_B' in name: param.fill_(.03)
    reference = deepcopy(model).eval().requires_grad_(False)
    original_buffers = {n:b.detach().clone() for n,b in model.named_buffers()}
    original_params = dict(model.named_parameters())
    for param in parameters: param.grad = torch.full_like(param,.123)
    old_grads = tuple(param.grad for param in parameters)
    old_values = tuple(g.clone() for g in old_grads)
    ids = torch.tensor([[1]*12+[125,126,126,127,2,3,4],[2]*12+[125,126,126,127,2,3,4]])
    helper = torch.randn(4,8,dtype=dtype,requires_grad=True)
    inputs = dict(input_ids=ids,attention_mask=torch.ones_like(ids),
                  position_ids=torch.arange(19).view(1,1,-1).expand(3,2,-1),
                  latent_hidden_states=helper.detach(),image_out_mask=ids.eq(126),use_cache=False)
    blocks = (V3Block(0,1,12,15,(16,18),(2,4),tuple(range(1,12))),
              V3Block(1,1,12,15,(16,),(2,),tuple(range(1,12))))
    spec = Stage1StochasticContext(42,3,((17,19),(18,19)),sites)
    inventory = {(g,('decoder',0)):sites for g in ('F','gain_clean','packed_replay')}
    life = V3Lifecycle(spec,inventory,has_auxiliary=True)
    calls,arities,bindings = [],[],[]
    qhook = qwen.model.register_forward_pre_hook(lambda *a:calls.append(life.frame[0]))
    rhook = resolve_qwen2_5_vl_model(reference).model.register_forward_pre_hook(lambda *a:calls.append('reference'))
    bhook = qwen.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda m,a:bindings.append((life.frame[0],next(p for p in m.parameters() if p.requires_grad))))
    from stage1.v3_lifecycle import current_v3_lifecycle
    def injected(module,args):
        active=current_v3_lifecycle()
        if fault and active is not None and active.frame[2:] == ('recompute','rv'):
            raise RuntimeError('injected PEFT RV failure')
    failure_hook=qwen.model.layers[0].self_attn.q_proj.register_forward_pre_hook(injected)
    original_grad = torch.autograd.grad
    def counted(outputs,*a,**kw):
        arities.append(len(outputs) if isinstance(outputs,tuple) else 1)
        return original_grad(outputs,*a,**kw)
    selected = {}
    from stage1.v3_selection import select_text_keys
    def record_selection(*a,**kw):
        selected.update(select_text_keys(*a,**kw)); return selected
    monkeypatch.setattr(module,'select_text_keys',record_selection)
    try:
        with set_checkpoint_early_stop(mode!='full'),life.activate():
            with life.graph('F'),bind_v3_execution(qwen.model,lambda i:life.checkpoint_contexts('F',('decoder',i))):
                f = model(**inputs).logits
            if gain_weighting=='token_entropy':
                # Keep real replay logits, but make the supplied F uncertainty
                # differ sharply from gain-clean/R/RV to authenticate the source.
                f=f.clone()
                with torch.no_grad():
                    f[0,15].fill_(-1000);f[0,15,0]=1000
                    f[0,17].zero_()
                    f[1,15].fill_(-1000);f[1,15,0]=math.log(3);f[1,15,1]=0
            f.retain_grad()
            with monkeypatch.context() as patch:
                patch.setattr(torch.autograd,'grad',counted)
                result = module.compute_v3_gradients(model,reference,inputs,V3BatchMetadata(blocks),f,
                                                    parameters,Stage1V3Config(text_selector=selector,gain_weighting=gain_weighting,
                                                        gain_entropy_mode=gain_entropy_mode),life)
            assert arities == [2,1]
            assert calls == (['F','gain_clean','reference','packed_replay'] if selector=='kl'
                             else ['F','gain_clean','packed_replay'])
            assert result.block_count == 2
            assert all(p.grad is old and torch.equal(old,value) for p,old,value in zip(parameters,old_grads,old_values))
            assert f.grad is None and helper.grad is None
            assert all(p.grad is None for p in reference.parameters())
            with life.backward('ordinary'):
                result.kl_proxy.backward()
    finally:
        for hook in (qhook,rhook,bhook,failure_hook): hook.remove()
        assert current_v3_lifecycle() is None
        assert all(dict(model.named_parameters())[n] is p for n,p in original_params.items())
        assert all(torch.equal(dict(model.named_buffers())[n],b) for n,b in original_buffers.items())
    assert any(g=='gain_clean' and all(value is not p for p in parameters) for g,value in bindings)
    proxy_grads = tuple(param.grad-old for param,old in zip(parameters,old_values))
    # Independent masked full sequence calls and literal objective formulas.
    qwen.model.gradient_checkpointing = False
    oracle_r,oracle_rv,per_sample_kl = [],[],[]
    for block in blocks:
        row = block.batch_index
        from types import SimpleNamespace
        oracle_block = SimpleNamespace(**{name:getattr(block,name) for name in ('start','end','target_positions')},
                    blocked_text_keys=tuple(selected[row,t-1] for t in block.target_positions))
        single = {key:(value[row:row+1] if key in ('input_ids','image_out_mask') else value) for key,value in inputs.items()}
        single['position_ids']=inputs['position_ids'][:,row:row+1]
        single['latent_hidden_states']=helper.detach()[row*2:row*2+2]
        for rv in (False,True):
            single['attention_mask']=full_mask(19,oracle_block,rv,dtype)
            with activate_stage1_stochastic_context(Stage1StochasticContext(42,3,(spec.rows[row],),sites)):
                logits=model(**single).logits[0,list(block.query_positions)].float()
            ids_tensor=torch.tensor(block.target_ids)
            token_scores=logits.log_softmax(-1).gather(1,ids_tensor[:,None]).flatten()
            (oracle_rv if rv else oracle_r).append(token_scores)
            if not rv:
                clean=(f[row,list(block.query_positions)].detach().float()/2).log_softmax(-1)
                restricted=(logits/2).log_softmax(-1)
                per_sample_kl.append((4*clean.exp()*(clean-restricted)).sum(-1).mean())
    expected_kl=torch.stack(per_sample_kl).mean()
    expected_kl_grads=original_grad(expected_kl,parameters,allow_unused=True,retain_graph=True)
    if gain_weighting=='token_entropy':
        weighted=[]
        for block,r,rv in zip(blocks,oracle_r,oracle_rv):
            # Oracle uses literal T1 full-vocabulary formula, independent of helper.
            logits=f[block.batch_index,list(block.query_positions)].detach().double()
            logp=logits.log_softmax(-1)
            w=-(logp.exp()*logp).sum(-1)
            if gain_entropy_mode=='normalized':w=w/math.log(logp.shape[-1])
            w=w.float()
            weighted.append((w*(r-rv)).mean())
        expected_r=original_grad(sum(weighted),parameters,allow_unused=True,retain_graph=True)
        assert result.r_sum_grads is None
        actual_gain=result.gain_sum_grads
    else:
        expected_r=original_grad(sum(r.mean() for r in oracle_r),parameters,allow_unused=True,retain_graph=True)
        assert result.gain_sum_grads is None
        actual_gain=result.r_sum_grads
    expected_rv=original_grad(sum(rv.mean() for rv in oracle_rv),parameters,allow_unused=True)
    torch.testing.assert_close(result.kl_value,expected_kl)
    for actuals,expecteds in ((proxy_grads,expected_kl_grads),(actual_gain,expected_r),(result.rv_sum_grads,expected_rv)):
        for actual,expected in zip(actuals,expecteds):
            expected=torch.zeros_like(actual) if expected is None else expected.to(actual)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-4)


@pytest.mark.parametrize('selector',['kl','entropy'])
@pytest.mark.parametrize('mode',['off','full'])
@pytest.mark.parametrize('gain_entropy_mode',['normalized','raw'])
def test_token_entropy_actual_masked_qwen_gradient_and_no_extra_calls(monkeypatch,selector,mode,gain_entropy_mode):
    test_joint_vjp_preserves_optimizer_grads_and_call_budget(torch.float32,.05,mode,monkeypatch,selector,
        gain_weighting='token_entropy',gain_entropy_mode=gain_entropy_mode)


def test_proxy_has_no_unused_edge_and_beta_applies_once():
    module=api(); a=torch.nn.Parameter(torch.ones(2)); unused=torch.nn.Parameter(torch.ones(2))
    events=[];unused.register_hook(lambda g:events.append('unused'))
    proxy=module.optimizer_gradient_proxy(torch.tensor(3.),(torch.tensor([2.,4.]),None),(a,unused))
    (.005*proxy/4).backward()
    torch.testing.assert_close(a.grad,torch.tensor([.0025,.005]))
    assert unused.grad is None and events==[]
    a.grad=None
    (0*module.optimizer_gradient_proxy(torch.tensor(3.),(torch.ones(2),None),(a,unused))).backward()
    assert torch.count_nonzero(a.grad)==0 and unused.grad is None


@pytest.mark.parametrize('weighting',['none','token_entropy'])
def test_no_valid_blocks_has_no_auxiliary_forward_or_vjp(monkeypatch,weighting):
    module=api();param=torch.nn.Parameter(torch.ones(2))
    def unexpected(*args,**kwargs):pytest.fail('empty auxiliary must not execute')
    monkeypatch.setattr(torch.autograd,'grad',unexpected)
    result=module.compute_v3_gradients(None,None,{},V3BatchMetadata(()),torch.zeros(1,2,3),
                                      (param,),Stage1V3Config(gain_weighting=weighting),None)
    assert result.block_count==0 and result.kl_value.item()==0 and result.kl_proxy.item()==0
    primary=result.r_sum_grads if weighting=='none' else result.gain_sum_grads
    assert all(torch.count_nonzero(g)==0 for g in primary+result.rv_sum_grads)


@pytest.mark.parametrize('mode',['early','full'])
def test_tied_unused_parameter_and_peft_exception_cleanup(monkeypatch,mode):
    test_joint_vjp_preserves_optimizer_grads_and_call_budget(torch.float32,.05,mode,monkeypatch,'kl',tied=True)
    with pytest.raises(RuntimeError,match='injected PEFT RV failure'):
        test_joint_vjp_preserves_optimizer_grads_and_call_budget(torch.float32,.05,mode,monkeypatch,'kl',tied=True,fault=True)
