"""Native v5 runtime compared with independent full-sequence oracle.

Mutations caught: detached F K/V, V-off inheriting Text-off, omitted trainable
row paths, different dropout coordinates, reference mutation, wrong reductions.
Oracle uses full-sequence calls and literal masks; never candidate K/V/planning.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest
import torch
import peft
import transformers
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from peft_common import resolve_qwen2_5_vl_model
from stage1.peft import build_stage1_tuned_model
from stage1.types import Stage1TuningConfig
from stage1.equivalence import SampleStableStage1Dropout, stage1_sample_seed


# Declared before any candidate GREEN run; FP32 native RMSNorm is untouched.
ATOL, RTOL = 1e-5, 1e-4
COORD = ContextVar('probe_source_coordinates', default=None)
IDS = torch.tensor([[11,12,13,21,22,23,125,126,126,127,31,32,33,34,35,125,126,126,126,127,41,42,43,44,45,46]])
BODY = (7,8,16,17,18)
# (START, END, actual target token positions, per-predictor previous-text keys)
BLOCKS = ((6,9,(10,11,12),((3,),(4,),(3,5))),
          (15,19,(20,21,22,23,24),((10,),(11,),(12,),(3,10),(4,13))))
PREDICTORS = (9,10,11,19,20,21,22,23)
TARGETS = IDS[0, [10,11,12,20,21,22,23,24]]
POSITIONS = torch.stack((torch.arange(26)*2,torch.arange(26)*3+2,torch.arange(26)+1))[:,None,:]
RECORDS = []


def digest_state(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tensor.dtype,tuple(tensor.shape))).encode())
        digest.update(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class SourceCoordinates:
    """Checkpoint retains these managers across diagnostic and ordinary VJPs."""
    def __init__(self,value):
        self.value = value
        self.tokens = []

    def __enter__(self):
        self.tokens.append(COORD.set(self.value))
        return self

    def __exit__(self,*exc):
        COORD.reset(self.tokens.pop())


class NativeRunner:
    """Common execution-only harness: deterministic source-coordinate dropout.

    Narrowly derived from pinned v3 layer checkpoint closure/context binding,
    not its masks/replay, and schema-v1 native sample-stable dropout seed/draws.
    Actual layer forward and native normalization/DoRA remain untouched.
    """
    def __init__(self, student, reference, checkpointed):
        self.checkpointed = checkpointed
        self.original_calls, self.attention_calls = [], []
        self.layer_completions = []
        self.handles = []
        for qwen, owner in ((student,'student'),(reference,'reference')):
            decoder = qwen.model
            self.handles.append(decoder.register_forward_pre_hook(
                lambda _m,_a,_kw,o=owner: self.original_calls.append((o,COORD.get()[1])), with_kwargs=True))
            for layer in decoder.layers:
                self.handles.append(layer.self_attn.register_forward_pre_hook(
                    lambda _m,_a,_kw,o=owner,i=layer.self_attn.layer_idx:
                    self.attention_calls.append((o,COORD.get()[1],COORD.get()[2],i)), with_kwargs=True))
            for module in qwen.modules():
                if isinstance(module,SampleStableStage1Dropout):
                    def source_dropout(x,m=module):
                        if not m.training or m.p == 0:
                            return x
                        positions,_,_ = COORD.get()
                        generator = torch.Generator(device=x.device)
                        generator.manual_seed(stage1_sample_seed(42,0,7,m.site_name))
                        draws = torch.rand((26,x.shape[-1]), generator=generator,device=x.device)
                        mask = draws[positions].ge(m.p).to(x.dtype)/(1-m.p)
                        return x*mask
                    module.forward = source_dropout

    def __call__(self, decoder, embeds, positions, *, role, source_positions=None,
                 mask=None, capture=False, prefix=None):
        coordinates = (torch.arange(26)[None,:] if source_positions is None else source_positions)
        value = (coordinates,role,'original')
        restorations = []
        try:
            # Own per-layer checkpoints, including packed prefix API that native
            # decoder checkpoint routing intentionally does not checkpoint.
            for layer in decoder.layers:
                original = layer.forward
                restorations.append((layer,'forward' in layer.__dict__,original))
                def wrapped(*args,_original=original,_index=layer.self_attn.layer_idx,**kwargs):
                    def call(*a,**kw):
                        output = _original(*a,**kw)
                        self.layer_completions.append((role,COORD.get()[2],_index))
                        return output
                    if self.checkpointed and decoder.training:
                        return checkpoint(call,*args,use_reentrant=False,
                            context_fn=lambda:(SourceCoordinates(value),SourceCoordinates((coordinates,role,'recompute'))),**kwargs)
                    return call(*args,**kwargs)
                layer.forward = wrapped
            with SourceCoordinates(value):
                return decoder(inputs_embeds=embeds,position_ids=positions,attention_mask=mask,
                               stage2_capture_kv=capture,stage2_prefix_key_values=prefix,
                               use_cache=False,return_dict=True)
        finally:
            for layer,had_instance,original in reversed(restorations):
                if had_instance:
                    layer.forward = original
                else:
                    del layer.forward

    def close(self):
        for handle in self.handles:
            handle.remove()


def fixture(dropout,backend="sdpa"):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config = Qwen2_5_VLConfig(vocab_size=128,hidden_size=8,intermediate_size=16,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
        vision_config=dict(depth=1,hidden_size=8,intermediate_size=16,num_heads=2,
            patch_size=2,spatial_merge_size=1,temporal_patch_size=1,window_size=2,
            out_hidden_size=8,fullatt_block_indexes=[0]),
        image_token_id=80,vision_start_token_id=81,video_token_id=82,
        stage='stage1',latent_size=2,latent_token_id=126,attn_implementation=backend,
        rope_scaling={'type':'mrope','mrope_section':[1,1,0]})
    student = build_stage1_tuned_model(Qwen2_5_VLForConditionalGeneration(config),
        Stage1TuningConfig(method='dora',rank=4,alpha=8,dropout=.05), (125,126,127))
    for module in student.modules():
        if isinstance(module, SampleStableStage1Dropout): module.p = dropout
    # Activate all DoRA/row effects, rather than all-zero LoRA B initialization.
    with torch.no_grad():
        for parameter in student.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter)*.035)
        # Scale 1 produced detached-prefix delta below ATOL. This fixed x4
        # output scale makes both OFF auxiliaries sensitive without changing
        # any objective, normalization, mask or tolerance (sensitivity log).
        head = resolve_qwen2_5_vl_model(student).lm_head
        head.base_head.weight.mul_(4.)
        head.output_delta.mul_(4.)
    reference = deepcopy(student)
    with torch.no_grad():
        for parameter in reference.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter)*.023)
    reference.eval().requires_grad_(False)
    oracle_student,oracle_reference = deepcopy(student),deepcopy(reference)
    live = (torch.randn(1,26,8)*.08).requires_grad_()
    helper = torch.randn(1,26,8,requires_grad=True)
    noise = torch.randn(5,8)
    labels = IDS.clone()
    labels[:,BODY] = -100
    return student,reference,oracle_student,oracle_reference,live,helper,noise,labels


def realize(model,live,helper):
    qwen = resolve_qwen2_5_vl_model(model)
    body = torch.zeros(1,26,1,dtype=torch.bool)
    body[:,BODY] = True
    x = torch.where(body,helper.detach(),qwen.model.embed_tokens(IDS)+live)
    return qwen,x


def oracle(qwen,reference,x,runner):
    # No candidate mask/planning/K/V access: actual independent full sequences.
    clean_hidden = runner(qwen.model,x,POSITIONS,role='F').last_hidden_state
    clean = qwen.lm_head(clean_hidden)
    with torch.no_grad():
        ref = reference.lm_head(runner(reference.model,x.detach(),POSITIONS,role='reference').last_hidden_state)
    off = {'v':[],'text':[]}
    for start,end,targets,selected in BLOCKS:
        for kind in ('v','text'):
            visible = torch.ones(26,26,dtype=torch.bool).tril()
            if kind == 'v':
                # Every replay query from END onward blocks body; START/END
                # stay visible. Earlier prefix rows remain clean/causal.
                visible[end:,start+1:end] = False
            else:
                for target,keys in zip(targets,selected):
                    visible[target-1,list(keys)] = False
            bias = torch.zeros(26,26).masked_fill(~visible,-torch.inf)[None,None]
            output = runner(qwen.model,x,POSITIONS,mask=bias,role=f'{kind}-{start}')
            selected_hidden = output.last_hidden_state[0,torch.tensor(targets)-1]
            off[kind].append(qwen.lm_head(selected_hidden))
    return dict(clean=clean,reference=ref,hidden=clean_hidden,v=torch.cat(off['v']),text=torch.cat(off['text']))


def block_mean(values):
    # Literal fixture reductions: unequal 3/5 lengths, two original blocks.
    return (values[:3].mean()+values[3:].mean())/2


def oracle_losses(output,x,noise,labels,support):
    clean = output['clean'][0,list(PREDICTORS)].float()
    logp = clean.log_softmax(-1)
    logmu = output['reference'][0,list(PREDICTORS)].detach().float().log_softmax(-1)
    gt = (logp-logmu).gather(1,TARGETS[:,None]).squeeze(1)
    dil = block_mean(F.softplus(-gt)+(logmu.exp()*F.softplus(logp-logmu)).sum(-1))
    selected = output['v'].float().gather(-1,support)
    logq = selected.log_softmax(-1)
    entropy = block_mean(torch.log(torch.tensor(50.))+(logq.exp()*logq).sum(-1))
    target = (clean.detach()/2).softmax(-1)
    text_logp = (output['text'].float()/2).log_softmax(-1)
    kl = 4*(target*((clean.detach()/2).log_softmax(-1)-text_logp)).sum(-1).mean()
    ce = F.cross_entropy(output['clean'][:,:-1].float().reshape(-1,128),labels[:,1:].reshape(-1))
    pred = output['hidden'][0,[k-1 for k in BODY]].float()
    goal = x[0,list(BODY)].detach().float()+.01*noise
    alignment = 1-F.cosine_similarity(goal,pred,dim=-1).mean()
    terms = dict(ce=ce,alignment=alignment,dil=dil,entropy=entropy,text_kl=kl)
    terms['total'] = .1*ce+.1*alignment+dil+.1*entropy+.01*kl
    return terms


def runtime_api():
    import importlib.util
    assert importlib.util.find_spec('stage1.v5_execution') is not None, 'missing native v5 execution'
    from stage1.v5_execution import V5Execution
    from stage1.equivalence import Stage1StochasticContext
    from transformers.models.qwen2_5_vl.stage1_v5_trace import capture_v5_trace, bind_v5_execution
    from transformers.models.qwen2_5_vl.stage1_v5_replay import V5ReplayBlock, run_v5_packed_replay
    return V5Execution, Stage1StochasticContext, capture_v5_trace, bind_v5_execution, V5ReplayBlock, run_v5_packed_replay


@contextmanager
def native_execute(qwen, reference, x, checkpointed, control=None, branches=(True,True)):
    Execution, Spec, capture, bind, Block, replay = runtime_api()
    sites = tuple(sorted(m.site_name for m in qwen.modules() if isinstance(m,SampleStableStage1Dropout)))
    owner = Execution(Spec(42,0,((7,26),),sites))
    qwen.model.gradient_checkpointing = checkpointed
    blocks = tuple(Block(0,s,e,t,tuple(IDS[0,list(t)].tolist()),k) for s,e,t,k in BLOCKS)
    keys = tuple((0,p) for p in PREDICTORS)
    candidates = tuple(tuple(range(3,s)) for s,e,t,k in BLOCKS for _ in t)
    calls = []
    handles = [qwen.model.register_forward_pre_hook(lambda *_:calls.append('F' if not calls else 'packed')),
               reference.model.register_forward_pre_hook(lambda *_:calls.append('reference'))]
    try:
        owner.__enter__()
        with owner.graph('F'), bind(qwen.model,owner,role='F',visual=qwen.visual), capture(qwen,keys,candidates) as trace:
            clean = qwen.model(inputs_embeds=x,position_ids=POSITIONS,use_cache=False,return_dict=True)
        clean_logits = qwen.lm_head(clean.last_hidden_state)
        with torch.no_grad():
            frozen = reference(inputs_embeds=x.detach(),position_ids=POSITIONS,use_cache=False,
                               stage1_selected_positions=keys,return_dict=True)
        ref = frozen.stage1_selected_logits
        if control == 'detach_prefix':
            trace.key_values = tuple(tuple(v.detach() for v in pair) for pair in trace.key_values)
        if control == 'contaminate_v':
            def contaminate(_module,args,kwargs):
                if kwargs.get('stage2_prefix_key_values') is not None:
                    mask = kwargs['attention_mask'].clone()
                    for row,block in enumerate(blocks):
                        for target,keys in zip(block.target_positions,block.blocked_text_keys):
                            mask[row,0,target-1-block.end,list(keys)] = -torch.inf
                    return args,dict(kwargs,attention_mask=mask)
            handles.append(qwen.model.register_forward_pre_hook(contaminate,with_kwargs=True))
        with owner.graph('packed'):
            off = replay(qwen.model,qwen.lm_head,trace,blocks,execution=owner,
                         need_voff=branches[0],need_textoff=branches[1],chunk_positions=3)
        assert off.query_keys == keys
        assert all(not row.requires_grad for row in trace.attention_rows.values())
        text = off.textoff_logits.detach() if control == 'detach_text' else off.textoff_logits
        yield dict(clean=clean_logits,reference=ref,hidden=clean.last_hidden_state,
                    v=off.voff_logits,text=text),owner,calls,trace
    finally:
        for handle in handles: handle.remove()
        if owner.state in ('ACTIVE','BACKWARD'):
            owner.__exit__(*sys.exc_info())


def terms(output,x,noise,labels,support):
    # Independent literal equations; selected reference already has [N,V].
    full_ref = output['clean'].detach().clone()
    full_ref[0,list(PREDICTORS)] = output['reference']
    return oracle_losses(dict(output,reference=full_ref),x,noise,labels,support)


def gradients(model,live,loss=None):
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]+[('LIVE_INPUT',live)]
    if loss is not None:
        values = torch.autograd.grad(loss,[p for _,p in named],retain_graph=True,allow_unused=True) if loss.requires_grad else [None]*len(named)
    else:
        values = [p.grad for _,p in named]
        assert all(v is not None for v in values)
    return {n:torch.zeros_like(p) if v is None else v.detach().clone() for (n,p),v in zip(named,values)}


def equal(got,want):
    for name in got:
        torch.testing.assert_close(got[name],want[name],atol=ATOL,rtol=RTOL,msg=name)


@pytest.mark.parametrize('dropout',[0.,.05])
@pytest.mark.parametrize('checkpoint_mode',['off','early','full'])
def test_native_combined_and_raw_K_match_literal_full_masks(dropout,checkpoint_mode):
    runtime_api()
    student,reference,os,oref,live,helper,noise,labels = fixture(dropout)
    q,x = realize(student,live,helper)
    ol = live.detach().clone().requires_grad_()
    oq,ox = realize(os,ol,helper)
    rq,orq = resolve_qwen2_5_vl_model(reference),resolve_qwen2_5_vl_model(oref)
    runner = NativeRunner(oq,orq,checkpoint_mode!='off')
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad])
    owned = {id(p) for group in optimizer.param_groups for p in group['params']}
    try:
        with set_checkpoint_early_stop(checkpoint_mode!='full'):
            with native_execute(q,rq,x,checkpoint_mode!='off') as (got,owner,calls,trace):
                wanted = oracle(oq,orq,ox,runner)
                equal({k:got[k] for k in ('clean','hidden','v','text')},{k:wanted[k] for k in ('clean','hidden','v','text')})
                torch.testing.assert_close(got['reference'],wanted['reference'][0,list(PREDICTORS)],atol=ATOL,rtol=RTOL)
                support = got['v'].detach().topk(50,dim=-1).indices
                losses,expected = terms(got,x,noise,labels,support),oracle_losses(wanted,ox,noise,labels,support)
                equal(losses,expected)
                with owner.backward():
                    diagnostic_start = len(owner.events)
                    raw_got,raw_want = gradients(student,live,losses['text_kl']),gradients(os,ol,expected['text_kl'])
                    equal(raw_got,raw_want)
                    diagnostic = [e for e in owner.events[diagnostic_start:] if e['site'] is None and e['phase']=='recompute']
                    ordinary_start = len(owner.events)
                    losses['total'].backward()
                    ordinary = [e for e in owner.events[ordinary_start:] if e['site'] is None and e['phase']=='recompute']
                    assert bool(diagnostic) == (checkpoint_mode!='off')
                    assert bool(ordinary) == (checkpoint_mode!='off')
                expected['total'].backward()
        assert calls == ['F','reference','packed']
        equal(gradients(student,live),gradients(os,ol))
        print(json.dumps(dict(case=[dropout,checkpoint_mode],original_calls=calls,
            raw_K_max_gradient_error=max(float((raw_got[n]-raw_want[n]).abs().max()) for n in raw_got),
            total_max_gradient_error=max(float((gradients(student,live)[n]-gradients(os,ol)[n]).abs().max()) for n in raw_got),
            diagnostic_recompute_units=[e['role']+':'+str(e['unit']) for e in diagnostic],
            ordinary_recompute_units=[e['role']+':'+str(e['unit']) for e in ordinary]),sort_keys=True))

        assert {id(p) for p in student.parameters() if p.requires_grad and p.grad is not None} == owned
        assert helper.grad is None
        assert all(p.grad is None for p in reference.parameters())
        assert torch.count_nonzero(live.grad[0,list(BODY)]) == 0
        assert live.grad[0,[3,4,5,10,11,12,13,14]].abs().sum()>0
        assert q.model.gradient_checkpointing == (checkpoint_mode!='off')
        assert all('forward' not in layer.__dict__ for layer in q.model.layers)
        assert bool([e for e in owner.events if e['phase']=='recompute']) == (checkpoint_mode!='off')
    finally:
        runner.close()


@pytest.mark.parametrize('control',['detach_prefix','contaminate_v','detach_text'])
def test_causal_mutations_fail_actual_oracle(control):
    runtime_api()
    student,reference,os,oref,live,helper,noise,labels = fixture(.05)
    q,x = realize(student,live,helper)
    ol = live.detach().clone().requires_grad_()
    oq,ox = realize(os,ol,helper)
    rq,orq = resolve_qwen2_5_vl_model(reference),resolve_qwen2_5_vl_model(oref)
    runner = NativeRunner(oq,orq,False)
    try:
        with native_execute(q,rq,x,False,control) as (got,owner,calls,_):
            wanted = oracle(oq,orq,ox,runner)
            support = wanted['v'].detach().topk(50,dim=-1).indices
            losses,expected = terms(got,x,noise,labels,support),oracle_losses(wanted,ox,noise,labels,support)
            with owner.backward():
                if control == 'contaminate_v':
                    with pytest.raises(AssertionError) as caught:
                        torch.testing.assert_close(got['v'],wanted['v'],atol=ATOL,rtol=RTOL)
                    print('NEGATIVE_CONTROL '+control+': '+str(caught.value))
                    losses['total'].backward()
                else:
                    objective = 'text_kl' if control=='detach_text' else 'total'
                    with pytest.raises(AssertionError) as caught:
                        equal(gradients(student,live,losses[objective]),gradients(os,ol,expected[objective]))
                    print('NEGATIVE_CONTROL '+control+': '+str(caught.value))
                    losses['total'].backward()
    finally:
        runner.close()


@pytest.mark.parametrize('branches',[(True,False),(False,True),(False,False)])
def test_optional_branches_prune_and_match_independent_oracle(branches):
    runtime_api()
    student,reference,os,oref,live,helper,noise,labels = fixture(.05)
    q,x = realize(student,live,helper)
    oq,ox = realize(os,live.detach(),helper)
    rq,orq = resolve_qwen2_5_vl_model(reference),resolve_qwen2_5_vl_model(oref)
    runner = NativeRunner(oq,orq,False)
    try:
        with native_execute(q,rq,x,False,branches=branches) as (got,owner,calls,_):
            want = oracle(oq,orq,ox,runner)
            for enabled,key in zip(branches,('v','text')):
                if enabled: torch.testing.assert_close(got[key],want[key],atol=ATOL,rtol=RTOL)
                else: assert got[key] is None
            assert calls == (['F','reference','packed'] if any(branches) else ['F','reference'])
            with owner.backward(): got['clean'].sum().backward()
    finally: runner.close()
