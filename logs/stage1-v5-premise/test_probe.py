"""Throwaway CPU design-premise experiment; not Stage1-v5 product tests.

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

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'results/stage1-v5-premise/result.json'
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


@contextmanager
def execution_coordinates(value):
    token = COORD.set(value)
    try:
        yield
    finally:
        COORD.reset(token)


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
                            context_fn=lambda:(execution_coordinates(value),execution_coordinates((coordinates,role,'recompute'))),**kwargs)
                    return call(*args,**kwargs)
                layer.forward = wrapped
            with execution_coordinates(value):
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


def fixture(dropout):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config = Qwen2_5_VLConfig(vocab_size=128,hidden_size=8,intermediate_size=16,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
        vision_config=dict(depth=1,hidden_size=8,intermediate_size=16,num_heads=2,
            patch_size=2,spatial_merge_size=1,temporal_patch_size=1,window_size=2,
            out_hidden_size=8,fullatt_block_indexes=[0]),
        image_token_id=80,vision_start_token_id=81,video_token_id=82,
        stage='stage1',latent_size=2,latent_token_id=126,attn_implementation='sdpa',
        rope_scaling={'type':'mrope','mrope_section':[1,1,0]})
    student = build_stage1_tuned_model(Qwen2_5_VLForConditionalGeneration(config),
        Stage1TuningConfig(method='dora',rank=4,alpha=8,dropout=dropout), (125,126,127))
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


def candidate_adapter():
    try:
        module = importlib.import_module('probe_adapter')
    except ModuleNotFoundError as error:
        if error.name != 'probe_adapter':
            raise
        module = None
    assert module is not None, 'candidate shared-F packing adapter is missing (expected TDD RED)'
    return module


def assert_equivalent(got,want):
    errors = {}
    for name in got:
        torch.testing.assert_close(got[name],want[name],atol=ATOL,rtol=RTOL,msg=name)
        errors[name] = float((got[name].detach()-want[name].detach()).abs().max())
    return errors


def gradients(model,live):
    answer = {}
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            answer[name] = parameter.grad.detach().clone()
    assert live.grad is not None
    answer['LIVE_INPUT'] = live.grad.detach().clone()
    return answer


@pytest.mark.parametrize('dropout',[0.,.05])
@pytest.mark.parametrize('checkpoint_mode',['off','early','full'])
def test_fused_shared_F_matches_literal_masked_calls_all_gradients(dropout,checkpoint_mode):
    adapter = candidate_adapter()
    student,reference,os,oref,live,helper,noise,labels = fixture(dropout)
    qwen,x = realize(student,live,helper)
    ol = live.detach().clone().requires_grad_()
    oq,ox = realize(os,ol,helper)
    rq,orq = resolve_qwen2_5_vl_model(reference),resolve_qwen2_5_vl_model(oref)
    assert digest_state(student) != digest_state(reference)
    runner,orr = NativeRunner(qwen,rq,checkpoint_mode!='off'),NativeRunner(oq,orq,checkpoint_mode!='off')
    owned = {name:id(p) for name,p in student.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=1e-5)
    other_optimizer = torch.optim.AdamW([p for p in os.parameters() if p.requires_grad],lr=1e-5)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == set(owned.values())
    reference_before = digest_state(reference)
    try:
        with set_checkpoint_early_stop(checkpoint_mode!='full'):
            got = adapter.execute(qwen,rq,x,POSITIONS,BLOCKS,runner)
            support = got['v'].detach().float().topk(50,dim=-1).indices
            want = oracle(oq,orq,ox,orr)
            outputs_error = assert_equivalent(got,want)
            terms = adapter.losses(got,x,noise,labels,support,IDS,BODY,PREDICTORS,BLOCKS)
            expected = oracle_losses(want,ox,noise,labels,support)
            loss_error = assert_equivalent(terms,expected)
            calls_before = list(runner.original_calls)
            assert calls_before == [('student','F'),('reference','reference'),('student','packed')]
            terms['total'].backward()
            expected['total'].backward()
        assert runner.original_calls == calls_before, 'layer replay must not introduce new decoder calls'
        gradient_error = assert_equivalent(gradients(student,live),gradients(os,ol))
        nonzero = [name for name,p in student.named_parameters() if p.requires_grad and p.grad.abs().sum()>0]
        assert any('lora_B' in n for n in nonzero)
        assert any('lora_A' in n for n in nonzero)
        assert any('lora_magnitude_vector' in n for n in nonzero)
        assert any('trainable_tokens' in n for n in nonzero)
        assert any('output_delta' in n for n in nonzero)
        assert live.grad[0,[3,4,5,10,11,12,13,14]].abs().sum()>0
        assert torch.count_nonzero(live.grad[0,list(BODY)]) == 0
        assert helper.grad is None
        assert not rq.training and all(not p.requires_grad and p.grad is None for p in reference.parameters())
        assert digest_state(reference) == reference_before
        optimizer.step()
        other_optimizer.step()
        updates_error = assert_equivalent(
            {n:p for n,p in student.named_parameters() if p.requires_grad},
            {n:p for n,p in os.named_parameters() if p.requires_grad})
        assert {name:id(p) for name,p in student.named_parameters() if p.requires_grad} == owned
        assert digest_state(reference) == reference_before
        # Same realized history still produces identical ref outputs after student update.
        with torch.no_grad():
            repeat = rq.lm_head(runner(rq.model,x.detach(),POSITIONS,role='reference_post_update').last_hidden_state)
        torch.testing.assert_close(repeat,got['reference'],atol=0,rtol=0)
        recomputations = [v for v in runner.attention_calls if v[2]=='recompute']
        assert bool(recomputations) == (checkpoint_mode!='off')
        completed_replays = [v for v in runner.layer_completions if v[1]=='recompute']
        assert len(completed_replays) == (4 if checkpoint_mode=='full' else 0)
        record = dict(dropout=dropout,checkpoint_mode=checkpoint_mode,status='PASS',
            output_max_errors=outputs_error,loss_max_errors=loss_error,
            loss_values={name:float(value.detach()) for name,value in terms.items()},
            gradient_max_error=max(gradient_error.values()),
            update_max_error=max(updates_error.values()),trainable_parameters=len(owned),
            nonzero_trainable_gradients=len(nonzero),original_calls=calls_before,
            recomputation_attention_calls=len(recomputations),
            recomputation_completed_layers=len(completed_replays),
            reference_state_sha256=reference_before,
            boundaries=dict(reference_frozen=True,reference_immutable=True,helper_grad_none=True,
                live_input_gradient=True,optimizer_parameter_identity=True),
            fixed_candidates_sha256=hashlib.sha256(support.numpy().tobytes()).hexdigest())
        RECORDS.append(record)
        print(json.dumps(dict(record,gradient_errors=gradient_error,
                              attention_calls=runner.attention_calls,
                              layer_completions=runner.layer_completions),sort_keys=True))
    finally:
        runner.close()
        orr.close()


@pytest.mark.parametrize('control',['detach_prefix','contaminate_v'])
def test_causal_negative_controls_fail_the_intended_equivalence(control):
    adapter = candidate_adapter()
    student,reference,os,oref,live,helper,noise,labels = fixture(.05)
    qwen,x = realize(student,live,helper)
    ol = live.detach().clone().requires_grad_()
    oq,ox = realize(os,ol,helper)
    rq,orq = resolve_qwen2_5_vl_model(reference),resolve_qwen2_5_vl_model(oref)
    runner,orr = NativeRunner(qwen,rq,False),NativeRunner(oq,orq,False)
    try:
        ordinary = adapter.execute(qwen,rq,x,POSITIONS,BLOCKS,runner)
        support = ordinary['v'].detach().float().topk(50,dim=-1).indices
        broken = adapter.execute(qwen,rq,x,POSITIONS,BLOCKS,runner,control=control)
        want = oracle(oq,orq,ox,orr)
        terms = adapter.losses(broken,x,noise,labels,support,IDS,BODY,PREDICTORS,BLOCKS)
        expected = oracle_losses(want,ox,noise,labels,support)
        if control == 'detach_prefix':
            assert_equivalent(broken,want)  # Same forward; only autograd path changes.
            assert_equivalent(terms,expected)
            terms['total'].backward()
            expected['total'].backward()
            bad,wanted = gradients(student,live),gradients(os,ol)
            with pytest.raises(AssertionError,match='LIVE_INPUT') as caught:
                torch.testing.assert_close(bad['LIVE_INPUT'],wanted['LIVE_INPUT'],atol=ATOL,rtol=RTOL,
                    msg=lambda diagnostic:'LIVE_INPUT detached prefix\n'+diagnostic)
            error = float((bad['LIVE_INPUT']-wanted['LIVE_INPUT']).abs().max())
        else:
            with pytest.raises(AssertionError,match='V_OFF') as caught:
                torch.testing.assert_close(broken['v'],want['v'],atol=ATOL,rtol=RTOL,
                    msg=lambda diagnostic:'V_OFF contaminated text mask\n'+diagnostic)
            error = float((broken['v'].detach()-want['v'].detach()).abs().max())
        # Keep intended assertion evidence in logs, including mismatch diagnostics.
        print(f'NEGATIVE_CONTROL {control}: expected equivalence rejection\n{caught.value}')
        RECORDS.append(dict(control=control,status='EXPECTED_EQUIVALENCE_REJECTION',max_error=error,
            fixed_candidates_sha256=hashlib.sha256(support.numpy().tobytes()).hexdigest()))
    finally:
        runner.close()
        orr.close()


@pytest.fixture(scope='session',autouse=True)
def save_results():
    yield
    source_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob('*.py')}
    used_sources = ('src/stage1/peft.py','src/stage1/equivalence.py','src/stage1/trainer.py',
                    'src/stage1/token_rows.py','src/step_preference/dil_bce_orig_loss.py',
                    'src/step_preference/topk_entropy_loss.py',
                    'transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py')
    payload = dict(artifact_kind='throwaway_CPU_premise_probe',base_commit='3441a1ce9d205b175681ef229aedb8643f23718e',
        overall_outcome='supported_within_probe_scope' if len(RECORDS)==8 else 'incomplete_or_failed',
        expected_cases=8,observed_case_records=len(RECORDS),
        command="PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q -s logs/stage1-v5-premise/test_probe.py",
        tolerances=dict(atol=ATOL,rtol=RTOL),cases=RECORDS,source_sha256=source_hashes,
        limits=['Fixed helper fixtures; no helper pass or full four-call Trainer witness',
                'CPU FP32 SDPA tiny Qwen/DoRA; no GPU, distributed, Trainer, EMA lifecycle or ZeRO witness',
                'Fixed text-key selections/candidate IDs/noise; does not test selectors or data mapping',
                'Source-coordinate adapter dropout harness; no replacement of native RMSNorm arithmetic'],
        environment=dict(python=sys.version,executable=sys.executable,
                         torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__),
        fixture=dict(sequence_length=26,hidden_size=8,layers=2,vocab=128,dora_rank=4,alpha=8,
                     blocks=BLOCKS,body=BODY,predictors=PREDICTORS,head_scale=4.),
        runtime_source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in used_sources},
        origin=dict(checkpoint_contexts='pinned e555a2b6ddf43d689cceca7342fa3fb6a17b90c3 stage1_v3_trace.py per-layer closure/context idea',
                    dropout='current stage1.equivalence.stage1_sample_seed and native full source draw shape'))
    RESULT.write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
