"""OFF checkpoint opt-in, actual decoder replay, and ordinary diagnostic contracts."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch


def test_off_checkpoint_reaches_config_and_both_worker_environments():
    from utils_deepseed import parse_args
    from src.multinode_runtime import validated_stage2_environment
    from tests.stage2.test_launcher import run_launcher, dry_run_tokens
    base = ['--stage','stage2','--stage2_execution_profile','latest_external_until']
    assert not getattr(parse_args(base).step_preference_config,'off_activation_checkpointing',False)
    assert parse_args(base+['--step_off_activation_checkpointing']).step_preference_config.off_activation_checkpointing
    environment = validated_stage2_environment(dict(STAGE1_CHECKPOINT='/models/stage1',DATA_PATH='/data/train',
        IMAGE_ROOT='/data/images',STAGE2_EXECUTION_PROFILE='latest_external_until',STEP_OFF_ACTIVATION_CHECKPOINTING='true'),
        output_dir='/checkpoints/probe',log_file='/logs/probe',deepspeed_config='/config.json')
    assert environment['STEP_OFF_ACTIVATION_CHECKPOINTING']=='true'
    assert '--step_off_activation_checkpointing' in dry_run_tokens(run_launcher(env_overrides=environment))
    assert '--step_off_activation_checkpointing' not in dry_run_tokens(run_launcher(env_overrides={'STEP_OFF_ACTIVATION_CHECKPOINTING':'false'}))


def test_off_checkpoint_real_dora_decoder_preserves_named_ce_preference_gradients_rng_and_cache():
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora.dora import DoraLinearLayer
    from stage2.dora import dora_microbatch_scope
    from step_preference.config import StepPreferenceConfig
    from step_preference.equivalence import _trainer
    from step_preference.shared_prefix_measurement import tiny_measurement_case
    from scripts.verify_until_trainer import probed_trainer_class, off_replay_witness
    from stage2.trainer import Stage2Trainer
    model,batch=tiny_measurement_case(steps=2,total_target_tokens=4,device='cpu')
    model.config.stage2_execution_profile='latest_external_until'
    model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,lora_dropout=.2,use_dora=True,
        target_modules=['q_proj','k_proj','v_proj','o_proj'])).get_base_model().train()
    model.model.embed_tokens.weight.requires_grad_(True)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'lora_B' in name: p.normal_(std=.02)
    batch['step_preference']['steps']=tuple({'span':s} for s in batch['reasoning_step_spans'])
    records=[]
    for enabled in (False,True):
        current=copy.deepcopy(model)
        config=StepPreferenceConfig(enabled=True,mode='stage2b',direct_from_stage1=True,
            beta=.5,lambda_entropy=0.,shared_prefix_enabled=True,sidecar_attention_backend='dense',
            target_scope='until_next_latent',visual_off_suppress_latent_boundaries=False)
        # Exercise the actual consumer even on baseline, where the new field is ignored.
        object.__setattr__(config,'off_activation_checkpointing',enabled)
        cls=probed_trainer_class(Stage2Trainer,{'path':'aligned'}, {})
        trainer=cls.__new__(cls); trainer.__dict__.update(_trainer(config).__dict__)
        trainer.stage2_execution_profile='latest_external_until'
        trainer.args=SimpleNamespace(stage2_debugging=False)
        trainer._gate_capture=None
        captured=[]
        trainer.gate_observer=SimpleNamespace(observe=lambda total,pref,inputs,scores:captured.append((pref,scores)))
        calls=[]; phase=['forward']; norm_calls=[]
        magnitudes=[m for m in current.modules() if isinstance(m,DoraLinearLayer)]
        for m in magnitudes:
            original=m.get_weight_norm
            def norm(*args,_original=original,**kw):
                norm_calls.append(phase[0]); return _original(*args,**kw)
            m.get_weight_norm=norm
        def entry(module,args,kwargs):
            cache=kwargs.get('past_key_values')
            if cache is not None:
                calls.append((phase[0],cache,cache.get_seq_length(),all('forward' in m.__dict__ for m in magnitudes),
                              tuple(t for pair in zip(cache.key_cache,cache.value_cache) for t in pair)))
        hook=current.model.register_forward_pre_hook(entry,with_kwargs=True)
        named=[(n,p) for n,p in current.named_parameters() if p.requires_grad]
        torch.manual_seed(42)
        with off_replay_witness(trainer,current) as off_calls, dora_microbatch_scope(current):
            total=trainer.compute_loss(current,copy.deepcopy(batch))
            assert len(captured)==1, 'actual local reducer was not intercepted'
            pref,scores=captured[0]; ce=total-pref
            prefix_named=[(f'prefix/{i//2}/{"key" if i%2==0 else "value"}',tensor)
                          for i,tensor in enumerate(calls[0][4])]
            rng=torch.get_rng_state().clone(); values={}
            for label,loss in [('ce',ce),('preference',pref),('total',total)]:
                phase[0]=label
                grads=torch.autograd.grad(loss,[p for _,p in named+prefix_named],allow_unused=True,retain_graph=True)
                values[label]={n:None if g is None else g.clone() for (n,p),g in zip(named+prefix_named,grads)}
                assert torch.equal(rng,torch.get_rng_state())
                assert all(p.grad is None for _,p in named)
            phase[0]='ordinary';total.backward()
            for n,p in named:
                if p.grad is not None: torch.testing.assert_close(p.grad,values['total'][n],atol=1e-7,rtol=1e-5)
            assert torch.equal(rng,torch.get_rng_state())
        hook.remove()
        assert all('forward' not in m.__dict__ for m in magnitudes)
        assert all(row[3] for row in calls), 'checkpoint replay escaped DoRA scope'
        assert len(norm_calls)==len(magnitudes), 'checkpoint replay recomputed detached norms'
        assert len({id(row[1]) for row in calls})==len(calls), 'reused mutated sidecar cache'
        assert all(row[2]==batch['input_ids'].shape[1] for row in calls)
        if enabled: assert any(row[0]=='ordinary' for row in calls), 'OFF checkpoint was ignored'
        else: assert len(calls)==1
        assert off_calls['forward']==1 and (off_calls['backward']>0)==enabled
        assert values['preference']['model.embed_tokens.weight'].abs().max()>0
        assert all(values['preference'][name].abs().max()>0 for name,_ in prefix_named)
        records.append((scores,values,rng))
    for label in records[0][0]: torch.testing.assert_close(records[0][0][label],records[1][0][label],atol=0,rtol=0)
    assert torch.equal(records[0][2],records[1][2])
    for loss in records[0][1]:
        for name,left in records[0][1][loss].items():
            right=records[1][1][loss][name]
            assert (left is None)==(right is None),name
            if left is not None: torch.testing.assert_close(left,right,atol=1e-7,rtol=1e-5,msg=f'{loss}/{name}')


def test_aligned_memory_probe_selects_explicit_source_groups_and_preserves_strict_mode():
    from scripts.verify_until_trainer import memory_diagnostic,snapshot_inventory,probe_compiler_context
    from stage2.until_alignment_probe import select_windows
    from torch._functorch import config
    spec=dict(path='aligned',mode='ordinary_backward_memory',source_updates=[60,1],off_activation_checkpointing=True)
    assert memory_diagnostic(spec)
    order=list(range(3379))+[7];lengths=[10]*3379;latent=[2]*3379
    replay=select_windows(order,lengths,latent,source_updates=spec['source_updates'])
    assert replay['indices']==order[59*16:60*16]+order[:16]
    assert replay['source_updates']==[60,1] and len(replay['coefficients'])==32
    assert replay['lengths']==[10]*32 and replay['latent_counts']==[2]*32
    assert 'rank-0/update-2-total-actual.pt' in snapshot_inventory(spec,ranks=[0])
    with config.patch(donated_buffer=True):
        with probe_compiler_context(spec): assert config.donated_buffer is True
        with probe_compiler_context({'path':'external'}): assert config.donated_buffer is False
        assert config.donated_buffer is True
    for change in ({'save_on_cpu':True},{'source_updates':[1,1]},{'source_updates':[1,212]}, {'off_activation_checkpointing':1}):
        with pytest.raises(ValueError): memory_diagnostic(dict(spec,**change))


def test_memory_probe_snapshots_only_existing_accumulated_gradients(tmp_path,monkeypatch):
    from stage2.until_alignment_probe import MemoryDiagnosticObserver
    from deepspeed.utils import safe_get_full_grad
    observer=MemoryDiagnosticObserver.__new__(MemoryDiagnosticObserver)
    observer.capture_ordinary_gradients=True
    observer.rank=observer.update=0;observer.micro=3
    observer.tensor_dir=tmp_path;observer.stream=(tmp_path/'events.jsonl').open('w')
    observer.replay={'source_updates':[60,1]}
    parameter=torch.nn.Parameter(torch.tensor([2.,3.]));unused=torch.nn.Parameter(torch.tensor([1.]))
    observer.named=[('w',parameter),('unused',unused)]
    # Emulate only the ZeRO master accessor; the accumulation/update is real CPU math.
    parameter._hp_mapping=True
    parameter.get_full_hp_param=lambda:parameter.data
    observer.engine=SimpleNamespace(device='cpu',optimizer=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{'lr':0.}]),cpu_offload=False))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda d:0)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda d:0)
    monkeypatch.setattr(torch.autograd,'grad',lambda *a,**kw:pytest.fail('extra gradient read'))
    for _ in range(4): (parameter.square().sum()/4).backward()
    rng=torch.get_rng_state().clone();original=parameter.grad.clone()
    observer.before_step()
    snapshot=torch.load(tmp_path/'update-1-total-actual.pt',weights_only=True)
    assert torch.equal(snapshot['w'],original) and snapshot['unused'] is None
    assert torch.equal(parameter.grad,original) and torch.equal(rng,torch.get_rng_state())
    torch.optim.SGD([parameter],lr=.1).step()
    observer.after_step();observer.stream.close()
    event=json.loads((tmp_path/'events.jsonl').read_text().splitlines()[-1])
    assert event['original_update_index']==60 and event['ordinary_gradient_snapshot']=='update-1-total-actual.pt'
    assert event['master_witness_name']=='w' and event['master_changed'] is True


def test_aligned_capacity_acceptance_requires_real_replay_gradients_and_nonzero_update():
    from stage2.until_alignment_probe import check_memory_runtime
    replay={'sample_ids':[f's{i}' for i in range(32)],'source_updates':[60,61],
            'lengths':[100]*32,'latent_counts':[2]*32}
    records=[]
    for rank in range(4):
        records += [dict(event='init',rank=rank,path='aligned',hostname=f'n{rank//2}',world_size=4,
            microbatch=1,gas=4,zero_stage=2,schedule_steps=212,warmup_steps=10),
            dict(event='ordinary_probe_config',rank=rank,off_activation_checkpointing=True,
                 save_on_cpu=False,optimizer_cpu_offload=False,donated_buffer=True),
            dict(event='compiler',rank=rank,calls=[dict(callable='torch.nn.attention.flex_attention.flex_attention',
                 dynamic=True,donated_buffer=True)])]
        for micro in range(8):
            for event in ('micro_entry','forward','backward'):
                records.append(dict(event=event,rank=rank,micro=micro,sample_id=f's{micro*4+rank}',
                    allocated=1,peak_allocated=2,tokens=100,latent_blocks=2,original_update_index=[60,61][micro//4],
                    actual_backward_hooks=1,dora_scope_cleaned=True,off_decoder_calls={'forward':1,'backward':1}))
        for step in (1,2):
            records.append(dict(event='memory_update',rank=rank,probe_optimizer_step=step,lr=[(step-1)*5e-6],
                original_update_index=[60,61][step-1],ordinary_gradient_snapshot=f'update-{step}-total-actual.pt',
                master_witness_name='w',master_changed=step==2))
    assert check_memory_runtime(records,replay)['complete']
    assert not check_memory_runtime(records,replay)['comparison_pass']
    for event,key,value in [('backward','off_decoder_calls',{'forward':1,'backward':0}),
                            ('memory_update','ordinary_gradient_snapshot',None),
                            ('memory_update','master_changed',False),
                            ('ordinary_probe_config','optimizer_cpu_offload',True),
                            ('forward','original_update_index',1)]:
        altered=copy.deepcopy(records)
        row=next(r for r in reversed(altered) if r['event']==event)
        row[key]=value
        assert not check_memory_runtime(altered,replay)['complete'],(event,key)


def test_default_replay_preserves_frozen_strict_envelope_schema():
    """Unrequested geometry keys invalidate previously frozen structural identity."""
    from stage2.until_alignment_probe import select_windows
    order=list(range(84));lengths=[10]*84;lengths[65]=100
    # Baseline frozen schema and hand-selected first three/challenge/tail groups.
    expected={'source_updates':[1,2,3,5,6],'indices':list(range(48))+list(range(64,84)),
        'coefficients':[1/16]*68,'challenge_source_update':5,
        'challenge_proxy':'sum(length^2*(1+latent_count))','historical_optimizer_state_reproduced':False}
    assert select_windows(order,lengths,[1]*84)==expected
