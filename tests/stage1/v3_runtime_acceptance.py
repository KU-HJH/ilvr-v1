"""Fixed coordinator-owned tiny Qwen acceptance; never a single-node launcher.

The CPU tests check instrumentation/validation. Only an actual four-rank run
can produce acceptance evidence. Snapshots containing tensors stay in checkpoints.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import replace
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import socket
from importlib.metadata import version
import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainerCallback, set_seed
from transformers.integrations.deepspeed import unset_hf_deepspeed_config
from peft_common import resolve_qwen2_5_vl_model
from stage1.trainer import CustomTrainerStage1
from stage1.v3_lifecycle import current_v3_lifecycle
from stage1.v3_resume import V3_FRESH_PROFILE, read_v3_contract, validate_v3_resume, teacher_digest

KIND='stage1-v3-runtime-acceptance'
ROOT=Path(__file__).resolve().parents[2]
PACKAGES=('torch','transformers','peft','trl','accelerate','deepspeed')
ATOL,RTOL=1e-6,1e-4


@contextmanager
def observe_microstep(trainer):
    """Observe the actual product boundary, including inherited ordinary backward."""
    event=dict(core_forwards=[],aux_vjp_arities=[],ordinary_backwards=0,helper_builds=0,sites=[])
    qwen=resolve_qwen2_5_vl_model(trainer.model)
    teacher=resolve_qwen2_5_vl_model(trainer.ema_teacher.model)
    original_build=trainer.target_builder.build
    original_backward=trainer.accelerator.backward
    original_grad=torch.autograd.grad
    in_helper=False
    lives=[]
    def build(*a,**kw):
        nonlocal in_helper
        event['helper_builds']+=1;in_helper=True
        try:return original_build(*a,**kw)
        finally:in_helper=False
    def backward(*a,**kw):
        event['ordinary_backwards']+=1
        life=current_v3_lifecycle();lives.append(life)
        return original_backward(*a,**kw)
    def grad(outputs,*a,**kw):
        life=current_v3_lifecycle()
        if life and life.frame[3] in ('joint_r_kl','rv'):
            event['aux_vjp_arities'].append(len(outputs) if isinstance(outputs,tuple) else 1)
        return original_grad(outputs,*a,**kw)
    hooks=[qwen.model.register_forward_pre_hook(lambda *a:event['core_forwards'].append(current_v3_lifecycle().frame[0])),
           teacher.model.register_forward_pre_hook(lambda *a:None if in_helper else event['core_forwards'].append('reference'))]
    trainer.target_builder.build=build;trainer.accelerator.backward=backward;torch.autograd.grad=grad
    try:yield event
    finally:
        trainer.target_builder.build=original_build;trainer.accelerator.backward=original_backward;torch.autograd.grad=original_grad
        for hook in hooks:hook.remove()
        if lives:
            event['sites']=list(lives[-1].events)
            event['lifecycle']=lives[-1].state


def _digest(value):
    """Deterministic content hash for nested CPU tensor/runtime state."""
    h=hashlib.sha256()
    def add(v):
        if isinstance(v,torch.Tensor):
            x=v.detach().cpu().contiguous();h.update(str((str(x.dtype),tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((str(v.dtype),v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):add(k);add(v[k])
        elif isinstance(v,(tuple,list)):
            h.update(str(type(v)).encode())
            for x in v:add(x)
        else:h.update(repr(v).encode())
    add(value);return h.hexdigest()


def _finite(record,key,limit):
    value=record.get(key)
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=limit:
        raise ValueError('invalid v3 numeric witness: '+key)


def _events(events,rank,count):
    if not isinstance(events,list) or len(events)!=count:raise ValueError('missing v3 microstep events')
    for e in events:
        if (type(e.get('occurrence')) is not int or not 0<=e['occurrence']<24
                or e.get('helper_builds')!=1 or e.get('ordinary_backwards')!=1 or e.get('lifecycle')!='CLOSED'
                or e.get('core_forwards')!=(['F'] if rank==0 else ['F','gain_clean','reference','packed_replay'])
                or e.get('aux_vjp_arities')!=([] if rank==0 else [2,1])):
            raise ValueError('v3 forward/VJP/helper runtime witness mismatch')
        sites=e.get('sites')
        if not isinstance(sites,list) or not sites:raise ValueError('missing v3 layer runtime observations')
        # Original site inventory is observed independently before the forward.
        expected={(g,tuple(u),s) for g,u,s in e.get('inventory',[])}
        original={(x['graph'],tuple(x['unit']),x['site']) for x in sites if x['phase']=='original'}
        if not expected or original!=expected:raise ValueError('v3 original site inventory mismatch')
        roles={x['backward'] for x in sites if x['phase']=='recompute'}
        if roles!=({'ordinary'} if rank==0 else {'joint_r_kl','rv','ordinary'}):
            raise ValueError('v3 checkpoint replay role witness mismatch')


def validate_v3_rank_records(records,*,commit):
    """Fail closed on missing ranks, events, exact state comparisons and numeric evidence."""
    if (not re.fullmatch('[a-f0-9]{40}',commit) or len(records)!=4
            or {r.get('global_rank') for r in records}!={0,1,2,3}):
        raise ValueError('v3 acceptance requires exactly four ranks')
    first=records[0];packages=first.get('packages');roots=first.get('roots');contract=first.get('contract')
    if not isinstance(packages,dict) or set(packages)!=set(PACKAGES) or any(not isinstance(v,str) or not v for v in packages.values()):
        raise ValueError('v3 package identity missing')
    if not isinstance(roots,dict) or set(roots)!={'local','remote','host'} or any(not Path(x).is_absolute() or '..' in Path(x).parts for x in roots.values()):
        raise ValueError('v3 root identity missing')
    validate_v3_resume(contract,contract)
    profile=contract['profile']
    if any(profile.get(k)!=v for k,v in V3_FRESH_PROFILE.items()):raise ValueError('v3 resolved profile mismatch')
    if (profile.get('world_size'),profile.get('effective_global_batch'),profile.get('zero_stage'),profile.get('bf16'))!=(4,16,1,True):
        raise ValueError('v3 topology/profile mismatch')
    schedule=contract['schedule']
    if (schedule['microsteps_per_rank'],schedule['accumulation_steps'],schedule['planned_steps'],schedule['remainder_microsteps'])!=(6,4,2,2):
        raise ValueError('v3 acceptance schedule mismatch')
    hosts={0:set(),1:set()}
    for r in records:
        rank=r['global_rank'];machine=rank//2
        if (r.get('schema_version')!=1 or r.get('kind')!=KIND or r.get('commit')!=commit
                or r.get('packages')!=packages or r.get('roots')!=roots or r.get('world_size')!=4
                or r.get('zero_stage')!=1 or r.get('machine_rank')!=machine or r.get('local_rank')!=rank%2
                or r.get('repository')!=roots['remote' if machine==0 else 'local']):
            raise ValueError('v3 runtime/root identity mismatch')
        if not isinstance(r.get('hostname'),str) or not r['hostname']:raise ValueError('missing v3 hostname')
        hosts[machine].add(r['hostname'])
        validate_v3_resume(r.get('contract'),contract)
        for name,count,steps in (('baseline',6,[4,6]),('interrupted',4,[4]),('resumed',2,[6])):
            trajectory=r.get(name,{})
            _events(trajectory.get('events'),rank,count)
            boundaries=trajectory.get('boundaries',[])
            if [b.get('completed_microsteps') for b in boundaries]!=steps:raise ValueError('v3 completed boundary mismatch')
            for b in boundaries:
                if b.get('applied') is not True or b.get('global_blocks')!=(24 if b['completed_microsteps']==4 else 12):
                    raise ValueError('v3 applied step/global block count mismatch')
                for k,limit in (('master_error_ratio',1.),('sync_residual',0.),('moment_residual',0.)):_finite(b,k,limit)
                if type(b.get('movement')) not in (float,int) or not math.isfinite(b['movement']) or b['movement']<=0:
                    raise ValueError('v3 master displacement not witnessed')
        baseline=r['baseline'];interrupted=r['interrupted'];resumed=r['resumed']
        if ([e['occurrence'] for e in baseline['events']]!=[e['occurrence'] for e in interrupted['events']+resumed['events']]
                or baseline['state']!=resumed['state'] or interrupted['state']!=resumed.get('restored_state')):
            raise ValueError('v3 exact occurrence/resume state mismatch')
        for state in (baseline['state'],interrupted['state'],resumed['state'],resumed['restored_state']):
            if set(state)!={'masters','optimizer','scheduler','rng','teacher'} or any(not isinstance(v,str) or not re.fullmatch('[a-f0-9]{64}',v) for v in state.values()):
                raise ValueError('v3 state hashes missing')
        if r.get('teacher_before')!=baseline['state']['teacher'] or r['teacher_before']!=resumed['state']['teacher']:
            raise ValueError('v3 frozen teacher changed')
        skip=r.get('skip',{})
        if skip.get('cause')!='injected_overflow_decision' or skip.get('overflow_calls')!=1 or skip.get('applied') is not False or skip.get('gain_steps_delta')!=0:
            raise ValueError('v3 real-engine skip witness missing')
        for key in ('master_residual','moment_residual','sync_residual'):_finite(skip,key,0.)
        faults=r.get('faults',{})
        if (faults.get('cause')!='injected' or faults.get('nonfinite_checkpoint_blocked') is not True
                or faults.get('post_apply_checkpoint_blocked') is not True):raise ValueError('v3 failed boundary checkpoint guard missing')
    if any(len(v)!=1 for v in hosts.values()) or hosts[0]==hosts[1]:raise ValueError('v3 requires two distinct node witnesses')


def _snapshot(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_snapshot(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(_snapshot(v) for v in value)
    return value


def state_snapshot(trainer):
    from deepspeed.utils import safe_get_full_fp32_param
    state=dict(masters=[safe_get_full_fp32_param(p).detach().cpu().clone() for p in trainer.v3_gain_controller.parameters],
        optimizer=_snapshot(trainer.model_wrapped.optimizer.optimizer.state_dict()),
        scheduler=_snapshot(trainer.lr_scheduler.state_dict()),
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()),
        teacher=trainer.ema_teacher.state_dict())
    return _snapshot(state)


def state_hashes(trainer):return {k:_digest(v) for k,v in state_snapshot(trainer).items()}


def _ratio(got,want):
    torch.testing.assert_close(got,want.to(got),atol=ATOL,rtol=RTOL)
    return float(((got-want.to(got)).abs()/(ATOL+RTOL*want.to(got).abs())).max())


class AcceptanceTrainer(CustomTrainerStage1):
    """Observers around the product loss, adapter and actual AdamW step."""
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.events=[];self.boundaries=[];self.restored_state=None;self._adam_ratio=0.
        self._offset=0;self.last_batch=None

    def _install_deepspeed_step_adapter(self):
        super()._install_deepspeed_step_adapter()
        from tests.stage1.vision_gain_zero_probe import adamw_reference
        adam=self.model_wrapped.optimizer.optimizer
        original=adam.step
        def observed(*a,**kw):
            expectations=[]
            for group in adam.param_groups:
                for p in group['params']:
                    if p.grad is None:continue
                    state=adam.state[p]
                    want=adamw_reference(p.detach().float().clone(),p.grad.detach().float().clone(),
                        state.get('exp_avg',torch.zeros_like(p)),state.get('exp_avg_sq',torch.zeros_like(p)),int(state.get('step',0)),group)
                    expectations.append((p,want))
            result=original(*a,**kw)
            for p,(want,first,second) in expectations:
                self._adam_ratio=max(self._adam_ratio,_ratio(p,want),_ratio(adam.state[p]['exp_avg'],first),_ratio(adam.state[p]['exp_avg_sq'],second))
            return result
        adam.step=observed
        self._offset=4 if self.state.global_step==1 else 0

    def training_step(self,model,inputs,num_items_in_batch=None):
        if not self.events and self._offset:
            self.restored_state=state_hashes(self)
        from stage1.equivalence import SampleStableStage1Dropout
        qwen=resolve_qwen2_5_vl_model(self.model)
        metadata=inputs['stage1_batch_metadata']
        graphs=('F','gain_clean','packed_replay') if metadata.v3.blocks else ('F',)
        inventory=[(g,('decoder',i),m.site_name) for g in graphs for i,layer in enumerate(qwen.model.layers)
                   for m in layer.modules() if isinstance(m,SampleStableStage1Dropout)]
        self.last_batch=inputs
        with observe_microstep(self) as event:
            event.update(occurrence=metadata.samples[0].source_index,inventory=inventory)
            self.events.append(event)
            return super().training_step(model,inputs,num_items_in_batch)

    def _finish_optimizer_step(self,model=None,*,successful):
        from deepspeed.utils import safe_get_full_fp32_param
        gain=self.v3_gain_controller
        masters=[safe_get_full_fp32_param(p).detach().clone() for p in gain.parameters]
        directions=[x.clone() for x in gain.direction]
        moments=_digest(model.optimizer.optimizer.state_dict())
        blocks=gain.global_block_count
        super()._finish_optimizer_step(model,successful=successful)
        residual=self._adam_ratio;movement=0.;sync=0.
        for p,before,direction in zip(gain.parameters,masters,directions):
            after=safe_get_full_fp32_param(p)
            want=before+gain.config.gain_alpha*direction if successful and blocks else before
            residual=max(residual,_ratio(after,want))
            movement=max(movement,float((after-before).abs().max()))
            assert torch.equal(p.detach(),after.to(p.dtype))
            sync=max(sync,float((p.detach()-after.to(p.dtype)).abs().max()))
        assert moments==_digest(model.optimizer.optimizer.state_dict())
        self.boundaries.append(dict(completed_microsteps=self._offset+len(self.events),applied=successful,
            global_blocks=blocks,master_error_ratio=residual,movement=movement,sync_residual=sync,moment_residual=0.))
        self._adam_ratio=0.


class StopAtOne(TrainerCallback):
    def on_save(self,args,state,control,**kw):
        if state.global_step==1:control.should_training_stop=True
        return control


def acceptance_dependencies():
    from tests.stage1.vision_gain_zero_probe import probe_dependencies,ProbeCollator
    from stage1.v3_metadata import V3Block,V3BatchMetadata
    deps=probe_dependencies()
    class Collator(ProbeCollator):
        def __call__(self,rows):
            batch=super().__call__(rows)
            ids=batch['input_ids'];ids=torch.cat((ids[:,:5],torch.ones(1,12,dtype=ids.dtype),ids[:,5:]),dim=1)
            batch.update(input_ids=ids,labels=ids.clone(),attention_mask=torch.ones_like(ids))
            old=batch['stage1_batch_metadata'];sample=old.samples[0]
            spans=tuple(replace(s,start=s.start+12,end=s.end+12,latent_positions=tuple(t+12 for t in s.latent_positions)) for s in sample.latent_spans)
            sample=replace(sample,latent_spans=spans,unpadded_length=ids.shape[1])
            rank=dist.get_rank()
            blocks=tuple(V3Block(0,i,s.start,s.end,(s.end+1,s.end+2),(2,3),tuple(range(4,17))) for i,s in enumerate(spans[:rank]))
            batch['stage1_batch_metadata']=replace(old,samples=(sample,),sequence_length=ids.shape[1],v3=V3BatchMetadata(blocks))
            return batch
    return replace(deps,trainer_factory=AcceptanceTrainer,
        collator_factory=lambda processor,token_ids,latent_size,*,v3_enabled:Collator(token_ids))


def _training_args(args,output,fixture,resume=False):
    from utils_deepseed import parse_args
    argv=['--stage','stage1','--stage1_variant','v3','--save_model_path',str(output),
        '--data_path',str(fixture/'TRAIN.jsonl'),'--image_root',str(fixture/'images'),
        '--save_strategy','steps','--save_steps','1','--deepspeed_config',str(args.deepspeed)]
    if resume:argv+=['--resume_from_checkpoint_path',str(output/'checkpoint-1')]
    return parse_args(argv)


def _train(args,output,fixture,*,resume=False,stop=False):
    from stage1.composition import build_stage1_training_components
    from deepspeed_runtime import load_deepspeed_runtime_profile
    unset_hf_deepspeed_config();set_seed(42)
    components=build_stage1_training_components(_training_args(args,output,fixture,resume),
        dependencies=acceptance_dependencies(),deepspeed_runtime_profile=load_deepspeed_runtime_profile(args.deepspeed,world_size=4))
    trainer=components.trainer
    before=_digest(trainer.ema_teacher.state_dict())
    if stop:trainer.add_callback(StopAtOne())
    trainer.train(resume_from_checkpoint=str(output/'checkpoint-1') if resume else None)
    snapshot=state_snapshot(trainer)
    return trainer,trajectory_record(trainer,snapshot,before,resume=resume),snapshot


def trajectory_record(trainer,snapshot,before,*,resume=False):
    record=dict(events=trainer.events,boundaries=trainer.boundaries,state={k:_digest(v) for k,v in snapshot.items()},teacher_before=before)
    if resume:record['restored_state']=trainer.restored_state
    return _snapshot(record)


def _blocked_save(trainer,path):
    try:trainer.save_model(str(path))
    except RuntimeError as error:
        if 'checkpoint' not in str(error):raise
        if path.exists():raise AssertionError('failed boundary published an artifact')
        return True
    raise AssertionError('failed boundary allowed checkpoint')


def fault_witness(trainer,root):
    """Explicitly injected decisions; successful comparison trajectory already saved."""
    from unittest.mock import patch
    from stage1.v3_update import V3GainController
    from stage1.v3_gradients import V3Gradients
    import stage1.v3_update as update
    engine=trainer.model_wrapped;zero=engine.optimizer;gain=trainer.v3_gain_controller
    before=state_snapshot(trainer);steps=gain.successful_steps;calls=[]
    def overflow(*a,**kw):calls.append(1);zero.overflow=True
    engine.set_gradient_accumulation_boundary(True)
    # Actual product backward -> engine.step -> optimizer.step -> overflow branch.
    with patch.object(zero,'check_grad_overflow',True),patch.object(zero,'check_overflow',overflow):
        trainer.training_step(engine,trainer.last_batch)
    after=state_snapshot(trainer)
    assert not engine.was_step_applied() and len(calls)==1 and gain.successful_steps==steps
    for key in ('masters','optimizer'):assert _digest(before[key])==_digest(after[key])
    assert gain.state=='IDLE'
    skip=dict(cause='injected_overflow_decision',overflow_calls=len(calls),applied=engine.was_step_applied(),
              gain_steps_delta=gain.successful_steps-steps,master_residual=0.,moment_residual=0.,sync_residual=0.)
    zero.overflow=False
    # Nonfinite decision aborts before ordinary optimizer execution on every rank.
    zeros=tuple(torch.zeros_like(p,dtype=torch.float32) for p in gain.parameters)
    invalid=tuple(x.clone() for x in zeros)
    if dist.get_rank()==0:invalid[0].reshape(-1)[0]=float('nan')
    gain.accumulate(V3Gradients(invalid,zeros,0,torch.tensor(0.),torch.tensor(0.)))
    try:engine.step()
    except RuntimeError as error:
        if 'nonfinite' not in str(error):raise
    else:raise AssertionError('nonfinite auxiliary accepted')
    nonfinite=_blocked_save(trainer,root/'forbidden-nonfinite')
    # One extra failed attempt: ordinary Adam is real, but injected Gain write
    # fails and the overall update never reaches a successful boundary or save.
    trainer.v3_gain_controller=V3GainController(gain.parameters,gain.config)
    with patch.object(update,'safe_set_full_fp32_param',side_effect=RuntimeError('injected post-Adam apply failure')):
        try:trainer.training_step(engine,trainer.last_batch)
        except RuntimeError as error:
            if 'injected post-Adam' not in str(error):raise
        else:raise AssertionError('post-apply injection was not reached')
    post=_blocked_save(trainer,root/'forbidden-post-apply')
    return skip,dict(cause='injected',nonfinite_checkpoint_blocked=nonfinite,
                     post_apply_checkpoint_blocked=post,failed_post_adam_attempts=1)


def run_acceptance(args:argparse.Namespace)->dict[str,object]:
    if int(os.environ.get('WORLD_SIZE','0'))!=4:
        raise ValueError('v3 acceptance requires coordinator-owned two-node/four-rank execution')
    if os.environ.get('ILVR_REPOSITORY_COMMIT')!=args.commit:raise ValueError('v3 owned commit mismatch')
    from tests.stage1.zero_compat_diagnostic import _initialize_distributed,_publish_witness
    from artifact_paths import resolve_run_artifact_paths
    _initialize_distributed()
    if dist.get_world_size()!=4 or int(os.environ['LOCAL_RANK'])!=dist.get_rank()%2:
        raise ValueError('v3 rank topology mismatch')
    paths=resolve_run_artifact_paths(args.run_id)
    fixture=ROOT/'data'/args.run_id
    if int(os.environ['LOCAL_RANK'])==0 and args.phase=='initial':
        (fixture/'images').mkdir(parents=True,exist_ok=False);(fixture/'TRAIN.jsonl').write_text('{}\n')
    dist.barrier()
    state_dir=paths.checkpoint_dir/'acceptance-state';state_dir.mkdir(parents=True,exist_ok=True)
    state_path=state_dir/f'rank-{dist.get_rank()}.pt'
    if args.phase=='initial':
        baseline,base_record,base_snapshot=_train(args,paths.checkpoint_dir/'baseline',fixture)
        assert baseline.state.global_step==2
        baseline.accelerator.free_memory();del baseline;gc.collect();torch.cuda.empty_cache()
        # The next Trainer constructs its own DeepSpeedPlugin. Accelerate 1.6
        # rejects a second explicit plugin until its process-local state is reset.
        # Keep PartialState and the live four-rank process group intact.
        from accelerate.state import AcceleratorState
        AcceleratorState._reset_state(reset_partial_state=False)
        interrupted,first_record,first_snapshot=_train(args,paths.checkpoint_dir,fixture,stop=True)
        assert interrupted.state.global_step==1
        record=dict(schema_version=1,kind=KIND,commit=args.commit,packages={n:version(n) for n in PACKAGES},
            global_rank=dist.get_rank(),local_rank=int(os.environ['LOCAL_RANK']),machine_rank=dist.get_rank()//2,
            hostname=socket.gethostname(),world_size=4,zero_stage=1,repository=str(ROOT),
            roots=dict(local=args.local_repo_root,remote=args.remote_repo_root,host=args.remote_host_repo_root),
            contract=interrupted.v3_contract,baseline=base_record,interrupted=first_record,teacher_before=base_record['teacher_before'])
        torch.save(dict(record=record,baseline=base_snapshot,interrupted=first_snapshot),state_path)
    else:
        saved=torch.load(state_path,map_location='cpu',weights_only=False)
        record=saved['record']
        resumed,resume_record,resume_snapshot=_train(args,paths.checkpoint_dir,fixture,resume=True)
        assert resumed.state.global_step==2
        assert resume_record['state']==record['baseline']['state']
        assert resume_record['restored_state']==record['interrupted']['state']
        record['resumed']=resume_record
        record['skip'],record['faults']=fault_witness(resumed,paths.checkpoint_dir)
    records=[None]*4;dist.all_gather_object(records,record)
    if args.phase=='resume':validate_v3_rank_records(records,commit=args.commit)
    witness=dict(schema_version=1,kind=KIND,commit=args.commit,phase=args.phase,ranks=records)
    _publish_witness(paths,witness,filename=f'{args.phase}-witness.json')
    return witness


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--phase',choices=('initial','resume'),required=True)
    parser.add_argument('--zero-stage',type=int,choices=(1,),required=True)
    parser.add_argument('--deepspeed',type=Path,required=True)
    for field in ('run-id','commit','local-repo-root','remote-repo-root','remote-host-repo-root'):
        parser.add_argument('--'+field,required=True)
    run_acceptance(parser.parse_args(argv))
    return 0


if __name__=='__main__':raise SystemExit(main())
