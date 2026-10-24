#!/usr/bin/env python
"""Bounded, coordinator-owned weights-only Stage1-v5 timing replay.

CPU intervals measure Python dispatch/wall time. CUDA events measure elapsed
stream timeline intervals, including idle time/dependencies, not kernel sums.
All nested spans are inclusive; only the explicit partitions may be added.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from transformers import TrainerCallback

REPOSITORY = Path(__file__).resolve().parents[1]


class CudaEventClock:
    def record(self):
        import torch
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def drain(self):
        import torch
        torch.cuda.synchronize()

    def elapsed(self, start, end):
        return start.elapsed_time(end)


class ForwardTimingObserver:
    """Scoped wrappers around actual original calls and backward recomputation.

    No tensor, return value, RNG, model mode or execution context is replaced.
    Events are recorded on the current stream and drained once after training.
    """
    def __init__(self, trainer, *, rank, event_clock=None):
        self.trainer, self.rank = trainer, rank
        self.clock = event_clock if event_clock is not None else CudaEventClock()
        self.records, self.windows, self._restore = [], [], []
        self.current = None
        self.stack = []
        self.helper_active = self.in_backward = False
        self.failed = False
        self.window = None
        self._engine = None
        self.origin = time.perf_counter_ns()
        self.cuda_origin = self.clock.record()
        self._finished = False

    def _now(self):
        return (time.perf_counter_ns()-self.origin)/1e6

    def _patch(self, obj, name, replacement):
        existed = name in vars(obj)
        original = vars(obj).get(name)
        self._restore.append((obj,name,existed,original))
        setattr(obj,name,replacement)

    def _wrap(self, obj, name, phase, *, original=False):
        real = getattr(obj,name)
        def wrapped(*args,**kwargs):
            if self.current is None:
                return real(*args,**kwargs)
            label = phase() if callable(phase) else phase
            if label is None:
                return real(*args,**kwargs)
            with self.span(label):
                if original:
                    self.current['original_order'].append(label)
                return real(*args,**kwargs)
        self._patch(obj,name,wrapped)

    @contextmanager
    def span(self, name):
        span = dict(id=len(self.current['spans']),name=name,
                    parent=self.stack[-1] if self.stack else None,
                    cpu_start_ms=self._now(),_start=self.clock.record())
        self.current['spans'].append(span)
        self.stack.append(span['id'])
        try:
            yield span
        finally:
            span['_end']=self.clock.record()
            span['cpu_end_ms']=self._now()
            span['cpu_ms']=span['cpu_end_ms']-span['cpu_start_ms']
            self.stack.pop()

    def __enter__(self):
        from peft_common import resolve_qwen2_5_vl_model
        from transformers.models.qwen2_5_vl import stage1_v5_replay as replay
        trainer = self.trainer
        qwen = resolve_qwen2_5_vl_model(trainer.model)
        teacher = resolve_qwen2_5_vl_model(trainer.ema_teacher.model)
        try:
            real_step = trainer.training_step
            def step(model, inputs, *args, **kwargs):
                if self.failed:
                    raise ValueError('timing attribution failed; collection stopped')
                if self.current is not None:
                    raise ValueError('nested timing step')
                metadata=inputs['stage1_v5_metadata']
                samples=inputs['stage1_batch_metadata'].samples
                self.current=dict(rank=self.rank,microbatch=len(self.records),
                    window=self.window['index'] if self.window is not None else trainer.state.global_step,
                    warmup=(self.window['index'] if self.window is not None else trainer.state.global_step)==0,
                    original_order=[],spans=[],geometry=dict(
                        source_indices=[s.source_index for s in samples],
                        source_ids=[s.source_id for s in samples],
                        unpadded_lengths=[s.unpadded_length for s in samples],
                        input_shape=list(inputs['input_ids'].shape),
                        query_count=len(metadata.query_keys),original_blocks=metadata.original_step_count,
                        active_blocks=sum(bool(b.target_positions) for b in metadata.blocks),
                        original_pixel_shape=list(inputs.get('pixel_values',()).shape) if 'pixel_values' in inputs else None,
                        helper_pixel_shape=list(inputs.get('pixel_values_latent',()).shape) if 'pixel_values_latent' in inputs else None,
                        grid_shapes={k:list(inputs[k].shape) for k in ('image_grid_thw','image_grid_thw_latent') if k in inputs}))
                try:
                    # Trainer prepares DeepSpeed after observer attachment. Wrap
                    # its existing adapter only once it exists, without replacing
                    # the engine/optimizer lifecycle.
                    wrapper=getattr(trainer.accelerator,'deepspeed_engine_wrapped',None)
                    engine=getattr(wrapper,'engine',None)
                    if engine is not None and self._engine is None:
                        self._engine=engine
                        self._wrap(engine,'step','deepspeed_step')
                    with self.span('training_step'):
                        loss=real_step(model,inputs,*args,**kwargs)
                    self.current['_loss']=loss.detach()
                    validate_microbatch(self.current, cuda=False)
                    self.records.append(self.current)
                    return loss
                except BaseException:
                    self.failed=True
                    raise
                finally:
                    self.current=None
            self._patch(trainer,'training_step',step)
            real_build=trainer.target_builder.build
            def build(*args,**kwargs):
                if self.current is None:
                    return real_build(*args,**kwargs)
                with self.span('helper_targets'):
                    self.helper_active=True
                    try: return real_build(*args,**kwargs)
                    finally: self.helper_active=False
            self._patch(trainer.target_builder,'build',build)
            self._wrap(trainer,'compute_loss','compute_loss')
            self._wrap(trainer.model,'forward','clean',original=True)
            self._wrap(trainer.ema_teacher.model,'forward',
                lambda:'helper' if self.helper_active else 'reference',original=True)
            self._wrap(teacher.visual,'forward',lambda:'helper_vision' if self.helper_active else 'reference_vision')
            self._wrap(qwen.visual,'forward','clean_vision')
            real_replay=replay.run_v5_packed_replay
            def packed(*args,**kwargs):
                if self.current is None:
                    return real_replay(*args,**kwargs)
                with self.span('packed'):
                    self.current['original_order'].append('packed')
                    self.current['geometry']['packed_branches']=[name for name,key in
                        (('voff','need_voff'),('textoff','need_textoff')) if kwargs[key]]
                    return real_replay(*args,**kwargs)
            self._patch(replay,'run_v5_packed_replay',packed)
            real_decoder=qwen.model.forward
            def decoder(*args,**kwargs):
                if self.current is None:
                    return real_decoder(*args,**kwargs)
                packed=kwargs.get('stage2_prefix_key_values') is not None
                phase='recompute_decoder' if self.in_backward else 'packed_decoder' if packed else 'clean_decoder'
                with self.span(phase):
                    if packed and not self.in_backward:
                        self.current['geometry'].update(
                            packed_input_shape=list(kwargs['inputs_embeds'].shape),
                            packed_bias_shape=list(kwargs['attention_mask'].shape),
                            packed_prefix_shape=list(kwargs['stage2_prefix_key_values'][0][0].shape))
                    return real_decoder(*args,**kwargs)
            self._patch(qwen.model,'forward',decoder)
            for layer in qwen.model.layers:
                self._wrap(layer,'forward',lambda:'recompute_layer' if self.in_backward else None)
            real_backward=trainer.accelerator.backward
            def backward(*args,**kwargs):
                if self.current is None:
                    return real_backward(*args,**kwargs)
                with self.span('backward'):
                    self.in_backward=True
                    try: return real_backward(*args,**kwargs)
                    finally: self.in_backward=False
            self._patch(trainer.accelerator,'backward',backward)
        except BaseException:
            self.__exit__(None,None,None)
            raise
        return self

    def __exit__(self, *exc):
        for obj,name,existed,original in reversed(self._restore):
            if existed: setattr(obj,name,original)
            else: delattr(obj,name)
        self._restore.clear()

    def begin_window(self):
        if self.window is not None or len(self.windows)>=4:
            raise ValueError('invalid replay window boundary')
        self.window=dict(index=len(self.windows),cpu_start_ms=self._now(),
            _start=self.clock.record(),microbatch_start=len(self.records),
            _updates=self.trainer._successful_optimizer_steps)

    def end_window(self):
        window=self.window
        window['_end']=self.clock.record()
        window['cpu_end_ms']=self._now()
        window['cpu_ms']=window['cpu_end_ms']-window['cpu_start_ms']
        window['successful_updates']=self.trainer._successful_optimizer_steps-window.pop('_updates')
        window['microbatch_end']=len(self.records)
        if window['successful_updates']!=1:
            raise ValueError('replay window requires one successful optimizer update')
        self.windows.append(window)
        self.window=None

    def finish(self):
        if self._finished: return self.records
        if self._restore or self.current is not None or self.window is not None or self.failed:
            raise ValueError('cannot finalize active or failed timing observation')
        self.clock.drain()
        def resolve(span):
            start,end=span.pop('_start'),span.pop('_end')
            span['cuda_ms']=self.clock.elapsed(start,end)
            span['cuda_start_ms']=self.clock.elapsed(self.cuda_origin,start)
            span['cuda_end_ms']=self.clock.elapsed(self.cuda_origin,end)
        for row in self.records:
            packed=next((s for s in row['spans'] if s['name']=='packed'),None)
            if packed is not None:
                decoder=next(s for s in row['spans'] if s['name']=='packed_decoder')
                row['packed_partition'].update(
                    setup_cuda_ms=self.clock.elapsed(packed['_start'],decoder['_start']),
                    projection_and_remainder_cuda_ms=self.clock.elapsed(decoder['_end'],packed['_end']))
            for span in row['spans']: resolve(span)
            row['finite_loss']=bool(row.pop('_loss').isfinite().all().item())
            if not row['finite_loss']: raise ValueError('nonfinite replay loss')
            validate_microbatch(row)
        for window in self.windows:
            resolve(window)
            rows=self.records[window['microbatch_start']:window['microbatch_end']]
            if not rows or any(r['window']!=window['index'] for r in rows):
                raise ValueError('replay window/microbatch mismatch')
            window['training_step_cpu_ms']=sum(r['spans'][0]['cpu_ms'] for r in rows)
            window['window_remainder_cpu_ms']=window['cpu_ms']-window['training_step_cpu_ms']
            window['training_step_cuda_ms']=sum(r['spans'][0]['cuda_ms'] for r in rows)
            window['window_remainder_cuda_ms']=window['cuda_ms']-window['training_step_cuda_ms']
            if window['window_remainder_cpu_ms']<0 or window['window_remainder_cuda_ms']<-.1:
                raise ValueError('negative window accounting')
        self._finished=True
        return self.records


def validate_microbatch(row, *, cuda=True):
    """First nail: conditional originals, backward boundary, disjoint coverage."""
    spans=row['spans']
    by_name=defaultdict(list)
    for s in spans:
        by_name[s['name']].append(s)
        for key in ('cpu_start_ms','cpu_end_ms','cpu_ms')+(('cuda_ms','cuda_start_ms','cuda_end_ms') if cuda else ()):
            if not math.isfinite(s[key]) or s[key]<0:
                raise ValueError('timing intervals must be finite and nonnegative')
        if cuda and not math.isclose(s['cuda_end_ms']-s['cuda_start_ms'],s['cuda_ms'],abs_tol=.1):
            raise ValueError('inconsistent CUDA interval')
        if not math.isclose(s['cpu_end_ms']-s['cpu_start_ms'],s['cpu_ms'],abs_tol=1e-6):
            raise ValueError('inconsistent timing interval')
        if s['parent'] is not None:
            parent=spans[s['parent']]
            if not parent['cpu_start_ms']<=s['cpu_start_ms']<=s['cpu_end_ms']<=parent['cpu_end_ms']:
                raise ValueError('timing nesting escaped parent')
            if cuda and not parent['cuda_start_ms']-.1<=s['cuda_start_ms']<=s['cuda_end_ms']<=parent['cuda_end_ms']+.1:
                raise ValueError('CUDA interval escaped parent')
    expected=['helper','clean']+(['reference','packed'] if row['geometry']['query_count'] else [])
    if row['original_order']!=expected or any(len(by_name[n])!=expected.count(n) for n in ('helper','clean','reference','packed')):
        raise ValueError('conditional original call order/count mismatch')
    if len(by_name['training_step'])!=1 or len(by_name['backward'])!=1 or len(by_name['compute_loss'])!=1:
        raise ValueError('missing step/loss/backward boundary')
    step,backward,loss=(by_name[n][0] for n in ('training_step','backward','compute_loss'))
    if loss['cpu_end_ms']>backward['cpu_start_ms']:
        raise ValueError('backward overlaps original loss/forward')
    outer=[by_name[n][0] for n in expected]+[backward]
    if any(left['cpu_end_ms']>right['cpu_start_ms'] for left,right in zip(outer,outer[1:])):
        raise ValueError('original forward/backward spans overlap')
    total=sum(s['cpu_ms'] for s in outer)
    remainder=step['cpu_ms']-total
    if remainder<0: raise ValueError('outer timing exceeds enclosing step')
    row['accounting']=dict(outer_forward_cpu_ms=sum(s['cpu_ms'] for s in outer[:-1]),
        backward_cpu_ms=backward['cpu_ms'],step_remainder_cpu_ms=remainder,
        enclosing_step_cpu_ms=step['cpu_ms'])
    if cuda:
        cuda_remainder=step['cuda_ms']-sum(s['cuda_ms'] for s in outer)
        if cuda_remainder<-.1:
            raise ValueError('CUDA outer timing exceeds enclosing step')
        row['accounting'].update(outer_forward_cuda_ms=sum(s['cuda_ms'] for s in outer[:-1]),
            backward_cuda_ms=backward['cuda_ms'],step_remainder_cuda_ms=cuda_remainder,
            enclosing_step_cuda_ms=step['cuda_ms'])
        for s in outer:
            if not step['cuda_start_ms']-.1<=s['cuda_start_ms']<=s['cuda_end_ms']<=step['cuda_end_ms']+.1:
                raise ValueError('CUDA outer timing escaped step')
        if any(left['cuda_end_ms']>right['cuda_start_ms']+.1 for left,right in zip(outer,outer[1:])):
            raise ValueError('CUDA original/backward intervals overlap')
    if 'packed' in expected:
        if len(by_name['packed_decoder'])!=1:
            raise ValueError('packed requires one combined decoder invocation')
        packed,decoder=by_name['packed'][0],by_name['packed_decoder'][0]
        if decoder['parent']!=packed['id']: raise ValueError('packed decoder not nested')
        row.setdefault('packed_partition',{}).update(setup_cpu_ms=decoder['cpu_start_ms']-packed['cpu_start_ms'],
            decoder_cpu_ms=decoder['cpu_ms'],projection_and_remainder_cpu_ms=packed['cpu_end_ms']-decoder['cpu_end_ms'],
            cpu_ms=packed['cpu_ms'])
        if cuda:
            partition=row['packed_partition']
            if any(not math.isfinite(partition[k]) or partition[k]<0 for k in ('setup_cuda_ms','projection_and_remainder_cuda_ms')):
                raise ValueError('nonfinite packed partition')
            if not math.isclose(partition['setup_cuda_ms']+decoder['cuda_ms']+
                    partition['projection_and_remainder_cuda_ms'],packed['cuda_ms'],abs_tol=.1):
                raise ValueError('CUDA packed partition does not reconcile')
        shape=row['geometry']['packed_input_shape']
        if shape[0]!=row['geometry']['active_blocks']*len(row['geometry']['packed_branches']):
            raise ValueError('packed branch geometry mismatch')
    for s in by_name['recompute_layer']+by_name['recompute_decoder']:
        if not backward['cpu_start_ms']<=s['cpu_start_ms']<=s['cpu_end_ms']<=backward['cpu_end_ms']:
            raise ValueError('recomputation escaped backward')


class BoundedReplayCallback(TrainerCallback):
    def __init__(self, observer): self.observer=observer
    def on_step_begin(self,args,state,control,**kwargs):
        self.observer.begin_window()
    def on_step_end(self,args,state,control,**kwargs):
        self.observer.end_window()
        if len(self.observer.windows)==4: control.should_training_stop=True
        return control


def rank_record(observer, rows, *, run_id, identity):
    return dict(kind='stage1-v5-forward-timing',rank=observer.rank,world_size=4,
        run_id=run_id,status='pass',identity=identity,window_count=len(observer.windows),
        windows=observer.windows,microbatches=rows,weights_only=True,
        optimizer_and_global_step_restored=False,wrappers_removed=not observer._restore)


def _statistics(values):
    return dict(median=statistics.median(values),minimum=min(values),maximum=max(values))


def summarize_records(records, *, run_id):
    if len(records)!=4 or {r['rank'] for r in records}!={0,1,2,3}:
        raise ValueError('successful timing requires four ranks')
    identity=records[0]['identity']
    measured=[]
    for record in records:
        if (record['run_id']!=run_id or record['status']!='pass' or record['world_size']!=4
                or not record['wrappers_removed'] or record['identity']!=identity or record['window_count']!=4 or len(record['windows'])!=4):
            raise ValueError('incompatible or incomplete rank timing record')
        rows=record['microbatches']
        if len(rows)!=16 or [r['microbatch'] for r in rows]!=list(range(16)):
            raise ValueError('bounded replay requires sixteen microbatches per rank')
        for index,window in enumerate(record['windows']):
            if (window['index']!=index or window['successful_updates']!=1
                    or window['microbatch_start']!=4*index or window['microbatch_end']!=4*(index+1)):
                raise ValueError('incomplete optimizer window')
            for key in ('cpu_ms','cuda_ms','window_remainder_cpu_ms'):
                if not math.isfinite(window[key]) or window[key]<0: raise ValueError('nonfinite window timing')
        for i,row in enumerate(rows):
            if row['rank']!=record['rank'] or row['window']!=i//4 or row['warmup']!=(i<4) or not row['finite_loss']:
                raise ValueError('rank/window/finite execution mismatch')
            validate_microbatch(row)
        measured.extend(rows[4:])
    def phases(rows):
        groups=defaultdict(list)
        for row in rows:
            for span in row['spans']: groups[span['name']].append(span)
        return {name:dict(count=len(spans),cpu_ms=_statistics([s['cpu_ms'] for s in spans]),
            cuda_ms=_statistics([s['cuda_ms'] for s in spans])) for name,spans in groups.items()}
    return dict(kind='stage1-v5-forward-timing',status='pass',run_id=run_id,ranks=[0,1,2,3],
        measured_microbatches=len(measured),warmup_microbatches_excluded=16,identity=identity,
        phases=phases(measured),per_rank={str(r['rank']):phases(r['microbatches'][4:]) for r in records},
        geometry=[dict(rank=r['rank'],microbatch=r['microbatch'],**r['geometry']) for r in measured],
        windows={str(r['rank']):r['windows'] for r in records},
        definitions=dict(cpu='perf_counter wall/dispatch intervals; asynchronous GPU work may outlive dispatch',
            cuda='current-stream CUDA-event elapsed timeline intervals; not kernel active-duration sums',
            coverage='spans inclusive with explicit parent IDs; add only accounting outer spans and remainder',
            packed='combined independent V-off+Text-off rows; setup includes bias/KV and Python preparation; remainder includes projection',
            backward='accelerator.backward inclusive of recomputation and any DeepSpeed engine step within that call',
            deepspeed_step='nested actual engine.step: clipping, optimizer, zeroing, scheduler and successful-step adapter',
            window_remainder='window outside training_step, including optimizer bookkeeping; excludes loader prefetch before on_step_begin'),
        limitations=['Weights-only replay: optimizer, scheduler and global step start fresh; not historical timing recovery.',
            'One warmup and three measurement windows only; replay starts at the initial seeded sampler positions, not source final-step positions.',
            'No convergence, accuracy or historical phase-percentage claim.',
            'CUDA events observe current-stream dependency timeline, not all-stream GPU utilization.',
            'Nested phases overlap their enclosing spans; packed branches are not independently timed.'])


def validate_source_contract(saved):
    from stage1.v5_config import Stage1V5Config
    from stage1.v5_resume import validate_v5_resume
    from stage1.query import resolve_query_config
    config=validate_v5_resume(saved,{})
    expected=Stage1V5Config(selection='clean',preference_entropy_top_percent=20,
        lambda_topk_entropy=.1,lambda_topk_mass=.05,preference_topk_k=50)
    if config!=expected or saved['coefficients']!=dict(ce_weight=.1,sim_weight=.1):
        raise ValueError('timing requires completed clean20 source objective')
    if any(saved['schedule'].get(k)!=v for k,v in dict(configured_max_steps=2160,num_train_epochs=10).items()):
        raise ValueError('timing must retain original 2160-update/10-epoch schedule')
    execution=saved['execution']
    required=dict(tuning_method='dora',zero_stage=1,ema_tau=1.,per_device_train_batch_size=1,
        gradient_accumulation_steps=4,bf16=True,seed=42,learning_rate=5e-5,warmup_steps=10,
        lora_r=32,lora_alpha=64,lora_dropout=.05,latent_size=8,attention_implementation='sdpa',
        vision_attention_implementation='flash_attention_2',gradient_checkpointing=True,textual_states='previous')
    if any(execution.get(k)!=v for k,v in required.items()):
        raise ValueError('source execution differs from approved timing profile')
    if resolve_query_config({},saved=execution.get('query_config')).mode!='legacy':
        raise ValueError('four-forward timing requires original legacy query path')


def replay_cli(args):
    """Composition restores weights/reference; Trainer.train receives no resume."""
    return ['--stage','stage1','--stage1_variant','v5',
        '--resume_from_checkpoint_path',str(args.source_checkpoint),'--model',str(args.model),
        '--data_path',str(args.data_path),'--image_root',str(args.image_root),
        '--save_model_path',str(REPOSITORY/'checkpoints'/args.run_id/'replay'),
        '--log_file',str(REPOSITORY/'logs'/args.run_id/'replay.log'),
        '--cache_dir',str(REPOSITORY/'data/cache'),
        '--deepspeed_config',str(args.deepspeed),'--save_strategy','no',
        '--ce_weight','0.1','--sim_weight','0.1','--lambda_preference','1',
        '--step_preference_beta','1','--step_preference_margin','0',
        '--lambda_topk_entropy','0.1','--lambda_topk_mass','0.05',
        '--preference_topk_k','50','--preference_topk_mode','independent',
        '--no-topk_include_gt','--stage1_v5_selection','clean',
        '--preference_entropy_top_percent','20','--stage1_v5_text_kl_weight','0.01',
        '--max_steps','2160','--epochs','10','--max_seq_length_train','32768',
        '--no-stage1_v5_debug']


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def compose_seeded_replay(parsed):
    from utils_deepseed import seed_everything
    from stage1.composition import build_stage1_training_components
    # Match src/main.py before model/reference construction and contract capture.
    seed_everything(parsed.seed)
    return build_stage1_training_components(parsed)


def replay(args):
    from scripts.verify_stage1_v5 import require_distributed_entry
    from src.artifact_paths import require_canonical_artifact_path
    from stage1.v5_resume import load_v5_contract
    import torch
    import torch.distributed as dist
    require_distributed_entry()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    if not dist.is_initialized(): dist.init_process_group('nccl')
    if dist.get_world_size()!=4: raise ValueError('timing requires four ranks')
    source=require_canonical_artifact_path(args.source_checkpoint,kind='checkpoint')
    saved=load_v5_contract(source)
    validate_source_contract(saved)
    source_state=json.loads((source/'trainer_state.json').read_text())
    if source_state['global_step']!=2160: raise ValueError('source must be completed checkpoint-2160')
    destination=require_canonical_artifact_path(REPOSITORY/'checkpoints'/args.run_id/'replay',kind='checkpoint')
    # Exclusive coordinator preflight also rejects run ID reuse on either node.
    if destination.exists(): raise ValueError('timing replay destination already exists')
    dist.barrier()  # Every rank checks freshness before composition publishes its reference.
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPOSITORY,text=True).strip()
    if commit!=os.environ['ILVR_REPOSITORY_COMMIT']:
        raise ValueError('runtime commit differs from owned command')
    subprocess.run(['git','diff','--exit-code','HEAD','--','scripts/profile_stage1_v5.py',
        'src/multinode.py','src/multinode_runtime.py','src/stage1','transformers/src/transformers'],
        cwd=REPOSITORY,check=True,capture_output=True)
    identity=dict(commit=commit,source_checkpoint=str(source.relative_to(REPOSITORY/'checkpoints')),
        source_files={name:_sha(source/name) for name in ('stage1_v5.json','stage1_v5_reference.pt',
            'stage1_peft_metadata.json','adapter_config.json','adapter_model.safetensors','ilvr_token_rows.safetensors')},
        source_reference=saved['reference'],data_sha256=_sha(args.data_path),
        code_sha256={name:_sha(REPOSITORY/name) for name in ('scripts/profile_stage1_v5.py',
            'src/multinode.py','src/multinode_runtime.py')})
    from utils_deepseed import parse_args
    from stage1.v5_reference import _state_identity
    from transformers.integrations.deepspeed import unset_hf_deepspeed_config
    unset_hf_deepspeed_config()
    parsed=parse_args(replay_cli(args))
    components=compose_seeded_replay(parsed)
    trainer=components.trainer
    before=_state_identity(trainer.ema_teacher,trainer.ema_teacher.state_dict())
    observer=ForwardTimingObserver(trainer,rank=dist.get_rank())
    callback=BoundedReplayCallback(observer)
    trainer.add_callback(callback)
    try:
        with observer:
            # Deliberately NOT components.resume_decision.checkpoint_path.
            # Existing v5 train() validates saved/current full contracts normally.
            output=trainer.train(resume_from_checkpoint=None)
    finally:
        trainer.remove_callback(callback)
    rows=observer.finish()
    after=_state_identity(trainer.ema_teacher,trainer.ema_teacher.state_dict())
    if (not trainer._v5_contract_ready or trainer.state.global_step!=4
            or not math.isfinite(output.training_loss) or before!=after
            or before!=trainer.v5_reference_identity.effective_state_sha256):
        raise ValueError('incomplete/nonfinite replay or changed immutable reference')
    identity['effective_contract']=trainer._effective_v5_contract(trainer._v5_contract)
    record=rank_record(observer,rows,run_id=args.run_id,identity=identity)
    record['reference_unchanged']=True
    record['runtime']=dict(host=os.uname().nodename,local_rank=int(os.environ['LOCAL_RANK']),
        machine_rank=0 if dist.get_rank()<2 else 1,device=torch.cuda.get_device_name(),
        training_loss=output.training_loss,global_step=trainer.state.global_step)
    records=[None]*4
    dist.all_gather_object(records,record)
    summary=summarize_records(records,run_id=args.run_id)
    from src.multinode_runtime import _publish_bytes_no_replace
    root=REPOSITORY/'results'/args.run_id
    root.mkdir(parents=True,exist_ok=True)
    _publish_bytes_no_replace(root/f'forward-timing-rank-{dist.get_rank()}.json',
        json.dumps(record,sort_keys=True,allow_nan=False).encode())
    if int(os.environ['LOCAL_RANK'])==0:
        _publish_bytes_no_replace(root/'forward-timing-summary.json',
            json.dumps(summary,sort_keys=True,indent=2,allow_nan=False).encode())
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--mode',choices=('replay','record-show'),required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--rank',type=int,choices=range(4))
    for name in ('source-checkpoint','model','data-path','image-root','deepspeed'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args(argv)
    from src.multinode import _RUN_ID_PATTERN
    if not _RUN_ID_PATTERN.fullmatch(args.run_id): parser.error('invalid run ID')
    if args.mode=='record-show':
        if args.rank is None: parser.error('rank is required')
        from src.multinode_runtime import _read_regular_file
        print(_read_regular_file(REPOSITORY/'results'/args.run_id/f'forward-timing-rank-{args.rank}.json',
            maximum_size=16*1024*1024).decode())
        return 0
    if any(getattr(args,name) is None for name in ('source_checkpoint','model','data_path','image_root','deepspeed')):
        parser.error('source-checkpoint, model, data-path, image-root and deepspeed required')
    return replay(args)


if __name__=='__main__':
    raise SystemExit(main())
