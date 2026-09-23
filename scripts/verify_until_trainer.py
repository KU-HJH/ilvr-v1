#!/usr/bin/env python
"""Frozen-external/aligned Trainer gate; launch only through the two-node coordinator.

Set ILVR_UNTIL_ALIGNMENT_GATE_SPEC to a JSON spec for the normal Stage2 launcher.
Run three external specs first, freeze their envelopes, then run the aligned spec.
CLI freeze/compare operate on gathered evidence and never launch training.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.verify_until_native import ExternalSource, external_modules, describe_batch, state_digest
from scripts import verify_voff_model_path as baseline
from stage2.until_alignment_probe import (
    NUMERICAL_PROTOCOL, AlignmentObserver, check_runtime, compare_snapshot,
    freeze_envelope, select_windows, tensor_destination, required_snapshots, expected_snapshots,
)


class TrainerExternalSource(ExternalSource):
    """Also retain the frozen parent Trainer; dependencies remain explicitly shared."""
    def __init__(self,root):
        super().__init__(root)
        for original,relative in (
            ('transformers.trainer','transformers/src/transformers/trainer'),
            ('transformers.integrations.flex_attention','transformers/src/transformers/integrations/flex_attention'),
        ):
            parent,leaf=original.rsplit('.',1)
            alias=parent+'.'+self.prefix+'_'+leaf
            self.mapping[original]=alias
            self.folders[alias]=self.root/relative

    def exec_module(self,module):
        import builtins
        from types import SimpleNamespace
        def supplied_import(name,globals=None,locals=None,fromlist=(),level=0):
            if level==0 and name=='transformers' and tuple(fromlist)==('Trainer',):
                parent=self.module('transformers.trainer')
                self.redirects.add((module.__file__,name+'.Trainer',parent.__file__))
                return SimpleNamespace(Trainer=parent.Trainer)
            redirected=self.translated(name) if not level else name
            if redirected!=name:
                self.redirects.add((module.__file__,name,redirected))
            return builtins.__import__(redirected,globals,locals,fromlist,level)
        module.__dict__['__builtins__']=dict(vars(builtins),__import__=supplied_import)
        exec(compile(Path(module.__file__).read_bytes(),module.__file__,'exec'),module.__dict__)


def bind_external_flex(source,external):
    """The model imports a shared registry; make only its frozen namespace private."""
    function=source.module('transformers.integrations.flex_attention').flex_attention_forward
    registry=dict(external.model.ALL_ATTENTION_FUNCTIONS)
    registry['flex_attention']=function
    external.model.ALL_ATTENTION_FUNCTIONS=registry
    return function


def authenticate_source(spec):
    manifest = json.loads(Path(spec['external_manifest']).read_text())
    root = Path(spec.get('external_root',manifest['root'])).resolve(strict=True)
    for name,expected in manifest['files'].items():
        if baseline.sha(root/name) != expected:
            raise ValueError(f'frozen external source hash mismatch: {name}')
    return root


def load_spec(path):
    spec=json.loads(Path(path).read_text())
    if spec.get('path') not in ('external','aligned'):
        raise ValueError('gate path must be external or aligned')
    for key in ('result_dir','tensor_dir','external_manifest','manifest','data','image_root','base','stage1'):
        if not isinstance(spec.get(key),str) or not spec[key]:
            raise ValueError(f'missing gate field: {key}')
    if spec['path']=='aligned':
        envelope=Path(spec.get('envelope',''))
        if not envelope.is_file():
            raise ValueError('freeze external controls before aligned execution')
        if json.loads(envelope.read_text()).get('protocol') != NUMERICAL_PROTOCOL:
            raise ValueError('frozen numerical protocol differs')
    tensor_destination(spec['tensor_dir'])
    result=Path(spec['result_dir']).resolve()
    if not result.is_relative_to((Path.cwd()/'results').resolve()):
        raise ValueError('gate results must be under current results tree')
    return spec


def build_inputs(spec, processor, external, source):
    from stage2 import data
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
    manifest=json.loads(Path(spec['manifest']).read_text())
    if baseline.sha(spec['data']) != manifest['data_sha256']:
        raise ValueError('data differs from retained manifest')
    if (manifest['latent_size'],manifest['supervise_latent_boundaries'],manifest['assistant_plan_rendering']) != (8,True,'concatenated-v1'):
        raise ValueError('retained manifest profile differs')
    raw=data.load_jsonl_examples(Path(spec['data']))
    retained=manifest['retained_indices']
    if len(retained)!=3379:
        raise ValueError('approved cohort requires 3379 retained examples')
    samples=[raw[i] for i in retained]
    supplied=[external.data.prepare_stage2_example(s,image_root=Path(spec['image_root']),latent_size=8) for s in samples]
    external_sampler=source.module('stage2.sampler').LatentBlockBalancedSampler(supplied,num_replicas=2,seed=42)
    order=list(external_sampler)
    if len(order)!=3380 or sorted(set(order))!=list(range(3379)):
        raise ValueError('external retained stream/padding differs')
    local=[data.prepare_stage2_example(s,image_root=Path(spec['image_root']),latent_size=8,
             assistant_plan_rendering='concatenated-v1',require_reasoning_units=True) for s in samples]
    from stage2.sampler import ExternalUntilBalancedSampler
    aligned=list(ExternalUntilBalancedSampler(local,num_replicas=4,seed=42,gradient_accumulation_steps=4))
    if aligned!=order:
        raise ValueError('aligned full212 stream differs from independent frozen-source sampler')
    replay=select_windows(order,[manifest['lengths'][i] for i in retained],
                          [manifest['latent_block_counts'][i] for i in retained])
    replay['sample_ids']=[supplied[i].sample_id for i in replay['indices']]
    replay['full_stream_sha256']=hashlib.sha256(json.dumps(order).encode()).hexdigest()
    replay['retained_indices_sha256']=hashlib.sha256(json.dumps(retained).encode()).hexdigest()
    ids=baseline.token_ids(processor.tokenizer)[:3]
    if spec['path']=='aligned':
        dataset=[local[i] for i in replay['indices']]
        collator=data.Stage2DataCollator(processor,LatentTokenIds(*ids),8,True,
            step_preference_enabled=True,preference_target_scope='until_next_latent')
    else:
        dataset=[supplied[i] for i in replay['indices']]
        collator=external.data.Stage2DataCollator(processor,external.rollout.LatentTokenIds(*ids),8,True,
            context_preference_enabled=True)
    return dataset,collator,replay


def build_model(spec, processor, external):
    """Same initializer and loader as the existing native comparator, one path/process."""
    import torch
    from stage2.attention_backend import load_qwen2_5_vl_with_attention
    from stage2.checkpointing import configure_stage2_metadata
    from stage2.peft import Stage2TuningConfig,load_stage1_dora_for_fresh_stage2
    from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
    torch.manual_seed(42)
    supplied=spec['path']=='external'
    cls=external.model.Qwen2_5_VLForConditionalGeneration if supplied else Qwen2_5_VLForConditionalGeneration
    kwargs=dict(torch_dtype=torch.bfloat16,local_files_only=True)
    if supplied:
        kwargs['stage2_split_attention']=True
    model=load_qwen2_5_vl_with_attention(spec['base'],attention_implementation='sdpa',
        vision_attention_implementation='flash_attention_2',config_load_kwargs={'local_files_only':True},
        model_load_kwargs=kwargs,load_model=cls.from_pretrained)
    base=model
    model=load_stage1_dora_for_fresh_stage2(base,processor,spec['stage1'],
        Stage2TuningConfig('dora',32,64,.05,baseline.TARGETS),requested_base_model_reference=spec['base'])
    configure_stage2_metadata(base.config,LatentTokenIds(*baseline.token_ids(processor.tokenizer)[:3]),
        latent_size=8,supervise_latent_boundaries=True,latent_activation_checkpointing=True)
    if not supplied:
        base.config.stage2_execution_profile='latest_external_until'
        base.config.vision_config.stage2_execution_profile='latest_external_until'
        for block in base.visual.blocks:
            block.attn.stage2_execution_profile='latest_external_until'
    base.config.use_cache=False
    model.gradient_checkpointing_disable()
    return model.to(dtype=torch.bfloat16).train()


def probed_trainer_class(base, spec, replay):
    import torch
    from transformers import Trainer,TrainerCallback
    from torch.utils.data import SequentialSampler
    from peft.tuners.lora.dora import DoraLinearLayer

    class ProbeCallback(TrainerCallback):
        def on_train_begin(self,args,state,control,**kwargs):
            trainer=self.owner
            engine=trainer.accelerator.deepspeed_engine_wrapped.engine
            trainer.gate_observer=AlignmentObserver(trainer,engine,spec,replay)
            original=engine.step
            def step(*a,**kw):
                boundary=engine.is_gradient_accumulation_boundary()
                skipped=engine.skipped_steps
                if boundary:
                    trainer.gate_observer.before_step()
                value=original(*a,**kw)
                if boundary:
                    if engine.skipped_steps!=skipped:
                        raise RuntimeError('optimizer step skipped')
                    trainer.gate_observer.after_step()
                return value
            engine.step=step
            return control
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step==5:
                control.should_training_stop=True
            if state.global_step>5:
                raise RuntimeError('bounded probe exceeded five updates')
            return control

    class Probed(base):
        def __init__(self,*args,**kwargs):
            callback=ProbeCallback()
            kwargs['callbacks']=[callback]
            super().__init__(*args,**kwargs)
            callback.owner=self
            self.model_accepts_loss_kwargs=False
            self.accelerator.even_batches=False
            self.gate_observer=None
            self._gate_capture=None
            self._norms={}
            self._norm_calls=0
            self._magnitude_modules=[]
            for name,module in self.model.named_modules():
                if isinstance(module,DoraLinearLayer):
                    self._magnitude_modules.append(module)
                    original=module.get_weight_norm
                    def norm(*a,_original=original,_name=name,**kw):
                        value=_original(*a,**kw)
                        self._norms[_name]=value.detach().cpu().clone()
                        self._norm_calls+=1
                        return value
                    module.get_weight_norm=norm
        def _get_train_sampler(self):
            return SequentialSampler(self.train_dataset)
        def _ensure_stage2_sampler_identity(self):
            return None
        def set_initial_training_values(self,args,dataloader,total_train_batch_size):
            # Shared runtime adaptation for replay; source forward/backward remains native.
            values=list(base.__bases__[0].set_initial_training_values(self,args,dataloader,total_train_batch_size))
            values[1]=math.ceil(values[5]/args.gradient_accumulation_steps)
            values[0]=math.ceil(values[6]/values[1])
            return tuple(values)
        def training_step(self,model,inputs,num_items_in_batch=None):
            self._norms,self._norm_calls={},0
            torch.cuda.reset_peak_memory_stats(self.gate_observer.engine.device)
            description=describe_batch(inputs)
            # Names of implementation-specific metadata differ; actual tensors and targets may not.
            coordinates={k:description[k] for k in ('tensors','sample_ids','targets')}
            before=self.gate_observer.hook_calls
            value=super().training_step(model,inputs,num_items_in_batch)
            observer=self.gate_observer
            observer.save(f'micro-{observer.micro}-dora-norms',self._norms)
            observer.write(dict(event='backward',rank=observer.rank,micro=observer.micro,
                coordinates=coordinates,dora_norm_calls=self._norm_calls,
                dora_scope_cleaned=all('forward' not in m.__dict__ for m in self._magnitude_modules),
                actual_backward_hooks=observer.hook_calls-before))
            return value
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            value=super().compute_loss(model,inputs,return_outputs=return_outputs,num_items_in_batch=num_items_in_batch)
            if self._gate_capture is None:
                raise RuntimeError('native preference reducer was not observed')
            preference,scores=self._gate_capture
            self.gate_observer.observe(value[0] if return_outputs else value,preference,inputs,scores)
            self._gate_capture=None
            return value
        def _reduce_context_objective(self,ce,clean,entropy,results):
            value=super()._reduce_context_objective(ce,clean,entropy,results)
            scores={f'clean/{i}':v.detach().cpu() for i,v in enumerate(clean)}
            scores.update({f'off/{i}':results[(i,'visual')]['mean_logp'].detach().cpu() for i in range(len(clean))})
            scores['ce']=ce.detach().cpu().reshape(())
            scores['preference']=value[1]['loss/context_preference'].detach().cpu().reshape(())
            self._gate_capture=(value[1]['loss/context_preference'],scores)
            return value
        def _reduce_step_preference_components(self,ce,clean,entropy,items,branches):
            value=super()._reduce_step_preference_components(ce,clean,entropy,items,branches)
            scores={f'clean/{i}':clean[(item.sample_id,item.span.step_id)].detach().cpu().reshape(()) for i,item in enumerate(items)}
            scores.update({f'off/{i}':self._mean_logp(branches[self._preference_branch_key(item,'visual','chosen_entropy')],
                          item.chosen_target_ids).detach().cpu().reshape(()) for i,item in enumerate(items)})
            scores['ce']=ce.detach().cpu().reshape(())
            scores['preference']=value[1]['loss/context_preference_all'].detach().cpu().reshape(())
            self._gate_capture=(value[1]['loss/context_preference_all'],scores)
            return value
    return Probed


def runtime_origins(trainer,source,external_flex):
    import inspect
    import importlib
    modules={}
    for name in ('transformers.trainer','transformers.cache_utils','transformers.integrations.deepspeed',
                 'transformers.modeling_utils','transformers.integrations.flex_attention','accelerate','deepspeed','peft'):
        module=importlib.import_module(name)
        modules[name]={'file':module.__file__,'sha256':baseline.sha(module.__file__),'shared':True}
    parent=type(trainer).__bases__[0].__bases__[0]
    parent_file=inspect.getfile(parent)
    modules['actual_parent_Trainer']={'file':parent_file,'sha256':baseline.sha(parent_file),
                                     'supplied':Path(parent_file).is_relative_to(source.root)}
    flex_file=inspect.getfile(external_flex)
    modules['external_private_registry_flex']={'file':flex_file,'sha256':baseline.sha(flex_file),
                                             'supplied':Path(flex_file).is_relative_to(source.root)}
    modules['redirected_imports']=sorted(source.redirects)
    return modules


def run_from_training_args(args, spec_path):
    """Called by src/main.py only for an explicitly requested gate spec."""
    import os
    import torch
    from torch._functorch import config as functorch_config
    from transformers import AutoProcessor,set_seed
    from main import build_stage2_training_arguments
    from deepspeed_runtime import bind_deepspeed_runtime_profile
    from stage2.trainer import Stage2Trainer
    from step_preference.config import StepPreferenceConfig
    spec=load_spec(spec_path)
    spec_sha256=baseline.sha(spec_path)
    run_id=Path(args.save_model_path).name
    if int(os.environ.get('WORLD_SIZE','1'))!=4:
        raise ValueError('launch gate with existing two-node four-rank coordinator')
    if getattr(args,'stage2_execution_profile','legacy')!='latest_external_until':
        raise ValueError('explicit latest_external_until launch profile required')
    if Path(args.load_model_path).resolve()!=Path(spec['stage1']).resolve():
        # PEFT launchers may resolve the initializer through the dedicated adapter flag.
        if Path(getattr(args,'stage1_adapter_path','') or '.').resolve()!=Path(spec['stage1']).resolve():
            raise ValueError('launch Stage1 initializer differs from gate spec')
    root=authenticate_source(spec)
    # The observer retains each compiled graph for two gradient reads before
    # ordinary Trainer backward. Disable one-use saved-buffer donation before
    # either source can compile; restore the process setting even on failure.
    # This adaptation belongs only to the diagnostic gate, never full training.
    with functorch_config.patch(donated_buffer=False), TrainerExternalSource(root) as source:
        external=external_modules(source)
        external_flex=bind_external_flex(source,external)
        processor=AutoProcessor.from_pretrained(spec['stage1'],local_files_only=True)
        dataset,collator,replay=build_inputs(spec,processor,external,source)
        model=build_model(spec,processor,external)
        initial_digest=state_digest(model)
        train_args=build_stage2_training_arguments(args)
        train_args.output_dir=str(tensor_destination(Path(spec['tensor_dir'])/'trainer'))
        train_args.max_steps=212
        train_args.warmup_steps=10
        train_args.save_strategy='no'
        train_args.logging_strategy='no'
        train_args.learning_rate=5e-5
        train_args.dataloader_num_workers=0
        if spec['path']=='external':
            base=external.trainer.Stage2Trainer
            config=external.config.StepPreferenceConfig(enabled=True,beta=.5,lambda_preference=1.,lambda_entropy=0.,modalities='visual')
            extra={}
        else:
            base=Stage2Trainer
            config=StepPreferenceConfig(enabled=True,mode='stage2b',target_scope='until_next_latent',
                beta=.5,lambda_preference=1.,lambda_entropy=0.,shared_prefix_enabled=True,
                visual_only_context_preference=True,visual_off_suppress_latent_boundaries=False,
                direct_from_stage1=True,sidecar_attention_backend='flex',sidecar_query_budget=1024)
            extra=dict(stage2_execution_profile='latest_external_until',stage2_sampler_policy='external_until_balanced')
        cls=probed_trainer_class(base,spec,replay)
        trainer=cls(model=model,args=train_args,train_dataset=dataset,data_collator=collator,
            processing_class=processor,step_preference_config=config,
            deepspeed_runtime_profile=bind_deepspeed_runtime_profile(args,world_size=4),**extra)
        rank=trainer.args.process_index
        # Both paths start from exactly the same per-rank RNG state, reset after all construction.
        set_seed(42+rank)
        result_dir=Path(spec['result_dir'])
        result_dir.mkdir(parents=True,exist_ok=True)
        metadata=dict(spec=spec,protocol=NUMERICAL_PROTOCOL,replay=replay,initial_digest=initial_digest,
            source_manifest_sha256=baseline.sha(spec['external_manifest']),rank=rank,
            envelope_sha256=baseline.sha(spec['envelope']) if spec['path']=='aligned' else None,
            source_adaptations=['frozen parent Trainer and frozen Flex callable; shared remaining framework dependencies and Stage1 loader','reference grouping2 stream replay sharded by Accelerator over4 ranks',
                'fixed Trainer GAS4 denominator and ceil replay epoch accounting','schedule212/warmup10; stop callback after5 updates',
                'gate-only AOT donated_buffer=False before compilation for retained observer gradients; restored on exit'],
            probe_compiler_config={'donated_buffer':functorch_config.donated_buffer},
            runtime_origins=runtime_origins(trainer,source,external_flex),
            final_tail_scope='actual partial optimizer event for original212 IDs, not historical optimizer state212')
        with (result_dir/f'metadata-rank-{rank}.json').open('x') as f:
            json.dump(metadata,f,indent=2)
        # Observe the compiler invocation itself, not an inferred config flag.
        compiler_calls=[]
        original_compile=torch.compile
        def observed_compile(function=None,*a,**kw):
            if function is not None:
                compiler_calls.append({'callable':getattr(function,'__module__','')+'.'+getattr(function,'__name__',type(function).__name__),
                                       'dynamic':kw.get('dynamic'),
                                       'donated_buffer':functorch_config.donated_buffer})
            return original_compile(function,*a,**kw)
        torch.compile=observed_compile
        try:
            trainer.train()
        finally:
            torch.compile=original_compile
        flex_calls=[call for call in compiler_calls if call['callable']=='torch.nn.attention.flex_attention.flex_attention']
        trainer.gate_observer.write(dict(event='compiler',rank=rank,calls=compiler_calls,
            flex_dynamic_true=bool(flex_calls) and all(call['dynamic'] is True for call in flex_calls)))
        if not flex_calls or any(call['dynamic'] is not True for call in flex_calls):
            raise RuntimeError('actual Flex compiler dynamic=True dispatch was not witnessed')
        if trainer.state.global_step!=5 or trainer.gate_observer.micro!=16:
            raise RuntimeError('bounded event count differs')
        authenticate_source(spec)
        trainer.accelerator.wait_for_everyone()
        if baseline.sha(spec_path)!=spec_sha256:
            raise ValueError('gate spec changed during execution')
        completion=dict(kind='until_alignment_probe',run_id=run_id,rank=rank,
            commit=os.environ['ILVR_REPOSITORY_COMMIT'],spec_sha256=spec_sha256,
            schedule_steps=212,probe_optimizer_steps=trainer.state.global_step,
            microsteps=trainer.gate_observer.micro+1,status='complete')
        with (result_dir/f'completion-rank-{rank}.json').open('x') as stream:
            json.dump(completion,stream,indent=2)
        trainer.accelerator.wait_for_everyone()
    return None


def records(directory):
    return [json.loads(line) for path in sorted(Path(directory).glob('rank-*.jsonl')) for line in path.read_text().splitlines()]


def metadata(directory):
    return json.loads((Path(directory)/'metadata-rank-0.json').read_text())


def load_snapshot(directory,relative):
    import torch
    return torch.load(Path(directory)/relative,map_location='cpu',weights_only=True)


def structural_identity(directory):
    result=[]
    for rank in range(4):
        payload=json.loads((Path(directory)/f'metadata-rank-{rank}.json').read_text())
        if payload['rank']!=rank:
            raise ValueError('rank metadata identity differs')
        result.append({k:payload[k] for k in ('replay','initial_digest','source_manifest_sha256')})
    coordinates=sorted([(r['rank'],r['micro'],r['coordinates']) for r in records(directory)
                        if r['event']=='backward'],key=lambda r:r[:2])
    return {'ranks':result,'forward_coordinates':coordinates}


def freeze(control_dirs,output):
    controls=[metadata(d) for d in control_dirs]
    identities=[structural_identity(d) for d in control_dirs]
    if any(i!=identities[0] for i in identities[1:]):
        raise ValueError('control inputs, rank initial states or sources differ')
    if len(controls)!=3 or any(c['spec']['path']!='external' for c in controls):
        raise ValueError('three external-only controls required')
    reports=[check_runtime(records(d),c['replay']) for d,c in zip(control_dirs,controls)]
    if not all(r['pass'] for r in reports):
        raise ValueError(f'external runtime evidence failed: {reports}')
    directories=[Path(c['spec']['tensor_dir']) for c in controls]
    files=sorted(str(p.relative_to(directories[0])) for p in directories[0].glob('rank-*/*.pt'))
    required_snapshots(set(files))
    if any(sorted(str(p.relative_to(d)) for p in d.glob('rank-*/*.pt'))!=files for d in directories[1:]):
        raise ValueError('control tensor artifacts differ')
    if any(any(c[k]!=controls[0][k] for k in ('replay','initial_digest','source_manifest_sha256')) for c in controls[1:]):
        raise ValueError('control initial/source/stream identity differs')
    envelopes={}
    for name in files:
        snapshots=[load_snapshot(d,name) for d in directories]
        if name.endswith('-rng.pt'):
            if any(any(not __import__('torch').equal(v,s[k]) for k,v in snapshots[0].items()) for s in snapshots[1:]):
                raise ValueError('control RNG differs')
            continue
        envelopes[name]=freeze_envelope(snapshots)
    payload=dict(protocol=NUMERICAL_PROTOCOL,controls=control_dirs,envelopes=envelopes,
                 structural_identity=identities[0],
                 control_metadata_sha256=[baseline.sha(Path(d)/'metadata-rank-0.json') for d in control_dirs],
                 reference_artifact_sha256={name:baseline.sha(directories[0]/name) for name in files})
    with Path(output).open('x') as f:
        json.dump(payload,f,indent=2)
    return {'frozen':str(output),'tensor_snapshots':len(envelopes)}


def compare(envelope_path,aligned_dir):
    import torch
    envelope=json.loads(Path(envelope_path).read_text())
    if envelope['protocol']!=NUMERICAL_PROTOCOL:
        raise ValueError('numerical protocol changed')
    reference_dir=envelope['controls'][0]
    reference,candidate=metadata(reference_dir),metadata(aligned_dir)
    if baseline.sha(Path(reference_dir)/'metadata-rank-0.json')!=envelope['control_metadata_sha256'][0]:
        raise ValueError('frozen reference metadata changed')
    for name,digest in envelope['reference_artifact_sha256'].items():
        if baseline.sha(Path(reference['spec']['tensor_dir'])/name)!=digest:
            raise ValueError('frozen reference artifact changed')
    required_snapshots(set(str(p.relative_to(candidate['spec']['tensor_dir'])) for p in Path(candidate['spec']['tensor_dir']).glob('rank-*/*.pt')))
    if candidate['spec']['path']!='aligned':
        raise ValueError('comparison candidate must be aligned')
    checks=[check_runtime(records(d),m['replay']) for d,m in ((reference_dir,reference),(aligned_dir,candidate))]
    exact={k:reference[k]==candidate[k] for k in ('replay','initial_digest','source_manifest_sha256')}
    exact['all_rank_structure']=structural_identity(reference_dir)==structural_identity(aligned_dir)==envelope['structural_identity']
    exact['predeclared_envelope']=candidate.get('envelope_sha256')==baseline.sha(envelope_path)
    def coordinates(d):
        return sorted([(r['rank'],r['micro'],r['coordinates']) for r in records(d) if r['event']=='backward'],key=lambda r:r[:2])
    exact['forward_inputs_targets']=coordinates(reference_dir)==coordinates(aligned_dir)
    reports={}
    for name,thresholds in envelope['envelopes'].items():
        reports[name]=compare_snapshot(load_snapshot(reference['spec']['tensor_dir'],name),
                                      load_snapshot(candidate['spec']['tensor_dir'],name),thresholds)
    for path in Path(reference['spec']['tensor_dir']).glob('rank-*/*-rng.pt'):
        name=path.relative_to(reference['spec']['tensor_dir'])
        a,b=load_snapshot(reference['spec']['tensor_dir'],name),load_snapshot(candidate['spec']['tensor_dir'],name)
        exact[str(name)]=a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    return {'pass':all(r['pass'] for r in checks) and all(exact.values()) and all(r['pass'] for r in reports.values()),
            'runtime':checks,'exact':exact,'comparisons':reports,'protocol':NUMERICAL_PROTOCOL,
            'full212_authorized':False,'scope':'bounded comparison only; controller must assess gate before launching full212'}


def validate_node_entry_names(kind, entries, ranks):
    if kind=='tensors':
        expected=expected_snapshots(ranks)
    elif kind=='results':
        expected={f'{prefix}{rank}{suffix}' for rank in ranks for prefix,suffix in (
            ('rank-','.jsonl'),('metadata-rank-','.json'),('completion-rank-','.json'))}
    else:
        raise ValueError('unknown alignment artifact kind')
    names=[e.get('path') for e in entries]
    if len(names)!=len(expected) or set(names)!=expected:
        raise ValueError('alignment node artifact inventory differs')


def _completion_records(spec_path, *, ranks, run_id, commit, spec_sha256):
    if baseline.sha(spec_path)!=spec_sha256:
        raise ValueError('alignment completion spec digest differs')
    spec=json.loads(Path(spec_path).read_text())
    for rank in ranks:
        payload=json.loads((Path(spec['result_dir'])/f'completion-rank-{rank}.json').read_text())
        expected=dict(kind='until_alignment_probe',run_id=run_id,rank=rank,commit=commit,
            spec_sha256=spec_sha256,schedule_steps=212,probe_optimizer_steps=5,microsteps=17,status='complete')
        if payload!=expected:
            raise ValueError('alignment terminal completion differs from five-step probe')
        meta=json.loads((Path(spec['result_dir'])/f'metadata-rank-{rank}.json').read_text())
        if meta['rank']!=rank or meta['spec']!=spec:
            raise ValueError('alignment metadata differs from owned spec')
    return spec


def node_evidence(spec_path, *, ranks, run_id, commit, spec_sha256):
    """Export only closed, complete rank-owned metrics and tensor inventory."""
    spec=_completion_records(spec_path,ranks=ranks,run_id=run_id,commit=commit,spec_sha256=spec_sha256)
    result={'ranks':ranks,'spec_sha256':spec_sha256}
    for kind,field in (('results','result_dir'),('tensors','tensor_dir')):
        root=Path(spec[field])
        names=expected_snapshots(ranks) if kind=='tensors' else {
            f'{prefix}{rank}{suffix}' for rank in ranks for prefix,suffix in (
                ('rank-','.jsonl'),('metadata-rank-','.json'),('completion-rank-','.json'))}
        entries=[]
        for name in sorted(names):
            path=root/name
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('alignment artifact is missing or escapes its owned directory')
            with path.open('rb') as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            entries.append({'path':name,'size':path.stat().st_size,'sha256':digest})
        validate_node_entry_names(kind,entries,ranks)
        result[kind]=entries
    return result


def validate_collected_completion(spec_path, *, run_id, commit, spec_sha256):
    spec=_completion_records(spec_path,ranks=range(4),run_id=run_id,commit=commit,spec_sha256=spec_sha256)
    required_snapshots({str(p.relative_to(spec['tensor_dir'])) for p in Path(spec['tensor_dir']).glob('rank-*/*.pt')})
    report=check_runtime(records(spec['result_dir']),metadata(spec['result_dir'])['replay'])
    if not report['pass']:
        raise ValueError(f'alignment runtime witness failed: {report["failures"]}')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    f=commands.add_parser('freeze'); f.add_argument('--controls',nargs=3,required=True); f.add_argument('--output',required=True)
    c=commands.add_parser('compare'); c.add_argument('--envelope',required=True); c.add_argument('--aligned',required=True); c.add_argument('--output',required=True)
    n=commands.add_parser('node-evidence'); n.add_argument('--spec',required=True); n.add_argument('--ranks',nargs='+',type=int,required=True)
    n.add_argument('--run-id',required=True); n.add_argument('--commit',required=True); n.add_argument('--spec-sha256',required=True)
    args=parser.parse_args()
    if args.command=='node-evidence':
        print(json.dumps(node_evidence(args.spec,ranks=args.ranks,run_id=args.run_id,commit=args.commit,spec_sha256=args.spec_sha256)))
        return
    result=freeze(args.controls,args.output) if args.command=='freeze' else compare(args.envelope,args.aligned)
    if args.command=='compare':
        with Path(args.output).open('x') as f:
            json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'pass':result.get('pass'), 'output':args.output}))
    if result.get('pass') is False:
        raise SystemExit(1)


if __name__=='__main__':
    main()
