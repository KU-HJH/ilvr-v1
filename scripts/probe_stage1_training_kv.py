#!/usr/bin/env python
"""Standalone historical Stage1-v5 local loss/4 diagnostic; never trains/saves.

Full-prefix replay intentionally shares F's live input-preparation graph. Only
its decoder prefix graph is recomputed. No distributed-engine claim follows.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import dataclass, asdict
import hashlib
import importlib
from importlib import metadata as packages
import inspect
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]


def tensor_hash(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype),list(value.shape)]).encode()
    return hashlib.sha256(header+value.view(torch.uint8).numpy().tobytes()).hexdigest()


def file_hash(path: Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def diagnostic_window(original_blocks: int):
    from stage1.v5_reduction import V5Window
    if type(original_blocks) is not int or original_blocks <= 0:
        raise ValueError('positive original block count required')
    return V5Window(16*original_blocks,4,4,4)


def image_identity(image) -> dict[str, object]:
    from PIL import Image
    if isinstance(image, Image.Image):
        return dict(mode=image.mode,size=list(image.size),sha256=hashlib.sha256(image.tobytes()).hexdigest())
    path=Path(image)
    with Image.open(path) as opened: opened.load()
    return dict(path=str(path),sha256=file_hash(path))


def select_example(examples, collator):
    for example in sorted(examples,key=lambda row:row.source_index):
        # Native preparation/collation errors are prerequisites, never a skip filter.
        batch=collator([example])
        metadata=batch['stage1_v5_metadata']
        length=batch['stage1_batch_metadata'].samples[0].unpadded_length
        count=metadata.original_step_count
        if 1 <= count <= 2 and metadata.query_keys and length <= 512:
            if not example.original_image_paths or len(example.helper_image_paths)!=count:
                raise ValueError('eligible row has incomplete original/helper images')
            evidence=dict(source_index=example.source_index,source_id=example.source_id,
                canonical_slot_id=(example.occurrence.canonical_slot_id if example.occurrence else example.source_index),
                length=length,original_blocks=count,scored_targets=len(metadata.query_keys),
                blocks=[asdict(b) for b in metadata.blocks],
                tensors={k:tensor_hash(v) for k,v in batch.items() if isinstance(v,Tensor)},
                original_images=[image_identity(p) for p in example.original_image_paths],
                helper_images=[image_identity(p) for p in example.helper_image_paths])
            return example,batch,evidence
    raise ValueError('no eligible retained TRAIN row (1-2 blocks, scored target, length <=512)')


def authenticate_runtime(saved=None):
    names=('peft_common','task_deepseed','utils_deepseed','stage2.attention_backend','stage1.trainer','stage1.v5_trainer','stage1.data','stage1.targets','stage1.checkpointing',
        'stage1.equivalence','stage1.v5_execution','stage1.v5_objective','stage1.v5_selection',
        'step_preference.dil_bce_orig_loss','step_preference.topk_entropy_loss',
        'transformers.trainer','transformers.models.qwen2_5_vl.modeling_qwen2_5_vl',
        'transformers.models.qwen2_5_vl.stage1_v5_trace','transformers.models.qwen2_5_vl.stage1_v5_replay')
    modules={}
    for name in names:
        module=importlib.import_module(name)
        path=Path(module.__file__).resolve()
        if not path.is_relative_to(ROOT.resolve()): raise ValueError(f'foreign native import: {name}: {path}')
        modules[name]=dict(path=str(path),sha256=file_hash(path))
    # Check every loaded local native module, including transitive preparation imports.
    for name,module in tuple(sys.modules.items()):
        if name.startswith(('stage1.','step_preference.','transformers.models.qwen2_5_vl.')):
            path=getattr(module,'__file__',None)
            if path and not Path(path).resolve().is_relative_to(ROOT.resolve()):
                raise ValueError(f'foreign native import: {name}: {path}')
    aliases=dict(stage1_trainer='stage1.trainer',v5_trainer='stage1.v5_trainer',transformers_trainer='transformers.trainer')
    recorded=(saved or {}).get('execution',{}).get('source_sha256',{})
    trainer_hashes={key:dict(actual=modules[name]['sha256'],recorded=recorded.get(key),
        matches=modules[name]['sha256']==recorded.get(key)) for key,name in aliases.items()}
    if recorded and any(not value['matches'] for key,value in trainer_hashes.items() if key in recorded):
        raise ValueError('recorded Trainer hash differs from isolated runtime')
    versions={key:packages.version(key) for key in ('torch','transformers','accelerate','deepspeed','peft','trl')}
    recorded_packages=(saved or {}).get('execution',{}).get('packages',{})
    if any(versions.get(key)!=value for key,value in recorded_packages.items()):
        raise ValueError('package versions differ from saved runtime')
    return dict(modules=modules,trainer_hashes=trainer_hashes,packages=versions,
        recorded_packages=(saved or {}).get('execution',{}).get('packages',{}),
        authentication_scope='module roots and current hashes; three recorded Trainer hashes; partial historical authentication')


def read_contract(checkpoint):
    from stage1.v5_resume import load_v5_contract
    saved=load_v5_contract(Path(checkpoint))
    if saved['coefficients'] != dict(ce_weight=.1,sim_weight=.1):
        raise ValueError('diagnostic requires saved CE0.1/sim0.1')
    return saved


def restore_added_tokens(tokenizer, directory):
    """Recover original AddedToken flags, rather than inference string coercion."""
    from transformers import AddedToken
    payload=json.loads((Path(directory)/'tokenizer_config.json').read_text())
    entries=payload.get('added_tokens_decoder',{})
    for raw_id,entry in entries.items():
        token=AddedToken(**{k:v for k,v in entry.items() if k!='__type'})
        if tokenizer.convert_tokens_to_ids(token.content)!=int(raw_id):
            raise ValueError('AddedToken source ID conflict')
        tokenizer.add_tokens([token],special_tokens=token.special)
    return tokenizer


def apply_backend(saved):
    from stage1.equivalence import read_stage1_backend_settings
    backend=saved['execution']['backend']
    torch.backends.cudnn.deterministic=backend['cudnn_deterministic']
    torch.backends.cudnn.benchmark=backend['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32=backend['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32=backend['cudnn_allow_tf32']
    torch.set_float32_matmul_precision(backend['float32_matmul_precision'])
    if read_stage1_backend_settings()!=backend: raise ValueError('saved backend could not be restored')
    return read_stage1_backend_settings()


def restore_models(checkpoint, base, processor, saved, *, device):
    from transformers import Qwen2_5_VLConfig
    from stage1.composition import _configure_latent_model,_load_stage1_model,DEFAULT_STAGE1_DEPENDENCIES
    from stage1.checkpointing import load_stage1_peft_metadata,preflight_stage1_peft_artifact,load_stage1_peft_artifact
    from stage1.peft import build_stage1_tuned_model,install_stage1_sample_stable_dropout
    from stage1.types import Stage1TuningConfig
    from stage1.ema import Stage1EmaTeacher
    from stage1.v5_reference import restore_v5_reference,V5ReferenceIdentity
    checkpoint,base=Path(checkpoint),Path(base)
    meta=load_stage1_peft_metadata(checkpoint)
    config=Qwen2_5_VLConfig.from_pretrained(base)
    _configure_latent_model(config,meta.latent_token_ids,latent_size=meta.latent_size)
    config.compress_strategy=saved['execution']['compress_strategy']; config.use_cache=False
    load_args=SimpleNamespace(stage1_variant='v5',cache_dir=None,
        attention_implementation=saved['execution']['attention_implementation'],
        vision_attention_implementation=saved['execution']['vision_attention_implementation'])
    preflight=preflight_stage1_peft_artifact(processor,config,checkpoint,
        requested_base_model_reference=str(base),v5_resume=True)
    original=_load_stage1_model(str(base),config,load_args,DEFAULT_STAGE1_DEPENDENCIES)
    student=load_stage1_peft_artifact(original,processor,checkpoint,requested_base_model_reference=str(base),
        is_trainable=True,preflight=preflight,v5_resume=True)
    # PEFT restores trainable adapters as FP32 by default; restore saved BF16 storage.
    student.to(dtype=torch.bfloat16)
    sites=install_stage1_sample_stable_dropout(student)
    student.stage1_stochastic_sites=tuple(sites)
    reference_base=_load_stage1_model(str(base),deepcopy(config),load_args,DEFAULT_STAGE1_DEPENDENCIES)
    reference_base.resize_token_embeddings(len(processor.tokenizer))
    tuning=Stage1TuningConfig(method='dora',rank=meta.rank,alpha=meta.alpha,dropout=meta.dropout,
        target_suffixes=meta.requested_target_suffixes)
    reference=build_stage1_tuned_model(reference_base,tuning,meta.latent_token_ids)
    teacher=Stage1EmaTeacher(student,1.,'dora',initial_reference_model=reference)
    state=torch.load(checkpoint/'stage1_v5_reference.pt',map_location='cpu',weights_only=True)
    named=dict(teacher.model.named_parameters())
    for name,value in state['parameters'].items(): named[name].data=named[name].data.to(value.dtype)
    restore_v5_reference(teacher,student,processor,base_path=base,directory=checkpoint,
        expected=V5ReferenceIdentity(**saved['reference']))
    if saved['execution']['gradient_checkpointing']:
        from stage1.equivalence import stage1_checkpoint_context_fn
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False,context_fn=stage1_checkpoint_context_fn))
    student.to(device); teacher.model.to(device)
    student.train(); teacher.model.eval()
    validate_model_profile(student,saved)
    validate_model_profile(teacher.model,saved,reference=True)
    return student,teacher


def prepare_inputs(args):
    from transformers import AutoProcessor
    from stage1.composition import _add_fresh_latent_tokens
    from stage1.checkpointing import load_stage1_peft_metadata
    from stage1.data import prepare_stage1_comt_examples,Stage1DataCollator
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
    from stage1.v5_reference import _processor_identity
    saved=read_contract(args.checkpoint)
    runtime=authenticate_runtime(saved)
    meta=load_stage1_peft_metadata(Path(args.checkpoint))
    processor=AutoProcessor.from_pretrained(args.base)
    _,ids=_add_fresh_latent_tokens(processor.tokenizer)
    restore_added_tokens(processor.tokenizer,args.checkpoint)
    if tuple(ids)!=tuple(meta.latent_token_ids): raise ValueError('original processor latent ID mismatch')
    examples=prepare_stage1_comt_examples(Path(args.data_path),Path(args.image_root))
    _,batch,evidence=select_example(examples,Stage1DataCollator(processor,LatentTokenIds(*ids),meta.latent_size,stage1_variant='v5'))
    evidence.update(processor_sha256=_processor_identity(processor),train_sha256=file_hash(Path(args.data_path)),
        checkpoint=str(Path(args.checkpoint).resolve()),base=str(Path(args.base).resolve()))
    if evidence['processor_sha256']!=saved['reference']['processor_sha256']:
        raise ValueError('original AddedToken processor identity does not match authenticated reference')
    return processor,batch,evidence,saved,runtime


def _write_json(path,payload):
    with Path(path).open('x') as stream: json.dump(payload,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')


def run_probe(args):
    from stage1.equivalence import Stage1StochasticContext
    destination=ROOT/'results'/args.run_id
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'): raise ValueError('run-id must be one path component')
    destination.mkdir(parents=True,exist_ok=True)
    prepared=destination/'prepared-input.json'; report_path=destination/'report.json'
    if report_path.exists() or (args.prepare_only and prepared.exists()): raise FileExistsError('refusing existing experiment evidence')
    report=blank_report(); started=time.monotonic(); arms={}
    try:
        processor,batch,evidence,saved,runtime=prepare_inputs(args)
        report.update(input=evidence,objective=saved,runtime=runtime)
        frozen=finite_json(dict(schema_version=1,input=evidence,objective=saved))
        if args.prepare_only:
            _write_json(prepared,frozen)
            return dict(report,status='prepared')
        if not prepared.is_file() or json.loads(prepared.read_text())!=frozen:
            raise ValueError('missing or mismatched frozen CPU prepared-input identity')
        report['runtime']['backend']=apply_backend(saved)
        model,teacher=restore_models(args.checkpoint,args.base,processor,saved,device=args.device)
        report['runtime']['constructed']=runtime_model_profile(model,teacher)
        trainer=make_carrier(model,teacher,processor,saved)
        working={k:v.to(args.device) if isinstance(v,Tensor) else v for k,v in batch.items()}
        # The original selected sample's canonical ID defines source dropout. It is not the raw source string.
        sample=batch['stage1_batch_metadata'].samples[0]
        slot=sample.source_index if sample.canonical_slot_id is None else sample.canonical_slot_id
        state=json.loads((Path(args.checkpoint)/'trainer_state.json').read_text())
        stochastic=Stage1StochasticContext(42,int(state['global_step']),((slot,sample.unpadded_length),),
            tuple(model.stage1_stochastic_sites))
        report['objective']['diagnostic_window']=asdict(diagnostic_window(batch['stage1_v5_metadata'].original_step_count))
        report['objective']['local_backward_divisor']=4
        report['objective']['normalization_scope']='same example repeated over nominal global16; not historical heterogeneous window or distributed participation'
        four_passes(trainer,working,stochastic=stochastic,results=arms)
        report.update(summarize_results(arms))
    except Exception as error:
        if arms: report.update(summarize_results(arms))
        report.update(status='incomplete',reason=f'{type(error).__name__}: {error}')
    report['wall_seconds']=time.monotonic()-started
    _write_json(report_path,finite_json(report))
    return finite_json(report)


@dataclass(frozen=True)
class OriginalDecoderInputs:
    inputs_embeds: Tensor
    position_ids: Tensor
    padding: Tensor


@contextmanager
def capture_original_inputs(decoder: nn.Module) -> Iterator[list[OriginalDecoderInputs]]:
    records=[]
    def before(module,args,kwargs):
        if records: return
        embeds=kwargs.get('inputs_embeds')
        if embeds is None: raise ValueError('original F must expose live prepared inputs_embeds')
        batch,length,_=embeds.shape
        padding=kwargs.get('attention_mask')
        if padding is None: padding=torch.ones((batch,length),dtype=torch.long,device=embeds.device)
        if padding.shape!=(batch,length): raise ValueError('original F padding must be [B,T]')
        positions=kwargs.get('position_ids')
        if positions is None: positions=torch.arange(length,device=embeds.device)[None,None].expand(3,batch,-1)
        if positions.ndim==2: positions=positions[None].expand(3,-1,-1)
        records.append(OriginalDecoderInputs(embeds,positions,padding))
    handle=decoder.register_forward_pre_hook(before,with_kwargs=True,prepend=True)
    try: yield records
    finally: handle.remove()


def full_prefix_replay(decoder,lm_head,original,blocks,*,execution,need_voff,need_textoff,chunk_positions=128):
    from transformers.models.qwen2_5_vl.stage1_v5_trace import bind_v5_execution,_PACKED
    from transformers.models.qwen2_5_vl.stage1_v5_replay import V5ReplayOutput
    if type(need_voff) is not bool or type(need_textoff) is not bool or type(chunk_positions) is not int or chunk_positions<=0:
        raise ValueError('boolean branches and positive chunk_positions required')
    keys=tuple((b.batch_index,t-1) for b in blocks for t in b.target_positions)
    active=tuple(b for b in blocks if b.target_positions)
    rows=tuple((kind,b) for kind,enabled in (('v',need_voff),('text',need_textoff)) if enabled for b in active)
    if not rows: return V5ReplayOutput(None,None,keys)
    x=original.inputs_embeds
    length=x.shape[1]
    samples=torch.tensor([b.batch_index for _,b in rows],device=x.device)[:,None].expand(-1,length)
    positions=torch.arange(length,device=x.device)[None].expand(len(rows),-1)
    valid=original.padding[samples,positions].bool()
    embeds=x[samples,positions]
    ids=original.position_ids[:,samples,positions]
    causal=torch.ones((length,length),device=x.device,dtype=torch.bool).tril()
    bias=x.new_full((len(rows),1,length,length),float('-inf'))
    for row,(kind,b) in enumerate(rows):
        visible=causal & original.padding[b.batch_index].bool()[None,:]
        if kind=='v': visible[b.end:,b.start+1:b.end]=False
        else:
            for target,blocked in zip(b.target_positions,b.blocked_text_keys): visible[target-1,list(blocked)]=False
        # Padded query rows cannot influence valid causal queries.
        visible[~valid[row],0]=True
        bias[row,0].masked_fill_(visible,0.)
    token=_PACKED.set((execution.dropout_mask,samples,positions,valid,{}))
    try:
        with bind_v5_execution(decoder,execution,role='packed'):
            output=decoder(inputs_embeds=embeds,position_ids=ids,attention_mask=bias,use_cache=False,return_dict=True)
    finally: _PACKED.reset(token)
    row_indices,predictors=zip(*[(i,t-1) for i,(_,b) in enumerate(rows) for t in b.target_positions])
    selected=output.last_hidden_state[row_indices,predictors]
    logits=torch.cat([lm_head(chunk) for chunk in selected.split(chunk_positions)])
    count=len(keys)
    return V5ReplayOutput(logits[:count] if need_voff else None,
        logits[count:] if need_voff and need_textoff else logits if need_textoff else None,keys)


def oracle_adapter(original, evidence):
    def replay(decoder,lm_head,_trace,blocks,**kwargs):
        # Deliberately never access _trace, including its embeddings/positions.
        if len(original)!=1: raise ValueError('oracle requires exactly one original F input view')
        if evidence is not None:
            evidence['blocks']=[asdict(b) for b in blocks]
        return full_prefix_replay(decoder,lm_head,original[0],blocks,**kwargs)
    return replay


@contextmanager
def temporary_attribute(owner,name,value):
    had=name in vars(owner)
    previous=getattr(owner,name,None)
    setattr(owner,name,value)
    try: yield
    finally:
        if had: setattr(owner,name,previous)
        else: delattr(owner,name)


@dataclass
class ArmResult:
    raw_loss: float
    scaled_loss: float
    components: dict[str,float]
    gradients: dict[str,Tensor | None]
    evidence: dict[str,object]


def parameter_identity(model,*,trainable=True):
    return {name:dict(dtype=str(p.dtype),shape=list(p.shape),sha256=tensor_hash(p))
        for name,p in model.named_parameters() if not trainable or p.requires_grad}


def run_local_backward(trainer,batch,*,stochastic,oracle,observe):
    from peft_common import resolve_qwen2_5_vl_model
    from stage1.v5_execution import V5Execution
    import transformers.models.qwen2_5_vl.stage1_v5_replay as replay
    if trainer.stage1_v5_debug or trainer.collect_stage1_diagnostics:
        raise ValueError('all diagnostic arms require saved stage1_v5_debug=False')
    if getattr(trainer,'_v5_step_stack',None) is not None: raise RuntimeError('nested native carrier')
    qwen=resolve_qwen2_5_vl_model(trainer.model)
    started=time.monotonic()
    ambient_before=tensor_hash(torch.get_rng_state())
    devices=sorted({p.device.index for p in trainer.model.parameters() if p.device.type=='cuda'})
    ambient_cuda_before=[tensor_hash(torch.cuda.get_rng_state(i)) for i in devices]
    named={name:p for name,p in trainer.model.named_parameters() if p.requires_grad}
    for p in trainer.model.parameters(): p.grad=None
    before=parameter_identity(trainer.model,trainable=False)
    reference_before=parameter_identity(trainer.ema_teacher.model,trainable=False)
    evidence=dict(blocks_original=[asdict(b) for b in batch['stage1_v5_metadata'].blocks],parameters_before=before,
        trainable_inventory={name:before[name] for name in named},
        parameter_identity_scope='all named student and reference parameters, including frozen values',
        stochastic=asdict(stochastic),oracle=oracle,observed=observe,
        autocast_enabled=torch.is_autocast_enabled(),debug=False,
        bypasses=['Trainer/SFT constructor and profile initialization','distributed wrappers/engine/optimizer/scheduler',
            'native parent training_step (CPU prerequisite tests cover local loss/4 only)'])
    components={}
    working=dict(batch,stage1_v5_window=diagnostic_window(batch['stage1_v5_metadata'].original_step_count))
    with ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=devices))
        torch.random.default_generator.manual_seed(stochastic.base_seed)
        for device in devices:
            with torch.cuda.device(device): torch.cuda.manual_seed(stochastic.base_seed)
        evidence["ambient_rng"]=dict(seed=stochastic.base_seed,cpu_sha256=tensor_hash(torch.get_rng_state()),
            cuda_sha256=[tensor_hash(torch.cuda.get_rng_state(i)) for i in sorted({p.device.index for p in trainer.model.parameters() if p.device.type=="cuda"})],
            scope="common local alignment-noise realization; not historical noise draw")
        execution=stack.enter_context(V5Execution(stochastic,debug=False))
        stack.enter_context(temporary_attribute(trainer,'_v5_step_stack',stack))
        stack.enter_context(temporary_attribute(trainer,'_v5_execution',execution))
        stack.enter_context(temporary_attribute(trainer,'_latest_stage1_diagnostics',None))
        if hasattr(qwen,'rope_deltas'):
            stack.enter_context(temporary_attribute(qwen,'rope_deltas',qwen.rope_deltas))
        if observe: stack.enter_context(observe_arm(trainer,evidence,components))
        if oracle:
            original=stack.enter_context(capture_original_inputs(qwen.model))
            stack.enter_context(temporary_attribute(replay,'run_v5_packed_replay',oracle_adapter(original,evidence)))
        loss,outputs=trainer.compute_loss(trainer.model,working,return_outputs=True)
        raw=float(loss.detach()); scaled=loss/4
        evidence['clean_logits_sha256']=tensor_hash(outputs.logits)
        evidence['clean_logits_dtype']=str(outputs.logits.dtype)
        evidence['original_inputs_sha256']=tensor_hash(outputs.inputs_embeds)
        evidence['original_inputs_dtype']=str(outputs.inputs_embeds.dtype)
        scaled.backward()
        grads={name:None if p.grad is None else p.grad.detach().cpu().clone() for name,p in named.items()}
    evidence['parameters_after']=parameter_identity(trainer.model,trainable=False)
    evidence['parameters_unchanged']=evidence['parameters_after']==before
    evidence['execution_closed']=execution.state=='CLOSED'
    evidence['ambient_cpu_rng_restored']=ambient_before==tensor_hash(torch.get_rng_state())
    evidence['ambient_cuda_rng_restored']=ambient_cuda_before==[tensor_hash(torch.cuda.get_rng_state(i)) for i in devices]
    evidence['reference_before']=reference_before
    evidence['reference_after']=parameter_identity(trainer.ema_teacher.model,trainable=False)
    evidence['reference_unchanged']=reference_before==evidence['reference_after']
    evidence['wall_seconds']=time.monotonic()-started
    return ArmResult(raw,float(scaled.detach()),components,grads,evidence)


@contextmanager
def observe_supports(records):
    import stage1.v5_objective as objective
    original=objective.build_entropy_support
    def observe(*args,**kwargs):
        support=original(*args,**kwargs)
        records.append(dict(indices=support.ids.detach().cpu().tolist(),valid=support.valid.detach().cpu().tolist(),
            sha256=tensor_hash(support.ids),mode=kwargs['mode']))
        return support
    with temporary_attribute(objective,'build_entropy_support',observe): yield records


@contextmanager
def observe_arm(trainer,evidence,components):
    """Passive wrappers return the exact native objects, with debug kept false."""
    import stage1.v5_trainer as native
    from peft_common import resolve_qwen2_5_vl_model
    preparation=trainer._prepare_stage1_loss_inputs
    losses=trainer._stage1_native_losses
    objective=native.compute_v5_loss_sums
    selection=native.select_v5_text_keys
    evidence['supports']=[]; evidence['decoder_calls']=[]; evidence['selected_text_keys']=[]
    evidence['branches']=dict(voff='unexercised',textoff='unexercised')
    evidence['voff_edges']=0; evidence['textoff_edges']=0
    def prepare(*args,**kwargs):
        result=preparation(*args,**kwargs)
        targets=result[2]
        evidence['helper']=dict(values_sha256=tensor_hash(targets.latent_values),
            dtype=str(targets.latent_values.dtype),mask_sha256=tensor_hash(targets.latent_mask),
            selected_counts=targets.selected_counts,selected_candidate_indices=targets.selected_candidate_indices,
            detached=not result[0]['latent_hidden_states'].requires_grad)
        return result
    def native_losses(*args,**kwargs):
        result=losses(*args,**kwargs)
        components.update(ce=float(result[0].detach()),alignment=float(result[1].detach()))
        return result
    def sums(*args,**kwargs):
        result=objective(*args,**kwargs)
        evidence['branches']=dict(voff='exercised' if evidence['voff_edges'] else 'unexercised',
            textoff='exercised' if evidence['textoff_edges'] else 'unexercised')
        evidence['scored_queries']=len(args[4].query_keys)
        for name in ('dil_sum','entropy_sum','mass_sum','text_kl'):
            components[name]=float(getattr(result,name).detach())
        return result
    def select(*args,**kwargs):
        result=selection(*args,**kwargs)
        evidence['selected_text_keys']=[dict(query=list(key),keys=list(value)) for key,value in result.items()]
        return result
    def reference(module,args,kwargs,output):
        logits=getattr(output,'stage1_selected_logits',None)
        if logits is not None:
            evidence['reference_logits_sha256']=tensor_hash(logits)
            evidence['reference_logits_dtype']=str(logits.dtype)
    def decoder(module,args,kwargs):
        embeds=kwargs['inputs_embeds']
        mask=kwargs.get('attention_mask')
        if mask is not None and mask.ndim==4:
            blocks=[b for b in evidence['blocks_original'] if b['target_positions']]
            config=trainer.v5_config
            need_v=config.lambda_topk_entropy>0 or config.selection in ('voff','gap','intersection')
            rows=[(kind,b) for kind,needed in (('v',need_v),('text',config.text_kl_weight>0)) if needed for b in blocks]
            selected={tuple(item['query']):item['keys'] for item in evidence['selected_text_keys']}
            prefix=kwargs.get('stage2_prefix_key_values') is not None
            for row,(kind,b) in enumerate(rows):
                if kind=='v':
                    first=0 if prefix else b['end']
                    stop=max(b['target_positions'])-b['end'] if prefix else max(b['target_positions'])
                    evidence['voff_edges']+=int(torch.isneginf(mask[row,0,first:stop,b['start']+1:b['end']]).sum())
                else:
                    for target in b['target_positions']:
                        keys=selected.get((b['batch_index'],target-1),[])
                        query=target-1-b['end'] if prefix else target-1
                        evidence['textoff_edges']+=int(torch.isneginf(mask[row,0,query,keys]).sum())
        evidence['decoder_calls'].append(dict(batch=embeds.shape[0],length=embeds.shape[1],dtype=str(embeds.dtype),
            past_key_values=kwargs.get('past_key_values') is not None,
            prefix_key_values=kwargs.get('stage2_prefix_key_values') is not None,
            mask_shape=None if mask is None else list(mask.shape),
            blocked_edges=None if mask is None or mask.ndim!=4 else int(torch.isneginf(mask).sum())))
    with ExitStack() as stack:
        stack.enter_context(temporary_attribute(trainer,'_prepare_stage1_loss_inputs',prepare))
        stack.enter_context(temporary_attribute(trainer,'_stage1_native_losses',native_losses))
        stack.enter_context(temporary_attribute(native,'compute_v5_loss_sums',sums))
        stack.enter_context(temporary_attribute(native,'select_v5_text_keys',select))
        stack.enter_context(observe_supports(evidence['supports']))
        handle=trainer.ema_teacher.model.register_forward_hook(reference,with_kwargs=True)
        stack.callback(handle.remove)
        handle=resolve_qwen2_5_vl_model(trainer.model).model.register_forward_pre_hook(decoder,with_kwargs=True,prepend=True)
        stack.callback(handle.remove)
        yield


def four_passes(trainer,batch,*,stochastic,results=None):
    # The caller keeps completed arms if a later pass fails. No graph crosses arms.
    if results is None: results={}
    for name,oracle,observe in (('U',False,False),('A',False,True),('A-repeat',False,True),('B',True,True)):
        results[name]=run_local_backward(trainer,batch,stochastic=stochastic,oracle=oracle,observe=observe)
    return results


def compare_gradients(left: Mapping[str,Tensor | None],right: Mapping[str,Tensor | None]):
    """FP64 is used ONLY on detached CPU statistics, never model arithmetic."""
    parameters={}; unused=[]; mismatches=[]; shape_mismatches=[]
    left_sq=right_sq=error_sq=dot=0.; maximum=0.; finite=True; exact=True
    for name in sorted(set(left)|set(right)):
        l,r=left.get(name),right.get(name)
        row=dict(left_key=name in left,right_key=name in right,left_present=l is not None,right_present=r is not None)
        if name not in left or name not in right or (l is None)!=(r is None):
            mismatches.append(name); exact=False; row['status']='presence_mismatch'
        elif l is None:
            unused.append(name); row['status']='matched_unused'
        elif l.shape!=r.shape:
            shape_mismatches.append(name); exact=False; row['status']='shape_mismatch'
        else:
            l64,r64=l.detach().cpu().double(),r.detach().cpu().double()
            good=bool(torch.isfinite(l64).all() and torch.isfinite(r64).all())
            same=torch.equal(l,r); finite=finite and good; exact=exact and same
            row.update(status='matched_present',left_dtype=str(l.dtype),right_dtype=str(r.dtype),shape=list(l.shape),finite=good,exact=same)
            if good:
                ls=float(l64.square().sum()); rs=float(r64.square().sum())
                es=float((r64-l64).square().sum()); dp=float((l64*r64).sum())
                mx=float((r64-l64).abs().max()) if l.numel() else 0.
                row.update(left_l2=math.sqrt(ls),right_l2=math.sqrt(rs),error_l2=math.sqrt(es),max_abs=mx,
                    both_zero=ls==0 and rs==0,zero_denominator=ls==0,
                    relative_l2=math.sqrt(es/ls) if ls else 0. if rs==0 else None,
                    cosine=max(-1.,min(1.,dp/math.sqrt(ls*rs))) if ls and rs else None)
                left_sq+=ls; right_sq+=rs; error_sq+=es; dot+=dp; maximum=max(maximum,mx)
            else:
                row.update(left_l2=None,right_l2=None,error_l2=None,max_abs=None,both_zero=False,
                    zero_denominator=None,relative_l2=None,cosine=None)
        # Nonfinite one-sided tensors must also invalidate finite reporting.
        for tensor in (l,r):
            if tensor is not None and not bool(torch.isfinite(tensor).all()): finite=False
        parameters[name]=row
    global_metrics=dict(scope='matched present tensors only',left_l2=math.sqrt(left_sq) if finite else None,
        right_l2=math.sqrt(right_sq) if finite else None,error_l2=math.sqrt(error_sq) if finite else None,
        max_abs=maximum if finite else None,both_zero=left_sq==right_sq==0 if finite else False,
        zero_denominator=left_sq==0 if finite else None,
        relative_l2=(math.sqrt(error_sq/left_sq) if left_sq else 0. if right_sq==0 else None) if finite else None,
        cosine=max(-1.,min(1.,dot/math.sqrt(left_sq*right_sq))) if finite and left_sq and right_sq else None)
    return dict(parameters=parameters,matched_unused=unused,presence_mismatches=mismatches,
        shape_mismatches=shape_mismatches,finite=finite,exact=exact and finite,**{'global':global_metrics})


def finite_json(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [finite_json(v) for v in value]
    return value


def _same_control(left,right,*,observed):
    metrics=compare_gradients(left.gradients,right.gradients)
    common=('clean_logits_sha256','original_inputs_sha256','original_inputs_dtype','reference_before','parameters_before','parameters_after','stochastic','ambient_rng')
    if observed: common+=('helper','reference_logits_sha256','supports','selected_text_keys')
    state={key:left.evidence.get(key)==right.evidence.get(key) for key in common}
    losses=math.isfinite(left.raw_loss) and math.isfinite(right.raw_loss) and left.raw_loss==right.raw_loss and left.scaled_loss==right.scaled_loss
    return dict(valid=losses and metrics['exact'] and all(state.values()),loss_exact=losses,
        gradients=metrics,state=state,scope=('observed helper/reference/clean/support/selection and gradients' if observed else
        'returned native loss/clean logits, complete gradients and boundary state; no helper/support observer in U'))


def summarize_results(arms):
    serialized={name:dict(raw_loss=result.raw_loss,scaled_loss=result.scaled_loss,components=result.components,
        evidence=result.evidence) for name,result in arms.items()}
    result=dict(status='incomplete',finite=False,arms=serialized,controls={},comparisons={})
    if tuple(arms)!=('U','A','A-repeat','B'):
        result['reason']='four ordered passes did not complete'
        return finite_json(result)
    observer=_same_control(arms['U'],arms['A'],observed=False)
    repeat=_same_control(arms['A'],arms['A-repeat'],observed=True)
    contrast=compare_gradients(arms['A'].gradients,arms['B'].gradients)
    finite=all(math.isfinite(r.raw_loss) and math.isfinite(r.scaled_loss) and
        all(math.isfinite(v) for v in r.components.values()) and
        all(v is None or bool(torch.isfinite(v).all()) for v in r.gradients.values()) for r in arms.values())
    native,oracle=arms['A'],arms['B']
    support_a=native.evidence.get('supports',[]); support_b=oracle.evidence.get('supports',[])
    supports_equal=support_a==support_b
    overlap=[]
    for l,r in zip(support_a,support_b):
        for li,ri,lv,rv in zip(l['indices'],r['indices'],l['valid'],r['valid']):
            ls={v for v,valid in zip(li,lv) if valid}; rs={v for v,valid in zip(ri,rv) if valid}
            overlap.append(dict(intersection=len(ls&rs),union=len(ls|rs),jaccard=len(ls&rs)/len(ls|rs) if ls|rs else None))
    common={key:native.evidence.get(key)==oracle.evidence.get(key) for key in
        ('helper','reference_logits_sha256','clean_logits_sha256','original_inputs_sha256','original_inputs_dtype','selected_text_keys','parameters_before','stochastic','ambient_rng')}
    unchanged=all(r.evidence.get('parameters_unchanged',False) and r.evidence.get('reference_unchanged',False)
        and r.evidence.get('ambient_cpu_rng_restored',False) and r.evidence.get('ambient_cuda_rng_restored',False) for r in arms.values())
    nonzero=all(any(v is not None and bool(v.count_nonzero()) for v in r.gradients.values()) for r in arms.values())
    valid=observer['valid'] and repeat['valid'] and all(common.values()) and unchanged and nonzero
    result.update(status='measured' if finite and valid else 'inconclusive',finite=finite,
        controls=dict(observer=observer,repeat=repeat,contrast_common_state=common,unchanged_state=unchanged,nonzero_gradients=nonzero),
        comparisons=dict(native_vs_full_prefix=dict(gradients=contrast,
            raw_loss_difference=oracle.raw_loss-native.raw_loss,scaled_loss_difference=oracle.scaled_loss-native.scaled_loss,
            component_differences={key:oracle.components[key]-native.components[key] for key in native.components if key in oracle.components},
            supports=dict(equal=supports_equal,classification='selection_mediated' if not supports_equal else 'same_observed_support',
                native_calls=len(support_a),oracle_calls=len(support_b),overlap=overlap),
            interpretation='local measurements only; no cross-path equivalence tolerance')))
    if not finite: result['reason']='nonfinite objective/component/gradient'
    elif not valid: result['reason']='observer/repeat/common-state/nonzero/state-restoration prerequisite failed'
    return finite_json(result)


def make_carrier(model,teacher,processor,saved):
    from stage1.v5_trainer import CustomTrainerStage1V5
    from stage1.v5_config import Stage1V5Config
    from stage1.targets import Stage1TeacherTargetBuilder
    trainer=CustomTrainerStage1V5.__new__(CustomTrainerStage1V5)
    trainer.model=model; trainer.ema_teacher=teacher
    trainer.stage1_v5_debug=False; trainer.collect_stage1_diagnostics=False
    trainer.ce_weight=saved['coefficients']['ce_weight']; trainer.sim_weight=saved['coefficients']['sim_weight']
    trainer.v5_config=Stage1V5Config(**saved['config'])
    execution=saved['execution']
    trainer.target_builder=Stage1TeacherTargetBuilder(processor.tokenizer,coverage_p=execution['coverage_p'],
        image_pool_k=execution['image_pool_k'],helper_group_L=execution['helper_group_L'],
        textual_states=execution['textual_states'],validate=False,collect_diagnostics=False)
    trainer._v5_step_stack=None; trainer._ema_execution_device=None
    return trainer


def blank_report():
    return dict(schema_version=1,status='incomplete',scope='historical Stage1-v5 standalone pre-engine local raw objective and loss/4 gradients',
        runtime={},input={},objective={},arms={},controls={},comparisons={},limitations=[
        'No optimizer, scheduler, checkpoint save, Trainer.train, distributed engine or training-trajectory comparison.',
        'Shared live F input-preparation graph; independent decoder full-prefix replay only.',
        'BF16 differences are measurements; FP32 CPU prerequisite tolerances do not certify BF16 equivalence.',
        'Native-repeat spread is repeatability evidence, never a cross-path tolerance.',
        'Total-gradient equality cannot establish each OFF-branch Jacobian.',
        'Diagnostic arm wall time includes observation/hashing overhead; not training throughput.',
        'Historical authentication is partial; three Trainer hashes do not authenticate every historical runtime byte.'])


def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('checkpoint','base','data-path','image-root','run-id'): parser.add_argument('--'+option,required=True)
    parser.add_argument('--device',default='cuda:0'); parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv)
    try:
        report=run_probe(args)
        print(json.dumps(dict(status=report['status'],reason=report.get('reason'),run_id=args.run_id),allow_nan=False),flush=True)
        return 0 if report['status'] in ('prepared','measured') else 1
    except Exception as error:
        print(json.dumps(dict(status='incomplete',reason=f'{type(error).__name__}: {error}'),allow_nan=False),file=sys.stderr,flush=True)
        return 1


def runtime_model_profile(model,teacher):
    from peft_common import resolve_qwen2_5_vl_model
    qwen=resolve_qwen2_5_vl_model(model)
    return dict(attention_implementation=qwen.model.config._attn_implementation,
        vision_attention_implementation=qwen.visual.config._attn_implementation,
        decoder_attention_class=type(qwen.model.layers[0].self_attn).__module__+'.'+type(qwen.model.layers[0].self_attn).__qualname__,
        vision_attention_class=type(qwen.visual.blocks[0].attn).__module__+'.'+type(qwen.visual.blocks[0].attn).__qualname__,
        decoder_checkpointing=qwen.model.gradient_checkpointing,vision_checkpointing=qwen.visual.gradient_checkpointing,
        trainables=parameter_identity(model),reference_dtypes={n:str(p.dtype) for n,p in teacher.model.named_parameters() if n in teacher.parameter_names},
        nonzero_lora_B=any(bool(p.detach().count_nonzero()) for n,p in model.named_parameters() if 'lora_B' in n),
        stochastic_sites=list(model.stage1_stochastic_sites),dropout=[dict(site=m.site_name,p=m.p) for m in model.modules() if hasattr(m,'site_name') and hasattr(m,'p')])


def validate_model_profile(model,saved,*,reference=False):
    from peft_common import resolve_qwen2_5_vl_model
    from stage2.attention_backend import validate_qwen2_5_vl_attention_backends
    qwen=resolve_qwen2_5_vl_model(model)
    execution=saved['execution']
    for actual,key in ((qwen.model.config._attn_implementation,'attention_implementation'),
                       (qwen.visual.config._attn_implementation,'vision_attention_implementation')):
        if actual!=execution[key]: raise ValueError(f'constructed backend mismatch: {key}')
    if execution['vision_attention_implementation']=='flash_attention_2':
        validate_qwen2_5_vl_attention_backends(qwen,attention_implementation=execution['attention_implementation'],
            vision_attention_implementation=execution['vision_attention_implementation'])
    if not reference:
        if qwen.model.gradient_checkpointing!=execution['gradient_checkpointing']:
            raise ValueError('constructed checkpointing differs from saved profile')
        if any(p.dtype!=torch.bfloat16 for p in model.parameters() if p.requires_grad):
            raise ValueError('student trainable representation must remain BF16')
        dropout=[m.p for m in model.modules() if hasattr(m,'site_name') and hasattr(m,'p')]
        if not dropout or any(p!=execution['lora_dropout'] for p in dropout):
            raise ValueError('source-coordinate dropout differs from saved profile')


if __name__=='__main__':
    raise SystemExit(main())
