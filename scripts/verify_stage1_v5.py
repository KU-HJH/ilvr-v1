#!/usr/bin/env python
"""Stage1-v5 read-only audit and coordinator-owned four-rank admission witness."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

REPOSITORY = Path(__file__).resolve().parents[1]
GEOMETRY_DIMENSIONS = ('branch_count','prefix_length','suffix_length','selected_queries','clean_length',
    'padded_suffix_tokens','dense_bias_elements','replicated_prefix_tokens')
VISUAL_SHAPE_FIELDS = ('original_image_grid_thw','helper_image_grid_thw',
    'original_pixel_shape','helper_pixel_shape')


def packed_geometry(metadata, *, source_index, source_id):
    active = [b for b in metadata.blocks if b.target_positions]
    prefix = max(active, key=lambda b:b.end) if active else None
    suffix = max(active, key=lambda b:max(b.target_positions)-b.end) if active else None
    p = prefix.end if prefix else 0
    s = max(suffix.target_positions)-suffix.end if suffix else 0
    branches = 2*len(active)
    return dict(source_index=source_index, source_id=source_id, original_blocks=len(metadata.blocks),
        branch_count=branches,prefix_length=p,suffix_length=s,
        prefix_source_item=prefix.source_item_index if prefix else None,
        suffix_source_item=suffix.source_item_index if suffix else None,
        selected_queries=2*len(metadata.query_keys), projection_chunk_positions=128,
        dense_bias_elements=branches*s*(p+s),replicated_prefix_tokens=branches*p,
        padded_suffix_tokens=branches*s)


def collated_geometry(batch):
    """Read the actual microbatch1 inputs before the helper consumes its tensors."""
    try:
        sample,=batch['stage1_batch_metadata'].samples
        ids=batch['input_ids']
        if len(ids.shape)!=2 or ids.shape[0]!=1:
            raise ValueError('admission geometry requires collated microbatch1')
        return dict(packed_geometry(batch['stage1_v5_metadata'],
            source_index=sample.source_index,source_id=sample.source_id),clean_length=ids.shape[1],
            original_image_grid_thw=batch['image_grid_thw'].detach().cpu().tolist(),
            helper_image_grid_thw=batch['image_grid_thw_latent'].detach().cpu().tolist(),
            original_pixel_shape=list(batch['pixel_values'].shape),
            helper_pixel_shape=list(batch['pixel_values_latent'].shape))
    except KeyError as error:
        raise ValueError(f'missing collated admission geometry: {error.args[0]}') from error


def _geometry_comparison(row):
    try:
        return tuple(row[k] for k in GEOMETRY_DIMENSIONS),tuple(row[k] for k in VISUAL_SHAPE_FIELDS)
    except KeyError as error:
        raise ValueError(f'missing whole-step admission geometry: {error.args[0]}') from error


def maximum_geometries(rows):
    """Keep all non-dominated actual rows in source order, including equal rows.

    Compare raw B/P/S/Q/L and derived allocations only for identical original
    and helper visual shapes. This is no allocator or FA2 workspace bound.
    """
    comparisons=[_geometry_comparison(row) for row in rows]
    return [row for row,(extents,visual) in zip(rows,comparisons) if not any(
        other_visual==visual and all(a>=b for a,b in zip(other,extents))
        and any(a>b for a,b in zip(other,extents)) for other,other_visual in comparisons)]


def audit_prepared_loader(dataset, *, output_dir):
    """Actual inherited Stage1 Trainer loader plus Accelerate's actual four-way sharder.

    Only device/process placement is supplied explicitly for CPU metadata audit;
    sampler, seedable conversion, even-batch repeats and epoch budgeting execute.
    No process group or runtime distributed participation is claimed.
    """
    import torch
    from accelerate.data_loader import prepare_data_loader
    from transformers import Trainer, TrainingArguments
    from stage1.v5_trainer import CustomTrainerStage1V5
    occurrences=[]; lengths=[]; budgets=[]
    for rank in range(4):
        trainer=Trainer(model=torch.nn.Linear(1,1),args=TrainingArguments(str(output_dir),
            use_cpu=True,per_device_train_batch_size=1,gradient_accumulation_steps=4,
            seed=42,report_to=[],remove_unused_columns=False,dataloader_num_workers=0),
            train_dataset=list(range(len(dataset))),data_collator=lambda rows:rows)
        config=trainer.accelerator.dataloader_config
        trainer.accelerator.prepare=lambda loader:prepare_data_loader(loader,device=torch.device('cpu'),
            num_processes=4,process_index=rank,split_batches=config.split_batches,
            dispatch_batches=config.dispatch_batches,even_batches=config.even_batches,
            use_seedable_sampler=config.use_seedable_sampler,data_seed=config.data_seed,
            rng_types=[],put_on_device=False)
        loader=trainer.get_train_dataloader()
        lengths.append(len(loader))
        # Exact v5 opt-in parent epoch-budget implementation, not a standalone ceil.
        budget_owner=CustomTrainerStage1V5.__new__(CustomTrainerStage1V5)
        budgets.append(budget_owner.set_initial_training_values(trainer.args,loader,16)[1])
        for microbatch,rows in enumerate(loader):
            for index in rows:
                source=dataset[index]
                occurrences.append(dict(rank=rank,microbatch=microbatch,
                    source_index=getattr(source,'source_index',index),source_id=getattr(source,'source_id',None)))
    if len(set(lengths))!=1 or len(set(budgets))!=1 or not lengths[0]:
        raise ValueError('prepared rank loaders disagree or dataset is empty')
    counts=Counter(row['source_index'] for row in occurrences)
    return dict(retained_samples=len(dataset),prepared_loader_microbatches_per_rank=lengths,
        sampler_occurrences=occurrences,source_occurrences=[dict(source_index=k,repeat_count=v) for k,v in sorted(counts.items())],
        configured_gas=4,optimizer_updates_for_one_exposure=budgets[0],
        final_window_microbatches=(lengths[0]-1)%4+1,native_tail_denominator=16,
        scope='CPU actual parent loader/Accelerate sharder; no process group',
        sampler_profile=asdict(config))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(args):
    from transformers import AutoProcessor, AutoConfig
    from stage1.composition import _add_fresh_latent_tokens
    from stage1.data import prepare_stage1_comt_examples, Stage1DataCollator
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
    config=AutoConfig.from_pretrained(args.model,local_files_only=True)
    _,ids=_add_fresh_latent_tokens(processor.tokenizer)
    examples=prepare_stage1_comt_examples(args.data_path,args.image_root)
    collator=Stage1DataCollator(processor,LatentTokenIds(*ids),8,stage1_variant='v5')
    rows=[]
    for example in examples:
        batch=collator([example])
        if batch['input_ids'].shape[1]>32768:
            raise ValueError(f'source {example.source_index} exceeds approved 32768 sequence capacity')
        rows.append(collated_geometry(batch))
    report=audit_prepared_loader(examples,output_dir=REPOSITORY/'checkpoints'/args.run_id/'audit-metadata')
    from importlib.metadata import version
    from stage1.v5_reference import _base_identity, _processor_identity
    base_digest,_=_base_identity(Path(args.model))
    report.update(max_packed_geometry=maximum_geometries(rows),retained_geometry=rows,
        source_identity=dict(data_sha256=_sha(args.data_path),data_path=str(args.data_path),image_root=str(args.image_root)),
        code_identity=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPOSITORY,text=True).strip(),
            files={name:_sha(REPOSITORY/name) for name in ('scripts/verify_stage1_v5.py','src/stage1/data.py',
                'src/stage1/v5_trainer.py','src/stage1/trainer.py','transformers/src/transformers/trainer.py')}),
        profile_identity=dict(model=str(args.model),base_content_sha256=base_digest,processor_sha256=_processor_identity(processor),config_sha256=_sha(Path(args.model)/'config.json'),
            model_type=config.model_type,latent_ids=ids,deepspeed_config_sha256=_sha(REPOSITORY/'configs/deepspeed_zero1.json'),packages={p:version(p) for p in ('torch','transformers','accelerate','deepspeed','peft','trl')}))
    return report


def require_distributed_entry():
    if os.environ.get('WORLD_SIZE')!='4': raise ValueError('witness requires four global ranks')
    if not os.environ.get('ILVR_REPOSITORY_COMMIT') or os.environ.get('ILVR_NUM_MACHINES')!='2':
        raise ValueError('witness requires coordinator-owned entry')
    import torch
    if not torch.cuda.is_available(): raise ValueError('witness requires actual GPU ZeRO1 execution')


def _contract_digest(contract):
    return hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_contract_profile(contract, audited):
    """Bind actual resolved execution/objective to the approved selected profile."""
    from stage1.v5_config import Stage1V5Config
    from stage1.v5_resume import validate_v5_resume
    validate_v5_resume(contract,{})
    if contract['config']!=asdict(Stage1V5Config(lambda_topk_entropy=.1,preference_topk_k=50)):
        raise ValueError('different resolved v5 objective')
    if contract['coefficients']!=dict(ce_weight=.1,sim_weight=.1):
        raise ValueError('different resolved CE/alignment coefficients')
    execution=contract['execution']
    fixed=dict(per_device_train_batch_size=1,gradient_accumulation_steps=4,bf16=True,seed=42,
        learning_rate=5e-5,warmup_steps=10,weight_decay=.01,max_grad_norm=1.,
        gradient_checkpointing=True,dataloader_num_workers=0,optim='adamw_torch_fused',
        lr_scheduler_type='linear',tuning_method='dora',ema_tau=1.,zero_stage=1,
        attention_implementation='sdpa',vision_attention_implementation='flash_attention_2',
        latent_size=8,max_sequence_length=32768,lora_r=32,lora_alpha=64,lora_dropout=.05,
        lora_target_modules='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',loss_chunk_positions=128,
        compress_strategy='average',coverage_p=.9,image_pool_k=8,helper_group_L=256,textual_states='previous',
        model=audited['profile_identity']['base_content_sha256'])
    for key,value in fixed.items():
        if execution.get(key)!=value: raise ValueError(f'different resolved execution {key}')
    source_paths={'v5_trainer':'src/stage1/v5_trainer.py','stage1_trainer':'src/stage1/trainer.py',
                  'transformers_trainer':'transformers/src/transformers/trainer.py'}
    if execution['source_sha256']!={key:audited['code_identity']['files'][path] for key,path in source_paths.items()}:
        raise ValueError('actual imported source profile differs from audit')
    if execution['packages']!=audited['profile_identity']['packages']:
        raise ValueError('audit/runtime package profile mismatch')
    digest=audited['profile_identity']['deepspeed_config_sha256']
    if execution['deepspeed_profile']!=dict(enabled=True,zero_stage=1,world_size=4,
                                           config_path=digest,config_digest=digest):
        raise ValueError('different resolved DeepSpeed profile')
    for key in ('base_content_sha256','processor_sha256'):
        if contract['reference'][key]!=audited['profile_identity'][key]:
            raise ValueError(f'audit/runtime reference mismatch: {key}')


def validate_runtime_binding(audited, binding):
    validate_contract_profile(binding['contract'],audited)
    observed=binding['original_loader']
    for key in ('retained_samples','prepared_loader_microbatches_per_rank','sampler_occurrences',
                'source_occurrences','configured_gas','optimizer_updates_for_one_exposure',
                'final_window_microbatches','native_tail_denominator','sampler_profile'):
        if observed[key]!=audited[key]: raise ValueError(f'original prepared loader/audit mismatch: {key}')
    schedule=binding['contract']['schedule']
    fixed=dict(dataset_samples=audited['retained_samples'],
        prepared_microbatches_per_rank=audited['prepared_loader_microbatches_per_rank'][0],
        gradient_accumulation_steps=4,effective_global_batch=16,world_size=4,
        configured_max_steps=audited['optimizer_updates_for_one_exposure'],num_train_epochs=1.,
        one_exposure_window_updates=audited['optimizer_updates_for_one_exposure'],
        parent_epoch_update_budget=audited['optimizer_updates_for_one_exposure'],
        final_window_microbatches=audited['final_window_microbatches'],dataloader_drop_last=False,
        tail_native_denominator='configured_gas_world_size',
        dil_denominator='actual_window_eligible_dil_tokens',
        auxiliary_denominator='actual_window_original_blocks_including_empty')
    for key,value in fixed.items():
        if schedule.get(key)!=value: raise ValueError(f'original contract/audit schedule mismatch: {key}')
    expected_input=dict(data_sha256=audited['source_identity']['data_sha256'],
        base_content_sha256=audited['profile_identity']['base_content_sha256'],
        processor_sha256=audited['profile_identity']['processor_sha256'],
        source_sha256=audited['code_identity']['files'])
    if binding['input_identity']!=expected_input: raise ValueError('runtime input content identity mismatch')


def observe_original_loader_rank(trainer, loader, rank):
    """Iterate actual prepared batches, including native collation/device placement."""
    occurrences=[]
    for microbatch,batch in enumerate(loader):
        for sample in batch['stage1_batch_metadata'].samples:
            occurrences.append(dict(rank=rank,microbatch=microbatch,
                source_index=sample.source_index,source_id=sample.source_id))
    return dict(rank=rank,occurrences=occurrences,length=len(loader),
        sampler_profile=asdict(trainer.accelerator.dataloader_config))


def observe_original_loader(trainer, loader, contract):
    import torch.distributed as dist
    local=observe_original_loader_rank(trainer,loader,dist.get_rank())
    gathered=[None]*4;dist.all_gather_object(gathered,local)
    if [r['rank'] for r in gathered]!=[0,1,2,3] or any(
            r['sampler_profile']!=local['sampler_profile'] for r in gathered):
        raise ValueError('original loader rank/sampler profile disagreement')
    occurrences=[o for r in gathered for o in r['occurrences']]
    counts=Counter(o['source_index'] for o in occurrences)
    schedule=contract['schedule']
    return dict(retained_samples=len(trainer.train_dataset),
        prepared_loader_microbatches_per_rank=[r['length'] for r in gathered],sampler_occurrences=occurrences,
        source_occurrences=[dict(source_index=k,repeat_count=v) for k,v in sorted(counts.items())],
        configured_gas=trainer.args.gradient_accumulation_steps,
        optimizer_updates_for_one_exposure=schedule['parent_epoch_update_budget'],
        final_window_microbatches=schedule['final_window_microbatches'],native_tail_denominator=16,
        sampler_profile=local['sampler_profile'])


def validate_rank_records(records, *, run_id, expected_geometry, audited):
    for row in expected_geometry: _geometry_comparison(row)
    if len(records)!=4 or {r['rank'] for r in records}!={0,1,2,3}: raise ValueError('missing/corrupt global ranks')
    contracts=set()
    for r in records:
        if r['world_size']!=4 or r['run_id']!=run_id or r['entry'] is not True or r['exit'] is not True:
            raise ValueError('incorrect real rank entry/exit')
        if r['reference_before']!=r['reference_after']: raise ValueError('reference mutated')
        validate_runtime_binding(audited,r['runtime_binding'])
        validate_contract_profile(r['fixture_contract'],audited)
        if r['contract_digest']!=_contract_digest(r['fixture_contract']):
            raise ValueError('unbound common contract digest')
        if r['runtime_binding']['contract']['reference']['effective_state_sha256']!=r['reference_before']:
            raise ValueError('reference state is not bound to actual original contract')
        fixture_schedule=r['fixture_contract']['schedule']
        for key,value in dict(dataset_samples=16*len(expected_geometry),prepared_microbatches_per_rank=4*len(expected_geometry),
                              configured_max_steps=len(expected_geometry),parent_epoch_update_budget=len(expected_geometry),
                              final_window_microbatches=4).items():
            if fixture_schedule.get(key)!=value: raise ValueError(f'fixture schedule mismatch: {key}')
        contracts.add(r['contract_digest'])
        if len(r['tiny_windows'])!=3: raise ValueError('missing full/tail windows')
        for i,(w,(n,m)) in enumerate(zip(r['tiny_windows'],((9,4),(9,4),(5,2)),strict=True)):
            if w['optimizer_step']!=i or not math.isclose(w['learning_rate'],tiny_learning_rate(i),rel_tol=0.,abs_tol=1e-12):
                raise ValueError('tiny learning rate/optimizer-step mismatch')
            if (w['global_original_steps'],w['actual_microbatches'],w['configured_gas'],w['native_denominator'])!=(n,m,4,16):
                raise ValueError('wrong actual global denominator/window')
            counts=w['all_rank_original_counts']
            expected_counts=([[2,1,0,0],[1,0,1,0],[0,0,0,0],[1,2,0,1]] if m==4
                             else [[1,0],[1,1],[0,0],[0,2]])
            if counts!=expected_counts or sum(map(sum,counts))!=n:
                raise ValueError('rank counts do not match global original-step denominator')
            token_counts=w['all_rank_dil_token_counts']
            expected_tokens=([[3,1,0,0],[1,0,1,0],[0,0,0,0],[1,3,0,1]] if m==4
                             else [[1,0],[1,1],[0,0],[0,3]])
            token_total=11 if m==4 else 6
            if (token_counts!=expected_tokens or sum(map(sum,token_counts))!=token_total
                    or w['global_dil_tokens']!=token_total
                    or w['dil_scale']!=16/token_total or w['auxiliary_scale']!=16/n):
                raise ValueError('rank counts/scales do not match global DIL-token denominator')
            for key,limit in (('gradient_error',.005),('update_error',2e-6)):
                if not math.isfinite(w[key]) or w[key]>limit: raise ValueError('global oracle mismatch')
            if not math.isfinite(w['gradient_norm']) or w['gradient_norm']<=0: raise ValueError('nonfinite/zero gradient')
        if len(r['continuations'])!=2: raise ValueError('missing engine continuation')
        for c,s in zip(r['continuations'],(1,2),strict=True):
            if (c['saved_step'],c['sources'],c['updates'],c['final_step'])!=(s,list(range(s*4,10)),3-s,3):
                raise ValueError('engine continuation lost sources/tail update')
            if c['learning_rates']!=[tiny_learning_rate(i) for i in range(s,3)]:
                raise ValueError('continuation learning rate mismatch')
        if any(r['hooks'].get(k,0)<=0 for k in ('helper','F','backward')):
            raise ValueError('missing real model/backward entries')
        p=r['production']
        expected_windows=[dict(step=i,source_index=g['source_index'],microbatches=[g['source_index']]*4)
                          for i,g in enumerate(expected_geometry)]
        if (p['global_step']!=len(expected_geometry) or p['source_windows']!=expected_windows
                or len(p['gradient_norms'])!=len(expected_windows)
                or p['gradient_norm_steps']!=list(range(len(expected_geometry)))
                or p['optimizer_steps']!=[dict(step=i,successful=True) for i in range(len(expected_geometry))]):
            raise ValueError('incomplete or uneven source/window/norm coverage')
        eligible_microbatches=4*sum(g['selected_queries']>0 for g in expected_geometry)
        expected_hooks=dict(helper=p['microbatches'],F=p['microbatches'],backward=p['microbatches'],
            reference=eligible_microbatches,packed=eligible_microbatches)
        if p['microbatches']!=4*p['global_step'] or any(
                r['hooks'].get(key,0)!=count for key,count in expected_hooks.items()):
            raise ValueError('incorrect conditional model/one-backward counts')
        if p['global_step']<1 or p['finite_loss'] is not True or not 0<p['peak_allocated']<=p['peak_reserved']:
            raise ValueError('production profile failed')
        if not p['gradient_norms'] or any(not math.isfinite(g) or g<=0 for g in p['gradient_norms']):
            raise ValueError('missing/nonfinite production gradient norms')
        if p['geometry']!=expected_geometry or set(p['source_indices'])!={g['source_index'] for g in expected_geometry}:
            raise ValueError('missing maximum joint geometry')
    if len({_contract_digest(r['runtime_binding']) for r in records})!=1:
        raise ValueError('original runtime binding differs across ranks')
    if len(contracts)!=1 or len({r['reference_before'] for r in records})!=1: raise ValueError('rank contract/reference disagreement')
    return dict(kind='stage1-v5-admission',ranks=[0,1,2,3],run_id=run_id,status='pass')


def tiny_learning_rate(optimizer_step):
    # Probe config: total3, warmup10, base5e-5 and linear schedule. All three
    # mutation boundaries remain in warmup; values are independently derived.
    if type(optimizer_step) is not int or not 0<=optimizer_step<3:
        raise ValueError('invalid tiny optimizer step')
    return 5e-5*optimizer_step/10


def tiny_engine_witness(directory, *, use_cpu=False, deepspeed=None):
    """Real parent training loop, checkpoint and optimizer; engine when distributed.

    Scalar logits feed the actual objective to isolate parent/engine reduction. Full Qwen calls are witnessed
    separately by production_witness; this fixture never claims model equivalence.
    """
    from copy import deepcopy
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import Trainer, TrainingArguments
    from stage1.v5_trainer import CustomTrainerStage1V5
    from stage1.v5_metadata import V5BatchMetadata,V5Block
    from stage1.v5_reduction import v5_auxiliary_scale, v5_dil_scale
    from stage1.v5_config import Stage1V5Config
    from stage1.v5_objective import compute_v5_loss_sums
    from tests.stage1.v5.test_reduction import FOUR_RANK_WINDOWS
    world=1 if use_cpu else 4
    counts=FOUR_RANK_WINDOWS[0]
    tails=FOUR_RANK_WINDOWS[1]
    sources=[]; windows=[]; local_counts=[]; local_token_rows=[]; oracle_states=[]
    config=Stage1V5Config(text_kl_weight=0.,lambda_topk_entropy=.1,preference_topk_k=2)
    class Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.p=torch.nn.Parameter(torch.tensor(.2))
        def forward(self,**kwargs): return self.p
    class WitnessTrainer(CustomTrainerStage1V5):
        v5_config=config
        # Scalar engine fixture uses v5 schedule/prefetch with parent lifecycle;
        # production composition below owns Qwen/reference/PEFT-specific hooks.
        __init__=Trainer.__init__
        train=Trainer.train
        save_model=Trainer.save_model
        _save=Trainer._save
        _save_checkpoint=Trainer._save_checkpoint
        _rotate_checkpoints=Trainer._rotate_checkpoints
        log=Trainer.log
        def _get_train_sampler(self): return SequentialSampler(self.train_dataset)
        def compute_loss(self,model,inputs,**kwargs):
            window=inputs['stage1_v5_window']
            metadata=inputs['stage1_v5_metadata']
            p=model(); n=len(metadata.query_keys)
            clean=p*p.new_tensor([.2,-.5,.8]).repeat(n,1)
            reference=p.new_tensor([.3,.6,-.4]).repeat(n,1)
            off=p*p.new_tensor([.6,-.2,.1]).repeat(n,1)
            sums=compute_v5_loss_sums(clean,reference,off,None,metadata,config)
            return (.1*p.square()+v5_dil_scale(window)*sums.dil_sum
                    +v5_auxiliary_scale(window)*.1*sums.entropy_sum)
        def training_step(self,model,inputs,*args,**kwargs):
            sources.append(inputs['source'])
            local_counts.append(inputs['stage1_v5_metadata'].original_step_count)
            local_token_rows.append(tuple(b.target_ids for b in inputs['stage1_v5_metadata'].blocks))
            self.latest_window=inputs['stage1_v5_window']
            if not use_cpu and not getattr(self,'engine_observed',False):
                engine=self.accelerator.deepspeed_engine_wrapped.engine
                original=engine.step
                def step(*a,**kw):
                    before_step(self,engine)
                    result=original(*a,**kw)
                    after_step(self,engine)
                    return result
                engine.step=step; self.engine_observed=True
            return Trainer.training_step(self,model,inputs,*args,**kwargs)
    initial_value=torch.tensor(.2,dtype=torch.float32 if use_cpu else torch.bfloat16).float()
    oracle=torch.nn.Parameter(initial_value.clone())
    oracle_optimizer=torch.optim.AdamW([oracle],lr=5e-5,weight_decay=.01)
    pending={}
    def before_step(trainer,engine=None):
        w=trainer.latest_window
        actual_counts=local_counts[-w.actual_microbatches:]
        token_rows=local_token_rows[-w.actual_microbatches:]
        if not use_cpu:
            import torch.distributed as dist
            gathered_rows=[None]*4; dist.all_gather_object(gathered_rows,(actual_counts,token_rows))
            gathered=[row[0] for row in gathered_rows]
            rank_tokens=[row[1] for row in gathered_rows]
            if sum(map(sum,gathered))!=w.global_original_steps: raise ValueError('actual count collective mismatch')
        else: gathered=[actual_counts]; rank_tokens=[token_rows]
        pending['all_rank_original_counts']=gathered
        token_counts=[[sum(map(len,batch)) for batch in rank] for rank in rank_tokens]
        if sum(map(sum,token_counts))!=w.global_dil_tokens:
            raise ValueError('actual DIL token count collective mismatch')
        pending['all_rank_dil_token_counts']=token_counts
        # Independent global arithmetic; no candidate scale/window helper used.
        oracle_optimizer.zero_grad()
        logp=(oracle*torch.tensor([.2,-.5,.8])).log_softmax(-1)
        logmu=torch.tensor([.3,.6,-.4]).log_softmax(-1)
        ratio=logp-logmu
        logq=(oracle*torch.tensor([.6,-.2,.1]))[[0,2]].log_softmax(-1)
        entropy=math.log(2)+(logq.exp()*logq).sum()
        dil,aux=oracle*0,oracle*0
        for rank in rank_tokens:
            for batch in rank:
                for targets in batch:
                    if targets:
                        dil=dil+torch.nn.functional.softplus(-ratio[list(targets)]).sum()
                        dil=dil+len(targets)*(logmu.exp()*torch.nn.functional.softplus(ratio)).sum()
                        aux=aux+.1*entropy
        objective=(.1*w.actual_microbatches/4*oracle.square()
            +(dil/w.global_dil_tokens if w.global_dil_tokens else dil*0)
            +(aux/w.global_original_steps if w.global_original_steps else aux*0))
        objective.backward()
        if engine is None:
            observed=trainer.model.p.grad.detach().float().cpu()
            before=trainer.model.p.detach().float().cpu()
            lr=trainer.optimizer.param_groups[0]['lr']
        else:
            from deepspeed.utils import safe_get_full_grad,safe_get_full_fp32_param
            observed=safe_get_full_grad(trainer.model.p).detach().float().cpu()
            before=safe_get_full_fp32_param(trainer.model.p).detach().float().cpu()
            lr=engine.optimizer.param_groups[0]['lr']
        expected_lr=tiny_learning_rate(trainer.state.global_step)
        if not math.isclose(lr,expected_lr,rel_tol=0.,abs_tol=1e-12):
            raise ValueError(f'tiny optimizer learning rate mismatch at step {trainer.state.global_step}: {lr} != {expected_lr}')
        pending['learning_rate']=lr
        pending['optimizer_step']=trainer.state.global_step
        if not torch.isfinite(observed): raise ValueError('tiny nonfinite gradient')
        pending.update(error=float((observed-oracle.grad).abs()),norm=float(observed.abs()),before=before)
        oracle_optimizer.param_groups[0]['lr']=expected_lr
        oracle_optimizer.step()
    def after_step(trainer,engine=None):
        if engine is None: actual=trainer.model.p.detach().float().cpu()
        else:
            from deepspeed.utils import safe_get_full_fp32_param
            actual=safe_get_full_fp32_param(trainer.model.p).detach().float().cpu()
        # Save only the independent scalar oracle's parameter and AdamW state,
        # after its literal token/block update, without retaining an autograd graph.
        oracle_states.append((oracle.detach().clone(),deepcopy(oracle_optimizer.state_dict())))
        w=trainer.latest_window
        windows.append(dict(global_original_steps=w.global_original_steps,configured_gas=4,
            global_dil_tokens=w.global_dil_tokens,dil_scale=v5_dil_scale(w),
            auxiliary_scale=v5_auxiliary_scale(w),
            actual_microbatches=w.actual_microbatches,native_denominator=4*world,
            optimizer_step=pending['optimizer_step'],learning_rate=pending['learning_rate'],
            all_rank_original_counts=pending['all_rank_original_counts'],
            all_rank_dil_token_counts=pending['all_rank_dil_token_counts'],
            gradient_error=pending['error'],gradient_norm=pending['norm'],update_error=float((actual-oracle.detach()).abs())))
    def collate(rows):
        slot=rows[0]; micro=slot if use_cpu else slot//4; rank=0 if use_cpu else slot%4
        active_counts=counts if micro<8 else tails
        position=micro%4 if micro<8 else micro-8
        rows=[(r,i) for r in (range(4) if use_cpu else (rank,))
              for i in range(active_counts[r][position])]
        blocks=tuple(V5Block(0,j,8*j,8*j+1,tuple(range(8*j+2,8*j+3+i)),
                            ((r+i)%3,)*(i+1),()) for j,(r,i) in enumerate(rows))
        return dict(source=micro,stage1_v5_metadata=V5BatchMetadata(blocks))
    def build(output):
        args=TrainingArguments(str(output),use_cpu=use_cpu,bf16=not use_cpu,
            per_device_train_batch_size=1,gradient_accumulation_steps=4,max_steps=3,
            learning_rate=5e-5,warmup_steps=10,weight_decay=.01,max_grad_norm=1.,
            save_strategy='steps',save_steps=1,save_on_each_node=True,report_to=[],
            remove_unused_columns=False,dataloader_num_workers=0,seed=42,deepspeed=deepspeed,
            disable_tqdm=True)
        trainer=WitnessTrainer(model=Scalar(),args=args,train_dataset=list(range(10*world)),data_collator=collate)
        trainer.model_accepts_loss_kwargs=False
        if use_cpu:
            trainer.create_optimizer()
            trainer.optimizer.register_step_pre_hook(lambda *_:before_step(trainer))
            trainer.optimizer.register_step_post_hook(lambda *_:after_step(trainer))
        return trainer
    directory=Path(directory)
    original=build(directory/'initial')
    original.train()
    full_windows=list(windows)
    saved_oracles=tuple(oracle_states)
    del original
    continuations=[]
    # Each independently restored parent+ZeRO continuation retains absolute source positions.
    for saved in (1,2):
        sources.clear(); windows.clear(); local_counts.clear()
        trainer=build(directory/f'continuation-{saved}')
        # Restore the exact independent new-objective state, including moments
        # from the zero-LR first update. Deep copies keep continuations isolated.
        parameter,state=saved_oracles[saved-1]
        oracle.data.copy_(parameter); oracle_optimizer=torch.optim.AdamW([oracle],lr=0.,weight_decay=.01)
        oracle_optimizer.load_state_dict(deepcopy(state))
        trainer.train(resume_from_checkpoint=str(directory/'initial'/f'checkpoint-{saved}'))
        continuations.append(dict(saved_step=saved,sources=list(sources),updates=len(windows),final_step=trainer.state.global_step,learning_rates=[w['learning_rate'] for w in windows]))
        if any(w['gradient_error']>.005 or w['update_error']>2e-6 for w in windows):
            raise ValueError(f'restored engine oracle mismatch: {windows}')
        del trainer
    return dict(tiny_windows=full_windows,continuations=continuations,
                scope='CPU actual parent loop' if use_cpu else 'four-rank actual Trainer/ZeRO1')


def observe_production_trainer(trainer, *, expected_geometry=None):
    from peft_common import resolve_qwen2_5_vl_model
    hooks=Counter(); observed=set(); observed_geometry={}; gradient_norms=[]; gradient_norm_steps=[]; source_windows=[]; optimizer_steps=[]
    original_finish=trainer._finish_optimizer_step
    def finish(*args,successful,**kwargs):
        optimizer_steps.append(dict(step=trainer.state.global_step,successful=successful))
        return original_finish(*args,successful=successful,**kwargs)
    trainer._finish_optimizer_step=finish
    original_norm=trainer._record_gradient_diagnostic
    def record_norm(norm):
        gradient_norm_steps.append(trainer.state.global_step)
        gradient_norms.append(float(norm.detach().cpu())); return original_norm(norm)
    trainer._record_gradient_diagnostic=record_norm
    student=resolve_qwen2_5_vl_model(trainer.model)
    reference=resolve_qwen2_5_vl_model(trainer.ema_teacher.model)
    helper=[False]
    original_build=trainer.target_builder.build
    def build(*a,**kw):
        helper[0]=True
        try: return original_build(*a,**kw)
        finally: helper[0]=False
    trainer.target_builder.build=build
    handles=[student.model.register_forward_pre_hook(lambda m,a,kw:hooks.update(['packed' if 'stage2_prefix_key_values' in kw else 'F']),with_kwargs=True),
        reference.model.register_forward_pre_hook(lambda *_:hooks.update(['helper' if helper[0] else 'reference']))]
    original_step=trainer.training_step
    expected_by_source=({g['source_index']:g for g in selected_admission_geometry(
        dict(max_packed_geometry=expected_geometry))} if expected_geometry is not None else None)
    def step(model,inputs,*a,**kw):
        sample=inputs['stage1_batch_metadata'].samples[0]
        geometry=collated_geometry(inputs)
        if expected_by_source is not None and geometry!=expected_by_source.get(sample.source_index):
            raise ValueError('live collated admission geometry differs from audit')
        observed.add(sample.source_index)
        step_index=trainer.state.global_step
        if not source_windows or source_windows[-1]['step']!=step_index:
            source_windows.append(dict(step=step_index,source_index=sample.source_index,microbatches=[]))
        source_windows[-1]['microbatches'].append(sample.source_index)
        observed_geometry[sample.source_index]=geometry
        return original_step(model,inputs,*a,**kw)
    trainer.training_step=step
    original_backward=trainer.accelerator.backward
    def backward(*a,**kw):
        hooks.update(['backward']); return original_backward(*a,**kw)
    trainer.accelerator.backward=backward
    return dict(hooks=hooks,source_indices=observed,geometry=observed_geometry,handles=handles,gradient_norms=gradient_norms,gradient_norm_steps=gradient_norm_steps,source_windows=source_windows,optimizer_steps=optimizer_steps)


def validate_audit_reference(audited, identity):
    for key in ('base_content_sha256','processor_sha256'):
        if audited['profile_identity'][key]!=getattr(identity,key):
            raise ValueError(f'audit/runtime reference identity mismatch: {key}')


def production_witness(args, audited):
    """Bounded original-base/r32/BF16/ZeRO1 training on every admission source."""
    import torch
    from stage1.composition import build_stage1_training_components
    from stage1.v5_reference import _state_identity
    from utils_deepseed import parse_args
    from transformers.integrations.deepspeed import unset_hf_deepspeed_config
    unset_hf_deepspeed_config()
    geometry=selected_admission_geometry(audited)
    path=REPOSITORY/'checkpoints'/args.run_id/'production'
    cli=['--stage','stage1','--model',str(args.model),'--data_path',str(args.data_path),
        '--image_root',str(args.image_root),'--save_model_path',str(path),
        '--stage1_variant','v5','--stage1_v5_debug','--stage1_loss_type','dil-bce-orig',
        '--tuning_method','dora','--lora_r','32','--lora_alpha','64','--lora_dropout','0.05',
        '--lora_target_modules','q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
        '--latent_size','8','--ema_tau','1','--ce_weight','0.1','--sim_weight','0.1',
        '--lambda_preference','1','--lambda_topk_entropy','0.1','--preference_topk_k','50',
        '--preference_topk_mode','independent','--lambda_topk_mass','0',
        '--stage1_v5_selection','all','--stage1_v5_text_kl_weight','0.01',
        '--step_preference_beta','1','--step_preference_margin','0','--loss_chunk_positions','128',
        '--per_device_train_batch_size','1','--gradient_accumulation_steps','4','--epochs','1',
        '--max_steps',str(audited['optimizer_updates_for_one_exposure']),'--learning_rate','5e-5','--warm_up_steps','10','--seed','42',
        '--attention_implementation','sdpa','--vision_attention_implementation','flash_attention_2',
        '--gradient_checkpointing','--textual_states','previous','--save_strategy','no',
        '--deepspeed_config',str(args.deepspeed)]
    parsed=parse_args(cli)
    components=build_stage1_training_components(parsed)
    trainer=components.trainer
    validate_audit_reference(audited,trainer.v5_reference_identity)
    # Observe original retained loader/profile before any diagnostic substitution.
    full_loader=trainer.get_train_dataloader()
    full_contract=trainer._effective_v5_contract(trainer._build_v5_runtime_contract(full_loader))
    validate_contract_profile(full_contract,audited)
    binding=dict(contract=full_contract,original_loader=observe_original_loader(trainer,full_loader,full_contract),
        input_identity=dict(data_sha256=_sha(args.data_path),
            base_content_sha256=trainer.v5_reference_identity.base_content_sha256,
            processor_sha256=trainer.v5_reference_identity.processor_sha256,
            source_sha256={path:_sha(REPOSITORY/path) for path in audited['code_identity']['files']}))
    validate_runtime_binding(audited,binding)
    del full_loader
    # Fixture sampler: each source repeated sixteen times, four microsteps on
    # every rank. Every actual retained maximum is executed unchanged.
    by_index={e.source_index:e for e in components.dataset}
    trainer.train_dataset=[by_index[g['source_index']] for g in geometry for _ in range(16)]
    from torch.utils.data import SequentialSampler
    trainer._get_train_sampler=lambda:SequentialSampler(trainer.train_dataset)
    trainer.args.max_steps=len(geometry)
    observation=observe_production_trainer(trainer,expected_geometry=geometry)
    hooks=observation['hooks']; observed=observation['source_indices']; observed_geometry=observation['geometry']; handles=observation['handles']
    before=_state_identity(trainer.ema_teacher,trainer.ema_teacher.state_dict())
    torch.cuda.reset_peak_memory_stats()
    output=trainer.train()
    after=_state_identity(trainer.ema_teacher,trainer.ema_teacher.state_dict())
    for h in handles: h.remove()
    contract=trainer._effective_v5_contract(trainer._v5_contract)
    result=dict(hooks=dict(hooks),reference_before=before,reference_after=after,
        runtime_binding=binding,fixture_contract=contract,contract_digest=_contract_digest(contract),
        production=dict(global_step=trainer.state.global_step,microbatches=hooks['backward'],finite_loss=math.isfinite(output.training_loss),
            gradient_norms=observation['gradient_norms'],gradient_norm_steps=observation['gradient_norm_steps'],source_windows=observation['source_windows'],optimizer_steps=observation['optimizer_steps'],
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            source_indices=sorted(observed),geometry=[observed_geometry[g['source_index']] for g in geometry]))
    del components,trainer
    return result


def selected_admission_geometry(audited):
    # Every actual non-dominated geometry is required; no coordinate-winner cap.
    rows=audited['max_packed_geometry']
    if not rows or len({r['source_index'] for r in rows})!=len(rows):
        raise ValueError('empty or duplicate admission geometry')
    for row in rows: _geometry_comparison(row)
    return rows


def _output_path(path):
    path=Path(path).absolute()
    if not path.resolve().is_relative_to((REPOSITORY/'results').resolve()):
        raise ValueError('metadata output must be under current repository results/')
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--mode',choices=('audit','distributed-witness','record-show'),required=True)
    parser.add_argument('--data-path',type=Path)
    parser.add_argument('--image-root',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--output-json',type=Path)
    parser.add_argument('--audit-json',type=Path)
    parser.add_argument('--deepspeed',type=Path)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--rank',type=int,choices=range(4))
    args=parser.parse_args(argv)
    from src.multinode import _RUN_ID_PATTERN
    if not _RUN_ID_PATTERN.fullmatch(args.run_id): parser.error('invalid run ID')
    try:
        if args.mode=='record-show':
            if args.rank is None: raise ValueError('rank is required')
            print((REPOSITORY/'results'/args.run_id/f'v5-rank-{args.rank}.json').read_text())
            return 0
        if any(v is None for v in (args.data_path,args.image_root,args.model,args.output_json)):
            raise ValueError('data-path, image-root, model and output-json required')
        output=_output_path(args.output_json)
        if args.mode=='audit':
            result=audit(args)
        else:
            require_distributed_entry()
            if args.audit_json is None or args.deepspeed is None: raise ValueError('audit-json and deepspeed required')
            audited=json.loads(args.audit_json.read_text())
            if audited['source_identity']['data_sha256']!=_sha(args.data_path): raise ValueError('audit data changed')
            for path,digest in audited['code_identity']['files'].items():
                if _sha(REPOSITORY/path)!=digest: raise ValueError(f'audit code changed: {path}')
            import torch
            import torch.distributed as dist
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            if not dist.is_initialized(): dist.init_process_group('nccl')
            record=dict(rank=dist.get_rank(),world_size=dist.get_world_size(),run_id=args.run_id,entry=True,exit=False)
            record.update(tiny_engine_witness(REPOSITORY/'checkpoints'/args.run_id/'tiny',deepspeed=str(args.deepspeed)))
            from transformers.integrations.deepspeed import unset_hf_deepspeed_config
            unset_hf_deepspeed_config()
            import gc
            gc.collect();torch.cuda.empty_cache()
            record.update(production_witness(args,audited)); record['exit']=True
            records=[None]*4;dist.all_gather_object(records,record)
            result=validate_rank_records(records,run_id=args.run_id,expected_geometry=selected_admission_geometry(audited),audited=audited)
            result['records']=records
            output.parent.mkdir(parents=True,exist_ok=True)
            from src.multinode_runtime import _publish_bytes_no_replace
            _publish_bytes_no_replace(output.parent/f'v5-rank-{record["rank"]}.json',json.dumps(record,sort_keys=True,allow_nan=False).encode())
            if int(os.environ['LOCAL_RANK'])!=0: return 0
        output.parent.mkdir(parents=True,exist_ok=True)
        from src.multinode_runtime import _publish_bytes_no_replace
        _publish_bytes_no_replace(output,json.dumps(result,sort_keys=True,indent=2,allow_nan=False).encode())
        print(json.dumps(dict(output_json=str(output),mode=args.mode)))
        return 0
    except (ValueError,RuntimeError,OSError) as error:
        parser.error(str(error))
    return 2


if __name__=='__main__': raise SystemExit(main())
