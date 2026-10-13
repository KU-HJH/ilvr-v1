"""Versioned v3 objective, effective profile and actual-loader schedule identity."""
from __future__ import annotations
from dataclasses import asdict,dataclass
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import torch
from .v3_config import Stage1V3Config

FILENAME='stage1_v3.json'
ROOT=Path(__file__).resolve().parents[2]
V3_FRESH_PROFILE=dict(model='Qwen/Qwen2.5-VL-3B-Instruct',tuning_method='dora',
    per_device_train_batch_size=1,gradient_accumulation_steps=4,epochs=1.,learning_rate=5e-5,
    seed=42,warm_up_steps=10,ce_weight=.1,sim_weight=1.,ema_tau=1.,latent_size=8,
    lora_r=32,lora_alpha=64,lora_dropout=.05,
    lora_target_modules='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
    attention_implementation='sdpa',vision_attention_implementation='flash_attention_2',gradient_checkpointing=True,
    deepspeed_config='configs/deepspeed_zero1.json',max_steps=-1,
    max_seq_length_train=32768,textual_states='previous')
FORMULAS=dict(selector='T2_KL_student_reference_sample_centered_full_causal_v1',
    masks='independent_R_scored_text_RV_all_suffix_latent_body_v1',
    gain='global_block_mean_project_global_dot_fp64_post_adam_fp32_v1',
    text_kl='T2_KL_detached_F_R_query_mean_sample_mean_v1',
    warmup='completed_step_over_ceil_005_planned_v1',
    ordinary='configured_accumulation_rank_denominator_including_tail_v1',
    dropout='original_sample_step_site_seed_v1')


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _formula_ids(config):
    formulas=dict(FORMULAS)
    if config.text_selector=='entropy':
        formulas['selector']='T2_entropy_student_sample_centered_full_causal_v1'
    if config.gain_weighting=='token_entropy':
        formulas['gain']='F_T1_detached_H_logV_token_gap_block_mean_raw_RV_project_fp64_post_adam_fp32_v1'
    if config.gain_update_mode=='adamw_projected':
        formulas['gain']='global_block_mean_same_state_adamw_joint_moments_active_lr_project_fp64_fp32_master_v1'
        if config.gain_weighting=='token_entropy':
            formulas['gain']='F_T1_detached_H_logV_token_gap_'+formulas['gain']
        formulas['gain_alpha_role']='inactive_legacy_displacement_v1'
    return formulas


def _contract_identity(value):
    # Only additive mode fields have legacy defaults; never fill other
    # missing fields or mutate/rewrite the original saved contract.
    normalized=dict(value,config=dict(value['config']))
    normalized['config'].setdefault('text_selector','kl')
    normalized['config'].setdefault('gain_weighting','none')
    normalized['config'].setdefault('gain_update_mode','post_adam')
    normalized['config'].setdefault('gain_optimizer_weight',None)
    return _canonical(normalized)


def _resolve_gain_update(saved_config: Stage1V3Config | None, mode: str | None,
                         weight: float | None) -> tuple[str, float | None]:
    if saved_config is not None:
        if ((mode is not None and mode != saved_config.gain_update_mode)
                or (weight is not None and weight != saved_config.gain_optimizer_weight)):
            raise ValueError('v3 resume Gain optimizer mismatch; changing modes/weights requires a fresh run')
        mode = saved_config.gain_update_mode if mode is None else mode
        weight = saved_config.gain_optimizer_weight if weight is None else weight
    config = Stage1V3Config(gain_update_mode='post_adam' if mode is None else mode,
                           gain_optimizer_weight=weight)
    return config.gain_update_mode, config.gain_optimizer_weight


@dataclass(frozen=True)
class V3ScheduleIdentity:
    retained_ids_sha256: str
    sampler_identity: dict[str,object]
    microsteps_per_rank: int
    accumulation_steps: int
    updates_per_epoch: int
    planned_steps: int
    remainder_microsteps: int


def build_v3_schedule_identity(*,retained_ids: tuple[str,...],sampler_identity: Mapping[str,object],
                               microsteps_per_rank: int,accumulation_steps: int) -> V3ScheduleIdentity:
    if (type(microsteps_per_rank) is not int or microsteps_per_rank<=0 or type(accumulation_steps) is not int
            or accumulation_steps<=0 or not retained_ids or any(type(i) is not str or not i for i in retained_ids)
            or len(set(retained_ids))!=len(retained_ids)):
        raise ValueError('invalid v3 retained identity or positive integer update budget')
    steps=math.ceil(microsteps_per_rank/accumulation_steps)
    sampler=json.loads(_canonical(dict(sampler_identity)))
    return V3ScheduleIdentity(hashlib.sha256(_canonical(retained_ids)).hexdigest(),sampler,
        microsteps_per_rank,accumulation_steps,steps,steps,microsteps_per_rank%accumulation_steps)


def build_v3_checkpoint_contract(config: Stage1V3Config,schedule: V3ScheduleIdentity,
                                 resolved_profile: Mapping[str,object]) -> dict[str,object]:
    profile=json.loads(_canonical(dict(resolved_profile)))
    teacher=profile.get('frozen_teacher_sha256')
    if not isinstance(teacher,str) or len(teacher)!=64 or any(c not in '0123456789abcdef' for c in teacher):
        raise ValueError('v3 checkpoint requires frozen teacher content identity')
    return dict(schema_version=1,variant='v3',config=config.to_dict(),formula_ids=_formula_ids(config),
                schedule=asdict(schedule),planned_steps=schedule.planned_steps,profile=profile,
                frozen_teacher_sha256=teacher,boundary='complete')


def _validate_contract(value):
    fields={'schema_version','variant','config','formula_ids','schedule','planned_steps','profile','frozen_teacher_sha256','boundary'}
    if (not isinstance(value,Mapping) or set(value)!=fields or type(value['schema_version']) is not int
            or value['schema_version']!=1 or value['variant']!='v3' or value['boundary']!='complete'):
        raise ValueError('unsupported or incomplete v3 resume identity')
    try:
        config=Stage1V3Config(**value['config'])
        schedule=V3ScheduleIdentity(**value['schedule'])
    except (TypeError, ValueError) as error:
        raise ValueError('invalid v3 resume config/schedule') from error
    if value['formula_ids']!=_formula_ids(config):
        raise ValueError('v3 resume selector/formula mismatch')
    if (not isinstance(schedule.retained_ids_sha256,str) or len(schedule.retained_ids_sha256)!=64
            or any(c not in '0123456789abcdef' for c in schedule.retained_ids_sha256)
            or type(schedule.sampler_identity) is not dict
            or type(schedule.remainder_microsteps) is not int):
        raise ValueError('invalid v3 resume schedule identity')
    numeric=(schedule.microsteps_per_rank,schedule.accumulation_steps,schedule.updates_per_epoch,schedule.planned_steps)
    if any(type(v) is not int or v<=0 for v in numeric):raise ValueError('invalid v3 resume schedule')
    if (schedule.updates_per_epoch!=math.ceil(schedule.microsteps_per_rank/schedule.accumulation_steps)
            or schedule.planned_steps!=schedule.updates_per_epoch
            or schedule.remainder_microsteps!=schedule.microsteps_per_rank%schedule.accumulation_steps):
        raise ValueError('inconsistent v3 resume remainder/budget')
    if _contract_identity(value)!=_canonical(build_v3_checkpoint_contract(config,schedule,value['profile'])):
        raise ValueError('inconsistent v3 resume contract')


def validate_v3_resume(saved: Mapping[str,object],requested: Mapping[str,object]) -> None:
    _validate_contract(saved);_validate_contract(requested)
    if _contract_identity(saved)!=_contract_identity(requested):raise ValueError('v3 resume objective/profile/schedule mismatch')


def read_v3_contract(checkpoint: str|Path):
    path=Path(checkpoint)/FILENAME
    if path.is_symlink() or not path.is_file() or path.stat().st_size>1024*1024:
        raise ValueError('v3 resume contract must be a bounded regular file')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate v3 resume key')
            result[key]=value
        return result
    value=json.loads(path.read_bytes(),object_pairs_hook=pairs)
    _validate_contract(value)
    return value


def resolve_v3_arguments(args,raw_argv):
    explicit={item.split('=',1)[0][2:] for item in raw_argv if item.startswith('--')}
    args.stage1_v3_explicit=explicit
    variant=getattr(args,'stage1_variant',None)
    selector=getattr(args,'stage1_v3_selector',None)
    weighting=getattr(args,'stage1_v3_gain_weighting',None)
    if args.stage!='stage1':
        if variant is not None:raise ValueError('stage1_variant is only valid for Stage 1')
        if selector is not None:raise ValueError('stage1_v3_selector is only valid for Stage1-v3')
        if weighting is not None:raise ValueError('stage1_v3_gain_weighting is only valid for Stage1-v3')
        return
    checkpoint=getattr(args,'resume_from_checkpoint_path',None)
    if checkpoint is None:
        output=Path(args.save_model_path)
        candidates=[p for p in output.glob('checkpoint-*') if p.is_dir() and p.name[11:].isdigit()]
        if candidates:checkpoint=max(candidates,key=lambda p:int(p.name[11:]))
    saved=read_v3_contract(checkpoint) if checkpoint and (Path(checkpoint)/FILENAME).exists() else None
    inherited='v3' if saved is not None else 'standard'
    if checkpoint and variant is not None and variant!=inherited:
        raise ValueError('stage1 variant resume mismatch; historical checkpoints cannot become v3')
    args.stage1_variant=variant or inherited
    args.stage1_v3_saved_contract=saved
    if args.stage1_variant!='v3':
        if selector is not None:raise ValueError('stage1_v3_selector is only valid for Stage1-v3')
        if weighting is not None:raise ValueError('stage1_v3_gain_weighting is only valid for Stage1-v3')
        return
    config=Stage1V3Config(**saved['config']) if saved else Stage1V3Config()
    saved_selector=config.text_selector
    if saved and selector is not None and selector!=saved_selector:
        raise ValueError('v3 resume text selector mismatch; changing modes requires a fresh run')
    args.stage1_v3_selector=selector or saved_selector
    if saved and weighting is not None and weighting!=config.gain_weighting:
        raise ValueError('v3 resume Gain weighting mismatch; changing modes requires a fresh run')
    args.stage1_v3_gain_weighting=weighting or config.gain_weighting
    profile=dict(V3_FRESH_PROFILE) if saved is None else dict(saved['profile'])
    # Older v3 checkpoints used one backend for both towers. Preserve their
    # execution and serialized identity rather than applying the fresh default.
    if saved and 'vision_attention_implementation' not in profile:
        profile['vision_attention_implementation']=profile.get('attention_implementation')
    for key in V3_FRESH_PROFILE:
        if key not in profile:raise ValueError('v3 resume profile missing '+key)
        value=profile[key]
        current=getattr(args,key,None)
        if key=='model':
            # Metadata identifies the original base; each node may use its local cache.
            if current and Path(current).name=='Qwen2.5-VL-3B-Instruct':value=current
        if key=='deepspeed_config':
            if saved and key in explicit and Path(str(current)).name!=Path(str(value)).name:
                raise ValueError('v3 resume deepspeed_config mismatch')
            value=str(ROOT/'configs'/Path(value).name)
        elif saved and key in explicit and current!=value:
            raise ValueError('v3 resume '+key+' mismatch')
        setattr(args,key,value)
    args.tuning_method_explicit=True
    args.use_lora=False


def retained_source_ids(dataset):
    result=[]
    for i,example in enumerate(dataset):
        index=getattr(example,'source_index',i)
        source=getattr(example,'source_id',None)
        result.append(f'{index}:{source if source is not None else index}')
    return tuple(result)


def prepared_sampler_identity(dataloader,world_size):
    batch_sampler=dataloader.batch_sampler
    sampler=getattr(batch_sampler,'sampler',None)
    inner=getattr(batch_sampler,'batch_sampler',None)
    if sampler is None and inner is not None:sampler=getattr(inner,'sampler',None)
    return dict(world_size=world_size,prepared_loader=type(dataloader).__qualname__,
        batch_sampler=type(batch_sampler).__qualname__,sampler=type(sampler).__qualname__,
        batch_size=getattr(batch_sampler,'batch_size',None),drop_last=bool(getattr(batch_sampler,'drop_last',False)),
        even_batches=bool(getattr(batch_sampler,'even_batches',False)),
        split_batches=bool(getattr(batch_sampler,'split_batches',False)),
        seed=getattr(sampler,'initial_seed',None),replacement=bool(getattr(sampler,'replacement',False)))


def teacher_digest(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state['parameters'].items()):
        value=tensor.detach().cpu().contiguous()
        digest.update(_canonical((name,str(value.dtype),tuple(value.shape))))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_v3_contract(directory,contract):
    _validate_contract(contract)
    from src.multinode_runtime import _publish_bytes_no_replace
    path=Path(directory).resolve()/FILENAME
    raw=_canonical(contract)+b'\n'
    if path.exists():
        if path.is_symlink():raise ValueError('refusing to replace different v3 checkpoint contract')
        if path.read_bytes()!=raw:
            existing=read_v3_contract(path.parent)
            if 'text_selector' in existing['config'] and 'gain_weighting' in existing['config']:
                raise ValueError('refusing to replace different v3 checkpoint contract')
            validate_v3_resume(existing,contract)
            # Equivalent additive legacy metadata is already authoritative. Leave
            # its original bytes intact when saving to the same final output.
        return
    _publish_bytes_no_replace(path,raw)


def attach_v3_identity(trainer,args,dataset,checkpoint=None):
    from importlib.metadata import version
    from peft_common import resolve_qwen2_5_vl_model
    qwen=resolve_qwen2_5_vl_model(trainer.model)
    saved=getattr(args,'stage1_v3_saved_contract',None)
    profile={name:getattr(args,name) for name in V3_FRESH_PROFILE}
    if saved and 'vision_attention_implementation' not in saved['profile']:
        profile.pop('vision_attention_implementation')
    profile['model']=V3_FRESH_PROFILE['model']
    profile['deepspeed_config']='configs/'+Path(args.deepspeed_config).name
    profile.update(world_size=4,effective_global_batch=16,bf16=True,weight_decay=.01,
        max_grad_norm=1.,optimizer='AdamW',scheduler='linear',helper_noise_std=.01,
        zero_stage=1,offload=False,data_workers=0,non_reentrant_checkpoint=True,
        vision_backend=type(qwen.visual.blocks[0].attn).__name__,
        packages={name:version(name) for name in ('torch','transformers','peft','trl','accelerate','deepspeed')})
    if checkpoint is None:
        profile['frozen_teacher_sha256']=teacher_digest(trainer.ema_teacher.state_dict())
    else:
        teacher=torch.load(Path(checkpoint)/'stage1_ema_teacher.pt',map_location='cpu',weights_only=True)
        profile['frozen_teacher_sha256']=teacher_digest(teacher['state'])
        if saved is None or profile!=saved['profile']:
            raise ValueError('v3 resume execution/teacher profile mismatch')
    trainer.v3_resolved_profile=profile
    trainer.v3_retained_ids=retained_source_ids(dataset)
    trainer.v3_saved_contract=saved
    trainer.stage1_variant='v3'



LAUNCHER_ALIASES={'--method':'tuning_method','--model':'model','--epochs':'epochs','--learning-rate':'learning_rate',
    '--ce-weight':'ce_weight','--gradient-accumulation':'gradient_accumulation_steps','--micro-batch':'per_device_train_batch_size',
    '--ema-tau':'ema_tau','--seed':'seed','--deepspeed-config':'deepspeed_config','--attention-implementation':'attention_implementation',
    '--textual_states':'textual_states','--max-steps':'max_steps','--stage1-v3-selector':'stage1_v3_selector',
    '--stage1-v3-gain-weighting':'stage1_v3_gain_weighting'}


def resolve_coordinator_arguments(arguments,output):
    """Use the same effective profile for worker argv, plan and result validation."""
    import os
    from utils_deepseed import parse_args
    from src.multinode import _argument_value
    values={option:value for option in (*LAUNCHER_ALIASES,'--stage1-variant','--resume')
            if (value:=_argument_value(tuple(arguments),option)) is not None}
    variant=values.get('--stage1-variant') or os.environ.get('STAGE1_VARIANT')
    checkpoint=values.get('--resume')
    has_saved=checkpoint is not None and (Path(checkpoint)/FILENAME).is_file()
    if variant!='v3' and not has_saved:
        if '--stage1-v3-selector' in values:raise ValueError('stage1-v3-selector is only valid for Stage1-v3')
        if '--stage1-v3-gain-weighting' in values:raise ValueError('stage1-v3-gain-weighting is only valid for Stage1-v3')
        return arguments
    argv=['--stage','stage1','--save_model_path',str(output)]
    if variant:argv+=['--stage1_variant',variant]
    if checkpoint:argv+=['--resume_from_checkpoint_path',checkpoint]
    for option,key in LAUNCHER_ALIASES.items():
        if option in values:argv+=['--'+key,values[option]]
    args=parse_args(argv)
    if args.stage1_variant!='v3':return arguments
    from src.multinode import _replace_argument_value
    # Coordinator validation accepts --flag=value; workers accept separate
    # tokens. Normalize only this opt-in route, preserving literal value text.
    result=tuple(part for token in arguments for part in
                 (token.split('=',1) if token.startswith('--') and '=' in token else (token,)))
    def upsert(items,option,value):
        return (_replace_argument_value(items,option,value) if option in items
                else (*items,option,value))
    result=upsert(result,'--stage1-variant','v3')
    for option,key in LAUNCHER_ALIASES.items():
        if option in ('--max-steps','--deepspeed-config','--textual_states'):continue
        result=upsert(result,option,str(getattr(args,key)))
    if '--max-steps' in result:
        i=result.index('--max-steps');result=result[:i]+result[i+2:]
    return result


def _launcher_main():
    import argparse
    from utils_deepseed import parse_args
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',default='')
    parser.add_argument('--resume',default='')
    parser.add_argument('--output',required=True)
    parser.add_argument('original',nargs=argparse.REMAINDER)
    options=parser.parse_args()
    argv=['--stage','stage1','--save_model_path',options.output]
    if options.variant:argv+=['--stage1_variant',options.variant]
    if options.resume:argv+=['--resume_from_checkpoint_path',options.resume]
    for i,item in enumerate(options.original):
        if item in LAUNCHER_ALIASES:
            argv+=['--'+LAUNCHER_ALIASES[item],options.original[i+1]]
    args=parse_args(argv)
    values={'stage1_variant':args.stage1_variant}
    if args.stage1_variant=='v3':
        values.update({name:getattr(args,name) for name in V3_FRESH_PROFILE})
        values['stage1_v3_selector']=args.stage1_v3_selector
        values['stage1_v3_gain_weighting']=args.stage1_v3_gain_weighting
    import sys
    for name,value in values.items():
        sys.stdout.write(name+'\0'+(str(value).lower() if isinstance(value,bool) else str(value))+'\0')


if __name__=='__main__':_launcher_main()
