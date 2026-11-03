import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import pytest
from stage1.v3_config import Stage1V3Config
from utils_deepseed import parse_args

ROOT=Path(__file__).resolve().parents[2]


def test_fresh_profile_overrides_stale_coefficients_and_execution_values():
    args=parse_args(['--stage','stage1','--stage1_variant','v3','--ce_weight','1',
        '--ema_tau','.999','--epochs','3','--gradient_accumulation_steps','2','--learning_rate','1e-4',
        '--attention_implementation','flash_attention_2','--deepspeed_config','configs/deepspeed_zero3.json'])
    assert args.stage1_variant=='v3'
    assert (args.ce_weight,args.ema_tau,args.epochs,args.gradient_accumulation_steps)==(.1,1.,1.,4)
    assert args.learning_rate==5e-5 and args.warm_up_steps==10 and args.seed==42
    assert args.attention_implementation=='sdpa' and args.gradient_checkpointing
    assert args.vision_attention_implementation=='flash_attention_2'
    assert Path(args.deepspeed_config).name=='deepspeed_zero1.json'
    assert (args.tuning_method,args.lora_r,args.lora_alpha,args.lora_dropout,args.latent_size)==('dora',32,64,.05,8)
    assert args.model.endswith('Qwen2.5-VL-3B-Instruct')


def saved(tmp_path,selector='kl',legacy=False):
    module=importlib.import_module('stage1.v3_resume')
    profile=dict(module.V3_FRESH_PROFILE,frozen_teacher_sha256='a'*64)
    # A supported saved coefficient is authoritative on resume.
    profile['ce_weight']=.2
    schedule=module.build_v3_schedule_identity(retained_ids=('a','b'),sampler_identity={'world_size':4},
        microsteps_per_rank=6,accumulation_steps=4)
    contract=module.build_v3_checkpoint_contract(Stage1V3Config(text_selector=selector),schedule,profile)
    if legacy:contract['config'].pop('text_selector')
    path=tmp_path/'checkpoint-1';path.mkdir()
    (path/'stage1_v3.json').write_text(json.dumps(contract))
    return path,contract


def test_omitted_resume_inherits_saved_and_explicit_conflicts_fail(tmp_path):
    checkpoint,contract=saved(tmp_path)
    args=parse_args(['--stage','stage1','--resume_from_checkpoint_path',str(checkpoint)])
    assert args.stage1_variant=='v3' and args.ce_weight==.2
    assert args.stage1_v3_saved_contract==contract
    for extra in (['--stage1_variant','standard'],['--ce_weight','.1'],['--epochs','3']):
        with pytest.raises(ValueError,match='resume|mismatch'):
            parse_args(['--stage','stage1','--resume_from_checkpoint_path',str(checkpoint),*extra])


def test_historical_checkpoint_cannot_be_reinterpreted_as_v3(tmp_path):
    with pytest.raises(ValueError,match='resume|historical'):
        parse_args(['--stage1_variant','v3','--resume_from_checkpoint_path',str(tmp_path)])


def test_standard_omission_does_not_change_historical_defaults():
    args=parse_args([])
    assert args.stage1_variant=='standard' and args.epochs==2 and args.ema_tau==.999


def test_coordinator_allowlist_and_launcher_enforce_fresh_profile_without_writes():
    from src.multinode_runtime import validate_stage1_arguments
    assert validate_stage1_arguments(('--stage1-variant','v3'))==('--stage1-variant','v3')
    env=dict(os.environ,STAGE1_VARIANT='v3',STAGE1_EPOCHS='3',STAGE1_CE_WEIGHT='1',STAGE1_EMA_TAU='.999',
        STAGE1_GRADIENT_ACCUMULATION='2',STAGE1_DEEPSPEED_CONFIG='configs/deepspeed_zero3.json',
        ILVR_NUM_MACHINES='2',ILVR_NUM_PROCESSES='4',ILVR_MACHINE_RANK='1',ILVR_MAIN_PROCESS_IP='172.16.11.41',
        ILVR_MAIN_PROCESS_PORT='29579',ILVR_SOCKET_IFNAME='bond0.11')
    result=subprocess.run([str(ROOT/'run_stage1_training.sh'),'--dry-run','--method','dora','--model','models/Qwen2.5-VL-3B-Instruct',
        '--data-path',str(ROOT/'data/TRAIN.jsonl'),'--image-root',str(ROOT/'data/images'),
        '--output-dir',str(ROOT/'checkpoints/v3-profile-test')],env=env,cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    words=shlex.split(result.stdout)
    for key,value in {'stage1_variant':'v3','epochs':'1','ce_weight':'0.1','ema_tau':'1.0',
                      'gradient_accumulation_steps':'4','attention_implementation':'sdpa',
                      'vision_attention_implementation':'flash_attention_2'}.items():
        assert words[words.index('--'+key)+1]==value
    assert Path(words[words.index('--deepspeed_config')+1]).name=='deepspeed_zero1.json'


@pytest.mark.parametrize('selector',['kl','entropy'])
@pytest.mark.parametrize('weighting',['none','token_entropy'])
@pytest.mark.parametrize('update_mode',['post_adam','adamw_projected'])
@pytest.mark.parametrize('stop',[None,2])
def test_composition_passes_opt_in_metadata_objective_and_identity(tmp_path,monkeypatch,selector,weighting,update_mode,stop):
    from dataclasses import replace
    from types import SimpleNamespace
    from tests.stage1.test_composition import stage1_args,fake_dependencies
    import stage1.composition as composition
    import stage1.v3_resume as resume
    extra=['--stage1_v3_gain_update_mode',update_mode]
    if stop is not None:extra+=['--stage1_v3_stop_after_checkpoint',str(stop),'--no_final_save']
    if update_mode=='adamw_projected':extra+=['--stage1_v3_gain_optimizer_weight','0.25']
    args=stage1_args(tmp_path,'--stage1_variant','v3','--stage1_v3_selector',selector,'--stage1_v3_gain_weighting',weighting,*extra)
    events=[];deps,_,_,_=fake_dependencies(events)
    # This fixture isolates composition metadata, not CUDA availability. The
    # separate real-Qwen regression authenticates the concrete backend pair.
    original_config=deps.load_config
    def load_config(*a,**kw):
        config=original_config(*a,**kw);config.vision_config=SimpleNamespace()
        return config
    monkeypatch.setattr(composition.Qwen2_5_VLForConditionalGeneration,
        '_check_and_enable_flash_attn_2',lambda config,**kw:config)
    deps=replace(deps,load_config=load_config)
    seen={}
    def collator(*a,**kw):seen['collator']=kw;return object()
    original=deps.trainer_factory
    def trainer(**kw):seen['trainer']=kw;return original(**kw)
    monkeypatch.setattr(resume,'attach_v3_identity',lambda *a:seen.setdefault('identity',a))
    monkeypatch.setattr(composition,'build_stage1_training_arguments',lambda args:SimpleNamespace(output_dir=args.save_model_path))
    composition.build_stage1_training_components(args,
        dependencies=replace(deps,collator_factory=collator,trainer_factory=trainer),
        deepspeed_runtime_profile=__import__('deepspeed_runtime').load_deepspeed_runtime_profile(args.deepspeed_config,world_size=4))
    assert seen['collator']=={'v3_enabled':True}
    assert isinstance(seen['trainer']['v3_config'],Stage1V3Config)
    assert seen['trainer']['v3_config'].text_selector==selector
    assert seen['trainer']['v3_config'].gain_weighting==weighting
    assert seen['trainer']['v3_config'].gain_update_mode==update_mode
    assert seen['trainer']['v3_config'].gain_optimizer_weight==(.25 if update_mode=='adamw_projected' else None)
    assert seen['trainer']['v3_config'].gain_entropy_mode=='raw'
    assert seen['trainer']['v3_config'].text_kl_warmup_ratio==0.
    assert seen['trainer']['sim_weight']==.1
    assert seen['trainer']['ce_weight']==.1 and seen['trainer']['ema_tau']==1.
    assert seen['identity'][1] is args
    assert seen['trainer'].get('stage1_v3_stop_after_checkpoint')==stop


def test_cli_default_and_explicit_entropy_preserve_training_profile():
    default=parse_args(['--stage1_variant','v3'])
    entropy=parse_args(['--stage1_variant','v3','--stage1_v3_selector','entropy'])
    assert default.stage1_v3_selector=='kl' and entropy.stage1_v3_selector=='entropy'
    for name in importlib.import_module('stage1.v3_resume').V3_FRESH_PROFILE:
        assert getattr(default,name)==getattr(entropy,name)


@pytest.mark.parametrize('extra', [[],['--stage1_variant','standard'],['--stage','stage2']])
def test_selector_is_rejected_outside_v3(extra):
    with pytest.raises(ValueError,match='selector|Stage1-v3'):
        parse_args([*extra,'--stage1_v3_selector','entropy'])


@pytest.mark.parametrize('selector,legacy', [('kl',False),('entropy',False),('kl',True)])
def test_resume_inherits_selector_and_rejects_explicit_mode_switch(tmp_path,selector,legacy):
    checkpoint,contract=saved(tmp_path,selector,legacy)
    common=['--resume_from_checkpoint_path',str(checkpoint)]
    args=parse_args(common)
    assert args.stage1_v3_selector==selector and args.stage1_v3_saved_contract==contract
    assert parse_args([*common,'--stage1_v3_selector',selector]).stage1_v3_selector==selector
    opposite='kl' if selector=='entropy' else 'entropy'
    with pytest.raises(ValueError,match='selector|mismatch'):
        parse_args([*common,'--stage1_v3_selector',opposite])


@pytest.mark.parametrize('selector,resumed', [('kl',False),('entropy',False),('entropy',True)])
def test_launcher_passes_selector_to_worker_argv(tmp_path,selector,resumed):
    extra=['--stage1-variant','v3','--stage1-v3-selector',selector]
    if resumed:
        checkpoint,_=saved(tmp_path,selector)
        extra=['--resume',str(checkpoint)]
    env=dict(os.environ,ILVR_NUM_MACHINES='2',ILVR_NUM_PROCESSES='4',ILVR_MACHINE_RANK='1',
        ILVR_MAIN_PROCESS_IP='172.16.11.41',ILVR_MAIN_PROCESS_PORT='29579',ILVR_SOCKET_IFNAME='bond0.11')
    env.pop('STAGE1_VARIANT',None)
    result=subprocess.run([str(ROOT/'run_stage1_training.sh'),'--dry-run','--method','dora',
        '--model','models/Qwen2.5-VL-3B-Instruct','--data-path',str(ROOT/'data/TRAIN.jsonl'),
        '--image-root',str(ROOT/'data/images'),'--output-dir',str(ROOT/'checkpoints/v3-entropy-dry'),*extra],
        env=env,cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    argv=shlex.split(result.stdout)
    assert argv[argv.index('--stage1_v3_selector')+1]==selector
    assert argv[argv.index('--stage1_variant')+1]=='v3'


@pytest.mark.parametrize('selector',['entropy','unknown'])
def test_launcher_rejects_selector_without_v3_before_launch(selector):
    result=subprocess.run([str(ROOT/'run_stage1_training.sh'),'--dry-run',
        '--stage1-v3-selector',selector],env={**os.environ,'STAGE1_VARIANT':'standard'},
        cwd=ROOT,text=True,capture_output=True)
    assert result.returncode!=0 and ('selector' in result.stderr or 'unrecognized' in result.stderr)


@pytest.mark.parametrize('inline',[False,True])
def test_coordinator_normalizes_selector_and_preserves_resume_mode(tmp_path,inline):
    from src.multinode_runtime import validate_stage1_arguments
    from stage1.v3_resume import resolve_coordinator_arguments
    flags=('--stage1-variant=v3','--stage1-v3-selector=entropy') if inline else (
        '--stage1-variant','v3','--stage1-v3-selector','entropy')
    resolved=resolve_coordinator_arguments(validate_stage1_arguments(flags),tmp_path/'out')
    assert resolved[resolved.index('--stage1-v3-selector')+1]=='entropy'
    checkpoint,_=saved(tmp_path,'entropy')
    resumed=resolve_coordinator_arguments(('--resume',str(checkpoint)),tmp_path/'out')
    assert resumed[resumed.index('--stage1-v3-selector')+1]=='entropy'
    with pytest.raises(ValueError,match='selector|mismatch'):
        resolve_coordinator_arguments(('--resume',str(checkpoint),'--stage1-v3-selector','kl'),tmp_path/'out')


def test_coordinator_rejects_selector_without_v3(tmp_path):
    from stage1.v3_resume import resolve_coordinator_arguments
    with pytest.raises(ValueError,match='selector|Stage1-v3'):
        resolve_coordinator_arguments(('--stage1-v3-selector','entropy'),tmp_path)


@pytest.mark.parametrize('legacy',[False,True])
def test_resume_preserves_saved_vision_and_rejects_backend_changes(tmp_path,legacy):
    checkpoint,contract=saved(tmp_path)
    if legacy:
        contract['profile'].pop('vision_attention_implementation',None)
        contract['profile']['vision_backend']='Qwen2_5_VLVisionSdpaAttention'
        (checkpoint/'stage1_v3.json').write_text(json.dumps(contract))
    common=['--resume_from_checkpoint_path',str(checkpoint)]
    expected='sdpa' if legacy else 'flash_attention_2'
    assert parse_args(common).vision_attention_implementation==expected
    opposite='flash_attention_2' if legacy else 'sdpa'
    with pytest.raises(ValueError,match='vision_attention_implementation.*mismatch'):
        parse_args([*common,'--vision_attention_implementation',opposite])


def test_legacy_vision_profile_identity_is_unchanged_when_reattached_and_saved(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import torch
    import peft_common
    module=importlib.import_module('stage1.v3_resume')
    teacher_state={'parameters':{'weight':torch.ones(1)}}
    trainer=SimpleNamespace(model=object(),ema_teacher=SimpleNamespace(state_dict=lambda:teacher_state))
    vision=type('Qwen2_5_VLVisionSdpaAttention',(),{})()
    monkeypatch.setattr(peft_common,'resolve_qwen2_5_vl_model',lambda model:
        SimpleNamespace(visual=SimpleNamespace(blocks=[SimpleNamespace(attn=vision)])))
    args=parse_args(['--stage1_variant','v3'])
    module.attach_v3_identity(trainer,args,['row'])
    profile=dict(trainer.v3_resolved_profile)
    profile.pop('vision_attention_implementation',None)
    schedule=module.build_v3_schedule_identity(retained_ids=('0:0',),sampler_identity={'world_size':4},
        microsteps_per_rank=1,accumulation_steps=4)
    contract=module.build_v3_checkpoint_contract(Stage1V3Config(),schedule,profile)
    checkpoint=tmp_path/'checkpoint-1';checkpoint.mkdir()
    module.write_v3_contract(checkpoint,contract)
    before=(checkpoint/module.FILENAME).read_bytes()
    torch.save({'state':teacher_state},checkpoint/'stage1_ema_teacher.pt')
    resumed=parse_args(['--resume_from_checkpoint_path',str(checkpoint)])
    module.attach_v3_identity(trainer,resumed,['row'],checkpoint)
    assert trainer.v3_resolved_profile==profile
    requested=module.build_v3_checkpoint_contract(Stage1V3Config(),schedule,trainer.v3_resolved_profile)
    module.validate_v3_resume(contract,requested)
    module.write_v3_contract(checkpoint,requested)
    assert (checkpoint/module.FILENAME).read_bytes()==before
