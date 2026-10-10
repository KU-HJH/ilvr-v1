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
    assert Path(args.deepspeed_config).name=='deepspeed_zero1.json'
    assert (args.tuning_method,args.lora_r,args.lora_alpha,args.lora_dropout,args.latent_size)==('dora',32,64,.05,8)
    assert args.model.endswith('Qwen2.5-VL-3B-Instruct')


def saved(tmp_path):
    module=importlib.import_module('stage1.v3_resume')
    profile=dict(module.V3_FRESH_PROFILE,frozen_teacher_sha256='a'*64)
    # A supported saved coefficient is authoritative on resume.
    profile['ce_weight']=.2
    schedule=module.build_v3_schedule_identity(retained_ids=('a','b'),sampler_identity={'world_size':4},
        microsteps_per_rank=6,accumulation_steps=4)
    contract=module.build_v3_checkpoint_contract(Stage1V3Config(),schedule,profile)
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
        ILVR_NUM_MACHINES='2',ILVR_NUM_PROCESSES='4',ILVR_MACHINE_RANK='1',ILVR_MAIN_PROCESS_IP='10.1.102.41',
        ILVR_MAIN_PROCESS_PORT='29579',ILVR_SOCKET_IFNAME='bond1.102')
    result=subprocess.run([str(ROOT/'run_stage1_training.sh'),'--dry-run','--method','dora','--model','models/Qwen2.5-VL-3B-Instruct',
        '--data-path',str(ROOT/'data/TRAIN.jsonl'),'--image-root',str(ROOT/'data/images'),
        '--output-dir',str(ROOT/'checkpoints/v3-profile-test')],env=env,cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    words=shlex.split(result.stdout)
    for key,value in {'stage1_variant':'v3','epochs':'1','ce_weight':'0.1','ema_tau':'1.0',
                      'gradient_accumulation_steps':'4','attention_implementation':'sdpa'}.items():
        assert words[words.index('--'+key)+1]==value
    assert Path(words[words.index('--deepspeed_config')+1]).name=='deepspeed_zero1.json'


def test_composition_passes_opt_in_metadata_objective_and_identity(tmp_path,monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from tests.stage1.test_composition import stage1_args,fake_dependencies
    import stage1.composition as composition
    import stage1.v3_resume as resume
    args=stage1_args(tmp_path,'--stage1_variant','v3')
    events=[];deps,_,_,_=fake_dependencies(events)
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
    assert seen['trainer']['ce_weight']==.1 and seen['trainer']['ema_tau']==1.
    assert seen['identity'][1] is args
