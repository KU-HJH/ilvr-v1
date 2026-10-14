"""Run the actual sequential shell recipe against cheap external boundaries."""
import os
from pathlib import Path
import shlex
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[3]
RECIPE=ROOT/'scripts/stage1_v5_train_eval.sh'
ARGS=['--run-id','v5-flow-test','--main-process-port','29683',
    '--remote-repo-root','/root/workspace/ILVR/.worktrees/v5 remote',
    '--remote-host-repo-root','/home/korea/Workspace/hjh/workspace/ILVR/.worktrees/v5 remote',
    '--max-steps','3']

def run(*args,env=None):
    return subprocess.run(['bash',str(RECIPE),*ARGS,*args],text=True,capture_output=True,env=env)

def test_recipe_dry_run_overrides_sourced_stale_profile(tmp_path):
    settings=tmp_path/'stale.sh'
    settings.write_text('export STAGE1_CE_WEIGHT=1 CE_WEIGHT=1 STAGE1_GRADIENT_ACCUMULATION=2 STAGE1_SEED=123 STAGE1_MODEL=trained/previous\n')
    done=run('--settings',str(settings),'--dry-run')
    assert done.returncode==0,done.stderr
    lines=[shlex.split(x) for x in done.stdout.splitlines()]
    assert len(lines)==3
    train,prepare,evaluate=lines
    for flag,value in [('--ce-weight','0.1'),('--sim-weight','0.1'),('--gradient-accumulation','4'),
        ('--seed','42'),('--warm-up-steps','10'),('--loss-chunk-positions','128'),
        ('--lambda-topk-entropy','0.1'),('--preference-topk-k','50'),('--stage1-variant','v5')]:
        assert train[train.index(flag)+1]==value
    assert train[train.index('--model')+1]=='/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct'
    assert '--resume' not in train
    assert prepare[prepare.index('--checkpoint-step')+1]=='3'
    assert Path(prepare[prepare.index('--destination')+1]).is_relative_to(ROOT/'checkpoints')
    assert evaluate[evaluate.index('--test-data-path')+1]==str(ROOT/'data/TEST.jsonl')
    for flag,value in [('--max-new-tokens','4096'),('--max-sequence-length','128000'),
        ('--repetition-penalty','1'),('--temperature','0'),('--top-p','1'),
        ('--generation-limit-policy','external_full_latent_block'),('--vision-ablation','none')]:
        assert evaluate[evaluate.index(flag)+1]==value
    assert evaluate[evaluate.index('--run-id')+1]=='v5-flow-test-test400'

@pytest.mark.parametrize('fail,events',[('', ['train_success','prepare_success','evaluate_TEST400']),
    ('train',['train_failure']),('prepare',['train_success','prepare_failure'])])
def test_recipe_only_evaluates_selected_successful_final_artifact(tmp_path,fail,events):
    # Copy the real recipe into an isolated checkout with boundary stubs. This
    # executes its parsing, path derivation, sequencing, and shell fail-fast logic.
    assert RECIPE.is_file(), 'approved sequential recipe is missing'
    scripts=tmp_path/'scripts'; scripts.mkdir()
    recipe=scripts/RECIPE.name; recipe.write_bytes(RECIPE.read_bytes())
    (tmp_path/'data').mkdir(); (tmp_path/'data/TEST.jsonl').write_text('{}\n'*400)
    (tmp_path/'venv/ilvr-venv/bin').mkdir(parents=True)
    capture=tmp_path/'events'
    train=tmp_path/'run_multinode_stage1_training.sh'
    train.write_text('#!/usr/bin/env bash\nif [[ $FAIL_STAGE == train ]]; then echo train_failure >> "$EVENTS"; exit 7; fi\necho train_success >> "$EVENTS"\n')
    train.chmod(0o755)
    py=tmp_path/'venv/ilvr-venv/bin/python'
    py.write_text('#!/usr/bin/env bash\nif [[ $3 == prepare-inference ]]; then\n if [[ $FAIL_STAGE == prepare ]]; then echo prepare_failure >> "$EVENTS"; exit 8; fi\n echo prepare_success >> "$EVENTS"\nelif [[ $3 == evaluate ]]; then echo evaluate_TEST400 >> "$EVENTS"; else exit 9; fi\n')
    py.chmod(0o755)
    done=subprocess.run(['bash',str(recipe),*ARGS],env={**os.environ,'EVENTS':str(capture),'FAIL_STAGE':fail},text=True,capture_output=True)
    assert done.returncode == (0 if not fail else 7 if fail=='train' else 8),done.stderr
    assert capture.read_text().splitlines()==events

def test_decoder_binds_managed_remote_producer_to_local_checkpoint(tmp_path,monkeypatch):
    import json
    from stage1 import training_evidence as evidence
    local=tmp_path/'checkpoints/v5-evidence-test/checkpoint-3'; local.mkdir(parents=True)
    # This replaces only the storage policy root with this test's owned tree.
    from src import artifact_paths
    monkeypatch.setattr(artifact_paths,'CANONICAL_CHECKPOINT_ROOT',tmp_path/'checkpoints')
    payload=evidence.build_stage1_training_evidence(run_id='v5-evidence-test',method='dora',zero_stage=1,
        world_size=4,micro_batch=1,gradient_accumulation_steps=4,ema_tau=1.,seed=42,
        global_step=3,checkpoint_path=local,loss=.3)
    producer='/remote/v5/checkpoints/v5-evidence-test/checkpoint-3'
    payload['checkpoint_path']=producer
    kwargs=dict(expected_run_id='v5-evidence-test',expected_method='dora',expected_zero_stage=1,
        expected_world_size=4,expected_checkpoint=local,expected_global_step=3,expected_producer_checkpoint=producer)
    assert evidence.decode_stage1_training_evidence(json.dumps(payload).encode(),**kwargs)==payload
    with pytest.raises(ValueError,match='producer identity'):
        evidence.decode_stage1_training_evidence(json.dumps(payload).encode(),**{**kwargs,'expected_producer_checkpoint':'/wrong/checkpoint-3'})

@pytest.mark.parametrize('changed',[False,True])
def test_actual_v5_resume_composition_authenticates_relocated_base_before_helper(tmp_path,monkeypatch,changed):
    import copy
    from dataclasses import replace
    import json
    import shutil
    from types import SimpleNamespace
    import torch
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    from stage1 import composition, checkpointing, v5_trainer
    from stage1.ema import Stage1EmaTeacher
    from stage1.peft import build_stage1_tuned_model
    from stage1.v5_reference import capture_v5_reference
    from stage1.v5_resume import build_v5_contract
    from stage1.v5_config import Stage1V5Config
    from tests.stage1.v5.test_reference import tiny_original_qwen, stage1_tuning, TinyStage1Processor, LATENT_TOKEN_IDS, RESIZED_VOCAB_SIZE
    from tests.stage1.test_checkpointing import sample_metadata
    from utils_deepseed import parse_args
    from deepspeed_runtime import load_deepspeed_runtime_profile
    import artifact_paths
    ckroot=tmp_path/'checkpoints'; monkeypatch.setattr(artifact_paths,'CANONICAL_CHECKPOINT_ROOT',ckroot)
    original=tiny_original_qwen(); old=tmp_path/'old-base'; original.save_pretrained(old)
    new=tmp_path/'relocated-base'; shutil.copytree(old,new)
    tuning=stage1_tuning(); base=Qwen2_5_VLForConditionalGeneration.from_pretrained(old,local_files_only=True)
    base.resize_token_embeddings(RESIZED_VOCAB_SIZE)
    composition._configure_latent_model(base.config,LATENT_TOKEN_IDS,latent_size=2)
    base.config.compress_strategy="average"
    base.config.use_cache=False
    base.config._attn_implementation_autoset=True
    base.config.vision_config._attn_implementation_autoset=True
    base.config.vision_config.torch_dtype=torch.float32
    student=build_stage1_tuned_model(base,tuning,LATENT_TOKEN_IDS)
    processor=TinyStage1Processor()
    artifact=ckroot/'run/checkpoint-3'
    metadata=replace(sample_metadata(original,tuning),base_model_reference=str(old),ce_weight=.1,sim_weight=.1)
    student.peft_config['default'].base_model_name_or_path=str(old)
    checkpointing.save_stage1_peft_artifact(student,processor,artifact,metadata)
    teacher=Stage1EmaTeacher(student,1.,'dora')
    identity=capture_v5_reference(teacher,student,processor,base_path=old,directory=artifact)
    contract=build_v5_contract(Stage1V5Config(lambda_topk_entropy=.1,preference_topk_k=50),identity,
        ce_weight=.1,sim_weight=.1,schedule={},execution={'tuning_method':'dora','ema_tau':1.,'zero_stage':1})
    (artifact/'stage1_v5.json').write_text(json.dumps(contract))
    if changed:
        with (new/'model.safetensors').open('ab') as stream: stream.write(b'changed-content')
    data=tmp_path/'TRAIN.jsonl'; data.write_text('{}\n'); images=tmp_path/'images'; images.mkdir()
    args=parse_args(['--stage','stage1','--stage1_variant','v5','--model',str(new),'--tuning_method','dora',
        '--resume_from_checkpoint_path',str(artifact),'--save_model_path',str(ckroot/'out'),
        '--data_path',str(data),'--image_root',str(images),'--ce_weight','.1','--sim_weight','.1',
        '--per_device_train_batch_size','1','--gradient_accumulation_steps','4','--gradient_checkpointing',
        '--attention_implementation','sdpa','--vision_attention_implementation','flash_attention_2',
        '--latent_size','2','--lora_r','4','--lora_alpha','8','--lora_dropout','0',
        '--deepspeed_config',str(ROOT/'configs/deepspeed_zero1.json')])
    profile=load_deepspeed_runtime_profile(args.deepspeed_config,world_size=4)
    snapshot=checkpointing.read_stage1_peft_metadata_snapshot(artifact)
    decision=checkpointing.Stage1ResumeDecision(artifact,checkpointing.Stage1ArtifactKind.PEFT,False,snapshot)
    calls=[]
    monkeypatch.setattr(Qwen2_5_VLForConditionalGeneration, '_check_and_enable_flash_attn_2',
                        classmethod(lambda cls,config,**kw:config))
    class Trainer:
        def __init__(self,**kwargs): self.model=kwargs['model']; self.ema_teacher=kwargs['ema_teacher']; calls.append('constructed')
    monkeypatch.setattr(v5_trainer,'CustomTrainerStage1V5',Trainer)
    monkeypatch.setattr(composition,'build_stage1_training_arguments',lambda args:SimpleNamespace(device='cpu',should_save=False))
    monkeypatch.setattr(composition,'validate_stage1_checkpoint_context',lambda model:None)
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:True)
    def gather(out,value): out[:]=[value]*4
    monkeypatch.setattr(torch.distributed,'all_gather_object',gather)
    def load(source,*,config,**kw):
        calls.append('load')
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(source,config=config,local_files_only=True)
    deps=replace(composition.DEFAULT_STAGE1_DEPENDENCIES,
        resolve_resume=lambda *_:decision,load_processor=lambda *_a,**_k:processor,
        load_config=lambda source,**_k:Qwen2_5_VLConfig.from_pretrained(source,local_files_only=True),
        load_model=load,prepare_dataset=lambda *_:[],collator_factory=lambda *_a,**_k:None)
    if changed:
        with pytest.raises(ValueError,match='base.*content|original-base content'):
            composition.build_stage1_training_components(args,dependencies=deps,deepspeed_runtime_profile=profile)
        assert calls==[]
    else:
        with pytest.warns(UserWarning,match="base_model_name_or_path.*renamed"):
            bundle=composition.build_stage1_training_components(args,dependencies=deps,deepspeed_runtime_profile=profile)
        assert calls==['load','load','constructed']
        for name,value in teacher.state_dict()['parameters'].items():
            torch.testing.assert_close(bundle.trainer.ema_teacher.state_dict()['parameters'][name],value,rtol=0,atol=0)

def test_recipe_rejects_non_test400_input_before_training(tmp_path):
    scripts=tmp_path/'scripts'; scripts.mkdir()
    recipe=scripts/RECIPE.name; recipe.write_bytes(RECIPE.read_bytes())
    (tmp_path/'data').mkdir(); (tmp_path/'data/TEST.jsonl').write_text('{}\n')
    done=subprocess.run(['bash',str(recipe),*ARGS],text=True,capture_output=True)
    assert done.returncode==2
    assert 'TEST400 requires exactly 400' in done.stderr

def test_managed_dataset_identity_authenticates_equal_bytes_and_merges_without_relaxation(tmp_path):
    import json
    import hashlib
    from src import multinode_evaluation as evaluation
    assert hasattr(evaluation,'authenticated_dataset_identity'), 'managed worker dataset authentication missing'
    local=tmp_path/'local/TEST.jsonl'; local.parent.mkdir()
    remote=tmp_path/'remote/TEST.jsonl'; remote.parent.mkdir()
    local.write_bytes(b'{"a":1}\n'); remote.write_bytes(local.read_bytes())
    canonical=evaluation.dataset_identity(local)
    digest=hashlib.sha256(local.read_bytes()).hexdigest()
    kwargs=dict(canonical_identity_json=json.dumps(canonical),expected_sha256=digest,num_shards=4)
    assert evaluation.authenticated_dataset_identity(remote,**kwargs)==canonical
    assert evaluation.authenticated_dataset_identity(remote)==evaluation.dataset_identity(remote)
    manifests=[]
    gen={'max_new_tokens':4096,'max_sequence_length':128000,'repetition_penalty':1,
         'generation_limit_policy':'external_full_latent_block','temperature':0,'top_p':1,'vision_ablation':'none'}
    for rank in range(4):
        jsonl=tmp_path/f'shard-{rank}.jsonl'
        jsonl.write_text(json.dumps({'index':rank,'task_name':'zebra-cot','match':True})+'\n')
        manifest=Path(str(jsonl)+'.manifest.json'); manifests.append(manifest)
        evaluation.write_shard_manifest(manifest,shard_jsonl=jsonl,
            dataset_identity=evaluation.authenticated_dataset_identity(remote if rank<2 else local,**kwargs),
            model_identity='task6:'+'a'*64,generation_config=gen,shard_index=rank,num_shards=4,
            expected_source_indices=[rank],processed_source_indices=[rank],worker_exit_status=0)
    merged=evaluation.merge_shards(manifests,expected_sample_count=4,output_jsonl=tmp_path/'merged.jsonl',summary_json=tmp_path/'summary.json')
    assert merged['num_samples']==4
    from src.multinode_artifacts import validate_evaluation_sidecars
    validate_evaluation_sidecars(manifests,[tmp_path/f'shard-{i}.jsonl' for i in range(4)],
        expected_model_identity='task6:'+'a'*64,expected_sample_count=4,expected_generation_config=gen)
    altered=json.loads(manifests[0].read_text()); altered['dataset_identity']['path']=str(remote)
    manifests[0].write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='dataset identity mismatch'):
        evaluation.merge_shards(manifests,expected_sample_count=4,output_jsonl=tmp_path/'rejected.jsonl',summary_json=tmp_path/'rejected-summary.json')
    # Same size and mtime are deliberately held constant: content must decide.
    before=remote.stat(); remote.write_bytes(b'{"a":2}\n'); os.utime(remote,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(ValueError,match='content'):
        evaluation.authenticated_dataset_identity(remote,**kwargs)
    with pytest.raises(ValueError,match='paired'):
        evaluation.authenticated_dataset_identity(local,canonical_identity_json=json.dumps(canonical),num_shards=4)
    with pytest.raises(ValueError,match='four-shard'):
        evaluation.authenticated_dataset_identity(local,**{**kwargs,'num_shards':1})


def test_generated_stage1_worker_argv_parses_real_entrypoint(tmp_path,monkeypatch):
    import importlib.util
    from src.multinode import ClusterSpec
    from src.multinode_runtime import EvaluationPlan, _build_evaluation_commands_for_stage
    spec=importlib.util.spec_from_file_location('v5_actual_eval',ROOT/'eval.py'); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cluster=ClusterSpec.default(local_repo_root=ROOT,remote_repo_root=Path('/remote/v5'),remote_host_repo_root=Path('/host/v5'))
    plan=EvaluationPlan(run_id='v5-entrypoint-test',model_dir=ROOT/'checkpoints/evaluation-staging/v5-train/prepared-model',
        test_data_path=ROOT/'data/TEST.jsonl',task_name='zebra-cot',output_dir=ROOT/'results/v5-entrypoint-test',cluster=cluster)
    commands=_build_evaluation_commands_for_stage(plan,stage='stage1',model_identity='task6:'+'a'*64,commit='b'*40)
    import sys
    for worker in commands.workers:
        argv=worker.node_spec.command[worker.node_spec.command.index('--')+1:]
        monkeypatch.setattr(sys,'argv',argv[1:])
        args=module.get_eval_args()
        root=Path('/remote/v5') if worker.assignment.node=='remote' else ROOT
        assert Path(args.runtime_identity_path).is_relative_to(root/'results/v5-entrypoint-test')
        assert args.canonical_dataset_identity and args.expected_dataset_sha256

@pytest.mark.parametrize('tamper',['none','before','after'])
def test_actual_stage1_worker_checks_content_before_inference_and_before_publication(tmp_path,monkeypatch,tamper):
    import hashlib
    import importlib.util
    import json
    from types import SimpleNamespace
    from src.multinode_evaluation import dataset_identity
    spec=importlib.util.spec_from_file_location('v5_offline_worker',ROOT/'eval.py'); worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    local=tmp_path/'canonical.jsonl'; remote=tmp_path/'remote.jsonl'
    local.write_text(json.dumps({'text_input':'x','original_final_answer':'A','image_input':[]})+'\n')
    remote.write_bytes(local.read_bytes())
    canonical=dataset_identity(local); digest=hashlib.sha256(local.read_bytes()).hexdigest()
    output=tmp_path/'shard-0.jsonl'
    args=SimpleNamespace(num_shards=4,shard_index=0,test_data_path=str(remote),model_dir='/prepared/model',
        base_model_dir='/original/base',cache_dir=str(tmp_path/'cache'),task_name='zebra-cot',output_json_path=str(output),
        runtime_identity_path=None,max_new_tokens=4096,max_sequence_length=128000,repetition_penalty=1.,
        generation_limit_policy='external_full_latent_block',temperature=0.,top_p=1.,model_identity='task6:'+'a'*64,
        canonical_dataset_identity=json.dumps(canonical),expected_dataset_sha256=digest)
    monkeypatch.setattr(worker,'get_eval_args',lambda:args)
    events=[]
    def change():
        stat=remote.stat(); remote.write_bytes(remote.read_bytes().replace(b'"x"',b'"y"')); os.utime(remote,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    if tamper=='before': change()
    def load(**kw):
        if tamper=='before': pytest.fail('changed content reached model loading')
        events.append('load'); return None,None
    def infer(*_a):
        events.append('infer')
        if tamper=='after': change()
        return {'raw_output':'\\boxed{A}','assistant_only':'\\boxed{A}','inference_time':.01}
    monkeypatch.setattr(worker,'load_evaluation_artifacts',load)
    monkeypatch.setattr(worker,'run_one_example',infer)
    manifest=Path(str(output)+'.manifest.json')
    if tamper!='none':
        with pytest.raises(ValueError,match='content'): worker.main()
        assert not manifest.exists()
        assert events==([] if tamper=='before' else ['load','infer'])
    else:
        worker.main()
        assert events==['load','infer']
        assert json.loads(manifest.read_text())['dataset_identity']==canonical

@pytest.mark.parametrize('variant',['v5','standard','v5-missing','v5-resume','v5-resume-rotated'])
def test_actual_coordinator_collects_then_retains_v5_on_both_nodes_before_success(tmp_path,monkeypatch,variant):
    import json
    import shutil
    from types import SimpleNamespace
    from src import multinode as coordinator, multinode_runtime as runtime
    from src.multinode_artifacts import ArtifactEntry, NodeArtifactManifest
    from stage1.training_evidence import build_stage1_training_evidence
    local=tmp_path/'local'; remote=tmp_path/'remote'
    (local/'configs').mkdir(parents=True)
    shutil.copy2(ROOT/'configs/deepspeed_zero1.json',local/'configs/deepspeed_zero1.json')
    monkeypatch.setattr(coordinator,'_REPOSITORY_ROOT',local)
    monkeypatch.setattr(coordinator,'_repository_git_stdout',lambda _: 'b'*40)
    monkeypatch.setattr(coordinator,'collect_node_snapshot',lambda *_a,**_k:None)
    monkeypatch.setattr(coordinator,'validate_preflight',lambda *_:SimpleNamespace(commit='b'*40))
    events=[]
    run='v5-retain-test'; left=local/'checkpoints'/run/'checkpoint-2'; right=remote/'checkpoints'/run/'checkpoint-2'
    for root,ranks in [(right,(0,1)),(left,(2,3))]:
        root.mkdir(parents=True)
        for rank in ranks:
            (root/f'global_step2').mkdir(exist_ok=True)
            (root/f'global_step2/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt').write_bytes(b'optimizer')
            (root/f'rng_state_{rank}.pth').write_bytes(b'rng')
        (root/'global_step2/mp_rank_00_model_states.pt').write_bytes(b'replicated')
    for name in ['latest','trainer_state.json','scheduler.pt','deepspeed-runtime.json','stage1_ema_teacher.pt',
        'stage1_peft_metadata.json','adapter_config.json','adapter_model.safetensors','ilvr_token_rows.safetensors',
        'config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json']:
        (right/name).write_bytes(b'fixture')
    if variant=='v5' or variant.startswith('v5-resume'):
        (right/'stage1_v5.json').write_bytes(b'saved contract')
        (right/'stage1_v5_reference.pt').write_bytes(b'immutable initial reference')
    class BoundaryRunner:
        def read_artifact_manifest(self,node,checkpoint,identity,canonical_owner=False):
            root=right if node==0 else left
            return NodeArtifactManifest(1,run,node,(0,1,2,3) if canonical_owner else (0,1) if node==0 else (2,3),
                tuple(ArtifactEntry(p.relative_to(root),p.stat().st_size,node) for p in root.rglob('*') if p.is_file()),identity,True)
        def read_canonical_artifact_manifest(self,checkpoint,identity):
            return self.read_artifact_manifest(0,checkpoint,identity,canonical_owner=True)
        def validate_push_destination(self,manifest,entries): pass
        def push_artifacts(self,manifest,entries):
            for entry in entries:
                target=right/entry.relative_path; assert not target.exists()
                target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(left/entry.relative_path,target)
            events.append('collect')
        def validate_resume_destination(self,entries,destination):
            assert destination==left
            for entry in entries: assert not (left/entry.relative_path).exists()
        def pull_resume_artifacts(self,manifest,entries,destination):
            for entry in entries:
                target=left/entry.relative_path; assert not target.exists()
                target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(right/entry.relative_path,target)
            events.append('redistribute')
        def read_stage1_training_evidence(self,path):
            return json.dumps(build_stage1_training_evidence(run_id=run,method='dora',zero_stage=1,world_size=4,
                micro_batch=1,gradient_accumulation_steps=4,ema_tau=1.,seed=77 if variant.startswith('v5-resume') else 42,global_step=2,checkpoint_path=right,loss=.2)).encode()
    runner=BoundaryRunner()
    if variant.startswith('v5-resume'):
        from stage1.v5_resume import build_v5_contract
        from stage1.v5_config import Stage1V5Config
        from stage1.v5_reference import V5ReferenceIdentity
        resume=left.parent/'checkpoint-1'; resume.mkdir()
        contract=build_v5_contract(Stage1V5Config(lambda_topk_entropy=.1,preference_topk_k=50),
            V5ReferenceIdentity(1,'a'*64,'b'*64,'c'*64,('weight',)),ce_weight=.1,sim_weight=.1,
            schedule={'configured_max_steps':2,'num_train_epochs':1},execution={
                'per_device_train_batch_size':1,'gradient_accumulation_steps':4,'ema_tau':1.,'seed':77,
                'attention_implementation':'sdpa'})
        (resume/'stage1_v5.json').write_text(json.dumps(contract))
        # Existing redistribution is independently exercised above. This case
        # isolates post-training validation of the already-available contract.
        monkeypatch.setattr(coordinator,'validate_resume_path',lambda _run,path:Path(path))
        monkeypatch.setattr(coordinator,'redistribute_resume_checkpoint',lambda *_a,**_k:())
    monkeypatch.setattr(coordinator,'SubprocessCoordinatorRunner',lambda *_a,**_k:runner)
    builder=runtime.build_owned_tmux_commands
    monkeypatch.setattr(coordinator,'build_owned_tmux_commands',lambda plan,**kwargs:builder(plan,
        checkpoint_root=local/'checkpoints',log_root=local/'logs/multinode',stage_log_root=local/'logs/zero-stage-compatibility',**kwargs))
    monkeypatch.setattr(coordinator,'start_rank0_then_rank1',lambda *_a,**_k:events.append('train'))
    def completed(*_a,**_k):
        if variant=='v5-resume-rotated': shutil.rmtree(resume)
        return {0:0,1:0}
    monkeypatch.setattr(coordinator,'monitor_owned_jobs',completed)
    def success(*_a,**_k):
        if variant=='v5':
            for filename in ['stage1_v5.json','stage1_v5_reference.pt']:
                assert (left/filename).is_file(), 'v5 completion lacks local saved reference/contract'
                assert (left/filename).read_bytes()==(right/filename).read_bytes()
        events.append('success'); return local/'results/witness.json'
    monkeypatch.setattr(coordinator,'write_training_witness',success)
    argv=['--run-id',run,'--zero-stage','1','--main-process-port','29683',
        '--remote-repo-root',str(remote),'--remote-host-repo-root','/host/v5','--stage1-variant','v5' if variant=='v5-missing' else variant,
        '--method','dora','--model','/original/base','--data-path',str(local/'data/TRAIN.jsonl'),
        '--image-root',str(local/'data/images'),'--max-steps','2','--gradient-accumulation','4']
    if variant=='v5-missing':
        with pytest.raises(SystemExit): coordinator._training_main('stage1',argv)
        assert events==['train','collect']
        return
    if variant.startswith('v5-resume'):
        argv[argv.index('--stage1-variant')+1]='v5'
        i=argv.index('--gradient-accumulation'); del argv[i:i+2]
        i=argv.index('--max-steps'); del argv[i:i+2]
        argv += ['--attempt-id','resume','--resume',str(resume)]
    assert coordinator._training_main('stage1',argv)==0
    assert events==(['train','collect','redistribute','success'] if variant=='v5' else ['train','collect','success'])
