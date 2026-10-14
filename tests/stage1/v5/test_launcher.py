"""Behavioral coordinator and worker argv witnesses; no ranks are launched."""
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
REMOTE = Path('/root/workspace/ILVR/.worktrees/v5 managed')
HOST = Path('/home/korea/Workspace/hjh/workspace/ILVR/.worktrees/v5 managed')
PROFILE = ['--stage1-variant','v5','--stage1-loss-type','dil-bce-orig',
    '--sim-weight','.1','--ce-weight','.1','--lambda-preference','1',
    '--lambda-topk-entropy','.1','--preference-topk-k','50',
    '--preference-topk-mode','independent','--no-topk-include-gt',
    '--lambda-topk-mass','0','--stage1-v5-selection','all',
    '--stage1-v5-text-kl-weight','.01','--loss-chunk-positions','128',
    '--vision-attention-implementation','flash_attention_2','--warm-up-steps','10',
    '--micro-batch','1','--gradient-accumulation','4','--seed','42',
    '--attention-implementation','sdpa','--max-steps','3']

def test_managed_coordinator_generates_each_node_worker_and_preflight_paths():
    completed = subprocess.run([str(ROOT/'run_multinode_stage1_training.sh'),
        '--run-id','v5-command-test','--zero-stage','1','--main-process-port','29683',
        '--local-repo-root',str(ROOT),'--remote-repo-root',str(REMOTE),
        '--remote-host-repo-root',str(HOST),'--dry-run','--method','dora',
        '--model','/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct',
        '--data-path',str(ROOT/'data/TRAIN.jsonl'),'--image-root',str(ROOT/'data/images_comt'),
        *PROFILE], text=True,capture_output=True,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"})
    assert completed.returncode == 0, completed.stderr
    value=json.loads(completed.stdout)
    assert value['plan']['world_size']==4
    assert value['plan']['cluster']['local_repo_root']==str(ROOT)
    from src.multinode import decode_launch_plan
    from src.multinode_runtime import build_owned_tmux_commands, build_preflight_commands
    plan=decode_launch_plan(json.dumps(value['plan']).encode())
    commands=build_owned_tmux_commands(plan,commit='a'*40,stage_arguments=[
        '--method','dora','--model','/original/base','--data-path',str(ROOT/'data/TRAIN.jsonl'),
        '--image-root',str(ROOT/'data/images_comt'),*PROFILE])
    assert [x.machine_rank for x in commands.node_specs]==[0,1]
    for spec,root in zip(commands.node_specs,(REMOTE,ROOT)):
        env=dict(spec.environment)
        assert env['ILVR_NUM_PROCESSES']=='4' and env['ILVR_NUM_MACHINES']=='2'
        assert str(root/'run_stage1_training.sh') in spec.command
        assert str(root/'data/TRAIN.jsonl') in spec.command
        assert spec.ownership_path.is_relative_to(root/'logs')
        argv=spec.command[spec.command.index('--')+1:]
        dry=subprocess.run(['bash',str(ROOT/'run_stage1_training.sh'),*argv[2:],'--dry-run'],
            env={**os.environ,**env},text=True,capture_output=True)
        assert dry.returncode==0,dry.stderr
        worker=shlex.split(dry.stdout)
        vals={worker[i].lstrip('-'):worker[i+1] for i in range(len(worker)-1) if worker[i].startswith('--')}
        assert vals['stage1_variant']=='v5'
        assert vals['ce_weight']==vals['sim_weight']=='.1'
        assert vals['lambda_topk_entropy']=='.1' and vals['preference_topk_k']=='50'
        assert vals['loss_chunk_positions']=='128'
        assert '--no-topk_include_gt' in worker
    pre=build_preflight_commands(plan,input_paths=(ROOT/'data/TRAIN.jsonl',),artifact_roots=(),remote_run_paths=(),local_run_paths=(),remote_tmux_sessions=(),local_tmux_sessions=())
    assert str(REMOTE/'data/TRAIN.jsonl') in pre.remote_node_info[-1]
    assert str(ROOT/'data/TRAIN.jsonl') in pre.local_node_info

def test_managed_evaluation_uses_node_roots_for_workers_and_remote_control():
    from src.multinode import ClusterSpec
    from src.multinode_runtime import EvaluationPlan, EvaluationSubprocessRunner, _build_evaluation_commands_for_stage
    cluster=ClusterSpec.default(local_repo_root=ROOT,remote_repo_root=REMOTE,remote_host_repo_root=HOST)
    plan=EvaluationPlan(run_id='v5-eval-test',model_dir=ROOT/'checkpoints/evaluation-staging/v5-train-test/prepared-model',
        test_data_path=ROOT/'data/TEST.jsonl',task_name='zebra-cot',output_dir=ROOT/'results/v5-eval-test',cluster=cluster)
    commands=_build_evaluation_commands_for_stage(plan,stage='stage1',model_identity='task6:'+'a'*64,commit='b'*40)
    assert len(commands.workers)==4
    runner=EvaluationSubprocessRunner(plan,commit='b'*40)
    for worker in commands.workers:
        remote=worker.assignment.node=='remote'; root=REMOTE if remote else ROOT
        spec=worker.node_spec
        assert spec.ownership_path.is_relative_to(root/'logs')
        assert dict(spec.environment)["PATH"].split(":")[0] == str(root/"venv/ilvr-venv/bin")
        argv=spec.command[spec.command.index('--')+1:]
        assert argv[0]==str(root/'venv/ilvr-venv/bin/python')
        assert argv[1]==str(root/'eval.py')
        assert str(root/'data/TEST.jsonl') in argv
        assert str(root/'results/v5-eval-test'/f'shard-{worker.assignment.shard_index}.jsonl') in argv
        command=runner._internal_command(worker,'exit-show',['--exit-path',str(spec.exit_path)])
        if remote: assert str(REMOTE/'venv/ilvr-venv/bin/python') in command[-1]
        else: assert str(ROOT/'venv/ilvr-venv/bin/python') in command
    assert str(REMOTE/'results/v5-eval-test') in commands.remote_pull[-2]


def test_training_collection_keeps_v5_reference_but_inference_excludes_it():
    from dataclasses import replace
    from tests.multinode.test_evaluation_coordinator import _manifest
    from src.multinode_artifacts import ArtifactEntry, select_common_metadata_entries, select_inference_entries
    manifest=_manifest()
    additions=tuple(ArtifactEntry(Path(name),1,0) for name in ['stage1_v5.json','stage1_v5_reference.pt','stage1_ema_teacher.pt'])
    manifest=replace(manifest,entries=manifest.entries+additions)
    common={e.relative_path.name for e in select_common_metadata_entries(manifest)}
    assert {'stage1_v5.json','stage1_v5_reference.pt'} <= common
    inference={e.relative_path.name for e in select_inference_entries(manifest)}
    assert 'stage1_v5.json' not in inference and 'stage1_v5_reference.pt' not in inference
    assert 'stage1_ema_teacher.pt' not in inference

def test_native_default_requires_explicit_roots_and_topology_cannot_drift():
    import pytest
    from dataclasses import replace
    from src.multinode import ClusterSpec, validate_cluster_topology
    if not ROOT.is_relative_to('/root/workspace/ILVR'):
        with pytest.raises(ValueError,match='explicit remote repository roots'): ClusterSpec.default()
    cluster=ClusterSpec.default(local_repo_root=ROOT,remote_repo_root=REMOTE,remote_host_repo_root=HOST)
    with pytest.raises(ValueError,match='fixed ILVR topology'):
        validate_cluster_topology(replace(cluster,num_processes=2))
    with pytest.raises(ValueError,match='both remote repository roots'):
        ClusterSpec.default(remote_repo_root=REMOTE)

def test_managed_preparation_mirrors_only_allowlisted_model_to_remote_root(tmp_path):
    from src.multinode import build_launch_plan
    from src.multinode_runtime import SubprocessCoordinatorRunner
    from tests.multinode.test_evaluation_coordinator import _manifest
    plan=build_launch_plan(stage='stage1',tuning_method='dora',zero_stage=1,run_id='trained-model-run',
        main_process_port=29683,local_repo_root=ROOT,remote_repo_root=REMOTE,remote_host_repo_root=HOST)
    manifest=_manifest()
    recorded=[]
    def boundary(command,**kwargs):
        recorded.append(tuple(command))
        return subprocess.CompletedProcess(command,0,stdout=json.dumps(manifest.to_json()),stderr='')
    runner=SubprocessCoordinatorRunner(plan,commands=None,commit='b'*40,log_root=tmp_path,command_runner=boundary)
    destination=ROOT/'checkpoints/evaluation-staging/trained-model-run/prepared-model'
    runner.mirror_prepared_inference(manifest,destination)
    reserve,transfer,show=recorded
    assert str(REMOTE/'checkpoints/evaluation-staging/trained-model-run/prepared-model') in reserve[-1]
    assert str(REMOTE/'checkpoints/evaluation-staging/trained-model-run/prepared-model') in show[-1]
    assert str(REMOTE/'checkpoints/evaluation-staging/trained-model-run') in transfer[-1]
    files=(tmp_path/'trained-model-run/initial/artifact-inference-mirror.files').read_bytes()
    assert b'v5_reference' not in files and b'optim_states' not in files

def test_managed_training_preparation_pulls_exact_remote_checkpoint(tmp_path,monkeypatch):
    from src.multinode import build_launch_plan
    from src.multinode_runtime import SubprocessCoordinatorRunner
    from src import multinode_artifacts as artifacts
    from tests.multinode.test_evaluation_coordinator import _manifest
    plan=build_launch_plan(stage='stage1',tuning_method='dora',zero_stage=1,run_id='trained-model-run',
        main_process_port=29683,local_repo_root=ROOT,remote_repo_root=REMOTE,remote_host_repo_root=HOST)
    recorded=[]
    def boundary(command,**kwargs):
        recorded.append(tuple(command)); return subprocess.CompletedProcess(command,0,stdout='',stderr='')
    runner=SubprocessCoordinatorRunner(plan,commands=None,commit='b'*40,log_root=tmp_path,command_runner=boundary)
    # Transfer effects and inode inspection are separately tested by the existing
    # artifact suite. This witness exercises this actual caller's rsync source.
    monkeypatch.setattr(artifacts,'validate_reserved_inference_destination',lambda _:None)
    monkeypatch.setattr(artifacts,'validate_transferred_entries',lambda *_:None)
    runner.pull_inference_artifacts(_manifest(),tmp_path/'prepared')
    assert str(REMOTE/'checkpoints/trained-model-run/checkpoint-2') in recorded[0][-2]

def test_actual_first_worker_start_records_bound_dataset_digest_before_launch(tmp_path):
    import hashlib
    from src.multinode import ClusterSpec
    from src.multinode_runtime import EvaluationPlan, EvaluationSubprocessRunner, _build_evaluation_commands_for_stage
    cluster=ClusterSpec.default(local_repo_root=ROOT,remote_repo_root=REMOTE,remote_host_repo_root=HOST)
    plan=EvaluationPlan(run_id='v5-identity-record',model_dir=ROOT/'checkpoints/evaluation-staging/v5-train/prepared-model',
        test_data_path=ROOT/'data/TEST.jsonl',task_name='zebra-cot',output_dir=ROOT/'results/v5-identity-record',cluster=cluster)
    commands=_build_evaluation_commands_for_stage(plan,stage='stage1',model_identity='task6:'+'a'*64,commit='b'*40)
    record=tmp_path/'v5-identity-record/initial/dataset-identity.json'
    def boundary(command,**kwargs):
        assert record.is_file(), 'dataset digest was not durably recorded before launch'
        return subprocess.CompletedProcess(command,0,stdout='',stderr='')
    runner=EvaluationSubprocessRunner(plan,commit='b'*40,log_root=tmp_path,command_runner=boundary)
    runner._controls_prepared=True
    runner.start_worker(commands.workers[0])
    before=record.read_bytes()
    runner.start_worker(commands.workers[1])
    value=json.loads(before)
    data=plan.test_data_path; stat=data.stat()
    assert value=={'canonical_identity':{'path':str(data),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns},
        'content_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
    assert record.read_bytes()==before


def test_public_resume_shell_preserves_omissions_on_both_mapped_nodes(tmp_path,monkeypatch):
    import pytest
    import shutil
    from src import multinode
    from src.multinode_runtime import build_owned_tmux_commands
    from stage1.v5_config import Stage1V5Config
    from stage1.v5_reference import V5ReferenceIdentity
    from stage1.v5_resume import build_v5_contract
    from utils_deepseed import parse_args
    local,remote=tmp_path/'local',tmp_path/'remote'
    (local/'configs').mkdir(parents=True)
    shutil.copy2(ROOT/'configs/deepspeed_zero1.json',local/'configs/deepspeed_zero1.json')
    monkeypatch.setattr(multinode,'_REPOSITORY_ROOT',local)
    saved=build_v5_contract(Stage1V5Config(lambda_topk_entropy=.1,preference_topk_k=50),
        V5ReferenceIdentity(1,'a'*64,'b'*64,'c'*64,('weight',)),ce_weight=.1,sim_weight=.1,
        schedule={'configured_max_steps':3,'num_train_epochs':2.},execution={
            'per_device_train_batch_size':1,'gradient_accumulation_steps':4,'learning_rate':3e-5,
            'ema_tau':1.,'seed':77,'attention_implementation':'sdpa','vision_attention_implementation':'flash_attention_2',
            'gradient_checkpointing':True,'textual_states':'all','lora_r':32,'lora_alpha':64,
            'lora_dropout':.05,'lora_target_modules':'q_proj','latent_size':8,'warmup_steps':10})
    for root in (local,remote):
        cp=root/'checkpoints/v5-resume-command/checkpoint-1'; cp.mkdir(parents=True)
        (cp/'stage1_v5.json').write_text(json.dumps(saved))
    plan=multinode.build_launch_plan(stage='stage1',tuning_method='dora',zero_stage=1,
        run_id='v5-resume-command',main_process_port=29683,attempt_id='resume',
        remote_repo_root=remote,remote_host_repo_root=Path('/host/v5'))
    base=['--method','dora','--model','/original/base','--data-path',str(local/'data/TRAIN.jsonl'),
        '--image-root',str(local/'data/images'),'--resume',str(local/'checkpoints/v5-resume-command/checkpoint-1')]
    for extra,overrides in (([],{}),(['--gradient-accumulation','2'],{}),(['--attention-implementation','eager'],{}),
        ([],{'STAGE1_SEED':'123'}),([],{ 'STAGE1_GRADIENT_ACCUMULATION':'2'})):
        commands=build_owned_tmux_commands(plan,commit='a'*40,stage_arguments=base+extra,
            checkpoint_root=local/'checkpoints',log_root=local/'logs/multinode',
            stage_log_root=local/'logs/zero-stage-compatibility',result_root=local/'results')
        for node,root in zip(commands.node_specs,(remote,local)):
            stage=node.command[node.command.index('--')+1:]
            completed=subprocess.run(['bash',str(ROOT/'run_stage1_training.sh'),*stage[2:],'--dry-run'],
                env={**{k:v for k,v in os.environ.items() if not k.startswith('STAGE1_') and k!='CE_WEIGHT'},
                     **dict(node.environment),'PYTHONDONTWRITEBYTECODE':'1',**overrides},text=True,capture_output=True)
            assert completed.returncode==0,completed.stderr
            worker=shlex.split(completed.stdout); argv=worker[worker.index('src/main.py')+1:]
            assert argv[argv.index('--resume_from_checkpoint_path')+1]==str(root/'checkpoints/v5-resume-command/checkpoint-1')
            if extra or overrides:
                with pytest.raises(ValueError,match='conflicts with saved v5 contract'): parse_args(argv)
                continue
            args=parse_args(argv)
            assert args.stage1_variant=='v5' and args.ce_weight==args.sim_weight==.1
            expected={'per_device_train_batch_size':1,'gradient_accumulation_steps':4,'learning_rate':3e-5,
                'ema_tau':1.,'seed':77,'attention_implementation':'sdpa','vision_attention_implementation':'flash_attention_2',
                'gradient_checkpointing':True,'textual_states':'all','lora_r':32,'lora_alpha':64,
                'lora_dropout':.05,'latent_size':8,'warm_up_steps':10,'max_steps':3,'epochs':2.}
            for name,value in expected.items(): assert getattr(args,name)==value
            from stage1.types import resolve_stage1_tuning_config
            assert resolve_stage1_tuning_config(args).target_suffixes==('q_proj',)
