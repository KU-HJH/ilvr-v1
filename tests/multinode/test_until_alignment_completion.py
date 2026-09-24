"""A bounded gate must complete through the real coordinator without checkpoint212."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import src.multinode as coordinator
import src.multinode_runtime as runtime


@pytest.mark.parametrize('diagnostic',[False,True])
def test_owned_coordinator_collects_probe_after_zero_exits_without_checkpoint212(tmp_path,monkeypatch,capsys,diagnostic):
    run_id='until-gate-coordinator-test'
    repository=coordinator._REPOSITORY_ROOT
    spec_path=tmp_path/'spec.json'
    spec_path.write_text(json.dumps({'path':'external','result_dir':str(repository/'results'/run_id),
        'tensor_dir':str(repository/'checkpoints'/run_id/'gate-tensors'),
        'mode':'ordinary_backward_memory' if diagnostic else 'comparison'}))
    for key,value in {'STAGE1_CHECKPOINT':'/models/base','DATA_PATH':'/data/train.jsonl',
        'IMAGE_ROOT':'/data','TUNING_METHOD':'dora','STAGE2_EXECUTION_PROFILE':'latest_external_until',
        'ILVR_UNTIL_ALIGNMENT_GATE_SPEC':str(spec_path)}.items():
        monkeypatch.setenv(key,value)
    commit=coordinator._repository_git_stdout(('rev-parse','HEAD'))
    events=[]
    monkeypatch.setattr(coordinator,'collect_node_snapshot',lambda *a,**k:events.append('preflight'))
    monkeypatch.setattr(coordinator,'validate_preflight',lambda *a:SimpleNamespace(commit=commit))
    witness=repository/'results'/run_id/'initial-probe-witness.json'
    class Runner:
        def collect_alignment_gate(self,spec_path,**kwargs):
            assert kwargs['exit_codes']=={0:0,1:0}
            assert kwargs['schedule_steps']==212
            events.append('collect-real-probe5')
            return witness
    monkeypatch.setattr(coordinator,'SubprocessCoordinatorRunner',lambda *a,**k:Runner())
    monkeypatch.setattr(coordinator,'start_rank0_then_rank1',lambda *a,**k:events.append('owned-start'))
    def monitor(*a,**kw):
        events.append('owned-monitor')
        return {0:0,1:0}
    monkeypatch.setattr(coordinator,'monitor_owned_jobs',monitor)
    def wrong_checkpoint(*a,**k):
        pytest.fail('bounded gate incorrectly requested training checkpoint212')
    monkeypatch.setattr(coordinator,'collect_checkpoint_artifacts',wrong_checkpoint)
    assert coordinator._training_main('stage2',['--run-id',run_id,'--zero-stage','2',
        '--main-process-port','29631','--max_steps','212','--save_steps','212'])==0
    assert events==['preflight','preflight','owned-start','owned-monitor','collect-real-probe5']
    key='memory_diagnostic_witness' if diagnostic else 'alignment_probe_witness'
    assert json.loads(capsys.readouterr().out)[key]==str(witness)


def test_probe_launch_paths_must_belong_to_owned_run(tmp_path):
    function=getattr(runtime,'validate_alignment_gate_launch',None)
    assert callable(function), 'gate launch ownership validation missing'
    spec=tmp_path/'spec.json'
    spec.write_text(json.dumps({'path':'external','result_dir':str(coordinator._REPOSITORY_ROOT/'results'/'foreign-run'),
                               'tensor_dir':str(coordinator._REPOSITORY_ROOT/'checkpoints'/'our-gate'/'tensors')}))
    plan=coordinator.build_launch_plan(stage='stage2',tuning_method='dora',zero_stage=2,
                                    run_id='our-gate',main_process_port=29631)
    with pytest.raises(ValueError,match='run-owned'):
        function(spec,plan=plan,result_root=coordinator._REPOSITORY_ROOT/'results',schedule_steps=212)


def _completed_artifacts(repository,run_id):
    """Hand-specified event fixture; no gate comparator computes its expectations."""
    from stage2.until_alignment_probe import expected_snapshots
    root=repository/'results'/run_id
    tensors=repository/'checkpoints'/run_id/'tensors'
    root.mkdir(parents=True); tensors.mkdir(parents=True)
    spec={'path':'external','result_dir':str(root),'tensor_dir':str(tensors)}
    path=repository/'spec.json'; path.write_text(json.dumps(spec))
    import hashlib
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    replay={'sample_ids':[f's{i}' for i in range(68)],'source_updates':[1,2,3,99,212]}
    for rank in range(4):
        metadata={'rank':rank,'spec':spec,'replay':replay}
        (root/f'metadata-rank-{rank}.json').write_text(json.dumps(metadata))
        completion=dict(kind='until_alignment_probe',run_id=run_id,rank=rank,commit='a'*40,
            spec_sha256=digest,schedule_steps=212,probe_optimizer_steps=5,microsteps=17,status='complete')
        (root/f'completion-rank-{rank}.json').write_text(json.dumps(completion))
        records=[dict(event='init',rank=rank,hostname=f'host{rank//2}',world_size=4,gas=4,microbatch=1,
                      fixed_denominator=16,zero_stage=2,schedule_steps=212,warmup_steps=10),
                 dict(event='compiler',rank=rank,calls=[dict(callable='torch.nn.attention.flex_attention.flex_attention',dynamic=True)])]
        for micro in range(17):
            records.append(dict(event='example',rank=rank,micro=micro,sample_id=f's{micro*4+rank}',coefficient=1/16,weight=1.,peak_allocated=1024))
            records.append(dict(event='backward',rank=rank,micro=micro,dora_norm_calls=2,dora_scope_cleaned=True,actual_backward_hooks=3))
        for step in range(1,6):
            records.append(dict(event='update',rank=rank,probe_optimizer_step=step,completed=True,fixed_denominator=16,
                actual_count=4 if step==5 else 16,original_update_index=[1,2,3,99,212][step-1],lr=[(step-1)*5e-6],
                master_delta_nonzero=step>1,preference_gradient_nonzero=True,
                fixed_gradient={'coordinate':{'finite':True,'expected_l2':1.,'l2_error':.001}}))
        (root/f'rank-{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    for name in expected_snapshots():
        target=tensors/name; target.parent.mkdir(exist_ok=True)
        target.write_bytes(b'opaque tensor transport fixture')
    return path,spec,digest


def test_completion_requires_actual_five_step_markers_and_four_rank_runtime(tmp_path):
    from scripts.verify_until_trainer import validate_collected_completion
    path,spec,digest=_completed_artifacts(tmp_path,'our-gate')
    kwargs=dict(run_id='our-gate',commit='a'*40,spec_sha256=digest)
    assert validate_collected_completion(path,**kwargs)['pass']
    marker=Path(spec['result_dir'])/'completion-rank-0.json'
    payload=json.loads(marker.read_text()); payload['probe_optimizer_steps']=212
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='five-step'):
        validate_collected_completion(path,**kwargs)
    payload['probe_optimizer_steps']=5; marker.write_text(json.dumps(payload))
    rank=Path(spec['result_dir'])/'rank-3.jsonl'; rank.unlink()
    with pytest.raises(ValueError,match='runtime witness failed'):
        validate_collected_completion(path,**kwargs)


@pytest.mark.parametrize('corrupt',[False,True])
@pytest.mark.parametrize('diagnostic',[False,True])
def test_real_collector_gathers_remote_tensors_and_rejects_transport_corruption(tmp_path,monkeypatch,corrupt,diagnostic):
    import shutil
    import subprocess
    from scripts.verify_until_trainer import node_evidence
    builder=_completed_memory_artifacts if diagnostic else _completed_artifacts
    path,spec,digest=builder(tmp_path,'our-gate')
    remote=node_evidence(path,ranks=[0,1],run_id='our-gate',commit='a'*40,spec_sha256=digest)
    holding=tmp_path/'remote'; holding.mkdir()
    for kind,field in (('results','result_dir'),('tensors','tensor_dir')):
        for entry in remote[kind]:
            source=Path(spec[field])/entry['path']; dest=holding/kind/entry['path']
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.move(source,dest)
    monkeypatch.setattr(runtime,'_CHECKPOINT_ROOT',tmp_path/'checkpoints')
    monkeypatch.setattr(runtime,'_REPOSITORY_ROOT',tmp_path)
    def boundary(command,**kwargs):
        assert kwargs['timeout_seconds']==runtime._ARTIFACT_TRANSFER_TIMEOUT_SECONDS
        if command[0]=='/usr/bin/rsync':
            destination=Path(command[-1]); kind='tensors' if destination==Path(spec['tensor_dir']) else 'results'
            listing=Path(command[command.index('--files-from')+1])
            for name in listing.read_bytes().split(b'\0'):
                if not name: continue
                target=destination/name.decode(); target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(holding/kind/name.decode(),target)
                if corrupt and kind=='tensors': target.write_bytes(b'corrupted')
            return subprocess.CompletedProcess(command,0,'','')
        return subprocess.CompletedProcess(command,0,json.dumps(remote),'')
    plan=coordinator.build_launch_plan(stage='stage2',tuning_method='dora',zero_stage=2,run_id='our-gate',main_process_port=29631)
    runner=runtime.SubprocessCoordinatorRunner(plan,commands=None,commit='a'*40,log_root=tmp_path/'logs',
        checkpoint_root=tmp_path/'checkpoints',command_runner=boundary)
    kwargs=dict(exit_codes={0:0,1:0},schedule_steps=212,expected_spec_sha256=digest,result_root=tmp_path/'results')
    if corrupt:
        with pytest.raises(ValueError,match='artifact differs'):
            runner.collect_alignment_gate(path,**kwargs)
    else:
        witness=runner.collect_alignment_gate(path,**kwargs)
        report=json.loads(witness.read_text())
        assert report['probe_optimizer_steps']==(2 if diagnostic else 5) and report['schedule_steps']==212
        assert report['runtime']['complete' if diagnostic else 'pass'] and report['training_checkpoint'] is None
        count=32 if diagnostic else 240
        assert report['rank_artifact_counts']=={'remote':count,'local':count}
        if diagnostic:
            assert witness.name=='initial-memory-diagnostic-witness.json'
            assert report['kind']=='until_memory_diagnostic'
            assert report['runtime']['comparison_pass'] is False and report['full212_authorized'] is False
        assert not (tmp_path/'checkpoints'/'our-gate'/'checkpoint-212').exists()


def _completed_memory_artifacts(tmp_path,run_id):
    path,spec,_ = _completed_artifacts(tmp_path,run_id)
    spec['mode']='ordinary_backward_memory'
    path.write_text(json.dumps(spec))
    import hashlib
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    root=Path(spec['result_dir']); tensors=Path(spec['tensor_dir'])
    for rank in range(4):
        meta=root/f'metadata-rank-{rank}.json'
        payload=json.loads(meta.read_text()); payload['spec']=spec; meta.write_text(json.dumps(payload))
        marker=root/f'completion-rank-{rank}.json'
        payload=json.loads(marker.read_text()); payload.update(kind='until_memory_diagnostic',probe_optimizer_steps=2,microsteps=8,spec_sha256=digest)
        marker.write_text(json.dumps(payload))
        entries=[dict(event='init',rank=rank,hostname=f'host{rank//2}',world_size=4,gas=4,microbatch=1,zero_stage=2,schedule_steps=212,warmup_steps=10),
                 dict(event='compiler',rank=rank,calls=[dict(callable='torch.nn.attention.flex_attention.flex_attention',dynamic=True,donated_buffer=False)])]
        for micro in range(8):
            for event in ['micro_entry','forward','backward']:
                entries.append(dict(event=event,rank=rank,micro=micro,sample_id=f's{micro*4+rank}',allocated=100,peak_allocated=120,actual_backward_hooks=3,coordinates={'targets':[7]}))
        entries.extend(dict(event='memory_update',rank=rank,probe_optimizer_step=step,lr=[(step-1)*5e-6]) for step in [1,2])
        (root/f'rank-{rank}.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in entries))
    for tensor in tensors.glob('rank-*/*.pt'):
        name=tensor.name
        keep=any(name==f'micro-{m}-{k}.pt' for m in range(8) for k in ['scores','rng'])
        if not keep: tensor.unlink()
    return path,spec,digest


def test_memory_completion_is_separate_and_cannot_pass_alignment_gate(tmp_path):
    from scripts.verify_until_trainer import validate_collected_completion, node_evidence
    path,spec,digest=_completed_memory_artifacts(tmp_path,'memory-test')
    root=Path(spec['result_dir'])
    kwargs=dict(run_id='memory-test',commit='a'*40,spec_sha256=digest)
    report=validate_collected_completion(path,**kwargs)
    assert report['complete'] is True and report['comparison_pass'] is False
    assert len(node_evidence(path,ranks=[0,1],**kwargs)['tensors'])==32
    # A missing ordinary backward is not a completed memory diagnosis.
    target=root/'rank-0.jsonl'
    rows=[json.loads(x) for x in target.read_text().splitlines()]
    target.write_text(''.join(json.dumps(x)+'\n' for x in rows if not(x['event']=='backward' and x.get('micro')==4)))
    with pytest.raises(ValueError,match='memory diagnostic'):
        validate_collected_completion(path,**kwargs)


@pytest.mark.parametrize('checkpoint',[False,True])
def test_aligned_ordinary_probe_reaches_real_coordinator_dry_run(tmp_path,monkeypatch,capsys,checkpoint):
    """The approved worker mode must survive the coordinator's launch validator."""
    run_id='off-checkpoint-coordinator-dry-run'
    repository=coordinator._REPOSITORY_ROOT
    spec_path=tmp_path/'spec.json'
    spec_path.write_text(json.dumps(dict(path='aligned',mode='ordinary_backward_memory',
        source_updates=[60,61],off_activation_checkpointing=checkpoint,save_on_cpu=False,
        result_dir=str(repository/'results'/run_id),tensor_dir=str(repository/'checkpoints'/run_id/'tensors'))))
    for key,value in dict(STAGE1_CHECKPOINT='/models/stage1',DATA_PATH='/data/train.jsonl',IMAGE_ROOT='/data',
        TUNING_METHOD='dora',STAGE2_EXECUTION_PROFILE='latest_external_until',
        STEP_OFF_ACTIVATION_CHECKPOINTING=str(checkpoint).lower(),ILVR_UNTIL_ALIGNMENT_GATE_SPEC=str(spec_path)).items():
        monkeypatch.setenv(key,value)
    assert coordinator._training_main('stage2',['--run-id',run_id,'--zero-stage','2',
        '--main-process-port','29631','--max_steps','212','--save_steps','212','--dry-run'])==0
    output=json.loads(capsys.readouterr().out)
    assert output['remote'] and output['local']
    assert output['plan']['world_size']==4


@pytest.mark.parametrize('change',[
    {'mode':'unknown'}, {'source_updates':None}, {'source_updates':[1,1]},
    {'source_updates':[1,212]}, {'source_updates':[True,2]},
    {'off_activation_checkpointing':1}, {'save_on_cpu':True},
])
def test_aligned_ordinary_launch_rejects_unsupported_replay_or_offload(tmp_path,change):
    run_id='off-checkpoint-invalid'
    repository=coordinator._REPOSITORY_ROOT
    spec=dict(path='aligned',mode='ordinary_backward_memory',source_updates=[1,2],
        off_activation_checkpointing=True,save_on_cpu=False,
        result_dir=str(repository/'results'/run_id),tensor_dir=str(repository/'checkpoints'/run_id/'tensors'))
    spec.update(change)
    path=tmp_path/'spec.json';path.write_text(json.dumps(spec))
    plan=coordinator.build_launch_plan(stage='stage2',tuning_method='dora',zero_stage=2,
                                    run_id=run_id,main_process_port=29631)
    with pytest.raises(ValueError):
        runtime.validate_alignment_gate_launch(path,plan=plan,result_root=repository/'results',schedule_steps=212)
