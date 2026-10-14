"""CPU protocol evidence; no distributed execution is implied."""
import importlib.util
from pathlib import Path

import pytest
import torch
from stage1.v5_metadata import V5Block, V5BatchMetadata


def api():
    assert importlib.util.find_spec('scripts.verify_stage1_v5'), 'Stage1-v5 admission diagnostic missing'
    from scripts import verify_stage1_v5
    return verify_stage1_v5


def test_geometry_combines_prefix_and_suffix_from_different_blocks():
    metadata=V5BatchMetadata((V5Block(0,1,5,10,(11,30),(2,3),()),
                              V5Block(0,3,35,40,(41,42),(2,3),())))
    geometry=api().packed_geometry(metadata, source_index=17,source_id='unequal')
    assert geometry['branch_count']==4
    assert geometry['prefix_length']==40 and geometry['suffix_length']==20
    assert geometry['prefix_source_item']==3 and geometry['suffix_source_item']==1
    assert geometry['dense_bias_elements']==4800
    assert geometry['replicated_prefix_tokens']==160
    assert geometry['selected_queries']==8


def test_worst_geometry_keeps_non_dominated_actual_rows():
    rows=[dict(source_index=1,dense_bias_elements=100,replicated_prefix_tokens=900,selected_queries=5),
          dict(source_index=2,dense_bias_elements=1000,replicated_prefix_tokens=100,selected_queries=50),
          dict(source_index=3,dense_bias_elements=10,replicated_prefix_tokens=10,selected_queries=1)]
    assert [r['source_index'] for r in api().maximum_geometries(rows)]==[1,2]


def test_audit_uses_actual_parent_loader_sampler_and_tail(tmp_path):
    report=api().audit_prepared_loader(list(range(37)),output_dir=tmp_path)
    assert report['prepared_loader_microbatches_per_rank']==[10]*4
    assert report['optimizer_updates_for_one_exposure']==3
    assert report['final_window_microbatches']==2
    occurrences=report['sampler_occurrences']
    assert len(occurrences)==40 and set(r['source_index'] for r in occurrences)==set(range(37))
    assert sorted(r['repeat_count'] for r in report['source_occurrences'])==[1]*34+[2]*3
    assert report['native_tail_denominator']==16


def valid_records():
    return [dict(rank=r,world_size=4,run_id='v5-protocol',entry=True,exit=True,
        reference_before='a'*64,reference_after='a'*64,contract_digest='b'*64,
        tiny_windows=[dict(global_original_steps=n,configured_gas=4,actual_microbatches=m,
            native_denominator=16,all_rank_original_counts=([[2,1,0,0],[1,0,1,0],[0,0,0,0],[1,2,0,1]] if m==4 else [[1,0],[1,1],[0,0],[0,2]]),gradient_error=0.,update_error=0.,gradient_norm=1.) for n,m in ((9,4),(9,4),(5,2))],
        continuations=[dict(saved_step=s,sources=list(range(4*s,10)),updates=3-s,final_step=3) for s in (1,2)],
        hooks=dict(helper=8,F=8,reference=8,packed=8,backward=8),
        production=dict(global_step=2,microbatches=8,finite_loss=True,gradient_norms=[1.],peak_allocated=123,peak_reserved=456,
                        source_indices=[17],geometry=[dict(source_index=17,dense_bias_elements=4800)])) for r in range(4)]


@pytest.mark.parametrize('mutation', ['rank','denominator','counts','gradient','reference','continuation','hook','extra_call','geometry','norm'])
def test_rank_verification_rejects_corrupt_runtime_evidence(mutation):
    records=valid_records()
    kwargs=dict(run_id='v5-protocol',expected_geometry=[dict(source_index=17,dense_bias_elements=4800)])
    assert api().validate_rank_records(records,**kwargs)['ranks']==[0,1,2,3]
    if mutation=='rank': records.pop()
    elif mutation=='denominator': records[2]['tiny_windows'][-1]['native_denominator']=8
    elif mutation=='counts': records[0]['tiny_windows'][0]['all_rank_original_counts'][0][0]=1
    elif mutation=='norm': records[0]['production']['gradient_norms']=[]
    elif mutation=='gradient': records[0]['tiny_windows'][0]['gradient_error']=float('nan')
    elif mutation=='reference': records[1]['reference_after']='c'*64
    elif mutation=='continuation': records[3]['continuations'][0]['sources']=[4,5,6,7,8]
    elif mutation=='extra_call': records[0]['hooks']['F']+=1
    elif mutation=='hook': records[0]['hooks']['packed']=0
    else: records[0]['production']['geometry']=[]
    with pytest.raises(ValueError): api().validate_rank_records(records,**kwargs)


def test_distributed_mode_refuses_cpu_or_unowned_entry(tmp_path,monkeypatch):
    monkeypatch.setenv('WORLD_SIZE','1')
    with pytest.raises(ValueError,match='four'): api().require_distributed_entry()
    monkeypatch.setenv('WORLD_SIZE','4')
    monkeypatch.delenv('ILVR_REPOSITORY_COMMIT',raising=False)
    with pytest.raises(ValueError,match='owned'): api().require_distributed_entry()


def test_exact_owned_builder_maps_both_nodes_and_rejects_wrong_profile():
    from src.multinode import build_launch_plan
    from src import multinode_runtime as runtime
    assert hasattr(runtime,'build_stage1_v5_witness_commands'), 'owned Stage1-v5 command dispatch missing'
    local=Path(__file__).resolve().parents[3]
    plan=build_launch_plan(stage='stage1',tuning_method='dora',zero_stage=1,run_id='v5-protocol',
        main_process_port=29579,local_repo_root=local,
        remote_repo_root=Path('/remote/ILVR'),remote_host_repo_root=Path('/host/ILVR'))
    commands=runtime.build_stage1_v5_witness_commands(plan,commit='a'*40,data_path=local/'data/TRAIN.jsonl',
        image_root=local/'data/images',model=Path('/base'),audit_path=local/'results/audit.json')
    assert '/remote/ILVR/venv/ilvr-venv/bin/accelerate' in commands.node_specs[0].command
    assert str(local/'venv/ilvr-venv/bin/accelerate') in commands.node_specs[1].command
    for node,root in zip(commands.node_specs,('/remote/ILVR',str(local))):
        assert str(root+'/scripts/verify_stage1_v5.py') in node.command
        assert root+'/data/TRAIN.jsonl' in node.command
        assert root+'/results/audit.json' in node.command
        assert '--mode' in node.command and 'distributed-witness' in node.command
    bad=build_launch_plan(stage='stage1',tuning_method='dora',zero_stage=2,run_id='v5-bad',main_process_port=29579,
        local_repo_root=local,remote_repo_root=Path('/remote/ILVR'),remote_host_repo_root=Path('/host/ILVR'))
    with pytest.raises(ValueError): runtime.build_stage1_v5_witness_commands(bad,commit='a'*40,
        data_path=Path('/data'),image_root=Path('/images'),model=Path('/base'),audit_path=Path('/audit'))


def test_actual_parent_loop_checkpoint_continuation_retains_partial_tail(tmp_path):
    directory=Path(__file__).resolve().parents[3]/'checkpoints/stage1-v5-task7'/tmp_path.parent.name/tmp_path.name
    report=api().tiny_engine_witness(directory,use_cpu=True)
    assert [w['global_original_steps'] for w in report['tiny_windows']]==[9,9,5]
    assert report['scope']=='CPU actual parent loop'
    assert [w['native_denominator'] for w in report['tiny_windows']]==[4]*3
    assert [w['actual_microbatches'] for w in report['tiny_windows']]==[4,4,2]
    assert all(w['gradient_error']<1e-6 and w['update_error']<1e-6 for w in report['tiny_windows'])
    assert report['continuations']==[
        dict(saved_step=1,sources=[4,5,6,7,8,9],updates=2,final_step=3),
        dict(saved_step=2,sources=[8,9],updates=1,final_step=3)]


def test_runtime_observer_records_real_helper_f_reference_packed_backward(tmp_path):
    from tests.stage1.v5.test_trainer import integrated_fixture
    trainer,batch,*_=integrated_fixture(tmp_path,real_helper=True)
    assert hasattr(api(),'observe_production_trainer'), 'actual entry observer missing'
    observed=api().observe_production_trainer(trainer)
    batches,_=trainer.get_batch_samples(iter([batch]),4)
    trainer.training_step(trainer.model,batches[0])
    assert dict(observed['hooks'])==dict(helper=1,F=1,reference=1,packed=1,backward=1)
    assert observed['source_indices']=={7}
    assert observed['geometry'][7]['prefix_length']==19
    for handle in observed['handles']: handle.remove()


def test_coordinator_dry_run_routes_exact_diagnostic_without_launch(capsys):
    from src.multinode import _main as main
    root=Path(__file__).resolve().parents[3]
    code=main(['train-stage1','--stage1-v5-witness','--run-id','v5-dry-protocol','--zero-stage','1',
        '--main-process-port','29579','--local-repo-root',str(root),'--remote-repo-root','/remote/ILVR',
        '--remote-host-repo-root','/host/ILVR','--data-path',str(root/'data/TRAIN.jsonl'),
        '--image-root',str(root/'data/images'),'--model','/original/base',
        '--audit-json',str(root/'results/v5-audit.json'),'--dry-run'])
    import json
    result=json.loads(capsys.readouterr().out)
    assert code==0 and result['kind']=='stage1-v5-witness'
    assert len(result['node_specs'])==2
    assert result['node_specs'][0]['command'][-1]=='/remote/ILVR/results/v5-dry-protocol/stage1-v5-witness.json'


def test_audit_reference_profile_is_bound_to_actual_composed_reference():
    from types import SimpleNamespace
    audited=dict(profile_identity=dict(base_content_sha256='a'*64,processor_sha256='b'*64))
    assert hasattr(api(),'validate_audit_reference'), 'audit-to-runtime profile binding missing'
    api().validate_audit_reference(audited,SimpleNamespace(base_content_sha256='a'*64,processor_sha256='b'*64))
    with pytest.raises(ValueError):
        api().validate_audit_reference(audited,SimpleNamespace(base_content_sha256='c'*64,processor_sha256='b'*64))
