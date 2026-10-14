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


def geometry_row(source_index=17, **changes):
    row=dict(source_index=source_index,source_id=str(source_index),original_blocks=2,
        branch_count=4,prefix_length=40,suffix_length=20,selected_queries=8,
        prefix_source_item=3,suffix_source_item=1,projection_chunk_positions=128,
        dense_bias_elements=4800,replicated_prefix_tokens=160,padded_suffix_tokens=80,
        clean_length=64,original_image_grid_thw=[[1,2,2]],helper_image_grid_thw=[[1,2,2],[1,2,2]],
        original_pixel_shape=[4,12],helper_pixel_shape=[8,12])
    row.update(changes)
    return row


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
    assert geometry.get('padded_suffix_tokens')==80


def test_worst_geometry_keeps_non_dominated_actual_rows():
    rows=[geometry_row(1,dense_bias_elements=100,replicated_prefix_tokens=900,selected_queries=5),
          geometry_row(2,dense_bias_elements=1000,replicated_prefix_tokens=100,selected_queries=50),
          geometry_row(3,branch_count=2,prefix_length=10,suffix_length=1,clean_length=32,
              padded_suffix_tokens=2,dense_bias_elements=10,replicated_prefix_tokens=10,selected_queries=1)]
    assert [r['source_index'] for r in api().maximum_geometries(rows)]==[1,2]


def test_frontier_retains_reviewed_padded_suffix_counterexample():
    a=V5BatchMetadata((V5Block(0,0,1991,2000,tuple(range(2001,3501)),(1,)*1500,()),))
    b=V5BatchMetadata(tuple(V5Block(0,i,15*(i+1)-9,15*(i+1),
        tuple(range(151,751)) if i==9 else (15*(i+1)+1,),
        (1,)*600 if i==9 else (1,),()) for i in range(10)))
    # Equal visual signatures isolate the previously discarded packed shape.
    rows=[dict(geometry_row(),**api().packed_geometry(a,source_index=0,source_id='A'),clean_length=3501),
          dict(geometry_row(),**api().packed_geometry(b,source_index=1,source_id='B'),clean_length=751)]
    assert [r['source_index'] for r in api().maximum_geometries(rows)]==[0,1]
    assert [(r['dense_bias_elements'],r['replicated_prefix_tokens'],r['selected_queries'],
             r['padded_suffix_tokens']) for r in rows]==[(10500000,4000,3000,3000),(9000000,3000,1218,12000)]


def test_frontier_retains_independent_clean_extent_and_prunes_equal_visual_dominance():
    smaller=geometry_row(2,branch_count=2,prefix_length=20,suffix_length=10,selected_queries=4,
        dense_bias_elements=600,replicated_prefix_tokens=40,padded_suffix_tokens=20,clean_length=32)
    rows=[geometry_row(1),smaller,dict(smaller,source_index=3,clean_length=80),geometry_row(4)]
    assert [r['source_index'] for r in api().maximum_geometries(rows)]==[1,3,4]
    assert [r['source_index'] for r in api().maximum_geometries(
        [geometry_row(1),geometry_row(2,clean_length=80)])]==[2]


@pytest.mark.parametrize('field,value',[
    ('original_image_grid_thw',[[1,1,4]]),('helper_image_grid_thw',[[1,2,2],[1,1,4]]),
    ('original_pixel_shape',[4,24]),('helper_pixel_shape',[8,24])])
def test_frontier_does_not_compare_distinct_actual_visual_signatures(field,value):
    smaller=geometry_row(2,branch_count=2,prefix_length=20,suffix_length=10,selected_queries=4,
        dense_bias_elements=600,replicated_prefix_tokens=40,padded_suffix_tokens=20,clean_length=32)
    smaller[field]=value
    assert [r['source_index'] for r in api().maximum_geometries([geometry_row(1),smaller])]==[1,2]


def test_collated_geometry_reads_actual_tensor_shapes_without_consuming_helpers():
    from tests.stage1.v5.test_metadata import source_collator,source_example
    batch=source_collator(enabled=True)([source_example(['pre',None,'xy',None,'z'])])
    assert hasattr(api(),'collated_geometry'), 'whole-step collated shape producer missing'
    geometry=api().collated_geometry(batch)
    assert geometry['source_index']==17 and geometry['source_id']=='source-17'
    assert geometry['original_image_grid_thw']==[[1,2,2]]
    assert geometry['helper_image_grid_thw']==[[1,2,2],[1,2,2]]
    assert geometry['original_pixel_shape']==[4,1] and geometry['helper_pixel_shape']==[8,1]
    assert geometry['clean_length']==batch['input_ids'].shape[1]
    assert 'pixel_values_latent' in batch and 'image_grid_thw_latent' in batch
    # The admission extent is the live collated tensor, not a cached sample length.
    batch['input_ids']=torch.cat([batch['input_ids'],torch.ones(1,7,dtype=torch.long)],dim=1)
    assert api().collated_geometry(batch)['clean_length']==geometry['clean_length']+7
    del batch['pixel_values_latent']
    with pytest.raises(ValueError,match='geometry'):
        api().collated_geometry(batch)


def empty_geometry_row(source_index=29, **changes):
    row=geometry_row(source_index,branch_count=0,prefix_length=0,suffix_length=0,selected_queries=0,
        prefix_source_item=None,suffix_source_item=None,dense_bias_elements=0,
        replicated_prefix_tokens=0,padded_suffix_tokens=0,clean_length=80)
    row.update(changes)
    return row


def test_frontier_preserves_empty_target_source_with_independent_clean_or_visual_extent():
    empty=empty_geometry_row()
    assert [g['source_index'] for g in api().maximum_geometries([geometry_row(),empty])]==[17,29]
    empty['clean_length']=32; empty['helper_image_grid_thw']=[[1,4,4]]
    empty['helper_pixel_shape']=[16,12]
    assert [g['source_index'] for g in api().maximum_geometries([geometry_row(),empty])]==[17,29]


def test_production_witness_does_not_reject_empty_target_frontier_before_composition(monkeypatch):
    from types import SimpleNamespace
    import stage1.composition
    def stop_before_model_construction(_args):
        raise RuntimeError('empty frontier reached normal composition')
    monkeypatch.setattr(stage1.composition,'build_stage1_training_components',stop_before_model_construction)
    args=SimpleNamespace(run_id='v5-empty-entry',model=Path('/unused/base'),data_path=Path('/unused/data'),
        image_root=Path('/unused/images'),deepspeed=Path('configs/deepspeed_zero1.json'))
    with pytest.raises(RuntimeError,match='empty frontier reached normal composition'):
        api().production_witness(args,dict(max_packed_geometry=[empty_geometry_row()],optimizer_updates_for_one_exposure=1))


def test_audit_uses_actual_parent_loader_sampler_and_tail(tmp_path):
    report=api().audit_prepared_loader(list(range(37)),output_dir=tmp_path)
    assert report['prepared_loader_microbatches_per_rank']==[10]*4
    assert report['optimizer_updates_for_one_exposure']==3
    assert report['final_window_microbatches']==2
    occurrences=report['sampler_occurrences']
    assert len(occurrences)==40 and set(r['source_index'] for r in occurrences)==set(range(37))
    assert sorted(r['repeat_count'] for r in report['source_occurrences'])==[1]*34+[2]*3
    assert report['native_tail_denominator']==16


def valid_records(geometry=None):
    from copy import deepcopy
    import hashlib,json
    if geometry is None: geometry=[geometry_row()]
    audited,binding=binding_fixture()
    fixture=deepcopy(binding['contract'])
    fixture['schedule'].update(dataset_samples=16*len(geometry),prepared_microbatches_per_rank=4*len(geometry),
        configured_max_steps=len(geometry),parent_epoch_update_budget=len(geometry),
        one_exposure_window_updates=len(geometry),final_window_microbatches=4)
    digest=hashlib.sha256(json.dumps(fixture,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return [dict(rank=r,world_size=4,run_id='v5-protocol',entry=True,exit=True,
        reference_before='b'*64,reference_after='b'*64,contract_digest=digest,
        runtime_binding=deepcopy(binding),fixture_contract=deepcopy(fixture),
        tiny_windows=[dict(global_original_steps=n,configured_gas=4,actual_microbatches=m,
            optimizer_step=i,learning_rate=lr,native_denominator=16,
            all_rank_original_counts=([[2,1,0,0],[1,0,1,0],[0,0,0,0],[1,2,0,1]] if m==4 else [[1,0],[1,1],[0,0],[0,2]]),
            gradient_error=0.,update_error=0.,gradient_norm=1.) for i,(n,m,lr) in enumerate(((9,4,0.),(9,4,5e-6),(5,2,1e-5)))],
        continuations=[dict(saved_step=s,sources=list(range(4*s,10)),updates=3-s,final_step=3,
            learning_rates=([5e-6,1e-5] if s==1 else [1e-5])) for s in (1,2)],
        hooks={k:4*(sum(g['selected_queries']>0 for g in geometry) if k in ('reference','packed')
                    else len(geometry)) for k in ('helper','F','reference','packed','backward')},
        production=dict(global_step=len(geometry),microbatches=4*len(geometry),finite_loss=True,
            gradient_norms=[1.]*len(geometry),gradient_norm_steps=list(range(len(geometry))),peak_allocated=123,peak_reserved=456,
            source_windows=[dict(step=i,source_index=g['source_index'],microbatches=[g['source_index']]*4) for i,g in enumerate(geometry)],
            optimizer_steps=[dict(step=i,successful=True) for i in range(len(geometry))],
            source_indices=sorted(g['source_index'] for g in geometry),geometry=deepcopy(geometry))) for r in range(4)]


@pytest.mark.parametrize('mutation', ['rank','denominator','counts','gradient','reference','continuation','hook','extra_call','geometry','norm'])
def test_rank_verification_rejects_corrupt_runtime_evidence(mutation):
    records=valid_records()
    kwargs=dict(run_id='v5-protocol',audited=binding_fixture()[0],expected_geometry=[geometry_row()])
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
        dict(saved_step=1,sources=[4,5,6,7,8,9],updates=2,final_step=3,learning_rates=[5e-6,1e-5]),
        dict(saved_step=2,sources=[8,9],updates=1,final_step=3,learning_rates=[1e-5])]


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
    assert observed['geometry'][7].get('clean_length')==26
    assert observed['geometry'][7].get('padded_suffix_tokens')==20
    assert observed['geometry'][7].get('original_image_grid_thw')==[[1,2,2]]
    assert observed['geometry'][7].get('helper_image_grid_thw')==[[1,2,2],[1,2,2]]
    assert observed['geometry'][7].get('original_pixel_shape')==[4,12]
    assert observed['geometry'][7].get('helper_pixel_shape')==[8,12]
    # Native target preparation consumes a working copy; the observer retains
    # both helper shapes while the original collated batch stays reusable.
    assert 'pixel_values_latent' in batch and 'image_grid_thw_latent' in batch
    trainer._successful_optimizer_steps=0
    trainer._validate_gradient_norm(trainer.model)
    trainer.optimizer.step()
    trainer._finish_optimizer_step(trainer.model,successful=True)
    assert observed['source_windows']==[dict(step=0,source_index=7,microbatches=[7])]
    assert observed['optimizer_steps']==[dict(step=0,successful=True)]
    assert observed['gradient_norm_steps']==[0]
    assert len(observed['gradient_norms'])==1 and observed['gradient_norms'][0]>0
    for handle in observed['handles']: handle.remove()


@pytest.mark.parametrize('field',['input_ids','image_grid_thw','image_grid_thw_latent',
                                'pixel_values','pixel_values_latent'])
def test_runtime_observer_rejects_changed_collated_shape_before_helper(tmp_path,field):
    from tests.stage1.v5.test_trainer import integrated_fixture
    trainer,batch,*_=integrated_fixture(tmp_path,real_helper=True)
    expected=geometry_row(7,source_id='cpu',prefix_length=19,suffix_length=5,selected_queries=16,
        prefix_source_item=1,suffix_source_item=1,dense_bias_elements=480,
        replicated_prefix_tokens=76,padded_suffix_tokens=20,clean_length=26)
    observed=api().observe_production_trainer(trainer,expected_geometry=[expected])
    if field=='input_ids':
        batch[field]=torch.cat([batch[field],torch.ones(1,1,dtype=torch.long)],dim=1)
    elif 'grid' in field:
        batch[field][0]=torch.tensor([1,1,4])
    else:
        batch[field]=torch.cat([batch[field],batch[field][:,:1]],dim=1)
    batches,_=trainer.get_batch_samples(iter([batch]),4)
    try:
        with pytest.raises(ValueError,match='geometry'):
            trainer.training_step(trainer.model,batches[0])
        assert dict(observed['hooks'])=={}
        assert 'pixel_values_latent' in batch and 'image_grid_thw_latent' in batch
    finally:
        for handle in observed['handles']: handle.remove()


def test_runtime_observer_covers_real_helper_empty_target_shape(tmp_path):
    from tests.stage1.v5.test_trainer import integrated_fixture
    trainer,batch,*_=integrated_fixture(tmp_path,real_helper=True,empty=True)
    expected=empty_geometry_row(7,source_id='cpu',clean_length=26)
    observed=api().observe_production_trainer(trainer,expected_geometry=[expected])
    batches,_=trainer.get_batch_samples(iter([batch]),4)
    try:
        assert torch.isfinite(trainer.training_step(trainer.model,batches[0]))
        assert dict(observed['hooks'])==dict(helper=1,F=1,backward=1)
        assert observed['geometry'][7]==expected
    finally:
        for handle in observed['handles']: handle.remove()


@pytest.mark.parametrize('all_empty',[False,True])
def test_admission_accepts_conditional_calls_for_every_empty_target_source(all_empty):
    geometry=[empty_geometry_row()] if all_empty else [geometry_row(),empty_geometry_row()]
    records=valid_records(geometry)
    kwargs=dict(run_id='v5-protocol',expected_geometry=geometry,audited=binding_fixture()[0])
    assert api().validate_rank_records(records,**kwargs)['status']=='pass'
    records[0]['hooks']['packed']+=1
    with pytest.raises(ValueError): api().validate_rank_records(records,**kwargs)
    records=valid_records(geometry); records[0]['hooks']['helper']-=1
    with pytest.raises(ValueError): api().validate_rank_records(records,**kwargs)


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


def test_admission_covers_joint_frontier_row_without_coordinate_maximum():
    rows=[geometry_row(i,branch_count=2,prefix_length=p,suffix_length=s,selected_queries=q,
               dense_bias_elements=2*s*(p+s),replicated_prefix_tokens=2*p,padded_suffix_tokens=2*s)
          for i,p,s,q in ((1,1000,1,2),(2,100,1300,2),(3,10,200,400),(4,990,900,390))]
    frontier=api().maximum_geometries(rows)
    assert [g['source_index'] for g in frontier]==[1,2,3,4]
    assert api().selected_admission_geometry(dict(max_packed_geometry=frontier))==rows


@pytest.mark.parametrize('mutation',['insufficient_window','uneven_sources','missing_norm','order','rank_order','applied_step','norm_step'])
def test_admission_rejects_incomplete_or_uneven_source_windows(mutation):
    from copy import deepcopy
    geometry=[geometry_row(17),geometry_row(29,dense_bias_elements=9000)]
    records=valid_records(geometry)
    for r in records:
        r['production'].update(geometry=deepcopy(geometry),source_indices=[17,29],
            source_windows=[dict(step=i,source_index=s,microbatches=[s]*4) for i,s in enumerate((17,29))],
            gradient_norms=[1.,2.])
    assert api().validate_rank_records(records,run_id='v5-protocol',expected_geometry=geometry,audited=binding_fixture()[0])['status']=='pass'
    p=records[2]['production']
    if mutation=='insufficient_window':
        for r in records:
            r['production'].update(global_step=1,microbatches=4,gradient_norms=[1.])
            r['hooks']={k:4 for k in r['hooks']}
    elif mutation=='uneven_sources': p['source_windows'][0]['microbatches']=[17,17,17,29]
    elif mutation=='missing_norm': p['gradient_norms']=[1.]
    elif mutation=='norm_step': p['gradient_norm_steps']=[0,0]
    elif mutation=='applied_step': p['optimizer_steps'][0]['successful']=False
    elif mutation=='order':
        for r in records: r['production']['source_windows'].reverse()
    else: p['source_windows'].reverse()
    with pytest.raises(ValueError): api().validate_rank_records(records,run_id='v5-protocol',expected_geometry=geometry,audited=binding_fixture()[0])


@pytest.mark.parametrize('field',[
    'branch_count','prefix_length','suffix_length','selected_queries','padded_suffix_tokens','clean_length',
    'original_image_grid_thw','helper_image_grid_thw','original_pixel_shape','helper_pixel_shape'])
def test_admission_rejects_changed_or_missing_whole_step_shape(field):
    from copy import deepcopy
    geometry=[geometry_row()]
    records=valid_records(geometry)
    del records[0]['production']['geometry'][0][field]
    with pytest.raises(ValueError):
        api().validate_rank_records(records,run_id='v5-protocol',expected_geometry=geometry,audited=binding_fixture()[0])
    # A legacy/incomplete audit must fail even when every rank agrees with it.
    incomplete=deepcopy(geometry); del incomplete[0][field]
    incomplete_records=valid_records(geometry)
    for record in incomplete_records:
        del record['production']['geometry'][0][field]
    with pytest.raises(ValueError,match='geometry'):
        api().validate_rank_records(incomplete_records,run_id='v5-protocol',
            expected_geometry=incomplete,audited=binding_fixture()[0])
    with pytest.raises(ValueError,match='geometry'):
        api().selected_admission_geometry(dict(max_packed_geometry=incomplete))
    records=valid_records(geometry)
    old=records[0]['production']['geometry'][0][field]
    records[0]['production']['geometry'][0][field]=old+1 if isinstance(old,int) else []
    with pytest.raises(ValueError):
        api().validate_rank_records(records,run_id='v5-protocol',expected_geometry=geometry,audited=binding_fixture()[0])


def test_tiny_parent_oracle_rejects_wrong_final_tail_learning_rate(tmp_path,monkeypatch):
    from transformers import Trainer
    original=Trainer.create_scheduler
    def incorrect_schedule(self,*args,**kwargs):
        scheduler=original(self,*args,**kwargs)
        for i,fn in enumerate(scheduler.lr_lambdas):
            scheduler.lr_lambdas[i]=lambda step,fn=fn:fn(step)*(2 if step==2 else 1)
        return scheduler
    monkeypatch.setattr(Trainer,'create_scheduler',incorrect_schedule)
    directory=Path(__file__).resolve().parents[3]/'checkpoints/stage1-v5-task7'/tmp_path.parent.name/tmp_path.name
    with pytest.raises(ValueError,match='learning rate'):
        api().tiny_engine_witness(directory,use_cpu=True)


def binding_fixture():
    from tests.stage1.v5.test_resume import contract
    payload=contract()
    execution=dict(per_device_train_batch_size=1,gradient_accumulation_steps=4,bf16=True,seed=42,
        learning_rate=5e-5,warmup_steps=10,weight_decay=.01,max_grad_norm=1.,gradient_checkpointing=True,
        dataloader_num_workers=0,optim='adamw_torch_fused',lr_scheduler_type='linear',
        tuning_method='dora',ema_tau=1.,zero_stage=1,attention_implementation='sdpa',
        vision_attention_implementation='flash_attention_2',latent_size=8,max_sequence_length=32768,
        lora_r=32,lora_alpha=64,lora_dropout=.05,
        lora_target_modules='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
        loss_chunk_positions=128,compress_strategy='average',coverage_p=.9,image_pool_k=8,helper_group_L=256,textual_states='previous',model='a'*64,source_sha256=dict(v5_trainer='f'*64,stage1_trainer='f'*64,transformers_trainer='f'*64),packages=dict(torch='actual-torch',accelerate='actual-accelerate'),
        deepspeed_profile=dict(enabled=True,zero_stage=1,world_size=4,config_path='d'*64,config_digest='d'*64))
    payload['execution']=execution
    payload['schedule']=dict(dataset_samples=37,prepared_microbatches_per_rank=10,
        gradient_accumulation_steps=4,effective_global_batch=16,world_size=4,configured_max_steps=3,
        num_train_epochs=1.,one_exposure_window_updates=3,parent_epoch_update_budget=3,
        final_window_microbatches=2,dataloader_drop_last=False,
        tail_native_denominator='configured_gas_world_size',
        auxiliary_denominator='actual_window_original_blocks_including_empty')
    occurrences=[dict(rank=r,microbatch=i,source_index=(4*i+r)%37,source_id=str((4*i+r)%37))
                 for r in range(4) for i in range(10)]
    original=dict(retained_samples=37,prepared_loader_microbatches_per_rank=[10]*4,
        sampler_occurrences=occurrences,source_occurrences=[dict(source_index=i,repeat_count=2 if i<3 else 1) for i in range(37)],
        configured_gas=4,optimizer_updates_for_one_exposure=3,final_window_microbatches=2,native_tail_denominator=16,
        sampler_profile=dict(split_batches=False,even_batches=True,use_seedable_sampler=True,data_seed=None))
    audit={**original,'profile_identity':dict(base_content_sha256='a'*64,processor_sha256='c'*64,
        packages=execution['packages'],deepspeed_config_sha256='d'*64), 'source_identity':dict(data_sha256='e'*64),
        'code_identity':dict(files={'src/stage1/v5_trainer.py':'f'*64,'src/stage1/trainer.py':'f'*64,'transformers/src/transformers/trainer.py':'f'*64})}
    binding=dict(contract=payload,original_loader=original,
        input_identity=dict(data_sha256='e'*64,base_content_sha256='a'*64,processor_sha256='c'*64,
                            source_sha256=audit['code_identity']['files']))
    return audit,binding


@pytest.mark.parametrize('mutation',['coefficient','r','warmup','package','sampler','budget','occurrences','tail','input','helper_group','code_profile'])
def test_runtime_binding_rejects_stale_audit_or_different_effective_profile(mutation):
    audited,binding=binding_fixture()
    assert hasattr(api(),'validate_runtime_binding'), 'expected-bound original loader/profile missing'
    api().validate_runtime_binding(audited,binding)
    if mutation=='coefficient': binding['contract']['coefficients']['ce_weight']=1.
    elif mutation=='r': binding['contract']['execution']['lora_r']=4
    elif mutation=='warmup': binding['contract']['execution']['warmup_steps']=0
    elif mutation=='package': audited['profile_identity']['packages']=dict(torch='stale',accelerate='actual-accelerate')
    elif mutation=='sampler': audited['sampler_profile']=dict(audited['sampler_profile'],even_batches=False)
    elif mutation=='budget': audited['optimizer_updates_for_one_exposure']=2
    elif mutation=='occurrences': audited['sampler_occurrences']=list(reversed(audited['sampler_occurrences']))
    elif mutation=='tail': audited['final_window_microbatches']=1
    elif mutation=='code_profile': binding['contract']['execution']['source_sha256']['v5_trainer']='0'*64
    elif mutation=='helper_group': binding['contract']['execution']['helper_group_L']=128
    else: binding['input_identity']['data_sha256']='0'*64
    with pytest.raises(ValueError): api().validate_runtime_binding(audited,binding)


def test_original_loader_observer_reads_actual_prepared_source_order(tmp_path):
    from types import SimpleNamespace
    from transformers import Trainer,TrainingArguments
    # Metadata are supplied by this controlled collator; sampler, preparation,
    # device placement and iterator dispatch are actual parent/Accelerate code.
    def collate(rows):
        return dict(input_ids=torch.tensor(rows),stage1_batch_metadata=SimpleNamespace(
            samples=[SimpleNamespace(source_index=i,source_id=str(i)) for i in rows]))
    trainer=Trainer(model=torch.nn.Linear(1,1),args=TrainingArguments(str(tmp_path),
        use_cpu=True,per_device_train_batch_size=1,seed=42,report_to=[],remove_unused_columns=False),
        train_dataset=list(range(10)),data_collator=collate)
    loader=trainer.get_train_dataloader()
    observed=api().observe_original_loader_rank(trainer,loader,0)
    assert observed['length']==10
    assert observed['occurrences']==[dict(rank=0,microbatch=j,source_index=i,source_id=str(i))
        for j,i in enumerate([2,6,1,8,4,5,0,9,3,7])]
    assert observed['sampler_profile']['use_seedable_sampler'] is True


@pytest.mark.parametrize('mutation',['digest','fixture_coefficient','original_budget','lr','continuation_lr'])
def test_admission_rejects_unbound_common_profile_and_learning_rate(mutation):
    audited,_=binding_fixture();records=valid_records()
    if mutation=='digest':
        for r in records: r['contract_digest']='unbound-common-profile'
    elif mutation=='fixture_coefficient':
        for r in records: r['fixture_contract']['coefficients']['ce_weight']=1.
    elif mutation=='original_budget':
        for r in records: r['runtime_binding']['contract']['schedule']['configured_max_steps']=1
    elif mutation=='lr': records[0]['tiny_windows'][2]['learning_rate']=2e-5
    else: records[0]['continuations'][1]['learning_rates']=[2e-5]
    with pytest.raises(ValueError):
        api().validate_rank_records(records,run_id='v5-protocol',expected_geometry=[geometry_row()],audited=audited)
