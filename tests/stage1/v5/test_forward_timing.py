"""Timing must observe the real v5 graph without changing its output or gradients."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.stage1.v5.test_trainer import integrated_fixture


def api():
    assert importlib.util.find_spec('scripts.profile_stage1_v5'), 'bounded timing observer missing'
    from scripts import profile_stage1_v5
    return profile_stage1_v5


class EventClock:
    """CPU-only stand-in for recorded CUDA events, drained only after replay."""
    def __init__(self):
        self.count = 0
        self.drains = 0
    def record(self):
        self.count += 1
        return self.count
    def drain(self):
        self.drains += 1
    def elapsed(self, start, end):
        assert self.drains == 1
        return float(end - start)


def prepare(trainer, batch):
    return trainer.get_batch_samples(iter([batch]), 4)[0][0]


@pytest.mark.parametrize('empty', [False, True])
def test_observer_real_parent_preserves_loss_gradients_and_conditional_order(tmp_path, empty):
    module = api()
    baseline, batch, _, _, _, baseline_values = integrated_fixture(tmp_path/'baseline', empty=empty)
    observed, batch2, calls, _, _, observed_values = integrated_fixture(tmp_path/'observed', empty=empty)
    observed.model.load_state_dict(baseline.model.state_dict())
    observed.ema_teacher.model.load_state_dict(baseline.ema_teacher.model.state_dict())
    # Fixture helper values are separately random; retain identical targets.
    observed_values.copy_(baseline_values)
    for trainer in (baseline,observed):
        trainer.v5_config=replace(trainer.v5_config,selection='clean',preference_entropy_top_percent=20,lambda_topk_mass=.05)
    reference_state = deepcopy(observed.ema_teacher.model.state_dict())
    rng = torch.random.get_rng_state()
    expected = baseline.training_step(baseline.model, prepare(baseline,batch))
    torch.random.set_rng_state(rng)
    clock = EventClock()
    originals = (observed.training_step, observed.compute_loss, observed.model.forward,
                 observed.ema_teacher.model.forward, observed.accelerator.backward)
    inventory={id(m):m.forward for owner in (observed.model,observed.ema_teacher.model) for m in owner.modules()}
    with module.ForwardTimingObserver(observed, rank=2, event_clock=clock) as observer:
        actual = observed.training_step(observed.model, prepare(observed,batch2))
        assert clock.drains == 0
    rows = observer.finish()
    assert clock.drains == 1
    assert all(inventory[id(m)]==m.forward for owner in (observed.model,observed.ema_teacher.model) for m in owner.modules())
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for (_,left),(_,right) in zip(baseline.model.named_parameters(), observed.model.named_parameters()):
        if left.requires_grad:
            torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)
    assert all(torch.equal(v, reference_state[k]) for k,v in observed.ema_teacher.model.state_dict().items())
    assert originals == (observed.training_step, observed.compute_loss, observed.model.forward,
        observed.ema_teacher.model.forward, observed.accelerator.backward)
    row, = rows
    assert row['rank'] == 2 and row['finite_loss'] is True
    assert row['original_order'] == (['helper','clean'] if empty else ['helper','clean','reference','packed'])
    assert row['geometry']['query_count'] == (0 if empty else 8)
    assert row['accounting']['step_remainder_cpu_ms'] >= 0
    assert row['accounting']['step_remainder_cuda_ms'] >= 0
    spans = row['spans']
    backward, = [s for s in spans if s['name']=='backward']
    assert all(s['cpu_end_ms'] <= backward['cpu_start_ms'] for s in spans if s['name'] in row['original_order'])
    assert all(s['cuda_ms'] >= 0 for s in spans)
    if not empty:
        packed, = [s for s in spans if s['name']=='packed']
        decoder, = [s for s in spans if s['name']=='packed_decoder']
        assert decoder['parent'] == packed['id']
        assert row['packed_partition']['cpu_ms'] == pytest.approx(packed['cpu_ms'])
        partition=row['packed_partition']
        assert partition['setup_cuda_ms']+decoder['cuda_ms']+partition['projection_and_remainder_cuda_ms'] == pytest.approx(packed['cuda_ms'])
    # Actual checkpoint replay is nested under backward, never another original.
    recompute = [s for s in spans if s['name']=='recompute_layer']
    assert recompute and all(s['cpu_start_ms'] >= backward['cpu_start_ms'] for s in recompute)


def test_observer_removes_wrappers_on_helper_exception(tmp_path):
    module=api()
    trainer,batch,*_=integrated_fixture(tmp_path,fail=True)
    old = trainer.model.forward,trainer.target_builder.build,trainer.training_step
    with pytest.raises(ValueError,match='helper failure'):
        with module.ForwardTimingObserver(trainer,rank=0,event_clock=EventClock()):
            trainer.training_step(trainer.model,prepare(trainer,batch))
    assert old == (trainer.model.forward,trainer.target_builder.build,trainer.training_step)


def test_first_nail_rejects_missing_original_call_and_stops_next_collection(tmp_path):
    module=api()
    trainer,batch,*_=integrated_fixture(tmp_path)
    with module.ForwardTimingObserver(trainer,rank=0,event_clock=EventClock()) as observer:
        # A missing helper model call can otherwise return valid target values.
        real=trainer.target_builder.build
        def no_helper(*args):
            teacher=args[0]
            old=teacher.forward
            teacher.forward=lambda *a,**k: None
            try: return real(*args)
            finally: teacher.forward=old
        trainer.target_builder.build=no_helper
        with pytest.raises(ValueError,match='original call order'):
            trainer.training_step(trainer.model,prepare(trainer,batch))
        assert observer.failed
        with pytest.raises(ValueError,match='attribution failed'):
            trainer.training_step(trainer.model,prepare(trainer,batch))


def test_four_windows_excludes_warmup_and_requires_real_updates(tmp_path):
    module=api()
    trainer,batch,*_=integrated_fixture(tmp_path)
    trainer._successful_optimizer_steps=0
    clock=EventClock()
    with module.ForwardTimingObserver(trainer,rank=0,event_clock=clock) as observer:
        callback=module.BoundedReplayCallback(observer)
        control=SimpleNamespace(should_training_stop=False)
        for window in range(4):
            trainer.state.global_step=window
            callback.on_step_begin(trainer.args,trainer.state,control)
            for _ in range(4): trainer.training_step(trainer.model,prepare(trainer,batch))
            trainer.optimizer.step();trainer.optimizer.zero_grad()
            trainer._successful_optimizer_steps += 1
            trainer.state.global_step += 1
            callback.on_step_end(trainer.args,trainer.state,control)
        assert control.should_training_stop
    records=observer.finish()
    rank_record=module.rank_record(observer,records,run_id='timing-test',identity={'commit':'a'*40})
    assert rank_record['window_count'] == 4
    assert [r['warmup'] for r in records] == [True]*4+[False]*12
    records4=[deepcopy(rank_record) for _ in range(4)]
    for rank,record in enumerate(records4):
        record['rank']=rank
        for row in record['microbatches']: row['rank']=rank
    result=module.summarize_records(records4,run_id='timing-test')
    assert result['status']=='pass' and result['measured_microbatches']==48
    assert result['phases']['clean']['count']==48
    assert result['phases']['clean']['cpu_ms']['median']>0
    assert result['phases']['backward']['count']==48
    assert len(result['per_rank'])==4
    bad=deepcopy(records4);bad[3]['microbatches'][6]['spans'][0]['cpu_ms']=float('nan')
    with pytest.raises(ValueError,match='finite'): module.summarize_records(bad,run_id='timing-test')
    with pytest.raises(ValueError,match='four ranks'): module.summarize_records(records4[:3],run_id='timing-test')
    badcuda=deepcopy(records4)
    badcuda[1]['microbatches'][4]['spans'][0]['cuda_ms']+=10000
    with pytest.raises(ValueError,match='CUDA interval'): module.summarize_records(badcuda,run_id='timing-test')
    badorder=deepcopy(records4)
    badorder[2]['microbatches'][8]['original_order'].append('clean')
    with pytest.raises(ValueError,match='original call order'): module.summarize_records(badorder,run_id='timing-test')
    records4[0]['windows'][1]['successful_updates']=0
    with pytest.raises(ValueError,match='optimizer'): module.summarize_records(records4,run_id='timing-test')


def test_coordinator_dry_run_maps_source_and_owned_four_rank_entry(capsys):
    from src.multinode import _main
    root=Path(__file__).resolve().parents[3]
    code=_main(['train-stage1','--stage1-v5-forward-timing','--run-id','v5-timing-dry',
        '--zero-stage','1','--main-process-port','29583','--local-repo-root',str(root),
        '--remote-repo-root','/remote/ILVR','--remote-host-repo-root','/host/ILVR',
        '--source-checkpoint',str(root/'checkpoints/source/checkpoint-2160'),
        '--model',str(root/'checkpoints/pretrained/base'),'--data-path',str(root/'data/TRAIN.jsonl'),
        '--image-root',str(root/'data/images'),'--dry-run'])
    result=json.loads(capsys.readouterr().out)
    assert code==0 and result['kind']=='stage1-v5-forward-timing'
    for rank,spec in enumerate(result['node_specs']):
        command=spec['command'];node='/remote/ILVR' if rank==0 else str(root)
        assert command[command.index('--source-checkpoint')+1]==node+'/checkpoints/source/checkpoint-2160'
        assert command[command.index('--machine_rank')+1]==str(rank)
        assert command[command.index('--num_processes')+1]=='4'
        assert node+'/scripts/profile_stage1_v5.py' in command
        assert dict(spec['environment'])['NCCL_SOCKET_IFNAME']=='bond0.11'
        assert 'node-run' in ' '.join(result['remote'] if rank==0 else result['local'])


def test_replay_cli_inherits_saved_full_schedule_and_rejects_another_objective(tmp_path):
    module=api()
    from stage1.v5_config import Stage1V5Config
    from stage1.v5_resume import build_v5_contract
    from stage1.v5_reference import V5ReferenceIdentity
    from tests.stage1.v5.test_resume import contract
    from utils_deepseed import parse_args
    old=contract()
    saved=build_v5_contract(Stage1V5Config(selection='clean',preference_entropy_top_percent=20,
        lambda_topk_entropy=.1,lambda_topk_mass=.05,preference_topk_k=50),
        V5ReferenceIdentity(**old['reference']),
        ce_weight=.1,sim_weight=.1,schedule=dict(configured_max_steps=2160,num_train_epochs=10,
            dil_denominator='actual_window_selected_dil_tokens'),execution=dict(
            tuning_method='dora',zero_stage=1,ema_tau=1.,per_device_train_batch_size=1,
            gradient_accumulation_steps=4,bf16=True,seed=42,learning_rate=5e-5,warmup_steps=10,
            lora_r=32,lora_alpha=64,lora_dropout=.05,latent_size=8,
            attention_implementation='sdpa',vision_attention_implementation='flash_attention_2',
            gradient_checkpointing=True,textual_states='previous'))
    (tmp_path/'stage1_v5.json').write_text(json.dumps(saved))
    args=SimpleNamespace(source_checkpoint=tmp_path,model=tmp_path/'base',
        data_path=tmp_path/'TRAIN.jsonl',image_root=tmp_path/'images',run_id='timing-test',
        deepspeed=Path(__file__).resolve().parents[3]/'configs/deepspeed_zero1.json')
    module.validate_source_contract(saved)
    cli=module.replay_cli(args)
    parsed=parse_args(cli)
    assert parsed.max_steps==2160 and parsed.epochs==10
    assert parsed.resume_from_checkpoint_path==str(tmp_path)
    assert parsed.stage1_v5_config.selection=='clean' and parsed.lambda_topk_mass==.05
    assert parsed.save_strategy=='no' and parsed.gradient_checkpointing
    wrong=deepcopy(saved);wrong['config']['lambda_topk_mass']=0
    with pytest.raises(ValueError): module.validate_source_contract(wrong)


def test_rank_record_collection_maps_each_global_rank_to_its_node(tmp_path):
    module=api()
    from src.multinode import build_launch_plan
    from src.multinode_runtime import SubprocessCoordinatorRunner
    root=Path(__file__).resolve().parents[3]
    plan=build_launch_plan(stage='stage1',tuning_method='dora',zero_stage=1,
        run_id='timing-map',main_process_port=29583,local_repo_root=root,
        remote_repo_root=Path('/remote/ILVR'),remote_host_repo_root=Path('/host/ILVR'))
    plan=replace(plan,cluster=replace(plan.cluster,local_repo_root=tmp_path/'local'),
        deepspeed_config=tmp_path/'local/configs/deepspeed_zero1.json')
    runner=SubprocessCoordinatorRunner(plan,commands=None,commit='a'*40)
    for rank in (2,3):
        target=tmp_path/'local/results/timing-map'/f'forward-timing-rank-{rank}.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(dict(rank=rank)))
        assert json.loads(runner.read_stage1_v5_forward_timing_record(rank))==dict(rank=rank)
    def remote(command,**kwargs):
        text=' '.join(command)
        assert 'llm_kd' in text and 'korea@172.16.11.41' in text
        assert '-m scripts.profile_stage1_v5' in text
        rank=int(text.rsplit('--rank ',1)[1].rstrip("'"))
        return SimpleNamespace(stdout=json.dumps(dict(rank=rank)),returncode=0)
    runner._run=remote
    for rank in (0,1):
        assert json.loads(runner.read_stage1_v5_forward_timing_record(rank))==dict(rank=rank)
    with pytest.raises(ValueError): runner.read_stage1_v5_forward_timing_record(4)


def test_replay_seeds_original_entry_backend_before_composition(monkeypatch):
    module=api()
    import stage1.composition as composition
    monkeypatch.setattr(torch.backends.cudnn,'deterministic',False)
    monkeypatch.setattr(torch.backends.cudnn,'benchmark',True)
    sentinel=object()
    def loading_boundary(parsed, **kwargs):
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.initial_seed()==42
        return sentinel
    # The expensive model-loading boundary is replaced; seed helper is real.
    monkeypatch.setattr(composition,'build_stage1_training_components',loading_boundary)
    assert module.compose_seeded_replay(SimpleNamespace(seed=42)) is sentinel


def test_incomplete_optimizer_window_is_rejected_before_reporting(tmp_path):
    module=api()
    trainer,*_=integrated_fixture(tmp_path)
    trainer._successful_optimizer_steps=0
    with module.ForwardTimingObserver(trainer,rank=0,event_clock=EventClock()) as observer:
        observer.begin_window()
        with pytest.raises(ValueError,match='successful optimizer'):
            observer.end_window()
    assert not observer.windows


def test_prepared_engine_optimizer_step_is_nested_inside_accelerator_backward(tmp_path):
    module=api()
    trainer,batch,*_=integrated_fixture(tmp_path)
    # Only the prepared-engine boundary is supplied; the optimizer update and
    # parent backward are real. Accelerate's native DS wrapper steps here.
    engine=SimpleNamespace(step=trainer.optimizer.step)
    trainer.accelerator.deepspeed_engine_wrapped=SimpleNamespace(engine=engine)
    real=trainer.accelerator.backward
    def parent_backward(*args,**kwargs):
        result=real(*args,**kwargs)
        engine.step()
        return result
    trainer.accelerator.backward=parent_backward
    before={n:p.detach().clone() for n,p in trainer.model.named_parameters() if p.requires_grad}
    original=engine.step
    with module.ForwardTimingObserver(trainer,rank=0,event_clock=EventClock()) as observer:
        trainer.training_step(trainer.model,prepare(trainer,batch))
    row,=observer.finish()
    span,=[s for s in row['spans'] if s['name']=='deepspeed_step']
    backward,=[s for s in row['spans'] if s['name']=='backward']
    assert span['parent']==backward['id'] and engine.step==original
    assert any(not torch.equal(p,before[n]) for n,p in trainer.model.named_parameters() if n in before)


def test_replay_output_passes_real_cross_experiment_peft_resume_guard(tmp_path, monkeypatch):
    module=api()
    from stage1 import checkpointing
    repository=tmp_path/'repo'
    canonical_root=repository/'checkpoints'
    source=canonical_root/'completed-source'/'checkpoint-2160'
    source.mkdir(parents=True)
    monkeypatch.setattr(module,'REPOSITORY',repository)
    monkeypatch.setattr(checkpointing,'CANONICAL_CHECKPOINT_ROOT',canonical_root)
    args=SimpleNamespace(source_checkpoint=source,model=tmp_path/'base',
        data_path=repository/'data/TRAIN.jsonl',image_root=repository/'data/images',
        run_id='forward-timing-fresh',deepspeed=repository/'configs/deepspeed_zero1.json')
    cli=module.replay_cli(args)
    output=Path(cli[cli.index('--save_model_path')+1])
    # Exercise the unchanged cross-experiment guard that real composition calls.
    # The former <run-id>/replay destination is rejected by this real validator.
    checkpointing._validate_peft_resume_location(output,args.source_checkpoint)


@pytest.mark.parametrize('drift', ['none', 'image', 'tokens'])
def test_replay_processor_roundtrip_restores_only_exact_original_identity(tmp_path, drift):
    module=api()
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoProcessor, Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2_5_VLProcessor
    from stage1.composition import LATENT_TOKEN_STRINGS, _add_fresh_latent_tokens
    from stage1.v5_reference import _processor_identity
    from tests.stage1.v5.test_resume import contract
    backend=Tokenizer(WordLevel({'<unk>':0,'<eos>':1,'<pad>':2,'hello':3,'world':4},unk_token='<unk>'))
    backend.pre_tokenizer=Whitespace()
    tokenizer=Qwen2TokenizerFast(tokenizer_object=backend,unk_token='<unk>',eos_token='<eos>',pad_token='<pad>')
    _add_fresh_latent_tokens(tokenizer)
    processor=Qwen2_5_VLProcessor(image_processor=Qwen2VLImageProcessor(min_pixels=16,max_pixels=64),
        tokenizer=tokenizer,chat_template='{{ messages }}')
    expected=_processor_identity(processor)
    checkpoint=tmp_path/'checkpoint-2160'
    processor.save_pretrained(checkpoint)
    saved=contract()
    saved['reference']['processor_sha256']=expected
    (checkpoint/'stage1_v5.json').write_text(json.dumps(saved))
    reloaded=AutoProcessor.from_pretrained(checkpoint,local_files_only=True)
    # Real HF save/reload loses the AddedToken representation in this map.
    assert _processor_identity(reloaded, normalize_added_tokens=False)!=expected
    assert _processor_identity(reloaded)==expected
    sample='hello '+ ' '.join(LATENT_TOKEN_STRINGS)+' world'
    before_vocab=reloaded.tokenizer.get_vocab()
    before_ids=tuple(reloaded.tokenizer.convert_tokens_to_ids(t) for t in LATENT_TOKEN_STRINGS)
    before_encoding=reloaded.tokenizer.encode(sample)
    before_size=len(reloaded.tokenizer)
    if drift=='image':
        config=checkpoint/'preprocessor_config.json'
        settings=json.loads(config.read_text())
        settings['min_pixels']+=1
        config.write_text(json.dumps(settings))
    elif drift=='tokens':
        reloaded.tokenizer.additional_special_tokens=list(LATENT_TOKEN_STRINGS[:2])
        reloaded.save_pretrained(checkpoint)
    if drift!='none':
        with pytest.raises(ValueError,match='processor.*identity|latent.*tokens'):
            module.load_replay_processor(checkpoint,local_files_only=True)
    else:
        restored=module.load_replay_processor(checkpoint,local_files_only=True)
        assert _processor_identity(restored)==expected
        assert restored.tokenizer.get_vocab()==before_vocab
        assert len(restored.tokenizer)==before_size
        assert tuple(restored.tokenizer.convert_tokens_to_ids(t) for t in LATENT_TOKEN_STRINGS)==before_ids
        assert restored.tokenizer.encode(sample)==before_encoding
        assert all(not isinstance(t,str) for t in restored.tokenizer.special_tokens_map_extended['additional_special_tokens'])


def test_remote_timing_record_entry_imports_real_modules_without_pythonpath(tmp_path):
    import os
    import shlex
    import shutil
    import subprocess
    from src.multinode import build_launch_plan
    from src.multinode_runtime import SubprocessCoordinatorRunner
    root=Path(__file__).resolve().parents[3]
    repository=tmp_path/'remote-repo'
    for relative in ('scripts/profile_stage1_v5.py','src/multinode.py','src/multinode_runtime.py',
                     'src/artifact_paths.py','src/deepspeed_runtime.py'):
        destination=repository/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/relative,destination)
    (repository/'venv').mkdir()
    (repository/'venv/ilvr-venv').symlink_to(root/'venv/ilvr-venv',target_is_directory=True)
    run_id='timing-reader-subprocess'
    payload=dict(rank=0,run_id=run_id,identity=dict(commit='87087b3a42dc3131084b242e610f6586c2efc46f'))
    serialized=json.dumps(payload,sort_keys=True)
    record=repository/'results'/run_id/'forward-timing-rank-0.json'
    record.parent.mkdir(parents=True)
    record.write_text(serialized)
    plan=build_launch_plan(stage='stage1',tuning_method='dora',zero_stage=1,
        run_id=run_id,main_process_port=29583,local_repo_root=root,
        remote_repo_root=repository,remote_host_repo_root=tmp_path/'host-repo')
    environment=os.environ.copy()
    environment.pop('PYTHONPATH',None)
    environment['PYTHONDONTWRITEBYTECODE']='1'
    def local_container_boundary(command, *, input_bytes, timeout_seconds):
        # Replace only SSH/container transport. Execute the exact real reader
        # argv and cwd constructed by production, with actual repository imports.
        docker=shlex.split(command[-1])
        working_directory=Path(docker[docker.index('--workdir')+1])
        argv=shlex.split(docker[-1])
        return subprocess.run(argv,cwd=working_directory,env=environment,
            capture_output=True,text=True,check=False,timeout=timeout_seconds)
    runner=SubprocessCoordinatorRunner(plan,commands=None,commit='a'*40,
        command_runner=local_container_boundary)
    raw=runner.read_stage1_v5_forward_timing_record(0)
    assert raw==(serialized+'\n').encode()
    assert json.loads(raw)==payload
