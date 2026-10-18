"""Real entrypoint and four-worker contracts for original-vs-continuous identity."""
import dataclasses
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from tests.evaluation.test_original_ilvr import tiny_original, START, PAD, END
from tests.multinode.test_evaluation_coordinator import _plan, _manifest
from src.multinode_runtime import build_evaluation_commands
from src.multinode_evaluation import write_shard_manifest, merge_shards


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)


class NativeProcessor:
    tokenizer = None
    def __init__(self):
        self.conversations=[]
        self.decoded=[]
        self.tokenizer=self
    def convert_tokens_to_ids(self,token):
        return {'<|latent_start|>':START,'<|latent_pad|>':PAD,'<|latent_end|>':END}[token]
    def __call__(self,token=None,return_tensors=None,**kwargs):
        if token is not None: return {'input_ids':torch.tensor([[self.convert_tokens_to_ids(token)]])}
        assert kwargs['padding'] is True
        return {'input_ids':torch.tensor([[7,8]]),'attention_mask':torch.ones(1,2,dtype=torch.long)}
    def apply_chat_template(self,conversations,**kwargs):
        self.conversations.append(conversations)
        return 'prompt'
    def batch_decode(self,ids,**kwargs):
        self.decoded.append(ids.tolist())
        return ['<|im_start|>user\nQ<|im_end|><|im_start|>assistant\n<|im_end|>']


def test_stage1_cli_selects_original_while_direct_call_default_stays_continuous(monkeypatch):
    module=importlib.import_module('eval')
    monkeypatch.setattr(sys,'argv',['eval.py','--model_dir','model','--test_data_path','data',
        '--task_name','zebra-cot','--runtime_identity_path','runtime'])
    args=module.get_eval_args()
    assert getattr(args,'evaluation_protocol',None) == 'auto'
    assert args.repetition_penalty is None
    assert args.generation_limit_policy is None
    api,model,_=tiny_original([3])
    processor=NativeProcessor()
    result=module.run_one_example(model,processor,{'text_input':'Q'},
        {'max_new_tokens':5,'temperature':0.,'top_p':1.,'do_sample':False}, evaluation_protocol='original')
    assert result['extracted_final_answer'] == ''
    assert processor.decoded == [[[7,8,3]]], 'original parses full raw decode exactly once'
    assert result['evaluation_protocol']=='original'


def test_original_main_uses_original_empty_gold_match_and_all_source_denominator(tmp_path,monkeypatch):
    module=importlib.import_module('eval')
    _,model,_=tiny_original([3])
    processor=NativeProcessor()
    dataset=tmp_path/'test.jsonl'
    dataset.write_text('{}\n{"original_final_answer":""}\n{"original_final_answer":"yes"}\n')
    output=tmp_path/'worker.jsonl'
    monkeypatch.setattr(sys,'argv',['eval.py','--model_dir','model','--test_data_path',str(dataset),
        '--task_name','zebra-cot','--output_json_path',str(output),'--runtime_identity_path',str(tmp_path/'runtime.json')])
    monkeypatch.setattr(module,'load_evaluation_artifacts',lambda **kwargs:(model,processor))
    monkeypatch.setattr(module,'publish_evaluation_runtime_identity',lambda *args,**kwargs:None)
    monkeypatch.setattr(module,'authenticated_dataset_identity',lambda *args,**kwargs:{})
    from step_preference import capacity_evidence
    monkeypatch.setattr(capacity_evidence,'git_identity',lambda:'a'*40)
    module.main()
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    assert [r['match'] for r in rows[:-1]]==[True,True,False]
    assert rows[-1]['success']==2 and rows[-1]['accuracy']==2/3
    assert rows[-1]['evaluation_protocol']=='original'
    assert rows[-1]['execution_identity']['generation_config']['use_cache'] is True


@pytest.mark.parametrize('stage,protocol', [('stage1','original'),('stage2','continuous')])
def test_real_coordinator_workers_resolve_stage_protocol(stage,protocol):
    commands=build_evaluation_commands(_plan(),inference_manifest=_manifest(stage=stage),commit='b'*40)
    for worker in commands.workers:
        command=worker.node_spec.command
        assert '--evaluation-protocol' in command
        assert command[command.index('--evaluation-protocol')+1]==protocol
        if protocol=='original':
            assert '--repetition_penalty' not in command
            assert command[command.index('--generation-limit-policy')+1]=='native'


def test_explicit_continuous_keeps_legacy_limits_and_original_rejects_conflict():
    plan=_plan()
    assert hasattr(plan,'evaluation_protocol'), 'coordinator cannot identify the selected protocol'
    continuous=dataclasses.replace(plan,evaluation_protocol='continuous',generation_limit_policy='external_full_latent_block')
    commands=build_evaluation_commands(continuous,inference_manifest=_manifest(),commit='b'*40)
    assert all(w.node_spec.command[w.node_spec.command.index('--evaluation-protocol')+1]=='continuous' for w in commands.workers)
    with pytest.raises(ValueError,match='original.*native'):
        build_evaluation_commands(dataclasses.replace(plan,evaluation_protocol='original',generation_limit_policy='external_full_latent_block'),
            inference_manifest=_manifest(),commit='b'*40)


def identity(protocol='original',penalty=1.05,backend='Qwen2_5_VLFlashAttention2'):
    return {'evaluation_protocol':protocol,'source_commit':'13a67ed2ee975d6ca56b4e9eba18d68b352b166b' if protocol=='original' else None,
        'generation_config':{'use_cache':True,'eos_token_id':[3,4],'repetition_penalty':penalty},
        'attention_backend':{'language':[backend],'vision':['Qwen2_5_VLVisionFlashAttention2']}}


def shards(tmp_path,identities):
    manifests=[]
    for i in range(4):
        path=tmp_path/f'shard-{i}.jsonl'
        path.write_text(json.dumps({'index':i,'task_name':'task','match':True,'evaluation_protocol':identities[i]['evaluation_protocol']})+'\n')
        manifest=Path(str(path)+'.manifest.json')
        write_shard_manifest(manifest,shard_jsonl=path,dataset_identity={'path':'/data/test','size':1,'mtime_ns':0},
            model_identity='same-model',generation_config={'max_new_tokens':10,'max_sequence_length':128,
                'repetition_penalty':1.05,'temperature':0.,'top_p':1.},
            execution_identity=identities[i],shard_index=i,num_shards=4,
            expected_source_indices=[i],processed_source_indices=[i],worker_exit_status=0)
        manifests.append(manifest)
    return manifests


@pytest.mark.parametrize('changed',[identity('continuous'),identity(penalty=1.),identity(backend='Qwen2_5_VLSdpaAttention')])
def test_merge_rejects_mixed_protocol_resolved_config_or_actual_backend(tmp_path,changed):
    assert 'execution_identity' in __import__('inspect').signature(write_shard_manifest).parameters
    manifests=shards(tmp_path,[identity()]*3+[changed])
    with pytest.raises(ValueError,match='execution identity'):
        merge_shards(manifests,expected_sample_count=4,output_jsonl=tmp_path/'merged.jsonl',summary_json=tmp_path/'summary.json')


def test_original_merge_counts_legacy_matches_including_empty_gold(tmp_path):
    assert 'execution_identity' in __import__('inspect').signature(write_shard_manifest).parameters
    manifests=shards(tmp_path,[identity()]*4)
    result=merge_shards(manifests,expected_sample_count=4,output_jsonl=tmp_path/'merged.jsonl',summary_json=tmp_path/'summary.json')
    assert result['accuracy']==1. and result['success']==4
    assert result['execution_identity']==identity()
    assert 'valid_gold_accuracy' not in result


def test_offline_original_scorer_has_no_torch_model_or_reference_dependency():
    code='''import builtins
real=builtins.__import__
def deny(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','transformers','Reference'}: raise AssertionError(name)
    return real(name,*args,**kwargs)
builtins.__import__=deny
from evaluation.original_scoring import score_original
assert score_original('',{})['match'] is True
'''
    completed=subprocess.run([str(Path('venv/ilvr-venv/bin/python').resolve()),'-c',code],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr


def test_independent_pristine_source_oracle_runs_its_own_model_sampler_and_eval():
    spec=importlib.util.find_spec('evaluation.original_oracle')
    assert spec is not None, 'the independent pinned-source oracle is missing'
    oracle=importlib.import_module('evaluation.original_oracle')
    api,restored,calls=tiny_original([START,5,5,5,3])
    cls=oracle.source_model_class()
    assert cls is not type(restored)
    assert cls.generate is not type(restored).generate
    source=cls(restored.config).eval()
    source.load_state_dict({k:v for k,v in restored.state_dict().items() if not k.startswith('lm_head.')},strict=False)
    from tests.evaluation.test_original_ilvr import ScriptedHead
    source.lm_head=ScriptedHead([START,5,5,5,3])
    source_processor=NativeProcessor()
    native=oracle.source_evaluation_functions()
    assert native['run_one_example'].__code__.co_filename.endswith('original_eval.py.txt')
    result=native['run_one_example'](source,source_processor,{'text_input':'Q'},dict(max_new_tokens=20,do_sample=False))
    module=importlib.import_module('eval')
    recovered=module.run_one_example(restored,NativeProcessor(),{'text_input':'Q'},dict(max_new_tokens=20,do_sample=False),evaluation_protocol='original')
    assert source_processor.decoded[0][0][2:]==recovered['generated_suffix_token_ids']==[START,PAD,PAD,END,3]
    assert result['raw_output']==recovered['raw_output']
    assert result['assistant_only']==recovered['assistant_only']
    assert result['extracted_final_answer']==recovered['extracted_final_answer']


def test_gpu_witness_cli_caps_are_checked_before_loading_any_model(tmp_path):
    spec=importlib.util.find_spec('evaluation.original_witness')
    assert spec is not None, 'the bounded source-vs-restored witness is missing'
    witness=importlib.import_module('evaluation.original_witness')
    for indices,tokens in [('0,1,2,3',512),('0,0',512),('0',513)]:
        with pytest.raises(ValueError,match='budget|distinct'):
            witness.validate_budget(indices,tokens)
    assert witness.validate_budget('0,2,5',512)==[0,2,5]


def test_complete_bounded_witness_compares_source_and_restored_actual_tiny_models(tmp_path,monkeypatch):
    from evaluation import original_witness as witness, original_oracle
    from tests.evaluation.test_original_ilvr import ScriptedHead
    def load(mode,*args):
        _,model,_=tiny_original([START,5,5,5,3])
        if mode=='source':
            source=original_oracle.source_model_class()(model.config).eval()
            source.load_state_dict(model.state_dict(),strict=False)
            source.lm_head=ScriptedHead([START,5,5,5,3])
            model=source
        return model,NativeProcessor()
    monkeypatch.setattr(witness,'load_for_mode',load)
    path=tmp_path/'test.jsonl'
    path.write_text('{"text_input":"Q"}\n')
    kwargs=dict(model_dir=str(tmp_path/'model'),base_model_dir=None,dataset_path=path,
        indices=[0],cap=6,cache_dir='data/cache',deadline=__import__('time').monotonic()+60)
    source=witness.run_mode('source',**kwargs)
    restored=witness.run_mode('restored',**kwargs)
    assert witness.compare(source,restored)==[]
    assert source['metadata']['call_owners']['generate']['module'].endswith('_ilvr_source_oracle')
    assert restored['metadata']['call_owners']['generate']['module'].endswith('_ilvr_original_utils')
    assert len(source['observations'][0]['decoder_calls'])==5


def test_actual_unmerged_dora_sparse_rows_load_on_pinned_source_and_inherit_base_config(tmp_path,monkeypatch):
    from tests.stage2.test_stage1_checkpoint_handoff import write_stage1_dora_artifact
    from tests.stage1.test_checkpointing import tiny_original_qwen
    from stage1.checkpointing import load_stage1_peft_artifact
    from peft import PeftModel
    from peft_common import resolve_qwen2_5_vl_model
    from evaluation.original_ilvr import original_model_class,config_resolution
    from evaluation.original_oracle import source_model_class
    from evaluation.original_witness import parameter_identity
    processor,_=write_stage1_dora_artifact(tmp_path)
    saved=json.loads((tmp_path/'config.json').read_text())
    saved['use_cache']=False
    (tmp_path/'config.json').write_text(json.dumps(saved))
    def base(cls):
        current=tiny_original_qwen()
        current.config.use_cache=True
        model=cls(current.config).eval()
        model.load_state_dict(current.state_dict(),strict=True)
        return model
    native_base=base(original_model_class())
    oracle_base=base(source_model_class())
    module=importlib.import_module('eval')
    monkeypatch.setattr(module,'load_original_model',lambda *args:native_base)
    monkeypatch.setattr(module.AutoProcessor,'from_pretrained',lambda *args,**kwargs:processor)
    torch.manual_seed(42)
    restored,_=module.load_evaluation_artifacts(model_dir=str(tmp_path),base_model_dir='tiny/original-qwen',cache_dir='cache',evaluation_protocol='original')
    torch.manual_seed(42)
    source=load_stage1_peft_artifact(oracle_base,processor,tmp_path,requested_base_model_reference='tiny/original-qwen',is_trainable=False)
    assert isinstance(restored,PeftModel)
    assert parameter_identity(source)==parameter_identity(restored)
    qwen=resolve_qwen2_5_vl_model(restored)
    assert qwen.generation_config.use_cache is True
    assert qwen.config.use_cache is True
    assert config_resolution(tmp_path,'tiny/original-qwen',qwen)['artifact_config']['use_cache'] is False
    assert any('trainable_tokens_delta' in name for name,_ in restored.named_parameters())
    assert any('lora_magnitude_vector' in name for name,_ in restored.named_parameters())


def test_automatic_stage1_recipe_reaches_actual_original_evaluation_coordinator():
    import os
    import shlex
    from tests.stage1.v5.test_workflow import run
    recipe=run('--dry-run')
    assert recipe.returncode==0,recipe.stderr
    evaluate=shlex.split(recipe.stdout.splitlines()[2])
    assert '--repetition-penalty' not in evaluate
    completed=subprocess.run([*evaluate,'--dry-run'],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    payload=json.loads(completed.stdout)
    assert payload['plan']['evaluation_protocol']=='original'
    stage1=payload['worker_commands_by_observed_stage']['stage1']
    assert len(stage1)==4
    for command in stage1:
        text=' '.join(command)
        assert '--evaluation-protocol original' in text
        assert '--generation-limit-policy native' in text
        assert '--repetition_penalty' not in text


def test_real_unmerged_peft_full_run_mode_captures_source_and_restored_suffix(tmp_path,monkeypatch):
    from dataclasses import replace
    from tests.stage1.test_checkpointing import (tiny_original_qwen,sample_metadata,stage1_tuning,
        TinyStage1Processor,TinyStage1Tokenizer)
    from stage1.peft import build_stage1_tuned_model
    from stage1.checkpointing import save_stage1_peft_artifact,load_stage1_peft_artifact
    from evaluation import original_witness as witness,original_oracle
    from evaluation.original_ilvr import original_model_class
    from peft import PeftModel
    base=tiny_original_qwen(pretrained_vocab_size=PAD)
    metadata=replace(sample_metadata(base),base_vocab_size=PAD,resized_vocab_size=END+1,
        latent_token_ids=(PAD,START,END))
    base_state={key:value.detach().clone() for key,value in base.state_dict().items()}
    base_config=__import__('copy').deepcopy(base.config)
    base.resize_token_embeddings(END+1,mean_resizing=False)
    base.config.latent_token_id=PAD
    base.config.latent_start_id=START
    base.config.latent_end_id=END
    tuned=build_stage1_tuned_model(base,stage1_tuning(),(PAD,START,END))
    processor=TinyStage1Processor(TinyStage1Tokenizer((PAD,START,END),size=END+1))
    artifact=tmp_path/'artifact'
    save_stage1_peft_artifact(tuned,processor,artifact,metadata)
    def load(mode,*args):
        cls=original_oracle.source_model_class() if mode=='source' else original_model_class()
        base=cls(__import__('copy').deepcopy(base_config))
        base.load_state_dict(base_state,strict=True)
        model=load_stage1_peft_artifact(base,processor,artifact,
            requested_base_model_reference='tiny/original-qwen',is_trainable=False).eval()
        assert isinstance(model,PeftModel)
        return model,NativeProcessor()
    monkeypatch.setattr(witness,'load_for_mode',load)
    dataset=tmp_path/'test.jsonl';dataset.write_text('{"text_input":"Q"}\n')
    kwargs=dict(model_dir=str(artifact),base_model_dir='tiny/original-qwen',dataset_path=dataset,
        indices=[0],cap=1,cache_dir='data/cache',deadline=__import__('time').monotonic()+60)
    source=witness.run_mode('source',**kwargs)
    restored=witness.run_mode('restored',**kwargs)
    assert witness.compare(source,restored)==[]
    assert len(restored['observations'][0]['generated_suffix_token_ids'])==1
    assert source['metadata']['parameters']==restored['metadata']['parameters']


def test_four_actual_original_worker_mains_validate_and_merge_inherited_config(tmp_path,monkeypatch):
    from src.multinode_artifacts import validate_evaluation_sidecars
    from step_preference import capacity_evidence
    module=importlib.import_module('eval')
    _,model,_=tiny_original([3])
    dataset=tmp_path/'test.jsonl';dataset.write_text('{}\n'*4)
    monkeypatch.setattr(module,'load_evaluation_artifacts',lambda **kw:(model,NativeProcessor()))
    monkeypatch.setattr(module,'publish_evaluation_runtime_identity',lambda *a,**kw:None)
    monkeypatch.setattr(capacity_evidence,'git_identity',lambda:'a'*40)
    paths=[];manifests=[]
    for shard in range(4):
        output=tmp_path/f'shard-{shard}.jsonl'
        monkeypatch.setattr(sys,'argv',['eval.py','--model_dir','model','--test_data_path',str(dataset),
            '--task_name','zebra-cot','--output_json_path',str(output),'--runtime_identity_path',str(tmp_path/f'runtime-{shard}.json'),
            '--num-shards','4','--shard-index',str(shard),'--model-identity','same-model',
            '--evaluation-protocol','original','--generation-limit-policy','native'])
        assert module.main() == 0
        paths.append(output);manifests.append(Path(str(output)+'.manifest.json'))
    validate_evaluation_sidecars(manifests,paths,expected_model_identity='same-model',expected_sample_count=4,
        expected_generation_config={'max_new_tokens':4096,'max_sequence_length':32768,
            'temperature':0.,'top_p':1.,'vision_ablation':'none','generation_limit_policy':'native'},
        expected_evaluation_protocol='original')
    result=merge_shards(manifests,expected_sample_count=4,output_jsonl=tmp_path/'merged.jsonl',summary_json=tmp_path/'summary.json')
    assert result['success']==4 and result['accuracy']==1.
    assert result['evaluation_protocol']=='original'
