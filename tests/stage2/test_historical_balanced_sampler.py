"""Historical membership replay only; CPU evidence does not prove GPU safety."""
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as N

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler, DataLoader
from transformers import Trainer, TrainingArguments

from deepspeed_runtime import DeepSpeedRuntimeProfile
from stage2 import sampler as sampling
from stage2.trainer import Stage2Trainer
from tests.stage2.test_reference_random_sampler import ScalarModel, cpu_trainer
from tests.stage2.test_reference_sampler_resume_runtime import run_resume
from tests.step_preference.test_dil_trainer import dil_config

POLICY = 'historical_balanced_2rank'
REPO = Path(__file__).resolve().parents[2]
PROBE_ORDER = [15,14,12,9,0,7,2,1,11,10,19,18,16,17,4,5,8,13,6,3]


def examples(counts):
    return [N(sample_id=f's{i}', latent_block_count=c) for i,c in enumerate(counts)]


def sampler(data, **changes):
    assert hasattr(sampling, 'HistoricalBalanced2RankSampler'), 'historical policy is missing'
    return sampling.HistoricalBalanced2RankSampler(data, **{'num_replicas':4, 'seed':42, 'gradient_accumulation_steps':4, **changes})


@pytest.fixture
def storage():
    # Any model-bearing fixtures remain under the existing checkpoints tree.
    with tempfile.TemporaryDirectory(prefix='historical-sampler-test-', dir=REPO/'checkpoints') as path:
        yield Path(path)


def test_full_order_matches_independently_saved_historical_checksum():
    counts = json.loads((Path(__file__).parent/'fixtures/historical-retained-counts.json').read_text())
    data = examples(counts)
    rng = torch.get_rng_state().clone()
    historical = sampler(data)
    shards = [list(BatchSamplerShard(BatchSampler(historical,1,False), num_processes=4,
        process_index=r, even_batches=False)) for r in range(4)]
    order = [b[0] for row in zip(*shards) for b in row]
    assert torch.equal(rng, torch.get_rng_state())
    assert [len(s) for s in shards] == [845]*4
    # Literal checksum predates this implementation; covers all 212 ordered groups.
    assert hashlib.sha256(json.dumps(order).encode()).hexdigest() == '647d775a13d229386fe4e499c6b1ebdcd1db95c183c65ae6a0bc2db20ff62540'
    assert Counter(order) == Counter(range(3379)) + Counter([2544])
    assert order[-4:] == [1523,1968,232,132]
    assert [counts[i] for i in order[-4:]] == [3,3,1,1]
    assert historical.identity.to_dict()['padding_weight'] == 1
    assert historical.num_replicas == 4
    assert not isinstance(data[2544], sampling.ReferenceSample)
    for old in (sampling.LatentBlockBalancedSampler(data,num_replicas=4,seed=42),
                sampling.ReferenceRandomSampler(data,num_replicas=4,seed=42,gradient_accumulation_steps=4)):
        assert list(old) != order


def test_probe_order_epoch_and_rng_are_preserved():
    s = sampler(examples([1]*8+[3]*8+[6]*4))
    assert list(s) == PROBE_ORDER
    s.set_epoch(1)
    assert list(s) != PROBE_ORDER
    s.set_epoch(0)
    assert list(s) == PROBE_ORDER


@pytest.mark.parametrize('changes', [{'num_replicas':2},{'num_replicas':True},
    {'gradient_accumulation_steps':2},{'gradient_accumulation_steps':4.0},{'seed':True}])
def test_invalid_geometry_is_rejected(changes):
    with pytest.raises(ValueError):
        sampler(examples([1]*4), **changes)


def test_nondivisible_cohort_is_rejected_without_extra_padding():
    with pytest.raises(ValueError, match='divisible'):
        sampler(examples([1,1]))


def test_strict_identity_roundtrip_and_reordered_equal_count_cohort(storage):
    data = examples([1]*8+[3]*8+[6]*4)
    s = sampler(data)
    sampling.write_stage2_sampler_sidecar(storage,s.identity,global_step=1)
    assert sampling.read_stage2_sampler_sidecar(storage) == s.identity
    sampling.validate_stage2_sampler_resume_identity(storage,s.identity)
    reordered = list(data)
    reordered[0],reordered[1] = reordered[1],reordered[0]
    with pytest.raises(ValueError,match='ordered_dataset_sha256'):
        sampling.validate_stage2_sampler_resume_identity(storage,sampler(reordered).identity)
    with pytest.raises(ValueError,match='seed'):
        sampling.validate_stage2_sampler_resume_identity(storage,sampler(data,seed=43).identity)
    profile = DeepSpeedRuntimeProfile(True,2,None,None,4)
    assert sampling.select_stage2_sampler_policy('auto',profile=profile,config=dil_config(),resume=storage) == POLICY
    with pytest.raises(ValueError,match='policy'):
        sampling.select_stage2_sampler_policy('reference_random',profile=profile,config=dil_config(),resume=storage)


@pytest.mark.parametrize('field,value', [('world_size',2),('world_size',4.0),
    ('logical_group_size',4),('logical_group_size',2.0),('gradient_accumulation_steps',2),
    ('gradient_accumulation_steps',4.0),('policy','latent_balanced'),('padding_weight',0),
    ('padding_weight',True),('update_accounting','floor'),('ordering_version','other'),
    ('seed',True),('ordered_dataset_sha256','bad'),('bucket_sizes',{'1':8.,'3':8,'6':4})])
def test_strict_identity_rejects_corrupt_or_coercive_values(field,value):
    payload = sampler(examples([1]*8+[3]*8+[6]*4)).identity.to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        sampling.Stage2SamplerIdentity.from_dict(payload)


def test_strict_identity_rejects_missing_extra_fields_and_legacy_import(storage):
    data = examples([1]*4)
    identity = sampler(data).identity
    for field in identity.to_dict():
        payload = identity.to_dict();payload.pop(field)
        with pytest.raises(ValueError):sampling.Stage2SamplerIdentity.from_dict(payload)
    payload = identity.to_dict();payload['extra'] = 0
    with pytest.raises(ValueError):sampling.Stage2SamplerIdentity.from_dict(payload)
    legacy = sampling.LatentBlockBalancedSampler(data,num_replicas=2,seed=42).identity
    sampling.write_stage2_sampler_sidecar(storage,legacy,global_step=1)
    with pytest.raises(ValueError,match='policy'):
        sampling.validate_stage2_sampler_resume_identity(storage,identity)


def actual_trainer(storage,monkeypatch,**changes):
    monkeypatch.setattr(TrainingArguments,'world_size',property(lambda self:4))
    gas = changes.pop('gas',4)
    args = TrainingArguments(output_dir=str(storage/'output'),use_cpu=True,report_to=[],
        per_device_train_batch_size=1,gradient_accumulation_steps=gas,disable_tqdm=True)
    values = dict(model=ScalarModel(),args=args,train_dataset=examples([1]*8+[3]*8+[6]*4),
        step_preference_config=dil_config(),stage2_sampler_policy=POLICY,
        deepspeed_runtime_profile=DeepSpeedRuntimeProfile(True,2,None,None,4))
    values.update(changes)
    return Stage2Trainer(**values)


@pytest.mark.parametrize('kind', ['dil_bce_rf_topk', 'dil_bce_rf_topk_reverse'])
def test_real_trainer_constructs_unwrapped_sampler_with_fixed_denominator(storage,monkeypatch,kind):
    t = actual_trainer(storage,monkeypatch,step_preference_config=dil_config(kind))
    assert t.model_accepts_loss_kwargs is False and t.accelerator.even_batches is False
    assert t.train_dataset[0].sample_id == 's0'
    assert list(t._get_train_sampler()) == PROBE_ORDER
    # data_seed must not change historical seed+epoch ordering.
    t.args.data_seed = 917
    assert list(t._get_train_sampler()) == PROBE_ORDER
    t.compute_loss = lambda model,inputs,**kw:model(**inputs).loss
    t.training_step(t.model,{'input_ids':torch.tensor([[1]])})
    assert t.model.weight.grad.item() == pytest.approx(.5)
    assert t.model.weight.grad.item()/4 == pytest.approx(2./16)


@pytest.mark.parametrize('changes', [{'gas':2},{'compute_loss_func':lambda *a:0.},
    {'step_preference_config':None},{'step_preference_config':dil_config('dpo')},
    {'deepspeed_runtime_profile':DeepSpeedRuntimeProfile(True,2,None,None,2)}])
def test_real_trainer_rejects_incompatible_contract_before_parent(storage,monkeypatch,changes):
    monkeypatch.setattr(Trainer,'__init__',lambda *a,**kw:pytest.fail('parent constructed invalid policy'))
    with pytest.raises(ValueError):actual_trainer(storage,monkeypatch,**changes)


def test_actual_845_microstep_accounting_and_pre_tail_resume(storage):
    t = cpu_trainer(storage/'accounting',4,policy=POLICY)
    values = t.set_initial_training_values(t.args,DataLoader(t.train_dataset,batch_size=1),16)
    assert (values[0],values[1],values[6]) == (1,212,212)
    (storage/'tail').mkdir()
    t,backwards = run_resume(storage/'tail',4,211,212,policy=POLICY)
    assert backwards == [{'index':844,'sync':True,'step':211}]
    assert t.state.global_step == 212 and t.state.epoch == 1.
    assert t.model.weight.item() == pytest.approx(.95)


def test_completed_212_resume_is_epoch_boundary(storage):
    t,backwards = run_resume(storage,4,212,213,policy=POLICY)
    assert [b['index'] for b in backwards] == [0,1,2,3]
    assert [b['sync'] for b in backwards] == [False,False,False,True]
    assert t.state.global_step == 213 and t.model.weight.item() == pytest.approx(.8)


def test_pre_tail_resume_synchronizes_the_only_remaining_backward(storage):
    t,backwards = run_resume(storage,4,211,212,policy=POLICY)
    assert backwards == [{'index':844,'sync':True,'step':211}]
    assert t.state.global_step == 212 and t.model.weight.item() == pytest.approx(.95)


def test_reordered_cached_cohort_fails_before_parent_checkpoint_load(storage,monkeypatch):
    from deepspeed_runtime import write_deepspeed_runtime_sidecar, load_deepspeed_runtime_profile
    from step_preference.dil_identity import build_dil_identity
    t = actual_trainer(storage,monkeypatch,deepspeed_runtime_profile=
        load_deepspeed_runtime_profile(REPO/'configs/deepspeed_zero2.json',world_size=4))
    identity = t._get_train_sampler().identity
    checkpoint = storage/'checkpoint-1';checkpoint.mkdir()
    (checkpoint/'config.json').write_text(json.dumps({'ilvr_context_preference':build_dil_identity(t.step_preference_config)}))
    write_deepspeed_runtime_sidecar(checkpoint,t.deepspeed_runtime_profile)
    sampling.write_stage2_sampler_sidecar(checkpoint,identity,global_step=1)
    t.train_dataset[0],t.train_dataset[1] = t.train_dataset[1],t.train_dataset[0]
    monkeypatch.setattr(Trainer,'train',lambda *a,**kw:pytest.fail('checkpoint state loaded before identity rejection'))
    with pytest.raises(ValueError,match='ordered_dataset_sha256'):
        t.train(resume_from_checkpoint=str(checkpoint))


def test_existing_worker_cli_main_to_real_trainer_forwarding(storage,monkeypatch):
    import main
    from utils_deepseed import parse_args
    from tests.multinode.test_dil_forwarding import worker_commands
    for rank,env,result in worker_commands(dict(STAGE2_SAMPLER_POLICY=POLICY,
            STEP_PREFERENCE_LOSS_TYPE='dil_bce_rf_topk',STEP_PREFERENCE_TOPK='1516',
            STEP_PREFERENCE_LAMBDA_TOPK_MASS='1',STEP_PREFERENCE_ENTROPY_TOP_PERCENT='20',
            STEP_PREFERENCE_ENTROPY_SOURCE='voff',STEP_PREFERENCE_TARGET_SCOPE='until_next_latent',
            STAGE2_ASSISTANT_PLAN_RENDERING='concatenated-v1',LEARNING_RATE='5e-5',SEED='42',
            LATENT_SIZE='8',LORA_R='32',LORA_ALPHA='64',LORA_DROPOUT='0.05',
            ATTENTION_IMPLEMENTATION='sdpa',VISION_ATTENTION_IMPLEMENTATION='flash_attention_2',
            STEP_SIDECAR_ATTENTION_BACKEND='flex',STEP_SIDECAR_QUERY_BUDGET='1024',
            MAX_SEQUENCE_LENGTH='8192'), ('--max_steps','212','--warm_up_steps','10')):
        assert result.returncode == 0,result.stderr
        argv = json.loads(result.stdout)
        for key,value in {'--stage2_sampler_policy':POLICY,'--num_machines':'2','--num_processes':'4',
            '--machine_rank':str(rank),'--per_device_train_batch_size':'1','--gradient_accumulation_steps':'4',
            '--learning_rate':'5e-5','--seed':'42','--warm_up_steps':'10','--max_steps':'212',
            '--latent_size':'8','--lora_r':'32','--lora_alpha':'64','--lora_dropout':'0.05',
            '--attention_implementation':'sdpa','--vision_attention_implementation':'flash_attention_2',
            '--step_sidecar_attention_backend':'flex','--step_sidecar_query_budget':'1024',
            '--step_preference_beta':'0.5','--step_preference_weight':'1.0','--step_entropy_weight':'0',
            '--step_preference_loss_type':'dil_bce_rf_topk','--step_preference_topk':'1516',
            '--step_preference_lambda_topk_mass':'1','--step_preference_entropy_top_percent':'20',
            '--step_preference_entropy_source':'voff','--step_preference_target_scope':'until_next_latent',
            '--stage2_assistant_plan_rendering':'concatenated-v1','--max_seq_length_train':'8192',
            '--dataloader_num_workers':'0','--tuning_method':'dora'}.items():
            assert argv[argv.index(key)+1] == value
        assert '--no-gradient_checkpointing' in argv
        assert '--stage2_latent_activation_checkpointing' in argv
        assert '--no-step_visual_off_suppress_latent_boundaries' in argv
    args = parse_args(['--stage2_sampler_policy',POLICY])
    args.save_final_model = False
    profile = DeepSpeedRuntimeProfile(True,2,None,None,4)
    monkeypatch.setattr(main,'bind_deepspeed_runtime_profile',lambda *a,**kw:profile)
    monkeypatch.setattr(TrainingArguments,'world_size',property(lambda self:4))
    monkeypatch.setattr(main,'build_stage2_training_arguments',lambda args:TrainingArguments(
        output_dir=str(storage),use_cpu=True,report_to=[],gradient_accumulation_steps=4,
        per_device_train_batch_size=1))
    monkeypatch.setattr(main,'build_stage2_training_components',lambda args:N(
        model=ScalarModel(),train_dataset=examples([1]*8+[3]*8+[6]*4),
        data_collator=lambda x:x,processor=None,peft_metadata=None,
        step_preference_config=dil_config(),trainer_resume_path=None,trainer_class=Stage2Trainer))
    monkeypatch.setattr(main,'publish_stage2_training_evidence',lambda *a,**kw:None)
    monkeypatch.setattr(main,'publish_step_preference_phase_a_summary',lambda *a,**kw:None)
    seen = []
    def parent_train(self,*a,**kw):
        seen.append((self.stage2_sampler_policy,list(self._get_train_sampler())))
        return N(training_loss=0.)
    monkeypatch.setattr(Trainer,'train',parent_train)
    main.run_stage2(args)
    assert seen == [(POLICY,PROBE_ORDER)]
