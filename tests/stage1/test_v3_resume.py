import importlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.utils.data import SequentialSampler
from transformers import Trainer,TrainingArguments,TrainerCallback
from stage1.trainer import CustomTrainerStage1
from stage1.v3_config import Stage1V3Config


def api():return importlib.import_module('stage1.v3_resume')


def schedule():
    return api().build_v3_schedule_identity(retained_ids=tuple(map(str,range(6))),
        sampler_identity={'sampler':'SequentialSampler','world_size':1},microsteps_per_rank=6,accumulation_steps=4)


def test_schedule_and_contract_strict_identity():
    value=schedule()
    assert (value.updates_per_epoch,value.planned_steps,value.remainder_microsteps)==(2,2,2)
    contract=api().build_v3_checkpoint_contract(Stage1V3Config(),value,{'frozen_teacher_sha256':'a'*64})
    api().validate_v3_resume(contract,deepcopy(contract))
    for field,changed in [('schema_version',2),('planned_steps',1),('boundary','partial')]:
        bad=deepcopy(contract);bad[field]=changed
        with pytest.raises(ValueError):api().validate_v3_resume(bad,contract)
    bad=deepcopy(contract);bad['config']['text_kl_beta']=.02
    with pytest.raises(ValueError):api().validate_v3_resume(bad,contract)
    bad=deepcopy(contract);bad['schedule']['retained_ids_sha256']='b'*64
    with pytest.raises(ValueError):api().validate_v3_resume(bad,contract)


@pytest.mark.parametrize('micro,acc',[ (0,4),(6,0),(True,4),(6,1.5)])
def test_invalid_update_budget(micro,acc):
    with pytest.raises(ValueError):api().build_v3_schedule_identity(
        retained_ids=('x',),sampler_identity={},microsteps_per_rank=micro,accumulation_steps=acc)


class Scalar(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(1.))
    def forward(self,input_ids,labels=None):
        loss=(self.weight*input_ids.float()).square().mean()
        return {'loss':loss,'logits':self.weight*input_ids.float()}


class BudgetTrainer(CustomTrainerStage1):
    # Test actual parent training/save/resume with only the v3 schedule override;
    # no GPU training or simulated claim about the v3 ZeRO update is made here.
    def __init__(self,path,stop=False):
        self.v3_config=Stage1V3Config();self.stage1_variant='v3'
        self.v3_gain_controller=None;self.v3_resolved_profile={'frozen_teacher_sha256':'a'*64}
        self.v3_retained_ids=tuple(map(str,range(6)));self.v3_saved_contract=None
        self.observed=[]
        Trainer.__init__(self,model=Scalar(),args=TrainingArguments(output_dir=str(path),use_cpu=True,
            report_to=[],per_device_train_batch_size=1,gradient_accumulation_steps=4,num_train_epochs=1,
            learning_rate=.01,warmup_steps=0,save_strategy='steps',save_steps=1,save_safetensors=False,
            remove_unused_columns=False,seed=42,max_grad_norm=0),
            train_dataset=[{'input_ids':torch.tensor([i+1]),'labels':torch.tensor([0])} for i in range(6)])
        if stop:
            class Stop(TrainerCallback):
                def on_step_end(self,args,state,control,**kw):
                    if state.global_step==1:control.should_training_stop=True
                    return control
            self.add_callback(Stop())
    def train(self,*a,**kw):return Trainer.train(self,*a,**kw)
    def _save(self,*a,**kw):return Trainer._save(self,*a,**kw)
    def log(self,*a,**kw):return Trainer.log(self,*a,**kw)
    def _rotate_checkpoints(self,*a,**kw):return Trainer._rotate_checkpoints(self,*a,**kw)
    def _get_train_sampler(self):return SequentialSampler(self.train_dataset)
    def training_step(self,model,inputs,num_items_in_batch=None):
        self.observed.extend(inputs['input_ids'].flatten().tolist())
        return Trainer.training_step(self,model,inputs,num_items_in_batch)
    def compute_loss(self,*a,**kw):return Trainer.compute_loss(self,*a,**kw)
    def _save_checkpoint(self,*a,**kw):return Trainer._save_checkpoint(self,*a,**kw)
    def save_model(self,*a,**kw):return Trainer.save_model(self,*a,**kw)
    def _load_from_checkpoint(self,*a,**kw):return Trainer._load_from_checkpoint(self,*a,**kw)


def test_partial_accumulation_resume_matches_uninterrupted(tmp_path):
    baseline=BudgetTrainer(tmp_path/'baseline');baseline.train()
    interrupted=BudgetTrainer(tmp_path/'resumed',stop=True);interrupted.train()
    checkpoint=tmp_path/'resumed'/'checkpoint-1'
    resumed=BudgetTrainer(tmp_path/'resumed');resumed.train(resume_from_checkpoint=str(checkpoint))
    assert baseline.observed==[1,2,3,4,5,6]
    assert interrupted.observed==[1,2,3,4] and resumed.observed==[5,6]
    assert baseline.state.global_step==resumed.state.global_step==2
    assert baseline.v3_planned_steps==resumed.v3_planned_steps==2
    assert torch.equal(baseline.model.weight,resumed.model.weight)
    a=baseline.optimizer.state_dict();b=resumed.optimizer.state_dict()
    assert a['param_groups']==b['param_groups']
    for index,values in a['state'].items():
        for key,value in values.items():assert torch.equal(value,b['state'][index][key])
    assert baseline.lr_scheduler.state_dict()==resumed.lr_scheduler.state_dict()
    rng_a=torch.load(tmp_path/'baseline/checkpoint-2/rng_state.pth',weights_only=False)
    rng_b=torch.load(tmp_path/'resumed/checkpoint-2/rng_state.pth',weights_only=False)
    assert torch.equal(rng_a['cpu'],rng_b['cpu'])


def test_artifact_classification_rejects_bad_v3_identity_before_model_use(tmp_path):
    from stage1.checkpointing import classify_stage1_artifact
    (tmp_path/'stage1_v3.json').write_text('{"schema_version":99}')
    with pytest.raises(ValueError,match='v3 resume'):
        classify_stage1_artifact(tmp_path)


def test_v3_contract_is_resumed_as_common_metadata_from_both_node_roots():
    from dataclasses import replace
    from src.multinode_artifacts import ArtifactEntry,select_common_metadata_entries,select_resume_entries
    from tests.multinode.test_artifacts import strict_manifest
    manifest=strict_manifest(0)
    entry=ArtifactEntry(Path('stage1_v3.json'),123,0)
    manifest=replace(manifest,entries=manifest.entries+(entry,),process_ranks=(0,1,2,3))
    assert entry in select_common_metadata_entries(manifest)
    assert entry in select_resume_entries(manifest)


def test_full_save_publishes_contract_and_rejects_partial_boundary(tmp_path,monkeypatch):
    # Observe the product save hook without serializing a heavyweight model.
    from stage1.trainer import CustomTrainerStage1
    from transformers import Trainer
    from types import SimpleNamespace
    from stage1.v3_update import V3GainController
    from stage1.v3_gradients import V3Gradients
    trainer=CustomTrainerStage1.__new__(CustomTrainerStage1)
    param=nn.Parameter(torch.ones(1))
    trainer.v3_config=Stage1V3Config();trainer.v3_gain_controller=V3GainController((param,),trainer.v3_config)
    trainer.v3_contract=api().build_v3_checkpoint_contract(trainer.v3_config,schedule(),{'frozen_teacher_sha256':'a'*64})
    trainer.stage1_peft_metadata=None;trainer.args=SimpleNamespace(output_dir=str(tmp_path),should_save=True)
    monkeypatch.setattr(Trainer,'save_model',lambda *a,**kw:None)
    trainer.save_model(tmp_path)
    assert api().read_v3_contract(tmp_path)==trainer.v3_contract
    trainer.v3_gain_controller.accumulate(V3Gradients((torch.zeros(1),),(torch.zeros(1),),0,torch.tensor(0.),torch.tensor(0.)))
    with pytest.raises(RuntimeError,match='checkpoint'):trainer.save_model(tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('field,value', [('retained_ids_sha256','invalid'),('remainder_microsteps',False),('sampler_identity',[])])
def test_contract_rejects_malformed_schedule(field,value):
    contract=api().build_v3_checkpoint_contract(Stage1V3Config(),schedule(),{'frozen_teacher_sha256':'a'*64})
    contract['schedule'][field]=value
    with pytest.raises(ValueError,match='v3 resume'):
        api().validate_v3_resume(contract,contract)


def test_contract_rejects_unknown_nested_fields_as_validation_error():
    contract=api().build_v3_checkpoint_contract(Stage1V3Config(),schedule(),{'frozen_teacher_sha256':'a'*64})
    contract['config']['unknown']=True
    with pytest.raises(ValueError,match='v3 resume'):
        api().validate_v3_resume(contract,contract)


@pytest.mark.parametrize('selector,formula', [
    ('kl','T2_KL_student_reference_sample_centered_full_causal_v1'),
    ('entropy','T2_entropy_student_sample_centered_full_causal_v1'),
])
def test_contract_records_selector_and_checks_its_formula(tmp_path,selector,formula):
    contract=api().build_v3_checkpoint_contract(Stage1V3Config(text_selector=selector),schedule(),
        {'frozen_teacher_sha256':'a'*64})
    assert contract['config']['text_selector']==selector
    assert contract['formula_ids']['selector']==formula
    api().write_v3_contract(tmp_path,contract)
    assert api().read_v3_contract(tmp_path)==contract
    bad=deepcopy(contract)
    bad['formula_ids']['selector']=('T2_entropy_student_sample_centered_full_causal_v1' if selector=='kl'
                                  else 'T2_KL_student_reference_sample_centered_full_causal_v1')
    with pytest.raises(ValueError,match='resume|selector'):
        api().validate_v3_resume(bad,bad)


def test_legacy_missing_selector_remains_kl_without_rewriting_contract(tmp_path):
    current=api().build_v3_checkpoint_contract(Stage1V3Config(),schedule(),{'frozen_teacher_sha256':'a'*64})
    legacy=deepcopy(current)
    legacy['config'].pop('text_selector')
    path=tmp_path/'stage1_v3.json'
    original=json.dumps(legacy,indent=2).encode()
    path.write_bytes(original)
    read=api().read_v3_contract(tmp_path)
    assert read==legacy and path.read_bytes()==original
    api().validate_v3_resume(read,current)
    api().validate_v3_resume(current,read)
    assert read==legacy and path.read_bytes()==original
    entropy=api().build_v3_checkpoint_contract(Stage1V3Config(text_selector='entropy'),schedule(),
        {'frozen_teacher_sha256':'a'*64})
    with pytest.raises(ValueError,match='mismatch'):
        api().validate_v3_resume(read,entropy)
    malformed=deepcopy(legacy)
    malformed['config'].pop('text_cap')
    with pytest.raises(ValueError,match='resume'):
        api().validate_v3_resume(malformed,malformed)
