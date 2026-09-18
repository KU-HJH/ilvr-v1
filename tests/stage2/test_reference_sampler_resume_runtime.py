"""Actual inherited loop CPU optimizer witnesses; no distributed-training claim."""
import pytest
import torch
from torch.utils.data import DataLoader
from transformers.trainer_callback import TrainerState

from tests.stage2.test_reference_random_sampler import cpu_trainer


def run_resume(tmp_path, gas, saved_step, max_steps, policy='reference_random'):
    trainer=cpu_trainer(tmp_path/'output',gas,policy=policy,max_steps=max_steps)
    trainer.args.save_strategy='no'
    trainer.args.logging_strategy='no'
    trainer.args.max_grad_norm=0.
    trainer.train_dataset=[{'input_ids':torch.tensor([i])} for i in range(845)]
    trainer.get_train_dataloader=lambda:DataLoader(trainer.train_dataset,batch_size=1)
    trainer.compute_loss=lambda model,inputs,**kw:model(**inputs).loss
    trainer.model_accepts_loss_kwargs=False
    trainer._record_representative_gradient_norm=lambda model:None
    trainer.optimizer=torch.optim.SGD(trainer.model.parameters(),lr=.1)
    trainer.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(trainer.optimizer,lambda step:1.)
    checkpoint=tmp_path/'checkpoint'
    checkpoint.mkdir()
    TrainerState(global_step=saved_step,train_batch_size=1).save_to_json(str(checkpoint/'trainer_state.json'))
    backwards=[]
    training_step=trainer.training_step
    def observe(model,inputs,*args,**kwargs):
        backwards.append({'index':int(inputs['input_ids'].item()),
                          'sync':trainer.accelerator.sync_gradients,'step':trainer.state.global_step})
        return training_step(model,inputs,*args,**kwargs)
    trainer.training_step=observe
    trainer._inner_training_loop(batch_size=1,args=trainer.args,resume_from_checkpoint=str(checkpoint))
    return trainer,backwards


@pytest.mark.parametrize('gas,saved,final', [(2,422,423),(4,211,212)])
def test_pre_tail_resume_performs_the_remaining_actual_optimizer_step(tmp_path,gas,saved,final):
    # Ceiling budget alone passes resume arithmetic but leaves the lone tail unsynchronized.
    trainer,backwards=run_resume(tmp_path,gas,saved,final)
    assert backwards==[{'index':844,'sync':True,'step':saved}]
    assert trainer.state.global_step==final
    assert trainer.state.epoch==1.
    # d(w^2)/dw=2, one tail microstep retains the configured GAS divisor.
    assert trainer.model.weight.item()==pytest.approx(1.-.2/gas)


@pytest.mark.parametrize('gas,saved', [(2,423),(4,212)])
def test_completed_epoch_resume_starts_next_epoch_and_updates_normally(tmp_path,gas,saved):
    trainer,backwards=run_resume(tmp_path,gas,saved,saved+1)
    assert [b['index'] for b in backwards]==list(range(gas))
    assert [b['sync'] for b in backwards]==[False]*(gas-1)+[True]
    assert trainer.state.global_step==saved+1
    assert trainer.model.weight.item()==pytest.approx(.8)


def test_legacy_policy_retains_its_existing_floor_epoch_resume(tmp_path):
    trainer,backwards=run_resume(tmp_path,2,422,423,policy='latent_balanced')
    assert backwards==[{'index':0,'sync':False,'step':422},{'index':1,'sync':True,'step':422}]
    assert trainer.state.global_step==423
    assert trainer.model.weight.item()==pytest.approx(.8)
