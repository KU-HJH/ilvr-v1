"""CPU product entry witness, not a distributed training run."""
import importlib
import pytest
from tests.stage1.v3_helpers import product_v3_trainer


@pytest.mark.parametrize('empty',[False,True])
def test_product_integration_call_budget(tmp_path,empty):
    api=importlib.import_module('tests.stage1.v3_runtime_acceptance')
    trainer,batch=product_v3_trainer(tmp_path,checkpointed=True,image=True,empty=empty)
    with api.observe_microstep(trainer) as event:
        trainer.training_step(trainer.model,batch)
    assert event['core_forwards']==(['F'] if empty else ['F','gain_clean','reference','packed_replay'])
    assert event['aux_vjp_arities']==([] if empty else [2,1])
    assert event['ordinary_backwards']==1 and event['helper_builds']==1
    assert {e['backward'] for e in event['sites'] if e['phase']=='recompute'}==({'ordinary'} if empty else {'joint_r_kl','rv','ordinary'})
    assert event['lifecycle']=='CLOSED'
