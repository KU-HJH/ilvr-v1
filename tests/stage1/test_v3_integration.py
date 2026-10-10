"""CPU product entry witness, not a distributed training run."""
import importlib
import pytest
from tests.stage1.v3_helpers import product_v3_trainer
from stage1.v3_config import Stage1V3Config


@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('selector',['kl','entropy'])
def test_product_integration_call_budget(tmp_path,empty,selector):
    api=importlib.import_module('tests.stage1.v3_runtime_acceptance')
    trainer,batch=product_v3_trainer(tmp_path,checkpointed=True,image=True,empty=empty,
        config=Stage1V3Config(text_selector=selector))
    with api.observe_microstep(trainer) as event:
        trainer.training_step(trainer.model,batch)
    expected=['F'] if empty else (['F','gain_clean','reference','packed_replay']
        if selector=='kl' else ['F','gain_clean','packed_replay'])
    assert event['core_forwards']==expected
    assert event['aux_vjp_arities']==([] if empty else [2,1])
    assert event['ordinary_backwards']==1 and event['helper_builds']==1
    assert {e['backward'] for e in event['sites'] if e['phase']=='recompute'}==({'ordinary'} if empty else {'joint_r_kl','rv','ordinary'})
    assert event['lifecycle']=='CLOSED'
