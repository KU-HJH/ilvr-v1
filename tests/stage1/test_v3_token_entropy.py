"""Independent entropy algebra and actual entrypoint contracts for Gain weighting."""
from copy import deepcopy
import json
import math
import os
import shlex
import subprocess
import pytest
import torch
from stage1.v3_config import Stage1V3Config
from stage1.v3_gradients import V3Gradients
from stage1.v3_metadata import V3Block
from stage1.v3_update import V3GainController
from stage1 import v3_objective, v3_resume
from tests.stage1.test_v3_resume import schedule
from tests.stage1.test_v3_profile import ROOT, saved
from utils_deepseed import parse_args


def test_token_entropy_weights_gaps_before_original_block_means_and_detaches_f():
    # Independent binary entropies: H(.5)=1, H(.75)=.811278..., H(.9)=.468996...
    clean=torch.nn.Parameter(torch.tensor([[[0.,0.],[math.log(3),0.],[math.log(9),0.]]]))
    keys=((0,0),(0,1),(0,2))
    weights=v3_objective.normalized_token_entropy(clean,keys)
    torch.testing.assert_close(weights,torch.tensor([1.,.8112781244591328,.4689955935892812]))
    assert not weights.requires_grad and weights.dtype==torch.float32
    theta=torch.tensor([math.log(3),math.log(2),0.],requires_grad=True)
    r=torch.stack((theta,torch.zeros_like(theta)),dim=-1);rv=r.flip(-1)
    blocks=(V3Block(0,1,2,3,(4,5),(0,0),()),V3Block(0,2,6,7,(8,),(0,),()))
    gain,rv_score=v3_objective.entropy_weighted_gain_and_rv_sums(r,rv,blocks,weights)
    g,f_grad=torch.autograd.grad(gain,(theta,clean),allow_unused=True,retain_graph=True)
    h,=torch.autograd.grad(rv_score,theta)
    torch.testing.assert_close(g,torch.tensor([.5,.4056390622295664,.4689955935892812]))
    torch.testing.assert_close(h,torch.tensor([-.375,-1/3,-.5]))
    assert f_grad is None
    torch.testing.assert_close(gain,torch.tensor((math.log(3)+.8112781244591328*math.log(2))/2))
    # Reuse real accumulation/projection; simulate only global sum transport.
    for rank in range(2):
        p=torch.nn.Parameter(torch.ones(3));zero=torch.tensor(0.)
        controller=V3GainController((p,),Stage1V3Config(gain_weighting='token_entropy'))
        local_g=g.clone();local_h=h.clone()
        local_g[2 if rank==0 else slice(0,2)]=0
        local_h[2 if rank==0 else slice(0,2)]=0
        controller.accumulate(V3Gradients(None,(local_h,),1,zero,zero,gain_sum_grads=(local_g,)))
        totals=[torch.tensor(0),torch.tensor(2),g,h,torch.tensor(0)]
        calls=[]
        def reduce(tensor,**kwargs):
            calls.append(tensor.detach().clone());tensor.copy_(totals[len(calls)-1])
        from unittest.mock import patch
        with patch('stage1.v3_update.dist.is_initialized',return_value=True),patch('stage1.v3_update.dist.all_reduce',reduce):
            controller.prepare()
        want_g=torch.tensor([.25,.2028195311147832,.2344977967946406],dtype=torch.float64)
        want_h=torch.tensor([-.1875,-1/6,-.25],dtype=torch.float64)
        expected=want_g-(want_g@want_h)/(want_h@want_h)*want_h
        torch.testing.assert_close(controller.direction[0],expected.float())
        assert controller.global_block_count==2 and len(calls)==5


def test_entropy_query_order_chunking_uniform_and_confident_predictions():
    logits=torch.tensor([[[1000.,-1000.],[0.,0.],[float('nan'),0.]]],requires_grad=True)
    keys=((0,1),(0,0))
    for size in (1,128):
        result=v3_objective.normalized_token_entropy(logits,keys,query_chunk_size=size)
        torch.testing.assert_close(result,torch.tensor([1.,0.]))
        assert not result.requires_grad
    # Invalid unselected rows were ignored, but selected ones must fail.
    with pytest.raises(ValueError,match='nonfinite'):
        v3_objective.normalized_token_entropy(logits,((0,2),))
    with pytest.raises(ValueError):
        v3_objective.normalized_token_entropy(torch.zeros(1,1,1),((0,0),))


def test_token_entropy_controller_does_not_subtract_rv_from_nonconflicting_gain():
    p=torch.nn.Parameter(torch.ones(2));zero=torch.tensor(0.)
    controller=V3GainController((p,),Stage1V3Config(gain_weighting='token_entropy'))
    controller.accumulate(V3Gradients(None,(torch.tensor([2.,0.]),),2,zero,zero,
        gain_sum_grads=(torch.tensor([4.,2.]),)))
    controller.prepare()
    torch.testing.assert_close(controller.direction[0],torch.tensor([2.,1.]))


@pytest.mark.parametrize('selector',['kl','entropy'])
def test_token_entropy_checkpoint_mode_and_legacy_bytes_are_authoritative(tmp_path,selector):
    profile={'frozen_teacher_sha256':'a'*64}
    plain=v3_resume.build_v3_checkpoint_contract(Stage1V3Config(text_selector=selector),schedule(),profile)
    legacy=deepcopy(plain);legacy['config'].pop('gain_weighting')
    original=json.dumps(legacy,indent=2).encode();path=tmp_path/v3_resume.FILENAME;path.write_bytes(original)
    v3_resume.validate_v3_resume(legacy,plain)
    v3_resume.write_v3_contract(tmp_path,plain)
    assert path.read_bytes()==original
    weighted=v3_resume.build_v3_checkpoint_contract(Stage1V3Config(text_selector=selector,gain_weighting='token_entropy'),schedule(),profile)
    assert weighted['formula_ids']['gain']!=plain['formula_ids']['gain']
    with pytest.raises(ValueError,match='mismatch'):
        v3_resume.validate_v3_resume(legacy,weighted)
    wrong=deepcopy(weighted);wrong['formula_ids']['gain']=plain['formula_ids']['gain']
    with pytest.raises(ValueError,match='formula'):
        v3_resume.validate_v3_resume(wrong,wrong)


@pytest.mark.parametrize('selector',['kl','entropy'])
@pytest.mark.parametrize('resumed',[False,True])
def test_token_entropy_cli_coordinator_and_worker_mode(tmp_path,selector,resumed):
    from src.multinode_runtime import validate_stage1_arguments
    default=parse_args(['--stage1_variant','v3'])
    assert default.stage1_v3_gain_weighting=='token_entropy'
    checkpoint,contract=saved(tmp_path,selector)
    # Real pre-feature checkpoint must inherit none.
    contract['config'].pop('gain_weighting')
    (checkpoint/v3_resume.FILENAME).write_text(json.dumps(contract))
    assert parse_args(['--resume_from_checkpoint_path',str(checkpoint)]).stage1_v3_gain_weighting=='none'
    with pytest.raises(ValueError,match='weighting|mismatch'):
        parse_args(['--resume_from_checkpoint_path',str(checkpoint),'--stage1_v3_gain_weighting','token_entropy'])
    contract=v3_resume.build_v3_checkpoint_contract(Stage1V3Config(text_selector=selector,gain_weighting='token_entropy'),
        v3_resume.V3ScheduleIdentity(**contract['schedule']),contract['profile'])
    (checkpoint/v3_resume.FILENAME).write_text(json.dumps(contract))
    args=parse_args(['--resume_from_checkpoint_path',str(checkpoint)])
    assert args.stage1_v3_gain_weighting=='token_entropy'
    with pytest.raises(ValueError,match='weighting|mismatch'):
        parse_args(['--resume_from_checkpoint_path',str(checkpoint),'--stage1_v3_gain_weighting','none'])
    flags=validate_stage1_arguments(('--stage1-variant=v3','--stage1-v3-gain-weighting=token_entropy'))
    resolved=v3_resume.resolve_coordinator_arguments(flags,tmp_path/'out')
    assert resolved[resolved.index('--stage1-v3-gain-weighting')+1]=='token_entropy'
    resolved=v3_resume.resolve_coordinator_arguments(('--resume',str(checkpoint)),tmp_path/'out')
    assert resolved[resolved.index('--stage1-v3-gain-weighting')+1]=='token_entropy'
    env={**os.environ,'STAGE1_VARIANT':'','ILVR_NUM_MACHINES':'2','ILVR_NUM_PROCESSES':'4','ILVR_MACHINE_RANK':'1',
        'ILVR_MAIN_PROCESS_IP':'10.1.102.41','ILVR_MAIN_PROCESS_PORT':'29579','ILVR_SOCKET_IFNAME':'bond1.102'}
    extra=['--resume',str(checkpoint)] if resumed else ['--stage1-variant','v3','--stage1-v3-selector',selector,
        '--stage1-v3-gain-weighting','token_entropy']
    result=subprocess.run([str(ROOT/'run_stage1_training.sh'),'--dry-run','--method','dora',
        '--model','models/Qwen2.5-VL-3B-Instruct','--data-path',str(ROOT/'data/TRAIN.jsonl'),
        '--image-root',str(ROOT/'data/images'),'--output-dir',str(ROOT/'checkpoints/v3-token-entropy-dry'),*extra],
        env=env,cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    argv=shlex.split(result.stdout)
    assert argv[argv.index('--stage1_v3_gain_weighting')+1]=='token_entropy'
    assert argv[argv.index('--stage1_v3_selector')+1]==selector


@pytest.mark.parametrize('extra',[[],['--stage1_variant','standard'],['--stage','stage2']])
def test_token_entropy_cli_rejects_non_v3_usage(extra):
    with pytest.raises(ValueError,match='weighting|Stage1-v3'):
        parse_args([*extra,'--stage1_v3_gain_weighting','token_entropy'])


def test_token_entropy_worker_and_coordinator_reject_option_outside_v3(tmp_path):
    with pytest.raises(ValueError,match='Stage1-v3'):
        v3_resume.resolve_coordinator_arguments(('--stage1-v3-gain-weighting','token_entropy'),tmp_path)
    result=subprocess.run([str(ROOT/'run_stage1_training.sh'),'--dry-run','--stage1-v3-gain-weighting','token_entropy'],
        env={**os.environ,'STAGE1_VARIANT':'standard'},cwd=ROOT,text=True,capture_output=True)
    assert result.returncode!=0 and 'Stage1-v3' in result.stderr
