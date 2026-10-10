import json
from dataclasses import replace
from pathlib import Path
import pytest
from src import multinode as mn,multinode_runtime as rt
from tests.multinode.test_v3_roots import LOCAL,REMOTE,HOST,native_plan


def test_fixed_acceptance_route_uses_each_node_root_and_four_ranks():
    builder=getattr(rt,'build_owned_v3_acceptance_commands',None)
    assert callable(builder),'missing fixed v3 acceptance route'
    p=native_plan()
    for phase in ('initial','resume'):
        commands=builder(replace(p,attempt_id=phase),commit='a'*40)
        for rank,root in ((0,REMOTE),(1,LOCAL)):
            cmd=commands.node_specs[rank].command
            assert str(root/'tests/stage1/v3_runtime_acceptance.py') in cmd
            assert cmd[cmd.index('--num_processes')+1]=='4'
            assert cmd[cmd.index('--num_machines')+1]=='2'
            assert cmd[cmd.index('--phase')+1]==phase
            assert cmd[cmd.index('--deepspeed')+1]==str(root/'configs/deepspeed_zero1.json')
    with pytest.raises(ValueError):builder(replace(p,zero_stage=2),commit='a'*40)
    with pytest.raises(TypeError):builder(p,commit='a'*40,harness_path='other.py')


def test_acceptance_coordinator_dry_run_is_bounded_and_preserves_native_roots(capsys):
    rc=mn._training_main('stage1',['--stage1-v3-runtime-acceptance','--run-id','v3-acceptance-dry',
        '--zero-stage','1','--main-process-port','29689','--dry-run',
        '--local-repo-root',str(LOCAL),'--remote-repo-root',str(REMOTE),'--remote-host-repo-root',str(HOST)])
    assert rc==0
    record=json.loads(capsys.readouterr().out)
    assert set(record['attempts'])=={'initial','resume'}
    assert record['successful_update_budget']==4
    assert record['profile']['gradient_accumulation_steps']==4
    assert str(REMOTE/'tests/stage1/v3_runtime_acceptance.py') in str(record)
    assert str(LOCAL/'tests/stage1/v3_runtime_acceptance.py') in str(record)


def test_normal_v3_coordinator_resolves_profile_before_launch_and_result_validation(monkeypatch,capsys):
    calls=[];original=mn.build_owned_tmux_commands
    def observed(*a,**kw):calls.append((a,kw));return original(*a,**kw)
    monkeypatch.setattr(mn,'build_owned_tmux_commands',observed)
    assert mn._training_main('stage1',['--run-id','v3-profile-route','--zero-stage','1','--main-process-port','29689','--dry-run',
        '--remote-repo-root',str(REMOTE),'--remote-host-repo-root',str(HOST),'--stage1-variant','v3',
        '--method','full','--micro-batch','2','--gradient-accumulation','2','--ce-weight','1','--ema-tau','0.999','--max-steps','431'])==0
    args,kwargs=calls[0];forwarded=kwargs['stage_arguments']
    assert args[0].tuning_method=='dora'
    assert forwarded[forwarded.index('--micro-batch')+1]=='1'
    assert forwarded[forwarded.index('--gradient-accumulation')+1]=='4'
    assert forwarded[forwarded.index('--ce-weight')+1]=='0.1'
    assert forwarded[forwarded.index('--ema-tau')+1]=='1.0'
    assert '--max-steps' not in forwarded
