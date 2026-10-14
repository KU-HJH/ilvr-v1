"""Debug gates must skip diagnostics without changing the real training math."""
from dataclasses import replace
import json
import os
import shlex
import subprocess

import pytest
import torch

from tests.stage1.v5.test_trainer import integrated_fixture, api
from tests.stage1.v5.test_objective import fixture, metadata
from tests.stage1.v5.test_launcher import ROOT, PROFILE, REMOTE, HOST


def test_public_coordinator_shell_parser_carries_debug_and_default_off():
    from utils_deepseed import parse_args
    assert parse_args(['--stage1_variant','v5']).stage1_v5_debug is False
    for flag, expected in [('--stage1-v5-debug',True),('--no-stage1-v5-debug',False)]:
        done=subprocess.run([str(ROOT/'run_multinode_stage1_training.sh'),
            '--run-id','v5-debug-command','--zero-stage','1','--main-process-port','29683',
            '--local-repo-root',str(ROOT),'--remote-repo-root',str(REMOTE),
            '--remote-host-repo-root',str(HOST),'--dry-run','--method','dora',
            '--model','/original/base','--data-path',str(ROOT/'data/TRAIN.jsonl'),
            '--image-root',str(ROOT/'data/images_comt'),*PROFILE,flag],
            text=True,capture_output=True,env=os.environ)
        assert done.returncode==0,done.stderr
        from src.multinode import decode_launch_plan
        from src.multinode_runtime import build_owned_tmux_commands
        plan=decode_launch_plan(json.dumps(json.loads(done.stdout)['plan']).encode())
        commands=build_owned_tmux_commands(plan,commit='a'*40,stage_arguments=[
            '--method','dora','--model','/original/base','--data-path',str(ROOT/'data/TRAIN.jsonl'),
            '--image-root',str(ROOT/'data/images_comt'),*PROFILE,flag])
        for spec in commands.node_specs:
            argv=spec.command[spec.command.index('--')+1:]
            shell=subprocess.run(['bash',str(ROOT/'run_stage1_training.sh'),*argv[2:],'--dry-run'],
                text=True,capture_output=True,env={**os.environ,**dict(spec.environment)})
            assert shell.returncode==0,shell.stderr
            worker=shlex.split(shell.stdout)
            parsed=parse_args(worker[worker.index('src/main.py')+1:])
            assert parsed.stage1_v5_debug is expected


def test_recipe_overrides_sourced_debug_on(tmp_path):
    from tests.stage1.v5.test_workflow import run
    settings=tmp_path/'stale.sh'; settings.write_text('export STAGE1_V5_DEBUG=true\n')
    done=run('--settings',str(settings),'--dry-run')
    assert done.returncode==0,done.stderr
    train=shlex.split(done.stdout.splitlines()[0])
    assert '--no-stage1-v5-debug' in train


def test_real_helper_checkpoint_dropout_loss_all_gradients_rng_match_debug_modes(tmp_path):
    observed=[]
    for debug in (False,True):
        torch.manual_seed(213)
        trainer,batch,calls,_,backs,_=integrated_fixture(tmp_path,real_helper=True,debug=debug)
        from peft_common import resolve_qwen2_5_vl_model
        layer_calls=[]
        handle=resolve_qwen2_5_vl_model(trainer.model).model.layers[0].self_attn.register_forward_pre_hook(
            lambda *_:layer_calls.append(1))
        batches,_=trainer.get_batch_samples(iter([batch]),4)
        loss=trainer.training_step(trainer.model,batches[0])
        handle.remove()
        assert len(layer_calls)==4  # F/packed originals and both backward recomputations.
        grads={name:p.grad.clone() if p.grad is not None else None
               for name,p in trainer.model.named_parameters() if p.requires_grad}
        observed.append((loss,grads,torch.get_rng_state().clone()))
        assert calls==['helper','F','reference','packed'] and backs[0].state=='CLOSED'
        assert bool(trainer._latest_stage1_diagnostics) is debug
        assert bool(trainer.target_builder.targets.diagnostics) is debug
    torch.testing.assert_close(observed[0][0],observed[1][0],rtol=0,atol=0)
    assert observed[0][1].keys()==observed[1][1].keys()
    for name,gradient in observed[0][1].items():
        other=observed[1][1][name]
        if gradient is None: assert other is None,name
        else: torch.testing.assert_close(gradient,other,rtol=0,atol=0,msg=name)
    assert torch.equal(observed[0][2],observed[1][2])


@pytest.mark.parametrize('debug',[False,True])
def test_native_target_consistency_tripwire_gated_but_backward_completes(tmp_path,debug):
    trainer,batch,*_=integrated_fixture(tmp_path,debug=debug)
    original=trainer.target_builder.build
    def build(*args):
        targets=original(*args)
        return replace(targets,selected_counts=((0,0),))
    trainer.target_builder.build=build
    batches,_=trainer.get_batch_samples(iter([batch]),4)
    if debug:
        with pytest.raises(RuntimeError,match='matching teacher target'):
            trainer.training_step(trainer.model,batches[0])
    else:
        assert torch.isfinite(trainer.training_step(trainer.model,batches[0]))
        assert any(p.grad is not None for p in trainer.model.parameters() if p.requires_grad)


@pytest.mark.parametrize('debug',[False,True])
def test_native_deepspeed_diagnostic_tripwire_preserves_actual_step_and_ema(tmp_path,debug):
    from stage1.trainer import _DeepSpeedStepAdapter
    trainer,*_=integrated_fixture(tmp_path,debug=debug)
    trainer._successful_optimizer_steps=0
    parameter=next(p for p in trainer.model.parameters() if p.requires_grad)
    parameter.grad=torch.ones_like(parameter)
    before=parameter.detach().clone()
    from tests.stage1.v5.test_replay import digest_state
    reference_before=digest_state(trainer.ema_teacher.model)
    applied=[]
    class Engine:
        def is_gradient_accumulation_boundary(self): return True
        def was_step_applied(self): return True
        def step(self):
            trainer.optimizer.step(); applied.append('step')
    engine=Engine()
    def invalid(_engine): raise FloatingPointError('invalid injected diagnostic norm')
    trainer._validate_deepspeed_gradient_norm=invalid
    adapter=_DeepSpeedStepAdapter(trainer,engine,engine.step)
    if debug:
        with pytest.raises(FloatingPointError,match='diagnostic norm'): adapter()
        assert applied==[] and trainer._successful_optimizer_steps==0
        assert torch.equal(parameter,before)
    else:
        adapter(); assert applied==['step']
        assert trainer._successful_optimizer_steps==1
        assert not torch.equal(parameter,before)
    assert digest_state(trainer.ema_teacher.model)==reference_before


@pytest.mark.parametrize('validate',[False,True])
def test_helper_checkpoint_retains_validation_mode_after_original_call(validate,monkeypatch):
    from step_preference.dil_bce_orig_loss import orig_components_chunked
    clean,ref,_,_=fixture()
    terms=orig_components_chunked(clean,ref,torch.tensor([0,3,1]),validate=validate,
                                  collect_diagnostics=validate,chunk_positions=2)
    def invalid(tensor): return torch.zeros_like(tensor,dtype=torch.bool)
    monkeypatch.setattr(torch,'isfinite',invalid)
    if validate:
        with pytest.raises(ValueError,match='log probabilities'): (terms.chosen+terms.reference).sum().backward()
    else:
        (terms.chosen+terms.reference).sum().backward()
        assert clean.grad is not None
        assert terms.reference_prob_sum_error is None


def test_default_helpers_still_reject_nonfinite_and_objective_off_has_no_diagnostics():
    from step_preference.dil_bce_orig_loss import orig_components_chunked
    from stage1.v5_objective import compute_v5_loss_sums
    from stage1.v5_config import Stage1V5Config
    clean,ref,off,text=fixture()
    with pytest.raises(ValueError,match='finite'):
        orig_components_chunked(clean*float('nan'),ref,torch.tensor([0,3,1]))
    sums=compute_v5_loss_sums(clean,ref,off,text,metadata(),
        Stage1V5Config(lambda_topk_entropy=.1,preference_topk_k=2,lambda_topk_mass=.2),
        validate=False,collect_diagnostics=False)
    assert sums.diagnostics=={}
    (sums.dil_sum+sums.entropy_sum+sums.mass_sum+sums.text_kl).backward()
    assert clean.grad is not None and off.grad is not None and text.grad is not None


def test_debug_is_runtime_instrumentation_and_not_resume_objective(tmp_path):
    from tests.stage1.v5.test_resume import contract
    from utils_deepseed import parse_args
    from stage1.v5_resume import validate_v5_resume
    saved=contract(); saved['execution']['stage1_v5_debug']=False
    (tmp_path/'stage1_v5.json').write_text(json.dumps(saved))
    args=parse_args(['--resume_from_checkpoint_path',str(tmp_path),'--stage1_v5_debug'])
    assert args.stage1_v5_debug is True
    assert args.stage1_v5_config==validate_v5_resume(saved,{})
    on=json.loads(json.dumps(saved)); on['execution']['stage1_v5_debug']=True
    assert api()._effective_v5_contract(saved)==api()._effective_v5_contract(on)


def test_mass_diagnostics_off_skips_gap_computation_and_preserves_loss_gradient(monkeypatch):
    from step_preference.dil_topk_loss import _group_mass_terms
    from step_preference.topk_entropy_loss import build_entropy_support, entropy_support_mass_components
    clean,_,off,_=fixture()
    support=torch.tensor([[True,True,False,False]]*3)
    strict=_group_mass_terms(clean.detach(),off,support)
    expected_grad=torch.autograd.grad(strict[0].sum(),off,retain_graph=True)[0]
    def forbidden_abs(self): raise AssertionError('diagnostic-only abs gap computed')
    monkeypatch.setattr(torch.Tensor,'abs',forbidden_abs)
    lean=_group_mass_terms(clean.detach(),off,support,collect_diagnostics=False)
    torch.testing.assert_close(lean[0],strict[0],rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(lean[0].sum(),off,retain_graph=True)[0],expected_grad,rtol=0,atol=0)
    assert lean[1:]==(None,None,None)
    with pytest.raises(AssertionError,match='abs gap'):
        _group_mass_terms(clean.detach(),off,support)
    candidates=build_entropy_support(clean,off,torch.tensor([0,3,1]),k=2,mode='independent',include_gt=False,
        validate=False,collect_diagnostics=False)
    lean_components=entropy_support_mass_components(clean,off,candidates,validate=False,collect_diagnostics=False)
    assert lean_components.mass_abs_gap is None


@pytest.mark.parametrize('debug',[False,True])
def test_repeated_teacher_device_scan_tripwire_and_startup_guard(tmp_path,debug):
    trainer,batch,*_=integrated_fixture(tmp_path,debug=debug)
    trainer._module_devices=lambda _module:{torch.device('meta')}
    batches,_=trainer.get_batch_samples(iter([batch]),4)
    if debug:
        with pytest.raises(RuntimeError,match='share a device'):
            trainer.training_step(trainer.model,batches[0])
    else:
        assert torch.isfinite(trainer.training_step(trainer.model,batches[0]))
    # Startup placement validation always runs, even with debug OFF.
    with pytest.raises(RuntimeError,match='placement failed'):
        trainer._prepare_ema_teacher_for_training()


@pytest.mark.parametrize('debug',[False,True])
def test_v5_full_native_numeric_check_tripwire(tmp_path,debug,monkeypatch):
    trainer,batch,*_=integrated_fixture(tmp_path,debug=debug,real_helper=True)
    batches,_=trainer.get_batch_samples(iter([batch]),4)
    monkeypatch.setattr(torch,'isfinite',lambda value:torch.zeros_like(value,dtype=torch.bool))
    if debug:
        with pytest.raises(FloatingPointError,match='finite'):
            trainer.training_step(trainer.model,batches[0])
    else:
        trainer.training_step(trainer.model,batches[0])
        assert any(p.grad is not None for p in trainer.model.parameters() if p.requires_grad)


def test_strict_entropy_support_without_collection_remains_usable():
    from step_preference.topk_entropy_loss import build_entropy_support, topk_entropy_components
    clean,_,off,_=fixture()
    support=build_entropy_support(clean,off,torch.tensor([0,3,1]),k=2,mode='independent',include_gt=False,
                                  collect_diagnostics=False)
    result=topk_entropy_components(off,support,collect_diagnostics=False)
    assert result.normalized_entropy is None
    result.kl.sum().backward()
    assert off.grad is not None


def test_native_hidden_state_helper_retains_static_strict_default():
    from types import SimpleNamespace
    from stage1.targets import Stage1TeacherTargetBuilder
    states=torch.full((1,2,3),float('nan'))
    outputs=SimpleNamespace(hidden_states=(states,states))
    with pytest.raises(FloatingPointError,match='finite'):
        Stage1TeacherTargetBuilder._hidden_states(outputs,(1,2))
    pre,last=Stage1TeacherTargetBuilder._hidden_states(outputs,(1,2),validate=False)
    assert pre is states and last is states


@pytest.mark.parametrize('debug',[False,True])
def test_native_compatibility_optimizer_step_gates_diagnostics_and_retains_update(tmp_path,debug):
    trainer,*_=integrated_fixture(tmp_path,debug=debug)
    trainer._successful_optimizer_steps=0
    parameter=next(p for p in trainer.model.parameters() if p.requires_grad)
    parameter.grad=torch.ones_like(parameter)
    before=parameter.detach().clone()
    def invalid(_model): raise FloatingPointError('injected diagnostic gradient')
    trainer._validate_gradient_norm=invalid
    if debug:
        with pytest.raises(FloatingPointError,match='diagnostic gradient'): trainer.optimizer_step()
        assert trainer._successful_optimizer_steps==0 and torch.equal(parameter,before)
    else:
        trainer.optimizer_step()
        assert trainer._successful_optimizer_steps==1 and not torch.equal(parameter,before)
