"""Runtime witness contracts: upstream hook behavior and independent update math."""
import importlib.util
from pathlib import Path
import torch
import pytest


def module():
    path = Path(__file__).resolve().parents[2] / 'src/stage2/reproduction_probe.py'
    assert path.exists(), 'runtime witness implementation missing'
    spec = importlib.util.spec_from_file_location('runtime_probe', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_reference_preserves_pinned_zero_hook_backward_and_dropout_rng():
    # A backward-based reference would prematurely fire the pinned reduction hook.
    from torch.utils.checkpoint import checkpoint
    from deepspeed.utils.torch import register_grad_hook
    p = torch.nn.Parameter(torch.arange(1., 5.))
    events = []
    hook = register_grad_hook(p, lambda value: events.append(value.grad.clone()))
    torch.manual_seed(42)
    loss = checkpoint(lambda x: torch.nn.functional.dropout(x, .3, True).square().sum(),
                      p, use_reentrant=False)
    before = torch.get_rng_state().clone()
    grads = module().reference_gradients(loss, [p])
    assert events == []
    assert p.grad is None
    assert torch.equal(before, torch.get_rng_state())
    loss.backward()
    assert len(events) == 1
    assert torch.equal(grads[0], p.grad)
    hook.remove()


def test_reference_does_not_replace_existing_accumulation():
    p = torch.nn.Parameter(torch.tensor(2.))
    p.grad = torch.tensor(7.)
    assert module().reference_gradients(p.square(), [p])[0] == 4
    assert p.grad == 7


def test_adam_reference_clips_once_and_keeps_moments_at_zero_lr():
    # g=[3,4] clips to [.6,.8]; beta=0, epsilon=0 => sign update.
    ref = module().AdamReference()
    p = [torch.tensor([2., -2.])]
    g = [torch.tensor([3., 4.])]
    options = [dict(lr=0., betas=(0., 0.), eps=0., weight_decay=.1)]
    delta = ref.delta(p, g, options, clip=1.)
    assert torch.equal(delta[0], torch.zeros(2))
    options[0]['lr'] = .01
    delta = ref.delta(p, g, options, clip=1.)
    assert torch.allclose(delta[0], torch.tensor([-.012, -.008]), atol=1e-7)
    assert ref.steps == 2


def test_comparison_fails_quarter_scaled_partial_and_nonfinite():
    compare = module().compare_vectors
    expected = [torch.tensor([4., 8.])]
    assert compare(expected, expected)['pass']
    assert not compare(expected, [torch.tensor([1., 2.])])['pass']
    assert not compare(expected, [torch.tensor([float('nan'), 8.])])['pass']


def test_probe_flag_is_disabled_by_default_and_coordinator_preserves_opt_in():
    from utils_deepseed import parse_args
    from src.multinode_runtime import validated_stage2_environment
    assert parse_args([]).stage2_reproduction_probe_dir == ''
    assert parse_args(['--stage2_reproduction_probe_dir', 'results/probe']).stage2_reproduction_probe_dir == 'results/probe'
    env = validated_stage2_environment(dict(STAGE1_CHECKPOINT='a',DATA_PATH='b',IMAGE_ROOT='c',
        STAGE2_REPRODUCTION_PROBE_DIR='results/probe'),output_dir='checkpoints/test',
        log_file='logs/test',deepspeed_config='configs/deepspeed_zero2.json')
    assert env['STAGE2_REPRODUCTION_PROBE_DIR'] == 'results/probe'


def test_trainer_probe_receives_unscaled_loss_and_real_input_identity():
    from stage2.trainer import Stage2Trainer
    from types import SimpleNamespace
    trainer = object.__new__(Stage2Trainer)
    trainer.stage2_diagnostics_interval = 0
    trainer._stage2_microbatch_index = 0
    p = torch.nn.Parameter(torch.tensor(2.))
    seen = []
    trainer._reproduction_observer = SimpleNamespace(observe_loss=lambda loss, inputs: seen.append((loss,inputs)))
    inputs = {'input_ids':torch.tensor([[1,2]])}
    result = trainer.compute_loss(lambda **kwargs: SimpleNamespace(loss=p.square()), inputs)
    assert result.item() == 4
    assert len(seen) == 1 and seen[0][0] is result and seen[0][1] is inputs


def test_tiny_supplied_model_path_compares_real_forward_and_all_parameter_gradients():
    import os, subprocess, sys, json
    repo = Path(__file__).resolve().parents[2]
    script = repo/'scripts/verify_voff_model_path.py'
    assert script.exists(), 'tiny model-path differential missing'
    result = subprocess.run([sys.executable,str(script)],cwd=repo,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),
        capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert len(payload['cases']) == 2
    assert payload['source_hashes'] and payload['shared_dependency_hashes']
    assert payload['external_files_unchanged']
    for case in payload['cases']:
        assert case['loss']['pass']
        assert case['parameter_gradients']
        assert all(g['pass'] for g in case['parameter_gradients'].values())
        assert case['connected_parameter_count'] > 10
        assert case['preference_loss']['pass']
        assert case['clean_scores']['pass'] and case['visual_off_scores']['pass']
        assert case['preference_gradient_nonzero']


def test_adam_zero_lr_first_step_still_changes_second_step_moments():
    ref = module().AdamReference()
    options=[dict(lr=0.,betas=(.5,.5),eps=0.,weight_decay=0.)]
    ref.delta([torch.tensor([2.])],[torch.tensor([1.])],options,1.)
    options[0]['lr']=.01
    delta=ref.delta([torch.tensor([2.])],[torch.tensor([0.])],options,1.)
    assert delta[0].item() == pytest.approx(-.005773502691896258,abs=1e-8)


def test_vector_comparison_rejects_missing_or_wrong_shape():
    assert not module().compare_vectors([torch.ones(2)],[])['pass']
    assert not module().compare_vectors([torch.ones(2)],[torch.ones(1)])['pass']


def test_observer_records_full_full_partial_vectors_and_real_adam_deltas(tmp_path, monkeypatch):
    # CPU boundary contract, NOT real ZeRO/DDP: mocks only transport/read APIs.
    # It catches swapped numerator denominators, lost group identities, missed
    # clipping observation, BF16-only delta reads, and premature stop on mismatch.
    import json
    from types import SimpleNamespace as N
    import torch.distributed as dist
    import deepspeed.utils as ds
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dist,'get_rank',lambda:0)
    monkeypatch.setattr(dist,'get_world_size',lambda:4)
    monkeypatch.setattr(dist,'all_gather_object',lambda output,value:output.__setitem__(slice(None),[list(value) for _ in range(4)]))
    monkeypatch.setattr(dist,'all_reduce',lambda value:value.mul_(4))
    monkeypatch.setattr(ds,'safe_get_full_grad',lambda p:p.grad.clone())
    monkeypatch.setattr(ds,'safe_get_full_fp32_param',lambda p:p.detach().clone())
    model=torch.nn.Linear(1,1,bias=False)
    model.weight.data.fill_(2.)
    p=model.weight
    optimizer=torch.optim.AdamW([p],lr=0.,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    zero=N(optimizer=optimizer,bit16_groups=[[p]],single_partition_of_fp32_groups=[p],loss_scale=1.,clip_grad=1.,
           process_gradients=lambda *args:None,
           unscale_and_clip_grads=lambda *args:torch.nn.utils.clip_grad_norm_([p],1.))
    trainer=N(deepspeed_runtime_profile=N(zero_stage=2),state=N(global_step=0),
              args=N(per_device_train_batch_size=1,gradient_accumulation_steps=4),
              model_accepts_loss_kwargs=False,compute_loss_func=None)
    observer=module().ReproductionObserver(trainer,N(module=model,optimizer=zero),'results/probe')
    for update,count in enumerate((4,4,1)):
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]['lr']=update*.01
        for i in range(count):
            loss=(i+1)*p.sum()
            observer.observe_loss(loss,{'sample_ids':[f'{update}-{i}'],'input_ids':torch.ones(1,i+2,dtype=torch.long)})
            (loss/4).backward()
        observer.before_step()
        zero.unscale_and_clip_grads([p.grad],p.grad.norm())
        optimizer.step()
        observer.after_step()
    rows=[json.loads(line) for line in (tmp_path/'results/probe/rank-0.jsonl').read_text().splitlines()]
    updates=[r for r in rows if r['event']=='update']
    assert [r['actual_count'] for r in updates]==[16,16,4]
    assert updates[1]['group_ids_by_rank']==[[f'1-{i}' for i in range(4)]]*4
    assert [r['actual_group_gradient']['pass'] for r in updates]==[True,True,False]
    assert all(r['historical_fixed_gradient']['pass'] for r in updates)
    assert all(r['effective_gradient_adam_delta']['pass'] for r in updates)
    assert all(r['clip_once_per_partition_group'] for r in updates)
    assert all(r['clip_group_indices']==[0] for r in updates)
    assert all(r['clip_shared_global_factor'] for r in updates)
    assert updates[0]['observed_delta_l2']==0
    assert updates[1]['observed_delta_l2']==pytest.approx(.0102,abs=1e-6)
    assert updates[2]['observed_delta_l2']>0
    assert all(r['full_run_authorized'] is False for r in updates)
