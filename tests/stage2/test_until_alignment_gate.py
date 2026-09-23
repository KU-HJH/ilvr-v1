"""Gate failures: permissive envelopes, disconnected zeros and wrong tail weights."""
import importlib.util
import math
import pytest
import torch


def probe():
    assert importlib.util.find_spec('stage2.until_alignment_probe') is not None, 'comparative gate missing'
    from stage2 import until_alignment_probe
    return until_alignment_probe


def test_repeat_envelope_rejects_drift_and_disconnected_zero():
    p = probe()
    controls = [{'w': torch.tensor([1., 2.])} for _ in range(3)]
    envelope = p.freeze_envelope(controls)
    assert p.compare_snapshot(controls[0], controls[1], envelope)['pass']
    assert not p.compare_snapshot(controls[0], {'w': torch.tensor([1., 2.0001])}, envelope)['pass']
    assert not p.compare_snapshot({'w': None}, {'w': torch.zeros(2)}, p.freeze_envelope([{'w': None}]*3))['pass']


def test_control_instability_cannot_create_permissive_tolerance():
    p = probe()
    with pytest.raises(ValueError, match='unstable'):
        p.freeze_envelope([{'w': torch.tensor([1.])}, {'w': torch.tensor([1.03])}, {'w': torch.tensor([1.])}])
    envelope = p.freeze_envelope([{'w': torch.tensor([1.])}, {'w': torch.tensor([1.005])}, {'w': torch.tensor([1.])}])
    assert 0.009 < envelope['w']['relative_l2'] < .01
    assert p.compare_snapshot({'w': torch.tensor([1.])}, {'w': torch.tensor([1.009])}, envelope)['pass']
    assert not p.compare_snapshot({'w': torch.tensor([1.])}, {'w': torch.tensor([1.02])}, envelope)['pass']


def test_zero_nonfinite_and_dtype_mismatch_fail_closed():
    p = probe()
    zero = {'w': torch.zeros(1)}
    envelope = p.freeze_envelope([zero]*3)
    for candidate in ({'w': torch.tensor([1e-30])}, {'w': torch.tensor([float('nan')])}, {'w': torch.zeros(1, dtype=torch.bfloat16)}, {}):
        assert not p.compare_snapshot(zero, candidate, envelope)['pass']


def test_replay_carries_real_duplicate_and_independent_fixed_tail_weights():
    p = probe()
    # Five full groups and four real tail presentations. Duplicate is early.
    order = list(range(83))
    order.insert(3, 1)
    lengths = [10]*83
    lengths[65] = 100
    replay = p.select_windows(order, lengths, [1]*83)
    assert replay['source_updates'] == [1, 2, 3, 5, 6]
    assert replay['indices'][-4:] == [79, 80, 81, 82]
    assert replay['coefficients'][-4:] == [1/16]*4
    assert len(replay['indices']) == 68
    assert replay['indices'].count(1) == 2


def test_tensor_artifacts_cannot_be_written_under_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = probe()
    with pytest.raises(ValueError, match='checkpoints'):
        p.tensor_destination(tmp_path/'results'/'probe')
    assert p.tensor_destination(tmp_path/'checkpoints'/'probe').is_dir()


def test_gate_report_rejects_missing_tail_and_post_update_evidence():
    p = probe()
    records = []
    for rank in range(4):
        records.append(dict(event='init', rank=rank, hostname=f'node{rank//2}', world_size=4,
                            gas=4, microbatch=1, fixed_denominator=16, zero_stage=2,
                            schedule_steps=212, warmup_steps=10))
        records.append(dict(event='compiler',rank=rank,flex_dynamic_true=True,
                            calls=[{'callable':'torch.nn.attention.flex_attention.flex_attention','dynamic':True}]))
        for micro in range(17):
            records.append(dict(event='example', rank=rank, micro=micro, sample_id=f's{micro*4+rank}',
                                coefficient=1/16,weight=1.,peak_allocated=123))
            records.append(dict(event='backward', rank=rank, micro=micro, dora_norm_calls=7,
                                dora_scope_cleaned=True, actual_backward_hooks=10))
        for step in range(1,6):
            records.append(dict(event='update',rank=rank,probe_optimizer_step=step,
                                original_update_index=[1,2,3,99,212][step-1], completed=True,
                                actual_count=4 if step==5 else 16,fixed_denominator=16,
                                lr=[(step-1)*5e-6],master_delta_nonzero=step>1,preference_gradient_nonzero=True,
                                fixed_gradient={'coordinate':{'finite':True,'expected_l2':1.,'l2_error':.001}}))
    replay={'sample_ids':[f's{i}' for i in range(68)],'source_updates':[1,2,3,99,212]}
    assert p.check_runtime(records,replay)['pass']
    for event in records:
        if event['event']=='update':
            event['preference_gradient_nonzero']=False
    assert not p.check_runtime(records,replay)['pass']
    for event in records:
        if event['event']=='update':
            event['preference_gradient_nonzero']=True
    compiler=next(r for r in records if r['event']=='compiler')
    compiler['calls'][0]['dynamic']=False
    assert not p.check_runtime(records,replay)['pass']
    compiler['calls'][0]['dynamic']=True
    records[-1]['fixed_denominator']=4
    assert not p.check_runtime(records,replay)['pass']
    records[-1]['fixed_denominator']=16
    records[:]=[r for r in records if not(r['event']=='backward' and r['micro']==8)]
    assert not p.check_runtime(records,replay)['pass']


def test_gate_spec_refuses_aligned_run_before_envelope_is_frozen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from scripts.verify_until_trainer import load_spec
    import json
    spec=dict(path='aligned',result_dir=str(tmp_path/'results'/'a'),tensor_dir=str(tmp_path/'checkpoints'/'a'),
              external_manifest='source.json',manifest='retained.json',data='data.jsonl',image_root='data',base='base',stage1='stage1')
    path=tmp_path/'spec.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='freeze external controls'):
        load_spec(path)


def test_snapshot_inventory_must_cover_every_rank_and_required_event():
    p=probe()
    with pytest.raises(ValueError,match='inventory'):
        p.required_snapshots({'rank-0/update-1-master-delta.pt'})


def test_external_sidecar_dataclass_matches_actual_deepspeed_hook_traversal():
    # DeepSpeed registers outputs in dict/list/tuple; dataclass sidecars are opaque.
    from dataclasses import dataclass
    from deepspeed.runtime.utils import OutputBackwardHookManager
    p=probe()
    @dataclass
    class Sidecars:
        logits: object
    parameter=torch.tensor([2.],requires_grad=True)
    output={'loss':parameter.square().sum(),'sidecars':Sidecars(parameter*3)}
    manager=OutputBackwardHookManager(lambda:None)
    manager.register_hooks_on_outputs(output)
    observer=p.AlignmentObserver.__new__(p.AlignmentObserver)
    observer._reference_output_managers=[]
    observer._capture_output_hooks(None,(),output)
    assert observer._reference_output_managers==[manager]
    # Backprop through the dataclass is real; capture must not require nonexistent hooks.
    assert torch.autograd.grad(output['sidecars'].logits.sum(),parameter)[0].tolist()==[3.]


def test_reference_uses_frozen_parent_trainer_and_flex_callable():
    from pathlib import Path
    import scripts.verify_until_trainer as runner
    source_root=Path('/raid/workspace/temp2/ILVR/.worktrees/ours1')
    if not source_root.is_dir():
        pytest.skip('frozen source is not installed in this CPU environment')
    assert hasattr(runner,'TrainerExternalSource'), 'frozen framework loader missing'
    with runner.TrainerExternalSource(source_root) as source:
        external=runner.external_modules(source)
        runner.bind_external_flex(source,external)
        import inspect
        assert Path(inspect.getfile(external.trainer.Stage2Trainer.__bases__[0])).resolve()==source_root/'transformers/src/transformers/trainer.py'
        actual=external.model.ALL_ATTENTION_FUNCTIONS['flex_attention']
        assert Path(inspect.getfile(actual)).resolve()==source_root/'transformers/src/transformers/integrations/flex_attention.py'
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        assert actual is not ALL_ATTENTION_FUNCTIONS['flex_attention']


def test_gate_scope_allows_compiled_retained_gradients_and_restores_config(monkeypatch):
    """Removing the gate-only compiler scope breaks warmed AOT retained backward."""
    from types import SimpleNamespace
    from torch._functorch import config
    import scripts.verify_until_trainer as runner

    class ProbeCompleted(Exception):
        pass

    class CpuSourceBoundary:
        def __init__(self, root):
            pass

        def __enter__(self):
            # AOT eager uses real compiled autograd, without a CUDA/kernel dependency.
            # Warm an ordinary backward first: otherwise lazy compilation notices
            # retain_graph and removes the donated-buffer metadata automatically.
            torch._dynamo.reset()
            compiled = torch.compile(lambda x, w: (x @ w).sin().sum(),
                                     backend='aot_eager', fullgraph=True)
            x = torch.tensor([[.2, .4], [.6, .8]], requires_grad=True)
            w = torch.tensor([[.1, .3], [.5, .7]], requires_grad=True)
            compiled(x, w).backward()
            x.grad = w.grad = None
            value = compiled(x, w)
            expected = torch.autograd.grad((x @ w).sin().sum(), (x, w))
            rng = torch.get_rng_state().clone()
            for _ in range(2):
                observed = torch.autograd.grad(value, (x, w), retain_graph=True)
                for actual, target in zip(observed, expected):
                    torch.testing.assert_close(actual, target)
                assert x.grad is None and w.grad is None
                assert torch.equal(torch.get_rng_state(), rng)
            value.backward()
            torch.testing.assert_close(x.grad, expected[0])
            torch.testing.assert_close(w.grad, expected[1])
            raise ProbeCompleted

        def __exit__(self, *args):
            return False

    monkeypatch.setenv('WORLD_SIZE', '4')
    monkeypatch.setattr(runner, 'load_spec', lambda path: {'stage1': '/same-stage1'})
    monkeypatch.setattr(runner.baseline, 'sha', lambda path: 'unused-cpu-boundary')
    monkeypatch.setattr(runner, 'authenticate_source', lambda spec: '/unused-source')
    monkeypatch.setattr(runner, 'TrainerExternalSource', CpuSourceBoundary)
    args = SimpleNamespace(save_model_path='/checkpoints/cpu-probe',
                           stage2_execution_profile='latest_external_until',
                           load_model_path='/same-stage1')
    try:
        with config.patch(donated_buffer=True):
            with pytest.raises(ProbeCompleted):
                runner.run_from_training_args(args, '/unused-spec')
            assert config.donated_buffer is True
    finally:
        torch._dynamo.reset()


def test_memory_diagnostic_observes_without_extra_gradients_and_keeps_backward(tmp_path, monkeypatch):
    from types import SimpleNamespace
    p = probe()
    assert hasattr(p, 'MemoryDiagnosticObserver'), 'ordinary-backward observer missing'
    observer = p.MemoryDiagnosticObserver.__new__(p.MemoryDiagnosticObserver)
    observer.micro, observer.rank, observer.update = -1, 0, 0
    observer.replay = {'sample_ids': ['s0'], 'source_updates': [1, 2]}
    observer.engine = SimpleNamespace(device='cpu')
    observer._reference_output_managers = []
    observer.stream = (tmp_path/'memory.jsonl').open('w')
    observer.tensor_dir = tmp_path
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 100)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 120)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda device: torch.tensor([1],dtype=torch.uint8))
    monkeypatch.setattr(torch.autograd, 'grad', lambda *a, **k: pytest.fail('diagnostic ran extra autograd.grad'))
    parameter = torch.tensor(2., requires_grad=True)
    total = parameter.square()
    observer.memory('micro_entry', sample_id='s0', micro=0, coordinates={'targets': [7]})
    observer.observe(total, total/2, {'sample_ids':['s0'], 'input_ids':torch.tensor([[7]])}, {'ce':total.detach()})
    # Flushed before ordinary backward: available even if backward OOMs.
    events = [__import__('json').loads(x) for x in (tmp_path/'memory.jsonl').read_text().splitlines()]
    assert [x['event'] for x in events] == ['micro_entry', 'forward']
    assert parameter.grad is None
    total.backward()
    assert parameter.grad.item() == 4.
    assert sorted(x.name for x in tmp_path.glob('*.pt')) == ['micro-0-rng.pt','micro-0-scores.pt']
    observer.stream.close()


def test_memory_diagnostic_mode_is_external_only_and_unknown_modes_fail():
    from scripts.verify_until_trainer import memory_diagnostic
    assert memory_diagnostic({'path':'external', 'mode':'ordinary_backward_memory'})
    assert not memory_diagnostic({'path':'external'})
    with pytest.raises(ValueError, match='external'):
        memory_diagnostic({'path':'aligned', 'mode':'ordinary_backward_memory'})
    with pytest.raises(ValueError, match='mode'):
        memory_diagnostic({'path':'external', 'mode':'typo'})
