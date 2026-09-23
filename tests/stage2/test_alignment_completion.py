"""Causal contracts for the remaining approved external-until profile deltas."""
from types import SimpleNamespace
import pytest
import torch


def diagnostic_trainer(profile='latest_external_until', entropy=0.):
    from accelerate import Accelerator
    from transformers.trainer_callback import CallbackHandler, TrainerState, TrainerControl, DefaultFlowCallback
    from step_preference.config import StepPreferenceConfig
    from step_preference.equivalence import _trainer
    trainer = _trainer(StepPreferenceConfig(enabled=True, mode='stage2b',
        visual_only_context_preference=True, direct_from_stage1=True, beta=.5, lambda_entropy=entropy))
    trainer.stage2_execution_profile = profile
    trainer.args = SimpleNamespace(logging_steps=20, logging_first_step=False,
        include_num_input_tokens_seen=False, stage2_debugging=False)
    trainer.accelerator = Accelerator(cpu=True)
    trainer.state = TrainerState(global_step=19)
    trainer.control = TrainerControl()
    trainer.callback_handler = CallbackHandler([DefaultFlowCallback], None, None, None, None)
    return trainer


def metrics(trainer, ce):
    # Two equal-length steps: margins are +2,-2, mean=0. Mean of softplus
    # losses is 0.8132616875 at beta=.5, not softplus(mean margin)=.693147.
    values = trainer._reduce_context_objective(torch.tensor(float(ce)),
        [torch.tensor(-1.), torch.tensor(-4.)], [torch.tensor([1.]), torch.tensor([2.])],
        {(0,'visual'): {'mean_logp':torch.tensor(-3.), 'entropies':torch.tensor([3.])},
         (1,'visual'): {'mean_logp':torch.tensor(-2.), 'entropies':torch.tensor([5.])}})[1]
    values.update({'stats/scored_tokens':4., 'stats/logical_scoring_groups':2.})
    return values


def test_profile_consumes_four_workers_with_persistent_prefetch(monkeypatch):
    # Catches profile CLI resolving4 but SFT config retaining0/nonpersistent workers.
    import main
    from utils_deepseed import parse_args
    args = parse_args(['--stage','stage2','--stage2_execution_profile','latest_external_until',
                      '--dataloader_num_workers','0'])
    # SFTConfig otherwise initializes CUDA; this test only exercises our config boundary.
    monkeypatch.setattr(main, 'SFTConfig', lambda **kw: SimpleNamespace(**kw))
    resolved = main.build_stage2_training_arguments(args)
    assert resolved.dataloader_num_workers == 4
    assert resolved.dataloader_prefetch_factor == 2
    assert resolved.dataloader_persistent_workers is True
    assert resolved.logging_steps == 20
    legacy = parse_args(['--stage','stage2'])
    assert legacy.dataloader_num_workers == 0


def test_profile_logs_last_microbatch_exact_reference_schema_and_formula():
    trainer = diagnostic_trainer()
    trainer.accelerator.gradient_state._set_sync_gradients(False)
    trainer._accumulate_step_preference_metrics(metrics(trainer, 100.))
    trainer.accelerator.gradient_state._set_sync_gradients(True)
    trainer._accumulate_step_preference_metrics(metrics(trainer, 3.))
    trainer.state.global_step = 20
    trainer.log({'loss': 7.5})
    log = trainer.state.log_history[-1]
    assert log == pytest.approx({'loss':7.5, 'step':20,
        'loss/ce':3., 'loss/context_preference_all':.8132616875,
        'loss/total':3.8132616875, 'loss/context_preference_visual':.8132616875,
        'loss/context_preference_text':0., 'stats/context_margin_visual':0.,
        'stats/context_margin_text':0., 'stage2/debugging':0, 'stage2/entropy_computed':0})
    trainer.log({'train_loss': 8.})
    assert trainer.state.log_history[-1] == {'train_loss':8., 'step':20}


def test_profile_diagnostics_follow_next_update_cadence_and_first_step():
    trainer = diagnostic_trainer()
    trainer.state.global_step = 0
    trainer._accumulate_step_preference_metrics(metrics(trainer, 3.))
    trainer.log({'loss':7.})
    assert trainer.state.log_history[-1] == {'loss':7., 'step':0}
    trainer.args.logging_first_step = True
    trainer._accumulate_step_preference_metrics(metrics(trainer, 3.))
    trainer.state.global_step = 1
    trainer.log({'loss':7.})
    assert trainer.state.log_history[-1]['loss/ce'] == 3.


def test_legacy_keeps_all_microbatch_diagnostics():
    trainer = diagnostic_trainer('legacy')
    trainer._accumulate_step_preference_metrics(metrics(trainer, 100.))
    trainer._accumulate_step_preference_metrics(metrics(trainer, 3.))
    trainer.log({'loss':7.})
    assert trainer.state.log_history[-1]['loss/ce'] == 51.5
    assert 'loss/context_preference' in trainer.state.log_history[-1]
    assert 'stats/entropy_clean' in trainer.state.log_history[-1]


@pytest.mark.parametrize('path', ['external', 'aligned'])
def test_observer_cpu_saved_tensors_wraps_only_native_forward_and_preserves_gradients(path):
    # An outer hook witnesses which scope owns saved tensors on CPU too. The
    # optional native-forward hook must override it, then leave observer reads
    # and ordinary backward outside, without a second forward or RNG change.
    from scripts.verify_until_trainer import probed_trainer_class
    class Native:
        def compute_loss(self, model, inputs, **kwargs):
            self.calls += 1
            loss = (model * inputs['x']).sin().square().sum()
            self._gate_capture = (loss, {'loss':loss.detach().clone()})
            return loss
    outcomes = []
    for offload in (False, True):
        cls = probed_trainer_class(Native, {'path':path, 'save_on_cpu':offload}, {})
        trainer = cls.__new__(cls)
        trainer.calls = 0
        trainer._gate_capture = None
        parameter = torch.tensor([.2,.4], requires_grad=True)
        packed, observer_gradients = [], []
        def pack(tensor):
            packed.append(tensor)
            return tensor
        def observe(loss, preference, inputs, scores):
            observer_gradients.append(torch.autograd.grad(preference, parameter, retain_graph=True)[0])
            # Outside the offload context, this operation uses our outer hook.
            parameter.square()
            observer_gradients.append(torch.autograd.grad(loss, parameter, retain_graph=True)[0])
        trainer.gate_observer = SimpleNamespace(observe=observe)
        rng = torch.get_rng_state().clone()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor:tensor):
            value = trainer.compute_loss(parameter, {'x':torch.tensor([2.,3.])})
        assert torch.equal(torch.get_rng_state(), rng)
        assert trainer.calls == 1
        value.backward()
        assert all(torch.equal(g, parameter.grad) for g in observer_gradients)
        outcomes.append((value.detach(), parameter.grad, len(packed)))
    assert torch.equal(outcomes[0][0], outcomes[1][0])
    assert torch.equal(outcomes[0][1], outcomes[1][1])
    assert outcomes[0][2] > 1
    assert outcomes[1][2] == 1


def test_gate_retains_workers_and_uses_first_update_for_native_log_witness():
    import scripts.verify_until_trainer as runner
    assert hasattr(runner, 'configure_probe_training_args'), 'gate still overrides workers and disables logs'
    args = SimpleNamespace(dataloader_num_workers=4, dataloader_persistent_workers=True,
                           dataloader_prefetch_factor=2)
    runner.configure_probe_training_args(args)
    assert args.dataloader_num_workers == 4
    assert args.dataloader_persistent_workers is True
    assert args.logging_strategy == 'steps'
    assert args.logging_steps == 20 and args.logging_first_step is True
    assert args.stage2_debugging is False
    assert args.max_steps == 212 and args.warmup_steps == 10


def test_profile_skips_zero_weight_clean_and_sidecar_entropy_without_changing_gradients(monkeypatch):
    import copy
    import stage2.trainer as module
    from step_preference.shared_prefix_measurement import tiny_measurement_case
    from step_preference.equivalence import _trainer
    from step_preference.config import StepPreferenceConfig
    model, batch = tiny_measurement_case(steps=2, total_target_tokens=4, device='cpu')
    model.config.stage2_execution_profile = 'latest_external_until'
    batch['step_preference']['steps'] = tuple({'span':s} for s in batch['reasoning_step_spans'])
    config = StepPreferenceConfig(enabled=True, mode='stage2b', direct_from_stage1=True,
        sidecar_attention_backend='dense', beta=.5, lambda_entropy=0.,
        visual_only_context_preference=True, target_scope='until_next_latent')
    def run(debug):
        model.zero_grad(set_to_none=True)
        trainer = _trainer(config)
        trainer.stage2_execution_profile = 'latest_external_until'
        trainer.args = SimpleNamespace(stage2_debugging=debug)
        torch.manual_seed(42)
        loss, outputs = trainer.compute_loss(model, copy.deepcopy(batch), return_outputs=True)
        loss.backward()
        gradients = {name:None if p.grad is None else p.grad.clone() for name,p in model.named_parameters()}
        return loss.detach(), gradients, outputs.stage2_sidecar_results, torch.get_rng_state().clone()
    expected = run(True)
    def forbidden(logits):
        pytest.fail('zero-weight profile still computes clean entropy')
    monkeypatch.setattr(module, '_selected_entropies', forbidden)
    actual = run(False)
    assert torch.equal(actual[0], expected[0]) and torch.equal(actual[3], expected[3])
    assert all(result['entropies'].numel() == 0 for result in actual[2])
    for name, value in actual[1].items():
        target = expected[1][name]
        assert (value is None) == (target is None), name
        if value is not None:
            torch.testing.assert_close(value, target, rtol=0, atol=0, msg=name)


def test_cpu_offload_keeps_compiled_checkpoint_dropout_named_reads_and_backward():
    from scripts.verify_until_trainer import probed_trainer_class
    from torch.utils.checkpoint import checkpoint
    from torch._functorch import config
    compiled = torch.compile(lambda x: torch.nn.functional.dropout(x.sin(), p=.25, training=True),
                             backend='aot_eager', fullgraph=True)
    class Native:
        def compute_loss(self, model, inputs, **kwargs):
            self.calls += 1
            hidden = checkpoint(compiled, model['weight'] * inputs['x'], use_reentrant=False,
                                preserve_rng_state=True)
            pref = hidden.square().sum()
            total = pref + model['bias'].square()
            self._gate_capture = (pref, {'preference':pref.detach(), 'total':total.detach()})
            return total
    outcomes = []
    try:
        with config.patch(donated_buffer=False):
            # First compilation initializes device metadata; do it outside a checkpoint.
            compiled(torch.tensor([.2,.4], requires_grad=True)).sum().backward()
            for offload in (False, True):
                cls = probed_trainer_class(Native, {'path':'external', 'save_on_cpu':offload}, {})
                trainer = cls.__new__(cls)
                trainer.calls, trainer._gate_capture = 0, None
                named = {'weight':torch.tensor([.2,.4], requires_grad=True),
                         'bias':torch.tensor(.3, requires_grad=True)}
                reads, hooks = {}, []
                def observe(total, preference, inputs, scores):
                    rng = torch.get_rng_state().clone()
                    for kind, loss in [('preference',preference), ('total',total)]:
                        values = torch.autograd.grad(loss, tuple(named.values()), retain_graph=True, allow_unused=True)
                        reads[kind] = dict(zip(named, values))
                        assert torch.equal(rng, torch.get_rng_state())
                        assert all(p.grad is None for p in named.values())
                    named['bias'].square()  # Must use the restored outer hook.
                trainer.gate_observer = SimpleNamespace(observe=observe)
                torch.manual_seed(42)
                with torch.autograd.graph.saved_tensors_hooks(lambda t: hooks.append(t) or t, lambda t:t):
                    total = trainer.compute_loss(named, {'x':torch.tensor([2.,3.])})
                rng = torch.get_rng_state().clone()
                total.backward()
                assert torch.equal(rng, torch.get_rng_state())
                assert trainer.calls == 1
                assert reads['preference']['bias'] is None
                assert all(torch.equal(p.grad, reads['total'][n]) for n,p in named.items())
                outcomes.append((total.detach(), reads, rng, len(hooks)))
        assert outcomes[1][3] == 1 and outcomes[0][3] > 1
        assert torch.equal(outcomes[0][0], outcomes[1][0])
        assert torch.equal(outcomes[0][2], outcomes[1][2])
        for kind in ('preference','total'):
            for name in named:
                left, right = outcomes[0][1][kind][name], outcomes[1][1][kind][name]
                assert (left is None and right is None) or torch.equal(left, right)
    finally:
        torch._dynamo.reset()


def test_native_prefix_guard_blocks_drift_missing_named_gradients_and_next_micro(tmp_path):
    import scripts.verify_until_trainer as runner
    assert hasattr(runner, 'NativePrefixCheck'), 'offload can continue before exact native prefix checks'
    reference_dir = tmp_path/'native'/'rank-2'
    reference_dir.mkdir(parents=True)
    expected = {'weight':torch.tensor([1.,2.]), 'unused':None}
    reference_file = reference_dir/'micro-0-preference.pt'
    torch.save(expected, reference_file)
    reference = {'tensor_dir':str(reference_dir.parent), 'dataloader_num_workers':0,
                 'ranks':{'2':{'0':{'kinds':['preference'], 'coordinates':{'sample_ids':['a']},
                    'sha256':{'preference':runner.baseline.sha(reference_file)}}}}}
    events = []
    observer = SimpleNamespace(rank=2, write=events.append)
    check = runner.NativePrefixCheck(reference, observer, workers=4)
    check.enter(0, {'sample_ids':['a']})
    with pytest.raises(ValueError, match='incomplete'):
        check.enter(1, {'sample_ids':['b']})
    for wrong in ({'weight':torch.tensor([1.,2.])},
                  {'weight':torch.tensor([1.,2.],dtype=torch.float64),'unused':None},
                  {'weight':torch.tensor([1.,2.00001]),'unused':None},
                  {'weight':torch.tensor([1.,2.]),'unused':torch.tensor(0.)}):
        with pytest.raises(ValueError, match='native prefix'):
            check.snapshot('micro-0-preference', wrong)
    check.snapshot('micro-0-preference', expected)
    check.enter(1, {'sample_ids':['b']})
    assert events[-1]['event'] == 'native_prefix_snapshot'
    assert events[-1]['reference_workers'] == 0 and events[-1]['workers'] == 4
    with pytest.raises(ValueError, match='coordinates'):
        check.enter(0, {'sample_ids':['wrong']})
    reference_file.unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        runner.NativePrefixCheck(reference, observer, workers=4)


def test_profile_projects_full_off_suffix_before_selecting_predictor_rows():
    from transformers.models.qwen2_5_vl.stage2_shared_prefix import score_selected_hidden
    head = torch.nn.Linear(3, 7, bias=False)
    model = SimpleNamespace(lm_head=head, config=SimpleNamespace(stage2_execution_profile='latest_external_until'))
    hidden = torch.arange(12, dtype=torch.float32).reshape(1,4,3).requires_grad_()
    projected_rows = []
    hook = head.register_forward_pre_hook(lambda module,args:projected_rows.append(args[0].shape[1]))
    scores = score_selected_hidden(model, hidden, torch.tensor([0,3]), torch.tensor([2,5]), [2])
    hook.remove()
    expected = head(hidden)[0].float().log_softmax(-1)[torch.tensor([0,3]),torch.tensor([2,5])].mean()
    assert projected_rows == [4], 'reference projects entire OFF suffix before row selection'
    assert torch.equal(scores[0], expected)
    actual_gradient = torch.autograd.grad(scores[0], hidden)[0]
    expected_gradient = torch.autograd.grad(expected, hidden)[0]
    assert torch.equal(actual_gradient, expected_gradient)


def test_diagnostic_collection_does_not_change_loss_rng_or_named_gradients():
    outcomes = []
    for step in (18, 19):
        trainer = diagnostic_trainer()
        trainer.state.global_step = step
        clean = torch.tensor(-1., requires_grad=True)
        off = torch.tensor(-3., requires_grad=True)
        total, values = trainer._reduce_context_objective(torch.tensor(3.), [clean], [],
            {(0,'visual'):{'mean_logp':off, 'entropies':torch.empty(0)}})
        before = torch.get_rng_state().clone()
        trainer._accumulate_step_preference_metrics(values)
        gradients = torch.autograd.grad(total, (clean, off))
        assert torch.equal(before, torch.get_rng_state())
        outcomes.append((total.detach(), gradients))
    assert torch.equal(outcomes[0][0], outcomes[1][0])
    assert all(torch.equal(a,b) for a,b in zip(outcomes[0][1],outcomes[1][1]))
