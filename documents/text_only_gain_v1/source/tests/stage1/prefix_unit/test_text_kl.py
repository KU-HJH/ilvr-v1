"""Text consistency and optimizer routing, using CPU tensor graphs only."""
import importlib
import math

import pytest
import torch
from torch.utils.checkpoint import checkpoint


def api():
    assert importlib.util.find_spec('stage1.text_kl') is not None, 'text KL is missing'
    return importlib.import_module('stage1.text_kl')


@pytest.mark.parametrize('chunk', [1, 3])
def test_forward_kl_sample_balance_detaches_clean_and_trains_masked(chunk):
    module = api()
    clean = torch.tensor([[[2 * math.log(3), 0.]] * 2, [[0., 0.]] * 2], requires_grad=True)
    masked = torch.zeros(3, 2, requires_grad=True)
    keys = ((0, 0), (0, 1), (1, 0))
    collector = module.TextKLAccumulator(clean, keys, temperature=2.)
    for start in range(0, 3, chunk):
        collector.consume(keys[start:start + chunk], masked[start:start + chunk])
    loss, metrics = collector.finish()
    assert loss.item() == pytest.approx(.2616240719, abs=1e-6)
    loss.backward()
    assert clean.grad is None
    torch.testing.assert_close(masked.grad, torch.tensor([[-.125, .125], [-.125, .125], [0., 0.]]))
    assert metrics['text_kl_query_count'].item() == 3


def test_query_coverage_is_exact_but_chunk_order_can_follow_blocks():
    collector = api().TextKLAccumulator(torch.zeros(1, 3, 2), ((0, 0), (0, 2)), temperature=1.)
    collector.consume(((0, 2),), torch.zeros(1, 2, requires_grad=True))
    with pytest.raises(ValueError, match='incomplete'):
        collector.finish()
    with pytest.raises(ValueError, match='query'):
        collector.consume(((0, 2),), torch.zeros(1, 2))
    with pytest.raises(ValueError, match='query'):
        collector.consume(((0, 1),), torch.zeros(1, 2))
    collector.consume(((0, 0),), torch.zeros(1, 2, requires_grad=True))
    assert collector.finish()[0].item() == 0.


def test_joint_vjp_preserves_gain_and_routes_scaled_kl_without_third_recompute():
    # A separate KL autograd.grad would replay this checkpoint a third time.
    api()
    from stage1.vision_gain_execution import Stage1GainScores
    from stage1.vision_gain_update import GainGradientAccumulator
    param = torch.nn.Parameter(torch.tensor([.2, .7]))
    unused = torch.nn.Parameter(torch.tensor([1.]))
    alias = param.detach().requires_grad_(True)
    unused_alias = unused.detach().requires_grad_(True)
    calls = []
    def tensor_graph(x):
        calls.append(True)
        return x.sin()
    h = checkpoint(tensor_graph, alias, use_reentrant=False)
    r, rv, kl = h[0] + 2 * h[1], 3 * h[0] - h[1], (h ** 2).sum()
    accumulator = GainGradientAccumulator((param, unused))
    scores = Stage1GainScores(r, rv, 1, (alias, unused_alias))
    proxy = accumulator.accumulate(scores, optimizer_loss=kl)
    assert len(calls) == 3  # One tensor evaluation + the existing two recomputations.
    torch.testing.assert_close(accumulator.clean_grads[0], param.detach().cos() * torch.tensor([1., 2.]))
    torch.testing.assert_close(accumulator.masked_grads[0], param.detach().cos() * torch.tensor([3., -1.]))
    assert proxy.item() == pytest.approx(kl.detach().item())
    (.5 * (param.square().sum() + .3 * proxy)).backward()
    expected = param.detach() + .3 * param.detach().sin() * param.detach().cos()
    torch.testing.assert_close(param.grad, expected)
    assert unused.grad is None and alias.grad is None
    assert len(calls) == 3  # Optimizer backward must not revisit the alias graph.


def test_optimizer_receives_sum_of_microbatch_gradients():
    api()
    from stage1.vision_gain_execution import Stage1GainScores
    from stage1.vision_gain_update import GainGradientAccumulator
    param = torch.nn.Parameter(torch.tensor([2.]))
    accumulator = GainGradientAccumulator((param,))
    for target in (0., 1.):
        alias = param.detach().requires_grad_(True)
        proxy = accumulator.accumulate(Stage1GainScores(alias.sum(), -alias.sum(), 1, (alias,)),
                                       optimizer_loss=((alias - target) ** 2).sum())
        (proxy / 2).backward()
    torch.testing.assert_close(param.grad, torch.tensor([3.]))
    assert accumulator.block_count == 2


def test_checkpoint_dropout_shares_mask_across_vjp_lanes_and_preserves_rng():
    module = api()
    x = torch.tensor([.2, .7, 1., 2.], requires_grad=True)
    calls = []
    def tensor_graph(t):
        calls.append(True)
        return torch.nn.functional.dropout(t.sin(), p=.5, training=True)
    torch.manual_seed(42)
    initial_rng = torch.get_rng_state()
    h = checkpoint(tensor_graph, x, use_reentrant=False)
    after_forward = torch.get_rng_state()
    r, kl, rv = h.sum(), h.square().sum(), -h.sum()
    r_grad, kl_grad = module.joint_positive_kl_gradients(r, kl, (x,))
    torch.autograd.grad(rv, x)
    assert len(calls) == 3
    assert torch.equal(after_forward, torch.get_rng_state())
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(initial_rng)
        mask = torch.nn.functional.dropout(torch.ones(4), p=.5, training=True)
    torch.testing.assert_close(r_grad[0], mask * x.detach().cos())
    torch.testing.assert_close(kl_grad[0], 2 * mask.square() * x.detach().sin() * x.detach().cos())


def test_proxy_omits_unused_parameter_graph_edges_for_reducer_hook_count():
    p = torch.nn.Parameter(torch.tensor(1.))
    unused = torch.nn.Parameter(torch.tensor(2.))
    unused_node = unused.expand_as(unused).grad_fn.next_functions[0][0]
    expected_hooks = []
    p.register_hook(lambda grad: expected_hooks.append(torch._C._will_engine_execute_node(unused_node)))
    loss = api().optimizer_gradient_loss(torch.tensor(3.), (torch.tensor(4.), None), (p, unused))
    loss.backward()
    assert expected_hooks == [False], 'unused graph edges overcount DeepSpeed post-accumulate hooks'
    assert p.grad.item() == 4. and unused.grad is None


def test_cli_reads_independent_text_kl_settings(tmp_path):
    import json
    from dataclasses import asdict
    from utils_deepseed import parse_args
    from stage1.prefix_mask_config import PrefixConditioningConfig
    path = tmp_path / 'prefix.json'
    path.write_text(json.dumps(asdict(PrefixConditioningConfig(enabled=True, image_max_mask_ratio=0.))))
    config = api().TextKLConfig.from_args(parse_args(['--stage1_text_kl_beta', '.2',
        '--stage1_text_kl_warmup_ratio', '.1', '--stage1_prefix_conditioning_config', str(path),
        '--stage1_vision_gain_execution', 'packed', '--stage1_vision_gain_alpha', '.001',
        '--attention_implementation', 'sdpa']))
    assert config.beta == .2 and config.warmup_ratio == .1


def test_conditioning_request_consumes_only_existing_positive_logits():
    from test_selection import fixture
    from stage1.prefix_mask_accounting import DecoderCallLedger
    from stage1.prefix_mask_config import PrefixConditioningConfig
    from stage1.vision_gain_packed_execution import PrefixConditioningContext
    snapshot, _ = fixture()
    collector = api().TextKLAccumulator(torch.zeros(1, 40, 4), snapshot.gt_queries, temperature=2.)
    context = PrefixConditioningContext(PrefixConditioningConfig(enabled=True, image_max_mask_ratio=0.),
        snapshot, None, DecoderCallLedger(), dict(parameter_version=3, teacher_version=2))
    context.text_kl = collector
    request = context.request({})
    logits = torch.randn(len(snapshot.gt_queries), 4, requires_grad=True)
    request.consume_positive_logits(snapshot.gt_queries, logits)
    loss, metrics = collector.finish()
    loss.backward()
    assert logits.grad.abs().sum() > 0 and metrics['text_kl_query_count'] == len(snapshot.gt_queries)


@pytest.mark.parametrize('weight', [0., .1])
def test_packed_executor_collects_kl_and_preserves_original_gain(monkeypatch, weight):
    from types import SimpleNamespace
    from test_execution import _context_fixture
    from stage1.vision_gain_execution import capture_gain_rng
    from stage1.vision_gain_packed_execution import execute_packed_stage1_gain
    from stage1.vision_gain_update import GainGradientAccumulator
    context, inputs, plan = _context_fixture()
    model = torch.nn.Module()  # Parameter container; its forward is never called.
    model.weight = torch.nn.Parameter(torch.tensor(2.))
    clean = torch.zeros(1, 6, 2, requires_grad=True)
    context.text_kl = api().TextKLAccumulator(clean, context.snapshot.gt_queries, temperature=2.)
    context.text_kl_weight = weight
    def tensor_boundary(module, state, args, kwargs, **options):
        alias = state[0]['weight']
        logits = torch.stack((alias, -alias))[None]
        kwargs['stage1_vision_gain_packed_request'].conditioning.consume_positive_logits(((0, 4),), logits)
        return SimpleNamespace(logits=None, stage1_gain_positive_scores=(alias.square() * 3).view(1),
                               stage1_gain_masked_scores=(alias.square() * 2).view(1))
    monkeypatch.setattr(torch.func, 'functional_call', tensor_boundary)
    accumulator = GainGradientAccumulator((model.weight,))
    rng = capture_gain_rng(model)
    execute_packed_stage1_gain(model, inputs, plan, None, rng, rng,
                              accumulator=accumulator, conditioning_context=context)
    assert accumulator.clean_grads[0].item() == 12. and accumulator.masked_grads[0].item() == 8.
    assert model.weight.grad is None and clean.grad is None
    loss = model.weight * 0 + weight * context.text_kl.optimizer_loss
    loss.backward()
    assert model.weight.grad.item() == pytest.approx(weight * 2 * math.tanh(1.), abs=1e-6)
    assert context.text_kl.metrics['text_kl_query_count'].item() == 1
    assert not context.text_kl.losses and clean.grad is None


@pytest.mark.parametrize('beta,step,fail_gain', [(0., 5, False), (.1, 0, False),
                                              (.1, 5, False), (.1, 5, True), (.1, 5, 'nonfinite')])
def test_trainer_adds_text_kl_optimizer_gradient_but_no_clean_target_gradient(
        monkeypatch, beta, step, fail_gain):
    from types import SimpleNamespace
    from test_metadata import snapshot_inputs
    from stage1.trainer import CustomTrainerStage1
    from stage1.prefix_mask_config import PrefixConditioningConfig
    from stage1.prefix_mask_accounting import DecoderCallLedger
    from stage1.vision_gain_execution import Stage1GainScores
    from stage1.vision_gain_update import GainGradientAccumulator
    import stage1.vision_gain_execution as execution
    targets, metadata, _ = snapshot_inputs()
    clean = torch.zeros(1, 18, 18)
    clean[..., 0] = 2.
    clean.requires_grad_()
    param = torch.nn.Parameter(torch.zeros(18))
    ledger = DecoderCallLedger()
    class ModelBoundary:
        training = True
        def parameters(self):
            return iter((param,))
        def __call__(self, **kwargs):
            # Supply existing tensors; never instantiate or execute a model.
            return SimpleNamespace(logits=clean, loss=clean.sum() * 0 + param.sum() * 0 + 1.)
    model = ModelBoundary()
    config = api().TextKLConfig(beta=beta)
    gain = SimpleNamespace(completed_windows=0, accumulator=GainGradientAccumulator((param,)))
    def execute(*args, conditioning_context, **kwargs):
        collector = conditioning_context.text_kl
        if not config.enabled:
            assert collector is None
            return
        assert collector is not None, 'trainer must attach the text KL collector'
        if fail_gain == 'nonfinite':
            collector.consume(conditioning_context.snapshot.gt_queries, torch.full((3, 18), float('nan')))
        if fail_gain:
            gain.accumulator.fail('test failure')
            return
        alias = param.detach().requires_grad_(True)
        keys = conditioning_context.snapshot.gt_queries
        collector.consume(keys, alias.expand(len(keys), -1))
        kl, _ = collector.finish()
        scores = Stage1GainScores(alias.sum(), -alias.sum(), 1, (alias,))
        collector.optimizer_loss = gain.accumulator.accumulate(scores, optimizer_loss=kl)
    monkeypatch.setattr(execution, 'execute_stage1_gain', execute)
    trainer = SimpleNamespace(model=model, _validate_ema_teacher_device=lambda _: None,
        _validate_targets=lambda *a: None, target_builder=SimpleNamespace(build=lambda *a: targets),
        ema_teacher=SimpleNamespace(model=object()), stage1_preference=SimpleNamespace(enabled=False),
        text_kl_config=config, state=SimpleNamespace(global_step=step, max_steps=100),
        vision_gain_config=SimpleNamespace(execution='packed', branch_mode='independent'),
        _vision_gain=gain, prefix_conditioning_config=PrefixConditioningConfig(enabled=True, image_max_mask_ratio=0.),
        _prefix_teacher_version=0, _prefix_parameter_version=0,
        _sample_balanced_ce_loss=lambda outputs, labels, loss: loss,
        _alignment_loss=lambda *a: torch.tensor(0.), ce_weight=1., sim_weight=1.,
        collect_stage1_diagnostics=True)
    ids = torch.arange(18)[None]
    loss = CustomTrainerStage1._compute_loss(trainer, model, dict(input_ids=ids, labels=ids,
        attention_mask=torch.ones_like(ids), stage1_batch_metadata=metadata),
        conditioning_ledger=ledger, conditioning_report=dict(parameter_version=0, teacher_version=0))
    loss.backward()
    weight = 0. if fail_gain else config.weight(step, 100)
    assert not clean.grad.any()
    assert bool(param.grad.abs().sum()) == bool(weight)
    if config.enabled and not fail_gain:
        metrics = trainer._latest_stage1_diagnostics
        assert metrics['text_kl_beta'].item() == pytest.approx(weight)
        assert loss.item() == pytest.approx(1 + weight * metrics['text_kl_raw'].item())


def test_text_kl_configuration_and_resume_contract(tmp_path):
    from stage1.prefix_mask_config import PrefixConditioningConfig
    from stage1.vision_gain import Stage1VisionGainConfig
    module = api()
    config = module.TextKLConfig(beta=.1, warmup_ratio=.1)
    config.validate_runtime(PrefixConditioningConfig(enabled=True, image_max_mask_ratio=0.),
                            Stage1VisionGainConfig(.001, execution='packed'))
    with pytest.raises(ValueError, match='text-only'):
        config.validate_runtime(PrefixConditioningConfig(enabled=True),
                                Stage1VisionGainConfig(.001, execution='packed'))
    module.TextKLConfig().validate_resume(tmp_path)
    with pytest.raises(ValueError, match='missing'):
        config.validate_resume(tmp_path)
    config.save(tmp_path)
    config.validate_resume(tmp_path)
    with pytest.raises(ValueError, match='mismatch'):
        module.TextKLConfig().validate_resume(tmp_path)
    assert config.weight(0, 100) == 0 and config.weight(5, 100) == .05
    assert config.weight(10, 100) == .1


@pytest.mark.parametrize('settings', [dict(beta=-1), dict(beta=True), dict(beta=float('nan')),
                                    dict(warmup_ratio=-1), dict(warmup_ratio=2)])
def test_invalid_config_rejected(settings):
    with pytest.raises(ValueError):
        api().TextKLConfig(**settings)
