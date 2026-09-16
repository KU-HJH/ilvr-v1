"""Catch clean-prefix reuse, asymmetric image access, and detached gain scores."""
import copy
from dataclasses import replace

import pytest
import torch

from step_preference.config import StepPreferenceConfig
from step_preference.equivalence import _tiny_model, _trainer
from step_preference.sidecar import build_context_sidecar_plan
from step_preference.spans import ReasoningStepSpan
from transformers.models.qwen2_5_vl import stage2_shared_prefix as replay


def case():
    torch.manual_seed(481)
    config = _tiny_model().config
    config.num_hidden_layers = 2
    config._attn_implementation = 'sdpa'
    model = type(_tiny_model())(config).train()
    model.visual.requires_grad_(False)
    ids = torch.tensor([[11, 80, 14, 91, 90, 90, 92, 12, 13]])
    span = ReasoningStepSpan(0, 0, 3, 4, 6, 7, 9, None)
    positions = torch.arange(9)[None, None].expand(3, 1, -1).clone()
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
                 position_ids=positions, sample_ids=('image-evidence-gain',),
                 reasoning_step_spans=(span,), step_preference={'steps': ({'span': span},)})
    return model, batch


def config(**kwargs):
    return StepPreferenceConfig(enabled=True, mode='stage2b',
        stage2a_checkpoint_path='tiny-stage2a', visual_only_context_preference=True,
        visual_off_suppress_latent_boundaries=False, lambda_preference=10.,
        lambda_entropy=0., sidecar_attention_backend='dense', **kwargs)


def test_clean_prefix_remains_the_default():
    """An opt-in change must not silently select the new scoring route."""
    assert getattr(config(), 'prefix_mode', None) == 'clean_prefix'
    assert config().block_original_image is False


@pytest.mark.parametrize('blocked', [False, True])
def test_native_gain_uses_two_recomputed_scores_and_clean_ce(blocked):
    """Catches using clean-root logits for the positive image-blocked score."""
    model, batch = case()
    trainer = _trainer(config(prefix_mode='v_off_prefix', block_original_image=blocked))
    loss, output = trainer.compute_loss(model, batch, return_outputs=True)
    scores = {r['kind']: r['mean_logp'] for r in output.stage2_sidecar_results}
    assert set(scores) == {'visual_on', 'visual_off'}
    expected_pref = -torch.nn.functional.logsigmoid(.1*(scores['visual_on']-scores['visual_off']))
    torch.testing.assert_close(loss, output.loss + 10*expected_pref)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert trainer._step_preference_metric_totals['stats/logp_evidence_on'] == pytest.approx(float(scores['visual_on'].detach()))


def test_image_allowed_gain_matches_legacy_loss_and_gradients_without_dropout():
    """Catches changing targets, positions, latent generation, or CE in the new path."""
    model, batch = case()
    legacy = copy.deepcopy(model)
    old_loss = _trainer(config()).compute_loss(legacy, batch)
    new_loss = _trainer(config(prefix_mode='v_off_prefix')).compute_loss(model, batch)
    torch.testing.assert_close(new_loss, old_loss, atol=1e-5, rtol=1e-5)
    old_loss.backward()
    new_loss.backward()
    for (name, p), (_, old) in zip(model.named_parameters(), legacy.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None and old.grad is not None, name
            torch.testing.assert_close(p.grad, old.grad, atol=2e-5, rtol=2e-3, msg=name)


@pytest.mark.parametrize('blocked', [False, True])
def test_fixed_latents_image_perturbation_and_derivatives(blocked):
    """Blocking only target attention to already-clean KV must fail this test."""
    function = getattr(replay, 'score_evidence_gain', None)
    assert callable(function), 'full-prefix evidence-gain scorer is missing'
    model, batch = case()
    plan = build_context_sidecar_plan(clean_input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'], spans=batch['reasoning_step_spans'], visual_only=True)
    options = dict(plan=plan, prefix_mode='v_off_prefix', block_original_image=blocked,
                   checkpoint=False, backend='dense')
    # Freeze every decoder input except the raw image position. In particular,
    # image-derived latent vectors are held fixed: they are the tested evidence.
    embeds = model.model.embed_tokens(batch['input_ids']).detach().requires_grad_()
    original, _ = function(model, options, batch['input_ids'], embeds, batch['position_ids'])
    changed = embeds.detach().clone()
    changed[:, 1, 0] += 5
    perturbed, _ = function(model, options, batch['input_ids'], changed, batch['position_ids'])
    for a, b in zip(original, perturbed):
        if blocked:
            torch.testing.assert_close(a['mean_logp'], b['mean_logp'], atol=0, rtol=0)
        else:
            assert abs(float(a['mean_logp']-b['mean_logp'])) > 1e-6
        derivative = torch.autograd.grad(a['mean_logp'], embeds, retain_graph=True)[0]
        assert bool(derivative[:, 1].count_nonzero()) is (not blocked)
        if a['kind'] == 'visual_off':
            assert derivative[:, 4:6].count_nonzero() == 0
        else:
            assert derivative[:, 4:6].count_nonzero() > 0
        # START and END must still contribute to predictions on BOTH branches.
        assert derivative[:, 3].count_nonzero() > 0
        assert derivative[:, 6].count_nonzero() > 0


def test_identical_visibility_is_an_exact_null_with_dropout():
    """Independent dropout draws must not manufacture an evidence gain."""
    function = getattr(replay, 'score_evidence_gain', None)
    assert callable(function)
    model, batch = case()
    for module in model.modules():
        if hasattr(module, 'attention_dropout'):
            module.attention_dropout = .2
    plan = build_context_sidecar_plan(clean_input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'], spans=batch['reasoning_step_spans'], visual_only=True)
    branch = plan.groups[0][0]
    plan = replace(plan, groups=((replace(branch, blocked_end=branch.blocked_start),), ()))
    embeds = model.model.embed_tokens(batch['input_ids'])
    result, _ = function(model, dict(plan=plan, prefix_mode='v_off_prefix',
        block_original_image=True, checkpoint=True, backend='dense'),
        batch['input_ids'], embeds, batch['position_ids'])
    a, b = [r['mean_logp'] for r in result]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    gradient = torch.autograd.grad(a-b, embeds)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-7, rtol=0)


@pytest.mark.parametrize('kwargs', [dict(prefix_mode='unknown'),
    dict(block_original_image=True), dict(prefix_mode='v_off_prefix', block_original_image=1)])
def test_invalid_prefix_or_image_policy_fails_closed(kwargs):
    """Do not silently ignore a requested ablation or interpret an integer as bool."""
    with pytest.raises(ValueError):
        config(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Flex integration needs CUDA')
@pytest.mark.parametrize('blocked', [False, True])
def test_cuda_flex_checkpoint_matches_dense_loss_and_gradients(blocked):
    model, batch = case()
    model = model.cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    dense = copy.deepcopy(model)
    cfg = config(prefix_mode='v_off_prefix', block_original_image=blocked)
    dense_loss = _trainer(cfg).compute_loss(dense, batch)
    flex_loss = _trainer(replace(cfg, sidecar_attention_backend='flex')).compute_loss(model, batch)
    torch.testing.assert_close(flex_loss, dense_loss, atol=1e-5, rtol=1e-5)
    dense_loss.backward()
    flex_loss.backward()
    for (name, p), (_, old) in zip(model.named_parameters(), dense.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, old.grad, atol=2e-5, rtol=2e-3, msg=name)
