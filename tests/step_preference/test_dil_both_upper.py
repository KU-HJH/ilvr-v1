"""Both-upper DIL applies each softplus before its averaging axes."""
import json
from dataclasses import replace

import pytest
import torch

from step_preference.dil_config import uses_entropy_selection, validate_dil_runtime
from step_preference.dil_identity import build_dil_identity, read_dil_identity, validate_dil_resume
from step_preference.dil_reduction import reduce_dil_context_loss
from step_preference.dil_topk_loss import topk_token_terms
from tests.step_preference.test_dil_config import _bce_config
from utils_deepseed import parse_args


BOTH_UPPER = "dil_bce_rf_topk_both_upper"


@pytest.mark.parametrize("percent", [None, 50., 20.])
def test_both_upper_value_and_full_gradients_preserve_original_step_denominator(percent):
    # Catches mean-before-chosen-softplus, either rejected mean-before-softplus,
    # frozen q weights, wrong masks, retained-step denominator and live clean mass.
    config = _bce_config(preference_loss_type=BOTH_UPPER, preference_topk=2,
        preference_entropy_top_percent=percent, beta=1., lambda_ce=.1)
    clean = [torch.tensor([[-.2, .7, 1.2, -1.4], [.5, -2., .1, 1.8],
                           [1.1, .2, -.6, 2.]], requires_grad=True),
             torch.tensor([[.7, -1., .4, 1.2]], requires_grad=True)]
    off = [torch.tensor([[3., 1., -2., 0.], [.2, 2., .8, -1.],
                         [1., -2., .9, .3]], requires_grad=True),
           torch.tensor([[-1., .4, 2., .8]], requires_grad=True)]
    ids = [torch.tensor([0, 1, 2]), torch.tensor([3])]
    gt = [c.log_softmax(-1).gather(-1, i[:, None]).squeeze(-1) for c, i in zip(clean, ids)]
    off_gt = [o.log_softmax(-1).gather(-1, i[:, None]).squeeze(-1) for o, i in zip(off, ids)]
    terms = [topk_token_terms(c, o, i, beta=1., k=2, chunk_size=1)
             for c, o, i in zip(clean, off, ids)]
    ce = torch.tensor(.8, requires_grad=True)
    entropies = [torch.tensor([8., 7., 1.]), torch.tensor([0.])]
    actual, metrics = reduce_dil_context_loss(ce, gt, off_gt, config=config,
        topk_terms=terms, selection_entropies=entropies)
    # Hand-derived sample-level top entropy masks: ceil(4*p/100).
    masks = {None: [[0, 1, 2], [0]], 50.: [[0, 1], []], 20.: [[0], []]}[percent]
    positive, negative, mass, standard_positive, reverse_negative = [], [], [], [], []
    frozen_negative = []
    for c, o, selected, clean_gt in zip(clean, off, masks, gt):
        if not selected:
            continue
        positions = torch.tensor(selected)
        chosen_gt = clean_gt[positions]
        positive.append(torch.log1p(torch.exp(-chosen_gt)).mean())
        standard_positive.append(torch.log1p(torch.exp(-chosen_gt.mean())))
        support = o[positions].detach().argsort(-1, descending=True)[:, :2]
        q = o[positions].gather(-1, support).softmax(-1)
        negative.append((q * torch.log1p(q)).sum(-1).mean())
        reverse_negative.append(torch.log1p(torch.exp((q*q.log()).sum(-1).mean())))
        frozen_negative.append((q.detach() * torch.log1p(q)).sum(-1).mean())
        cm = c[positions].detach().softmax(-1).gather(-1, support).sum(-1)
        om = o[positions].softmax(-1).gather(-1, support).sum(-1)
        mass.append((cm*(cm/om).log() + (1-cm)*((1-cm)/(1-om)).log()).mean())
    want_positive, want_negative, want_mass = sum(positive)/2, sum(negative)/2, sum(mass)/2
    expected = .1*ce + want_positive + want_negative + want_mass
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    for key, want in [("loss/dil_positive", want_positive), ("loss/dil_negative", want_negative),
                      ("loss/topk_mass", want_mass)]:
        torch.testing.assert_close(metrics[key], want, atol=1e-6, rtol=1e-5)
    if percent != 20.:
        assert abs(float(want_positive-sum(standard_positive)/2)) > 1e-4
    assert abs(float(want_negative-sum(reverse_negative)/2)) > 1e-4
    assert metrics["stats/eligible_steps"].item() == 2
    assert metrics["stats/selected_steps"].item() == (2 if percent is None else 1)
    assert metrics["stats/selected_tokens"].item() == sum(map(len, masks))
    parameters = [ce, *clean, *off]
    got = torch.autograd.grad(actual, parameters, retain_graph=True)
    want = torch.autograd.grad(expected, parameters, allow_unused=True, retain_graph=True)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, torch.zeros_like(a) if b is None else b, atol=1e-6, rtol=1e-5)
    rejected_grad = torch.autograd.grad(metrics["loss/dil_negative"], off[0], retain_graph=True)[0]
    frozen_grad = torch.autograd.grad(sum(frozen_negative)/2, off[0], retain_graph=True)[0]
    assert rejected_grad.abs().sum() > 0
    assert not torch.allclose(rejected_grad, frozen_grad, atol=1e-5)
    mass_grad = torch.autograd.grad(metrics["loss/topk_mass"], clean, allow_unused=True, retain_graph=True)
    assert all(g is None or torch.count_nonzero(g) == 0 for g in mass_grad)
    for historical in ("dil_bce_rf_topk", "dil_bce_rf_topk_reverse"):
        _, old = reduce_dil_context_loss(ce, gt, off_gt, config=replace(config, preference_loss_type=historical),
            topk_terms=terms, selection_entropies=entropies)
        torch.testing.assert_close(old["loss/topk_mass"], metrics["loss/topk_mass"], atol=0, rtol=0)


@pytest.mark.parametrize("historical", ["dil_bce_rf_topk", "dil_bce_rf_topk_reverse"])
def test_both_upper_cli_defaults_and_saved_identity_reject_cross_variant_resume(tmp_path, historical):
    # Catches missing CLI admission, ordinary coefficients, missing resume route,
    # shared formula identity, or a checkpoint accepted across variant boundaries.
    args = parse_args([
        "--stage", "stage2", "--step_preference_enabled", "--step_preference_mode", "stage2b",
        "--resume_from_checkpoint_path", str(tmp_path), "--max_steps", "212",
        "--step_shared_prefix_enabled", "--no-step_visual_off_suppress_latent_boundaries",
        "--step_entropy_weight", "0", "--step_preference_weight", "1",
        "--step_preference_loss_type", BOTH_UPPER, "--step_preference_topk", "1516",
        "--step_preference_lambda_topk_mass", "1", "--step_preference_entropy_top_percent", "20",
        "--step_preference_target_scope", "until_next_latent",
    ])
    config = args.step_preference_config
    assert (config.beta, config.preference_margin, config.lambda_ce) == (1., 0., .1)
    assert config.is_dil_resume and uses_entropy_selection(config)
    validate_dil_runtime(config, vocab_size=151668, supervise_latent_boundaries=True)
    with pytest.raises(ValueError, match="margin"):
        replace(config, preference_margin=.5)
    identity = build_dil_identity(config)
    assert identity["formula_version"] == "dil_bce_rf_topk_both_upper_v1"
    (tmp_path / "config.json").write_text(json.dumps({"ilvr_context_preference": identity}))
    assert read_dil_identity(json.loads((tmp_path / "config.json").read_text())) == identity
    validate_dil_resume(tmp_path, config)
    old = replace(config, preference_loss_type=historical)
    with pytest.raises(ValueError, match="identity"):
        validate_dil_resume(tmp_path, old)
    (tmp_path / "config.json").write_text(json.dumps({"ilvr_context_preference": build_dil_identity(old)}))
    with pytest.raises(ValueError, match="identity"):
        validate_dil_resume(tmp_path, config)
