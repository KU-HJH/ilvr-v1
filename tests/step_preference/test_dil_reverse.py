"""Reverse reductions must move softplus across both rejected averaging axes."""
import json
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from step_preference.dil_config import uses_entropy_selection, validate_dil_runtime
from step_preference.dil_identity import build_dil_identity, read_dil_identity, validate_dil_resume
from step_preference.dil_reduction import reduce_dil_context_loss
from step_preference.dil_topk_loss import topk_token_terms
from tests.step_preference.test_dil_config import _bce_config
from utils_deepseed import parse_args


REVERSE = "dil_bce_rf_topk_reverse"


@pytest.mark.parametrize("percent", [None, 50.])
def test_reverse_value_and_live_gradients_follow_both_axis_formula(percent):
    # Catches old chosen reduction, candidate-only softplus, token weighting,
    # detached q, selecting the wrong tokens, and softplus(0) for empty steps.
    config = _bce_config(preference_loss_type=REVERSE, preference_topk=2,
                         preference_entropy_top_percent=percent, beta=.5)
    clean = [torch.tensor([[-.2, .7, 1.2, -1.4], [.5, -2., .1, 1.8],
                           [1.1, .2, -.6, 2.]], requires_grad=True),
             torch.tensor([[.7, -1., .4, 1.2]], requires_grad=True)]
    off = [torch.tensor([[3., 1., -2., 0.], [.2, 2., .8, -1.],
                         [1., -2., .9, .3]], requires_grad=True),
           torch.tensor([[-1., .4, 2., .8]], requires_grad=True)]
    ids = [torch.tensor([0, 1, 2]), torch.tensor([3])]
    gt = [c.log_softmax(-1).gather(-1, i[:, None]).squeeze(-1) for c, i in zip(clean, ids)]
    off_gt = [o.log_softmax(-1).gather(-1, i[:, None]).squeeze(-1) for o, i in zip(off, ids)]
    terms = [topk_token_terms(c, o, i, beta=.5, k=2, chunk_size=1)
             for c, o, i in zip(clean, off, ids)]
    ce = torch.tensor(.8, requires_grad=True)
    entropies = [torch.tensor([8., 7., 1.]), torch.tensor([0.])]
    actual, metrics = reduce_dil_context_loss(ce, gt, off_gt, config=config,
        topk_terms=terms, selection_entropies=entropies)
    # These masks are hand-derived: top50 of four tokens keeps only first two.
    masks = [[0, 1, 2], [0]] if percent is None else [[0, 1], []]
    positive, negative, mass, candidate_only = [], [], [], []
    for c, o, i, selected, clean_gt in zip(clean, off, ids, masks, gt):
        if not selected:
            continue
        positions = torch.tensor(selected)
        chosen_gt = clean_gt[positions]
        positive.append(torch.log1p(torch.exp(-.5 * chosen_gt)).mean())
        # Independent softmax / log expression retains the q weighting gradient.
        support = o[positions].detach().argsort(-1, descending=True)[:, :2]
        q = o[positions].gather(-1, support).softmax(-1)
        expected_logq = (q * q.log()).sum(-1)
        negative.append(torch.log1p(torch.exp(.5 * expected_logq.mean())))
        candidate_only.append(torch.log1p(torch.exp(.5 * expected_logq)).mean())
        cm = c[positions].detach().softmax(-1).gather(-1, support).sum(-1)
        om = o[positions].softmax(-1).gather(-1, support).sum(-1)
        mass.append((cm * (cm/om).log() + (1-cm) * ((1-cm)/(1-om)).log()).mean())
    want_positive = sum(positive)/2
    want_negative = sum(negative)/2
    want_mass = sum(mass)/2
    expected = ce + want_positive + want_negative + want_mass
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(metrics["loss/dil_positive"], want_positive)
    torch.testing.assert_close(metrics["loss/dil_negative"], want_negative)
    torch.testing.assert_close(metrics["loss/topk_mass"], want_mass, atol=1e-6, rtol=1e-5)
    assert abs(float(want_negative - sum(candidate_only)/2)) > 1e-4
    assert metrics["stats/eligible_steps"].item() == 2
    assert metrics["stats/selected_steps"].item() == (2 if percent is None else 1)
    parameters = [ce, *clean, *off]
    got = torch.autograd.grad(actual, parameters, retain_graph=True)
    want = torch.autograd.grad(expected, parameters, allow_unused=True, retain_graph=True)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, torch.zeros_like(a) if b is None else b, atol=1e-6, rtol=1e-5)
    # Compare just rejected gradients: freezing q changes this derivative.
    negative_grad = torch.autograd.grad(metrics["loss/dil_negative"], off[0], retain_graph=True)[0]
    positions = torch.tensor(masks[0])
    support = off[0][positions].detach().argsort(-1, descending=True)[:, :2]
    logq = off[0][positions].gather(-1, support).log_softmax(-1)
    frozen_q_loss = F.softplus(.5 * (logq.exp().detach()*logq).sum(-1).mean())/2
    frozen_grad = torch.autograd.grad(frozen_q_loss, off[0])[0]
    assert negative_grad.abs().sum() > 0
    assert not torch.allclose(negative_grad, frozen_grad, atol=1e-5)


def test_reverse_parser_checkpoint_roundtrip_rejects_original_topk_identity(tmp_path):
    # Catches dropped selector, wrong resume gate, or sharing old formula identity.
    args = parse_args([
        "--stage", "stage2", "--step_preference_enabled", "--step_preference_mode", "stage2b",
        "--resume_from_checkpoint_path", str(tmp_path), "--max_steps", "212",
        "--step_shared_prefix_enabled", "--no-step_visual_off_suppress_latent_boundaries",
        "--step_entropy_weight", "0", "--step_preference_beta", ".5",
        "--step_preference_weight", "1", "--step_preference_loss_type", REVERSE,
        "--step_preference_topk", "1516", "--step_preference_lambda_topk_mass", "1",
        "--step_preference_entropy_top_percent", "20", "--step_preference_target_scope", "until_next_latent",
    ])
    config = args.step_preference_config
    assert config.is_dil_resume and uses_entropy_selection(config)
    validate_dil_runtime(config, vocab_size=151668, supervise_latent_boundaries=True)
    with pytest.raises(ValueError, match="vocab|topk"):
        validate_dil_runtime(config, vocab_size=1516, supervise_latent_boundaries=True)
    identity = build_dil_identity(config)
    assert identity["loss_type"] == REVERSE
    assert identity["formula_version"] == "dil_bce_rf_topk_reverse_v1"
    (tmp_path / "config.json").write_text(json.dumps({"ilvr_context_preference": identity}))
    assert read_dil_identity(json.loads((tmp_path / "config.json").read_text())) == identity
    validate_dil_resume(tmp_path, config)
    original = replace(config, preference_loss_type="dil_bce_rf_topk")
    with pytest.raises(ValueError, match="identity"):
        validate_dil_resume(tmp_path, original)
    (tmp_path / "config.json").write_text(json.dumps({"ilvr_context_preference": build_dil_identity(original)}))
    with pytest.raises(ValueError, match="identity"):
        validate_dil_resume(tmp_path, config)
