"""Real tiny forwards compared with independent full-vocabulary objectives."""
import copy
import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from step_preference.equivalence import _trainer
from step_preference.shared_prefix_measurement import tiny_measurement_case
from tests.step_preference.test_context_preference import context_config


def dil_config(kind="dil_bce_rf_topk", **kwargs):
    values = dict(preference_loss_type=kind, beta=.5, lambda_entropy=0.,
                  visual_only_context_preference=True, shared_prefix_enabled=True, visual_off_suppress_latent_boundaries=False)
    if kind in ("dil_bce_rf_topk", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_gt"):
        values.update(preference_topk=7, lambda_topk_mass=.7)
    if kind == "dil_bce_rf_topk_gt":
        values.update(beta=1., lambda_ce=.1, lambda_topk_mass=1.)
    values.update(kwargs)
    return context_config(**values)


def dil_case(steps=2, unequal=True):
    model, batch = tiny_measurement_case(steps=steps, total_target_tokens=3*steps, device="cpu")
    model.config.supervise_latent_boundaries = True
    spans = list(batch["reasoning_step_spans"])
    if unequal:
        spans = [replace(s, target_text_end=s.target_text_start + min(i+1, 3)) for i, s in enumerate(spans)]
        spans = [replace(s, prev_text_end=spans[i-1].target_text_end) if i else s for i, s in enumerate(spans)]
    batch["reasoning_step_spans"] = tuple(spans)
    batch["step_preference"] = {"steps": tuple({"span": s} for s in spans)}
    return model, batch


def observed_forward(model, batch, config):
    projections = []
    hook = model.lm_head.register_forward_hook(lambda _m, _a, out: projections.append(out))
    try:
        loss, output = _trainer(config).compute_loss(model, batch, return_outputs=True)
    finally:
        hook.remove()
    return loss, output, projections[-1][0] if len(projections) > 1 else None


def full_logit_oracle(output, off, batch, config):
    spans = batch["reasoning_step_spans"]
    if not spans:
        return config.lambda_ce * output.loss
    clean = output.stage2_selected_logits[0].float()
    clean_lp, off_lp = clean.log_softmax(-1), off.float().log_softmax(-1)
    ent_source = clean_lp if config.preference_entropy_source == "clean" else off_lp
    entropy = -(ent_source.exp() * ent_source).sum(-1)
    chosen = set(range(clean.shape[0]))
    if config.preference_entropy_top_percent is not None and config.preference_entropy_top_percent < 100:
        chosen = set(sorted(chosen, key=lambda i: (-float(entropy[i].detach()), i))[
            :math.ceil(len(chosen)*config.preference_entropy_top_percent/100)])
    total = config.lambda_ce * output.loss
    cursor = 0
    for span in spans:
        size = span.target_text_end-span.target_text_start
        indices = [i for i in range(cursor, cursor+size) if i in chosen]
        if indices:
            ids = batch["input_ids"][0, span.target_text_start:span.target_text_end]
            local = [i-cursor for i in indices]
            positive = F.softplus(-config.beta*clean_lp[indices].gather(1, ids[local, None]).mean())
            if config.preference_loss_type == "dil_bce_rf_topk_reverse":
                positive = F.softplus(-config.beta*clean_lp[indices].gather(1, ids[local, None])).mean()
            if config.preference_loss_type == "dil_bce_rf":
                negative = F.softplus(config.beta*off_lp[indices].gather(1, ids[local, None]).mean())
                mass = total*0
            elif config.preference_loss_type == "dil_bce_rf_topk_gt":
                gt_logq, masses = [], []
                for index, target in zip(indices, ids[local]):
                    # Independent Python union; no production support or loss helper.
                    support_ids = set(off[index].detach().argsort(descending=True)[:config.preference_topk].tolist())
                    support_ids.add(int(target))
                    support = sorted(support_ids)
                    mc = clean_lp[index].detach().exp()[support].sum()
                    mo = off_lp[index].exp()[support].sum()
                    gt_logq.append((off_lp[index, target].exp() / mo).log())
                    masses.append(mc * (mc / mo).log() + (1-mc) * ((1-mc)/(1-mo)).log())
                negative = F.softplus(config.beta * torch.stack(gt_logq).mean())
                mass = torch.stack(masses).mean()
            else:
                support = off[indices].detach().topk(config.preference_topk, -1).indices
                q = off[indices].float().gather(-1, support).softmax(-1)
                negative = (q * F.softplus(config.beta*q.log())).sum(-1).mean()
                if config.preference_loss_type == "dil_bce_rf_topk_reverse":
                    negative = F.softplus(config.beta*(q*q.log()).sum(-1).mean())
                mc = clean_lp[indices].detach().exp().gather(-1, support).sum(-1)
                mo = off_lp[indices].exp().gather(-1, support).sum(-1)
                mass = (mc*(mc/mo).log() + (1-mc)*((1-mc)/(1-mo)).log()).mean()
            total = total + (config.lambda_preference*(positive+negative)+config.lambda_topk_mass*mass)/len(spans)
        cursor += size
    return total


def compare_grads(actual, expected, model, *, exact=False):
    parameters = [p for p in model.parameters() if p.requires_grad]
    got = torch.autograd.grad(actual, parameters, retain_graph=True, allow_unused=True)
    want = torch.autograd.grad(expected, parameters, retain_graph=True, allow_unused=True)
    for a, b in zip(got, want):
        if b is not None:
            assert a is not None
            torch.testing.assert_close(a, b, atol=0 if exact else 1e-5, rtol=0 if exact else 1e-4)
    return got


@pytest.mark.parametrize("config", [dil_config("dil_bce_rf"), dil_config(),
    dil_config("dil_bce_rf_topk_reverse"),
    dil_config("dil_bce_rf_topk_reverse", preference_entropy_top_percent=20),
    dil_config("dil_bce_rf_topk_reverse", preference_entropy_top_percent=20, preference_entropy_source="voff"),
    dil_config(preference_entropy_top_percent=20),
    dil_config(preference_entropy_top_percent=20, preference_entropy_source="voff"),
    dil_config(lambda_preference=0., lambda_topk_mass=2.),
    dil_config("dil_bce_rf_topk_gt"),
    dil_config("dil_bce_rf_topk_gt", lambda_preference=0.),
    dil_config("dil_bce_rf_topk_gt", preference_entropy_top_percent=20),
    dil_config("dil_bce_rf_topk_gt", preference_entropy_top_percent=20, preference_entropy_source="voff")])
def test_actual_unequal_step_loss_and_parameter_gradients_match_full_logits(config):
    # Catches DPO dispatch, token-weighted steps, detached q, and coupling mass to lambdaP.
    model, batch = dil_case()
    actual, output, off = observed_forward(model, batch, config)
    oracle = full_logit_oracle(output, off, batch, config)
    assert actual.requires_grad
    torch.testing.assert_close(actual, oracle, atol=1e-6, rtol=1e-5)
    compare_grads(actual, oracle, model)


@pytest.mark.parametrize("kind", ["dil_bce_rf", "dil_bce_rf_topk", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_gt"])
@pytest.mark.parametrize("padding", [False, True])
def test_direct_trainer_validates_resolved_boundaries_before_forward(kind, padding):
    model, batch = dil_case()
    model.config.supervise_latent_boundaries = False
    trainer = _trainer(dil_config(kind))
    trainer.stage2_sampler_policy = "reference_random"
    if padding:
        batch["stage2_sample_weight"] = 0.
    calls = []
    hook = model.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        with pytest.raises(ValueError, match="supervised latent boundaries"):
            trainer.compute_loss(model, batch)
    finally:
        hook.remove()
    assert not calls


@pytest.mark.parametrize("kind", ["dil_bce_rf_topk", "dil_bce_rf_topk_gt"])
def test_direct_trainer_uses_output_head_vocabulary_before_empty_forward(kind):
    model, batch = dil_case()
    model.lm_head = torch.nn.Linear(model.config.hidden_size, 5, bias=False)
    batch["reasoning_step_spans"] = ()
    batch["step_preference"] = {"steps": ()}
    calls = []
    hook = model.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        with pytest.raises(ValueError, match="smaller than output vocab"):
            _trainer(dil_config(kind)).compute_loss(model, batch)
    finally:
        hook.remove()
    assert not calls


@pytest.mark.parametrize("config", [dil_config("dil_bce_rf"), dil_config(), dil_config(preference_entropy_top_percent=20),
    dil_config("dil_bce_rf_topk_reverse"), dil_config("dil_bce_rf_topk_gt")])
def test_empty_sample_keeps_clean_ce(config):
    model, batch = dil_case()
    batch["reasoning_step_spans"] = ()
    batch["step_preference"] = {"steps": ()}
    actual, out, _ = observed_forward(model, batch, config)
    torch.testing.assert_close(actual, config.lambda_ce*out.loss, atol=0, rtol=0)
    compare_grads(actual, config.lambda_ce*out.loss, model, exact=True)
