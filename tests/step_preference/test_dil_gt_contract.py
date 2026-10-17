"""GT DIL objective reduction, entry defaults, and saved formula contracts."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from step_preference.config import StepPreferenceConfig
from step_preference.dil_config import validate_dil_config, validate_dil_runtime
from step_preference.dil_identity import build_dil_identity, read_dil_identity, validate_dil_resume
from step_preference.dil_reduction import reduce_dil_context_loss
from step_preference.dil_topk_loss import TopkGtTokenResult, TopkTokenResult
from tests.step_preference.test_dil_config import _bce_config
from tests.step_preference.test_loss_defaults import cli
from utils_deepseed import parse_args


GT = "dil_bce_rf_topk_gt"


def gt_config(**overrides):
    return _bce_config(**(dict(preference_loss_type=GT, preference_topk=2,
                               beta=1., lambda_ce=.1, lambda_topk_mass=1.) | overrides))


def reduction_fixture():
    clean = [torch.tensor(v, requires_grad=True) for v in ([-.1, -4., -2.], [-.7])]
    off = [torch.tensor(v, requires_grad=True) for v in ([-9., -8., -7.], [-6.])]
    logq = [torch.tensor(v, requires_grad=True) for v in ([-.2, -3., -.5], [-.6])]
    mass = [torch.tensor(v, requires_grad=True) for v in ([.2, .4, .8], [.9])]
    terms = [TopkGtTokenResult(q, m, *(torch.zeros_like(q) for _ in range(5)))
             for q, m in zip(logq, mass)]
    ce = torch.tensor(2., requires_grad=True)
    entropies = [torch.tensor([8., 7., 1.]), torch.tensor([0.])]
    return ce, clean, off, logq, mass, terms, entropies


def test_gt_reduction_empty_selected_step():
    # Breaks: tokenwise softplus, old negative loss, removing an empty step,
    # selecting conditional entropy, or replacing the full-vocabulary OFF metric.
    ce, clean, off, logq, mass, terms, entropies = reduction_fixture()
    total, metrics = reduce_dil_context_loss(ce, clean, off,
        config=gt_config(preference_entropy_top_percent=50.), topk_terms=terms,
        selection_entropies=entropies)
    want_pos = F.softplus(torch.tensor(2.05)) / 2
    want_neg = F.softplus(torch.tensor(-1.6)) / 2
    want_mass = torch.tensor(.15)
    torch.testing.assert_close(metrics["loss/dil_positive"], want_pos)
    torch.testing.assert_close(metrics["loss/dil_negative"], want_neg)
    torch.testing.assert_close(metrics["loss/topk_mass"], want_mass)
    torch.testing.assert_close(metrics["stats/topk_gt_logq"], torch.tensor(-.8))
    torch.testing.assert_close(metrics["stats/logp_visual_off"], torch.tensor(-4.25))
    torch.testing.assert_close(total, .1*ce + want_pos + want_neg + want_mass)
    assert metrics["stats/selected_steps"].item() == 1
    assert metrics["stats/eligible_steps"].item() == 2
    assert metrics["stats/selected_tokens"].item() == 2
    # Independently hand-select first two leaves and retain the two-step divisor.
    expected = .1*ce + F.softplus(-(clean[0][0]+clean[0][1])/2)/2
    expected = expected + F.softplus((logq[0][0]+logq[0][1])/2)/2
    expected = expected + (mass[0][0]+mass[0][1])/4
    leaves = [ce, *clean, *off, *logq, *mass]
    actual_grads = torch.autograd.grad(total, leaves, allow_unused=True, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, leaves, allow_unused=True, retain_graph=True)
    for leaf, actual, expected_grad in zip(leaves, actual_grads, expected_grads):
        torch.testing.assert_close(torch.zeros_like(leaf) if actual is None else actual,
                                   torch.zeros_like(leaf) if expected_grad is None else expected_grad)
    for grad in actual_grads[1:]:
        if grad is not None:
            assert grad[-1].item() == 0.  # discarded last token or wholly empty step
    # The entirely discarded clean, OFF, conditional and mass leaves participate.
    assert all(actual_grads[index] is not None for index in (2, 4, 6, 8))
    stat_grads = torch.autograd.grad(metrics["stats/topk_gt_logq"], logq)
    torch.testing.assert_close(stat_grads[0], torch.tensor([.25, .25, 0.]))
    torch.testing.assert_close(stat_grads[1], torch.zeros(1))


@pytest.mark.parametrize("preference,mass_weight", [(1., 0.), (0., 1.), (1., 1.)])
def test_gt_independent_coefficients_and_empty_sample(preference, mass_weight):
    # Breaks: mass multiplied by lambda_P or CE unweighted/double weighted.
    ce, clean, off, _, _, terms, _ = reduction_fixture()
    cfg = gt_config(lambda_preference=preference, lambda_topk_mass=mass_weight)
    total, metrics = reduce_dil_context_loss(ce, clean, off, config=cfg, topk_terms=terms)
    want_positive = (F.softplus(torch.tensor(6.1/3)) + F.softplus(torch.tensor(.7)))/2
    want_negative = (F.softplus(torch.tensor(-3.7/3)) + F.softplus(torch.tensor(-.6)))/2
    want_mass = torch.tensor(((.2+.4+.8)/3 + .9)/2)
    torch.testing.assert_close(metrics["loss/dil_positive"], want_positive)
    torch.testing.assert_close(metrics["loss/dil_negative"], want_negative)
    torch.testing.assert_close(total, .1*ce + preference*(want_positive+want_negative) + mass_weight*want_mass)
    torch.testing.assert_close(torch.autograd.grad(total, ce)[0], torch.tensor(.1))
    empty, empty_metrics = reduce_dil_context_loss(ce, [], [], config=cfg, topk_terms=[])
    torch.testing.assert_close(empty, .1*ce)
    torch.testing.assert_close(empty_metrics["stats/topk_gt_logq"], torch.tensor(0.))


def test_gt_config_defaults_and_invalid_contract():
    # Breaks: CLI drops GT, ordinary beta/margin leaks, or invalid settings pass.
    for mass_args, mass_value in (([], 0.), (["--step_preference_lambda_topk_mass", "1"], 1.)):
        parsed = parse_args(cli(GT) + mass_args)
        direct = gt_config(beta=None, lambda_ce=None, lambda_topk_mass=mass_value)
        for cfg in (parsed.step_preference_config, direct, StepPreferenceConfig.from_args(SimpleNamespace(**vars(parsed)))):
            assert (cfg.beta, cfg.preference_margin, cfg.lambda_ce, cfg.lambda_preference, cfg.lambda_topk_mass) == (1., 0., .1, 1., mass_value)
            validate_dil_config(cfg)
            validate_dil_runtime(cfg, vocab_size=16, supervise_latent_boundaries=True)
    for scope in ("immediate_text", "until_next_latent"):
        validate_dil_runtime(gt_config(target_scope=scope), vocab_size=4, supervise_latent_boundaries=True)
    for updates in ({"preference_margin": .1}, {"lambda_entropy": .1},
                    {"preference_topk": None}, {"preference_topk": 1}, {"preference_topk": True},
                    {"prefix_mode": "v_off_prefix"}, {"shared_prefix_enabled": False}):
        with pytest.raises(ValueError):
            gt_config(**updates)
    with pytest.raises(ValueError, match="vocab|topk"):
        validate_dil_runtime(gt_config(), vocab_size=2, supervise_latent_boundaries=True)


def save_identity(path, config):
    path.mkdir(parents=True, exist_ok=True)
    identity = build_dil_identity(config)
    (path / "config.json").write_text(json.dumps({"ilvr_context_preference": identity}))
    return identity


def test_gt_identity_resume_roundtrip(tmp_path):
    # Breaks: GT adopts historical formula or cross-selector resume is allowed.
    cfg = gt_config()
    identity = save_identity(tmp_path / "gt", cfg)
    assert identity["formula_version"] == "dil_bce_rf_topk_gt_v1"
    assert read_dil_identity(json.loads((tmp_path / "gt/config.json").read_text())) == identity
    validate_dil_resume(tmp_path / "gt", cfg)
    for selector in ("dil_bce_rf_topk", "dil_bce_rf_topk_reverse"):
        old = replace(cfg, preference_loss_type=selector)
        save_identity(tmp_path / selector, old)
        with pytest.raises(ValueError, match="identity"):
            validate_dil_resume(tmp_path / selector, cfg)
        with pytest.raises(ValueError, match="identity"):
            validate_dil_resume(tmp_path / "gt", old)
    corrupted = {**identity, "formula_version": "dil_bce_rf_topk_v1"}
    with pytest.raises(ValueError, match="identity"):
        read_dil_identity({"ilvr_context_preference": corrupted})
    (tmp_path / "gt/config.json").write_text(json.dumps({"ilvr_context_preference": corrupted}))
    with pytest.raises(ValueError, match="identity"):
        validate_dil_resume(tmp_path / "gt", cfg)


def test_gt_resume_omission_retains_saved_coefficients(tmp_path):
    # Breaks: fresh CE/beta replace omitted saved settings during real CLI resolution.
    save_identity(tmp_path, gt_config(beta=.5, lambda_ce=1.))
    argv = cli(GT)
    argv.remove("--stage2b_direct_from_stage1")
    adapter = argv.index("--stage1_adapter_path")
    del argv[adapter:adapter+2]
    argv += ["--resume_from_checkpoint_path", str(tmp_path), "--max_steps", "212",
             "--step_preference_lambda_topk_mass", "1"]
    k = argv.index("--step_preference_topk")
    argv[k+1] = "2"
    parsed = parse_args(argv).step_preference_config
    direct = gt_config(beta=None, lambda_ce=None, direct_from_stage1=False,
                       resume_from_checkpoint_path=str(tmp_path))
    for cfg in (parsed, direct):
        assert cfg.is_dil_resume
        assert (cfg.beta, cfg.lambda_ce, cfg.preference_margin, cfg.lambda_topk_mass) == (.5, 1., 0., 1.)
        validate_dil_resume(tmp_path, cfg)
    for explicit in (["--ce_weight", ".1"], ["--step_preference_beta", "1"]):
        mismatch = parse_args(argv + explicit).step_preference_config
        with pytest.raises(ValueError, match="identity"):
            validate_dil_resume(tmp_path, mismatch)


def test_gt_statistics_type_and_shape_validation():
    # Breaks: reinterpretation of NamedTuple position zero across distinct formulas.
    ce, clean, off, _, _, terms, _ = reduction_fixture()
    old_terms = [TopkTokenResult(*term) for term in terms]
    with pytest.raises(ValueError, match="statistics"):
        reduce_dil_context_loss(ce, clean, off, config=gt_config(), topk_terms=old_terms)
    for selector in ("dil_bce_rf_topk", "dil_bce_rf_topk_reverse"):
        with pytest.raises(ValueError, match="statistics"):
            reduce_dil_context_loss(ce, clean, off,
                config=gt_config(preference_loss_type=selector), topk_terms=terms)
    wrong = [terms[0]._replace(gt_logq=torch.zeros(2)), terms[1]]
    with pytest.raises(ValueError, match="statistics"):
        reduce_dil_context_loss(ce, clean, off, config=gt_config(), topk_terms=wrong)
