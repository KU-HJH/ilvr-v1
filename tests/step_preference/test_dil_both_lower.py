"""Both-mean DIL: independent CUDA value/gradient witnesses and host contracts."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from step_preference.config import StepPreferenceConfig
from step_preference.dil_config import uses_entropy_selection, validate_dil_runtime
from step_preference.dil_identity import attach_dil_identity, build_dil_identity, read_dil_identity, validate_dil_resume
from step_preference.dil_reduction import reduce_dil_context_loss
from step_preference.dil_topk_loss import topk_token_terms
from tests.step_preference.test_dil_config import _bce_config
from utils_deepseed import parse_args

LOWER = "dil_bce_rf_topk_both_lower"


@pytest.mark.parametrize("percent", [None, 50., 20.])
@pytest.mark.parametrize("beta", [1., .5])
def test_cuda_both_means_precede_step_softplus_with_live_q_and_fixed_denominator(percent, beta):
    # Breaks on either tokenwise softplus, frozen q, cross-step softplus,
    # wrong entropy mask, surviving-step normalization or clean mass gradient.
    config = _bce_config(preference_loss_type=LOWER, preference_topk=2,
        preference_entropy_top_percent=percent, beta=beta, lambda_ce=.1)
    clean = [torch.tensor([[-.2, .7, 1.2, -1.4], [.5, -2., .1, 1.8],
                           [1.1, .2, -.6, 2.]], device="cuda", requires_grad=True),
             torch.tensor([[.7, -1., .4, 1.2]], device="cuda", requires_grad=True)]
    off = [torch.tensor([[3., 1., -2., 0.], [.2, 2., .8, -1.],
                         [1., -2., .9, .3]], device="cuda", requires_grad=True),
           torch.tensor([[-1., .4, 8., .8]], device="cuda", requires_grad=True)]
    ids = [torch.tensor([0, 1, 2], device="cuda"), torch.tensor([3], device="cuda")]
    gt = [c.log_softmax(-1).gather(-1, i[:, None]).squeeze(-1) for c, i in zip(clean, ids)]
    off_gt = [o.log_softmax(-1).gather(-1, i[:, None]).squeeze(-1) for o, i in zip(off, ids)]
    terms = [topk_token_terms(c, o, i, beta=beta, k=2, chunk_size=1)
             for c, o, i in zip(clean, off, ids)]
    ce = torch.tensor(.8, device="cuda", requires_grad=True)
    entropies = [torch.tensor([8., 7., 1.], device="cuda"), torch.tensor([0.], device="cuda")]
    actual, metrics = reduce_dil_context_loss(ce, gt, off_gt, config=config,
        topk_terms=terms, selection_entropies=entropies)
    # Hand-derived sample masks from ceil(4*p/100); the denominator stays two.
    masks = {None: [[0, 1, 2], [0]], 50.: [[0, 1], []], 20.: [[0], []]}[percent]
    positive, negative, mass, upper_positive, upper_negative = [], [], [], [], []
    token_negative, frozen_negative, chosen_scores, rejected_scores = [], [], [], []
    for c, o, selected, target_ids in zip(clean, off, masks, ids):
        if not selected:
            continue
        positions = torch.tensor(selected, device="cuda")
        chosen_gt = c.double().log_softmax(-1).gather(-1, target_ids[:, None]).squeeze(-1)[positions]
        chosen_scores.append(chosen_gt.mean())
        positive.append(torch.log1p(torch.exp(-beta * chosen_gt.mean())))
        upper_positive.append(torch.log1p(torch.exp(-beta * chosen_gt)).mean())
        support = o[positions].detach().argsort(-1, descending=True)[:, :2]
        q = o[positions].double().gather(-1, support).softmax(-1)
        expectation = (q * q.log()).sum(-1)
        rejected_scores.append(expectation.mean())
        negative.append(torch.log1p(torch.exp(beta * expectation.mean())))
        upper_negative.append((q * torch.log1p(torch.exp(beta * q.log()))).sum(-1).mean())
        token_negative.append(torch.log1p(torch.exp(beta * expectation)).mean())
        frozen_negative.append(torch.log1p(torch.exp(beta * (q.detach() * q.log()).sum(-1).mean())))
        cm = c[positions].detach().double().softmax(-1).gather(-1, support).sum(-1)
        om = o[positions].double().softmax(-1).gather(-1, support).sum(-1)
        mass.append((cm * (cm/om).log() + (1-cm) * ((1-cm)/(1-om)).log()).mean())
    want_positive, want_negative, want_mass = sum(positive)/2, sum(negative)/2, sum(mass)/2
    expected = .1 * ce + want_positive + want_negative + want_mass
    torch.testing.assert_close(actual, expected.float(), atol=1e-6, rtol=1e-5)
    for key, want in [("loss/dil_positive", want_positive), ("loss/dil_negative", want_negative),
                      ("loss/topk_mass", want_mass)]:
        torch.testing.assert_close(metrics[key], want.float(), atol=1e-6, rtol=1e-5)
    if percent != 20.:
        assert abs(float(want_positive - sum(upper_positive)/2)) > 1e-4
        assert abs(float(want_negative - sum(token_negative)/2)) > 1e-4
    assert abs(float(want_negative - sum(upper_negative)/2)) > 1e-4
    if percent is None:
        assert abs(float(want_positive - F.softplus(-beta * torch.stack(chosen_scores).mean()))) > 1e-4
        assert abs(float(want_negative - F.softplus(beta * torch.stack(rejected_scores).mean()))) > 1e-4
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
    assert rejected_grad.is_cuda and rejected_grad.abs().sum() > 0
    assert not torch.allclose(rejected_grad, frozen_grad, atol=1e-5)
    mass_grad = torch.autograd.grad(metrics["loss/topk_mass"], clean, allow_unused=True, retain_graph=True)
    assert all(g is None or torch.count_nonzero(g) == 0 for g in mass_grad)
    for historical in ("dil_bce_rf_topk", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_both_upper"):
        _, old = reduce_dil_context_loss(ce, gt, off_gt, config=replace(config, preference_loss_type=historical),
            topk_terms=terms, selection_entropies=entropies)
        torch.testing.assert_close(old["loss/topk_mass"], metrics["loss/topk_mass"], atol=0, rtol=0)


def lower_cli(resume=None):
    argv = ["--stage", "stage2", "--step_preference_enabled", "--step_preference_mode", "stage2b",
        "--max_steps", "212", "--step_shared_prefix_enabled", "--no-step_visual_off_suppress_latent_boundaries",
        "--step_entropy_weight", "0", "--step_preference_weight", "1", "--step_preference_loss_type", LOWER,
        "--step_preference_topk", "1516", "--step_preference_lambda_topk_mass", "1",
        "--step_preference_entropy_top_percent", "20", "--step_preference_target_scope", "until_next_latent"]
    return argv + (["--resume_from_checkpoint_path", str(resume)] if resume is not None else
        ["--stage2b_direct_from_stage1", "--stage1_adapter_path", "stage1"])


@pytest.mark.parametrize("historical", ["dil_bce_rf_topk", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_both_upper"])
def test_host_cli_defaults_saved_identity_and_cross_variant_resume(tmp_path, historical):
    # Missing admission/default resolution/identity route could resume the wrong objective.
    fresh = parse_args(lower_cli()).step_preference_config
    config = parse_args(lower_cli(tmp_path)).step_preference_config
    direct = _bce_config(preference_loss_type=LOWER, preference_topk=1516, lambda_ce=None,
        target_scope="until_next_latent", preference_entropy_top_percent=20.)
    for c in (fresh, config, direct, StepPreferenceConfig.from_args(SimpleNamespace(**vars(parse_args(lower_cli()))))):
        assert (c.beta, c.preference_margin, c.lambda_ce) == (1., 0., .1)
        assert uses_entropy_selection(c)
    assert config.is_dil_resume
    validate_dil_runtime(config, vocab_size=151668, supervise_latent_boundaries=True)
    with pytest.raises(ValueError, match="margin"):
        replace(config, preference_margin=.5)
    identity = build_dil_identity(config)
    assert identity["formula_version"] == "dil_bce_rf_topk_both_lower_v1"
    # Exercise actual attach -> serialized configuration -> read/resume, without weights.
    model = SimpleNamespace(config=SimpleNamespace(), model=SimpleNamespace(), lm_head=SimpleNamespace())
    attach_dil_identity(model, config)
    (tmp_path / "config.json").write_text(json.dumps(vars(model.config)))
    assert read_dil_identity(json.loads((tmp_path / "config.json").read_text())) == identity
    validate_dil_resume(tmp_path, config)
    old = replace(config, preference_loss_type=historical)
    with pytest.raises(ValueError, match="identity"):
        validate_dil_resume(tmp_path, old)
    (tmp_path / "config.json").write_text(json.dumps({"ilvr_context_preference": build_dil_identity(old)}))
    with pytest.raises(ValueError, match="identity"):
        validate_dil_resume(tmp_path, config)


@pytest.mark.parametrize("resume", [False, True])
def test_host_both_worker_shell_commands_preserve_profile_and_resume(tmp_path, resume):
    # Wrong shell selector/default/resume route loses the requested two-node profile.
    from tests.multinode.test_dil_forwarding import worker_commands
    settings = dict(STEP_PREFERENCE_LOSS_TYPE=LOWER, STEP_PREFERENCE_TOPK="1516",
        STEP_PREFERENCE_LAMBDA_TOPK_MASS="1", STEP_PREFERENCE_ENTROPY_TOP_PERCENT="20",
        STEP_PREFERENCE_ENTROPY_SOURCE="clean", STEP_PREFERENCE_BETA="", CE_WEIGHT="",
        STEP_PREFERENCE_TARGET_SCOPE="until_next_latent", STAGE2_SAMPLER_POLICY="historical_balanced_2rank")
    if resume:
        saved_config = parse_args(lower_cli(tmp_path)).step_preference_config
        (tmp_path / "config.json").write_text(json.dumps({"ilvr_context_preference": build_dil_identity(saved_config)}))
        settings.update(STAGE2B_DIRECT_FROM_STAGE1="false", STAGE1_ADAPTER_PATH="", RESUME_CHECKPOINT=str(tmp_path))
    arguments = ("--max_steps", "212") + (() if resume else ("--stage2_stop_after_checkpoint", "2"))
    for rank, _, result in worker_commands(settings, arguments):
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        expected = {"--step_preference_loss_type": LOWER, "--step_preference_topk": "1516",
            "--step_preference_beta": "1.0", "--step_preference_margin": "0.0", "--ce_weight": "0.1",
            "--step_preference_weight": "1.0", "--step_preference_lambda_topk_mass": "1",
            "--step_preference_entropy_top_percent": "20", "--step_preference_entropy_source": "clean",
            "--step_preference_target_scope": "until_next_latent", "--stage2_sampler_policy": "historical_balanced_2rank",
            "--num_machines": "2", "--num_processes": "4", "--machine_rank": str(rank),
            "--gradient_accumulation_steps": "4", "--max_steps": "212"}
        for key, value in expected.items():
            assert argv[argv.index(key)+1] == value
        if resume:
            assert argv[argv.index("--resume_from_checkpoint_path")+1] == str(tmp_path)
            assert "--stage2b_direct_from_stage1" not in argv
            assert "--stage1_adapter_path" not in argv
        else:
            assert argv[argv.index("--stage2_stop_after_checkpoint")+1] == "2"
            entry = next(i for i, arg in enumerate(argv) if arg.endswith("/main.py"))
            parsed = parse_args(argv[entry+1:])
            assert parsed.stage2_stop_after_checkpoint == 2
            assert parsed.step_preference_config.target_scope == "until_next_latent"


@pytest.mark.parametrize("percent", [None, 20.])
def test_cuda_actual_trainer_sidecar_matches_full_logit_loss_and_gradients(percent):
    # Real forward/recomputed checkpoint gradients catch missing trainer or sidecar admission.
    from step_preference.shared_prefix_measurement import tiny_measurement_case
    from tests.step_preference.test_dil_trainer import observed_forward, compare_grads
    config = _bce_config(preference_loss_type=LOWER, preference_topk=7, beta=1., lambda_ce=.1,
        target_scope="until_next_latent", preference_entropy_top_percent=percent, sidecar_attention_backend="dense")
    with torch.device("cuda:0"):
        model, batch = tiny_measurement_case(steps=2, total_target_tokens=6, device="cuda:0")
    model.config.supervise_latent_boundaries = True
    spans = tuple(replace(s, target_text_end=s.target_text_start+i+1) for i,s in enumerate(batch["reasoning_step_spans"]))
    spans = tuple(replace(s, prev_text_end=spans[i-1].target_text_end) if i else s for i,s in enumerate(spans))
    batch["reasoning_step_spans"] = spans
    batch["step_preference"] = {"steps": tuple({"span": s} for s in spans)}
    actual, output, off = observed_forward(model, batch, config)
    assert actual.is_cuda and off.is_cuda
    clean = output.stage2_selected_logits[0].float()
    lp = clean.log_softmax(-1)
    off_lp = off.float().log_softmax(-1)
    entropy = -(lp.exp()*lp).sum(-1)
    selected = set(range(len(clean))) if percent is None else {max(range(len(clean)), key=lambda i: float(entropy[i].detach()))}
    expected = .1*output.loss
    cursor = 0
    for span in spans:
        size = span.target_text_end-span.target_text_start
        positions = [i for i in range(cursor,cursor+size) if i in selected]
        if positions:
            ids = batch["input_ids"][0,span.target_text_start:span.target_text_end]
            local = [i-cursor for i in positions]
            chosen = torch.log1p(torch.exp(-lp[positions].gather(-1,ids[local,None]).mean()))
            support = off[positions].detach().argsort(-1,descending=True)[:,:7]
            q = off[positions].float().gather(-1,support).softmax(-1)
            rejected = torch.log1p(torch.exp((q*q.log()).sum(-1).mean()))
            cm = lp[positions].detach().exp().gather(-1,support).sum(-1)
            om = off_lp[positions].exp().gather(-1,support).sum(-1)
            mass = (cm*(cm/om).log()+(1-cm)*((1-cm)/(1-om)).log()).mean()
            expected = expected+(chosen+rejected+mass)/2
        cursor += size
    torch.testing.assert_close(actual, expected.float(), atol=1e-6, rtol=1e-5)
    grads = compare_grads(actual, expected, model)
    assert any(g is not None and g.is_cuda and g.abs().sum()>0 for g in grads)


def test_cuda_historical_sampler_trainer_admits_new_objective(monkeypatch):
    # The actual trainer's selector admission is separate from parser/sampler policy resolution.
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    from stage2.trainer import Stage2Trainer
    from tests.stage2.test_historical_balanced_sampler import examples
    from tests.stage2.test_reference_random_sampler import ScalarModel
    from transformers import TrainingArguments
    monkeypatch.setattr(TrainingArguments,"world_size",property(lambda self:4))
    config = _bce_config(preference_loss_type=LOWER, preference_topk=7, beta=1., lambda_ce=.1)
    with torch.device("cuda:0"):
        model = ScalarModel()
    args = TrainingArguments(output_dir="checkpoints/dil-both-mean-tests",report_to=[],
        per_device_train_batch_size=1,gradient_accumulation_steps=4)
    trainer = Stage2Trainer(model=model,args=args,train_dataset=examples([1]*8+[3]*8+[6]*4),
        step_preference_config=config,stage2_sampler_policy="historical_balanced_2rank",
        deepspeed_runtime_profile=DeepSpeedRuntimeProfile(True,2,None,None,4))
    assert model.weight.is_cuda
    assert trainer.model_accepts_loss_kwargs is False and trainer.accelerator.even_batches is False
    assert trainer._get_train_sampler().identity.policy == "historical_balanced_2rank"


def test_host_historical_sampler_generator_preserves_approved_index_order():
    # Pure host index-generation contract; the production sampler owns CPU RNG.
    from stage2.sampler import HistoricalBalanced2RankSampler
    from tests.stage2.test_historical_balanced_sampler import examples, PROBE_ORDER
    indices = HistoricalBalanced2RankSampler(examples([1]*8+[3]*8+[6]*4),
        num_replicas=4,seed=42,gradient_accumulation_steps=4)
    assert list(indices) == PROBE_ORDER
