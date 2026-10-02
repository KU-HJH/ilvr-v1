"""Consumer-visible defaults, CE-only gradients, and outside-beta preference margin."""
import json
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from step_preference.config import StepPreferenceConfig
from step_preference.dil_identity import build_dil_identity, read_dil_identity, validate_dil_resume
from step_preference.dpo_loss import context_preference_loss
from step_preference.equivalence import _trainer
from utils_deepseed import parse_args

REPO = Path(__file__).resolve().parents[2]
SELECTORS = ("dpo", "dil_bce_rf", "dil_bce_rf_topk", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_both_upper")


def config(selector="dpo", **changes):
    values = dict(enabled=True, mode="stage2b", direct_from_stage1=True,
                  shared_prefix_enabled=True, visual_off_suppress_latent_boundaries=False,
                  lambda_entropy=0., preference_loss_type=selector, sidecar_attention_backend="dense")
    if "topk" in selector:
        values["preference_topk"] = 7
    return StepPreferenceConfig(**(values | changes))


def cli(selector):
    values = ["--stage", "stage2", "--step_preference_enabled", "--step_preference_mode", "stage2b",
              "--stage2b_direct_from_stage1", "--stage1_adapter_path", "checkpoint",
              "--step_shared_prefix_enabled", "--no-step_visual_off_suppress_latent_boundaries",
              "--step_entropy_weight", "0", "--step_preference_loss_type", selector]
    return values + (["--step_preference_topk", "7"] if "topk" in selector else [])


@pytest.mark.parametrize("selector", SELECTORS)
def test_fresh_cli_direct_and_namespace_resolve_same_loss(selector):
    # Break: ordinary defaults leak into DIL or fresh CE is ignored by a constructor.
    parsed = parse_args(cli(selector))
    direct = config(selector)
    namespace = SimpleNamespace(**vars(parsed))
    for actual in (parsed.step_preference_config, direct, StepPreferenceConfig.from_args(namespace)):
        assert actual.lambda_ce == .1
        assert actual.beta == (10. if selector == "dpo" else 1.)
        assert actual.preference_margin == (.5 if selector == "dpo" else 0.)
    assert parsed.step_dpo_beta == direct.beta
    assert parsed.step_preference_margin == direct.preference_margin


@pytest.mark.parametrize("alias", ["--step_preference_beta", "--step_dpo_beta"])
@pytest.mark.parametrize("before", [False, True])
def test_explicit_historical_coefficients_are_order_independent(alias, before):
    options = [alias, ".5", "--ce_weight", "1", "--step_preference_margin", "0"]
    args = parse_args(options + cli("dil_bce_rf") if before else cli("dil_bce_rf") + options)
    assert (args.step_preference_config.beta, args.step_preference_config.lambda_ce,
            args.step_preference_config.preference_margin) == (.5, 1., 0.)


def test_namespace_beta_alias_survives_an_omitted_legacy_alias():
    args = SimpleNamespace(step_dpo_beta=None, step_preference_beta=.37)
    assert StepPreferenceConfig.from_args(args).beta == .37


def test_dil_resume_path_is_validated_before_identity_io():
    with pytest.raises(ValueError, match="resume_from_checkpoint_path"):
        config("dil_bce_rf", direct_from_stage1=False, resume_from_checkpoint_path=True)


def test_context_margin_is_subtracted_after_beta_with_both_gradients():
    # delta=1 -> logit=2*1-.5=1.5. Putting margin inside beta yields logit=1.
    clean = torch.tensor([-1.], requires_grad=True)
    off = torch.tensor([-2.], requires_grad=True)
    loss = context_preference_loss(clean, off, beta=2., margin=.5)
    assert loss.item() == pytest.approx(math.log1p(math.exp(-1.5)))
    loss.backward()
    assert clean.grad.item() == pytest.approx(-2/(1+math.exp(1.5)))
    assert off.grad.item() == pytest.approx(2/(1+math.exp(1.5)))


@pytest.mark.parametrize("replacement", [False, True])
def test_reducers_weight_only_raw_ce_and_use_outside_beta_margin(replacement):
    settings = {"prefix_mode": "evidence_replacement"} if replacement else {}
    trainer = _trainer(config(beta=2., preference_margin=.5, lambda_ce=.1, **settings))
    ce = torch.tensor(3., requires_grad=True)
    clean = torch.tensor(-1., requires_grad=True)
    off = torch.tensor(-2., requires_grad=True)
    if replacement:
        total, metrics = trainer._reduce_evidence_replacement_objective(ce, [
            {"step_id": 0, "kind": "visual_original", "mean_logp": clean},
            {"step_id": 0, "kind": "visual_replacement", "source_kind": "image", "mean_logp": off},
        ])
    else:
        total, metrics = trainer._reduce_context_objective(ce, [clean], [torch.zeros(1)],
            {(0, "visual"): {"mean_logp": off, "entropies": torch.zeros(1)}})
    assert total.item() == pytest.approx(.3 + math.log1p(math.exp(-1.5)))
    assert metrics["loss/ce"].item() == 3.  # metric remains raw CE
    total.backward()
    assert ce.grad.item() == pytest.approx(.1)
    assert clean.grad.item() == pytest.approx(-2/(1+math.exp(1.5)))


@pytest.mark.parametrize("selector", SELECTORS)
def test_real_sidecar_loss_scales_ce_once_without_scaling_preference(selector):
    # Compare a second real objective on the identical forward graph: only CE changes.
    from dataclasses import replace
    from tests.step_preference.test_dil_trainer import dil_case
    model, batch = dil_case()
    cfg = config(selector, lambda_ce=.1, beta=2., preference_margin=.5 if selector == "dpo" else 0.)
    trainer = _trainer(cfg)
    actual, output = trainer.compute_loss(model, batch, return_outputs=True)
    raw_ce = output.loss
    other, _ = _trainer(replace(cfg, lambda_ce=1.)).compute_loss(model, batch, return_outputs=True)
    torch.testing.assert_close(other - actual, .9*raw_ce, atol=1e-6, rtol=1e-5)
    parameter = model.model.embed_tokens.weight
    delta_grad = torch.autograd.grad(other-actual, parameter, retain_graph=True)[0]
    ce_grad = torch.autograd.grad(raw_ce, parameter)[0]
    torch.testing.assert_close(delta_grad, .9*ce_grad, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("enabled", [False, True])
def test_plain_stage2_and_stage2a_scale_ce_once(enabled):
    from step_preference.equivalence import _tiny_model
    model = _tiny_model().train()
    ids = torch.tensor([[11, 12, 13]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
    raw_ce = model(**inputs, stage2_loss_only=False).loss
    loss, output = _trainer(StepPreferenceConfig(enabled=enabled, lambda_ce=.1)).compute_loss(model, inputs, return_outputs=True)
    torch.testing.assert_close(loss, .1*raw_ce)
    assert output.loss is loss
    parameter = model.model.embed_tokens.weight
    actual = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
    raw = torch.autograd.grad(raw_ce, parameter)[0]
    torch.testing.assert_close(actual, .1*raw)


@pytest.mark.parametrize("ce,beta", [(1., .5), (.1, 1.)])
def test_old_and_new_dil_identity_roundtrip_and_resume(ce, beta):
    from step_preference.dil_identity import attach_dil_identity
    from step_preference.equivalence import _tiny_model
    from transformers import Qwen2_5_VLConfig
    cfg = config("dil_bce_rf", lambda_ce=ce, beta=beta, preference_margin=0.)
    identity = build_dil_identity(cfg)
    path = REPO/"checkpoints/loss-defaults-tests"/f"identity-ce{ce}-beta{beta}"
    path.mkdir(parents=True, exist_ok=True)
    model = _tiny_model()
    attach_dil_identity(model, cfg)
    model.config.save_pretrained(path)
    restored = Qwen2_5_VLConfig.from_pretrained(path)
    assert read_dil_identity(restored.to_dict()) == identity
    validate_dil_resume(path, cfg)


@pytest.mark.parametrize("selector", SELECTORS)
def test_shell_fallbacks_emit_objective_aware_coefficients(selector):
    from tests.multinode.test_dil_forwarding import worker_commands
    settings = {"CE_WEIGHT": "", "STEP_PREFERENCE_BETA": "", "STEP_DPO_BETA": "",
                "STEP_PREFERENCE_MARGIN": "", "STEP_PREFERENCE_LOSS_TYPE": selector,
                "STEP_PREFERENCE_TOPK": "7" if "topk" in selector else ""}
    for rank, _, result in worker_commands(settings):
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        expected = {"--ce_weight": "0.1", "--step_preference_beta": "10.0" if selector == "dpo" else "1.0",
                    "--step_preference_margin": "0.5" if selector == "dpo" else "0.0"}
        for option, value in expected.items():
            assert argv[argv.index(option)+1] == value
        assert argv[argv.index("--machine_rank")+1] == str(rank)


def test_fresh_dil_profile_replaces_stale_coefficients():
    result = subprocess.run(["bash", "-c", "source scripts/stage2/dil_bce_profile.sh; printf '%s\\n' \"$CE_WEIGHT $STEP_PREFERENCE_BETA $STEP_DPO_BETA $STEP_PREFERENCE_MARGIN\""],
        cwd=REPO, env={**os.environ, "CE_WEIGHT": "1", "STEP_PREFERENCE_BETA": ".5", "STEP_DPO_BETA": ".5", "STEP_PREFERENCE_MARGIN": ".5"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1 1.0 1.0 0"


def test_dil_resume_omitted_coefficients_use_saved_values():
    path = REPO/"checkpoints/loss-defaults-tests/saved-defaults"
    path.mkdir(parents=True, exist_ok=True)
    identity = build_dil_identity(config("dil_bce_rf", beta=.5, lambda_ce=1., preference_margin=0.))
    (path/"config.json").write_text(json.dumps({"ilvr_context_preference": identity}))
    saved = config("dil_bce_rf", direct_from_stage1=False, resume_from_checkpoint_path=str(path))
    assert (saved.beta, saved.lambda_ce, saved.preference_margin) == (.5, 1., 0.)
    validate_dil_resume(path, saved)
    explicit = config("dil_bce_rf", beta=1., lambda_ce=.1, direct_from_stage1=False, resume_from_checkpoint_path=str(path))
    with pytest.raises(ValueError, match="identity"):
        validate_dil_resume(path, explicit)


def test_stage1_resume_omitted_ce_uses_saved_metadata():
    # The metadata loader still validates the complete record before model construction.
    path = REPO/"checkpoints/loss-defaults-tests/stage1-saved-coefficient"
    path.mkdir(parents=True, exist_ok=True)
    from tests.stage1.test_checkpointing import sample_metadata
    (path/"stage1_peft_metadata.json").write_text(json.dumps(sample_metadata().to_dict()))
    arguments = ["--stage", "stage1", "--resume_from_checkpoint_path", str(path)]
    assert parse_args(arguments).ce_weight == 1.
    assert parse_args(arguments + ["--ce_weight", ".1"]).ce_weight == .1


def test_ordinary_nonzero_margin_resume_is_rejected_before_restore():
    # Historical ordinary margins were beta-scaled and lack exact formula identity.
    with pytest.raises(ValueError, match="mutually exclusive"):
        config(beta=2., preference_margin=.5, direct_from_stage1=False,
               stage2a_checkpoint_path="stage2a", resume_from_checkpoint_path="saved")


def test_legacy_reference_difference_stays_inside_beta_and_tunable_margin_outside():
    # delta = (-1 - -3) - (-2 - -3) = 1; logit = 2*1-.5 = 1.5.
    from step_preference.dpo_loss import dpo_pair_losses
    chosen, rejected, ref_chosen, ref_rejected = [torch.tensor([x], requires_grad=True) for x in (-1., -3., -2., -3.)]
    got = dpo_pair_losses(chosen, rejected, ref_chosen, ref_rejected, beta=2., margin=.5)
    assert got.item() == pytest.approx(math.log1p(math.exp(-1.5)))
    got.sum().backward()
    assert chosen.grad.item() == pytest.approx(-2/(1+math.exp(1.5)))
    assert rejected.grad.item() == pytest.approx(2/(1+math.exp(1.5)))
    assert ref_chosen.grad is None and ref_rejected.grad is None


def test_legacy_trainer_reducer_forwards_outside_beta_margin():
    cfg = StepPreferenceConfig(enabled=True, mode="stage2b", objective="legacy_dpo", beta=2.,
        preference_margin=.5, lambda_ce=.1, lambda_entropy=0., negative_path="n", negative_manifest="m",
        stage2a_checkpoint_path="a")
    ce = torch.tensor(3., requires_grad=True)
    values = dict(clean_ce=ce, chosen_scores=[torch.tensor(-1.)], visual_rejected_scores=[torch.tensor(-3.)],
        textual_rejected_scores=[torch.tensor(-3.)], reference_chosen=[torch.tensor(-2.)],
        reference_visual_rejected=[torch.tensor(-3.)], reference_textual_rejected=[torch.tensor(-3.)],
        valid_visual=[True], valid_textual=[False], visual_entropy_losses=[torch.zeros(1)],
        textual_entropy_losses=[torch.zeros(1)], clean_entropy_values=[torch.zeros(1)],
        visual_entropy_values=[torch.zeros(1)], textual_entropy_values=[torch.zeros(1)], branch_forwards=1, scored_tokens=1)
    got = _trainer(cfg)._reduce_stage2b_objective(**values)
    assert got.item() == pytest.approx(.3+math.log1p(math.exp(-1.5)))
    got.backward()
    assert ce.grad.item() == pytest.approx(.1)


def test_stage1_shell_forwards_explicit_ce_coefficient():
    result = subprocess.run(["bash", "run_stage1_training.sh", "--method", "dora", "--model", "model", "--data-path", "data.jsonl",
        "--image-root", "data", "--output-dir", str(REPO/"checkpoints/loss-defaults-tests/stage1-dry"), "--dry-run"],
        cwd=REPO, env={**os.environ, "STAGE1_CE_WEIGHT": ".1"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    import shlex
    argv = shlex.split(result.stdout)
    assert argv[argv.index("--ce_weight")+1] == ".1"


def test_stage1_coordinator_forwards_ce_to_both_workers():
    from src.multinode import build_launch_plan
    from src.multinode_runtime import build_owned_tmux_commands
    plan = build_launch_plan(stage="stage1", tuning_method="dora", zero_stage=1,
                             run_id="loss-defaults-stage1", main_process_port=29631)
    commands = build_owned_tmux_commands(plan, commit="a"*40,
        stage_arguments=("--method", "dora", "--model", "model", "--data-path", "data.jsonl", "--image-root", "data", "--ce-weight", ".1"))
    for spec in commands.node_specs:
        import shlex
        result = subprocess.run([*spec.command[spec.command.index("--")+1:], "--dry-run"],
            cwd=REPO, env={**os.environ, **dict(spec.environment)}, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        argv = shlex.split(result.stdout)
        assert argv[argv.index("--ce_weight")+1] == ".1"


def test_dil_worker_resume_restores_omitted_coefficients_from_identity():
    from tests.multinode.test_dil_forwarding import worker_commands
    path = REPO/"checkpoints/loss-defaults-tests/worker-resume"
    path.mkdir(parents=True, exist_ok=True)
    saved = build_dil_identity(config("dil_bce_rf", beta=.5, lambda_ce=1., preference_margin=0.))
    (path/"config.json").write_text(json.dumps({"ilvr_context_preference": saved}))
    settings = {"CE_WEIGHT": "", "STEP_PREFERENCE_BETA": "", "STEP_DPO_BETA": "", "RESUME_CHECKPOINT": str(path),
                "STEP_PREFERENCE_LOSS_TYPE": "dil_bce_rf", "STAGE1_ADAPTER_PATH": "", "STAGE2B_DIRECT_FROM_STAGE1": "false"}
    for _, _, result in worker_commands(settings, ("--max_steps", "212")):
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        assert argv[argv.index("--ce_weight")+1] == "1.0"
        assert argv[argv.index("--step_preference_beta")+1] == "0.5"
