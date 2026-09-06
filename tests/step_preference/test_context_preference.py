"""New default Stage 2B: clean targets only, independent runtime witnesses."""

import copy
import math
from dataclasses import replace

import pytest
import torch

from step_preference import dpo_loss
from step_preference.config import StepPreferenceConfig
from step_preference.equivalence import _tiny_model, _trainer
from step_preference.shared_prefix_measurement import RuntimeCounters, tiny_measurement_case
from utils_deepseed import parse_args


def context_config(**kwargs):
    return StepPreferenceConfig(enabled=True, mode="stage2b",
        stage2a_checkpoint_path="tiny-stage2a", sidecar_attention_backend="dense", **kwargs)


def forbidden(*args, **kwargs):
    pytest.fail("new default reached legacy negative/reference/rejected computation")


def test_context_preference_averages_individual_losses_and_keeps_both_gradients():
    """Catches sigmoid-after-mean and detached ablated scores."""
    clean = torch.tensor([-1., -3.], requires_grad=True)
    off = torch.tensor([-3., -1.], requires_grad=True)
    loss_fn = getattr(dpo_loss, "context_preference_loss", None)
    assert callable(loss_fn), "context_preference_loss is missing"
    loss = loss_fn(clean, off, beta=1., margin=0.)
    assert float(loss.detach()) == pytest.approx((math.log1p(math.exp(-2)) + math.log1p(math.exp(2)))/2)
    loss.backward()
    assert torch.all(clean.grad < 0) and torch.all(off.grad > 0)


@pytest.mark.parametrize("margin", [0.0, 0.4])
def test_context_loss_uses_margin_before_sigmoid(margin):
    clean = torch.tensor([-1., -3.], requires_grad=True)
    off = torch.tensor([-3., -1.], requires_grad=True)
    fn = getattr(dpo_loss, "context_preference_loss", None)
    assert callable(fn), "context_preference_loss is missing"
    got = fn(clean, off, beta=1., margin=margin)
    expected = (math.log1p(math.exp(margin-2.)) +
                math.log1p(math.exp(margin+2.))) / 2
    assert float(got.detach()) == pytest.approx(expected)
    got.backward()
    assert torch.all(clean.grad < 0) and torch.all(off.grad > 0)


@pytest.mark.parametrize("clean,off", [
    (torch.empty(0), torch.empty(0)),
    (torch.zeros(2), torch.zeros(1)),
    (torch.tensor([float("nan")]), torch.zeros(1)),
])
def test_context_loss_rejects_invalid_scores(clean, off):
    fn = getattr(dpo_loss, "context_preference_loss", None)
    assert callable(fn), "context_preference_loss is missing"
    with pytest.raises(ValueError):
        fn(clean, off, beta=0.1, margin=0.0)


@pytest.mark.parametrize("k", [1, 4, 8, 16])
def test_default_actual_scoring_calls_are_three_without_legacy(k, monkeypatch):
    """Catches a legacy default, per-step/chunk forwards, or skipped steps."""
    model, batch = tiny_measurement_case(steps=k, total_target_tokens=2*k, device="cpu")
    # Legacy fixture metadata is deliberately absent at the training boundary.
    batch["step_preference"] = {"steps": tuple(dict(span=s) for s in batch["reasoning_step_spans"])}
    trainer = _trainer(context_config(sidecar_query_budget=1))
    monkeypatch.setattr(model, "generate", forbidden)
    monkeypatch.setattr(trainer, "_compute_stage2b_loss", forbidden)
    for name in ("_compute_stage2b_shared_prefix_loss", "_compute_stage2b_packed_loss"):
        monkeypatch.setattr(trainer, name, forbidden)
    monkeypatch.setattr("stage2.trainer.aggregate_step_dpo", forbidden)
    monkeypatch.setattr("step_preference.sidecar._rejected_branch", forbidden)
    with RuntimeCounters(model) as counters:
        loss, output = trainer.compute_loss(model, batch, return_outputs=True)
        before_backward = counters.snapshot()
        counters.phase = "backward"
        loss.backward()
        counts = counters.snapshot()
    assert before_backward["decoder_calls_forward"] - before_backward["latent_decoder_calls_forward"] == 3
    assert before_backward["latent_decoder_calls_forward"] == 2*k
    assert before_backward["sidecar_chunks_forward"] == 2
    assert len(output.stage2_sidecar_results) == 2*k
    assert {r["kind"] for r in output.stage2_sidecar_results} == {"visual_off", "textual_off"}
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())
    metrics = trainer._step_preference_metric_totals
    assert metrics["stats/scored_tokens"] == 6*k
    assert "loss/context_preference" in metrics and "loss/dpo" not in metrics
    print(dict(K=k, counts=counts, metrics=metrics))


def test_default_preflight_and_collator_need_no_negative_artifacts(inputs, monkeypatch):
    """Catches hidden negative requirements above the trainer."""
    import main
    from stage2.data import Stage2DataCollator
    from tests.step_preference.test_packed_admission_cli import AdmissionProcessor
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds

    source = inputs()
    source.negative_path.unlink()
    source.negative_manifest.unlink()
    args = parse_args(["--stage", "stage2", "--step_preference_enabled",
        "--step_preference_mode", "stage2b", "--stage2a_checkpoint_path", str(source.checkpoint_path),
        "--load_model_path", str(source.checkpoint_path), "--data_path", str(source.clean_data_path),
        "--image_root", str(source.image_root), "--attention_implementation", "sdpa",
        "--vision_attention_implementation", "flash_attention_2"])
    monkeypatch.setattr(main, "read_negative_manifest", forbidden)
    monkeypatch.setattr(main, "load_negative_store", forbidden)
    prepared = main.preflight_stage2b_artifacts(args)
    assert prepared.negative_store is None
    collator = Stage2DataCollator(AdmissionProcessor(), LatentTokenIds(90, 91, 92),
        8, True, step_preference_enabled=True)
    batch = collator([prepared.prepared_examples[0]])
    assert batch["reasoning_step_spans"]
    assert all(set(s) == {"span"} for s in batch["step_preference"]["steps"])
    config = copy.deepcopy(_tiny_model().config)
    config.vocab_size = 512
    config.latent_size = 8
    config.image_token_id = 70
    config.num_hidden_layers = 2
    config.vision_config.window_size = 8
    config.vision_config.spatial_merge_size = 2
    model = type(_tiny_model())(config).float().train()
    model.visual.requires_grad_(False)
    patch_width = (
        config.vision_config.in_channels
        * config.vision_config.temporal_patch_size
        * config.vision_config.patch_size**2
    )
    batch["pixel_values"] = torch.zeros(16, patch_width)
    batch["position_ids"] = (
        torch.arange(batch["input_ids"].shape[1]).view(1, 1, -1).expand(3, 1, -1)
    )
    trainer = _trainer(
        replace(args.step_preference_config, sidecar_attention_backend="dense")
    )
    monkeypatch.setattr(model, "generate", forbidden)
    monkeypatch.setattr(trainer, "_compute_stage2b_loss", forbidden)
    monkeypatch.setattr("stage2.trainer.aggregate_step_dpo", forbidden)
    monkeypatch.setattr("step_preference.sidecar._rejected_branch", forbidden)
    with RuntimeCounters(model) as counters:
        loss = trainer.compute_loss(model, batch)
        before_backward = counters.snapshot()
        counters.phase = "backward"
        loss.backward()
    assert torch.isfinite(loss)
    assert (
        before_backward["decoder_calls_forward"]
        - before_backward["latent_decoder_calls_forward"]
        == 3
    )
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


# Reuse input preparation, not negative data, from the existing admission tests.
from tests.step_preference.test_packed_admission_cli import inputs
