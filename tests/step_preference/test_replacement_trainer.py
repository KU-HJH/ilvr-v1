"""Trainer contracts for same-image evidence replacement."""

import pytest
import torch

from step_preference.config import StepPreferenceConfig
from step_preference.equivalence import _trainer
from step_preference.spans import ReasoningStepSpan


def _config(**updates):
    values = dict(
        enabled=True,
        mode="stage2b",
        stage2a_checkpoint_path="tiny-stage2a",
        prefix_mode="evidence_replacement",
        lambda_preference=10.0,
        lambda_entropy=0.0,
        sidecar_attention_backend="dense",
        visual_only_context_preference=True,
        visual_off_suppress_latent_boundaries=False,
        replacement_seed=42,
    )
    values.update(updates)
    return StepPreferenceConfig(**values)


@pytest.mark.parametrize("target_scope", ["immediate_text", "until_next_latent"])
def test_trainer_uses_original_minus_replacement_step_mean_and_clean_ce(target_scope, monkeypatch):
    """Catches using clean logits, source scores in reverse, or a detached replacement loss."""
    from tests.step_preference.insertion_helpers import native_case

    model, ids, _, grid, _, pixels = native_case()
    spans = (
        ReasoningStepSpan(0, 0, 5, 6, 8, 9, 11, None),
        ReasoningStepSpan(1, 9, 11, 12, 14, 15, 17, 0),
    )
    batch = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=ids.clone(),
        pixel_values=pixels,
        image_grid_thw=grid,
        sample_ids=("replacement-trainer",),
        reasoning_step_spans=spans,
        step_preference={"steps": tuple({"span": span} for span in spans)},
    )
    trainer = _trainer(_config(target_scope=target_scope))
    visual_calls, rollout_calls = [], []
    visual_hook = model.visual.register_forward_pre_hook(
        lambda *_: visual_calls.append(1)
    )
    original_rollout = model.stage2_rollout_engine.rollout

    def counted_rollout(*args, **kwargs):
        rollout_calls.append(1)
        return original_rollout(*args, **kwargs)

    monkeypatch.setattr(model.stage2_rollout_engine, "rollout", counted_rollout)
    monkeypatch.setattr(model, "generate", lambda *args, **kwargs: pytest.fail("replacement scoring generated tokens"))
    loss, output = trainer.compute_loss(model, batch, return_outputs=True)
    visual_hook.remove()
    by_step = {span.step_id: {} for span in spans}
    for result in output.stage2_sidecar_results:
        by_step[result["step_id"]][result["kind"]] = result["mean_logp"]
    margins = torch.stack(
        [value["visual_original"] - value["visual_replacement"] for value in by_step.values()]
    )
    expected = output.loss - trainer.step_preference_config.lambda_preference * torch.nn.functional.logsigmoid(
        trainer.step_preference_config.beta * (margins - trainer.step_preference_config.preference_margin)
    ).mean()
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-4)
    assert {result["kind"] for result in output.stage2_sidecar_results} == {
        "visual_original", "visual_replacement"
    }
    assert len(visual_calls) == 1
    assert len(rollout_calls) == 1
    loss.backward()
    assert model.model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.model.embed_tokens.weight.grad).all()


def test_replacement_zero_steps_keeps_clean_ce_and_zero_source_counts():
    """Catches an empty context requiring an image replay or inventing source metrics."""
    from step_preference.equivalence import _tiny_model

    model = _tiny_model().train()
    ids = torch.tensor([[11, 12, 13]])
    batch = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=ids.clone(),
        sample_ids=("replacement-empty",),
        reasoning_step_spans=(),
        step_preference={"steps": ()},
    )
    trainer = _trainer(_config())
    loss, output = trainer.compute_loss(model, batch, return_outputs=True)
    torch.testing.assert_close(loss, output.loss)
    assert output.stage2_sidecar_results == ()
    assert trainer._step_preference_metric_totals["stats/replacement_donor_margin_count"] == 0
    assert trainer._step_preference_metric_totals["stats/replacement_image_margin_count"] == 0


def test_replacement_identity_survives_model_config_save_and_reload(tmp_path):
    """Catches a saved replacement checkpoint losing its non-resumable objective identity."""
    from step_preference.equivalence import _tiny_model
    from stage2.trainer import attach_evidence_replacement_identity
    from transformers import Qwen2_5_VLConfig

    model = _tiny_model()
    attach_evidence_replacement_identity(model, _config(replacement_seed=917))
    model.config.save_pretrained(tmp_path)
    restored = Qwen2_5_VLConfig.from_pretrained(tmp_path)
    assert restored.stage2_evidence_replacement_objective == {
        "mode": "evidence_replacement",
        "training_seed": 917,
        "sampling_version": "ilvr-evidence-replacement-v1",
        "selection": "earlier_latent_or_same_image_encoder_features_detached",
        "replacement_gradient": "detached_copy_original_occurrence_differentiable",
    }


def test_replacement_logs_globally_weighted_source_margins(monkeypatch):
    """Catches averaging per-source means instead of raw margin sums and counts."""
    from transformers import Trainer

    trainer = _trainer(_config())
    clean_ce = torch.tensor(1.0)
    values = []
    for step_id, (margin, source_kind) in enumerate(((2.0, "image"), (4.0, "donor"), (4.0, "donor"), (4.0, "donor"))):
        replacement = torch.tensor(-3.0 - step_id)
        values.extend(
            [
                {"kind": "visual_original", "step_id": step_id, "mean_logp": replacement + margin},
                {"kind": "visual_replacement", "step_id": step_id, "mean_logp": replacement, "source_kind": source_kind},
            ]
        )
    _, metrics = trainer._reduce_evidence_replacement_objective(clean_ce, values)
    metrics.update({"stats/scored_tokens": 8, "stats/logical_scoring_groups": 8})
    trainer._accumulate_step_preference_metrics(metrics)

    class SingleRankAccelerator:
        device = torch.device("cpu")

        @staticmethod
        def reduce(value, reduction):
            assert reduction == "sum"
            return value

    trainer.accelerator = SingleRankAccelerator()
    logged = []
    monkeypatch.setattr(Trainer, "log", lambda self, logs, **kwargs: logged.append(logs))
    trainer.log({"loss": 0.0})
    values = logged[0]
    assert values["stats/replacement_margin_sum"] == pytest.approx(14.0)
    assert values["stats/replacement_margin_count"] == pytest.approx(4.0)
    assert values["stats/replacement_margin"] == pytest.approx(3.5)
    assert values["stats/replacement_donor_margin_sum"] == pytest.approx(12.0)
    assert values["stats/replacement_donor_margin_count"] == pytest.approx(3.0)
    assert values["stats/replacement_donor_margin"] == pytest.approx(4.0)
    assert values["stats/replacement_image_margin"] == pytest.approx(2.0)
