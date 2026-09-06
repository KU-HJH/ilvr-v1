"""Full Trainer objective and routing witnesses for shared suffix replay."""

import copy
from dataclasses import replace

import pytest
import torch

from step_preference.config import StepPreferenceConfig
from step_preference.equivalence import _config, _trainer
from utils_deepseed import parse_args
from test_shared_prefix_model import fixture


@pytest.mark.parametrize("updates", [
    {"packing_enabled": True, "max_packed_sequence_length": 4608},
    {"enabled": False}, {"mode": "stage2a"},
    {"sidecar_query_budget": 0}, {"sidecar_query_budget": True},
    {"sidecar_query_budget": -1}, {"sidecar_attention_backend": "unknown"},
    {"packed_admission_manifest": "old.json", "packed_admission_manifest_sha256": "a"*64},
])
def test_shared_config_rejects_contradictory_routes_and_invalid_budget(updates):
    """Catches accepting legacy admission, disabled preference, or unusable chunk settings."""
    with pytest.raises(ValueError):
        replace(_config(packing=False), shared_prefix_enabled=True, **updates)


def test_cli_selects_shared_without_legacy_capacity():
    """Catches CLI options lost before Trainer config or enabling sharing by default."""
    assert not StepPreferenceConfig().shared_prefix_enabled
    args = parse_args(["--stage", "stage2", "--step_preference_enabled",
        "--step_preference_mode", "stage2b", "--stage2a_checkpoint_path", "checkpoint",
        "--step_preference_objective", "legacy_dpo",
        "--step_negative_path", "negative.jsonl", "--step_negative_manifest", "manifest.json",
        "--step_shared_prefix_enabled", "--step_sidecar_attention_backend", "dense",
        "--step_sidecar_query_budget", "1"])
    config = args.step_preference_config
    assert config.shared_prefix_enabled and not config.packing_enabled
    assert config.sidecar_attention_backend == "dense" and config.sidecar_query_budget == 1


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_shared_trainer_two_layer_multistep_all_components_gradients_and_budgets(backend):
    """Catches changed CE/DPO/entropy denominators, lost clean gradients, or per-chunk reduction."""
    model, ids, spans, steps, _ = fixture(backend)
    # A third valid pair of another modality and length exposes global denominators.
    start = spans[1].target_text_start
    rejected_ids = torch.cat((ids[:, :start], torch.tensor([[61, 62, 63]])), 1)
    steps[1] = dict(steps[1], negative=replace(steps[1]["negative"],
        valid_textual_negative=True, rejected_textual_text="textual reject",
        reference_textual_rejected_score=-3.), textual_rejected=dict(
            model_inputs=dict(input_ids=rejected_ids, attention_mask=torch.ones_like(rejected_ids)),
            completion_token_positions=tuple(range(start, start+3)), completion_token_ids=(61, 62, 63)))
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
                 sample_ids=("shared",), reasoning_step_spans=spans,
                 step_preference=dict(steps=steps))
    configs = [_config(packing=False)] + [replace(_config(packing=False),
        shared_prefix_enabled=True, sidecar_attention_backend="dense", sidecar_query_budget=b)
        for b in (1, 1024)]
    records = []
    for config in configs:
        current = copy.deepcopy(model)
        trainer = _trainer(config)
        loss, out = trainer.compute_loss(current, batch, return_outputs=True)
        assert all(p.grad is None for p in current.parameters())
        loss.backward()
        records.append((loss.detach(), trainer._step_preference_metric_totals,
            {n: p.grad.clone() for n, p in current.named_parameters() if p.requires_grad}))
        if config.shared_prefix_enabled:
            assert out.stage2_sidecar_stats["query_rows"] == 19
            assert trainer._step_preference_metric_totals["stats/reference_cache_entries_consumed"] == 5
            assert trainer._step_preference_metric_totals["stats/reference_cache_entries_missing"] == 0
    expected, metrics, grads = records[0]
    for loss, actual_metrics, actual_grads in records[1:]:
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-4)
        for key, value in metrics.items():
            if key.startswith("loss/") or key.startswith("stats/entropy") or "margin" in key:
                assert actual_metrics[key] == pytest.approx(value, abs=1e-5, rel=1e-4), key
        assert grads.keys() == actual_grads.keys()
        errors = []
        for name in grads:
            torch.testing.assert_close(actual_grads[name], grads[name], atol=1e-5, rtol=1e-4, msg=name)
            errors.append(float((actual_grads[name]-grads[name]).abs().max()))
        print(dict(backend=backend, max_gradient_abs=max(errors), parameters=len(grads),
                   max_component_abs=max(abs(actual_metrics[k]-v) for k,v in metrics.items() if k.startswith("loss/")),
                   total_abs=float((loss-expected).abs()),
                   near_zero_elements=sum(int((g.abs()<1e-8).sum()) for g in grads.values())))


def test_shared_zero_steps_and_invalid_negatives_keep_clean_ce():
    """Catches empty selected-logit handling or dropping entropy with invalid DPO pairs."""
    model, ids, spans, steps, _ = fixture()
    config = replace(_config(packing=False), shared_prefix_enabled=True, sidecar_attention_backend="dense")
    for zero_steps in (False, True):
        joined = [] if zero_steps else [dict(s, negative=replace(s["negative"],
            valid_visual_negative=False, reference_visual_rejected_score=None,
            rejected_visual_text=None), visual_rejected=None) for s in steps]
        batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids,
            sample_ids=("shared",), reasoning_step_spans=() if zero_steps else spans,
            step_preference=dict(steps=joined))
        trainer = _trainer(config)
        loss = trainer.compute_loss(model, batch)
        metrics = trainer._step_preference_metric_totals
        assert metrics["loss/dpo"] == 0
        assert metrics["loss/entropy"] == 0 if zero_steps else metrics["loss/entropy"] > 0
        assert torch.isfinite(loss)
