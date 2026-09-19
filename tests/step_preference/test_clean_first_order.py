"""Runtime witnesses that clean projections and CE finish before sidecar work."""

import importlib

import pytest
import torch

from tests.step_preference.insertion_helpers import native_case


@pytest.mark.parametrize("prefix_mode,scorer_module,scorer_name", [
    ("clean_prefix", "stage2_shared_prefix", "score_sidecars"),
    ("v_off_insertion", "stage2_v_off_insertion", "score_v_off_insertion"),
    ("evidence_replacement", "stage2_evidence_replacement", "score_evidence_replacement"),
])
@pytest.mark.parametrize("output_path,expected_clean_events", [
    ("full", ["head", "ce"]),
    ("loss_only", ["head", "ce", "head"]),
    ("selective", ["head"]),
])
def test_clean_projection_and_ce_finish_before_shared_sidecars(
    prefix_mode, scorer_module, scorer_name, output_path, expected_clean_events, monkeypatch,
):
    """Moving a scorer before clean LM/CE (or between clean projections) fails."""
    model, ids, _, grid, options, pixels = native_case()
    options = dict(options, prefix_mode=prefix_mode)
    if prefix_mode == "evidence_replacement":
        options.update(sample_id="clean-first", replacement_seed=42,
                       replacement_version="ilvr-evidence-replacement-v1")
    labels = ids.clone()
    labels[:, :9] = -100
    positions = torch.tensor([14, 8, 9])
    kwargs = dict(stage2_logit_positions=positions)
    if output_path == "selective":
        kwargs["stage2_selective_logits_only"] = True
    else:
        kwargs["labels"] = labels
        kwargs["stage2_loss_only"] = output_path == "loss_only"

    events = []
    projections = []
    module = importlib.import_module(f"transformers.models.qwen2_5_vl.{scorer_module}")
    real_scorer = getattr(module, scorer_name)

    def observed_scorer(*args, **kwargs):
        events.append("sidecar_enter")
        result = real_scorer(*args, **kwargs)
        events.append("sidecar_exit")
        return result

    real_ce = torch.nn.CrossEntropyLoss.forward

    def observed_ce(self, *args, **kwargs):
        result = real_ce(self, *args, **kwargs)
        events.append("ce")
        return result

    def observed_head(_module, _args, output):
        events.append("head")
        projections.append(output)

    monkeypatch.setattr(module, scorer_name, observed_scorer)
    monkeypatch.setattr(torch.nn.CrossEntropyLoss, "forward", observed_ce)
    handle = model.lm_head.register_forward_hook(observed_head)
    try:
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       pixel_values=pixels, image_grid_thw=grid, use_cache=False,
                       stage2_shared_prefix_inputs=options, **kwargs)
    finally:
        handle.remove()

    assert events.count("sidecar_enter") == events.count("sidecar_exit") == 1
    first_sidecar = events.index("sidecar_enter")
    assert events[:first_sidecar] == expected_clean_events, events
    assert events[-1] == "sidecar_exit", events
    assert output.stage2_sidecar_results
    assert all(torch.isfinite(result["mean_logp"]) for result in output.stage2_sidecar_results)
    if output_path == "full":
        torch.testing.assert_close(output.logits, projections[0], atol=0, rtol=0)
        torch.testing.assert_close(output.stage2_selected_logits,
                                   projections[0].index_select(1, positions), atol=0, rtol=0)
    else:
        assert output.logits is None
        clean_selected = projections[1 if output_path == "loss_only" else 0]
        torch.testing.assert_close(output.stage2_selected_logits, clean_selected, atol=0, rtol=0)
    if output_path == "selective":
        assert output.loss is None
    else:
        assert torch.isfinite(output.loss)
