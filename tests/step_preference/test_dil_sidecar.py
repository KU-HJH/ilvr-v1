"""Sidecar alignment, checkpoint replay and execution-cost witnesses."""
import copy
from dataclasses import replace

import pytest
import torch

from step_preference.equivalence import _trainer
from step_preference.shared_prefix_measurement import RuntimeCounters
from step_preference.sidecar import build_context_sidecar_plan
from tests.step_preference.test_dil_trainer import dil_case, dil_config, observed_forward


def test_sidecar_returns_only_per_token_statistics():
    model, batch = dil_case()
    loss, output, _ = observed_forward(model, batch, dil_config())
    assert loss.requires_grad
    for span, result in zip(batch["reasoning_step_spans"], output.stage2_sidecar_results):
        assert "dil_tokens" in result, "top-k checkpoint result contract is absent"
        assert all(t.ndim == 1 for t in result["dil_tokens"])
        assert result["predictor_positions"] == tuple(range(span.target_text_start-1, span.target_text_end-1))
        assert torch.equal(result["target_ids"], batch["input_ids"][0, span.target_text_start:span.target_text_end])
        assert result["target_logps"].shape == result["target_ids"].shape
        assert all(not isinstance(t, torch.Tensor) or t.ndim < 2 for t in result.values())


@pytest.mark.parametrize("corruption", ["step", "target", "position"])
def test_sidecar_rejects_permuted_step_identity(corruption):
    model, batch = dil_case(unequal=False)
    plan = build_context_sidecar_plan(clean_input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        spans=batch["reasoning_step_spans"], visual_only=True)
    a, b = plan.groups[0]
    changes = {"step_id": b.step_id} if corruption == "step" else (
        {"target_token_ids": b.target_token_ids} if corruption == "target" else {"logical_positions": b.logical_positions})
    plan = replace(plan, groups=((replace(a, **changes), b), ()))
    with pytest.raises(ValueError, match="DIL.*(identity|alignment)"):
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], use_cache=False,
            stage2_logit_positions=torch.tensor(plan.root_selected_positions),
            stage2_shared_prefix_inputs=dict(plan=plan, backend="dense",
                dil_objective=dict(loss_type="dil_bce_rf_topk", beta=.5, k=7)))


@pytest.mark.parametrize("mass_only", [False, True])
@pytest.mark.parametrize("kind", ["dil_bce_rf_topk", "dil_bce_rf_topk_reverse"])
def test_dil_checkpoint_replay_preserves_gradients(mass_only, kind):
    model, batch = dil_case()
    baseline = copy.deepcopy(model)
    outcomes = []
    for current, enabled in [(model, True), (baseline, False)]:
        observed_kv, observed_latent = [], []
        def capture(_module, _args, kwargs, out):
            if kwargs.get("stage2_capture_kv"):
                for pair in out.stage2_key_values:
                    for value in pair:
                        value.register_hook(lambda grad: observed_kv.append(grad.detach()))
                kwargs["inputs_embeds"].register_hook(lambda grad: observed_latent.append(grad[:, [2, 3, 9, 10]].detach()))
        def options(_module, _args, kwargs):
            kwargs["stage2_shared_prefix_inputs"]["checkpoint"] = enabled
        pre = current.register_forward_pre_hook(options, with_kwargs=True)
        hook = current.model.register_forward_hook(capture, with_kwargs=True)
        try:
            loss, out, _ = observed_forward(current, batch, dil_config(kind, lambda_preference=0. if mass_only else 1.))
            (loss-out.loss).backward()
        finally:
            hook.remove(); pre.remove()
        assert observed_kv and any(torch.count_nonzero(g) for g in observed_kv)
        assert observed_latent and any(torch.count_nonzero(g) for g in observed_latent)
        outcomes.append((loss.detach(), [p.grad for p in current.parameters()]))
        print({"checkpoint": enabled, "mass_only": mass_only, "kv_nonzero": sum(int(torch.count_nonzero(g)) for g in observed_kv),
               "latent_nonzero": sum(int(torch.count_nonzero(g)) for g in observed_latent)})
    torch.testing.assert_close(outcomes[0][0], outcomes[1][0], atol=0, rtol=0)
    for a, b in zip(outcomes[0][1], outcomes[1][1]):
        if b is not None:
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_dil_adds_no_forward_calls():
    counts = []
    for config in [dil_config("dpo"), dil_config("dil_bce_rf"), dil_config(),
                   dil_config(preference_entropy_top_percent=20),
                   dil_config(preference_entropy_top_percent=20, preference_entropy_source="voff")]:
        model, batch = dil_case()
        with RuntimeCounters(model) as counters:
            head_calls = {"forward": 0, "backward": 0}
            def count_head(_module, _args, _output):
                head_calls[counters.phase] += 1
            hook = model.lm_head.register_forward_hook(count_head)
            try:
                loss = _trainer(config).compute_loss(model, batch)
                before = counters.snapshot()
                before["lm_head_calls_forward"] = head_calls["forward"]
                counters.phase = "backward"
                loss.backward()
                after = counters.snapshot()
                after["lm_head_calls_backward"] = head_calls["backward"]
            finally:
                hook.remove()
        counts.append(before)
        print({"kind": config.preference_loss_type, "source": config.preference_entropy_source,
               "percent": config.preference_entropy_top_percent, "forward": before, "with_replay": after})
    assert counts[1:] == [counts[0]]*4
