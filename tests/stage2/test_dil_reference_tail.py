"""Sample-local BCE objectives through real configured-GAS training_step."""
import copy

import pytest
import torch

from tests.stage2.test_reference_random_sampler import actual_reference_trainer
from tests.step_preference.test_dil_trainer import dil_case, dil_config, observed_forward, full_logit_oracle


@pytest.mark.parametrize("config", [dil_config("dil_bce_rf"), dil_config(lambda_preference=0., lambda_topk_mass=2.),
    dil_config(preference_entropy_top_percent=20, preference_entropy_source="voff")])
def test_three_unequal_samples_plus_padding_use_global_16_denominator(tmp_path, monkeypatch, config):
    # Removing fixed GAS4, using step-weighted rank means, or leaking padding RNG/metrics fails.
    model, _ = dil_case()
    reference = copy.deepcopy(model)
    batches = [dil_case(steps=n)[1] for n in (1, 2, 3)]
    trainer = actual_reference_trainer(tmp_path, model, [batches[0]]*19, config, monkeypatch)
    parameters = [p for p in model.parameters() if p.requires_grad]
    ref_parameters = [p for p in reference.parameters() if p.requires_grad]
    expected_grads, rank_grads, expected_losses = [], [], []
    for batch in batches:
        _, out, off = observed_forward(reference, batch, config)
        oracle = full_logit_oracle(out, off, batch, config)
        expected_losses.append(oracle.detach())
        expected_grads.append(torch.autograd.grad(oracle, ref_parameters, allow_unused=True))
        model.zero_grad(set_to_none=True)
        actual = trainer.training_step(model, dict(batch, stage2_sample_weight=1.))
        torch.testing.assert_close(actual, oracle.detach()/4, atol=1e-6, rtol=1e-5)
        rank_grads.append([p.grad.detach().clone() if p.grad is not None else None for p in parameters])
    before_metrics = dict(trainer._step_preference_metric_totals)
    before_batches = trainer._step_preference_metric_batches
    before_counts = dict(trainer._runtime_objective_observation_counts)
    before_rng = torch.get_rng_state().clone()
    model.zero_grad(set_to_none=True)
    padding_loss = trainer.training_step(model, dict(batches[-1], stage2_sample_weight=0.,
        stage2_sample_index=-1, stage2_sample_latent_block_count=3))
    assert padding_loss.item() == 0
    assert torch.equal(before_rng, torch.get_rng_state())
    assert trainer._step_preference_metric_totals == before_metrics
    assert trainer._step_preference_metric_batches == before_batches
    assert trainer._runtime_objective_observation_counts == before_counts
    for i, parameter in enumerate(parameters):
        if expected_grads[0][i] is not None:
            assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
            observed = sum(g[i] for g in rank_grads)/4
            oracle = sum(g[i] for g in expected_grads)/16
            torch.testing.assert_close(observed, oracle, atol=1e-5, rtol=1e-4)
    print({"kind": config.preference_loss_type, "preference_weight": config.lambda_preference,
           "sample_step_counts": [1, 2, 3], "sample_losses": [float(x) for x in expected_losses],
           "gas": trainer.args.gradient_accumulation_steps, "tail_denominator": 16,
           "padding_loss": float(padding_loss), "padding_metrics_delta": trainer._step_preference_metric_batches-before_batches})
    trainer.log({})
    logged = trainer.state.log_history[-1]
    assert logged["stats/preference_loss_type"] == config.preference_loss_type
    assert logged["stats/entropy_source"] == config.preference_entropy_source
    assert logged["stats/entropy_top_percent"] == (config.preference_entropy_top_percent or 100.)
    assert logged["stats/eligible_steps"] == 2.  # sample mean of 1, 2 and 3; padding absent
