"""Runtime tests for diagnostic-only Stage 2 checks, not arithmetic changes."""
import copy
import sys

import pytest
import torch

from step_preference.equivalence import _trainer
from step_preference.spans import ReasoningStepSpan
from tests.step_preference.insertion_helpers import native_case
from tests.step_preference.test_v_off_insertion import insertion_config


def training_batch(ids, grid, pixels):
    spans = (ReasoningStepSpan(0, 0, 5, 6, 8, 9, 11, None),
             ReasoningStepSpan(1, 9, 11, 12, 14, 15, 17, 0))
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
                pixel_values=pixels, image_grid_thw=grid, sample_ids=('debug-probe',),
                reasoning_step_spans=spans,
                step_preference={'steps': tuple({'span': s} for s in spans)})


@pytest.mark.parametrize('argv,want', [([], False), (['--stage2_debugging'], True),
                                      (['--no-stage2_debugging'], False)])
def test_cli_selects_diagnostic_mode(argv, want):
    from utils_deepseed import parse_args
    assert getattr(parse_args(['--stage', 'stage2', *argv]), 'stage2_debugging', None) is want


@pytest.mark.parametrize('debugging', [False, True])
def test_native_forward_only_runs_detailed_checks_in_debug(debugging, monkeypatch):
    model, ids, _, grid, _, pixels = native_case()
    trainer = _trainer(insertion_config(lambda_preference=1.))
    trainer.stage2_debugging = debugging
    observed = []
    for name in ('equal', 'isfinite'):
        original = getattr(torch, name)
        def record(*args, _name=name, _original=original, **kwargs):
            caller = sys._getframe(1)
            if caller.f_code.co_name == '<genexpr>':
                caller = caller.f_back
            observed.append((_name, caller.f_code.co_name))
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, record)
    loss = trainer.compute_loss(model, training_batch(ids, grid, pixels))
    assert observed.count(('isfinite', '_selected_logits')) == int(debugging)
    assert observed.count(('isfinite', '_validate_score_tensors')) == 2 * int(debugging)
    assert observed.count(('equal', 'score_v_off_insertion')) == 2 * int(debugging)
    assert observed.count(('equal', 'validate_shared_inputs')) == int(debugging)
    assert ('isfinite', '_reduce_context_objective') in observed
    assert torch.isfinite(loss)


@pytest.mark.parametrize('debugging', [False, True])
def test_invalid_input_mask_still_fails_before_model_execution(debugging):
    model, ids, _, grid, _, pixels = native_case()
    trainer = _trainer(insertion_config())
    trainer.stage2_debugging = debugging
    batch = training_batch(ids, grid, pixels)
    batch['attention_mask'][0, 3] = 0
    with pytest.raises(ValueError, match='contiguous prefix'):
        trainer.compute_loss(model, batch)


@pytest.mark.parametrize('debugging', [False, True])
def test_final_nonfinite_loss_always_fails(debugging):
    trainer = _trainer(insertion_config())
    trainer.stage2_debugging = debugging
    with pytest.raises(ValueError, match='total loss must be finite'):
        trainer._reduce_context_objective(torch.tensor(float('nan')), [], [], {})


@pytest.mark.parametrize('mode', ['independent', 'cumulative'])
def test_debug_mode_preserves_loss_gradients_rng_and_metrics(mode):
    from peft import LoraConfig, get_peft_model
    model, ids, _, grid, _, pixels = native_case()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=.05,
        use_dora=True, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                      'gate_proj', 'up_proj', 'down_proj']))
    for name, parameter in model.named_parameters():
        if 'lora_B' in name:
            with torch.no_grad():
                parameter.normal_(0, .01)
    batch = training_batch(ids, grid, pixels)
    results = []
    for debugging in (True, False):
        current = copy.deepcopy(model).train()
        trainer = _trainer(insertion_config(branch_mode=mode, lambda_preference=1.))
        trainer.stage2_debugging = debugging
        torch.manual_seed(42)
        loss = trainer.compute_loss(current, batch)
        loss.backward()
        results.append((loss.detach(),
            {name: p.grad.clone() for name, p in current.named_parameters() if p.grad is not None},
            torch.get_rng_state().clone(), dict(trainer._step_preference_metric_totals)))
    for left, right in zip(results[0][:3], results[1][:3]):
        pairs = [(left, right)] if isinstance(left, torch.Tensor) else [(left[k], right[k]) for k in left]
        if isinstance(left, dict):
            assert left.keys() == right.keys()
        for a, b in pairs:
            assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                               b.contiguous().reshape(-1).view(torch.uint8))
    assert results[0][3] == results[1][3]
