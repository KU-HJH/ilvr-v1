"""Remove unused trainer work while retaining the independently pinned baseline."""
import ast
import copy
from pathlib import Path
import subprocess
from types import MethodType

import pytest
import torch

from step_preference.equivalence import _trainer
from step_preference.numerical_tolerance import compare_numerical_tolerance
from step_preference.spans import ReasoningStepSpan
from tests.step_preference.insertion_helpers import native_case
from tests.step_preference.test_v_off_insertion import insertion_config


def training_batch(ids, grid, pixels):
    spans = (ReasoningStepSpan(0, 0, 5, 6, 8, 9, 11, None),
             ReasoningStepSpan(1, 9, 11, 12, 14, 15, 17, 0))
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
                pixel_values=pixels, image_grid_thw=grid, sample_ids=('no-op-audit',),
                reasoning_step_spans=spans,
                step_preference={'steps': tuple({'span': span} for span in spans)})


@pytest.mark.parametrize('mode', ['independent', 'cumulative'])
def test_loss_only_insertion_does_not_project_unused_clean_logits(mode):
    # Reused from 0dc81c0: catches the unused 4-row clean projection after CE.
    model, ids, _, grid, _, pixels = native_case()
    projected_rows = []
    handle = model.lm_head.register_forward_hook(
        lambda module, args, output: projected_rows.append(output.numel() // output.shape[-1]))
    try:
        loss = _trainer(insertion_config(branch_mode=mode, lambda_preference=1.)).compute_loss(
            model, training_batch(ids, grid, pixels), return_outputs=False)
    finally:
        handle.remove()
    assert projected_rows == ([4, 2, 2, 16] if mode == 'independent' else [4, 4, 16])
    assert torch.isfinite(loss)


@pytest.mark.parametrize('mode', ['independent', 'cumulative'])
def test_requested_outputs_keep_full_and_selected_clean_logits(mode):
    model, ids, _, grid, _, pixels = native_case()
    _, outputs = _trainer(insertion_config(branch_mode=mode, lambda_preference=1.)).compute_loss(
        model, training_batch(ids, grid, pixels), return_outputs=True)
    assert outputs.logits.shape == (1, 17, 128)
    assert outputs.stage2_selected_logits.shape == (1, 4, 128)
    torch.testing.assert_close(outputs.stage2_selected_logits,
                               outputs.logits[:, [8, 9, 14, 15]], atol=0, rtol=0)


@pytest.mark.parametrize('prefix', ['clean_prefix', 'v_off_prefix'])
def test_other_prefix_loss_only_keeps_clean_selected_projection(prefix):
    model, ids, _, grid, _, pixels = native_case()
    projected_rows = []
    handle = model.lm_head.register_forward_hook(
        lambda module, args, output: projected_rows.append(output.numel() // output.shape[-1]))
    try:
        _trainer(insertion_config(prefix_mode=prefix, lambda_preference=1.)).compute_loss(
            model, training_batch(ids, grid, pixels), return_outputs=False)
    finally:
        handle.remove()
    assert projected_rows[-2:] == [16, 4]


@pytest.mark.parametrize('prefix', ['v_off_insertion', 'clean_prefix', 'v_off_prefix'])
@pytest.mark.parametrize('weight', [0., .1])
def test_zero_entropy_edge_removed_only_in_insertion_and_metrics_retained(prefix, weight):
    # First two 0dc81c0 regression families, extended for untouched modes.
    trainer = _trainer(insertion_config(prefix_mode=prefix, lambda_preference=1., lambda_entropy=weight))
    ce = torch.tensor(2., requires_grad=True)
    on = torch.tensor(-1., requires_grad=True)
    off = torch.tensor(-2., requires_grad=True)
    on_entropy = torch.tensor([.4], requires_grad=True)
    off_entropy = torch.tensor([.2], requires_grad=True)
    total, metrics = trainer._reduce_context_objective(
        ce, [on], [on_entropy],
        {(0, 'visual'): {'mean_logp': off, 'entropies': off_entropy}})
    expected = ce - torch.nn.functional.logsigmoid(.1 * (on - off)) + weight * .3
    torch.testing.assert_close(total, expected, atol=0, rtol=0)
    assert float(metrics['loss/entropy'].detach()) == pytest.approx(.3)
    assert metrics['loss/entropy'].requires_grad
    total.backward()
    if prefix == 'v_off_insertion' and weight == 0:
        assert on_entropy.grad is None and off_entropy.grad is None
    else:
        torch.testing.assert_close(on_entropy.grad, torch.tensor([weight]), atol=0, rtol=0)
        torch.testing.assert_close(off_entropy.grad, torch.tensor([-weight]), atol=0, rtol=0)
    assert on.grad < 0 and off.grad > 0 and ce.grad == 1


@pytest.mark.parametrize('debugging', [False, True])
def test_zero_coefficient_still_rejects_nan_entropy_with_debugging_on_or_off(debugging):
    trainer = _trainer(insertion_config(lambda_preference=1., lambda_entropy=0.))
    trainer.stage2_debugging = debugging
    with pytest.raises(ValueError, match='total loss must be finite'):
        trainer._reduce_context_objective(torch.tensor(2.), [torch.tensor(-1.)],
            [torch.tensor([float('nan')])],
            {(0, 'visual'): {'mean_logp': torch.tensor(-2.), 'entropies': torch.tensor([.2])}})


@pytest.fixture(scope='module')
def unchanged_methods():
    """Execute the two original methods from immutable d29bb35, not candidate code."""
    import stage2.trainer as trainer_module
    source = subprocess.run(
        ['git', 'show', 'd29bb358c0a7c185069ac47bcee5fbd31b0776ca:src/stage2/trainer.py'],
        cwd=Path(__file__).resolve().parents[2], check=True, capture_output=True, text=True,
    ).stdout
    original_class = next(node for node in ast.parse(source).body
                          if isinstance(node, ast.ClassDef) and node.name == 'Stage2Trainer')
    names = {'_compute_context_preference_loss', '_reduce_context_objective'}
    methods = [node for node in original_class.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(vars(trainer_module))
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<unchanged-d29bb35-trainer>', 'exec'), namespace)
    return {name: namespace[name] for name in names}


@pytest.mark.parametrize('mode', ['independent', 'cumulative'])
@pytest.mark.parametrize('debugging', [False, True])
def test_tiny_cpu_loss_metrics_and_all_gradients_match_unchanged_base(mode, debugging, unchanged_methods):
    with torch.random.fork_rng(devices=[]):
        baseline, ids, _, grid, _, pixels = native_case()
        baseline.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        candidate = copy.deepcopy(baseline)
        batch = training_batch(ids, grid, pixels)
        config = insertion_config(branch_mode=mode, lambda_preference=1., lambda_entropy=0.)
        baseline_trainer, candidate_trainer = _trainer(config), _trainer(config)
        for name, method in unchanged_methods.items():
            setattr(baseline_trainer, name, MethodType(method, baseline_trainer))
        rng = torch.get_rng_state().clone()
        observations = []
        for model, trainer in ((baseline, baseline_trainer), (candidate, candidate_trainer)):
            trainer.stage2_debugging = debugging
            torch.set_rng_state(rng)
            loss = trainer.compute_loss(model, batch, return_outputs=False)
            loss.backward()
            metrics = dict(trainer._step_preference_metric_totals)
            gradients = {name: None if p.grad is None else p.grad.detach().clone()
                         for name, p in model.named_parameters() if p.requires_grad}
            observations.append((loss.detach(), metrics, gradients, torch.get_rng_state().clone()))
        reference, actual = observations
        torch.testing.assert_close(actual[0], reference[0], atol=0, rtol=0)
        assert actual[1] == reference[1]
        assert torch.equal(actual[3], reference[3])
        measured = compare_numerical_tolerance(reference[1], actual[1], reference[2], actual[2])
        assert measured['passed'], measured
