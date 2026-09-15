"""Eliminate work with no loss contribution, without changing numerical paths."""
from collections import Counter

import pytest
import torch

from step_preference.equivalence import _trainer
from step_preference.spans import ReasoningStepSpan
from tests.step_preference.insertion_helpers import native_case, preference
from tests.step_preference.test_v_off_insertion import insertion_config, scorer


def training_batch(ids, grid, pixels):
    spans = (ReasoningStepSpan(0, 0, 5, 6, 8, 9, 11, None),
             ReasoningStepSpan(1, 9, 11, 12, 14, 15, 17, 0))
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
                pixel_values=pixels, image_grid_thw=grid, sample_ids=('no-op-audit',),
                reasoning_step_spans=spans,
                step_preference={'steps': tuple({'span': span} for span in spans)})


@pytest.mark.parametrize('mode', ['independent', 'cumulative'])
def test_loss_only_insertion_does_not_project_unused_clean_logits(mode):
    # The baseline projects 4 clean predictor rows AFTER the 16-row CE head,
    # despite using only evidence-on scores. This catches restoring that work.
    model, ids, _, grid, _, pixels = native_case()
    projected_rows = []
    handle = model.lm_head.register_forward_hook(
        lambda module, args, output: projected_rows.append(output.numel() // output.shape[-1]))
    loss = _trainer(insertion_config(branch_mode=mode)).compute_loss(
        model, training_batch(ids, grid, pixels), return_outputs=False)
    handle.remove()
    assert projected_rows == ([4, 2, 2, 16] if mode == 'independent' else [4, 4, 16])
    assert torch.isfinite(loss)


@pytest.mark.parametrize('weight', [0., .1])
def test_zero_weight_entropy_does_not_receive_backward_but_keeps_metric(weight):
    trainer = _trainer(insertion_config(lambda_preference=1., lambda_entropy=weight))
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
    total.backward()
    if weight == 0:
        assert on_entropy.grad is None and off_entropy.grad is None
    else:
        torch.testing.assert_close(on_entropy.grad, torch.tensor([weight]), atol=0, rtol=0)
        torch.testing.assert_close(off_entropy.grad, torch.tensor([-weight]), atol=0, rtol=0)
    assert on.grad < 0 and off.grad > 0 and ce.grad == 1


@pytest.mark.parametrize('mode', ['independent', 'cumulative'])
@pytest.mark.parametrize('backend', ['dense', 'flex'])
def test_mask_reuse_avoids_flex_rebuild_without_retaining_dense_masks(mode, backend, monkeypatch):
    from transformers.models.qwen2_5_vl import stage2_v_off_insertion as insertion
    if backend == 'flex' and not torch.cuda.is_available():
        pytest.skip('actual Flex mask replay requires CUDA')
    model, ids, _, grid, options, pixels = native_case()
    if backend == 'flex':
        model, ids, grid, pixels = model.cuda(), ids.cuda(), grid.cuda(), pixels.cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    options.update(branch_mode=mode, checkpoint=True, backend=backend)
    original = insertion._causal_mask
    counts = Counter()

    def observe(model, embeds, prefix_length, backend):
        counts[(prefix_length, embeds.shape[1])] += 1
        return original(model, embeds, prefix_length, backend)

    monkeypatch.setattr(insertion, '_causal_mask', observe)
    base = model.prepare_stage2_base_inputs_embeds(input_ids=ids, pixel_values=pixels, image_grid_thw=grid)
    values, _ = scorer(model, options, ids, base, grid)
    loss = preference({(row['kind'], row['step_id']): row['mean_logp'] for row in values})
    loss.backward()
    # Dense masks are quadratic and must retain their baseline release/rebuild
    # lifecycle. Only compact Flex masks may be retained across replay.
    assert counts and all(count == (1 if backend == 'flex' else 2)
                          for count in counts.values()), counts
    assert model.model.embed_tokens.weight.grad.isfinite().all()
