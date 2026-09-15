"""ON-only checkpoint selection must survive outer replay and preserve gradients."""
import copy
from collections import Counter

import pytest
import torch

from step_preference.equivalence import _trainer
from tests.step_preference.insertion_helpers import native_case
from tests.step_preference.test_v_off_insertion import insertion_config
from step_preference.spans import ReasoningStepSpan


@pytest.mark.parametrize('stride,selected', [(1, {0, 1, 2, 3}), (2, {0, 2}), (4, {0})])
def test_trainer_limits_only_on_checkpoint_layers_and_preserves_loss_gradients(stride, selected, monkeypatch):
    model, ids, _, grid, _, pixels = native_case(num_hidden_layers=4)
    baseline = copy.deepcopy(model)
    spans = (ReasoningStepSpan(0, 0, 5, 6, 8, 9, 11, None),
             ReasoningStepSpan(1, 9, 11, 12, 14, 15, 17, 0))
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
                 pixel_values=pixels, image_grid_thw=grid, sample_ids=('stride-test',),
                 reasoning_step_spans=spans,
                 step_preference={'steps': tuple({'span': s} for s in spans)})
    for m in (model, baseline):
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    routed = {'on': set(), 'clean_or_off': set()}
    executions = Counter()
    phase = ['forward']
    original = model.model._gradient_checkpointing_func
    layer_ids = {id(layer): i for i, layer in enumerate(model.model.layers)}

    def observe(function, *args, **kwargs):
        kind = 'on' if args[-1] is not None else 'clean_or_off'
        routed[kind].add(layer_ids[id(function.__self__)])
        return original(function, *args, **kwargs)

    monkeypatch.setattr(model.model, '_gradient_checkpointing_func', observe)
    handles = []
    for i, layer in enumerate(model.model.layers):
        def pre_hook(module, args, kwargs, index=i):
            prefix = args[-1] if len(args) > 10 else kwargs.get('stage2_prefix_key_value')
            executions[(phase[0], 'on' if prefix is not None else 'clean_or_off', index)] += 1
        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
    trainer = _trainer(insertion_config(on_layer_checkpoint_stride=stride))
    actual, output = trainer.compute_loss(model, batch, return_outputs=True)
    assert routed == {'on': selected, 'clean_or_off': {0, 1, 2, 3}}
    phase[0] = 'backward'
    actual.backward()
    for handle in handles:
        handle.remove()
    assert routed['on'] == selected  # Replay must not silently reset the policy.
    if stride != 1:
        assert executions['backward', 'on', 0] > executions['backward', 'on', 1]
    ref, ref_output = _trainer(insertion_config()).compute_loss(baseline, batch, return_outputs=True)
    ref.backward()
    torch.testing.assert_close(actual, ref, atol=0, rtol=0)
    torch.testing.assert_close(output.loss, ref_output.loss, atol=0, rtol=0)
    for (name, param), (ref_name, ref_param) in zip(model.named_parameters(), baseline.named_parameters()):
        assert name == ref_name
        if param.grad is None or ref_param.grad is None:
            assert param.grad is ref_param.grad is None
        else:
            torch.testing.assert_close(param.grad, ref_param.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize('bad', [0, 3, True, 2.0])
def test_invalid_stride_is_rejected(bad):
    with pytest.raises(ValueError, match='on_layer_checkpoint_stride'):
        insertion_config(on_layer_checkpoint_stride=bad)


def test_reduced_stride_cannot_silently_apply_to_clean_prefix():
    from step_preference.config import StepPreferenceConfig
    with pytest.raises(ValueError, match='v_off_insertion'):
        StepPreferenceConfig(on_layer_checkpoint_stride=2)
