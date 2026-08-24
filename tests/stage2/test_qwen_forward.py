from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from main import configure_stage2_model_backend
from helpers import ToyCausalLanguageModel, tiny_stage2_qwen
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentRolloutEngine


class OutputAwareToyCausalLanguageModel(ToyCausalLanguageModel):
    """Toy decoder that mirrors Qwen's config-default output flags."""

    def __init__(self, hidden_size, *, output_hidden_states=False, output_attentions=False):
        super().__init__(hidden_size)
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        position_ids,
        use_cache,
        return_dict,
        generate_mode=False,
        output_hidden_states=None,
        output_attentions=None,
    ):
        outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            return_dict=return_dict,
            generate_mode=generate_mode,
        )
        output_hidden_states = (
            self.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        output_attentions = self.output_attentions if output_attentions is None else output_attentions
        return SimpleNamespace(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=(inputs_embeds, outputs.last_hidden_state) if output_hidden_states else None,
            attentions=(torch.zeros(1, device=inputs_embeds.device),) if output_attentions else None,
        )


class CountingVisual(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.calls = 0
        self.grad_enabled = []

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, pixel_values, grid_thw):
        self.calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.weight.unsqueeze(0).expand(grid_thw.shape[0], -1)


class PixelFreeToyCausalLanguageModel(ToyCausalLanguageModel):
    """Records only kwargs the Stage 2 wrapper sends to the decoder."""

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        self.extra_kwargs = []

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        position_ids,
        use_cache,
        return_dict,
        generate_mode=False,
        **kwargs,
    ):
        self.extra_kwargs.append(kwargs)
        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            return_dict=return_dict,
            generate_mode=generate_mode,
        )


def stage2_scaffold():
    input_ids = torch.tensor([[1, 81, 80, 80, 80, 80, 2, 91, 90, 90, 92, 3]])
    attention_mask = torch.ones_like(input_ids)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    return input_ids, attention_mask, image_grid_thw


class RecordingRolloutEngine:
    def __init__(self):
        self.delegate = LatentRolloutEngine()
        self.calls = 0

    def rollout(self, **kwargs):
        self.calls += 1
        return self.delegate.rollout(**kwargs)


def test_stage2_freezes_full_visual_module_at_composition():
    model = tiny_stage2_qwen()

    assert all(not parameter.requires_grad for parameter in model.visual.parameters())


def test_stage2_training_uses_model_owned_rollout_engine():
    model = tiny_stage2_qwen()
    model.model = OutputAwareToyCausalLanguageModel(8)
    engine = RecordingRolloutEngine()
    model.stage2_rollout_engine = engine
    input_ids, attention_mask, image_grid_thw = stage2_scaffold()

    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
    )

    assert engine.calls == 1


def _run_tiny_stage2_train_step(model):
    input_ids = torch.tensor([[1, 91, 90, 90, 92, 2, 3]])
    labels = input_ids.masked_fill(input_ids == 90, -100)
    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        return_stage2_latent_states=True,
        retain_stage2_latent_grads=True,
    )
    outputs.loss.backward()
    return outputs


def test_stage2_gradient_checkpointing_preserves_loss_latents_and_gradients():
    """Catches checkpointing that detaches or numerically changes the latent recurrence."""
    torch.manual_seed(7)
    plain = tiny_stage2_qwen().train()
    checkpointed = deepcopy(plain).train()
    plain.config.use_cache = False
    checkpointed.config.use_cache = False
    checkpointed.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    plain_outputs = _run_tiny_stage2_train_step(plain)
    checkpointed_outputs = _run_tiny_stage2_train_step(checkpointed)

    assert torch.isfinite(plain_outputs.loss)
    assert torch.isfinite(checkpointed_outputs.loss)
    assert torch.allclose(
        plain_outputs.loss,
        checkpointed_outputs.loss,
        atol=1e-6,
        rtol=1e-5,
    )
    assert len(plain_outputs.stage2_latent_states) == len(
        checkpointed_outputs.stage2_latent_states
    ) == 2
    for plain_state, checkpointed_state in zip(
        plain_outputs.stage2_latent_states,
        checkpointed_outputs.stage2_latent_states,
        strict=True,
    ):
        assert torch.allclose(plain_state, checkpointed_state, atol=1e-6, rtol=1e-5)
        assert plain_state.grad is not None and torch.isfinite(plain_state.grad).all()
        assert checkpointed_state.grad is not None and torch.isfinite(
            checkpointed_state.grad
        ).all()

    plain_grad = plain.model.layers[0].self_attn.q_proj.weight.grad
    checkpointed_grad = checkpointed.model.layers[0].self_attn.q_proj.weight.grad
    assert plain_grad is not None and torch.isfinite(plain_grad).all()
    assert checkpointed_grad is not None and torch.isfinite(checkpointed_grad).all()
    assert plain_grad.abs().sum() > 0
    assert checkpointed_grad.abs().sum() > 0
    assert torch.allclose(plain_grad, checkpointed_grad, atol=1e-6, rtol=1e-5)
    assert plain_outputs.past_key_values is None
    assert checkpointed_outputs.past_key_values is None


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_stage2_backend_configuration_controls_cache_and_checkpointing(
    gradient_checkpointing,
):
    """Catches cache activation or checkpointing that ignores the Stage 2 flag."""
    model = tiny_stage2_qwen()
    model.config.use_cache = True
    if not gradient_checkpointing:
        model.gradient_checkpointing_enable()

    configure_stage2_model_backend(
        model,
        gradient_checkpointing=gradient_checkpointing,
    )

    assert model.config.use_cache is False
    assert model.is_gradient_checkpointing is gradient_checkpointing


def test_stage2_encodes_frozen_visual_once_and_backpropagates_through_lm():
    model = tiny_stage2_qwen()
    model.model = PixelFreeToyCausalLanguageModel(8)
    model.visual = CountingVisual(8)
    model.visual.requires_grad_(False)
    input_ids = torch.tensor([[81, 80, 1, 91, 90, 90, 92, 2]])
    labels = input_ids.clone()
    labels[:, :3] = -100
    labels[input_ids == 90] = -100

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        pixel_values=torch.ones(1, 3),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
    )
    outputs.loss.backward()

    assert model.visual.calls == 1
    assert model.visual.grad_enabled == [False]
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.visual.parameters())
    assert model.model.scale.grad is not None
    expected_loss = torch.nn.functional.cross_entropy(
        outputs.logits[..., :-1, :].contiguous().view(-1, model.config.vocab_size),
        labels[..., 1:].contiguous().view(-1),
    )
    assert torch.allclose(outputs.loss, expected_loss)
    assert all(
        not {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}.intersection(kwargs)
        for kwargs in model.model.extra_kwargs
    )


def test_stage2_rejects_image_token_feature_count_mismatch():
    model = tiny_stage2_qwen()
    model.visual = CountingVisual(8)

    with pytest.raises(ValueError, match="Stage 2 image token/feature count mismatch"):
        model(
            input_ids=torch.tensor([[81, 80, 80, 1, 91, 90, 90, 92, 2]]),
            attention_mask=torch.ones(1, 9, dtype=torch.long),
            pixel_values=torch.ones(1, 3),
            image_grid_thw=torch.tensor([[1, 1, 1]]),
        )


def test_stage2_forward_slices_one_full_mrope_tensor(monkeypatch):
    model = tiny_stage2_qwen()
    recorder = OutputAwareToyCausalLanguageModel(8)
    model.model = recorder
    input_ids, attention_mask, image_grid_thw = stage2_scaffold()
    expected_positions, _ = model.get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    rope_calls = 0
    original_get_rope_index = model.get_rope_index

    def record_get_rope_index(*args, **kwargs):
        nonlocal rope_calls
        rope_calls += 1
        return original_get_rope_index(*args, **kwargs)

    monkeypatch.setattr(model, "get_rope_index", record_get_rope_index)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        labels=input_ids.masked_fill(input_ids == 90, -100),
        return_stage2_latent_states=True,
    )

    assert len(outputs.stage2_latent_states) == 2
    assert rope_calls == 1
    assert len(recorder.calls) == 3
    assert torch.equal(recorder.calls[0][2], expected_positions[..., :8])
    assert torch.equal(recorder.calls[1][2], expected_positions[..., :9])
    assert torch.equal(recorder.calls[2][2], expected_positions)
    assert outputs.logits.shape == (1, input_ids.shape[1], model.config.vocab_size)
    assert outputs.loss.ndim == 0


def test_stage2_outputs_do_not_retain_latents_or_diagnostic_graphs_by_default():
    model = tiny_stage2_qwen()
    model.model = OutputAwareToyCausalLanguageModel(8)
    input_ids, attention_mask, image_grid_thw = stage2_scaffold()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        collect_stage2_diagnostics=True,
    )

    assert outputs.stage2_latent_states is None
    assert torch.equal(outputs.stage2_diagnostics["block_count"], torch.tensor(1))
    assert outputs.stage2_diagnostics["latent_l2_norms"].requires_grad is False
    assert outputs.stage2_diagnostics["adjacent_cosine_similarities"].requires_grad is False


@pytest.mark.parametrize(
    ("config_default", "call_override"),
    [(False, True), (True, False)],
)
def test_stage2_forward_honors_per_call_output_overrides(config_default, call_override):
    model = tiny_stage2_qwen()
    model.model = OutputAwareToyCausalLanguageModel(
        8,
        output_hidden_states=config_default,
        output_attentions=config_default,
    )
    input_ids, attention_mask, image_grid_thw = stage2_scaffold()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        output_hidden_states=call_override,
        output_attentions=call_override,
    )

    assert (outputs.hidden_states is not None) is call_override
    assert (outputs.attentions is not None) is call_override


def test_stage2_return_dict_false_preserves_qwen_order_before_opt_in_extras():
    model = tiny_stage2_qwen()
    model.model = OutputAwareToyCausalLanguageModel(8)
    input_ids, attention_mask, image_grid_thw = stage2_scaffold()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        labels=input_ids.masked_fill(input_ids == 90, -100),
        output_hidden_states=True,
        output_attentions=True,
        return_dict=False,
        return_stage2_latent_states=True,
        collect_stage2_diagnostics=True,
    )

    loss, logits, hidden_states, attentions, latent_states, diagnostics = outputs
    assert loss.ndim == 0
    assert logits.shape == (1, input_ids.shape[1], model.config.vocab_size)
    assert len(hidden_states) == model.config.num_hidden_layers + 1
    assert len(attentions) == model.config.num_hidden_layers
    assert len(latent_states) == model.config.latent_size
    assert diagnostics["block_count"].item() == 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("pixel_values_latent", torch.zeros(1, 1)),
        ("image_grid_thw_latent", torch.tensor([[1, 1, 1]])),
        ("image_out_mask", torch.ones(1, 12, dtype=torch.bool)),
        ("latent_hidden_states", torch.zeros(2, 8)),
    ],
)
def test_stage2_forward_rejects_legacy_helper_inputs(name, value):
    model = tiny_stage2_qwen()
    input_ids, attention_mask, image_grid_thw = stage2_scaffold()

    with pytest.raises(ValueError, match=name):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            **{name: value},
        )
