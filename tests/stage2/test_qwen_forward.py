from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from main import configure_stage2_model_backend
from .helpers import (
    ToyCausalLanguageModel,
    activate_stage2_adapter_deltas,
    tiny_stage2_qwen,
)
from stage2.peft import (
    ALLOWED_STAGE2_TARGET_SUFFIXES,
    Stage2TuningConfig,
    build_stage2_tuned_model,
    resolve_stage2_qwen_model,
)
from transformers.models.qwen2_5_vl import stage2_latent_rollout
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentRolloutEngine,
    LatentSpan,
)


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


def settings(method: str) -> Stage2TuningConfig:
    return Stage2TuningConfig(
        method=method,
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
    )


class RecordingRolloutEngine:
    def __init__(self):
        self.delegate = LatentRolloutEngine()
        self.calls = 0

    def rollout(self, **kwargs):
        self.calls += 1
        return self.delegate.rollout(**kwargs)


def _reference_full_copy_rollout(
    *,
    language_model,
    base_inputs_embeds,
    attention_mask,
    full_position_ids,
    latent_spans,
):
    """Independent oracle for the original full-sequence-copy recurrence."""
    current = base_inputs_embeds
    states = []
    prefix_lengths = []
    for span in latent_spans:
        for position in span.latent_positions:
            outputs = language_model(
                inputs_embeds=current[:, :position],
                attention_mask=attention_mask[:, :position],
                position_ids=full_position_ids[..., :position],
                use_cache=False,
                return_dict=True,
            )
            latent = outputs.last_hidden_state[:, -1, :]
            states.append(latent)
            prefix_lengths.append(position)
            current = torch.cat(
                (current[:, :position], latent.unsqueeze(1), current[:, position + 1 :]),
                dim=1,
            )
    return current, tuple(states), tuple(prefix_lengths)


def test_prefix_sized_rollout_matches_full_copy_oracle():
    """Catches prefix assembly changing the no-cache recurrence or its gradients."""
    torch.manual_seed(23)
    input_ids = torch.arange(11).unsqueeze(0)
    reference_embedding = nn.Embedding(32, 4)
    candidate_embedding = deepcopy(reference_embedding)
    reference_decoder = ToyCausalLanguageModel(4)
    candidate_decoder = deepcopy(reference_decoder)
    attention_mask = torch.ones_like(input_ids)
    full_position_ids = torch.arange(11).view(1, 1, 11).expand(3, -1, -1)
    latent_spans = (
        LatentSpan(1, (2, 3), 4),
        LatentSpan(6, (7, 8), 9),
    )

    reference_base = reference_embedding(input_ids)
    candidate_base = candidate_embedding(input_ids)
    reference_final, reference_states, reference_prefix_lengths = _reference_full_copy_rollout(
        language_model=reference_decoder,
        base_inputs_embeds=reference_base,
        attention_mask=attention_mask,
        full_position_ids=full_position_ids,
        latent_spans=latent_spans,
    )
    candidate = LatentRolloutEngine().rollout(
        language_model=candidate_decoder,
        base_inputs_embeds=candidate_base,
        attention_mask=attention_mask,
        full_position_ids=full_position_ids,
        latent_spans=latent_spans,
    )

    assembled = stage2_latent_rollout._assemble_latent_inputs(
        candidate_base,
        list(zip((2, 3, 7, 8), candidate.latent_states, strict=True)),
        stop=candidate_base.shape[1],
    )
    torch.testing.assert_close(assembled, candidate.final_inputs_embeds, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        candidate.final_inputs_embeds, reference_final, atol=1e-6, rtol=1e-5
    )
    assert candidate.prefix_lengths == reference_prefix_lengths == (2, 3, 7, 8)
    for candidate_state, reference_state in zip(
        candidate.latent_states, reference_states, strict=True
    ):
        torch.testing.assert_close(candidate_state, reference_state, atol=1e-6, rtol=1e-5)

    reference_outputs = reference_decoder(
        inputs_embeds=reference_final,
        attention_mask=attention_mask,
        position_ids=full_position_ids,
        use_cache=False,
        return_dict=True,
    )
    candidate_outputs = candidate_decoder(
        inputs_embeds=candidate.final_inputs_embeds,
        attention_mask=attention_mask,
        position_ids=full_position_ids,
        use_cache=False,
        return_dict=True,
    )
    reference_loss = reference_outputs.last_hidden_state.square().mean()
    candidate_loss = candidate_outputs.last_hidden_state.square().mean()
    torch.testing.assert_close(candidate_loss, reference_loss, atol=1e-6, rtol=1e-5)
    candidate_loss.backward()
    reference_loss.backward()

    assert [inputs.shape[1] for inputs, *_ in candidate_decoder.calls] == [2, 3, 7, 8, 11]
    assert [inputs.shape[1] for inputs, *_ in reference_decoder.calls] == [2, 3, 7, 8, 11]
    torch.testing.assert_close(
        candidate_embedding.weight.grad, reference_embedding.weight.grad, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        candidate_decoder.scale.grad, reference_decoder.scale.grad, atol=1e-6, rtol=1e-5
    )


@pytest.mark.parametrize("method", ["full", "lora", "dora"])
def test_stage2_loss_only_matches_full_loss_and_all_trainable_gradients(method):
    """Catches selective LM-head projection changing any trainable gradient."""
    torch.manual_seed(17)
    full = build_stage2_tuned_model(tiny_stage2_qwen(), settings(method)).train()
    selective = deepcopy(full).train()
    selective_qwen = resolve_stage2_qwen_model(selective)
    input_ids = torch.tensor([[1, 91, 90, 90, 92, 2, 3]])
    labels = input_ids.clone()
    labels[:, :2] = -100
    labels[input_ids == 90] = -100
    seen = []
    handle = selective_qwen.lm_head.register_forward_pre_hook(
        lambda _module, args: seen.append(tuple(args[0].shape))
    )

    full_output = full(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
    )
    selective_output = selective(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        stage2_loss_only=True,
    )
    handle.remove()
    full_output.loss.backward()
    selective_output.loss.backward()

    active = int((labels[:, 1:] != -100).sum())
    assert seen == [(active, selective_qwen.config.hidden_size)]
    assert selective_output.logits is None
    torch.testing.assert_close(
        selective_output.loss, full_output.loss, atol=1e-6, rtol=1e-5
    )
    for (full_name, full_parameter), (fast_name, fast_parameter) in zip(
        full.named_parameters(), selective.named_parameters(), strict=True
    ):
        assert full_name == fast_name
        if full_parameter.requires_grad:
            torch.testing.assert_close(
                fast_parameter.grad,
                full_parameter.grad,
                atol=1e-6,
                rtol=1e-5,
            )


@pytest.mark.parametrize(
    ("labels", "return_dict", "message"),
    [
        (None, True, "Stage 2 loss-only forward requires labels"),
        (
            torch.full((1, 7), -100, dtype=torch.long),
            True,
            "Stage 2 loss-only forward requires at least one supervised label",
        ),
        (
            torch.tensor([[-100, -100, -100, -100, 92, 2, 3]]),
            False,
            "Stage 2 loss-only forward requires return_dict=True",
        ),
    ],
)
def test_stage2_loss_only_rejects_invalid_contracts(labels, return_dict, message):
    """Catches loss-only calls returning undefined or ambiguous output contracts."""
    model = tiny_stage2_qwen()
    input_ids = torch.tensor([[1, 91, 90, 90, 92, 2, 3]])

    with pytest.raises(ValueError, match=message):
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            return_dict=return_dict,
            stage2_loss_only=True,
        )


def test_stage2_loss_only_default_call_retains_full_logits():
    """Catches the optimized projection becoming the ordinary forward default."""
    model = tiny_stage2_qwen()
    input_ids = torch.tensor([[1, 91, 90, 90, 92, 2, 3]])

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
    )

    assert outputs.logits.shape == (1, input_ids.shape[1], model.config.vocab_size)


def test_stage2_bfloat16_labels_preserve_full_fp32_logits_contract():
    """Catches labels casting only shifted logits while returning BF16 full logits."""
    model = tiny_stage2_qwen().to(dtype=torch.bfloat16)
    input_ids = torch.tensor([[1, 91, 90, 90, 92, 2, 3]])
    labels = input_ids.clone()
    labels[:, :2] = -100
    labels[input_ids == 90] = -100

    ordinary = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        stage2_loss_only=False,
    )
    loss_only = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        stage2_loss_only=True,
    )

    assert ordinary.logits.shape == (
        1,
        input_ids.shape[1],
        model.config.vocab_size,
    )
    assert ordinary.logits.dtype == torch.float32
    assert loss_only.logits is None


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


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"latent_size": 0}, "latent_size must be a positive integer"),
        ({"latent_size": True}, "latent_size must be a positive integer"),
        ({"latent_start_token_id": None}, "latent_start_token_id.*integer"),
        ({"latent_start_token_id": "91"}, "latent_start_token_id.*integer"),
        (
            {"latent_start_token_id": 90},
            "latent pad/start/end token IDs must be distinct",
        ),
        (
            {"latent_start_id": 99},
            "latent_start_token_id.*latent_start_id.*disagree",
        ),
    ],
)
def test_stage2_direct_forward_rejects_malformed_config(updates, message):
    """Catches malformed external Stage 2 configs bypassing composition checks."""
    model = tiny_stage2_qwen()
    for name, value in updates.items():
        setattr(model.config, name, value)

    input_ids = torch.tensor([[1, 91, 90, 90, 92, 2]])
    with pytest.raises(ValueError, match=message):
        model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


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
    configure_stage2_model_backend(
        checkpointed,
        gradient_checkpointing=True,
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
    if gradient_checkpointing:
        assert model.model._gradient_checkpointing_func.keywords == {
            "use_reentrant": False
        }


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


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_peft_stage2_multiblock_rollout_ignores_latent_pad_embeddings(method):
    """Catches PEFT recurrence using discrete latent-pad vectors in either block."""
    wrapped = build_stage2_tuned_model(tiny_stage2_qwen(), settings(method)).eval()
    inactive = deepcopy(wrapped).eval()
    activate_stage2_adapter_deltas(wrapped)
    qwen = resolve_stage2_qwen_model(wrapped)
    input_ids = torch.tensor([[1, 91, 90, 90, 92, 2, 91, 90, 90, 92, 3]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.masked_fill(input_ids == 90, -100)
    inactive_outputs = inactive(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_stage2_latent_states=True,
    )
    projection_calls = []
    decoder_inputs = []

    def record_decoder_inputs(_module, _args, kwargs):
        decoder_inputs.append(kwargs["inputs_embeds"].detach().clone())

    projection_handle = qwen.model.layers[0].self_attn.q_proj.register_forward_hook(
        lambda *unused: projection_calls.append(1)
    )
    decoder_handle = qwen.model.register_forward_pre_hook(
        record_decoder_inputs, with_kwargs=True
    )
    reference = wrapped(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_stage2_latent_states=True,
    )
    projection_handle.remove()
    decoder_handle.remove()
    with torch.no_grad():
        qwen.get_input_embeddings().weight[90].fill_(10_000)
    modified = wrapped(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_stage2_latent_states=True,
    )

    assert reference.loss.isfinite()
    assert modified.loss.isfinite()
    assert len(projection_calls) == 5
    assert [value.shape[1] for value in decoder_inputs] == [2, 3, 7, 8, 11]
    assert len(reference.stage2_latent_states) == len(
        modified.stage2_latent_states
    ) == 4
    assert any(
        not torch.allclose(inactive_state, active_state)
        for inactive_state, active_state in zip(
            inactive_outputs.stage2_latent_states,
            reference.stage2_latent_states,
            strict=True,
        )
    )
    torch.testing.assert_close(
        decoder_inputs[2][:, 2], reference.stage2_latent_states[0]
    )
    torch.testing.assert_close(
        decoder_inputs[2][:, 3], reference.stage2_latent_states[1]
    )
    for reference_state, modified_state in zip(
        reference.stage2_latent_states,
        modified.stage2_latent_states,
        strict=True,
    ):
        torch.testing.assert_close(reference_state, modified_state)


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
