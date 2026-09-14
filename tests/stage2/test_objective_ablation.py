from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn

from transformers.models.qwen2_5_vl import stage2_latent_rollout

from stage2.peft import (
    ALLOWED_STAGE2_TARGET_SUFFIXES,
    Stage2TuningConfig,
    build_stage2_tuned_model,
)

from .helpers import activate_stage2_adapter_deltas, tiny_stage2_qwen


def _two_layer_qwen(*, attn_implementation="eager"):
    model = tiny_stage2_qwen(attn_implementation=attn_implementation)
    model.model.layers.append(deepcopy(model.model.layers[0]))
    model.config.num_hidden_layers = 2
    return model


def _two_block_inputs():
    input_ids = torch.tensor(
        [[7, 8, 91, 90, 90, 92, 9, 10, 91, 90, 90, 92, 11, 12]]
    )
    return input_ids, torch.ones_like(input_ids)


def _dora_model(*, attn_implementation="eager"):
    model = _two_layer_qwen(attn_implementation=attn_implementation)
    wrapped = build_stage2_tuned_model(
        model,
        Stage2TuningConfig(
            method="dora",
            r=4,
            alpha=8,
            dropout=0.05,
            target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
        ),
    ).train()
    activate_stage2_adapter_deltas(wrapped, value=0.125)
    wrapped.config.use_cache = False
    return wrapped


def _forward_with_generated_delta(model, *, delta, stage2_ablation=None):
    input_ids, attention_mask = _two_block_inputs()
    original_replace = stage2_latent_rollout._replace_embedding

    def replace_generated_latent(embeds, position, value):
        if position in (3, 4):
            value = value + delta.to(device=value.device, dtype=value.dtype)
        return original_replace(embeds, position, value)

    with patch.object(
        stage2_latent_rollout,
        "_replace_embedding",
        side_effect=replace_generated_latent,
    ):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stage2_ablation=stage2_ablation,
            return_stage2_latent_states=True,
        )


def test_selected_generated_latents_affect_f_but_not_masked_m():
    """Catches missing q>=e to s<j<e blocking, including end-mediated leakage."""
    torch.manual_seed(17)
    from stage2.rng import capture_rng_state, replay_rng_state

    model = _dora_model()
    zero = torch.zeros(model.config.hidden_size)
    delta = torch.tensor([9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0])
    paired_state = capture_rng_state()

    with torch.no_grad(), replay_rng_state(paired_state):
        f_original = _forward_with_generated_delta(model, delta=zero)
    with torch.no_grad(), replay_rng_state(paired_state):
        f_changed = _forward_with_generated_delta(model, delta=delta)
    with torch.no_grad(), replay_rng_state(paired_state):
        m_original = _forward_with_generated_delta(
            model, delta=zero, stage2_ablation=(2, 5)
        )
    with torch.no_grad(), replay_rng_state(paired_state):
        m_changed = _forward_with_generated_delta(
            model, delta=delta, stage2_ablation=(2, 5)
        )

    assert not torch.allclose(f_original.logits[:, 5:8], f_changed.logits[:, 5:8])
    torch.testing.assert_close(m_original.logits[:, 5:8], m_changed.logits[:, 5:8])
    torch.testing.assert_close(m_original.logits[:, :3], f_original.logits[:, :3])
    for original, changed in zip(
        m_original.stage2_latent_states[2:],
        m_changed.stage2_latent_states[2:],
        strict=True,
    ):
        torch.testing.assert_close(original, changed)


def test_sdpa_blocks_selected_generated_latents_independently():
    """Catches SDPA silently dropping the explicit local intervention mask."""
    from stage2.rng import capture_rng_state, replay_rng_state

    torch.manual_seed(19)
    model = _two_layer_qwen(attn_implementation="sdpa").eval()
    zero = torch.zeros(model.config.hidden_size)
    delta = torch.tensor([9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0])
    state = capture_rng_state()
    with replay_rng_state(state):
        f_original = _forward_with_generated_delta(model, delta=zero)
    with replay_rng_state(state):
        f_changed = _forward_with_generated_delta(model, delta=delta)
    with replay_rng_state(state):
        m_original = _forward_with_generated_delta(
            model, delta=zero, stage2_ablation=(2, 5)
        )
    with replay_rng_state(state):
        m_changed = _forward_with_generated_delta(
            model, delta=delta, stage2_ablation=(2, 5)
        )

    assert not torch.allclose(f_original.logits[:, 5:8], f_changed.logits[:, 5:8])
    torch.testing.assert_close(m_original.logits[:, 5:8], m_changed.logits[:, 5:8])


@pytest.mark.parametrize(
    "stage2_ablation",
    [
        (True, 5),
        (2.0, 5),
        (3, 5),
        (2, 4),
        (2, 14),
        [2, 5],
    ],
)
def test_stage2_ablation_rejects_non_marker_or_malformed_spans(stage2_ablation):
    """Catches interventions that do not name one exact parsed nonempty block."""
    model = _two_layer_qwen().eval()
    input_ids, attention_mask = _two_block_inputs()

    with pytest.raises((ValueError, NotImplementedError)):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stage2_ablation=stage2_ablation,
        )


def test_stage2_ablation_rejects_a_latent_span_in_padding():
    """Catches an exact token span being accepted when its markers are padding."""
    model = _two_layer_qwen().eval()
    input_ids, attention_mask = _two_block_inputs()
    attention_mask[:, 2:6] = 0

    with pytest.raises(ValueError):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stage2_ablation=(2, 5),
        )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("_attn_implementation", "flash_attention_2"),
        ("use_sliding_window", True),
        ("attention_dropout", 0.1),
    ],
)
def test_stage2_ablation_rejects_unsupported_attention_modes(mutation, value):
    """Catches an intervention being advertised where mask semantics are unverified."""
    model = _two_layer_qwen().eval()
    setattr(model.config, mutation, value)
    input_ids, attention_mask = _two_block_inputs()

    with pytest.raises(NotImplementedError):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stage2_ablation=(2, 5),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stage2_ablation": (2, 5)},
        {"stage2_finite_ce": True},
    ],
)
def test_stage2_only_controls_are_rejected_outside_stage2(kwargs):
    """Catches Stage 2 objective controls leaking into the legacy Stage 1 path."""
    model = _two_layer_qwen().eval()
    model.config.stage = "stage1"
    input_ids, attention_mask = _two_block_inputs()

    with pytest.raises(ValueError):
        model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def test_stage2_finite_ce_returns_graph_connected_zero_only_when_shifted_labels_empty():
    """Catches NaN CE for an all-ignore shifted target without changing normal CE."""
    torch.manual_seed(23)
    model = _two_layer_qwen().train()
    input_ids, attention_mask = _two_block_inputs()
    labels = torch.full_like(input_ids, -100)
    labels[:, 0] = input_ids[:, 0]

    legacy = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    finite = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        stage2_finite_ce=True,
    )

    assert torch.isnan(legacy.loss)
    assert finite.loss.dtype == torch.float32
    assert finite.loss.item() == 0.0
    assert finite.loss.grad_fn is not None
    finite.loss.backward()
    assert model.lm_head.weight.grad is not None
    assert torch.count_nonzero(model.lm_head.weight.grad) == 0


def test_stage2_finite_ce_preserves_normal_legacy_ce():
    """Catches the opt-in finite fallback altering ordinary CE reduction."""
    torch.manual_seed(29)
    model = _two_layer_qwen().eval()
    input_ids, attention_mask = _two_block_inputs()
    labels = input_ids.masked_fill(input_ids == 90, -100)

    legacy = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    finite = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        stage2_finite_ce=True,
    )

    torch.testing.assert_close(finite.loss, legacy.loss)
    torch.testing.assert_close(finite.logits, legacy.logits)


class _RecordingVisual(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(
            torch.linspace(-1.0, 1.0, hidden_size), requires_grad=False
        )
        self.calls = []

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, pixel_values, grid_thw):
        self.calls.append((pixel_values.detach().clone(), grid_thw.detach().clone()))
        return self.weight.unsqueeze(0)


def test_ablation_preserves_inputs_images_positions_and_builds_exact_additive_mask():
    """Catches padding-as-ablation, changed RoPE, or loss of the original image input."""
    model = _two_layer_qwen().eval()
    model.visual = _RecordingVisual(model.config.hidden_size)
    input_ids = torch.tensor([[81, 80, 91, 90, 90, 92, 9, 10, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]])
    pixel_values = torch.tensor([[0.25, -0.5, 0.75]])
    image_grid_thw = torch.tensor([[1, 1, 1]])
    original = tuple(value.clone() for value in (input_ids, attention_mask, pixel_values, image_grid_thw))
    expected_positions, _ = model.get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    decoder_calls = []

    def record_decoder_call(_module, _args, kwargs):
        decoder_calls.append(
            (
                kwargs["attention_mask"].detach().clone(),
                kwargs["position_ids"].detach().clone(),
            )
        )

    handle = model.model.layers[0].register_forward_pre_hook(
        record_decoder_call, with_kwargs=True
    )
    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        stage2_ablation=(2, 5),
    )
    handle.remove()

    for actual, before in zip(
        (input_ids, attention_mask, pixel_values, image_grid_thw), original, strict=True
    ):
        torch.testing.assert_close(actual, before)
    assert len(model.visual.calls) == 1
    torch.testing.assert_close(model.visual.calls[0][0], pixel_values)
    torch.testing.assert_close(model.visual.calls[0][1], image_grid_thw)
    assert [positions.shape[-1] for _, positions in decoder_calls] == [3, 4, 9]
    for (_, actual_positions), length in zip(decoder_calls, (3, 4, 9), strict=True):
        torch.testing.assert_close(actual_positions, expected_positions[..., :length])

    final_mask = decoder_calls[-1][0]
    assert final_mask.shape == (1, 1, 9, 9)
    min_value = torch.finfo(final_mask.dtype).min
    assert torch.all(final_mask[0, 0, 5:8, 3:5] == min_value)
    assert torch.all(final_mask[0, 0, 4, 3:5] == 0)
    assert torch.all(final_mask[0, 0, :8, 8] == min_value)
    assert final_mask[0, 0, 7, 2] == 0


def _assert_rng_state_equal(actual, expected):
    torch.testing.assert_close(actual.cpu_state, expected.cpu_state)
    assert tuple(device for device, _ in actual.cuda_states) == tuple(
        device for device, _ in expected.cuda_states
    )
    for (_, actual_state), (_, expected_state) in zip(
        actual.cuda_states, expected.cuda_states, strict=True
    ):
        torch.testing.assert_close(actual_state, expected_state)


def test_rng_replay_is_nested_and_restores_the_outer_stream():
    """Catches paired replay consuming or overwriting the surrounding RNG stream."""
    from stage2.rng import capture_rng_state, replay_rng_state

    torch.manual_seed(31)
    paired = capture_rng_state()
    expected = torch.rand(5)
    outer = capture_rng_state()

    with replay_rng_state(paired):
        torch.testing.assert_close(torch.rand(5), expected)
        with replay_rng_state(paired):
            torch.testing.assert_close(torch.rand(5), expected)
    _assert_rng_state_equal(capture_rng_state(), outer)


def test_rng_capture_does_not_initialize_unused_cuda(monkeypatch):
    """Catches CPU objective setup eagerly initializing every visible CUDA device."""
    from stage2.rng import capture_rng_state

    if torch.cuda.is_initialized():
        pytest.skip("CUDA was already initialized by this process")

    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda *args, **kwargs: pytest.fail("uninitialized CUDA RNG was queried"),
    )
    state = capture_rng_state()

    assert state.cuda_states == ()
    assert not torch.cuda.is_initialized()


def test_rng_replay_rejects_an_incompatible_initialized_device_set():
    """Catches replay silently creating a CUDA stream absent from the outer scope."""
    from stage2.rng import RNGState, replay_rng_state

    if torch.cuda.is_initialized():
        pytest.skip("requires the CPU-only initialized-device-set case")
    incompatible = RNGState(
        cpu_state=torch.get_rng_state().clone(),
        cuda_states=((0, torch.get_rng_state().clone()),),
    )

    with pytest.raises(RuntimeError):
        with replay_rng_state(incompatible):
            pass
    assert not torch.cuda.is_initialized()


def test_train_mode_dora_no_intervention_replay_matches_f_and_restores_rng():
    """Catches no_grad/eval confusion or unmatched DoRA dropout in paired replay."""
    from stage2.rng import capture_rng_state, replay_rng_state

    torch.manual_seed(37)
    model = _dora_model()
    input_ids, attention_mask = _two_block_inputs()
    paired = capture_rng_state()
    f = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_stage2_latent_states=True,
    )
    after_f = capture_rng_state()

    with torch.no_grad(), replay_rng_state(paired):
        replayed = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_stage2_latent_states=True,
        )

    assert model.training
    assert not replayed.logits.requires_grad
    torch.testing.assert_close(replayed.logits, f.logits, atol=1e-6, rtol=1e-5)
    for actual, expected in zip(
        replayed.stage2_latent_states, f.stage2_latent_states, strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    _assert_rng_state_equal(capture_rng_state(), after_f)


def test_post_only_f_loss_reaches_generated_latent_graph():
    """Catches post predictions detaching the selected self-generated latents."""
    torch.manual_seed(41)
    model = _dora_model()
    input_ids, attention_mask = _two_block_inputs()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_stage2_latent_states=True,
        retain_stage2_latent_grads=True,
    )

    outputs.logits[:, 5:8].float().square().mean().backward()

    assert all(
        state.grad is not None
        and torch.isfinite(state.grad).all()
        and state.grad.abs().sum() > 0
        for state in outputs.stage2_latent_states[:2]
    )


def _post_gradient_snapshot(model, *, run_m, paired_state):
    from stage2.rng import replay_rng_state

    input_ids, attention_mask = _two_block_inputs()
    model.zero_grad(set_to_none=True)
    with replay_rng_state(paired_state):
        f = model(input_ids=input_ids, attention_mask=attention_mask)
        if run_m:
            with torch.no_grad(), replay_rng_state(paired_state):
                m = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    stage2_ablation=(2, 5),
                )
            assert not m.logits.requires_grad
        f.logits[:, 5:8].float().square().mean().backward()
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }


def test_m_before_f_backward_does_not_contaminate_checkpoint_recomputation():
    """Catches mutable intervention state or RNG leakage during F checkpoint backward."""
    from stage2.rng import capture_rng_state

    torch.manual_seed(43)
    initial = _dora_model()
    plain_only = deepcopy(initial)
    plain_with_m = deepcopy(initial)
    checkpoint_only = deepcopy(initial)
    checkpoint_with_m = deepcopy(initial)
    checkpoint_only.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    checkpoint_with_m.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    assert checkpoint_only.base_model.model.model.gradient_checkpointing
    assert checkpoint_with_m.base_model.model.model.gradient_checkpointing
    paired = capture_rng_state()

    snapshots = {
        "plain_only": _post_gradient_snapshot(
            plain_only, run_m=False, paired_state=paired
        ),
        "plain_with_m": _post_gradient_snapshot(
            plain_with_m, run_m=True, paired_state=paired
        ),
        "checkpoint_only": _post_gradient_snapshot(
            checkpoint_only, run_m=False, paired_state=paired
        ),
        "checkpoint_with_m": _post_gradient_snapshot(
            checkpoint_with_m, run_m=True, paired_state=paired
        ),
    }
    names = set(snapshots["plain_only"])
    assert names
    assert all(set(snapshot) == names for snapshot in snapshots.values())
    assert any("lora_magnitude_vector" in name for name in names)

    for candidate_name in ("plain_with_m", "checkpoint_only", "checkpoint_with_m"):
        max_error = 0.0
        for name in sorted(names):
            expected = snapshots["plain_only"][name]
            actual = snapshots[candidate_name][name]
            max_error = max(max_error, (actual - expected).abs().max().item())
            torch.testing.assert_close(
                actual,
                expected,
                atol=2e-6,
                rtol=2e-5,
                msg=lambda message, *, n=name, c=candidate_name: f"{c} {n}: {message}",
            )
        assert max_error <= 2e-6, f"{candidate_name} max gradient error {max_error}"
