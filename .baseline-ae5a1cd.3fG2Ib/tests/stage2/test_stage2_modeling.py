import pytest
import torch

from step_preference.ablation import (
    StepAblationSpec,
    build_additive_step_attention_mask,
)
from tests.stage2.helpers import tiny_stage2_qwen


def _stage2_tokens():
    return torch.tensor([[1, 91, 90, 90, 92, 2, 3, 4]])


def test_stage2_ablation_uses_shifted_mask_only_on_final_decoder_call():
    """Catches a final-pass mask applied unshifted or leaked into latent rollout."""
    torch.manual_seed(31)
    model = tiny_stage2_qwen(attn_implementation="eager").eval()
    input_ids = _stage2_tokens()
    attention_mask = torch.ones_like(input_ids)
    selected_positions = torch.tensor([3, 4, 5, 6])
    spec = StepAblationSpec(
        modality="visual",
        target_start=5,
        target_end=7,
        source_start=1,
        source_end=4,
    )
    final_mask = build_additive_step_attention_mask(
        attention_mask,
        spec,
        dtype=model.dtype,
        device=model.device,
        attn_implementation="eager",
    )

    with torch.no_grad():
        clean = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_stage2_latent_states=True,
        )

    decoder_calls = []

    def record_decoder_call(_module, _args, kwargs):
        decoder_calls.append(
            (
                kwargs["attention_mask"].detach().clone(),
                kwargs["position_ids"].detach().clone(),
            )
        )

    handle = model.model.register_forward_pre_hook(record_decoder_call, with_kwargs=True)
    try:
        with torch.no_grad():
            ablated = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stage2_final_attention_mask=final_mask,
                stage2_logit_positions=selected_positions,
                stage2_selective_logits_only=True,
                return_stage2_latent_states=True,
            )
    finally:
        handle.remove()

    assert [mask.shape for mask, _ in decoder_calls] == [
        (1, 2),
        (1, 3),
        (1, 1, 8, 8),
    ]
    assert torch.equal(decoder_calls[0][0], attention_mask[:, :2])
    assert torch.equal(decoder_calls[1][0], attention_mask[:, :3])
    assert torch.equal(decoder_calls[2][0], final_mask)
    expected_positions = torch.arange(8).view(1, 1, 8).expand(3, 1, 8)
    assert torch.equal(decoder_calls[2][1], expected_positions)
    for clean_state, ablated_state in zip(
        clean.stage2_latent_states,
        ablated.stage2_latent_states,
        strict=True,
    ):
        torch.testing.assert_close(clean_state, ablated_state, rtol=0, atol=0)

    clean_selected = clean.logits.index_select(1, selected_positions)
    assert ablated.logits is None
    assert ablated.stage2_selected_logits.shape == clean_selected.shape
    torch.testing.assert_close(
        ablated.stage2_selected_logits[:, 0], clean_selected[:, 0], rtol=1e-6, atol=1e-7
    )
    assert not torch.allclose(
        ablated.stage2_selected_logits[:, 1], clean_selected[:, 1], rtol=1e-6, atol=1e-7
    )
    assert not torch.allclose(
        ablated.stage2_selected_logits[:, 2], clean_selected[:, 2], rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        ablated.stage2_selected_logits[:, 3], clean_selected[:, 3], rtol=1e-6, atol=1e-7
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for SDPA row handling")
def test_stage2_sdpa_passes_caller_owned_4d_mask_unchanged_to_decoder_layers():
    """Catches the CUDA padding workaround reopening a caller-blocked 4-D row."""
    model = tiny_stage2_qwen(attn_implementation="sdpa").cuda().eval()
    input_ids = _stage2_tokens().cuda()
    attention_mask = torch.ones_like(input_ids)
    final_mask = build_additive_step_attention_mask(
        attention_mask,
        StepAblationSpec("textual", 5, 6, 0, 5),
        dtype=model.dtype,
        device=model.device,
        attn_implementation="sdpa",
    )
    layer_masks = []

    def record_layer_mask(_module, _args, kwargs):
        if kwargs["attention_mask"] is not None and kwargs["attention_mask"].ndim == 4:
            layer_masks.append(kwargs["attention_mask"].detach().clone())

    handle = model.model.layers[0].register_forward_pre_hook(
        record_layer_mask, with_kwargs=True
    )
    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stage2_final_attention_mask=final_mask,
            )
    finally:
        handle.remove()

    assert len(layer_masks) == 1
    assert torch.equal(layer_masks[0], final_mask)


def test_stage2_selective_logits_select_hidden_rows_before_lm_head_and_keep_gradients():
    """Catches full-sequence LM-head projection in selective-only mode."""
    torch.manual_seed(37)
    model = tiny_stage2_qwen().eval()
    input_ids = _stage2_tokens()
    attention_mask = torch.ones_like(input_ids)
    selected_positions = torch.tensor([6, 1, 4])
    with torch.no_grad():
        full = model(input_ids=input_ids, attention_mask=attention_mask)

    seen_shapes = []
    handle = model.lm_head.register_forward_pre_hook(
        lambda _module, args: seen_shapes.append(tuple(args[0].shape))
    )
    try:
        selected = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stage2_logit_positions=selected_positions,
            stage2_selective_logits_only=True,
        )
    finally:
        handle.remove()

    assert seen_shapes == [(1, 3, model.config.hidden_size)]
    assert selected.logits is None
    torch.testing.assert_close(
        selected.stage2_selected_logits,
        full.logits.index_select(1, selected_positions),
        rtol=1e-6,
        atol=1e-7,
    )
    selected.stage2_selected_logits.square().mean().backward()
    lm_head_grad = model.lm_head.weight.grad
    decoder_grad = model.model.layers[0].self_attn.q_proj.weight.grad
    assert lm_head_grad is not None and torch.isfinite(lm_head_grad).all()
    assert decoder_grad is not None and torch.isfinite(decoder_grad).all()
    assert lm_head_grad.abs().sum() > 0
    assert decoder_grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (torch.tensor([], dtype=torch.long), "nonempty 1-D integer"),
        (torch.tensor([[2, 3]]), "nonempty 1-D integer"),
        (torch.tensor([2.0, 3.0]), "nonempty 1-D integer"),
        (torch.tensor([2, 2]), "unique"),
        (torch.tensor([-1]), "range"),
        (torch.tensor([8]), "range"),
    ],
)
def test_stage2_rejects_invalid_logit_positions(positions, message):
    """Catches ambiguous or invalid hidden-row selection."""
    model = tiny_stage2_qwen()
    input_ids = _stage2_tokens()

    with pytest.raises(ValueError, match=message):
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            stage2_logit_positions=positions,
            stage2_selective_logits_only=True,
        )


def test_stage2_selective_only_requires_positions():
    """Catches a selective-only call with no defined logits to return."""
    model = tiny_stage2_qwen()
    input_ids = _stage2_tokens()

    with pytest.raises(ValueError, match="requires stage2_logit_positions"):
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            stage2_selective_logits_only=True,
        )


def test_stage2_selective_only_rejects_full_ce_labels():
    """Catches silently dropping ordinary full-sequence CE labels."""
    model = tiny_stage2_qwen()
    input_ids = _stage2_tokens()

    with pytest.raises(ValueError, match="cannot compute full-sequence label loss"):
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids,
            stage2_logit_positions=torch.tensor([4]),
            stage2_selective_logits_only=True,
        )


@pytest.mark.parametrize("with_labels", [False, True])
def test_stage2_loss_only_and_selective_only_are_mutually_exclusive_before_projection(
    with_labels,
):
    """Catches contradictory projection modes doing decoder or LM-head work."""
    model = tiny_stage2_qwen()
    input_ids = _stage2_tokens()
    decoder_calls = []
    lm_head_calls = []
    decoder_handle = model.model.register_forward_pre_hook(
        lambda _module, _args: decoder_calls.append(1)
    )
    lm_head_handle = model.lm_head.register_forward_pre_hook(
        lambda _module, _args: lm_head_calls.append(1)
    )
    try:
        with pytest.raises(ValueError, match="mutually exclusive"):
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=input_ids if with_labels else None,
                stage2_loss_only=True,
                stage2_logit_positions=torch.tensor([4]),
                stage2_selective_logits_only=True,
            )
    finally:
        decoder_handle.remove()
        lm_head_handle.remove()

    assert decoder_calls == []
    assert lm_head_calls == []


@pytest.mark.parametrize(
    "final_mask",
    [
        torch.zeros(1, 8),
        torch.zeros(1, 1, 8, 8, dtype=torch.float64),
    ],
)
def test_stage2_rejects_invalid_final_attention_mask(final_mask):
    """Catches a malformed or wrong-compute-dtype mask crossing the decoder boundary."""
    model = tiny_stage2_qwen()
    input_ids = _stage2_tokens()

    with pytest.raises(ValueError, match="stage2_final_attention_mask"):
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            stage2_final_attention_mask=final_mask,
        )


def test_stage2_default_output_has_no_selected_logits():
    """Catches selective output changing the ordinary Stage 2 default path."""
    model = tiny_stage2_qwen()
    input_ids = _stage2_tokens()

    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    assert outputs.logits.shape == (1, 8, model.config.vocab_size)
    assert outputs.stage2_selected_logits is None
