from types import SimpleNamespace

import pytest
import torch
from torch import nn

from evaluation.continuous_latent import (
    ContinuousLatentController,
    ContinuousLatentGenerationConfig,
)
from evaluation.visual_interventions import (
    VisualIntervention,
    build_image_attention_mask,
    identify_image_positions,
)
from tests.evaluation.test_continuous_latent import TOKENS, scripted_model
from tests.stage2.helpers import tiny_stage2_qwen
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentSpan


def _allowed(mask):
    return mask[0, 0].eq(0).tolist()


def test_mask_modes_match_independently_enumerated_allowed_edges():
    """Catches E3/E4 masking the wrong query rows or weakening causality."""
    validity = torch.ones(1, 7, dtype=torch.long)
    causal = [
        [True, False, False, False, False, False, False],
        [True, True, False, False, False, False, False],
        [True, True, True, False, False, False, False],
        [True, True, True, True, False, False, False],
        [True, True, True, True, True, False, False],
        [True, True, True, True, True, True, False],
        [True, True, True, True, True, True, True],
    ]
    e3 = [row[:] for row in causal]
    for query in (4, 5, 6):
        e3[query][1] = False
        e3[query][2] = False
    e4 = [row[:] for row in causal]
    for query in (0, 3, 4, 5, 6):
        e4[query][1] = False
        e4[query][2] = False

    assert _allowed(
        build_image_attention_mask(
            validity, (1, 2), (4, 5, 6), "baseline", torch.float32
        )
    ) == causal
    assert _allowed(
        build_image_attention_mask(
            validity, (1, 2), (4, 5, 6), "causal_mask_control", torch.float32
        )
    ) == causal
    assert _allowed(
        build_image_attention_mask(
            validity,
            (1, 2),
            (4, 5, 6),
            "latent_image_attention_block",
            torch.float32,
        )
    ) == e3
    assert _allowed(
        build_image_attention_mask(
            validity,
            (1, 2),
            (4, 5, 6),
            "global_image_attention_block",
            torch.float32,
        )
    ) == e4


@pytest.mark.parametrize("mode", ["baseline", "causal_mask_control"])
def test_mask_preserves_padding_and_sliding_window_constraints(mode):
    """Catches an intervention mask that restores padded or expired causal keys."""
    validity = torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.long)
    expected = [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [False, True, False, False, False],
        [False, False, False, True, False],
        [False, False, False, True, True],
    ]

    mask = build_image_attention_mask(
        validity,
        image_positions=(1,),
        latent_query_positions=(3,),
        mode=mode,
        dtype=torch.float32,
        sliding_window=2,
    )

    assert _allowed(mask) == expected
    assert torch.isfinite(mask).all()


def test_image_positions_are_physical_initial_sequence_indices():
    """Catches flattening batch coordinates or returning token IDs as positions."""
    assert identify_image_positions(torch.tensor([[80, 4, 80, 80, 7]]), 80) == (0, 2, 3)
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        identify_image_positions(torch.tensor([[80], [80]]), 80)


class _CaptureDecoder:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])


def _decoder_kwargs(length, *, token=0):
    return {
        "inputs_embeds": torch.full((1, length, 4), float(token)),
        "attention_mask": torch.ones(1, length, dtype=torch.long),
        "position_ids": torch.arange(length).view(1, 1, -1).expand(3, 1, -1),
        "use_cache": False,
        "return_dict": True,
    }


def test_causal_mask_control_retains_image_edges_and_logits():
    """Catches the neutral control removing an image edge or changing free logits."""
    decoder = _CaptureDecoder()
    policy = VisualIntervention("causal_mask_control", image_token_id=80)
    policy.begin(torch.tensor([[7, 80, 8]]), TOKENS)
    policy.register_latent_span(
        SimpleNamespace(start_position=2, latent_positions=(3,), end_position=4)
    )
    kwargs = _decoder_kwargs(5)

    policy.wrap_decoder(decoder)(**kwargs)

    expected = [[key <= query for key in range(5)] for query in range(5)]
    assert _allowed(decoder.calls[0]["attention_mask"]) == expected
    assert decoder.calls[0]["position_ids"] is kwargs["position_ids"]
    logits = torch.zeros(1, 128)
    assert policy.process_logits(
        torch.tensor([[7, 80, 8]]), logits, logits, TOKENS
    ) is logits


def test_wrapper_masks_registered_start_and_pad_producers_on_every_replay():
    """Catches E3 masking only the final call or masking END/text query rows."""
    decoder = _CaptureDecoder()
    intervention = VisualIntervention("latent_image_attention_block", image_token_id=80)
    intervention.begin(torch.tensor([[7, 80, 80, 8]]), TOKENS)
    intervention.register_latent_span(
        SimpleNamespace(start_position=4, latent_positions=(5, 6), end_position=7)
    )
    wrapped = intervention.wrap_decoder(decoder)

    wrapped(**_decoder_kwargs(6))
    wrapped(**_decoder_kwargs(8, token=1))

    first = _allowed(decoder.calls[0]["attention_mask"])
    replay = _allowed(decoder.calls[1]["attention_mask"])
    assert first[4][1:3] == [False, False]
    assert first[5][1:3] == [False, False]
    assert replay[4][1:3] == [False, False]
    assert replay[5][1:3] == [False, False]
    assert replay[6][1:3] == [False, False]
    assert replay[7][1:3] == [True, True]
    assert [record["prefix_length"] for record in intervention.forward_records] == [6, 8]


def test_wrapper_uses_only_initial_image_positions_and_rejects_cache():
    """Catches generated image-like IDs becoming keys or a foreign KV cache leaking in."""
    decoder = _CaptureDecoder()
    intervention = VisualIntervention("global_image_attention_block", image_token_id=80)
    intervention.begin(torch.tensor([[7, 80, 8]]), TOKENS)
    wrapped = intervention.wrap_decoder(decoder)

    wrapped(**_decoder_kwargs(5))
    allowed = _allowed(decoder.calls[0]["attention_mask"])
    assert allowed[4][1] is False
    assert allowed[4][3] is True
    with pytest.raises(ValueError, match="past_key_values"):
        wrapped(**_decoder_kwargs(5), past_key_values=(object(),))
    with pytest.raises(ValueError, match="use_cache=False"):
        wrapped(**{**_decoder_kwargs(5), "use_cache": True})


@pytest.mark.parametrize("mode", ["baseline", "image_replacement", "no_input_image"])
def test_baseline_wrapper_is_an_exact_identity_for_mask_logits_and_positions(mode):
    """Catches baseline instrumentation changing numerical decoder inputs or logits."""
    decoder = _CaptureDecoder()
    intervention = VisualIntervention(mode, image_token_id=80)
    intervention.begin(torch.tensor([[7, 80, 8]]), TOKENS)
    wrapped = intervention.wrap_decoder(decoder)
    kwargs = _decoder_kwargs(3)
    mask = kwargs["attention_mask"]
    positions = kwargs["position_ids"]

    wrapped(**kwargs)
    raw = torch.tensor([[0.0, 1.0, 2.0, 3.0] + [-4.0] * 89 + [4.0] + [-4.0] * 34])
    decision = raw.clone()
    result = intervention.process_logits(torch.tensor([[1, 2, 3]]), raw, decision, TOKENS)

    assert decoder.calls[0]["attention_mask"] is mask
    assert decoder.calls[0]["position_ids"] is positions
    assert result is decision
    torch.testing.assert_close(result, decision, rtol=0, atol=0)


class _StartAndTextHead(nn.Module):
    def forward(self, hidden):
        logits = hidden.new_full((*hidden.shape[:-1], 128), -20.0)
        logits[..., TOKENS.start_token_id] = 5.0
        logits[..., 11] = 4.0
        return logits


def test_e2_changes_only_start_and_controller_still_forces_complete_template():
    """Catches E2 changing ordinary logits or bypassing forced PAD/PAD/END insertion."""
    intervention = VisualIntervention("no_latent_execution", image_token_id=80)
    intervention.begin(torch.tensor([[80, 4]]), TOKENS)
    raw = torch.arange(128, dtype=torch.float32).view(1, -1)
    decision = raw.clone()

    adjusted = intervention.process_logits(torch.tensor([[80, 4]]), raw, decision, TOKENS)

    unchanged = torch.ones(128, dtype=torch.bool)
    unchanged[TOKENS.start_token_id] = False
    torch.testing.assert_close(adjusted[:, unchanged], decision[:, unchanged], rtol=0, atol=0)
    assert torch.isneginf(adjusted[0, TOKENS.start_token_id])

    model = scripted_model(latent_size=2, emitted_ids=[])
    model.lm_head = _StartAndTextHead()
    baseline = VisualIntervention("baseline", image_token_id=80)
    result = ContinuousLatentController().generate(
        model=model,
        input_ids=torch.tensor([[80, 4]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=ContinuousLatentGenerationConfig(max_new_tokens=4),
        intervention=baseline,
    )
    assert result.generated_suffix_ids == (91, 90, 90, 92)
    assert baseline.events[0]["pre_argmax_token_id"] == TOKENS.start_token_id
    assert baseline.events[0]["changed"] is False


@pytest.mark.parametrize(
    ("mode", "expected_allowed"),
    [
        (
            "latent_image_attention_block",
            [
                [[True, False], [True, True]],
                [[True, False, False], [True, True, False], [False, True, True]],
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [False, True, True, False],
                    [False, True, True, True],
                ],
                [
                    [True, False, False, False, False, False],
                    [True, True, False, False, False, False],
                    [False, True, True, False, False, False],
                    [False, True, True, True, False, False],
                    [False, True, True, True, True, False],
                    [True, True, True, True, True, True],
                ],
                [
                    [True, False, False, False, False, False],
                    [True, True, False, False, False, False],
                    [False, True, True, False, False, False],
                    [False, True, True, True, False, False],
                    [False, True, True, True, True, False],
                    [True, True, True, True, True, True],
                ],
            ],
        ),
        (
            "causal_mask_control",
            [
                [[True, False], [True, True]],
                [[True, False, False], [True, True, False], [True, True, True]],
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, False],
                    [True, True, True, True],
                ],
                [
                    [True, False, False, False, False, False],
                    [True, True, False, False, False, False],
                    [True, True, True, False, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, True, False],
                    [True, True, True, True, True, True],
                ],
                [
                    [True, False, False, False, False, False],
                    [True, True, False, False, False, False],
                    [True, True, True, False, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, True, False],
                    [True, True, True, True, True, True],
                ],
            ],
        ),
    ],
)
def test_controller_routes_normal_rollout_post_end_and_next_text_through_one_policy(
    mode, expected_allowed
):
    """Catches controller routes bypassing the selected mask, mRoPE, or no-cache policy."""
    model = scripted_model(
        latent_size=2,
        emitted_ids=[TOKENS.start_token_id, 11, 3],
    )
    input_ids = torch.tensor([[80, 7]])
    intervention = VisualIntervention(mode, image_token_id=80)
    decoder_calls = []

    def capture_decoder_call(_module, _args, kwargs):
        decoder_calls.append(
            {
                "attention_mask": kwargs["attention_mask"].clone(),
                "position_ids": kwargs["position_ids"].clone(),
                "use_cache": kwargs["use_cache"],
            }
        )

    handle = model.model.register_forward_pre_hook(
        capture_decoder_call, with_kwargs=True
    )
    try:
        result = ContinuousLatentController().generate(
            model=model,
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            token_ids=TOKENS,
            generation_config=ContinuousLatentGenerationConfig(
                max_new_tokens=5, eos_token_id=3
            ),
            return_latent_states=True,
            return_post_end_logits=True,
            intervention=intervention,
        )
    finally:
        handle.remove()

    assert result.generated_suffix_ids == (91, 90, 90, 92, 3)
    assert len(result.post_end_logits_by_block) == 1
    assert len(decoder_calls) == len(expected_allowed)
    for call, allowed in zip(decoder_calls, expected_allowed, strict=True):
        prefix_length = len(allowed)
        assert call["use_cache"] is False
        assert _allowed(call["attention_mask"]) == allowed
        torch.testing.assert_close(
            call["position_ids"], _position_ids(prefix_length), rtol=0, atol=0
        )
    assert [record["role"] for record in intervention.forward_records] == [
        "text",
        "latent_rollout",
        "latent_rollout",
        "text_or_post_end",
        "text_or_post_end",
    ]


def test_begin_resets_generation_local_events_records_spans_and_image_positions():
    """Catches per-example evidence and masks carrying into the next generation."""
    intervention = VisualIntervention("latent_image_attention_block", image_token_id=80)
    intervention.begin(torch.tensor([[80, 7]]), TOKENS)
    intervention.register_latent_span(
        SimpleNamespace(start_position=2, latent_positions=(3,), end_position=4)
    )
    intervention.process_logits(
        torch.tensor([[80, 7]]), torch.zeros(1, 128), torch.zeros(1, 128), TOKENS
    )
    intervention.wrap_decoder(_CaptureDecoder())(**_decoder_kwargs(4))

    intervention.begin(torch.tensor([[7, 80, 8]]), TOKENS)

    assert intervention.events == []
    assert intervention.forward_records == []
    decoder = _CaptureDecoder()
    intervention.wrap_decoder(decoder)(**_decoder_kwargs(3))
    assert _allowed(decoder.calls[0]["attention_mask"])[2][1] is True


def _position_ids(length):
    positions = torch.arange(length)
    return positions.view(1, 1, -1).expand(3, 1, -1)


def test_tiny_qwen_eager_reports_exact_zero_for_forbidden_image_attention():
    """Catches a numerically ineffective E4 additive mask in actual eager Qwen attention."""
    torch.manual_seed(42)
    model = tiny_stage2_qwen(attn_implementation="eager").float().eval()
    embeds = torch.randn(1, 7, model.config.hidden_size)
    mask = build_image_attention_mask(
        torch.ones(1, 7, dtype=torch.long),
        (1, 2),
        (),
        "global_image_attention_block",
        embeds.dtype,
    )

    with torch.no_grad():
        output = model.model(
            inputs_embeds=embeds,
            attention_mask=mask,
            position_ids=_position_ids(7),
            use_cache=False,
            return_dict=True,
            output_attentions=True,
            generate_mode=True,
        )

    attention = output.attentions[0]
    assert torch.count_nonzero(attention[:, :, (0, 3, 4, 5, 6), 1:3]).item() == 0


def test_tiny_qwen_noop_preserves_logits_and_explicit_mrope_positions_exactly():
    """Catches baseline instrumentation rebuilding masks or position IDs in actual Qwen."""
    torch.manual_seed(42)
    model = tiny_stage2_qwen(attn_implementation="sdpa").float().eval()
    embeds = torch.randn(1, 7, model.config.hidden_size)
    validity = torch.ones(1, 7, dtype=torch.long)
    positions = _position_ids(7)
    seen_positions = []

    def decoder(**kwargs):
        seen_positions.append(kwargs["position_ids"])
        return model.model(generate_mode=True, **kwargs)

    with torch.no_grad():
        direct = decoder(
            inputs_embeds=embeds,
            attention_mask=validity,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        )
        intervention = VisualIntervention("baseline", image_token_id=80)
        intervention.begin(torch.tensor([[7, 80, 80, 8, 9, 10, 11]]), TOKENS)
        wrapped = intervention.wrap_decoder(decoder)(
            inputs_embeds=embeds,
            attention_mask=validity,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        )

    torch.testing.assert_close(
        model.lm_head(direct.last_hidden_state),
        model.lm_head(wrapped.last_hidden_state),
        rtol=0,
        atol=0,
    )
    assert seen_positions == [positions, positions]


def test_tiny_qwen_sdpa_e4_blocks_image_perturbation_from_nonimage_recurrence():
    """Catches E4 applied only after image changes have entered recurrent hidden states."""
    torch.manual_seed(42)
    model = tiny_stage2_qwen(attn_implementation="sdpa").float().eval()
    embeds = torch.randn(1, 7, model.config.hidden_size)
    changed = embeds.clone()
    torch.manual_seed(42)
    changed[:, 1:3] += torch.randn_like(changed[:, 1:3]) * 3
    positions = _position_ids(7)
    validity = torch.ones(1, 7, dtype=torch.long)
    span = LatentSpan(start_position=3, latent_positions=(4, 5), end_position=6)

    def recurrent_decode(values, *, masked):
        decoder = lambda **kwargs: model.model(generate_mode=True, **kwargs)
        if masked:
            intervention = VisualIntervention(
                "global_image_attention_block", image_token_id=80
            )
            intervention.begin(torch.tensor([[7, 80, 80, 8, 9, 10, 11]]), TOKENS)
            intervention.register_latent_span(span)
            decoder = intervention.wrap_decoder(decoder)
        with torch.no_grad():
            rollout = model.stage2_rollout_engine.rollout(
                language_model=decoder,
                base_inputs_embeds=values,
                attention_mask=validity,
                full_position_ids=positions,
                latent_spans=(span,),
            )
            hidden = decoder(
                inputs_embeds=rollout.final_inputs_embeds,
                attention_mask=validity,
                position_ids=positions,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            return rollout.latent_states, hidden, model.lm_head(hidden)

    base_states, base_hidden, base_logits = recurrent_decode(embeds, masked=True)
    changed_states, changed_hidden, changed_logits = recurrent_decode(changed, masked=True)
    nonimage = torch.tensor([0, 3, 4, 5, 6])
    for base_state, changed_state in zip(base_states, changed_states, strict=True):
        torch.testing.assert_close(base_state, changed_state, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        base_hidden[:, nonimage], changed_hidden[:, nonimage], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        base_logits[:, nonimage], changed_logits[:, nonimage], rtol=1e-5, atol=1e-6
    )

    _, _, unmasked_logits = recurrent_decode(changed, masked=False)
    _, _, unmasked_base_logits = recurrent_decode(embeds, masked=False)
    assert (
        unmasked_logits[:, nonimage] - unmasked_base_logits[:, nonimage]
    ).abs().max().item() > 1e-5
