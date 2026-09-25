import pytest
import torch
from evaluation.continuous_latent import (
    ContinuousLatentController,
    ContinuousLatentGenerationConfig,
)
from evaluation.visual_interventions import VisualIntervention, build_image_attention_mask
from stage2.inference import Stage2GenerationConfig, Stage2InferenceController
from tests.evaluation.test_continuous_latent import TOKENS, scripted_model


def test_latent_mask_blocks_only_latent_rows():
    m = build_image_attention_mask(torch.ones(1, 8), (1, 2), (5, 6), "latent_mask", torch.float32)
    allowed = m[0, 0] == 0
    assert not allowed[5, 1] and not allowed[6, 2] and allowed[4, 1] and allowed[7, 2]
    assert torch.equal(allowed, allowed.tril() & allowed)


def test_causal_control_is_plain_causal():
    m = build_image_attention_mask(torch.ones(1, 5), (1,), (3,), "causal_mask", torch.float32)
    assert torch.equal(m[0, 0] == 0, torch.ones(5, 5, dtype=torch.bool).tril())


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown"):
        build_image_attention_mask(torch.ones(1, 3), (), (), "global_mask", torch.float32)


def test_wrapper_injects_4d_mask():
    calls = []
    iv = VisualIntervention("latent_mask", image_token_id=80)
    iv.begin(torch.tensor([[1, 80, 2]]), None)
    wrapped = iv.wrap_decoder(lambda **kw: calls.append(kw) or "ok")
    assert wrapped(inputs_embeds=torch.zeros(1, 3, 4), attention_mask=torch.ones(1, 3),
                   position_ids=None, use_cache=False, return_dict=True) == "ok"
    assert calls[0]["attention_mask"].shape == (1, 1, 3, 3)


def test_intervention_wraps_rollout_decoder_and_registers_every_latent_span():
    """The rollout engine must run through the wrapped decoder, not the raw one."""
    model = scripted_model(
        latent_size=2, emitted_ids=[TOKENS.start_token_id, 6, TOKENS.start_token_id, 3]
    )
    input_ids = torch.tensor([[80, 1]])
    intervention = VisualIntervention("latent_mask", image_token_id=80)
    registered_spans = []
    original_register = intervention.register_latent_span
    intervention.register_latent_span = lambda span: (
        registered_spans.append(span),
        original_register(span),
    )[-1]

    result = ContinuousLatentController().generate(
        model=model,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        token_ids=TOKENS,
        generation_config=ContinuousLatentGenerationConfig(max_new_tokens=20, eos_token_id=3),
        intervention=intervention,
    )

    assert result.latent_block_count == 2
    # register_latent_span is called once per generated latent block.
    assert len(registered_spans) == 2
    # Forward calls attributed to latent rollout only occur when the rollout
    # engine invoked the wrapped decoder (role derives from wrap_decoder's
    # own bookkeeping), and every decoder call the underlying model actually
    # received carries the 4-D additive mask wrap_decoder builds.
    rollout_records = [r for r in intervention.forward_records if r["role"] == "latent_rollout"]
    assert rollout_records
    assert all(attn.ndim == 4 for _, attn, _ in model.model.calls)


def test_stage2_inference_controller_forwards_intervention_to_parent():
    """Stage2InferenceController must accept and pass intervention through, not swallow it."""
    model = scripted_model(latent_size=2, emitted_ids=[TOKENS.start_token_id])
    intervention = VisualIntervention("latent_mask", image_token_id=80)
    input_ids = torch.tensor([[80, 1]])

    result = Stage2InferenceController().generate(
        model=model,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=1),
        intervention=intervention,
    )

    # begin() is only invoked by ContinuousLatentController.generate when it
    # actually received this intervention instance, so a non-empty
    # forward_records here proves the argument reached the parent instead of
    # being dropped by the Stage2InferenceController override.
    assert intervention.forward_records
    assert result.latent_start_count == 1


def test_stage2_inference_controller_omitted_intervention_matches_none():
    """Callers that omit intervention (eval.py) must behave exactly as before."""
    model = scripted_model(latent_size=2, emitted_ids=[3])

    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=1, eos_token_id=3),
    )

    assert result.stop_reason == "eos"
