"""BF16 sparse input-row gradients for a separately embedded context group."""

import copy

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from step_preference.equivalence import _tiny_model
from step_preference.sidecar import build_context_sidecar_plan
from step_preference.spans import ReasoningStepSpan


def _case():
    torch.manual_seed(42)
    config = copy.deepcopy(_tiny_model().config)
    config.num_hidden_layers = 3
    config._attn_implementation = "sdpa"
    config.stage2_latent_activation_checkpointing = True
    model = type(_tiny_model())(config).bfloat16().train()
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=.05, use_dora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        trainable_token_indices={"model.embed_tokens": [90, 91, 92]},
        task_type="CAUSAL_LM",
    ))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.05)
    ids = torch.tensor([[11, 91, 90, 90, 92, 12, 13, 91, 90, 90, 92, 14, 15]])
    spans = (ReasoningStepSpan(0, 0, 1, 2, 4, 5, 7, None),
             ReasoningStepSpan(1, 5, 7, 8, 10, 11, 13, None))
    plan = build_context_sidecar_plan(
        clean_input_ids=ids, attention_mask=torch.ones_like(ids),
        spans=spans, visual_only=True,
    )
    labels = ids.clone()
    labels[:, :5] = -100
    labels[ids == 90] = -100
    return model, ids, labels, plan


def _literal_group(model, ids, labels):
    """Replay fixed END/text rows against clean K/V, without sidecar helpers."""
    decoder = model.base_model.model.model
    captured = []

    def request_capture(_module, _args, kwargs):
        if kwargs["inputs_embeds"].shape[1] == 13:
            kwargs["stage2_capture_kv"] = True

    def capture(_module, _args, kwargs, output):
        if kwargs.get("stage2_capture_kv"):
            captured.append((kwargs["position_ids"], output.stage2_key_values))

    before = decoder.register_forward_pre_hook(request_capture, with_kwargs=True)
    after = decoder.register_forward_hook(capture, with_kwargs=True)
    try:
        clean = model(input_ids=ids, labels=labels, use_cache=False,
                      stage2_loss_only=True,
                      stage2_logit_positions=torch.tensor([4, 5, 10, 11]))
    finally:
        before.remove()
        after.remove()
    assert len(captured) == 1
    positions, prefix = captured[0]
    # Two independent two-row branches: each sees its own causal suffix and
    # the clean prefix before END, excluding its current latent body.
    visible = torch.zeros((4, 17), dtype=torch.bool)
    visible[:2, :2] = True
    visible[2:, :8] = True
    visible[:2, 13:15] = torch.ones((2, 2), dtype=torch.bool).tril()
    visible[2:, 15:17] = torch.ones((2, 2), dtype=torch.bool).tril()
    mask = torch.zeros((1, 1, 4, 17), dtype=torch.bfloat16)
    mask.masked_fill_(~visible, torch.finfo(torch.bfloat16).min)
    embeds = decoder.embed_tokens(torch.tensor([[92, 12, 92, 14]]))
    replay = decoder(inputs_embeds=embeds,
                     position_ids=positions[:, :, [4, 5, 10, 11]],
                     attention_mask=mask, stage2_prefix_key_values=prefix,
                     use_cache=False, return_dict=True)
    logits = model.base_model.model.lm_head(replay.last_hidden_state)
    off = logits[0].float().log_softmax(-1).gather(
        -1, torch.tensor([[12], [13], [14], [15]])
    ).reshape(2, 2).mean(-1)
    return clean, off


def _objective(output, off):
    clean = output.stage2_selected_logits[0].float().log_softmax(-1).gather(
        -1, torch.tensor([[12], [13], [14], [15]])
    ).reshape(2, 2).mean(-1)
    return output.loss - F.logsigmoid(.1 * (clean - off)).mean()


@pytest.mark.parametrize("checkpoint", [False, True])
def test_bf16_sparse_embedding_gradients_match_literal_group_replay(checkpoint):
    """Reusing clean embeddings or splitting group lookups changes sparse gradients."""
    model, ids, labels, plan = _case()
    reference = copy.deepcopy(model)
    torch.manual_seed(173)
    actual = model(input_ids=ids, labels=labels, use_cache=False,
                   stage2_loss_only=True,
                   stage2_logit_positions=torch.tensor([4, 5, 10, 11]),
                   stage2_shared_prefix_inputs=dict(
                       plan=plan, backend="dense", checkpoint=checkpoint,
                       prefix_mode="clean_prefix", query_budget=1))
    actual_off = torch.stack([row["mean_logp"] for row in actual.stage2_sidecar_results])
    actual_loss = _objective(actual, actual_off)
    actual_loss.backward()
    torch.manual_seed(173)
    expected, expected_off = _literal_group(reference, ids, labels)
    expected_loss = _objective(expected, expected_off)
    expected_loss.backward()
    torch.testing.assert_close(actual_off, expected_off, atol=0, rtol=0)
    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
    sparse_name = "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta.default"
    actual_parameters = dict(model.named_parameters())
    expected_parameters = dict(reference.named_parameters())
    sparse_gradient = actual_parameters[sparse_name].grad
    assert sparse_gradient is not None and sparse_gradient.abs().max() > 0
    for name, parameter in actual_parameters.items():
        if not parameter.requires_grad:
            continue
        expected_gradient = expected_parameters[name].grad
        assert parameter.grad is not None and expected_gradient is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, expected_gradient,
                                   atol=0, rtol=0, msg=name)
