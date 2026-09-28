"""Exact disabled/100% equivalence and selected-step denominators."""
import copy

import pytest
import torch

from tests.step_preference.test_dil_trainer import dil_case, dil_config, observed_forward, full_logit_oracle
from step_preference.equivalence import _trainer


@pytest.mark.parametrize("source", ["clean", "voff"])
def test_entropy_100_matches_disabled(source, monkeypatch):
    import step_preference.entropy_selection as selection
    def forbidden(*args, **kwargs):
        pytest.fail("100%/disabled must bypass selection")
    monkeypatch.setattr(selection, "select_entropy_masks", forbidden)
    model, batch = dil_case()
    reference = copy.deepcopy(model)
    actual, a, _ = observed_forward(model, batch, dil_config(preference_entropy_top_percent=100, preference_entropy_source=source))
    expected, b, _ = observed_forward(reference, batch, dil_config())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(a.logits, b.logits, atol=0, rtol=0)
    actual.backward(); expected.backward()
    for x, y in zip(model.parameters(), reference.parameters()):
        if y.grad is not None:
            torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)


def test_selection_concentrated_on_one_step_keeps_original_step_denominator():
    model, batch = dil_case(steps=3)
    config = dil_config(preference_entropy_top_percent=1.)
    actual, out, off = observed_forward(model, batch, config)
    oracle = full_logit_oracle(out, off, batch, config)
    torch.testing.assert_close(actual, oracle, atol=1e-6, rtol=1e-5)
    trainer = _trainer(config)
    trainer.compute_loss(model, batch)
    assert trainer._step_preference_metric_totals["stats/selected_steps"] == 1
    assert trainer._step_preference_metric_totals["stats/eligible_steps"] == 3


def test_dpo_unchanged():
    # The reference performs literal clean-KV replay, never calling the shared-prefix scorer.
    model, batch = dil_case()
    clone = copy.deepcopy(model)
    config = dil_config("dpo")
    actual, out, _ = observed_forward(model, batch, config)
    positions = torch.tensor([4, 11, 12])
    captured = []
    def request_capture(_module, _args, kwargs):
        if kwargs["inputs_embeds"].shape[1] == 15:
            kwargs["stage2_capture_kv"] = True
    def capture(_module, _args, kwargs, output):
        if kwargs.get("stage2_capture_kv"):
            captured.append((kwargs["position_ids"], output.stage2_key_values))
    pre = clone.model.register_forward_pre_hook(request_capture, with_kwargs=True)
    post = clone.model.register_forward_hook(capture, with_kwargs=True)
    try:
        ref = clone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"], use_cache=False, stage2_logit_positions=positions)
    finally:
        pre.remove()
        post.remove()
    clean_positions, prefix = captured[0]
    visible = torch.zeros((3, 18), dtype=torch.bool)
    visible[0, :2] = True
    visible[1:, :9] = True
    visible[0, 15] = True
    visible[1:, 16:] = torch.ones((2, 2), dtype=torch.bool).tril()
    mask = torch.zeros((1, 1, 3, 18)).masked_fill(~visible, torch.finfo(torch.float32).min)
    replay = clone.model(inputs_embeds=clone.model.embed_tokens(torch.tensor([[92, 92, 13]])),
                         position_ids=clean_positions.index_select(2, positions), attention_mask=mask,
                         stage2_prefix_key_values=prefix, use_cache=False, return_dict=True)
    off = clone.lm_head(replay.last_hidden_state)[0].float().log_softmax(-1).gather(
        1, torch.tensor([[12], [13], [13]])).squeeze(-1)
    c = ref.stage2_selected_logits[0].float().log_softmax(-1)
    values, cursor = [], 0
    for span in batch["reasoning_step_spans"]:
        n = span.target_text_end-span.target_text_start
        ids = batch["input_ids"][0, span.target_text_start:span.target_text_end]
        score = c[cursor:cursor+n].gather(1, ids[:, None]).mean()
        values.append(-torch.nn.functional.logsigmoid(config.beta*(score-off[cursor:cursor+n].mean())))
        cursor += n
    expected = ref.loss + torch.stack(values).mean()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(out.logits, ref.logits, atol=0, rtol=0)
    actual.backward(); expected.backward()
    for x, y in zip(model.parameters(), clone.parameters()):
        if y.grad is not None:
            torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)
