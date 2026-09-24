import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from main import configure_stage2_model_backend
from stage2.peft import Stage2TuningConfig, build_stage2_tuned_model, resolve_stage2_qwen_model
from step_preference.equivalence import _trainer
from tests.stage2.helpers import activate_stage2_adapter_deltas, tiny_stage2_qwen
from tests.step_preference.test_context_preference import context_config
from tests.step_preference.test_context_sidecars import two_step_case
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentRolloutEngine, LatentSpan

class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6, 6)
        self.dropout = nn.Dropout(0.2)
        self.calls = 0

    def forward(self, *, inputs_embeds, **_kwargs):
        self.calls += 1
        hidden = self.dropout(torch.tanh(self.proj(inputs_embeds)))
        return SimpleNamespace(last_hidden_state=hidden)


def run_case(*, checkpointing):
    embedding = nn.Embedding(32, 6)
    decoder = TinyDecoder().train()
    output_head = nn.Linear(6, 32)
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    inputs = embedding(token_ids)
    inputs.retain_grad()
    result = LatentRolloutEngine().rollout(
        language_model=decoder,
        base_inputs_embeds=inputs,
        attention_mask=torch.ones(1, 10, dtype=torch.long),
        full_position_ids=torch.arange(10).view(1, 1, 10).expand(3, -1, -1),
        latent_spans=(LatentSpan(1, (2, 3), 4), LatentSpan(6, (7, 8), 9)),
        retain_latent_grads=True,
        latent_activation_checkpointing=checkpointing,
    )
    logits = output_head(result.final_inputs_embeds)
    targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    return SimpleNamespace(
        result=result,
        loss=loss.detach(),
        input_grad=inputs.grad.detach().clone(),
        parameter_grads={
            "embedding": embedding.weight.grad.detach().clone(),
            "decoder": tuple(
                parameter.grad.detach().clone()
                for parameter in decoder.parameters()
            ),
            "output_head": tuple(
                parameter.grad.detach().clone()
                for parameter in output_head.parameters()
            ),
        },
        calls=decoder.calls,
    )


def test_checkpointed_prefix_replay_preserves_outputs_loss_and_gradients():
    """Catches replay changing dropout RNG, latent connectivity, or decoder gradients."""
    torch.manual_seed(42)
    plain = run_case(checkpointing=False)
    torch.manual_seed(42)
    checked = run_case(checkpointing=True)

    torch.testing.assert_close(
        plain.result.final_inputs_embeds, checked.result.final_inputs_embeds
    )
    torch.testing.assert_close(plain.loss, checked.loss)
    torch.testing.assert_close(plain.input_grad, checked.input_grad)
    torch.testing.assert_close(
        plain.parameter_grads["embedding"],
        checked.parameter_grads["embedding"],
    )
    for component in ("decoder", "output_head"):
        for actual, expected in zip(
            plain.parameter_grads[component], checked.parameter_grads[component]
        ):
            torch.testing.assert_close(actual, expected)
    assert all(state.grad is not None for state in checked.result.latent_states)
    assert checked.calls > plain.calls == 4


def test_checkpointing_is_opt_in_and_skipped_under_no_grad():
    """Catches the default or inference path unexpectedly replaying decoder prefixes."""
    decoder = TinyDecoder().eval()
    inputs = torch.randn(1, 5, 6)
    kwargs = dict(
        language_model=decoder,
        base_inputs_embeds=inputs,
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        full_position_ids=torch.arange(5).view(1, 1, 5).expand(3, -1, -1),
        latent_spans=(LatentSpan(1, (2, 3), 4),),
    )
    with torch.no_grad():
        LatentRolloutEngine().rollout(**kwargs)
        LatentRolloutEngine().rollout(**kwargs, latent_activation_checkpointing=True)
    assert decoder.calls == 4


def test_native_and_latent_checkpointing_conflict():
    with pytest.raises(ValueError, match="native|gradient|checkpoint"):
        configure_stage2_model_backend(
            tiny_stage2_qwen(), gradient_checkpointing=True,
            latent_activation_checkpointing=True,
        )


def test_only_latent_decoder_calls_replay(monkeypatch):
    import transformers.models.qwen2_5_vl.stage2_shared_prefix as shared

    model, batch = two_step_case()
    model.train()
    configure_stage2_model_backend(model, gradient_checkpointing=False,
                                   latent_activation_checkpointing=True)
    model.config.stage2_sidecar_activation_checkpointing = False
    records = []
    phase = ["forward"]

    def observed(_module, _args, kwargs):
        kind = ("sidecar" if kwargs.get("stage2_prefix_key_values") is not None
                else "clean" if kwargs.get("stage2_capture_kv", False) else "latent")
        records.append((phase[0], kind, kwargs["inputs_embeds"].shape[1]))

    def forbidden_checkpoint(*args, **kwargs):
        pytest.fail("sidecar checkpoint was invoked")

    monkeypatch.setattr(shared, "checkpoint", forbidden_checkpoint)
    handle = model.model.register_forward_pre_hook(observed, with_kwargs=True)
    try:
        trainer = _trainer(context_config(target_scope="until_next_latent"))
        loss = trainer.compute_loss(model, batch)
        phase[0] = "backward"
        loss.backward()
    finally:
        handle.remove()
    assert not model.is_gradient_checkpointing
    forward_latent = [length for phase_name, kind, length in records
                      if phase_name == "forward" and kind == "latent"]
    backward = [(kind, length) for phase_name, kind, length in records
                if phase_name == "backward"]
    assert forward_latent and backward
    assert all(kind == "latent" for kind, _ in backward)
    assert sorted(length for _, length in backward) == sorted(forward_latent)


def test_visual_off_gradient_reaches_live_clean_kv(monkeypatch):
    import transformers.models.qwen2_5_vl.stage2_shared_prefix as shared

    model, batch = two_step_case()
    model.train()
    configure_stage2_model_backend(model, gradient_checkpointing=False,
                                   latent_activation_checkpointing=True)
    model.config.stage2_sidecar_activation_checkpointing = False
    captured = []
    original_score = shared.score_sidecars

    def observed_score(model, options, input_ids, clean_embeds, clean_positions, prefix_kv):
        assert options["checkpoint"] is False
        for key, value in prefix_kv:
            assert key.requires_grad and value.requires_grad
            key.retain_grad()
            value.retain_grad()
            captured.extend((key, value))
        return original_score(model, options, input_ids, clean_embeds, clean_positions, prefix_kv)

    monkeypatch.setattr(shared, "score_sidecars", observed_score)
    _, outputs = _trainer(context_config(target_scope="until_next_latent")).compute_loss(
        model, batch, return_outputs=True,
    )
    voff = [r["mean_logp"] for r in outputs.stage2_sidecar_results if r["kind"] == "visual_off"]
    assert voff
    (-torch.stack(voff).sum()).backward()
    assert captured
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in captured)
    assert any(torch.count_nonzero(t.grad).item() for t in captured)


@pytest.mark.parametrize("method", ["full", "dora"])
def test_selective_checkpoint_preserves_tiny_gradients(method):
    base, batch = two_step_case()
    if method == "dora":
        base = build_stage2_tuned_model(base, Stage2TuningConfig(
            method="dora", r=4, alpha=8, dropout=0.05,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        ))
        activate_stage2_adapter_deltas(base)
    records = []
    for enabled in (False, True):
        model = copy.deepcopy(base).train()
        qwen = resolve_stage2_qwen_model(model)
        configure_stage2_model_backend(qwen, gradient_checkpointing=False,
                                       latent_activation_checkpointing=enabled)
        qwen.config.stage2_sidecar_activation_checkpointing = False
        torch.manual_seed(42)
        loss, out = _trainer(context_config(target_scope="until_next_latent")).compute_loss(
            model, batch, return_outputs=True,
        )
        loss.backward()
        grads = {name: None if p.grad is None else p.grad.detach().clone()
                 for name, p in model.named_parameters() if p.requires_grad}
        records.append((loss.detach(), out.logits.detach(), grads))
    for index in (0, 1):
        torch.testing.assert_close(records[0][index], records[1][index], atol=1e-5, rtol=1e-4)
    assert records[0][2].keys() == records[1][2].keys()
    maxima = []
    for name, left in records[0][2].items():
        right = records[1][2][name]
        assert (left is None) == (right is None), name
        if left is not None:
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4, msg=name)
            maxima.append(float((left - right).abs().max()))
    assert maxima
    print({"method": method, "gradient_tensors": len(maxima), "max_abs_error": max(maxima)})
