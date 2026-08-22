from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import main as main_module
import trainer_stage2 as trainer_stage2_module
from evidence_window.stage2_config import (
    Stage2EvidenceWindowConfig,
    Stage2PreferenceConfig,
    Stage2ReferenceConfig,
)
from evidence_window.stage2_layout import Stage2InterleavedLayout
from trainer_stage2 import CustomTrainerStage2


TOKEN_IDS = {
    "start": 90,
    "end": 91,
    "pad": 92,
    "padding": 0,
    "special": (0, 90, 91, 92, 99),
}


def _online_type():
    module = importlib.import_module("evidence_window.stage2_reference")
    reference_type = getattr(module, "OnlineFrozenStage2Reference", None)
    assert reference_type is not None, "online frozen Stage-2 reference is missing"
    return reference_type


class _Accelerator:
    def __init__(self):
        self.prepared = []
        self.process_index = 0

    def prepare_model(self, model, *, evaluation_mode=False):
        assert evaluation_mode is True
        assert model.training is False
        assert all(not parameter.requires_grad for parameter in model.parameters())
        self.prepared.append(model)
        return model


class _Output:
    def __init__(self, loss, logits, hidden):
        self.loss = loss
        self.logits = logits
        self.hidden_states = (hidden,)


class _TinyStage2Model(torch.nn.Module):
    def __init__(self, *, fail_reference=False):
        super().__init__()
        self.config = SimpleNamespace(stage="stage1")
        self.embedding = torch.nn.Embedding(128, 4)
        self.head = torch.nn.Linear(4, 128)
        self.calls = []
        self.fail_reference = fail_reference

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        attention_block_mask=None,
        attention_backend_override=None,
        output_hidden_states=False,
        return_dict=True,
        use_cache=None,
    ):
        del return_dict, use_cache
        if self.fail_reference and labels is None:
            raise RuntimeError("reference replay failed")
        self.calls.append(
            {
                "block": attention_block_mask,
                "backend": attention_backend_override,
                "grad_enabled": torch.is_grad_enabled(),
                "hidden": output_hidden_states,
            }
        )
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        hidden = hidden.cumsum(dim=1)
        if attention_block_mask is not None:
            blocked_queries = attention_block_mask.any(dim=-1).unsqueeze(-1)
            hidden = hidden - blocked_queries.to(hidden.dtype) * 0.25
        logits = self.head(hidden)
        loss = logits.sum() * 0.0
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return _Output(loss, logits, hidden)


def _one_edge_layout():
    return Stage2InterleavedLayout(
        evidence_token_mask=torch.tensor([[[1, 1, 1, 0, 0]]], dtype=torch.bool),
        window_token_mask=torch.tensor([[[0, 0, 0, 1, 1]]], dtype=torch.bool),
        evidence_valid=torch.tensor([[True]]),
        window_valid=torch.tensor([[True]]),
        causal_pair_mask=torch.tensor([[[True]]]),
    )


def _no_edge_layout():
    return Stage2InterleavedLayout(
        evidence_token_mask=torch.zeros((1, 0, 5), dtype=torch.bool),
        window_token_mask=torch.tensor([[[0, 0, 0, 1, 1]]], dtype=torch.bool),
        evidence_valid=torch.zeros((1, 0), dtype=torch.bool),
        window_valid=torch.tensor([[True]]),
        causal_pair_mask=torch.zeros((1, 0, 1), dtype=torch.bool),
    )


def _batch():
    return {
        "input_ids": torch.tensor([[90, 92, 91, 5, 6]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 5, 6]]),
        "ew_sample_indices": torch.tensor([17]),
        "ew_plan_types": (("latent", "text"),),
    }


def _online_config():
    return Stage2EvidenceWindowConfig(
        reference=Stage2ReferenceConfig(
            mode="online_frozen", checkpoint="/approved/stage1"
        ),
        preference=Stage2PreferenceConfig(enabled=True, lambda_ew=1.0, beta=0.4),
    )


def test_zero3_rejects_before_factory_or_model_allocation():
    """A ZeRO-3 run must fail before allocating the second full model."""
    calls = []

    def factory():
        calls.append("allocated")
        return _TinyStage2Model()

    with pytest.raises(ValueError, match="ZeRO-3"):
        _online_type().create(factory, _Accelerator(), zero_stage=3)

    assert calls == []


@pytest.mark.parametrize("zero_stage", [1, 2])
def test_independent_reference_is_stage2_frozen_eval_before_prepare(zero_stage):
    """Missing freeze-before-prepare can leak gradients or optimizer ownership."""
    policy = _TinyStage2Model()
    created = []

    def factory():
        model = _TinyStage2Model()
        created.append(model)
        return model

    accelerator = _Accelerator()
    reference = _online_type().create(factory, accelerator, zero_stage=zero_stage)

    assert reference.model is created[0]
    assert reference.model is not policy
    assert reference.model.config.stage == "stage2"
    assert reference.model.training is False
    assert accelerator.prepared == [reference.model]
    assert reference.witness.trainable_parameter_count == 0
    assert all(not parameter.requires_grad for parameter in reference.parameters())


def test_online_reference_uses_matched_sdpa_edges_and_dummy_slots_without_gradients():
    """Reference effects must use the same exact intervention and replay count as policy."""
    model = _TinyStage2Model()
    reference = _online_type().create(lambda: model, _Accelerator(), zero_stage=2)
    inputs = {
        "input_ids": torch.tensor([[90, 92, 91, 5, 6]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
    }
    labels = torch.tensor([[-100, -100, -100, 5, 6]])

    targets = reference.get_batch(
        torch.tensor([17]),
        _one_edge_layout(),
        model_inputs=inputs,
        labels=labels,
        contribution_reduction="mean",
        max_evidence_edges_per_window=0,
        edge_count_coordinator=lambda _count, _device: (3, 2),
    )

    assert targets.sample_indices == (17,)
    assert targets.pair_mask.tolist() == [[[True]]]
    assert torch.isfinite(targets.contributions).all()
    assert targets.contributions.requires_grad is False
    assert len(model.calls) == 3
    assert model.calls[0]["block"] is None
    assert model.calls[1]["block"].dtype is torch.bool
    assert not model.calls[2]["block"].any()
    assert all(call["backend"] == "sdpa" for call in model.calls)
    assert all(call["grad_enabled"] is False for call in model.calls)

    no_edge_model = _TinyStage2Model()
    no_edge_reference = _online_type().create(
        lambda: no_edge_model, _Accelerator(), zero_stage=1
    )
    no_edge_targets = no_edge_reference.get_batch(
        [23],
        _no_edge_layout(),
        model_inputs=inputs,
        labels=labels,
        contribution_reduction="mean",
        max_evidence_edges_per_window=0,
        edge_count_coordinator=lambda _count, _device: (3, 2),
    )
    assert no_edge_targets.pair_mask.numel() == 0
    assert len(no_edge_model.calls) == 3


def test_reference_checkpoint_loader_sets_stage2_bf16_flash_attention(monkeypatch):
    """Loading with Stage-1/helper semantics would make online targets incomparable."""
    loaded = []
    config = SimpleNamespace()

    monkeypatch.setattr(
        main_module.Qwen2_5_VLConfig,
        "from_pretrained",
        lambda checkpoint, **kwargs: loaded.append(("config", checkpoint, kwargs)) or config,
    )

    class FakeQwen:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            loaded.append(("model", checkpoint, kwargs))
            model = _TinyStage2Model()
            model.config = kwargs["config"]
            model.resized_to = None
            model.resize_token_embeddings = lambda size: setattr(model, "resized_to", size)
            return model

    monkeypatch.setattr(main_module, "Qwen2_5_VLForConditionalGeneration", FakeQwen)
    args = SimpleNamespace(cache_dir="/cache", compress_strategy="average", latent_size=8)

    model = main_module.load_stage2_online_reference_model(
        args,
        _online_config(),
        tokenizer_size=131,
        latent_token_ids={"pad": 92, "start": 90, "end": 91},
    )

    assert config.stage == "stage2"
    assert config.compress_strategy == "average"
    assert config.latent_size == 8
    assert config.latent_token_id == 92
    assert config.latent_start_id == 90
    assert config.latent_end_id == 91
    assert model.resized_to == 131
    model_call = next(item for item in loaded if item[0] == "model")
    assert model_call[1] == "/approved/stage1"
    assert model_call[2]["torch_dtype"] is torch.bfloat16
    assert model_call[2]["attn_implementation"] == "flash_attention_2"


def test_online_dpo_keeps_reference_fixed_and_policy_trainable(monkeypatch, tmp_path):
    """One optimizer step must update only policy while finite EW-DPO backpropagates."""
    accelerator = _Accelerator()

    def fake_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]
        instance.args = SimpleNamespace(output_dir=str(tmp_path))
        instance.state = SimpleNamespace(global_step=1, max_steps=1)
        instance.accelerator = accelerator

    monkeypatch.setattr(trainer_stage2_module.SFTTrainer, "__init__", fake_init)
    torch.manual_seed(7)
    policy = _TinyStage2Model()
    reference_model = _TinyStage2Model()
    snapshot = {
        name: parameter.detach().clone()
        for name, parameter in reference_model.named_parameters()
    }
    trainer = CustomTrainerStage2(
        model=policy,
        evidence_window_config=_online_config(),
        latent_token_ids=TOKEN_IDS,
        reference_model_factory=lambda: reference_model,
        deepspeed_zero_stage=2,
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    optimizer.zero_grad()
    loss = trainer.compute_loss(policy, _batch())
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())
    assert all(parameter.grad is None for parameter in reference_model.parameters())
    optimizer.step()

    assert all(
        torch.equal(snapshot[name], parameter)
        for name, parameter in reference_model.named_parameters()
    )
    assert all(call["grad_enabled"] is False for call in reference_model.calls)
    assert len(reference_model.calls) == len(policy.calls) - 1


def test_online_reference_failure_is_reported_before_policy_auxiliary_replay(
    monkeypatch, tmp_path
):
    """A failed provider must stop before policy ranks enter auxiliary collectives."""
    def fake_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]
        instance.args = SimpleNamespace(output_dir=str(tmp_path))
        instance.state = SimpleNamespace(global_step=1, max_steps=1)
        instance.accelerator = _Accelerator()

    monkeypatch.setattr(trainer_stage2_module.SFTTrainer, "__init__", fake_init)
    policy = _TinyStage2Model()
    trainer = CustomTrainerStage2(
        model=policy,
        evidence_window_config=_online_config(),
        latent_token_ids=TOKEN_IDS,
        reference_model_factory=lambda: _TinyStage2Model(fail_reference=True),
        deepspeed_zero_stage=1,
    )

    with pytest.raises(ValueError, match="reference.*failed"):
        trainer.compute_loss(policy, _batch())

    assert len(policy.calls) == 1
