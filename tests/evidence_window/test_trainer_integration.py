from __future__ import annotations

from types import MethodType
from types import SimpleNamespace

import pytest
import torch

from evidence_window.config import (
    EvidenceWindowConfig,
    PreferenceConfig,
    ReferenceConfig,
    RelationConfig,
    ResidualConfig,
)
from evidence_window.engine import EvidenceWindowEngine
from evidence_window.scores import reference_adjusted_effect


TOKEN_IDS = {
    "start": 90,
    "end": 91,
    "pad": 92,
    "padding": 0,
    "special": (0, 90, 91, 92, 99),
}


def _enabled_config() -> EvidenceWindowConfig:
    return EvidenceWindowConfig(
        reference=ReferenceConfig(mode="frozen_copy"),
        preference=PreferenceConfig(beta=0.7, lambda_ew=0.4),
        relation=RelationConfig(lambda_magnitude=0.3),
    )


def _assert_snapshot_equal(snapshot, module):
    assert snapshot.keys() == dict(module.named_parameters()).keys()
    assert all(
        torch.equal(before, dict(module.named_parameters())[name])
        for name, before in snapshot.items()
    )


def test_disabled_constructor_consumes_new_kwargs_without_building_runtime(
    tiny_trainer_factory,
    monkeypatch,
):
    import trainer as trainer_module

    def unexpected_runtime(*args, **kwargs):
        del args, kwargs
        raise AssertionError("disabled configuration constructed evidence-window runtime")

    monkeypatch.setattr(trainer_module, "EvidenceWindowEngine", unexpected_runtime)
    monkeypatch.setattr(trainer_module, "build_reference_provider", unexpected_runtime)
    trainer, _ = tiny_trainer_factory(
        evidence_window_config=EvidenceWindowConfig.disabled(),
        latent_token_ids=TOKEN_IDS,
    )

    assert trainer._ew_engine is None
    assert trainer._reference is None
    assert "evidence_window_config" not in trainer._parent_init_kwargs
    assert "latent_token_ids" not in trainer._parent_init_kwargs


def test_enabled_constructor_builds_native_runtime_after_parent_initialization(
    tiny_trainer_factory,
):
    trainer, model = tiny_trainer_factory(
        evidence_window_config=_enabled_config(),
        latent_token_ids=TOKEN_IDS,
    )

    assert isinstance(trainer._ew_engine, EvidenceWindowEngine)
    assert trainer._reference.model is not model
    assert trainer._reference.model in trainer.accelerator.prepared_models
    assert trainer._token_ids.latent_start == 90
    assert trainer._token_ids.latent_end == 91
    assert trainer._token_ids.latent_pad == 92
    assert trainer._token_ids.padding == 0
    assert trainer._token_ids.special == frozenset({0, 90, 91, 92, 99})


def test_enabled_loss_composes_with_original_loss_and_preserves_fixed_models(
    tiny_trainer_factory,
    legacy_latent_case,
    snapshot_parameters,
    monkeypatch,
):
    trainer, model = tiny_trainer_factory(
        evidence_window_config=_enabled_config(),
        latent_token_ids=TOKEN_IDS,
    )
    reference_before = snapshot_parameters(trainer._reference.model)
    ema_before = snapshot_parameters(trainer._ema.teacher)
    captured = []
    original = trainer._compute_evidence_window

    def capture_evidence_window(policy, context):
        result = original(policy, context)
        captured.append(result)
        return result

    monkeypatch.setattr(trainer, "_compute_evidence_window", capture_evidence_window)
    torch.manual_seed(1234)
    loss, outputs = trainer.compute_loss(
        model,
        dict(legacy_latent_case["inputs"]),
        return_outputs=True,
    )

    assert len(captured) == 1
    assert torch.allclose(
        loss,
        legacy_latent_case["recorded_legacy_loss"] + captured[0].total_loss,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        outputs.logits.sum(), legacy_latent_case["recorded_logits_sum"], atol=1e-6, rtol=1e-6
    )
    captured[0].total_loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert all(parameter.grad is None for parameter in trainer._reference.parameters())
    assert all(parameter.grad is None for parameter in trainer._ema.teacher.parameters())

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    _assert_snapshot_equal(reference_before, trainer._reference.model)
    _assert_snapshot_equal(ema_before, trainer._ema.teacher)
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(
        id(parameter) not in optimizer_parameter_ids
        for parameter in trainer._reference.parameters()
    )


def test_three_spans_add_one_loo_forward_per_enabled_role(
    tiny_trainer_factory,
):
    trainer, model = tiny_trainer_factory(
        evidence_window_config=_enabled_config(),
        latent_token_ids=TOKEN_IDS,
        sim_weight=0.0,
    )
    input_ids = torch.tensor(
        [[5, 90, 92, 91, 7, 90, 92, 91, 8, 90, 92, 91, 9, 0]]
    )
    inputs = {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "labels": torch.tensor(
            [[-100, -100, -100, -100, 7, -100, -100, -100, 8, -100, -100, -100, 9, -100]]
        ),
        "image_out_mask": input_ids.eq(92),
        "pixel_values_latent": torch.ones((3, 1)),
        "image_grid_thw_latent": torch.ones((3, 3), dtype=torch.long),
    }
    latent_values = torch.tensor(
        [
            [
                [0.25, -0.5, 0.75, 1.0],
                [-0.75, 0.5, 0.25, -0.25],
                [0.6, 0.2, -0.4, 0.8],
            ]
        ]
    )
    active_mask = input_ids.eq(92)

    def three_latents(_self, supplied_inputs, k, p):
        del supplied_inputs, k, p
        return latent_values.clone(), active_mask.clone()

    trainer._teacher_build_latents = MethodType(three_latents, trainer)
    model.reset_counts()
    trainer._reference.model.reset_counts()
    trainer._ema.teacher.reset_counts()

    trainer.compute_loss(model, inputs)

    assert model.full_forward_count == 1
    assert model.loo_forward_count == 3
    assert model.all_null_forward_count == 0
    assert trainer._reference.model.full_forward_count == 1
    assert trainer._reference.model.loo_forward_count == 3
    assert trainer._ema.teacher.full_forward_count == 1
    assert trainer._ema.teacher.loo_forward_count == 3


def test_condition_copy_is_local_and_preserves_fixed_token_history(
    tiny_trainer_factory,
    legacy_latent_case,
):
    config = EvidenceWindowConfig(
        reference=ReferenceConfig(mode="frozen_copy"),
        preference=PreferenceConfig(lambda_window=1.0),
    )
    trainer, model = tiny_trainer_factory(
        evidence_window_config=config,
        latent_token_ids=TOKEN_IDS,
        sim_weight=0.0,
    )
    caller_inputs = {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in legacy_latent_case["inputs"].items()
    }
    before = {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in caller_inputs.items()
    }
    calls = []
    original_forward = model.forward

    def recording_forward(_self, *args, **kwargs):
        calls.append(
            {
                name: value.clone() if torch.is_tensor(value) else value
                for name, value in kwargs.items()
            }
        )
        return original_forward(*args, **kwargs)

    model.forward = MethodType(recording_forward, model)
    trainer.compute_loss(model, caller_inputs)

    assert len(calls) == 2
    full_call, all_null_call = calls
    assert "labels" in full_call
    assert "labels" not in all_null_call
    assert "pixel_values_latent" not in all_null_call
    assert "image_grid_thw_latent" not in all_null_call
    assert torch.equal(all_null_call["input_ids"], before["input_ids"])
    assert torch.equal(all_null_call["attention_mask"], before["attention_mask"])
    assert torch.equal(all_null_call["position_ids"], before["position_ids"])
    assert torch.equal(all_null_call["image_out_mask"], before["image_out_mask"])
    assert all_null_call["latent_hidden_states"].shape == (1, 2, 4)
    assert torch.count_nonzero(all_null_call["latent_hidden_states"]) == 0
    assert all_null_call["return_dict"] is True
    assert all_null_call["output_hidden_states"] is False
    assert caller_inputs.keys() == before.keys()
    assert all(
        torch.equal(caller_inputs[name], value)
        for name, value in before.items()
        if torch.is_tensor(value)
    )


def test_enabled_no_latents_returns_original_ce_without_condition_forwards(
    tiny_trainer_factory,
    legacy_no_latent_case,
):
    trainer, model = tiny_trainer_factory(
        evidence_window_config=_enabled_config(),
        latent_token_ids=TOKEN_IDS,
    )

    def no_latents(_self, inputs, k, p):
        del k, p
        return None, torch.zeros_like(inputs["input_ids"], dtype=torch.bool)

    trainer._teacher_build_latents = MethodType(no_latents, trainer)
    model.reset_counts()
    trainer._reference.model.reset_counts()
    trainer._ema.teacher.reset_counts()
    loss = trainer.compute_loss(model, dict(legacy_no_latent_case["inputs"]))

    assert isinstance(trainer._ew_engine, EvidenceWindowEngine)
    assert torch.allclose(loss, legacy_no_latent_case["recorded_ce_loss"])
    assert model.forward_count == 1
    assert trainer._reference.model.forward_count == 0
    assert trainer._ema.teacher.forward_count == 0


def test_relation_only_empty_pair_set_keeps_original_loss_and_skips_ema_conditions(
    tiny_trainer_factory,
):
    config = EvidenceWindowConfig(
        relation=RelationConfig(lambda_magnitude=1.0),
    )
    trainer, model = tiny_trainer_factory(
        evidence_window_config=config,
        latent_token_ids=TOKEN_IDS,
        sim_weight=0.0,
    )
    input_ids = torch.tensor([[5, 6, 90, 92, 91, 0]])
    inputs = {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "labels": torch.tensor([[-100, 6, -100, -100, -100, -100]]),
        "image_out_mask": input_ids.eq(92),
    }
    latent_values = torch.tensor([[[0.25, -0.5, 0.75, 1.0]]])

    def one_latent(_self, supplied_inputs, k, p):
        del supplied_inputs, k, p
        return latent_values.clone(), input_ids.eq(92)

    trainer._teacher_build_latents = MethodType(one_latent, trainer)
    model.reset_counts()
    trainer._ema.teacher.reset_counts()
    loss = trainer.compute_loss(model, inputs)

    assert isinstance(trainer._ew_engine, EvidenceWindowEngine)
    assert torch.isfinite(loss)
    assert model.forward_count == 1
    assert trainer._ema.teacher.forward_count == 0
    assert trainer._reference is None


def test_residual_kd_rejects_any_output_head_bias_at_startup(tiny_trainer_factory):
    config = EvidenceWindowConfig(residual=ResidualConfig(lambda_logit=1.0))

    with pytest.raises(ValueError, match="bias-free"):
        tiny_trainer_factory(
            evidence_window_config=config,
            latent_token_ids=TOKEN_IDS,
            output_bias=True,
        )


@pytest.mark.parametrize("condition_logits_dtype", [torch.bfloat16, torch.float16])
def test_auxiliary_scores_normalize_labels_dependent_logits_to_float32(
    condition_logits_dtype,
    tiny_trainer_factory,
    legacy_latent_case,
    monkeypatch,
):
    try:
        torch.zeros((1, 2), dtype=condition_logits_dtype).log_softmax(-1)
    except RuntimeError:
        pytest.skip(f"CPU log_softmax does not support {condition_logits_dtype}")

    config = EvidenceWindowConfig(
        reference=ReferenceConfig(mode="frozen_copy"),
        preference=PreferenceConfig(beta=1.0, lambda_ew=1.0),
        relation=RelationConfig(lambda_magnitude=1.0),
    )
    trainer, model = tiny_trainer_factory(
        evidence_window_config=config,
        latent_token_ids=TOKEN_IDS,
        condition_logits_dtype=condition_logits_dtype,
        sim_weight=0.0,
    )
    outputs_by_condition = {}
    results = []
    original_compute = trainer._ew_engine.compute

    def recording_compute(layout, latent_values, policy_full, forward, **kwargs):
        outputs_by_condition[("policy", "full", None)] = policy_full

        def recording_forward(role, condition, values, need_hidden):
            output = forward(role, condition, values, need_hidden)
            outputs_by_condition[(role, condition.kind, condition.evidence_index)] = output
            return output

        result = original_compute(
            layout,
            latent_values,
            policy_full,
            recording_forward,
            **kwargs,
        )
        results.append(result)
        return result

    monkeypatch.setattr(trainer._ew_engine, "compute", recording_compute)
    trainer.compute_loss(model, dict(legacy_latent_case["inputs"]))

    assert len(results) == 1
    reference_full = outputs_by_condition[("reference", "full", None)].token_logps
    for evidence_index in (0, 1):
        token_margin = reference_adjusted_effect(
            outputs_by_condition[("policy", "full", None)].token_logps,
            outputs_by_condition[("policy", "loo", evidence_index)].token_logps,
            reference_full,
            outputs_by_condition[("reference", "loo", evidence_index)].token_logps,
        )
        assert torch.allclose(
            token_margin,
            torch.zeros_like(token_margin),
            atol=1e-7,
            rtol=0.0,
        )
    assert torch.allclose(
        results[0].ew_margins,
        torch.zeros_like(results[0].ew_margins),
        atol=1e-7,
        rtol=0.0,
    )
    assert {
        output.token_logps.dtype for output in outputs_by_condition.values()
    } == {torch.float32}
    assert {role for role, _, _ in outputs_by_condition} == {
        "policy",
        "reference",
        "ema",
    }


def test_main_wires_loaded_config_and_resolved_token_ids_into_trainer(
    monkeypatch,
    tmp_path,
):
    import main as main_module

    class FakeTokenizer:
        all_special_ids = [0, 190, 191, 192, 199]
        ids = {
            "<|latent_start|>": 190,
            "<|latent_end|>": 191,
            "<|latent_pad|>": 192,
            "<|endoftext|>": 0,
        }

        def add_tokens(self, tokens, special_tokens):
            assert special_tokens
            return len(tokens)

        def __call__(self, value, return_tensors):
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[self.ids[value]]])}

        def __len__(self):
            return 200

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

        def save_pretrained(self, path):
            self.saved_path = path

    class FakeModel:
        def __init__(self):
            self.visual = torch.nn.Linear(1, 1, bias=False)

        def resize_token_embeddings(self, size):
            assert size == 200

    class FakeTrainer:
        captured = None

        def __init__(self, **kwargs):
            type(self).captured = kwargs

        def train(self, resume_from_checkpoint):
            assert resume_from_checkpoint is None

        def is_world_process_zero(self):
            return False

        def save_model(self, path):
            self.saved_path = path

    args = SimpleNamespace(
        ew_config=None,
        log_file=str(tmp_path / "train.log"),
        cache_dir=str(tmp_path / "cache"),
        model="tiny",
        compress_strategy="none",
        latent_size=2,
        stage="stage1",
        task="tiny",
        data_path=str(tmp_path / "data.jsonl"),
        use_lora=False,
        save_model_path=str(tmp_path / "output"),
        epochs=1,
        gradient_accumulation_steps=1,
        warm_up_steps=0,
        save_steps=1,
        max_seq_length_train=32,
        deepspeed_config="zero.json",
        sim_weight=1.0,
        ema_tau=0.9,
        coverage_p=0.8,
        image_pool_k=2,
        helper_group_L=4,
        ce_weight=1.0,
    )
    processor = FakeProcessor()
    monkeypatch.setattr(main_module, "seed_everything", lambda seed: None)
    monkeypatch.setattr(main_module, "get_args", lambda: args)
    monkeypatch.setattr(
        main_module.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: processor,
    )
    monkeypatch.setattr(
        main_module.Qwen2_5_VLConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        main_module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(main_module, "task_preporcess_config", {"tiny": lambda sample: sample})
    monkeypatch.setattr(main_module, "load_jsonl_dataset", lambda path: [])
    monkeypatch.setattr(
        main_module,
        "SFTConfig",
        lambda **kwargs: SimpleNamespace(output_dir=kwargs["output_dir"]),
    )
    monkeypatch.setattr(main_module, "CustomTrainerStage1", FakeTrainer)
    monkeypatch.setattr(main_module, "get_last_checkpoint", lambda path: None)
    monkeypatch.setattr(main_module.torch.cuda, "current_device", lambda: 0)

    main_module.main_train()

    assert isinstance(
        FakeTrainer.captured["evidence_window_config"], EvidenceWindowConfig
    )
    assert FakeTrainer.captured["evidence_window_config"].is_disabled
    assert FakeTrainer.captured["latent_token_ids"] == {
        "start": 190,
        "end": 191,
        "pad": 192,
        "padding": 0,
        "special": (0, 190, 191, 192, 199),
    }

    loaded_config = EvidenceWindowConfig(
        relation=RelationConfig(lambda_magnitude=0.25)
    )
    args.ew_config = str(tmp_path / "enabled.json")
    monkeypatch.setattr(
        main_module.EvidenceWindowConfig,
        "from_json",
        lambda path: loaded_config,
    )

    main_module.main_train()

    assert FakeTrainer.captured["evidence_window_config"] is loaded_config
    assert FakeTrainer.captured["latent_token_ids"] == {
        "start": 190,
        "end": 191,
        "pad": 192,
        "padding": 0,
        "special": (0, 190, 191, 192, 199),
    }
