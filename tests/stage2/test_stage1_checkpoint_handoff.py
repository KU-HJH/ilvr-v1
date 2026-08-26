from __future__ import annotations

import json
import random
import shutil
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate.state import AcceleratorState, GradientState
from transformers import TrainerCallback, TrainingArguments, default_data_collator
from transformers.integrations.deepspeed import unset_hf_deepspeed_config
from transformers.utils.peft_utils import find_adapter_config_file

import main
from peft_common import resolve_qwen2_5_vl_model
from stage1.checkpointing import (
    STAGE1_PEFT_METADATA_FILENAME,
    load_stage1_peft_artifact,
    save_stage1_peft_artifact,
)
from stage1.peft import build_stage1_tuned_model
from stage1.token_rows import find_sparse_output_token_rows
from stage2.checkpointing import (
    build_stage2_peft_metadata,
    classify_stage2_artifact,
    configure_stage2_metadata,
    load_stage2_peft_metadata,
    prepare_stage2_sparse_token_base,
    resolve_stage2_load_decision,
)
from stage2.peft import (
    Stage2TuningConfig,
    build_stage2_tuned_model,
    load_stage1_dora_for_fresh_stage2,
    load_stage2_model_for_inference,
    validate_stage2_parameter_boundary,
)
from stage2.trainer import Stage2Trainer
from tests.stage1.test_checkpointing import (
    LATENT_TOKEN_IDS,
    RESIZED_VOCAB_SIZE,
    TinyStage1Processor,
    sample_metadata,
    stage1_tuning,
    tiny_original_qwen,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
from tests.stage2.test_routing import stage2_args


@pytest.fixture(autouse=True)
def isolated_accelerate_state(monkeypatch):
    """Keep real Trainer construction independent of earlier distributed tests."""
    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
    unset_hf_deepspeed_config()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    yield
    unset_hf_deepspeed_config()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


def stage2_dora_tuning(**overrides) -> Stage2TuningConfig:
    values = dict(
        method="dora",
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=stage1_tuning().target_suffixes,
    )
    values.update(overrides)
    return Stage2TuningConfig(**values)


def write_stage1_dora_artifact(
    directory, *, pretrained_vocab_size=None, tie_word_embeddings=False
):
    base = tiny_original_qwen(
        **(
            {}
            if pretrained_vocab_size is None
            else {"pretrained_vocab_size": pretrained_vocab_size}
        )
    )
    base.config.tie_word_embeddings = tie_word_embeddings
    if tie_word_embeddings:
        base.tie_weights()
    metadata = sample_metadata(base)
    base.resize_token_embeddings(RESIZED_VOCAB_SIZE)
    base.config.latent_token_id = LATENT_TOKEN_IDS[0]
    base.config.latent_start_id = LATENT_TOKEN_IDS[1]
    base.config.latent_end_id = LATENT_TOKEN_IDS[2]
    model = build_stage1_tuned_model(base, stage1_tuning(), LATENT_TOKEN_IDS)
    with torch.no_grad():
        for index, (name, parameter) in enumerate(model.named_parameters(), start=1):
            if parameter.requires_grad:
                parameter.fill_(0.005 * (index % 9 + 1))
    processor = TinyStage1Processor()
    save_stage1_peft_artifact(model, processor, directory, metadata)
    return processor, metadata


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_fresh_stage2_handoff_accepts_complete_legacy_stage1_v1_metadata(
    tmp_path, tie_word_embeddings
):
    processor, metadata = write_stage1_dora_artifact(
        tmp_path, tie_word_embeddings=tie_word_embeddings
    )
    metadata_path = tmp_path / STAGE1_PEFT_METADATA_FILENAME
    legacy = metadata.to_dict()
    legacy["schema_version"] = 1
    legacy.pop("pretrained_model_vocab_size")
    metadata_path.write_text(
        json.dumps(legacy, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source = tiny_original_qwen()
    source.config.tie_word_embeddings = tie_word_embeddings
    if tie_word_embeddings:
        source.tie_weights()
    continued = load_stage1_dora_for_fresh_stage2(
        source,
        processor,
        tmp_path,
        stage2_dora_tuning(),
    )

    qwen = resolve_qwen2_5_vl_model(continued)
    assert qwen.config.vocab_size == RESIZED_VOCAB_SIZE
    assert qwen.model.embed_tokens.token_adapter.base_layer.num_embeddings == (
        RESIZED_VOCAB_SIZE
    )
    assert qwen.config.tie_word_embeddings is False


def tiny_tag_free_original_qwen(
    *, pretrained_vocab_size=None, tie_word_embeddings=False
):
    """Return the actual sparse-resume base shape without Stage 1 ILVR tags."""
    model = tiny_original_qwen(
        **(
            {}
            if pretrained_vocab_size is None
            else {"pretrained_vocab_size": pretrained_vocab_size}
        )
    )
    model.config.tie_word_embeddings = tie_word_embeddings
    if tie_word_embeddings:
        model.tie_weights()
    model.config.__dict__.pop("stage", None)
    model.config.__dict__.pop("latent_size", None)
    return model


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_padded_stage1_v2_to_self_contained_stage2_direct_and_inference_parity(
    tmp_path, tie_word_embeddings,
):
    pretrained_vocab_size = 132
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    processor, stage1_metadata = write_stage1_dora_artifact(
        stage1_dir,
        pretrained_vocab_size=pretrained_vocab_size,
        tie_word_embeddings=tie_word_embeddings,
    )
    tuning = stage2_dora_tuning()
    continuation_base = tiny_original_qwen(
        pretrained_vocab_size=pretrained_vocab_size
    )
    continuation_base.config.tie_word_embeddings = tie_word_embeddings
    if tie_word_embeddings:
        continuation_base.tie_weights()
    continued = load_stage1_dora_for_fresh_stage2(
        continuation_base,
        processor,
        stage1_dir,
        tuning,
    )
    qwen = resolve_qwen2_5_vl_model(continued)
    configure_stage2_metadata(
        qwen.config,
        LatentTokenIds(*LATENT_TOKEN_IDS),
        latent_size=stage1_metadata.latent_size,
        supervise_latent_boundaries=True,
    )
    metadata = build_stage2_peft_metadata(
        tuning=tuning,
        base_model_reference=stage1_metadata.base_model_reference,
        model_config=qwen.config,
        token_ids=LatentTokenIds(*LATENT_TOKEN_IDS),
        resolved_target_names=stage1_metadata.resolved_target_names,
        token_row_source="sparse_adapter",
        stage1_adapter_reference=str(stage1_dir.resolve()),
        tokenizer_base_vocab_size=stage1_metadata.base_vocab_size,
        pretrained_model_vocab_size=stage1_metadata.pretrained_model_vocab_size,
        resized_vocab_size=stage1_metadata.resized_vocab_size,
        source_base_model_architecture=stage1_metadata.base_model_architecture,
    )
    assert metadata.base_model_architecture["decoder"]["tie_word_embeddings"] is (
        tie_word_embeddings
    )
    assert qwen.config.tie_word_embeddings is False
    Stage2Trainer(
        model=continued,
        args=TrainingArguments(
            output_dir=str(stage2_dir), report_to=[], use_cpu=True
        ),
        stage2_peft_metadata=metadata,
    ).save_model(stage2_dir)
    processor.save_pretrained(stage2_dir)
    fixed_hidden = torch.linspace(-0.4, 0.9, 16).reshape(2, 8)
    expected_logits = qwen.lm_head(fixed_hidden).detach()
    shutil.rmtree(stage1_dir)

    direct_base = tiny_tag_free_original_qwen(
        pretrained_vocab_size=pretrained_vocab_size,
        tie_word_embeddings=tie_word_embeddings,
    )
    prepare_stage2_sparse_token_base(
        direct_base, processor.tokenizer, load_stage2_peft_metadata(stage2_dir)
    )
    direct = build_stage2_tuned_model(
        direct_base,
        tuning,
        adapter_path=stage2_dir,
        is_trainable=True,
        token_row_source="sparse_adapter",
        latent_token_ids=LATENT_TOKEN_IDS,
    )
    torch.testing.assert_close(
        resolve_qwen2_5_vl_model(direct).lm_head(fixed_hidden), expected_logits
    )

    inference = load_stage2_model_for_inference(
        "tiny/original-qwen",
        stage2_dir,
        processor,
        discover_adapter_config=lambda unused, **unused_kwargs: None,
        load_config=lambda unused, **unused_kwargs: deepcopy(
            tiny_tag_free_original_qwen(
                pretrained_vocab_size=pretrained_vocab_size,
                tie_word_embeddings=tie_word_embeddings,
            ).config
        ),
        load_model=lambda unused, **unused_kwargs: tiny_tag_free_original_qwen(
            pretrained_vocab_size=pretrained_vocab_size,
            tie_word_embeddings=tie_word_embeddings,
        ),
    )
    torch.testing.assert_close(
        resolve_qwen2_5_vl_model(inference).lm_head(fixed_hidden), expected_logits
    )


def active_stage1_state(model) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if any(
            component in name
            for component in (
                "lora_A",
                "lora_B",
                "lora_magnitude_vector",
                "trainable_tokens_delta",
                "output_delta",
            )
        )
    }


def write_sparse_stage2_artifact(stage1_dir, stage2_dir):
    processor, stage1_metadata = write_stage1_dora_artifact(stage1_dir)
    tuning = stage2_dora_tuning()
    continued = load_stage1_dora_for_fresh_stage2(
        tiny_original_qwen(), processor, stage1_dir, tuning
    )
    qwen = resolve_qwen2_5_vl_model(continued)
    configure_stage2_metadata(
        qwen.config,
        LatentTokenIds(*LATENT_TOKEN_IDS),
        latent_size=stage1_metadata.latent_size,
        supervise_latent_boundaries=True,
    )
    metadata = build_stage2_peft_metadata(
        tuning=tuning,
        base_model_reference=stage1_metadata.base_model_reference,
        model_config=qwen.config,
        token_ids=LatentTokenIds(*LATENT_TOKEN_IDS),
        resolved_target_names=stage1_metadata.resolved_target_names,
        token_row_source="sparse_adapter",
        stage1_adapter_reference=str(stage1_dir.resolve()),
        tokenizer_base_vocab_size=stage1_metadata.base_vocab_size,
        pretrained_model_vocab_size=stage1_metadata.pretrained_model_vocab_size,
        resized_vocab_size=stage1_metadata.resized_vocab_size,
        source_base_model_architecture=stage1_metadata.base_model_architecture,
    )
    Stage2Trainer(
        model=continued,
        args=TrainingArguments(
            output_dir=str(stage2_dir), report_to=[], use_cpu=True
        ),
        stage2_peft_metadata=metadata,
    ).save_model(stage2_dir)
    processor.save_pretrained(stage2_dir)
    return processor, tuning, metadata, continued


class StopStage2AfterFirstStep(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_training_stop = True
        return control


def _stage2_resume_training_args(output_dir, *, save_strategy):
    return TrainingArguments(
        output_dir=str(output_dir),
        max_steps=2,
        per_device_train_batch_size=1,
        save_strategy=save_strategy,
        save_steps=1,
        report_to=[],
        disable_tqdm=True,
        use_cpu=True,
        seed=47,
        data_seed=47,
    )


def _stage2_sparse_batch():
    return {
        "input_ids": torch.tensor([1, 126, 125, 125, 127, 2]),
        "attention_mask": torch.ones(6, dtype=torch.long),
        "labels": torch.tensor([-100, 126, -100, -100, 127, 2]),
    }


def _capture_rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )


def _assert_rng_state_equal(actual, expected):
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0]
    assert np.array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    assert torch.equal(actual[2], expected[2])


def _assert_nested_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _capture_trainable_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _capture_logits(model):
    model.eval()
    batch = _stage2_sparse_batch()
    with torch.no_grad():
        return model(
            input_ids=batch["input_ids"].unsqueeze(0),
            attention_mask=batch["attention_mask"].unsqueeze(0),
        ).logits.detach().cpu()


def test_stage1_dora_handoff_preserves_same_adapter_rows_and_logits(tmp_path):
    """Catches merging, stacking, or dropping any learned Stage 1 sparse delta."""
    processor, _ = write_stage1_dora_artifact(tmp_path)
    before = load_stage1_peft_artifact(
        tiny_original_qwen(), processor, tmp_path, is_trainable=False
    )
    after = load_stage1_dora_for_fresh_stage2(
        tiny_original_qwen(), processor, tmp_path, stage2_dora_tuning()
    )
    fixed_hidden = torch.linspace(-0.6, 0.8, 16).reshape(2, 8)

    before_state = active_stage1_state(before)
    after_state = active_stage1_state(after)
    assert set(after_state) == set(before_state)
    for name, expected in before_state.items():
        torch.testing.assert_close(after_state[name], expected)
    torch.testing.assert_close(
        resolve_qwen2_5_vl_model(after).lm_head(fixed_hidden),
        resolve_qwen2_5_vl_model(before).lm_head(fixed_hidden),
    )
    trainable = {
        name for name, parameter in after.named_parameters() if parameter.requires_grad
    }
    assert any("lora_A" in name for name in trainable)
    assert any("lora_B" in name for name in trainable)
    assert any("lora_magnitude_vector" in name for name in trainable)
    assert any("trainable_tokens_delta" in name for name in trainable)
    assert any("output_delta" in name for name in trainable)
    assert find_sparse_output_token_rows(after).output_delta.requires_grad
    report = validate_stage2_parameter_boundary(
        after, stage2_dora_tuning(), allow_sparse_token_rows=True
    )
    assert report.trainable_parameters > 0


def handoff_decision_kwargs(stage1_dir, **overrides):
    values = dict(
        load_model_path="tiny/original-qwen",
        stage1_adapter_path=str(stage1_dir),
        resume_from_checkpoint_path=None,
        load_config=lambda unused: tiny_original_qwen().config,
        latent_size=2,
        supervise_latent_boundaries=True,
        tuning_method="dora",
        tuning_method_explicit=True,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=stage1_tuning().target_suffixes,
    )
    values.update(overrides)
    return values


def test_stage1_adapter_source_routes_a_fresh_stage2_decision(tmp_path):
    """Catches treating Stage 1 adapter initialization as Stage 2 resume."""
    write_stage1_dora_artifact(tmp_path)

    decision = resolve_stage2_load_decision(**handoff_decision_kwargs(tmp_path))

    assert decision.base_model_source == "tiny/original-qwen"
    assert decision.processor_source == str(tmp_path)
    assert decision.stage1_adapter_source == str(tmp_path)
    assert decision.adapter_source is None
    assert decision.trainer_resume_path is None
    assert decision.stage1_metadata_snapshot is not None


def test_stage1_adapter_source_and_stage2_resume_fail_before_source_loading():
    """Catches ambiguous checkpoint precedence reaching either artifact loader."""
    calls = []

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_stage2_load_decision(
            **handoff_decision_kwargs(
                "stage1-adapter",
                resume_from_checkpoint_path="stage2/checkpoint-3",
                load_config=lambda source: calls.append(source),
            )
        )

    assert calls == []


def _stub_empty_stage2_data(monkeypatch):
    monkeypatch.setattr(main, "load_jsonl_examples", lambda unused: [])
    monkeypatch.setattr(
        main, "build_prepared_example_length_measure", lambda unused: "measure"
    )
    monkeypatch.setattr(
        main,
        "filter_stage2_examples",
        lambda **unused: ([], SimpleNamespace(retained=0)),
    )
    monkeypatch.setattr(main, "Stage2DataCollator", lambda *unused: object())


def _injected_stage2_source_dependencies(find_adapter_config):
    """Bundle installed source seams; finder-only injection is forbidden."""

    return main.Stage2SourceDependencies(
        find_adapter_config=find_adapter_config,
        load_config=main.AutoConfig.from_pretrained,
        load_processor=main.AutoProcessor.from_pretrained,
        load_model=main.Qwen2_5_VLForConditionalGeneration.from_pretrained,
    )


def test_main_handoff_consumes_preflight_before_large_base_load(
    tmp_path, monkeypatch
):
    """Catches reopening Stage 1 payloads or validating them after base weights."""
    processor, _ = write_stage1_dora_artifact(
        tmp_path, pretrained_vocab_size=132
    )
    args = stage2_args(
        stage1_adapter_path=str(tmp_path),
        load_model_path="tiny/original-qwen",
        latent_size=2,
        tuning_method="dora",
        tuning_method_explicit=True,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=stage1_tuning().target_suffixes,
        gradient_checkpointing=False,
    )
    calls = SimpleNamespace(preflight=False, model_loads=0)
    real_preflight = __import__(
        "stage1.checkpointing", fromlist=["preflight_stage1_peft_artifact"]
    ).preflight_stage1_peft_artifact

    def record_preflight(*positional, **keywords):
        witness = real_preflight(*positional, **keywords)
        calls.preflight = True
        return witness

    def load_large_base(source, **unused):
        assert source == "tiny/original-qwen"
        assert calls.preflight is True
        calls.model_loads += 1
        return tiny_original_qwen(pretrained_vocab_size=132)

    monkeypatch.setattr(
        main.AutoProcessor, "from_pretrained", lambda source: processor
    )
    monkeypatch.setattr(
        main.AutoConfig,
        "from_pretrained",
        lambda source: tiny_original_qwen(pretrained_vocab_size=132).config,
    )
    monkeypatch.setattr(
        main,
        "preflight_stage1_peft_artifact",
        record_preflight,
        raising=False,
    )
    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        load_large_base,
    )
    _stub_empty_stage2_data(monkeypatch)

    components = main.build_stage2_training_components(args)

    assert calls.model_loads == 1
    assert components.trainer_resume_path is None
    assert components.peft_metadata.token_row_source == "sparse_adapter"
    assert components.peft_metadata.stage1_adapter_reference == str(tmp_path.resolve())
    assert components.peft_metadata.tokenizer_base_vocab_size == 125
    assert components.peft_metadata.pretrained_model_vocab_size == 132
    assert components.peft_metadata.resized_vocab_size == 128
    assert resolve_qwen2_5_vl_model(components.model).config.stage == "stage2"


def test_malformed_stage1_adapter_fails_before_large_base_load(tmp_path, monkeypatch):
    """Catches corrupted Stage 1 tensors being discovered after expensive loading."""
    processor, _ = write_stage1_dora_artifact(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"corrupt")
    args = stage2_args(
        stage1_adapter_path=str(tmp_path),
        load_model_path="tiny/original-qwen",
        latent_size=2,
        tuning_method="dora",
        tuning_method_explicit=True,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=stage1_tuning().target_suffixes,
    )
    calls = SimpleNamespace(model_loads=0)
    monkeypatch.setattr(
        main.AutoProcessor, "from_pretrained", lambda source: processor
    )
    monkeypatch.setattr(
        main.AutoConfig,
        "from_pretrained",
        lambda source: tiny_original_qwen().config,
    )

    def reject_large_load(*unused, **unused_keywords):
        calls.model_loads += 1
        raise AssertionError("large base load must not run")

    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        reject_large_load,
    )

    with pytest.raises(ValueError, match="adapter safetensors"):
        main.build_stage2_training_components(args)

    assert calls.model_loads == 0


@pytest.mark.parametrize(
    ("tuning", "message"),
    [
        (stage2_dora_tuning(method="full"), "full"),
        (stage2_dora_tuning(method="lora"), "LoRA"),
        (stage2_dora_tuning(r=2), "rank"),
        (stage2_dora_tuning(alpha=4), "alpha"),
        (stage2_dora_tuning(dropout=0.25), "dropout"),
        (
            stage2_dora_tuning(target_modules=("q_proj",)),
            "requested target",
        ),
    ],
)
def test_stage1_dora_handoff_rejects_changed_or_new_stage2_adapter_before_mutation(
    tmp_path, tuning, message
):
    """Catches unsupported transitions reaching PEFT attachment or Trainer setup."""
    processor, _ = write_stage1_dora_artifact(tmp_path)
    fresh = tiny_original_qwen()
    original_vocab_size = fresh.config.vocab_size

    with pytest.raises(ValueError, match=message):
        load_stage1_dora_for_fresh_stage2(fresh, processor, tmp_path, tuning)

    assert fresh.config.vocab_size == original_vocab_size
    assert not hasattr(fresh.model.embed_tokens, "token_adapter")


def test_stage1_lora_cannot_initialize_fresh_stage2_dora(tmp_path):
    """Catches the legacy Stage 1 LoRA path being silently promoted to DoRA."""
    base = tiny_original_qwen()
    tuning = stage1_tuning("lora")
    metadata = sample_metadata(base, tuning)
    base.resize_token_embeddings(RESIZED_VOCAB_SIZE)
    base.config.latent_token_id = LATENT_TOKEN_IDS[0]
    base.config.latent_start_id = LATENT_TOKEN_IDS[1]
    base.config.latent_end_id = LATENT_TOKEN_IDS[2]
    model = build_stage1_tuned_model(base, tuning, LATENT_TOKEN_IDS)
    processor = TinyStage1Processor()
    save_stage1_peft_artifact(model, processor, tmp_path, metadata)

    with pytest.raises(ValueError, match="Stage 1 DoRA"):
        load_stage1_dora_for_fresh_stage2(
            tiny_original_qwen(), processor, tmp_path, stage2_dora_tuning()
        )


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_saved_sparse_stage2_adapter_reloads_after_stage1_source_is_deleted(tmp_path):
    """Catches Stage 2 save/load retaining a hidden dependency on Stage 1 paths."""
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    processor, stage1_metadata = write_stage1_dora_artifact(stage1_dir)
    tuning = stage2_dora_tuning()
    continued = load_stage1_dora_for_fresh_stage2(
        tiny_original_qwen(), processor, stage1_dir, tuning
    )
    qwen = resolve_qwen2_5_vl_model(continued)
    configure_stage2_metadata(
        qwen.config,
        LatentTokenIds(*LATENT_TOKEN_IDS),
        latent_size=stage1_metadata.latent_size,
        supervise_latent_boundaries=True,
    )
    metadata = build_stage2_peft_metadata(
        tuning=tuning,
        base_model_reference=stage1_metadata.base_model_reference,
        model_config=qwen.config,
        token_ids=LatentTokenIds(*LATENT_TOKEN_IDS),
        resolved_target_names=stage1_metadata.resolved_target_names,
        token_row_source="sparse_adapter",
        stage1_adapter_reference=str(stage1_dir.resolve()),
        tokenizer_base_vocab_size=stage1_metadata.base_vocab_size,
        pretrained_model_vocab_size=stage1_metadata.pretrained_model_vocab_size,
        resized_vocab_size=stage1_metadata.resized_vocab_size,
        source_base_model_architecture=stage1_metadata.base_model_architecture,
    )
    expected_state = active_stage1_state(continued)
    fixed_hidden = torch.linspace(-0.4, 0.9, 16).reshape(2, 8)
    expected_logits = qwen.lm_head(fixed_hidden).detach()
    trainer = Stage2Trainer(
        model=continued,
        args=TrainingArguments(
            output_dir=str(stage2_dir), report_to=[], use_cpu=True
        ),
        stage2_peft_metadata=metadata,
    )

    assert trainer.state.global_step == 0
    assert trainer.optimizer is None
    assert trainer.lr_scheduler is None

    trainer.save_model(stage2_dir)
    processor.save_pretrained(stage2_dir)

    assert (stage2_dir / "adapter_model.safetensors").is_file()
    assert (stage2_dir / "ilvr_token_rows.safetensors").is_file()
    assert (stage2_dir / "stage2_peft_metadata.json").is_file()
    assert (stage2_dir / "preprocessor_config.json").is_file()
    shutil.rmtree(stage1_dir)
    reloaded_metadata = load_stage2_peft_metadata(stage2_dir)
    assert classify_stage2_artifact(stage2_dir).value == "peft"
    assert reloaded_metadata.stage1_adapter_reference == str(stage1_dir.resolve())

    reloaded_base = tiny_original_qwen()
    reloaded_base.resize_token_embeddings(RESIZED_VOCAB_SIZE)
    configure_stage2_metadata(
        reloaded_base.config,
        LatentTokenIds(*LATENT_TOKEN_IDS),
        latent_size=stage1_metadata.latent_size,
        supervise_latent_boundaries=True,
    )
    reloaded = build_stage2_tuned_model(
        reloaded_base,
        tuning,
        adapter_path=stage2_dir,
        is_trainable=True,
        token_row_source="sparse_adapter",
        latent_token_ids=LATENT_TOKEN_IDS,
    )

    reloaded_state = active_stage1_state(reloaded)
    assert set(reloaded_state) == set(expected_state)
    for name, expected in expected_state.items():
        torch.testing.assert_close(reloaded_state[name], expected)
    torch.testing.assert_close(
        resolve_qwen2_5_vl_model(reloaded).lm_head(fixed_hidden), expected_logits
    )


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_sparse_stage2_resume_decision_uses_tag_free_original_base(tmp_path):
    """Catches sparse v4 resume being routed through legacy Stage 1 tag checks."""
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "checkpoint-1"
    _, tuning, metadata, _ = write_sparse_stage2_artifact(
        stage1_dir, stage2_dir
    )
    shutil.rmtree(stage1_dir)
    config_sources = []

    decision = resolve_stage2_load_decision(
        load_model_path="tiny/original-qwen",
        resume_from_checkpoint_path=str(stage2_dir),
        load_config=lambda source: config_sources.append(source)
        or tiny_tag_free_original_qwen().config,
        discover_adapter_config=lambda unused, **unused_kwargs: None,
        latent_size=metadata.latent_size,
        supervise_latent_boundaries=metadata.supervise_latent_boundaries,
        tuning_method=None,
        tuning_method_explicit=False,
        lora_r=tuning.r,
        lora_alpha=tuning.alpha,
        lora_dropout=tuning.dropout,
        lora_target_modules=tuning.target_modules,
    )

    assert config_sources == ["tiny/original-qwen"]
    assert decision.base_model_source == "tiny/original-qwen"
    assert decision.processor_source == str(stage2_dir)
    assert decision.adapter_source == str(stage2_dir)
    assert decision.trainer_resume_path == str(stage2_dir)
    assert decision.peft_metadata == metadata


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_main_sparse_stage2_resume_is_independent_of_stage1_source(
    tmp_path, monkeypatch
):
    """Catches normal Stage 2 resume reopening its provenance-only Stage 1 path."""
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2-checkpoint"
    processor, tuning, metadata, continued = write_sparse_stage2_artifact(
        stage1_dir, stage2_dir
    )
    expected_state = active_stage1_state(continued)
    fixed_hidden = torch.linspace(-0.4, 0.9, 16).reshape(2, 8)
    expected_logits = resolve_qwen2_5_vl_model(continued).lm_head(
        fixed_hidden
    ).detach()
    shutil.rmtree(stage1_dir)
    args = stage2_args(
        stage1_adapter_path=None,
        load_model_path="tiny/original-qwen",
        resume_from_checkpoint_path=str(stage2_dir),
        latent_size=2,
        tuning_method=None,
        tuning_method_explicit=False,
        gradient_checkpointing=False,
    )
    processor_sources = []
    config_sources = []
    model_sources = []
    monkeypatch.setattr(
        main.AutoProcessor,
        "from_pretrained",
        lambda source: processor_sources.append(source) or processor,
    )
    monkeypatch.setattr(
        main.AutoConfig,
        "from_pretrained",
        lambda source: config_sources.append(source)
        or tiny_tag_free_original_qwen().config,
    )
    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda source, **unused: model_sources.append(source)
        or tiny_tag_free_original_qwen(),
    )
    _stub_empty_stage2_data(monkeypatch)

    components = main.build_stage2_training_components(
        args,
        source_dependencies=_injected_stage2_source_dependencies(
            lambda unused, **unused_kwargs: None
        ),
    )

    assert processor_sources == [str(stage2_dir)]
    assert config_sources == ["tiny/original-qwen"]
    assert model_sources == ["tiny/original-qwen"]
    assert components.trainer_resume_path == str(stage2_dir)
    assert components.peft_metadata == metadata
    validate_stage2_parameter_boundary(
        components.model, tuning, allow_sparse_token_rows=True
    )
    restored_state = active_stage1_state(components.model)
    assert set(restored_state) == set(expected_state)
    for name, expected in expected_state.items():
        torch.testing.assert_close(restored_state[name], expected)
    torch.testing.assert_close(
        resolve_qwen2_5_vl_model(components.model).lm_head(fixed_hidden),
        expected_logits,
    )


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_sparse_resume_default_loaders_share_one_source_provenance(
    tmp_path, monkeypatch
):
    """Catches Hub source selectors diverging between discovery and loading."""
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2-checkpoint"
    original_base = tmp_path / "clean-original-base"
    original_base.mkdir()
    processor, _, metadata, _ = write_sparse_stage2_artifact(
        stage1_dir, stage2_dir
    )
    shutil.rmtree(stage1_dir)
    token = "".join(("private", "-token"))
    args = stage2_args(
        stage1_adapter_path=None,
        load_model_path=str(original_base),
        resume_from_checkpoint_path=str(stage2_dir),
        latent_size=metadata.latent_size,
        tuning_method=None,
        tuning_method_explicit=False,
        gradient_checkpointing=False,
        cache_dir=str(tmp_path / "hub-cache"),
        revision="tested-revision",
        token=token,
        local_files_only=True,
    )
    expected_source_kwargs = {
        name: getattr(args, name)
        for name in ("cache_dir", "revision", "token", "local_files_only")
    }
    calls = []
    real_finder = find_adapter_config_file

    def record_finder(source, **kwargs):
        calls.append(("finder", source, kwargs))
        return real_finder(source, **kwargs)

    def record_config(source, **kwargs):
        calls.append(("config", source, kwargs))
        return tiny_tag_free_original_qwen().config

    def record_processor(source, **kwargs):
        calls.append(("processor", source, kwargs))
        return processor

    def record_model(source, **kwargs):
        calls.append(("model", source, kwargs))
        return tiny_tag_free_original_qwen()

    monkeypatch.setattr(main, "find_adapter_config_file", record_finder)
    monkeypatch.setattr(main.AutoConfig, "from_pretrained", record_config)
    monkeypatch.setattr(main.AutoProcessor, "from_pretrained", record_processor)
    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        record_model,
    )
    _stub_empty_stage2_data(monkeypatch)

    components = main.build_stage2_training_components(args)

    assert [name for name, _, _ in calls] == [
        "finder",
        "config",
        "processor",
        "model",
    ]
    assert calls[0][1] == calls[1][1] == calls[3][1] == str(original_base)
    assert calls[2][1] == str(stage2_dir)
    for name, _, kwargs in calls:
        source_kwargs = {
            key: kwargs[key] for key in expected_source_kwargs
        }
        assert source_kwargs == expected_source_kwargs, name
        assert all(
            source_kwargs[key] is expected
            for key, expected in expected_source_kwargs.items()
        ), name
    assert calls[0][2] == expected_source_kwargs
    assert calls[1][2] == expected_source_kwargs
    assert calls[2][2] == expected_source_kwargs
    assert calls[3][2] == {
        **expected_source_kwargs,
        "attn_implementation": "sdpa",
    }
    assert components.trainer_resume_path == str(stage2_dir)


def test_stage2_source_provenance_is_immutable_and_excludes_model_options():
    """Catches later loader layers mutating selectors or adding model-only args."""
    args = stage2_args(
        cache_dir="/cache/models",
        revision="tested-revision",
        token="tested-token",
        local_files_only=False,
        torch_dtype="bfloat16",
    )

    source_kwargs = main.build_stage2_source_provenance_kwargs(args)

    assert dict(source_kwargs) == {
        "cache_dir": "/cache/models",
        "revision": "tested-revision",
        "token": "tested-token",
        "local_files_only": False,
    }
    with pytest.raises(TypeError):
        source_kwargs["revision"] = "mutated"


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_sparse_resume_injected_source_dependencies_share_provenance(
    tmp_path, monkeypatch
):
    """Catches dependency injection restoring a finder-only bypass."""
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2-checkpoint"
    processor, _, metadata, _ = write_sparse_stage2_artifact(
        stage1_dir, stage2_dir
    )
    shutil.rmtree(stage1_dir)
    args = stage2_args(
        stage1_adapter_path=None,
        load_model_path="org/clean-original-base",
        resume_from_checkpoint_path=str(stage2_dir),
        latent_size=metadata.latent_size,
        tuning_method=None,
        tuning_method_explicit=False,
        gradient_checkpointing=False,
        cache_dir="/cache/models",
        revision="immutable-revision",
        token="immutable-token",
        local_files_only=True,
    )
    expected_source_kwargs = {
        name: getattr(args, name)
        for name in ("cache_dir", "revision", "token", "local_files_only")
    }
    observed = []

    def finder(source, **kwargs):
        observed.append(("finder", kwargs))
        return None

    def config(source, **kwargs):
        observed.append(("config", kwargs))
        return tiny_tag_free_original_qwen().config

    def processor_loader(source, **kwargs):
        observed.append(("processor", kwargs))
        return processor

    def model_loader(source, **kwargs):
        observed.append(("model", kwargs))
        return tiny_tag_free_original_qwen()

    source_dependencies = main.Stage2SourceDependencies(
        find_adapter_config=finder,
        load_config=config,
        load_processor=processor_loader,
        load_model=model_loader,
    )
    _stub_empty_stage2_data(monkeypatch)

    main.build_stage2_training_components(
        args, source_dependencies=source_dependencies
    )

    assert [name for name, _ in observed] == [
        "finder",
        "config",
        "processor",
        "model",
    ]
    for name, kwargs in observed[:-1]:
        assert kwargs == expected_source_kwargs, name
        assert all(
            kwargs[key] is expected for key, expected in expected_source_kwargs.items()
        ), name
    assert observed[-1][1] == {
        **expected_source_kwargs,
        "attn_implementation": "sdpa",
    }


@pytest.mark.parametrize("source_kind", ("local", "remote"))
@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_sparse_stage2_resume_discovers_adapter_bearing_original_source_before_load(
    tmp_path, monkeypatch, source_kind
):
    """Catches original-base routing silently following a local or Hub adapter."""
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2-checkpoint"
    write_sparse_stage2_artifact(stage1_dir, stage2_dir)
    shutil.rmtree(stage1_dir)
    discovery_calls = []
    calls = SimpleNamespace(
        configs=0, processors=0, model_loads=0, attaches=0, trainers=0
    )

    if source_kind == "local":
        original_source = tmp_path / "adapter-source"
        original_source.mkdir()
        (original_source / "adapter_config.json").write_text(
            '{"base_model_name_or_path": "tiny/original-qwen"}\n',
            encoding="utf-8",
        )

        def discover_adapter(source, **kwargs):
            discovery_calls.append((source, kwargs))
            return find_adapter_config_file(source, **kwargs)

    else:
        original_source = "org/remote-adapter"

        def discover_adapter(source, **kwargs):
            discovery_calls.append((source, kwargs))
            return "/cache/adapter_config.json"

    args = stage2_args(
        stage1_adapter_path=None,
        load_model_path=str(original_source),
        resume_from_checkpoint_path=str(stage2_dir),
        latent_size=2,
        tuning_method=None,
        tuning_method_explicit=False,
        gradient_checkpointing=False,
        cache_dir="/cache/models",
        revision="tested-revision",
        token="tested-token",
        local_files_only=True,
    )

    def reject_config(*unused, **unused_kwargs):
        calls.configs += 1
        raise AssertionError("config load must not run after adapter discovery")

    def reject_processor(*unused, **unused_kwargs):
        calls.processors += 1
        raise AssertionError("processor load must not run after adapter discovery")

    def reject_model(*unused, **unused_kwargs):
        calls.model_loads += 1
        raise AssertionError("large model load must not run after adapter discovery")

    def reject_attach(*unused, **unused_kwargs):
        calls.attaches += 1
        raise AssertionError("adapter attach must not run")

    def reject_trainer(*unused, **unused_kwargs):
        calls.trainers += 1
        raise AssertionError("Trainer must not run")

    monkeypatch.setattr(main.AutoConfig, "from_pretrained", reject_config)
    monkeypatch.setattr(main.AutoProcessor, "from_pretrained", reject_processor)
    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration, "from_pretrained", reject_model
    )
    monkeypatch.setattr(main, "build_stage2_tuned_model", reject_attach)
    monkeypatch.setattr(main, "Stage2Trainer", reject_trainer)

    with pytest.raises(ValueError, match="original.*base.*adapter|adapter.*original.*base"):
        main.build_stage2_training_components(
            args,
            source_dependencies=_injected_stage2_source_dependencies(
                discover_adapter
            ),
        )

    assert discovery_calls == [
        (
            str(original_source),
            {
                "cache_dir": "/cache/models",
                "revision": "tested-revision",
                "token": "tested-token",
                "local_files_only": True,
            },
        )
    ]
    assert vars(calls) == {
        "configs": 0,
        "processors": 0,
        "model_loads": 0,
        "attaches": 0,
        "trainers": 0,
    }


@pytest.mark.parametrize(
    ("marker", "value"),
    [
        ("peft_type", "LORA"),
        ("base_model_name_or_path", "tiny/original-qwen"),
        ("inference_mode", False),
    ],
)
@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_sparse_stage2_resume_rejects_peft_metadata_in_original_config_before_load(
    tmp_path, monkeypatch, marker, value
):
    """Catches PEFT adapter metadata masquerading as a tag-free Qwen config."""
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2-checkpoint"
    write_sparse_stage2_artifact(stage1_dir, stage2_dir)
    shutil.rmtree(stage1_dir)
    config = tiny_tag_free_original_qwen().config
    setattr(config, marker, value)
    calls = SimpleNamespace(
        configs=0, processors=0, model_loads=0, attaches=0, trainers=0
    )

    def load_config(unused, **unused_kwargs):
        calls.configs += 1
        return config

    def reject_processor(*unused, **unused_kwargs):
        calls.processors += 1
        raise AssertionError("processor load must not run after config rejection")

    def reject_model(*unused, **unused_kwargs):
        calls.model_loads += 1
        raise AssertionError("large model load must not run after config rejection")

    def reject_attach(*unused, **unused_kwargs):
        calls.attaches += 1
        raise AssertionError("adapter attach must not run")

    def reject_trainer(*unused, **unused_kwargs):
        calls.trainers += 1
        raise AssertionError("Trainer must not run")

    monkeypatch.setattr(main.AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(main.AutoProcessor, "from_pretrained", reject_processor)
    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration, "from_pretrained", reject_model
    )
    monkeypatch.setattr(main, "build_stage2_tuned_model", reject_attach)
    monkeypatch.setattr(main, "Stage2Trainer", reject_trainer)
    args = stage2_args(
        stage1_adapter_path=None,
        load_model_path="tiny/original-qwen",
        resume_from_checkpoint_path=str(stage2_dir),
        latent_size=2,
        tuning_method=None,
        tuning_method_explicit=False,
        gradient_checkpointing=False,
    )

    with pytest.raises(ValueError, match=f"PEFT.*{marker}|{marker}.*PEFT"):
        main.build_stage2_training_components(
            args,
            source_dependencies=_injected_stage2_source_dependencies(
                lambda unused, **unused_kwargs: None
            ),
        )

    assert vars(calls) == {
        "configs": 1,
        "processors": 0,
        "model_loads": 0,
        "attaches": 0,
        "trainers": 0,
    }


@pytest.mark.parametrize(
    ("runtime_marker", "runtime_value"),
    [
        ("peft_config", {"default": object()}),
        ("_hf_peft_config_loaded", True),
    ],
)
@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_sparse_stage2_resume_rejects_adapter_bearing_loaded_base_before_resize(
    tmp_path, monkeypatch, runtime_marker, runtime_value
):
    """Catches a dynamically adapter-loaded Qwen reaching resize or attachment."""
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2-checkpoint"
    processor, _, _, _ = write_sparse_stage2_artifact(stage1_dir, stage2_dir)
    shutil.rmtree(stage1_dir)
    loaded = tiny_tag_free_original_qwen()
    setattr(loaded, runtime_marker, runtime_value)
    original_vocab = loaded.config.vocab_size
    calls = SimpleNamespace(model_loads=0, attaches=0, trainers=0)

    monkeypatch.setattr(
        main.AutoConfig,
        "from_pretrained",
        lambda unused: tiny_tag_free_original_qwen().config,
    )
    monkeypatch.setattr(main.AutoProcessor, "from_pretrained", lambda unused: processor)

    def load_model(*unused, **unused_kwargs):
        calls.model_loads += 1
        return loaded

    def reject_attach(*unused, **unused_kwargs):
        calls.attaches += 1
        raise AssertionError("adapter attach must not run")

    def reject_trainer(*unused, **unused_kwargs):
        calls.trainers += 1
        raise AssertionError("Trainer must not run")

    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration, "from_pretrained", load_model
    )
    monkeypatch.setattr(main, "build_stage2_tuned_model", reject_attach)
    monkeypatch.setattr(main, "Stage2Trainer", reject_trainer)
    _stub_empty_stage2_data(monkeypatch)
    args = stage2_args(
        stage1_adapter_path=None,
        load_model_path="tiny/original-qwen",
        resume_from_checkpoint_path=str(stage2_dir),
        latent_size=2,
        tuning_method=None,
        tuning_method_explicit=False,
        gradient_checkpointing=False,
    )

    with pytest.raises(
        ValueError,
        match="adapter-free original base.*adapter-bearing|adapter-bearing.*original base",
    ):
        main.build_stage2_training_components(
            args,
            source_dependencies=_injected_stage2_source_dependencies(
                lambda unused, **unused_kwargs: None
            ),
        )

    assert calls.model_loads == 1
    assert calls.attaches == 0
    assert calls.trainers == 0
    assert loaded.config.vocab_size == original_vocab


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("architecture", "original base architecture mismatch.*hidden_act"),
        ("base_vocab", "original base architecture mismatch.*vocab_size"),
        ("processor_vocab", "processor vocabulary mismatch"),
        ("token_ids", "latent token IDs"),
        ("token_strings", "latent token string round-trip"),
    ],
)
@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_sparse_stage2_resume_rejects_wrong_original_sources_before_attach(
    tmp_path, monkeypatch, mutation, message
):
    """Catches incompatible sources reaching PEFT attachment or Trainer."""
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2-checkpoint"
    processor, _, _, _ = write_sparse_stage2_artifact(stage1_dir, stage2_dir)
    shutil.rmtree(stage1_dir)
    config = tiny_tag_free_original_qwen().config
    if mutation == "architecture":
        config.hidden_act = "gelu"
    elif mutation == "base_vocab":
        config.vocab_size = RESIZED_VOCAB_SIZE - 4
    elif mutation == "processor_vocab":
        processor.tokenizer._size = RESIZED_VOCAB_SIZE - 1
    elif mutation == "token_ids":
        processor.tokenizer._ids["<|latent_start|>"] = LATENT_TOKEN_IDS[0] - 1
    else:
        processor.tokenizer._tokens[LATENT_TOKEN_IDS[1]] = "<|wrong_latent|>"

    args = stage2_args(
        stage1_adapter_path=None,
        load_model_path="tiny/original-qwen",
        resume_from_checkpoint_path=str(stage2_dir),
        latent_size=2,
        tuning_method=None,
        tuning_method_explicit=False,
        gradient_checkpointing=False,
    )
    calls = SimpleNamespace(model_loads=0, adapter_attaches=0, trainers=0)
    monkeypatch.setattr(
        main.AutoProcessor, "from_pretrained", lambda unused: processor
    )
    monkeypatch.setattr(
        main.AutoConfig, "from_pretrained", lambda unused: config
    )

    def load_large_base(*unused, **unused_keywords):
        calls.model_loads += 1
        model = tiny_tag_free_original_qwen()
        if mutation == "architecture":
            model.config.hidden_act = "gelu"
        elif mutation == "base_vocab":
            model.config.vocab_size = RESIZED_VOCAB_SIZE - 4
        return model

    def reject_adapter_attach(*unused, **unused_keywords):
        calls.adapter_attaches += 1
        raise AssertionError("adapter attach must not run")

    def reject_trainer(*unused, **unused_keywords):
        calls.trainers += 1
        raise AssertionError("Trainer must not run")

    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        load_large_base,
    )
    monkeypatch.setattr(main, "build_stage2_tuned_model", reject_adapter_attach)
    monkeypatch.setattr(main, "Stage2Trainer", reject_trainer)
    _stub_empty_stage2_data(monkeypatch)

    with pytest.raises(ValueError, match=message):
        main.build_stage2_training_components(
            args,
            source_dependencies=_injected_stage2_source_dependencies(
                lambda unused, **unused_kwargs: None
            ),
        )

    assert calls.model_loads == 0
    assert calls.adapter_attaches == 0
    assert calls.trainers == 0


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_sparse_stage2_numbered_resume_matches_uninterrupted_training(tmp_path):
    """Catches Stage 2 resume dropping any Trainer-owned or sparse adapter state."""
    stage1_dir = tmp_path / "stage1"
    processor, stage1_metadata = write_stage1_dora_artifact(stage1_dir)
    tuning = stage2_dora_tuning()
    initial = load_stage1_dora_for_fresh_stage2(
        tiny_original_qwen(), processor, stage1_dir, tuning
    )
    qwen = resolve_qwen2_5_vl_model(initial)
    configure_stage2_metadata(
        qwen.config,
        LatentTokenIds(*LATENT_TOKEN_IDS),
        latent_size=stage1_metadata.latent_size,
        supervise_latent_boundaries=True,
    )
    metadata = build_stage2_peft_metadata(
        tuning=tuning,
        base_model_reference=stage1_metadata.base_model_reference,
        model_config=qwen.config,
        token_ids=LatentTokenIds(*LATENT_TOKEN_IDS),
        resolved_target_names=stage1_metadata.resolved_target_names,
        token_row_source="sparse_adapter",
        stage1_adapter_reference=str(stage1_dir.resolve()),
        tokenizer_base_vocab_size=stage1_metadata.base_vocab_size,
        pretrained_model_vocab_size=stage1_metadata.pretrained_model_vocab_size,
        resized_vocab_size=stage1_metadata.resized_vocab_size,
        source_base_model_architecture=stage1_metadata.base_model_architecture,
    )
    dataset = [_stage2_sparse_batch()]

    uninterrupted = Stage2Trainer(
        model=deepcopy(initial),
        args=_stage2_resume_training_args(
            tmp_path / "uninterrupted", save_strategy="no"
        ),
        train_dataset=dataset,
        data_collator=default_data_collator,
        processing_class=processor,
        stage2_peft_metadata=metadata,
    )
    uninterrupted.train()
    expected_model = _capture_trainable_state(uninterrupted.model)
    expected_optimizer = deepcopy(uninterrupted.optimizer.state_dict())
    expected_scheduler = deepcopy(uninterrupted.lr_scheduler.state_dict())
    expected_rng = _capture_rng_state()
    expected_logits = _capture_logits(uninterrupted.model)

    output = tmp_path / "resumed"
    interrupted = Stage2Trainer(
        model=deepcopy(initial),
        args=_stage2_resume_training_args(output, save_strategy="steps"),
        train_dataset=dataset,
        data_collator=default_data_collator,
        processing_class=processor,
        stage2_peft_metadata=metadata,
        callbacks=[StopStage2AfterFirstStep()],
    )
    interrupted.train()

    checkpoint = output / "checkpoint-1"
    for filename in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "ilvr_token_rows.safetensors",
        "stage2_peft_metadata.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "tokenizer.json",
        "preprocessor_config.json",
    ):
        assert (checkpoint / filename).is_file(), filename
    assert classify_stage2_artifact(checkpoint).value == "peft"

    shutil.rmtree(stage1_dir)
    assert not stage1_dir.exists()
    resumed_processor = TinyStage1Processor.from_pretrained(checkpoint)
    resumed_metadata = load_stage2_peft_metadata(checkpoint)
    assert resumed_metadata.stage1_adapter_reference == str(stage1_dir.resolve())
    resumed_base = tiny_tag_free_original_qwen()
    prepare_stage2_sparse_token_base(
        resumed_base, resumed_processor.tokenizer, resumed_metadata
    )
    resumed_model = build_stage2_tuned_model(
        resumed_base,
        tuning,
        adapter_path=checkpoint,
        is_trainable=True,
        token_row_source="sparse_adapter",
        latent_token_ids=LATENT_TOKEN_IDS,
    )
    resumed = Stage2Trainer(
        model=resumed_model,
        args=_stage2_resume_training_args(output, save_strategy="no"),
        train_dataset=dataset,
        data_collator=default_data_collator,
        processing_class=resumed_processor,
        stage2_peft_metadata=resumed_metadata,
    )

    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.state.global_step == 2
    trainable_names = set(_capture_trainable_state(resumed.model))
    for component in (
        "lora_A",
        "lora_B",
        "lora_magnitude_vector",
        "trainable_tokens_delta",
        "output_delta",
    ):
        assert any(component in name for name in trainable_names), component
    _assert_nested_state_equal(_capture_trainable_state(resumed.model), expected_model)
    _assert_nested_state_equal(resumed.optimizer.state_dict(), expected_optimizer)
    _assert_nested_state_equal(resumed.lr_scheduler.state_dict(), expected_scheduler)
    _assert_rng_state_equal(_capture_rng_state(), expected_rng)
    torch.testing.assert_close(
        _capture_logits(resumed.model), expected_logits, rtol=0, atol=0
    )
