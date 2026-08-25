from __future__ import annotations

import shutil
from types import SimpleNamespace

import pytest
import torch
from accelerate.state import AcceleratorState, GradientState
from transformers import TrainingArguments
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

import main
from peft_common import resolve_qwen2_5_vl_model
from stage1.checkpointing import (
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
    resolve_stage2_load_decision,
)
from stage2.peft import (
    Stage2TuningConfig,
    build_stage2_tuned_model,
    load_stage1_dora_for_fresh_stage2,
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


def write_stage1_dora_artifact(directory):
    base = tiny_original_qwen()
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


def test_main_handoff_consumes_preflight_before_large_base_load(
    tmp_path, monkeypatch
):
    """Catches reopening Stage 1 payloads or validating them after base weights."""
    processor, _ = write_stage1_dora_artifact(tmp_path)
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
        return tiny_original_qwen()

    monkeypatch.setattr(
        main.AutoProcessor, "from_pretrained", lambda source: processor
    )
    monkeypatch.setattr(
        main.AutoConfig,
        "from_pretrained",
        lambda source: tiny_original_qwen().config,
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
def test_main_sparse_stage2_resume_is_independent_of_stage1_source(
    tmp_path, monkeypatch
):
    """Catches normal Stage 2 resume reopening its provenance-only Stage 1 path."""
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2-checkpoint"
    processor, stage1_metadata = write_stage1_dora_artifact(stage1_dir)
    tuning = stage2_dora_tuning()
    continued = load_stage1_dora_for_fresh_stage2(
        tiny_original_qwen(), processor, stage1_dir, tuning
    )
    qwen = resolve_qwen2_5_vl_model(continued)
    configure_stage2_metadata(
        qwen.config,
        LatentTokenIds(*LATENT_TOKEN_IDS),
        latent_size=2,
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
    )
    Stage2Trainer(
        model=continued,
        args=TrainingArguments(
            output_dir=str(stage2_dir), report_to=[], use_cpu=True
        ),
        stage2_peft_metadata=metadata,
    ).save_model(stage2_dir)
    processor.save_pretrained(stage2_dir)
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
    monkeypatch.setattr(
        main.AutoProcessor,
        "from_pretrained",
        lambda source: processor_sources.append(source) or processor,
    )
    monkeypatch.setattr(
        main.AutoConfig,
        "from_pretrained",
        lambda source: tiny_original_qwen().config,
    )
    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda source, **unused: tiny_original_qwen(),
    )
    _stub_empty_stage2_data(monkeypatch)

    components = main.build_stage2_training_components(args)

    assert processor_sources == [str(stage2_dir)]
    assert components.trainer_resume_path == str(stage2_dir)
    assert components.peft_metadata == metadata
    validate_stage2_parameter_boundary(
        components.model, tuning, allow_sparse_token_rows=True
    )
