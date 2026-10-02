"""Same-run DIL resume must reach real preflight/model loading and Trainer dispatch."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest
from transformers import Trainer, TrainingArguments

import main
from deepspeed_runtime import write_deepspeed_runtime_sidecar
from peft_common import resolve_qwen2_5_vl_model
from stage2.trainer import Stage2Trainer
from step_preference.config import StepPreferenceConfig
from step_preference.dil_identity import build_dil_identity
from tests.stage1.test_checkpointing import TinyStage1Processor
from tests.stage2.test_stage1_checkpoint_handoff import (
    isolated_accelerate_state, tiny_tag_free_original_qwen, write_sparse_stage2_artifact,
)
from utils_deepseed import parse_args

REPO = Path(__file__).resolve().parents[2]


def resume_argv(checkpoint, base, data, selector="dil_bce_rf"):
    argv = [
        "--stage", "stage2", "--load_model_path", str(base),
        "--resume_from_checkpoint_path", str(checkpoint),
        "--data_path", str(data), "--image_root", str(REPO / "data"),
        "--step_preference_enabled", "--step_preference_mode", "stage2b",
        "--step_preference_loss_type", selector, "--step_preference_beta", "0.5",
        "--step_shared_prefix_enabled", "--no-step_visual_off_suppress_latent_boundaries",
        "--step_entropy_weight", "0", "--attention_implementation", "sdpa",
        "--vision_attention_implementation", "flash_attention_2",
        "--latent_size", "2", "--no-gradient_checkpointing", "--max_steps", "212",
    ]
    if selector in ("dil_bce_rf_topk", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_both_upper"):
        argv += ["--step_preference_topk", "8", "--step_preference_lambda_topk_mass", "1"]
    return argv


@pytest.fixture
def resume_artifact(tmp_path):
    """Real tiny sparse DoRA artifact, original-base weights and processor files."""
    artifact, base = tmp_path / "checkpoint-1", tmp_path / "base"
    processor, _, metadata, model = write_sparse_stage2_artifact(tmp_path / "stage1", artifact)
    tiny_tag_free_original_qwen().save_pretrained(base)
    payload = json.loads((base / "config.json").read_text())
    payload["architectures"] = []
    (base / "config.json").write_text(json.dumps(payload))
    (artifact / "trainer_state.json").write_text(json.dumps({"max_steps": 212, "global_step": 1}))
    with tempfile.TemporaryDirectory(prefix="dil-resume-test-", dir=REPO / "data") as directory:
        data = Path(directory) / "empty.jsonl"
        data.write_text("")
        yield artifact, base, data, processor, metadata, model


def save_identity(artifact, selector):
    values = dict(enabled=True, mode="stage2b", objective="context_preference",
                  direct_from_stage1=True, preference_loss_type=selector, beta=0.5,
                  shared_prefix_enabled=True, visual_off_suppress_latent_boundaries=False,
                  lambda_entropy=0.)
    if selector in ("dil_bce_rf_topk", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_both_upper"):
        values.update(preference_topk=8, lambda_topk_mass=1.)
    payload = json.loads((artifact / "config.json").read_text())
    payload["ilvr_context_preference"] = build_dil_identity(StepPreferenceConfig(**values))
    (artifact / "config.json").write_text(json.dumps(payload))


def cpu_loading_boundaries(monkeypatch):
    # Only GPU attention selection is substituted: actual config, processor,
    # base weights, PEFT weights, sparse rows, resolver and validators stay real.
    monkeypatch.setattr(main.AutoProcessor, "from_pretrained",
                        lambda source, **kwargs: TinyStage1Processor.from_pretrained(source))
    def cpu_mixed_load(source, **kwargs):
        return kwargs["load_model"](source, attn_implementation="sdpa")
    monkeypatch.setattr(main, "load_qwen2_5_vl_with_attention", cpu_mixed_load)
    monkeypatch.setattr(main, "validate_qwen2_5_vl_attention_backends", lambda *a, **k: None)
    # No image/tokenization workload in this checkpoint-routing witness.
    monkeypatch.setattr(main, "filter_stage2_examples", lambda **kwargs: ([], None))
    monkeypatch.setattr(main, "Stage2DataCollator", lambda *a, **k: lambda batch: batch)


@pytest.mark.parametrize("selector", ["dil_bce_rf", "dil_bce_rf_topk", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_both_upper"])
def test_real_resume_entry_reloads_sparse_weights_and_reaches_parent_train(
    resume_artifact, monkeypatch, selector,
):
    """Catches fresh-init routing, checkpoint loss or bypassed identity validation."""
    artifact, base, data, _, metadata, original = resume_artifact
    save_identity(artifact, selector)
    args = parse_args(resume_argv(artifact, base, data, selector))
    cpu_loading_boundaries(monkeypatch)
    components = main.build_stage2_training_components(args)
    assert components.trainer_resume_path == str(artifact)
    assert components.peft_metadata == metadata
    assert args.step_preference_config.phase_steps is None
    assert components.processor.tokenizer.convert_tokens_to_ids("<|latent_pad|>") == 125
    import torch
    hidden = torch.linspace(-.4, .9, 16).reshape(2, 8)
    torch.testing.assert_close(
        resolve_qwen2_5_vl_model(components.model).lm_head(hidden),
        resolve_qwen2_5_vl_model(original).lm_head(hidden),
    )
    # CPU configuration replaces BF16/GPU SFT setup; run_stage2, component
    # construction, Stage2Trainer.train and its resume identity gates stay real.
    monkeypatch.setattr(main, "build_stage2_training_arguments", lambda selected: TrainingArguments(
        output_dir=str(artifact.parent / "output"), use_cpu=True, report_to=[], max_steps=selected.max_steps))
    calls = []
    def parent_train(self, *unused, resume_from_checkpoint=None, **kwargs):
        calls.append(resume_from_checkpoint)
        return SimpleNamespace(training_loss=0.)
    monkeypatch.setattr(Trainer, "train", parent_train)
    trainer = Stage2Trainer(model=components.model, args=TrainingArguments(
        output_dir=str(artifact.parent / "output"), use_cpu=True, report_to=[]),
        step_preference_config=components.step_preference_config)
    write_deepspeed_runtime_sidecar(artifact, trainer.deepspeed_runtime_profile)
    args.save_final_model = False
    main.run_stage2(args)
    assert calls == [str(artifact)]
    # The trainer itself must still reject changes after preflight/model loading.
    payload = json.loads((artifact / "config.json").read_text())
    payload["ilvr_context_preference"]["beta"] = .75
    (artifact / "config.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity"):
        trainer.train(resume_from_checkpoint=str(artifact))
    assert calls == [str(artifact)]


@pytest.mark.parametrize("mutation", ["missing_identity", "changed_identity", "budget", "missing_budget"])
def test_resume_preflight_rejects_identity_or_budget_before_model_load(resume_artifact, monkeypatch, mutation):
    artifact, base, data, *_ = resume_artifact
    save_identity(artifact, "dil_bce_rf")
    if mutation in ("missing_identity", "changed_identity"):
        path = artifact / "config.json"
        payload = json.loads(path.read_text())
        if mutation == "missing_identity":
            payload.pop("ilvr_context_preference")
        else:
            payload["ilvr_context_preference"]["beta"] = .75
        path.write_text(json.dumps(payload))
    elif mutation == "missing_budget":
        (artifact / "trainer_state.json").unlink()
    argv = resume_argv(artifact, base, data)
    if mutation == "budget":
        argv[-1] = "106"
    args = parse_args(argv)
    monkeypatch.setattr(main.Qwen2_5_VLForConditionalGeneration, "from_pretrained",
                        lambda *a, **k: pytest.fail("loaded weights for invalid resume"))
    with pytest.raises((ValueError, FileNotFoundError), match="identity|max_steps|trainer_state"):
        main.build_stage2_training_components(args)


@pytest.mark.parametrize("extra", [
    ["--stage2b_direct_from_stage1"], ["--stage1_adapter_path", "fresh-stage1"],
    ["--stage2a_checkpoint_path", "fresh-stage2a"], ["--stage2_total_steps", "212"],
])
def test_resume_parser_rejects_mixed_fresh_initialization_and_ambiguous_budget(extra):
    with pytest.raises(ValueError, match="resume|mutually exclusive"):
        parse_args(resume_argv("saved", "base", "data") + extra)


def test_dpo_stage2b_resume_remains_rejected():
    argv = resume_argv("saved", "base", "data", "dpo") + ["--stage2a_checkpoint_path", "stage2a"]
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_args(argv)


@pytest.mark.parametrize("mutation,match", [
    ("vocabulary", "topk|vocab"), ("boundaries", "supervise_latent_boundaries"),
    ("original_base", "original base|stage1"), ("sparse_tokens", "token"),
])
def test_resume_keeps_resolved_model_and_sparse_provenance_guards(
    resume_artifact, monkeypatch, mutation, match,
):
    """Catches resume routing around existing base/token/runtime validation."""
    artifact, base, data, *_ = resume_artifact
    save_identity(artifact, "dil_bce_rf_topk")
    argv = resume_argv(artifact, base, data, "dil_bce_rf_topk")
    if mutation == "vocabulary":
        argv[argv.index("--step_preference_topk") + 1] = "128"
        path = artifact / "config.json"
        payload = json.loads(path.read_text())
        payload["ilvr_context_preference"]["topk"] = 128
        path.write_text(json.dumps(payload))
    elif mutation == "boundaries":
        argv += ["--no-supervise_latent_boundaries"]
    elif mutation == "original_base":
        path = base / "config.json"
        payload = json.loads(path.read_text())
        payload["stage"] = "stage2"
        path.write_text(json.dumps(payload))
    else:
        path = artifact / "tokenizer.json"
        payload = json.loads(path.read_text())
        payload["ids"]["<|latent_pad|>"] = 124
        path.write_text(json.dumps(payload))
    cpu_loading_boundaries(monkeypatch)
    args = parse_args(argv)
    with pytest.raises(ValueError, match=match):
        main.build_stage2_training_components(args)


def test_resume_training_arguments_keep_explicit_saved_budget(monkeypatch):
    """Catches a fresh allocation being reintroduced before SFTConfig creation."""
    args = parse_args(resume_argv("saved", "base", "data"))
    # Observe our emitted SFT settings without requiring a BF16 CUDA device.
    monkeypatch.setattr(main, "SFTConfig", SimpleNamespace)
    arguments = main.build_stage2_training_arguments(args)
    assert arguments.max_steps == 212
    assert arguments.warmup_steps == 10
