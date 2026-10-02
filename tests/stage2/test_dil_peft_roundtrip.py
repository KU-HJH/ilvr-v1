"""DIL objective identity survives real PEFT artifact save and inference reload."""

import json

import pytest
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments

from peft_common import resolve_qwen2_5_vl_model
from stage2.peft import load_stage2_model_for_inference
from stage2.trainer import Stage2Trainer
from deepspeed_runtime import write_deepspeed_runtime_sidecar
from step_preference.config import StepPreferenceConfig
from step_preference.dil_identity import build_dil_identity
from tests.stage2.test_stage1_checkpoint_handoff import (
    tiny_tag_free_original_qwen,
    write_sparse_stage2_artifact,
)
from tests.stage2.helpers import tiny_stage2_qwen


def bce_config(kind="dil_bce_rf"):
    return StepPreferenceConfig(
        enabled=True, mode="stage2b", objective="context_preference",
        preference_loss_type=kind, beta=0.5,
        preference_topk=8 if kind in ("dil_bce_rf_topk_reverse", "dil_bce_rf_topk_both_upper") else None,
        lambda_topk_mass=1. if kind in ("dil_bce_rf_topk_reverse", "dil_bce_rf_topk_both_upper") else 0.,
        lambda_entropy=0.0, direct_from_stage1=True,
        shared_prefix_enabled=True,
        visual_off_suppress_latent_boundaries=False,
    )


@pytest.fixture(params=["dil_bce_rf", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_both_upper"])
def bce_kind(request):
    return request.param


@pytest.fixture
def bce_artifact(tmp_path, bce_kind):
    stage1, artifact, base = (tmp_path / name for name in ("stage1", "stage2", "base"))
    processor, _, metadata, trained = write_sparse_stage2_artifact(stage1, artifact)
    Stage2Trainer(
        model=trained,
        args=TrainingArguments(output_dir=str(artifact), report_to=[], use_cpu=True),
        stage2_peft_metadata=metadata,
        step_preference_config=bce_config(bce_kind),
    ).save_model(artifact)
    tiny_tag_free_original_qwen().save_pretrained(base)
    source_config = base / "config.json"
    source_payload = json.loads(source_config.read_text())
    source_payload["architectures"] = []
    source_config.write_text(json.dumps(source_payload))
    return processor, trained, artifact, base


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_bce_peft_save_reload_preserves_identity_logits_and_latents(bce_artifact, bce_kind):
    processor, trained, artifact, base = bce_artifact
    identity = build_dil_identity(bce_config(bce_kind))
    assert json.loads((artifact / "config.json").read_text())["ilvr_context_preference"] == identity
    assert not (artifact / "optimizer.pt").exists()
    reloaded = load_stage2_model_for_inference(base, artifact, processor)
    assert resolve_qwen2_5_vl_model(reloaded).config.ilvr_context_preference == identity
    scaffold = torch.tensor([[1, 126, 125, 125, 127, 2, 126, 125, 125, 127, 3]])
    mask = torch.ones_like(scaffold)
    with torch.inference_mode():
        before = trained.eval()(input_ids=scaffold, attention_mask=mask,
                                return_stage2_latent_states=True)
        after = reloaded(input_ids=scaffold, attention_mask=mask,
                         return_stage2_latent_states=True)
    torch.testing.assert_close(after.logits, before.logits)
    for actual, expected in zip(after.stage2_latent_states, before.stage2_latent_states, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_malformed_identity_rejected_before_base_weight_load(bce_artifact, tmp_path):
    processor, _, artifact, base = bce_artifact
    payload_path = artifact / "config.json"
    payload = json.loads(payload_path.read_text())
    payload["ilvr_context_preference"]["future"] = 1
    payload_path.write_text(json.dumps(payload))
    model_loads = []
    with pytest.raises(ValueError, match="identity"):
        load_stage2_model_for_inference(
            base, artifact, processor,
            load_model=lambda *args, **kwargs: model_loads.append((args, kwargs)),
        )
    assert model_loads == []


def test_train_resume_normalizes_true_false_and_path_before_trainer_restore(
    bce_artifact, bce_kind, monkeypatch
):
    _, trained, artifact, _ = bce_artifact
    config = bce_config(bce_kind)
    numbered = artifact / "checkpoint-1"
    saver = Stage2Trainer(
        model=trained,
        args=TrainingArguments(output_dir=str(artifact), report_to=[], use_cpu=True),
        step_preference_config=config,
    )
    saver.save_model(numbered)
    write_deepspeed_runtime_sidecar(numbered, saver.deepspeed_runtime_profile)
    assert json.loads((numbered / "config.json").read_text())["ilvr_context_preference"] == build_dil_identity(config)
    calls = []
    monkeypatch.setattr(Trainer, "train", lambda self, *args, resume_from_checkpoint=None, **kwargs:
                        calls.append(resume_from_checkpoint))
    saver.train(resume_from_checkpoint=False)
    saver.train(resume_from_checkpoint=True)
    saver.train(resume_from_checkpoint=numbered)
    assert calls == [None, str(numbered), numbered]
    payload_path = numbered / "config.json"
    payload = json.loads(payload_path.read_text())
    payload["ilvr_context_preference"]["beta"] = 0.75
    payload_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity"):
        saver.train(resume_from_checkpoint=numbered)
    with pytest.raises(ValueError, match="identity"):
        saver.train(resume_from_checkpoint=True)
    assert len(calls) == 3


def test_save_rejects_unsupervised_boundaries_before_config_write(bce_artifact, tmp_path):
    _, trained, _, _ = bce_artifact
    resolved = resolve_qwen2_5_vl_model(trained)
    resolved.config.supervise_latent_boundaries = False
    target = tmp_path / "invalid-save"
    trainer = Stage2Trainer(
        model=trained,
        args=TrainingArguments(output_dir=str(target), report_to=[], use_cpu=True),
        step_preference_config=bce_config(),
    )
    with pytest.raises(ValueError, match="supervised latent boundaries"):
        trainer.save_model(target)
    assert not (target / "config.json").exists()


def test_inference_rejects_symlinked_identity_config_before_base_load(bce_artifact, tmp_path):
    processor, _, artifact, base = bce_artifact
    config = artifact / "config.json"
    separate = tmp_path / "separate-config.json"
    separate.write_bytes(config.read_bytes())
    config.unlink()
    config.symlink_to(separate)
    model_loads = []
    with pytest.raises(ValueError, match="config|identity"):
        load_stage2_model_for_inference(
            base, artifact, processor,
            load_model=lambda *args, **kwargs: model_loads.append((args, kwargs)),
        )
    assert model_loads == []


def test_bce_full_model_save_reload_preserves_identity(tmp_path):
    model = tiny_stage2_qwen()
    model.config.supervise_latent_boundaries = True
    trainer = Stage2Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True),
        step_preference_config=bce_config(),
    )

    trainer.save_model(tmp_path)
    restored = Qwen2_5_VLForConditionalGeneration.from_pretrained(tmp_path)

    assert restored.config.ilvr_context_preference == build_dil_identity(bce_config())
    torch.testing.assert_close(restored.lm_head.weight, model.lm_head.weight)
    assert (tmp_path / "model.safetensors").is_file()
