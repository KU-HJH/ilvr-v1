"""Historical metadata may enable inference, never training or disk rewriting."""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import Qwen2_5_VLForConditionalGeneration

import main
from stage2.checkpointing import (
    classify_stage2_artifact,
    decode_stage2_inference_metadata_snapshot,
    inspect_stage2_artifact,
    load_stage2_peft_metadata,
    resolve_stage2b_initialization_decision,
)
from stage2.peft import load_stage2_model_for_inference, resolve_stage2_qwen_model
from tests.stage2.test_routing import stage2_args
from tests.stage2.test_stage1_checkpoint_handoff import (
    tiny_tag_free_original_qwen,
    write_sparse_stage2_artifact,
)


HISTORICAL_EXTRAS = {
    "rollout_backend": "no_cache",
    "prefix_dropout_semantics": "resample_per_prefix",
    "stage2_cache_activation_checkpointing": False,
    "execution_backend": "shared_prefix_3pass",
}
pytestmark = pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")


@pytest.fixture(scope="module")
def historical_artifact():
    storage = Path("checkpoints/dil-bce-schema7-cpu")
    storage.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="inference-", dir=storage) as temporary:
        root = Path(temporary)
        stage1 = root / "stage1"
        current = root / "current"
        historical = root / "historical"
        base = root / "base"
        processor, _, _, trained = write_sparse_stage2_artifact(stage1, current)
        tiny_tag_free_original_qwen().save_pretrained(base)
        config_path = base / "config.json"
        config = json.loads(config_path.read_bytes())
        config["architectures"] = []
        config_path.write_text(json.dumps(config), encoding="utf-8")
        shutil.copytree(current, historical)
        metadata_path = historical / "stage2_peft_metadata.json"
        metadata = json.loads(metadata_path.read_bytes())
        metadata.update(schema_version=7, **HISTORICAL_EXTRAS)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        (historical / "config.json").unlink(missing_ok=True)
        shutil.rmtree(stage1)
        yield SimpleNamespace(root=root, current=current, historical=historical,
                              base=base, processor=processor, trained=trained)


def _assert_same_model(actual, expected, *, exact_runtime=False):
    # Missing LoRA A/B, DoRA magnitudes, selective input rows or output rows
    # must fail even if logits happen to agree for one scaffold.
    def adapter_and_rows(model):
        return {
            name: tensor for name, tensor in model.state_dict().items()
            if "lora_" in name or "trainable_tokens_delta" in name
            or name.endswith("output_delta") or name.endswith("token_ids")
        }

    # PEFT replaces the three effective input rows. Their unused underlying
    # base embeddings are randomly resized and are not serialized adapter state.
    actual_state, expected_state = adapter_and_rows(actual), adapter_and_rows(expected)
    assert actual_state.keys() == expected_state.keys()
    for name in expected_state:
        torch.testing.assert_close(actual_state[name], expected_state[name], rtol=0, atol=0)
    scaffold = torch.tensor([[1, 126, 125, 125, 127, 2, 126, 125, 125, 127, 3]])
    with torch.inference_mode():
        actual_output = actual(input_ids=scaffold, attention_mask=torch.ones_like(scaffold),
                               return_stage2_latent_states=True)
        expected_output = expected(input_ids=scaffold, attention_mask=torch.ones_like(scaffold),
                                   return_stage2_latent_states=True)
    # Trainable versus frozen PEFT DoRA uses slightly different floating-point
    # arithmetic; v4 versus v7 frozen reloads must agree exactly.
    tolerances = {"rtol": 0, "atol": 0} if exact_runtime else {}
    torch.testing.assert_close(actual_output.logits, expected_output.logits, **tolerances)
    assert len(actual_output.stage2_latent_states) == len(expected_output.stage2_latent_states) == 4
    for actual_state, expected_state in zip(actual_output.stage2_latent_states,
                                          expected_output.stage2_latent_states, strict=True):
        torch.testing.assert_close(actual_state, expected_state, **tolerances)
    assert not any(parameter.requires_grad for parameter in actual.parameters())


def test_schema7_actual_inference_roundtrip_preserves_raw_artifact(historical_artifact):
    artifact = historical_artifact
    metadata_path = artifact.historical / "stage2_peft_metadata.json"
    before = metadata_path.read_bytes()
    assert not (artifact.historical / "config.json").exists()
    native = load_stage2_model_for_inference(artifact.base, artifact.current, artifact.processor)
    historical = load_stage2_model_for_inference(artifact.base, artifact.historical, artifact.processor)
    _assert_same_model(historical, native, exact_runtime=True)
    _assert_same_model(historical, artifact.trained.eval())
    assert not hasattr(resolve_stage2_qwen_model(historical).config, "ilvr_context_preference")
    assert metadata_path.read_bytes() == before
    assert json.loads(before)["schema_version"] == 7
    inspection = inspect_stage2_artifact(artifact.historical, for_inference=True)
    decoded = decode_stage2_inference_metadata_snapshot(inspection.peft_metadata_snapshot)
    assert decoded.schema_version == 7
    assert inspection.peft_metadata_snapshot.metadata_json == before
    with pytest.raises(ValueError, match="unsupported Stage 2 PEFT metadata schema"):
        decoded.to_dict()


@pytest.mark.parametrize(("field", "value"), [
    ("rollout_backend", "differentiable_cache"), ("rollout_backend", None),
    ("prefix_dropout_semantics", "reuse_cached_prefix"), ("prefix_dropout_semantics", 0),
    ("stage2_cache_activation_checkpointing", True), ("stage2_cache_activation_checkpointing", 0),
    ("execution_backend", None), ("execution_backend", "gated"),
    ("execution_backend", "insertion"), ("execution_backend", "unknown"),
    ("schema_version", True), ("schema_version", 5), ("schema_version", 6), ("schema_version", 8),
    ("latent_size", "8"), ("latent_pad_token_id", 1),
    ("unknown_field", "ignored"),
    *[(field, "DELETE") for field in HISTORICAL_EXTRAS],
    ("pretrained_model_vocab_size", "DELETE"),
])
def test_schema7_invalid_metadata_fails_before_base_loading(historical_artifact, field, value):
    artifact = historical_artifact
    with TemporaryDirectory(prefix="invalid-", dir=artifact.root) as temporary:
        path = Path(temporary) / "artifact"
        shutil.copytree(artifact.historical, path)
        metadata_path = path / "stage2_peft_metadata.json"
        metadata = json.loads(metadata_path.read_bytes())
        if value == "DELETE":
            metadata.pop(field)
        else:
            metadata[field] = value
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        calls = []
        with pytest.raises(ValueError):
            load_stage2_model_for_inference(artifact.base, path, artifact.processor,
                                           load_model=lambda *args, **kwargs: calls.append("base"))
        assert calls == []


@pytest.mark.parametrize("corruption", ["adapter_rank", "missing_magnitude", "sparse_shape"])
def test_schema7_retains_adapter_and_sparse_tensor_checks(historical_artifact, corruption):
    artifact = historical_artifact
    with TemporaryDirectory(prefix="corrupt-", dir=artifact.root) as temporary:
        path = Path(temporary) / "artifact"
        shutil.copytree(artifact.historical, path)
        if corruption == "adapter_rank":
            config_path = path / "adapter_config.json"
            config = json.loads(config_path.read_bytes())
            config["r"] = 8
            config_path.write_text(json.dumps(config), encoding="utf-8")
        elif corruption == "missing_magnitude":
            tensor_path = path / "adapter_model.safetensors"
            tensors = load_file(tensor_path)
            tensors.pop(next(name for name in tensors if "lora_magnitude_vector" in name))
            save_file(tensors, tensor_path)
        else:
            tensor_path = path / "ilvr_token_rows.safetensors"
            with safe_open(tensor_path, framework="pt") as payload:
                metadata = payload.metadata()
            tensors = load_file(tensor_path)
            tensors["output_delta"] = tensors["output_delta"][:, :-1].contiguous()
            save_file(tensors, tensor_path, metadata=metadata)
        calls = []
        with pytest.raises(ValueError):
            load_stage2_model_for_inference(
                artifact.base, path, artifact.processor,
                load_model=lambda *args, **kwargs: calls.append("base"),
            )
        assert calls == []


@pytest.mark.parametrize("consumer", ["load", "inspect", "classify", "training", "stage2b"])
def test_schema7_remains_rejected_by_default_and_training_entrypoints(historical_artifact, consumer):
    artifact = historical_artifact
    with pytest.raises(ValueError, match="unsupported Stage 2 PEFT metadata schema"):
        if consumer == "load":
            load_stage2_peft_metadata(artifact.historical)
        elif consumer == "inspect":
            inspect_stage2_artifact(artifact.historical)
        elif consumer == "classify":
            classify_stage2_artifact(artifact.historical)
        elif consumer == "training":
            # Real main entrypoint, before processor/model/Trainer or optimizer load.
            main.build_stage2_training_components(stage2_args(
                load_model_path=str(artifact.base),
                resume_from_checkpoint_path=str(artifact.historical),
                tuning_method="dora", latent_size=2,
            ))
        else:
            resolve_stage2b_initialization_decision(
                stage2a_checkpoint_path=str(artifact.historical),
                load_model_path=str(artifact.base),
            )


@pytest.mark.parametrize("module_name", ["eval", "evaluate_deepseed"])
def test_schema7_evaluator_routes_load_real_model_from_immutable_inspection(
    historical_artifact, module_name, monkeypatch,
):
    artifact = historical_artifact
    module = importlib.import_module(module_name)
    with TemporaryDirectory(prefix="route-", dir=artifact.root) as temporary:
        path = Path(temporary) / "artifact"
        shutil.copytree(artifact.historical, path)
        metadata_path = path / "stage2_peft_metadata.json"
        before = metadata_path.read_bytes()
        original_inspect = module.inspect_stage2_artifact
        inspections = []

        def inspect(*args, **kwargs):
            result = original_inspect(*args, **kwargs)
            inspections.append(result)
            return result

        def processor(*args, **kwargs):
            # Model loading must use the retained bytes, not re-open rewritten metadata.
            metadata_path.write_bytes(b"invalid after inspection")
            return artifact.processor

        monkeypatch.setattr(module, "inspect_stage2_artifact", inspect)
        monkeypatch.setattr(module.AutoProcessor, "from_pretrained", processor)
        # Replace only GPU/FlashAttention policy; the real CPU base and PEFT loader run.
        monkeypatch.setattr(module, "load_model", lambda source, cache:
                            Qwen2_5_VLForConditionalGeneration.from_pretrained(source))
        if module_name == "eval":
            monkeypatch.setattr(module, "validate_qwen2_5_vl_attention_backends", lambda *a, **kw: None)
        loaded, loaded_processor = module.load_evaluation_artifacts(
            model_dir=str(path), base_model_dir=str(artifact.base), cache_dir="cache",
        )
        assert loaded_processor is artifact.processor
        assert len(inspections) == 1
        assert inspections[0].peft_metadata_snapshot.metadata_json == before
        assert json.loads(before)["schema_version"] == 7
        _assert_same_model(loaded, artifact.trained.eval())
