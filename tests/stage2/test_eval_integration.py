import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers.modeling_utils as modeling_utils
from peft import PeftModel
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import PretrainedConfig

from .helpers import (
    ToyCausalLanguageModel,
    activate_stage2_adapter_deltas,
    tiny_stage2_qwen,
)
from stage2.checkpointing import (
    build_stage2_peft_metadata,
    save_stage2_peft_metadata,
)
from stage2.peft import (
    ALLOWED_STAGE2_TARGET_SUFFIXES,
    Stage2TuningConfig,
    build_stage2_tuned_model,
    resolve_stage2_qwen_model,
    resolve_stage2_target_module_names,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentRolloutEngine,
    LatentTokenIds,
)
from tests.stage2.test_stage1_checkpoint_handoff import (
    tiny_tag_free_original_qwen,
    write_stage1_dora_artifact,
    write_sparse_stage2_artifact,
)


@pytest.fixture(autouse=True)
def restore_evaluator_backend_precision():
    """Evaluator loading enables TF32; it must not leak into later numeric tests."""
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


MODULE_NAMES = ("eval", "evaluate_deepseed")


@pytest.mark.gpu
def test_evaluation_runtime_sidecar_counts_concrete_blocks_exclusively(monkeypatch, tmp_path):
    module = importlib.import_module("eval")
    from tests.stage2.test_attention_backend import _load_tiny_mixed_model
    model, *_ = _load_tiny_mixed_model()
    assert hasattr(module, "publish_evaluation_runtime_identity"), "evaluation runtime publisher missing"
    from step_preference import capacity_evidence
    monkeypatch.setattr(capacity_evidence, "git_identity", lambda: "a"*40)
    output=tmp_path / "runtime.json"
    module.publish_evaluation_runtime_identity(model, tmp_path, output)
    payload=json.loads(output.read_bytes())
    assert payload["language_attention_layer_count"] == payload["vision_attention_block_count"] == 2
    with pytest.raises(ValueError): module.publish_evaluation_runtime_identity(model,tmp_path,output)
    model.visual.blocks[0].attn=nn.Identity()
    rejected=tmp_path / "rejected.json"
    with pytest.raises(ValueError): module.publish_evaluation_runtime_identity(model,tmp_path,rejected)
    assert not rejected.exists()
LATENT_TOKEN_IDS = {
    "<|latent_pad|>": 90,
    "<|latent_start|>": 91,
    "<|latent_end|>": 92,
}


class FakeTokenizer:
    eos_token_id = 3
    unk_token_id = 0
    unk_token = "<unk>"

    def convert_tokens_to_ids(self, token):
        return LATENT_TOKEN_IDS.get(token, self.unk_token_id)

    def __call__(self, token, return_tensors):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[self.convert_tokens_to_ids(token)]])}


class FakeProcessor:
    def __init__(self, *, include_attention_mask=True, include_visual_inputs=True):
        self.tokenizer = FakeTokenizer()
        self.include_attention_mask = include_attention_mask
        self.include_visual_inputs = include_visual_inputs
        self.images = None
        self.decoded_ids = None
        self.decoded_id_batches = []
        self.input_ids = torch.tensor([[10, 11]])
        self.attention_mask = torch.tensor([[1, 1]])
        self.pixel_values = torch.tensor([[1.5, 2.5]])
        self.image_grid_thw = torch.tensor([[1, 2, 3]])

    def apply_chat_template(self, conversations, *, tokenize, add_generation_prompt):
        assert conversations[0]["content"][-1] == {
            "type": "text",
            "text": "question",
        }
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered prompt"

    def __call__(self, *, text, images, return_tensors, padding):
        assert text == ["rendered prompt"]
        assert return_tensors == "pt"
        assert padding is True
        self.images = images
        result = {"input_ids": self.input_ids}
        if self.include_attention_mask:
            result["attention_mask"] = self.attention_mask
        if self.include_visual_inputs:
            result["pixel_values"] = self.pixel_values
            result["image_grid_thw"] = self.image_grid_thw
        return result

    def batch_decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        self.decoded_ids = ids
        self.decoded_id_batches.append(ids)
        return [
            "<|im_start|>user question<|im_end|>"
            "<|im_start|>assistant Final answer: 42<|im_end|>"
        ]


class ModelWhoseGenerateRaises:
    device = torch.device("cpu")

    def __init__(self, *, start_token_id=91, batch_size=1):
        self.config = SimpleNamespace(
            stage="stage2",
            latent_size=2,
            latent_pad_token_id=90,
            latent_start_token_id=start_token_id,
            latent_end_token_id=92,
            max_position_embeddings=128,
        )
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.batch_size = batch_size
        self.model = ToyCausalLanguageModel(4)
        self.lm_head = nn.Linear(4, 128, bias=False)
        self.stage2_rollout_engine = LatentRolloutEngine()

    def prepare_stage2_base_inputs_embeds(
        self, *, input_ids, pixel_values, image_grid_thw
    ):
        return self.model.embed_tokens(input_ids)

    def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        return positions, torch.zeros(input_ids.shape[0], 1, device=input_ids.device)

    def generate(self, **kwargs):
        raise AssertionError("legacy model.generate was called")


class RecordingStage1Model:
    device = torch.device("cpu")

    def __init__(self):
        self.config = SimpleNamespace(
            stage="stage1",
            latent_size=2,
            latent_pad_token_id=90,
            latent_start_token_id=91,
            latent_end_token_id=92,
        )
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.model = ToyCausalLanguageModel(4)
        self.lm_head = nn.Linear(4, 128, bias=False)
        self.stage2_rollout_engine = LatentRolloutEngine()

    def prepare_stage2_base_inputs_embeds(
        self, *, input_ids, pixel_values, image_grid_thw
    ):
        return self.model.embed_tokens(input_ids)

    def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        return positions, torch.zeros(input_ids.shape[0], 1, device=input_ids.device)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def generate(self, **_kwargs):
        raise AssertionError("legacy model.generate was called")


class RecordingController:
    def __init__(self):
        self.calls = []
        self.sequences = torch.tensor([[10, 11, 21, 22]])

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            sequences=self.sequences,
            generated_suffix_ids=(21, 22),
            latent_start_count=1,
            latent_block_count=1,
            stop_reason="eos",
        )


class DeepSpeedShapedWrapper:
    device = torch.device("cpu")

    def __init__(self, module):
        self.module = module


class StartTokenHead(nn.Module):
    def forward(self, hidden):
        logits = hidden.new_full((hidden.shape[0], 128), -1e9)
        logits[:, LATENT_TOKEN_IDS["<|latent_start|>"]] = 0
        return logits


class RealControllerCapModel(ModelWhoseGenerateRaises):
    def __init__(self):
        super().__init__()
        self.config.max_position_embeddings = 5
        self.model = ToyCausalLanguageModel(4)
        self.lm_head = StartTokenHead()
        self.stage2_rollout_engine = LatentRolloutEngine()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def prepare_stage2_base_inputs_embeds(
        self, *, input_ids, pixel_values, image_grid_thw
    ):
        return self.get_input_embeddings()(input_ids)

    def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        return positions, torch.zeros(input_ids.shape[0], 1, device=input_ids.device)


def _import_entry_point(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    if module_name == "eval":
        monkeypatch.setattr(
            module,
            "load_qwen2_5_vl_with_attention",
            lambda source, **kwargs: kwargs["load_model"](
                source, **dict(kwargs["model_load_kwargs"])
            ),
        )
        monkeypatch.setattr(
            module,
            "validate_qwen2_5_vl_attention_backends",
            lambda *_args, **_kwargs: None,
        )
    return module


def _capture_transformers_flash_attention_error(model_class, monkeypatch, failure):
    config = PretrainedConfig()
    config._name_or_path = "tiny"

    if failure == "unsupported-model":
        class UnsupportedModel(model_class):
            _supports_flash_attn_2 = False

        checked_model_class = UnsupportedModel
    else:
        checked_model_class = model_class
        monkeypatch.setattr(
            modeling_utils, "is_flash_attn_2_available", lambda: False
        )
        monkeypatch.setattr(modeling_utils, "is_torch_npu_available", lambda: False)
        monkeypatch.setattr(
            modeling_utils.importlib.util, "find_spec", lambda _name: None
        )

    with pytest.raises((ImportError, ValueError)) as raised:
        checked_model_class._check_and_enable_flash_attn_2(config)
    return raised.value


def _saved_adapter_artifact(tmp_path, method="lora"):
    tuning = Stage2TuningConfig(
        method=method,
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
    )
    source_base = tiny_stage2_qwen()
    source_base.config.supervise_latent_boundaries = True
    fresh_base = copy.deepcopy(source_base)
    targets = resolve_stage2_target_module_names(source_base, tuning.target_modules)
    source = build_stage2_tuned_model(source_base, tuning).eval()
    activate_stage2_adapter_deltas(source)
    artifact_dir = tmp_path / "adapter"
    # Synthetic base has no on-disk config; PEFT deliberately assumes unchanged vocab.
    with pytest.warns(UserWarning, match=r"Could not find a config file in  - will assume that the vocabulary was not modified\."):
        source.save_pretrained(artifact_dir)
    save_stage2_peft_metadata(
        artifact_dir,
        build_stage2_peft_metadata(
            tuning=tuning,
            base_model_reference="tiny-stage1",
            model_config=resolve_stage2_qwen_model(source).config,
            token_ids=LatentTokenIds(
                pad_token_id=90,
                start_token_id=91,
                end_token_id=92,
            ),
            resolved_target_names=targets,
        ),
    )
    fresh_base.config.stage = "stage1"
    return artifact_dir, fresh_base


def _save_exact_base_source(model, directory):
    expected_architectures = list(model.config.architectures or [])
    model.save_pretrained(directory)
    model.config.architectures = expected_architectures
    config_path = directory / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["architectures"] = expected_architectures
    config_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )


@pytest.fixture(scope="module")
def sparse_eval_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("sparse-eval-inference")
    stage1_dir = root / "stage1"
    stage2_dir = root / "stage2"
    base_dir = root / "original-base"
    processor, _, _, source = write_sparse_stage2_artifact(
        stage1_dir, stage2_dir
    )
    _save_exact_base_source(tiny_tag_free_original_qwen(), base_dir)
    shutil.rmtree(stage1_dir)
    return SimpleNamespace(
        stage2_dir=stage2_dir,
        base_dir=base_dir,
        processor=processor,
        source=source.eval(),
    )


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_eval_cli_accepts_explicit_stage1_base(module_name, monkeypatch):
    """Omitting the base-model CLI route must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module_name,
            "--model_dir",
            "adapter",
            "--base_model_dir",
            "stage1",
            "--test_data_path",
            "test.jsonl",
            "--task_name",
            "task",
        ],
    )

    if module_name == "eval":
        monkeypatch.setattr(sys, "argv", [*sys.argv,"--runtime_identity_path","runtime.json"])
    assert module.get_eval_args().base_model_dir == "stage1"


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_eval_cli_defaults_to_legacy_single_worker_mode(module_name, monkeypatch):
    """Changing the default entrypoint output mode must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module_name,
            "--model_dir",
            "model",
            "--test_data_path",
            "test.jsonl",
            "--task_name",
            "task",
        ],
    )

    if module_name == "eval":
        monkeypatch.setattr(sys, "argv", [*sys.argv,"--runtime_identity_path","runtime.json"])
    args = module.get_eval_args()

    assert args.num_shards == 1
    assert args.shard_index == 0
    assert args.model_identity is None
    assert args.max_new_tokens == 4096
    assert args.max_sequence_length == 32768
    assert args.repetition_penalty == 1.05
    assert args.generation_limit_policy == "bounded"


def _run_main_with_fake_sample(
    module, monkeypatch, tmp_path, *, num_shards, shard_index=0, fail_index=None
):
    (tmp_path / "image.png").write_bytes(b"image")
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(
        "".join(
            json.dumps(
                {
                    "text_input": f"question-{index}",
                    "image_input": ["image.png"],
                    "original_final_answer": "42",
                }
            )
            + "\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "worker.jsonl"
    monkeypatch.setattr(
        module,
        "get_eval_args",
        lambda: SimpleNamespace(
            model_dir="relative-model",
            base_model_dir=None,
            test_data_path=str(dataset),
            task_name="zebra-cot",
            output_json_path=str(output),
            cache_dir="cache",
            max_new_tokens=8,
            max_sequence_length=32768,
            repetition_penalty=1.05,
            temperature=0.0,
            top_p=1.0,
            num_shards=num_shards,
            shard_index=shard_index,
            model_identity=None,
        ),
    )
    monkeypatch.setattr(module, "load_evaluation_artifacts", lambda **_kwargs: (object(), object()))

    def run_one(_model, _processor, sample, _gen_kwargs):
        assert _gen_kwargs["max_sequence_length"] == 32768
        assert _gen_kwargs["repetition_penalty"] == 1.05
        index = int(sample["text_input"].rsplit("-", 1)[1])
        if index == fail_index:
            raise RuntimeError("causal sample failure")
        result = {
            "raw_output": "raw answer",
            "assistant_only": "Final answer: 42",
            "extracted_final_answer": "42",
            "prompt_identity": "p" * 64,
            "prompt_token_length": 2,
            "generated_suffix_token_ids": [21, 22],
            "latent_start_count": 1,
            "latent_block_count": 1,
            "stop_reason": "eos",
        }
        if module.__name__ == "eval":
            result["inference_time"] = 1.25
        return result

    monkeypatch.setattr(module, "run_one_example", run_one)
    return dataset, output, module.main()


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_eval_entrypoint_resolves_dataset_relative_image_before_run_one(
    module_name, monkeypatch, tmp_path
):
    module = _import_entry_point(module_name, monkeypatch)
    dataset_root = tmp_path / "dataset"
    image = dataset_root / "images_comt/creation/12003.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), color="white").save(image)
    dataset = dataset_root / "subset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "text_input": "question",
                "image_input": ["images_comt/creation/12003.png"],
                "original_final_answer": "42",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.jsonl"
    monkeypatch.setattr(
        module,
        "get_eval_args",
        lambda: SimpleNamespace(
            model_dir="relative-model",
            base_model_dir=None,
            test_data_path=str(dataset),
            task_name="zebra-cot",
            output_json_path=str(output),
            cache_dir="cache",
            max_new_tokens=8,
            max_sequence_length=32768,
            repetition_penalty=1.05,
            temperature=0.0,
            top_p=1.0,
            num_shards=1,
            shard_index=0,
            model_identity=None,
        ),
    )
    monkeypatch.setattr(
        module, "load_evaluation_artifacts", lambda **_kwargs: (object(), object())
    )
    observed = []

    def run_one(_model, _processor, sample, _gen_kwargs):
        observed.append(sample["image_input"])
        result = {
            "raw_output": "raw answer",
            "assistant_only": "answer",
            "extracted_final_answer": "42",
        }
        if module_name == "eval":
            result["inference_time"] = 0.25
        return result

    monkeypatch.setattr(module, "run_one_example", run_one)

    module.main()

    assert observed == [[str(image.resolve())]]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_default_single_worker_records_shared_answer_scoring(
    module_name, monkeypatch, tmp_path
):
    """Omitting parse fields or a denominator must fail this output contract."""
    module = _import_entry_point(module_name, monkeypatch)
    _dataset, output, result = _run_main_with_fake_sample(
        module, monkeypatch, tmp_path, num_shards=1
    )

    if module_name == "eval":
        row = {
            "index": 0,
            "task_name": "zebra-cot",
            "image_input": ["image.png"],
            "prompt_identity": "p" * 64,
            "prompt_token_length": 2,
            "generated_suffix_token_ids": [21, 22],
            "raw_output": "raw answer",
            "assistant_only": "Final answer: 42",
            "prediction": "Final answer: 42",
            "extracted_final_answer": "42",
            "prediction_parse_status": "parsed",
            "prediction_parse_detail": None,
            "prediction_normalized": "42",
            "gold_final_answer": "42",
            "gold_parse_status": "parsed",
            "gold_parse_detail": None,
            "gold_normalized": "42",
            "invalid_gold_reason": None,
            "latent_start_count": 1,
            "latent_block_count": 1,
            "stop_reason": "eos",
            "match": True,
            "inference_time_sec": 1.25,
        }
        rows = [dict(row, index=index) for index in range(6)]
        rows.append(
            {
                "summary": True,
                "num_samples": 6,
                "average_inference_time_sec": 1.25,
                "total_inference_time_sec": 7.5,
                "accuracy": 1.0,
                "success": 6,
                "correct": 6,
                "source_count": 6,
                "valid_gold_count": 6,
                "invalid_gold_count": 0,
                "invalid_gold": [],
                "coverage_accuracy": 1.0,
                "valid_gold_accuracy": 1.0,
            }
        )
    else:
        row = {
            "index": 0,
            "task_name": "zebra-cot",
            "prompt_identity": "p" * 64,
            "prompt_token_length": 2,
            "generated_suffix_token_ids": [21, 22],
            "raw_output": "raw answer",
            "assistant_only": "Final answer: 42",
            "prediction": "Final answer: 42",
            "extracted_final_answer": "42",
            "prediction_parse_status": "parsed",
            "prediction_parse_detail": None,
            "prediction_normalized": "42",
            "gold_final_answer": "42",
            "gold_parse_status": "parsed",
            "gold_parse_detail": None,
            "gold_normalized": "42",
            "invalid_gold_reason": None,
            "latent_start_count": 1,
            "latent_block_count": 1,
            "stop_reason": "eos",
            "vision_ablation": None,
            "input_image_count": None,
            "match": True,
        }
        rows = [dict(row, index=index) for index in range(6)]

    expected = "".join(
        json.dumps(row, ensure_ascii=False, **({"sort_keys":True,"separators":(",",":")} if module_name == "eval" else {})) + "\n" for row in rows
    )
    assert result is None
    assert output.read_text(encoding="utf-8") == expected
    assert not Path(f"{output}.manifest.json").exists()


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_standard_tasks_score_assistant_content_and_record_parse_fields(
    module_name, monkeypatch, tmp_path
):
    """Scoring a stale extracted field instead of assistant output must fail."""
    module = _import_entry_point(module_name, monkeypatch)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "text_input": "question",
                "image_input": [],
                "original_final_answer": "42",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.jsonl"
    monkeypatch.setattr(
        module,
        "get_eval_args",
        lambda: SimpleNamespace(
            model_dir="model",
            base_model_dir=None,
            test_data_path=str(dataset),
            task_name="zebra-cot",
            output_json_path=str(output),
            cache_dir="cache",
            max_new_tokens=8,
            max_sequence_length=32768,
            repetition_penalty=1.05,
            temperature=0.0,
            top_p=1.0,
            num_shards=1,
            shard_index=0,
            model_identity=None,
        ),
    )
    monkeypatch.setattr(
        module, "load_evaluation_artifacts", lambda **_kwargs: (object(), object())
    )

    def run_one(_model, _processor, _sample, _gen_kwargs):
        result = {
            "raw_output": "raw output",
            "assistant_only": "not 42",
            "extracted_final_answer": "42",
        }
        if module_name == "eval":
            result["inference_time"] = 0.25
        return result

    monkeypatch.setattr(module, "run_one_example", run_one)
    module.main()

    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["match"] is False
    assert row["prediction_parse_status"] == "parsed"
    assert row["prediction_normalized"] == "not 42"
    assert row["gold_parse_status"] == "parsed"
    assert row["gold_normalized"] == "42"
    assert row["invalid_gold_reason"] is None


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_inference_error_rows_preserve_invalid_gold_for_summary(
    module_name, monkeypatch, tmp_path
):
    """Dropping gold validity when inference fails must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "text_input": "failure",
                    "image_input": [],
                    "original_final_answer": None,
                },
                {
                    "text_input": "success",
                    "image_input": [],
                    "original_final_answer": "42",
                },
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output.jsonl"
    monkeypatch.setattr(
        module,
        "get_eval_args",
        lambda: SimpleNamespace(
            model_dir="model",
            base_model_dir=None,
            test_data_path=str(dataset),
            task_name="zebra-cot",
            output_json_path=str(output),
            cache_dir="cache",
            max_new_tokens=8,
            max_sequence_length=32768,
            repetition_penalty=1.05,
            temperature=0.0,
            top_p=1.0,
            num_shards=1,
            shard_index=0,
            model_identity=None,
        ),
    )
    monkeypatch.setattr(
        module, "load_evaluation_artifacts", lambda **_kwargs: (object(), object())
    )

    def run_one(_model, _processor, sample, _gen_kwargs):
        if sample["text_input"] == "failure":
            raise RuntimeError("causal inference failure")
        result = {
            "raw_output": "raw output",
            "assistant_only": "42",
            "extracted_final_answer": "42",
        }
        if module_name == "eval":
            result["inference_time"] = 0.25
        return result

    monkeypatch.setattr(module, "run_one_example", run_one)
    exit_status = module.main()
    if module_name == "eval":
        assert exit_status == 1, "caught inference errors must fail the evaluation process"

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    failed = next(row for row in rows if row.get("index") == 0)
    assert failed["error"] == "causal inference failure"
    assert failed["gold_final_answer"] is None
    assert failed["gold_parse_status"] == "invalid_gold"
    assert failed["gold_parse_detail"] == "missing"
    assert failed["gold_normalized"] == ""
    assert failed["invalid_gold_reason"] == "missing"

    if module_name == "eval":
        summary = rows[-1]
        assert summary["valid_gold_count"] == 1
        assert summary["invalid_gold"] == [{"index": 0, "reason": "missing"}]
        assert summary["coverage_accuracy"] == pytest.approx(1 / 2)
        assert summary["valid_gold_accuracy"] == 1.0


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_four_shard_worker_uses_global_indices_and_writes_success_sidecar(
    module_name, monkeypatch, tmp_path
):
    """Enumerating the filtered shard locally must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    dataset, output, result = _run_main_with_fake_sample(
        module, monkeypatch, tmp_path, num_shards=4, shard_index=1
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    manifest = json.loads(
        Path(f"{output}.manifest.json").read_text(encoding="utf-8")
    )
    dataset_stat = dataset.stat()
    assert result == 0
    assert [row["index"] for row in rows] == [1, 5]
    assert all("summary" not in row for row in rows)
    assert manifest["dataset_identity"] == {
        "path": str(dataset.absolute()),
        "size": dataset_stat.st_size,
        "mtime_ns": dataset_stat.st_mtime_ns,
    }
    assert manifest["model_identity"] == f"direct:{Path('relative-model').absolute()}"
    assert manifest["generation_config"] == {
        "max_new_tokens": 8,
        "max_sequence_length": 32768,
        "repetition_penalty": 1.05,
        "temperature": 0.0,
        "top_p": 1.0,
        "vision_ablation": "none",
        "generation_limit_policy": "bounded",
    }
    assert manifest["expected_source_indices"] == [1, 5]
    assert manifest["processed_source_indices"] == [1, 5]
    assert manifest["worker_exit_status"] == 0


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_four_shard_worker_retains_error_and_returns_nonzero(
    module_name, monkeypatch, tmp_path
):
    """Treating a partial diagnostic shard as successful must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    _dataset, output, result = _run_main_with_fake_sample(
        module,
        monkeypatch,
        tmp_path,
        num_shards=4,
        shard_index=1,
        fail_index=5,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    manifest = json.loads(Path(f"{output}.manifest.json").read_text())
    assert result == 1
    assert rows[-1]["index"] == 5
    assert rows[-1]["error"] == "causal sample failure"
    assert manifest["processed_source_indices"] == [1, 5]
    assert manifest["worker_exit_status"] == 1


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_empty_four_shard_worker_still_publishes_mergeable_success_sidecar(
    module_name, monkeypatch, tmp_path
):
    """Returning before the zero-sample shard sidecar must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("", encoding="utf-8")
    output = tmp_path / "worker.jsonl"
    monkeypatch.setattr(
        module,
        "get_eval_args",
        lambda: SimpleNamespace(
            model_dir="model",
            base_model_dir=None,
            test_data_path=str(dataset),
            task_name="task",
            output_json_path=str(output),
            cache_dir="cache",
            max_new_tokens=8,
            max_sequence_length=32768,
            repetition_penalty=1.05,
            temperature=0.0,
            top_p=1.0,
            num_shards=4,
            shard_index=3,
            model_identity="validated-checkpoint-identity",
        ),
    )
    monkeypatch.setattr(
        module,
        "load_evaluation_artifacts",
        lambda **_kwargs: (object(), object()),
    )

    result = module.main()

    manifest = json.loads(Path(f"{output}.manifest.json").read_text())
    assert result == 0
    assert output.read_bytes() == b""
    assert manifest["expected_source_indices"] == []
    assert manifest["processed_source_indices"] == []
    assert manifest["model_identity"] == "validated-checkpoint-identity"
    assert manifest["worker_exit_status"] == 0


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_peft_eval_loads_processor_from_adapter_and_model_from_stage1_base(
    module_name, monkeypatch, tmp_path
):
    """Loading the adapter directory as a full model must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    artifact_dir, fresh_base = _saved_adapter_artifact(tmp_path)
    base_dir = tmp_path / "stage1"
    _save_exact_base_source(fresh_base, base_dir)
    processor = FakeProcessor()
    processor_sources = []
    model_sources = []
    validations = []

    def load_processor(source, **_kwargs):
        processor_sources.append(source)
        return processor

    def load_model(source, **_kwargs):
        model_sources.append(source)
        return fresh_base

    monkeypatch.setattr(module.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        load_model,
    )
    if module_name == "eval":
        monkeypatch.setattr(
            module,
            "validate_qwen2_5_vl_attention_backends",
            lambda selected, **kwargs: validations.append((selected, kwargs)),
        )

    model, loaded_processor = module.load_evaluation_artifacts(
        model_dir=str(artifact_dir),
        base_model_dir=str(base_dir),
        cache_dir="cache",
    )

    assert isinstance(model, PeftModel)
    assert loaded_processor is processor
    assert processor_sources == [str(artifact_dir)]
    assert model_sources == [str(base_dir)]
    assert resolve_stage2_qwen_model(model).config.stage == "stage2"
    if module_name == "eval":
        assert validations[-1] == (
            resolve_stage2_qwen_model(model),
            {
                "attention_implementation": "sdpa",
                "vision_attention_implementation": "flash_attention_2",
            },
        )


def test_eval_load_model_requires_fixed_mixed_backend_and_validates_runtime(
    monkeypatch,
):
    """Catches evaluation flattening both towers onto one attention backend."""
    module = _import_entry_point("eval", monkeypatch)
    model = tiny_stage2_qwen()
    loads = []
    validations = []

    def load_mixed(source, **kwargs):
        loads.append((source, kwargs))
        return model

    monkeypatch.setattr(module, "load_qwen2_5_vl_with_attention", load_mixed)
    monkeypatch.setattr(
        module,
        "validate_qwen2_5_vl_attention_backends",
        lambda selected, **kwargs: validations.append((selected, kwargs)),
    )

    assert module.load_model("model", "cache") is model
    assert loads[0][0] == "model"
    assert loads[0][1]["attention_implementation"] == "sdpa"
    assert loads[0][1]["vision_attention_implementation"] == "flash_attention_2"
    assert loads[0][1]["config_load_kwargs"] == {"cache_dir": "cache"}
    assert loads[0][1]["model_load_kwargs"] == {
        "cache_dir": "cache",
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    assert "attn_implementation" not in loads[0][1]["model_load_kwargs"]
    assert validations == [
        (
            model,
            {
                "attention_implementation": "sdpa",
                "vision_attention_implementation": "flash_attention_2",
            },
        )
    ]


@pytest.mark.parametrize(
    "failure",
    [
        ImportError("cannot import flash_attn CUDA extension"),
        ValueError("FlashAttention2 class resolution failed"),
    ],
    ids=("initialization", "class-resolution"),
)
def test_eval_mixed_backend_failure_propagates_without_eager_retry(
    monkeypatch, failure
):
    """Catches evaluation retrying an unauthenticated eager model after mixed failure."""
    module = _import_entry_point("eval", monkeypatch)
    calls = []

    def fail_mixed(source, **kwargs):
        calls.append(
            (
                source,
                kwargs["attention_implementation"],
                kwargs["vision_attention_implementation"],
            )
        )
        raise failure

    monkeypatch.setattr(module, "load_qwen2_5_vl_with_attention", fail_mixed)

    with pytest.raises(type(failure), match=str(failure)) as raised:
        module.load_model("model", "cache")

    assert raised.value is failure
    assert calls == [("model", "sdpa", "flash_attention_2")]


def test_stage1_eval_loads_validated_stage1_dora_adapter(monkeypatch, tmp_path):
    """Stage 1 PEFT evaluation uses its validated loader and original base."""
    module = _import_entry_point("eval", monkeypatch)
    artifact_dir = tmp_path / "stage1-dora"
    processor, _metadata = write_stage1_dora_artifact(artifact_dir)
    base_model = tiny_tag_free_original_qwen()
    processor_sources = []
    model_sources = []

    monkeypatch.setattr(
        module.AutoProcessor,
        "from_pretrained",
        lambda source, **_kwargs: processor_sources.append(source) or processor,
    )
    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda source, **_kwargs: model_sources.append(source) or base_model,
    )

    model, loaded_processor = module.load_evaluation_artifacts(
        model_dir=str(artifact_dir),
        base_model_dir="tiny/original-qwen",
        cache_dir="cache",
    )

    assert isinstance(model, PeftModel)
    assert loaded_processor is processor
    assert processor_sources == [str(artifact_dir)]
    assert model_sources == ["tiny/original-qwen"]


def test_stage1_eval_uses_continuous_latent_controller_without_model_generate(
    monkeypatch,
):
    module = _import_entry_point("eval", monkeypatch)
    model = RecordingStage1Model()
    processor = FakeProcessor()
    controller = RecordingController()
    monkeypatch.setattr(
        module,
        "Stage2InferenceController",
        lambda: (_ for _ in ()).throw(
            AssertionError("Stage 2 controller was constructed for Stage 1")
        ),
    )
    monkeypatch.setattr(
        module, "ContinuousLatentController", lambda: controller, raising=False
    )

    result = module.run_one_example(
        model,
        processor,
        {"text_input": "question"},
        {
            "max_new_tokens": 5,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        },
    )

    assert len(controller.calls) == 1
    call = controller.calls[0]
    assert call["model"] is model
    assert torch.equal(call["input_ids"], processor.input_ids)
    assert torch.equal(call["attention_mask"], processor.attention_mask)
    assert call["generation_config"].max_new_tokens == 5
    assert call["return_latent_states"] is False
    assert result["extracted_final_answer"] == "42"
    assert result["generated_suffix_token_ids"] == [21, 22]
    assert result["latent_start_count"] == 1
    assert result["latent_block_count"] == 1
    assert result["stop_reason"] == "eos"


def test_prompt_preparation_excludes_gold_from_tensors_and_identity(tmp_path):
    """Changing or removing gold must not alter the model's prompt lineage."""
    from evaluation.runtime import prepare_evaluation_input

    image = tmp_path / "prompt.png"
    Image.new("RGB", (2, 2), color="white").save(image)
    processor = FakeProcessor()
    images = [Image.open(image).convert("RGB")]
    base = {
        "text_input": "question",
        "image_input": [str(image)],
        "original_final_answer": "42",
    }
    mutated = dict(base, original_final_answer="SECRET-GOLD")
    removed = {key: value for key, value in base.items() if key != "original_final_answer"}

    prepared = [
        prepare_evaluation_input(processor, sample, images)
        for sample in (base, mutated, removed)
    ]

    assert all(
        torch.equal(item.tensors["input_ids"], prepared[0].tensors["input_ids"])
        for item in prepared
    )
    assert all(
        torch.equal(
            item.tensors["pixel_values"], prepared[0].tensors["pixel_values"]
        )
        for item in prepared
    )
    assert {item.prompt_identity for item in prepared} == {
        prepared[0].prompt_identity
    }


def test_evaluation_runtime_import_does_not_load_stage1_training_dependencies():
    """A patched Python version string must not fail before evaluator startup."""
    repo = Path(__file__).resolve().parents[2]
    program = (
        "import sys\n"
        "import evaluation.runtime\n"
        "assert 'datasets' not in sys.modules\n"
        "assert 'stage1.data' not in sys.modules\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": f"{repo}:{repo / 'src'}:{repo / 'transformers/src'}",
        },
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("module_name", MODULE_NAMES)
@pytest.mark.parametrize("payload_mutation", ["delete", "replace"])
def test_sparse_eval_reuses_one_route_inspection_after_payload_mutation(
    module_name,
    payload_mutation,
    monkeypatch,
    tmp_path,
    sparse_eval_artifact,
):
    """Route inspection must own all bytes used after artifact mutation."""

    module = _import_entry_point(module_name, monkeypatch)
    artifact_dir = tmp_path / f"{module_name}-{payload_mutation}"
    shutil.copytree(sparse_eval_artifact.stage2_dir, artifact_dir)
    payload_paths = {
        artifact_dir / name
        for name in (
            "stage2_peft_metadata.json",
            "adapter_config.json",
            "adapter_model.safetensors",
            "ilvr_token_rows.safetensors",
        )
    }
    reads = {path: 0 for path in payload_paths}
    original_read_bytes = Path.read_bytes

    def counted_read_bytes(path):
        if path in reads:
            reads[path] += 1
        return original_read_bytes(path)

    def load_processor(source, **_kwargs):
        assert source == str(artifact_dir)
        assert reads == {path: 1 for path in payload_paths}
        for path in payload_paths:
            if payload_mutation == "delete":
                path.unlink()
            else:
                path.write_bytes(b"replaced after route inspection")
        return sparse_eval_artifact.processor

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    monkeypatch.setattr(module.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda source, **_kwargs: tiny_tag_free_original_qwen(),
    )

    restored, processor = module.load_evaluation_artifacts(
        model_dir=str(artifact_dir),
        base_model_dir=str(sparse_eval_artifact.base_dir),
        cache_dir="cache",
    )

    assert reads == {path: 1 for path in payload_paths}
    assert processor is sparse_eval_artifact.processor
    expected_state = {
        name: value.detach().cpu()
        for name, value in sparse_eval_artifact.source.state_dict().items()
        if "lora_" in name or "trainable_tokens_delta" in name
    }
    actual_state = {
        name: value.detach().cpu()
        for name, value in restored.state_dict().items()
        if "lora_" in name or "trainable_tokens_delta" in name
    }
    assert actual_state.keys() == expected_state.keys()
    for name, expected in expected_state.items():
        torch.testing.assert_close(actual_state[name], expected, rtol=0, atol=0)
    fixed_hidden = torch.linspace(-0.4, 0.9, 16).reshape(2, 8)
    torch.testing.assert_close(
        resolve_stage2_qwen_model(restored).lm_head(fixed_hidden),
        resolve_stage2_qwen_model(sparse_eval_artifact.source).lm_head(
            fixed_hidden
        ),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_sparse_eval_rejects_noncanonical_adapter_key_before_large_load(
    module_name,
    monkeypatch,
    tmp_path,
    sparse_eval_artifact,
):
    """A prefix-renamed saved key must fail before the eval model load."""

    module = _import_entry_point(module_name, monkeypatch)
    artifact_dir = tmp_path / f"{module_name}-malformed"
    shutil.copytree(sparse_eval_artifact.stage2_dir, artifact_dir)
    adapter_path = artifact_dir / "adapter_model.safetensors"
    state = load_file(adapter_path)
    key = next(name for name in state if "lora_A.weight" in name)
    state["renamed." + key.removeprefix("base_model.model.")] = state.pop(key)
    save_file(state, adapter_path)
    model_loads = []
    monkeypatch.setattr(
        module.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: sparse_eval_artifact.processor,
    )
    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda *args, **kwargs: model_loads.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="canonical adapter tensor fields"):
        module.load_evaluation_artifacts(
            model_dir=str(artifact_dir),
            base_model_dir=str(sparse_eval_artifact.base_dir),
            cache_dir="cache",
        )

    assert model_loads == []


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_full_eval_uses_model_dir_only_and_rejects_stray_base(
    module_name, monkeypatch, tmp_path
):
    """Treating a full artifact as base-plus-adapter must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    (full_dir / "model.safetensors").touch()
    full_model = tiny_stage2_qwen()
    processor = FakeProcessor()
    model_sources = []
    monkeypatch.setattr(
        module.AutoProcessor,
        "from_pretrained",
        lambda source, **_kwargs: processor,
    )
    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda source, **_kwargs: model_sources.append(source) or full_model,
    )

    loaded_model, loaded_processor = module.load_evaluation_artifacts(
        model_dir=str(full_dir), base_model_dir=None, cache_dir="cache"
    )

    assert loaded_model is full_model
    assert loaded_processor is processor
    assert model_sources == [str(full_dir)]
    with pytest.raises(ValueError, match="base_model_dir.*full"):
        module.load_evaluation_artifacts(
            model_dir=str(full_dir),
            base_model_dir="stray-stage1",
            cache_dir="cache",
        )
    assert model_sources == [str(full_dir)]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
@pytest.mark.parametrize(
    "markers",
    [
        ("stage2_peft_metadata.json",),
        ("adapter_config.json",),
        ("adapter_model.safetensors",),
        ("adapter_model.bin",),
    ],
    ids=(
        "sidecar-only",
        "config-only",
        "safe-weights-only",
        "unsupported-bin-weights",
    ),
)
def test_eval_rejects_incomplete_peft_artifact_before_loading(
    module_name, markers, monkeypatch, tmp_path
):
    module = _import_entry_point(module_name, monkeypatch)
    artifact_dir = tmp_path / "incomplete"
    artifact_dir.mkdir()
    for marker in markers:
        (artifact_dir / marker).touch()
    monkeypatch.setattr(
        module.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("processor load must not start"),
    )
    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("model load must not start"),
    )

    with pytest.raises(ValueError, match="incomplete Stage 2 PEFT artifact"):
        module.load_evaluation_artifacts(
            model_dir=str(artifact_dir),
            base_model_dir="stage1",
            cache_dir="cache",
        )


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_eval_rejects_saved_adapter_after_sidecar_is_removed(
    module_name, monkeypatch, tmp_path
):
    module = _import_entry_point(module_name, monkeypatch)
    artifact_dir, _fresh_base = _saved_adapter_artifact(tmp_path)
    (artifact_dir / "stage2_peft_metadata.json").unlink()
    monkeypatch.setattr(
        module.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("processor load must not start"),
    )
    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("model load must not start"),
    )

    with pytest.raises(ValueError, match="incomplete Stage 2 PEFT artifact"):
        module.load_evaluation_artifacts(
            model_dir=str(artifact_dir),
            base_model_dir="stage1",
            cache_dir="cache",
        )


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_eval_rejects_mixed_full_and_peft_artifact_before_loading(
    module_name, monkeypatch, tmp_path
):
    module = _import_entry_point(module_name, monkeypatch)
    artifact_dir = tmp_path / "mixed"
    artifact_dir.mkdir()
    for marker in (
        "stage2_peft_metadata.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "model.safetensors",
    ):
        (artifact_dir / marker).touch()
    monkeypatch.setattr(
        module.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("processor load must not start"),
    )
    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("model load must not start"),
    )

    with pytest.raises(ValueError, match="mixed Stage 2 full and PEFT artifact"):
        module.load_evaluation_artifacts(
            model_dir=str(artifact_dir),
            base_model_dir="stage1",
            cache_dir="cache",
        )


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_peft_eval_requires_base_before_loading_or_iterating(
    module_name, monkeypatch, tmp_path
):
    """A missing Stage 1 base must fail before any model or processor load."""
    module = _import_entry_point(module_name, monkeypatch)
    artifact_dir, _fresh_base = _saved_adapter_artifact(tmp_path)
    monkeypatch.setattr(
        module.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("processor load must not start"),
    )
    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("model load must not start"),
    )

    with pytest.raises(ValueError, match="base_model_dir.*required"):
        module.load_evaluation_artifacts(
            model_dir=str(artifact_dir), base_model_dir=None, cache_dir="cache"
        )


@pytest.mark.parametrize("module_name", ("evaluate_deepseed",))
@pytest.mark.parametrize(
    "error_type,error_message",
    [
        (
            FileNotFoundError,
            "checkpoint /checkpoints/flashattention-experiment/missing does not exist",
        ),
        (ValueError, "unknown model ID flash_attention_2-experiment"),
        (ValueError, "FlashAttention metadata mismatch"),
        (ValueError, "Flash Attention latent token mismatch"),
        (ValueError, "flash-attention adapter mismatch"),
        (ValueError, "malformed config: attn_implementation=flash_attention_2"),
        (ValueError, "FlashAttention adapter dtype metadata mismatch"),
        (
            ValueError,
            "Flash Attention adapter is not compatible with the base model",
        ),
        (ValueError, "adapter config package flash_attn version mismatch"),
        (
            ValueError,
            "Flash Attention 2 adapter only supports torch.float16 metadata",
        ),
        (ValueError, "Flash Attention 2 adapter backend initialization failed"),
        (ImportError, "cannot import flash_attn CUDA extension"),
        (RuntimeError, "Flash Attention 2 is unavailable"),
    ],
    ids=(
        "missing-path",
        "model-id",
        "metadata",
        "token",
        "adapter",
        "malformed-config",
        "adapter-dtype-metadata",
        "adapter-base-incompatible",
        "package-version-metadata",
        "adapter-dtype-support",
        "invented-backend-initialization",
        "flash-attn-import-text",
        "synthetic-unavailable",
    ),
)
def test_non_backend_model_error_is_not_hidden_by_flash_fallback(
    module_name, error_type, error_message, monkeypatch
):
    """Retrying metadata/config errors as eager attention must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    calls = []
    original_error = error_type(error_message)

    def fail_load(_source, **kwargs):
        calls.append(kwargs["attn_implementation"])
        raise original_error

    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        fail_load,
    )

    with pytest.raises(error_type, match=error_message) as raised:
        module.load_model("model", "cache")
    assert raised.value is original_error
    assert calls == ["flash_attention_2"]


@pytest.mark.parametrize("module_name", ("evaluate_deepseed",))
@pytest.mark.parametrize("nested", [False, True], ids=("direct", "nested"))
@pytest.mark.parametrize(
    "error_message",
    [
        "No module named 'flash_attn_extra'",
        "cannot import my_flash_attn extension",
    ],
    ids=("suffix", "prefix"),
)
def test_near_flash_attn_module_name_does_not_retry_eager(
    module_name, nested, error_message, monkeypatch
):
    """Substring package-name matching must fail this actual-loader test."""
    module = _import_entry_point(module_name, monkeypatch)
    calls = []
    near_name_error = ImportError(error_message)
    if nested:
        original_error = RuntimeError("attention model load failed")
        original_error.__cause__ = near_name_error
    else:
        original_error = near_name_error

    def fail_load(_source, **kwargs):
        calls.append(kwargs["attn_implementation"])
        raise original_error

    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        fail_load,
    )

    with pytest.raises(type(original_error)) as raised:
        module.load_model("model", "cache")
    assert raised.value is original_error
    assert calls == ["flash_attention_2"]


@pytest.mark.parametrize("module_name", ("evaluate_deepseed",))
def test_nested_non_backend_error_does_not_combine_classifier_signals(
    module_name, monkeypatch
):
    """Combining classifier signals across exceptions must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    calls = []
    original_error = RuntimeError("FlashAttention adapter load failed")
    original_error.__cause__ = ValueError("adapter dtype config mismatch")

    def fail_load(_source, **kwargs):
        calls.append(kwargs["attn_implementation"])
        raise original_error

    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        fail_load,
    )

    with pytest.raises(RuntimeError, match="adapter load failed") as raised:
        module.load_model("model", "cache")
    assert raised.value is original_error
    assert calls == ["flash_attention_2"]


@pytest.mark.parametrize("module_name", ("evaluate_deepseed",))
@pytest.mark.parametrize("nesting", ["direct", "cause", "context"])
@pytest.mark.parametrize(
    "failure", ["unsupported-model", "missing-package"]
)
def test_real_transformers_flash_attention_error_retries_eager(
    module_name, nesting, failure, monkeypatch
):
    """Ignoring the real Transformers traceback provenance must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    model = tiny_stage2_qwen()
    calls = []
    backend_error = _capture_transformers_flash_attention_error(
        module.Qwen2_5_VLForConditionalGeneration, monkeypatch, failure
    )
    if nesting == "cause":
        original_error = RuntimeError("model initialization failed")
        original_error.__cause__ = backend_error
    elif nesting == "context":
        original_error = RuntimeError("model initialization failed")
        original_error.__context__ = backend_error
    else:
        original_error = backend_error

    def load(_source, **kwargs):
        calls.append(kwargs["attn_implementation"])
        if kwargs["attn_implementation"] == "flash_attention_2":
            raise original_error
        return model

    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        load,
    )

    assert module.load_model("model", "cache") is model
    assert calls == ["flash_attention_2", "eager"]


@pytest.mark.parametrize("module_name", ("evaluate_deepseed",))
@pytest.mark.parametrize("nesting", ["direct", "cause", "context"])
@pytest.mark.parametrize(
    "failure", ["unsupported-model", "missing-package"]
)
def test_identical_flash_attention_text_without_transformers_provenance_does_not_retry(
    module_name, nesting, failure, monkeypatch
):
    """Restoring message-based classification must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    calls = []
    backend_error = _capture_transformers_flash_attention_error(
        module.Qwen2_5_VLForConditionalGeneration, monkeypatch, failure
    )
    fabricated_error = type(backend_error)(str(backend_error))
    if nesting == "cause":
        original_error = RuntimeError("model initialization failed")
        original_error.__cause__ = fabricated_error
    elif nesting == "context":
        original_error = RuntimeError("model initialization failed")
        original_error.__context__ = fabricated_error
    else:
        original_error = fabricated_error

    def fail_load(_source, **kwargs):
        calls.append(kwargs["attn_implementation"])
        raise original_error

    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        fail_load,
    )

    with pytest.raises(type(original_error)) as raised:
        module.load_model("model", "cache")
    assert raised.value is original_error
    assert calls == ["flash_attention_2"]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_peft_eval_token_mismatch_fails_during_artifact_loading(
    module_name, monkeypatch, tmp_path
):
    """Deferring token validation until sample iteration must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    artifact_dir, fresh_base = _saved_adapter_artifact(tmp_path)
    base_dir = tmp_path / "stage1"
    _save_exact_base_source(fresh_base, base_dir)
    processor = FakeProcessor()

    class MismatchedTokenizer(FakeTokenizer):
        def convert_tokens_to_ids(self, token):
            if token == "<|latent_start|>":
                return 999
            return super().convert_tokens_to_ids(token)

    processor.tokenizer = MismatchedTokenizer()
    model_calls = []
    monkeypatch.setattr(
        module.AutoProcessor,
        "from_pretrained",
        lambda _source, **_kwargs: processor,
    )
    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda source, **_kwargs: model_calls.append(source) or fresh_base,
    )

    with pytest.raises(ValueError, match="latent_start_token_id mismatch"):
        module.load_evaluation_artifacts(
            model_dir=str(artifact_dir),
            base_model_dir=str(base_dir),
            cache_dir="cache",
        )
    assert model_calls == []


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_eval_uses_controller_with_original_visual_inputs_and_preserves_decoding(
    module_name,
    monkeypatch,
    tmp_path,
):
    """Removing the controller call or forwarding the wrong model inputs must fail."""
    module = _import_entry_point(module_name, monkeypatch)
    controller = RecordingController()
    monkeypatch.setattr(
        module, "Stage2InferenceController", lambda: controller, raising=False
    )
    original_image = tmp_path / "original.png"
    Image.new("RGB", (2, 2), color="white").save(original_image)
    processor = FakeProcessor()
    model = ModelWhoseGenerateRaises()

    result = module.run_one_example(
        model,
        processor,
        {
            "text_input": "question",
            "image_input": str(original_image),
            "helper_images": ["must-not-be-opened.png"],
        },
        {
            "max_new_tokens": 9,
            "temperature": 0.7,
            "top_p": 0.75,
            "do_sample": True,
            "eos_token_id": 8,
            "max_sequence_length": 64,
            "repetition_penalty": 1.05,
        },
    )

    assert len(controller.calls) == 1
    call = controller.calls[0]
    assert call["model"] is model
    assert torch.equal(call["input_ids"], processor.input_ids)
    assert torch.equal(call["attention_mask"], processor.attention_mask)
    assert torch.equal(call["pixel_values"], processor.pixel_values)
    assert torch.equal(call["image_grid_thw"], processor.image_grid_thw)
    assert call["token_ids"].pad_token_id == 90
    assert call["token_ids"].start_token_id == 91
    assert call["token_ids"].end_token_id == 92
    assert call["generation_config"].max_new_tokens == 9
    assert call["generation_config"].temperature == 0.7
    assert call["generation_config"].top_p == 0.75
    assert call["generation_config"].eos_token_id == 8
    assert call["generation_config"].max_sequence_length == 64
    assert call["generation_config"].repetition_penalty == 1.05
    assert call["return_latent_states"] is False
    assert len(processor.images) == 1
    assert processor.images[0].mode == "RGB"
    assert len(processor.decoded_id_batches) == 2
    assert torch.equal(processor.decoded_id_batches[0], controller.sequences)
    assert torch.equal(
        processor.decoded_id_batches[1], controller.sequences[:, 2:]
    )
    assert result["assistant_only"] == "Final answer: 42"
    assert result["extracted_final_answer"] == "42"
    expected_result_fields = {
        "prompt_identity",
        "prompt_token_length",
        "generated_suffix_token_ids",
        "raw_output",
        "assistant_only",
        "extracted_final_answer",
        "latent_start_count",
        "latent_block_count",
        "stop_reason",
    }
    if module_name == "eval":
        expected_result_fields.add("inference_time")
    else:
        expected_result_fields.update({"vision_ablation", "input_image_count"})
    assert set(result) == expected_result_fields


@pytest.mark.parametrize(
    ("vision_ablation", "expected_image_count"),
    (("all", 0), ("interleaved", 1)),
)
def test_stage2_vision_ablation_controls_processor_images_and_disables_latents(
    vision_ablation, expected_image_count, monkeypatch, tmp_path
):
    """Ablation must change only the approved image boundary and latent control."""
    module = _import_entry_point("evaluate_deepseed", monkeypatch)
    controller = RecordingController()
    controller.generate = lambda **kwargs: (
        controller.calls.append(kwargs)
        or SimpleNamespace(
            sequences=controller.sequences,
            generated_suffix_ids=(21, 22),
            latent_start_count=0,
            latent_block_count=0,
            stop_reason="eos",
        )
    )
    monkeypatch.setattr(module, "Stage2InferenceController", lambda: controller)
    original_image = tmp_path / "original.png"
    Image.new("RGB", (2, 2), color="white").save(original_image)
    processor = FakeProcessor()

    result = module.run_one_example(
        ModelWhoseGenerateRaises(),
        processor,
        {"text_input": "question", "image_input": str(original_image)},
        {
            "max_new_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "vision_ablation": vision_ablation,
        },
    )

    call = controller.calls[0]
    assert call["disable_latent_generation"] is True
    assert result["vision_ablation"] == vision_ablation
    assert result["input_image_count"] == expected_image_count
    if vision_ablation == "all":
        assert processor.images is None
        assert call["pixel_values"] is None
        assert call["image_grid_thw"] is None
    else:
        assert len(processor.images) == 1
        assert torch.equal(call["pixel_values"], processor.pixel_values)
        assert torch.equal(call["image_grid_thw"], processor.image_grid_thw)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_eval_maps_explicit_greedy_settings_to_controller(module_name, monkeypatch):
    """Accidentally sampling when do_sample=False must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    controller = RecordingController()
    monkeypatch.setattr(
        module, "Stage2InferenceController", lambda: controller, raising=False
    )

    module.run_one_example(
        ModelWhoseGenerateRaises(),
        FakeProcessor(),
        {"text_input": "question"},
        {
            "max_new_tokens": 5,
            "temperature": 0.8,
            "top_p": 0.6,
            "do_sample": False,
        },
    )

    config = controller.calls[0]["generation_config"]
    assert config.max_new_tokens == 5
    assert config.temperature == 0.0
    assert config.top_p == 0.6
    assert config.eos_token_id == 3
    assert config.max_sequence_length == 128


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_eval_forwards_external_generation_limit_policy(module_name, monkeypatch):
    """Dropping the evaluator option must leave the controller on bounded semantics."""
    module = _import_entry_point(module_name, monkeypatch)
    controller = RecordingController()
    monkeypatch.setattr(
        module, "Stage2InferenceController", lambda: controller, raising=False
    )

    module.run_one_example(
        ModelWhoseGenerateRaises(),
        FakeProcessor(),
        {"text_input": "question"},
        {
            "max_new_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "generation_limit_policy": "external_full_latent_block",
        },
    )

    assert (
        controller.calls[0]["generation_config"].generation_limit_policy
        == "external_full_latent_block"
    )


def test_deepspeed_eval_unwraps_model_for_controller_and_uses_wrapper_device(
    monkeypatch,
):
    """Stopping at DeepSpeed's PeftModel wrapper must fail this test."""
    module = _import_entry_point("evaluate_deepseed", monkeypatch)
    controller = RecordingController()
    monkeypatch.setattr(module, "Stage2InferenceController", lambda: controller)
    tuning = Stage2TuningConfig(
        method="lora",
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
    )
    adapter_model = build_stage2_tuned_model(tiny_stage2_qwen(), tuning).eval()
    activate_stage2_adapter_deltas(adapter_model)
    wrapper = DeepSpeedShapedWrapper(adapter_model)

    module.run_one_example(
        wrapper,
        FakeProcessor(),
        {"text_input": "question"},
        {"max_new_tokens": 1, "temperature": 0.0, "top_p": 1.0},
    )

    resolved = wrapper.module.get_base_model()
    assert controller.calls[0]["model"] is resolved
    assert any(hasattr(module, "lora_A") for module in resolved.modules())
    assert controller.calls[0]["input_ids"].device == torch.device("cpu")


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_eval_resolves_direct_peft_wrapper_for_controller(module_name, monkeypatch):
    """Passing a direct PeftModel to the controller must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    controller = RecordingController()
    monkeypatch.setattr(module, "Stage2InferenceController", lambda: controller)
    tuning = Stage2TuningConfig(
        method="lora",
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
    )
    adapter_model = build_stage2_tuned_model(tiny_stage2_qwen(), tuning).eval()
    activate_stage2_adapter_deltas(adapter_model)

    module.run_one_example(
        adapter_model,
        FakeProcessor(),
        {"text_input": "question"},
        {"max_new_tokens": 1, "temperature": 0.0, "top_p": 1.0},
    )

    resolved = adapter_model.get_base_model()
    assert controller.calls[0]["model"] is resolved
    assert any(hasattr(child, "lora_A") for child in resolved.modules())


@pytest.mark.parametrize("module_name", MODULE_NAMES)
@pytest.mark.parametrize("source", ["explicit", "model_fallback"])
def test_eval_preserves_multiple_eos_ids(module_name, source, monkeypatch):
    """Narrowing an HF EOS list to scalar or leaving it mutable must fail."""
    module = _import_entry_point(module_name, monkeypatch)
    controller = RecordingController()
    monkeypatch.setattr(module, "Stage2InferenceController", lambda: controller)
    model = ModelWhoseGenerateRaises()
    kwargs = {"max_new_tokens": 2, "temperature": 0.0, "top_p": 1.0}
    if source == "explicit":
        kwargs["eos_token_id"] = [8, 9]
        expected = (8, 9)
    else:
        model.generation_config.eos_token_id = [3, 4]
        expected = (3, 4)

    module.run_one_example(
        model,
        FakeProcessor(),
        {"text_input": "question"},
        kwargs,
    )

    assert controller.calls[0]["generation_config"].eos_token_id == expected


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_eval_derives_model_sequence_cap_for_real_controller(module_name, monkeypatch):
    """The model-derived cap must truncate the latent scaffold without overflow."""
    module = _import_entry_point(module_name, monkeypatch)

    result = module.run_one_example(
        RealControllerCapModel(),
        FakeProcessor(include_visual_inputs=False),
        {"text_input": "question"},
        {"max_new_tokens": 16, "temperature": 0.0, "top_p": 1.0},
    )

    assert result["generated_suffix_token_ids"] == [91, 90, 90]
    assert result["latent_block_count"] == 0
    assert result["stop_reason"] == "max_sequence_length"


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_text_only_eval_defaults_missing_attention_mask(module_name, monkeypatch):
    """Indexing optional processor fields or forwarding visual placeholders must fail."""
    module = _import_entry_point(module_name, monkeypatch)
    controller = RecordingController()
    monkeypatch.setattr(module, "Stage2InferenceController", lambda: controller)
    processor = FakeProcessor(
        include_attention_mask=False,
        include_visual_inputs=False,
    )

    module.run_one_example(
        ModelWhoseGenerateRaises(),
        processor,
        {"text_input": "question"},
        {"max_new_tokens": 1, "temperature": 0.0, "top_p": 1.0},
    )

    call = controller.calls[0]
    assert torch.equal(call["attention_mask"], torch.ones_like(processor.input_ids))
    assert call["pixel_values"] is None
    assert call["image_grid_thw"] is None


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_eval_rejects_mismatched_checkpoint_token_metadata(module_name, monkeypatch):
    """Replacing validated metadata with direct token probing must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)

    with pytest.raises(ValueError, match="latent_start_token_id mismatch"):
        module.run_one_example(
            ModelWhoseGenerateRaises(start_token_id=999),
            FakeProcessor(),
            {"text_input": "question"},
            {"max_new_tokens": 1, "temperature": 0.0, "top_p": 1.0},
        )


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_eval_rejects_missing_latent_size_metadata(module_name, monkeypatch):
    """Restoring the legacy latent-size fallback must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    controller = RecordingController()
    monkeypatch.setattr(
        module, "Stage2InferenceController", lambda: controller, raising=False
    )
    model = ModelWhoseGenerateRaises()
    del model.config.latent_size

    with pytest.raises(ValueError, match="latent_size"):
        module.run_one_example(
            model,
            FakeProcessor(),
            {"text_input": "question"},
            {"max_new_tokens": 1, "temperature": 0.0, "top_p": 1.0},
        )

    assert controller.calls == []


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_eval_rejects_processor_batches_larger_than_one(module_name, monkeypatch):
    """Removing the local B=1 guard must fail before controller invocation."""
    module = _import_entry_point(module_name, monkeypatch)
    controller = RecordingController()
    monkeypatch.setattr(
        module, "Stage2InferenceController", lambda: controller, raising=False
    )
    processor = FakeProcessor()
    processor.input_ids = torch.tensor([[10, 11], [12, 13]])
    processor.attention_mask = torch.ones_like(processor.input_ids)

    with pytest.raises(NotImplementedError, match="batch_size=1"):
        module.run_one_example(
            ModelWhoseGenerateRaises(),
            processor,
            {"text_input": "question"},
            {"max_new_tokens": 1, "temperature": 0.0, "top_p": 1.0},
        )

    assert controller.calls == []


def test_evaluator_precision_fixture_restores_backend_flags():
    """Removing fixture teardown must expose evaluator TF32 state to later tests."""
    initial = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    isolation = restore_evaluator_backend_precision.__wrapped__()
    next(isolation)
    try:
        torch.backends.cuda.matmul.allow_tf32 = not initial[0]
        torch.backends.cudnn.allow_tf32 = not initial[1]
    finally:
        isolation.close()
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == initial
