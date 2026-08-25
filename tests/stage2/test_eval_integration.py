import copy
import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
from peft import PeftModel
from PIL import Image
from torch import nn

from helpers import (
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


MODULE_NAMES = ("eval", "evaluate_deepseed")
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


class RecordingController:
    def __init__(self):
        self.calls = []
        self.sequences = torch.tensor([[10, 11, 21, 22]])

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(sequences=self.sequences)


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
    return module


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

    assert module.get_eval_args().base_model_dir == "stage1"


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_peft_eval_loads_processor_from_adapter_and_model_from_stage1_base(
    module_name, monkeypatch, tmp_path
):
    """Loading the adapter directory as a full model must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    artifact_dir, fresh_base = _saved_adapter_artifact(tmp_path)
    base_dir = tmp_path / "stage1"
    processor = FakeProcessor()
    processor_sources = []
    model_sources = []

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


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_full_eval_uses_model_dir_only_and_rejects_stray_base(
    module_name, monkeypatch, tmp_path
):
    """Treating a full artifact as base-plus-adapter must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    full_dir = tmp_path / "full"
    full_dir.mkdir()
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


@pytest.mark.parametrize("module_name", MODULE_NAMES)
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


@pytest.mark.parametrize("module_name", MODULE_NAMES)
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


@pytest.mark.parametrize("module_name", MODULE_NAMES)
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


@pytest.mark.parametrize("module_name", MODULE_NAMES)
@pytest.mark.parametrize("nested", [False, True], ids=("direct", "nested"))
def test_flash_attn_module_import_error_retries_with_eager(
    module_name, nested, monkeypatch
):
    """Missing the real flash_attn package spelling must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    model = tiny_stage2_qwen()
    calls = []
    package_error = ModuleNotFoundError("No module named 'flash_attn'")
    if nested:
        original_error = RuntimeError("attention backend initialization failed")
        original_error.__cause__ = package_error
    else:
        original_error = package_error

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


@pytest.mark.parametrize("module_name", MODULE_NAMES)
@pytest.mark.parametrize(
    "error_message",
    [
        (
            "FlashAttention2 has been toggled on, but it cannot be used due to "
            "the following error: Flash Attention 2 is not available on CPU. "
            "Please make sure torch can access a CUDA device."
        ),
        (
            "Qwen2_5_VLForConditionalGeneration does not support Flash Attention "
            "2.0 yet. Please request to add support where the model is hosted, on "
            "its model hub page: https://huggingface.co/tiny/discussions/new or in "
            "the Transformers GitHub repo: "
            "https://github.com/huggingface/transformers/issues/new"
        ),
        (
            "Flash Attention 2 and BetterTransformer API are not compatible. "
            "Please make sure to disable BetterTransformers by doing "
            "model.reverse_bettertransformer()"
        ),
        (
            "You are attempting to use Flash Attention 2.0 with a model not "
            "initialized on GPU and with no GPU available. This is not supported "
            "yet. Please make sure to have access to a GPU and either initialise "
            "the model on a GPU by passing a device_map or initialising the model "
            "on CPU and then moving it to GPU."
        ),
        (
            "You are attempting to use Flash Attention 2.0 with a model dispatched "
            "on CPU or disk. This is not supported. Please make sure to initialise "
            "the model on a GPU by passing a device_map that contains only GPU "
            "devices as keys."
        ),
    ],
    ids=(
        "toggled-unavailable",
        "unsupported-model",
        "bettertransformer",
        "no-gpu",
        "cpu-disk-dispatch",
    ),
)
def test_local_transformers_flash_attention_errors_retry_eager(
    module_name, error_message, monkeypatch
):
    """Removing any locally raised ValueError family must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    model = tiny_stage2_qwen()
    calls = []
    original_error = ValueError(error_message)

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


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_flash_attn_import_error_retries_with_eager(module_name, monkeypatch):
    """An ImportError naming flash_attn must trigger the backend fallback."""
    module = _import_entry_point(module_name, monkeypatch)
    model = tiny_stage2_qwen()
    calls = []
    original_error = ImportError("cannot import flash_attn CUDA extension")

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


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_flash_backend_initialization_error_retries_with_eager(
    module_name, monkeypatch
):
    """Removing the intended backend-only fallback must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    model = tiny_stage2_qwen()
    calls = []

    def load(_source, **kwargs):
        calls.append(kwargs["attn_implementation"])
        if kwargs["attn_implementation"] == "flash_attention_2":
            raise RuntimeError("Flash Attention 2 is unavailable")
        return model

    monkeypatch.setattr(
        module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        load,
    )

    assert module.load_model("model", "cache") is model
    assert calls == ["flash_attention_2", "eager"]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_peft_eval_token_mismatch_fails_during_artifact_loading(
    module_name, monkeypatch, tmp_path
):
    """Deferring token validation until sample iteration must fail this test."""
    module = _import_entry_point(module_name, monkeypatch)
    artifact_dir, fresh_base = _saved_adapter_artifact(tmp_path)
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
            base_model_dir=str(tmp_path / "stage1"),
            cache_dir="cache",
        )
    assert model_calls == [str(tmp_path / "stage1")]


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
    assert call["return_latent_states"] is False
    assert len(processor.images) == 1
    assert processor.images[0].mode == "RGB"
    assert torch.equal(processor.decoded_ids, controller.sequences)
    assert result["assistant_only"] == "Final answer: 42"
    assert result["extracted_final_answer"] == "42"
    assert set(result) == (
        {"raw_output", "assistant_only", "extracted_final_answer", "inference_time"}
        if module_name == "eval"
        else {"raw_output", "assistant_only", "extracted_final_answer"}
    )


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
    """Dropping the model-derived cap would let this latent block overflow."""
    module = _import_entry_point(module_name, monkeypatch)

    with pytest.raises(ValueError, match="latent block would exceed"):
        module.run_one_example(
            RealControllerCapModel(),
            FakeProcessor(include_visual_inputs=False),
            {"text_input": "question"},
            {"max_new_tokens": 1, "temperature": 0.0, "top_p": 1.0},
        )


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
