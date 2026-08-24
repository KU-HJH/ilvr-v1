import importlib
from types import SimpleNamespace

import pytest
import torch
from PIL import Image


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
    def __init__(self):
        self.tokenizer = FakeTokenizer()
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
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "pixel_values": self.pixel_values,
            "image_grid_thw": self.image_grid_thw,
        }

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
            latent_size=2,
            latent_pad_token_id=90,
            latent_start_token_id=start_token_id,
            latent_end_token_id=92,
        )
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.batch_size = batch_size

    def generate(self, **kwargs):
        raise AssertionError("legacy model.generate was called")


class RecordingController:
    def __init__(self):
        self.calls = []
        self.sequences = torch.tensor([[10, 11, 21, 22]])

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(sequences=self.sequences)


def _import_entry_point(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    return module


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
