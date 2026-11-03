from dataclasses import replace

import pytest
import torch

from evaluation.io import parse_config, read_manifest
from evaluation.inference_bridge import generate_one, runtime_metadata
from evaluation.schema import VspTarget
from tests.stage2 import test_eval_integration as legacy_fixtures
from vsp_fixtures import make_vsp_case


class VspRecordingProcessor(legacy_fixtures.FakeProcessor):
    """Exercise the real prompt path while making prompt/suffix decoding distinct."""

    def __init__(self):
        super().__init__()
        self.messages = []
        self.image_pixels = []
        self.chat_template = "active-processor-template"
        self.tokenizer.chat_template = "active-tokenizer-template"

    def apply_chat_template(self, conversations, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages.append(conversations)
        return "rendered VSP prompt"

    def __call__(self, *, text, images, return_tensors, padding):
        assert text == ["rendered VSP prompt"]
        assert return_tensors == "pt"
        assert padding is True
        self.images = images
        self.image_pixels.append([image.getpixel((0, 0)) for image in images])
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "pixel_values": self.pixel_values,
            "image_grid_thw": self.image_grid_thw,
        }

    def batch_decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        self.decoded_id_batches.append(ids.clone())
        if ids.shape[1] == 4:
            return ["user prompt Action plan: L"]
        if ids.shape[1] == 2 and ids[0].tolist() == [21, 22]:
            return ["<|im_start|>assistant Action plan: R<|im_end|>"]
        raise AssertionError(f"unexpected decoded token IDs: {ids.tolist()}")


@pytest.mark.parametrize(
    ("stage", "model_factory", "controller_name"),
    [
        (
            "stage1",
            legacy_fixtures.RecordingStage1Model,
            "ContinuousLatentController",
        ),
        (
            "stage2",
            legacy_fixtures.ModelWhoseGenerateRaises,
            "Stage2InferenceController",
        ),
    ],
)
def test_bridge_matches_real_legacy_dispatch_with_three_images_and_no_target_leakage(
    stage, model_factory, controller_name, monkeypatch
):
    runtime = legacy_fixtures._import_entry_point("eval", monkeypatch)
    case = make_vsp_case(1)
    config = parse_config(case["config"], case["root"])
    sample = read_manifest(case["manifest"], config, case["root"]).samples[0]
    mutated = replace(
        sample,
        target=VspTarget(((1, 0, 2), (0, 0, 0), (0, 0, 0)), 100),
    )
    assert mutated.model_input is sample.model_input

    controllers = []

    def controller_factory():
        controller = legacy_fixtures.RecordingController()
        controllers.append(controller)
        return controller

    monkeypatch.setattr(runtime, controller_name, controller_factory, raising=False)
    model = model_factory()
    direct_processor = VspRecordingProcessor()
    bridge_processor = VspRecordingProcessor()
    absolute_images = [
        str((config.dataset_root / relative).resolve())
        for relative in sample.model_input.images
    ]

    direct = runtime.run_one_example(
        model,
        direct_processor,
        {"text_input": sample.model_input.text, "image_input": absolute_images},
        config.run_metadata.generation_config,
    )
    bridged = generate_one(
        model,
        bridge_processor,
        mutated.model_input,
        config.dataset_root,
        config.run_metadata.generation_config,
        run_one_example=runtime.run_one_example,
    )

    expected_content = [
        {"type": "image"},
        {"type": "image"},
        {"type": "image"},
        {"type": "text", "text": sample.model_input.text},
    ]
    assert direct_processor.messages == bridge_processor.messages == [
        [{"role": "user", "content": expected_content}]
    ]
    assert direct_processor.image_pixels == bridge_processor.image_pixels == [
        [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    ]
    assert len(controllers) == 2
    for controller in controllers:
        assert len(controller.calls) == 1
        call = controller.calls[0]
        assert call["model"] is model
        assert torch.equal(call["input_ids"], bridge_processor.input_ids)
        assert torch.equal(call["attention_mask"], bridge_processor.attention_mask)
        assert torch.equal(call["pixel_values"], bridge_processor.pixel_values)
        assert torch.equal(call["image_grid_thw"], bridge_processor.image_grid_thw)
        assert call["generation_config"].max_new_tokens == 8
        assert call["generation_config"].max_sequence_length == 32768
        assert call["generation_config"].repetition_penalty == 1.05
        assert call["generation_config"].temperature == 0.0
        assert call["generation_config"].top_p == 1.0
        assert call["return_latent_states"] is False
    assert vars(controllers[0].calls[0]["generation_config"]) == vars(
        controllers[1].calls[0]["generation_config"]
    )
    assert direct["prompt_token_length"] == bridged["prompt_token_length"] == 2
    assert direct["generated_suffix_token_ids"] == bridged[
        "generated_suffix_token_ids"
    ] == [21, 22]
    assert direct["assistant_only"] == bridged["assistant_only"] == "Action plan: R"
    assert direct["raw_output"] == bridged["raw_output"] == "user prompt Action plan: L"
    assert [batch.tolist() for batch in bridge_processor.decoded_id_batches] == [
        [[10, 11, 21, 22]],
        [[21, 22]],
    ]


@pytest.mark.filterwarnings("ignore:Could not find a config file in.*:UserWarning")
def test_runtime_metadata_handles_an_adapter_loaded_by_the_existing_runtime(
    monkeypatch,
):
    runtime = legacy_fixtures._import_entry_point("eval", monkeypatch)
    case = make_vsp_case(1)
    staging_root = case["model_dir"].parent
    adapter_dir, fresh_base = legacy_fixtures._saved_adapter_artifact(staging_root)
    base_dir = staging_root / "actual-base"
    legacy_fixtures._save_exact_base_source(fresh_base, base_dir)
    processor = VspRecordingProcessor()
    monkeypatch.setattr(
        runtime.AutoProcessor,
        "from_pretrained",
        lambda _source, **_kwargs: processor,
    )
    monkeypatch.setattr(
        runtime.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda _source, **_kwargs: fresh_base,
    )
    model, loaded_processor = runtime.load_evaluation_artifacts(
        model_dir=str(adapter_dir),
        base_model_dir=str(base_dir),
        cache_dir=str(case["root"] / "data/cache/evaluation" / case["run_id"]),
    )

    metadata = runtime_metadata(
        model,
        loaded_processor,
        runtime,
        adapter_dir,
        base_dir,
        case["config"]["run_metadata"]["generation_config"],
    )

    assert metadata["model_dir"] == str(adapter_dir.resolve())
    assert metadata["base_model_dir"] == str(base_dir.resolve())
    assert metadata["qwen_config"] == {"stage": "stage2", "latent_size": 2}
    assert metadata["latent_token_ids"] == {"pad": 90, "start": 91, "end": 92}
    assert "adapter_config.json" in metadata["metadata_sha256"]
    assert "stage2_peft_metadata.json" in metadata["metadata_sha256"]
