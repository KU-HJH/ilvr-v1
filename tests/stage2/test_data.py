import json
from dataclasses import replace

import pytest
import torch
from PIL import Image

from stage2.data import (
    Stage2DataCollator,
    build_prepared_example_length_measure,
    build_stage2_conversation,
    build_stage2_labels,
    filter_stage2_examples,
    load_jsonl_examples,
    prepare_stage2_example,
    resolve_original_images,
)
from step_preference.spans import (
    ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    PlanCharSpan,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentSpan,
    LatentStructureError,
    LatentTokenIds,
    parse_latent_spans,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor


TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)


@pytest.mark.parametrize("mutation", [None, "unpadded_length", "padded_length", "sample_id", "cap"])
def test_packed_admission_collator_rechecks_natural_plan(tmp_path, monkeypatch, mutation):
    from tests.step_preference.test_packed_admission_cli import AdmissionProcessor
    from step_preference.negatives import StepNegative
    from step_preference.packed_admission import PackedAdmissionRecord
    from step_preference import packing

    Image.new("RGB", (2, 2)).save(tmp_path / "input.png")
    example = prepare_stage2_example(
        dict(sample_id="sample-a", image_input="input.png", text_input="question",
             sequence_plan=[dict(type="text", content="A"),
                            dict(type="latent", helper_image="ignored.png"),
                            dict(type="text", content="BC")]),
        image_root=tmp_path, latent_size=8, require_reasoning_units=True,
        assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    )
    processor = AdmissionProcessor()
    store = {("sample-a", 0): StepNegative("sample-a", 0, "BC", "DD", None, True, False, -1.0, -2.0, None)}
    kwargs = dict(processor=processor, token_ids=TOKENS, latent_size=8,
                  supervise_latent_boundaries=True, step_preference_enabled=True, negative_store=store)
    original = Stage2DataCollator(**kwargs)([example])
    natural = packing.build_packed_tree_plan(
        clean_input_ids=original["input_ids"], attention_mask=original["attention_mask"],
        reasoning_spans=original["reasoning_step_spans"], joined_steps=original["step_preference"]["steps"],
    )
    record = PackedAdmissionRecord(0, "sample-a", natural.root_length, natural.unpadded_length,
                                   natural.padded_length, 4, 1, 1, 3, 8, "retained")
    records = {"sample-a": record}
    if mutation == "sample_id":
        records = {}
    if mutation in ("unpadded_length", "padded_length"):
        planner = packing.build_packed_tree_plan
        def changed(**kwargs):
            plan = planner(**kwargs)
            return replace(plan, **{mutation: getattr(plan, mutation) + 128})
        monkeypatch.setattr(packing, "build_packed_tree_plan", changed)
    collator_kwargs = dict(kwargs, packed_admission_records_by_sample_id=records,
                           max_packed_sequence_length=128 if mutation == "cap" else 4608)
    if mutation is not None:
        with pytest.raises(ValueError, match="admission"):
            Stage2DataCollator(**collator_kwargs)([example])
        return
    collator = Stage2DataCollator(**collator_kwargs)
    records.clear()  # The caller cannot mutate the collator's authenticated snapshot.
    actual = collator([example])
    assert actual.keys() == original.keys()
    for key in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"):
        assert torch.equal(actual[key], original[key])
    assert actual["sample_ids"] == ("sample-a",)
    assert actual["reasoning_step_spans"] == original["reasoning_step_spans"]


class RecordingTokenizer:
    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert text == "<|im_start|>assistant"
        assert add_special_tokens is False
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[31, 32]])}


class RecordingProcessor:
    def __init__(self, input_ids=None):
        self.tokenizer = RecordingTokenizer()
        self.input_ids = (
            input_ids
            if input_ids is not None
            else torch.tensor([[1, 31, 32, 91, 90, 90, 92, 7]])
        )
        self.calls = []

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        return "rendered CoMT conversation"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "input_ids": self.input_ids.clone(),
            "attention_mask": torch.ones_like(self.input_ids),
            "pixel_values": torch.ones((1, 3, 2, 2)),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }


class QwenImageBoundaryProcessor(RecordingProcessor):
    """Run supplied images through the real local Qwen image processor."""

    def __init__(self):
        super().__init__()
        self.image_processor = Qwen2VLImageProcessor()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        image_inputs = self.image_processor(
            images=kwargs["images"], return_tensors=kwargs["return_tensors"]
        )
        return {
            "input_ids": self.input_ids.clone(),
            "attention_mask": torch.ones_like(self.input_ids),
            "pixel_values": image_inputs["pixel_values"],
            "image_grid_thw": image_inputs["image_grid_thw"],
        }


@pytest.fixture
def prepared_example(valid_sample, tmp_path):
    valid_sample["sequence_plan"] = [
        {"type": "text", "content": "before"},
        {"type": "latent", "helper_image": "ignored.png"},
        {"type": "text", "content": "after"},
    ]
    return prepare_stage2_example(valid_sample, image_root=tmp_path, latent_size=2)


def make_collator(input_ids=None):
    processor = RecordingProcessor(input_ids=input_ids)
    return Stage2DataCollator(
        processor=processor,
        token_ids=TOKENS,
        latent_size=2,
        supervise_latent_boundaries=True,
    )


@pytest.fixture
def valid_sample(tmp_path):
    Image.new("RGB", (56, 56)).save(tmp_path / "input.png")
    return {
        "image_input": "input.png",
        "text_input": "question",
        "sequence_plan": [{"type": "text", "content": "answer"}],
    }


def test_helper_steps_become_latent_blocks_without_helper_images(tmp_path):
    Image.new("RGB", (2, 2)).save(tmp_path / "input.png")
    sample = {
        "image_input": ["input.png"],
        "text_input": "question",
        "sequence_plan": [
            {"type": "text", "content": "A"},
            {"type": "latent", "helper_image": "helper.png"},
            {"type": "text", "content": "B"},
        ],
    }

    conversation = build_stage2_conversation(sample, image_root=tmp_path, latent_size=2)

    assert conversation[1]["content"][0]["text"] == (
        "A<|latent_start|><|latent_pad|><|latent_pad|><|latent_end|>B"
    )
    assert all(item["type"] != "image" for item in conversation[1]["content"])


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda sample: sample.update(sequence_plan="bad"), "sequence_plan must be a list"),
        (
            lambda sample: sample["sequence_plan"].append({"type": "latent"}),
            "latent step is missing helper_image",
        ),
        (
            lambda sample: sample["sequence_plan"].append({"type": "unknown"}),
            "unsupported sequence_plan step type",
        ),
        (
            lambda sample: sample["sequence_plan"].append({"type": "text", "content": 1}),
            "text step content must be a string",
        ),
    ],
)
def test_rejects_malformed_comt_trajectory(valid_sample, mutation, message, tmp_path):
    mutation(valid_sample)

    with pytest.raises(ValueError, match=message):
        build_stage2_conversation(valid_sample, image_root=tmp_path, latent_size=2)


def test_rejects_missing_original_image(valid_sample, tmp_path):
    valid_sample["image_input"] = "missing.png"

    with pytest.raises(FileNotFoundError, match="missing original image"):
        build_stage2_conversation(valid_sample, image_root=tmp_path, latent_size=2)


@pytest.mark.parametrize(
    "image_input",
    ("images_comt/creation/10003.png", "creation/10003.png"),
)
def test_original_image_resolver_accepts_canonical_and_root_relative_paths(
    tmp_path, image_input
):
    image_root = tmp_path / "images_comt"
    image = image_root / "creation/10003.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(image)

    assert resolve_original_images(
        {"image_input": image_input}, image_root
    ) == (str(image),)


def test_original_image_resolver_strips_the_root_name_only_once(tmp_path):
    image_root = tmp_path / "images_comt"
    image = image_root / "creation/10003.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(image)

    with pytest.raises(FileNotFoundError, match="missing original image"):
        resolve_original_images(
            {"image_input": "images_comt/images_comt/creation/10003.png"},
            image_root,
        )


@pytest.mark.parametrize(
    "image_input",
    ("/tmp/absolute.png", "../escape.png", "creation/control\n.png"),
)
def test_original_image_resolver_rejects_unsafe_paths(tmp_path, image_input):
    image_root = tmp_path / "images_comt"
    image_root.mkdir()

    with pytest.raises(ValueError, match="safe relative path"):
        resolve_original_images({"image_input": image_input}, image_root)


def test_original_image_resolver_rejects_a_symlink_escape(tmp_path):
    image_root = tmp_path / "images_comt"
    image_root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    (image_root / "escape.png").symlink_to(outside)

    with pytest.raises(ValueError, match="remain inside image_root"):
        resolve_original_images({"image_input": "escape.png"}, image_root)


def test_prepare_example_preserves_original_image_paths_and_latent_count(tmp_path):
    Image.new("RGB", (2, 2)).save(tmp_path / "first.png")
    Image.new("RGB", (2, 2)).save(tmp_path / "second.png")
    sample = {
        "image_input": ["first.png", "second.png"],
        "text_input": "question",
        "sequence_plan": [
            {"type": "latent", "helper_image": "ignored.png"},
            {"type": "text", "content": "answer"},
        ],
    }

    prepared = prepare_stage2_example(sample, image_root=tmp_path, latent_size=2)

    assert prepared.original_image_paths == (
        str(tmp_path / "first.png"),
        str(tmp_path / "second.png"),
    )
    assert prepared.latent_block_count == 1


def test_prepare_example_uses_one_adjacent_text_rendering_for_content_and_spans(
    tmp_path,
):
    """Catches conversation text and recorded spans using different renderings."""
    Image.new("RGB", (2, 2)).save(tmp_path / "input.png")
    sample = {
        "image_input": "input.png",
        "text_input": "question",
        "sequence_plan": [
            {"type": "text", "content": "A."},
            {"type": "text", "content": "The final answer"},
        ],
    }

    prepared = prepare_stage2_example(
        sample,
        image_root=tmp_path,
        latent_size=1,
        assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    )

    assert prepared.conversation[1]["content"][0]["text"] == "A. \nThe final answer"
    assert prepared.plan_char_spans == (
        PlanCharSpan(0, "text", 0, 2),
        PlanCharSpan(1, "text", 4, 20),
    )


def test_ordinary_stage2_prepare_and_collate_matches_the_legacy_golden(tmp_path):
    """Catches opt-in rendering changing any ordinary Stage 2 model input."""
    Image.new("RGB", (2, 2)).save(tmp_path / "input.png")
    sample = {
        "sample_id": "ordinary-golden",
        "image_input": "input.png",
        "text_input": "question",
        "sequence_plan": [
            {"type": "text", "content": "A"},
            {"type": "latent", "helper_image": "ignored.png"},
            {"type": "text", "content": "B"},
            {"type": "text", "content": "C"},
        ],
    }

    prepared = prepare_stage2_example(sample, image_root=tmp_path, latent_size=2)
    collator = make_collator()
    batch = collator([prepared])

    assert prepared.conversation == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(tmp_path / "input.png")},
                {"type": "text", "text": "question"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "A<|latent_start|><|latent_pad|><|latent_pad|><|latent_end|>BC",
                }
            ],
        },
    ]
    assert prepared.plan_char_spans == (
        PlanCharSpan(0, "text", 0, 1),
        PlanCharSpan(1, "latent", 1, 59),
        PlanCharSpan(2, "text", 59, 60),
        PlanCharSpan(3, "text", 60, 61),
    )
    assert torch.equal(
        batch["input_ids"], torch.tensor([[1, 31, 32, 91, 90, 90, 92, 7]])
    )
    assert torch.equal(batch["attention_mask"], torch.ones((1, 8), dtype=torch.long))
    assert torch.equal(
        batch["labels"],
        torch.tensor([[-100, -100, -100, 91, -100, -100, 92, 7]]),
    )
    assert torch.equal(batch["pixel_values"], torch.ones((1, 3, 2, 2)))
    assert torch.equal(batch["image_grid_thw"], torch.tensor([[1, 2, 2]]))
    assert parse_latent_spans(
        batch["input_ids"][0], token_ids=TOKENS, latent_size=2
    ) == (LatentSpan(3, (4, 5), 6),)


def test_load_jsonl_examples_ignores_blank_lines(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")

    assert load_jsonl_examples(path) == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize("supervise, expected", [(True, [91, 92]), (False, [-100, -100])])
def test_stage2_labels_mask_slots_and_pair_boundaries(supervise, expected):
    ids = torch.tensor([[5, 6, 91, 90, 90, 92, 7, 8]])

    labels = build_stage2_labels(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        assistant_content_start=2,
        token_ids=TOKENS,
        latent_size=2,
        supervise_latent_boundaries=supervise,
    )

    assert labels[0, 3:5].tolist() == [-100, -100]
    assert [labels[0, 2].item(), labels[0, 5].item()] == expected
    assert torch.all(labels[0, :2] == -100)


def test_stage2_labels_validate_tokenized_latent_structure():
    ids = torch.tensor([[5, 91, 90, 90]])

    with pytest.raises(LatentStructureError, match="missing latent_end"):
        build_stage2_labels(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            assistant_content_start=1,
            token_ids=TOKENS,
            latent_size=2,
            supervise_latent_boundaries=True,
        )


def test_stage2_collator_emits_only_original_visual_tensors(prepared_example):
    """Catches passing helper-image inputs or silently truncating a sample."""
    collator = make_collator()

    batch = collator([prepared_example])

    assert "pixel_values" in batch
    assert "image_grid_thw" in batch
    assert "pixel_values_latent" not in batch
    assert "image_grid_thw_latent" not in batch
    assert batch["input_ids"].shape[0] == 1
    assert collator.processor.calls[-1]["truncation"] is False
    images = collator.processor.calls[-1]["images"]
    assert len(images) == 1
    assert len(images[0]) == len(prepared_example.original_image_paths)
    assert all(isinstance(image, Image.Image) for image in images[0])
    assert batch["labels"][0, :3].tolist() == [-100, -100, -100]


def test_stage2_collator_rejects_two_samples(prepared_example):
    """Catches accidental batching before Stage 2 supports it."""
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        make_collator()([prepared_example, prepared_example])


def test_stage2_collator_revalidates_tokenized_structure(prepared_example):
    """Catches a processor whose final tokenization corrupts a latent block."""
    with pytest.raises(LatentStructureError, match="expected 2 latent_pad tokens"):
        make_collator(torch.tensor([[1, 31, 32, 91, 90, 92, 7]]))([prepared_example])


def test_prepared_length_measure_uses_original_images_without_truncation(prepared_example):
    """Catches length measurement that drops visual inputs or truncates evidence."""
    processor = RecordingProcessor()

    length = build_prepared_example_length_measure(processor)(prepared_example)

    assert length == 8
    call = processor.calls[-1]
    assert call["text"] == ["rendered CoMT conversation"]
    assert call["padding"] is False
    assert call["truncation"] is False
    images = call["images"]
    assert len(images) == 1
    assert len(images[0]) == len(prepared_example.original_image_paths)
    assert all(isinstance(image, Image.Image) for image in images[0])


def test_stage2_collator_materializes_original_images_for_qwen_processor(prepared_example):
    """Catches passing filesystem strings into Qwen's real image-processing boundary."""
    processor = QwenImageBoundaryProcessor()
    collator = Stage2DataCollator(
        processor=processor,
        token_ids=TOKENS,
        latent_size=2,
        supervise_latent_boundaries=True,
    )

    batch = collator([prepared_example])

    assert isinstance(batch["pixel_values"], torch.Tensor)
    assert batch["pixel_values"].numel() > 0
    images = processor.calls[-1]["images"]
    assert len(images) == 1
    assert len(images[0]) == len(prepared_example.original_image_paths)
    assert all(isinstance(image, Image.Image) for image in images[0])


def test_prepared_length_measure_materializes_images_for_qwen_processor(prepared_example):
    """Catches length measurement passing filesystem strings into Qwen image processing."""
    processor = QwenImageBoundaryProcessor()

    length = build_prepared_example_length_measure(processor)(prepared_example)

    assert length == 8
    assert processor.calls[-1]["truncation"] is False
    images = processor.calls[-1]["images"]
    assert len(images) == 1
    assert len(images[0]) == len(prepared_example.original_image_paths)
    assert all(isinstance(image, Image.Image) for image in images[0])


def test_length_filter_drops_whole_samples_and_reports_distribution():
    """Catches token-level truncation or statistics computed after filtering."""
    examples = [
        {"length": 4, "latent_block_count": 0},
        {"length": 8, "latent_block_count": 1},
        {"length": 12, "latent_block_count": 1},
    ]

    kept, stats = filter_stage2_examples(
        examples=examples,
        measure_length=lambda item: item["length"],
        max_length=8,
    )

    assert [item["length"] for item in kept] == [4, 8]
    assert (stats.total, stats.retained, stats.filtered) == (3, 2, 1)
    assert stats.filtered_percentage == pytest.approx(100 / 3)
    assert (stats.minimum, stats.median, stats.maximum) == (4, 8.0, 12)
    assert stats.latent_block_distribution == {0: 1, 1: 2}
