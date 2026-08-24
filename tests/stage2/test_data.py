import json

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
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds, LatentStructureError


TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)


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
    Image.new("RGB", (2, 2)).save(tmp_path / "input.png")
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
    assert collator.processor.calls[-1]["images"] == [
        list(prepared_example.original_image_paths)
    ]
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
    assert processor.calls == [{
        "text": ["rendered CoMT conversation"],
        "images": [list(prepared_example.original_image_paths)],
        "padding": False,
        "truncation": False,
        "return_tensors": "pt",
    }]


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
