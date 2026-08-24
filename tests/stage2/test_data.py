import json

import pytest
import torch
from PIL import Image

from stage2.data import (
    build_stage2_conversation,
    build_stage2_labels,
    load_jsonl_examples,
    prepare_stage2_example,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds, LatentStructureError


TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)


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
