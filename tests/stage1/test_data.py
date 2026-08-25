import json

from PIL import Image
import pytest

from helpers import make_stage1_collator, prepared_example
from stage1.data import prepare_stage1_comt_examples


@pytest.fixture
def stage1_collator():
    return make_stage1_collator()


def test_collator_records_per_sample_ownership_and_row_identity(stage1_collator):
    """Catches flattened image grids or latent blocks crossing sample ownership."""
    batch = stage1_collator(
        [
            prepared_example(
                source_index=7, original_images=1, helper_images=2, blocks=2
            ),
            prepared_example(
                source_index=11, original_images=2, helper_images=1, blocks=1
            ),
        ]
    )

    metadata = batch["stage1_batch_metadata"]
    assert metadata.batch_size == 2
    assert [sample.source_index for sample in metadata.samples] == [7, 11]
    assert metadata.samples[0].original_grid_range == (0, 1)
    assert metadata.samples[1].original_grid_range == (1, 3)
    assert metadata.samples[0].helper_grid_range == (0, 2)
    assert metadata.samples[1].helper_grid_range == (2, 3)
    assert [len(sample.latent_spans) for sample in metadata.samples] == [2, 1]
    assert metadata.samples[0].helper_to_block == (0, 1)
    assert metadata.samples[1].helper_to_block == (0,)
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["image_out_mask"].shape == batch["input_ids"].shape
    assert batch["pixel_values_latent"].shape[0] == 3
    assert batch["image_grid_thw_latent"].shape[0] == 3


def test_collator_repads_replaced_rows_and_records_unpadded_lengths(stage1_collator):
    """Catches stale attention/label widths after unequal rows expand to latent slots."""
    batch = stage1_collator(
        [
            prepared_example(
                source_index=3,
                original_images=1,
                helper_images=1,
                blocks=1,
                trailing_words=1,
            ),
            prepared_example(
                source_index=4,
                original_images=1,
                helper_images=2,
                blocks=2,
                trailing_words=5,
            ),
        ]
    )

    metadata = batch["stage1_batch_metadata"]
    assert metadata.samples[0].unpadded_length < metadata.samples[1].unpadded_length
    assert metadata.sequence_length == batch["input_ids"].shape[1]
    assert batch["attention_mask"][0].sum().item() == metadata.samples[0].unpadded_length
    assert batch["attention_mask"][1].sum().item() == metadata.samples[1].unpadded_length
    assert [len(sample.latent_spans) for sample in metadata.samples] == [1, 2]
    assert batch["image_out_mask"].sum().item() == 9


@pytest.mark.parametrize(
    ("tokens", "helper_count", "message"),
    [
        (["start", "pad", "end"], 2, "latent blocks.*helper images"),
        (["start", "pad", "pad", "end"], 1, "exactly 3 latent slots"),
        (["start", "start", "pad", "end", "end"], 1, "nonnested"),
        (["start", "pad"], 1, "incomplete"),
    ],
)
def test_collator_rejects_malformed_sample(tokens, helper_count, message):
    """Catches malformed latent scaffolds being silently paired or redistributed."""
    example = prepared_example(
        source_index=19,
        original_images=1,
        helper_images=helper_count,
        blocks=0,
        tokens=tokens,
    )

    with pytest.raises(ValueError, match=message) as error:
        make_stage1_collator()([example])

    assert "source_index=19" in str(error.value)
    assert "observed" in str(error.value)
    assert "expected" in str(error.value)


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        (["end"], "latent_end.*without"),
        (["pad"], "latent_pad.*outside"),
        (["start", "pad", "end", "end"], "latent_end.*without"),
    ],
)
def test_collator_rejects_missing_or_duplicated_special_tokens(tokens, message):
    """Catches unmatched special tokens being ignored by start/end zip pairing."""
    example = prepared_example(
        source_index=23,
        original_images=1,
        helper_images=1,
        blocks=0,
        tokens=tokens,
    )

    with pytest.raises(ValueError, match=message) as error:
        make_stage1_collator()([example])

    assert "source_index=23" in str(error.value)
    assert "observed" in str(error.value)
    assert "expected" in str(error.value)


def test_prepare_examples_uses_row_index_as_unique_identity_and_raw_id_as_provenance(
    tmp_path,
):
    """Catches duplicate raw IDs replacing unique JSONL row ownership keys."""
    for filename in ("first.png", "second.png", "helper-a.png", "helper-b.png"):
        Image.new("RGB", (2, 2)).save(tmp_path / filename)
    rows = [
        {
            "id": "duplicate",
            "image_input": "first.png",
            "text_input": "question one",
            "sequence_plan": [
                {"type": "latent", "helper_image": "helper-a.png"}
            ],
        },
        {
            "id": "duplicate",
            "image_input": "second.png",
            "text_input": "question two",
            "sequence_plan": [
                {"type": "latent", "helper_image": "helper-b.png"}
            ],
        },
    ]
    data_path = tmp_path / "TRAIN.jsonl"
    data_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    examples = prepare_stage1_comt_examples(data_path, tmp_path)

    assert [example.source_index for example in examples] == [0, 1]
    assert [example.source_id for example in examples] == ["duplicate", "duplicate"]
    assert examples[0].original_image_paths == (str(tmp_path / "first.png"),)
    assert examples[1].helper_image_paths == (str(tmp_path / "helper-b.png"),)
    assert examples[0].conversation[1]["content"][0]["image"] == str(
        tmp_path / "helper-a.png"
    )


@pytest.mark.parametrize(
    ("processor_kwargs", "message"),
    [
        ({"original_grid_rows": 1}, "original image grids"),
        ({"helper_grid_rows": 1}, "helper image grids"),
    ],
)
def test_collator_rejects_processor_grid_count_drift(processor_kwargs, message):
    """Catches processor output rows being silently assigned to the wrong sample."""
    examples = [
        prepared_example(source_index=5, original_images=1, helper_images=1, blocks=1),
        prepared_example(source_index=6, original_images=1, helper_images=1, blocks=1),
    ]

    with pytest.raises(ValueError, match=message) as error:
        make_stage1_collator(**processor_kwargs)(examples)

    assert "source_index=" in str(error.value)
    assert "observed" in str(error.value)
    assert "expected" in str(error.value)


def test_collator_rejects_tensor_shape_drift_with_sample_context():
    """Catches a processor attention shape that no longer matches token rows."""
    example = prepared_example(
        source_index=29, original_images=1, helper_images=1, blocks=1
    )

    with pytest.raises(ValueError, match="B,T") as error:
        make_stage1_collator(bad_mask=True)([example])

    assert "source_index=29" in str(error.value)
    assert "observed" in str(error.value)
    assert "expected" in str(error.value)
