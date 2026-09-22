import json
from pathlib import Path

from PIL import Image
import pytest
from transformers import AutoProcessor

from evaluation.vision_block_inputs import (
    load_prepared_samples,
    prepare_manifest,
    prepare_sample,
)


ARTIFACT = Path(
    "/root/workspace/ILVR/.worktrees/voff-reproduction-20260917/checkpoints/"
    "evaluation-staging/stage2-ce-matched-gb16-s212-20260919-155759/"
    "prepared-checkpoint-212"
)


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(ARTIFACT, local_files_only=True)


def _row(sequence_plan, *, final_answer="1"):
    return {
        "text_input": "Question?",
        "image_input": ["original.png"],
        "original_final_answer": final_answer,
        "sequence_plan": sequence_plan,
    }


def _latent():
    return {"type": "latent", "helper_image": "must-not-be-loaded.png"}


def test_blank_final_answer_keeps_reasoning_site(tmp_path, processor):
    """Catches treating blank answer metadata as a reason to drop a valid site."""
    Image.new("RGB", (56, 56)).save(tmp_path / "original.png")
    row = _row(
        [
            {"type": "text", "content": "Before."},
            _latent(),
            {"type": "text", "content": "Reason one."},
            {"type": "text", "content": "The final answer is: 1"},
        ],
        final_answer="",
    )

    sample, exclusions = prepare_sample(row, 57, processor, tmp_path)

    assert sample is not None
    assert len(sample.sites) == 1
    assert exclusions == []
    site = sample.sites[0]
    assert site.plan_index == 1
    assert site.target_positions[0] == site.end + 1
    assert "final answer" not in processor.tokenizer.decode(site.target_ids).lower()
    assert "pixel_values_latent" not in sample.model_inputs
    assert "labels" not in sample.model_inputs


def test_valid_first_site_survives_ineligible_later_site(tmp_path, processor):
    """Catches discarding a whole sample when only one latent has no text target."""
    Image.new("RGB", (56, 56)).save(tmp_path / "original.png")
    row = _row(
        [
            {"type": "text", "content": "Before."},
            _latent(),
            {"type": "text", "content": "Mapped reasoning."},
            _latent(),
            {"type": "text", "content": ""},
        ]
    )

    sample, exclusions = prepare_sample(row, 8, processor, tmp_path)

    assert sample is not None
    assert [site.plan_index for site in sample.sites] == [1]
    assert exclusions == [
        {
            "source_index": 8,
            "plan_index": 3,
            "reason": "ineligible_reasoning_unit",
            "explanation": "latent step lacks immediate non-empty text neighbors",
        }
    ]


def test_adjacent_final_answer_and_crossing_token_are_not_site_targets(
    tmp_path, processor
):
    """Catches including an adjacent plan-entry crossing token in target identity."""
    Image.new("RGB", (56, 56)).save(tmp_path / "original.png")
    row = _row(
        [
            {"type": "text", "content": "Before."},
            _latent(),
            {"type": "text", "content": "individuals."},
            {"type": "text", "content": "Leveraging final answer: 1"},
        ]
    )

    sample, exclusions = prepare_sample(row, 4, processor, tmp_path)

    assert sample is not None and not exclusions
    site = sample.sites[0]
    target = processor.tokenizer.decode(site.target_ids)
    assert "individuals" in target
    assert "Leveraging" not in target
    assert (
        sample.model_inputs["input_ids"][0, site.target_positions[0]].item()
        == site.target_ids[0]
    )
    assert (
        sample.model_inputs["input_ids"].shape
        == sample.model_inputs["attention_mask"].shape
    )


@pytest.mark.parametrize("indices", [[0] * 100, list(range(99)) + [1000]])
def test_manifest_rejects_duplicate_or_out_of_range_selection(
    tmp_path, processor, indices
):
    """Catches silently replacing malformed fixed-selection source indices."""
    data_path = tmp_path / "TEST.jsonl"
    data_path.write_text(json.dumps(_row([])) + "\n", encoding="utf-8")
    selection_path = tmp_path / "metadata.json"
    selection_path.write_text(
        json.dumps({"selection": {"requested_indices": indices}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="100 unique in-range"):
        prepare_manifest(
            data_path, selection_path, processor, tmp_path, tmp_path / "manifest.json"
        )


def test_loader_rejects_changed_target_identity(tmp_path, processor):
    """Catches replay proceeding after a persisted target token was changed."""
    Image.new("RGB", (56, 56)).save(tmp_path / "original.png")
    row = _row(
        [
            {"type": "text", "content": "Before."},
            _latent(),
            {"type": "text", "content": "Reason."},
        ]
    )
    sample, exclusions = prepare_sample(row, 0, processor, tmp_path)
    assert sample is not None and not exclusions
    manifest = {
        "schema_version": 1,
        "rows": [row],
        "samples": [
            {
                "source_index": 0,
                "identity": "changed",
                "sites": [],
            }
        ],
    }

    with pytest.raises(ValueError, match="identity changed"):
        list(load_prepared_samples(manifest, processor, tmp_path))
