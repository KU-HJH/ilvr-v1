import json
from pathlib import Path

from PIL import Image
import pytest
from transformers import AutoProcessor

from evaluation.vision_block_inputs import (
    _sample_identity,
    load_prepared_samples,
    prepare_manifest,
    prepare_sample,
)
from stage2.data import prepare_stage2_example
from step_preference.spans import expand_qwen_image_placeholders


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
    prepared = prepare_stage2_example(
        row,
        image_root=tmp_path,
        latent_size=8,
        assistant_plan_rendering="concatenated-v1",
    )
    rendered = processor.apply_chat_template(
        prepared.conversation, tokenize=False, add_generation_prompt=False
    )
    expanded = expand_qwen_image_placeholders(
        rendered,
        image_grid_thw=sample.model_inputs["image_grid_thw"],
        image_token=processor.image_token,
        merge_size=processor.image_processor.merge_size,
    )
    offsets = processor.tokenizer(
        [expanded],
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )["offset_mapping"][0].tolist()
    assistant_start = expanded.rfind(prepared.conversation[1]["content"][0]["text"])
    target_span = prepared.plan_char_spans[2]
    target_start = assistant_start + target_span.start
    boundary = assistant_start + target_span.end
    crossing = [
        index
        for index, (start, end) in enumerate(offsets)
        if start < boundary < end
    ]
    expected_positions = tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start >= target_start and end <= boundary
    )
    expected_ids = tuple(
        int(sample.model_inputs["input_ids"][0, index])
        for index in expected_positions
    )

    assert crossing and crossing[0] not in site.target_positions
    assert (
        sample.model_inputs["input_ids"][0, crossing[0]].item()
        == processor.tokenizer.convert_tokens_to_ids(".L")
    )
    assert site.target_positions == expected_positions
    assert site.target_ids == expected_ids
    assert site.target_positions[0] == site.end + 1


def test_global_span_mapping_failure_excludes_the_sample_once(
    tmp_path, processor, monkeypatch
):
    """Catches labeling a globally invalid rendered plan as many site failures."""
    Image.new("RGB", (56, 56)).save(tmp_path / "original.png")
    row = _row(
        [
            {"type": "text", "content": "Before."},
            _latent(),
            {"type": "text", "content": "First."},
            {"type": "text", "content": "Again."},
            _latent(),
            {"type": "text", "content": "Second."},
        ]
    )
    import evaluation.vision_block_inputs as inputs

    real_mapper = inputs.map_reasoning_step_spans

    def globally_invalid(**kwargs):
        if not kwargs["reasoning_steps"]:
            raise ValueError("token 41 crosses a plan boundary")
        return real_mapper(**kwargs)

    monkeypatch.setattr(inputs, "map_reasoning_step_spans", globally_invalid)

    sample, exclusions = prepare_sample(row, 9, processor, tmp_path)

    assert sample is None
    assert exclusions == [
        {
            "source_index": 9,
            "plan_index": None,
            "reason": "sample_span_mapping_invalid",
            "explanation": "token 41 crosses a plan boundary",
        }
    ]


def test_site_local_mapping_failure_preserves_an_earlier_valid_site(
    tmp_path, processor, monkeypatch
):
    """Catches global validation accidentally mapping and requiring every site."""
    Image.new("RGB", (56, 56)).save(tmp_path / "original.png")
    row = _row(
        [
            {"type": "text", "content": "Before."},
            _latent(),
            {"type": "text", "content": "First."},
            {"type": "text", "content": "Again."},
            _latent(),
            {"type": "text", "content": "Second."},
        ]
    )
    import evaluation.vision_block_inputs as inputs

    real_mapper = inputs.map_reasoning_step_spans

    def later_site_invalid(**kwargs):
        steps = kwargs["reasoning_steps"]
        if steps and steps[0].latent.plan_index == 4:
            raise ValueError(
                "character span does not map to an exact contiguous token interval"
            )
        return real_mapper(**kwargs)

    monkeypatch.setattr(inputs, "map_reasoning_step_spans", later_site_invalid)

    sample, exclusions = prepare_sample(row, 10, processor, tmp_path)

    assert sample is not None
    assert [site.plan_index for site in sample.sites] == [1]
    assert [(item["plan_index"], item["reason"]) for item in exclusions] == [
        (4, "unmappable_site")
    ]


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


@pytest.mark.parametrize("field", ["target_positions", "target_ids"])
def test_loader_rejects_tampered_persisted_site(tmp_path, processor, field):
    """Catches replay trusting identity while explicit persisted site data changed."""
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
    persisted_site = dict(sample.sites[0].__dict__)
    persisted_site[field] = [*persisted_site[field][:-1], 999999]
    manifest = {
        "schema_version": 1,
        "rows": [row],
        "samples": [
            {
                "source_index": 0,
                "identity": _sample_identity(sample),
                "sites": [persisted_site],
            }
        ],
    }

    with pytest.raises(ValueError, match="identity changed"):
        list(load_prepared_samples(manifest, processor, tmp_path))


def test_loader_rejects_non_integer_persisted_site_field(tmp_path, processor):
    """Catches Python equality allowing bool to impersonate integer plan index one."""
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
    persisted_site = dict(sample.sites[0].__dict__)
    persisted_site["plan_index"] = True
    manifest = {
        "schema_version": 1,
        "rows": [row],
        "samples": [
            {
                "source_index": 0,
                "identity": _sample_identity(sample),
                "sites": [persisted_site],
            }
        ],
    }

    with pytest.raises(ValueError, match="identity changed"):
        list(load_prepared_samples(manifest, processor, tmp_path))
