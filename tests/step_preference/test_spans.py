import hashlib
import json
import pickle

import pytest
from PIL import Image

from step_preference.spans import (
    PlanCharSpan,
    ReasoningStepPlan,
    ReasoningStepSpan,
    build_assistant_plan,
    canonical_sample_id,
    reject_duplicate_sample_ids,
)
from stage2.data import prepare_stage2_example


def test_canonical_sample_id_preserves_nonempty_explicit_value():
    """Catches replacing a source-provided identity with a content hash."""
    row = {"sample_id": "row-17", "text_input": "ignored"}

    assert canonical_sample_id(row) == "row-17"


def test_canonical_sample_id_hashes_exact_canonical_utf8_json():
    """Catches hashing a differently ordered, spaced, or ASCII-escaped row."""
    row = {"text_input": "café", "sample_id": "", "image_input": ["z.png"]}
    expected_json = '{"image_input":["z.png"],"sample_id":"","text_input":"café"}'
    expected = hashlib.sha256(expected_json.encode("utf-8")).hexdigest()

    assert canonical_sample_id(row) == expected
    assert json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == expected_json


def test_reject_duplicate_sample_ids_reports_the_reused_identity():
    """Catches accepting two samples that would join to one negative-artifact row."""
    with pytest.raises(ValueError, match="duplicate sample_id: duplicate"):
        reject_duplicate_sample_ids(["first", "duplicate", "duplicate"])


@pytest.mark.parametrize(
    "sequence_plan",
    [
        [
            {"type": "text", "content": "before"},
            {"type": "latent", "helper_image": "helper.png"},
            {"type": "text", "content": ""},
        ],
        [
            {"type": "text", "content": "before"},
            {"type": "text", "content": "also-before"},
            {"type": "latent", "helper_image": "helper.png"},
        ],
    ],
)
def test_build_assistant_plan_requires_immediate_nonempty_text_neighbors(sequence_plan):
    """Catches allowing a latent plan entry without its exact reasoning-unit triple."""
    with pytest.raises(ValueError, match="non-empty text"):
        build_assistant_plan(sequence_plan, latent_size=2, require_reasoning_units=True)


def test_build_assistant_plan_uses_latent_ordinals_and_half_open_character_intervals():
    """Catches raw plan indices, helper-index drift, or inclusive assistant offsets."""
    sequence_plan = [
        {"type": "text", "content": "A"},
        {"type": "latent", "helper_image": "first.png"},
        {"type": "text", "content": "BC"},
        {"type": "latent", "helper_image": "second.png"},
        {"type": "text", "content": "D"},
    ]
    latent_block = "<|latent_start|><|latent_pad|><|latent_pad|><|latent_end|>"
    assistant_text, plan_char_spans, reasoning_steps = build_assistant_plan(
        sequence_plan, latent_size=2, require_reasoning_units=True
    )

    assert assistant_text == f"A{latent_block}BC{latent_block}D"
    assert reasoning_steps == (
        ReasoningStepPlan(
            step_id=0,
            prev_text=PlanCharSpan(0, "text", 0, 1),
            latent=PlanCharSpan(1, "latent", 1, 59),
            target_text=PlanCharSpan(2, "text", 59, 61),
            helper_image_idx=0,
        ),
        ReasoningStepPlan(
            step_id=1,
            prev_text=PlanCharSpan(2, "text", 59, 61),
            latent=PlanCharSpan(3, "latent", 61, 119),
            target_text=PlanCharSpan(4, "text", 119, 120),
            helper_image_idx=1,
        ),
    )
    assert plan_char_spans == (
        PlanCharSpan(0, "text", 0, 1),
        PlanCharSpan(1, "latent", 1, 59),
        PlanCharSpan(2, "text", 59, 61),
        PlanCharSpan(3, "latent", 61, 119),
        PlanCharSpan(4, "text", 119, 120),
    )
    assert assistant_text[0:1] == "A"
    assert assistant_text[17:45] == "<|latent_pad|><|latent_pad|>"
    assert assistant_text[59:61] == "BC"


def test_plan_records_are_immutable_and_token_spans_have_contract_fields():
    """Catches mutable span metadata or a token-span record that loses helper provenance."""
    plan_span = PlanCharSpan(0, "text", 3, 5)
    token_span = ReasoningStepSpan(2, 10, 13, 14, 16, 17, 19, 2)

    with pytest.raises(Exception):
        plan_span.start = 0
    assert token_span == ReasoningStepSpan(
        step_id=2,
        prev_text_start=10,
        prev_text_end=13,
        latent_start=14,
        latent_end=16,
        target_text_start=17,
        target_text_end=19,
        helper_image_idx=2,
    )


def test_prepare_example_keeps_identity_and_plan_metadata_without_changing_assistant_output(tmp_path):
    """Catches Stage 2 dropping metadata required by the opt-in span-mapping path."""
    Image.new("RGB", (2, 2)).save(tmp_path / "input.png")
    sample = {
        "sample_id": "stable-row",
        "image_input": ["input.png"],
        "text_input": "question",
        "sequence_plan": [
            {"type": "text", "content": "A"},
            {"type": "latent", "helper_image": "helper.png"},
            {"type": "text", "content": "B"},
        ],
    }

    prepared = prepare_stage2_example(sample, image_root=tmp_path, latent_size=1)

    assert prepared.sample_id == "stable-row"
    assert prepared.sequence_plan == (
        {"type": "text", "content": "A"},
        {"type": "latent", "helper_image": "helper.png"},
        {"type": "text", "content": "B"},
    )
    assert prepared.plan_char_spans == (
        PlanCharSpan(0, "text", 0, 1),
        PlanCharSpan(1, "latent", 1, 45),
        PlanCharSpan(2, "text", 45, 46),
    )
    assert prepared.reasoning_steps[0].helper_image_idx == 0
    assert prepared.conversation[1]["content"][0]["text"] == (
        "A<|latent_start|><|latent_pad|><|latent_end|>B"
    )


def test_prepared_sequence_plan_snapshot_is_pickle_safe_and_immutable(tmp_path):
    """Catches worker-queue serialization failures from an unpickleable snapshot."""
    Image.new("RGB", (2, 2)).save(tmp_path / "input.png")
    sample = {
        "image_input": ["input.png"],
        "text_input": "question",
        "sequence_plan": [{"type": "text", "content": "answer"}],
    }

    restored = pickle.loads(
        pickle.dumps(prepare_stage2_example(sample, image_root=tmp_path, latent_size=1))
    )

    assert restored.sequence_plan == ({"type": "text", "content": "answer"},)
    with pytest.raises(TypeError, match="immutable dictionary"):
        restored.sequence_plan[0]["content"] = "changed"
