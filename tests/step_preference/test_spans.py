import hashlib
import json
import pickle
from pathlib import Path

import pytest
from PIL import Image
from transformers import AutoTokenizer

from step_preference.spans import (
    ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    PlanCharSpan,
    ReasoningStepPlan,
    ReasoningStepSpan,
    build_assistant_plan,
    canonical_sample_id,
    map_reasoning_step_spans,
    reject_duplicate_sample_ids,
)
from stage2.data import prepare_stage2_example


STAGE2A_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827"
)


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


def test_adjacent_text_space_lf_rendering_separates_spans_without_normalizing_content():
    """Catches fused adjacent text while preserving exact source whitespace."""
    sequence_plan = [
        {"type": "text", "content": "A."},
        {"type": "text", "content": "The final answer"},
    ]

    assistant, spans, _ = build_assistant_plan(
        sequence_plan,
        1,
        assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    )

    assert assistant == "A. \nThe final answer"
    assert (spans[0].start, spans[0].end) == (0, 2)
    assert (spans[1].start, spans[1].end) == (4, 20)
    assert assistant[spans[0].end : spans[1].start] == " \n"

    whitespace_plan = [
        {"type": "text", "content": " leading "},
        {"type": "text", "content": " trailing\n"},
    ]
    preserved, _, _ = build_assistant_plan(
        whitespace_plan,
        1,
        assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    )
    assert preserved == " leading  \n trailing\n"


def test_assistant_plan_rendering_defaults_to_concatenated_and_rejects_unknown_values():
    """Catches changing ordinary Stage 2 bytes or silently accepting a new identity."""
    sequence_plan = [
        {"type": "text", "content": "A."},
        {"type": "text", "content": "The final answer"},
    ]

    assistant, _, _ = build_assistant_plan(sequence_plan, 1)

    assert assistant == "A.The final answer"
    with pytest.raises(ValueError, match="unsupported assistant plan rendering: unknown-v1"):
        build_assistant_plan(
            sequence_plan,
            1,
            assistant_plan_rendering="unknown-v1",
        )


def test_adjacent_text_space_lf_rendering_prevents_cross_boundary_stage2a_tokens():
    """Catches `.The` or `.\\n` BPE merges defeating exact reasoning spans."""
    tokenizer = AutoTokenizer.from_pretrained(STAGE2A_CHECKPOINT)
    assert tokenizer.is_fast
    sequence_plan = [
        {"type": "text", "content": "before"},
        {"type": "latent", "helper_image": "helper.png"},
        {"type": "text", "content": "A."},
        {"type": "text", "content": "The final answer"},
    ]

    legacy_assistant, legacy_spans, _ = build_assistant_plan(sequence_plan, 1)
    legacy = tokenizer(
        [legacy_assistant],
        add_special_tokens=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    legacy_boundary = legacy_spans[2].end
    assert any(
        start < legacy_boundary < end
        and legacy_assistant[start:end] == ".The"
        for start, end in legacy["offset_mapping"][0].tolist()
    )

    lf_only = tokenizer(
        ["A.\nThe final answer"],
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    assert any(
        start < 2 < end and "A.\nThe final answer"[start:end] == ".\n"
        for start, end in lf_only["offset_mapping"][0]
    )

    assistant, spans, reasoning_steps = build_assistant_plan(
        sequence_plan,
        1,
        require_reasoning_units=True,
        assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    )
    encoded = tokenizer(
        [assistant],
        add_special_tokens=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"][0].tolist()
    target = spans[2]
    final_answer = spans[3]
    assert not any(
        start < endpoint < end
        for start, end in offsets
        for endpoint in (target.end, final_answer.start)
    )

    mapped = map_reasoning_step_spans(
        rendered_text=assistant,
        assistant_text=assistant,
        plan_char_spans=spans,
        reasoning_steps=reasoning_steps,
        offset_mapping=encoded["offset_mapping"],
        input_ids=encoded["input_ids"],
        latent_pad_token_id=tokenizer.convert_tokens_to_ids("<|latent_pad|>"),
    )[0]
    assert offsets[mapped.target_text_start][0] == target.start
    assert offsets[mapped.target_text_end - 1][1] == target.end

    delimiter_index = offsets.index([target.end, final_answer.start])
    final_answer_indices = {
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start >= final_answer.start and end <= final_answer.end
    }
    assert delimiter_index not in range(mapped.target_text_start, mapped.target_text_end)
    assert delimiter_index not in final_answer_indices


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
