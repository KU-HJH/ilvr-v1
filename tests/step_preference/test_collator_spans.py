import torch
from PIL import Image
import pytest

from stage2.data import Stage2DataCollator, prepare_stage2_example
from step_preference import spans
from step_preference.spans import (
    PlanCharSpan,
    ReasoningStepPlan,
    ReasoningStepSpan,
)
from step_preference.negatives import StepNegative
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)
LATENT_START = "<|latent_start|>"
LATENT_PAD = "<|latent_pad|>"
LATENT_END = "<|latent_end|>"
IMAGE_PAD = "<|image_pad|>"


def test_expand_qwen_image_placeholders_uses_each_grid_in_rendered_order():
    """Catches skipping or reordering Qwen's per-image placeholder expansion."""
    rendered = f"left{IMAGE_PAD}middle{IMAGE_PAD}right"

    expanded = spans.expand_qwen_image_placeholders(
        rendered,
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 2, 2]]),
        image_token=IMAGE_PAD,
        merge_size=2,
    )

    assert expanded == (
        f"left{IMAGE_PAD * 4}middle{IMAGE_PAD}right"
    )


def test_map_reasoning_step_spans_uses_full_rendered_offsets_and_excludes_controls():
    """Catches assistant-relative token indices or latent controls entering a step span."""
    assistant = f"A{LATENT_START}{LATENT_PAD}{LATENT_PAD}{LATENT_END}BC"
    rendered = f"prompt::{assistant}::suffix"
    # Hand-derived full-string offsets: prompt token, A, start, two pads, end, BC, suffix.
    offsets = torch.tensor(
        [[[0, 8], [8, 9], [9, 25], [25, 39], [39, 53], [53, 67], [67, 69], [69, 77]]]
    )
    input_ids = torch.tensor([[10, 11, 91, 90, 90, 92, 12, 13]])
    step = ReasoningStepPlan(
        step_id=0,
        prev_text=PlanCharSpan(0, "text", 0, 1),
        latent=PlanCharSpan(1, "latent", 1, 59),
        target_text=PlanCharSpan(2, "text", 59, 61),
        helper_image_idx=0,
    )

    mapped = spans.map_reasoning_step_spans(
        rendered_text=rendered,
        assistant_text=assistant,
        plan_char_spans=(step.prev_text, step.latent, step.target_text),
        reasoning_steps=(step,),
        offset_mapping=offsets,
        input_ids=input_ids,
        latent_pad_token_id=90,
    )

    assert mapped == (
        ReasoningStepSpan(
            step_id=0,
            prev_text_start=1,
            prev_text_end=2,
            latent_start=3,
            latent_end=5,
            target_text_start=6,
            target_text_end=7,
            helper_image_idx=0,
        ),
    )


def test_map_reasoning_step_spans_rejects_a_token_crossing_a_plan_boundary():
    """Catches silently assigning a tokenizer merge shared by adjacent plan entries."""
    assistant = f"A{LATENT_START}{LATENT_PAD}{LATENT_END}B"
    rendered = f"xx{assistant}yy"
    step = ReasoningStepPlan(
        step_id=0,
        prev_text=PlanCharSpan(0, "text", 0, 1),
        latent=PlanCharSpan(1, "latent", 1, 45),
        target_text=PlanCharSpan(2, "text", 45, 46),
        helper_image_idx=0,
    )
    # Token 1 starts in A and ends inside latent_start, crossing assistant char 1.
    offsets = torch.tensor(
        [[[0, 2], [2, 4], [4, 18], [18, 32], [32, 46], [46, 48], [48, 50]]]
    )

    with pytest.raises(ValueError, match="crosses a plan boundary"):
        spans.map_reasoning_step_spans(
            rendered_text=rendered,
            assistant_text=assistant,
            plan_char_spans=(step.prev_text, step.latent, step.target_text),
            reasoning_steps=(step,),
            offset_mapping=offsets,
            input_ids=torch.tensor([[10, 11, 91, 90, 92, 12, 13]]),
            latent_pad_token_id=90,
        )


def test_immediate_text_excludes_tokens_crossing_adjacent_text_joins():
    """Catches rejecting BPE merges at the allowed previous/target text joins."""
    assistant = f"PABC{LATENT_START}{LATENT_PAD}{LATENT_END}DE.F"
    rendered = f"xx{assistant}yy"
    plan_spans = (
        PlanCharSpan(0, "text", 0, 1),
        PlanCharSpan(1, "text", 1, 4),
        PlanCharSpan(2, "latent", 4, 48),
        PlanCharSpan(3, "text", 48, 51),
        PlanCharSpan(4, "text", 51, 52),
    )
    step = ReasoningStepPlan(
        step_id=0,
        prev_text=plan_spans[1],
        latent=plan_spans[2],
        target_text=plan_spans[3],
        helper_image_idx=0,
    )
    # Tokens 1 and 7 cross recognized text/text joins. The fully contained
    # `BC` and `DE` tokens remain the immediate preference intervals.
    offsets = torch.tensor(
        [[[0, 2], [2, 4], [4, 6], [6, 22], [22, 36], [36, 50],
          [50, 52], [52, 54], [54, 56]]]
    )

    mapped = spans.map_reasoning_step_spans(
        rendered_text=rendered,
        assistant_text=assistant,
        plan_char_spans=plan_spans,
        reasoning_steps=(step,),
        offset_mapping=offsets,
        input_ids=torch.tensor([[10, 11, 12, 91, 90, 92, 13, 14, 15]]),
        latent_pad_token_id=90,
        preference_target_scope="immediate_text",
    )

    assert mapped == (
        ReasoningStepSpan(
            step_id=0,
            prev_text_start=2,
            prev_text_end=3,
            latent_start=4,
            latent_end=5,
            target_text_start=6,
            target_text_end=7,
            helper_image_idx=0,
        ),
    )


def test_immediate_text_rejects_an_endpoint_gap_without_a_crossing_token():
    """Catches treating every inexact endpoint at a text join as an allowed merge."""
    assistant = f"A{LATENT_START}{LATENT_PAD}{LATENT_END}DE.F"
    rendered = f"xx{assistant}yy"
    plan_spans = (
        PlanCharSpan(0, "text", 0, 1),
        PlanCharSpan(1, "latent", 1, 45),
        PlanCharSpan(2, "text", 45, 48),
        PlanCharSpan(3, "text", 48, 49),
    )
    step = ReasoningStepPlan(0, plan_spans[0], plan_spans[1], plan_spans[2], 0)
    # The target's period at [49, 50) has no token. No token crosses the
    # recognized join at 50, so the target interval must remain invalid.
    offsets = torch.tensor(
        [[[0, 2], [2, 3], [3, 19], [19, 33], [33, 47], [47, 49],
          [50, 51], [51, 53]]]
    )

    with pytest.raises(ValueError, match="exact contiguous token interval"):
        spans.map_reasoning_step_spans(
            rendered_text=rendered,
            assistant_text=assistant,
            plan_char_spans=plan_spans,
            reasoning_steps=(step,),
            offset_mapping=offsets,
            input_ids=torch.tensor([[10, 11, 91, 90, 92, 12, 13, 14]]),
            latent_pad_token_id=90,
            preference_target_scope="immediate_text",
        )


def test_immediate_text_keeps_assistant_start_strict_with_leading_empty_text():
    """Catches an empty/text join admitting a token that starts before the assistant."""
    assistant, plan_spans, reasoning_steps = spans.build_assistant_plan(
        [
            {"type": "text", "content": ""},
            {"type": "text", "content": "ABC"},
            {"type": "latent", "helper_image": "helper.png"},
            {"type": "text", "content": "D"},
        ],
        latent_size=1,
        require_reasoning_units=True,
    )

    with pytest.raises(ValueError, match="crosses a plan boundary"):
        spans.map_reasoning_step_spans(
            rendered_text=f"xx{assistant}yy",
            assistant_text=assistant,
            plan_char_spans=plan_spans,
            reasoning_steps=reasoning_steps,
            offset_mapping=torch.tensor(
                [[[0, 3], [3, 5], [5, 21], [21, 35], [35, 49],
                  [49, 50], [50, 52]]]
            ),
            input_ids=torch.tensor([[10, 11, 91, 90, 92, 12, 13]]),
            latent_pad_token_id=90,
            preference_target_scope="immediate_text",
        )


def test_immediate_text_keeps_assistant_end_strict_with_trailing_empty_text():
    """Catches a text/empty join admitting a token that ends after the assistant."""
    assistant, plan_spans, reasoning_steps = spans.build_assistant_plan(
        [
            {"type": "text", "content": "A"},
            {"type": "latent", "helper_image": "helper.png"},
            {"type": "text", "content": "BCD"},
            {"type": "text", "content": ""},
        ],
        latent_size=1,
        require_reasoning_units=True,
    )

    with pytest.raises(ValueError, match="crosses a plan boundary"):
        spans.map_reasoning_step_spans(
            rendered_text=f"xx{assistant}yy",
            assistant_text=assistant,
            plan_char_spans=plan_spans,
            reasoning_steps=reasoning_steps,
            offset_mapping=torch.tensor(
                [[[0, 2], [2, 3], [3, 19], [19, 33], [33, 47],
                  [47, 49], [49, 52]]]
            ),
            input_ids=torch.tensor([[10, 11, 91, 90, 92, 12, 13]]),
            latent_pad_token_id=90,
            preference_target_scope="immediate_text",
        )


def test_map_reasoning_step_spans_rejects_start_to_pad_crossing_for_unmapped_latent():
    """Catches skipping the inner start/pad boundary of an incomplete latent unit."""
    assistant = f"{LATENT_START}{LATENT_PAD}{LATENT_END}"

    with pytest.raises(ValueError, match="crosses a plan boundary"):
        spans.map_reasoning_step_spans(
            rendered_text=f"xx{assistant}yy",
            assistant_text=assistant,
            plan_char_spans=(PlanCharSpan(0, "latent", 0, 44),),
            reasoning_steps=(),
            offset_mapping=torch.tensor(
                [[[0, 2], [2, 19], [19, 32], [32, 46], [46, 48]]]
            ),
            input_ids=torch.tensor([[10, 91, 90, 92, 11]]),
            latent_pad_token_id=90,
        )


def test_map_reasoning_step_spans_rejects_pad_to_end_crossing_for_unmapped_latent():
    """Catches skipping the inner pad/end boundary of an incomplete latent unit."""
    assistant = f"{LATENT_START}{LATENT_PAD}{LATENT_END}"

    with pytest.raises(ValueError, match="crosses a plan boundary"):
        spans.map_reasoning_step_spans(
            rendered_text=f"xx{assistant}yy",
            assistant_text=assistant,
            plan_char_spans=(PlanCharSpan(0, "latent", 0, 44),),
            reasoning_steps=(),
            offset_mapping=torch.tensor(
                [[[0, 2], [2, 18], [18, 33], [33, 46], [46, 48]]]
            ),
            input_ids=torch.tensor([[10, 91, 90, 92, 11]]),
            latent_pad_token_id=90,
        )


class CharacterOffsetTokenizer:
    """Small deterministic fast-tokenizer boundary for collator contract tests."""

    is_fast = True

    def __init__(self):
        self.calls = []
        self._special_ids = {
            "<|im_start|>": 31,
            "<|im_end|>": 32,
            IMAGE_PAD: 70,
            LATENT_START: 91,
            LATENT_PAD: 90,
            LATENT_END: 92,
        }

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_tensors=None,
        return_offsets_mapping=False,
    ):
        self.calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "padding": padding,
                "truncation": truncation,
                "return_tensors": return_tensors,
                "return_offsets_mapping": return_offsets_mapping,
            }
        )
        texts = [text] if isinstance(text, str) else text
        encoded_ids = []
        encoded_offsets = []
        for value in texts:
            ids = []
            offsets = []
            cursor = 0
            while cursor < len(value):
                special = next(
                    (token for token in self._special_ids if value.startswith(token, cursor)),
                    None,
                )
                if special is not None:
                    ids.append(self._special_ids[special])
                    offsets.append((cursor, cursor + len(special)))
                    cursor += len(special)
                else:
                    ids.append(100 + ord(value[cursor]))
                    offsets.append((cursor, cursor + 1))
                    cursor += 1
            encoded_ids.append(ids)
            encoded_offsets.append(offsets)

        result = {
            "input_ids": torch.tensor(encoded_ids),
            "attention_mask": torch.ones(
                (len(encoded_ids), len(encoded_ids[0])), dtype=torch.long
            ),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor(encoded_offsets)
        return result


class SinglePassQwenProcessor:
    image_token = IMAGE_PAD

    class ImageProcessor:
        merge_size = 2

    image_processor = ImageProcessor()

    def __init__(self):
        self.tokenizer = CharacterOffsetTokenizer()
        self.image_processing_calls = 0
        self.chat_template_calls = []
        self.processor_calls = []

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        self.chat_template_calls.append(
            {
                "conversation": conversation,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        assert tokenize is False
        assert add_generation_prompt is False
        assistant = conversation[1]["content"][0]["text"]
        return (
            "<|im_start|>user\n<|image_pad|>question<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
        )

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        self.processor_calls.append(
            {
                "text": text,
                "images": images,
                "padding": padding,
                "truncation": truncation,
                "return_tensors": return_tensors,
            }
        )
        assert len(images) == 1 and len(images[0]) == 1
        self.image_processing_calls += 1
        # One 4x4 grid with merge_size=2 produces exactly four image placeholders.
        expanded = text[0].replace(IMAGE_PAD, IMAGE_PAD * 4, 1)
        encoded = self.tokenizer(
            [expanded],
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        encoded["pixel_values"] = torch.arange(12).reshape(1, 3, 2, 2)
        encoded["image_grid_thw"] = torch.tensor([[1, 4, 4]])
        return encoded


@pytest.fixture
def mapped_example(tmp_path):
    Image.new("RGB", (2, 2)).save(tmp_path / "input.png")
    return prepare_stage2_example(
        {
            "sample_id": "sample-7",
            "image_input": "input.png",
            "text_input": "question",
            "sequence_plan": [
                {"type": "text", "content": "A"},
                {"type": "latent", "helper_image": "helper.png"},
                {"type": "text", "content": "BC"},
            ],
        },
        image_root=tmp_path,
        latent_size=2,
    )


def make_mapped_collator(processor, *, enabled):
    return Stage2DataCollator(
        processor=processor,
        token_ids=TOKENS,
        latent_size=2,
        supervise_latent_boundaries=True,
        step_preference_enabled=enabled,
    )


def test_stage2b_collator_joins_exact_step_and_tokenizes_each_valid_rejection_as_one_chat(
    mapped_example,
):
    """Catches loose joins or separately tokenized prefix/completion fragments."""
    processor = SinglePassQwenProcessor()
    negative = StepNegative(
        sample_id="sample-7",
        step_id=0,
        chosen_text="BC",
        rejected_visual_text="XY",
        rejected_textual_text="ZQ",
        valid_visual_negative=True,
        valid_textual_negative=True,
        reference_chosen_score=-0.2,
        reference_visual_rejected_score=-0.8,
        reference_textual_rejected_score=-0.7,
    )
    collator = Stage2DataCollator(
        processor=processor,
        token_ids=TOKENS,
        latent_size=2,
        supervise_latent_boundaries=True,
        step_preference_enabled=True,
        negative_store={("sample-7", 0): negative},
    )

    batch = collator([mapped_example])

    step = batch["step_preference"]["steps"][0]
    assert step["negative"] is negative
    assert step["span"] == batch["reasoning_step_spans"][0]
    assert step["visual_rejected"]["completion_token_positions"]
    assert step["textual_rejected"]["completion_token_positions"]
    assert processor.image_processing_calls == 3
    processed_texts = [call["text"][0] for call in processor.processor_calls]
    assert processed_texts[1] == (
        "<|im_start|>user\n<|image_pad|>question<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"A{LATENT_START}{LATENT_PAD * 2}{LATENT_END}XY<|im_end|>\n"
    )
    assert processed_texts[2].endswith(
        f"A{LATENT_START}{LATENT_PAD * 2}{LATENT_END}ZQ<|im_end|>\n"
    )


def test_stage2b_collator_keeps_invalid_pairs_for_entropy_without_rejected_tokenization(
    mapped_example,
):
    """Catches invalid DPO pairs removing a structural entropy step or adding a branch."""
    processor = SinglePassQwenProcessor()
    negative = StepNegative(
        sample_id="sample-7",
        step_id=0,
        chosen_text="BC",
        rejected_visual_text=None,
        rejected_textual_text=None,
        valid_visual_negative=False,
        valid_textual_negative=False,
        reference_chosen_score=-0.2,
        reference_visual_rejected_score=None,
        reference_textual_rejected_score=None,
    )
    collator = Stage2DataCollator(
        processor=processor,
        token_ids=TOKENS,
        latent_size=2,
        supervise_latent_boundaries=True,
        step_preference_enabled=True,
        negative_store={("sample-7", 0): negative},
    )

    batch = collator([mapped_example])

    assert len(batch["step_preference"]["steps"]) == 1
    assert batch["step_preference"]["steps"][0]["visual_rejected"] is None
    assert batch["step_preference"]["steps"][0]["textual_rejected"] is None
    assert processor.image_processing_calls == 1


def test_disabled_collator_matches_captured_prechange_batch_exactly(mapped_example):
    """Catches any shared branch regression hidden by comparing new paths together."""
    processor = SinglePassQwenProcessor()

    batch = make_mapped_collator(processor, enabled=False)([mapped_example])

    assert set(batch) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "labels",
    }
    assert torch.equal(
        batch["input_ids"],
        torch.tensor(
            [[
                31,
                217,
                215,
                201,
                214,
                110,
                70,
                70,
                70,
                70,
                213,
                217,
                201,
                215,
                216,
                205,
                211,
                210,
                32,
                110,
                31,
                197,
                215,
                215,
                205,
                215,
                216,
                197,
                210,
                216,
                110,
                165,
                91,
                90,
                90,
                92,
                166,
                167,
                32,
                110,
            ]]
        ),
    )
    assert torch.equal(batch["attention_mask"], torch.ones((1, 40), dtype=torch.long))
    assert torch.equal(
        batch["labels"],
        torch.tensor(
            [[-100] * 30 + [110, 165, 91, -100, -100, 92, 166, 167, 32, 110]]
        ),
    )
    assert torch.equal(
        batch["pixel_values"],
        torch.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]]),
    )
    assert torch.equal(batch["image_grid_thw"], torch.tensor([[1, 4, 4]]))
    assert processor.image_processing_calls == 1
    assert processor.chat_template_calls == [
        {
            "conversation": mapped_example.conversation,
            "tokenize": False,
            "add_generation_prompt": False,
        }
    ]
    assert len(processor.processor_calls) == 1
    processor_call = processor.processor_calls[0]
    assert set(processor_call) == {
        "text",
        "images",
        "padding",
        "truncation",
        "return_tensors",
    }
    assert processor_call["text"] == [
        "<|im_start|>user\n<|image_pad|>question<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"A{LATENT_START}{LATENT_PAD * 2}{LATENT_END}BC<|im_end|>\n"
    ]
    assert processor_call["padding"] is True
    assert processor_call["truncation"] is False
    assert processor_call["return_tensors"] == "pt"
    assert len(processor_call["images"]) == 1
    assert len(processor_call["images"][0]) == 1
    assert isinstance(processor_call["images"][0][0], Image.Image)
    assert processor.tokenizer.calls == [
        {
            "text": "<|im_start|>assistant",
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
            "return_tensors": "pt",
            "return_offsets_mapping": False,
        },
        {
            "text": [
                "<|im_start|>user\n"
                f"{IMAGE_PAD * 4}question<|im_end|>\n"
                "<|im_start|>assistant\n"
                f"A{LATENT_START}{LATENT_PAD * 2}{LATENT_END}BC<|im_end|>\n"
            ],
            "add_special_tokens": True,
            "padding": True,
            "truncation": False,
            "return_tensors": "pt",
            "return_offsets_mapping": False,
        },
    ]


def test_enabled_collator_maps_expanded_full_string_without_changing_normal_tensors(
    mapped_example,
):
    """Catches fragmented tokenization, double image processing, or processor drift."""
    disabled_processor = SinglePassQwenProcessor()
    enabled_processor = SinglePassQwenProcessor()
    disabled = make_mapped_collator(disabled_processor, enabled=False)([mapped_example])

    enabled = make_mapped_collator(enabled_processor, enabled=True)([mapped_example])

    for key, value in disabled.items():
        assert torch.equal(enabled[key], value), key
    assert enabled_processor.image_processing_calls == 1
    offset_call = enabled_processor.tokenizer.calls[-1]
    assert offset_call["return_offsets_mapping"] is True
    assert isinstance(offset_call["text"], list) and len(offset_call["text"]) == 1
    assert IMAGE_PAD * 4 in offset_call["text"][0]
    assert enabled["sample_ids"] == ("sample-7",)
    assert enabled["reasoning_step_spans"] == (
        ReasoningStepSpan(
            step_id=0,
            prev_text_start=31,
            prev_text_end=32,
            latent_start=33,
            latent_end=35,
            target_text_start=36,
            target_text_end=38,
            helper_image_idx=0,
        ),
    )
    assert (
        enabled["step_preference"]["expanded_rendered_text"]
        == offset_call["text"][0]
    )
    assert (
        enabled["step_preference"]["offset_mapping"].shape
        == enabled["input_ids"].shape + (2,)
    )
