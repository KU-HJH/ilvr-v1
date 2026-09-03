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

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        assistant = conversation[1]["content"][0]["text"]
        return (
            "<|im_start|>user\n<|image_pad|>question<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
        )

    def __call__(self, *, text, images, padding, truncation, return_tensors):
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


def test_disabled_collator_keeps_the_prechange_keys_and_tensors(mapped_example):
    """Catches opt-in metadata or offset tokenization leaking into Stage 2A batches."""
    processor = SinglePassQwenProcessor()

    batch = make_mapped_collator(processor, enabled=False)([mapped_example])

    assert set(batch) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "labels",
    }
    assert processor.image_processing_calls == 1
    assert all(not call["return_offsets_mapping"] for call in processor.tokenizer.calls)
    assert torch.equal(batch["pixel_values"], torch.arange(12).reshape(1, 3, 2, 2))


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
