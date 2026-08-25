from __future__ import annotations

from copy import deepcopy

from PIL import Image
import torch

from stage1.data import PreparedStage1Example, Stage1DataCollator
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)


class DeterministicTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_tensors="pt",
    ):
        assert return_tensors == "pt"
        assert text == "<|im_start|>assistant"
        return {"input_ids": torch.tensor([[31, 32]])}


class DeterministicImageProcessor:
    def __init__(self, owner):
        self.owner = owner

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        count = len(images)
        if self.owner.helper_grid_rows is not None:
            count = self.owner.helper_grid_rows
        return {
            "pixel_values": torch.arange(count, dtype=torch.float32).unsqueeze(1),
            "image_grid_thw": torch.ones((count, 3), dtype=torch.long),
        }


class DeterministicStage1Processor:
    """Small processor double whose token rows are derived from conversations."""

    def __init__(self, *, original_grid_rows=None, helper_grid_rows=None, bad_mask=False):
        self.tokenizer = DeterministicTokenizer()
        self.image_processor = DeterministicImageProcessor(self)
        self.original_grid_rows = original_grid_rows
        self.helper_grid_rows = helper_grid_rows
        self.bad_mask = bad_mask
        self._rendered = {}
        self._render_count = 0

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize,
        add_generation_prompt=False,
    ):
        assert tokenize is False
        assert add_generation_prompt is False
        key = f"rendered-{self._render_count}"
        self._render_count += 1
        self._rendered[key] = deepcopy(conversation)
        return key

    @staticmethod
    def _assistant_tokens(conversation):
        tokens = []
        token_map = {"start": 91, "pad": 90, "end": 92, "word": 70}
        for turn in conversation:
            if turn.get("role") != "assistant":
                continue
            for item in turn.get("content", []):
                if item.get("type") == "image":
                    tokens.extend((91, 777, 92))
                elif item.get("type") == "text":
                    text = item.get("text", "")
                    if text.startswith("TOKENS:"):
                        names = [
                            name
                            for name in text.removeprefix("TOKENS:").split(",")
                            if name
                        ]
                        tokens.extend(token_map[name] for name in names)
                    else:
                        tokens.extend([70] * max(1, len(text.split())))
        return tokens

    def __call__(
        self,
        *,
        text,
        images,
        padding,
        truncation,
        return_tensors,
    ):
        assert padding is True
        assert truncation is False
        assert return_tensors == "pt"
        rows = [
            [10, 31, 32, *self._assistant_tokens(self._rendered[key]), 7]
            for key in text
        ]
        width = max(map(len, rows))
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row_index, row in enumerate(rows):
            input_ids[row_index, : len(row)] = torch.tensor(row)
            attention_mask[row_index, : len(row)] = 1
        if self.bad_mask:
            attention_mask = attention_mask[:, :-1]

        grid_rows = sum(len(group) for group in images)
        if self.original_grid_rows is not None:
            grid_rows = self.original_grid_rows
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.arange(grid_rows, dtype=torch.float32).unsqueeze(1),
            "image_grid_thw": torch.ones((grid_rows, 3), dtype=torch.long),
        }


def prepared_example(
    *,
    source_index,
    original_images,
    helper_images,
    blocks,
    source_id=None,
    trailing_words=1,
    tokens=None,
):
    user_content = [
        {"type": "image", "image": Image.new("RGB", (2, 2))}
        for _ in range(original_images)
    ]
    user_content.append({"type": "text", "text": "question"})
    if tokens is None:
        assistant_content = []
        for block in range(blocks):
            assistant_content.append(
                {"type": "image", "image": Image.new("RGB", (2, 2))}
            )
            assistant_content.append({"type": "text", "text": f"step-{block}"})
        assistant_content.append(
            {"type": "text", "text": " ".join(["tail"] * trailing_words)}
        )
    else:
        assistant_content = [{"type": "text", "text": "TOKENS:" + ",".join(tokens)}]

    originals = tuple(
        item["image"] for item in user_content if item["type"] == "image"
    )
    helpers = tuple(Image.new("RGB", (2, 2)) for _ in range(helper_images))
    return PreparedStage1Example(
        source_index=source_index,
        source_id=source_id,
        conversation=[
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        original_image_paths=originals,
        helper_image_paths=helpers,
    )


def make_stage1_collator(**processor_kwargs):
    return Stage1DataCollator(
        processor=DeterministicStage1Processor(**processor_kwargs),
        token_ids=TOKENS,
        latent_size=3,
    )
