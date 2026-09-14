import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from stage2.data import Stage2DataCollator, Stage2PreparedExample
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)


def _windows():
    module = importlib.import_module("stage2.objective_windows")
    required = ("build_text_vocabulary", "select_objective_windows", "TextVocabulary")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    assert not missing, f"missing window callables: {missing}"
    return module


def test_objective_windows_api_exists():
    try:
        module = importlib.import_module("stage2.objective_windows")
    except ModuleNotFoundError:
        module = None
    assert module is not None, "stage2.objective_windows must implement selection"


class VocabularyTokenizer:
    def __init__(self):
        self._vocab = {
            "text": 0,
            "ordinary_added": 1,
            "<|im_start|>": 2,
            "<tool_call>": 3,
            "<|fim_prefix|>": 4,
            "<|image_pad|>": 5,
            "<|latent_pad|>": 6,
            "[PAD]": 7,
            "<|latent_start|>": 8,
            "<|latent_end|>": 9,
        }
        self.all_special_ids = [2, 5, 6]
        self.pad_token_id = 7

    def get_vocab(self):
        return dict(self._vocab)


def test_text_vocabulary_excludes_controls_structural_and_unmapped_rows():
    module = _windows()
    tokenizer = VocabularyTokenizer()
    config = SimpleNamespace(
        image_token_id=5,
        latent_token_id=6,
        latent_pad_token_id=6,
        latent_start_token_id=8,
        latent_end_token_id=9,
    )

    vocabulary = module.build_text_vocabulary(tokenizer, config, output_size=11)

    assert vocabulary.token_ids == (0, 1)
    assert vocabulary.excluded == {
        2: "special_token",
        3: "structural_token:<tool_call>",
        4: "structural_token:<|fim_prefix|>",
        5: "special_token",
        6: "special_token",
        7: "padding_token",
        8: "structural_token:<|latent_start|>",
        9: "structural_token:<|latent_end|>",
        10: "unmapped_output_row",
    }
    assert vocabulary.identity == {
        "output_size": 11,
        "token_ids": [0, 1],
        "excluded": [
            {"token_id": token_id, "reason": reason}
            for token_id, reason in sorted(vocabulary.excluded.items())
        ],
    }
    json.dumps(vocabulary.identity)


@pytest.mark.parametrize(
    "tokenizer, config, output_size, message",
    [
        (VocabularyTokenizer(), SimpleNamespace(image_token_id=12), 11, "output_size"),
        (VocabularyTokenizer(), SimpleNamespace(image_token_id="bad"), 11, "image_token_id"),
        (VocabularyTokenizer(), SimpleNamespace(image_token_id=10), 11, "cannot resolve"),
        (VocabularyTokenizer(), SimpleNamespace(), 5, "output_size"),
    ],
)
def test_text_vocabulary_rejects_unresolved_id_mismatches(
    tokenizer, config, output_size, message
):
    with pytest.raises(ValueError, match=message):
        _windows().build_text_vocabulary(tokenizer, config, output_size=output_size)


def _text_vocabulary(*ids):
    return _windows().TextVocabulary(tuple(ids), {}, 100)


def _select(ids, labels, segments, *, attention=None, pre=1, post=1, latent_size=2):
    input_ids = torch.tensor(ids, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    if attention is None:
        attention = torch.ones_like(input_ids)
    return _windows().select_objective_windows(
        input_ids,
        label_tensor,
        attention,
        segments,
        token_ids=TOKENS,
        latent_size=latent_size,
        text_vocabulary=_text_vocabulary(*range(90)),
        pre_window=pre,
        post_window=post,
    )


def test_eight_token_literal_alignment_uses_boundary_state_and_post_target():
    selection = _select(
        [[10, 11, 91, 90, 90, 92, 12, 13]],
        [[10, 11, 91, -100, -100, 92, 12, 13]],
        [[(0, 2), (6, 8)]],
    )

    assert selection.excluded == {}
    assert len(selection.blocks) == 1
    block = selection.blocks[0]
    assert (block.batch_index, block.block_index) == (0, 0)
    assert (block.start_position, block.end_position) == (2, 5)
    assert block.pre_positions == (1,)
    assert block.post_positions == (5,)
    assert block.target_ids == (12,)


def test_windows_support_batched_left_and_right_padding_without_crossing_attention():
    ids = [
        [99, 10, 11, 91, 90, 90, 92, 12, 13],
        [10, 11, 91, 90, 90, 92, 12, 13, 99],
    ]
    labels = [
        [-100, 10, 11, 91, -100, -100, 92, 12, 13],
        [10, 11, 91, -100, -100, 92, 12, 13, -100],
    ]
    attention = torch.tensor(
        [[0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0]]
    )

    selection = _select(
        ids,
        labels,
        [[(1, 3), (7, 9)], [(0, 2), (6, 8)]],
        attention=attention,
        pre=2,
        post=2,
    )

    assert [block.pre_positions for block in selection.blocks] == [(1, 2), (0, 1)]
    assert [block.post_positions for block in selection.blocks] == [(6, 7), (5, 6)]
    assert [block.target_ids for block in selection.blocks] == [(12, 13), (12, 13)]


def test_overlap_positions_are_scoped_to_each_batch_row():
    selection = _select(
        [
            [10, 11, 12, 13, 14, 91, 90, 90, 92, 15],
            [10, 91, 90, 90, 92, 11, 12, 13, 14, 99],
        ],
        [
            [10, 11, 12, 13, 14, 91, -100, -100, 92, 15],
            [10, 91, -100, -100, 92, 11, 12, 13, 14, -100],
        ],
        [[(0, 5), (9, 10)], [(0, 1), (5, 9)]],
        pre=1,
        post=1,
    )

    assert len(selection.blocks) == 2
    assert selection.blocks[0].pre_positions == (4,)
    assert selection.blocks[1].post_positions == (4,)


def test_pre_window_uses_text_state_even_though_next_label_is_marker():
    selection = _select(
        [[10, 91, 90, 90, 92, 11]],
        [[10, 91, -100, -100, 92, 11]],
        [[(0, 1), (5, 6)]],
    )
    assert selection.blocks[0].pre_positions == (0,)


def test_overlap_is_removed_from_both_sides_simultaneously_without_refill():
    selection = _select(
        [[10, 91, 90, 90, 92, 11, 12, 91, 90, 90, 92, 13]],
        [[10, 91, -100, -100, 92, 11, 12, 91, -100, -100, 92, 13]],
        [[(0, 1), (5, 7), (11, 12)]],
        pre=2,
        post=2,
    )

    assert len(selection.blocks) == 2
    assert selection.blocks[0].post_positions == (4,)
    assert selection.blocks[1].pre_positions == (6,)
    assert selection.blocks[0].target_ids == (11,)


def test_incomplete_block_still_participates_in_simultaneous_overlap_pass():
    selection = _select(
        [[91, 90, 90, 92, 10, 11, 91, 90, 90, 92, 12]],
        [[91, -100, -100, 92, 10, 11, 91, -100, -100, 92, 12]],
        [[(0, 0), (4, 6), (10, 11)]],
        pre=2,
        post=3,
    )

    assert len(selection.blocks) == 1
    assert selection.blocks[0].block_index == 1
    assert selection.blocks[0].pre_positions == (5,)
    assert selection.excluded == {"empty_pre": 1}


def test_empty_short_and_nontext_segments_are_excluded_with_reasons():
    selection = _select(
        [[91, 90, 90, 92, 10, 91, 90, 90, 92, 93]],
        [[91, -100, -100, 92, 10, 91, -100, -100, 92, 93]],
        [[(0, 0), (4, 5), (9, 10)]],
    )

    assert selection.blocks == ()
    assert selection.excluded == {"empty_pre": 1, "empty_post": 1}


@pytest.mark.parametrize(
    "ids, labels, segments, message",
    [
        ([[10, 91, 90]], [[10, 91, -100]], [[(0, 1), (3, 3)]], "batch_index=0"),
        (
            [[10, 91, 90, 12, 92, 13]],
            [[10, 91, -100, 12, 92, 13]],
            [[(0, 1), (5, 6)]],
            "batch_index=0",
        ),
        (
            [[10, 91, 90, 90, 92, 13]],
            [[10, 91, -100, -100, 92, 13]],
            [[(0, 2), (5, 6)]],
            "text_segments",
        ),
    ],
)
def test_windows_reject_malformed_markers_and_segment_boundaries(
    ids, labels, segments, message
):
    with pytest.raises(ValueError, match=message):
        _select(ids, labels, segments)


def test_windows_return_zero_blocks_when_no_latent_blocks_exist():
    selection = _select([[10, 11]], [[10, 11]], [[(0, 2)]])
    assert selection.blocks == ()
    assert selection.excluded == {}


def _fast_tokenizer():
    vocab = {
        "[UNK]": 0,
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "system": 3,
        "user": 4,
        "assistant": 5,
        "question": 6,
        "before": 7,
        "after": 8,
        "\n": 9,
        "<|vision_start|>": 10,
        "<|image_pad|>": 11,
        "<|vision_end|>": 12,
        "<|latent_start|>": 91,
        "<|latent_pad|>": 90,
        "<|latent_end|>": 92,
    }
    vocab.update({f"unused_{token_id}": token_id for token_id in range(13, 90)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                AddedToken(token, normalized=False, special=True)
                for token in (
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|vision_start|>",
                    "<|image_pad|>",
                    "<|vision_end|>",
                    "<|latent_start|>",
                    "<|latent_pad|>",
                    "<|latent_end|>",
                )
            ]
        }
    )
    return tokenizer


class SelfContainedProcessor:
    def __init__(self):
        self.tokenizer = _fast_tokenizer()

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        assistant = conversation[-1]["content"][0]["text"]
        return (
            "<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|> question"
            "<|im_end|>\n<|im_start|>assistant\n"
            f"{assistant}<|im_end|>\n"
        )

    def __call__(self, **kwargs):
        encoded = self.tokenizer(
            kwargs["text"], add_special_tokens=False, return_tensors="pt"
        )
        ids = encoded["input_ids"][0].tolist()
        image_position = ids.index(11)
        ids[image_position : image_position + 1] = [11, 11, 11]
        input_ids = torch.tensor([ids])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.ones((1, 3, 2, 2)),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }


def _prepared(assistant):
    return Stage2PreparedExample(
        conversation=[
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant}]},
        ],
        original_image_paths=(),
        latent_block_count=1,
    )


def test_collator_feature_off_preserves_legacy_output_exactly():
    processor = SelfContainedProcessor()
    example = _prepared(
        "before<|latent_start|><|latent_pad|><|latent_pad|><|latent_end|>after"
    )
    legacy = Stage2DataCollator(processor, TOKENS, 2, True)([example])
    explicit_off = Stage2DataCollator(
        processor, TOKENS, 2, True, enable_stage2_objective=False
    )([example])

    assert legacy.keys() == explicit_off.keys()
    assert "stage2_text_segments" not in legacy
    for key in legacy:
        torch.testing.assert_close(legacy[key], explicit_off[key])


def test_collator_feature_on_emits_verified_segments_after_image_expansion():
    processor = SelfContainedProcessor()
    example = _prepared(
        "before<|latent_start|><|latent_pad|><|latent_pad|><|latent_end|>after"
    )
    legacy_batch = Stage2DataCollator(processor, TOKENS, 2, True)([example])
    batch = Stage2DataCollator(
        processor, TOKENS, 2, True, enable_stage2_objective=True
    )([example])

    segments = batch.pop("stage2_text_segments")
    assert len(segments) == 1
    assert len(segments[0]) == 2
    (pre_start, pre_end), (post_start, post_end) = segments[0]
    assert batch["input_ids"][0, pre_start:pre_end].tolist() == [7]
    assert batch["input_ids"][0, post_start:post_end].tolist() == [8]
    assert batch["labels"][0, pre_start].item() == 7
    assert batch["labels"][0, post_start].item() == 8
    assert batch.keys() == legacy_batch.keys()
    for key in batch:
        torch.testing.assert_close(batch[key], legacy_batch[key])


def test_collator_leading_template_newline_does_not_create_pre_text_window():
    processor = SelfContainedProcessor()
    example = _prepared(
        "<|latent_start|><|latent_pad|><|latent_pad|><|latent_end|>after"
    )
    batch = Stage2DataCollator(
        processor, TOKENS, 2, True, enable_stage2_objective=True
    )([example])

    assert batch["stage2_text_segments"][0][0][0] == batch["stage2_text_segments"][0][0][1]
    selection = _windows().select_objective_windows(
        batch["input_ids"],
        batch["labels"],
        batch["attention_mask"],
        batch["stage2_text_segments"],
        token_ids=TOKENS,
        latent_size=2,
        text_vocabulary=_text_vocabulary(*range(90)),
        pre_window=1,
        post_window=1,
    )
    assert selection.blocks == ()
    assert selection.excluded == {"empty_pre": 1}


def test_collator_rejects_batch_size_greater_than_one_with_objective_enabled():
    processor = SelfContainedProcessor()
    example = _prepared("before<|latent_start|><|latent_pad|><|latent_pad|><|latent_end|>after")
    collator = Stage2DataCollator(
        processor, TOKENS, 2, True, enable_stage2_objective=True
    )
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        collator([example, example])
