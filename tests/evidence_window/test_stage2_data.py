import pytest
import torch

from evidence_window.layout import TokenIds
from evidence_window.stage2_layout import extract_stage2_layout, validate_plan_types
from stage2_data import build_stage2_record, collate_fn_stage2


TOKENS = TokenIds(90, 91, 92, 0, frozenset({0, 90, 91, 92, 99}))


def test_record_preserves_plan_without_checking_helper(monkeypatch, raw_sample):
    """Restoring helper existence checks would reject an inert Stage-2 placeholder."""
    monkeypatch.setattr(
        "task_deepseed._assert_exists",
        lambda path: (_ for _ in ()).throw(AssertionError(path)),
    )

    record = build_stage2_record(raw_sample, 17, validate_user_images=False)

    assert record["ew_sample_index"] == 17
    assert record["ew_plan_types"] == ("text", "latent", "text")


def test_record_collapses_adjacent_text_steps_to_token_semantic_window_runs():
    """Separate source text steps without a latent boundary form one parsed R window."""
    sample = {
        "image_input": [],
        "text_input": "question",
        "sequence_plan": [
            {"type": "text", "content": "before"},
            {"type": "latent", "helper_image": "unused.png"},
            {"type": "text", "content": "reasoning"},
            {"type": "text", "content": "final answer"},
        ],
    }

    record = build_stage2_record(sample, 920, validate_user_images=False)

    assert record["ew_plan_types"] == ("text", "latent", "text")
    ids = torch.tensor([[10, 90, 92, 91, 20, 21]])
    layout = extract_stage2_layout(ids, torch.ones_like(ids), ids.clone(), TOKENS)
    validate_plan_types(layout, (record["ew_plan_types"],), [920])


def test_record_does_not_normalize_an_unknown_plan_type():
    """Metadata normalization must not hide an invalid source sequence-plan step."""
    sample = {
        "image_input": [],
        "text_input": "question",
        "sequence_plan": [{"type": "audio", "content": "unsupported"}],
    }

    with pytest.raises(ValueError, match="Unknown step type: audio"):
        build_stage2_record(sample, 1, validate_user_images=False)


def test_collator_loads_only_user_images(
    monkeypatch,
    stage2_processor,
    stage2_record,
    stage2_args,
):
    """Passing assistant placeholders to vision decoding would fail this boundary check."""
    vision_calls = []

    def user_images_only(conversations):
        vision_calls.append(conversations)
        assert all(
            item["type"] != "image"
            for conversation in conversations
            for turn in conversation
            if turn["role"] == "assistant"
            for item in turn["content"]
        )
        return [["decoded-user-image"]], None

    monkeypatch.setattr("stage2_data.process_vision_info", user_images_only)

    batch = collate_fn_stage2([stage2_record], stage2_processor, stage2_args())

    assert len(vision_calls) == 1
    assert stage2_processor.image_inputs == [["decoded-user-image"]]
    for forbidden in (
        "pixel_values_latent",
        "image_grid_thw_latent",
        "latent_hidden_states",
        "image_out_mask",
    ):
        assert forbidden not in batch
    assert batch["ew_sample_indices"].tolist() == [17]
    assert batch["ew_plan_types"] == (("text", "latent", "text"),)


def test_collator_hides_latent_placeholders_from_qwen_image_expansion(
    monkeypatch,
    stage2_processor,
    stage2_record,
    stage2_args,
):
    """A latent placeholder must not consume a user-image grid row in Qwen processing."""
    monkeypatch.setattr(
        "stage2_data.process_vision_info",
        lambda conversations: ([["decoded-user-image"]], None),
    )

    collate_fn_stage2([stage2_record], stage2_processor, stage2_args())

    assert stage2_processor.text_inputs is not None
    rendered = stage2_processor.text_inputs[0]
    assert rendered.count("<|image_pad|>") == 1
    assert "<|latent_start|><|latent_pad|><|latent_end|>" in rendered
