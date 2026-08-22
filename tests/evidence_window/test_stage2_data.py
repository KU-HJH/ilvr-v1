from stage2_data import build_stage2_record, collate_fn_stage2


def test_record_preserves_plan_without_checking_helper(monkeypatch, raw_sample):
    """Restoring helper existence checks would reject an inert Stage-2 placeholder."""
    monkeypatch.setattr(
        "task_deepseed._assert_exists",
        lambda path: (_ for _ in ()).throw(AssertionError(path)),
    )

    record = build_stage2_record(raw_sample, 17, validate_user_images=False)

    assert record["ew_sample_index"] == 17
    assert record["ew_plan_types"] == ("text", "latent", "text")


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
