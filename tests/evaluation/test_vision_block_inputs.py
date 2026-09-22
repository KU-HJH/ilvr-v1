import copy
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from transformers import AutoProcessor

import evaluation.vision_block_inputs as inputs
from evaluation.vision_block_inputs import (
    PreparedSample,
    Site,
    _sample_identity,
    load_prepared_samples,
    prepare_manifest,
    prepare_sample,
)
from evaluation.vision_block_metrics import IMAGE_CONDITIONS, IMAGE_PROFILE
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


def _geometry_sample(
    *, input_ids=None, attention_mask=None, image_grid_thw=None, sites=None
):
    ids = input_ids
    if ids is None:
        ids = torch.tensor([[10, 20, 21, 21, 22, 30, 91, *([90] * 8), 92, 40]])
    if attention_mask is None:
        attention_mask = torch.ones_like(ids)
    if image_grid_thw is None:
        image_grid_thw = torch.tensor([[1, 1, 2]])
    if sites is None:
        sites = (Site(0, 1, 6, 7, 15, 15, (16,), (40,)),)
    return PreparedSample(
        0,
        {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "image_grid_thw": image_grid_thw,
        },
        sites,
    )


def _geometry_contract():
    return dict(
        image_token_id=21,
        vision_start_token_id=20,
        vision_end_token_id=22,
        merge_size=1,
    )


def test_processor_image_contract_uses_tokenizer_and_image_processor(processor):
    assert inputs.processor_image_contract(processor) == {
        "image_token_id": processor.tokenizer.convert_tokens_to_ids(
            processor.image_token
        ),
        "vision_start_token_id": processor.tokenizer.convert_tokens_to_ids(
            "<|vision_start|>"
        ),
        "vision_end_token_id": processor.tokenizer.convert_tokens_to_ids(
            "<|vision_end|>"
        ),
        "merge_size": processor.image_processor.merge_size,
    }


@pytest.mark.parametrize(
    ("image_id", "start_id", "end_id", "merge_size"),
    [
        (-1, 2, 3, 1),
        (True, 2, 3, 1),
        (1.5, 2, 3, 1),
        (999, 2, 3, 1),
        (1, 1, 3, 1),
        (1, 2, 2, 1),
        (1, 2, 3, 0),
        (1, 2, 3, -1),
        (1, 2, 3, True),
        (1, 2, 3, 1.5),
    ],
)
def test_processor_image_contract_rejects_invalid_values(
    image_id, start_id, end_id, merge_size
):
    token_ids = {
        "image": image_id,
        "<|vision_start|>": start_id,
        "<|vision_end|>": end_id,
    }
    tokenizer = SimpleNamespace(
        unk_token_id=999,
        convert_tokens_to_ids=lambda token: token_ids.get(token, 999),
    )
    candidate = SimpleNamespace(
        image_token="image",
        tokenizer=tokenizer,
        image_processor=SimpleNamespace(merge_size=merge_size),
    )

    with pytest.raises(ValueError):
        inputs.processor_image_contract(candidate)


@pytest.mark.parametrize(
    "candidate",
    [
        SimpleNamespace(
            tokenizer=SimpleNamespace(
                unk_token_id=999, convert_tokens_to_ids=lambda token: 1
            ),
            image_processor=SimpleNamespace(merge_size=1),
        ),
        SimpleNamespace(
            image_token="image",
            tokenizer=SimpleNamespace(
                unk_token_id=999, convert_tokens_to_ids=lambda token: 1
            ),
        ),
    ],
)
def test_processor_image_contract_rejects_missing_values(candidate):
    with pytest.raises(ValueError):
        inputs.processor_image_contract(candidate)


@pytest.mark.parametrize("image_token", ["missing", None, ""])
def test_processor_image_contract_rejects_missing_or_empty_image_token(image_token):
    """Catches accepting a missing processor image token via a permissive tokenizer."""
    tokenizer = SimpleNamespace(
        unk_token_id=999,
        convert_tokens_to_ids=lambda token: {
            "<|vision_start|>": 2,
            "<|vision_end|>": 3,
        }.get(token, 1),
    )
    candidate = SimpleNamespace(
        tokenizer=tokenizer,
        image_processor=SimpleNamespace(merge_size=1),
    )
    if image_token != "missing":
        candidate.image_token = image_token

    with pytest.raises(ValueError):
        inputs.processor_image_contract(candidate)


def test_image_payload_positions_exclude_boundaries():
    sample = _geometry_sample()
    contract = _geometry_contract()

    assert inputs.original_image_positions(sample, **contract) == (2, 3)

    bad_grid = dict(
        sample.model_inputs, image_grid_thw=torch.tensor([[1, 1, 3]])
    )
    with pytest.raises(ValueError):
        inputs.original_image_positions(
            PreparedSample(0, bad_grid, sample.sites), **contract
        )
    for column, value in [(1, 10), (4, 21), (5, 21)]:
        bad = sample.model_inputs["input_ids"].clone()
        bad[0, column] = value
        with pytest.raises(ValueError):
            inputs.original_image_positions(
                PreparedSample(
                    0, dict(sample.model_inputs, input_ids=bad), sample.sites
                ),
                **contract,
            )


@pytest.mark.parametrize(
    "grid",
    [
        torch.tensor([[0, 1, 2]]),
        torch.tensor([[-1, 1, 2]]),
        torch.tensor([[1.5, 1.0, 2.0]]),
        torch.tensor([[True, True, True]]),
        torch.tensor([1, 1, 2]),
        torch.tensor([[1, 1, 1], [1, 1, 1]]),
    ],
)
def test_original_image_positions_rejects_malformed_grid(grid):
    with pytest.raises(ValueError):
        inputs.original_image_positions(
            _geometry_sample(image_grid_thw=grid), **_geometry_contract()
        )


@pytest.mark.parametrize("marker", [20, 22])
def test_original_image_positions_rejects_multiple_markers(marker):
    sample = _geometry_sample()
    bad = sample.model_inputs["input_ids"].clone()
    bad[0, 0 if marker == 20 else 5] = marker

    with pytest.raises(ValueError):
        inputs.original_image_positions(
            PreparedSample(0, dict(sample.model_inputs, input_ids=bad), sample.sites),
            **_geometry_contract(),
        )


def test_original_image_positions_rejects_missing_payload():
    ids = torch.tensor([[10, 20, 22, 30, 91, *([90] * 8), 92, 40]])

    with pytest.raises(ValueError):
        inputs.original_image_positions(
            _geometry_sample(input_ids=ids, image_grid_thw=torch.tensor([[1, 1, 1]])),
            **_geometry_contract(),
        )


def test_original_image_positions_rejects_image_at_or_after_first_latent_start():
    sample = _geometry_sample(
        sites=(Site(0, 1, 3, 4, 12, 12, (13,), (40,)),)
    )

    with pytest.raises(ValueError):
        inputs.original_image_positions(sample, **_geometry_contract())


@pytest.mark.parametrize("merge_size", [0, -1, True, 1.5, 3])
def test_original_image_positions_rejects_invalid_merge(merge_size):
    with pytest.raises(ValueError):
        inputs.original_image_positions(
            _geometry_sample(), **dict(_geometry_contract(), merge_size=merge_size)
        )


@pytest.mark.parametrize(
    "sample",
    [
        _geometry_sample(
            input_ids=torch.tensor(
                [
                    [10, 20, 21, 21, 22, 30, 91, *([90] * 8), 92, 40],
                    [10, 20, 21, 21, 22, 30, 91, *([90] * 8), 92, 40],
                ]
            )
        ),
        _geometry_sample(
            attention_mask=torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1, *([1] * 8), 1, 0]]
            )
        ),
    ],
)
def test_original_image_positions_rejects_padded_or_non_batch_one_inputs(sample):
    with pytest.raises(ValueError):
        inputs.original_image_positions(sample, **_geometry_contract())


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


def test_malformed_latent_tokens_exclude_the_sample_once(tmp_path, processor):
    """Catches a stray latent token escaping as a collator exception."""
    Image.new("RGB", (56, 56)).save(tmp_path / "original.png")
    row = _row(
        [
            {"type": "text", "content": "Before <|latent_pad|>."},
            _latent(),
            {"type": "text", "content": "Reason."},
        ]
    )

    sample, exclusions = prepare_sample(row, 11, processor, tmp_path)

    assert sample is None
    assert exclusions == [
        {
            "source_index": 11,
            "plan_index": None,
            "reason": "sample_latent_structure_invalid",
            "explanation": "latent_pad outside latent block",
        }
    ]


@pytest.mark.parametrize(
    ("image_input", "message"),
    [
        (["missing.png"], "missing original image"),
        (["../outside.png"], "safe relative path"),
    ],
)
def test_operational_image_errors_propagate(tmp_path, processor, image_input, message):
    """Catches structural exclusion handling swallowing image-operation failures."""
    row = _row([{"type": "text", "content": "Before."}, _latent()])
    row["image_input"] = image_input

    with pytest.raises((FileNotFoundError, ValueError), match=message):
        prepare_sample(row, 12, processor, tmp_path)


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


def test_manifest_denominators_count_sites_and_target_tokens(tmp_path, processor, monkeypatch):
    """Catches confusing 100 samples, 101 sites and 203 target tokens."""
    import evaluation.vision_block_inputs as inputs
    from evaluation.vision_block_inputs import PreparedSample, Site

    rows = [{"image_input": [f"{category}/original.png"]}
            for category, count in inputs.EXPECTED_CATEGORY_COUNTS.items()
            for _ in range(count)]
    data = tmp_path / "TEST.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selection": {"requested_indices": list(range(100))}}))
    monkeypatch.setattr(inputs, "_sha256", lambda path: inputs.EXPECTED_DATA_SHA256
                        if path == data else inputs.EXPECTED_SELECTION_SHA256)

    def prepared(row, index, processor, image_root):
        sites = [Site(index, 1, 0, 1, 9, 9, (10, 11), (21, 22))]
        if index == 0:
            sites.append(Site(index, 3, 12, 13, 21, 21, (22, 23, 24), (31, 32, 33)))
        return PreparedSample(index, {}, tuple(sites)), []

    monkeypatch.setattr(inputs, "prepare_sample", prepared)
    manifest = prepare_manifest(data, selection, processor, tmp_path, tmp_path / "manifest.json")
    assert manifest["S"] == 101
    assert manifest["N"] == 203
    assert manifest["analysis_profile"] == "block-local-v2"
    assert manifest["condition_order"] == [
        "clean",
        "body_mask",
        "boundary_mask",
        "full_mask",
    ]


def test_image_profile_manifest_records_contract_and_payload_positions(
    tmp_path, processor, monkeypatch
):
    """Catches image-profile preparation omitting replay-critical geometry."""
    import evaluation.vision_block_inputs as inputs

    rows = [
        {"image_input": [f"{category}/original.png"]}
        for category, count in inputs.EXPECTED_CATEGORY_COUNTS.items()
        for _ in range(count)
    ]
    data = tmp_path / "TEST.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"selection": {"requested_indices": list(range(100))}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        inputs,
        "_sha256",
        lambda path: inputs.EXPECTED_DATA_SHA256
        if path == data
        else inputs.EXPECTED_SELECTION_SHA256,
    )
    contract = inputs.processor_image_contract(processor)
    merge_size = contract["merge_size"]
    input_ids = torch.tensor(
        [
            [
                contract["vision_start_token_id"],
                contract["image_token_id"],
                contract["image_token_id"],
                contract["vision_end_token_id"],
                90,
                90,
                40,
            ]
        ]
    )

    def prepared(row, index, processor, image_root):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "image_grid_thw": torch.tensor([[1, merge_size, 2 * merge_size]]),
        }
        site = Site(index, 1, 4, 5, 5, 5, (6,), (40,))
        return PreparedSample(index, model_inputs, (site,)), []

    monkeypatch.setattr(inputs, "prepare_sample", prepared)

    manifest = prepare_manifest(
        data,
        selection,
        processor,
        tmp_path,
        tmp_path / "manifest.json",
        analysis_profile=IMAGE_PROFILE,
    )

    assert manifest["analysis_profile"] == IMAGE_PROFILE
    assert manifest["condition_order"] == list(IMAGE_CONDITIONS)
    assert manifest["image_token_contract"] == contract
    assert all(
        record["original_image_positions"] == [1, 2]
        for record in manifest["samples"]
    )


def _real_image_replay_manifest(tmp_path, processor, monkeypatch):
    image_dir = tmp_path / "creation"
    image_dir.mkdir()
    Image.new("RGB", (56, 56)).save(image_dir / "original.png")
    row = _row(
        [
            {"type": "text", "content": "Before."},
            _latent(),
            {"type": "text", "content": "Reason."},
        ]
    )
    row["image_input"] = ["creation/original.png"]
    sample, exclusions = prepare_sample(row, 0, processor, tmp_path)
    assert sample is not None and not exclusions

    data = tmp_path / "TEST.jsonl"
    data.write_text(json.dumps(row) + "\n", encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"selection": {"requested_indices": [0]}}), encoding="utf-8"
    )
    monkeypatch.setattr(inputs, "EXPECTED_DATA_SHA256", inputs._sha256(data))
    monkeypatch.setattr(
        inputs, "EXPECTED_SELECTION_SHA256", inputs._sha256(selection)
    )
    monkeypatch.setattr(inputs, "EXPECTED_CATEGORY_COUNTS", {"creation": 1})
    monkeypatch.setattr(inputs, "_validate_selection", lambda indices, row_count: (0,))
    manifest = prepare_manifest(
        data,
        selection,
        processor,
        tmp_path,
        tmp_path / "manifest.json",
        analysis_profile=IMAGE_PROFILE,
    )
    assert manifest["samples"][0]["identity"] == _sample_identity(sample)
    assert tuple(
        inputs._load_site_record(site) for site in manifest["samples"][0]["sites"]
    ) == sample.sites
    return sample, manifest


def test_image_profile_loader_round_trip_preserves_identity_and_sites(
    tmp_path, processor, monkeypatch
):
    """Catches image replay changing the pre-existing sample identity contract."""
    sample, manifest = _real_image_replay_manifest(tmp_path, processor, monkeypatch)

    loaded = list(load_prepared_samples(manifest, processor, tmp_path))

    assert len(loaded) == 1
    assert _sample_identity(loaded[0]) == _sample_identity(sample)
    assert loaded[0].sites == sample.sites

    legacy = copy.deepcopy(manifest)
    legacy.pop("analysis_profile")
    legacy.pop("condition_order")
    legacy.pop("image_token_contract")
    legacy["samples"][0].pop("original_image_positions")
    legacy_loaded = list(load_prepared_samples(legacy, processor, tmp_path))
    assert _sample_identity(legacy_loaded[0]) == _sample_identity(sample)
    assert legacy_loaded[0].sites == sample.sites


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "moved", "bool", "string"],
)
def test_image_profile_loader_rejects_malformed_persisted_positions(
    tmp_path, processor, monkeypatch, mutation
):
    """Catches trusting a persisted image list despite unchanged input hashes."""
    _, manifest = _real_image_replay_manifest(tmp_path, processor, monkeypatch)
    record = manifest["samples"][0]
    positions = record["original_image_positions"]
    if mutation == "missing":
        record.pop("original_image_positions")
    elif mutation == "duplicate":
        record["original_image_positions"] = [positions[0], positions[0], *positions[2:]]
    elif mutation == "moved":
        record["original_image_positions"] = [positions[0] + 1, *positions[1:]]
    elif mutation == "bool":
        record["original_image_positions"] = [True, *positions[1:]]
    else:
        record["original_image_positions"] = [str(positions[0]), *positions[1:]]

    with pytest.raises(ValueError, match="image"):
        list(load_prepared_samples(manifest, processor, tmp_path))


def test_image_profile_loader_rejects_changed_processor_contract(
    tmp_path, processor, monkeypatch
):
    """Catches replaying image positions under a different token contract."""
    _, manifest = _real_image_replay_manifest(tmp_path, processor, monkeypatch)
    manifest["image_token_contract"]["image_token_id"] += 1

    with pytest.raises(ValueError, match="image"):
        list(load_prepared_samples(manifest, processor, tmp_path))


@pytest.mark.parametrize("profile", [None, "block-local-v2"])
def test_loader_rejects_image_metadata_under_legacy_or_default_profile(
    tmp_path, processor, monkeypatch, profile
):
    """Catches treating image-profile replay metadata as a legacy manifest."""
    _, manifest = _real_image_replay_manifest(tmp_path, processor, monkeypatch)
    if profile is None:
        manifest.pop("analysis_profile")
        manifest.pop("condition_order")
    else:
        manifest["analysis_profile"] = profile
        manifest["condition_order"] = [
            "clean",
            "body_mask",
            "boundary_mask",
            "full_mask",
        ]

    with pytest.raises(ValueError, match="image.*profile"):
        list(load_prepared_samples(manifest, processor, tmp_path))
