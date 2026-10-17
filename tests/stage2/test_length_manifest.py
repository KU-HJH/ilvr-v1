import hashlib
import json
import os
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from stage2.length_manifest import (
    Stage2LengthManifest,
    build_stage2_length_manifest,
    main as build_length_manifest_main,
    read_stage2_length_manifest,
    validate_stage2_length_manifest,
    write_stage2_length_manifest,
)
from stage2.data import summarize_stage2_lengths
from step_preference.spans import (
    ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    ASSISTANT_PLAN_RENDERING_CONCATENATED,
)


class StableToken:
    def __init__(self, content):
        self.content = content

    def __str__(self):
        return self.content


class ManifestTokenizer:
    def __init__(self):
        self.init_kwargs = {"padding_side": "right", "added": StableToken("<added>")}
        self.special_tokens_map = {
            "eos_token": "<eos>",
            "additional_special_tokens": [StableToken("<latent>")],
        }
        self.special_tokens_map_extended = dict(self.special_tokens_map)
        self.chat_template = "tokenizer template"

    def get_vocab(self):
        return {"z": 2, "<eos>": 0, "a": 1}


class ManifestImageProcessor:
    def to_dict(self):
        return {"size": {"shortest_edge": 4}, "resample": 3}


class ManifestProcessor:
    def __init__(self):
        self.image_processor = ManifestImageProcessor()
        self.tokenizer = ManifestTokenizer()
        self.chat_template = "processor template"
        self.measure_calls = 0

    def to_dict(self):
        return {"processor_setting": StableToken("normalized")}

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        return json.dumps(conversation, sort_keys=True)

    def __call__(self, **kwargs):
        assert kwargs["padding"] is False
        assert kwargs["truncation"] is False
        assert all(isinstance(image, Image.Image) for image in kwargs["images"][0])
        self.measure_calls += 1
        length = 10 if "long answer" in kwargs["text"][0] else 8
        return {"input_ids": torch.zeros((1, length), dtype=torch.long)}


class ImageMutatingProcessor(ManifestProcessor):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        Image.new("RGB", (7, 9), color="green").save(self.image_path)
        return result


@pytest.fixture
def processor():
    return ManifestProcessor()


@pytest.fixture
def manifest_inputs(tmp_path):
    Image.new("RGB", (2, 3), color="red").save(tmp_path / "first.png")
    Image.new("RGB", (4, 5), color="blue").save(tmp_path / "second.png")
    rows = [
        {
            "image_input": "first.png",
            "text_input": "first question",
            "sequence_plan": [{"type": "text", "content": "short answer"}],
        },
        {
            "image_input": "second.png",
            "text_input": "second question",
            "sequence_plan": [
                {"type": "latent", "helper_image": "unused.png"},
                {"type": "text", "content": "long answer"},
            ],
        },
    ]
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return data_path, rows


def build_manifest(
    manifest_inputs,
    processor,
    *,
    max_sequence_length=8,
    assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_CONCATENATED,
):
    data_path, _ = manifest_inputs
    return build_stage2_length_manifest(
        data_path=data_path,
        image_root=data_path.parent,
        processor=processor,
        processor_source="/models/stage1",
        latent_size=2,
        supervise_latent_boundaries=True,
        max_sequence_length=max_sequence_length,
        assistant_plan_rendering=assistant_plan_rendering,
    )


def test_manifest_selects_exact_raw_indices_and_round_trips(
    tmp_path, manifest_inputs, processor
):
    """Catches post-filter indexing, repeated measurement, or lossy JSON decoding."""
    manifest = build_manifest(manifest_inputs, processor)

    assert manifest.schema_version == 2
    assert manifest.assistant_plan_rendering == ASSISTANT_PLAN_RENDERING_CONCATENATED
    assert manifest.lengths == (8, 10)
    assert manifest.latent_block_counts == (0, 1)
    assert manifest.retained_indices == (0,)
    assert processor.measure_calls == 2
    assert [(image.width, image.height) for image in manifest.images] == [(2, 3), (4, 5)]
    assert (manifest.stats.total, manifest.stats.retained, manifest.stats.filtered) == (2, 1, 1)

    path = write_stage2_length_manifest(tmp_path / "manifest.json", manifest)
    first_bytes = path.read_bytes()
    assert read_stage2_length_manifest(path) == manifest
    assert write_stage2_length_manifest(path, manifest) == path
    assert path.read_bytes() == first_bytes


def test_manifest_cli_records_the_requested_assistant_rendering(
    tmp_path, manifest_inputs, monkeypatch
):
    """Catches the standalone audit dropping its rendering identity."""
    data_path, _ = manifest_inputs
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        lambda _source: ManifestProcessor(),
    )
    path = tmp_path / "cli-manifest.json"

    assert build_length_manifest_main(
        [
            "--data-path",
            str(data_path),
            "--image-root",
            str(data_path.parent),
            "--processor-source",
            "/models/stage2a",
            "--latent-size",
            "2",
            "--max-sequence-length",
            "8",
            "--assistant-plan-rendering",
            ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
            "--manifest-path",
            str(path),
        ]
    ) == 0
    assert (
        read_stage2_length_manifest(path).assistant_plan_rendering
        == ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF
    )


def test_manifest_validation_rejects_a_different_assistant_rendering(
    tmp_path, manifest_inputs
):
    """Catches Stage 2B consuming lengths audited from legacy assistant bytes."""
    data_path, _ = manifest_inputs
    processor = ManifestProcessor()
    manifest = build_manifest(manifest_inputs, processor)
    path = write_stage2_length_manifest(tmp_path / "manifest.json", manifest)

    with pytest.raises(ValueError, match="configuration does not match"):
        validate_stage2_length_manifest(
            path=path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            data_path=data_path,
            image_root=data_path.parent,
            processor=processor,
            latent_size=2,
            supervise_latent_boundaries=True,
            max_sequence_length=8,
            assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
        )


def test_manifest_consumption_rejects_missing_or_changed_dataset(tmp_path, manifest_inputs):
    """Catches ranks accepting an absent manifest or training bytes outside its digest."""
    data_path, _ = manifest_inputs
    manifest = build_manifest(manifest_inputs, ManifestProcessor())
    path = write_stage2_length_manifest(tmp_path / "manifest.json", manifest)
    original_bytes = data_path.read_bytes()

    with pytest.raises(ValueError, match="regular file"):
        validate_stage2_length_manifest(
            path=tmp_path / "missing.json",
            expected_sha256="0" * 64,
            data_path=data_path,
            image_root=data_path.parent,
            processor=ManifestProcessor(),
            latent_size=2,
            supervise_latent_boundaries=True,
            max_sequence_length=8,
        )

    data_path.write_bytes(original_bytes + b"\n")
    with pytest.raises(ValueError, match="data SHA-256"):
        validate_stage2_length_manifest(
            path=path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            data_path=data_path,
            image_root=data_path.parent,
            processor=ManifestProcessor(),
            latent_size=2,
            supervise_latent_boundaries=True,
            max_sequence_length=8,
        )


def test_manifest_consumption_rejects_processor_and_image_changes(
    tmp_path, manifest_inputs
):
    """Catches stale processor or image identities reaching rank-local preparation."""
    data_path, _ = manifest_inputs
    manifest = build_manifest(manifest_inputs, ManifestProcessor())
    path = write_stage2_length_manifest(tmp_path / "manifest.json", manifest)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    changed_processor = ManifestProcessor()
    changed_processor.chat_template = "different"

    with pytest.raises(ValueError, match="processor identity"):
        validate_stage2_length_manifest(
            path=path,
            expected_sha256=digest,
            data_path=data_path,
            image_root=data_path.parent,
            processor=changed_processor,
            latent_size=2,
            supervise_latent_boundaries=True,
            max_sequence_length=8,
        )

    Image.new("RGB", (7, 9), color="green").save(data_path.parent / "first.png")
    with pytest.raises(ValueError, match="image identities"):
        validate_stage2_length_manifest(
            path=path,
            expected_sha256=digest,
            data_path=data_path,
            image_root=data_path.parent,
            processor=ManifestProcessor(),
            latent_size=2,
            supervise_latent_boundaries=True,
            max_sequence_length=8,
        )


def test_manifest_consumption_returns_the_hashed_data_snapshot_despite_dataset_toctou(
    tmp_path, manifest_inputs
):
    """Catches re-opening the JSONL after validating a digest from different bytes."""
    data_path, rows = manifest_inputs
    processor = ManifestProcessor()
    manifest = build_manifest(manifest_inputs, processor)
    path = write_stage2_length_manifest(tmp_path / "manifest.json", manifest)

    snapshot = validate_stage2_length_manifest(
        path=path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        data_path=data_path,
        image_root=data_path.parent,
        processor=processor,
        latent_size=2,
        supervise_latent_boundaries=True,
        max_sequence_length=8,
    )

    data_path.write_text('{"changed": true}\n', encoding="utf-8")

    assert snapshot.raw_examples == tuple(rows)
    assert len(snapshot.prepared_examples) == len(rows)


def test_manifest_consumption_rejects_dataset_cardinality_mismatch(
    tmp_path, manifest_inputs
):
    """Catches a manifest whose internally valid length list names fewer raw rows."""
    data_path, _ = manifest_inputs
    processor = ManifestProcessor()
    original = build_manifest(manifest_inputs, processor)
    manifest = replace(
        original,
        lengths=(original.lengths[0],),
        latent_block_counts=(original.latent_block_counts[0],),
        retained_indices=(0,),
        stats=summarize_stage2_lengths(
            lengths=(original.lengths[0],),
            latent_block_counts=(original.latent_block_counts[0],),
            max_length=original.max_sequence_length,
        ),
    )
    path = write_stage2_length_manifest(tmp_path / "short-manifest.json", manifest)

    with pytest.raises(ValueError, match="cardinality"):
        validate_stage2_length_manifest(
            path=path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            data_path=data_path,
            image_root=data_path.parent,
            processor=processor,
            latent_size=2,
            supervise_latent_boundaries=True,
            max_sequence_length=8,
        )


def test_manifest_dataset_digest_binds_raw_order(manifest_inputs):
    """Catches hashing parsed records after losing byte order."""
    data_path, rows = manifest_inputs
    original = build_manifest(manifest_inputs, ManifestProcessor())
    data_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reversed(rows)),
        encoding="utf-8",
    )

    reordered = build_manifest(manifest_inputs, ManifestProcessor())

    assert reordered.data_sha256 != original.data_sha256


def test_manifest_image_identity_binds_dimensions_and_mtime(manifest_inputs):
    """Catches stale image metadata being omitted from manifest identity."""
    data_path, _ = manifest_inputs
    original = build_manifest(manifest_inputs, ManifestProcessor())

    first_path = data_path.parent / "first.png"
    Image.new("RGB", (3, 2), color="red").save(first_path)
    resized = build_manifest(manifest_inputs, ManifestProcessor())
    assert resized.images[0] != original.images[0]
    assert (resized.images[0].width, resized.images[0].height) == (3, 2)

    second_path = data_path.parent / "second.png"
    previous = second_path.stat()
    os.utime(
        second_path,
        ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000),
    )
    touched = build_manifest(manifest_inputs, ManifestProcessor())
    assert touched.images[1].mtime_ns != resized.images[1].mtime_ns


def test_processor_digest_is_deterministic_behavior_identity(manifest_inputs):
    """Catches source paths or address-bearing token reprs entering the digest."""
    data_path, _ = manifest_inputs
    first = build_manifest(manifest_inputs, ManifestProcessor())
    second = build_stage2_length_manifest(
        data_path=data_path,
        image_root=data_path.parent,
        processor=ManifestProcessor(),
        processor_source="/different/location/with-equivalent-processor",
        latent_size=2,
        supervise_latent_boundaries=True,
        max_sequence_length=8,
    )
    assert first.processor_sha256 == second.processor_sha256

    changed_processor = ManifestProcessor()
    changed_processor.chat_template = "changed template"
    changed = build_stage2_length_manifest(
        data_path=data_path,
        image_root=data_path.parent,
        processor=changed_processor,
        processor_source="/models/stage1",
        latent_size=2,
        supervise_latent_boundaries=True,
        max_sequence_length=8,
    )
    assert changed.processor_sha256 != first.processor_sha256


def _write_mutated_manifest(tmp_path, manifest, mutate):
    payload = asdict(manifest)
    mutate(payload)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda payload: payload.update(unknown=True), "unknown fields"),
        (lambda payload: payload.update(schema_version=1), "unsupported schema_version"),
        (lambda payload: payload.update(latent_size=True), "latent_size"),
        (lambda payload: payload.update(lengths=[8]), "same cardinality"),
        (lambda payload: payload.update(retained_indices=[1]), "retained_indices"),
    ],
)
def test_manifest_reader_rejects_invalid_schema(
    tmp_path, manifest_inputs, processor, mutate, message
):
    """Catches accepting metadata that cannot select the exact audited rows."""
    manifest = build_manifest(manifest_inputs, processor)
    path = _write_mutated_manifest(tmp_path, manifest, mutate)

    with pytest.raises(ValueError, match=message):
        read_stage2_length_manifest(path)


def test_manifest_reader_rejects_symlinks_and_non_regular_files(
    tmp_path, manifest_inputs, processor
):
    """Catches manifest publication being redirected or read from special files."""
    manifest = build_manifest(manifest_inputs, processor)
    regular = write_stage2_length_manifest(tmp_path / "regular.json", manifest)
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(regular)

    with pytest.raises(ValueError, match="regular file"):
        read_stage2_length_manifest(symlink)
    with pytest.raises(ValueError, match="regular file"):
        read_stage2_length_manifest(tmp_path)


def test_manifest_writer_never_overwrites_different_identity(
    tmp_path, manifest_inputs, processor
):
    """Catches a later process replacing an already-published manifest."""
    path = write_stage2_length_manifest(
        tmp_path / "manifest.json", build_manifest(manifest_inputs, processor)
    )
    different = build_manifest(
        manifest_inputs, ManifestProcessor(), max_sequence_length=10
    )

    with pytest.raises(ValueError, match="different Stage 2 length manifest"):
        write_stage2_length_manifest(path, different)


def test_manifest_dataclass_is_frozen(manifest_inputs, processor):
    """Catches callers mutating selection metadata after validation."""
    manifest = build_manifest(manifest_inputs, processor)

    with pytest.raises(FrozenInstanceError):
        manifest.max_sequence_length = 10
    assert isinstance(manifest, Stage2LengthManifest)


def test_manifest_nested_distribution_is_immutable(manifest_inputs, processor):
    """Catches mutation through the frozen stats dataclass's nested mapping."""
    manifest = build_manifest(manifest_inputs, processor)

    with pytest.raises(TypeError, match="immutable"):
        manifest.stats.latent_block_distribution[9] = 1


def test_manifest_writer_revalidates_before_creating_file(
    tmp_path, manifest_inputs, processor
):
    """Catches serialization of an in-memory manifest whose invariants were bypassed."""
    manifest = build_manifest(manifest_inputs, processor)
    dict.__setitem__(manifest.stats.latent_block_distribution, 9, 1)
    path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="stats do not match"):
        write_stage2_length_manifest(path, manifest)

    assert not path.exists()


def test_manifest_parses_the_same_dataset_byte_snapshot_it_hashes(
    monkeypatch, manifest_inputs
):
    """Catches reopening changed JSONL contents after hashing an earlier snapshot."""
    data_path, rows = manifest_inputs
    original_bytes = data_path.read_bytes()
    changed_rows = [dict(rows[0]), rows[1]]
    changed_rows[0]["sequence_plan"] = [
        {"type": "text", "content": "long answer"}
    ]
    changed_bytes = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in changed_rows
    ).encode("utf-8")
    original_read_bytes = Path.read_bytes

    def read_then_change(path):
        snapshot = original_read_bytes(path)
        if path == data_path:
            data_path.write_bytes(changed_bytes)
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", read_then_change)

    manifest = build_manifest(manifest_inputs, ManifestProcessor())

    assert manifest.data_sha256 == hashlib.sha256(original_bytes).hexdigest()
    assert manifest.lengths == (8, 10)


def test_manifest_rejects_image_changes_during_length_measurement(manifest_inputs):
    """Catches image identity describing a different file state than measurement used."""
    data_path, _ = manifest_inputs
    processor = ImageMutatingProcessor(data_path.parent / "first.png")

    with pytest.raises(ValueError, match="changed during length measurement"):
        build_manifest(manifest_inputs, processor)
