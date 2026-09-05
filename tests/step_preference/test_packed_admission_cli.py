"""Real source-chain, collator, planner, and publication contracts (CPU only)."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from PIL import Image
import pytest

from step_preference import packed_admission as admission
from stage2.length_manifest import build_stage2_length_manifest, write_stage2_length_manifest
from step_preference.negatives import StepNegative, inspect_stage2a_checkpoint_identity, write_negative_artifact
from tests.step_preference.test_collator_spans import CharacterOffsetTokenizer, SinglePassQwenProcessor


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AdmissionTokenizer(CharacterOffsetTokenizer):
    def get_vocab(self):
        return self._special_ids

    def convert_tokens_to_ids(self, token):
        return self._special_ids.get(token)


class AdmissionProcessor(SinglePassQwenProcessor):
    class ImageProcessor:
        merge_size = 2

        def to_dict(self):
            return {"merge_size": 2}

    image_processor = ImageProcessor()

    def __init__(self):
        super().__init__()
        self.tokenizer = AdmissionTokenizer()

    def to_dict(self):
        return {}


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    git("init", "-q")
    git("config", "user.name", "Admission Test")
    git("config", "user.email", "test@example.invalid")
    (tmp_path / ".gitignore").write_text("*\n")
    (tmp_path / "tracked").write_text("clean\n")
    git("add", "-f", ".gitignore", "tracked")
    git("commit", "-qm", "fixture")
    processor = AdmissionProcessor()
    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", lambda *a, **k: processor)
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *a, **k: SimpleNamespace(image_token_id=70, latent_size=8, supervise_latent_boundaries=True))
    Image.new("RGB", (2, 2)).save(tmp_path / "input.png")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name, raw in {"config.json": b'{}', "model.safetensors": b"weights", "tokenizer.json": b'{}'}.items():
        (checkpoint / name).write_bytes(raw)

    def make(*, count=2, selected=None, clean_cap=4608, long_last=False):
        rows = [dict(sample_id=f"sample-{i}", image_input="input.png", text_input="question", sequence_plan=[
            {"type": "text", "content": "A"},
            {"type": "latent", "helper_image": "helper.png"},
            {"type": "text", "content": "BC"},
        ]) for i in range(count)]
        if long_last:
            rows[-1]["sequence_plan"][-1]["content"] = "B" * 1500
        source = tmp_path / "original.jsonl"
        source.write_bytes(b"".join(canonical(row) for row in rows))
        manifest = build_stage2_length_manifest(data_path=source, image_root=tmp_path, processor=processor, processor_source=str(checkpoint), latent_size=8, supervise_latent_boundaries=True, max_sequence_length=clean_cap, assistant_plan_rendering="adjacent-text-space-lf-v1")
        if count == 3450:
            from stage2.data import summarize_stage2_lengths
            lengths = manifest.lengths[:3354] + (4700,) * 96
            manifest = replace(manifest, lengths=lengths, retained_indices=tuple(range(3354)), stats=summarize_stage2_lengths(lengths=lengths, latent_block_counts=manifest.latent_block_counts, max_length=clean_cap))
        length_path = write_stage2_length_manifest(tmp_path / "length.json", manifest)
        selected = tuple(manifest.retained_indices if selected is None else selected)
        clean = tmp_path / "clean.jsonl"
        clean.write_bytes(b"".join(canonical(rows[i]) for i in selected))
        mapping = tmp_path / "map.json"
        mapping.write_bytes(canonical(dict(schema_version=1, kind="stage2b-subset-source-map", original_source_sha256=digest(source), subset_sha256=digest(clean), original_source_indices=selected, sample_ids=[rows[i]["sample_id"] for i in selected])))
        inventory, tokenizer = inspect_stage2a_checkpoint_identity(checkpoint)
        negative = tmp_path / "negative.jsonl"
        negative_manifest = tmp_path / "negative-manifest.json"
        negatives = [StepNegative(f"sample-{i}", 0, rows[i]["sequence_plan"][-1]["content"], "D" * (1500 if long_last and i == selected[-1] else 2), None, True, False, -1.0, -2.0, None) for i in selected]
        write_negative_artifact(negative_path=negative, manifest_path=negative_manifest, negatives=negatives, source_path=clean, source_sha256=digest(clean), selected_row_indices=tuple(range(len(selected))), stage2a_checkpoint_path=checkpoint, checkpoint_files=inventory, tokenizer_config_identity=tokenizer, latent_size=8, assistant_plan_rendering="adjacent-text-space-lf-v1", generation_settings=dict(attention_implementation="sdpa", vision_attention_implementation="flash_attention_2", device="cuda:0", max_new_tokens=32, min_tokens=2, torch_dtype="bfloat16"))
        return SimpleNamespace(original_data_path=source, clean_data_path=clean, clean_length_manifest=length_path, clean_length_manifest_sha256=digest(length_path), image_root=tmp_path, checkpoint_path=checkpoint, source_map_path=mapping, source_map_sha256=digest(mapping), negative_path=negative, negative_manifest=negative_manifest, max_packed_sequence_length=4608, retained_output_path=tmp_path / "retained.jsonl", manifest_output_path=tmp_path / "admission.json")
    return make


def build(args):
    assert hasattr(admission, "build_packed_admission"), "missing admission producer"
    return admission.build_packed_admission(args)


def verify(args, expected_hash=None):
    assert hasattr(admission, "main"), "missing admission CLI"
    output = args.manifest_output_path.parent / "verification.json"
    code = admission.main(["verify", "--manifest-path", str(args.manifest_output_path), "--manifest-sha256", expected_hash or digest(args.manifest_output_path), "--output-path", str(output)])
    return code, json.loads(output.read_bytes())


def test_build_and_verify_preserve_source_order_and_reconstruct_features(inputs):
    args = inputs(long_last=True)
    result = build(args)
    assert result.retained_source_indices == (0,)
    assert result.filtered_source_indices == (1,)
    row = result.records[0]
    assert (row.vision_token_count, row.image_count, row.reasoning_block_count, row.branch_count, row.scored_token_count) == (4, 1, 1, 3, 8)
    assert args.retained_output_path.read_bytes() == args.clean_data_path.read_bytes().splitlines(keepends=True)[0]
    assert result.identity.git_commit == git("rev-parse", "HEAD")
    assert result.identity.tracked_worktree_clean is True
    code, verified = verify(args)
    assert code == 0 and verified["status"] == "pass"
    assert verified["verified_record_count"] == 2
    for key, path in [("manifest_sha256", args.manifest_output_path), ("clean_admissible_sha256", args.clean_data_path), ("packed_retained_sha256", args.retained_output_path), ("negative_sha256", args.negative_path), ("negative_manifest_sha256", args.negative_manifest)]:
        assert verified[key] == digest(path)


@pytest.mark.parametrize("selected,expected", [(None, 3354), ((1314,), 1)])
def test_all_clean_admissible_counts_and_original_ordinals_survive(inputs, selected, expected):
    args = inputs(count=3450, selected=selected)
    manifest = build(args)
    assert asdict(manifest.counts) == dict(original=3450, clean_admissible=3354, mapped=expected, retained=expected, filtered=0)
    if selected:
        assert manifest.records[0].source_index == 1314
        assert manifest.retained_source_indices == (1314,)
    else:
        assert manifest.retained_source_indices == tuple(range(3354))
    assert verify(args)[0] == 0


@pytest.mark.parametrize("mutation", ["packed_length", "vision_count", "image_count", "reason", "retained_order", "retained_byte", "negative_manifest_hash", "backend", "cap", "git"])
def test_verifier_rejects_mutations_and_never_repairs_manifest(inputs, mutation):
    args = inputs(selected=(0,) if mutation == "packed_length" else None)
    build(args)
    payload = json.loads(args.manifest_output_path.read_bytes())
    if mutation == "packed_length":
        payload["records"][0]["packed_unpadded_length"] += 1
    elif mutation == "vision_count":
        payload["records"][0]["vision_token_count"] += 1
    elif mutation == "image_count":
        payload["records"][0]["image_count"] += 1
    elif mutation == "reason":
        payload["records"][0]["admission_reason"] = "packed_length_exceeds_limit"
    elif mutation == "retained_order":
        payload["retained_source_indices"].reverse()
    elif mutation == "retained_byte":
        args.retained_output_path.write_bytes(args.retained_output_path.read_bytes() + b" ")
    else:
        key, value = {"negative_manifest_hash": ("negative_manifest_sha256", "a" * 64), "backend": ("attention_implementation", "eager"), "cap": ("max_packed_sequence_length", 4736), "git": ("git_commit", "b" * 40)}[mutation]
        payload["identity"][key] = value
    args.manifest_output_path.write_bytes(canonical(payload))
    before = args.manifest_output_path.read_bytes()
    code, result = verify(args)
    assert code == 2 and result["status"] == "fail"
    assert len(result["errors"]) == 1 and result["verified_record_count"] == 0
    assert result["retained_source_indices"] == result["filtered_source_indices"] == []
    assert args.manifest_output_path.read_bytes() == before


@pytest.mark.parametrize("failure", ["missing", "hash"])
def test_untrusted_manifest_produces_bounded_failure(inputs, failure):
    args = inputs()
    if failure == "hash":
        build(args)
    code, result = verify(args, "0" * 64)
    assert code == 2 and result["status"] == "fail"
    assert all(result[key] is None for key in ["manifest_sha256", "clean_admissible_sha256", "packed_retained_sha256", "negative_sha256", "negative_manifest_sha256"])
    assert result["verified_record_count"] == 0
    assert result["retained_source_indices"] == result["filtered_source_indices"] == []
    assert len(result["errors"]) == 1


@pytest.mark.parametrize("stage", ["build", "verify"])
@pytest.mark.parametrize("staged", [False, True])
def test_dirty_tracked_tree_never_publishes_pass(inputs, stage, staged):
    args = inputs()
    if stage == "verify":
        build(args)
    Path("tracked").write_text("dirty\n")
    if staged:
        git("add", "-f", "tracked")
    if stage == "build":
        with pytest.raises(ValueError, match="tracked"):
            build(args)
        assert not args.retained_output_path.exists() and not args.manifest_output_path.exists()
    else:
        code, result = verify(args)
        assert code == 2 and result["status"] == "fail"


@pytest.mark.parametrize("occupied", ["retained_output_path", "manifest_output_path"])
def test_build_preflights_both_outputs_before_loading(inputs, occupied):
    args = inputs()
    getattr(args, occupied).write_bytes(b"preserved")
    args.checkpoint_path = Path("missing-checkpoint")
    with pytest.raises(ValueError, match="already exists"):
        build(args)
    assert getattr(args, occupied).read_bytes() == b"preserved"


@pytest.mark.parametrize("corruption", ["clean_byte", "map_index", "map_hash", "length_cap", "checkpoint_byte", "negative_source", "symlink", "noncanonical"])
def test_source_chain_rejects_disagreement_before_publication(inputs, corruption):
    args = inputs(clean_cap=8192 if corruption == "length_cap" else 4608)
    if corruption == "clean_byte":
        args.clean_data_path.write_bytes(args.clean_data_path.read_bytes() + b"\n")
    elif corruption == "map_index":
        obj = json.loads(args.source_map_path.read_bytes())
        obj["original_source_indices"] = [0, 2]
        args.source_map_path.write_bytes(canonical(obj))
        args.source_map_sha256 = digest(args.source_map_path)
    elif corruption == "map_hash":
        args.source_map_sha256 = "0" * 64
    elif corruption == "checkpoint_byte":
        (args.checkpoint_path / "tokenizer.json").write_bytes(b"changed")
    elif corruption == "negative_source":
        obj = json.loads(args.negative_manifest.read_bytes())
        obj["source_sha256"] = "0" * 64
        args.negative_manifest.write_bytes(canonical(obj))
    elif corruption == "symlink":
        alias = args.clean_data_path.with_name("alias.jsonl")
        alias.symlink_to(args.clean_data_path)
        args.clean_data_path = alias
    elif corruption == "noncanonical":
        args.source_map_path.write_text(json.dumps(json.loads(args.source_map_path.read_bytes()), indent=2))
        args.source_map_sha256 = digest(args.source_map_path)
    with pytest.raises(ValueError):
        build(args)
    assert not args.retained_output_path.exists() and not args.manifest_output_path.exists()


def test_second_exclusive_write_failure_preserves_retained_orphan(inputs, monkeypatch):
    args = inputs()
    def load_config(*_args, **_kwargs):
        args.manifest_output_path.write_bytes(b"occupied during reconstruction")
        return SimpleNamespace(image_token_id=70, latent_size=8, supervise_latent_boundaries=True)
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", load_config)
    with pytest.raises(ValueError, match="already exists"):
        build(args)
    assert args.retained_output_path.read_bytes() == args.clean_data_path.read_bytes()
    assert args.manifest_output_path.read_bytes() == b"occupied during reconstruction"


def test_image_change_during_reconstruction_cannot_publish_admission(inputs, monkeypatch):
    args = inputs()
    def load_config(*_args, **_kwargs):
        Image.new("RGB", (3, 3)).save(args.image_root / "input.png")
        return SimpleNamespace(image_token_id=70, latent_size=8, supervise_latent_boundaries=True)
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", load_config)
    with pytest.raises(ValueError, match="image"):
        build(args)
    assert not args.retained_output_path.exists() and not args.manifest_output_path.exists()


@pytest.mark.parametrize("artifact,prefix", [("clean_data_path", 1), ("retained_output_path", 2), ("negative_path", 3), ("negative_manifest", 4)])
def test_verifier_hashes_form_only_authenticated_prefix(inputs, artifact, prefix):
    args = inputs()
    build(args)
    path = getattr(args, artifact)
    path.write_bytes(path.read_bytes() + b" ")
    code, result = verify(args)
    assert code == 2
    hashes = [result[key] for key in ["manifest_sha256", "clean_admissible_sha256", "packed_retained_sha256", "negative_sha256", "negative_manifest_sha256"]]
    assert all(value is not None for value in hashes[:prefix])
    assert hashes[prefix:] == [None] * (5 - prefix)


@pytest.mark.parametrize("artifact,hash_key,prefix", [("clean_data_path", "clean_data_sha256", 1), ("negative_path", "negative_sha256", 3)])
def test_syntax_invalid_source_is_not_published_as_authenticated_identity(inputs, artifact, hash_key, prefix):
    args = inputs()
    build(args)
    path = getattr(args, artifact)
    path.write_bytes(b'{"duplicate":1,"duplicate":2}\n')
    manifest = json.loads(args.manifest_output_path.read_bytes())
    manifest["identity"][hash_key] = digest(path)
    args.manifest_output_path.write_bytes(canonical(manifest))
    code, result = verify(args)
    assert code == 2
    hashes = [result[key] for key in ["manifest_sha256", "clean_admissible_sha256", "packed_retained_sha256", "negative_sha256", "negative_manifest_sha256"]]
    assert hashes[prefix:] == [None] * (5 - prefix)


def test_real_audit_collection_uses_shared_collator_feature_definitions(inputs):
    from scripts.audit_packed_stage2b_lengths import collect_packed_length_audit
    args = inputs()
    identity, rows = collect_packed_length_audit(SimpleNamespace(checkpoint_path=args.checkpoint_path, data_path=args.clean_data_path, image_root=args.image_root, negative_path=args.negative_path, negative_manifest=args.negative_manifest))
    assert identity["attention_implementation"] == "sdpa"
    assert [(row["vision_token_count"], row["image_count"], row["reasoning_block_count"], row["scored_token_count"]) for row in rows] == [(4, 1, 1, 8), (4, 1, 1, 8)]


def test_source_map_cannot_admit_an_existing_clean_excluded_original_row(inputs):
    args = inputs(count=3450, selected=(3354,))
    with pytest.raises(ValueError, match="clean-admissible"):
        build(args)
    assert not args.manifest_output_path.exists()


def test_clean_reencoding_is_rejected_even_with_matching_map_hash(inputs):
    args = inputs()
    args.clean_data_path.write_bytes(args.clean_data_path.read_bytes().replace(b'"sample_id":', b'"sample_id": '))
    payload = json.loads(args.source_map_path.read_bytes())
    payload["subset_sha256"] = digest(args.clean_data_path)
    args.source_map_path.write_bytes(canonical(payload))
    args.source_map_sha256 = digest(args.source_map_path)
    with pytest.raises(ValueError, match="mapped original lines"):
        build(args)
    assert not args.manifest_output_path.exists()


@pytest.mark.parametrize("field,value", [("latent_size", 7), ("supervise_latent_boundaries", False)])
def test_checkpoint_configuration_disagreement_fails_without_output(inputs, monkeypatch, field, value):
    args = inputs()
    config = dict(image_token_id=70, latent_size=8, supervise_latent_boundaries=True)
    config[field] = value
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *a, **k: SimpleNamespace(**config))
    with pytest.raises(ValueError, match="configuration"):
        build(args)
    assert not args.retained_output_path.exists() and not args.manifest_output_path.exists()


def test_build_cli_accepts_the_required_public_flags(inputs):
    args = inputs(selected=(1,))
    argv = ["build"]
    for name, value in vars(args).items():
        argv.extend(("--" + name.replace("_", "-"), str(value)))
    assert admission.main(argv) == 0
    payload = json.loads(args.manifest_output_path.read_bytes())
    assert payload["retained_source_indices"] == [1]
    assert args.retained_output_path.read_bytes() == args.clean_data_path.read_bytes()
