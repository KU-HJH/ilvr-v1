"""Training admission must authenticate the full cohort before loading weights."""

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

import main
from tests.step_preference.test_packed_admission_cli import (
    inputs, build, verify, digest, canonical,
)
from utils_deepseed import parse_args


def training_args(source, *, packed=True, artifacts=True):
    argv = [
        "--stage", "stage2", "--load_model_path", str(source.checkpoint_path),
        "--data_path", str(source.retained_output_path if packed else source.clean_data_path),
        "--image_root", str(source.image_root), "--step_preference_enabled",
        "--step_preference_mode", "stage2b", "--stage2a_checkpoint_path", str(source.checkpoint_path),
        "--step_preference_objective", "legacy_dpo",
        "--step_negative_path", str(source.negative_path),
        "--step_negative_manifest", str(source.negative_manifest),
        "--attention_implementation", "sdpa", "--vision_attention_implementation", "flash_attention_2",
    ]
    if packed:
        argv += ["--step_preference_packing"]
    if artifacts:
        verification = source.manifest_output_path.parent / "verification.json"
        argv += [
            "--step_packed_admission_manifest", str(source.manifest_output_path),
            "--step_packed_admission_manifest_sha256", digest(source.manifest_output_path),
            "--step_packed_admission_verification", str(verification),
            "--step_packed_admission_verification_sha256", digest(verification),
        ]
    return parse_args(argv)


def test_packed_admission_selects_retained_from_authenticated_clean_negatives(inputs):
    source = inputs(long_last=True)
    manifest = build(source)
    assert verify(source)[0] == 0
    result = main.preflight_stage2b_artifacts(training_args(source))
    assert tuple(row.sample_id for row in result.prepared_examples) == ("sample-0",)
    assert set(result.negative_store) == {("sample-0", 0), ("sample-1", 0)}
    assert result.packed_admission == manifest
    assert tuple(result.packed_admission_records_by_sample_id) == ("sample-0",)
    with pytest.raises(TypeError):
        result.packed_admission_records_by_sample_id["extra"] = manifest.records[0]
    with pytest.raises(FrozenInstanceError):
        result.packed_admission.records[0].packed_padded_length = 256


@pytest.mark.parametrize("mutation", [
    "original", "clean", "retained", "negative", "negative_manifest", "length", "map",
    "manifest_hash", "verification_hash", "verification_identity", "verification_fail",
    "retained_order", "data_path", "negative_path", "image_root", "cap", "sample_limit",
    "latent_size", "clean_limit", "boundaries", "record_sample_id",
])
def test_packed_admission_rejects_chain_mutation_before_model(inputs, monkeypatch, mutation):
    source = inputs()
    build(source)
    assert verify(source)[0] == 0
    args = training_args(source)
    paths = dict(original=source.original_data_path, clean=source.clean_data_path,
                 retained=source.retained_output_path, negative=source.negative_path,
                 negative_manifest=source.negative_manifest, length=source.clean_length_manifest,
                 map=source.source_map_path)
    if mutation in paths:
        paths[mutation].write_bytes(paths[mutation].read_bytes() + b" ")
    elif mutation.endswith("_hash"):
        field = "packed_admission_" + mutation.replace("_hash", "_sha256")
        args.step_preference_config = replace(args.step_preference_config, **{field: "0" * 64})
    elif mutation in ("verification_identity", "verification_fail"):
        path = Path(args.step_preference_config.packed_admission_verification)
        body = json.loads(path.read_bytes())
        if mutation == "verification_identity":
            body["clean_admissible_sha256"] = "0" * 64
        else:
            body.update(status="fail", errors=["rejected"], verified_record_count=0,
                        retained_source_indices=[], filtered_source_indices=[])
        path.write_bytes(canonical(body))
        args.step_preference_config = replace(args.step_preference_config, packed_admission_verification_sha256=digest(path))
    elif mutation == "retained_order":
        source.retained_output_path.write_bytes(b"".join(reversed(source.retained_output_path.read_bytes().splitlines(keepends=True))))
    elif mutation == "record_sample_id":
        body = json.loads(source.manifest_output_path.read_bytes())
        body["records"][0]["sample_id"] = body["retained_sample_ids"][0] = "forged"
        source.manifest_output_path.write_bytes(canonical(body))
        args.step_preference_config = replace(args.step_preference_config, packed_admission_manifest_sha256=digest(source.manifest_output_path))
    elif mutation == "negative_path":
        copy = source.negative_path.with_name("other-negative.jsonl")
        copy.write_bytes(source.negative_path.read_bytes())
        args.step_preference_config = replace(args.step_preference_config, negative_path=str(copy))
    elif mutation == "cap":
        args.step_preference_config = replace(args.step_preference_config, max_packed_sequence_length=4736)
    else:
        name, value = dict(data_path=("data_path", str(source.clean_data_path)),
                           image_root=("image_root", str(source.checkpoint_path)),
                           sample_limit=("max_train_samples", 1), latent_size=("latent_size", 4),
                           clean_limit=("max_seq_length_train", 4096),
                           boundaries=("supervise_latent_boundaries", False))[mutation]
        setattr(args, name, value)
    def forbidden(*a, **k):
        pytest.fail("model construction reached on rejected admission")
    monkeypatch.setattr(main.Qwen2_5_VLForConditionalGeneration, "from_pretrained", forbidden)
    monkeypatch.setattr(main, "resolve_stage2b_initialization_decision", forbidden)
    with pytest.raises((ValueError, OSError)):
        main.build_stage2_training_components(args)


def test_public_packed_training_requires_admission(inputs, monkeypatch):
    source = inputs()
    args = training_args(source, artifacts=False)
    args.data_path = str(source.clean_data_path)
    monkeypatch.setattr(main, "get_args", lambda: args)
    monkeypatch.setattr(main, "validate_generated_artifact_paths", lambda args: None)
    with pytest.raises(ValueError, match="admission"):
        main.main_train()


def test_nonpacked_stage2b_does_not_require_admission(inputs):
    source = inputs()
    result = main.preflight_stage2b_artifacts(training_args(source, packed=False, artifacts=False))
    assert tuple(row.sample_id for row in result.prepared_examples) == ("sample-0", "sample-1")


@pytest.mark.parametrize("length_manifest", [False, True])
def test_packed_admission_components_preserve_cohort_without_runtime_filter(inputs, monkeypatch, length_manifest):
    from types import SimpleNamespace
    from tests.stage2.helpers import tiny_stage2_qwen
    from tests.step_preference.test_packed_admission_cli import AdmissionProcessor
    from stage2.checkpointing import Stage2LoadDecision
    from stage2.peft import Stage2TuningConfig

    source = inputs(long_last=True)
    build(source)
    assert verify(source)[0] == 0
    args = training_args(source)
    args._stage2b_preflight = main.preflight_stage2b_artifacts(args)
    model = tiny_stage2_qwen()
    processor = AdmissionProcessor()
    decision = Stage2LoadDecision(
        base_model_source=str(source.checkpoint_path), processor_source=str(source.checkpoint_path),
        adapter_source=None, trainer_resume_path=None,
        tuning=Stage2TuningConfig(method="full", r=4, alpha=8, dropout=0.0, target_modules=("q_proj",)),
        latent_size=8, supervise_latent_boundaries=True,
    )
    monkeypatch.setattr(main, "resolve_stage2b_initialization_decision", lambda **kwargs: decision)
    monkeypatch.setattr(main, "load_qwen2_5_vl_with_attention", lambda *a, **k: model)
    monkeypatch.setattr(main, "validate_qwen2_5_vl_attention_backends", lambda *a, **k: None)
    if length_manifest:
        args.stage2_length_manifest = "validated-retained-length.json"
        args.stage2_length_manifest_sha256 = "a" * 64
        # Any second partition must be ignored; admission alone owns the cohort.
        monkeypatch.setattr(main, "validate_stage2_length_manifest", lambda **kwargs: SimpleNamespace(
            manifest=SimpleNamespace(retained_indices=()), prepared_examples=()))
    monkeypatch.setattr(main, "filter_stage2_examples", lambda **kwargs: pytest.fail("runtime filtering changed admitted cohort"))
    dependencies = main.Stage2SourceDependencies(
        find_adapter_config=lambda path: None, load_config=lambda *a, **k: model.config,
        load_processor=lambda *a, **k: processor, load_model=lambda *a, **k: model,
    )
    result = main.build_stage2_training_components(args, source_dependencies=dependencies)
    assert tuple(row.sample_id for row in result.train_dataset) == ("sample-0",)
    assert tuple(result.data_collator.packed_admission_records_by_sample_id) == ("sample-0",)
    assert result.data_collator(list(result.train_dataset))["sample_ids"] == ("sample-0",)


@pytest.mark.parametrize("option", ["--_stage2b_capacity_probe", "--stage2b_capacity_probe"])
def test_packed_capacity_probe_bypass_is_not_public_cli(option):
    with pytest.raises(SystemExit):
        parse_args([option])


@pytest.mark.parametrize("requested, expected", [(None, "sample-0"), ("sample-1", "sample-1"), ("absent", None)])
def test_capacity_probe_selects_exact_authenticated_clean_sample(inputs, monkeypatch, requested, expected):
    from types import SimpleNamespace
    import torch
    from step_preference import equivalence
    from tests.step_preference.test_packed_capacity_probe import _Model
    from tests.step_preference.test_packed_admission_cli import AdmissionProcessor
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds

    source = inputs()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(equivalence, "_to_device", lambda batch, device: batch)
    builds = []
    def components(args):
        builds.append(args)
        preflight = args._stage2b_preflight
        return SimpleNamespace(
            model=_Model(checkpointing=False), train_dataset=preflight.prepared_examples,
            step_preference_config=args.step_preference_config,
            data_collator=main.Stage2DataCollator(
                AdmissionProcessor(), LatentTokenIds(90, 91, 92), 8, True,
                step_preference_enabled=True, negative_store=preflight.negative_store),
        )
    monkeypatch.setattr(main, "build_stage2_training_components", components)
    kwargs = dict(checkpoint_path=source.checkpoint_path, data_path=source.clean_data_path,
                  image_root=source.image_root, negative_path=source.negative_path,
                  negative_manifest=source.negative_manifest, device="cuda:0", requested_sample_id=requested)
    if expected is None:
        with pytest.raises(ValueError, match="requested sample"):
            equivalence.load_production_context(**kwargs)
        assert builds == []
    else:
        context = equivalence.load_production_context(**kwargs)
        assert context.identity["sample_id"] == expected
        assert context.batch["sample_ids"] == (expected,)
        assert builds[0]._stage2b_capacity_probe is True
