from dataclasses import replace
import hashlib
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import peft
import pytest
import torch
from accelerate.state import AcceleratorState, GradientState, PartialState
from torch import nn
from transformers import Qwen2_5_VLConfig
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

from peft_common import capture_qwen2_5_vl_architecture
from stage1.checkpointing import (
    STAGE1_PEFT_METADATA_FILENAME,
    STAGE1_PEFT_SCHEMA_VERSION,
    STAGE1_REPOSITORY_FORMAT_VERSION,
    Stage1ArtifactKind,
    Stage1PeftMetadata,
    Stage1PeftMetadataSnapshot,
    Stage1ResumeDecision,
    decode_stage1_peft_metadata_snapshot,
    read_stage1_peft_metadata_snapshot,
    resolve_stage1_resume,
    save_stage1_peft_metadata,
)
from stage1.composition import (
    DEFAULT_STAGE1_DEPENDENCIES,
    LATENT_TOKEN_STRINGS,
    build_stage1_training_arguments,
    build_stage1_training_components,
    run_stage1,
)
from utils_deepseed import parse_args
from stage1.benchmark import config_digest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _clear_training_process_state() -> None:
    os.environ.pop("ACCELERATE_USE_DEEPSPEED", None)
    unset_hf_deepspeed_config()
    AcceleratorState._reset_state(reset_partial_state=True)
    PartialState._reset_state()
    GradientState._reset_state()


@pytest.fixture(autouse=True)
def isolate_deepspeed_training_argument_state():
    """Own cleanup for this module's DeepSpeed-configured SFTConfig fixtures."""
    _clear_training_process_state()
    try:
        yield
    finally:
        _clear_training_process_state()


class RecordingTokenizer:
    def __init__(self, size=12):
        self.vocab = {f"token-{index}": index for index in range(size)}
        self.add_calls = []

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def add_special_tokens(self, payload):
        tokens = tuple(payload["additional_special_tokens"])
        self.add_calls.append(tokens)
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


class RecordingProcessor:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or RecordingTokenizer()
        self.saved = []

    def save_pretrained(self, path):
        self.saved.append(str(path))


class RecordingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = nn.Linear(1, 1)
        self.language = nn.Linear(1, 1)
        self.resize_calls = []

    def resize_token_embeddings(self, size):
        self.resize_calls.append(size)
        self.config.vocab_size = size


class RecordingTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.args = kwargs["args"]
        self.events = []

    def train(self, *, resume_from_checkpoint):
        self.events.append(("train", resume_from_checkpoint))

    def save_model(self, path):
        self.events.append(("save", str(path)))

    def is_world_process_zero(self):
        return True


def stage1_args(tmp_path, *extra, method="dora"):
    (tmp_path / "TRAIN.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "images").mkdir(exist_ok=True)
    tuning_args = (
        ["--use_lora"]
        if method == "lora"
        else ["--tuning_method", method]
    )
    return parse_args(
        [
            "--stage",
            "stage1",
            *tuning_args,
            "--model",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "--data_path",
            str(tmp_path / "TRAIN.jsonl"),
            "--image_root",
            str(tmp_path / "images"),
            "--save_model_path",
            str(tmp_path / "output"),
            "--learning_rate",
            "1e-5",
            "--per_device_train_batch_size",
            "8",
            "--gradient_accumulation_steps",
            "2",
            "--epochs",
            "3",
            "--seed",
            "42",
            "--attention_implementation",
            "flash_attention_2",
            "--deepspeed_config",
            str(REPO_ROOT / "configs/config_stage3.json"),
            "--save_strategy",
            "no",
            *extra,
        ]
    )


@pytest.mark.parametrize(
    ("method", "expected_kwargs"),
    [
        ("full", {"use_reentrant": False}),
        ("lora", {"use_reentrant": True}),
        ("dora", {"use_reentrant": True}),
    ],
)
def test_stage1_peft_uses_zero3_safe_reentrant_gradient_checkpointing(
    tmp_path, method, expected_kwargs
):
    training_args = build_stage1_training_arguments(
        stage1_args(tmp_path, "--gradient_checkpointing", method=method)
    )

    assert training_args.gradient_checkpointing is True
    assert training_args.gradient_checkpointing_kwargs == expected_kwargs


@pytest.mark.parametrize("method", ["full", "lora", "dora"])
def test_disabled_stage1_gradient_checkpointing_has_no_backend_kwargs(
    tmp_path, method
):
    training_args = build_stage1_training_arguments(
        stage1_args(tmp_path, "--no-gradient_checkpointing", method=method)
    )

    assert training_args.gradient_checkpointing is False
    assert training_args.gradient_checkpointing_kwargs is None


def _write_benchmark_audit(tmp_path, args, indices=(7, 3)):
    data = Path(args.data_path).resolve()
    images = Path(args.image_root).resolve()
    stat = data.stat()
    payload = {
        "schema_version": 1,
        "kind": "stage1-high-cost-row-audit",
        "data_path": str(data),
        "image_root": str(images),
        "requested_count": 64,
        "row_count": len(indices),
        "input_stat": {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "selected_rows": [
            {
                "source_row_index": index,
                "token_estimate": 0,
                "original_image_count": 0,
                "helper_image_count": 0,
                "latent_block_count": 0,
                "example": {},
            }
            for index in indices
        ],
    }
    payload["audit_digest"] = config_digest(payload)
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _write_probe_run_config(tmp_path, args):
    audit_path, audit = _write_benchmark_audit(tmp_path, args)
    source = Path(args.data_path).resolve()
    stat = source.stat()
    config = {
        "schema_version": 1,
        "kind": "stage1-probe-run-manifest",
        "method": "full",
        "tuning_config": None,
        "model": args.model,
        "source_data_path": str(source),
        "data_path": str(source),
        "data_stat": {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "image_root": str(Path(args.image_root).resolve()),
        "output_dir": str(Path(args.save_model_path).resolve()),
        "micro_batch": 8,
        "world_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_global_batch": 32,
        "epochs": 3,
        "learning_rate": 1e-5,
        "seed": 42,
        "max_steps": 3,
        "final_save": False,
        "deepspeed_config": str(Path(args.deepspeed_config).resolve()),
        "deepspeed_config_digest": hashlib.sha256(
            Path(args.deepspeed_config).read_bytes()
        ).hexdigest(),
        "attention_implementation": "flash_attention_2",
        "save_strategy": "no",
        "audit_digest": audit["audit_digest"],
        "probe_rows": str(audit_path),
        "probe_row_indices": [7, 3],
        "maximum_config_digest": None,
        "prerequisite_chain": str(tmp_path / "prerequisite-chain.json"),
        "prerequisite_chain_digest": "f" * 64,
    }
    config["config_digest"] = config_digest(config)
    path = tmp_path / "run-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, config


def _write_peft_resume_checkpoint(args, *, step=9):
    checkpoint = Path(args.save_model_path) / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    architecture_config = Qwen2_5_VLConfig(
        vocab_size=12,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 2,
            "out_hidden_size": 8,
            "fullatt_block_indexes": [0],
        },
        image_token_id=5,
        vision_start_token_id=6,
        vision_end_token_id=7,
        video_token_id=8,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 0]},
    )
    target_suffixes = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    owners = {
        "q_proj": "self_attn",
        "k_proj": "self_attn",
        "v_proj": "self_attn",
        "o_proj": "self_attn",
        "gate_proj": "mlp",
        "up_proj": "mlp",
        "down_proj": "mlp",
    }
    metadata = Stage1PeftMetadata(
        schema_version=STAGE1_PEFT_SCHEMA_VERSION,
        stage="stage1",
        base_model_reference=args.model,
        base_model_architecture=capture_qwen2_5_vl_architecture(
            architecture_config
        ),
        base_vocab_size=12,
        pretrained_model_vocab_size=12,
        resized_vocab_size=15,
        latent_token_ids=(12, 13, 14),
        latent_token_strings=LATENT_TOKEN_STRINGS,
        latent_size=args.latent_size,
        ce_weight=args.ce_weight,
        sim_weight=args.sim_weight,
        coverage_p=args.coverage_p,
        image_pool_k=args.image_pool_k,
        tuning_method="dora",
        rank=32,
        alpha=64,
        dropout=0.05,
        requested_target_suffixes=target_suffixes,
        resolved_target_names=tuple(
            f"model.layers.0.{owners[suffix]}.{suffix}"
            for suffix in target_suffixes
        ),
        input_row_format="peft_trainable_tokens_v1",
        output_row_format="ilvr_sparse_output_rows_v1",
        output_row_shape=(3, 8),
        peft_version=peft.__version__,
        repository_format_version=STAGE1_REPOSITORY_FORMAT_VERSION,
    )
    for filename in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "ilvr_token_rows.safetensors",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / filename).write_bytes(b"fixture")
    save_stage1_peft_metadata(checkpoint, metadata)
    processor = RecordingProcessor(RecordingTokenizer(size=12))
    for token, token_id in zip(LATENT_TOKEN_STRINGS, (12, 13, 14)):
        processor.tokenizer.vocab[token] = token_id
    return checkpoint, metadata, processor


def _fake_metadata_snapshot(checkpoint):
    return Stage1PeftMetadataSnapshot(
        artifact_dir=Path(checkpoint).resolve(),
        metadata_json=b"test-owned immutable metadata",
    )


def fake_dependencies(
    events,
    *,
    decision=None,
    processor=None,
    metadata=None,
    pretrained_vocab_size=None,
):
    decision = decision or Stage1ResumeDecision(None, None, False)
    processor = processor or RecordingProcessor()
    config = SimpleNamespace(
        vocab_size=(
            len(processor.tokenizer)
            if pretrained_vocab_size is None
            else pretrained_vocab_size
        )
    )
    model = RecordingModel(config)
    parameter_report = SimpleNamespace(
        trainable_parameters=7,
        total_parameters=11,
        trainable_names=("adapter",),
    )

    def resolve_resume(output_dir, explicit_resume):
        events.append(("resolve", Path(output_dir), explicit_resume))
        return decision

    def load_processor(source, **kwargs):
        events.append(("processor", str(source), kwargs))
        return processor

    def load_config(source, **kwargs):
        events.append(("config", str(source), kwargs))
        return config

    def load_model(source, **kwargs):
        events.append(("model", str(source), kwargs))
        assert kwargs["config"] is config
        return model

    preflight_witness = object()

    def preflight_peft(selected_processor, selected_config, path, **kwargs):
        events.append(
            (
                "preflight",
                selected_processor,
                selected_config,
                Path(path),
                kwargs,
                preflight_witness,
            )
        )
        return preflight_witness

    def load_peft_model(base, selected_processor, path, **kwargs):
        events.append(("peft", base, selected_processor, Path(path), kwargs))
        assert kwargs["preflight"] is preflight_witness
        return model

    def build_tuned(base, tuning, latent_ids):
        events.append(("topology", tuning.method, tuple(latent_ids)))
        return base

    def validate_boundary(candidate, tuning, latent_ids):
        events.append(("boundary", tuning.method, tuple(latent_ids)))
        return parameter_report

    def prepare_dataset(data_path, image_root):
        events.append(("dataset", Path(data_path), Path(image_root)))
        return ["row-0", "row-1"]

    def collator_factory(selected_processor, token_ids, latent_size):
        events.append(("collator", token_ids, latent_size))
        return (selected_processor, token_ids, latent_size)

    def trainer_factory(**kwargs):
        events.append(("trainer", kwargs["model"]))
        return RecordingTrainer(**kwargs)

    return replace(
        DEFAULT_STAGE1_DEPENDENCIES,
        resolve_resume=resolve_resume,
        load_processor=load_processor,
        load_config=load_config,
        load_model=load_model,
        preflight_peft=preflight_peft,
        build_tuned_model=build_tuned,
        validate_boundary=validate_boundary,
        prepare_dataset=prepare_dataset,
        collator_factory=collator_factory,
        trainer_factory=trainer_factory,
        build_peft_metadata=lambda **kwargs: metadata,
        capture_architecture=lambda config: {
            "decoder": {"vocab_size": config.vocab_size, "hidden_size": 4}
        },
        resolve_target_names=lambda model, suffixes: tuple(
            f"model.layers.0.{suffix}" for suffix in suffixes
        ),
        validate_vocab_alignment=lambda selected_model, expected_size: events.append(
            ("vocab", selected_model, expected_size)
        ),
        load_peft_metadata=lambda path: object(),
        decode_peft_metadata=lambda snapshot: metadata,
        load_peft_model=load_peft_model,
    ), processor, model, parameter_report


@pytest.mark.parametrize("explicit", [True, False])
def test_real_peft_resume_reuses_classification_metadata_snapshot_through_preflight(
    tmp_path, monkeypatch, explicit
):
    """Catches composition reopening the selected PEFT metadata after resolve."""
    args = stage1_args(tmp_path)
    checkpoint, metadata, processor = _write_peft_resume_checkpoint(args)
    if explicit:
        args.resume_from_checkpoint_path = str(checkpoint)
    metadata_path = checkpoint / STAGE1_PEFT_METADATA_FILENAME
    witnessed_bytes = metadata_path.read_bytes()
    metadata_reads = []
    real_read_bytes = Path.read_bytes

    def count_metadata_read(path):
        if path == metadata_path:
            metadata_reads.append(path)
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", count_metadata_read)
    events = []
    deps, _, _, _ = fake_dependencies(
        events, processor=processor, metadata=metadata
    )
    real_load_processor = deps.load_processor
    real_preflight = deps.preflight_peft
    preflight_snapshots = []

    def mutate_metadata_after_resolve(source, **kwargs):
        metadata_path.write_bytes(b'{"schema_version":999}\n')
        return real_load_processor(source, **kwargs)

    def inspect_preflight(selected_processor, selected_config, path, **kwargs):
        snapshot = kwargs["metadata_snapshot"]
        preflight_snapshots.append(snapshot)
        assert snapshot.metadata_json == witnessed_bytes
        assert decode_stage1_peft_metadata_snapshot(snapshot) == metadata
        return real_preflight(
            selected_processor, selected_config, path, **kwargs
        )

    deps = replace(
        deps,
        resolve_resume=resolve_stage1_resume,
        load_processor=mutate_metadata_after_resolve,
        load_peft_metadata=read_stage1_peft_metadata_snapshot,
        decode_peft_metadata=decode_stage1_peft_metadata_snapshot,
        preflight_peft=inspect_preflight,
    )

    bundle = build_stage1_training_components(args, dependencies=deps)

    assert metadata_reads == [metadata_path]
    assert bundle.resume_decision.peft_metadata_snapshot is preflight_snapshots[0]
    assert preflight_snapshots[0].artifact_dir == checkpoint.resolve()
    assert bundle.trainer.kwargs["stage1_peft_metadata"] == metadata


def test_real_explicit_resume_rejects_malformed_metadata_before_large_load(
    tmp_path,
):
    """Catches malformed PEFT metadata surviving classification into loading."""
    args = stage1_args(tmp_path)
    checkpoint, metadata, processor = _write_peft_resume_checkpoint(args)
    args.resume_from_checkpoint_path = str(checkpoint)
    (checkpoint / STAGE1_PEFT_METADATA_FILENAME).write_bytes(
        b'{"schema_version":999}\n'
    )
    events = []
    deps, _, _, _ = fake_dependencies(
        events, processor=processor, metadata=metadata
    )
    deps = replace(deps, resolve_resume=resolve_stage1_resume)

    with pytest.raises(ValueError, match="schema|fields"):
        build_stage1_training_components(args, dependencies=deps)

    assert not any(event[0] in {"processor", "model"} for event in events)


def test_fresh_dora_composition_adds_tokens_once_and_propagates_runtime_args(tmp_path):
    """Catches repeated token remapping and hidden Stage 1 Trainer constants."""
    events = []
    metadata = object()
    deps, processor, model, report = fake_dependencies(
        events, metadata=metadata
    )
    args = stage1_args(tmp_path)

    bundle = build_stage1_training_components(args, dependencies=deps)

    assert events[0][0] == "resolve"
    assert [event[0] for event in events].index("resolve") < [
        event[0] for event in events
    ].index("model")
    assert processor.tokenizer.add_calls == [LATENT_TOKEN_STRINGS]
    assert model.resize_calls == [15]
    assert (
        model.config.latent_token_id,
        model.config.latent_start_id,
        model.config.latent_end_id,
    ) == (12, 13, 14)
    assert model.config.latent_size == args.latent_size
    assert model.config.stage == "stage1"
    assert bundle.parameter_report is report
    assert bundle.dataset == ["row-0", "row-1"]
    assert bundle.trainer.kwargs["stage1_peft_metadata"] is metadata
    assert bundle.trainer.kwargs["model"] is model
    assert bundle.trainer.args.learning_rate == 1e-5
    assert bundle.trainer.args.per_device_train_batch_size == 8
    assert bundle.trainer.args.gradient_accumulation_steps == 2
    assert bundle.trainer.args.num_train_epochs == 3
    assert bundle.trainer.args.seed == 42
    assert bundle.trainer.args.deepspeed == str(
        REPO_ROOT / "configs/config_stage3.json"
    )
    assert bundle.trainer.args.save_strategy.value == "no"
    model_load = next(event for event in events if event[0] == "model")
    assert model_load[2]["attn_implementation"] == "flash_attention_2"
    assert [event[0] for event in events].index("boundary") < [
        event[0] for event in events
    ].index("trainer")
    assert all(not parameter.requires_grad for parameter in model.visual.parameters())


@pytest.mark.parametrize("method", ["full", "dora"])
@pytest.mark.parametrize("pretrained_vocab_size", [12, 18])
def test_fresh_stage1_accepts_exact_or_padded_pretrained_vocabulary_and_shrinks_once(
    tmp_path, pretrained_vocab_size, method
):
    events = []
    metadata_calls = []
    metadata = object()
    deps, processor, model, _ = fake_dependencies(
        events,
        metadata=metadata,
        pretrained_vocab_size=pretrained_vocab_size,
    )
    deps = replace(
        deps,
        build_peft_metadata=lambda **kwargs: (
            metadata_calls.append(kwargs) or metadata
        ),
    )

    build_stage1_training_components(
        stage1_args(tmp_path, method=method), dependencies=deps
    )

    assert processor.tokenizer.add_calls == [LATENT_TOKEN_STRINGS]
    assert model.resize_calls == [15]
    assert model.config.vocab_size == 15
    assert [event[0] for event in events].index("vocab") < [
        event[0] for event in events
    ].index("topology")
    if method == "dora":
        assert len(metadata_calls) == 1
        assert metadata_calls[0]["base_vocab_size"] == 12
        assert metadata_calls[0]["pretrained_model_vocab_size"] == (
            pretrained_vocab_size
        )
        assert metadata_calls[0]["resized_vocab_size"] == 15
        assert metadata_calls[0]["latent_token_ids"] == (12, 13, 14)
        assert metadata_calls[0]["base_architecture"]["decoder"]["vocab_size"] == (
            pretrained_vocab_size
        )
    else:
        assert metadata_calls == []


@pytest.mark.parametrize("method", ["full", "dora"])
def test_fresh_stage1_rejects_pretrained_vocabulary_too_small_before_model_load(
    tmp_path, method
):
    events = []
    deps, processor, model, _ = fake_dependencies(
        events,
        metadata=object(),
        pretrained_vocab_size=11,
    )

    with pytest.raises(ValueError, match="pretrained model vocabulary.*too small"):
        build_stage1_training_components(
            stage1_args(tmp_path, method=method), dependencies=deps
        )

    assert processor.tokenizer.add_calls == [LATENT_TOKEN_STRINGS]
    assert model.resize_calls == []
    assert not any(event[0] == "model" for event in events)


@pytest.mark.parametrize(
    ("kind", "method"),
    [(Stage1ArtifactKind.FULL, "full"), (Stage1ArtifactKind.PEFT, "dora")],
)
def test_resume_composition_preserves_checkpoint_token_ids_without_readding(
    tmp_path, kind, method
):
    """Catches Stage 1 resume re-adding or remapping latent vocabulary rows."""
    checkpoint = tmp_path / "output" / "checkpoint-9"
    decision = Stage1ResumeDecision(
        checkpoint,
        kind,
        True,
        _fake_metadata_snapshot(checkpoint)
        if kind is Stage1ArtifactKind.PEFT
        else None,
    )
    processor = RecordingProcessor(RecordingTokenizer(size=12))
    for token, token_id in zip(LATENT_TOKEN_STRINGS, (12, 13, 14)):
        processor.tokenizer.vocab[token] = token_id
    metadata = SimpleNamespace(
        base_model_reference="Qwen/Qwen2.5-VL-3B-Instruct",
        base_vocab_size=12,
        resized_vocab_size=15,
        latent_token_ids=(12, 13, 14),
        latent_size=8,
        ce_weight=1.0,
        sim_weight=1.0,
        coverage_p=0.9,
        image_pool_k=8,
        tuning_method="dora",
        rank=32,
        alpha=64,
        dropout=0.05,
        requested_target_suffixes=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )
    events = []
    deps, _, model, _ = fake_dependencies(
        events, decision=decision, processor=processor, metadata=metadata
    )
    model.config.latent_token_id = 12
    model.config.latent_start_id = 13
    model.config.latent_end_id = 14
    model.config.latent_pad_token_id = 12
    model.config.latent_start_token_id = 13
    model.config.latent_end_token_id = 14
    model.config.latent_size = 8
    args = stage1_args(tmp_path, method=method)

    bundle = build_stage1_training_components(args, dependencies=deps)

    assert bundle.resume_decision is decision
    assert processor.tokenizer.add_calls == []
    assert model.resize_calls == []
    assert bundle.trainer.kwargs["stage1_peft_metadata"] is (
        metadata if kind is Stage1ArtifactKind.PEFT else None
    )
    if kind is Stage1ArtifactKind.PEFT:
        assert any(event[0] == "processor" and event[1] == str(checkpoint) for event in events)
        assert not any(event[0] == "topology" for event in events)
        event_names = [event[0] for event in events]
        assert event_names.index("preflight") < event_names.index("model")
        preflight_event = next(
            event for event in events if event[0] == "preflight"
        )
        assert set(preflight_event[4]) == {"metadata_snapshot"}
        peft_event = next(event for event in events if event[0] == "peft")
        assert peft_event[3] == checkpoint
    else:
        assert any(event[0] == "model" and event[1] == str(checkpoint) for event in events)


@pytest.mark.parametrize(
    ("field", "checkpoint_value"),
    [
        ("latent_size", 7),
        ("ce_weight", 2.0),
        ("sim_weight", 2.0),
        ("coverage_p", 0.8),
        ("image_pool_k", 4),
    ],
)
def test_peft_resume_rejects_semantic_mismatch_before_any_large_load(
    tmp_path, field, checkpoint_value
):
    """Catches checkpoint-owned loss/model semantics changing on resume."""
    checkpoint = tmp_path / "output" / "checkpoint-9"
    decision = Stage1ResumeDecision(
        checkpoint,
        Stage1ArtifactKind.PEFT,
        True,
        _fake_metadata_snapshot(checkpoint),
    )
    metadata_values = {
        "base_model_reference": "Qwen/Qwen2.5-VL-3B-Instruct",
        "base_vocab_size": 12,
        "resized_vocab_size": 15,
        "latent_token_ids": (12, 13, 14),
        "latent_size": 8,
        "ce_weight": 1.0,
        "sim_weight": 1.0,
        "coverage_p": 0.9,
        "image_pool_k": 8,
        "tuning_method": "dora",
        "rank": 32,
        "alpha": 64,
        "dropout": 0.05,
        "requested_target_suffixes": (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    }
    metadata_values[field] = checkpoint_value
    events = []
    deps, _, _, _ = fake_dependencies(
        events,
        decision=decision,
        metadata=SimpleNamespace(**metadata_values),
    )

    with pytest.raises(ValueError, match="resume semantic mismatch"):
        build_stage1_training_components(stage1_args(tmp_path), dependencies=deps)

    assert [event for event in events if event[0] in {"processor", "model"}] == []


@pytest.mark.parametrize(
    "corruption", ["architecture", "adapter tensor", "sparse rows"]
)
def test_peft_artifact_preflight_failure_prevents_large_model_load(
    tmp_path, corruption
):
    """Catches composition loading base weights before artifact validation."""
    checkpoint = tmp_path / "output" / "checkpoint-9"
    decision = Stage1ResumeDecision(
        checkpoint,
        Stage1ArtifactKind.PEFT,
        True,
        _fake_metadata_snapshot(checkpoint),
    )
    processor = RecordingProcessor(RecordingTokenizer(size=12))
    for token, token_id in zip(LATENT_TOKEN_STRINGS, (12, 13, 14)):
        processor.tokenizer.vocab[token] = token_id
    metadata = SimpleNamespace(
        base_model_reference="Qwen/Qwen2.5-VL-3B-Instruct",
        base_vocab_size=12,
        resized_vocab_size=15,
        latent_token_ids=(12, 13, 14),
        latent_size=8,
        ce_weight=1.0,
        sim_weight=1.0,
        coverage_p=0.9,
        image_pool_k=8,
        tuning_method="dora",
        rank=32,
        alpha=64,
        dropout=0.05,
        requested_target_suffixes=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )
    events = []
    deps, _, model, _ = fake_dependencies(
        events,
        decision=decision,
        processor=processor,
        metadata=metadata,
    )
    model.config.latent_size = 8

    def reject_preflight(*args, **kwargs):
        raise ValueError(f"corrupt {corruption}")

    deps = replace(
        deps,
        preflight_peft=reject_preflight,
    )

    with pytest.raises(ValueError, match=f"corrupt {corruption}"):
        build_stage1_training_components(stage1_args(tmp_path), dependencies=deps)

    assert not any(event[0] == "model" for event in events)


def test_full_resume_rejects_latent_ids_that_are_not_appended_rows(tmp_path):
    """Catches a full checkpoint silently remapping ordinary vocabulary rows."""
    checkpoint = tmp_path / "output" / "checkpoint-9"
    decision = Stage1ResumeDecision(checkpoint, Stage1ArtifactKind.FULL, True)
    tokenizer = RecordingTokenizer(size=15)
    for token_id, token in zip((2, 3, 4), LATENT_TOKEN_STRINGS):
        del tokenizer.vocab[f"token-{token_id}"]
        tokenizer.vocab[token] = token_id
    processor = RecordingProcessor(tokenizer)
    events = []
    deps, _, model, _ = fake_dependencies(
        events, decision=decision, processor=processor
    )
    model.config.latent_token_id = 2
    model.config.latent_start_id = 3
    model.config.latent_end_id = 4
    model.config.latent_pad_token_id = 2
    model.config.latent_start_token_id = 3
    model.config.latent_end_token_id = 4
    model.config.latent_size = 8

    with pytest.raises(ValueError, match="appended"):
        build_stage1_training_components(
            stage1_args(tmp_path, method="full"), dependencies=deps
        )


@pytest.mark.parametrize("final_save", [True, False])
def test_run_stage1_trains_once_and_only_final_saves_when_requested(
    tmp_path, caplog, final_save
):
    """Catches duplicate final/numbered saves or saving during no-save probes."""
    trainer = RecordingTrainer(args=SimpleNamespace())
    processor = RecordingProcessor()
    decision = Stage1ResumeDecision(
        tmp_path / "checkpoint-4", Stage1ArtifactKind.FULL, True
    )
    bundle = SimpleNamespace(
        trainer=trainer,
        processor=processor,
        resume_decision=decision,
    )
    args = SimpleNamespace(final_save=final_save, save_model_path=str(tmp_path / "out"))
    ticks = iter((10.0, 13.5, 20.0, 21.25))

    with caplog.at_level(logging.INFO):
        result = run_stage1(
            args,
            component_builder=lambda selected: bundle,
            monotonic=lambda: next(ticks),
        )

    assert result is None
    expected = [("train", str(decision.checkpoint_path))]
    if final_save:
        expected.append(("save", str(tmp_path / "out")))
        assert processor.saved == [str(tmp_path / "out")]
        assert "Stage 1 final save time: 1.250s" in caplog.text
    else:
        assert processor.saved == []
    assert trainer.events == expected
    assert "Stage 1 Trainer time: 3.500s" in caplog.text


def test_run_stage1_atomically_emits_small_producer_evidence_without_model_load(
    tmp_path, monkeypatch,
):
    """Catches the real launcher/trainer path never publishing Task 17 input."""
    args = stage1_args(tmp_path, "--max_steps", "3", "--no_final_save", method="full")
    evidence_path = tmp_path / "producer evidence.json"
    audit_path, audit = _write_benchmark_audit(tmp_path, args)
    config = {
        "schema_version": 1,
        "kind": "stage1-probe-run-manifest",
        "method": "full",
        "tuning_config": None,
        "model": args.model,
        "source_data_path": str(Path(args.data_path).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "data_stat": {
            "device": Path(args.data_path).stat().st_dev,
            "inode": Path(args.data_path).stat().st_ino,
            "size": Path(args.data_path).stat().st_size,
            "mtime_ns": Path(args.data_path).stat().st_mtime_ns,
        },
        "image_root": str(Path(args.image_root).resolve()),
        "output_dir": str(Path(args.save_model_path).resolve()),
        "micro_batch": 8,
        "world_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_global_batch": 32,
        "epochs": 3,
        "learning_rate": 1e-5,
        "seed": 42,
        "max_steps": 3,
        "final_save": False,
        "deepspeed_config": str(Path(args.deepspeed_config).resolve()),
        "deepspeed_config_digest": hashlib.sha256(
            Path(args.deepspeed_config).read_bytes()
        ).hexdigest(),
        "attention_implementation": "flash_attention_2",
        "save_strategy": "no",
        "audit_digest": audit["audit_digest"],
        "probe_rows": str(audit_path),
        "probe_row_indices": [7, 3],
        "maximum_config_digest": None,
        "prerequisite_chain": str(tmp_path / "prerequisite-chain.json"),
        "prerequisite_chain_digest": "f" * 64,
    }
    config["config_digest"] = config_digest(config)
    config_path = tmp_path / "run config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    args.benchmark_run_config = str(config_path)
    args.benchmark_evidence_path = str(evidence_path)
    args.benchmark_source_data_path = str(Path(args.data_path).resolve())

    class EvidenceTrainer(RecordingTrainer):
        def __init__(self):
            super().__init__(args=SimpleNamespace())
            self.state = SimpleNamespace(global_step=3)
            self._representative_gradient_norm = 0.25

        def train(self, *, resume_from_checkpoint):
            super().train(resume_from_checkpoint=resume_from_checkpoint)
            return SimpleNamespace(
                metrics={
                    "train_samples_per_second": 24.0,
                    "train_steps_per_second": 0.75,
                    "train_loss": 0.5,
                }
            )

    trainer = EvidenceTrainer()
    bundle = SimpleNamespace(
        trainer=trainer,
        processor=RecordingProcessor(),
        dataset=[{}] * 64,
        parameter_report=SimpleNamespace(
            trainable_parameters=20, total_parameters=100
        ),
        resume_decision=Stage1ResumeDecision(None, None, False),
    )
    observed = lambda selected: {
        "rank_peak_allocated_bytes": (10, 11),
        "rank_peak_reserved_bytes": (12, 13),
        "gpu_names": ("stub-0", "stub-1"),
        "git_commit": "abc1234",
        "dependency_versions": {"torch": "test"},
    }

    import stage1.benchmark as benchmark
    reads = []
    original_read = benchmark._read_json_once
    artifact_builds = 0
    original_artifact_builder = benchmark._build_artifact_manifest

    def counted_read(path):
        reads.append(Path(path).resolve())
        return original_read(path)

    monkeypatch.setattr(benchmark, "_read_json_once", counted_read)

    def counted_artifact_builder(*args, **kwargs):
        nonlocal artifact_builds
        artifact_builds += 1
        return original_artifact_builder(*args, **kwargs)

    monkeypatch.setattr(
        benchmark, "_build_artifact_manifest", counted_artifact_builder
    )

    def replace_manifest_after_preflight(selected):
        config_path.write_text('{"replaced": true}', encoding="utf-8")
        return bundle

    run_stage1(
        args,
        component_builder=replace_manifest_after_preflight,
        monotonic=iter((1.0, 2.0, 3.0, 5.0, 6.0)).__next__,
        observability_provider=observed,
        rank_config_digest_provider=lambda digest, world_size: (digest, digest),
    )

    sealed = json.loads(evidence_path.read_text())
    assert sealed["kind"] == "stage1-producer-evidence"
    assert sealed["benchmark_evidence"]["optimizer_steps"] == 3
    assert sealed["benchmark_evidence"]["samples_seen"] == 96
    assert sealed["artifact_manifest"]["artifact_kind"] == "none"
    assert sealed["run_config"]["config_digest"] == config["config_digest"]
    assert reads.count(config_path.resolve()) == 1
    assert reads.count(audit_path.resolve()) == 1
    assert artifact_builds == 1
    assert not list(tmp_path.glob(".producer evidence.json.*.tmp"))


def test_run_stage1_rejects_runtime_or_rank_config_mismatch_before_build(tmp_path):
    """Catches model construction before exact/rank-consistent config preflight."""
    args = stage1_args(tmp_path, "--max_steps", "3", "--no_final_save", method="full")
    audit_path, audit = _write_benchmark_audit(tmp_path, args)
    config = {
        "schema_version": 1, "kind": "stage1-probe-run-manifest",
        "method": "full", "tuning_config": None, "model": args.model,
        "source_data_path": str(Path(args.data_path).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "data_stat": {
            "device": Path(args.data_path).stat().st_dev,
            "inode": Path(args.data_path).stat().st_ino,
            "size": Path(args.data_path).stat().st_size,
            "mtime_ns": Path(args.data_path).stat().st_mtime_ns,
        },
        "image_root": str(Path(args.image_root).resolve()),
        "output_dir": str(Path(args.save_model_path).resolve()),
        "micro_batch": 8, "world_size": 2, "gradient_accumulation_steps": 2,
        "effective_global_batch": 32, "epochs": 3, "learning_rate": 1e-5,
        "seed": 42, "max_steps": 3, "final_save": False,
        "deepspeed_config": str(Path(args.deepspeed_config).resolve()),
        "deepspeed_config_digest": hashlib.sha256(Path(args.deepspeed_config).read_bytes()).hexdigest(),
        "attention_implementation": "flash_attention_2", "save_strategy": "no",
        "audit_digest": audit["audit_digest"], "probe_rows": str(audit_path),
        "probe_row_indices": [7, 3], "maximum_config_digest": None,
        "prerequisite_chain": str(tmp_path / "prerequisite-chain.json"),
        "prerequisite_chain_digest": "f" * 64,
    }
    config["config_digest"] = config_digest(config)
    config_path = tmp_path / "run-config.json"
    config_path.write_text(json.dumps(config))
    args.benchmark_run_config = str(config_path)
    args.benchmark_evidence_path = str(tmp_path / "evidence.json")
    args.benchmark_source_data_path = config["source_data_path"]
    builds = 0

    def builder(selected):
        nonlocal builds
        builds += 1
        raise AssertionError("builder must not run")

    args.per_device_train_batch_size = 16
    with pytest.raises(ValueError, match="micro_batch"):
        run_stage1(
            args, component_builder=builder,
            rank_config_digest_provider=lambda digest, world_size: (digest, digest),
        )
    assert builds == 0

    args.per_device_train_batch_size = 8
    def config_mismatch(local_result, world_size):
        remote = dict(local_result)
        remote["config_digest"] = "f" * 64
        return (local_result, remote)

    with pytest.raises(ValueError, match="config digest"):
        run_stage1(
            args, component_builder=builder,
            rank_config_digest_provider=config_mismatch,
        )
    assert builds == 0

    forged = dict(config)
    forged["audit_digest"] = "e" * 64
    forged["config_digest"] = config_digest(
        {key: value for key, value in forged.items() if key != "config_digest"}
    )
    config_path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="audit rows"):
        run_stage1(
            args, component_builder=builder,
            rank_config_digest_provider=lambda digest, world_size: (digest, digest),
        )
    assert builds == 0


def test_distributed_preflight_collects_one_rank_runtime_error_before_uniform_failure(
    tmp_path,
):
    """Catches one rank exiting local validation while its peer hangs in gather."""
    args = stage1_args(tmp_path, "--max_steps", "3", "--no_final_save", method="full")
    config_path, config = _write_probe_run_config(tmp_path, args)
    args.benchmark_run_config = str(config_path)
    args.benchmark_evidence_path = str(tmp_path / "evidence.json")
    args.benchmark_source_data_path = config["source_data_path"]
    args.per_device_train_batch_size = 16
    gathered = []
    builds = 0

    def collective(local_result, world_size):
        gathered.append(local_result)
        if type(local_result) is not dict:
            return (local_result, local_result)
        remote_success = {
            "status": "success",
            "config_digest": config["config_digest"],
            "runtime_digest": "d" * 64,
            "error_type": None,
            "error_message": None,
        }
        return (local_result, remote_success)

    def builder(selected):
        nonlocal builds
        builds += 1
        raise AssertionError("builder must not run")

    with pytest.raises(ValueError, match="distributed preflight"):
        run_stage1(
            args,
            component_builder=builder,
            rank_config_digest_provider=collective,
        )
    assert len(gathered) == 1
    assert gathered[0]["status"] == "error"
    assert "micro_batch" in gathered[0]["error_message"]
    assert builds == 0


def test_distributed_preflight_collects_manifest_read_error_before_uniform_failure(
    tmp_path,
):
    """Catches a rank-local manifest read error bypassing the collective."""
    args = stage1_args(tmp_path, "--max_steps", "3", "--no_final_save", method="full")
    args.benchmark_run_config = str(tmp_path / "missing-run-config.json")
    args.benchmark_evidence_path = str(tmp_path / "evidence.json")
    args.benchmark_source_data_path = str(Path(args.data_path).resolve())
    gathered = []

    def collective(local_result, world_size):
        gathered.append(local_result)
        return (local_result, local_result)

    with pytest.raises(ValueError, match="distributed preflight"):
        run_stage1(
            args,
            component_builder=lambda selected: pytest.fail("builder must not run"),
            rank_config_digest_provider=collective,
        )
    assert len(gathered) == 1
    assert gathered[0]["status"] == "error"
    assert gathered[0]["error_type"] == "FileNotFoundError"


def test_distributed_preflight_rejects_successful_ranks_with_runtime_digest_mismatch(
    tmp_path,
):
    """Catches config-equal ranks proceeding with different validated runtime state."""
    args = stage1_args(tmp_path, "--max_steps", "3", "--no_final_save", method="full")
    config_path, config = _write_probe_run_config(tmp_path, args)
    args.benchmark_run_config = str(config_path)
    args.benchmark_evidence_path = str(tmp_path / "evidence.json")
    args.benchmark_source_data_path = config["source_data_path"]

    def collective(local_result, world_size):
        if type(local_result) is not dict:
            return (local_result, "e" * 64)
        remote = dict(local_result)
        remote["runtime_digest"] = "e" * 64
        return (local_result, remote)

    with pytest.raises(ValueError, match="runtime digest"):
        run_stage1(
            args,
            component_builder=lambda selected: pytest.fail("builder must not run"),
            rank_config_digest_provider=collective,
        )


def test_run_stage1_prevalidates_smoke_marker_and_embeds_it_in_producer_evidence(
    tmp_path,
):
    """Catches a resumed smoke publishing evidence without its immutable marker."""
    args = stage1_args(tmp_path, "--max_steps", "4", "--no_final_save", method="full")
    args.per_device_train_batch_size = 2
    checkpoint = Path(args.save_model_path).resolve() / "checkpoint-3"
    audit_path, audit = _write_benchmark_audit(tmp_path, args)
    config = {
        "schema_version": 1, "kind": "stage1-smoke-run-manifest",
        "method": "full", "tuning_config": None, "model": args.model,
        "source_data_path": str(Path(args.data_path).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "data_stat": audit["input_stat"],
        "image_root": str(Path(args.image_root).resolve()),
        "output_dir": str(Path(args.save_model_path).resolve()),
        "micro_batch": 2, "world_size": 2,
        "gradient_accumulation_steps": 2, "effective_global_batch": 8,
        "epochs": 3, "learning_rate": 1e-5, "seed": 42,
        "max_steps": 4, "final_save": False,
        "deepspeed_config": str(Path(args.deepspeed_config).resolve()),
        "deepspeed_config_digest": hashlib.sha256(
            Path(args.deepspeed_config).read_bytes()
        ).hexdigest(),
        "attention_implementation": "flash_attention_2", "save_strategy": "no",
        "audit_digest": audit["audit_digest"], "probe_rows": str(audit_path),
        "probe_row_indices": [7, 3], "maximum_config_digest": None,
        "prerequisite_chain": None, "prerequisite_chain_digest": None,
    }
    config["config_digest"] = config_digest(config)
    config_path = tmp_path / "smoke-run-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    marker = {
        "schema_version": 1, "kind": "stage1-smoke-checkpoint-topology",
        "method": "full", "artifact_kind": "full",
        "checkpoint_path": str(checkpoint), "initial_final_step": 3,
        "resumed_max_steps": 4, "micro_batch": 2, "world_size": 2,
        "checkpoint_validated": True, "trainer_state_digest": "a" * 64,
        "checkpoint_files": [{"name": "trainer_state.json", "size": 10}],
        "deepspeed_state_files": [],
    }
    marker["marker_digest"] = config_digest(marker)
    marker_path = tmp_path / "checkpoint-topology.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    args.benchmark_run_config = str(config_path)
    args.benchmark_evidence_path = str(evidence_path)
    args.benchmark_source_data_path = config["source_data_path"]
    args.benchmark_smoke_resume_witness = str(marker_path)
    args.resume_from_checkpoint_path = str(checkpoint)

    class SmokeTrainer(RecordingTrainer):
        def __init__(self):
            super().__init__(args=SimpleNamespace())
            self.state = SimpleNamespace(global_step=4)
            self._representative_gradient_norm = 0.25

        def train(self, *, resume_from_checkpoint):
            super().train(resume_from_checkpoint=resume_from_checkpoint)
            return SimpleNamespace(
                metrics={
                    "train_samples_per_second": 1.0,
                    "train_steps_per_second": 1.0,
                    "train_loss": 0.5,
                }
            )

    trainer = SmokeTrainer()
    bundle = SimpleNamespace(
        trainer=trainer,
        processor=RecordingProcessor(),
        dataset=[{}] * 8,
        parameter_report=SimpleNamespace(
            trainable_parameters=20, total_parameters=100
        ),
        resume_decision=Stage1ResumeDecision(
            checkpoint, Stage1ArtifactKind.FULL, True
        ),
    )

    run_stage1(
        args,
        component_builder=lambda selected: bundle,
        monotonic=iter((1.0, 2.0, 3.0, 5.0, 6.0)).__next__,
        observability_provider=lambda selected: {
            "rank_peak_allocated_bytes": (10, 11),
            "rank_peak_reserved_bytes": (12, 13),
            "gpu_names": ("B200", "B200"),
            "git_commit": "abc1234",
            "dependency_versions": {"torch": "test"},
        },
        rank_config_digest_provider=lambda local, world_size: (local, local),
    )

    sealed = json.loads(evidence_path.read_text())
    assert trainer.events == [("train", str(checkpoint))]
    assert sealed["run_config"] == config
    assert sealed["smoke_resume_witness"]["resume_artifact_digest"] == marker[
        "marker_digest"
    ]
