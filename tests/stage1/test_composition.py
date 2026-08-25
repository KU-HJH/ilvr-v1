from dataclasses import replace
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stage1.checkpointing import Stage1ArtifactKind, Stage1ResumeDecision
from stage1.composition import (
    DEFAULT_STAGE1_DEPENDENCIES,
    LATENT_TOKEN_STRINGS,
    build_stage1_training_components,
    run_stage1,
)
from utils_deepseed import parse_args


REPO_ROOT = Path(__file__).resolve().parents[2]


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
    return parse_args(
        [
            "--stage",
            "stage1",
            "--tuning_method",
            method,
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


def fake_dependencies(events, *, decision=None, processor=None, metadata=None):
    decision = decision or Stage1ResumeDecision(None, None, False)
    processor = processor or RecordingProcessor()
    config = SimpleNamespace(vocab_size=len(processor.tokenizer))
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
        load_peft_metadata=lambda path: metadata,
        load_peft_model=load_peft_model,
    ), processor, model, parameter_report


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


@pytest.mark.parametrize(
    ("kind", "method"),
    [(Stage1ArtifactKind.FULL, "full"), (Stage1ArtifactKind.PEFT, "dora")],
)
def test_resume_composition_preserves_checkpoint_token_ids_without_readding(
    tmp_path, kind, method
):
    """Catches Stage 1 resume re-adding or remapping latent vocabulary rows."""
    checkpoint = tmp_path / "output" / "checkpoint-9"
    decision = Stage1ResumeDecision(checkpoint, kind, True)
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
        checkpoint, Stage1ArtifactKind.PEFT, True
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
        checkpoint, Stage1ArtifactKind.PEFT, True
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
