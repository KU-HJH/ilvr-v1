import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from peft import PeftModel
from transformers import Trainer
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds

import main
import stage2.length_manifest as length_manifest_module
from .helpers import tiny_stage2_qwen
from main import (
    CustomTrainerStage1,
    Stage2DataCollator,
    Stage2Trainer,
    build_stage2_training_arguments,
    build_stage2_training_components,
    collate_fn_stage1,
    main_train,
    parse_args,
    route_training,
    select_training_path,
)
from stage2.checkpointing import (
    Stage2LoadDecision,
    build_stage2_peft_metadata,
    configure_stage2_metadata,
    resolve_stage2_load_decision,
    validate_stage2_peft_compatibility,
)
from stage2.peft import (
    Stage2TuningConfig,
    build_stage2_tuned_model,
    resolve_stage2_target_module_names,
)
from stage2.sampler import LatentBlockBalancedSampler


class DiagnosticStage2Model:
    def __init__(self):
        self.last_inputs = None
        self.diagnostics_requests = []

    def __call__(self, **inputs):
        self.last_inputs = inputs
        self.diagnostics_requests.append(inputs["collect_stage2_diagnostics"])
        return SimpleNamespace(
            loss=torch.tensor(2.0, requires_grad=True),
            stage2_diagnostics={
                "block_count": torch.tensor(2),
                "latent_l2_norms": torch.tensor(
                    [[3.0, 4.0]], requires_grad=True
                ),
                "adjacent_cosine_similarities": torch.tensor(
                    [[0.5]], requires_grad=True
                ),
            },
        )


def stage2_trainer_for_compute_loss(*, diagnostics_interval):
    trainer = object.__new__(Stage2Trainer)
    trainer.stage2_diagnostics_interval = diagnostics_interval
    trainer._stage2_microbatch_index = 0
    return trainer


def test_stage2_trainer_collects_detached_model_native_diagnostics():
    """Catches generic Trainer calls silently disabling Stage 2 observability."""
    trainer = stage2_trainer_for_compute_loss(diagnostics_interval=1)
    model = DiagnosticStage2Model()

    loss, outputs = trainer.compute_loss(
        model, {"input_ids": torch.ones(1, 3)}, return_outputs=True
    )

    assert loss.item() == 2.0
    assert outputs.loss is loss
    assert model.last_inputs["collect_stage2_diagnostics"] is True
    assert model.last_inputs["stage2_loss_only"] is False
    assert set(trainer._latest_stage2_diagnostics) == {
        "block_count",
        "latent_l2_norms",
        "adjacent_cosine_similarities",
    }
    assert all(
        not value.requires_grad
        for value in trainer._latest_stage2_diagnostics.values()
    )


def test_stage2_trainer_requests_loss_only_when_outputs_are_not_returned():
    """Catches ordinary training projecting full logits unnecessarily."""
    trainer = stage2_trainer_for_compute_loss(diagnostics_interval=0)
    model = DiagnosticStage2Model()

    loss = trainer.compute_loss(
        model, {"input_ids": torch.ones(1, 3)}, return_outputs=False
    )

    assert loss.item() == 2.0
    assert model.last_inputs["stage2_loss_only"] is True


def test_stage2_trainer_diagnostics_interval_schedules_microbatches():
    """Catches diagnostics running every batch or drifting from a zero-based cadence."""
    trainer = stage2_trainer_for_compute_loss(diagnostics_interval=2)
    model = DiagnosticStage2Model()

    for _ in range(5):
        trainer.compute_loss(model, {"input_ids": torch.ones(1, 3)})

    assert model.diagnostics_requests == [True, False, True, False, True]


def test_stage2_trainer_diagnostics_interval_zero_disables_diagnostics():
    """Catches the opt-in default still requesting diagnostic tensor work."""
    trainer = stage2_trainer_for_compute_loss(diagnostics_interval=0)
    model = DiagnosticStage2Model()

    for _ in range(3):
        trainer.compute_loss(model, {"input_ids": torch.ones(1, 3)})

    assert model.diagnostics_requests == [False, False, False]


def test_stage2_trainer_enriches_one_ordinary_log_then_clears_diagnostics(
    monkeypatch,
):
    """Catches detached Stage 2 diagnostics being dropped or replayed."""
    emitted = []
    monkeypatch.setattr(
        Trainer,
        "log",
        lambda self, logs, start_time=None: emitted.append((dict(logs), start_time)),
    )
    trainer = stage2_trainer_for_compute_loss(diagnostics_interval=0)
    trainer._latest_stage2_diagnostics = {
        "block_count": torch.tensor(2),
        "latent_l2_norms": torch.tensor([[3.0, 4.0]]),
        "adjacent_cosine_similarities": torch.tensor([[0.5]]),
    }

    trainer.log({"loss": 1.25}, start_time=7.0)
    trainer.log({"loss": 1.0})

    assert emitted == [
        (
            {
                "loss": 1.25,
                "stage2/latent_block_count": 2.0,
                "stage2/latent_l2_norm_mean": 3.5,
                "stage2/latent_l2_norm_min": 3.0,
                "stage2/latent_l2_norm_max": 4.0,
                "stage2/adjacent_cosine_mean": 0.5,
            },
            7.0,
        ),
        ({"loss": 1.0}, None),
    ]
    assert trainer._latest_stage2_diagnostics is None


def test_stage2_trainer_rejects_helper_image_tensors_before_model_call():
    """Catches Stage 1 helper-image tensors leaking past the Trainer boundary."""
    trainer = stage2_trainer_for_compute_loss(diagnostics_interval=0)
    model = DiagnosticStage2Model()

    with pytest.raises(ValueError, match="helper-image tensors leaked"):
        trainer.compute_loss(
            model,
            {
                "input_ids": torch.ones(1, 3),
                "pixel_values_latent": torch.ones(1, 3),
            },
        )

    assert model.last_inputs is None


def test_stage2_single_process_keeps_framework_train_sampler(monkeypatch):
    """Catches balanced distributed ordering leaking into one-process runs."""
    framework_sampler = object()
    monkeypatch.setattr(
        Trainer,
        "_get_train_sampler",
        lambda selected: framework_sampler,
    )
    trainer = object.__new__(Stage2Trainer)
    trainer.args = SimpleNamespace(world_size=1, seed=42)
    trainer.train_dataset = [SimpleNamespace(latent_block_count=1)]
    trainer.stage2_sampler_identity = "stale"

    sampler = trainer._get_train_sampler()

    assert sampler is framework_sampler
    assert not isinstance(sampler, LatentBlockBalancedSampler)
    assert trainer.stage2_sampler_identity is None


def test_stage_routing_selects_independent_trainers():
    """Catches either stage accidentally sharing the other's trainer/collator."""
    assert select_training_path("stage1").trainer_class is CustomTrainerStage1
    assert select_training_path("stage1").collator_factory is collate_fn_stage1
    assert select_training_path("stage2").trainer_class is Stage2Trainer
    assert select_training_path("stage2").collator_factory is Stage2DataCollator


def test_stage2_route_returns_the_selected_handler_result():
    """Catches Stage 2 dispatch entering the Stage 1 route or falling through."""
    args = SimpleNamespace(stage="stage2")
    observed = []

    result = route_training(
        args,
        stage1_handler=lambda selected: pytest.fail("Stage 1 route was selected"),
        stage2_handler=lambda selected: observed.append(selected) or "stage2-result",
    )

    assert result == "stage2-result"
    assert observed == [args]


def test_stage2_route_does_not_import_stage1_trainer_in_a_fresh_process():
    """Catches Stage 2 startup paying the Stage 1 Trainer import/construction cost."""
    repo = Path(__file__).resolve().parents[2]
    program = """
import sys
from types import SimpleNamespace
import main
result = main.route_training(
    SimpleNamespace(stage='stage2'),
    stage2_handler=lambda args: 'stage2-result',
)
assert result == 'stage2-result'
assert 'stage1.trainer' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_route_rejects_unsupported_stage():
    """Catches unknown stages silently falling into Stage 1."""
    with pytest.raises(ValueError, match="Unsupported stage: stage3"):
        route_training(SimpleNamespace(stage="stage3"))


def test_boundary_supervision_is_one_default_true_boolean():
    """Catches duplicate or inverted latent-boundary CLI switches."""
    assert parse_args(["--stage", "stage2"]).supervise_latent_boundaries is True
    assert (
        parse_args(["--stage", "stage2", "--no-supervise_latent_boundaries"])
        .supervise_latent_boundaries
        is False
    )


def test_stage2_diagnostics_interval_defaults_to_disabled():
    """Catches Stage 2 diagnostics becoming an implicit training cost."""
    assert parse_args(["--stage", "stage2"]).stage2_diagnostics_interval == 0


def test_ordinary_stage2_vision_attention_defaults_to_language_backend():
    """Catches the new vision option changing ordinary Stage 2 loader behavior."""
    defaults = parse_args(["--stage", "stage2"])
    eager = parse_args(
        ["--stage", "stage2", "--attention_implementation", "eager"]
    )
    mixed = parse_args(
        [
            "--stage",
            "stage2",
            "--attention_implementation",
            "sdpa",
            "--vision_attention_implementation",
            "flash_attention_2",
        ]
    )

    assert defaults.vision_attention_implementation == "sdpa"
    assert eager.vision_attention_implementation == "eager"
    assert mixed.vision_attention_implementation == "flash_attention_2"


@pytest.mark.parametrize("interval", [-1, -2])
def test_stage2_rejects_negative_diagnostics_interval(interval):
    """Catches negative diagnostic cadences reaching the trainer."""
    with pytest.raises(ValueError, match="diagnostics interval"):
        parse_args(
            [
                "--stage",
                "stage2",
                "--stage2_diagnostics_interval",
                str(interval),
            ]
        )


def test_fresh_stage2_tuning_defaults_to_full():
    """Catches fresh Stage 2 runs defaulting to a PEFT method."""
    args = parse_args(["--stage", "stage2"])

    assert args.tuning_method == "full"
    assert args.tuning_method_explicit is False


@pytest.mark.parametrize("method", ["full", "dora"])
def test_full_stage1_checkpoint_routes_to_fresh_full_or_dora(method):
    """Catches full Stage 1 sources being confused with Stage 2 resume state."""
    decision = resolve_stage2_load_decision(
        load_model_path="stage1/full-checkpoint",
        stage1_adapter_path=None,
        resume_from_checkpoint_path=None,
        load_config=lambda source: SimpleNamespace(stage="stage1"),
        discover_adapter_config=lambda *args, **kwargs: None,
        latent_size=4,
        supervise_latent_boundaries=True,
        tuning_method=method,
        tuning_method_explicit=True,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=("q_proj", "v_proj"),
    )

    assert decision.base_model_source == "stage1/full-checkpoint"
    assert decision.processor_source == "stage1/full-checkpoint"
    assert decision.adapter_source is None
    assert decision.trainer_resume_path is None
    assert decision.tuning.method == method


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_stage2_accepts_explicit_peft_methods(method):
    """Catches Stage 2 rejecting its explicit LoRA/DoRA selectors."""
    args = parse_args(["--stage", "stage2", "--tuning_method", method])

    assert args.tuning_method == method
    assert args.tuning_method_explicit is True


def test_stage2_resume_leaves_omitted_method_for_sidecar_resolution():
    """Catches resume parsing overwriting the checkpoint-sidecar selector."""
    args = parse_args(
        [
            "--stage",
            "stage2",
            "--resume_from_checkpoint_path",
            "checkpoint-7",
        ]
    )

    assert args.tuning_method is None
    assert args.tuning_method_explicit is False


def test_stage2_rejects_legacy_use_lora():
    """Catches legacy Stage 1 LoRA mode leaking into Stage 2."""
    with pytest.raises(ValueError, match="--tuning_method lora"):
        parse_args(["--stage", "stage2", "--use_lora"])


def test_stage1_cli_contract_does_not_gain_stage2_tuning():
    """Catches the Stage 2 selector changing the Stage 1 CLI contract."""
    args = parse_args(["--stage", "stage1", "--use_lora"])

    assert args.use_lora is True
    assert args.tuning_method is None


def test_stage2_rejects_multi_sample_local_batches():
    """Catches Stage 2 training violating its local B==1 requirement."""
    with pytest.raises(ValueError, match="requires per_device_train_batch_size=1"):
        parse_args(["--stage", "stage2", "--per_device_train_batch_size", "2"])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--lora_r", "0"], "LoRA rank and alpha must be positive"),
        (["--lora_alpha", "0"], "LoRA rank and alpha must be positive"),
        (["--lora_dropout", "nan"], "LoRA dropout must be finite"),
        (["--lora_dropout", "1"], "LoRA dropout must be finite"),
    ],
)
def test_stage2_rejects_invalid_peft_hyperparameters_during_parsing(argv, message):
    """Catches invalid PEFT settings reaching Stage 2 model construction."""
    with pytest.raises(ValueError, match=message):
        parse_args(["--stage", "stage2", *argv])


def test_stage1_cli_defaults_are_preserved():
    """Catches Stage 2 parser additions silently changing Stage 1 defaults."""
    args = parse_args(["--stage", "stage1"])

    assert args.model == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert args.epochs == 2
    assert args.gradient_accumulation_steps == 1
    assert args.save_model_path == "./checkpoints/model_stage1"
    assert args.use_lora is False


class RecordingStage1Tokenizer:
    token_ids = {
        "<|latent_pad|>": 90,
        "<|latent_start|>": 91,
        "<|latent_end|>": 92,
        "<|endoftext|>": 0,
    }

    def __call__(self, text, return_tensors):
        if text == "<|im_start|>assistant":
            return {"input_ids": torch.tensor([[31, 32]])}
        return {"input_ids": torch.tensor([[self.token_ids[text]]])}


class RecordingStage1Processor:
    def __init__(self):
        self.tokenizer = RecordingStage1Tokenizer()

    def apply_chat_template(self, example, tokenize):
        assert tokenize is False
        return "<image><|im_start|>assistant<output_image>"

    def __call__(self, *, text, images, return_tensors, padding):
        assert return_tensors == "pt"
        assert padding is True
        has_user_image = any(
            turn["role"] == "user"
            and any(item.get("type") == "image" for item in turn["content"])
            for turn in images[0]
        )
        has_assistant_image = any(
            turn["role"] == "assistant"
            and any(item.get("type") == "image" for item in turn["content"])
            for turn in images[0]
        )
        pixel_value = 1.0 if has_user_image and not has_assistant_image else 2.0
        return {
            "input_ids": torch.tensor([[10, 31, 32, 91, 777, 92, 7, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]]),
            "pixel_values": torch.tensor([pixel_value]),
            "image_grid_thw": torch.tensor([[int(pixel_value), 1, 1]]),
        }


def test_stage1_collator_contract_is_unchanged(monkeypatch):
    """Catches Stage 2 routing stripping Stage 1 helper tensors or labels."""
    example = [
        {"role": "user", "content": [{"type": "image", "image": "input.png"}]},
        {
            "role": "assistant",
            "content": [{"type": "image", "image": "helper.png"}],
        },
    ]
    monkeypatch.setattr(main, "process_vision_info", lambda examples: (examples, None))

    batch = collate_fn_stage1(
        [example],
        processor=RecordingStage1Processor(),
        args=SimpleNamespace(latent_size=2),
    )

    assert torch.equal(batch["pixel_values"], torch.tensor([1.0]))
    assert torch.equal(batch["image_grid_thw"], torch.tensor([[1, 1, 1]]))
    assert torch.equal(batch["pixel_values_latent"], torch.tensor([2.0]))
    assert torch.equal(batch["image_grid_thw_latent"], torch.tensor([[2, 1, 1]]))
    assert torch.equal(
        batch["labels"],
        torch.tensor([[-100, -100, -100, 91, -100, -100, 92, 7]]),
    )


class CompositionTokenizer:
    unk_token_id = 0
    unk_token = "<unk>"

    def __init__(self):
        self.ids = {
            "<|latent_pad|>": 90,
            "<|latent_start|>": 91,
            "<|latent_end|>": 92,
        }

    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert text == "<|im_start|>assistant"
        return {"input_ids": torch.tensor([[31, 32]])}

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)


class CompositionProcessor:
    def __init__(self):
        self.tokenizer = CompositionTokenizer()


def stage2_args(**overrides):
    values = dict(
        load_model_path="stage1/model",
        resume_from_checkpoint_path=None,
        attention_implementation="sdpa",
        vision_attention_implementation="sdpa",
        latent_size=4,
        supervise_latent_boundaries=True,
        data_path="data/train.jsonl",
        image_root="images",
        max_train_samples=None,
        max_seq_length_train=8192,
        save_model_path="out",
        epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=3,
        warm_up_steps=7,
        learning_rate=2e-5,
        save_steps=11,
        max_steps=13,
        gradient_checkpointing=True,
        deepspeed_config=None,
        seed=42,
        tuning_method="full",
        tuning_method_explicit=False,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules="q_proj,v_proj",
        stage2_diagnostics_interval=0,
        stage2_length_manifest=None,
        stage2_length_manifest_sha256=None,
        dataloader_num_workers=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--stage2_length_manifest", "manifest.json"], "must be provided together"),
        (["--stage2_length_manifest_sha256", "a" * 64], "must be provided together"),
        (["--dataloader_num_workers", "-1"], "must be non-negative"),
    ],
)
def test_stage2_cli_rejects_incomplete_length_manifest_or_negative_dataloader_num_workers(argv, message):
    """Catches Stage 2 accepting an unverifiable manifest or invalid workers."""
    with pytest.raises(ValueError, match=message):
        parse_args(["--stage", "stage2", *argv])


def test_stage2_length_manifest_rejects_wrong_sha_or_configuration(
    monkeypatch, tmp_path
):
    """Catches ranks accepting a different sealed file or a conflicting length policy."""
    import hashlib

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b"sealed manifest")
    data_path = tmp_path / "data.jsonl"
    data_path.write_bytes(b"[]\n")
    manifest = SimpleNamespace(
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        processor_sha256="processor-digest",
        latent_size=2,
        supervise_latent_boundaries=True,
        max_sequence_length=8,
        assistant_plan_rendering="concatenated-v1",
    )
    monkeypatch.setattr(
        length_manifest_module, "read_stage2_length_manifest", lambda path: manifest
    )
    monkeypatch.setattr(
        length_manifest_module, "_processor_sha256", lambda processor: "processor-digest"
    )

    with pytest.raises(ValueError, match="SHA-256"):
        length_manifest_module.validate_stage2_length_manifest(
            path=manifest_path,
            expected_sha256="0" * 64,
            data_path=data_path,
            image_root=tmp_path,
            processor=object(),
            latent_size=2,
            supervise_latent_boundaries=True,
            max_sequence_length=8,
        )
    with pytest.raises(ValueError, match="configuration"):
        length_manifest_module.validate_stage2_length_manifest(
            path=manifest_path,
            expected_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            data_path=data_path,
            image_root=tmp_path,
            processor=object(),
            latent_size=2,
            supervise_latent_boundaries=True,
            max_sequence_length=9,
        )


def _stub_stage2_data_pipeline(monkeypatch, calls):
    monkeypatch.setattr(main, "load_jsonl_examples", lambda path: [1, 2, 3])

    def prepare(item, *, image_root, latent_size, assistant_plan_rendering):
        calls.prepared.append(
            (item, image_root, latent_size, assistant_plan_rendering)
        )
        return item

    monkeypatch.setattr(main, "prepare_stage2_example", prepare)
    monkeypatch.setattr(
        main, "build_prepared_example_length_measure", lambda selected: "measure"
    )

    def filter_examples(*, examples, measure_length, max_length):
        calls.filtered = (examples, measure_length, max_length)
        return list(examples), SimpleNamespace(retained=len(examples))

    monkeypatch.setattr(main, "filter_stage2_examples", filter_examples)


@pytest.mark.parametrize(
    ("method", "expected_peft", "sample_limit", "expected_examples"),
    [
        ("full", False, None, [1, 2, 3]),
        ("lora", True, 2, [1, 2]),
        ("dora", True, None, [1, 2, 3]),
    ],
)
def test_stage2_components_compose_full_or_adapter_models_from_decision(
    monkeypatch, caplog, method, expected_peft, sample_limit, expected_examples
):
    """Catches source drift, late metadata setup, wrong wrapping, or provenance loss."""
    calls = SimpleNamespace(prepared=[], filtered=None, sources=[], wrap_states=[])
    processor = CompositionProcessor()
    tokenizer_ids_before = dict(processor.tokenizer.ids)
    model = tiny_stage2_qwen()
    model.config.stage = "stage1"
    model.config.supervise_latent_boundaries = True
    tuning = Stage2TuningConfig(
        method=method,
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=("q_proj", "v_proj"),
    )
    decision = Stage2LoadDecision(
        base_model_source=f"stage1/{method}-base",
        processor_source=f"stage1/{method}-processor",
        adapter_source=None,
        trainer_resume_path=None,
        tuning=tuning,
        latent_size=2,
        supervise_latent_boundaries=True,
    )
    monkeypatch.setattr(main, "resolve_stage2_load_decision", lambda **kwargs: decision)

    def load_processor(source):
        calls.sources.append(("processor", source))
        return processor

    def load_model(source, **kwargs):
        calls.sources.append(("base", source, kwargs))
        return model

    monkeypatch.setattr(main.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration, "from_pretrained", load_model
    )
    real_build = build_stage2_tuned_model

    def record_build(base, selected_tuning, *, adapter_path, is_trainable):
        calls.wrap_states.append(
            (
                base.config.stage,
                all(not parameter.requires_grad for parameter in base.visual.parameters()),
                adapter_path,
                is_trainable,
            )
        )
        return real_build(
            base,
            selected_tuning,
            adapter_path=adapter_path,
            is_trainable=is_trainable,
        )

    monkeypatch.setattr(main, "build_stage2_tuned_model", record_build, raising=False)
    _stub_stage2_data_pipeline(monkeypatch, calls)

    with caplog.at_level(logging.INFO):
        components = build_stage2_training_components(
            stage2_args(tuning_method=method, max_train_samples=sample_limit)
        )

    assert calls.sources == [
        ("processor", f"stage1/{method}-processor"),
        ("base", f"stage1/{method}-base", {"attn_implementation": "sdpa"}),
    ]
    assert calls.wrap_states == [("stage2", True, None, True)]
    assert isinstance(components.model, PeftModel) is expected_peft
    assert processor.tokenizer.ids == tokenizer_ids_before
    assert [item for item, _, _, _ in calls.prepared] == expected_examples
    assert all(
        str(root) == "images"
        and size == 2
        and rendering == "concatenated-v1"
        for _, root, size, rendering in calls.prepared
    )
    assert calls.filtered == (expected_examples, "measure", 8192)
    assert components.processor is processor
    assert components.train_dataset == expected_examples
    assert components.trainer_class is Stage2Trainer
    assert components.trainer_resume_path is None
    assert components.data_collator.token_ids == LatentTokenIds(90, 91, 92)
    assert components.data_collator.latent_size == 2
    assert components.data_collator.supervise_latent_boundaries is True
    assert model.config.stage == "stage2"
    assert model.config.latent_size == 2
    assert model.config.supervise_latent_boundaries is True
    assert model.config.use_cache is False
    assert model.is_gradient_checkpointing is True
    assert model.model._gradient_checkpointing_func.keywords == {
        "use_reentrant": False
    }
    assert all(not parameter.requires_grad for parameter in model.visual.parameters())
    assert "resolved_targets=" in caplog.text
    assert "model.layers.0.self_attn.q_proj" in caplog.text
    assert "model.layers.0.self_attn.v_proj" in caplog.text
    if method == "full":
        assert components.peft_metadata is None
    else:
        assert components.peft_metadata.base_model_reference == f"stage1/{method}-base"
        assert components.peft_metadata.tuning_method == method
        assert components.peft_metadata.requested_target_modules == (
            "q_proj",
            "v_proj",
        )
        assert components.peft_metadata.resolved_target_module_names == (
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
        )


def test_stage2_components_route_distinct_vision_backend_through_shared_loader(
    monkeypatch,
):
    """Catches Stage 2 flattening a requested FlashAttention2 vision backend."""
    processor = CompositionProcessor()
    model = tiny_stage2_qwen(attn_implementation="sdpa")
    model.config.stage = "stage1"
    model.config.supervise_latent_boundaries = True
    decision = Stage2LoadDecision(
        base_model_source="stage1/base",
        processor_source="stage1/processor",
        adapter_source=None,
        trainer_resume_path=None,
        tuning=Stage2TuningConfig(
            method="full", r=4, alpha=8, dropout=0.0, target_modules=("q_proj",)
        ),
        latent_size=2,
        supervise_latent_boundaries=True,
    )
    monkeypatch.setattr(main, "resolve_stage2_load_decision", lambda **_kwargs: decision)
    calls = SimpleNamespace(prepared=[], filtered=None, mixed=[], validated=[])
    _stub_stage2_data_pipeline(monkeypatch, calls)

    load_config = lambda *_args, **_kwargs: model.config
    load_model = lambda *_args, **_kwargs: pytest.fail(
        "direct one-backend loader must not run"
    )
    dependencies = main.Stage2SourceDependencies(
        find_adapter_config=lambda *_args, **_kwargs: None,
        load_config=load_config,
        load_processor=lambda _source, **_kwargs: processor,
        load_model=load_model,
    )

    def load_mixed(source, **kwargs):
        calls.mixed.append((source, kwargs))
        return model

    monkeypatch.setattr(main, "load_qwen2_5_vl_with_attention", load_mixed)
    monkeypatch.setattr(
        main,
        "validate_qwen2_5_vl_attention_backends",
        lambda selected, **kwargs: calls.validated.append((selected, kwargs)),
    )

    components = build_stage2_training_components(
        stage2_args(vision_attention_implementation="flash_attention_2"),
        source_dependencies=dependencies,
    )

    assert components.model is model
    assert calls.mixed == [
        (
            "stage1/base",
            {
                "attention_implementation": "sdpa",
                "vision_attention_implementation": "flash_attention_2",
                "config_load_kwargs": {},
                "model_load_kwargs": {},
                "load_config": load_config,
                "load_model": load_model,
            },
        )
    ]
    assert calls.validated == [
        (
            model,
            {
                "attention_implementation": "sdpa",
                "vision_attention_implementation": "flash_attention_2",
            },
        )
    ]


def test_stage2_components_select_length_manifest_indices_without_remeasuring_images(
    monkeypatch, tmp_path
):
    """Catches each rank repeating image-aware length filtering after a sealed audit."""
    calls = SimpleNamespace(prepared=[], filtered=None, sources=[])
    processor = CompositionProcessor()
    model = tiny_stage2_qwen()
    model.config.supervise_latent_boundaries = True
    decision = Stage2LoadDecision(
        base_model_source="stage1/base",
        processor_source="stage1/processor",
        adapter_source=None,
        trainer_resume_path=None,
        tuning=Stage2TuningConfig(
            method="full", r=4, alpha=8, dropout=0.0, target_modules=("q_proj",)
        ),
        latent_size=2,
        supervise_latent_boundaries=True,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b"sealed manifest")
    monkeypatch.setattr(main, "resolve_stage2_load_decision", lambda **kwargs: decision)
    monkeypatch.setattr(main.AutoProcessor, "from_pretrained", lambda source: processor)
    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda source, **kwargs: model,
    )
    _stub_stage2_data_pipeline(monkeypatch, calls)
    monkeypatch.setattr(
        main,
        "build_prepared_example_length_measure",
        lambda processor: pytest.fail("manifest path must not remeasure image-aware lengths"),
    )
    monkeypatch.setattr(
        main,
        "validate_stage2_length_manifest",
        lambda **kwargs: SimpleNamespace(
            manifest=SimpleNamespace(retained_indices=(2, 0)),
            raw_examples=(1, 2, 3),
            prepared_examples=(1, 2, 3),
        ),
        raising=False,
    )

    components = build_stage2_training_components(
        stage2_args(
            stage2_length_manifest=str(manifest_path),
            stage2_length_manifest_sha256=__import__("hashlib").sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        )
    )

    assert calls.filtered is None
    assert components.train_dataset == [3, 1]


@pytest.mark.parametrize(
    ("sample_limit", "expected_examples"),
    [(2, [1]), (3, [1, 3])],
)
def test_stage2_manifest_indices_respect_max_train_samples(
    monkeypatch, tmp_path, sample_limit, expected_examples
):
    """Catches retained raw indices outside a requested prefix indexing the prepared data."""
    processor = CompositionProcessor()
    model = tiny_stage2_qwen()
    model.config.supervise_latent_boundaries = True
    decision = Stage2LoadDecision(
        base_model_source="stage1/base",
        processor_source="stage1/processor",
        adapter_source=None,
        trainer_resume_path=None,
        tuning=Stage2TuningConfig(
            method="full", r=4, alpha=8, dropout=0.0, target_modules=("q_proj",)
        ),
        latent_size=2,
        supervise_latent_boundaries=True,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b"sealed manifest")
    monkeypatch.setattr(main, "resolve_stage2_load_decision", lambda **kwargs: decision)
    monkeypatch.setattr(main.AutoProcessor, "from_pretrained", lambda source: processor)
    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda source, **kwargs: model,
    )
    monkeypatch.setattr(
        main,
        "load_jsonl_examples",
        lambda path: pytest.fail("manifest path must consume its sealed data snapshot"),
    )
    monkeypatch.setattr(
        main,
        "validate_stage2_length_manifest",
        lambda **kwargs: SimpleNamespace(
            manifest=SimpleNamespace(retained_indices=(0, 2)),
            raw_examples=(1, 2, 3),
            prepared_examples=(1, 2, 3),
        ),
        raising=False,
    )

    components = build_stage2_training_components(
        stage2_args(
            max_train_samples=sample_limit,
            stage2_length_manifest=str(manifest_path),
            stage2_length_manifest_sha256=__import__("hashlib").sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        )
    )

    assert components.train_dataset == expected_examples


def test_stage2_peft_resume_validates_stage1_before_mutation_and_preserves_sidecar(
    monkeypatch, tmp_path
):
    """Catches compatibility validation being hidden by Stage 2 mutation or relocation."""
    processor = CompositionProcessor()
    tokenizer_ids_before = dict(processor.tokenizer.ids)
    tuning = Stage2TuningConfig(
        method="lora",
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=("q_proj", "v_proj"),
    )
    adapter_base = tiny_stage2_qwen()
    adapter_base.config.supervise_latent_boundaries = True
    resolved = resolve_stage2_target_module_names(adapter_base, tuning.target_modules)
    adapter = build_stage2_tuned_model(adapter_base, tuning)
    with pytest.warns(UserWarning, match="Could not find a config file"):
        adapter.save_pretrained(tmp_path)
    metadata = build_stage2_peft_metadata(
        tuning=tuning,
        base_model_reference="original/stage1-base",
        model_config=adapter_base.config,
        token_ids=LatentTokenIds(90, 91, 92),
        resolved_target_names=resolved,
    )
    decision = Stage2LoadDecision(
        base_model_source="relocated/stage1-base",
        processor_source="adapter/checkpoint-7",
        adapter_source=str(tmp_path),
        trainer_resume_path=str(tmp_path),
        tuning=tuning,
        latent_size=2,
        supervise_latent_boundaries=True,
        peft_metadata=metadata,
    )
    resumed_base = tiny_stage2_qwen()
    resumed_base.config.stage = "stage1"
    resumed_base.config.supervise_latent_boundaries = True
    events = []
    monkeypatch.setattr(main, "resolve_stage2_load_decision", lambda **kwargs: decision)
    monkeypatch.setattr(main.AutoProcessor, "from_pretrained", lambda source: processor)
    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda source, **kwargs: resumed_base,
    )
    real_validate = validate_stage2_peft_compatibility
    real_configure = configure_stage2_metadata

    def record_validate(base, tokenizer, selected_metadata, adapter_config, adapter_path):
        events.append(("validate", base.config.stage, selected_metadata, adapter_path))
        return real_validate(
            base, tokenizer, selected_metadata, adapter_config, adapter_path
        )

    def record_configure(config, token_ids, **kwargs):
        events.append(("configure", config.stage))
        return real_configure(config, token_ids, **kwargs)

    monkeypatch.setattr(
        main, "validate_stage2_peft_compatibility", record_validate, raising=False
    )
    monkeypatch.setattr(main, "configure_stage2_metadata", record_configure)
    calls = SimpleNamespace(prepared=[], filtered=None)
    _stub_stage2_data_pipeline(monkeypatch, calls)

    components = build_stage2_training_components(
        stage2_args(
            load_model_path="relocated/stage1-base",
            resume_from_checkpoint_path=str(tmp_path),
            tuning_method=None,
        )
    )

    assert events == [
        ("validate", "stage1", metadata, str(tmp_path)),
        ("configure", "stage1"),
    ]
    assert isinstance(components.model, PeftModel)
    assert components.peft_metadata is metadata
    assert components.peft_metadata.base_model_reference == "original/stage1-base"
    assert processor.tokenizer.ids == tokenizer_ids_before
    assert components.trainer_resume_path == str(tmp_path)


def test_stage2_training_arguments_create_fresh_optimizer_configuration():
    """Catches Stage 1 optimizer state or truncating data settings entering Stage 2."""
    training_args = build_stage2_training_arguments(stage2_args())

    assert training_args.output_dir == "out"
    assert training_args.num_train_epochs == 2
    assert training_args.per_device_train_batch_size == 1
    assert training_args.gradient_accumulation_steps == 3
    assert training_args.warmup_steps == 7
    assert training_args.learning_rate == 2e-5
    assert training_args.optim.value == "adamw_torch"
    assert training_args.remove_unused_columns is False
    assert training_args.max_seq_length == 8192
    assert training_args.max_steps == 13
    assert training_args.gradient_checkpointing is True
    assert training_args.gradient_checkpointing_kwargs == {"use_reentrant": False}


@pytest.mark.parametrize("worker_count", (0, 2))
def test_stage2_training_arguments_set_requested_dataloader_workers(worker_count):
    """Catches Stage 2 silently falling back to an implicit data-loader worker count."""
    training_args = build_stage2_training_arguments(
        stage2_args(dataloader_num_workers=worker_count)
    )

    assert training_args.dataloader_num_workers == worker_count
    assert training_args.dataloader_prefetch_factor == (
        None if worker_count == 0 else 2
    )


def test_stage2_training_arguments_omit_checkpoint_kwargs_when_disabled():
    """Catches inactive checkpointing carrying mode-specific runtime kwargs."""
    training_args = build_stage2_training_arguments(
        stage2_args(gradient_checkpointing=False)
    )

    assert training_args.gradient_checkpointing is False
    assert training_args.gradient_checkpointing_kwargs is None


def test_stage2_peft_zero3_rejects_gradient_checkpointing():
    """Catches the disproven repeated reentrant rollout path being selected."""
    args = stage2_args(
        tuning_method="lora",
        deepspeed_config=str(
            Path(main.__file__).resolve().parents[1]
            / "configs"
            / "deepspeed_zero3.json"
        ),
    )

    model = tiny_stage2_qwen()
    profile = main.bind_deepspeed_runtime_profile(args)
    with pytest.raises(ValueError, match="requires gradient checkpointing disabled"):
        main.configure_stage2_model_backend(
            model,
            gradient_checkpointing=True,
            tuning_method="lora",
            deepspeed_runtime_profile=profile,
        )
    with pytest.raises(ValueError, match="requires gradient checkpointing disabled"):
        build_stage2_training_arguments(args)


class RecordingTrainer:
    instance = None
    world_process_zero = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.resume_path = None
        self.saved_path = None
        type(self).instance = self

    def train(self, *, resume_from_checkpoint):
        self.resume_path = resume_from_checkpoint

    def save_model(self, path):
        self.saved_path = path

    def is_world_process_zero(self):
        return self.world_process_zero


class RecordingSavedProcessor:
    def __init__(self):
        self.saved_path = None

    def save_pretrained(self, path):
        self.saved_path = path


def test_main_keeps_stage2b_preflight_out_of_startup_log_and_reuses_snapshot(
    monkeypatch,
):
    """Catches validated prompts and rejections leaking through argparse logging."""
    secret_prompt = "stage2b-secret-prompt-7bbf85"
    secret_rejected = "stage2b-secret-rejected-40a32c"
    preflight = main.Stage2BPreflight(
        prepared_examples=(SimpleNamespace(prompt=secret_prompt),),
        negative_store={
            ("sample-1", 0): SimpleNamespace(rejected_text=secret_rejected)
        },
    )
    args = stage2_args(
        stage="stage2",
        log_file="unused.log",
        step_preference_config=main.StepPreferenceConfig(
            enabled=True,
            mode="stage2b",
            objective="legacy_dpo",
            stage2a_checkpoint_path="stage2a/checkpoint",
            negative_path="negatives.jsonl",
            negative_manifest="negatives.manifest.json",
        ),
    )
    logged_records = []
    routed_preflights = []

    monkeypatch.setattr(main, "get_args", lambda: args)
    monkeypatch.setattr(main, "validate_generated_artifact_paths", lambda selected: None)
    monkeypatch.setattr(main, "preflight_stage2b_artifacts", lambda selected: preflight)
    monkeypatch.setattr(main, "bind_deepspeed_runtime_profile", lambda selected: None)
    monkeypatch.setattr(main, "seed_everything", lambda **kwargs: None)
    monkeypatch.setattr(main.logging, "FileHandler", lambda *args, **kwargs: object())
    monkeypatch.setattr(main.logging, "StreamHandler", lambda *args, **kwargs: object())
    monkeypatch.setattr(main.logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(
        main.logging, "info", lambda value: logged_records.append(str(value))
    )

    def record_stage2(selected):
        routed_preflights.append(selected._stage2b_preflight)
        return "stage2b-result"

    monkeypatch.setattr(main, "run_stage2", record_stage2)

    result = main_train()

    startup_log = "\n".join(logged_records)
    assert result == "stage2b-result"
    assert routed_preflights == [preflight]
    assert routed_preflights[0] is preflight
    assert secret_prompt not in startup_log
    assert secret_rejected not in startup_log
    assert "_stage2b_preflight" not in startup_log


def test_stage2b_preflight_binds_space_lf_rendering_to_prepare_and_negative_load(
    monkeypatch, tmp_path
):
    """Catches Stage 2B preparing or authenticating a different assistant stream."""
    source = tmp_path / "data.jsonl"
    source.write_bytes(b'{"sample_id":"sample-a"}\n')
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    config = main.StepPreferenceConfig(
        enabled=True,
        mode="stage2b",
        objective="legacy_dpo",
        stage2a_checkpoint_path=str(tmp_path / "stage2a"),
        negative_path=str(tmp_path / "negatives.jsonl"),
        negative_manifest=str(tmp_path / "manifest.json"),
    )
    args = stage2_args(
        data_path=str(source),
        image_root=str(tmp_path / "images"),
        latent_size=8,
        max_train_samples=None,
        step_preference_config=config,
        vision_attention_implementation="flash_attention_2",
    )
    manifest = SimpleNamespace(
        source_sha256=source_sha256,
        generation_settings_dict={"min_tokens": 2, "max_new_tokens": 32},
    )
    prepared = SimpleNamespace(
        sample_id="sample-a",
        reasoning_steps=(
            SimpleNamespace(step_id=0, target_text=SimpleNamespace(start=0, end=6)),
        ),
        conversation=[
            {"role": "user", "content": []},
            {"role": "assistant", "content": [{"type": "text", "text": "chosen"}]},
        ],
    )
    prepare_calls = []
    load_calls = []
    monkeypatch.setattr(main, "read_negative_manifest", lambda _path: manifest)
    monkeypatch.setattr(main, "load_jsonl_examples", lambda _path: [object()])

    def prepare(item, **kwargs):
        prepare_calls.append((item, kwargs))
        return prepared

    monkeypatch.setattr(main, "prepare_stage2_example", prepare)
    monkeypatch.setattr(main, "validate_stage2b_peft_base_identity", lambda **_kwargs: None)
    monkeypatch.setattr(
        main,
        "inspect_stage2a_checkpoint_identity",
        lambda _path: ((), "tokenizer-v1"),
    )

    def load(**kwargs):
        load_calls.append(kwargs)
        return {("sample-a", 0): SimpleNamespace(chosen_text="chosen")}

    monkeypatch.setattr(main, "load_negative_store", load)

    result = main.preflight_stage2b_artifacts(args)

    assert result.prepared_examples == (prepared,)
    assert prepare_calls[0][1]["assistant_plan_rendering"] == (
        "adjacent-text-space-lf-v1"
    )
    assert load_calls[0]["expected_assistant_plan_rendering"] == (
        "adjacent-text-space-lf-v1"
    )
    assert load_calls[0]["expected_generation_settings"] == {
        "attention_implementation": "sdpa",
        "vision_attention_implementation": "flash_attention_2",
        "device": "cuda:0",
        "max_new_tokens": 32,
        "min_tokens": 2,
        "torch_dtype": "bfloat16",
        "seed": 42,
        "decoding": "greedy-argmax-v1",
    }


def test_direct_stage1_preflight_uses_fresh_dora_resolution_without_stage2a(
    monkeypatch, tmp_path
):
    """Catches direct V-only Stage 2B falling back to Stage 2A initialization."""
    source = tmp_path / "data.jsonl"
    source.write_bytes(b'{"sample_id":"sample-a"}\n')
    config = main.StepPreferenceConfig(
        enabled=True,
        mode="stage2b",
        objective="context_preference",
        visual_only_context_preference=True,
        direct_from_stage1=True,
    )
    args = stage2_args(
        data_path=str(source),
        image_root=str(tmp_path / "images"),
        stage1_adapter_path="stage1-adapter",
        step_preference_config=config,
        vision_attention_implementation="flash_attention_2",
    )
    prepared = SimpleNamespace(sample_id="sample-a", reasoning_steps=())
    decisions = []
    monkeypatch.setattr(main, "load_jsonl_examples", lambda _path: [object()])
    monkeypatch.setattr(main, "prepare_stage2_example", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        main,
        "resolve_stage2_load_decision",
        lambda **kwargs: decisions.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        main,
        "validate_stage2b_peft_base_identity",
        lambda **_kwargs: pytest.fail("direct mode used Stage 2A validation"),
    )

    result = main.preflight_stage2b_artifacts(args)

    assert result.prepared_examples == (prepared,)
    assert result.negative_store is None
    assert decisions[0]["resume_from_checkpoint_path"] is None
    assert decisions[0]["stage1_adapter_path"] == "stage1-adapter"


@pytest.mark.parametrize(
    ("attention_implementation", "vision_attention_implementation"),
    (("eager", "eager"), ("sdpa", "sdpa")),
)
def test_stage2b_preflight_rejects_noncanonical_attention_before_source_read(
    tmp_path,
    attention_implementation,
    vision_attention_implementation,
):
    """Catches Stage 2B authenticating a mutually matching noncanonical manifest."""
    config = main.StepPreferenceConfig(
        enabled=True,
        mode="stage2b",
        objective="legacy_dpo",
        stage2a_checkpoint_path=str(tmp_path / "stage2a"),
        negative_path=str(tmp_path / "negatives.jsonl"),
        negative_manifest=str(tmp_path / "manifest.json"),
    )
    args = stage2_args(
        data_path=str(tmp_path / "must-not-be-read.jsonl"),
        step_preference_config=config,
        attention_implementation=attention_implementation,
        vision_attention_implementation=vision_attention_implementation,
    )

    with pytest.raises(ValueError, match="Stage 2B requires"):
        main.preflight_stage2b_artifacts(args)


@pytest.mark.parametrize(
    ("attention_implementation", "vision_attention_implementation"),
    (("eager", "eager"), ("sdpa", "sdpa")),
)
def test_stage2b_components_reject_noncanonical_attention_with_cached_preflight(
    tmp_path,
    attention_implementation,
    vision_attention_implementation,
):
    """Catches an injected preflight bypass reaching configuration/model loading."""
    config = main.StepPreferenceConfig(
        enabled=True,
        mode="stage2b",
        objective="legacy_dpo",
        stage2a_checkpoint_path=str(tmp_path / "stage2a"),
        negative_path=str(tmp_path / "negatives.jsonl"),
        negative_manifest=str(tmp_path / "manifest.json"),
    )
    args = stage2_args(
        step_preference_config=config,
        attention_implementation=attention_implementation,
        vision_attention_implementation=vision_attention_implementation,
        _stage2b_preflight=main.Stage2BPreflight((), {}),
    )
    dependencies = main.Stage2SourceDependencies(
        find_adapter_config=lambda *_args, **_kwargs: pytest.fail(
            "artifact discovery must not start"
        ),
        load_config=lambda *_args, **_kwargs: pytest.fail(
            "configuration loading must not start"
        ),
        load_processor=lambda *_args, **_kwargs: pytest.fail(
            "processor loading must not start"
        ),
        load_model=lambda *_args, **_kwargs: pytest.fail(
            "model loading must not start"
        ),
    )

    with pytest.raises(ValueError, match="Stage 2B requires"):
        main.build_stage2_training_components(args, source_dependencies=dependencies)


def test_stage2b_preflight_rejects_changed_vision_backend_before_model_load(
    monkeypatch, tmp_path
):
    """Catches trusting a schema-3 manifest's producer-selected vision backend."""
    source = tmp_path / "data.jsonl"
    source.write_bytes(b'{"sample_id":"sample-a"}\n')
    stage2a = tmp_path / "stage2a"
    stage2a.mkdir()
    negatives = tmp_path / "negatives.jsonl"
    negatives.write_bytes(b"")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "selected_row_indices": [0],
                "stage2a_checkpoint_path": str(stage2a.resolve()),
                "checkpoint_files": [],
                "tokenizer_config_identity": "tokenizer-v1",
                "latent_size": 8,
                "assistant_plan_rendering": "adjacent-text-space-lf-v1",
                "generation_settings": {
                    "attention_implementation": "sdpa",
                    "vision_attention_implementation": "sdpa",
                    "device": "cuda:0",
                    "max_new_tokens": 32,
                    "min_tokens": 2,
                    "torch_dtype": "bfloat16",
                },
                "negative_sha256": hashlib.sha256(b"").hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    config = main.StepPreferenceConfig(
        enabled=True,
        mode="stage2b",
        objective="legacy_dpo",
        stage2a_checkpoint_path=str(stage2a),
        negative_path=str(negatives),
        negative_manifest=str(manifest_path),
    )
    args = stage2_args(
        data_path=str(source),
        image_root=str(tmp_path / "images"),
        latent_size=8,
        step_preference_config=config,
        vision_attention_implementation="flash_attention_2",
    )
    prepared = SimpleNamespace(
        sample_id="sample-a",
        reasoning_steps=(
            SimpleNamespace(step_id=0, target_text=SimpleNamespace(start=0, end=6)),
        ),
        conversation=[
            {"role": "user", "content": []},
            {"role": "assistant", "content": [{"type": "text", "text": "chosen"}]},
        ],
    )
    monkeypatch.setattr(main, "load_jsonl_examples", lambda _path: [object()])
    monkeypatch.setattr(
        main, "prepare_stage2_example", lambda *_args, **_kwargs: prepared
    )
    monkeypatch.setattr(
        main, "validate_stage2b_peft_base_identity", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        main,
        "inspect_stage2a_checkpoint_identity",
        lambda _path: ((), "tokenizer-v1"),
    )
    dependencies = main.Stage2SourceDependencies(
        find_adapter_config=lambda *_args, **_kwargs: None,
        load_config=lambda *_args, **_kwargs: pytest.fail("config load must not start"),
        load_processor=lambda *_args, **_kwargs: pytest.fail(
            "processor load must not start"
        ),
        load_model=lambda *_args, **_kwargs: pytest.fail("model load must not start"),
    )

    with pytest.raises(ValueError, match="generation settings do not match"):
        main.build_stage2_training_components(args, source_dependencies=dependencies)


def test_main_stage2b_rejects_legacy_negative_manifest_before_model_construction(
    monkeypatch, tmp_path
):
    """Catches enabled main routing accepting a schema-2 rendering-less artifact."""
    source = tmp_path / "data.jsonl"
    source.write_bytes(b'{"sample_id":"sample-a"}\n')
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "selected_row_indices": [0],
                "stage2a_checkpoint_path": str((tmp_path / "stage2a").resolve()),
                "checkpoint_files": [],
                "tokenizer_config_identity": "tokenizer-v1",
                "latent_size": 8,
                "generation_settings": {"min_tokens": 2, "max_new_tokens": 32},
                "negative_sha256": "0" * 64,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    config = main.StepPreferenceConfig(
        enabled=True,
        mode="stage2b",
        objective="legacy_dpo",
        stage2a_checkpoint_path=str(tmp_path / "stage2a"),
        negative_path=str(tmp_path / "negatives.jsonl"),
        negative_manifest=str(manifest_path),
    )
    args = stage2_args(
        stage="stage2",
        data_path=str(source),
        image_root=str(tmp_path / "images"),
        latent_size=8,
        log_file=str(tmp_path / "train.log"),
        step_preference_config=config,
        vision_attention_implementation="flash_attention_2",
    )
    prepared = SimpleNamespace(
        sample_id="sample-a",
        reasoning_steps=(
            SimpleNamespace(step_id=0, target_text=SimpleNamespace(start=0, end=6)),
        ),
        conversation=[
            {"role": "user", "content": []},
            {"role": "assistant", "content": [{"type": "text", "text": "chosen"}]},
        ],
    )
    monkeypatch.setattr(main, "get_args", lambda: args)
    monkeypatch.setattr(main, "validate_generated_artifact_paths", lambda _args: None)
    monkeypatch.setattr(main, "load_jsonl_examples", lambda _path: [object()])
    monkeypatch.setattr(main, "prepare_stage2_example", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(main, "validate_stage2b_peft_base_identity", lambda **_kwargs: None)
    monkeypatch.setattr(
        main,
        "inspect_stage2a_checkpoint_identity",
        lambda _path: ((), "tokenizer-v1"),
    )
    monkeypatch.setattr(
        main,
        "load_negative_store",
        lambda **_kwargs: {("sample-a", 0): SimpleNamespace(chosen_text="chosen")},
    )
    monkeypatch.setattr(main, "bind_deepspeed_runtime_profile", lambda _args: None)
    monkeypatch.setattr(main, "seed_everything", lambda **_kwargs: None)
    monkeypatch.setattr(main.logging, "FileHandler", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(main.logging, "StreamHandler", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(main.logging, "basicConfig", lambda **_kwargs: None)
    monkeypatch.setattr(
        main,
        "run_stage2",
        lambda _args: pytest.fail("model construction reached for legacy artifact"),
    )

    with pytest.raises(ValueError, match="unknown or missing fields|schema_version"):
        main_train()


def test_main_disabled_startup_log_has_no_private_preflight_field(monkeypatch):
    """Catches disabled startup logs gaining a private ``None`` payload field."""
    args = stage2_args(
        stage="stage2",
        log_file="unused.log",
        step_preference_config=main.StepPreferenceConfig(),
    )
    logged_records = []

    monkeypatch.setattr(main, "get_args", lambda: args)
    monkeypatch.setattr(main, "validate_generated_artifact_paths", lambda selected: None)
    monkeypatch.setattr(
        main,
        "preflight_stage2b_artifacts",
        lambda selected: pytest.fail("disabled training must not run Stage 2B preflight"),
    )
    monkeypatch.setattr(main, "bind_deepspeed_runtime_profile", lambda selected: None)
    monkeypatch.setattr(main, "seed_everything", lambda **kwargs: None)
    monkeypatch.setattr(main.logging, "FileHandler", lambda *args, **kwargs: object())
    monkeypatch.setattr(main.logging, "StreamHandler", lambda *args, **kwargs: object())
    monkeypatch.setattr(main.logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(
        main.logging, "info", lambda value: logged_records.append(str(value))
    )
    monkeypatch.setattr(main, "run_stage2", lambda selected: "stage2-result")

    result = main_train()

    assert result == "stage2-result"
    assert "_stage2b_preflight" not in "\n".join(logged_records)


@pytest.mark.parametrize(
    ("world_process_zero", "expected_processor_path"),
    [(True, "out"), (False, None)],
)
def test_main_stage2_uses_only_stage2_components_and_validated_resume(
    monkeypatch, tmp_path, world_process_zero, expected_processor_path
):
    """Catches Stage 2 falling through the Stage 1 trainer or auto-resume path."""
    args = stage2_args(
        stage="stage2",
        seed=99,
        log_file=str(tmp_path / "train.log"),
        cache_dir=str(tmp_path / "cache"),
        save_final_model=True,
        stage2_diagnostics_interval=3,
    )
    processor = RecordingSavedProcessor()
    model = object()
    dataset = ["sample"]
    collator = object()
    components = main.Stage2TrainingComponents(
        model=model,
        processor=processor,
        train_dataset=dataset,
        data_collator=collator,
        trainer_class=RecordingTrainer,
        trainer_resume_path="stage2/checkpoint-4",
        peft_metadata="preserved-sidecar",
    )
    training_args = object()
    RecordingTrainer.world_process_zero = world_process_zero
    monkeypatch.setattr(main, "get_args", lambda: args)
    monkeypatch.setattr(main, "validate_generated_artifact_paths", lambda selected: None)
    monkeypatch.setattr(main, "build_stage2_training_components", lambda selected: components)
    monkeypatch.setattr(main, "build_stage2_training_arguments", lambda selected: training_args)
    monkeypatch.setattr(
        main.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("Stage 1 model path was entered"),
    )

    main_train()

    trainer = RecordingTrainer.instance
    assert trainer.kwargs == {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "data_collator": collator,
        "processing_class": processor,
        "stage2_peft_metadata": "preserved-sidecar",
        "deepspeed_runtime_profile": args.deepspeed_runtime_profile,
        "stage2_diagnostics_interval": 3,
    }
    assert trainer.resume_path == "stage2/checkpoint-4"
    assert trainer.saved_path == "out"
    assert processor.saved_path == expected_processor_path
    assert torch.initial_seed() == 42


def test_main_stage1_uses_requested_global_seed(monkeypatch, tmp_path):
    """Catches the Stage 2 compatibility seed leaking into Stage 1."""
    args = stage2_args(
        stage="stage1",
        seed=73,
        log_file=str(tmp_path / "stage1.log"),
    )
    monkeypatch.setattr(main, "get_args", lambda: args)
    monkeypatch.setattr(main, "validate_generated_artifact_paths", lambda selected: None)
    monkeypatch.setattr(main, "route_training", lambda selected: "stage1-result")

    result = main_train()

    assert result == "stage1-result"
    assert torch.initial_seed() == 73


def test_main_establishes_artifact_and_deepspeed_identity_before_seeding(monkeypatch):
    """Catches CUDA-adjacent startup preceding canonical run preflight."""
    from src.artifact_paths import resolve_run_artifact_paths

    paths = resolve_run_artifact_paths("stage1-zero2-dora-mb4-ga2-candidate")
    args = stage2_args(
        stage="stage1",
        save_model_path=str(paths.checkpoint_dir),
        log_file=str(paths.log_dir / "train.log"),
        deepspeed_config=None,
    )
    observed = []
    monkeypatch.setattr(main, "get_args", lambda: args)
    monkeypatch.setattr(
        main,
        "validate_generated_artifact_paths",
        lambda selected: observed.append("artifacts"),
        raising=False,
    )
    monkeypatch.setattr(
        main,
        "bind_deepspeed_runtime_profile",
        lambda selected: observed.append("runtime"),
        raising=False,
    )
    monkeypatch.setattr(
        main, "seed_everything", lambda **kwargs: observed.append("seed")
    )
    monkeypatch.setattr(main, "route_training", lambda selected: observed.append("route"))
    monkeypatch.setattr(main.logging, "FileHandler", lambda *args, **kwargs: object())
    monkeypatch.setattr(main.logging, "StreamHandler", lambda *args, **kwargs: object())
    monkeypatch.setattr(main.logging, "basicConfig", lambda **kwargs: None)

    main_train()

    assert observed == ["artifacts", "runtime", "seed", "route"]


def test_direct_main_rejects_artifacts_before_importing_transformers(tmp_path):
    """Catches direct main execution importing model dependencies before preflight."""
    poison = tmp_path / "poison"
    transformers = poison / "transformers"
    transformers.mkdir(parents=True)
    (transformers / "__init__.py").write_text(
        "raise RuntimeError('transformers import reached before preflight')\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(poison)

    completed = subprocess.run(
        [
            str(Path(sys.executable)),
            str(Path(main.__file__).resolve()),
            "--save_model_path",
            "/tmp/noncanonical-checkpoint",
            "--log_file",
            "/tmp/noncanonical.log",
        ],
        cwd=Path(main.__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "canonical checkpoint root" in completed.stderr
    assert "transformers import reached before preflight" not in completed.stderr


def test_direct_main_help_uses_documented_top_level_pythonpath():
    """Catches optional evidence imports requiring the repository root on PYTHONPATH."""
    repo_root = Path(main.__file__).resolve().parents[1]
    run_id = "step-preference-phase-a-20260903T133145Z"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(repo_root / "transformers/src"), str(repo_root / "src"))
    )

    completed = subprocess.run(
        [
            str(Path(sys.executable)),
            str(Path(main.__file__).resolve()),
            "--save_model_path",
            str(repo_root / "checkpoints" / run_id),
            "--log_file",
            str(repo_root / "logs" / run_id / "stage2b.log"),
            "--help",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--step_preference_enabled" in completed.stdout
