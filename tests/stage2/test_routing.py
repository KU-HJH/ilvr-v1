from types import SimpleNamespace

import pytest
import torch
from transformers import Trainer
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds

import main
from main import (
    CustomTrainerStage1,
    Stage2DataCollator,
    Stage2Trainer,
    build_stage2_training_arguments,
    build_stage2_training_components,
    collate_fn_stage1,
    main_train,
    parse_args,
    select_training_path,
)


class FakeStage2Model:
    def __init__(self, loss):
        self.loss = loss
        self.last_inputs = None

    def __call__(self, **inputs):
        self.last_inputs = inputs
        return SimpleNamespace(loss=self.loss, stage2_diagnostics={"latent_blocks": 1})


def test_stage2_trainer_uses_model_ce_without_teacher():
    """Catches Stage 2 reintroducing a teacher, EMA, or altered model loss."""
    trainer = object.__new__(Stage2Trainer)
    trainer.model = FakeStage2Model(loss=torch.tensor(2.0, requires_grad=True))

    loss, outputs = trainer.compute_loss(
        trainer.model, {"input_ids": torch.ones(1, 3)}, return_outputs=True
    )

    assert loss.item() == 2.0
    assert outputs.loss is loss
    assert not hasattr(trainer, "_ema")
    assert trainer._latest_stage2_diagnostics == {"latent_blocks": 1}
    assert "pixel_values_latent" not in trainer.model.last_inputs
    assert trainer.model.last_inputs["collect_stage2_diagnostics"] is True


@pytest.mark.parametrize(
    ("diagnostics", "expected"),
    [
        (
            {
                "block_count": torch.tensor(2),
                "latent_l2_norms": torch.tensor([[3.0, 4.0]]),
                "adjacent_cosine_similarities": torch.tensor([[0.5]]),
            },
            {
                "stage2/latent_block_count": 2.0,
                "stage2/latent_l2_norm_mean": 3.5,
                "stage2/latent_l2_norm_min": 3.0,
                "stage2/latent_l2_norm_max": 4.0,
                "stage2/adjacent_cosine_mean": 0.5,
            },
        ),
        (
            {
                "block_count": torch.tensor(0),
                "latent_l2_norms": torch.empty(1, 0),
                "adjacent_cosine_similarities": torch.empty(1, 0),
            },
            {"stage2/latent_block_count": 0.0},
        ),
    ],
)
def test_stage2_trainer_logs_each_diagnostic_snapshot_once(
    monkeypatch, diagnostics, expected
):
    """Catches detached Stage 2 diagnostics being dropped or replayed forever."""
    emitted = []
    monkeypatch.setattr(
        Trainer,
        "log",
        lambda self, logs, start_time=None: emitted.append((dict(logs), start_time)),
    )
    trainer = object.__new__(Stage2Trainer)
    trainer._latest_stage2_diagnostics = diagnostics

    trainer.log({"loss": 1.25}, start_time=7.0)
    trainer.log({"loss": 1.0})

    assert emitted[0] == ({"loss": 1.25, **expected}, 7.0)
    assert emitted[1] == ({"loss": 1.0}, None)
    assert trainer._latest_stage2_diagnostics is None


def test_stage_routing_selects_independent_trainers():
    """Catches either stage accidentally sharing the other's trainer/collator."""
    assert select_training_path("stage1").trainer_class is CustomTrainerStage1
    assert select_training_path("stage1").collator_factory is collate_fn_stage1
    assert select_training_path("stage2").trainer_class is Stage2Trainer
    assert select_training_path("stage2").collator_factory is Stage2DataCollator


def test_boundary_supervision_is_one_default_true_boolean():
    """Catches duplicate or inverted latent-boundary CLI switches."""
    assert parse_args(["--stage", "stage2"]).supervise_latent_boundaries is True
    assert (
        parse_args(["--stage", "stage2", "--no-supervise_latent_boundaries"])
        .supervise_latent_boundaries
        is False
    )


def test_stage2_trainer_rejects_helper_tensors_before_model_call():
    """Catches Stage 1 helper-image tensors leaking into the Stage 2 model."""
    trainer = object.__new__(Stage2Trainer)
    model = FakeStage2Model(loss=torch.tensor(1.0))

    with pytest.raises(ValueError, match="helper-image tensors leaked"):
        trainer.compute_loss(
            model,
            {
                "input_ids": torch.ones(1, 3),
                "image_grid_thw_latent": torch.ones(1, 3),
            },
        )

    assert model.last_inputs is None


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--stage", "stage2", "--use_lora"], "does not support LoRA"),
        (
            ["--stage", "stage2", "--per_device_train_batch_size", "2"],
            "requires per_device_train_batch_size=1",
        ),
    ],
)
def test_stage2_cli_rejects_unsupported_training_modes(argv, message):
    """Catches unsupported Stage 2 LoRA or multi-sample local batches."""
    with pytest.raises(ValueError, match=message):
        parse_args(argv)


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


class RecordingVisual:
    def __init__(self):
        self.requires_grad = None

    def requires_grad_(self, value):
        self.requires_grad = value


class RecordingStage2Model:
    def __init__(self):
        self.config = SimpleNamespace()
        self.visual = RecordingVisual()
        self.is_gradient_checkpointing = False
        self.gradient_checkpointing_kwargs = None

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs=None):
        self.is_gradient_checkpointing = True
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False
        self.gradient_checkpointing_kwargs = None


class CompositionTokenizer:
    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert text == "<|im_start|>assistant"
        return {"input_ids": torch.tensor([[31, 32]])}


class CompositionProcessor:
    def __init__(self):
        self.tokenizer = CompositionTokenizer()


def stage2_args(**overrides):
    values = dict(
        load_model_path="stage1/model",
        resume_from_checkpoint_path=None,
        attention_implementation="sdpa",
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
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("sample_limit, expected", [(None, [1, 2, 3]), (2, [1, 2])])
def test_stage2_components_use_one_source_and_filter_complete_samples(
    monkeypatch, sample_limit, expected
):
    """Catches model/processor source drift, unbounded samples, or unfrozen vision."""
    calls = SimpleNamespace(prepared=[], filtered=None)
    processor = CompositionProcessor()
    model = RecordingStage2Model()
    decision = SimpleNamespace(
        model_source="selected/model",
        processor_source="selected/processor",
        trainer_resume_path="selected/checkpoint",
        latent_size=2,
        supervise_latent_boundaries=False,
    )
    monkeypatch.setattr(main, "resolve_stage2_load_decision", lambda **kwargs: decision)
    monkeypatch.setattr(
        main.AutoProcessor,
        "from_pretrained",
        lambda source: processor if source == "selected/processor" else None,
    )
    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda source, **kwargs: model
        if (source, kwargs) == ("selected/model", {"attn_implementation": "sdpa"})
        else None,
    )
    token_ids = LatentTokenIds(90, 91, 92)
    monkeypatch.setattr(main, "resolve_latent_token_ids", lambda tokenizer, config: token_ids)
    monkeypatch.setattr(main, "load_jsonl_examples", lambda path: [1, 2, 3])

    def prepare(item, *, image_root, latent_size):
        calls.prepared.append((item, image_root, latent_size))
        return item

    monkeypatch.setattr(main, "prepare_stage2_example", prepare)
    monkeypatch.setattr(
        main, "build_prepared_example_length_measure", lambda selected: "measure"
    )

    def filter_examples(*, examples, measure_length, max_length):
        calls.filtered = (examples, measure_length, max_length)
        return list(examples), SimpleNamespace(retained=len(examples))

    monkeypatch.setattr(main, "filter_stage2_examples", filter_examples)

    components = build_stage2_training_components(
        stage2_args(max_train_samples=sample_limit)
    )

    assert [item for item, _, _ in calls.prepared] == expected
    assert all(str(root) == "images" and size == 2 for _, root, size in calls.prepared)
    assert calls.filtered == (expected, "measure", 8192)
    assert components.model is model
    assert components.processor is processor
    assert components.train_dataset == expected
    assert components.trainer_class is Stage2Trainer
    assert components.trainer_resume_path == "selected/checkpoint"
    assert components.data_collator.token_ids == token_ids
    assert components.data_collator.latent_size == 2
    assert components.data_collator.supervise_latent_boundaries is False
    assert model.config.stage == "stage2"
    assert model.config.latent_size == 2
    assert model.config.supervise_latent_boundaries is False
    assert model.config.use_cache is False
    assert model.is_gradient_checkpointing is True
    assert model.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert model.visual.requires_grad is False


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


def test_stage2_training_arguments_omit_checkpoint_kwargs_when_disabled():
    """Catches inactive checkpointing carrying mode-specific runtime kwargs."""
    training_args = build_stage2_training_arguments(
        stage2_args(gradient_checkpointing=False)
    )

    assert training_args.gradient_checkpointing is False
    assert training_args.gradient_checkpointing_kwargs is None


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
        log_file=str(tmp_path / "train.log"),
        cache_dir=str(tmp_path / "cache"),
        save_final_model=True,
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
    )
    training_args = object()
    RecordingTrainer.world_process_zero = world_process_zero
    monkeypatch.setattr(main, "get_args", lambda: args)
    monkeypatch.setattr(main, "seed_everything", lambda seed: None)
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
    }
    assert trainer.resume_path == "stage2/checkpoint-4"
    assert trainer.saved_path == "out"
    assert processor.saved_path == expected_processor_path
