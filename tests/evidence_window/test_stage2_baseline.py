from types import SimpleNamespace

import pytest
import torch

import main as main_module
import trainer_stage2 as trainer_stage2_module
from evidence_window.config import EvidenceWindowConfig
from evidence_window.stage2_config import (
    Stage2EvidenceWindowConfig,
    Stage2PreferenceConfig,
)
from main import (
    prepare_training_dataset,
    resolve_evidence_window_config,
    resolve_policy_model_path,
    select_stage_components,
)
from stage2_data import collate_fn_stage2
from trainer import CustomTrainerStage1
from trainer_stage2 import CustomTrainerStage2


class TinyStage2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)
        self.head = torch.nn.Linear(4, 32)

    def forward(self, input_ids, attention_mask, labels):
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        logits = self.head(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)


def make_stage2_trainer(monkeypatch, model, config):
    def fake_sft_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]

    monkeypatch.setattr(trainer_stage2_module.SFTTrainer, "__init__", fake_sft_init)
    return CustomTrainerStage2(model=model, evidence_window_config=config)


def stage_args(stage, checkpoint=None, ew_config=None):
    return SimpleNamespace(
        stage=stage,
        model="registry/base-model",
        load_model_path=None if checkpoint is None else str(checkpoint),
        ew_config=None if ew_config is None else str(ew_config),
        task="tiny",
    )


def test_baseline_equals_direct_model_ce_and_strips_stage2_metadata(monkeypatch):
    model = TinyStage2Model()
    trainer = make_stage2_trainer(
        monkeypatch,
        model,
        Stage2EvidenceWindowConfig.disabled(),
    )
    model_inputs = {
        "input_ids": torch.tensor([[2, 3, 4, 5]]),
        "attention_mask": torch.ones((1, 4)),
        "labels": torch.tensor([[-100, 3, 4, 5]]),
    }
    stage2_batch = {
        **model_inputs,
        "ew_sample_indices": torch.tensor([17]),
        "ew_plan_types": (("text", "latent", "text"),),
    }

    direct = model(**model_inputs).loss
    actual, outputs = trainer.compute_loss(
        model,
        dict(stage2_batch),
        return_outputs=True,
    )

    torch.testing.assert_close(actual, direct)
    torch.testing.assert_close(outputs.loss, direct)
    assert not hasattr(trainer, "ema")
    assert not hasattr(trainer, "_reference")
    assert not hasattr(trainer, "_ew_engine")


def test_stage2_trainer_rejects_version1_config(monkeypatch):
    with pytest.raises(TypeError, match="Stage2EvidenceWindowConfig"):
        make_stage2_trainer(monkeypatch, TinyStage2Model(), EvidenceWindowConfig.disabled())


def test_stage_components_are_explicit_and_cross_stage_values_are_rejected():
    assert select_stage_components("stage1") == (
        CustomTrainerStage1,
        main_module.collate_fn_stage1,
    )
    assert select_stage_components("stage2") == (
        CustomTrainerStage2,
        collate_fn_stage2,
    )
    with pytest.raises(ValueError, match="stage1 or stage2"):
        select_stage_components("stage3")


def test_stage2_requires_existing_stage1_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="Stage 1 checkpoint"):
        resolve_policy_model_path(stage_args("stage2", tmp_path / "missing"))

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    assert resolve_policy_model_path(stage_args("stage2", checkpoint)) == str(checkpoint)
    assert resolve_policy_model_path(stage_args("stage1")) == "registry/base-model"


def test_stage1_accepts_enabled_historical_v1_configuration(tmp_path):
    disabled_path = tmp_path / "disabled-v1.json"
    disabled_path.write_text('{"version": 1}', encoding="utf-8")
    enabled_path = tmp_path / "enabled-v1.json"
    enabled_path.write_text(
        '{"version": 1, "relation": {"lambda_magnitude": 0.25}}',
        encoding="utf-8",
    )

    assert resolve_evidence_window_config(stage_args("stage1")).is_disabled
    assert resolve_evidence_window_config(
        stage_args("stage1", ew_config=disabled_path)
    ).is_disabled
    enabled = resolve_evidence_window_config(
        stage_args("stage1", ew_config=enabled_path)
    )
    assert isinstance(enabled, EvidenceWindowConfig)
    assert not enabled.is_disabled

    version2_path = tmp_path / "version2.json"
    version2_path.write_text('{"version": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="version must be 1"):
        resolve_evidence_window_config(stage_args("stage1", ew_config=version2_path))


def test_stage2_accepts_only_version2_configuration(tmp_path):
    version1_path = tmp_path / "v1.json"
    version1_path.write_text('{"version": 1}', encoding="utf-8")
    version2_path = tmp_path / "v2.json"
    version2_path.write_text('{"version": 2}', encoding="utf-8")

    assert isinstance(
        resolve_evidence_window_config(stage_args("stage2")),
        Stage2EvidenceWindowConfig,
    )
    assert isinstance(
        resolve_evidence_window_config(stage_args("stage2", ew_config=version2_path)),
        Stage2EvidenceWindowConfig,
    )
    with pytest.raises(ValueError, match="version must be 2"):
        resolve_evidence_window_config(stage_args("stage2", ew_config=version1_path))


def test_active_stage2_rejects_default_lora_dropout_before_processor_load(
    tmp_path, monkeypatch
):
    """The CLI default must fail before allocating processor, policy, or reference."""
    checkpoint = tmp_path / "stage1"
    checkpoint.mkdir()
    args = SimpleNamespace(
        stage="stage2",
        model="registry/base-model",
        load_model_path=str(checkpoint),
        ew_config=None,
        deepspeed_config="unused.json",
        log_file=str(tmp_path / "train.log"),
        cache_dir=str(tmp_path / "cache"),
        use_lora=True,
        lora_dropout=0.05,
    )
    active = Stage2EvidenceWindowConfig(
        preference=Stage2PreferenceConfig(enabled=True, lambda_ew=1.0)
    )
    loads = []
    monkeypatch.setattr(main_module, "seed_everything", lambda seed: None)
    monkeypatch.setattr(main_module, "get_args", lambda: args)
    monkeypatch.setattr(main_module, "resolve_evidence_window_config", lambda _args: active)
    monkeypatch.setattr(main_module, "read_deepspeed_zero_stage", lambda _path: 1)
    monkeypatch.setattr(
        main_module.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: loads.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="lora_dropout.*zero"):
        main_module.main_train()

    assert loads == []


def test_active_stage2_accepts_zero_lora_dropout_until_processor_load(
    tmp_path, monkeypatch
):
    """A zero-dropout LoRA launch must pass the cheap auxiliary preflight."""
    checkpoint = tmp_path / "stage1"
    checkpoint.mkdir()
    args = SimpleNamespace(
        stage="stage2",
        model="registry/base-model",
        load_model_path=str(checkpoint),
        ew_config=None,
        deepspeed_config="unused.json",
        log_file=str(tmp_path / "train.log"),
        cache_dir=str(tmp_path / "cache"),
        use_lora=True,
        lora_dropout=0.0,
    )
    active = Stage2EvidenceWindowConfig(
        preference=Stage2PreferenceConfig(enabled=True, lambda_ew=1.0)
    )
    loads = []

    class ProcessorReached(RuntimeError):
        pass

    def reached(*args, **kwargs):
        loads.append((args, kwargs))
        raise ProcessorReached

    monkeypatch.setattr(main_module, "seed_everything", lambda seed: None)
    monkeypatch.setattr(main_module, "get_args", lambda: args)
    monkeypatch.setattr(main_module, "resolve_evidence_window_config", lambda _args: active)
    monkeypatch.setattr(main_module, "read_deepspeed_zero_stage", lambda _path: 1)
    monkeypatch.setattr(main_module.AutoProcessor, "from_pretrained", reached)

    with pytest.raises(ProcessorReached):
        main_module.main_train()

    assert len(loads) == 1


def test_disabled_stage2_allows_default_lora_dropout_until_processor_load(
    tmp_path, monkeypatch
):
    """The CE-only baseline must not inherit matched-replay dropout constraints."""
    checkpoint = tmp_path / "stage1"
    checkpoint.mkdir()
    args = SimpleNamespace(
        stage="stage2",
        model="registry/base-model",
        load_model_path=str(checkpoint),
        ew_config=None,
        deepspeed_config="unused.json",
        log_file=str(tmp_path / "train.log"),
        cache_dir=str(tmp_path / "cache"),
        use_lora=True,
        lora_dropout=0.05,
    )
    loads = []

    class ProcessorReached(RuntimeError):
        pass

    def reached(*args, **kwargs):
        loads.append((args, kwargs))
        raise ProcessorReached

    monkeypatch.setattr(main_module, "seed_everything", lambda seed: None)
    monkeypatch.setattr(main_module, "get_args", lambda: args)
    monkeypatch.setattr(
        main_module,
        "resolve_evidence_window_config",
        lambda _args: Stage2EvidenceWindowConfig.disabled(),
    )
    monkeypatch.setattr(main_module, "read_deepspeed_zero_stage", lambda _path: 1)
    monkeypatch.setattr(main_module.AutoProcessor, "from_pretrained", reached)

    with pytest.raises(ProcessorReached):
        main_module.main_train()

    assert len(loads) == 1


def test_stage2_dataset_enumerates_raw_rows_into_helper_free_records():
    raw_rows = [
        {
            "image_input": [],
            "text_input": f"question-{index}",
            "sequence_plan": [
                {"type": "latent", "helper_image": f"missing-{index}.png"},
                {"type": "text", "content": f"answer-{index}"},
            ],
        }
        for index in range(2)
    ]

    records = prepare_training_dataset(stage_args("stage2"), raw_rows)

    assert [record["ew_sample_index"] for record in records] == [0, 1]
    assert [record["ew_plan_types"] for record in records] == [
        ("latent", "text"),
        ("latent", "text"),
    ]
    assert records[0]["conversations"][1]["content"][0]["image"].endswith(
        "missing-0.png"
    )


def test_stage1_dataset_keeps_task_preprocessor(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "task_preporcess_config",
        {"tiny": lambda row: {"processed": row["value"]}},
    )

    assert prepare_training_dataset(stage_args("stage1"), [{"value": 7}]) == [
        {"processed": 7}
    ]
