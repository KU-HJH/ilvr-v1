from __future__ import annotations

import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import trainer_stage2 as trainer_stage2_module
from evidence_window.stage2_config import (
    Stage2EvidenceWindowConfig,
    Stage2PreferenceConfig,
    Stage2ScheduleConfig,
)
from trainer_stage2 import CustomTrainerStage2


TOKEN_IDS = {
    "start": 90,
    "end": 91,
    "pad": 92,
    "padding": 0,
    "special": (0, 90, 91, 92, 99),
}


class _DropoutStage2Model(torch.nn.Module):
    def __init__(self, *, module_dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 4)
        self.dropout = torch.nn.Dropout(module_dropout)
        self.head = torch.nn.Linear(4, 128)
        self.config = SimpleNamespace(
            attention_dropout=attention_dropout,
            hidden_size=4,
        )
        self.calls = 0

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        attention_block_mask=None,
        attention_backend_override=None,
        output_hidden_states=False,
        return_dict=True,
        use_cache=None,
    ):
        del attention_backend_override, output_hidden_states, return_dict, use_cache
        self.calls += 1
        hidden = self.dropout(self.embedding(input_ids)) * attention_mask.unsqueeze(-1)
        if attention_block_mask is not None:
            hidden = hidden - attention_block_mask.any(dim=-1).unsqueeze(-1) * 0.1
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden)


def _config(*, schedule=None):
    return Stage2EvidenceWindowConfig(
        preference=Stage2PreferenceConfig(enabled=True, lambda_ew=1.0),
        schedule=Stage2ScheduleConfig() if schedule is None else schedule,
    )


def _batch(*, malformed=False):
    ids = [90, 92, 5, 5, 6] if malformed else [90, 92, 91, 5, 6]
    return {
        "input_ids": torch.tensor([ids]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 5, 6]]),
        "ew_sample_indices": torch.tensor([17]),
        "ew_plan_types": (("latent", "text"),),
    }


def _make_trainer(monkeypatch, tmp_path, model, config):
    def fake_sft_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]
        instance.args = SimpleNamespace(output_dir=str(tmp_path))
        instance.state = SimpleNamespace(global_step=1, max_steps=1)
        instance.accelerator = SimpleNamespace(process_index=0)

    monkeypatch.setattr(trainer_stage2_module.SFTTrainer, "__init__", fake_sft_init)
    return CustomTrainerStage2(
        model=model,
        evidence_window_config=config,
        latent_token_ids=TOKEN_IDS,
    )


@pytest.mark.parametrize(
    ("module_dropout", "attention_dropout", "name"),
    [(0.2, 0.0, "dropout"), (0.0, 0.1, "attention_dropout")],
)
def test_active_stage2_auxiliary_rejects_dropout_before_matched_replay(
    monkeypatch, tmp_path, module_dropout, attention_dropout, name
):
    """Nonzero training dropout must not inject independent full/LOO noise."""
    model = _DropoutStage2Model(
        module_dropout=module_dropout,
        attention_dropout=attention_dropout,
    ).train()
    trainer = _make_trainer(monkeypatch, tmp_path, model, _config())

    with pytest.raises(ValueError, match=name):
        trainer.compute_loss(model, _batch())

    assert model.calls == 1


def test_zero_dropout_allows_matched_full_and_loo_replay(monkeypatch, tmp_path):
    """Zero dropout keeps the production matched-condition path available."""
    model = _DropoutStage2Model().train()
    trainer = _make_trainer(monkeypatch, tmp_path, model, _config())

    loss = trainer.compute_loss(model, _batch())

    assert torch.isfinite(loss)
    assert model.calls == 3


def test_ce_only_warmup_does_not_enforce_auxiliary_dropout(monkeypatch, tmp_path):
    """A scheduled CE-only step must retain the baseline forward behavior."""
    model = _DropoutStage2Model(module_dropout=0.2).train()
    config = _config(schedule=Stage2ScheduleConfig(ew_warmup_ratio=0.5))
    trainer = _make_trainer(monkeypatch, tmp_path, model, config)
    trainer.state = SimpleNamespace(global_step=0, max_steps=10)

    loss = trainer.compute_loss(model, _batch())

    assert torch.isfinite(loss)
    assert model.calls == 1


def _asymmetric_layout_preflight_worker(rank, init_file, result_dir):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )

    def fake_sft_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]
        instance.args = SimpleNamespace(output_dir=result_dir)
        instance.state = SimpleNamespace(global_step=1, max_steps=1)
        instance.accelerator = SimpleNamespace(process_index=rank)

    class CountingEngine:
        calls = 0

        def compute(self, *args, **kwargs):
            del args, kwargs
            self.calls += 1
            raise AssertionError("auxiliary callback entered before rank consensus")

    trainer_stage2_module.SFTTrainer.__init__ = fake_sft_init
    model = _DropoutStage2Model().train()
    trainer = CustomTrainerStage2(
        model=model,
        evidence_window_config=_config(),
        latent_token_ids=TOKEN_IDS,
    )
    engine = CountingEngine()
    trainer._ew_engine = engine
    try:
        trainer.compute_loss(model, _batch(malformed=rank == 1))
        outcome = "unexpected success"
    except Exception as exc:
        outcome = (
            f"{type(exc).__name__}: {exc}; auxiliary_calls={engine.calls}; "
            f"model_calls={model.calls}"
        )
    finally:
        Path(result_dir, f"rank-{rank}.txt").write_text(outcome, encoding="utf-8")
        torch.distributed.destroy_process_group()


def test_two_rank_layout_failure_is_collective_before_any_auxiliary_callback(tmp_path):
    """One malformed rank must make both peers exit before reference/edge collectives."""
    context = multiprocessing.get_context("spawn")
    init_file = tmp_path / "gloo-init"
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    processes = [
        context.Process(
            target=_asymmetric_layout_preflight_worker,
            args=(rank, str(init_file), str(result_dir)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=5)

    assert alive == [], "rank-local layout validation stranded a peer"
    outcomes = [
        (result_dir / f"rank-{rank}.txt").read_text(encoding="utf-8")
        for rank in range(2)
    ]
    assert outcomes[0] == outcomes[1]
    assert "distributed Stage-2 auxiliary preflight failed" in outcomes[0]
    assert "unmatched latent" in outcomes[0]
    assert "auxiliary_calls=0" in outcomes[0]
    assert "model_calls=1" in outcomes[0]
