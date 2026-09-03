import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds

import main
from stage2.checkpointing import Stage2LoadDecision
from stage2.peft import Stage2TuningConfig
from stage2.trainer import Stage2Trainer
from step_preference.config import StepPreferenceConfig
from step_preference.negatives import StepNegative
from step_preference.spans import ReasoningStepSpan
from tests.stage2.helpers import tiny_stage2_qwen


def _config(**changes):
    values = dict(
        enabled=True,
        mode="stage2b",
        beta=0.5,
        lambda_dpo=2.0,
        lambda_entropy=0.3,
        entropy_margin=0.2,
        negative_path="negatives.jsonl",
        negative_manifest="manifest.json",
        stage2a_checkpoint_path="stage2a",
        attention_implementation="eager",
    )
    values.update(changes)
    return StepPreferenceConfig(**values)


def _negative(step_id, *, visual, textual):
    return StepNegative(
        sample_id="sample-a",
        step_id=step_id,
        chosen_text=f"chosen-{step_id}",
        rejected_visual_text=f"visual-{step_id}" if visual else None,
        rejected_textual_text=f"textual-{step_id}" if textual else None,
        valid_visual_negative=visual,
        valid_textual_negative=textual,
        reference_chosen_score=(-0.1 if step_id == 0 else -0.2),
        reference_visual_rejected_score=(
            (-1.0 if step_id == 0 else -0.4) if visual else None
        ),
        reference_textual_rejected_score=(-0.8 if textual else None),
    )


def _rejected(marker):
    return {
        "model_inputs": {
            "input_ids": torch.tensor([[marker, 0]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        },
        "completion_token_positions": (1,),
    }


def _batch():
    spans = (
        ReasoningStepSpan(0, 0, 1, 1, 2, 3, 5, 0),
        ReasoningStepSpan(1, 3, 5, 5, 6, 6, 8, 1),
    )
    rows = (_negative(0, visual=True, textual=False), _negative(1, visual=True, textual=True))
    return {
        "input_ids": torch.tensor([[3, 4, 5, 0, 0, 6, 0, 0]]),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 0, 0, -100, 0, 0]]),
        "sample_ids": ("sample-a",),
        "reasoning_step_spans": spans,
        "step_preference": {
            "steps": (
                {
                    "span": spans[0],
                    "negative": rows[0],
                    "visual_rejected": _rejected(21),
                    "textual_rejected": None,
                },
                {
                    "span": spans[1],
                    "negative": rows[1],
                    "visual_rejected": _rejected(22),
                    "textual_rejected": _rejected(23),
                },
            )
        },
    }


class _Policy:
    dtype = torch.float32

    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []
        self.ablation_index = 0

    def parameters(self):
        return iter((self.weight,))

    def _logits(self, positions, coefficient):
        row = torch.stack(
            (
                coefficient * self.weight,
                self.weight * 0.0,
                self.weight * 0.0,
                self.weight * 0.0,
            )
        )
        return row.reshape(1, 1, 4).expand(1, len(positions), 4)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        positions = kwargs.get("stage2_logit_positions")
        if "labels" in kwargs:
            return SimpleNamespace(
                loss=(self.weight - 0.25).square() + 0.75,
                stage2_selected_logits=(
                    None if positions is None else self._logits(positions, 0.6)
                ),
                stage2_diagnostics=None,
            )
        if kwargs.get("stage2_final_attention_mask") is not None:
            coefficient = (0.1, 0.2, 0.3, 0.4)[self.ablation_index]
            self.ablation_index += 1
        else:
            coefficient = {21: -0.2, 22: -0.4, 23: -0.6}[
                int(kwargs["input_ids"][0, 0])
            ]
        return SimpleNamespace(
            loss=None,
            stage2_selected_logits=self._logits(positions, coefficient),
            stage2_diagnostics=None,
        )


def _trainer(config):
    trainer = object.__new__(Stage2Trainer)
    trainer.stage2_diagnostics_interval = 0
    trainer._stage2_microbatch_index = 0
    trainer.step_preference_config = config
    return trainer


def _mean_logp(coefficient):
    logits = torch.tensor([coefficient, 0.0, 0.0, 0.0])
    return torch.log_softmax(logits, dim=0)[0]


def _entropy(coefficient):
    logits = torch.tensor([coefficient, 0.0, 0.0, 0.0])
    logps = torch.log_softmax(logits, dim=0)
    return -(logps.exp() * logps).sum()


def test_stage2b_runs_the_naive_graph_and_uses_global_pair_and_token_denominators(
    monkeypatch,
):
    """Catches merged masks, invalid-pair forwards, or per-modality reweighting."""
    emitted = []
    monkeypatch.setattr(
        Trainer,
        "log",
        lambda self, logs, start_time=None: emitted.append(dict(logs)),
    )
    trainer = _trainer(_config())
    model = _Policy()

    loss = trainer.compute_loss(model, _batch())
    trainer.log({"loss": float(loss.detach())})

    chosen = _mean_logp(0.6)
    policy_rejected = [_mean_logp(value) for value in (-0.2, -0.4, -0.6)]
    reference_margins = (0.9, 0.2, 0.6)
    pair_losses = [
        -F.logsigmoid(0.5 * ((chosen - rejected) - reference_margin))
        for rejected, reference_margin in zip(policy_rejected, reference_margins)
    ]
    expected_dpo = sum(pair_losses) / 3
    clean_entropy = _entropy(0.6)
    entropy_penalties = [
        F.relu(clean_entropy + 0.2 - _entropy(value))
        for value in (0.1, 0.2, 0.3, 0.4)
    ]
    expected_entropy = sum(entropy_penalties) / 4
    expected_ce = torch.tensor(1.3125)
    torch.testing.assert_close(
        loss.detach(), expected_ce + 2.0 * expected_dpo + 0.3 * expected_entropy
    )

    assert len(model.calls) == 8
    assert sum(call.get("stage2_final_attention_mask") is not None for call in model.calls) == 4
    assert sum(
        call.get("stage2_selective_logits_only") is True
        and call.get("stage2_final_attention_mask") is None
        and "labels" not in call
        for call in model.calls
    ) == 3
    assert all("step_preference" not in call for call in model.calls)
    assert all("sample_ids" not in call for call in model.calls)
    assert all("reasoning_step_spans" not in call for call in model.calls)

    metrics = emitted[0]
    required = {
        "loss/ce",
        "loss/dpo_visual",
        "loss/dpo_text",
        "loss/entropy_visual",
        "loss/entropy_text",
        "stats/dpo_margin_visual",
        "stats/dpo_margin_text",
        "stats/entropy_clean",
        "stats/entropy_visual_off",
        "stats/entropy_text_off",
        "stats/valid_visual_pairs",
        "stats/valid_textual_pairs",
        "stats/branch_forwards",
        "stats/scored_tokens",
    }
    assert required <= metrics.keys()
    assert all(math.isfinite(metrics[name]) for name in required)
    assert metrics["stats/valid_visual_pairs"] == 2.0
    assert metrics["stats/valid_textual_pairs"] == 1.0
    assert metrics["stats/branch_forwards"] == 8.0
    assert metrics["stats/scored_tokens"] == 15.0

    loss.backward()
    assert model.weight.grad is not None
    assert math.isfinite(float(model.weight.grad))
    assert float(model.weight.grad) != 0.0


def test_disabled_compute_loss_preserves_the_exact_existing_model_call():
    """Catches opt-in metadata or selective-logit kwargs leaking into Stage 2A."""
    trainer = _trainer(StepPreferenceConfig())
    model = _Policy()
    inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[-100, 2]]),
    }

    trainer.compute_loss(model, inputs, return_outputs=False)

    assert len(model.calls) == 1
    assert model.calls[0] == {
        "input_ids": inputs["input_ids"],
        "labels": inputs["labels"],
        "stage2_loss_only": True,
        "collect_stage2_diagnostics": False,
    }


def test_stage2b_rejects_missing_or_mismatched_private_join_metadata():
    """Catches silently dropping a source reasoning step at the trainer boundary."""
    trainer = _trainer(_config())
    model = _Policy()
    missing = _batch()
    del missing["step_preference"]
    with pytest.raises(ValueError, match="step_preference"):
        trainer.compute_loss(model, missing)

    mismatched = _batch()
    mismatched["step_preference"]["steps"] = mismatched["step_preference"]["steps"][:1]
    with pytest.raises(ValueError, match="reasoning step"):
        trainer.compute_loss(model, mismatched)


def test_stage2b_tiny_qwen_has_finite_loss_and_policy_gradients():
    """Catches trainer kwargs that violate the real selective Qwen Stage 2 contract."""
    torch.manual_seed(73)
    trainer = _trainer(_config())
    model = tiny_stage2_qwen(attn_implementation="eager").train()
    span = ReasoningStepSpan(0, 1, 2, 3, 5, 6, 8, 0)
    negative = StepNegative(
        sample_id="sample-real",
        step_id=0,
        chosen_text="chosen",
        rejected_visual_text="rejected",
        rejected_textual_text=None,
        valid_visual_negative=True,
        valid_textual_negative=False,
        reference_chosen_score=-0.3,
        reference_visual_rejected_score=-0.7,
        reference_textual_rejected_score=None,
    )
    input_ids = torch.tensor([[1, 5, 91, 90, 90, 92, 6, 7, 8]])
    labels = input_ids.clone()
    labels[:, :6] = -100
    rejected_ids = torch.tensor([[1, 5, 91, 90, 90, 92, 9, 10, 11]])
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "sample_ids": ("sample-real",),
        "reasoning_step_spans": (span,),
        "step_preference": {
            "steps": (
                {
                    "span": span,
                    "negative": negative,
                    "visual_rejected": {
                        "model_inputs": {
                            "input_ids": rejected_ids,
                            "attention_mask": torch.ones_like(rejected_ids),
                        },
                        "completion_token_positions": (6, 7),
                    },
                    "textual_rejected": None,
                },
            )
        },
    }

    loss = trainer.compute_loss(model, batch)
    loss.backward()

    assert torch.isfinite(loss)
    trainable_grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_grads
    assert all(torch.isfinite(gradient).all() for gradient in trainable_grads)


def test_stage2b_components_validate_exact_store_and_clear_stage2a_resume(
    monkeypatch, tmp_path
):
    """Catches permissive joins or a Stage 2A Trainer resume leaking through assembly."""
    source = tmp_path / "train.jsonl"
    source.write_bytes(b'{"sample_id":"sample-a"}\n')
    checkpoint = tmp_path / "stage2a"
    checkpoint.mkdir()
    negative_path = tmp_path / "negatives.jsonl"
    manifest_path = tmp_path / "manifest.json"
    args = main.parse_args(
        [
            "--stage",
            "stage2",
            "--data_path",
            str(source),
            "--image_root",
            str(tmp_path),
            "--step_preference_enabled",
            "--step_preference_mode",
            "stage2b",
            "--stage2a_checkpoint_path",
            str(checkpoint),
            "--step_negative_path",
            str(negative_path),
            "--step_negative_manifest",
            str(manifest_path),
            "--no-gradient_checkpointing",
        ]
    )
    decision = Stage2LoadDecision(
        base_model_source=str(checkpoint),
        processor_source=str(checkpoint),
        adapter_source=None,
        trainer_resume_path=None,
        tuning=Stage2TuningConfig(
            method="full", r=32, alpha=64, dropout=0.05, target_modules=("q_proj",)
        ),
        latent_size=8,
        supervise_latent_boundaries=True,
    )
    resolver_calls = []
    monkeypatch.setattr(
        main,
        "resolve_stage2b_initialization_decision",
        lambda **kwargs: resolver_calls.append(kwargs) or decision,
    )

    class Tokenizer:
        unk_token_id = -1

        def convert_tokens_to_ids(self, token):
            return {"<|latent_pad|>": 90, "<|latent_start|>": 91, "<|latent_end|>": 92}[token]

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[31, 32]])}

    processor = SimpleNamespace(tokenizer=Tokenizer())
    model = tiny_stage2_qwen()
    prepared = SimpleNamespace(
        sample_id="sample-a",
        reasoning_steps=(
            SimpleNamespace(step_id=0, target_text=SimpleNamespace(start=6, end=12)),
        ),
        conversation=[
            {"role": "user", "content": []},
            {"role": "assistant", "content": [{"type": "text", "text": "prefixchosen"}]},
        ],
    )
    prepare_calls = []
    monkeypatch.setattr(main, "load_jsonl_examples", lambda _path: [object()])
    monkeypatch.setattr(
        main,
        "prepare_stage2_example",
        lambda item, **kwargs: prepare_calls.append(kwargs) or prepared,
    )
    monkeypatch.setattr(main, "build_prepared_example_length_measure", lambda _p: object())
    monkeypatch.setattr(
        main,
        "filter_stage2_examples",
        lambda **kwargs: (list(kwargs["examples"]), SimpleNamespace(retained=1)),
    )
    monkeypatch.setattr(main, "resolve_latent_token_ids", lambda *_args: LatentTokenIds(90, 91, 92))
    checkpoint_files = ()
    manifest = SimpleNamespace(
        checkpoint_files=checkpoint_files,
        tokenizer_config_identity="tokenizer-v1",
        generation_settings_dict={"min_tokens": 2, "max_new_tokens": 32},
    )
    monkeypatch.setattr(main, "read_negative_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        main,
        "inspect_stage2a_checkpoint_identity",
        lambda _path: (checkpoint_files, "tokenizer-v1"),
    )
    negative = StepNegative(
        sample_id="sample-a",
        step_id=0,
        chosen_text="chosen",
        rejected_visual_text=None,
        rejected_textual_text=None,
        valid_visual_negative=False,
        valid_textual_negative=False,
        reference_chosen_score=-0.2,
        reference_visual_rejected_score=None,
        reference_textual_rejected_score=None,
    )
    store_calls = []
    monkeypatch.setattr(
        main,
        "load_negative_store",
        lambda **kwargs: store_calls.append(kwargs) or {("sample-a", 0): negative},
    )
    dependencies = main.Stage2SourceDependencies(
        find_adapter_config=lambda *_args, **_kwargs: None,
        load_config=lambda *_args, **_kwargs: SimpleNamespace(stage="stage2"),
        load_processor=lambda *_args, **_kwargs: processor,
        load_model=lambda *_args, **_kwargs: model,
    )

    components = main.build_stage2_training_components(
        args, source_dependencies=dependencies
    )

    assert resolver_calls[0]["stage2a_checkpoint_path"] == str(checkpoint)
    assert "resume_from_checkpoint_path" not in resolver_calls[0]
    assert prepare_calls == [
        {
            "image_root": tmp_path,
            "latent_size": 8,
            "require_reasoning_units": True,
        }
    ]
    assert store_calls[0]["expected_selected_row_indices"] == (0,)
    assert store_calls[0]["expected_keys"] == (("sample-a", 0),)
    assert components.trainer_resume_path is None
    assert components.step_preference_config.is_stage2b
    assert components.data_collator.negative_store == {("sample-a", 0): negative}
