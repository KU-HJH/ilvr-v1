import hashlib
import math
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("provided,expected",[({},42),({"seed":42},42),({"seed":73},73)])
def test_direct_stage2_arguments_default_seed_without_replacing_explicit_seed(monkeypatch,provided,expected):
    import main
    args=SimpleNamespace(save_model_path="out",epochs=1,per_device_train_batch_size=1,
        gradient_accumulation_steps=2,warm_up_steps=0,learning_rate=1e-4,save_steps=1,
        max_seq_length_train=32,max_steps=2,gradient_checkpointing=False,tuning_method="full",
        deepspeed_config=None,**provided)
    monkeypatch.setattr(main,"SFTConfig",lambda **kwargs:SimpleNamespace(**kwargs))
    arguments=main.build_stage2_training_arguments(args)
    assert arguments.seed == expected


@pytest.mark.parametrize("field",["step_packed_capacity_evidence_path","step_packed_capacity_sample_id",
    "step_packed_gpu_snapshot","step_packed_gpu_snapshot_sha256"])
def test_main_rejects_any_partial_capacity_request_before_model_construction(monkeypatch,field):
    import main
    monkeypatch.setattr(main,"bind_deepspeed_runtime_profile",lambda *a,**k:SimpleNamespace(world_size=2,zero_stage=2))
    monkeypatch.setattr(main,"build_stage2_training_components",lambda args:pytest.fail("constructed a model for partial capacity settings"))
    with pytest.raises(ValueError,match="capacity-only options must be supplied together"):
        main.run_stage2(SimpleNamespace(**{field:"provided"}))


@pytest.mark.parametrize("initial_step",[0,9])
def test_zero2_adapter_observes_only_true_boundaries_before_gradient_clear(monkeypatch,initial_step):
    from types import SimpleNamespace
    import stage2.trainer as module
    class Owner: pass
    trainer = Owner()
    trainer.deepspeed_runtime_profile = SimpleNamespace(zero_stage=2)
    trainer._observed_optimizer_step = None
    trainer.state = SimpleNamespace(global_step=initial_step)
    trainer._observed_pre_step_gradient_norms = []
    boundary = [False]
    optimizer = SimpleNamespace(norm=1.25)
    engine = SimpleNamespace(optimizer=optimizer,
        is_gradient_accumulation_boundary=lambda: boundary[0])
    monkeypatch.setattr(module, "collect_deepspeed_pre_step_norm", lambda opt, **kw: torch.tensor(opt.norm))
    seen=[]
    def step():
        seen.append(list(trainer._observed_pre_step_gradient_norms))
        optimizer.norm=0
    wrapped=module._Stage2DeepSpeedGradientAdapter(trainer, engine, step)
    wrapped(); boundary[0]=True; optimizer.norm=1.25; wrapped(); optimizer.norm=2.5; wrapped()
    assert seen == [[], [1.25], [1.25, 2.5]]
    assert trainer._observed_optimizer_step == initial_step + 2


def test_capacity_singleton_dataloader_has_four_real_observations_per_rank():
    from types import SimpleNamespace
    from accelerate.data_loader import prepare_data_loader
    from step_preference.capacity_evidence import CapacitySampleRecorder
    from stage2.trainer import Stage2Trainer
    for rank in (0,1):
        trainer=object.__new__(Stage2Trainer)
        trainer.args=SimpleNamespace(world_size=2,gradient_accumulation_steps=2,max_steps=2,
            per_device_train_batch_size=1,seed=42)
        trainer.train_dataset=[SimpleNamespace(sample_id="a",latent_block_count=1)]
        trainer.data_collator=CapacitySampleRecorder(lambda rows:{"sample_ids":[r.sample_id for r in rows]},"a")
        trainer._stage2_train_sampler=None
        loader=torch.utils.data.DataLoader(trainer.train_dataset,batch_size=1,
            sampler=trainer._get_train_sampler(),collate_fn=trainer.data_collator)
        sharded=prepare_data_loader(loader,num_processes=2,process_index=rank,split_batches=False,put_on_device=False)
        observed=list(sharded)
        assert len(observed) == 4
        assert trainer.data_collator.sample_ids == ["a"]*4


def test_capacity_main_captures_training_peak_before_collective_final_save(monkeypatch,tmp_path):
    import main
    from types import SimpleNamespace as N
    from step_preference import capacity_evidence as evidence
    assert hasattr(evidence,"finish_capacity_memory"),"missing pre-save memory capture"
    calls=[]
    args=N(step_packed_capacity_sample_id="a",save_final_model=True,save_model_path=str(tmp_path),stage2_diagnostics_interval=0)
    config=N(is_stage2b=True,packing_enabled=True)
    class Trainer:
        def __init__(self,**kw):
            self.accelerator=N(device="cpu",wait_for_everyone=lambda:calls.append("barrier"))
        def train(self,**kw): calls.append("train");return N(training_loss=1.)
        def save_model(self,path):calls.append("save")
        def is_world_process_zero(self):return True
    components=N(model=object(),train_dataset=[N(sample_id="a")],data_collator=lambda x:x,
        processor=N(save_pretrained=lambda p:calls.append("processor")),peft_metadata=None,
        step_preference_config=config,trainer_class=Trainer,trainer_resume_path=None)
    monkeypatch.setattr(main,"bind_deepspeed_runtime_profile",lambda *a,**k:N(world_size=2,zero_stage=2))
    monkeypatch.setattr(main,"build_stage2_training_components",lambda a:components)
    monkeypatch.setattr(main,"build_stage2_training_arguments",lambda a:object())
    monkeypatch.setattr(main,"publish_step_preference_phase_a_summary",lambda *a,**k:None)
    monkeypatch.setattr(main,"publish_stage2_training_evidence",lambda *a,**k:None)
    monkeypatch.setattr(evidence,"validate_capacity_options",lambda *a,**k:True)
    monkeypatch.setattr(evidence,"begin_capacity_memory",lambda *a:calls.append("baseline") or (1,2))
    monkeypatch.setattr(evidence,"finish_capacity_memory",lambda *a:calls.append("peaks") or 3)
    monkeypatch.setattr(evidence,"publish_training_runtime",lambda *a:calls.append("runtime") or (tmp_path/"runtime","a"*64))
    monkeypatch.setattr(evidence,"publish_distributed_capacity",lambda *a:calls.append("evidence"))
    main.run_stage2(args)
    assert calls == ["baseline","train","peaks","save","barrier","processor","barrier","runtime","evidence"]


def test_zero2_capacity_observation_does_not_accept_a_skipped_optimizer_step(monkeypatch):
    from types import SimpleNamespace as N
    import stage2.trainer as module
    class Owner: pass
    trainer=Owner();trainer.deepspeed_runtime_profile=N(zero_stage=2)
    engine=N(optimizer=object(),skipped_steps=0,is_gradient_accumulation_boundary=lambda:True)
    monkeypatch.setattr(module,"collect_deepspeed_pre_step_norm",lambda *a,**kw:torch.tensor(1.))
    wrapped=module._Stage2DeepSpeedGradientAdapter(trainer,engine,lambda:setattr(engine,"skipped_steps",1))
    with pytest.raises(RuntimeError,match="skipped"):wrapped()


def test_capacity_final_adapter_save_defers_processor_but_numbered_save_does_not(monkeypatch,tmp_path):
    import stage2.trainer as module
    from types import SimpleNamespace as N
    from step_preference.capacity_evidence import CapacitySampleRecorder
    events=[]
    class Model:
        def save_pretrained(self,target,**kw):events.append(("weights",target.name))
    model=Model();trainer=object.__new__(module.Stage2Trainer)
    trainer.model=model;trainer.accelerator=N(unwrap_model=lambda m:m)
    trainer.step_preference_config=module.StepPreferenceConfig()
    trainer.args=N(output_dir=str(tmp_path),save_safetensors=True)
    trainer.data_collator=CapacitySampleRecorder(lambda rows:rows,"a")
    trainer.processing_class=N(save_pretrained=lambda target:events.append(("processor",target.name)))
    monkeypatch.setattr(module,"PeftModel",Model)
    monkeypatch.setattr(module,"resolve_qwen2_5_vl_model",lambda m:N(config=N(save_pretrained=lambda target:None)))
    monkeypatch.setattr(torch,"save",lambda *a:None)
    trainer._save(tmp_path/"checkpoint-2");trainer._save(tmp_path)
    assert events == [("weights","checkpoint-2"),("processor","checkpoint-2"),("weights",tmp_path.name)]
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
        objective="legacy_dpo",
        beta=0.5,
        lambda_ce=1.0,
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
    if marker == 21:
        input_ids = torch.tensor([[3, 4, marker, 0, 0, 6, 0, 0]])
        completion_positions = (3,)
    else:
        input_ids = torch.tensor([[3, 4, 5, 0, 0, 6, marker, 0]])
        completion_positions = (7,)
    return {
        "model_inputs": {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        },
        "completion_token_positions": completion_positions,
        "completion_token_ids": (0,),
        "rendered_text": f"rejected-{marker}",
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
        if kwargs.get("stage2_packed_inputs") is not None:
            coefficients = (
                (0.6,) * 4
                + (0.1,) * 2
                + (0.2,) * 2
                + (-0.2,)
                + (0.3,) * 2
                + (0.4,) * 2
                + (-0.4,)
                + (-0.6,)
            )
            assert len(coefficients) == len(positions)
            logits = torch.cat(
                [self._logits((0,), coefficient) for coefficient in coefficients],
                dim=1,
            )
            return SimpleNamespace(
                loss=(self.weight - 0.25).square() + 0.75,
                stage2_selected_logits=logits,
                stage2_diagnostics=None,
            )
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
            ids = kwargs["input_ids"][0].tolist()
            marker = next(value for value in (21, 22, 23) if value in ids)
            coefficient = {21: -0.2, 22: -0.4, 23: -0.6}[marker]
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


def _gradient_vector(model):
    """Flatten every trainable slot, treating an untouched slot as zero."""
    return torch.cat(
        [
            (
                parameter.grad
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            ).reshape(-1)
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
    )


def _raw_policy_components(model):
    """Hand-derived objectives for the deterministic tiny policy."""
    def logp(coefficient):
        logits = torch.stack(
            (
                coefficient * model.weight,
                model.weight * 0.0,
                model.weight * 0.0,
                model.weight * 0.0,
            )
        )
        return torch.log_softmax(logits, dim=0)[0]

    def entropy(coefficient):
        logits = torch.stack(
            (
                coefficient * model.weight,
                model.weight * 0.0,
                model.weight * 0.0,
                model.weight * 0.0,
            )
        )
        logps = torch.log_softmax(logits, dim=0)
        return -(logps.exp() * logps).sum()

    chosen = logp(0.6)
    dpo_terms = [
        -F.logsigmoid(0.5 * ((chosen - logp(rejected)) - reference_margin))
        for rejected, reference_margin in zip(
            (-0.2, -0.4, -0.6), (0.9, 0.2, 0.6)
        )
    ]
    entropy_terms = [
        F.relu(entropy(0.6) + 0.2 - entropy(coefficient))
        for coefficient in (0.1, 0.2, 0.3, 0.4)
    ]
    return {
        "loss/ce": (model.weight - 0.25).square() + 0.75,
        "loss/dpo": sum(dpo_terms) / 3,
        "loss/entropy": sum(entropy_terms) / 4,
    }


@pytest.mark.parametrize(
    ("coefficients", "selected_metric", "expected_scalar"),
    [
        ((1.0, 0.0, 0.0), "loss/ce", 1.3125),
        ((0.0, 1.0, 0.0), "loss/dpo", 0.6574867963790894),
        ((0.0, 0.0, 1.0), "loss/entropy", 0.1679723560810089),
    ],
    ids=("ce-objective", "dpo-objective", "entropy-objective"),
)
def test_stage2b_objective_coefficients_isolate_scalar_and_gradient(
    coefficients, selected_metric, expected_scalar
):
    """Catches inactive objectives contributing loss or policy gradients."""
    lambda_ce, lambda_dpo, lambda_entropy = coefficients
    config = _config(
        lambda_ce=lambda_ce,
        lambda_dpo=lambda_dpo,
        lambda_entropy=lambda_entropy,
    )

    objective_trainer = _trainer(config)
    objective_metrics = {}
    objective_trainer._accumulate_step_preference_metrics = objective_metrics.update
    objective_model = _Policy()
    objective_loss = objective_trainer.compute_loss(objective_model, _batch())
    objective_loss.backward()

    component_model = _Policy()
    selected_component = _raw_policy_components(component_model)[selected_metric]
    selected_component.backward()

    assert len(objective_model.calls) == 8
    assert float(selected_component.detach()) == pytest.approx(expected_scalar)
    torch.testing.assert_close(
        objective_loss.detach(), selected_component.detach(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        _gradient_vector(objective_model),
        _gradient_vector(component_model),
        rtol=1e-5,
        atol=1e-7,
    )
    assert float(objective_metrics[selected_metric].detach()) == pytest.approx(
        expected_scalar
    )


def test_stage2b_component_metrics_are_raw_and_unweighted():
    """Catches coefficient scaling leaking into capacity-probe metrics."""
    trainer = _trainer(_config(lambda_ce=0.25, lambda_dpo=3.0, lambda_entropy=4.0))
    metrics = {}
    trainer._accumulate_step_preference_metrics = metrics.update

    trainer.compute_loss(_Policy(), _batch())

    assert float(metrics["loss/ce"].detach()) == pytest.approx(1.3125)
    assert float(metrics["loss/dpo"].detach()) == pytest.approx(0.6574867963790894)
    assert float(metrics["loss/entropy"].detach()) == pytest.approx(0.1679723560810089)


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
        "loss/total",
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
    assert metrics["loss/total"] == pytest.approx(float(loss.detach()))
    assert metrics["stats/valid_visual_pairs"] == 2.0
    assert metrics["stats/valid_textual_pairs"] == 1.0
    assert metrics["stats/branch_forwards"] == 8.0
    assert metrics["stats/scored_tokens"] == 15.0

    loss.backward()
    assert model.weight.grad is not None
    assert math.isfinite(float(model.weight.grad))
    assert float(model.weight.grad) != 0.0


def test_packed_stage2b_uses_one_policy_call_and_same_objective(monkeypatch):
    """Catches packed routing that repeats branches or changes the Stage 2B reduction."""
    monkeypatch.setattr(
        "step_preference.packing.build_flex_block_mask",
        lambda _plan, _device: "block-mask-sentinel",
    )
    naive_trainer = _trainer(_config())
    naive_model = _Policy()
    naive_loss = naive_trainer.compute_loss(naive_model, _batch())

    packed_trainer = _trainer(_config(packing_enabled=True, max_packed_sequence_length=4608))
    packed_model = _Policy()
    packed_loss = packed_trainer.compute_loss(packed_model, _batch())

    torch.testing.assert_close(packed_loss, naive_loss, rtol=0, atol=0)
    assert len(packed_model.calls) == 1
    call = packed_model.calls[0]
    assert call["stage2_packed_inputs"]["block_mask"] == "block-mask-sentinel"
    assert call["stage2_logit_positions"].numel() == 15
    metrics = packed_trainer._step_preference_metric_totals
    assert metrics["stats/branch_forwards"] == 1.0
    assert metrics["stats/packed_tokens"] == 128.0
    assert math.isfinite(metrics["stats/packed_mask_sparsity"])
    for name in (
        "loss/total",
        "loss/ce",
        "loss/dpo_visual",
        "loss/dpo_text",
        "loss/entropy_visual",
        "loss/entropy_text",
    ):
        assert metrics[name] == pytest.approx(
            naive_trainer._step_preference_metric_totals[name]
        )


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


def _tiny_qwen_batch():
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
    return {
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


def test_stage2b_tiny_qwen_has_finite_loss_and_policy_gradients():
    """Catches trainer kwargs that violate the real selective Qwen Stage 2 contract."""
    torch.manual_seed(73)
    trainer = _trainer(_config())
    model = tiny_stage2_qwen(attn_implementation="eager").train()

    loss = trainer.compute_loss(model, _tiny_qwen_batch())
    loss.backward()

    assert torch.isfinite(loss)
    trainable_grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_grads
    assert all(torch.isfinite(gradient).all() for gradient in trainable_grads)


def test_stage2b_return_outputs_keeps_full_clean_logits_and_combined_loss():
    """Catches returning the clean loss-only payload or stale CE as output.loss."""
    torch.manual_seed(79)
    trainer = _trainer(_config())
    model = tiny_stage2_qwen(attn_implementation="eager").train()
    outer_calls = []
    handle = model.register_forward_pre_hook(
        lambda _module, _args, kwargs: outer_calls.append(kwargs), with_kwargs=True
    )
    try:
        loss, outputs = trainer.compute_loss(
            model, _tiny_qwen_batch(), return_outputs=True
        )
    finally:
        handle.remove()

    assert outer_calls[0]["stage2_loss_only"] is False
    assert outputs.logits.shape[:2] == _tiny_qwen_batch()["input_ids"].shape
    assert outputs.stage2_selected_logits.shape[1] == 2
    assert outputs.loss is loss
    assert torch.isfinite(outputs.loss)


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
            "--step_preference_objective",
            "legacy_dpo",
            "--stage2a_checkpoint_path",
            str(checkpoint),
            "--step_negative_path",
            str(negative_path),
            "--step_negative_manifest",
            str(manifest_path),
            "--vision_attention_implementation",
            "flash_attention_2",
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
    # Keep the real mixed-backend construction and validation on this CPU fixture.
    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration, "_check_and_enable_flash_attn_2",
        lambda config, **kwargs: config,
    )
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
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checkpoint_files=checkpoint_files,
        tokenizer_config_identity="tokenizer-v1",
        generation_settings_dict={"min_tokens": 2, "max_new_tokens": 32},
    )
    monkeypatch.setattr(main, "read_negative_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        main, "validate_stage2b_peft_base_identity", lambda **_kwargs: None
    )
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
        load_config=lambda *_args, **_kwargs: model.config,
        load_processor=lambda *_args, **_kwargs: processor,
        load_model=lambda _source, *, config: main.Qwen2_5_VLForConditionalGeneration(config),
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
            "assistant_plan_rendering": "adjacent-text-space-lf-v1",
        }
    ]
    assert store_calls[0]["expected_selected_row_indices"] == (0,)
    assert store_calls[0]["expected_keys"] == (("sample-a", 0),)
    assert components.trainer_resume_path is None
    assert components.step_preference_config.is_stage2b
    assert components.data_collator.negative_store == {("sample-a", 0): negative}
