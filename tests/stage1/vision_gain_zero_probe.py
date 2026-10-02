"""Fixed, opt-in two-update DoRA/ZeRO1/2 vision-gain premise experiment.

All auxiliary machinery lives here, outside the production objective. Initial and
resume are separate coordinator-owned attempts with the same max_steps=2 schedule.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import platform
import socket

# Match the existing tiny lifecycle harness on this host's Python build marker.
_original_python_version = platform.python_version
if _original_python_version().endswith("+"):
    platform.python_version = lambda: _original_python_version().rstrip("+")

import torch
import torch.distributed as dist
from transformers import TrainerCallback
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

from artifact_paths import resolve_run_artifact_paths
from deepspeed_runtime import load_deepspeed_runtime_profile
from peft_common import resolve_qwen2_5_vl_model
from stage1.composition import build_stage1_training_components
from stage1.equivalence import Stage1StochasticContext, activate_stage1_stochastic_context
from stage1.peft import build_stage1_tuned_model, install_stage1_sample_stable_dropout
from stage1.trainer import CustomTrainerStage1
from stage1.types import Stage1BatchMetadata, Stage1LatentSpan, Stage1SampleMetadata, Stage1TuningConfig
from tests.stage1.test_integration import _tiny_dependencies, TinyLifecycleCollator
from tests.stage1.zero_compat_diagnostic import _initialize_distributed, _publish_witness
from utils_deepseed import parse_args as parse_training_args

platform.python_version = _original_python_version

KIND = "stage1-vision-gain-runtime-probe"
ALPHA = 0.001
PROFILE = dict(microbatch=1, accumulation=4, global_batch=16, learning_rate=5e-5,
               seed=42, warmup=10, ce_weight=0.1, alpha=ALPHA, max_steps=2)
BOOLEANS = ("aux_grad_isolated", "moments_base_only", "master_delta_matches",
            "second_step_retains_aux", "reload_retains_aux")
RESIDUALS = ("master_delta_residual", "moment_residual", "reload_residual", "sync_residual")


def validate_rank_records(records, *, commit, zero_stage):
    if len(records) != 4 or {r.get("global_rank") for r in records} != {0, 1, 2, 3}:
        raise ValueError("probe requires exactly all four global ranks")
    packages = records[0].get("packages")
    if not isinstance(packages, dict) or set(packages) != {"torch", "deepspeed", "accelerate", "trl"}:
        raise ValueError("probe package identity missing")
    if not re.fullmatch(r"[a-f0-9]{40}", commit) or zero_stage not in (1, 2):
        raise ValueError("invalid probe identity")
    for record in records:
        if (record.get("schema") != 1 or record.get("kind") != KIND
                or record.get("commit") != commit or record.get("packages") != packages
                or record.get("world_size") != 4 or record.get("zero_stage") != zero_stage
                or record.get("steps") != 2 or record.get("alpha") != ALPHA
                or record.get("successful_mutations") != 2
                or record.get("valid_blocks_per_microbatch") != record["global_rank"]
                or record.get("microbatches_per_step") != [4, 2]):
            raise ValueError("probe runtime identity mismatch")
        if any(record.get(k) is not True for k in BOOLEANS):
            raise ValueError("probe runtime evidence missing or false")
        for key in (*RESIDUALS, "aux_movement_max", "initial_aux_movement_max", "peak_memory_bytes"):
            value = record.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"probe numeric evidence invalid: {key}")
        if min(record["aux_movement_max"], record["initial_aux_movement_max"]) <= 100 * torch.finfo(torch.float32).eps:
            raise ValueError("probe has no resolvable auxiliary displacement")
        # Per-tensor allclose is checked in the runtime; enforce its global absolute witness too.
        if any(record[key] > 1e-7 for key in RESIDUALS):
            raise ValueError("probe numeric residual exceeds tolerance")
    return dict(schema=1, kind=KIND, commit=commit, zero_stage=zero_stage, steps=2,
                alpha=ALPHA, packages=packages, ranks=sorted(records, key=lambda r: r["global_rank"]))


def project_direction(clean, masked, count):
    if count == 0:
        return None
    g, h = (clean.double() - masked.double()) / count, masked.double() / count
    if not torch.isfinite(g).all() or not torch.isfinite(h).all():
        raise FloatingPointError("nonfinite probe gradients")
    h2, hg = h.dot(h), h.dot(g)
    if 0 < h2 < 1e-30:
        return None
    return (g if h2 == 0 else g - torch.minimum(hg, hg.new_zeros(())) / h2 * h).float()


def displacement_for_step(direction, *, applied):
    return direction if applied else None


def adamw_reference(parameter, gradient, first, second, step, group):
    """Independent FP32 AdamW calculation on exactly the clipped partition gradient."""
    b1, b2 = group["betas"]
    first = b1 * first + (1 - b1) * gradient
    second = b2 * second + (1 - b2) * gradient.square()
    step += 1
    denominator = second.sqrt() / math.sqrt(1 - b2 ** step) + group["eps"]
    updated = parameter * (1 - group["lr"] * group["weight_decay"])
    updated = updated - (group["lr"] / (1 - b1 ** step)) * first / denominator
    return updated, first, second


class ProbeCollator(TinyLifecycleCollator):
    def __call__(self, rows):
        assert len(rows) == 1
        ids = self.token_ids
        row = [80, 80, 80, 80, 1]
        spans = []
        for _ in range(3):
            left = len(row)
            row.extend([ids.start_token_id] + [ids.pad_token_id] * 8 + [ids.end_token_id, 2, 3])
            spans.append(Stage1LatentSpan(start=left, end=left + 9,
                                         latent_positions=tuple(range(left + 1, left + 9))))
        inputs = torch.tensor([row])
        metadata = Stage1SampleMetadata(source_index=int(rows[0]), source_id=f"probe-{rows[0]}",
                    original_grid_range=(0, 1), original_pixel_range=(0, 4),
                    helper_grid_range=(0, 3), helper_pixel_range=(0, 24),
                    latent_spans=tuple(spans), helper_to_block=(0, 1, 2), unpadded_length=len(row))
        return dict(input_ids=inputs, labels=inputs.clone(), attention_mask=torch.ones_like(inputs),
                    pixel_values=torch.cat((torch.zeros(4, 1), torch.ones(4, 63)), dim=1),
                    image_grid_thw=torch.tensor([[1, 2, 2]]),
                    pixel_values_latent=torch.ones(24, 64),
                    image_grid_thw_latent=torch.tensor([[1, 1, 8]] * 3),
                    stage1_batch_metadata=Stage1BatchMetadata(samples=(metadata,), batch_size=1, sequence_length=len(row)))


def probe_dependencies():
    dependencies = _tiny_dependencies()
    def config_loader(source, **kwargs):
        config = dependencies.load_config(source, **kwargs)
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.rope_scaling = {"rope_type": "default", "mrope_section": [2, 3, 3]}
        config._attn_implementation = "sdpa"
        return config
    def model_loader(source, **kwargs):
        return dependencies.load_model(source, **kwargs).to(dtype=torch.bfloat16)
    return replace(dependencies, load_config=config_loader, load_model=model_loader,
                   collator_factory=lambda processor, token_ids, latent_size: ProbeCollator(token_ids),
                   prepare_dataset=lambda data_path, image_root: list(range(24)),
                   trainer_factory=ProbeTrainer)


def score_inputs(model, batch, targets, *, masked, block=0):
    inputs = {k: v for k, v in batch.items() if k not in
              {"labels", "stage1_batch_metadata", "pixel_values_latent", "image_grid_thw_latent"}}
    qwen = resolve_qwen2_5_vl_model(model)
    position_ids, _ = qwen.get_rope_index(batch["input_ids"], batch["image_grid_thw"], None,
                                        batch["attention_mask"])
    inputs.update(position_ids=position_ids, latent_hidden_states=targets.latent_values.detach(),
                  image_out_mask=targets.latent_mask, use_cache=False)
    if masked:
        length = batch["input_ids"].shape[1]
        index = torch.arange(length, device=batch["input_ids"].device)
        span = batch["stage1_batch_metadata"].samples[0].latent_spans[block]
        blocked = (index[None, :] > index[:, None]) | ((index[:, None] >= span.end)
                  & (index[None, :] > span.start) & (index[None, :] < span.end))
        mask = torch.zeros((1, 1, length, length), dtype=next(model.parameters()).dtype, device=index.device)
        inputs["attention_mask"] = mask.masked_fill(blocked, torch.finfo(mask.dtype).min)
    return inputs


def _context(model, batch, step):
    sample = batch["stage1_batch_metadata"].samples[0]
    return Stage1StochasticContext(base_seed=42, optimizer_global_step=step,
             rows=((sample.source_index, sample.unpadded_length),),
             expected_sites=model.stage1_stochastic_sites)


def score_gradients(model, batch, targets, *, block_count, masked, optimizer_step, scores=None):
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    gradients = [torch.zeros_like(p, dtype=torch.float32) for p in parameters]
    blocks = range(block_count) if masked else (None,) if block_count else ()
    for block in blocks:
        with activate_stage1_stochastic_context(_context(model, batch, optimizer_step)):
            outputs = model(**score_inputs(model, batch, targets, masked=masked, block=block or 0))
            selected = [block] if masked else range(block_count)
            score = sum(outputs.logits[0, span.end:span.end + 2].float().log_softmax(-1)
                        .gather(1, batch["input_ids"][0, span.end + 1:span.end + 3, None]).mean()
                        for span in (batch["stage1_batch_metadata"].samples[0].latent_spans[b] for b in selected))
            if scores is not None:
                scores.append(float(score.detach()))
            values = torch.autograd.grad(score, parameters, allow_unused=True)
        for accumulator, value in zip(gradients, values):
            if value is not None:
                accumulator.add_(value.detach().float())
    return gradients


def reference_score_gradients(model, batch, targets, *, block_count, masked, optimizer_step, scores=None):
    """Independent FP32 score/backward reference through the same BF16 network.

    This model is a separate clone without the runtime's ZeRO hooks. Scoring and
    gradient collection deliberately do not call score_gradients/autograd.grad.
    """
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    gradients = [torch.zeros_like(p, dtype=torch.float32) for p in parameters]
    blocks = ([block] for block in range(block_count)) if masked else ([list(range(block_count))] if block_count else [])
    model.zero_grad(set_to_none=True)
    for selected in blocks:
        model.zero_grad(set_to_none=True)
        with activate_stage1_stochastic_context(_context(model, batch, optimizer_step)):
            logits = model(**score_inputs(model, batch, targets, masked=masked, block=selected[0])).logits
            terms = []
            for block in selected:
                span = batch["stage1_batch_metadata"].samples[0].latent_spans[block]
                positions = torch.arange(span.end + 1, span.end + 3, device=logits.device)
                rows = logits[0, positions - 1, :].float()
                log_probs = rows - torch.logsumexp(rows, dim=-1, keepdim=True)
                target_ids = batch["input_ids"][0, positions]
                terms.append(log_probs[torch.arange(positions.numel(), device=logits.device), target_ids].sum()
                             / positions.numel())
            score = torch.stack(terms).sum()
            if scores is not None:
                scores.append(float(score.detach()))
            score.backward()
        for accumulator, parameter in zip(gradients, parameters):
            if parameter.grad is not None:
                accumulator.add_(parameter.grad.detach().float())
    model.zero_grad(set_to_none=True)
    return gradients


def cpu_score_fixture():
    """Real tiny Qwen + DoRA with prior-backward support, no distributed engine."""
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
    from stage1.targets import Stage1TeacherTargetBuilder
    from stage1.ema import Stage1EmaTeacher
    dependencies = probe_dependencies()
    config = dependencies.load_config("tiny/original-qwen")
    model = dependencies.load_model("tiny/original-qwen", config=config).float()
    model.resize_token_embeddings(128, mean_resizing=False)
    ids = LatentTokenIds(pad_token_id=125, start_token_id=126, end_token_id=127)
    tuning = Stage1TuningConfig(method="dora", rank=32, alpha=64, dropout=0.05)
    model = build_stage1_tuned_model(model, tuning, (125, 126, 127))
    install_stage1_sample_stable_dropout(model)
    model.train()
    batch = ProbeCollator(ids)([0])
    teacher = Stage1EmaTeacher(model, decay=1.0, method="dora")
    processor = dependencies.load_processor("tiny/original-qwen")
    from stage1.composition import LATENT_TOKEN_STRINGS
    processor.tokenizer.add_special_tokens({"additional_special_tokens": LATENT_TOKEN_STRINGS})
    targets = Stage1TeacherTargetBuilder(processor.tokenizer).build(teacher.model, dict(batch), batch["stage1_batch_metadata"])
    return model, batch, targets


def _flatten(values):
    return torch.cat([value.reshape(-1).float() for value in values])


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_snapshot(item) for item in value]
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    if hasattr(value, "__dataclass_fields__"):
        return _snapshot(vars(value))
    return type(value).__name__


def _assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_same(a, b)
    else:
        assert left == right


def reducer_snapshot(zero):
    fields = ("averaged_gradients", "ipg_buckets", "params_already_reduced", "is_partition_reduced",
              "remaining_grads_in_partition", "all_grad_tensors", "_epilogue_ran_this_backward",
              "is_gradient_accumulation_boundary", "micro_step_id")
    state = {name: getattr(zero, name, None) for name in fields}
    manager = zero._backward_hook_state
    # Snapshot every installed manager counter/flag, including queued graph-task state.
    # Callback identities matter: two different callables must not collapse to "function".
    state["_backward_hook_state"] = {
        name: [id(hook) for hook in value] if name == "_grad_acc_post_hooks" else value
        for name, value in vars(manager).items()}
    return _snapshot(state)


class StopAfterFirstSave(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        if state.global_step == 1:
            self.trainer.save_probe_reference()
            control.should_training_stop = True
        return control


class ProbeTrainer(CustomTrainerStage1):
    """Test-only auxiliary path attached to the observed engine mutation boundary."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_model = deepcopy(self.model)
        self.reference_model.train()
        self.probe_clean = None
        self.probe_masked = None
        self.probe_count = 0
        self.probe_micros = 0
        self.probe_base_witness = []
        self._probe_expected_base_partitions = []
        self.probe_stats = dict(aux_grad_isolated=True, moments_base_only=True,
                               master_delta_matches=True, second_step_retains_aux=False,
                               reload_retains_aux=False, master_delta_residual=0.0,
                               moment_residual=0.0, reload_residual=0.0, sync_residual=0.0,
                               aux_movement_max=0.0, initial_aux_movement_max=0.0, successful_mutations=0,
                               score_residual=0.0, gradient_relative_l2=0.0)
        self._probe_verified_masters = None
        self._probe_base_entry_checked = False

    def _install_deepspeed_step_adapter(self):
        super()._install_deepspeed_step_adapter()
        engine = self.model_wrapped
        optimizer = engine.optimizer.optimizer
        original = optimizer.step
        def base_step(*args, **kwargs):
            if not self._probe_base_entry_checked:
                if self.state.global_step == 1:
                    assert self.probe_stats["reload_retains_aux"]
                    self._assert_verified_masters()
                self._probe_base_entry_checked = True
            references = []
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = optimizer.state[parameter]
                    self.probe_base_witness.append(dict(old=parameter.detach().float().cpu().clone(),
                        clipped_gradient=parameter.grad.detach().float().cpu().clone(),
                        old_state=_snapshot(state)))
                    references.append((parameter, adamw_reference(parameter.detach().float(), parameter.grad.detach().float(),
                        state.get("exp_avg", torch.zeros_like(parameter)), state.get("exp_avg_sq", torch.zeros_like(parameter)),
                        int(state.get("step", 0)), group)))
            result = original(*args, **kwargs)
            for parameter, (expected, first, second) in references:
                torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)
                self._probe_expected_base_partitions.append((parameter, expected.clone()))
                for key, want in (("exp_avg", first), ("exp_avg_sq", second)):
                    got = optimizer.state[parameter][key]
                    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7)
                    self.probe_stats["moment_residual"] = max(self.probe_stats["moment_residual"], float((got - want).abs().max()))
            return result
        optimizer.step = base_step
        if self.probe_phase == "resume":
            self._check_reload()

    def _parameters(self):
        return tuple(p for p in self.model.parameters() if p.requires_grad)

    def _reference_path(self):
        return Path(self.args.output_dir) / "probe-reference" / f"rank-{dist.get_rank()}.pt"

    def _check_reload(self):
        from deepspeed.utils import safe_get_full_fp32_param
        saved = torch.load(self._reference_path(), map_location="cpu", weights_only=True)
        masters = [safe_get_full_fp32_param(p) for p in self._parameters()]
        for got, want in zip(masters, saved["masters"]):
            torch.testing.assert_close(got, want.to(got.device), rtol=0, atol=0)
        _assert_same(_snapshot(self.model_wrapped.optimizer.optimizer.state_dict()), saved["optimizer"])
        assert self.state.global_step == 1
        assert self.lr_scheduler.state_dict()["last_epoch"] == 1
        self.probe_stats.update(saved["stats"])
        assert self.probe_stats["successful_mutations"] == 1
        assert self.probe_stats["initial_aux_movement_max"] > 100 * torch.finfo(torch.float32).eps
        self._probe_verified_masters = [p.clone() for p in saved["masters"]]
        self._probe_base_entry_checked = False
        self.probe_stats["second_step_retains_aux"] = False
        self.probe_stats["reload_retains_aux"] = True

    def _assert_verified_masters(self):
        from deepspeed.utils import safe_get_full_fp32_param
        assert self._probe_verified_masters is not None
        parameters = self._parameters()
        assert len(parameters) == len(self._probe_verified_masters)
        for parameter, expected in zip(parameters, self._probe_verified_masters):
            got = safe_get_full_fp32_param(parameter)
            torch.testing.assert_close(got, expected.to(got.device), rtol=0, atol=0)

    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = self._prepare_inputs(inputs)
        parameters = self._parameters()
        zero = model.optimizer
        before = reducer_snapshot(zero)
        before_grads = _snapshot([(p.grad, getattr(p, "grad_accum", None)) for p in parameters])
        with torch.random.fork_rng(devices=[self.accelerator.device.index]):
            targets = self.target_builder.build(self.ema_teacher.model, dict(inputs), inputs["stage1_batch_metadata"])
            count = dist.get_rank()
            clean_scores, masked_scores = [], []
            clean = score_gradients(self.model, inputs, targets, block_count=count, masked=False,
                                    optimizer_step=self.state.global_step, scores=clean_scores)
            masked = score_gradients(self.model, inputs, targets, block_count=count, masked=True,
                                     optimizer_step=self.state.global_step, scores=masked_scores)
            # Independent current-state BF16 recomputation; only scoring/buffers
            # use FP32. A FP32 network has different forward/backward rounding.
            self.reference_model.to(device=self.accelerator.device, dtype=next(self.model.parameters()).dtype)
            self.reference_model.load_state_dict(self.model.state_dict())
            for actual, is_masked, actual_scores in ((clean, False, clean_scores), (masked, True, masked_scores)):
                reference_scores = []
                expected = reference_score_gradients(self.reference_model, inputs, targets,
                            block_count=count, masked=is_masked, optimizer_step=self.state.global_step, scores=reference_scores)
                if count:
                    torch.testing.assert_close(torch.tensor(actual_scores), torch.tensor(reference_scores), rtol=1e-3, atol=1e-3)
                    a, b = _flatten(actual), _flatten(expected)
                    torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-5)
                    relative = float((a - b).norm() / b.norm().clamp_min(1e-30))
                    assert relative <= 0.01
                    self.probe_stats["gradient_relative_l2"] = max(self.probe_stats["gradient_relative_l2"], relative)
                    self.probe_stats["score_residual"] = max(self.probe_stats["score_residual"], max(abs(a-b) for a,b in zip(actual_scores, reference_scores)))
        _assert_same(before, reducer_snapshot(zero))
        _assert_same(before_grads, _snapshot([(p.grad, getattr(p, "grad_accum", None)) for p in parameters]))
        c, h = _flatten(clean), _flatten(masked)
        if self.probe_clean is None:
            self.probe_clean, self.probe_masked = torch.zeros_like(c), torch.zeros_like(h)
        self.probe_clean.add_(c); self.probe_masked.add_(h)
        self.probe_count += count
        self.probe_micros += 1
        # Base consumes the same detached targets; its existing stochastic context spans backward.
        builder = self.target_builder
        class CachedTargets:
            def build(self, teacher, working, metadata):
                return targets
        self.target_builder = CachedTargets()
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        finally:
            self.target_builder = builder

    def _finish_optimizer_step(self, model=None, *, successful):
        from deepspeed.utils import safe_get_full_fp32_param, safe_set_full_fp32_param
        assert successful, "probe requires successful optimizer update"
        assert self._probe_base_entry_checked and self.probe_base_witness
        assert self.probe_stats["successful_mutations"] == self.state.global_step
        assert self._probe_expected_base_partitions
        for parameter, expected in self._probe_expected_base_partitions:
            torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)
        count = torch.tensor(self.probe_count, dtype=torch.int64, device=self.accelerator.device)
        for tensor in (self.probe_clean, self.probe_masked, count):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        direction = displacement_for_step(project_direction(self.probe_clean, self.probe_masked, int(count)), applied=successful)
        expected_micros = 4 if self.state.global_step == 0 else 2
        assert self.probe_micros == expected_micros
        assert direction is not None, "probe requires an auxiliary mutation"
        if direction is not None:
            offset = 0
            movement = 0.0
            moments = _snapshot(model.optimizer.optimizer.state_dict())
            expected = []
            for parameter in self._parameters():
                master = safe_get_full_fp32_param(parameter)
                delta = ALPHA * direction[offset:offset + parameter.numel()].view_as(master)
                updated = master + delta
                movement = max(movement, float((updated - master).abs().max()))
                safe_set_full_fp32_param(parameter, updated)
                expected.append(updated)
                offset += parameter.numel()
            model.optimizer.update_lp_params()
            for parameter, want in zip(self._parameters(), expected):
                got = safe_get_full_fp32_param(parameter)
                torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7)
                self.probe_stats["master_delta_residual"] = max(self.probe_stats["master_delta_residual"], float((got-want).abs().max()))
                torch.testing.assert_close(parameter, got.to(parameter.dtype), rtol=0, atol=0)
            _assert_same(moments, _snapshot(model.optimizer.optimizer.state_dict()))
            assert movement > 100 * torch.finfo(torch.float32).eps
            self.probe_stats["aux_movement_max"] = max(self.probe_stats["aux_movement_max"], movement)
            self.probe_stats["successful_mutations"] += 1
            self._probe_verified_masters = [p.detach().cpu().clone() for p in expected]
            if self.state.global_step == 0:
                self.probe_stats["initial_aux_movement_max"] = movement
            else:
                # Entry retained the first displacement; independent AdamW and the
                # master/low-precision checks above verified the actual second result.
                self.probe_stats["second_step_retains_aux"] = True
        if successful:
            diagnostic = Path(self.args.output_dir) / "probe-reference" / f"rank-{dist.get_rank()}-step-{self.state.global_step + 1}.pt"
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(clean_sum=self.probe_clean.cpu(), masked_sum=self.probe_masked.cpu(),
                            global_blocks=int(count), direction=None if direction is None else direction.cpu(),
                            base_partitions=self.probe_base_witness,
                            fp32_masters=[safe_get_full_fp32_param(p).cpu() for p in self._parameters()]), diagnostic)
        self.probe_base_witness = []
        self._probe_expected_base_partitions = []
        self._probe_base_entry_checked = False
        self.probe_clean = self.probe_masked = None
        self.probe_count = self.probe_micros = 0
        super()._finish_optimizer_step(model, successful=successful)

    def save_probe_reference(self):
        assert self.probe_stats["successful_mutations"] == 1
        assert self.probe_stats["initial_aux_movement_max"] > 100 * torch.finfo(torch.float32).eps
        self._assert_verified_masters()
        path = self._reference_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(masters=self._probe_verified_masters,
                        optimizer=_snapshot(self.model_wrapped.optimizer.optimizer.state_dict()),
                        stats=self.probe_stats), path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--phase", choices=("initial", "resume"), required=True)
    parser.add_argument("--zero-stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--deepspeed", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    return parser


def probe_training_args(args, paths, fixture):
    """Resolve the fixed profile consumed by both Trainer attempts."""
    argv = ["--stage", "stage1", "--model", "tiny/original-qwen", "--data_path", str(fixture / "TRAIN.jsonl"),
            "--image_root", str(fixture / "images"), "--save_model_path", str(paths.checkpoint_dir),
            "--latent_size", "8", "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "4",
            "--max_steps", "2", "--save_strategy", "steps", "--save_steps", "1", "--attention_implementation", "sdpa",
            "--lora_r", "32", "--lora_alpha", "64", "--lora_dropout", "0.05", "--ce_weight", "0.1",
            "--learning_rate", "5e-5", "--warm_up_steps", "10", "--seed", "42", "--tuning_method", "dora",
            "--deepspeed_config", str(args.deepspeed), "--gradient_checkpointing",
            "--epochs", "1", "--ema_tau", "1.0", "--sim_weight", "1.0",
            "--coverage_p", "0.9", "--image_pool_k", "8", "--helper_group_L", "256",
            "--textual_states", "previous", "--compress_strategy", "average",
            "--vision_attention_implementation", "sdpa",
            "--lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"]
    if args.phase == "resume":
        argv += ["--resume_from_checkpoint_path", str(paths.checkpoint_dir / "checkpoint-1")]
    return parse_training_args(argv)


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise ValueError("probe requires coordinator-owned two-node/four-rank execution")
    if os.environ.get("ILVR_REPOSITORY_COMMIT") != args.commit:
        raise ValueError("probe commit differs from owned supervisor")
    _initialize_distributed()
    assert dist.get_world_size() == 4
    assert int(os.environ["LOCAL_RANK"]) == dist.get_rank() % 2
    paths = resolve_run_artifact_paths(args.run_id)
    fixture = Path(__file__).resolve().parents[2] / "data" / args.run_id
    if int(os.environ["LOCAL_RANK"]) == 0:
        if args.phase == "initial":
            fixture.mkdir(parents=True, exist_ok=False)
            (fixture / "images").mkdir()
            (fixture / "TRAIN.jsonl").write_text("{}\n")
        elif not (fixture / "TRAIN.jsonl").is_file():
            raise FileNotFoundError("probe resume fixture missing")
    dist.barrier()
    profile = load_deepspeed_runtime_profile(args.deepspeed, world_size=4)
    if profile.zero_stage != args.zero_stage:
        raise ValueError("probe ZeRO config mismatch")
    unset_hf_deepspeed_config()
    components = build_stage1_training_components(probe_training_args(args, paths, fixture), dependencies=probe_dependencies(),
                                                   deepspeed_runtime_profile=profile)
    trainer = components.trainer
    trainer.probe_phase = args.phase
    if args.phase == "initial":
        trainer.add_callback(StopAfterFirstSave(trainer))
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=str(paths.checkpoint_dir / "checkpoint-1") if args.phase == "resume" else None)
    assert trainer.state.global_step == (1 if args.phase == "initial" else 2)
    assert trainer._successful_optimizer_steps == 1
    assert trainer.probe_stats["successful_mutations"] == trainer.state.global_step
    assert trainer.probe_stats["initial_aux_movement_max"] > 100 * torch.finfo(torch.float32).eps
    record = dict(schema=1, kind=KIND, commit=args.commit,
                  packages={name: version(name) for name in ("torch", "deepspeed", "accelerate", "trl")},
                  global_rank=dist.get_rank(), local_rank=int(os.environ["LOCAL_RANK"]),
                  machine_rank=dist.get_rank() // 2, hostname=socket.gethostname(), world_size=4, zero_stage=args.zero_stage, steps=trainer.state.global_step,
                  alpha=ALPHA, valid_blocks_per_microbatch=dist.get_rank(), microbatches_per_step=[4, 2],
                  peak_memory_bytes=torch.cuda.max_memory_allocated(), **trainer.probe_stats)
    gathered = [None] * 4
    dist.all_gather_object(gathered, record)
    if args.phase == "resume":
        witness = validate_rank_records(gathered, commit=args.commit, zero_stage=args.zero_stage)
    else:
        witness = dict(schema=1, kind=KIND, phase="initial", commit=args.commit, ranks=gathered)
    _publish_witness(paths, witness, filename=f"{args.phase}-witness.json")
    return witness


def main(argv=None):
    witness = run_probe(build_parser().parse_args(argv))
    if dist.get_rank() == 0:
        print(json.dumps(witness, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
