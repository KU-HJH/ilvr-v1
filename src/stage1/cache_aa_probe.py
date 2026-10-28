"""Throwaway, explicit one-update A/A observers; never a cache implementation."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from transformers import TrainerCallback

from .equivalence import canonical_tensor_sha256
from .matched_resume import MatchedCheckpointStop


def fingerprint(value):
    if isinstance(value, torch.Tensor):
        return {"tensor_sha256": canonical_tensor_sha256(value),
                "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return fingerprint(torch.from_numpy(value.copy()))
    if is_dataclass(value):
        return fingerprint(asdict(value))
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [fingerprint(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported probe value: {type(value).__name__}")


def rng_state():
    return fingerprint({"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch_cpu": torch.get_rng_state(),
                        "torch_cuda_current": torch.cuda.get_rng_state()})


def parameters(model):
    return fingerprint({name: value for name, value in model.named_parameters() if value.requires_grad})


class Probe(TrainerCallback):
    def __init__(self, trainer, args):
        self.trainer, self.args = trainer, args
        self.rows, self.optimizer_steps = [], []
        self.current = None
        self.in_reference = False
        self.handles = []
        self.optimizer = None
        self.expected_optimizer_groups = None
        self.before = None
        self.after = None
        self.loader_workers = None

    def install(self):
        trainer = self.trainer
        original_loader = trainer.get_train_dataloader
        def get_loader(*args, **kwargs):
            loader = original_loader(*args, **kwargs)
            self.loader_workers = loader.num_workers
            if self.loader_workers != 4:
                raise ValueError("A/A probe requires actual prepared workers4")
            return loader
        trainer.get_train_dataloader = get_loader

        original_step = trainer.training_step
        def training_step(model, inputs, *args, **kwargs):
            if len(self.rows) >= 4 or self.current is not None:
                raise ValueError("A/A probe exceeded one accumulation window")
            engine = trainer.accelerator.deepspeed_engine_wrapped.engine
            if self.optimizer is None:
                self.optimizer = engine.optimizer.optimizer
                self.expected_optimizer_groups = len(self.optimizer.param_groups)
                self.handles.append(self.optimizer.register_step_pre_hook(self.before_optimizer))
                self.handles.append(self.optimizer.register_step_post_hook(self.after_optimizer))
            self.current = {"inputs": fingerprint(inputs), "rng_before": rng_state(),
                            "teacher_visual": [], "reference_head_inputs": []}
            try:
                loss = original_step(model, inputs, *args, **kwargs)
                self.current["returned_loss"] = fingerprint(loss.detach())
                self.current["rng_after"] = rng_state()
                self.rows.append(self.current)
                return loss
            finally:
                self.current = None
        trainer.training_step = training_step

        original_prepare = trainer._prepare_stage1_loss_inputs
        def prepare(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            if self.current is not None:
                target = result[2]
                self.current["targets"] = fingerprint({
                    "latent_values": target.latent_values, "latent_mask": target.latent_mask,
                    "selected_counts": target.selected_counts,
                    "selected_candidate_indices": target.selected_candidate_indices,
                    "selection_margins": target.selection_margins,
                    "rejected_blocks": target.rejected_blocks})
            return result
        trainer._prepare_stage1_loss_inputs = prepare

        original_loss = trainer.compute_loss
        def compute_loss(*args, **kwargs):
            output = original_loss(*args, **kwargs)
            if self.current is not None:
                loss = output[0] if isinstance(output, tuple) else output
                self.current["loss"] = fingerprint(loss.detach())
                self.current["loss_value"] = float(loss.detach().cpu())
            return output
        trainer.compute_loss = compute_loss

        teacher = trainer.ema_teacher.model
        original_forward = teacher.forward
        def teacher_forward(*args, **kwargs):
            previous = self.in_reference
            self.in_reference = "stage1_selected_positions" in kwargs
            try:
                output = original_forward(*args, **kwargs)
                if self.current is not None and self.in_reference:
                    self.current["reference_logits"] = fingerprint(output.stage1_selected_logits)
                return output
            finally:
                self.in_reference = previous
        teacher.forward = teacher_forward

        from peft_common import resolve_qwen2_5_vl_model
        qwen = resolve_qwen2_5_vl_model(teacher)
        def visual(module, inputs, output):
            if self.current is not None:
                self.current["teacher_visual"].append(fingerprint(output))
        def head(module, inputs):
            if self.current is not None and self.in_reference:
                self.current["reference_head_inputs"].append(fingerprint(inputs[0]))
        self.handles.extend((qwen.visual.register_forward_hook(visual),
                             qwen.lm_head.register_forward_pre_hook(head)))
        trainer.add_callback(self)
        trainer.add_callback(MatchedCheckpointStop(1))

    def before_optimizer(self, optimizer, args, kwargs):
        if len(self.optimizer_steps) >= self.expected_optimizer_groups:
            raise ValueError("A/A probe exceeded one update's optimizer groups")
        grads = [p.grad for group in optimizer.param_groups for p in group["params"]]
        if not grads or not any(g is not None for g in grads):
            raise ValueError("probe did not observe real optimizer gradients")
        self.optimizer_steps.append({"gradients": fingerprint(grads),
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "rng_before": rng_state()})

    def after_optimizer(self, optimizer, args, kwargs):
        # ZeRO invokes the inner optimizer separately for each temporarily
        # selected group. state_dict() is invalid while other groups are absent.
        self.optimizer_steps[-1]["group_state_after"] = fingerprint(
            [optimizer.state.get(p, {}) for group in optimizer.param_groups for p in group["params"]])
        self.optimizer_steps[-1]["master_parameters_after"] = fingerprint(
            [p for group in optimizer.param_groups for p in group["params"]])
        self.optimizer_steps[-1]["rng_after"] = rng_state()

    def snapshot(self):
        return {"student_trainable": parameters(self.trainer.model),
                "teacher_trainable_state": fingerprint(self.trainer.ema_teacher.state_dict()),
                "rng": rng_state()}

    def on_train_begin(self, args, state, control, **kwargs):
        self.before = self.snapshot()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step != 1:
            raise ValueError("probe update budget violated")
        self.after = self.snapshot()
        self.after["optimizer_state"] = fingerprint(self.optimizer.state_dict())
        self.after["scheduler"] = fingerprint(self.trainer.lr_scheduler.state_dict())
        self.after["successful_updates"] = self.trainer._successful_optimizer_steps
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if (state.global_step != 1 or len(self.rows) != 4
                or len(self.optimizer_steps) != self.expected_optimizer_groups
                or self.after is None or self.after["successful_updates"] != 1
                or self.loader_workers != 4):
            raise ValueError("incomplete one-update A/A witness")
        rank = torch.distributed.get_rank()
        root = Path(__file__).resolve().parents[2] / "results" / Path(args.output_dir).name
        root.mkdir(parents=True, exist_ok=True)
        record = {"rank": rank, "local_rank": int(os.environ["LOCAL_RANK"]),
            "host": os.uname().nodename, "pid": os.getpid(),
            "versions": {name: version(name) for name in ("torch", "deepspeed", "accelerate", "peft")},
            "profile": {"variant": self.args.stage1_variant, "ema_tau": self.args.ema_tau,
                "ce_weight": self.trainer.ce_weight, "sim_weight": self.trainer.sim_weight,
                "workers": self.loader_workers, "microbatch": args.per_device_train_batch_size,
                "accumulation": args.gradient_accumulation_steps, "world_size": 4,
                "max_steps": args.max_steps, "epochs": args.num_train_epochs,
                "warmup": args.warmup_steps, "learning_rate": args.learning_rate,
                "seed": args.seed},
            "before": self.before, "microbatches": self.rows,
            "optimizer_steps": self.optimizer_steps, "after": self.after,
            "limit": "Instrumented uncached A/A first window only; LR0 first warmup update; not cache A/B or speed evidence."}
        with (root / f"aa-rank-{rank}.json").open("x") as stream:
            json.dump(record, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(f"CACHE_AA_PROBE rank={rank} completed_updates=1 microbatches=4 workers=4", flush=True)
        return control


def attach(trainer, args):
    if (args.stage1_stop_after_checkpoint != 1 or args.resume_from_checkpoint_path is not None
            or args.max_steps != 2160 or args.epochs != 10
            or args.dataloader_num_workers != 4 or args.gradient_accumulation_steps != 4
            or args.per_device_train_batch_size != 1
            or trainer.deepspeed_runtime_profile.zero_stage != 1
            or not torch.distributed.is_initialized() or torch.distributed.get_world_size() != 4):
        raise ValueError("A/A probe requires fresh four-rank global16/10epoch schedule and stop1")
    observer = Probe(trainer, args)
    observer.install()
    return observer
