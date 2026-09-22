#!/usr/bin/env python
"""TRAIN2046 native loss/backward repetitions; shared framework, no optimizer or saves."""
import argparse
import builtins
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass, replace
import hashlib
import importlib
import importlib.abc
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

from scripts import verify_voff_model_path as baseline


class ExternalSource(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Execute unchanged supplied files, redirecting only their own absolute imports."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.redirects = set()
        self.prefix = "until_supplied_" + hashlib.sha256(str(self.root).encode()).hexdigest()[:12]
        self.mapping = {
            "stage2": self.prefix + "_stage2",
            "step_preference": self.prefix + "_step",
            "artifact_paths": self.prefix + "_artifact_paths",
            "transformers.models.qwen2_5_vl": "transformers.models." + self.prefix + "_qwen",
        }
        self.folders = {
            self.mapping["stage2"]: self.root / "src/stage2",
            self.mapping["step_preference"]: self.root / "src/step_preference",
            self.mapping["artifact_paths"]: self.root / "src/artifact_paths",
            self.mapping["transformers.models.qwen2_5_vl"]:
                self.root / "transformers/src/transformers/models/qwen2_5_vl",
        }

    def __enter__(self):
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *exc):
        sys.meta_path.remove(self)
        for name in list(sys.modules):
            if any(name == p or name.startswith(p + ".") for p in self.folders):
                del sys.modules[name]

    def translated(self, name):
        for old, new in self.mapping.items():
            if name == old or name.startswith(old + "."):
                return new + name[len(old):]
        return name

    def module(self, name):
        return importlib.import_module(self.translated(name))

    def find_spec(self, fullname, path=None, target=None):
        for prefix, folder in self.folders.items():
            if fullname == prefix or fullname.startswith(prefix + "."):
                tail = fullname[len(prefix):].lstrip(".").replace(".", "/")
                location = folder / tail if tail else folder
                package = location.is_dir()
                source = location / "__init__.py" if package else location.with_suffix(".py")
                if not source.is_file():
                    raise ImportError(f"required supplied module missing: {source}")
                return importlib.util.spec_from_file_location(fullname, source, loader=self,
                    submodule_search_locations=[str(location)] if package else None)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        def supplied_import(name, globals=None, locals=None, fromlist=(), level=0):
            redirected = self.translated(name) if not level else name
            if redirected != name:
                self.redirects.add((module.__file__, name, redirected))
            return builtins.__import__(redirected, globals, locals, fromlist, level)
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=supplied_import)
        exec(compile(Path(module.__file__).read_bytes(), module.__file__, "exec"), module.__dict__)


def external_modules(source):
    return SimpleNamespace(
        model=source.module("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"),
        data=source.module("stage2.data"), trainer=source.module("stage2.trainer"),
        config=source.module("step_preference.config"), spans=source.module("step_preference.spans"),
        rollout=source.module("transformers.models.qwen2_5_vl.stage2_latent_rollout"))


def tensor_record(value):
    import torch
    if value is None:
        return {"connected": False, "finite": None, "sha256": None, "l2_norm": None}
    value = value.detach().cpu()
    finite = bool(torch.isfinite(value).all())
    return {"connected": True, "shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": baseline.tensor_digest(value), "finite": finite,
            "l2_norm": float(value.double().norm()) if finite else None}


def compare_tensors(left, right):
    if left is not None and right is not None and (left.shape != right.shape or left.dtype != right.dtype):
        return {"comparable": False, "reason": "shape_or_dtype_difference",
                "left": tensor_record(left), "right": tensor_record(right)}
    result = baseline.comparison(left, right)
    # The old comparator's tolerance is supplementary; this diagnostic makes no pass claim.
    result.pop("pass")
    return {"comparable": True, **result}


def model_identity(left, right):
    import torch
    a, b = left.state_dict(), right.state_dict()
    differences = [n for n in sorted(a.keys() | b.keys()) if n not in a or n not in b
        or a[n].shape != b[n].shape or a[n].dtype != b[n].dtype or not torch.equal(a[n], b[n])]
    trainable = [[n for n, p in m.named_parameters() if p.requires_grad] for m in (left, right)]
    return {"all_state_exact_equal": not differences, "state_differences": differences,
            "trainability_equal": trainable[0] == trainable[1], "trainable_names": trainable,
            "distinct_parameter_objects": not bool({id(p) for p in left.parameters()} &
                                                    {id(p) for p in right.parameters()})}


def state_digest(model):
    digest = hashlib.sha256()
    # named_buffers includes nonpersistent buffers that state_dict omits.
    for kind, pairs in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, value in pairs:
            digest.update((kind + name + baseline.tensor_digest(value)).encode())
    return digest.hexdigest()


def move_batch(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(device)
    if is_dataclass(value):
        return replace(value, **{f.name: move_batch(getattr(value, f.name), device) for f in fields(value)})
    if isinstance(value, Mapping):
        return {k: move_batch(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_batch(v, device) for v in value)
    return value


def rng_record(device):
    import numpy as np
    import torch
    return {"cpu": baseline.tensor_digest(torch.get_rng_state()),
            "cuda": [baseline.tensor_digest(s) for s in torch.cuda.get_rng_state_all()]
                    if device.startswith("cuda") else [],
            "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest()}


def plan_metadata(value):
    """Only token/position plans enter here; no model or activation tensors."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return {f.name: plan_metadata(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [plan_metadata(v) for v in value]
    if isinstance(value, Mapping):
        return {k: plan_metadata(v) for k, v in value.items()}
    return value


class DispatchObserver:
    """Read Python call/return events; never replace a model or objective function."""
    def __init__(self, roots):
        self.roots = tuple(str(Path(p).resolve()) + "/" for p in roots)
        self.phase = "forward"
        self.counts = Counter()
        self.components = {}
        self.backward_calls = 0
        self.recomputes = 0

    def observe(self, frame, event, arg):
        code = frame.f_code
        filename, name = code.co_filename, code.co_name
        if event == "call":
            if filename.endswith("torch/_tensor.py") and name == "backward":
                self.backward_calls += 1
            if filename.endswith("torch/utils/checkpoint.py") and name == "recompute_fn" and self.phase == "backward":
                self.recomputes += 1
            if filename.startswith(self.roots) or ("torch/utils/checkpoint.py" in filename
                    and name in ("checkpoint", "recompute_fn", "_checkpoint_without_reentrant_generator")):
                self.counts[(filename, code.co_firstlineno, code.co_qualname, self.phase)] += 1
        if event == "return" and name in ("_reduce_context_objective", "_reduce_step_preference_components"):
            if isinstance(arg, tuple) and len(arg) > 1 and isinstance(arg[1], dict):
                self.components = {k: float(v.detach().float()) if hasattr(v, "detach") else float(v)
                                   for k, v in arg[1].items()}

    def __enter__(self):
        self.previous = sys.getprofile()
        if self.previous is not None:
            raise RuntimeError("native dispatch observer requires no existing Python profiler")
        sys.setprofile(self.observe)
        return self

    def __exit__(self, *exc):
        sys.setprofile(self.previous)

    def report(self):
        return {"observer": "sys.setprofile call/return events; forward hook observes output only",
                "backward_calls": self.backward_calls, "checkpoint_recompute_calls": self.recomputes,
                "calls": [dict(file=f, line=l, function=n, phase=p, count=c)
                          for (f, l, n, p), c in sorted(self.counts.items())]}


def run_repetitions(model, trainer, batch, device, *, repeats=3, observer_roots=()):
    import numpy as np
    import torch
    parameters = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    buffers = {n: p.detach().cpu().clone() for n, p in model.named_buffers()}
    initial = state_digest(model)
    records, gradients, values = [], [], []
    for repetition in range(repeats):
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(parameters[n])
            for n, p in model.named_buffers():
                p.copy_(buffers[n])
        model.zero_grad(set_to_none=True)
        if state_digest(model) != initial:
            raise RuntimeError("same-implementation initial state reset failed")
        prepared = move_batch(batch, device)
        random.seed(173)
        np.random.seed(173)
        torch.manual_seed(173)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(173)
        rng_before = rng_record(device)
        output_values = {}
        request = {}
        def capture_request(module, positional, kwargs):
            selected = kwargs.get("stage2_logit_positions")
            if selected is None and "stage2_score_token_mask" in kwargs:
                selected = kwargs["stage2_score_token_mask"][0].nonzero().flatten()
            request.update(stage2_loss_only=kwargs.get("stage2_loss_only"),
                selected_predictor_positions=None if selected is None else selected.detach().cpu().tolist(),
                selected_target_ids=None if selected is None else kwargs["input_ids"][0, selected + 1].detach().cpu().tolist(),
                model_inputs={k: tensor_record(v) for k, v in kwargs.items()
                              if isinstance(v, torch.Tensor) and k not in {"stage2_logit_positions", "stage2_score_token_mask"}},
                native_sidecar_plan=plan_metadata(kwargs.get("stage2_sidecar_plan",
                    kwargs.get("stage2_shared_prefix_inputs", {}).get("plan"))))
        def capture(module, args, output):
            for name, attribute in (("ce_loss", "loss"), ("selected_clean_logits", "stage2_selected_logits"),
                                    ("selected_clean_logits", "stage2_scored_logits")):
                value = getattr(output, attribute, None)
                if value is not None:
                    output_values[name] = value.detach().cpu().clone()
        hook = model.register_forward_hook(capture)
        prehook = model.register_forward_pre_hook(capture_request, with_kwargs=True)
        observer = DispatchObserver(observer_roots)
        cleared = all(p.grad is None for p in model.parameters())
        try:
            with observer, torch.autocast(device_type=torch.device(device).type, enabled=False):
                loss = trainer.compute_loss(model, prepared, return_outputs=False)
                output_values["loss"] = loss.detach().cpu().clone()
                observer.phase = "backward"
                (loss / 4).backward()
        finally:
            hook.remove()
            prehook.remove()
        grads = {n: None if p.grad is None else p.grad.detach().cpu().clone()
                 for n, p in model.named_parameters() if p.requires_grad}
        records.append({"repetition": repetition, "initial_state_sha256": initial,
            "state_after_backward_sha256": state_digest(model), "gradients_cleared": cleared,
            "rng_before": rng_before, "rng_after": rng_record(device),
            "loss": float(loss.detach().float()), "backward_loss": float((loss.detach() / 4).float()),
            "loss_components": observer.components, "dispatch": observer.report(),
            "forward_request": request,
            "gradients": {n: tensor_record(g) for n, g in grads.items()},
            "output_tensors": {n: tensor_record(v) for n, v in output_values.items()}})
        gradients.append(grads)
        values.append(output_values)
        del loss, prepared
    return records, gradients, values


def make_trainers(models, external, output_dir, device, *, backend="flex"):
    from stage2.trainer import Stage2Trainer
    from step_preference.config import StepPreferenceConfig
    from transformers import TrainingArguments
    configs = [StepPreferenceConfig(enabled=True, mode="stage2b", target_scope="until_next_latent",
        beta=.5, lambda_preference=1., lambda_entropy=0., shared_prefix_enabled=True,
        visual_only_context_preference=True, visual_off_suppress_latent_boundaries=False,
        direct_from_stage1=True, sidecar_attention_backend=backend, sidecar_query_budget=1024),
        external.config.StepPreferenceConfig(enabled=True, beta=.5, lambda_preference=1.,
                                            lambda_entropy=0., modalities="visual")]
    result = []
    for index, (model, cls, config) in enumerate(zip(models, (Stage2Trainer, external.trainer.Stage2Trainer), configs)):
        args = TrainingArguments(output_dir=str(Path(output_dir) / ("local" if index == 0 else "external")),
            use_cpu=device == "cpu", report_to=[], save_strategy="no", logging_first_step=True,
            per_device_train_batch_size=1, gradient_accumulation_steps=4, seed=42,
            bf16=False, fp16=False, remove_unused_columns=False)
        result.append(cls(model=model, args=args, step_preference_config=config))
    return result


def describe_batch(batch):
    import torch
    spans = batch["reasoning_step_spans"]
    if len(spans) == 1 and isinstance(spans[0], (list, tuple)):
        spans = spans[0]
    ids = batch["input_ids"]
    return {"tensors": {k: tensor_record(v) for k, v in batch.items() if isinstance(v, torch.Tensor)},
            "sample_ids": batch["sample_ids"], "spans": [asdict(s) for s in spans],
            "targets": [{"step_id": s.step_id,
                         "predictor_positions": list(range(s.target_text_start-1, s.target_text_end-1)),
                         "target_ids": ids[0, s.target_text_start:s.target_text_end].tolist()} for s in spans]}


def prepare_batches(args, processor, external):
    from stage2 import data
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
    manifest = json.loads(Path(args.manifest).read_text())
    if baseline.sha(args.data) != manifest["data_sha256"]:
        raise ValueError("TRAIN bytes differ from retained manifest")
    if 2046 not in manifest["retained_indices"] or manifest["latent_block_counts"][2046] != 2:
        raise ValueError("TRAIN2046 is not a retained two-block sample")
    if (manifest["latent_size"], manifest["supervise_latent_boundaries"], manifest["assistant_plan_rendering"]) != (8, True, "concatenated-v1"):
        raise ValueError("retained manifest preparation settings differ")
    sample = data.load_jsonl_examples(Path(args.data))[2046]
    ids = baseline.token_ids(processor.tokenizer)[:3]
    local = data.prepare_stage2_example(sample, image_root=Path(args.image_root), latent_size=8,
        assistant_plan_rendering="concatenated-v1", require_reasoning_units=True)
    supplied = external.data.prepare_stage2_example(sample, image_root=Path(args.image_root), latent_size=8)
    collators = [data.Stage2DataCollator(processor, LatentTokenIds(*ids), 8, True,
                    step_preference_enabled=True, preference_target_scope="until_next_latent"),
                external.data.Stage2DataCollator(processor, external.rollout.LatentTokenIds(*ids), 8, True,
                                                context_preference_enabled=True)]
    batches = [collator([example]) for collator, example in zip(collators, (local, supplied))]
    if batches[0]["input_ids"].shape[1] != manifest["lengths"][2046]:
        raise ValueError("local TRAIN2046 length differs from retained manifest")
    return batches


def comparisons(gradients, values, *, cross_allowed):
    output = {}
    for side in range(2):
        for a, b in itertools.combinations(range(3), 2):
            output[f"{'local' if side == 0 else 'external'}:{a}-{b}"] = {
                "gradients": {n: compare_tensors(gradients[side][a][n], gradients[side][b][n])
                              for n in gradients[side][a]},
                "outputs": {n: compare_tensors(values[side][a][n], values[side][b][n])
                            for n in values[side][a]}}
    if cross_allowed:
        for a, b in itertools.product(range(3), repeat=2):
            output[f"local:{a}-external:{b}"] = {
                "gradients": {n: compare_tensors(gradients[0][a][n], gradients[1][b][n])
                              for n in gradients[0][a]},
                "outputs": {n: compare_tensors(values[0][a][n], values[1][b][n])
                            for n in values[0][a].keys() & values[1][b].keys()}}
    return output


def run(args):
    import torch
    from transformers import AutoProcessor
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("native TRAIN2046 diagnostic requires CUDA")
    if torch.distributed.is_initialized():
        raise ValueError("diagnostic must not join an active distributed process group")
    root = Path(args.external_source).resolve(strict=True)
    source_before = {str(p): baseline.sha(p) for p in root.rglob("*.py")}
    inventory_before = baseline.inventory(root)
    torch.set_num_threads(2)
    with ExternalSource(root) as source:
        external = external_modules(source)
        processor = AutoProcessor.from_pretrained(args.stage1, local_files_only=True)
        roots = [str(Path("src").resolve()), str(Path("transformers").resolve()), str(root)]
        with DispatchObserver(roots) as preparation_observer:
            batches = prepare_batches(args, processor, external)
        models = baseline.real_models(args, processor, external.model, verify_initial_state=False)
        identity = model_identity(*models)
        runtime = [baseline.runtime_model_metadata(m, production=True) for m in models]
        trainers = make_trainers(models, external, args.log_dir, args.device)
        inputs = [describe_batch(b) for b in batches]
        input_equal = inputs[0]["tensors"] == inputs[1]["tensors"]
        target_equal = inputs[0]["targets"] == inputs[1]["targets"]
        cross_allowed = input_equal and target_equal and identity["all_state_exact_equal"] and identity["trainability_equal"]
        records, gradients, values = [], [], []
        for model, trainer, batch in zip(models, trainers, batches):
            r, g, v = run_repetitions(model, trainer, batch, args.device, observer_roots=roots)
            records.append(r)
            gradients.append(g)
            values.append(v)
        observed_coordinate_equal = all(
            records[0][a]["forward_request"][key] == records[1][b]["forward_request"][key]
            for a, b in itertools.product(range(3), repeat=2)
            for key in ("model_inputs", "selected_predictor_positions", "selected_target_ids"))
        cross_allowed = cross_allowed and observed_coordinate_equal
        source_after = {str(p): baseline.sha(p) for p in root.rglob("*.py")}
        unchanged = source_before == source_after and inventory_before == baseline.inventory(root)
        if not unchanged:
            raise RuntimeError("external source changed during diagnostic")
        modules = {name: {"file": m.__file__, "sha256": baseline.sha(m.__file__),
                          "supplied": Path(m.__file__).resolve().is_relative_to(root)}
                   for name, m in tuple(sys.modules.items()) if getattr(m, "__file__", None)
                   and str(m.__file__).endswith(".py") and Path(m.__file__).is_file()}
        return {"scope": "TRAIN2046 single-microbatch native objective under shared available framework; "
                         "external exported source is reconstructed execution, not authenticated historical runtime",
            "decision_rule": "Three repetitions are descriptive; no equivalence threshold or code-error attribution",
            "train_index": 2046, "repetitions_per_side": 3, "beta": .5, "preference_weight": 1.,
            "entropy_weight": 0., "common_backward_divisor": 4, "optimizer_steps": 0,
            "initialization": {"base": args.base, "stage1": args.stage1},
            "numerics": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "device": args.device, "hardware": torch.cuda.get_device_name(args.device),
                "parameter_dtype": "bfloat16", "autocast": False,
                "matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "native_decoder_checkpoint": False, "latent_activation_checkpoint": True,
                "local_sidecar_backend": "flex", "local_sidecar_query_budget": 1024,
                "external_sidecar_backend": "native split attention Flex; native packing budget"},
            "runtime_models": runtime, "initial_model_identity": identity,
            "inputs": inputs, "input_tensors_equal": input_equal, "targets_equal": target_equal,
            "observed_forward_coordinates_equal": observed_coordinate_equal,
            "isolated_cross_comparison_allowed": cross_allowed,
            "cross_comparison_block_reason": None if cross_allowed else "input, target, initial-state, or observed logit-coordinate inequality",
            "preparation_dispatch": preparation_observer.report(), "runs": records,
            "comparisons": comparisons(gradients, values, cross_allowed=cross_allowed),
            "external_files_unchanged": unchanged, "external_source_hashes": source_before,
            "loaded_modules": modules, "redirected_imports": sorted(source.redirects),
            "compatibility_shims": ["Supplied absolute imports redirected to isolated supplied namespaces; relative imports unchanged",
                "Shared current Transformers Trainer/framework, PEFT, stage1 loader/token rows, processor and dependencies",
                "Existing comparator normalizes platform.python_version trailing + for shared runtime compatibility",
                "No Trainer.train/accelerator backward/ZeRO engine; native compute_loss then tensor loss/4 backward",
                "BF16 whole-module cast follows existing comparator; TrainingArguments mixed precision is disabled"],
            "comparator_sha256": baseline.sha(__file__), "loader_sha256": baseline.sha(baseline.__file__),
            "data_sha256": baseline.sha(args.data), "manifest_sha256": baseline.sha(args.manifest)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("external-source", baseline.SOURCE), ("stage1", baseline.STAGE1),
                          ("base", baseline.BASE), ("manifest", baseline.MANIFEST),
                          ("data", "data/TRAIN.jsonl"), ("image-root", "data"), ("device", "cuda:0"),
                          ("log-dir", "logs/until-native-v2-20260922")):
        parser.add_argument("--" + name, default=default)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), allow_nan=False))
