#!/usr/bin/env python
"""TRAIN2046 native loss/backward repetitions; shared framework, no optimizer or saves."""
import argparse
import builtins
from contextlib import nullcontext
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
import threading
from types import SimpleNamespace
import weakref

from scripts import verify_voff_model_path as baseline


class ExternalSource(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Execute unchanged supplied files, redirecting only their own absolute imports."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.redirects = set()
        self.prefix = "until_supplied_" + hashlib.sha256(str(self.root).encode()).hexdigest()[:12]
        self.mapping = {
            "stage2": self.prefix + "_stage2",
            "stage2_objective": self.prefix + "_objective",
            "step_preference": self.prefix + "_step",
            "artifact_paths": self.prefix + "_artifact_paths",
            "transformers.models.qwen2_5_vl": "transformers.models." + self.prefix + "_qwen",
        }
        self.folders = {
            self.mapping["stage2"]: self.root / "src/stage2",
            self.mapping["stage2_objective"]: self.root / "src/stage2_objective",
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
    # Runtime buffers such as RoPE inv_freq can be nonpersistent in state_dict.
    a, b = [dict(itertools.chain(m.named_parameters(), m.named_buffers())) for m in (left, right)]
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


class NativeComponents:
    """Retain the actual native reducer graph, never reconstructed log objectives."""
    def __init__(self, trainer):
        self.trainer = trainer
        self.tensors, self.native, self.scores = {}, {}, []
        name = ("_reduce_context_objective" if hasattr(trainer, "_reduce_context_objective")
                else "_reduce_step_preference_components")
        self.reducer_code = getattr(trainer, name).__func__.__code__
        self.reducer_calls = 0

    def observe(self, frame, event, arg):
        if event != "return":
            return
        local = frame.f_locals
        if frame.f_code.co_name == "context_preference_loss" and "clean_logps" in local:
            # Both native implementations use this primitive; preserve its actual inputs.
            self.scores.append({key: local[name].detach().float().cpu().reshape(-1).tolist()
                                for key, name in (("clean", "clean_logps"), ("off", "ablated_logps"))})
        if frame.f_code is not self.reducer_code or local.get("self") is not self.trainer:
            return
        self.reducer_calls += 1
        if not isinstance(arg, tuple) or len(arg) < 2:
            raise ValueError("native component capture: unexpected reducer return")
        ce = local.get("clean_ce", local.get("ce_loss"))
        pref = local["preference"] if "clean_ce" in local else local["preference_all"]
        self.native = {"ce": ce, "preference": pref, "total": arg[0]}
        self.tensors = dict(self.native)

    def validate(self, total):
        import torch
        if self.reducer_calls != 1 or not self.scores or self.native.get("total") is not total:
            raise ValueError("native component capture missing or ambiguous")
        if any(self.tensors.get(k) is not v for k, v in self.native.items()):
            raise ValueError("native component capture identity changed")
        config = self.trainer.step_preference_config
        if (config.beta, config.lambda_preference, config.lambda_entropy) != (.5, 1., 0.):
            raise ValueError("native component capture requires beta0.5/preference1/entropy0")
        for value in self.tensors.values():
            if not isinstance(value, torch.Tensor) or value.numel() != 1 or not bool(torch.isfinite(value)):
                raise ValueError("native component capture missing or nonfinite scalar")
        # This tests graph membership, independently of scalar reconstruction. A detached
        # genuine native component is reported separately and cannot become a valid screen.
        connected = [self.tensors[k] for k in ("ce", "preference") if self.tensors[k].requires_grad]
        if not total.requires_grad:
            raise ValueError("native total is graph-disconnected; screen invalid")
        coefficients = torch.autograd.grad(total, connected, retain_graph=True, allow_unused=True)
        if any(c is None or not torch.equal(c, torch.ones_like(c)) for c in coefficients):
            raise ValueError("native component capture not connected to total with unit weight")
        residual = float((total.detach().double() - self.tensors["ce"].detach().double()
                          - self.tensors["preference"].detach().double()).abs())
        tolerance = 4 * torch.finfo(total.dtype).eps * max(1., abs(float(total.detach())))
        if residual > tolerance:
            raise ValueError("native scalar loss reconstruction failed")
        return {"identity_validated": True, "scalar_sum_residual": residual,
                "scalar_roundoff_bound": tolerance,
                "requires_grad": {k: v.requires_grad for k, v in self.tensors.items()},
                "native_graph_status": {k: "connected_to_native_total" if v.requires_grad
                                        else "native_requires_grad_false" for k, v in self.tensors.items()},
                "values": {k: float(v.detach().float()) for k, v in self.tensors.items()},
                "scores": self.scores}

    def select(self, component):
        if component not in self.tensors:
            raise ValueError("unknown or missing native component capture")
        return self.tensors[component]


def gradient_geometry(ce, preference, total):
    """Double-precision scalar measurements only; None remains distinct from zero."""
    import torch
    if ce.keys() != preference.keys() or ce.keys() != total.keys():
        raise ValueError("gradient parameter coordinates differ")
    sums = dict(ce=0., preference=0., total=0., dot=0., residual=0.)
    layers, connectivity, mismatches = {}, {}, []
    for name in ce:
        values = dict(ce=ce[name], preference=preference[name], total=total[name])
        connectivity[name] = {k: v is not None for k, v in values.items()}
        if (ce[name] is not None or preference[name] is not None) != (total[name] is not None):
            mismatches.append(name)
        live = [v for v in values.values() if v is not None]
        if not live:
            continue
        if any(v.shape != live[0].shape or not bool(torch.isfinite(v).all()) for v in live):
            raise ValueError("nonfinite gradient or changed parameter coordinates")
        a, b, t = [torch.zeros_like(live[0], dtype=torch.float64) if values[k] is None
                   else values[k].double() for k in ("ce", "preference", "total")]
        measures = dict(ce=float(a.square().sum()), preference=float(b.square().sum()),
                        total=float(t.square().sum()), dot=float((a*b).sum()),
                        residual=float((t-a-b).square().sum()))
        parts = name.split(".")
        layer = ".".join(parts[:parts.index("layers")+2]) if "layers" in parts else name.rsplit(".", 1)[0]
        group = layers.setdefault(layer, dict.fromkeys(sums, 0.))
        for key, value in measures.items():
            sums[key] += value
            group[key] += value
    def finish(group):
        a, b = group["ce"] ** .5, group["preference"] ** .5
        return {"ce_norm": a, "preference_norm": b, "total_norm": group["total"] ** .5,
                "cosine": group["dot"] / (a*b) if a*b else None,
                "preference_ce_norm_ratio": b/a if a else None,
                "residual_ce_norm_ratio": group["residual"] ** .5/a if a else None,
                "residual_preference_norm_ratio": group["residual"] ** .5/b if b else None,
                "sum_residual_l2": group["residual"] ** .5}
    return {**finish(sums), "layers": {k: finish(v) for k, v in layers.items()},
            "connectivity": connectivity, "connection_mismatches": mismatches}


class DispatchObserver:
    """Read Python call/return events; never replace a model or objective function."""
    def __init__(self, roots, *, model=None, capture=None, precision=False):
        self.roots = tuple(str(Path(p).resolve()) + "/" for p in roots)
        self.phase = "forward"
        self.counts = Counter()
        self.components = {}
        self.backward_calls = 0
        self.recomputes = 0
        self.model = model
        self.regions = {}
        self.recomputed = set()
        self.recompute_threads = set()
        self.native_before = None
        self.native_after = None
        self.capture = capture
        self.precision_modules = {}
        self.precision_activations = {}
        if precision:
            from peft_common import resolve_qwen2_5_vl_model
            qwen = resolve_qwen2_5_vl_model(model)
            self.precision_modules = {qwen.visual: "vision", qwen.model.embed_tokens: "embedding", qwen.lm_head: "head",
                **{layer.self_attn: f"attention.{i}" for i, layer in enumerate(qwen.model.layers)}}

    def observe_precision(self, module, args, kwargs, output):
        import torch
        def floating_dtypes(value):
            if isinstance(value, torch.Tensor):
                return {str(value.dtype)} if value.is_floating_point() else set()
            if isinstance(value, Mapping):
                value = value.values()
            elif not isinstance(value, (tuple, list)):
                return set()
            return set().union(*(floating_dtypes(item) for item in value))
        role = self.precision_modules[module]
        inputs = floating_dtypes((args, kwargs))
        outputs = floating_dtypes(output)
        expected = "torch.bfloat16" if role == "vision" else "torch.float32"
        if (inputs | outputs) - {expected} or not outputs:
            raise ValueError(f"diagnostic activation dtype mismatch: {role}")
        record = self.precision_activations.setdefault(role, {"calls": 0, "input_dtypes": [], "output_dtypes": []})
        record["calls"] += 1
        record["input_dtypes"] = sorted(set(record["input_dtypes"]) | inputs)
        record["output_dtypes"] = sorted(set(record["output_dtypes"]) | outputs)

    def observe_module(self, module, inputs):
        """Module hooks execute on the calling worker, unlike sys.setprofile.

        Weak checkpoint references prevent recycled Python frame IDs from merging
        distinct calls, without retaining checkpoint graphs or activations.
        """
        frame = sys._getframe(1)
        try:
            while frame is not None:
                if frame.f_code.co_filename.endswith("torch/utils/checkpoint.py"):
                    local = frame.f_locals
                    if frame.f_code.co_name == "checkpoint":
                        generator = local.get("gen")
                        generator_frame = getattr(generator, "gi_frame", None)
                        checkpoint_frame = None if generator_frame is None else generator_frame.f_locals.get("new_frame")
                        if checkpoint_frame is not None:
                            key = weakref.ref(checkpoint_frame)
                            function = local.get("function")
                            code = getattr(function, "__code__", None)
                            self.regions.setdefault(key, {"file": None if code is None else code.co_filename,
                                "function": type(function).__name__ if code is None else code.co_qualname})
                    elif frame.f_code.co_name == "unpack_hook" and "gid" in local and "frame" in local:
                        key = weakref.ref(local["frame"])
                        self.recomputed.add((key, local["gid"]))
                        self.recompute_threads.add(threading.get_ident())
                frame = frame.f_back
        finally:
            del frame

    def observe(self, frame, event, arg):
        if self.capture is not None and self.phase == "forward":
            self.capture.observe(frame, event, arg)
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
        self.hooks = [] if self.model is None else [
            module.register_forward_pre_hook(self.observe_module) for module in self.model.modules()]
        self.hooks += [module.register_forward_hook(self.observe_precision, with_kwargs=True)
                       for module in self.precision_modules]
        return self

    def __exit__(self, *exc):
        sys.setprofile(self.previous)
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def report(self):
        if self.precision_modules and set(self.precision_activations) != set(self.precision_modules.values()):
            raise ValueError("diagnostic activation dtype witness missing")
        module_count = len(self.recomputed)
        recomputed_regions = {key for key, graph_task in self.recomputed}
        native_delta = None if self.native_before is None or self.native_after is None else self.native_after - self.native_before
        native_matches = None if native_delta is None else native_delta == module_count
        complete = (bool(self.regions) and set(self.regions) == recomputed_regions
                    and native_matches is not False)
        return {**({"precision_activations": self.precision_activations} if self.precision_modules else {}),
                "observer": "Caller-thread sys.setprofile plus nonmutating module prehooks observing checkpoint stacks on execution threads",
                "backward_calls": self.backward_calls,
                "checkpoint_recompute_calls": module_count if complete else self.recomputes,
                "checkpoint_recompute_count_status": "module_regions_witnessed" if complete else "caller_thread_lower_bound_only",
                "caller_thread_checkpoint_recompute_calls": self.recomputes,
                "module_checkpoint_recompute_calls": module_count,
                "native_counter_before_backward": self.native_before,
                "native_counter_after_backward": self.native_after,
                "native_counter_matches_module_witness": native_matches,
                "checkpoint_coverage": {
                    "complete_for_observed_module_regions": complete,
                    "forward_module_checkpoint_regions": len(self.regions),
                    "recomputed_module_checkpoint_regions": len(recomputed_regions),
                    "module_recompute_thread_ids": sorted(self.recompute_threads),
                    "regions": list(self.regions.values()),
                    "scope": "Nonreentrant checkpoints reaching model module boundaries; caller Python profiler remains thread-local",
                    "limitation": "Pure tensor checkpoints without module boundaries are not covered; absent/mismatching region or native-counter evidence is incomplete"},
                "calls": [dict(file=f, line=l, function=n, phase=p, count=c)
                          for (f, l, n, p), c in sorted(self.counts.items())]}


def component_checkpoint_coverage(observer, component):
    report = observer.report()
    observed = {key for key, _ in observer.recomputed}
    known = set(observer.regions)
    complete = report["checkpoint_coverage"]["complete_for_observed_module_regions"]
    valid = bool(observed) and observed <= known and report["native_counter_matches_module_witness"] is not False
    if not valid or (component == "total" and not complete):
        raise ValueError("native checkpoint coverage unverified")
    return {"observed_component_valid": valid, "total_coverage_verified": complete if component == "total" else False,
            "omitted_forward_regions": len(known - observed),
            "omitted_regions": [observer.regions[key] for key in known - observed],
            "scope": "Observed component subset only; total coverage and gradient reconstruction are separate gates"}


def run_repetitions(model, trainer, batch, device, *, repeats=3, observer_roots=(), component=None, precision=False):
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
        component_capture = None if component is None else NativeComponents(trainer)
        observer = DispatchObserver(observer_roots, model=model, capture=component_capture, precision=precision)
        cleared = all(p.grad is None for p in model.parameters())
        try:
            with observer, torch.autocast(device_type=torch.device(device).type, enabled=False):
                loss = trainer.compute_loss(model, prepared, return_outputs=False)
                component_record = None if component_capture is None else component_capture.validate(loss)
                backward_value = loss if component_capture is None else component_capture.select(component)
                output_values["loss"] = loss.detach().cpu().clone()
                native_stats = getattr(trainer, "_latest_step_preference_compute_stats", None)
                observer.native_before = getattr(native_stats, "checkpoint_recompute_calls", None)
                observer.phase = "backward"
                if component_capture is None or backward_value.requires_grad:
                    (backward_value / 4).backward()
                observer.native_after = getattr(native_stats, "checkpoint_recompute_calls", None)
        finally:
            hook.remove()
            prehook.remove()
        grads = {n: None if p.grad is None else p.grad.detach().cpu().clone()
                 for n, p in model.named_parameters() if p.requires_grad}
        if precision and any(g is not None and g.dtype != torch.float32 for g in grads.values()):
            raise ValueError("diagnostic gradient dtype mismatch")
        records.append({"repetition": repetition, "initial_state_sha256": initial,
            "state_after_backward_sha256": state_digest(model), "gradients_cleared": cleared,
            "rng_before": rng_before, "rng_after": rng_record(device),
            "loss": float(loss.detach().float()), "backward_loss": float((backward_value.detach() / 4).float()),
            "component": component, "component_capture": component_record,
            "component_checkpoint_coverage": component_checkpoint_coverage(observer, component)
                if component is not None and observer.regions and backward_value.requires_grad else None,
            "loss_components": observer.components, "dispatch": observer.report(),
            "forward_request": request,
            "gradients": {n: tensor_record(g) for n, g in grads.items()},
            "output_tensors": {n: tensor_record(v) for n, v in output_values.items()}})
        gradients.append(grads)
        values.append(output_values)
        if component is not None:
            print(json.dumps({"progress": "component_repeat", "component": component,
                              "repetition": repetition, "loss": records[-1]["backward_loss"]}), file=sys.stderr, flush=True)
        del loss, prepared
        # Every component call must leave the identical initializer for the next call.
    if component is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(parameters[n])
            for n, p in model.named_buffers():
                p.copy_(buffers[n])
        model.zero_grad(set_to_none=True)
    return records, gradients, values


def make_trainers(models, external, output_dir, device, *, backend="flex"):
    from stage2.trainer import Stage2Trainer
    from step_preference.config import StepPreferenceConfig
    from transformers import TrainingArguments
    configs = [StepPreferenceConfig(enabled=True, mode="stage2b", target_scope="until_next_latent",
        beta=.5, lambda_ce=1., preference_margin=0., lambda_preference=1., lambda_entropy=0., shared_prefix_enabled=True,
        visual_only_context_preference=True, visual_off_suppress_latent_boundaries=False,
        direct_from_stage1=True, sidecar_attention_backend=backend, sidecar_query_budget=1024)]
    classes = [Stage2Trainer]
    if external is not None:
        configs.append(external.config.StepPreferenceConfig(enabled=True, beta=.5, lambda_preference=1.,
                                            lambda_entropy=0., modalities="visual"))
        classes.append(external.trainer.Stage2Trainer)
    result = []
    for index, (model, cls, config) in enumerate(zip(models, classes, configs)):
        if model is None:
            result.append(None)
            continue
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
    batches = [data.Stage2DataCollator(processor, LatentTokenIds(*ids), 8, True,
                    step_preference_enabled=True, preference_target_scope="until_next_latent")([local])]
    if external is not None:
        supplied = external.data.prepare_stage2_example(sample, image_root=Path(args.image_root), latent_size=8)
        batches.append(external.data.Stage2DataCollator(processor, external.rollout.LatentTokenIds(*ids), 8, True,
                                                context_preference_enabled=True)([supplied]))
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
    if args.decoder_dtype == "float32":
        validate_component_args(args)
    if args.mode == "components":
        return run_components(args)
    import torch
    from transformers import AutoProcessor
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("native TRAIN2046 diagnostic requires CUDA")
    if torch.distributed.is_initialized():
        raise ValueError("diagnostic must not join an active distributed process group")
    root = Path(args.external_source or baseline.SOURCE).resolve(strict=True)
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
        checkpoint_coverage_verified = all(r["dispatch"]["checkpoint_coverage"]["complete_for_observed_module_regions"]
                                           for side in records for r in side)
        cross_allowed = cross_allowed and checkpoint_coverage_verified
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
            "checkpoint_coverage_verified": checkpoint_coverage_verified,
            "isolated_cross_comparison_allowed": cross_allowed,
            "cross_comparison_block_reason": None if cross_allowed else "input, target, initial-state, logit-coordinate inequality or incomplete checkpoint observation",
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


def validate_component_args(args):
    if args.decoder_dtype == "float32" and (args.mode, args.side, args.initializer) != ("components", "local", "e1"):
        raise ValueError("FP32 diagnostic requires local e1 component mode")
    if args.side == "external" and not args.external_source:
        raise ValueError("component mode requires an explicit --external-source for external execution")


def prepare_component_precision(model, decoder_dtype, *, production=True):
    """Authenticate the BF16 initializer before an explicitly diagnostic promotion."""
    import torch
    from peft_common import resolve_qwen2_5_vl_model
    runtime = baseline.runtime_model_metadata(model, production=production)
    if decoder_dtype == "bfloat16":
        return runtime
    if decoder_dtype != "float32":
        raise ValueError("unsupported diagnostic decoder dtype")
    qwen = resolve_qwen2_5_vl_model(model)
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError("FP32 control requires an all-BF16 rounded initializer")
    if any(p.requires_grad for p in qwen.visual.parameters()):
        raise ValueError("FP32 control requires frozen vision")
    if torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("FP32 control requires highest matmul precision and TF32 disabled")
    vision_ids = {id(v) for v in itertools.chain(qwen.visual.parameters(), qwen.visual.buffers())}
    identities = {n: id(p) for n, p in model.named_parameters()}
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    initial = state_digest(model)
    expected = hashlib.sha256()
    vision_names = set()
    for kind, pairs in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, value in pairs:
            if id(value) in vision_ids:
                vision_names.add(name)
            elif value.is_floating_point():
                value = value.float()
            expected.update((kind + name + baseline.tensor_digest(value)).encode())
    # Preserve PEFT wrapper objects, shared parameters and the BF16 visual module.
    qwen.model.float()
    qwen.lm_head.float()
    for name, value in itertools.chain(model.named_parameters(), model.named_buffers()):
        dtype = torch.bfloat16 if name in vision_names else torch.float32
        if value.is_floating_point() and value.dtype != dtype:
            raise ValueError(f"diagnostic precision partition mismatch: {name}")
    if identities != {n: id(p) for n, p in model.named_parameters()} or trainable != [n for n, p in model.named_parameters() if p.requires_grad]:
        raise ValueError("diagnostic promotion changed parameter identity or trainability")
    actual = state_digest(model)
    if actual != expected.hexdigest():
        raise ValueError("diagnostic promotion changed BF16-rounded state values")
    runtime.update(parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        trainable_parameter_dtypes={n: str(p.dtype) for n, p in model.named_parameters() if p.requires_grad},
        precision_control={"initial_bf16_state_sha256": initial,
            "expected_promoted_sha256": expected.hexdigest(), "actual_promoted_sha256": actual,
            "promoted_state_exact": True, "parameter_identity_preserved": True,
            "vision_frozen": True, "vision_dtype": "torch.bfloat16", "decoder_dtype": "torch.float32",
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "scope": "BF16-rounded parameters promoted exactly; module dtypes do not establish kernel internals"})
    return runtime


def component_measurements(records, gradients):
    """Require observed repeat evidence, never an invented dtype tolerance."""
    import math
    flat = [r for rows in records.values() for r in rows]
    if set(records) != {"ce", "preference", "total"} or any(len(rows) != 3 for rows in records.values()):
        raise ValueError("three repeats per component required")
    disconnected = set()
    for row in flat:
        if row["initial_state_sha256"] != flat[0]["initial_state_sha256"]:
            raise ValueError("component initializer coordinates changed")
        if row["state_after_backward_sha256"] != row["initial_state_sha256"]:
            raise ValueError("native model state mutated")
        if row["forward_request"] != flat[0]["forward_request"] or row["rng_before"] != flat[0]["rng_before"]:
            raise ValueError("native input/target coordinates or RNG differ")
        if any(g["finite"] is False for g in row["gradients"].values()):
            raise ValueError("nonfinite native gradient")
        component = row.get("component")
        capture = row.get("component_capture") or {}
        native_disconnected = (component in ("ce", "preference") and capture.get("identity_validated") is True
            and capture.get("requires_grad", {}).get(component) is False
            and capture.get("native_graph_status", {}).get(component) == "native_requires_grad_false")
        if native_disconnected:
            if (row["component_checkpoint_coverage"] is not None or row["dispatch"]["backward_calls"] != 0
                    or any(g["connected"] for g in row["gradients"].values())):
                raise ValueError("native disconnected component has unexpected backward or gradient evidence")
            disconnected.add(component)
        elif row["component_checkpoint_coverage"] is None:
            raise ValueError("native graph or checkpoint coverage unverified")
    geometry = [gradient_geometry(*(gradients[k][i] for k in ("ce", "preference", "total"))) for i in range(3)]
    spreads = {}
    for key, repeats in gradients.items():
        distances = []
        for left, right in itertools.combinations(repeats, 2):
            squared = 0.
            for name in left:
                if (left[name] is None) != (right[name] is None):
                    raise ValueError("native connectivity varies across identical repeats")
                if left[name] is not None:
                    squared += float((left[name].double()-right[name].double()).square().sum())
            distances.append(squared ** .5)
        spreads[key] = max(distances)
    bound = sum(spreads.values())
    gradient_valid = all(not row["connection_mismatches"] and row["sum_residual_l2"] <= bound for row in geometry)
    forward = {}
    for component, rows in records.items():
        forward[component] = []
        for row in rows:
            capture = row.get("component_capture")
            if not capture or not capture["identity_validated"] or not capture["scores"]:
                raise ValueError("native forward component capture missing")
            values = dict(capture["values"])
            for group, pair in enumerate(capture["scores"]):
                for side in ("clean", "off"):
                    values.update({f"score.{group}.{side}.{i}": value for i, value in enumerate(pair[side])})
            if any(not math.isfinite(value) for value in values.values()):
                raise ValueError("nonfinite native forward value")
            forward[component].append(values)
    keys = forward["total"][0].keys()
    if any(values.keys() != keys for rows in forward.values() for values in rows):
        raise ValueError("native forward score coordinates differ")
    forward_metrics = {}
    for key in keys:
        spans = {component: max(v[key] for v in rows)-min(v[key] for v in rows) for component, rows in forward.items()}
        cross = max(abs(a[key]-b[key]) for left, right in itertools.combinations(forward.values(), 2)
                    for a, b in itertools.product(left, right))
        forward_metrics[key] = {"repeat_spreads": spans, "max_cross_component_difference": cross,
                                "within_observed_repeat_spread": cross <= sum(spans.values())}
    forward_valid = all(value["within_observed_repeat_spread"] for value in forward_metrics.values())
    valid = gradient_valid and forward_valid and not disconnected
    return {"geometry": geometry, "repeat_gradient_l2_spread": spreads, "sum_repeat_spread_bound": bound,
            "native_disconnected_components": sorted(disconnected),
            "forward_value_comparisons": forward_metrics,
            "forward_reconstruction_within_observed_repeat_spread": forward_valid,
            "gradient_reconstruction_within_observed_repeat_spread": gradient_valid,
            "reconstruction_within_observed_repeat_spread": valid,
            "reconstruction_status": "inconclusive_native_disconnection" if disconnected else (
                "within_observed_repeat_spread" if valid else "unresolved_beyond_observed_repeat_spread"),
            "interpretation": "Beyond-spread residual is unresolved, not a code-error diagnosis"}


def screen_initializers(cells):
    if set(cells) != {"e1", "e3"} or not all(c["measurements"]["reconstruction_within_observed_repeat_spread"] for c in cells.values()):
        return {"status": "inconclusive", "source_expansion_candidate": False,
                "reason": "Both cells and native gradient reconstruction required"}
    intervals = {}
    for key in ("cosine", "preference_ce_norm_ratio"):
        intervals[key] = {}
        for name, cell in cells.items():
            values = [r[key] for r in cell["measurements"]["geometry"]]
            if any(value is None for value in values):
                return {"status": "inconclusive", "source_expansion_candidate": False,
                        "reason": "Zero/disconnected gradient makes cosine or ratio undefined"}
            intervals[key][name] = [min(values), max(values)]
    def gap(pair):
        a, b = pair["e1"], pair["e3"]
        return max(a[0]-b[1], b[0]-a[1])
    cosine, ratio = intervals["cosine"], intervals["preference_ce_norm_ratio"]
    opposition = gap(cosine) > 0 and all(v[1] < 0 for v in cosine.values())
    scale = gap(ratio) > max(v[1]-v[0] for v in ratio.values())
    return {"status": "exploratory_candidate" if opposition or scale else "inconclusive",
            "opposition_candidate": opposition, "scale_candidate": scale, "intervals": intervals,
            "source_expansion_candidate": opposition or scale,
            "interpretation": "Initial single-sample geometry does not establish late-collapse causality"}


def run_component_cell(model, trainer, batch, device, roots, *, decoder_dtype=None, production=True):
    runtime = {} if decoder_dtype is None else {"runtime": prepare_component_precision(model, decoder_dtype, production=production)}
    records, gradients = {}, {}
    for component in ("total", "ce", "preference"):
        records[component], gradients[component], _ = run_repetitions(
            model, trainer, batch, device, observer_roots=roots, component=component, precision=decoder_dtype == "float32")
    return {**runtime, "runs": records, "measurements": component_measurements(records, gradients)}


def run_components(args):
    import gc
    import torch
    validate_component_args(args)
    if not args.device.startswith("cuda") or not torch.cuda.is_available() or torch.distributed.is_initialized():
        raise ValueError("component diagnostic requires independent CUDA execution")
    from stage2.attention_backend import load_qwen2_5_vl_with_attention
    from stage2.checkpointing import configure_stage2_metadata
    from stage2.peft import Stage2TuningConfig, load_stage1_dora_for_fresh_stage2
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
    torch.set_num_threads(2)
    root = Path(args.external_source).resolve(strict=True) if args.side == "external" else None
    before = None if root is None else baseline.inventory(root)
    hashes = {} if root is None else {str(p): baseline.sha(p) for p in root.rglob("*.py")}
    selected = {"e1": args.stage1, "e3": args.stage1_e3}
    if args.initializer != "both":
        selected = {args.initializer: selected[args.initializer]}
    cells, common_input = {}, None
    with (ExternalSource(root) if root is not None else nullcontext()) as source:
        external = None if source is None else external_modules(source)
        for name, checkpoint in selected.items():
            print(json.dumps({"progress": "initializer", "side": args.side, "initializer": name,
                              "checkpoint": checkpoint}), file=sys.stderr, flush=True)
            options = SimpleNamespace(**vars(args))
            options.stage1 = checkpoint
            processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
            batches = prepare_batches(options, processor, external)
            if external is not None:
                local, supplied = map(describe_batch, batches)
                if local["tensors"] != supplied["tensors"] or local["targets"] != supplied["targets"]:
                    raise ValueError("external input/target coordinates differ from local")
            batch = batches[-1]
            description = describe_batch(batch)
            if common_input is not None and description != common_input:
                raise ValueError("initializer input/target coordinates differ")
            common_input = description
            # Follow baseline.real_models' validated loading and attention path, with
            # only one selected class loaded. Never transfer state between implementations.
            torch.manual_seed(42)
            cls = Qwen2_5_VLForConditionalGeneration if external is None else external.model.Qwen2_5_VLForConditionalGeneration
            kwargs = dict(torch_dtype=torch.bfloat16, local_files_only=True)
            if external is not None:
                kwargs["stage2_split_attention"] = True
            base = load_qwen2_5_vl_with_attention(args.base, attention_implementation="sdpa",
                vision_attention_implementation="flash_attention_2", config_load_kwargs={"local_files_only": True},
                model_load_kwargs=kwargs, load_model=cls.from_pretrained)
            model = load_stage1_dora_for_fresh_stage2(base, processor, checkpoint,
                Stage2TuningConfig("dora", 32, 64, .05, baseline.TARGETS), requested_base_model_reference=args.base)
            configure_stage2_metadata(base.config, LatentTokenIds(*baseline.token_ids(processor.tokenizer)[:3]),
                latent_size=8, supervise_latent_boundaries=True, latent_activation_checkpointing=True)
            base.config.use_cache = False
            model.gradient_checkpointing_disable()
            model = model.to(device=args.device, dtype=torch.bfloat16).train()
            trainer = make_trainers([model] if external is None else [None, model], external,
                                    Path(args.log_dir) / name, args.device)[-1]
            roots = [str(Path("src").resolve()), str(Path("transformers").resolve())] + ([] if root is None else [str(root)])
            cells[name] = {"checkpoint": str(Path(checkpoint).resolve()), "input": description,
                           **run_component_cell(model, trainer, batch, args.device, roots, decoder_dtype=args.decoder_dtype)}
            del model, base, trainer, batch, batches, processor
            gc.collect()
            torch.cuda.empty_cache()
        if root is not None and (before != baseline.inventory(root) or hashes != {str(p): baseline.sha(p) for p in root.rglob("*.py")}):
            raise ValueError("external source mutated during component diagnostic")
    return {"mode": "components", "side": args.side, "train_index": 2046, "cells": cells,
            "screen": screen_initializers(cells), "beta": .5, "preference_weight": 1., "entropy_weight": 0.,
            "common_backward_divisor": 4, "optimizer_steps": 0, "repetitions_per_component": 3,
            "base": args.base, "external_source": None if root is None else str(root),
            "external_source_hashes": hashes, "external_files_unchanged": None if root is None else True,
            "diagnostic_sha256": baseline.sha(__file__), "loader_sha256": baseline.sha(baseline.__file__),
            "data_sha256": baseline.sha(args.data), "manifest_sha256": baseline.sha(args.manifest),
            "numerics": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device,
                         "hardware": torch.cuda.get_device_name(args.device), "dtype": args.decoder_dtype, "autocast": False,
                         "float32_matmul_precision": torch.get_float32_matmul_precision(),
                         "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
            "limitations": ["No optimizer or DDP/ZeRO execution; native loss/4 backward only",
                            "Three repeats and one initial sample cannot establish historical late-collapse causality",
                            "Original external weights and historical intermediate checkpoints unavailable",
                            "External expansion is a controller decision; this script never launches another side"]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "components"), default="native")
    parser.add_argument("--decoder-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--side", choices=("local", "external"), default="local")
    parser.add_argument("--initializer", choices=("e1", "e3", "both"), default="both")
    parser.add_argument("--stage1-e3", default="checkpoints/stage1-dora-egb16-e3-s648-lr5e5-20260920-03/checkpoint-648")
    for name, default in (("external-source", None), ("stage1", baseline.STAGE1),
                          ("base", baseline.BASE), ("manifest", baseline.MANIFEST),
                          ("data", "data/TRAIN.jsonl"), ("image-root", "data"), ("device", "cuda:0"),
                          ("log-dir", "logs/until-native-v2-20260922")):
        parser.add_argument("--" + name, default=default)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), allow_nan=False))
