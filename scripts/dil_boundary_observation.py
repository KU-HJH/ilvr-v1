"""Bounded native diagnostic observations, imported only for explicit opt-in.

CPU snapshots and profiling synchronize execution and can alter its scheduling.
Tensor hooks observe summed cotangents at tensor boundaries, not arbitrary
per-consumer edges. No returned object, gradient, RNG or native call is replaced.
"""
from contextlib import ExitStack
import hashlib
import inspect
from collections import Counter

DISCLOSURE = ("CPU snapshots/tensor hooks and CPU/CUDA profiler may change scheduling; "
              "matching scalars or controls cannot prove noninterference. Replay callbacks "
              "are observations, not causal graph execution order.")


def witness(tensor):
    import torch
    value = tensor.detach().cpu().clone(memory_format=torch.contiguous_format)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


class BoundaryObservation:
    """A single fixed case/repeat; keeps detached CPU tensors and hook handles only."""
    def __init__(self, case, repeat):
        self.case, self.repeat, self.phase = case, repeat, "forward"
        self.raw, self.records, self.calls = {}, [], []
        self.counts, self.stack = Counter(), ExitStack()
        self.bytes = 0
        self.chunk_rows = {}
        self.attention_events = []

    def __enter__(self):
        import torch
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(activities=activities, record_shapes=False,
                                               profile_memory=False, with_stack=False)
        try:
            self.stack.enter_context(self.profiler)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        try:
            return self.stack.__exit__(*exc)
        finally:
            # Report actual dispatcher/backward and CUDA kernel events. Parent
            # attribution is explicit when the profiler exposes it; never infer
            # a backward call from forward entry order.
            for event in self.profiler.events():
                name = event.name
                if any(term in name.lower() for term in
                       ("scaled_dot_product", "flash_attention", "efficient_attention", "attention_math")):
                    parent, marker = event.cpu_parent, None
                    while parent is not None:
                        if parent.name.startswith("dil_boundary/"):
                            marker = parent.name
                            break
                        parent = parent.cpu_parent
                    self.attention_events.append({"name": name, "device": str(event.device_type),
                                                  "native_marker": marker, "count": 1})
            counts = Counter((item["name"], item["device"], item["native_marker"]) for item in self.attention_events)
            self.attention_events = [{"name": name, "device": device, "native_marker": marker, "count": count}
                                     for (name, device, marker), count in counts.items()]

    def _snapshot(self, key, tensor):
        if tensor is None:
            self.raw[key] = None
            return
        if key in self.raw:
            raise ValueError("duplicate boundary snapshot")
        size = tensor.numel() * tensor.element_size()
        if self.bytes + size > 512 * 1024**2:
            raise ValueError("bounded boundary snapshot budget exceeded")
        import torch
        self.raw[key] = tensor.detach().cpu().clone(memory_format=torch.contiguous_format)
        self.bytes += size

    def capture(self, name, tensor, call, *, rows=None, gradient_kind="tensor_total", tail=False, flatten=False):
        import torch
        if not isinstance(tensor, torch.Tensor) or len(self.records) >= 256:
            raise ValueError("missing or excessive native boundary tensors")
        key = f"{call['identity']}/{call['phase']}/{name}"
        if any(item["key"] == key for item in self.records):
            raise ValueError("duplicate native boundary identity")
        def project(value):
            if tail:
                value = value[:, -1, :]
            if flatten:
                value = value.reshape(-1, value.shape[-1])
            return value
        self._snapshot(key + "/forward", project(tensor))
        self.raw[key + "/cotangent"] = None
        record = {"key": key, "name": name, "call": call["identity"], "phase": call["phase"], "rows": rows,
                  "requires_grad": bool(tensor.requires_grad), "cotangent_status": "not_delivered",
                  "gradient_kind": gradient_kind, "gradient_callbacks": 0}
        self.records.append(record)
        if tensor.requires_grad:
            def gradient(grad):
                if record["gradient_callbacks"]:
                    raise ValueError("multiple cotangents at one native tensor boundary")
                self.raw.pop(key + "/cotangent")
                self._snapshot(key + "/cotangent", project(grad))
                record["cotangent_status"] = "delivered"
                record["gradient_callbacks"] += 1
                return grad
            handle = tensor.register_hook(gradient)
            self.stack.callback(handle.remove)

    def watch_module(self, module, role, observe, before=None):
        """Wrap instance forward without hooks that create replacement views."""
        import torch
        original = module.forward
        owned = "forward" in module.__dict__
        old_owned = module.__dict__.get("forward")
        def call(*args, **kwargs):
            context = role(args, kwargs) if callable(role) else {"role": role}
            count_key = (self.phase, context["role"])
            ordinal = self.counts[count_key]
            self.counts[count_key] += 1
            item = {**context, "identity": f"{context['role']}:{ordinal}", "phase": self.phase, "status": "entered"}
            self.calls.append(item)
            with torch.profiler.record_function(f"dil_boundary/{item['phase']}/{item['identity']}"):
                try:
                    if before:
                        before(args, kwargs, item)
                    output = original(*args, **kwargs)
                    observe(args, kwargs, output, item)
                    item["status"] = "returned"
                    return output
                except BaseException as error:
                    item["status"] = ("checkpoint_early_stop" if type(error).__name__ == "_StopRecomputationError"
                                      else "raised")
                    raise
        module.forward = call
        def restore():
            if owned:
                module.forward = old_owned
            else:
                delattr(module, "forward")
        self.stack.callback(restore)

    def bind_native(self, model, batch, variant):
        """Trace each runtime independently; never synthesize common KV/latents."""
        import importlib
        import torch
        candidates = [module for module in model.modules()
                      if all(hasattr(module, name) for name in ("stage2_rollout_engine", "lm_head", "model"))
                      and "stage2_rollout_engine" in module.__dict__]
        if len(candidates) != 1:
            raise ValueError("ambiguous native Stage2 model boundary")
        native = candidates[0]
        sample = batch["sample_ids"][0]
        self.length = batch["input_ids"].shape[1]
        spans = batch["reasoning_step_spans"]
        if spans and isinstance(spans[0], (list, tuple)):
            spans = spans[0]
        logical = {p-1: [sample, int(span.step_id), p-1, int(batch["input_ids"][0, p])]
                   for span in spans for p in range(span.target_text_start, span.target_text_end)}
        def rows(positions):
            positions = [int(p) for p in positions]
            return [logical.get(p, [sample, -1, p, int(batch["input_ids"][0, p+1])]) for p in positions]
        self.logical = logical
        def frames():
            frame = inspect.currentframe().f_back
            # Keep no frames after callback. Only exact native entrypoint names
            # and live native locals are used, not synthetic dispatch schedules.
            for _ in range(24):
                if frame is None:
                    break
                yield frame.f_code.co_name, frame.f_locals
                frame = frame.f_back
        def source_layout():
            for name, local in frames():
                if name == "execute_stage2_sidecar_groups" and "layout" in local:
                    layout = local["layout"]
                    if layout.modality != "visual" or any(branch.role != "chosen_entropy" for branch in layout.branches):
                        raise ValueError("unapproved source sidecar boundary")
                    return [logical[branch.replay_boundary + int(offset)] for branch in layout.branches
                            for offset in branch.score_offsets.detach().cpu().tolist()]
            return None
        def decoder_role(args, kwargs):
            embeds = kwargs.get("inputs_embeds")
            if embeds is None:
                raise ValueError("native decoder boundary lacks inputs_embeds")
            suffix = kwargs.get("past_key_values") is not None or kwargs.get("stage2_prefix_key_values") is not None
            if suffix:
                mapped = (self.chunk_rows.get(id(kwargs.get("position_ids"))) if variant == "portTorch" else source_layout())
                return {"role": "suffix", "rows": mapped}
            length = embeds.shape[1]
            return {"role": "clean" if length == self.length else f"prefix:{length}", "rows": None}
        def decoder_before(args, kwargs, call):
            if call["role"] != "suffix":
                return
            if variant == "portTorch":
                pair = kwargs["stage2_prefix_key_values"][-1]
            else:
                cache = kwargs["past_key_values"]
                pair = (cache.key_cache[-1], cache.value_cache[-1])
            for kind, tensor in zip(("key", "value"), pair):
                self.capture(f"suffix.input_{kind}", tensor, call,
                             gradient_kind="distinct_view_contribution" if variant == "portTorch" else "shared_tensor_total")
        def decoder_after(args, kwargs, output, call):
            if call["role"].startswith("prefix:"):
                self.capture("prefix.hidden_tail", output.last_hidden_state, call, tail=True,
                             rows=[[sample, "latent_position", int(call["role"].split(":")[1])]])
            else:
                self.capture("decoder.hidden", output.last_hidden_state, call)
            if call["role"] == "clean":
                if variant == "portTorch":
                    pair = output.stage2_key_values[-1]
                else:
                    cache = output.past_key_values
                    pair = (cache.key_cache[-1], cache.value_cache[-1])
                for kind, tensor in zip(("key", "value"), pair):
                    self.capture(f"clean.{kind}", tensor, call, gradient_kind="shared_tensor_total")
        self.watch_module(native.model, decoder_role, decoder_after, before=decoder_before)
        def head_role(args, kwargs):
            for name, local in frames():
                if name == "_stage2_supervised_cross_entropy":
                    mask = local["supervised"]
                    return {"role": "clean_ce", "rows": rows(mask[0].nonzero().flatten().detach().cpu().tolist())}
                if name == "_stage2_selected_logits":
                    mask = local["score_token_mask"]
                    return {"role": "clean_target", "rows": rows(mask[0].nonzero().flatten().detach().cpu().tolist())}
                if name == "execute_stage2_sidecar_groups":
                    return {"role": "off_target", "rows": source_layout()}
                if name == "score" and "positions" in local and "targets" in local:
                    return {"role": "off_target", "rows": self.chunk_rows.get(id(local["positions"]))}
                if name == "_forward_stage2":
                    if args[0] is local.get("selected_hidden"):
                        selected = local.get("selected_positions")
                        return {"role": "clean_target", "rows": rows(selected.detach().cpu().tolist())}
                    if "active" in local:
                        return {"role": "clean_ce", "rows": rows(local["active"].nonzero().flatten().detach().cpu().tolist())}
            return {"role": "unmapped_head", "rows": None}
        def head_before(args, kwargs, call):
            self.capture("head.hidden", args[0], call, rows=call["rows"], flatten=True)
        def head_after(args, kwargs, output, call):
            self.capture("head.logits", output, call, rows=call["rows"], flatten=True)
        self.watch_module(native.lm_head, head_role, head_after, before=head_before)
        # Rollout return carries the actual checkpoint-returned latent views.
        engine = native.stage2_rollout_engine
        original = engine.rollout
        owned, old_owned = "rollout" in engine.__dict__, engine.__dict__.get("rollout")
        def rollout(*args, **kwargs):
            result = original(*args, **kwargs)
            call = {"identity": "rollout:0", "phase": self.phase}
            for index, tensor in enumerate(result.latent_states):
                self.capture(f"latent.{index}", tensor, call,
                             rows=[[sample, "latent_position", int(result.prefix_lengths[index])]],
                             gradient_kind="shared_tensor_total")
            return result
        engine.rollout = rollout
        def restore_rollout():
            if owned:
                engine.rollout = old_owned
            else:
                delattr(engine, "rollout")
        self.stack.callback(restore_rollout)
        if variant == "portTorch":
            owner = importlib.import_module("transformers.models.qwen2_5_vl.stage2_shared_prefix")
            original_factory = owner._score_chunk
            def factory(*args, **kwargs):
                output = original_factory(*args, **kwargs)
                branches = args[1]
                mapped = [logical[branch.logical_positions[position]] for branch in branches
                          for position in branch.selected_positions] if all(branch.kind == "visual_off" for branch in branches) else None
                self.chunk_rows[id(args[3])] = mapped
                return output
            owner._score_chunk = factory
            self.stack.callback(setattr, owner, "_score_chunk", original_factory)

    def record(self):
        return {"case": self.case, "repeat": self.repeat, "calls": self.calls,
                "records": [{**item, "forward": witness(self.raw[item["key"] + "/forward"]),
                             "cotangent": None if self.raw[item["key"] + "/cotangent"] is None
                             else witness(self.raw[item["key"] + "/cotangent"])} for item in self.records],
                "attention_events": self.attention_events, "observer_effect_disclosure": DISCLOSURE,
                "runtime_context": self.runtime_context(),
                "snapshot_bytes": self.bytes,
                "causal_limit": "First observed dependency frontier, not first bad operation; unsampled operations and merges remain unresolved."}

    @staticmethod
    def runtime_context():
        import torch
        import os
        return {"torch": torch.__version__, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def validate_raw(record, raw):
    items = record.get("records", [])
    keys = {item["key"] + suffix for item in items for suffix in ("/forward", "/cotangent")}
    if len(keys) != 2 * len(items) or set(raw) != keys:
        raise ValueError("missing or duplicate boundary tensor evidence")
    for item in items:
        for suffix, field in (("/forward", "forward"), ("/cotangent", "cotangent")):
            value = raw[item["key"] + suffix]
            if (None if value is None else witness(value)) != item.get(field):
                raise ValueError("boundary raw tensor witness differs")


def _stage(item, derivative):
    name = item["name"]
    if name == "head.logits":
        stage = 4
    elif name == "head.hidden":
        stage = 3
    elif name == "decoder.hidden":
        stage = 2
    elif name.startswith(("clean.", "suffix.input_")):
        stage = 1
    else:
        stage = 0
    position = 0
    if stage == 0 and item.get("rows") and item["rows"][0][1] == "latent_position":
        position = item["rows"][0][2]
    return (4-stage, -position) if derivative else (stage, position)


def compare_boundaries(left, lraw, right, rraw, comparator):
    validate_raw(left, lraw)
    validate_raw(right, rraw)
    li, ri = ({item["key"]: item for item in record.get("records", [])} for record in (left, right))
    unmapped = sorted(set(li) ^ set(ri))
    absent_cotangents = sorted({key for records in (li, ri) for key, item in records.items()
                               if item.get("cotangent") is None})
    unmapped.extend(key + "/cotangent-unavailable" for key in absent_cotangents)
    results, mismatches = {}, {"forward": [], "cotangent": []}
    for key in sorted(set(li) & set(ri)):
        a, b = li[key], ri[key]
        if a.get("rows") != b.get("rows") or "unmapped" in a.get("call", "") or "unmapped" in b.get("call", ""):
            unmapped.append(key)
            continue
        values = {}
        comparable = a.get("gradient_kind") == b.get("gradient_kind")
        for field in ("forward", "cotangent"):
            aw, bw = a.get(field), b.get(field)
            if field == "cotangent" and (aw is None or bw is None):
                # A native unused tensor or replay can legitimately deliver no
                # derivative. Preserve that observation without equating two
                # absences or inventing zeros/a numerical mismatch frontier.
                values[field] = None
                values["cotangent_availability"] = [a.get("cotangent_status"), b.get("cotangent_status")]
                continue
            if field == "cotangent" and not comparable:
                values[field] = None
                continue
            if aw is not None and bw is not None and any(aw[identity] != bw[identity] for identity in ("shape", "dtype")):
                values[field] = None
                unmapped.append(key + "/" + field + "-shape-dtype")
                continue
            comparison = comparator({key: lraw[key+"/"+field]}, {key: rraw[key+"/"+field]})
            values[field] = comparison
            if not comparison["pass"]:
                mismatches[field].append(a)
        values["cotangent_semantics"] = [a.get("gradient_kind"), b.get("gradient_kind")]
        # A shared source tensor and a distinct port view are different fan-in
        # claims. Numeric agreement does not make those contributions equal.
        if not comparable:
            values["cotangent_comparable"] = False
            unmapped.append(key + "/cotangent-semantics")
        else:
            values["cotangent_comparable"] = values.get("cotangent") is not None
        results[key] = values
    def frontier(field):
        items = mismatches[field]
        if not items:
            return []
        first = min(_stage(item, field == "cotangent") for item in items)
        return sorted(item["key"] for item in items if _stage(item, field == "cotangent") == first)
    forward_pass = not mismatches["forward"]
    cotangent_pass = (not mismatches["cotangent"] and not absent_cotangents
                      and bool(results) and len(results) == len(li) == len(ri)
                      and all(value["cotangent_comparable"] for value in results.values()))
    return {"pass": bool(li) and bool(ri) and not unmapped and forward_pass and cotangent_pass,
            "forward_pass": forward_pass, "cotangent_pass": cotangent_pass,
            "unmapped_or_incomparable": unmapped, "boundaries": results,
            "unavailable_cotangents": absent_cotangents,
            "earliest_forward_frontier": frontier("forward"), "earliest_cotangent_frontier": frontier("cotangent"),
            "causal_limit": "Grouped by traced dependency stage, never callback order; parallel paths can share the first frontier."}


def validate_trace(record, prescribed, repeat):
    """Missing observations remain a failed diagnostic, never equal boundaries."""
    if (record.get("case") != prescribed["raw_index"] or record.get("repeat") != repeat
            or not record.get("observer_effect_disclosure") or not record.get("attention_events")):
        raise ValueError("missing boundary case/repeat/profiler evidence")
    items = record.get("records", [])
    calls = record.get("calls", [])
    if not calls or any(item.get("phase") not in ("forward", "replay")
                        or item.get("status") not in ("returned", "checkpoint_early_stop") for item in calls):
        raise ValueError("missing native boundary call/replay completion evidence")
    if not any("backward" in event.get("name", "").lower() for event in record["attention_events"]):
        raise ValueError("missing actual attention backward profiler evidence")
    if len({item["key"] for item in items}) != len(items):
        raise ValueError("duplicate boundary identities")
    call_map = {(item.get("phase"), item.get("identity")): item for item in calls}
    if len(call_map) != len(calls):
        raise ValueError("duplicate native boundary call identities")
    for item in items:
        if item.get("key") != f"{item.get('call')}/{item.get('phase')}/{item.get('name')}":
            raise ValueError("inconsistent native boundary record identity")
        # Latents are observed at the native rollout return, outside a decoder
        # forward wrapper. Their separate representation is linked below to
        # the actual corresponding prefix position/call, not an invented call.
        rollout_return = item.get("call") == "rollout:0" and item.get("name", "").startswith("latent.")
        if not rollout_return and (item.get("phase"), item.get("call")) not in call_map:
            raise ValueError("missing native boundary call linkage")
    forward = [item for item in items if item.get("phase") == "forward"]
    by_boundary = {(item["call"], item["name"]): item for item in forward}
    head_calls = ("clean_ce:0", "clean_target:0", "off_target:0")
    required = {(call, name) for call in head_calls for name in ("head.hidden", "head.logits")}
    required.update(("clean:0", name) for name in ("decoder.hidden", "clean.key", "clean.value"))
    required.update(("rollout:0", f"latent.{index}") for index in range(prescribed["blocks"] * 8))
    if not required <= by_boundary.keys():
        raise ValueError("missing required native forward boundary identity")
    heads = [by_boundary[call, name] for call in head_calls for name in ("head.hidden", "head.logits")]
    if any(item.get("rows") is None or len(item["rows"]) != item["forward"]["shape"][0] for item in heads):
        raise ValueError("unmapped native head boundary rows")
    prefix_positions = []
    for index in range(prescribed["blocks"] * 8):
        latent = by_boundary["rollout:0", f"latent.{index}"]
        rows = latent.get("rows")
        if (not isinstance(rows, list) or len(rows) != 1 or len(rows[0]) != 3
                or rows[0][:2] != [prescribed["sample_id"], "latent_position"]
                or type(rows[0][2]) is not int or rows[0][2] <= 0):
            raise ValueError("missing native rollout boundary position")
        position = rows[0][2]
        prefix = by_boundary.get((f"prefix:{position}:0", "prefix.hidden_tail"))
        if prefix is None or prefix.get("rows") != rows:
            raise ValueError("missing native rollout/prefix boundary linkage")
        prefix_positions.append(position)
    if prefix_positions != sorted(set(prefix_positions)):
        raise ValueError("invalid native rollout boundary order")
    if len([item for item in forward if item.get("name") == "prefix.hidden_tail"]) != len(prefix_positions):
        raise ValueError("missing native latent prefix boundary")
    if any(call["phase"] == "forward" and call["status"] != "returned" for call in calls):
        raise ValueError("missing returned forward native boundary call")
    for item in items:
        if item.get("cotangent_status") not in ("delivered", "not_delivered"):
            raise ValueError("missing boundary cotangent status")
        if (item["cotangent_status"] == "delivered") != (item.get("cotangent") is not None):
            raise ValueError("inconsistent boundary cotangent witness")
        for field in ("forward", "cotangent"):
            value = item.get(field)
            if value is None and field == "cotangent":
                continue
            if (not isinstance(value, dict) or not isinstance(value.get("shape"), list)
                    or not isinstance(value.get("dtype"), str) or len(value.get("sha256", "")) != 64):
                raise ValueError("missing boundary tensor witness")
