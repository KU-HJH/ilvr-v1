"""First-update evidence for isolated pair probes; never valid speed timings."""
import dataclasses
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from step_preference.numerical_tolerance import compare_loss_tolerance, compare_numerical_tolerance

PHASE = "first-optimizer-update-pre-unscale-clip-step"
TIMING_SCOPE = "instrumented-not-valid-for-speed-comparison"


def hash_structured_value(value):
    """Type-framed, deterministic bytes, including tensor values in bounded chunks."""
    digest = hashlib.sha256()
    def emit(data):
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided:
                raise TypeError("unsupported paired-probe value: sparse tensor")
            emit(b"tensor")
            visit(str(item.dtype)); visit(list(item.shape))
            flat = item.detach().reshape(-1)
            for chunk in flat.split(262144):
                chunk = chunk.cpu().contiguous()
                if not bool(torch.isfinite(chunk).all()):
                    raise ValueError("paired-probe tensor must be finite")
                emit(chunk.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject or not np.isfinite(item).all():
                raise ValueError("paired-probe NumPy data must be finite and numeric")
            emit(b"numpy"); visit(str(item.dtype)); visit(list(item.shape)); emit(item.tobytes())
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            emit(b"dataclass"); visit(type(item).__module__ + "." + type(item).__qualname__)
            visit({field.name: getattr(item, field.name) for field in dataclasses.fields(item)})
        elif isinstance(item, Mapping) and all(isinstance(key, str) for key in item):
            emit(b"dict")
            for key in sorted(item):
                visit(key); visit(item[key])
            emit(b"end")
        elif isinstance(item, (list, tuple)):
            emit(type(item).__name__.encode())
            for child in item:
                visit(child)
            emit(b"end")
        elif item is None or type(item) in (str, bool, int, float):
            emit(type(item).__name__.encode()); emit(json.dumps(item, allow_nan=False).encode())
        else:
            raise TypeError(f"unsupported paired-probe value: {type(item).__name__}")
    visit(value)
    return digest.hexdigest()


def _rng_states(device=None):
    return dict(python=hash_structured_value(random.getstate()),
                numpy=hash_structured_value(np.random.get_state()),
                cpu=hash_structured_value(torch.get_rng_state()),
                cuda=hash_structured_value(torch.cuda.get_rng_state(device)))


def _verify_collective(value):
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, value)
    if any(item != gathered[0] for item in gathered) or value.get("errors"):
        raise RuntimeError(f"paired-probe rank coverage/validation mismatch: {gathered}")


def create_production_observer(owner, args):
    """Bind strict production checks; CPU tests replace only absent runtime boundaries."""
    if not torch.cuda.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("paired-probe requires initialized four-rank CUDA execution")
    adapter = getattr(owner, "_stage2_deepspeed_gradient_adapter", None)
    engine = getattr(adapter, "engine", None)
    if (engine is None or engine.step is not adapter or getattr(owner, "model_wrapped", None) is not engine
            or engine.zero_optimization_stage() != 2 or not engine.bfloat16_enabled()
            or engine.gradient_accumulation_steps() != 2):
        raise RuntimeError("paired-probe requires installed adapter, BF16 ZeRO2 and accumulation 2")
    from deepspeed.utils.tensor_fragment import get_full_hp_grad
    errors = []
    for name, parameter in sorted(owner.model.named_parameters()):
        if parameter.requires_grad and (
            parameter.device.type != "cuda"
            or getattr(getattr(parameter, "get_full_hp_grad", None), "__func__", None) is not get_full_hp_grad
            or not hasattr(parameter, "_hp_mapping") or not hasattr(parameter, "_dp_group")
            or torch.distributed.get_world_size(parameter._dp_group) != 4
        ):
            errors.append(f"unsupported ZeRO2 mapping: {name}")
    _verify_collective({"errors": errors})
    from multinode_artifacts import resolve_local_checkpoint_destination
    checkpoint_dir = Path(args.output_dir)
    resolve_local_checkpoint_destination(checkpoint_dir / "pair-probe")
    root = Path(__file__).resolve().parents[2]
    return PairedProbeObserver(owner, checkpoint_dir=checkpoint_dir,
        result_dir=root / "results" / checkpoint_dir.name,
        rank=torch.distributed.get_rank(), world_size=torch.distributed.get_world_size(),
        rng_reader=_rng_states, coverage_verifier=_verify_collective)


class PairedProbeObserver:
    def __init__(self, owner, *, checkpoint_dir, result_dir, rank, world_size,
                 rng_reader, coverage_verifier):
        self.owner, self.rank, self.world_size = owner, rank, world_size
        self.checkpoint_dir, self.result_dir = Path(checkpoint_dir), Path(result_dir)
        self.rng_reader, self.verify = rng_reader, coverage_verifier
        self.engine = owner._stage2_deepspeed_gradient_adapter.engine
        self.microbatches, self.gradient_tensors, self.gradient_coverage = [], {}, []
        self.installed = self.captured = self.finished = False

    def install(self):
        optimizer = self.engine.optimizer
        if (self.world_size != 4 or self.rank not in range(4) or not optimizer.partition_gradients
                or optimizer.cpu_offload or float(optimizer.loss_scale) != 1.):
            raise RuntimeError("paired-probe requires four ranks, GPU ZeRO2 and loss scale 1")
        self.parameters = sorted((name, p) for name, p in self.owner.model.named_parameters() if p.requires_grad)
        self.gradient_coverage = [dict(name=name, shape=list(p.shape)) for name, p in self.parameters]
        errors = [name for name, p in self.parameters if not callable(getattr(p, "get_full_hp_grad", None))]
        self.verify(dict(coverage=self.gradient_coverage, errors=errors))
        if errors or not self.parameters:
            raise RuntimeError("paired-probe requires full named gradient methods")
        state = []
        for kind, tensors in (("parameter", self.owner.model.named_parameters()),
                              ("buffer", self.owner.model.named_buffers())):
            for name, tensor in sorted(tensors):
                state.append(dict(kind=kind, name=name, shape=list(tensor.shape),
                                  dtype=str(tensor.dtype), sha256=hash_structured_value(tensor)))
        self.initial_state = dict(sha256=hash_structured_value(state), tensors=state)
        self.original_training_step, self.original_engine_step = self.owner.training_step, self.engine.step
        self.installed = True
        self.owner.training_step, self.engine.step = self._training_step, self._engine_step

    def _training_step(self, *args, **kwargs):
        try:
            if len(self.microbatches) >= 2:
                raise RuntimeError("paired-probe requires exactly two microbatches")
            inputs = args[1] if len(args) > 1 else kwargs["inputs"]
            self.microbatches.append(dict(index=len(self.microbatches),
                input_sha256=hash_structured_value(inputs), rng=self.rng_reader()))
            return self.original_training_step(*args, **kwargs)
        except BaseException:
            self.close()
            raise

    def _engine_step(self, *args, **kwargs):
        try:
            if self.engine.is_gradient_accumulation_boundary():
                if len(self.microbatches) != 2:
                    raise RuntimeError("paired-probe requires exactly two microbatches")
                if float(self.engine.optimizer.loss_scale) != 1.:
                    raise RuntimeError("paired-probe requires loss scale 1")
                errors = []
                for name, parameter in self.parameters:
                    # All ranks call the direct bound method, even with local grad=None.
                    value = parameter.get_full_hp_grad()
                    if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                            or not bool(torch.isfinite(value).all())):
                        errors.append(f"incomplete/nonfinite gradient: {name}")
                    elif self.rank == 0:
                        self.gradient_tensors[name] = value.detach().to(device="cpu", dtype=torch.float32).clone()
                self.verify(dict(coverage=self.gradient_coverage, errors=errors))
                if errors:
                    raise RuntimeError(str(errors))
                self.captured = True
                self.close()
            return self.original_engine_step(*args, **kwargs)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.installed:
            self.owner.training_step, self.engine.step = self.original_training_step, self.original_engine_step
            self.installed = False

    def finish_first_step(self, metrics):
        if self.finished:
            raise RuntimeError("paired-probe first step already written")
        if len(self.microbatches) != 2 or not self.captured:
            raise RuntimeError("paired-probe needs exactly two microbatches and pre-step gradients")
        checked = compare_loss_tolerance(metrics, metrics)
        if not checked["passed"]:
            raise RuntimeError("paired-probe needs finite objective terms")
        tensor_path = self.checkpoint_dir / "pair-probe" / "first-pre-step-gradients.pt"
        metadata_path = self.result_dir / "pair-probe" / f"rank-{self.rank}.json"
        payload = dict(schema_version=1, kind="stage2-paired-first-update", rank=self.rank,
            world_size=self.world_size, collection_phase=PHASE, initial_state=self.initial_state,
            microbatches=self.microbatches,
            losses={key: value["reference"] for key, value in checked["losses"].items()},
            gradient_coverage=self.gradient_coverage, gradient_artifact=str(tensor_path), timing_scope=TIMING_SCOPE)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if metadata_path.exists():
            raise FileExistsError(metadata_path)
        if self.rank == 0:
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            with tensor_path.open("xb") as stream:
                torch.save(self.gradient_tensors, stream)
        with metadata_path.open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        self.finished = True


def _read_metadata(paths, side, errors):
    by_rank = {}
    for path in paths:
        try:
            item = json.loads(Path(path).read_text())
            rank = item["rank"]
            if type(rank) is not int or rank not in range(4) or rank in by_rank:
                raise ValueError("duplicate/invalid rank")
            if (item["schema_version"] != 1 or item["kind"] != "stage2-paired-first-update"
                    or item["world_size"] != 4 or item["collection_phase"] != PHASE
                    or item["timing_scope"] != TIMING_SCOPE):
                raise ValueError("wrong capture protocol")
            if (not item["initial_state"]["sha256"] or not item["initial_state"]["tensors"]
                    or not item["gradient_coverage"] or not item["gradient_artifact"]
                    or not isinstance(item["losses"], dict)):
                raise ValueError("missing evidence")
            if len(item["microbatches"]) != 2:
                raise ValueError("exactly two microbatches required")
            for index, batch in enumerate(item["microbatches"]):
                if (batch["index"] != index or not batch["input_sha256"]
                        or any(not batch["rng"][key] for key in ("python", "numpy", "cpu", "cuda"))):
                    raise ValueError("incomplete input/RNG witness")
            by_rank[rank] = item
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{side} metadata {path}: {error}")
    if set(by_rank) != set(range(4)):
        errors.append(f"{side} requires all four rank metadata files")
    return by_rank


def compare_paired_runs(reference_metadata, candidate_metadata, reference_gradients, candidate_gradients):
    """Read trusted own tensor artifacts explicitly; compare one global vector once."""
    errors, rank_losses = [], {}
    reference = _read_metadata(reference_metadata, "reference", errors)
    candidate = _read_metadata(candidate_metadata, "candidate", errors)
    for rank in sorted(reference.keys() & candidate.keys()):
        left, right = reference[rank], candidate[rank]
        for field in ("initial_state", "gradient_coverage"):
            if left[field] != right[field]:
                errors.append(f"rank {rank} {field} mismatch")
        for index, (before, after) in enumerate(zip(left["microbatches"], right["microbatches"])):
            if before["input_sha256"] != after["input_sha256"]:
                errors.append(f"rank {rank} microbatch {index} input mismatch")
            if before["rng"] != after["rng"]:
                errors.append(f"rank {rank} microbatch {index} RNG mismatch")
        rank_losses[str(rank)] = compare_loss_tolerance(left["losses"], right["losses"])
        errors.extend(f"rank {rank}: {error}" for error in rank_losses[str(rank)]["errors"])
    gradients = {}
    if set(reference) == set(candidate) == set(range(4)):
        loaded = []
        try:
            for side, metadata, path in (("reference", reference, reference_gradients),
                                         ("candidate", candidate, candidate_gradients)):
                values = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(values, dict) or any(not isinstance(v, torch.Tensor) for v in values.values()):
                    raise ValueError(f"{side}: missing named gradient tensors")
                coverage = [dict(name=name, shape=list(value.shape)) for name, value in sorted(values.items())]
                if any(metadata[rank]["gradient_coverage"] != coverage for rank in range(4)):
                    errors.append(f"{side}: gradient artifact coverage mismatch")
                loaded.append(values)
            measured = compare_numerical_tolerance(reference[0]["losses"], candidate[0]["losses"], *loaded)
            gradients = measured["gradients"]
            errors.extend(measured["errors"])
        except (OSError, ValueError, TypeError, RuntimeError, EOFError) as error:
            errors.append(f"gradient artifacts: {error}")
    return dict(passed=not errors, errors=errors, gradients=gradients, rank_losses=rank_losses,
                scope="first-update only; no speed or final-accuracy equivalence")
