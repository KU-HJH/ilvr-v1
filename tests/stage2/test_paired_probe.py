"""CPU contracts for the isolated production paired-run observer."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stage2.paired_probe import (
    PairedProbeObserver,
    compare_paired_runs,
    hash_structured_value,
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int


def test_structured_hash_is_order_stable_and_detects_tensor_or_metadata_mutation():
    first = {
        "tensor": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        "metadata": (Span(2, 5), {"enabled": True, "score": 0.25}),
    }
    reordered = {"metadata": first["metadata"], "tensor": first["tensor"].clone()}
    assert hash_structured_value(first) == hash_structured_value(reordered)

    reordered["tensor"][0, 0] = 9
    assert hash_structured_value(first) != hash_structured_value(reordered)
    with pytest.raises(TypeError, match="unsupported paired-probe value"):
        hash_structured_value(object())


class _DropModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([3.0, 4.0]))
        self.adapter = nn.Module()
        self.adapter.lora_dropout = nn.ModuleDict({"default": nn.Dropout(0.25)})

    def forward(self, value):
        return self.adapter.lora_dropout["default"](value * self.weight)


class _GradientParameter(nn.Parameter):
    pass


def _owner_for_observer():
    model = _DropModel()
    gradient_before_step = torch.tensor([6.0, 8.0])
    calls = []

    def get_full_hp_grad():
        calls.append("collect")
        return gradient_before_step.clone()

    model.weight.get_full_hp_grad = get_full_hp_grad

    class Engine:
        def __init__(self):
            self.optimizer = SimpleNamespace(
                partition_gradients=True,
                cpu_offload=False,
                loss_scale=1.0,
                averaged_gradients={0: [gradient_before_step]},
            )
            self.step_calls = 0

        def is_gradient_accumulation_boundary(self):
            return True

        def step(self, marker=None):
            calls.append("step")
            self.step_calls += 1
            gradient_before_step.zero_()
            return marker

    engine = Engine()
    microbatch_count = 0

    def training_step(model_arg, inputs, num_items_in_batch=None):
        nonlocal microbatch_count
        microbatch_count += 1
        output = model_arg(inputs["values"])
        if microbatch_count == 2:
            assert engine.step("original-return") == "original-return"
        return output.sum()

    owner = SimpleNamespace(
        model=model,
        training_step=training_step,
        _stage2_deepspeed_gradient_adapter=SimpleNamespace(engine=engine),
    )
    return owner, engine, calls, gradient_before_step


def test_observer_captures_raw_before_step_once_and_restores_runtime_hooks(tmp_path):
    owner, engine, calls, gradient_source = _owner_for_observer()
    original_training_step = owner.training_step
    original_engine_step = engine.step
    checkpoint_dir = tmp_path / "checkpoints" / "ours1-zero-speed-pair-aa"
    result_dir = tmp_path / "results" / "ours1-zero-speed-pair-aa"
    observer = PairedProbeObserver(
        owner,
        checkpoint_dir=checkpoint_dir,
        result_dir=result_dir,
        rank=0,
        world_size=4,
        rng_reader=lambda device=None: {"cpu": "same", "cuda": f"cuda-{device}"},
        coverage_verifier=lambda value: None,
    )
    observer.install()
    first = {"values": torch.ones(2), "labels": torch.tensor([1]), "span": Span(1, 2)}
    second = {"values": torch.ones(2) * 2, "labels": torch.tensor([2]), "span": Span(2, 3)}
    owner.training_step(owner.model, first)
    owner.training_step(owner.model, second)

    assert calls == ["collect", "step"]
    assert engine.step_calls == 1
    saved = observer.gradient_tensors
    assert torch.equal(saved["weight"], torch.tensor([6.0, 8.0]))
    assert torch.equal(gradient_source, torch.zeros(2))
    assert owner.training_step is original_training_step
    assert engine.step == original_engine_step
    assert len(observer.microbatches) == 2
    assert [item["index"] for item in observer.microbatches] == [0, 1]
    assert not owner.model.adapter.lora_dropout["default"]._forward_pre_hooks

    observer.finish_first_step(
        {"loss/ce": 1.0, "loss/context_preference": 0.5}
    )
    tensor_path = checkpoint_dir / "pair-probe" / "first-pre-step-gradients.pt"
    metadata_path = result_dir / "pair-probe" / "rank-0.json"
    assert torch.equal(torch.load(tensor_path, weights_only=True)["weight"], torch.tensor([6.0, 8.0]))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["collection_phase"] == "first-optimizer-update-pre-unscale-clip-step"
    assert metadata["gradient_coverage"] == [{"name": "weight", "shape": [2]}]
    assert metadata["losses"] == {"loss/ce": 1.0, "loss/context_preference": 0.5}
    assert metadata["timing_scope"] == "instrumented-not-valid-for-speed-comparison"


def test_nonwriter_traverses_every_gradient_but_writes_no_tensor_artifact(tmp_path):
    owner, engine, calls, _ = _owner_for_observer()
    observer = PairedProbeObserver(
        owner,
        checkpoint_dir=tmp_path / "checkpoints" / "ours1-zero-speed-pair-aa",
        result_dir=tmp_path / "results" / "ours1-zero-speed-pair-aa",
        rank=2,
        world_size=4,
        rng_reader=lambda device=None: {"cpu": "same", "cuda": "same"},
        coverage_verifier=lambda value: None,
    )
    observer.install()
    owner.training_step(owner.model, {"values": torch.ones(2), "labels": torch.tensor([1])})
    owner.training_step(owner.model, {"values": torch.ones(2), "labels": torch.tensor([2])})
    observer.finish_first_step({"loss/ce": 1.0, "loss/context_preference": 0.5})
    assert calls == ["collect", "step"]
    assert observer.gradient_tensors == {}
    assert not (observer.checkpoint_dir / "pair-probe" / "first-pre-step-gradients.pt").exists()
    assert (observer.result_dir / "pair-probe" / "rank-2.json").is_file()


def _metadata(rank, *, state="start", input_hash="input", rng="rng"):
    return {
        "schema_version": 1,
        "kind": "stage2-paired-first-update",
        "rank": rank,
        "world_size": 4,
        "collection_phase": "first-optimizer-update-pre-unscale-clip-step",
        "initial_state": {"sha256": state, "tensors": [{"name": "weight", "shape": [2], "dtype": "torch.float32", "sha256": state}]},
        "microbatches": [
            {"index": 0, "input_sha256": input_hash + "0", "rng": {key: rng + "0" for key in ("python", "numpy", "cpu", "cuda")}},
            {"index": 1, "input_sha256": input_hash + "1", "rng": {key: rng + "1" for key in ("python", "numpy", "cpu", "cuda")}},
        ],
        "losses": {"loss/ce": 2.0 + rank, "loss/context_preference": 0.5},
        "gradient_coverage": [{"name": "weight", "shape": [2]}],
        "gradient_artifact": "different-run-path-is-not-identity",
        "timing_scope": "instrumented-not-valid-for-speed-comparison",
    }


def _write_run(root: Path, side: str, *, mutation=None):
    paths = []
    for rank in range(4):
        payload = _metadata(rank)
        if mutation is not None:
            mutation(rank, payload)
        path = root / side / f"rank-{rank}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        paths.append(path)
    gradients = root / side / "gradients.pt"
    torch.save({"weight": torch.tensor([3.0, 4.0])}, gradients)
    return paths, gradients


def test_comparison_uses_one_global_gradient_and_ignores_run_paths(tmp_path):
    reference, reference_gradients = _write_run(tmp_path, "reference")

    def diagnostic_only(rank, payload):
        payload["gradient_artifact"] = "candidate-location"

    candidate, candidate_gradients = _write_run(tmp_path, "candidate", mutation=diagnostic_only)
    result = compare_paired_runs(reference, candidate, reference_gradients, candidate_gradients)
    assert result["passed"] is True
    assert result["gradients"]["relative_l2"] == 0.0
    assert all(item["passed"] for item in result["rank_losses"].values())


def test_comparison_rejects_rng_identity_mismatch_and_rank_loss_outside_tolerance(tmp_path):
    reference, reference_gradients = _write_run(tmp_path, "reference")

    def mismatch(rank, payload):
        if rank == 2:
            payload["microbatches"][1]["rng"]["cuda"] = "wrong"
            payload["losses"]["loss/context_preference"] = 0.6

    candidate, candidate_gradients = _write_run(tmp_path, "candidate", mutation=mismatch)
    result = compare_paired_runs(reference, candidate, reference_gradients, candidate_gradients)
    assert result["passed"] is False
    assert any("rank 2 microbatch 1 RNG" in error for error in result["errors"])
    assert result["rank_losses"]["2"]["passed"] is False


def test_finish_rejects_incomplete_microbatch_capture(tmp_path):
    owner, _, _, _ = _owner_for_observer()
    observer = PairedProbeObserver(
        owner,
        checkpoint_dir=tmp_path / "checkpoints" / "ours1-zero-speed-pair-aa",
        result_dir=tmp_path / "results" / "ours1-zero-speed-pair-aa",
        rank=0,
        world_size=4,
        rng_reader=lambda device=None: {"cpu": "same", "cuda": "same"},
        coverage_verifier=lambda value: None,
    )
    observer.install()
    owner.training_step(owner.model, {"values": torch.ones(2), "labels": torch.tensor([1])})
    with pytest.raises(RuntimeError, match="exactly two microbatches"):
        observer.finish_first_step({"loss/ce": 1.0, "loss/context_preference": 0.5})
    observer.close()


def test_speed_probe_enables_pair_observer_only_for_pair_run(monkeypatch):
    import stage2.speed_probe as speed_probe
    from tests.step_preference.insertion_helpers import native_case

    model, *_ = native_case()
    owner = SimpleNamespace(
        model=model,
        _representative_gradient_norm=0.0,
        _step_preference_metric_totals={},
        _step_preference_metric_batches=0,
    )
    events = []

    class Observer:
        def install(self):
            events.append("install")

        def finish_first_step(self, metrics):
            events.append(("finish", metrics))

        def close(self):
            events.append("close")

    monkeypatch.setattr(speed_probe, "create_production_observer", lambda owner, args: Observer())
    monkeypatch.setattr(speed_probe.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(speed_probe.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(speed_probe.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(speed_probe.torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setenv("RANK", "0")

    ordinary = speed_probe.InsertionSpeedProbe(owner)
    ordinary.on_train_begin(SimpleNamespace(output_dir="/tmp/ours1-zero-speed-old"), None, None)
    ordinary.on_train_end(None, None, None)
    assert events == []

    pair = speed_probe.InsertionSpeedProbe(owner)
    pair.on_train_begin(SimpleNamespace(output_dir="/tmp/ours1-zero-speed-pair-aa"), None, None)
    pair.on_step_begin(None, None, None)
    owner._step_preference_metric_totals = {"loss/ce": 2.0, "loss/context_preference": 1.0}
    owner._step_preference_metric_batches = 2
    pair.on_step_end(None, SimpleNamespace(global_step=1), None)
    pair.on_train_end(None, None, None)
    assert events == [
        "install",
        ("finish", {"loss/ce": 1.0, "loss/context_preference": 0.5}),
        "close",
    ]


def _observer(owner, tmp_path, rank=0):
    return PairedProbeObserver(owner, checkpoint_dir=tmp_path / "checkpoints" / "run",
                               result_dir=tmp_path / "results" / "run", rank=rank, world_size=4,
                               rng_reader=lambda device=None: {"cpu": "same", "cuda": "same"},
                               coverage_verifier=lambda value: None)


def test_all_named_collectives_run_in_sorted_order_on_nonwriter(tmp_path):
    owner, engine, calls, _ = _owner_for_observer()
    owner.model.alpha = nn.Parameter(torch.ones(1))
    def alpha_gradient():
        calls.append("alpha")
        return torch.tensor([7.])
    owner.model.alpha.get_full_hp_grad = alpha_gradient
    observer = _observer(owner, tmp_path, rank=3)
    observer.install()
    for _ in range(2):
        owner.training_step(owner.model, {"values": torch.ones(2)})
    assert calls == ["alpha", "collect", "step"]
    assert observer.gradient_tensors == {}


def test_early_accumulation_boundary_rejects_before_updating(tmp_path):
    owner, engine, _, _ = _owner_for_observer()
    owner.training_step = lambda *args, **kwargs: engine.step()
    observer = _observer(owner, tmp_path)
    observer.install()
    with pytest.raises(RuntimeError, match="exactly two microbatches"):
        owner.training_step(owner.model, {"values": torch.ones(2)})
    assert engine.step_calls == 0
    assert not owner.model.adapter.lora_dropout["default"]._forward_pre_hooks


def test_scale_must_be_one_before_instrumentation(tmp_path):
    owner, engine, _, _ = _owner_for_observer()
    engine.optimizer.loss_scale = 2.
    with pytest.raises(RuntimeError, match="scale"):
        _observer(owner, tmp_path).install()


def test_observation_preserves_rng_delegation_and_exception_cleanup(tmp_path):
    owner, engine, _, _ = _owner_for_observer()
    marker = RuntimeError("original failure")
    def failed(model, inputs, num_items_in_batch=None):
        model(inputs["values"])
        raise marker
    owner.training_step = failed
    rng = torch.get_rng_state().clone()
    try:
        failed(owner.model, {"values": torch.ones(2)}, num_items_in_batch=9)
    except RuntimeError:
        pass
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    observer = _observer(owner, tmp_path)
    observer.install()
    with pytest.raises(RuntimeError) as caught:
        owner.training_step(owner.model, {"values": torch.ones(2)}, num_items_in_batch=9)
    assert observer.microbatches  # The installed wrapper really observed this call.
    assert caught.value is marker
    assert owner.training_step is failed
    assert not owner.model.adapter.lora_dropout["default"]._forward_pre_hooks
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_capture_and_finish_never_overwrite_existing_artifacts(tmp_path):
    owner, _, _, _ = _owner_for_observer()
    observer = _observer(owner, tmp_path)
    observer.install()
    for _ in range(2):
        owner.training_step(owner.model, {"values": torch.ones(2)})
    observer.finish_first_step({"loss/ce": 1., "loss/context_preference": .5})
    with pytest.raises((FileExistsError, RuntimeError)):
        observer.finish_first_step({"loss/ce": 1., "loss/context_preference": .5})


@pytest.mark.parametrize("field", ["initial_state", "microbatches", "gradient_coverage", "losses", "collection_phase"])
def test_missing_witness_never_passes_even_when_missing_from_both_runs(tmp_path, field):
    def remove(rank, payload):
        payload.pop(field)
    ref, ref_grad = _write_run(tmp_path, "reference", mutation=remove)
    cand, cand_grad = _write_run(tmp_path, "candidate", mutation=remove)
    result = compare_paired_runs(ref, cand, ref_grad, cand_grad)
    assert result["passed"] is False
    assert result["errors"]


@pytest.mark.parametrize("field", ["initial_state", "input", "gradient_coverage", "world_size"])
def test_start_input_or_coverage_mismatch_fails(tmp_path, field):
    ref, ref_grad = _write_run(tmp_path, "reference")
    def mutate(rank, payload):
        if rank != 1:
            return
        if field == "initial_state":
            payload[field]["sha256"] = "wrong"
        elif field == "input":
            payload["microbatches"][0]["input_sha256"] = "wrong"
        elif field == "gradient_coverage":
            payload[field][0]["shape"] = [1, 2]
        else:
            payload[field] = 2
    cand, cand_grad = _write_run(tmp_path, "candidate", mutation=mutate)
    result = compare_paired_runs(ref, cand, ref_grad, cand_grad)
    assert result["passed"] is False
    assert result["errors"]


def test_global_gradient_difference_uses_vectors_and_metadata_coverage(tmp_path):
    ref, ref_grad = _write_run(tmp_path, "reference")
    cand, cand_grad = _write_run(tmp_path, "candidate")
    torch.save({"weight": torch.tensor([-3., -4.])}, cand_grad)
    result = compare_paired_runs(ref, cand, ref_grad, cand_grad)
    assert result.get("gradients", {}).get("relative_l2") == 2.
    assert result["passed"] is False
    torch.save({"extra": torch.ones(1)}, cand_grad)
    result = compare_paired_runs(ref, cand, ref_grad, cand_grad)
    assert result["passed"] is False
    assert result["errors"]


def test_factory_requires_production_cuda_zero2_boundary(tmp_path):
    from stage2.paired_probe import create_production_observer
    owner, _, _, _ = _owner_for_observer()
    with pytest.raises(RuntimeError, match="CUDA"):
        create_production_observer(owner, SimpleNamespace(output_dir=str(tmp_path)))


@pytest.mark.parametrize("rng_key", ["python", "numpy"])
def test_missing_rng_component_cannot_pass_when_absent_in_both_runs(tmp_path, rng_key):
    def remove(rank, payload):
        payload["microbatches"][0]["rng"].pop(rng_key)
    ref, ref_grad = _write_run(tmp_path, "reference", mutation=remove)
    cand, cand_grad = _write_run(tmp_path, "candidate", mutation=remove)
    result = compare_paired_runs(ref, cand, ref_grad, cand_grad)
    assert result["passed"] is False
    assert result["errors"]
