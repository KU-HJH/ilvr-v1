"""Behavioral witnesses for the native diagnostic, with no training or GPU use."""
import importlib.util
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint


def diagnostic():
    path = Path(__file__).parents[2] / "scripts/verify_until_native.py"
    assert path.is_file(), "native diagnostic companion is missing"
    spec = importlib.util.spec_from_file_location("until_native_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_absolute_and_relative_imports_keep_supplied_identity(tmp_path):
    # A wrong import redirect would execute the installed local loss instead.
    for folder in ("stage2", "step_preference"):
        (tmp_path / "src" / folder).mkdir(parents=True)
        (tmp_path / "src" / folder / "__init__.py").write_text("")
    (tmp_path / "src/step_preference/dpo_loss.py").write_text("def score(): return 73\n")
    (tmp_path / "src/stage2/probe.py").write_text(
        "from step_preference.dpo_loss import score\ndef run(): return score()\n")
    d = diagnostic()
    with d.ExternalSource(tmp_path) as source:
        supplied = source.module("stage2.probe")
        assert supplied.run() == 73
        assert source.redirects
        from step_preference import dpo_loss
        assert not hasattr(dpo_loss, "score")


class StatefulModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.))
        self.unused = torch.nn.Parameter(torch.tensor(5.))
        self.register_buffer("counter", torch.tensor(0.))

    def forward(self, x):
        self.counter.add_(1)
        value = checkpoint(lambda w: w.sin() * x, self.weight,
                           use_reentrant=False)
        return SimpleNamespace(loss=value + torch.rand(()) * self.weight + self.counter.clone())


class ScalarTrainer:
    def compute_loss(self, model, inputs, return_outputs=False):
        assert return_outputs is False
        return model(**inputs).loss


def test_repeat_resets_buffers_rng_and_gradients_and_observes_single_backward():
    # Missing any reset makes repeated loss/gradient or initial-state hashes differ.
    d = diagnostic()
    model = StatefulModel()
    records, gradients, values = d.run_repetitions(
        model, ScalarTrainer(), {"x": torch.tensor(3.)}, "cpu",
        observer_roots=[str(Path(__file__).parent)])
    assert len(records) == 3
    assert len({r["initial_state_sha256"] for r in records}) == 1
    assert len({r["rng_before"]["cpu"] for r in records}) == 1
    assert len({r["loss"] for r in records}) == 1
    assert all(torch.equal(gradients[0]["weight"], g["weight"]) for g in gradients)
    assert all(r["gradients_cleared"] for r in records)
    assert all(r["dispatch"]["backward_calls"] == 1 for r in records)
    assert all(r["dispatch"]["checkpoint_recompute_calls"] > 0 for r in records)
    assert records[0]["gradients"]["unused"]["connected"] is False
    # Hand-derived derivative includes the one common /4 divisor.
    torch.manual_seed(173)
    expected = (3 * torch.cos(torch.tensor(2.)) + torch.rand(())) / 4
    torch.testing.assert_close(gradients[0]["weight"], expected)


def test_identity_and_tensor_mismatch_are_reported_without_coercing_coordinates():
    d = diagnostic()
    left, right = StatefulModel(), StatefulModel()
    with torch.no_grad():
        right.weight.add_(1)
    right.unused.requires_grad_(False)
    identity = d.model_identity(left, right)
    assert not identity["all_state_exact_equal"]
    assert "weight" in identity["state_differences"]
    assert not identity["trainability_equal"]
    assert d.compare_tensors(torch.ones(1), torch.ones(2))["comparable"] is False
    assert d.tensor_record(torch.tensor(float("inf")))["finite"] is False
    assert d.tensor_record(None)["connected"] is False


def test_identity_rejects_different_nonpersistent_rope_buffers():
    # state_dict omits these buffers although they change runtime rotary positions.
    d = diagnostic()
    left, right = StatefulModel(), StatefulModel()
    left.register_buffer("inv_freq", torch.tensor([1., .1]), persistent=False)
    right.register_buffer("inv_freq", torch.tensor([1., .2]), persistent=False)
    assert left.state_dict().keys() == right.state_dict().keys()
    identity = d.model_identity(left, right)
    assert not identity["all_state_exact_equal"]
    assert "inv_freq" in identity["state_differences"]


def test_checkpoint_worker_thread_has_module_witness_even_without_profile_events():
    # sys.setprofile alone misses this valid backward on a different Python thread.
    d = diagnostic()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Sigmoid())
    observer = d.DispatchObserver([str(Path(__file__).parent)], model=model)
    with observer:
        value = checkpoint(model, torch.ones(1, 2), use_reentrant=False)
        observer.phase = "backward"
        worker = threading.Thread(target=lambda: value.sum().backward())
        worker.start()
        worker.join()
    assert model[0].weight.grad is not None
    report = observer.report()
    assert report["caller_thread_checkpoint_recompute_calls"] == 0
    assert report["module_checkpoint_recompute_calls"] == 1
    assert report["checkpoint_coverage"]["complete_for_observed_module_regions"] is True
    assert report["checkpoint_coverage"]["module_recompute_thread_ids"] != [threading.get_ident()]


def test_each_native_trainer_runs_its_own_objective_and_checkpoint_backward(tmp_path):
    # Replacing either compute_loss with a common reducer loses the native source witness.
    d = diagnostic()
    root = Path(d.baseline.SOURCE)
    if not root.is_dir():
        pytest.skip("supplied source export is unavailable")
    class Tokenizer:
        def __len__(self): return 128
        def convert_tokens_to_ids(self, token):
            return dict(zip(("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>",
                             "<|image_pad|>", "<|vision_start|>", "<|video_pad|>"),
                            (90, 91, 92, 80, 81, 82)))[token]
    with d.ExternalSource(root) as source:
        external = d.external_modules(source)
        models, _ = d.baseline.tiny_models(Tokenizer(), external.model)
        case = d.baseline.tiny_cases(Tokenizer())[1]
        spans = case["spans"]
        local_batch = dict(case["inputs"], sample_ids=["tiny"], reasoning_step_spans=spans,
                           step_preference={"steps": [{"span": s} for s in spans]})
        ext_spans = [external.spans.ReasoningStepSpan(**asdict(s)) for s in spans]
        ext_batch = dict(case["inputs"], sample_ids=["tiny"], reasoning_step_spans=[ext_spans],
            step_preference_items=[external.data.StepPreferenceBatchItem(
                sample_id="tiny", span=s,
                chosen_predictor_positions=torch.arange(s.target_text_start-1, s.target_text_end-1),
                chosen_target_ids=case["inputs"]["input_ids"][0, s.target_text_start:s.target_text_end])
                for s in ext_spans])
        trainers = d.make_trainers(models, external, tmp_path, "cpu", backend="dense")
        for index, (model, trainer, batch) in enumerate(zip(models, trainers, (local_batch, ext_batch))):
            records, _, _ = d.run_repetitions(model, trainer, batch, "cpu", repeats=1,
                observer_roots=[str(root), str(Path(__file__).parents[2] / "src"),
                                str(Path(__file__).parents[2] / "transformers")])
            record = records[0]
            calls = record["dispatch"]["calls"]
            assert any("trainer.py" in c["file"] and c["function"].endswith("compute_loss")
                       and c["phase"] == "forward" for c in calls)
            native_reducer = "_reduce_context_objective" if index == 0 else "_reduce_step_preference_components"
            assert any(c["function"].endswith(native_reducer) for c in calls)
            assert record["dispatch"]["backward_calls"] == 1
            assert record["dispatch"]["checkpoint_recompute_calls"] > 0
            assert record["loss_components"]
            assert record["gradients"][next(iter(record["gradients"]))]["finite"] is not False
            assert record["output_tensors"]["selected_clean_logits"]["connected"]
            assert record["forward_request"]["stage2_loss_only"] is True
            assert record["forward_request"]["selected_predictor_positions"] == [10, 11, 16, 17, 18, 19]
            assert record["forward_request"]["native_sidecar_plan"]
            assert record["dispatch"]["checkpoint_coverage"]["complete_for_observed_module_regions"] is True
            if index == 1:
                assert record["dispatch"]["native_counter_matches_module_witness"] is True


def test_external_stage2_objective_dependency_is_supplied_not_local(tmp_path):
    for folder in ("stage2", "stage2_objective"):
        (tmp_path / "src" / folder).mkdir(parents=True)
        (tmp_path / "src" / folder / "__init__.py").write_text("")
    (tmp_path / "src/stage2_objective/config.py").write_text("VALUE = 137\n")
    (tmp_path / "src/stage2/probe.py").write_text(
        "from stage2_objective.config import VALUE\ndef run(): return VALUE\n")
    with diagnostic().ExternalSource(tmp_path) as source:
        assert source.translated("stage2_objective.config") != "stage2_objective.config"
        assert source.module("stage2.probe").run() == 137


def native_reducer_fixture():
    from step_preference.config import StepPreferenceConfig
    from step_preference.equivalence import _trainer
    trainer = _trainer(StepPreferenceConfig(enabled=True, mode="stage2b", direct_from_stage1=True,
        visual_only_context_preference=True, beta=.5, lambda_preference=1., lambda_entropy=0.))
    return trainer


def test_component_capture_uses_connected_native_reducer_tensors():
    # Detached log metrics or CE substituted for preference breaks these analytic derivatives.
    d = diagnostic()
    trainer = native_reducer_fixture()
    weight = torch.tensor(2., requires_grad=True)
    capture = d.NativeComponents(trainer)
    with d.DispatchObserver([str(Path(__file__).parents[2] / "src")], capture=capture):
        total, _ = trainer._reduce_context_objective(weight.square(), [weight], [weight.reshape(1)*0],
            {(0, "visual"): {"mean_logp": -weight, "entropies": weight.reshape(1)*0}})
    capture.validate(total)
    assert float(capture.scores[0]["clean"][0]) == 2.
    assert float(capture.scores[0]["off"][0]) == -2.
    assert torch.autograd.grad(capture.select("ce"), weight, retain_graph=True)[0].item() == 4.
    assert torch.autograd.grad(capture.select("preference"), weight, retain_graph=True)[0].item() == pytest.approx(-.11920292)
    assert torch.autograd.grad(capture.select("total"), weight)[0].item() == pytest.approx(3.88079708)


def test_component_capture_rejects_missing_or_detached_instrumentation():
    # A hook that misses the reducer or replaces captured tensors with detached metrics must fail closed.
    d = diagnostic()
    capture = d.NativeComponents(native_reducer_fixture())
    with pytest.raises(ValueError, match="capture"):
        capture.validate(torch.tensor(1., requires_grad=True))
    weight = torch.tensor(2., requires_grad=True)
    with d.DispatchObserver([str(Path(__file__).parents[2] / "src")], capture=capture):
        total, _ = capture.trainer._reduce_context_objective(weight.square(), [weight], [weight.reshape(1)*0],
            {(0, "visual"): {"mean_logp": -weight, "entropies": weight.reshape(1)*0}})
    capture.tensors["preference"] = capture.tensors["preference"].detach()
    with pytest.raises(ValueError, match="capture"):
        capture.validate(total)


def test_gradient_decomposition_detects_wrong_component_and_preserves_none_vs_zero():
    # An incorrect preference gradient changes the residual even when loss scalars agree.
    d = diagnostic()
    ce = {"layer.weight": torch.tensor([2., 0.]), "unused": None, "zero": torch.tensor(0.)}
    pref = {"layer.weight": torch.tensor([-1., 1.]), "unused": None, "zero": None}
    total = {"layer.weight": torch.tensor([1., 1.]), "unused": None, "zero": torch.tensor(0.)}
    report = d.gradient_geometry(ce, pref, total)
    assert report["sum_residual_l2"] == 0.
    assert report["cosine"] == pytest.approx(-2 ** -.5)
    assert report["preference_ce_norm_ratio"] == pytest.approx(2 ** -.5)
    assert report["connectivity"]["zero"] == {"ce": True, "preference": False, "total": True}
    total["layer.weight"] = torch.tensor([1., 2.])
    assert d.gradient_geometry(ce, pref, total)["sum_residual_l2"] == 1.
    total["zero"] = None
    assert d.gradient_geometry(ce, pref, total)["connection_mismatches"] == ["zero"]


def test_component_mode_defaults_local_and_requires_explicit_external_source():
    # Local-first must not silently load an obsolete exported source bundle.
    d = diagnostic()
    args = d.parse_args(["--mode", "components"])
    assert args.side == "local"
    assert args.initializer == "both"
    assert args.external_source is None
    with pytest.raises(ValueError, match="explicit.*external"):
        d.validate_component_args(d.parse_args(["--mode", "components", "--side", "external"]))


def test_component_checkpoint_coverage_allows_unused_sidecar_but_rejects_unknown_region():
    # CE legitimately skips sidecar recomputation; a recomputation absent from forward is invalid.
    d = diagnostic()
    model = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
    observer = d.DispatchObserver([], model=model)
    with observer:
        ce = checkpoint(model[0], torch.ones(1, 2), use_reentrant=False).square().sum()
        off = checkpoint(model[1], torch.ones(1, 2), use_reentrant=False).square().sum()
        observer.phase = "backward"
        ce.backward()
    record = d.component_checkpoint_coverage(observer, "ce")
    assert record["observed_component_valid"] is True
    assert record["omitted_forward_regions"] == 1
    assert record["total_coverage_verified"] is False
    with pytest.raises(ValueError, match="coverage"):
        d.component_checkpoint_coverage(observer, "total")
    observer.recomputed.add(("unobserved-region", 999))
    with pytest.raises(ValueError, match="coverage"):
        d.component_checkpoint_coverage(observer, "ce")


def test_repeat_component_selects_native_preference_and_restores_state():
    # Selecting total for every backward would yield 3.88/4 instead of -0.119/4.
    d = diagnostic()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.))
        def forward(self, input_ids):
            return SimpleNamespace(loss=self.weight.square(), score=self.weight)
    reducer = native_reducer_fixture()
    class Trainer:
        step_preference_config = reducer.step_preference_config
        _reduce_context_objective = type(reducer)._reduce_context_objective
        def compute_loss(self, model, inputs, return_outputs=False):
            output = model(**inputs)
            return self._reduce_context_objective(output.loss, [output.score], [output.score.reshape(1)*0],
                {(0, "visual"): {"mean_logp": -output.score, "entropies": output.score.reshape(1)*0}})[0]
    model = Model()
    before = d.state_digest(model)
    records, grads, _ = d.run_repetitions(model, Trainer(), {"input_ids": torch.tensor([[1, 2]])},
                                        "cpu", component="preference")
    assert len(records) == 3
    assert all(g["weight"].item() == pytest.approx(-.11920292 / 4) for g in grads)
    assert all(r["component_capture"]["identity_validated"] for r in records)
    assert d.state_digest(model) == before
    assert model.weight.grad is None


def measurement_fixture():
    from copy import deepcopy
    row = dict(initial_state_sha256="unchanged", state_after_backward_sha256="unchanged",
               forward_request={"selected_target_ids": [1]}, rng_before={"cpu": "fixed"},
               gradients={"w": {"finite": True}}, component_checkpoint_coverage={"observed_component_valid": True},
               component_capture={"identity_validated": True, "values": {"ce": 2., "preference": 1., "total": 3.},
                                  "scores": [{"clean": [2.], "off": [1.]}]})
    records = {key: [deepcopy(row) for _ in range(3)] for key in ("ce", "preference", "total")}
    gradients = {"ce": [{"w": torch.tensor([2.])} for _ in range(3)],
                 "preference": [{"w": torch.tensor([-1.])} for _ in range(3)],
                 "total": [{"w": torch.tensor([1.])} for _ in range(3)]}
    return records, gradients


def test_measurement_rejects_coordinate_mutation_and_leaves_beyond_repeat_residual_unresolved():
    # A blanket BF16 tolerance would conceal this small missing preference gradient.
    d = diagnostic()
    records, gradients = measurement_fixture()
    result = d.component_measurements(records, gradients)
    assert result["reconstruction_within_observed_repeat_spread"] is True
    gradients["total"] = [{"w": torch.tensor([1.001])} for _ in range(3)]
    result = d.component_measurements(records, gradients)
    assert result["sum_repeat_spread_bound"] == 0.
    assert result["reconstruction_within_observed_repeat_spread"] is False
    records["ce"][0]["forward_request"] = {"selected_target_ids": [2]}
    with pytest.raises(ValueError, match="coordinates"):
        d.component_measurements(records, gradients)
    records, gradients = measurement_fixture()
    records["ce"][0]["state_after_backward_sha256"] = "mutated"
    with pytest.raises(ValueError, match="mutat"):
        d.component_measurements(records, gradients)


def test_screen_requires_reconstruction_before_comparing_mechanisms():
    # Disjoint intervals must not authorize expansion when gradient reconstruction is unresolved.
    d = diagnostic()
    def cell(cosines, ratios, valid=True):
        return {"measurements": {"reconstruction_within_observed_repeat_spread": valid,
                "geometry": [{"cosine": c, "preference_ce_norm_ratio": r} for c, r in zip(cosines, ratios)]}}
    cells = {"e1": cell([-.9, -.8, -.85], [1., 1.1, 1.]),
             "e3": cell([-.4, -.3, -.35], [2., 2.1, 2.])}
    assert d.screen_initializers(cells)["opposition_candidate"] is True
    assert d.screen_initializers(cells)["scale_candidate"] is True
    cells["e3"]["measurements"]["reconstruction_within_observed_repeat_spread"] = False
    assert d.screen_initializers(cells)["source_expansion_candidate"] is False
    cells["e3"] = cell([-.85, -.7, -.75], [1.05, 1.15, 1.1])
    assert d.screen_initializers(cells)["source_expansion_candidate"] is False


def test_native_tiny_component_cell_executes_fresh_forward_per_backward(tmp_path, capsys):
    # Reusing a freed graph, substituting a common loss, or CE-sidecar coverage conflation breaks this native path.
    d = diagnostic()
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
    class Tokenizer:
        def __len__(self): return 128
        def convert_tokens_to_ids(self, token):
            return dict(zip(("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>",
                             "<|image_pad|>", "<|vision_start|>", "<|video_pad|>"),
                            (90, 91, 92, 80, 81, 82)))[token]
    models, _ = d.baseline.tiny_models(Tokenizer(), modeling_qwen2_5_vl)
    case = d.baseline.tiny_cases(Tokenizer())[1]
    spans = case["spans"]
    batch = dict(case["inputs"], sample_ids=["tiny"], reasoning_step_spans=spans,
                 step_preference={"steps": [{"span": span} for span in spans]})
    trainer = d.make_trainers([models[0]], None, tmp_path, "cpu", backend="dense")[0]
    result = d.run_component_cell(models[0], trainer, batch, "cpu", [str(Path(__file__).parents[2] / "src")])
    assert set(result["runs"]) == {"ce", "preference", "total"}
    assert all(len(rows) == 3 for rows in result["runs"].values())
    assert all(row["dispatch"]["backward_calls"] == 1 for rows in result["runs"].values() for row in rows)
    assert all(row["component_capture"]["identity_validated"] for rows in result["runs"].values() for row in rows)
    assert all(row["component_checkpoint_coverage"]["total_coverage_verified"] for row in result["runs"]["total"])
    assert any(row["component_checkpoint_coverage"]["omitted_forward_regions"] for row in result["runs"]["ce"])
    assert all(row["component_capture"]["scores"] for rows in result["runs"].values() for row in rows)
    import json
    captured = capsys.readouterr()
    progress = [json.loads(line) for line in captured.err.splitlines() if line.startswith('{"progress"')]
    assert len(progress) == 9
    assert {row["component"] for row in progress} == {"ce", "preference", "total"}
    assert not captured.out
    assert not any(path.is_file() for path in tmp_path.rglob("*"))
    json.dumps(result, allow_nan=False)  # No graph/gradient/model tensors escape into output.


def test_component_runner_refuses_cpu_real_checkpoint_execution():
    # Accidental production-model loading before CUDA/resource validation is forbidden.
    d = diagnostic()
    with pytest.raises(ValueError, match="CUDA"):
        d.run_components(d.parse_args(["--mode", "components", "--device", "cpu"]))


def test_capture_distinguishes_genuine_native_disconnection_from_instrumentation_failure():
    # A native detached preference is observable, whereas a detached observer copy is invalid.
    d = diagnostic()
    trainer = native_reducer_fixture()
    capture = d.NativeComponents(trainer)
    weight = torch.tensor(2., requires_grad=True)
    with d.DispatchObserver([], capture=capture):
        total, _ = trainer._reduce_context_objective(weight.square(), [torch.tensor(2.)], [torch.zeros(1)],
            {(0, "visual"): {"mean_logp": torch.tensor(-2.), "entropies": torch.zeros(1)}})
    result = capture.validate(total)
    assert result["native_graph_status"]["preference"] == "native_requires_grad_false"
    assert result["native_graph_status"]["ce"] == "connected_to_native_total"


def test_component_forward_values_must_reproduce_within_repeat_variation():
    # Identical gradients do not rescue component forwards that observed different scores.
    from copy import deepcopy
    d = diagnostic()
    records, gradients = measurement_fixture()
    records = deepcopy(records)
    records["preference"][0]["component_capture"]["scores"] = [{"clean": [2.], "off": [1.2]}]
    records["preference"][1]["component_capture"]["scores"] = [{"clean": [2.], "off": [1.2]}]
    records["preference"][2]["component_capture"]["scores"] = [{"clean": [2.], "off": [1.2]}]
    result = d.component_measurements(records, gradients)
    assert result["gradient_reconstruction_within_observed_repeat_spread"] is True
    assert result["forward_reconstruction_within_observed_repeat_spread"] is False
    assert result["reconstruction_within_observed_repeat_spread"] is False


@pytest.mark.parametrize("case", ["native_disconnected", "connected_without_checkpoints", "missing_capture"])
def test_component_cell_preserves_validated_native_disconnection_but_rejects_invalid_measurement(case):
    # A real reducer's detached preference must survive aggregation; broken instrumentation must not.
    d = diagnostic()
    reducer = native_reducer_fixture()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.layer.weight.fill_(2.)
        def forward(self, input_ids):
            x = input_ids.float()
            value = (self.layer(x) if case == "connected_without_checkpoints" else
                     checkpoint(self.layer, x, use_reentrant=False))
            score = value.sum() if case == "connected_without_checkpoints" else torch.tensor(2.)
            return SimpleNamespace(loss=value.square().sum(), score=score)
    class Trainer:
        step_preference_config = reducer.step_preference_config
        _reduce_context_objective = type(reducer)._reduce_context_objective
        def compute_loss(self, model, inputs, return_outputs=False):
            output = model(**inputs)
            if case == "missing_capture":
                return output.loss
            return self._reduce_context_objective(output.loss, [output.score], [output.score.reshape(1)*0],
                {(0, "visual"): {"mean_logp": -output.score, "entropies": output.score.reshape(1)*0}})[0]
    model = Model()
    batch = {"input_ids": torch.tensor([[1]])}
    if case != "native_disconnected":
        with pytest.raises(ValueError, match="coverage" if case == "connected_without_checkpoints" else "capture"):
            d.run_component_cell(model, Trainer(), batch, "cpu", [])
        return
    result = d.run_component_cell(model, Trainer(), batch, "cpu", [])
    assert all(len(rows) == 3 for rows in result["runs"].values())
    for row in result["runs"]["preference"]:
        assert row["component_capture"]["identity_validated"] is True
        assert row["component_capture"]["native_graph_status"]["preference"] == "native_requires_grad_false"
        assert row["dispatch"]["backward_calls"] == 0
        assert row["component_checkpoint_coverage"] is None
        assert all(g["connected"] is False for g in row["gradients"].values())
    assert all(row["component_checkpoint_coverage"]["total_coverage_verified"] for row in result["runs"]["total"])
    assert result["measurements"]["reconstruction_status"] == "inconclusive_native_disconnection"
    assert result["measurements"]["native_disconnected_components"] == ["preference"]
    assert result["measurements"]["reconstruction_within_observed_repeat_spread"] is False
    screen = d.screen_initializers({"e1": result, "e3": result})
    assert screen["status"] == "inconclusive"
    assert screen["source_expansion_candidate"] is False


def test_fp32_control_rejects_unapproved_mode_side_or_initializer_before_loading():
    # An ignored dtype flag or a broadened runner would execute an unapproved cell.
    d = diagnostic()
    assert d.parse_args([]).decoder_dtype == "bfloat16"
    allowed = d.parse_args(["--mode", "components", "--initializer", "e1", "--decoder-dtype", "float32"])
    d.validate_component_args(allowed)
    for extra in ([], ["--mode", "components"],
                  ["--mode", "components", "--initializer", "e3"],
                  ["--mode", "components", "--initializer", "e1", "--side", "external", "--external-source", "/absent"]):
        with pytest.raises(ValueError, match="FP32.*local.*e1.*component"):
            d.run(d.parse_args(["--decoder-dtype", "float32", *extra]))


def precision_tiny_fixture():
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
    from peft_common import resolve_qwen2_5_vl_model
    class Tokenizer:
        def __len__(self): return 128
        def convert_tokens_to_ids(self, token):
            return dict(zip(("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>",
                             "<|image_pad|>", "<|vision_start|>", "<|video_pad|>"),
                            (90, 91, 92, 80, 81, 82)))[token]
    d = diagnostic()
    models, _ = d.baseline.tiny_models(Tokenizer(), modeling_qwen2_5_vl)
    return d, models[0], resolve_qwen2_5_vl_model(models[0]), d.baseline.tiny_cases(Tokenizer())[1]


def test_fp32_promotion_preserves_bf16_rounded_values_wrappers_and_parameter_identity():
    # Loading original FP32 values or casting only the decoder misses wrapper parameters.
    d, model, qwen, _ = precision_tiny_fixture()
    before = {n: p.detach().clone() for n, p in list(model.named_parameters()) + list(model.named_buffers())}
    identities = {n: id(p) for n, p in model.named_parameters()}
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    embedding = qwen.model.embed_tokens
    head = qwen.lm_head
    runtime = d.prepare_component_precision(model, "float32", production=False)
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        expected_dtype = torch.bfloat16 if ".visual." in name else torch.float32
        if value.is_floating_point():
            assert value.dtype == expected_dtype
        assert torch.equal(value, before[name].to(value.dtype)), name
    assert identities == {n: id(p) for n, p in model.named_parameters()}
    assert trainable == [n for n, p in model.named_parameters() if p.requires_grad]
    assert qwen.model.embed_tokens is embedding and qwen.lm_head is head
    assert not any(p.requires_grad for p in qwen.visual.parameters())
    assert runtime["precision_control"]["promoted_state_exact"] is True
    assert runtime["precision_control"]["expected_promoted_sha256"] == runtime["precision_control"]["actual_promoted_sha256"]
    assert set(runtime["trainable_parameter_dtypes"].values()) == {"torch.float32"}
    assert runtime["precision_control"]["float32_matmul_precision"] == "highest"
    assert runtime["precision_control"]["cuda_matmul_allow_tf32"] is False
    with pytest.raises(ValueError, match="vision attention"):
        d.baseline.runtime_model_metadata(model, production=True)


@pytest.mark.parametrize("broken", ["vision_trainable", "initial_fp32", "decoder_attention", "production_vision", "tf32"])
def test_fp32_authentication_rejects_wrong_starting_profile_without_fallback(broken):
    # The control must never bless another initial precision, trainable vision, or backend.
    d, model, qwen, _ = precision_tiny_fixture()
    old_precision = torch.get_float32_matmul_precision()
    try:
        if broken == "vision_trainable":
            next(qwen.visual.parameters()).requires_grad_(True)
        elif broken == "initial_fp32":
            qwen.lm_head.float()
        elif broken == "decoder_attention":
            qwen.model.layers[0].self_attn = torch.nn.Identity()
        elif broken == "tf32":
            torch.set_float32_matmul_precision("high")
        with pytest.raises(ValueError):
            d.prepare_component_precision(model, "float32", production=broken == "production_vision")
    finally:
        torch.set_float32_matmul_precision(old_precision)


def test_default_precision_keeps_original_bf16_state_and_runtime_authentication():
    # Default callers must retain their original dtype and strict production profile gate.
    d, model, _, _ = precision_tiny_fixture()
    before = d.state_digest(model)
    runtime = d.prepare_component_precision(model, "bfloat16", production=False)
    assert d.state_digest(model) == before
    assert runtime["parameter_dtypes"] == ["torch.bfloat16"]
    assert "precision_control" not in runtime
    with pytest.raises(ValueError, match="vision attention"):
        d.prepare_component_precision(model, "bfloat16", production=True)


def test_fp32_native_cell_promotes_before_nine_fresh_backwards_and_observes_actual_dtypes(tmp_path):
    # Wiring promotion after a forward, omitting it, or assuming parameter dtype proves activation dtype fails here.
    d, model, qwen, case = precision_tiny_fixture()
    spans = case["spans"]
    batch = dict(case["inputs"], sample_ids=["tiny"], reasoning_step_spans=spans,
                 step_preference={"steps": [{"span": span} for span in spans]})
    trainer = d.make_trainers([model], None, tmp_path, "cpu", backend="dense")[0]
    original = d.state_digest(model)
    result = d.run_component_cell(model, trainer, batch, "cpu", [str(Path(__file__).parents[2] / "src")],
                                  decoder_dtype="float32", production=False)
    assert result["runtime"]["precision_control"]["initial_bf16_state_sha256"] == original
    rows = [row for group in result["runs"].values() for row in group]
    assert len(rows) == 9
    assert all(row["dispatch"]["backward_calls"] == 1 for row in rows)
    assert len({row["initial_state_sha256"] for row in rows}) == 1
    assert all(row["forward_request"] == rows[0]["forward_request"] for row in rows)
    assert all(row["component_capture"]["identity_validated"] for row in rows)
    assert all(row["component_checkpoint_coverage"]["total_coverage_verified"] for row in result["runs"]["total"])
    for row in rows:
        witness = row["dispatch"]["precision_activations"]
        assert witness["vision"]["output_dtypes"] == ["torch.bfloat16"]
        assert witness["embedding"]["output_dtypes"] == ["torch.float32"]
        assert witness["head"]["output_dtypes"] == ["torch.float32"]
        for i in range(3):
            assert witness[f"attention.{i}"]["input_dtypes"] == ["torch.float32"]
            assert witness[f"attention.{i}"]["output_dtypes"] == ["torch.float32"]
        assert {g["dtype"] for g in row["gradients"].values() if g["connected"]} == {"torch.float32"}
    assert all(p.grad is None for p in model.parameters())
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_precision_observer_rejects_actual_wrong_activation_dtype_and_removes_hooks():
    # A cast in the real output path must be caught even when parameters have the requested dtype.
    d, model, qwen, _ = precision_tiny_fixture()
    d.prepare_component_precision(model, "float32", production=False)
    wrong = qwen.lm_head.register_forward_hook(lambda module, args, output: output.bfloat16())
    counts = [len(module._forward_hooks) for module in model.modules()]
    try:
        with pytest.raises(ValueError, match="activation dtype"):
            with d.DispatchObserver([], model=model, precision=True):
                qwen.lm_head(torch.zeros(1, 2, 8))
        assert counts == [len(module._forward_hooks) for module in model.modules()]
    finally:
        wrong.remove()


def test_gradient_residual_reports_both_normalizations_without_relaxing_screen():
    # Reporting only CE-normalized residual can hide a large error relative to preference.
    d = diagnostic()
    geometry = d.gradient_geometry({"w": torch.tensor([4.])}, {"w": torch.tensor([2.])}, {"w": torch.tensor([7.])})
    assert geometry["residual_ce_norm_ratio"] == .25
    assert geometry["residual_preference_norm_ratio"] == .5
    zero = d.gradient_geometry({"w": torch.zeros(1)}, {"w": None}, {"w": torch.ones(1)})
    assert zero["residual_ce_norm_ratio"] is None
    assert zero["residual_preference_norm_ratio"] is None
    records, gradients = measurement_fixture()
    for repeat in gradients["total"]:
        repeat["w"] += 1
    assert d.component_measurements(records, gradients)["reconstruction_within_observed_repeat_spread"] is False
