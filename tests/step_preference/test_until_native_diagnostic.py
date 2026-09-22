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
