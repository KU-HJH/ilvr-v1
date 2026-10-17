"""Fixed diagnostic contracts; tensor fixtures stay inside pytest basetemp."""
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def full_rank_record(rank, digest="a"*64, variant="portTorch"):
    """Synthetic CPU contract metadata, never asserted as actual GPU evidence."""
    import copy
    p = probe()
    tensor = {"shape": [1], "dtype": "torch.float32", "sha256": "b"*64}
    state = {"weight": {**tensor, "trainable": True}, "unused": {**tensor, "trainable": True}}
    rng = {phase: {device: {"shape": [1], "dtype": "torch.uint8", "sha256": "b"*64}
                   for device in ("cpu", "cuda")} for phase in ("pre_forward", "post_forward", "post_backward")}
    cases = []
    for case in p.CASES:
        rows = [[[case["sample_id"], step, position, 7] for position in range(count)]
                for step, count in enumerate(case["branch_target_counts"])]
        selected = [group[0] for group in rows]
        keys = (["loss/ce", "loss/dil_positive", "loss/dil_negative", "loss/topk_mass", "loss/context_preference", "loss/total"]
                if variant == "portTorch" else ["loss/ce", "loss/dil_bce_positive", "loss/dil_bce_negative_topk",
                                               "loss/topk_mass_kl", "loss/topk_mass_weighted", "loss/context_preference_all", "loss/total"])
        objective = {"components": [{key: 1. for key in keys}],
                     "selections": [{"rows": selected, "inputs": [tensor], "masks": [[True]+[False]*(len(group)-1) for group in rows]}],
                     "support": [{"phase": "forward", "rows": selected, "targets": [7]*len(selected), "ids": [[1, 2]]*len(selected)}],
                     "original_step_denominator": case["blocks"], "support_mapping_unresolved": []}
        observations = []
        for repeat in (0, 1):
            obs = {"repeat": repeat, "loss": 1., "gradients": {"weight": tensor, "unused": None}, "rng": rng,
                   "tensor_path": str(ROOT / f"checkpoints/probe-contract/diagnostic-tensors/rank-{rank}/case-{case['raw_index']}-repeat-{repeat}.pt"),
                   "tensor_sha256": "b"*64, "objective": copy.deepcopy(objective)}
            if repeat:
                obs.update(repeat_comparison=p.compare_gradients({"weight": torch.ones(1), "unused": None}, {"weight": torch.ones(1), "unused": None}),
                           repeat_loss_pass=True, repeat_rng_equal=True, repeat_observation_equal=True,
                           repeat_objective_comparison=p.compare_repeat_objectives(objective, objective))
            observations.append(obs)
        cases.append({"raw_index": case["raw_index"], "sample_id": case["sample_id"], "repeats": [0, 1], "observations": observations,
                      "batch": {"sample_ids": [case["sample_id"]]}, "logical_rows": rows,
                      "batch_tensors": {key: tensor for key in ("input_ids", "labels", "attention_mask", "pixel_values", "image_grid_thw")}})
    return {"kind": "dil-full-model-probe-rank", "schema": 1, "run_id": "probe-contract",
            "manifest_sha256": digest, "variant": variant, "rank": rank, "world_size": 4,
            "machine_rank": rank//2, "local_rank": rank%2, "device": rank%2, "host": "remote" if rank < 2 else "local",
            "status": "complete", "model_state_unchanged": True, "deepspeed_engine_prepared": False,
            "loaded_state": state, "bf16_cast_state": state, "profile": p.PROFILE, "comparator": p.COMPARATOR, "cases": cases}


def probe():
    path = ROOT / "scripts/probe_dil_full_model.py"
    assert path.is_file(), "fixed full-model diagnostic is missing"
    spec = importlib.util.spec_from_file_location("dil_full_model_probe_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparator_preserves_none_names_shapes_and_reports_coordinates():
    p = probe()
    left = {"a": torch.tensor([1., 0.]), "unused": None}
    right = {"a": torch.tensor([1., 0.001]), "unused": None}
    result = p.compare_gradients(left, right)
    assert not result["pass"]
    assert result["parameters"]["a"]["failed_coordinates"] == 1
    assert result["max_abs_error"] == pytest.approx(.001)
    assert result["relative_l2"] == pytest.approx(.001)
    assert not p.compare_gradients(left, {"a": left["a"], "unused": torch.tensor(0.)})["pass"]
    assert not p.compare_gradients(left, {"renamed": left["a"], "unused": None})["pass"]
    assert not p.compare_gradients(left, {"a": torch.ones(1, 2), "unused": None})["pass"]
    assert not p.compare_gradients({"a": torch.ones(2)}, {"a": torch.ones(2, dtype=torch.float64)})["pass"]


def test_isolation_keeps_venv_and_removes_competing_repository(tmp_path):
    p = probe()
    source = tmp_path / "source"
    source.mkdir()
    paths = [str(ROOT / "src"), str(ROOT / "transformers/src"),
             "/repo/venv/lib/python3.11/site-packages", "/usr/lib/python3.11"]
    result = p.isolated_paths(source, ROOT, paths)
    assert result[:3] == [str(source / "transformers/src"), str(source / "src"), str(source)]
    assert "/repo/venv/lib/python3.11/site-packages" in result
    assert str(ROOT / "src") not in result


def test_observation_returns_original_once_restores_and_preserves_rng():
    p = probe()
    returned = object()
    owner = SimpleNamespace(call=lambda x: returned)
    seen = []
    state = torch.get_rng_state().clone()
    original = owner.call
    with p.observe_call(owner, "call", lambda args, kwargs, result: seen.append(result)):
        assert owner.call(3) is returned
    assert owner.call is original
    assert seen == [returned]
    assert torch.equal(state, torch.get_rng_state())


def test_native_scope_encloses_loss_and_all_parameter_derivatives():
    p = probe()
    model = torch.nn.Linear(2, 1)
    active = []
    @contextmanager
    def scope():
        active.append(True)
        yield
        active.pop()
    class Trainer:
        def compute_loss(self, model, batch, return_outputs=False):
            assert active and not return_outputs
            return model(batch).sum()
    model.weight.register_hook(lambda grad: grad if active else (_ for _ in ()).throw(AssertionError("scope ended")))
    loss, gradients = p.differentiate(Trainer(), model, torch.ones(1, 2), scope())
    assert loss == pytest.approx(float(model(torch.ones(1, 2)).sum()))
    assert set(gradients) == {"weight", "bias"}
    assert torch.equal(gradients["weight"], torch.ones(1, 2))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_fixed_config_mapping_uses_actual_port_parser(tmp_path, monkeypatch):
    p = probe()
    monkeypatch.syspath_prepend(str(ROOT / "transformers/src"))
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from utils_deepseed import parse_args
    manifest = {"base": "/models/base", "initializer": str(ROOT / "checkpoints/init/checkpoint-431"),
                "data_path": str(ROOT / "data/TRAIN.jsonl"), "image_root": str(ROOT / "data")}
    args = parse_args(p.runtime_arguments(manifest, ROOT, "probe-contract", "portTorch"))
    assert args.step_preference_config.target_scope == "until_next_latent"
    assert args.step_preference_config.preference_topk == 1516
    assert args.step_preference_config.beta == .5
    assert args.step_preference_config.lambda_topk_mass == 1.
    assert args.step_preference_config.preference_entropy_top_percent == 20.
    assert args.step_preference_config.lambda_entropy == 0.
    assert args.gradient_accumulation_steps == 4 and args.dataloader_num_workers == 4
    assert args.step_preference_config.phase_steps == 2 and args.stage2_total_steps == 2
    assert args.stage2_latent_activation_checkpointing and not args.gradient_checkpointing


def test_tensor_output_path_rejects_results_and_symlink_escape(tmp_path):
    p = probe()
    repository = tmp_path / "repo"
    (repository / "checkpoints").mkdir(parents=True)
    (repository / "results").mkdir()
    assert p.tensor_directory(repository, "probe-contract").is_relative_to(repository / "checkpoints")
    (repository / "checkpoints/probe-escape").symlink_to(repository / "results", target_is_directory=True)
    with pytest.raises(ValueError):
        p.tensor_directory(repository, "probe-escape")


def test_rank_records_require_four_actual_devices_and_complete_cases():
    p = probe()
    records = [full_rank_record(rank) for rank in range(4)]
    assert p.validate_rank_records(records, "probe-contract", "a"*64, "portTorch")["kind"] == "dil-full-model-probe"
    with pytest.raises(ValueError):
        p.validate_rank_records(records[:3], "probe-contract", "a"*64, "portTorch")
    records[3]["cases"].pop()
    with pytest.raises(ValueError):
        p.validate_rank_records(records, "probe-contract", "a"*64, "portTorch")


def test_rank_terminal_rejects_claim_without_model_gradient_or_native_objective():
    p = probe()
    records = [full_rank_record(rank) for rank in range(4)]
    records[1]["cases"][0]["observations"][0]["gradients"] = {}
    with pytest.raises(ValueError):
        p.validate_rank_records(records, "probe-contract", "a"*64, "portTorch")


def test_native_objective_comparison_maps_components_and_repeated_target_support():
    p = probe()
    # Same target token in two positions must still distinguish predictor rows.
    rows = [["sample", 0, 4, 7], ["sample", 0, 5, 7]]
    source = {"components": [{"loss/ce": 1., "loss/dil_bce_positive": 2., "loss/dil_bce_negative_topk": 3.,
                              "loss/topk_mass_kl": 4., "loss/topk_mass_weighted": 4., "loss/context_preference_all": 5., "loss/total": 10.}],
              "selections": [{"rows": rows}], "support": [{"phase": "forward", "rows": rows, "ids": [[1, 2], [2, 3]]}],
              "original_step_denominator": 1, "support_mapping_unresolved": []}
    port = {**source, "components": [{"loss/ce": 1., "loss/dil_positive": 2., "loss/dil_negative": 3.,
                                     "loss/topk_mass": 4., "loss/context_preference": 5., "loss/total": 10.}]}
    assert p.compare_objectives(source, port)["pass"]
    port = {**port, "support": [{"phase": "forward", "rows": rows, "ids": [[2, 3], [1, 2]]}]}
    assert not p.compare_objectives(source, port)["support_exact"]
    port["support"][0]["rows"] = None
    assert not p.compare_objectives(source, port)["pass"]


def test_result_comparison_rejects_incomplete_case_repeat_or_rank_pair(tmp_path):
    import json
    p = probe()
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    incomplete = {"rank": 2, "variant": "sourceFlash", "cases": []}
    left.write_text(json.dumps(incomplete))
    right.write_text(json.dumps({**incomplete, "variant": "sourceTorch"}))
    args = SimpleNamespace(left=str(left), right=str(right), repository=str(tmp_path), output=str(tmp_path / "results/comparison.json"))
    with pytest.raises(ValueError):
        p.compare_results(args)


def test_source_argument_mapping_is_accepted_by_actual_source_parser(tmp_path):
    import json
    import subprocess
    p = probe()
    source = Path('/raid/workspace/temp2/ILVR/.worktrees/ours1-dil-bce')
    manifest = {"base": "/models/base", "initializer": str(ROOT / "checkpoints/init/checkpoint-431"),
                "data_path": str(ROOT / "data/TRAIN.jsonl"), "image_root": str(ROOT / "data")}
    arguments = p.runtime_arguments(manifest, ROOT, "probe-contract", "sourceFlash")
    code = "import sys,json; sys.path[:0]="+repr([str(source / 'transformers/src'), str(source / 'src'), str(source)])+"; from utils_deepseed import parse_args; a=parse_args("+repr(arguments)+"); print(json.dumps({'scope':a.step_preference_target_scope,'phase':a.stage2_phase,'beta':a.step_preference_beta,'backend':a.stage2_rollout_backend}))"
    import os
    completed = subprocess.run([str(ROOT / "venv/ilvr-venv/bin/python"), "-c", code], cwd=tmp_path,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"scope": "until_next_latent", "phase": "stage2b", "beta": .5, "backend": "no_cache"}


def test_logical_identity_maps_source_sample_group_and_port_flat_spans():
    p = probe()
    span = SimpleNamespace(step_id=0, target_text_start=3, target_text_end=5)
    common = {"sample_ids": ["real-sample"], "input_ids": torch.tensor([[1, 2, 3, 7, 7]])}
    source = p.logical_rows({**common, "reasoning_step_spans": [[span]]})
    port = p.logical_rows({**common, "reasoning_step_spans": [span]})
    assert source == port == [[("real-sample", 0, 2, 7), ("real-sample", 0, 3, 7)]]


def test_native_port_support_maps_factory_bound_rows_in_forward_and_replay():
    p = probe()
    from tests.step_preference.test_dil_trainer import dil_case, dil_config
    from step_preference.equivalence import _trainer
    model, batch = dil_case()
    trainer = _trainer(dil_config(preference_entropy_top_percent=20))
    with p.ObjectiveObservation(trainer, "portTorch", batch) as observation:
        loss = trainer.compute_loss(model, batch, return_outputs=False)
        observation.phase = "backward_replay"
        torch.autograd.grad(loss, [parameter for parameter in model.parameters() if parameter.requires_grad], allow_unused=True)
        record = observation.record()
    forward = [chunk for chunk in record["support"] if chunk["phase"] == "forward"]
    assert forward and all(chunk["rows"] is not None for chunk in forward)
    assert not record["support_mapping_unresolved"]
    actual = [row for chunk in forward for row in chunk["rows"]]
    expected = [list(row) for rows in p.logical_rows(batch) for row in rows]
    assert actual == expected
    assert any(chunk["phase"] == "backward_replay" for chunk in record["support"])


def test_authentic_source_dora_scope_stays_patched_through_autograd(tmp_path):
    import subprocess
    import os
    code = '''
import importlib.util,sys,json
from pathlib import Path
source=Path('/raid/workspace/temp2/ILVR/.worktrees/ours1-dil-bce')
sys.path[:0]=[str(source/'transformers/src'),str(source/'src'),str(source)]
spec=importlib.util.spec_from_file_location('probe', PROBE)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
import torch
from peft import LoraConfig,get_peft_model
from stage2 import dora
assert Path(dora.__file__).is_relative_to(source)
model=get_peft_model(torch.nn.Sequential(torch.nn.Linear(2,2)),LoraConfig(target_modules=['0'],r=1,lora_alpha=2,lora_dropout=.05,use_dora=True))
seen={}
class Trainer:
 def compute_loss(self,model,batch,return_outputs=False):
  assert seen['patched']==1
  return model(batch).sum()
for param in model.parameters():
 if param.requires_grad:
  param.register_hook(lambda grad: grad if seen['patched']==1 else (_ for _ in ()).throw(AssertionError('ended scope')))
loss,grads=p.differentiate(Trainer(),model,torch.ones(1,2),p.native_dora_scope(model,'sourceFlash',seen))
assert seen['eligible']==seen['patched']==1 and seen['restored']
assert all(param.grad is None for param in model.parameters())
assert any(value is not None and torch.count_nonzero(value) for value in grads.values())
print(json.dumps(seen))
'''.replace('PROBE', repr(str(ROOT / 'scripts/probe_dil_full_model.py')))
    completed = subprocess.run([str(ROOT / "venv/ilvr-venv/bin/python"), "-c", code], cwd=tmp_path,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert '"restored": true' in completed.stdout


def test_source_torch_rotary_control_authenticates_callable_and_restores(tmp_path):
    import subprocess
    import os
    code = '''
import importlib.util,sys
from pathlib import Path
source=Path('/raid/workspace/temp2/ILVR/.worktrees/ours1-dil-bce')
sys.path[:0]=[str(source/'transformers/src'),str(source/'src'),str(source)]
spec=importlib.util.spec_from_file_location('probe', PROBE)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
import torch
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as m
original=m.apply_rotary_pos_emb_flashatt
observed={}
q=torch.tensor([[[1.,2.,3.,4.]]],requires_grad=True)
with p.rotary_scope(REPOSITORY,'sourceTorch',observed):
 out,k=m.apply_rotary_pos_emb_flashatt(q,q,torch.ones(1,4),torch.zeros(1,4))
 assert torch.equal(out,q) and torch.equal(k,q)
 torch.autograd.grad(out.sum(),q)
assert m.apply_rotary_pos_emb_flashatt is original
assert observed['calls']==1
assert len(observed['original_callable_sha256'])==len(observed['replacement_callable_sha256'])==64
'''.replace('PROBE', repr(str(ROOT / 'scripts/probe_dil_full_model.py'))).replace('REPOSITORY', repr(str(ROOT)))
    completed = subprocess.run([str(ROOT / "venv/ilvr-venv/bin/python"), "-c", code], cwd=tmp_path,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr


def comparison_fixture(tmp_path):
    import json
    p = probe()
    records = [full_rank_record(2, variant=variant) for variant in ("sourceFlash", "sourceTorch")]
    paths = []
    for record in records:
        for case in record["cases"]:
            for obs in case["observations"]:
                path = tmp_path / "checkpoints" / record["variant"] / f"{case['raw_index']}-{obs['repeat']}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = {"weight": torch.ones(1), "unused": None}
                torch.save(raw, path)
                obs.update(tensor_path=str(path), tensor_sha256=p.sha256(path),
                           gradients={name: None if value is None else p.tensor_record(value) for name, value in raw.items()})
                if obs["repeat"]:
                    obs["repeat_comparison"] = p.compare_gradients(raw, raw)
        path = tmp_path / (record["variant"]+".json")
        paths.append(path)
    args = SimpleNamespace(left=str(paths[0]), right=str(paths[1]), repository=str(tmp_path), output=str(tmp_path / "results/comparison.json"))
    def write():
        for path, record in zip(paths, records):
            path.write_text(json.dumps(record))
    write()
    return p, records, args, write


def test_valid_headers_without_substantive_evidence_cannot_pass(tmp_path):
    p, records, args, write = comparison_fixture(tmp_path)
    for record in records:
        for name in ("loaded_state", "bf16_cast_state", "model_state_unchanged"):
            record.pop(name)
        for case in record["cases"]:
            case.pop("batch_tensors")
            for obs in case["observations"]:
                obs.update(rng={}, gradients={})
                obs["objective"].update(selections=[], support=[], original_step_denominator=None)
    write()
    with pytest.raises(ValueError):
        p.compare_results(args)


@pytest.mark.parametrize("missing", ["rng", "repeat_comparison", "repeat_loss_pass", "repeat_rng_equal", "repeat_objective_comparison"])
def test_rank_validator_requires_rng_and_repeat_evidence(missing):
    p = probe()
    record = full_rank_record(2)
    record["cases"][0]["observations"][1].pop(missing)
    with pytest.raises(ValueError):
        p.validate_rank_record(record, "probe-contract", "a"*64, "portTorch")


@pytest.mark.parametrize("mutation", ["empty", "renamed", "dtype", "shape", "none", "bytes"])
def test_comparison_validates_raw_derivatives_against_complete_witness(tmp_path, mutation):
    p, records, args, write = comparison_fixture(tmp_path)
    obs = records[0]["cases"][0]["observations"][0]
    raw = {"weight": torch.ones(1), "unused": None}
    if mutation == "empty": raw = {}
    if mutation == "renamed": raw = {"other": torch.ones(1), "unused": None}
    if mutation == "dtype": raw["weight"] = torch.ones(1, dtype=torch.float64)
    if mutation == "shape": raw["weight"] = torch.ones(1, 1)
    if mutation == "none": raw["unused"] = torch.ones(1)
    if mutation == "bytes": raw["weight"] = torch.zeros(1)
    torch.save(raw, obs["tensor_path"])
    obs["tensor_sha256"] = p.sha256(obs["tensor_path"])
    write()
    with pytest.raises(ValueError):
        p.compare_results(args)


def test_repeat_components_use_fixed_tolerance_and_exact_discrete_evidence():
    import copy
    p = probe()
    left = full_rank_record(2)["cases"][0]["observations"][0]["objective"]
    right = copy.deepcopy(left)
    right["components"][0]["loss/ce"] += 5e-6
    result = p.compare_repeat_objectives(left, right)
    assert result["pass"] and result["max_abs_error"] == pytest.approx(5e-6)
    assert result["relative_l2"] > 0
    right["components"][0]["loss/ce"] += 1e-3
    assert not p.compare_repeat_objectives(left, right)["pass"]
    right = copy.deepcopy(left)
    right["selections"][0]["masks"][0][0] = False
    assert not p.compare_repeat_objectives(left, right)["discrete_equal"]


def test_complete_comparison_accepts_tolerated_repeat_metric_drift(tmp_path):
    import json
    p, records, args, write = comparison_fixture(tmp_path)
    for record in records:
        for case in record["cases"]:
            before, after = case["observations"]
            after["objective"]["components"][0]["loss/ce"] += 5e-6
            after["repeat_objective_comparison"] = p.compare_repeat_objectives(before["objective"], after["objective"])
    write()
    p.compare_results(args)
    result = json.loads(Path(args.output).read_text())
    assert result["pass"] and result["ranks_covered"] == [2] and not result["global_coverage_complete"]
    repeat = result["cases"][0]["comparisons"][1]["left_repeat"]["objective"]
    assert repeat["max_abs_error"] == pytest.approx(5e-6) and repeat["relative_l2"] > 0


@pytest.mark.parametrize("json_roundtrip", [False, True])
def test_native_logical_rows_validate_before_publication_and_after_json(json_roundtrip):
    import json
    p = probe()
    record = full_rank_record(2)
    case = record["cases"][0]
    case["logical_rows"] = p.logical_rows({
        "sample_ids": [case["sample_id"]], "input_ids": torch.full((1, 31), 7),
        "reasoning_step_spans": [SimpleNamespace(step_id=0, target_text_start=1, target_text_end=31)],
    })
    assert isinstance(case["logical_rows"][0][0], tuple)
    if json_roundtrip:
        record = json.loads(p.json_bytes(record))
        assert isinstance(record["cases"][0]["logical_rows"][0][0], list)
    assert set(p.validate_rank_record(record, "probe-contract", "a"*64, "portTorch")) == {"weight", "unused"}
    # Native tuple acceptance must retain the same sample-identity checks.
    rows = record["cases"][0]["logical_rows"][0]
    rows[0] = ("wrong-sample", *rows[0][1:])
    with pytest.raises(ValueError, match="invalid logical target row identity"):
        p.validate_rank_record(record, "probe-contract", "a"*64, "portTorch")


def test_snapshot_observes_native_batchfeature_without_mutation():
    from dataclasses import asdict
    from collections.abc import Mapping
    from transformers.feature_extraction_utils import BatchFeature
    from step_preference.spans import ReasoningStepSpan
    p = probe()
    span = ReasoningStepSpan(0, 0, 1, 1, 3, 3, 5, None)
    ids = torch.tensor([[1, 2, 3, 7, 7]])
    pixels = torch.tensor([[.25, .5]], requires_grad=True)
    spans = [[span]]
    batch = BatchFeature({"input_ids": ids, "pixel_values": pixels,
                          "sample_ids": [p.CASES[0]["sample_id"]], "reasoning_step_spans": spans,
                          "nested": {"target_rows": ((p.CASES[0]["sample_id"], 0, 2, 7),)}})
    assert isinstance(batch, Mapping) and not isinstance(batch, dict)
    original_data = batch.data
    original_ids, original_pixels = ids.clone(), pixels.detach().clone()
    rng = torch.get_rng_state().clone()
    expected = p.snapshot(batch.data)
    actual = p.snapshot(batch)
    assert actual == expected
    assert actual["reasoning_step_spans"] == [[asdict(span)]]
    assert actual["input_ids"] == p.tensor_record(ids)
    assert actual["pixel_values"] == p.tensor_record(pixels)
    assert batch.data is original_data and batch["input_ids"] is ids and batch["pixel_values"] is pixels
    assert batch["reasoning_step_spans"] is spans and spans[0][0] is span
    assert torch.equal(ids, original_ids) and torch.equal(pixels, original_pixels)
    assert pixels.requires_grad and pixels.grad is None
    assert torch.equal(rng, torch.get_rng_state())


def test_snapshot_still_rejects_unsupported_metadata_objects():
    with pytest.raises(ValueError, match="unsupported collator metadata"):
        probe().snapshot({"unsupported": object()})


def boundary_api():
    p = probe()
    factory = getattr(p, "boundary_api", None)
    assert callable(factory), "optional native boundary observation is missing"
    return factory()


def test_boundary_native_calls_preserve_outputs_cotangents_rng_and_exception_cleanup():
    # Break caught: wrapping a native call replaces its graph, output, gradient or state.
    api = boundary_api()
    layer = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.2))
    original = layer.forward
    x = torch.tensor([[1., 2.]], requires_grad=True)
    state = {k: v.clone() for k, v in layer.state_dict().items()}
    rng = torch.get_rng_state().clone()
    plain = layer(x)
    plain_grad = torch.autograd.grad(plain.sum(), x)[0]
    after_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    observer = api.BoundaryObservation(727, 0)
    returned = []
    def observe(args, kwargs, output, call):
        returned.append(output)
        observer.capture("head.hidden", args[0], call, gradient_kind="tensor_total")
        observer.capture("head.logits", output, call, gradient_kind="tensor_total")
    with observer:
        observer.watch_module(layer, "clean", observe)
        output = layer(x)
        assert output is returned[0]
        grad = torch.autograd.grad(output.sum(), x)[0]
    assert torch.equal(output, plain) and torch.equal(grad, plain_grad)
    assert torch.equal(torch.get_rng_state(), after_rng)
    assert all(torch.equal(state[k], v) for k, v in layer.state_dict().items())
    raw, metadata = observer.raw, observer.record()
    assert torch.equal(raw["clean:0/forward/head.logits/cotangent"], torch.ones(1, 2))
    assert metadata["case"] == 727 and metadata["repeat"] == 0
    assert layer.forward == original and not x._backward_hooks
    failed = api.BoundaryObservation(727, 1)
    with pytest.raises(RuntimeError, match="abort"):
        with failed:
            failed.watch_module(layer, "clean", observe)
            raise RuntimeError("abort")
    assert layer.forward == original


def test_boundary_nonreentrant_replay_keeps_call_identity_and_early_stop_explicit():
    # Break caught: replay observations get a new causal identity or abort leaks wrapper state.
    from torch.utils.checkpoint import checkpoint
    api = boundary_api()
    class Native(torch.nn.Module):
        def forward(self, x):
            return torch.sin(x).cos()
    layer = Native()
    original = layer.forward
    observer = api.BoundaryObservation(2046, 1)
    def observe(args, kwargs, output, call):
        observer.capture("hidden", output, call, gradient_kind="tensor_total")
    with observer:
        observer.watch_module(layer, "prefix:8", observe)
        x = torch.tensor([.3], requires_grad=True)
        y = checkpoint(layer, x, use_reentrant=False)
        observer.phase = "replay"
        torch.autograd.grad(y.sum(), x)
    calls = observer.record()["calls"]
    assert [call["identity"] for call in calls] == ["prefix:8:0", "prefix:8:0"]
    assert [call["phase"] for call in calls] == ["forward", "replay"]
    assert calls[0]["status"] == "returned" and calls[1]["status"] == "checkpoint_early_stop"
    assert layer.forward == original
    assert torch.equal(observer.raw["prefix:8:0/forward/hidden/cotangent"], torch.ones(1))


def test_boundary_comparison_detects_changed_derivative_and_missing_evidence():
    # Break caught: equal forward values hide a changed derivative or a missing tensor.
    api = boundary_api()
    def run(scale):
        observer = api.BoundaryObservation(727, 0)
        class Changed(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()
            @staticmethod
            def backward(ctx, grad):
                return grad * scale
        x = torch.tensor([2.], requires_grad=True)
        with observer:
            call = {"identity": "clean:0", "phase": "forward"}
            observer.capture("head.hidden", x, call, gradient_kind="tensor_total")
            y = Changed.apply(x)
            observer.capture("head.logits", y, call, gradient_kind="tensor_total")
            torch.autograd.grad(y.sum(), x)
        return observer
    left, right = run(1.), run(2.)
    result = api.compare_boundaries(left.record(), left.raw, right.record(), right.raw, probe().compare_gradients)
    assert not result["pass"]
    assert result["earliest_cotangent_frontier"] == ["clean:0/forward/head.hidden"]
    assert result["forward_pass"] and not result["cotangent_pass"]
    missing = dict(right.raw)
    missing.pop("clean:0/forward/head.logits/cotangent")
    with pytest.raises(ValueError, match="boundary"):
        api.compare_boundaries(left.record(), left.raw, right.record(), missing, probe().compare_gradients)
    assert not api.compare_boundaries(left.record(), left.raw, {"records": []}, {}, probe().compare_gradients)["pass"]


def test_boundary_profiler_records_executed_sdpa_dispatch_without_backend_forcing():
    # Break caught: reporting configured 'sdpa' as actual dispatched kernel evidence.
    api = boundary_api()
    x = torch.ones(1, 1, 2, 2, requires_grad=True)
    observer = api.BoundaryObservation(727, 0)
    with observer:
        y = torch.nn.functional.scaled_dot_product_attention(x, x, x)
        torch.autograd.grad(y.sum(), x)
    events = observer.record()["attention_events"]
    assert any("scaled_dot_product" in event["name"] for event in events)
    assert any("backward" in event["name"] for event in events)
    assert observer.record()["observer_effect_disclosure"]


def test_boundary_manifest_opt_in_preserves_unobserved_default(tmp_path):
    # Break caught: optional mode rejected, ambiguous opt-in accepted or ordinary mode changed.
    from tests.multinode.test_dil_full_model_probe import manifest_fixture
    import json
    p = probe()
    repository, path, value = manifest_fixture(tmp_path)
    assert "boundary_observation" not in p.load_manifest(path, repository)
    value["boundary_observation"] = True
    path.write_text(json.dumps(value))
    assert p.load_manifest(path, repository)["boundary_observation"] is True
    value["boundary_observation"] = "yes"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        p.load_manifest(path, repository)


@pytest.mark.parametrize("variant", ["sourceTorch", "portTorch"])
def test_boundary_adapter_observes_real_rollout_clean_cache_and_head_calls(variant):
    # Break caught: adapter misses native source cache or port KV, latent/call/row mapping.
    from functools import partial
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentRolloutEngine, LatentSpan
    api = boundary_api()
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
        def forward(self, *, inputs_embeds, **kwargs):
            hidden = self.linear(inputs_embeds).sin()
            key, value = hidden.unsqueeze(1), (hidden * 2).unsqueeze(1)
            fields = {"last_hidden_state": hidden}
            if kwargs.get("stage2_capture_kv"):
                fields["stage2_key_values"] = ((key, value),)
            if kwargs.get("use_cache"):
                fields["past_key_values"] = SimpleNamespace(key_cache=[key], value_cache=[value])
            return SimpleNamespace(**fields)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()
            self.lm_head = torch.nn.Linear(2, 3)
            self.stage2_rollout_engine = LatentRolloutEngine()
    model = Model()
    batch = {"input_ids": torch.tensor([[1, 1, 1, 2, 2]]), "labels": torch.tensor([[-100, -100, -100, 2, 2]]),
             "sample_ids": ["sample"], "reasoning_step_spans": [SimpleNamespace(step_id=0, target_text_start=3, target_text_end=5)]}
    observer = api.BoundaryObservation(727, 0)
    base = torch.ones(1, 5, 2, requires_grad=True)
    original = model.model.forward
    with observer:
        observer.bind_native(model, batch, variant)
        rollout = model.stage2_rollout_engine.rollout(language_model=partial(model.model, generate_mode=True),
            base_inputs_embeds=base, attention_mask=torch.ones(1, 5), full_position_ids=torch.zeros(3, 1, 5, dtype=torch.long),
            latent_spans=(LatentSpan(0, (1, 2), 3),), latent_activation_checkpointing=True)
        outputs = model.model(inputs_embeds=rollout.final_inputs_embeds,
            stage2_capture_kv=variant == "portTorch", use_cache=variant != "portTorch")
        def _stage2_selected_logits(hidden_states, score_token_mask):
            selected_hidden = hidden_states[score_token_mask].reshape(1, -1, 2)
            return model.lm_head(selected_hidden)
        clean = _stage2_selected_logits(outputs.last_hidden_state, torch.tensor([[False, False, True, True, False]]))
        pair = outputs.stage2_key_values[-1] if variant == "portTorch" else (outputs.past_key_values.key_cache[-1], outputs.past_key_values.value_cache[-1])
        loss = clean.sum() + pair[0].sum() + pair[1].sum()
        observer.phase = "replay"
        torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    names = {record["name"] for record in observer.record()["records"]}
    assert {"decoder.hidden", "clean.key", "clean.value", "latent.0", "latent.1", "head.hidden", "head.logits"} <= names
    head = next(record for record in observer.record()["records"] if record["name"] == "head.logits")
    assert head["rows"] == [["sample", 0, 2, 2], ["sample", 0, 3, 2]]
    assert head["call"] == "clean_target:0" and head["phase"] == "forward"
    assert model.model.forward == original
    assert len([call for call in observer.record()["calls"] if call["phase"] == "replay"]) == 2


def test_boundary_rank_validator_rejects_opt_in_without_trace_and_control():
    # Break caught: complete ordinary rank headers silently pass missing opted-in evidence.
    p = probe()
    record = full_rank_record(2)
    record["boundary_observation"] = True
    with pytest.raises(ValueError, match="boundary"):
        p.validate_rank_record(record, "probe-contract", "a"*64, "portTorch")


def test_boundary_observation_effect_reports_changed_envelope_despite_scalar_agreement():
    # Break caught: equal scalar objectives are mislabeled observer noninterference.
    p = probe()
    compare = getattr(p, "compare_observation_effect", None)
    assert callable(compare), "observation-effect control comparison is missing"
    objective = full_rank_record(2)["cases"][0]["observations"][0]["objective"]
    meta = {"loss": 1., "rng": {"fixed": 1}, "objective": objective}
    controls = [(meta, {"x": torch.tensor([1.])}), (meta, {"x": torch.tensor([1.])})]
    observed = [(meta, {"x": torch.tensor([1.])}), (meta, {"x": torch.tensor([1.001])})]
    result = compare(controls, observed, 727)
    assert result["scalar_agreement"] and result["repeat_envelope_changed"]
    assert result["exact_one_block_gradient_mismatch"]
    assert not result["matched_under_fixed_comparator"]
    assert result["noninterference_proven"] is False


def test_boundary_comparator_marks_fanin_and_missing_replay_incomparable():
    # Break caught: per-suffix view and shared-tensor total get a spurious causal mismatch.
    api = boundary_api()
    observers = []
    for kind, value in (("shared_tensor_total", 1.), ("distinct_view_contribution", 2.)):
        observer = api.BoundaryObservation(2046, 0)
        with observer:
            x = torch.tensor([1.], requires_grad=True)
            observer.capture("suffix.input_key", x, {"identity": "suffix:0", "phase": "forward"}, gradient_kind=kind)
            torch.autograd.grad((x*value).sum(), x)
        observers.append(observer)
    result = api.compare_boundaries(observers[0].record(), observers[0].raw, observers[1].record(), observers[1].raw, probe().compare_gradients)
    assert not result["pass"] and result["unmapped_or_incomparable"]
    assert result["earliest_cotangent_frontier"] == []
    assert result["boundaries"]["suffix:0/forward/suffix.input_key"]["cotangent"] is None


def test_boundary_unobserved_result_marks_boundaries_not_requested(tmp_path):
    # Break caught: an unobserved pair claims an intermediate comparison.
    p, records, args, write = comparison_fixture(tmp_path)
    records[0]["boundary_observation"] = False
    records[1]["boundary_observation"] = False
    write()
    p.compare_results(args)
    import json
    result = json.loads(Path(args.output).read_text())
    assert result.get("boundary_status") == "not_requested"


def test_boundary_comparison_rejects_zero_for_missing_cotangent_and_unmapped_rows():
    # Break caught: a nonexistent derivative gets normalized to an observed zero.
    api = boundary_api()
    left, right = api.BoundaryObservation(727, 0), api.BoundaryObservation(727, 0)
    for observer, used in ((left, False), (right, True)):
        with observer:
            x = torch.tensor([1.], requires_grad=True)
            observer.capture("head.logits", x, {"identity": "clean:0", "phase": "forward"}, rows=[["sample", 0, 1, 2]])
            if used:
                torch.autograd.grad((x*0).sum(), x)
    result = api.compare_boundaries(left.record(), left.raw, right.record(), right.raw, probe().compare_gradients)
    assert not result["pass"] and not result["cotangent_pass"]
    import copy
    unmapped = copy.deepcopy(right.record())
    unmapped["records"][0]["rows"] = None
    result = api.compare_boundaries(left.record(), left.raw, unmapped, right.raw, probe().compare_gradients)
    assert not result["pass"] and result["unmapped_or_incomparable"]


def test_boundary_native_exception_cleans_tensor_hooks_and_owned_forward():
    # Break caught: a native exception leaves tensor hooks or an instance patch active.
    api = boundary_api()
    class Fails(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("native failed")
    layer = Fails()
    original = layer.forward
    layer.forward = original
    x = torch.tensor([1.], requires_grad=True)
    observer = api.BoundaryObservation(727, 0)
    with pytest.raises(RuntimeError, match="native failed"):
        with observer:
            observer.watch_module(layer, "clean", lambda *args: None,
                before=lambda args, kwargs, call: observer.capture("head.hidden", args[0], call))
            layer(x)
    assert layer.forward is original and not x._backward_hooks
    assert observer.record()["calls"][0]["status"] == "raised"


def observed_comparison_fixture(tmp_path):
    import copy
    p, records, args, write = comparison_fixture(tmp_path)
    api = boundary_api()
    for record in records:
        record["boundary_observation"] = True
        for case in record["cases"]:
            case["unobserved_controls"] = copy.deepcopy(case["observations"])
            case["observation_effect"] = {"noninterference_proven": False}
            for obs in case["observations"]:
                observer = api.BoundaryObservation(case["raw_index"], obs["repeat"])
                with observer:
                    x = torch.ones(1, 1, 2, 2, requires_grad=True)
                    loss = torch.nn.functional.scaled_dot_product_attention(x, x, x).sum()
                    call = {"identity": "clean:0", "phase": "forward"}
                    observer.calls.append({**call, "status": "returned"})
                    for name in ("decoder.hidden", "clean.key", "clean.value"):
                        observer.capture(name, x, call)
                    suffix = {"identity": "suffix:0", "phase": "forward"}
                    observer.calls.append({**suffix, "status": "returned"})
                    observer.capture("decoder.hidden", x, suffix)
                    for index in range(len(case["logical_rows"])*8):
                        position = index + 1
                        latent_rows = [[case["sample_id"], "latent_position", position]]
                        observer.capture(f"latent.{index}", x, {"identity": "rollout:0", "phase": "forward"}, rows=latent_rows)
                        prefix = {"identity": f"prefix:{position}:0", "phase": "forward"}
                        observer.calls.append({**prefix, "status": "returned"})
                        observer.capture("prefix.hidden_tail", x, prefix, rows=latent_rows)
                    head_leaves = []
                    for role in ("clean_ce", "clean_target", "off_target"):
                        row_ids = [row for group in case["logical_rows"] for row in group]
                        y = torch.ones(len(row_ids), 2, requires_grad=True)
                        head_leaves.append(y)
                        head = {"identity": role+":0", "phase": "forward"}
                        observer.calls.append({**head, "status": "returned"})
                        for name in ("head.hidden", "head.logits"):
                            observer.capture(name, y, head, rows=row_ids)
                        loss = loss + y.sum()
                    torch.autograd.grad(loss, [x, *head_leaves], allow_unused=True)
                path = tmp_path / "checkpoints" / record["variant"] / f"boundary-{case['raw_index']}-{obs['repeat']}.pt"
                torch.save(observer.raw, path)
                obs["boundaries"] = {**observer.record(), "tensor_path": str(path), "tensor_sha256": p.sha256(path)}
    write()
    return p, records, args, write


def test_boundary_observed_results_validate_raw_snapshots_controls_and_repeat_comparisons(tmp_path):
    # Critical CPU integration: opt-in records through actual file loader/comparator.
    import json
    p, records, args, write = observed_comparison_fixture(tmp_path)
    p.compare_results(args)
    result = json.loads(Path(args.output).read_text())
    assert result["pass"] and result["boundary_status"] == "observed_partial_mapping"
    repeat = result["cases"][1]["comparisons"][1]
    assert repeat["boundaries"]["pass"] and repeat["left_boundary_repeat"]["pass"]
    assert result["cases"][0]["observation_effect"][0]["noninterference_proven"] is False
    trace = records[0]["cases"][0]["observations"][0]["boundaries"]
    raw = torch.load(trace["tensor_path"], weights_only=True)
    raw.pop(next(iter(raw)))
    torch.save(raw, trace["tensor_path"])
    trace["tensor_sha256"] = p.sha256(trace["tensor_path"])
    write()
    with pytest.raises(ValueError, match="boundary"):
        p.compare_results(args)


@pytest.mark.parametrize("missing", ["records", "calls", "attention_events"])
def test_boundary_observed_rank_rejects_missing_trace_evidence(tmp_path, missing):
    p, records, args, write = observed_comparison_fixture(tmp_path)
    records[0]["cases"][0]["observations"][0]["boundaries"].pop(missing)
    with pytest.raises(ValueError, match="boundary|attention"):
        p.validate_rank_record(records[0], "probe-contract", "a"*64, "sourceFlash")


def test_boundary_four_rank_terminal_requires_manifest_requested_observation():
    # Break caught: an opted-in coordinator accepts ordinary unobserved rank files.
    p = probe()
    with pytest.raises(ValueError, match="boundary"):
        p.validate_rank_records([full_rank_record(rank) for rank in range(4)], "probe-contract", "a"*64,
                                "portTorch", boundary_observation=True)


def test_boundary_shape_mismatch_is_incomparable_instead_of_numeric_frontier():
    # Break caught: an unmappable shape is reported as a measured numerical mismatch.
    api = boundary_api()
    observers = []
    for size in (1, 2):
        observer = api.BoundaryObservation(727, 0)
        with observer:
            x = torch.ones(size, requires_grad=True)
            observer.capture("head.hidden", x, {"identity": "clean:0", "phase": "forward"})
            torch.autograd.grad(x.sum(), x)
        observers.append(observer)
    result = api.compare_boundaries(observers[0].record(), observers[0].raw, observers[1].record(), observers[1].raw, probe().compare_gradients)
    assert not result["pass"] and result["unmapped_or_incomparable"]
    assert result["earliest_forward_frontier"] == result["earliest_cotangent_frontier"] == []


@pytest.mark.parametrize("removed", [
    "clean_ce:0/forward/head.hidden", "clean_ce:0/forward/head.logits",
    "clean_target:0/forward/head.hidden", "clean_target:0/forward/head.logits",
    "off_target:0/forward/head.hidden", "off_target:0/forward/head.logits",
    "clean:0/forward/decoder.hidden", "clean:0/forward/clean.key", "clean:0/forward/clean.value",
    "rollout:0/forward/latent.0", "prefix:1:0/forward/prefix.hidden_tail",
])
def test_boundary_bilateral_required_record_removal_rejected_before_comparison(tmp_path, removed):
    # Break caught: aggregate name/symmetric-difference checks hide missing required evidence.
    p, records, args, write = observed_comparison_fixture(tmp_path)
    api = boundary_api()
    traces = []
    for record in records:
        trace = record["cases"][0]["observations"][0]["boundaries"]
        raw = torch.load(trace["tensor_path"], map_location="cpu", weights_only=True)
        trace["records"] = [item for item in trace["records"] if item["key"] != removed]
        for suffix in ("/forward", "/cotangent"):
            raw.pop(removed + suffix)
        # Raw dictionaries still match the reduced metadata; semantic completeness
        # must be enforced by the required native trace gate.
        api.validate_raw(trace, raw)
        torch.save(raw, trace["tensor_path"])
        trace["tensor_sha256"] = p.sha256(trace["tensor_path"])
        traces.append(trace)
    write()
    for trace in traces:
        with pytest.raises(ValueError, match="boundary"):
            api.validate_trace(trace, p.CASES[0], 0)
    with pytest.raises(ValueError, match="boundary"):
        p.compare_results(args)


@pytest.mark.parametrize("mutation", ["head_call", "clean_call", "prefix_call", "record_identity", "prefix_position", "rollout_call"])
def test_boundary_required_records_link_to_native_calls_and_rollout_positions(tmp_path, mutation):
    # Break caught: correct tensor names attached to nonexistent/wrong calls pass the trace gate.
    p, records, args, write = observed_comparison_fixture(tmp_path)
    trace = records[0]["cases"][0]["observations"][0]["boundaries"]
    if mutation.endswith("_call") and mutation != "rollout_call":
        identity = {"head_call": "off_target:0", "clean_call": "clean:0", "prefix_call": "prefix:1:0"}[mutation]
        trace["calls"] = [item for item in trace["calls"] if item["identity"] != identity]
    else:
        item = next(item for item in trace["records"] if item["key"] == "rollout:0/forward/latent.0")
        if mutation == "record_identity":
            item["name"] = "latent.99"
        elif mutation == "prefix_position":
            item["rows"][0][2] = 999
        else:
            item["call"] = "other_rollout:0"
            item["key"] = "other_rollout:0/forward/latent.0"
    with pytest.raises(ValueError, match="boundary"):
        p.boundary_api().validate_trace(trace, p.CASES[0], 0)


def test_boundary_positive_fixture_delivers_every_head_cotangent(tmp_path):
    # Break caught: autograd prunes clean head leaves from the promised positive witness.
    p, records, args, write = observed_comparison_fixture(tmp_path)
    for record in records:
        for case in record["cases"]:
            for obs in case["observations"]:
                heads = [item for item in obs["boundaries"]["records"] if item["name"] in ("head.hidden", "head.logits")]
                assert len(heads) == 6
                assert all(item["cotangent_status"] == "delivered" and item["cotangent"] is not None for item in heads)


@pytest.mark.parametrize("missing", [
    "clean_ce:0/forward/head.hidden", "clean_ce:0/forward/head.logits",
    "clean_target:0/forward/head.hidden", "clean_target:0/forward/head.logits",
    "off_target:0/forward/head.hidden", "off_target:0/forward/head.logits",
    "clean:0/forward/clean.key", "clean:0/forward/clean.value", "rollout:0/forward/latent.0", "all",
])
def test_boundary_bilateral_absent_cotangent_is_unavailable_at_native_comparison_entrypoint(tmp_path, missing):
    # Break caught: None/None is called observed derivative equality after semantic validation.
    import json
    p, records, args, write = observed_comparison_fixture(tmp_path)
    api = boundary_api()
    for record in records:
        trace = record["cases"][0]["observations"][0]["boundaries"]
        raw = torch.load(trace["tensor_path"], map_location="cpu", weights_only=True)
        for item in trace["records"]:
            if item["key"] == missing or missing == "all":
                item.update(cotangent=None, cotangent_status="not_delivered", gradient_callbacks=0)
                raw[item["key"] + "/cotangent"] = None
        # Absence can be a legitimate native unused/replay observation. Keep
        # inspection usable; it must never become measured derivative agreement.
        api.validate_trace(trace, p.CASES[0], 0)
        api.validate_raw(trace, raw)
        torch.save(raw, trace["tensor_path"])
        trace["tensor_sha256"] = p.sha256(trace["tensor_path"])
    write()
    p.compare_results(args)
    result = json.loads(Path(args.output).read_text())["cases"][0]["comparisons"][0]["boundaries"]
    assert not result["pass"] and not result["cotangent_pass"]
    assert result["unmapped_or_incomparable"]
    assert result["earliest_cotangent_frontier"] == []
    key = missing if missing != "all" else "clean_ce:0/forward/head.hidden"
    assert result["boundaries"][key]["cotangent"] is None
    assert result["boundaries"][key]["cotangent_comparable"] is False


def test_boundary_delivered_zero_is_comparable_but_unused_replay_is_unavailable():
    # Break caught: an observed zero is treated as absence or unused replay is claimed equal.
    api = boundary_api()
    observers = []
    for repeat in (0, 1):
        observer = api.BoundaryObservation(727, repeat)
        with observer:
            x = torch.ones(1, requires_grad=True)
            observer.capture("head.hidden", x, {"identity": "clean_ce:0", "phase": "forward"})
            torch.autograd.grad((x*0).sum(), x)
        observers.append(observer)
    result = api.compare_boundaries(observers[0].record(), observers[0].raw, observers[1].record(), observers[1].raw, probe().compare_gradients)
    assert result["pass"] and result["cotangent_pass"]
    assert result["boundaries"]["clean_ce:0/forward/head.hidden"]["cotangent_comparable"]
    for observer in observers:
        with observer:
            unused = torch.ones(1, requires_grad=True)
            observer.capture("head.hidden", unused, {"identity": "off_target:0", "phase": "replay"})
    result = api.compare_boundaries(observers[0].record(), observers[0].raw, observers[1].record(), observers[1].raw, probe().compare_gradients)
    assert not result["pass"] and not result["cotangent_pass"]
    assert result["boundaries"]["off_target:0/replay/head.hidden"]["cotangent"] is None
    assert result["earliest_cotangent_frontier"] == []
