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
    state = {"weight": {"shape": [1], "dtype": "torch.float32", "sha256": "b"*64, "trainable": True}}
    observations = [{"repeat": repeat, "loss": 1., "gradients": {"weight": {key: value for key, value in state["weight"].items() if key != "trainable"}},
                     "tensor_path": str(ROOT / f"checkpoints/probe-contract/diagnostic-tensors/rank-{rank}/case-{index}-repeat-{repeat}.pt"),
                     "tensor_sha256": "b"*64, "objective": {"components": [{"loss/total": 1.}], "selections": [{"rows": [["sample", 0, 1, 7]]}],
                                                                       "support": [{"phase": "forward", "rows": [["sample", 0, 1, 7]], "ids": [[1, 2]]}]}}
                    for index in (727, 2046) for repeat in (0, 1)]
    return {"kind": "dil-full-model-probe-rank", "schema": 1, "run_id": "probe-contract",
            "manifest_sha256": digest, "variant": variant, "rank": rank, "world_size": 4,
            "machine_rank": rank//2, "local_rank": rank%2, "device": rank%2, "host": "remote" if rank < 2 else "local",
            "status": "complete", "model_state_unchanged": True, "deepspeed_engine_prepared": False,
            "loaded_state": state, "bf16_cast_state": state,
            "cases": [{"raw_index": index, "repeats": [0, 1], "observations": observations[ordinal*2:ordinal*2+2],
                       "batch_tensors": {key: state["weight"] for key in ("input_ids", "labels", "attention_mask", "pixel_values", "image_grid_thw")}}
                      for ordinal, index in enumerate((727, 2046))]}


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
