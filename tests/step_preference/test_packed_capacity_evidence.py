"""CPU witnesses for the strict distributed acceptance and runtime schemas."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import importlib.util
import json
from types import SimpleNamespace

import pytest


def test_local_observation_failure_is_gathered_before_collective_rejection(monkeypatch):
    m=api()
    assert hasattr(m,"gather_checked"),"missing bounded collective error observation"
    records=[]
    def gather(local):
        records.append(local)
        return [local,{"value":{},"error":None}]
    monkeypatch.setattr(m,"gather",gather)
    def failed():raise ValueError("rank-local GPU mismatch")
    with pytest.raises(ValueError,match="rank-local GPU mismatch"):m.gather_checked(failed)
    assert records == [{"value":None,"error":"ValueError: rank-local GPU mismatch"}]


@pytest.mark.parametrize("rank",[0,1])
def test_collective_publication_is_rank_zero_then_gather_then_barrier(monkeypatch,tmp_path,rank):
    m=api()
    assert hasattr(m,"publish_collective"),"missing rank-zero publication boundary"
    import torch.distributed as dist
    events=[]
    monkeypatch.setattr(dist,"get_rank",lambda:rank)
    monkeypatch.setattr(dist,"is_initialized",lambda:True)
    monkeypatch.setattr(dist,"get_world_size",lambda:2)
    monkeypatch.setattr(dist,"barrier",lambda:events.append("barrier"))
    monkeypatch.setattr(m,"publish_json",lambda *a:events.append("publish"))
    def all_gather(values,local):
        events.append("gather");values[:]=[local,local]
    monkeypatch.setattr(dist,"all_gather_object",all_gather)
    m.publish_collective(tmp_path/"runtime.json",{},dict)
    assert events == (["publish"] if rank == 0 else [])+["gather","barrier"]


def api():
    name = "step_preference.capacity_evidence"
    assert importlib.util.find_spec(name) is not None, "capacity evidence implementation missing"
    return importlib.import_module(name)


def snapshot():
    return {"schema_version": 1, "kind": "stage2b-packed-gpu-snapshot", "gpus": [
        {"host_physical_index": r, "visible_cuda_index": r, "name": "NVIDIA H100 NVL",
         "uuid": f"01234567-0123-4567-89ab-0123456789a{r}", "total_memory_bytes": 10000}
        for r in range(2)]}


def rank_records():
    return [{"rank": r, "sample_ids": ["sample-a"] * 4,
        "pre_step_gradient_norms": [1.25, 2.5], "training_loss": 0.5,
        "peak_reserved_bytes": 8500, "total_memory_bytes": 10000,
        "peak_reserved_fraction": 0.85} for r in range(2)]


def evidence_payload(tmp_path):
    root = str(tmp_path / "step-pref-packed-admission-test-zero2-7")
    return dict(schema_version=1, kind="stage2b-packed-distributed-capacity",
        run_id="step-pref-packed-admission-test-zero2-7", world_size=2, process_ranks=[0, 1],
        global_step=2, optimizer_step=2, gradient_accumulation_steps=2, microbatch_count=4,
        sample_id="sample-a", rank_observations=rank_records(), gpu_snapshot_path=str(tmp_path / "gpu.json"),
        gpu_snapshot_sha256="a" * 64, gpu_identities=[dict(rank=r, physical_index=r,
            name="NVIDIA H100 NVL", uuid=g["uuid"], total_memory_bytes=10000)
            for r, g in enumerate(snapshot()["gpus"])], checkpoint_path=root,
        training_runtime_identity_path=root + "/stage2_training_runtime_identity.json",
        training_runtime_identity_sha256="b" * 64, method="dora", zero_stage=2,
        finite_loss=True, finite_gradient=True, nonzero_gradient=True,
        memory_within_limit=True, worker_errors=[])


def test_distributed_schema_accepts_exact_two_rank_boundary(tmp_path):
    m = api()
    payload = evidence_payload(tmp_path)
    assert asdict(m.decode(m.PackedDistributedCapacityEvidence, payload)) == payload


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(world_size=4), lambda p: p.update(global_step=1),
    lambda p: p.update(optimizer_step=True), lambda p: p.update(gradient_accumulation_steps=1),
    lambda p: p.update(microbatch_count=2), lambda p: p.update(zero_stage=3),
    lambda p: p.update(worker_errors=["failed"]), lambda p: p.update(finite_loss=False),
    lambda p: p["rank_observations"][1]["sample_ids"].pop(),
    lambda p: p["rank_observations"][1]["sample_ids"].__setitem__(0, "other"),
    lambda p: p["rank_observations"][0]["pre_step_gradient_norms"].__setitem__(0, 0.0),
    lambda p: p["rank_observations"][0].update(training_loss=float("nan")),
    lambda p: p["rank_observations"][1].update(peak_reserved_bytes=8501, peak_reserved_fraction=0.8501),
    lambda p: p["gpu_identities"][1].update(physical_index=0),
    lambda p: p["gpu_identities"][1].update(name="NVIDIA B200"),
    lambda p: p.update(extra=1),
])
def test_distributed_schema_rejects_one_bad_rank_or_claim(tmp_path, mutate):
    m = api(); payload = evidence_payload(tmp_path); mutate(payload)
    with pytest.raises(ValueError): m.decode(m.PackedDistributedCapacityEvidence, payload)


@pytest.mark.parametrize("mutation", ["uuid", "total_memory_bytes", "name", "visible_cuda_index"])
def test_live_snapshot_match_rejects_each_changed_identity(mutation):
    m = api(); actual = snapshot(); changed = deepcopy(actual)
    changed["gpus"][1][mutation] = "wrong" if mutation in ("uuid", "name") else 0
    with pytest.raises(ValueError): m.require_matching_snapshot(actual, changed)


def test_strict_reader_rejects_noncanonical_duplicate_symlink_and_mutation(tmp_path):
    m = api(); path = tmp_path / "gpu.json"
    digest = m.publish_json(path, snapshot(), m.PackedGpuSnapshot)
    assert m.read_json(path, m.PackedGpuSnapshot, digest).gpus[1].host_physical_index == 1
    with pytest.raises(ValueError): m.publish_json(path, snapshot(), m.PackedGpuSnapshot)
    alias = tmp_path / "alias.json"; alias.symlink_to(path)
    with pytest.raises(ValueError): m.read_json(alias, m.PackedGpuSnapshot, digest)
    path.write_text(json.dumps(snapshot()))
    with pytest.raises(ValueError): m.read_json(path, m.PackedGpuSnapshot, digest)
    path.write_text('{"schema_version":1,"schema_version":1}\n')
    with pytest.raises(ValueError): m.read_json(path, m.PackedGpuSnapshot)


def test_private_selector_requires_authenticated_unique_retained_sample():
    m = api(); samples = (SimpleNamespace(sample_id="a"), SimpleNamespace(sample_id="b"))
    admission = SimpleNamespace(retained_sample_ids=("a", "b"))
    assert m.select_capacity_sample(samples, admission, "b") == (samples[1],)
    for selected, manifest, sample in [(samples, None, "a"), (samples, admission, "missing"),
        (samples + samples, admission, "a"), (samples, admission, "")]:
        with pytest.raises(ValueError): m.select_capacity_sample(selected, manifest, sample)


def test_capacity_collator_records_returned_authenticated_batch_only():
    m = api(); recorder = m.CapacitySampleRecorder(lambda items: {"sample_ids": ("a",)}, "a")
    recorder([object()]); recorder([object()])
    assert recorder.sample_ids == ["a", "a"]
    bad = m.CapacitySampleRecorder(lambda items: {"sample_ids": ("b",)}, "a")
    with pytest.raises(ValueError): bad([object()])
    assert bad.sample_ids == []


def test_capacity_options_validate_canonical_output_and_snapshot_before_runtime(tmp_path,monkeypatch):
    m=api();import artifact_paths
    results=tmp_path/"results";results.mkdir()
    monkeypatch.setattr(artifact_paths,"CANONICAL_RESULT_ROOT",results)
    path=tmp_path/"gpu.json";path.write_bytes(m.canonical(snapshot()))
    args=SimpleNamespace(step_packed_capacity_evidence_path=str(results/"evidence.json"),step_packed_capacity_sample_id="a",
        step_packed_gpu_snapshot=str(path),step_packed_gpu_snapshot_sha256=m.digest(path.read_bytes()),
        step_preference_config=SimpleNamespace(is_stage2b=True,packing_enabled=True),max_steps=2,
        per_device_train_batch_size=1,gradient_accumulation_steps=2,dataloader_num_workers=0,tuning_method="dora",gradient_checkpointing=True)
    gpu=m.GpuIdentity(0,"NVIDIA H100 NVL",snapshot()["gpus"][0]["uuid"],10000)
    monkeypatch.setattr(m,"observe_gpu",lambda rank:gpu)
    assert m.validate_capacity_options(args,world_size=2,zero_stage=2)
    for world,zero in [(1,2),(4,2),(2,3)]:
        with pytest.raises(ValueError):m.validate_capacity_options(args,world_size=world,zero_stage=zero)
    args.step_packed_capacity_evidence_path=str(tmp_path/"outside.json")
    with pytest.raises(ValueError,match="canonical result"):m.validate_capacity_options(args,world_size=2,zero_stage=2)


def runtime_payload(tmp_path):
    return dict(schema_version=1,kind="stage2-training-runtime-identity",run_id="step-pref-dora-ce-test",
        git_commit="a"*40,tracked_worktree_clean=True,coefficient_triplet=[1.0,0.0,0.0],
        tuning_method="dora",seed=42,bf16=True,per_device_train_batch_size=1,
        gradient_accumulation_steps=2,gradient_checkpointing=True,use_reentrant=False,
        global_step=2,optimizer_step=2,objective_metric_name="loss/ce",objective_metric_observation_count=4,
        finite_objective_metrics=True,positive_pre_step_gradient_count=2,nonfinite_pre_step_gradient_count=0,
        first_positive_pre_step_gradient_norms=[1.25,2.5],training_length_manifest_path=str(tmp_path / "length.json"),
        training_length_manifest_sha256="a"*64,admission_manifest_sha256="b"*64,verification_sha256="c"*64,
        world_size=2,process_ranks=[0,1],rank_observations=[dict(rank=r,
            language_attention_class="Qwen2_5_VLSdpaAttention",language_attention_layer_count=2,
            vision_attention_class="Qwen2_5_VLVisionFlashAttention2",vision_attention_block_count=2,
            packed_attention_backend="flex_attention",sdpa_forward_counts=[4,4],
            vision_flash_attention_forward_counts=[4,4],flex_attention_forward_counts=[4,4]) for r in range(2)])


@pytest.mark.parametrize("triplet,metric", [([1,0,0],"loss/ce"),([0,1,0],"loss/dpo"),([0,0,1],"loss/entropy"),([1,1,.1],"loss/total")])
def test_training_runtime_schema_selects_actual_objective(tmp_path,triplet,metric):
    m=api(); payload=runtime_payload(tmp_path); payload.update(coefficient_triplet=triplet,objective_metric_name=metric)
    assert m.decode(m.TrainingRuntimeIdentity,payload).objective_metric_name == metric


@pytest.mark.parametrize("mutate", [
    lambda p:p.update(seed=41),lambda p:p.update(bf16=False),lambda p:p.update(use_reentrant=True),
    lambda p:p.update(global_step=0),lambda p:p.update(optimizer_step=1),
    lambda p:p.update(objective_metric_name="loss/dpo"),lambda p:p.update(objective_metric_observation_count=0),
    lambda p:p.update(positive_pre_step_gradient_count=0),lambda p:p.update(nonfinite_pre_step_gradient_count=1),
    lambda p:p.update(training_length_manifest_sha256="0"*64),
    lambda p:p["rank_observations"][1]["flex_attention_forward_counts"].__setitem__(1,0),
    lambda p:p["rank_observations"][0].update(vision_attention_class="fake"),
])
def test_training_runtime_rejects_one_unobserved_claim(tmp_path, mutate):
    m=api(); payload=runtime_payload(tmp_path); mutate(payload)
    with pytest.raises(ValueError):m.decode(m.TrainingRuntimeIdentity,payload)


def test_adapter_publication_reopens_all_three_files_and_rejects_extra_model_state(tmp_path):
    m=api()
    from tests.stage2.test_peft_checkpointing import sample_metadata
    metadata=sample_metadata().to_dict()
    metadata.update(lora_r=32,lora_alpha=64,lora_dropout=.05)
    from stage2.checkpointing import Stage2PeftMetadata,save_stage2_peft_metadata
    save_stage2_peft_metadata(tmp_path,Stage2PeftMetadata.from_dict(metadata))
    (tmp_path/"adapter_config.json").write_bytes(m.canonical(dict(use_dora=True,r=32,lora_alpha=64,lora_dropout=.05)))
    (tmp_path/"adapter_model.safetensors").write_bytes(b"stub safetensors")
    (tmp_path/"training_args.bin").write_bytes(b"ordinary Trainer configuration")
    files=m.adapter_files(tmp_path)
    assert [f["relative_path"] for f in files] == ["adapter_config.json","adapter_model.safetensors","stage2_peft_metadata.json"]
    (tmp_path/"optimizer.pt").write_bytes(b"unaccepted state")
    with pytest.raises(ValueError):m.adapter_files(tmp_path)
