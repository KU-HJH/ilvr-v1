"""CPU witnesses for the strict distributed acceptance and runtime schemas."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import importlib.util
import json
from types import SimpleNamespace

import pytest
from tests.step_preference.test_packed_admission_cli import inputs,build,verify,AdmissionProcessor


def test_ignore_preflight_checks_checkpoint_symlink_leaf_and_real_result_descendants(tmp_path,monkeypatch):
    import subprocess
    root=tmp_path/"repository";root.mkdir()
    storage=tmp_path/"checkpoint-storage";storage.mkdir()
    (root/"checkpoints").symlink_to(storage,target_is_directory=True)
    for name in ("results","logs"):(root/name).mkdir()
    subprocess.run(["git","init","-q",str(root)],check=True)
    excludes=root/"empty-excludes";excludes.write_text("")
    subprocess.run(["git","config","core.excludesFile",str(excludes)],cwd=root,check=True)
    rules={"results":"/results/*\n","logs":"/logs/*\n","checkpoints":"/checkpoints\n"}
    ignore=root/".gitignore";ignore.write_text("".join(rules.values()))
    calls=[];run=subprocess.run
    def observe(argv,**kwargs):
        calls.append((argv,kwargs["cwd"]))
        return run(argv,**kwargs)
    monkeypatch.setattr(subprocess,"run",observe)
    api().require_ignored_roots(root)
    assert calls == [(["git","check-ignore","-q","--no-index",path],root) for path in
        ("results/.codex-ignore-probe","logs/.codex-ignore-probe","checkpoints")]
    for missing in rules:
        ignore.write_text("".join(rule for name,rule in rules.items() if name != missing))
        with pytest.raises(ValueError,match=missing):api().require_ignored_roots(root)
    assert (root/"checkpoints").is_symlink()
    assert not list(storage.iterdir())
    assert all(not list((root/name).iterdir()) for name in ("results","logs"))


@pytest.fixture
def runtime_inputs(inputs,tmp_path):
    from pathlib import Path
    from stage2.length_manifest import build_stage2_length_manifest,write_stage2_length_manifest
    m=api();args=inputs();manifest=build(args)
    assert verify(args)[0] == 0
    length=tmp_path/"training-length.json"
    value=build_stage2_length_manifest(data_path=args.retained_output_path,image_root=args.image_root,
        processor=AdmissionProcessor(),processor_source=str(args.checkpoint_path),latent_size=8,
        supervise_latent_boundaries=True,max_sequence_length=8192,assistant_plan_rendering="adjacent-text-space-lf-v1")
    write_stage2_length_manifest(length,value)
    return SimpleNamespace(admission_args=args,manifest=manifest,length=length,
        admission_sha=m.digest(Path(args.manifest_output_path).read_bytes()),
        verification_sha=m.digest((tmp_path/"verification.json").read_bytes()))


@pytest.mark.parametrize("format",["indented","reordered","duplicate","nonfinite","unknown","schema"])
def test_training_length_manifest_requires_semantic_canonical_bytes(runtime_inputs,format):
    m=api();path=runtime_inputs.length;payload=json.loads(path.read_bytes())
    if format == "indented":raw=(json.dumps(payload,sort_keys=True,indent=2)+"\n").encode()
    elif format == "reordered":raw=(json.dumps(dict(reversed(list(payload.items()))),separators=(",",":"))+"\n").encode()
    elif format == "duplicate":raw=m.canonical(payload).replace(b'"schema_version":2',b'"schema_version":2,"schema_version":2')
    else:
        if format == "nonfinite":payload["stats"]["median"]=float("nan")
        if format == "unknown":payload["extra"]=1
        if format == "schema":payload["schema_version"]=1
        raw=(json.dumps(payload,sort_keys=True,separators=(",",":"))+"\n").encode()
    path.write_bytes(raw)
    # The supplied hash authenticates these exact bad bytes, not the original file.
    with pytest.raises(ValueError):
        m.training_length_manifest(path,m.digest(raw),runtime_inputs.manifest.identity)


def test_training_manifest_canonicalization_matches_named_writer_numeric_keys(runtime_inputs,tmp_path):
    from dataclasses import replace
    from stage2.data import summarize_stage2_lengths
    from stage2.length_manifest import read_stage2_length_manifest,write_stage2_length_manifest
    m=api();original=read_stage2_length_manifest(runtime_inputs.length)
    counts=(2,10)
    value=replace(original,latent_block_counts=counts,stats=summarize_stage2_lengths(
        lengths=original.lengths,latent_block_counts=counts,max_length=8192))
    path=write_stage2_length_manifest(tmp_path/"numeric-keys.json",value)
    assert b'"latent_block_distribution":{"2":1,"10":1}' in path.read_bytes()
    assert m.training_length_manifest(path,m.digest(path.read_bytes()),runtime_inputs.manifest.identity) == value


@pytest.fixture
def producing_ranks(runtime_inputs,tmp_path,monkeypatch):
    """Two real producer calls; only hardware/collective transport and policy kernels are doubled."""
    import threading
    from dataclasses import replace
    from concurrent.futures import ThreadPoolExecutor
    import torch
    import torch.distributed as dist
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen
    from step_preference.config import StepPreferenceConfig
    from stage2.trainer import _Stage2DeepSpeedGradientAdapter
    from tests.step_preference.test_stage2b_trainer import _trainer,_Policy,_batch
    from tests.stage2.test_attention_backend import _load_tiny_mixed_model
    from tests.step_preference.test_packed_capacity_gate import write_adapter

    m=api();local=threading.local();barrier=threading.Barrier(2,timeout=15)
    records={};events=[]
    def gather(output,value):
        index=getattr(local,"gather_index",0);local.gather_index=index+1
        records.setdefault(index,{})[local.rank]=deepcopy(value)
        barrier.wait()
        output[:]=[deepcopy(records[index][rank]) for rank in (0,1)]
        events.append((local.rank,"gather",index));barrier.wait()
    def synchronize():
        events.append((local.rank,"barrier"));barrier.wait()
    monkeypatch.setattr(dist,"get_rank",lambda:local.rank)
    monkeypatch.setattr(dist,"is_initialized",lambda:True)
    monkeypatch.setattr(dist,"get_world_size",lambda:2)
    monkeypatch.setattr(dist,"all_gather_object",gather)
    monkeypatch.setattr(dist,"barrier",synchronize)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda device:None)
    monkeypatch.setattr(torch.nn.functional,"scaled_dot_product_attention",lambda q,k,v,**kw:q)
    monkeypatch.setattr(qwen,"flex_attention_forward",lambda owner,q,k,v,mask,**kw:(q.transpose(1,2),None))
    monkeypatch.setattr(qwen,"flash_attn_varlen_func",lambda q,k,v,*a:q)
    monkeypatch.setattr("step_preference.packing.build_flex_block_mask",lambda *a,**k:object())
    gpu_path=tmp_path/"producer-gpu.json"
    gpu_sha=m.publish_json(gpu_path,snapshot(),m.PackedGpuSnapshot)
    live_gpus=[m.GpuIdentity(rank,"NVIDIA H100 NVL",gpu["uuid"],10000)
        for rank,gpu in enumerate(snapshot()["gpus"])]
    monkeypatch.setattr(m,"observe_gpu",lambda device:live_gpus[local.rank])

    def prepare(triplet,*,zero_gradient=False):
        arm={tuple([1,0,0]):"ce",tuple([0,1,0]):"dpo",tuple([0,0,1]):"entropy"}.get(tuple(triplet))
        identifier=f"step-pref-dora-{arm}-producer" if arm else "step-pref-packed-admission-producer-zero2-0"
        checkpoint=tmp_path/"checkpoints"/identifier;write_adapter(checkpoint)
        source=runtime_inputs.admission_args
        config=StepPreferenceConfig(enabled=True,mode="stage2b",packing_enabled=True,max_packed_sequence_length=4608,
            lambda_ce=triplet[0],lambda_dpo=triplet[1],lambda_entropy=triplet[2],
            stage2a_checkpoint_path=str(source.checkpoint_path),negative_path=str(source.negative_path),
            negative_manifest=str(source.negative_manifest),packed_admission_manifest=str(source.manifest_output_path),
            packed_admission_manifest_sha256=runtime_inputs.admission_sha,
            packed_admission_verification=str(tmp_path/"verification.json"),
            packed_admission_verification_sha256=runtime_inputs.verification_sha)
        args=SimpleNamespace(save_model_path=str(checkpoint),tuning_method="dora",
            stage2_length_manifest=str(runtime_inputs.length),stage2_length_manifest_sha256=m.digest(runtime_inputs.length.read_bytes()),
            step_packed_gpu_snapshot=str(gpu_path),step_packed_gpu_snapshot_sha256=gpu_sha,
            step_packed_capacity_sample_id="sample-0",step_packed_capacity_evidence_path=str(tmp_path/"distributed.json"))
        ranks=[]
        for rank in (0,1):
            trainer=_trainer(config);model,*_=_load_tiny_mixed_model();trainer.model=model
            trainer.args=SimpleNamespace(seed=42,bf16=True,per_device_train_batch_size=1,gradient_accumulation_steps=2,
                gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False})
            trainer.state=SimpleNamespace(global_step=0)
            trainer.deepspeed_runtime_profile=SimpleNamespace(zero_stage=2)
            trainer.accelerator=SimpleNamespace(device=torch.device("cpu"),unwrap_model=lambda value:value)
            policy=_Policy()
            if zero_gradient:policy.weight.data.fill_(.25)
            optimizer=torch.optim.SGD(policy.parameters(),lr=.01)
            boundary=[False];norms_at_step=[]
            zero_optimizer=SimpleNamespace(partition_gradients=True,loss_scale=1.,
                scaled_global_norm=lambda:policy.weight.grad.detach().norm())
            engine=SimpleNamespace(optimizer=zero_optimizer,is_gradient_accumulation_boundary=lambda:boundary[0],skipped_steps=0)
            def step():
                if boundary[0]:
                    norms_at_step.append(float(policy.weight.grad.norm()))
                    optimizer.step();optimizer.zero_grad()
            wrapped=_Stage2DeepSpeedGradientAdapter(trainer,engine,step)
            batch=_batch();batch["sample_ids"]=("sample-0",)
            for item in batch["step_preference"]["steps"]:
                item["negative"]=replace(item["negative"],sample_id="sample-0")
            recorder=m.CapacitySampleRecorder(lambda items:batch,"sample-0")
            losses=[]
            for microbatch in range(4):
                inputs=recorder([object()])
                loss=trainer.compute_loss(policy,inputs);losses.append(float(loss.detach()))
                (loss/2).backward()
                embeddings=(torch.ones(3,1,2,4),torch.zeros(3,1,2,4))
                for layer in model.model.layers:
                    layer.self_attn(torch.zeros(1,2,8),position_embeddings=embeddings)
                    layer.self_attn.forward_flex(torch.zeros(1,2,8),block_mask=object(),position_embeddings=embeddings)
                for block in model.visual.blocks:
                    block.attn(torch.zeros(2,8),torch.tensor([0,2]),position_embeddings=(torch.ones(2,4),torch.zeros(2,4)))
                boundary[0]=microbatch % 2 == 1
                wrapped()
                if boundary[0]:trainer.state.global_step+=1
            assert len(policy.calls) == 4 and trainer.state.global_step == trainer._observed_optimizer_step == 2
            if not zero_gradient:
                assert trainer._observed_pre_step_gradient_norms == norms_at_step
                assert len(norms_at_step) == 2 and all(norm > 0 for norm in norms_at_step)
            ranks.append(SimpleNamespace(trainer=trainer,recorder=recorder,loss=SimpleNamespace(training_loss=sum(losses)/4),peak=8500))
        def execute(distributed=False):
            def run(rank):
                local.rank=rank;local.gather_index=0
                state=ranks[rank]
                runtime=m.publish_training_runtime(args,state.trainer)
                evidence=None
                if distributed:
                    evidence=m.publish_distributed_capacity(args,state.trainer,state.loss,state.recorder,state.peak,*runtime)
                return runtime,evidence
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(run,rank) for rank in (0,1)]
                outputs=[];errors=[]
                for future in futures:
                    try:outputs.append(future.result(timeout=30))
                    except Exception as error:errors.append(error)
            return outputs,errors
        return SimpleNamespace(args=args,ranks=ranks,execute=execute,events=events,live_gpus=live_gpus)
    return prepare


@pytest.mark.parametrize("triplet,metric",[([1,0,0],"loss/ce"),([0,1,0],"loss/dpo"),([0,0,1],"loss/entropy"),([1,1,.1],"loss/total")])
def test_actual_publishers_bind_real_objectives_boundaries_and_two_ranks(producing_ranks,triplet,metric):
    from pathlib import Path
    m=api();state=producing_ranks(triplet);distributed=metric == "loss/total"
    outputs,errors=state.execute(distributed)
    assert not errors
    assert outputs[0] == outputs[1]
    path,sha=outputs[0][0];runtime=m.read_json(path,m.TrainingRuntimeIdentity,sha)
    assert runtime.objective_metric_name == metric and runtime.objective_metric_observation_count == 4
    assert runtime.first_positive_pre_step_gradient_norms == state.ranks[0].trainer._observed_pre_step_gradient_norms
    assert runtime.positive_pre_step_gradient_count == 2 and runtime.global_step == runtime.optimizer_step == 2
    assert [record.rank for record in runtime.rank_observations] == [0,1]
    for record in runtime.rank_observations:
        assert record.sdpa_forward_counts == record.flex_attention_forward_counts == record.vision_flash_attention_forward_counts == [4,4]
    assert len([event for event in state.events if event[1] == "barrier"]) == (4 if distributed else 2)
    if distributed:
        evidence=m.read_json(state.args.step_packed_capacity_evidence_path,m.PackedDistributedCapacityEvidence)
        assert evidence == outputs[0][1]
        assert evidence.training_runtime_identity_sha256 == sha
        assert all(record.sample_ids == ["sample-0"]*4 and record.peak_reserved_fraction == .85 for record in evidence.rank_observations)
    assert {path.name for path in Path(state.args.save_model_path).iterdir()} == {*m.ADAPTER_FILES,"stage2_training_runtime_identity.json"}
    original=path.read_bytes()
    _,errors=state.execute(distributed)
    assert len(errors) == 2 and all("publication" in str(error) for error in errors)
    assert path.read_bytes() == original


@pytest.mark.parametrize("failure",["zero_gradient","unexecuted_layer","rank_step_disagreement"])
def test_actual_runtime_publisher_rejects_failed_observation_on_both_ranks(producing_ranks,failure):
    from pathlib import Path
    state=producing_ranks([1,0,0],zero_gradient=failure == "zero_gradient")
    if failure == "unexecuted_layer":state.ranks[1].trainer.model.model.layers[1].self_attn._flex_attention_forward_count=0
    if failure == "rank_step_disagreement":state.ranks[1].trainer.state.global_step=1
    _,errors=state.execute()
    assert len(errors) == 2
    assert not (Path(state.args.save_model_path)/"stage2_training_runtime_identity.json").exists()


@pytest.mark.parametrize("failure",["loss","memory","gpu","sample"])
def test_actual_distributed_publisher_rejects_failed_rank_after_runtime(producing_ranks,failure):
    from pathlib import Path
    from dataclasses import replace
    state=producing_ranks([1,1,.1])
    if failure == "loss":state.ranks[1].loss.training_loss=float("nan")
    if failure == "memory":state.ranks[1].peak=8501
    if failure == "gpu":state.live_gpus[1]=replace(state.live_gpus[1],uuid="00000000-0000-4000-8000-000000000001")
    if failure == "sample":state.ranks[1].recorder.sample_ids[3]="other"
    _,errors=state.execute(distributed=True)
    assert len(errors) == 2
    assert (Path(state.args.save_model_path)/"stage2_training_runtime_identity.json").exists()
    assert not Path(state.args.step_packed_capacity_evidence_path).exists()


@pytest.mark.parametrize("format",["indented","reordered"])
def test_actual_runtime_publisher_rejects_noncanonical_sampler_with_matching_hash(producing_ranks,runtime_inputs,format):
    from pathlib import Path
    m=api();state=producing_ranks([1,0,0]);payload=json.loads(runtime_inputs.length.read_bytes())
    raw=(json.dumps(payload,indent=2)+"\n").encode() if format == "indented" else (json.dumps(dict(reversed(list(payload.items()))),separators=(",",":"))+"\n").encode()
    runtime_inputs.length.write_bytes(raw);state.args.stage2_length_manifest_sha256=m.digest(raw)
    _,errors=state.execute()
    assert len(errors) == 2 and all("canonical" in str(error) for error in errors)
    assert not (Path(state.args.save_model_path)/"stage2_training_runtime_identity.json").exists()


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
