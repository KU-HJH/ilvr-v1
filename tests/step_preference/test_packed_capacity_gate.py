import importlib
import importlib.util
import pytest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace as N
import json

from tests.step_preference.test_packed_admission_cli import inputs, build, verify, digest
from tests.step_preference.test_packed_v13_premise import evidence as historical_evidence, _authenticate


def test_pilot_reopens_rejected_chain_and_rejects_status_cap_mutation(historical_evidence,tmp_path):
    m=gate(); from step_preference.capacity_evidence import canonical
    pilot=_authenticate(historical_evidence); path=tmp_path/"pilot.json"
    path.write_bytes(canonical(pilot))
    assert m.authenticate_pilot(path,digest(path)) == pilot
    for key,value in [("status","fail"),("candidate_cap",4864),("original_data_sha256","0"*64)]:
        changed={**pilot,key:value};path.write_bytes(canonical(changed))
        with pytest.raises(ValueError):m.authenticate_pilot(path,digest(path))


def write_adapter(root):
    from tests.stage2.test_peft_checkpointing import sample_metadata
    from stage2.checkpointing import save_stage2_peft_metadata
    from step_preference.capacity_evidence import canonical
    root.mkdir(parents=True)
    save_stage2_peft_metadata(root,replace(sample_metadata(),lora_r=32,lora_alpha=64,lora_dropout=.05))
    (root/"adapter_config.json").write_bytes(canonical(dict(use_dora=True,r=32,lora_alpha=64,lora_dropout=.05)))
    (root/"adapter_model.safetensors").write_bytes(b"adapter weights")


@pytest.fixture
def capacity_fixture(inputs,tmp_path,monkeypatch):
    from tests.step_preference.test_packed_capacity_evidence import snapshot,runtime_payload,evidence_payload
    from tests.step_preference.test_packed_admission_cli import AdmissionProcessor
    from stage2.length_manifest import build_stage2_length_manifest,write_stage2_length_manifest
    from step_preference.capacity_evidence import canonical
    m=gate(); args=inputs(count=3450); manifest=build(args)
    code,_=verify(args); assert code == 0
    pilot=tmp_path/"pilot.json"; pilot.write_bytes(canonical({"fixture":"historical boundary separately tested"}))
    gpu=tmp_path/"gpu.json"; gpu.write_bytes(canonical(snapshot()))
    monkeypatch.setattr(m,"capture_gpu_snapshot",snapshot)
    monkeypatch.setattr(m,"authenticate_pilot",lambda *a:dict(original_data_path=manifest.identity.original_data_path,
        original_data_sha256=manifest.identity.original_data_sha256,checkpoint_path=manifest.identity.checkpoint_path,
        checkpoint_files=[asdict(f) for f in manifest.identity.checkpoint_files],
        gpu=dict(physical_index=0,name="NVIDIA H100 NVL",uuid=snapshot()["gpus"][0]["uuid"],total_memory_bytes=10000)))
    commands=tmp_path/"commands.json"; commands.write_bytes(canonical(dict(schema_version=1,kind="stage2b-packed-command-log",commands=[["producer"]])))
    lengths=build_stage2_length_manifest(data_path=args.retained_output_path,image_root=args.image_root,
        processor=AdmissionProcessor(),processor_source=str(args.checkpoint_path),latent_size=8,supervise_latent_boundaries=True,
        max_sequence_length=8192,assistant_plan_rendering="adjacent-text-space-lf-v1")
    reports=[]; evidences=[]; checkpoints=[]
    for index in manifest.boundary_witnesses.final_probe_source_indices:
        row=next(r for r in manifest.records if r.source_index == index)
        root=tmp_path/"checkpoints"/f"step-pref-packed-admission-test-zero2-{index}"
        write_adapter(root)
        length=tmp_path/f"length-{index}.json";write_stage2_length_manifest(length,lengths)
        runtime=runtime_payload(tmp_path)
        runtime.update(run_id=root.name,git_commit=manifest.identity.git_commit,coefficient_triplet=[1.,1.,.1],objective_metric_name="loss/total",
            training_length_manifest_path=str(length),training_length_manifest_sha256=digest(length),
            admission_manifest_sha256=digest(args.manifest_output_path),verification_sha256=digest(tmp_path/"verification.json"))
        runtime_path=root/"stage2_training_runtime_identity.json";runtime_path.write_bytes(canonical(runtime))
        ev=evidence_payload(tmp_path)
        ev.update(run_id=root.name,sample_id=row.sample_id,checkpoint_path=str(root),gpu_snapshot_path=str(gpu),gpu_snapshot_sha256=digest(gpu),
            training_runtime_identity_path=str(runtime_path),training_runtime_identity_sha256=digest(runtime_path))
        for rank in ev["rank_observations"]:rank["sample_ids"]=[row.sample_id]*4
        ev_path=tmp_path/f"evidence-{index}.json";ev_path.write_bytes(canonical(ev))
        report=dict(schema_version=1,status="pass",identity=dict(checkpoint_path=manifest.identity.checkpoint_path,device="cuda:0",dtype="bfloat16",
            attention_implementation="sdpa",vision_attention_implementation="flash_attention_2",tuning_method="dora",gradient_checkpointing=True,
            git_commit=manifest.identity.git_commit,gpu_snapshot_path=str(gpu),gpu_snapshot_sha256=digest(gpu)),
            sample=dict(source_row_index=index,sample_id=row.sample_id,clean_length=row.clean_length,step_count=row.reasoning_block_count,
                branch_count=row.branch_count,packed_unpadded_length=row.packed_unpadded_length,packed_padded_length=row.packed_padded_length),
            losses=dict(ce=1.,dpo=1.,entropy=.1,total=2.1),gradient_norm=1.25,
            memory=dict(device_total_bytes=10000,baseline_allocated_bytes=100,peak_allocated_bytes=5000,
                baseline_reserved_bytes=1000,peak_reserved_bytes=8500,peak_reserved_fraction=.85),
            gpu=dict(physical_index=0,name="NVIDIA H100 NVL",uuid=snapshot()["gpus"][0]["uuid"],total_memory_bytes=10000),
            elapsed_seconds=1.,errors=[])
        report_path=tmp_path/f"capacity-{index}.json";report_path.write_bytes(canonical(report))
        reports.append(str(report_path));evidences.append(str(ev_path));checkpoints.append(str(root))
    return N(pilot_summary=str(pilot),pilot_summary_sha256=digest(pilot),gpu_snapshot=str(gpu),gpu_snapshot_sha256=digest(gpu),
        admission_manifest=str(args.manifest_output_path),admission_manifest_sha256=digest(args.manifest_output_path),
        verification=str(tmp_path/"verification.json"),verification_sha256=digest(tmp_path/"verification.json"),
        commands_path=str(commands),commands_sha256=digest(commands),capacity_report=reports,distributed_evidence=evidences,
        distributed_checkpoint=checkpoints,output_path=str(tmp_path/"capacity-summary.json"))


def test_final_gate_publishes_complete_witnesses_matches_by_source_and_reopens_all(capacity_fixture):
    m=gate(); args=capacity_fixture
    from step_preference.capacity_evidence import canonical,publish_json
    args.distributed_checkpoint.reverse()
    summary=m.build_summary(args)
    assert [w.source_index for w in summary.witness_reports] == [0,3353]
    assert summary.counts.original == 3450 and summary.counts.clean_admissible == 3354
    sha=publish_json(args.output_path,summary,m.PackedCapacitySummary)
    assert m.verify_summary(args.output_path,sha) == summary
    paths=[Path(args.capacity_report[0]),Path(args.distributed_evidence[0]),
        Path(args.distributed_checkpoint[0])/"adapter_model.safetensors",
        Path(summary.distributed_witnesses[0].training_runtime_identity_path),Path(args.commands_path)]
    for path in paths:
        raw=path.read_bytes();path.write_bytes(raw+b" ")
        with pytest.raises(ValueError):m.verify_summary(args.output_path,sha)
        path.write_bytes(raw)
    for field in ("capacity_report","distributed_evidence"):
        original=getattr(args,field);setattr(args,field,original[:1])
        with pytest.raises(ValueError):m.build_summary(args)
        setattr(args,field,original)


def test_failed_gate_publishes_one_bounded_error_without_trusted_witnesses(tmp_path):
    m=gate(); argv=["publish"]
    for flag in ("admission-manifest","pilot-summary","gpu-snapshot","commands-path","verification"):
        argv += ["--"+flag,str(tmp_path/(flag+".json")),"--"+("commands-sha256" if flag == "commands-path" else flag+"-sha256"),"a"*64]
    for flag in ("capacity-report","distributed-evidence","distributed-checkpoint"):
        argv += ["--"+flag,str(tmp_path/flag)]
    output=tmp_path/"failed.json";argv += ["--output-path",str(output)]
    assert m.main(argv) == 2
    assert output.exists(),"bounded failure summary missing"
    payload=json.loads(output.read_bytes())
    assert payload["status"] == "fail" and len(payload["errors"]) == 1
    assert payload["witness_reports"] == payload["distributed_witnesses"] == []


def test_gate_binds_report_digest_to_exact_decoded_descriptor_bytes(capacity_fixture,monkeypatch):
    from step_preference import capacity_evidence as wire
    m=gate();args=capacity_fixture;target=Path(args.capacity_report[0]);original=target.read_bytes()
    reader=wire.admission._read_strict_bytes;swapped=[]
    def read(path,label):
        raw=reader(path,label)
        if Path(path) == target and not swapped:
            changed=json.loads(raw);changed["elapsed_seconds"]=2.
            target.write_bytes(wire.canonical(changed));swapped.append(True)
        return raw
    monkeypatch.setattr(wire.admission,"_read_strict_bytes",read)
    summary=m.build_summary(args)
    assert swapped and summary.witness_reports[0].sha256 == wire.digest(original)


def gate():
    name = "scripts.validate_packed_capacity_gate"
    assert importlib.util.find_spec(name) is not None, "capacity gate missing"
    return importlib.import_module(name)


def test_gate_requires_complete_manifest_witness_set():
    m = gate()
    from types import SimpleNamespace as N
    manifest = N(boundary_witnesses=N(final_probe_source_indices=(7, 11)),
        records=(N(source_index=7, sample_id="a"), N(source_index=11, sample_id="b")))
    assert m.match_witnesses(manifest, [(11, "b"), (7, "a")]) == [(7, "a"), (11, "b")]
    for pairs in [[(7, "a")], [(7, "a"), (7, "a")], [(7, "a"), (11, "wrong")]]:
        with pytest.raises(ValueError): m.match_witnesses(manifest, pairs)


def test_gate_cli_requires_all_shared_hashes_and_distributed_pair(tmp_path):
    m = gate()
    with pytest.raises(SystemExit): m.main(["publish", "--output-path", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()
