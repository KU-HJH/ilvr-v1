import importlib
import importlib.util
from pathlib import Path
import subprocess
import pytest
from tests.step_preference.test_packed_v13_premise import evidence as historical_evidence, _authenticate
from tests.step_preference.test_packed_capacity_gate import capacity_fixture,inputs,write_adapter
from scripts.run_packed_admission_protocol import authenticate_pilot as actual_authenticate_pilot


@pytest.fixture
def complete_protocol(capacity_fixture,tmp_path,monkeypatch):
    """CPU subprocess producers; admission and final publish/verify remain real."""
    import contextlib
    import io
    import json
    import shutil
    from copy import deepcopy
    from dataclasses import asdict,replace
    from types import SimpleNamespace as N
    from stage2.length_manifest import build_stage2_length_manifest,write_stage2_length_manifest,read_stage2_length_manifest
    from step_preference import packed_admission as admission
    from step_preference.negatives import inspect_stage2a_checkpoint_identity,write_negative_artifact,_row_from_payload
    from step_preference.capacity_evidence import canonical,digest,decode,PackedGpuSnapshot,PackedCommandLog,read_json
    from tests.step_preference.test_packed_admission_cli import AdmissionProcessor
    from tests.step_preference.test_packed_capacity_evidence import snapshot,runtime_payload,evidence_payload
    from scripts import validate_packed_capacity_gate as gate

    m=protocol();fixture=capacity_fixture
    identity=json.loads(Path(fixture.admission_manifest).read_bytes())["identity"]
    data=tmp_path/"data";data.mkdir()
    shutil.copyfile(tmp_path/"input.png",data/"input.png")
    source=data/"TRAIN.jsonl";source.write_bytes(Path(identity["original_data_path"]).read_bytes())
    checkpoint=tmp_path/m.CHECKPOINT;shutil.copytree(identity["checkpoint_path"],checkpoint)
    inventory,tokenizer=inspect_stage2a_checkpoint_identity(checkpoint)
    for name in ("results","logs"):(tmp_path/name).mkdir()
    # Rebind the existing historical fixture's exact descriptors to this fixture's
    # source/checkpoint, then run the actual premise and transitive pilot readers.
    from scripts import publish_packed_v13_premise as premise
    historical_root=tmp_path/"historical";historical_root.mkdir()
    history=historical_evidence.__wrapped__(historical_root,monkeypatch)
    inventory_payload=[asdict(item) for item in inventory];frozen={}
    for row in (1241,1314,1372):
        directory=historical_root/("v11" if row == 1372 else f"row-{row}")
        negative_manifest=directory/f"row-{row}-negative-manifest.json"
        payload=json.loads(negative_manifest.read_bytes())
        payload.update(stage2a_checkpoint_path=str(checkpoint),checkpoint_files=inventory_payload,source_sha256=digest(source.read_bytes()))
        negative_manifest.write_bytes(canonical(payload))
        audit=directory/f"row-{row}-packed-audit.json";payload=json.loads(audit.read_bytes())
        payload["identity"].update(checkpoint_path=str(checkpoint),data_path=str(source),image_root=str(data))
        audit.write_bytes(canonical(payload))
        capacity=directory/f"row-{row}-capacity.json";payload=json.loads(capacity.read_bytes())
        payload["identity"]["checkpoint_path"]=str(checkpoint);capacity.write_bytes(canonical(payload))
        if row != 1372:
            frozen.update({f"row{row}_{name}":digest(path.read_bytes()) for name,path in
                (("manifest",negative_manifest),("audit",audit),("capacity",capacity))})
    v11_path=history["v11"];v11=json.loads(v11_path.read_bytes())
    v11["original_train"]=dict(path=str(source),sha256=digest(source.read_bytes()))
    v11["checkpoint"].update(path=str(checkpoint),checkpoint_files=inventory_payload)
    for reference in [*v11["checkpoint"]["identity_source_manifests"],*v11["audits"],*v11["capacity_reports"]]:
        reference["sha256"]=digest(Path(reference["path"]).read_bytes())
    v11_path.write_bytes(canonical(v11))
    monkeypatch.setattr(premise,"HISTORICAL_V11_SUMMARY_SHA256",digest(v11_path.read_bytes()))
    monkeypatch.setattr(premise,"FROZEN_INPUT_SHA256",frozen)
    history.update(original=source,checkpoint=checkpoint)
    pilot=_authenticate(history)
    pilot_path=tmp_path/"protocol-pilot.json";pilot_path.write_bytes(canonical(pilot))
    monkeypatch.setattr(m,"authenticate_pilot",actual_authenticate_pilot)
    monkeypatch.setattr(gate,"authenticate_pilot",actual_authenticate_pilot)
    gpu=snapshot();gpu["gpus"][0].update(uuid=pilot["gpu"]["uuid"],total_memory_bytes=pilot["gpu"]["total_memory_bytes"])
    monkeypatch.setattr(m,"capture_gpu_snapshot",lambda:decode(PackedGpuSnapshot,gpu))
    monkeypatch.setattr(gate,"capture_gpu_snapshot",lambda:decode(PackedGpuSnapshot,gpu))
    negatives=[_row_from_payload(json.loads(line)) for line in Path(identity["negative_path"]).read_bytes().splitlines()]
    generation=json.loads(Path(identity["negative_manifest_path"]).read_bytes())["generation_settings"]
    report_template=json.loads(Path(fixture.capacity_report[0]).read_bytes())
    report_template["gpu"]=pilot["gpu"]
    total=pilot["gpu"]["total_memory_bytes"];peak=int(total*.8)
    report_template["memory"].update(device_total_bytes=total,peak_reserved_bytes=peak,peak_reserved_fraction=peak/total)
    original_run=subprocess.run;seen=[];phases=[];failure=[None];exclusive=[False]
    identifier="step-pref-packed-admission-complete"
    result=tmp_path/"results"/identifier;logs=tmp_path/"logs"/identifier
    def option(command,name):return command[command.index(name)+1]
    def body(command,environment):
        if command[:2] == ["-m","stage2.length_manifest"]:
            phase="clean"
        elif command[:2] == ["-m","step_preference.packed_admission"]:
            phase="admission-"+command[2]
        elif command[0].endswith("generate_step_negatives.py"):
            phase="negative"
        elif command[0].endswith("probe_packed_stage2b_capacity.py"):
            phase="single-"+option(command,"--source_row_index")
        elif command[0] == "./run_stage2_training.sh":
            phase="distributed-"+environment["OUTPUT_DIR"].rsplit("-",1)[-1]
        else:phase="gate-"+command[1]
        phases.append(phase)
        if phase == failure[0]:raise subprocess.CalledProcessError(9,seen[-1])
        if phase == "clean":
            measured=build_stage2_length_manifest(data_path=source,image_root=data,processor=AdmissionProcessor(),
                processor_source=str(checkpoint),latent_size=8,supervise_latent_boundaries=True,max_sequence_length=4608,
                assistant_plan_rendering="adjacent-text-space-lf-v1")
            old=read_stage2_length_manifest(identity["clean_length_manifest_path"])
            write_stage2_length_manifest(option(command,"--manifest-path"),replace(measured,lengths=old.lengths,
                retained_indices=old.retained_indices,stats=old.stats))
        elif phase == "negative":
            clean=Path(option(command,"--data_path"))
            write_negative_artifact(negative_path=option(command,"--output_path"),manifest_path=option(command,"--manifest_path"),
                negatives=negatives,source_path=clean,source_sha256=digest(clean.read_bytes()),selected_row_indices=range(3354),
                stage2a_checkpoint_path=checkpoint,checkpoint_files=inventory,tokenizer_config_identity=tokenizer,
                latent_size=8,assistant_plan_rendering="adjacent-text-space-lf-v1",generation_settings=generation)
        elif phase.startswith("admission-"):
            assert admission.main(command[2:]) == 0
        elif phase.startswith("single-"):
            source_index=int(option(command,"--source_row_index"))
            manifest=json.loads((result/"packed-admission.json").read_bytes())
            record=next(item for item in manifest["records"] if item["source_index"] == source_index)
            report=deepcopy(report_template)
            report["identity"].update(checkpoint_path=str(checkpoint),git_commit=manifest["identity"]["git_commit"],
                gpu_snapshot_path=str(result/"gpu-snapshot.json"),gpu_snapshot_sha256=digest((result/"gpu-snapshot.json").read_bytes()))
            report["sample"].update(source_row_index=source_index,sample_id=record["sample_id"],clean_length=record["clean_length"],
                step_count=record["reasoning_block_count"],branch_count=record["branch_count"],
                packed_unpadded_length=record["packed_unpadded_length"],packed_padded_length=record["packed_padded_length"])
            Path(option(command,"--output_path")).write_bytes(canonical(report))
        elif phase.startswith("distributed-"):
            root=Path(environment["OUTPUT_DIR"]);log=Path(environment["LOG_FILE"])
            assert not root.exists() and not log.parent.exists()
            write_adapter(root);log.parent.mkdir();log.write_text("launcher-owned training stdout/stderr\n")
            witness_result=tmp_path/"results"/root.name;witness_result.mkdir()
            length=witness_result/"stage2-length-manifest.json"
            write_stage2_length_manifest(length,build_stage2_length_manifest(data_path=environment["DATA_PATH"],image_root=tmp_path/"data",
                processor=AdmissionProcessor(),processor_source=str(checkpoint),latent_size=8,supervise_latent_boundaries=True,
                max_sequence_length=8192,assistant_plan_rendering="adjacent-text-space-lf-v1"))
            runtime=runtime_payload(tmp_path);manifest=json.loads((result/"packed-admission.json").read_bytes())
            runtime.update(run_id=root.name,git_commit=manifest["identity"]["git_commit"],coefficient_triplet=[1.,1.,.1],objective_metric_name="loss/total",
                training_length_manifest_path=str(length),training_length_manifest_sha256=digest(length.read_bytes()),
                admission_manifest_sha256=environment["STEP_PACKED_ADMISSION_MANIFEST_SHA256"],
                verification_sha256=environment["STEP_PACKED_ADMISSION_VERIFICATION_SHA256"])
            runtime_path=root/"stage2_training_runtime_identity.json";runtime_path.write_bytes(canonical(runtime))
            evidence=evidence_payload(tmp_path);sample=environment["STEP_PACKED_CAPACITY_SAMPLE_ID"]
            evidence.update(run_id=root.name,sample_id=sample,checkpoint_path=str(root),training_runtime_identity_path=str(runtime_path),
                training_runtime_identity_sha256=digest(runtime_path.read_bytes()),gpu_snapshot_path=environment["STEP_PACKED_GPU_SNAPSHOT"],
                gpu_snapshot_sha256=environment["STEP_PACKED_GPU_SNAPSHOT_SHA256"])
            for observation,rank_gpu in zip(evidence["rank_observations"],gpu["gpus"]):
                total=rank_gpu["total_memory_bytes"];peak=int(total*.8)
                observation.update(sample_ids=[sample]*4,total_memory_bytes=total,peak_reserved_bytes=peak,peak_reserved_fraction=peak/total)
            evidence["gpu_identities"]=[dict(rank=rank,physical_index=rank,name=rank_gpu["name"],uuid=rank_gpu["uuid"],
                total_memory_bytes=rank_gpu["total_memory_bytes"]) for rank,rank_gpu in enumerate(gpu["gpus"])]
            Path(environment["STEP_PACKED_CAPACITY_EVIDENCE_PATH"]).write_bytes(canonical(evidence))
        else:
            if phase == "gate-publish":
                command_path=Path(option(command,"--commands-path"))
                observed=read_json(command_path,PackedCommandLog,option(command,"--commands-sha256"))
                assert observed.commands == seen[:-1]
                raw=command_path.read_bytes()
                with pytest.raises(ValueError):m.publish_json(command_path,observed,PackedCommandLog)
                assert command_path.read_bytes() == raw;exclusive[0]=True
            assert gate.main(command[1:]) == 0
    def run(argv,**kwargs):
        if argv[0] == "nvidia-smi":return subprocess.CompletedProcess(argv,0,stdout="")
        if argv[0] != "/usr/bin/env":return original_run(argv,**kwargs)
        assert "env" not in kwargs and kwargs["check"] is True
        seen.append(list(argv));environment={};index=1
        while "=" in argv[index] and not argv[index].startswith("/"):
            key,value=argv[index].split("=",1);environment[key]=value;index+=1
        if argv[index].endswith("/bin/python"):index+=1
        command=argv[index:]
        if command[0] == "./run_stage2_training.sh":
            assert kwargs == {"check":True}
            body(command,environment)
        else:
            assert kwargs["stderr"] is subprocess.STDOUT
            capture=io.StringIO()
            with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):body(command,environment)
            kwargs["stdout"].write(capture.getvalue().encode())
        return subprocess.CompletedProcess(argv,0)
    monkeypatch.setattr(subprocess,"run",run)
    return N(module=m,run=lambda:m.run_protocol(tmp_path,identifier,pilot_path,digest(pilot_path.read_bytes())),
        seen=seen,phases=phases,failure=failure,exclusive=exclusive,result=result,logs=logs,root=tmp_path,identifier=identifier)


def test_complete_protocol_all_witnesses_command_log_and_real_final_verify(complete_protocol,capsys):
    import json
    from step_preference.capacity_evidence import read_json,PackedCommandLog,digest
    from scripts.validate_packed_capacity_gate import PackedCapacitySummary
    state=complete_protocol;result=state.run()
    assert state.phases == ["clean","negative","admission-build","admission-verify","single-0","single-3353",
        "distributed-0","distributed-3353","gate-publish","gate-verify"]
    assert state.exclusive == [True]
    command_log=read_json(result/"packed-command-log.json",PackedCommandLog)
    assert command_log.commands == state.seen[:8]
    snapshot_sha=digest((result/"gpu-snapshot.json").read_bytes())
    manifest_sha=digest((result/"packed-admission.json").read_bytes())
    verification_sha=digest((result/"packed-admission-verification.json").read_bytes())
    root=state.root;python=state.module.python;checkpoint=state.module.CHECKPOINT
    assert state.seen[:4] == [
        python(root,"-m","stage2.length_manifest","--data-path","data/TRAIN.jsonl","--image-root","data",
            "--processor-source",checkpoint,"--latent-size","8","--supervise-latent-boundaries","--max-sequence-length","4608",
            "--assistant-plan-rendering","adjacent-text-space-lf-v1","--manifest-path",result/"clean-length-manifest.json"),
        python(root,root/"scripts/generate_step_negatives.py","--data_path",result/"clean-admissible.jsonl","--image_root","data",
            "--stage2a_checkpoint_path",checkpoint,"--output_path",result/"negatives.jsonl","--manifest_path",result/"negative-manifest.json",
            "--latent_size","8","--min_tokens","2","--max_new_tokens","32","--seed","42",
            "--attention_implementation","sdpa","--vision_attention_implementation","flash_attention_2","--device","cuda:0",cuda="0"),
        python(root,"-m","step_preference.packed_admission","build","--original-data-path","data/TRAIN.jsonl",
            "--clean-data-path",result/"clean-admissible.jsonl","--clean-length-manifest",result/"clean-length-manifest.json",
            "--clean-length-manifest-sha256",digest((result/"clean-length-manifest.json").read_bytes()),"--image-root","data",
            "--source-map-path",result/"clean-source-map.json","--source-map-sha256",digest((result/"clean-source-map.json").read_bytes()),
            "--checkpoint-path",checkpoint,"--negative-path",result/"negatives.jsonl","--negative-manifest",result/"negative-manifest.json",
            "--max-packed-sequence-length","4608","--retained-output-path",result/"packed-retained.jsonl",
            "--manifest-output-path",result/"packed-admission.json"),
        python(root,"-m","step_preference.packed_admission","verify","--manifest-path",result/"packed-admission.json",
            "--manifest-sha256",manifest_sha,"--output-path",result/"packed-admission-verification.json")]
    assert state.seen[4:6] == [state.module.probe_command(state.root,state.identifier,index,f"sample-{index}",snapshot_sha) for index in (0,3353)]
    assert state.seen[6:8] == [state.module.distributed_command(state.root,state.identifier,index,f"sample-{index}",snapshot_sha,manifest_sha,verification_sha) for index in (0,3353)]
    summary=read_json(result/"capacity-summary.json",PackedCapacitySummary)
    assert summary.command_log_sha256 == digest((result/"packed-command-log.json").read_bytes())
    assert [w.source_index for w in summary.witness_reports] == [w.source_index for w in summary.distributed_witnesses] == [0,3353]
    for name,index in (("capacity-validator.log",8),("capacity-summary-verify.log",9)):
        assert json.loads((state.logs/name).read_bytes().splitlines()[0]) == state.seen[index]
    assert "VERIFIED_CAPACITY_V13 status=pass cap=4608\n" in (state.logs/"capacity-summary-verify.log").read_text()
    assert f"CAPACITY_SUMMARY_SHA256={digest((result/'capacity-summary.json').read_bytes())}" in capsys.readouterr().out
    assert not any(path.name.startswith("zero2-") for path in state.logs.iterdir())


@pytest.mark.parametrize("phase,expected_count",[("single-3353",6),("distributed-0",7),("gate-publish",9)])
def test_later_protocol_failure_stops_all_subsequent_commands(complete_protocol,phase,expected_count,capsys):
    state=complete_protocol;state.failure[0]=phase
    with pytest.raises(subprocess.CalledProcessError):state.run()
    assert len(state.seen) == expected_count and state.phases[-1] == phase
    assert not (state.logs/"capacity-summary-verify.log").exists()
    assert not (state.result/"capacity-summary.json").exists()
    assert (state.result/"packed-command-log.json").exists() == (phase == "gate-publish")
    assert "FAILED_PHASE=" in capsys.readouterr().err


def protocol():
    name = "scripts.run_packed_admission_protocol"
    assert importlib.util.find_spec(name) is not None, "one-shot protocol missing"
    return importlib.import_module(name)


def test_recorded_environment_is_executed_argv_and_failure_stops(tmp_path, monkeypatch):
    m = protocol(); seen = []
    def run(argv, **kw):
        assert "env" not in kw and kw["check"] is True
        seen.append(argv)
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(subprocess, "run", run)
    commands = []
    command = ["/usr/bin/env", "CUDA_VISIBLE_DEVICES=0", "PYTHONPATH=/r/transformers/src:/r/src", "/r/venv/ilvr-venv/bin/python", "producer.py"]
    with pytest.raises(subprocess.CalledProcessError):
        m.execute(command, tmp_path / "generate.log", "negative_generation", commands)
        m.execute(["later"], tmp_path / "later.log", "admission_build", commands)
    assert seen == [command] and commands == [command]
    assert not (tmp_path / "later.log").exists()


def test_protocol_exact_probe_and_distributed_shared_artifacts(tmp_path):
    m = protocol(); root = tmp_path.resolve(); run = "step-pref-packed-admission-test"
    shared = root / "results" / run
    probe = m.probe_command(root, run, 7, "sample-a", "a" * 64)
    assert probe[:4] == ["/usr/bin/env", "CUDA_VISIBLE_DEVICES=0", f"PYTHONPATH={root}/transformers/src:{root}/src", "TOKENIZERS_PARALLELISM=false"]
    assert probe[probe.index("--sample_id") + 1] == "sample-a"
    assert probe[probe.index("--source_row_index") + 1] == "7"
    assert probe[probe.index("--negative_path") + 1] == str(shared / "negatives.jsonl")
    checkpoint=root/"checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827"
    assert probe == ["/usr/bin/env","CUDA_VISIBLE_DEVICES=0",f"PYTHONPATH={root}/transformers/src:{root}/src","TOKENIZERS_PARALLELISM=false",
        str(root/"venv/ilvr-venv/bin/python"),str(root/"scripts/probe_packed_stage2b_capacity.py"),
        "--checkpoint_path",str(checkpoint),"--data_path",str(shared/"clean-admissible.jsonl"),"--image_root",str(root/"data"),
        "--negative_path",str(shared/"negatives.jsonl"),"--negative_manifest",str(shared/"negative-manifest.json"),
        "--output_path",str(shared/"witness-7-capacity.json"),"--source_row_index","7","--sample_id","sample-a","--device","cuda:0",
        "--attention_implementation","sdpa","--vision_attention_implementation","flash_attention_2","--tuning_method","dora",
        "--gradient_checkpointing","--gpu-snapshot",str(shared/"gpu-snapshot.json"),"--gpu-snapshot-sha256","a"*64,
        "--max_peak_reserved_fraction","0.85"]
    command = m.distributed_command(root, run, 7, "sample-a", "a" * 64, "b" * 64, "c" * 64)
    assert command[1] == "CUDA_VISIBLE_DEVICES=0,1"
    assert f"DATA_PATH={shared}/packed-retained.jsonl" in command
    assert f"STEP_NEGATIVE_PATH={shared}/negatives.jsonl" in command
    assert "STEP_PACKED_CAPACITY_SAMPLE_ID=sample-a" in command
    assert command[-6:] == ["--max_steps", "2", "--epochs", "1", "--save_steps", "2"]
    assert command[-7] == "./run_stage2_training.sh"
    witness=run+"-zero2-7"
    settings=[("STAGE1_CHECKPOINT",checkpoint),("STAGE2A_CHECKPOINT_PATH",checkpoint),
        ("DATA_PATH",shared/"packed-retained.jsonl"),("IMAGE_ROOT",root/"data"),("OUTPUT_DIR",root/"checkpoints"/witness),
        ("LOG_FILE",root/"logs"/witness/"train.log"),("STEP_PACKED_CAPACITY_EVIDENCE_PATH",shared/"zero2-7-evidence.json"),
        ("STEP_PACKED_CAPACITY_SAMPLE_ID","sample-a"),("STEP_PACKED_GPU_SNAPSHOT",shared/"gpu-snapshot.json"),("STEP_PACKED_GPU_SNAPSHOT_SHA256","a"*64),
        ("STEP_PREFERENCE_ENABLED","true"),("STEP_PREFERENCE_MODE","stage2b"),("STEP_PREFERENCE_OBJECTIVE","legacy_dpo"),("STEP_PREFERENCE_PACKING","true"),("STEP_MAX_PACKED_SEQUENCE_LENGTH","4608"),
        ("STEP_NEGATIVE_PATH",shared/"negatives.jsonl"),("STEP_NEGATIVE_MANIFEST",shared/"negative-manifest.json"),
        ("STEP_PACKED_ADMISSION_MANIFEST",shared/"packed-admission.json"),("STEP_PACKED_ADMISSION_MANIFEST_SHA256","b"*64),
        ("STEP_PACKED_ADMISSION_VERIFICATION",shared/"packed-admission-verification.json"),("STEP_PACKED_ADMISSION_VERIFICATION_SHA256","c"*64),
        ("CE_WEIGHT","1"),("STEP_DPO_WEIGHT","1"),("STEP_ENTROPY_WEIGHT","0.1"),("STEP_DPO_BETA","0.1"),("STEP_ENTROPY_MARGIN","0.1"),
        ("TUNING_METHOD","dora"),("LORA_R","32"),("LORA_ALPHA","64"),("LORA_DROPOUT","0.05"),
        ("LORA_TARGET_MODULES","q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"),("DEEPSPEED_CONFIG",root/"configs/deepspeed_zero2.json"),
        ("LEARNING_RATE","5e-5"),("SEED","42"),("PER_DEVICE_TRAIN_BATCH_SIZE","1"),("GRADIENT_ACCUMULATION_STEPS","2"),
        ("GRADIENT_CHECKPOINTING","true"),("ATTENTION_IMPLEMENTATION","sdpa"),("VISION_ATTENTION_IMPLEMENTATION","flash_attention_2"),
        ("MAX_SEQUENCE_LENGTH","8192"),("SUPERVISE_LATENT_BOUNDARIES","true"),("DATALOADER_NUM_WORKERS","0"),("STAGE2_NUM_PROCESSES","2")]
    assert command == ["/usr/bin/env","CUDA_VISIBLE_DEVICES=0,1",f"PYTHONPATH={root}/transformers/src:{root}/src","TOKENIZERS_PARALLELISM=false",
        *[f"{key}={value}" for key,value in settings],"./run_stage2_training.sh","--max_steps","2","--epochs","1","--save_steps","2"]


def test_protocol_path_grammar_and_existing_target_reject_before_creation(tmp_path):
    m = protocol(); occupied = tmp_path / "occupied"; occupied.mkdir()
    with pytest.raises(ValueError): m.require_fresh(occupied)
    for value in ["../x", "step-pref-packed-admission-X", "step-pref-packed-admission-a/b"]:
        with pytest.raises(ValueError): m.run_paths(tmp_path, value)


def test_distributed_execution_leaves_log_creation_to_launcher(tmp_path,monkeypatch):
    m=protocol()
    assert hasattr(m,"execute_distributed"),"missing launcher-owned logging route"
    log=tmp_path/"witness"/"train.log"; seen=[]; commands=[]
    def run(argv,**kw):
        assert not log.parent.exists()
        assert kw == {"check":True}
        seen.append(argv)
    monkeypatch.setattr(subprocess,"run",run)
    command=["/usr/bin/env","LOG_FILE="+str(log),"./run_stage2_training.sh"]
    m.execute_distributed(command,log,commands)
    assert commands == seen == [command]
    assert not log.parent.exists()


@pytest.mark.parametrize("occupied",[None,"results","logs"])
def test_driver_reauthenticates_pilot_and_gates_fresh_targets_before_first_launch(historical_evidence,tmp_path,monkeypatch,occupied):
    from step_preference.capacity_evidence import canonical,decode,PackedGpuSnapshot,digest
    from tests.step_preference.test_packed_capacity_evidence import snapshot
    m=protocol();original_run=subprocess.run
    assert m.authenticate_pilot is actual_authenticate_pilot
    monkeypatch.chdir(tmp_path)
    for command in (["git","init","-q"],["git","config","user.name","Protocol Test"],["git","config","user.email","test@example.invalid"]):original_run(command,check=True)
    (tmp_path/".gitignore").write_text("*\n");(tmp_path/"tracked").write_text("clean\n")
    original_run(["git","add","-f",".gitignore","tracked"],check=True)
    original_run(["git","commit","-qm","fixture"],check=True)
    (tmp_path/"data").symlink_to(tmp_path,target_is_directory=True)
    checkpoint=tmp_path/m.CHECKPOINT;checkpoint.parent.mkdir(parents=True);checkpoint.symlink_to(historical_evidence["checkpoint"],target_is_directory=True)
    for name in ("results","logs"):(tmp_path/name).mkdir()
    pilot=_authenticate(historical_evidence);path=tmp_path/"pilot.json";path.write_bytes(canonical(pilot))
    gpu=snapshot();gpu["gpus"][0].update(uuid=pilot["gpu"]["uuid"],total_memory_bytes=pilot["gpu"]["total_memory_bytes"])
    monkeypatch.setattr(m,"capture_gpu_snapshot",lambda:decode(PackedGpuSnapshot,gpu))
    identifier="step-pref-packed-admission-test";seen=[]
    if occupied:(tmp_path/occupied/identifier).mkdir()
    def run(argv,**kw):
        if argv[0] == "nvidia-smi":return subprocess.CompletedProcess(argv,0,stdout="")
        if argv[0] != "/usr/bin/env":return original_run(argv,**kw)
        assert "env" not in kw and kw["check"] is True
        seen.append(argv)
        raise subprocess.CalledProcessError(7,argv)
    monkeypatch.setattr(subprocess,"run",run)
    with pytest.raises(ValueError if occupied else subprocess.CalledProcessError):
        m.run_protocol(tmp_path,identifier,path,digest(path.read_bytes()))
    assert len(seen) == (0 if occupied else 1)
    if not occupied:
        assert seen[0] == m.python(tmp_path,"-m","stage2.length_manifest","--data-path","data/TRAIN.jsonl","--image-root","data",
            "--processor-source",m.CHECKPOINT,"--latent-size","8","--supervise-latent-boundaries","--max-sequence-length","4608",
            "--assistant-plan-rendering","adjacent-text-space-lf-v1","--manifest-path",tmp_path/"results"/identifier/"clean-length-manifest.json")
        assert {p.name for p in (tmp_path/"logs"/identifier).iterdir()} == {"clean-length-manifest.log"}
        assert not (tmp_path/"results"/identifier/"packed-command-log.json").exists()
