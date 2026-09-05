import importlib
import importlib.util
from pathlib import Path
import subprocess
import pytest
from tests.step_preference.test_packed_v13_premise import evidence as historical_evidence, _authenticate


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
        ("STEP_PREFERENCE_ENABLED","true"),("STEP_PREFERENCE_MODE","stage2b"),("STEP_PREFERENCE_PACKING","true"),("STEP_MAX_PACKED_SEQUENCE_LENGTH","4608"),
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
