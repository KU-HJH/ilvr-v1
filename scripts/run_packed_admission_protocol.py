#!/usr/bin/env python3
"""Execute the fixed v13 admission protocol once, stopping at its first failure."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from step_preference import packed_admission as admission
from step_preference.capacity_evidence import (
    run_id, require, git_identity, require_ignored_roots, require_idle,
    capture_gpu_snapshot, require_matching_snapshot, PackedGpuSnapshot,
    PackedCapacityReport, PackedDistributedCapacityEvidence, PackedCommandLog,
    read_json, read_bytes, digest, canonical, publish_json,
)
from scripts.validate_packed_capacity_gate import authenticate_pilot, authenticated_admission


CHECKPOINT = "checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827"


def require_fresh(*paths):
    for path in paths:
        path=Path(path)
        require(not path.is_symlink() and not path.exists(),f"fresh target already exists: {path}")


def below(root,*parts):
    candidate=root.joinpath(*parts)
    require(not candidate.is_symlink(),"derived output cannot be a symlink")
    path=candidate.resolve(strict=False)
    path.relative_to(root.resolve(strict=False))
    return path


def run_paths(root,identifier):
    run_id(identifier,"step-pref-packed-admission-")
    return below(root/"results",identifier),below(root/"logs",identifier),below(root/"checkpoints",identifier)


def env_command(root,argv,cuda=None,tokenizer=False):
    values=[]
    if cuda is not None: values.append("CUDA_VISIBLE_DEVICES="+cuda)
    values.append(f"PYTHONPATH={root}/transformers/src:{root}/src")
    if tokenizer: values.append("TOKENIZERS_PARALLELISM=false")
    return ["/usr/bin/env",*values,*[str(token) for token in argv]]


def python(root,*argv,cuda=None,tokenizer=False):
    return env_command(root,[root/"venv/ilvr-venv/bin/python",*argv],cuda,tokenizer)


def probe_command(root,identifier,source,sample,snapshot_sha):
    shared=root/"results"/identifier
    return python(root,root/"scripts/probe_packed_stage2b_capacity.py",
        "--checkpoint_path",root/CHECKPOINT,"--data_path",shared/"clean-admissible.jsonl",
        "--image_root",root/"data","--negative_path",shared/"negatives.jsonl",
        "--negative_manifest",shared/"negative-manifest.json","--output_path",shared/f"witness-{source}-capacity.json",
        "--source_row_index",source,"--sample_id",sample,"--device","cuda:0",
        "--attention_implementation","sdpa","--vision_attention_implementation","flash_attention_2",
        "--tuning_method","dora","--gradient_checkpointing","--gpu-snapshot",shared/"gpu-snapshot.json",
        "--gpu-snapshot-sha256",snapshot_sha,"--max_peak_reserved_fraction","0.85",cuda="0",tokenizer=True)


def distributed_command(root,identifier,source,sample,snapshot_sha,manifest_sha,verification_sha):
    shared=root/"results"/identifier; witness=f"{identifier}-zero2-{source}"
    settings=[
        ("STAGE1_CHECKPOINT",root/CHECKPOINT),("STAGE2A_CHECKPOINT_PATH",root/CHECKPOINT),
        ("DATA_PATH",shared/"packed-retained.jsonl"),("IMAGE_ROOT",root/"data"),
        ("OUTPUT_DIR",root/"checkpoints"/witness),("LOG_FILE",root/"logs"/witness/"train.log"),
        ("STEP_PACKED_CAPACITY_EVIDENCE_PATH",shared/f"zero2-{source}-evidence.json"),
        ("STEP_PACKED_CAPACITY_SAMPLE_ID",sample),("STEP_PACKED_GPU_SNAPSHOT",shared/"gpu-snapshot.json"),
        ("STEP_PACKED_GPU_SNAPSHOT_SHA256",snapshot_sha),("STEP_PREFERENCE_ENABLED","true"),("STEP_PREFERENCE_MODE","stage2b"),
        ("STEP_PREFERENCE_PACKING","true"),("STEP_MAX_PACKED_SEQUENCE_LENGTH","4608"),
        ("STEP_NEGATIVE_PATH",shared/"negatives.jsonl"),("STEP_NEGATIVE_MANIFEST",shared/"negative-manifest.json"),
        ("STEP_PACKED_ADMISSION_MANIFEST",shared/"packed-admission.json"),("STEP_PACKED_ADMISSION_MANIFEST_SHA256",manifest_sha),
        ("STEP_PACKED_ADMISSION_VERIFICATION",shared/"packed-admission-verification.json"),("STEP_PACKED_ADMISSION_VERIFICATION_SHA256",verification_sha),
        ("CE_WEIGHT","1"),("STEP_DPO_WEIGHT","1"),("STEP_ENTROPY_WEIGHT","0.1"),("STEP_DPO_BETA","0.1"),("STEP_ENTROPY_MARGIN","0.1"),
        ("TUNING_METHOD","dora"),("LORA_R","32"),("LORA_ALPHA","64"),("LORA_DROPOUT","0.05"),
        ("LORA_TARGET_MODULES","q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"),
        ("DEEPSPEED_CONFIG",root/"configs/deepspeed_zero2.json"),("LEARNING_RATE","5e-5"),("SEED","42"),
        ("PER_DEVICE_TRAIN_BATCH_SIZE","1"),("GRADIENT_ACCUMULATION_STEPS","2"),("GRADIENT_CHECKPOINTING","true"),
        ("ATTENTION_IMPLEMENTATION","sdpa"),("VISION_ATTENTION_IMPLEMENTATION","flash_attention_2"),
        ("MAX_SEQUENCE_LENGTH","8192"),("SUPERVISE_LATENT_BOUNDARIES","true"),("DATALOADER_NUM_WORKERS","0"),("STAGE2_NUM_PROCESSES","2")]
    return env_command(root,[*[f"{key}={value}" for key,value in settings],"./run_stage2_training.sh",
        "--max_steps","2","--epochs","1","--save_steps","2"],cuda="0,1",tokenizer=True)


def execute(command,log_path,phase,commands=None):
    require_fresh(log_path)
    if commands is not None:
        commands.append(list(command))
    with Path(log_path).open("xb") as log:
        # Preserve the validator/verifier argv here without a command-log hash cycle.
        if commands is None:
            log.write(canonical(command)); log.flush()
        try:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(f"FAILED_PHASE={phase} LOG_PATH={Path(log_path).resolve()}",file=sys.stderr)
            raise


def execute_distributed(command,log_path,commands):
    """Let the launcher's exclusive directory creation and capture own its log."""
    require_fresh(log_path,Path(log_path).parent)
    commands.append(command)
    try:
        subprocess.run(command,check=True)
    except subprocess.CalledProcessError:
        print(f"FAILED_PHASE=distributed_capacity LOG_PATH={Path(log_path).resolve()}",file=sys.stderr)
        raise


def run_protocol(root,identifier,pilot_path,pilot_sha):
    root=Path(root).resolve(strict=True)
    require(root == Path.cwd().resolve(strict=True),"protocol must run from its implementation root")
    result,logs,_=run_paths(root,identifier)
    # All preflights precede the first result/log allocation.
    pilot=authenticate_pilot(pilot_path,pilot_sha)
    require(Path(pilot["original_data_path"]) == (root/"data/TRAIN.jsonl").resolve(strict=True)
            and Path(pilot["checkpoint_path"]) == (root/CHECKPOINT).resolve(strict=True),"pilot frozen inputs differ from protocol")
    git_identity(); require_ignored_roots(root); require_fresh(result,logs); require_idle()
    initial_gpu=capture_gpu_snapshot()
    from step_preference.capacity_evidence import require_gpu_match,decode,GpuIdentity
    require_gpu_match(initial_gpu,decode(GpuIdentity,pilot["gpu"]),0)
    result.mkdir(); logs.mkdir()
    commands=[]
    execute(python(root,"-m","stage2.length_manifest","--data-path","data/TRAIN.jsonl","--image-root","data",
        "--processor-source",CHECKPOINT,"--latent-size","8","--supervise-latent-boundaries",
        "--max-sequence-length","4608","--assistant-plan-rendering","adjacent-text-space-lf-v1",
        "--manifest-path",result/"clean-length-manifest.json"),logs/"clean-length-manifest.log","clean_build",commands)
    from stage2.length_manifest import read_stage2_length_manifest
    from stage2.data import prepare_stage2_example
    from scripts.select_jsonl_rows import select_jsonl_rows
    length=read_stage2_length_manifest(result/"clean-length-manifest.json")
    require((length.stats.total,length.stats.retained,length.stats.filtered)==(3450,3354,96),"clean cohort counts differ")
    source=root/"data/TRAIN.jsonl"; clean=result/"clean-admissible.jsonl"
    select_jsonl_rows(source,clean,length.retained_indices)
    original_rows=admission._source_rows(read_bytes(source))
    ids=[prepare_stage2_example(original_rows[i],image_root=root/"data",latent_size=8,require_reasoning_units=True,
        assistant_plan_rendering="adjacent-text-space-lf-v1").sample_id for i in length.retained_indices]
    mapping=admission.SubsetSourceMap(1,"stage2b-subset-source-map",digest(read_bytes(source)),digest(read_bytes(clean)),length.retained_indices,tuple(ids))
    admission._write_exclusively(result/"clean-source-map.json",canonical(asdict(mapping)),"source map")
    require_idle()
    execute(python(root,root/"scripts/generate_step_negatives.py","--data_path",clean,"--image_root","data",
        "--stage2a_checkpoint_path",CHECKPOINT,"--output_path",result/"negatives.jsonl","--manifest_path",result/"negative-manifest.json",
        "--latent_size","8","--min_tokens","2","--max_new_tokens","32","--seed","42",
        "--attention_implementation","sdpa","--vision_attention_implementation","flash_attention_2","--device","cuda:0",cuda="0"),
        logs/"generate.log","negative_generation",commands)
    execute(python(root,"-m","step_preference.packed_admission","build","--original-data-path","data/TRAIN.jsonl",
        "--clean-data-path",clean,"--clean-length-manifest",result/"clean-length-manifest.json",
        "--clean-length-manifest-sha256",digest(read_bytes(result/"clean-length-manifest.json")),"--image-root","data",
        "--source-map-path",result/"clean-source-map.json","--source-map-sha256",digest(read_bytes(result/"clean-source-map.json")),
        "--checkpoint-path",CHECKPOINT,"--negative-path",result/"negatives.jsonl","--negative-manifest",result/"negative-manifest.json",
        "--max-packed-sequence-length","4608","--retained-output-path",result/"packed-retained.jsonl",
        "--manifest-output-path",result/"packed-admission.json"),logs/"packed-admission-build.log","admission_build",commands)
    manifest_sha=digest(read_bytes(result/"packed-admission.json"))
    execute(python(root,"-m","step_preference.packed_admission","verify","--manifest-path",result/"packed-admission.json",
        "--manifest-sha256",manifest_sha,"--output-path",result/"packed-admission-verification.json"),
        logs/"packed-admission-verify.log","admission_verify",commands)
    verification_sha=digest(read_bytes(result/"packed-admission-verification.json"))
    manifest=authenticated_admission(result/"packed-admission.json",manifest_sha,result/"packed-admission-verification.json",verification_sha)
    require(manifest.counts.retained > 0 and manifest.counts.filtered < 34,"packed exclusion capacity gate failed")
    by_source={r.source_index:r for r in manifest.records}
    witnesses=[(i,by_source[i].sample_id) for i in manifest.boundary_witnesses.final_probe_source_indices]
    require(1 <= len(witnesses) <= 8 and all(not any(ord(c)<32 or ord(c)==127 for c in sample) for _,sample in witnesses),"invalid witness/sample set")
    require_idle(); snapshot=capture_gpu_snapshot(); require_matching_snapshot(initial_gpu,snapshot)
    snapshot_sha=publish_json(result/"gpu-snapshot.json",snapshot,PackedGpuSnapshot)
    for source,sample in witnesses:
        require_fresh(result/f"witness-{source}-capacity.json",logs/f"witness-{source}-capacity.log")
        require_idle(); require_matching_snapshot(snapshot,capture_gpu_snapshot())
        execute(probe_command(root,identifier,source,sample,snapshot_sha),logs/f"witness-{source}-capacity.log","single_capacity",commands)
        report=read_json(result/f"witness-{source}-capacity.json",PackedCapacityReport)
        require(report.status == "pass" and report.sample.sample_id == sample and report.sample.source_row_index == source,
                "capacity witness did not pass with exact source identity")
    for source,sample in witnesses:
        witness=f"{identifier}-zero2-{source}"
        witness_result,witness_logs,witness_checkpoint=run_paths(root,witness)
        require_fresh(witness_result,witness_logs,witness_checkpoint,result/f"zero2-{source}-evidence.json")
        require_idle(); require_matching_snapshot(snapshot,capture_gpu_snapshot())
        # The launcher creates its exclusive log directory and captures LOG_FILE.
        command=distributed_command(root,identifier,source,sample,snapshot_sha,manifest_sha,verification_sha)
        execute_distributed(command,witness_logs/"train.log",commands)
        evidence=read_json(result/f"zero2-{source}-evidence.json",PackedDistributedCapacityEvidence)
        require(evidence.sample_id == sample and evidence.checkpoint_path == str(witness_checkpoint),"distributed witness source/output differs")
    command_sha=publish_json(result/"packed-command-log.json",PackedCommandLog(1,"stage2b-packed-command-log",commands),PackedCommandLog)
    gate=python(root,root/"scripts/validate_packed_capacity_gate.py","publish","--pilot-summary",Path(pilot_path).resolve(strict=True),
        "--pilot-summary-sha256",pilot_sha,"--gpu-snapshot",result/"gpu-snapshot.json","--gpu-snapshot-sha256",snapshot_sha,
        "--commands-path",result/"packed-command-log.json","--commands-sha256",command_sha,
        "--admission-manifest",result/"packed-admission.json","--admission-manifest-sha256",manifest_sha,
        "--verification",result/"packed-admission-verification.json","--verification-sha256",verification_sha)
    for source,_ in witnesses: gate += ["--capacity-report",str(result/f"witness-{source}-capacity.json")]
    for source,_ in witnesses: gate += ["--distributed-evidence",str(result/f"zero2-{source}-evidence.json")]
    for source,_ in witnesses: gate += ["--distributed-checkpoint",str(root/"checkpoints"/f"{identifier}-zero2-{source}")]
    gate += ["--output-path",str(result/"capacity-summary.json")]
    execute(gate,logs/"capacity-validator.log","capacity_verify")
    summary_sha=digest(read_bytes(result/"capacity-summary.json"))
    execute(python(root,root/"scripts/validate_packed_capacity_gate.py","verify","--summary",result/"capacity-summary.json",
        "--summary-sha256",summary_sha,"--gpu-snapshot",result/"gpu-snapshot.json","--gpu-snapshot-sha256",snapshot_sha),
        logs/"capacity-summary-verify.log","capacity_verify")
    print(f"ADMISSION_RESULT_PATH={result} CAPACITY_SUMMARY_SHA256={summary_sha}")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("run-id","pilot-summary","pilot-summary-sha256"): parser.add_argument("--"+flag,required=True)
    args=parser.parse_args(argv)
    try:
        run_protocol(Path(__file__).resolve().parents[1],args.run_id,args.pilot_summary,args.pilot_summary_sha256)
        return 0
    except (ValueError,OSError,subprocess.CalledProcessError) as error:
        print(str(error),file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
