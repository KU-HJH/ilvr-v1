#!/usr/bin/env python3
"""Freeze, validate, publish and verify the three v13 DoRA ablation artifacts."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from step_preference import packed_admission as admission
from step_preference.capacity_evidence import (
    StrictRecord, TrainingRuntimeIdentity, EvaluationRuntimeIdentity, PackedGpuSnapshot,
    read_bytes, read_json, read_json_with_hash, decode, digest, canonical, publish_json, require, run_id,
    adapter_files, git_identity, require_ignored_roots, require_idle, capture_gpu_snapshot,
    require_matching_snapshot, training_length_manifest,
)
from scripts.validate_packed_capacity_gate import verify_summary, authenticated_admission, validate_runtime
from evaluation.scoring import EvaluationScoreSummary, score_answer


ARMS={"ce":[1,0,0],"dpo":[0,1,0],"entropy":[0,0,1]}
NORMALIZE={"--save_model_path":"<CHECKPOINT_DIR>","--log_file":"<LOG_FILE>",
    "--stage2_length_manifest":"<LENGTH_MANIFEST>","--ce_weight":"<CE_WEIGHT>",
    "--step_dpo_weight":"<DPO_WEIGHT>","--step_entropy_weight":"<ENTROPY_WEIGHT>"}
TEST_SHA="6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618"


@dataclass(frozen=True)
class LaunchConfig(StrictRecord):
    schema_version: int
    arm: str
    run_id: str
    coefficient_triplet: list[float]
    normalized_command: list[str]
    capacity_summary_path: str
    capacity_summary_sha256: str
    retained_data_sha256: str
    negative_sha256: str
    negative_manifest_sha256: str
    admission_manifest_sha256: str
    verification_sha256: str

    def validate(self):
        require(self.schema_version == 1 and self.arm in ARMS and self.coefficient_triplet == ARMS[self.arm],"invalid launch arm/triplet")
        run_id(self.run_id,f"step-pref-dora-{self.arm}-")
        require(self.normalized_command and all(self.normalized_command),"launch command must be a nonempty string array")
        for option,value in NORMALIZE.items():
            require(option_value(self.normalized_command,option) == value,"launch command was not normalized exactly")
        require(not any(self.run_id in token for token in self.normalized_command),"launch run ID remains unnormalized")


@dataclass(frozen=True)
class ComparisonRun(StrictRecord):
    rank: int
    arm: str
    run_id: str
    launch_config_sha256: str
    checkpoint_sha256: dict[str,str]
    training_runtime_identity_sha256: str
    predictions_sha256: str
    evaluation_runtime_identity_sha256: str
    coefficient_triplet: list[float]
    correct: int
    source_count: int
    coverage_accuracy: float
    valid_gold_accuracy: float
    invalid_gold_count: int
    average_inference_time_sec: float

    def validate(self):
        require(self.arm in ARMS and self.coefficient_triplet == ARMS[self.arm],"comparison arm/triplet differs")
        run_id(self.run_id,f"step-pref-dora-{self.arm}-")
        require(set(self.checkpoint_sha256) == {"adapter_config.json","adapter_model.safetensors","stage2_peft_metadata.json"},"comparison adapter mapping differs")
        for value in self.checkpoint_sha256.values(): admission._require_sha256("adapter",value)
        require(1 <= self.rank <= 3 and self.source_count == 400 and 0 <= self.correct <= 400
                and 0 <= self.invalid_gold_count <= 400 and self.correct <= 400-self.invalid_gold_count
                and self.coverage_accuracy == self.correct/400 and 0 <= self.valid_gold_accuracy <= 1
                and self.average_inference_time_sec >= 0,"comparison metric is invalid")


@dataclass(frozen=True)
class Comparison(StrictRecord):
    schema_version: int
    kind: str
    primary_metric: str
    training_population: str
    runs: list[ComparisonRun]

    def validate(self):
        require((self.schema_version,self.kind,self.primary_metric,self.training_population)
                == (1,"stage2b-dora-ablation-comparison","coverage_accuracy","capacity_admitted_subset"),"invalid comparison schema")
        require(len(self.runs) == 3 and {r.arm for r in self.runs} == set(ARMS),"comparison requires exactly three arms")
        values=[asdict(r) for r in self.runs]
        require(values == rank_runs(values),"comparison ranks/order were not recomputed")


@dataclass(frozen=True)
class EvaluationSample(StrictRecord):
    index: int
    task_name: str
    image_input: str | list[str]
    prompt_identity: str
    prompt_token_length: int
    generated_suffix_token_ids: list[int]
    raw_output: str
    assistant_only: str
    prediction: str
    extracted_final_answer: str
    prediction_parse_status: str
    prediction_parse_detail: str | None
    prediction_normalized: str
    gold_final_answer: str | None
    gold_parse_status: str
    gold_parse_detail: str | None
    gold_normalized: str
    invalid_gold_reason: str | None
    latent_start_count: int
    latent_block_count: int
    stop_reason: str
    match: bool
    inference_time_sec: float

    def validate(self):
        require(0 <= self.index < 400 and self.task_name == "comt" and self.prompt_identity
                and self.prompt_token_length >= 0 and all(i >= 0 for i in self.generated_suffix_token_ids)
                and self.latent_start_count >= 0 and self.latent_block_count >= 0 and self.inference_time_sec >= 0,
                "invalid evaluation observation")
        require(self.prediction == self.assistant_only and self.prediction_parse_status in ("parsed","empty_prediction","malformed")
                and self.gold_parse_status in ("parsed","invalid_gold","malformed"),"invalid evaluation parse status")
        require((self.prediction_parse_detail is not None) == (self.prediction_parse_status == "malformed"),"prediction parse detail differs")


@dataclass(frozen=True)
class InvalidGold(StrictRecord):
    index: int
    reason: str

    def validate(self):
        require(self.index >= 0 and self.reason,"invalid gold reason missing")


@dataclass(frozen=True)
class EvaluationSummary(StrictRecord):
    summary: bool
    num_samples: int
    average_inference_time_sec: float
    total_inference_time_sec: float
    accuracy: float
    success: int
    correct: int
    source_count: int
    valid_gold_count: int
    invalid_gold_count: int
    invalid_gold: list[InvalidGold]
    coverage_accuracy: float
    valid_gold_accuracy: float

    def validate(self):
        require(self.summary and self.num_samples == self.source_count == 400
                and self.success == self.correct and self.valid_gold_count+self.invalid_gold_count == 400
                and min(self.average_inference_time_sec,self.total_inference_time_sec,self.accuracy,
                        self.coverage_accuracy,self.valid_gold_accuracy) >= 0,"invalid evaluation summary")


def option_value(command,option):
    require(command.count(option) == 1,f"launcher requires exactly one {option}")
    index=command.index(option)
    require(index+1 < len(command) and not command[index+1].startswith("--"),f"launcher option {option} lacks value")
    return command[index+1]


def normalize_command(raw,identifier,triplet,identifiers):
    command=admission._parse_json(raw.encode("utf-8"),"launcher dry-run")
    require(type(command) is list and command and all(type(token) is str and token for token in command),"dry-run must be one JSON string array")
    for option,value in zip(("--ce_weight","--step_dpo_weight","--step_entropy_weight"),triplet):
        parsed=float(option_value(command,option))
        require(math.isfinite(parsed) and parsed == value,"launcher coefficient differs")
    normalized=list(command)
    for option,placeholder in NORMALIZE.items():
        option_value(command,option)
        normalized[normalized.index(option)+1]=placeholder
    require(not any(identifier in token for identifier in identifiers for token in normalized),"run ID or output path remains unnormalized")
    return normalized


def rank_runs(runs):
    require(len(runs) == 3 and {r["arm"] for r in runs} == set(ARMS),"ranking requires exactly three arms")
    ranked=sorted((dict(r) for r in runs),key=lambda r:(-r["coverage_accuracy"],r["arm"]))
    for record in ranked:
        record["rank"]=1+sum(other["coverage_accuracy"] > record["coverage_accuracy"] for other in ranked)
    return ranked


def comparison_markdown(runs):
    lines=["Training comparison applies only to the capacity-admitted subset; TEST evaluation covers all 400 rows.","",
        "Capacity witnesses are an empirical gate; they do not prove memory bounds for the cohort or two epochs.","",
        "| Rank | Arm | Correct / 400 | Coverage accuracy | Valid-gold accuracy | Invalid gold | Average inference (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        lines.append(f"| {r['rank']} | {r['arm']} | {r['correct']} / 400 | {r['coverage_accuracy']:.6f} | {r['valid_gold_accuracy']:.6f} | {r['invalid_gold_count']} | {r['average_inference_time_sec']:.6f} |")
    return "\n".join(lines)+"\n"


def validate_evaluation(predictions,test_data,*,with_identity=False,test_data_sha256=None):
    raw=read_bytes(predictions)
    lines=raw.splitlines(keepends=True)
    require(len(lines) == 401,"evaluation requires exactly 400 samples plus summary")
    source=admission._source_rows(read_bytes(test_data,test_data_sha256))
    require(len(source) == 400,"TEST must contain exactly 400 rows")
    rows=[]
    for index,line in enumerate(lines[:400]):
        payload=admission._parse_json(line,"evaluation row")
        require("error" not in payload and payload.get("prediction_parse_status") != "inference_error","evaluation contains inference errors")
        value=decode(EvaluationSample,payload)
        require(line == canonical(payload),"evaluation row must be canonical JSON")
        require(value.index == index and value.gold_final_answer == source[index].get("original_final_answer")
                and value.image_input == source[index].get("image_input",[]),"evaluation source identity differs")
        score=score_answer(value.assistant_only,source[index].get("original_final_answer"))
        expected=dict(extracted_final_answer=score.prediction.extracted,prediction_parse_status=score.prediction.status,
            prediction_parse_detail=score.prediction.detail,prediction_normalized=score.prediction.normalized,
            gold_parse_status=score.gold.status,gold_parse_detail=score.gold.detail,gold_normalized=score.gold.normalized,
            invalid_gold_reason=score.invalid_gold_reason,match=score.match)
        require(all(payload[name] == observation for name,observation in expected.items()),"evaluation scores/parse claims differ")
        rows.append(payload)
    summary_payload=admission._parse_json(lines[-1],"evaluation summary")
    summary=decode(EvaluationSummary,summary_payload)
    require(lines[-1] == canonical(summary_payload),"evaluation summary must be canonical JSON")
    score=EvaluationScoreSummary.from_rows(rows,400)
    total=sum(r["inference_time_sec"] for r in rows)
    expected=dict(summary=True,num_samples=400,average_inference_time_sec=total/400,total_inference_time_sec=total,
        accuracy=score.coverage_accuracy,success=score.correct,**asdict(score))
    require(summary_payload == expected,"evaluation summary differs from independently recomputed metrics")
    return (summary,digest(raw)) if with_identity else summary


def comparison_directory(path,root,*,fresh=False):
    path=Path(path)
    run_id(path.name,"step-pref-dora-comparison-")
    require(path.is_absolute() and not path.is_symlink(),"comparison directory must be absolute and not a symlink")
    resolved=path.resolve(strict=not fresh)
    resolved.relative_to((root/"results").resolve(strict=True))
    require(resolved.parent == (root/"results").resolve(strict=True),"comparison must be directly under results")
    if fresh:
        require(not path.exists(),"comparison directory already exists")
    else:
        require(path.is_dir(),"comparison directory is missing")
    return resolved


def arm_paths(root,identifier,arm):
    run_id(identifier,f"step-pref-dora-{arm}-")
    paths={name:(root/name/identifier).resolve(strict=False) for name in ("checkpoints","results","logs")}
    for name,path in paths.items():
        path.relative_to((root/name).resolve(strict=True))
        require(not (root/name/identifier).is_symlink(),"arm directory cannot be a symlink")
    return paths


def capacity_and_manifest(args):
    summary=verify_summary(args.capacity_summary,args.capacity_summary_sha256)
    manifest=authenticated_admission(summary.admission_manifest_path,summary.admission_manifest_sha256,
        summary.verification_path,summary.verification_sha256)
    return summary,manifest


def shared_launch_fields(args,summary,manifest):
    identity=manifest.identity
    return dict(capacity_summary_path=str(Path(args.capacity_summary).resolve(strict=True)),capacity_summary_sha256=args.capacity_summary_sha256,
        retained_data_sha256=identity.retained_data_sha256,negative_sha256=identity.negative_sha256,
        negative_manifest_sha256=identity.negative_manifest_sha256,admission_manifest_sha256=summary.admission_manifest_sha256,
        verification_sha256=summary.verification_sha256)


def validate_launch_tokens(command,paths,triplet,summary,identity):
    checks={"--save_model_path":paths["checkpoints"],"--log_file":paths["logs"]/"train.log",
        "--stage2_length_manifest":paths["results"]/"stage2-length-manifest.json",
        "--data_path":identity.retained_data_path,"--step_negative_path":identity.negative_path,
        "--step_negative_manifest":identity.negative_manifest_path,"--step_packed_admission_manifest":summary.admission_manifest_path,
        "--step_packed_admission_verification":summary.verification_path,"--stage2a_checkpoint_path":identity.checkpoint_path,
        "--load_model_path":identity.checkpoint_path,"--image_root":identity.image_root_path}
    for option,expected in checks.items():
        require(Path(option_value(command,option)).resolve(strict=False) == Path(expected),f"launcher {option} path differs")
    fixed={"--step_packed_admission_manifest_sha256":summary.admission_manifest_sha256,
        "--step_packed_admission_verification_sha256":summary.verification_sha256,"--stage2_length_manifest_sha256":"0"*64,
        "--step_preference_mode":"stage2b","--step_max_packed_sequence_length":"4608","--tuning_method":"dora",
        "--lora_r":"32","--lora_alpha":"64","--lora_dropout":"0.05","--seed":"42",
        "--per_device_train_batch_size":"1","--gradient_accumulation_steps":"2","--dataloader_num_workers":"0",
        "--attention_implementation":"sdpa","--vision_attention_implementation":"flash_attention_2",
        "--max_seq_length_train":"8192","--epochs":"2","--warm_up_steps":"10","--save_steps":"612"}
    fixed.update({"--stage":"stage2","--latent_size":"8","--learning_rate":"5e-5","--step_dpo_beta":"0.1",
        "--step_entropy_margin":"0.1","--lora_target_modules":"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--num_processes":"2"})
    for option,expected in fixed.items():
        require(option_value(command,option) == expected,f"launcher {option} setting differs")
    root=paths["results"].parent.parent
    require(Path(option_value(command,"--deepspeed_config")).resolve(strict=False) == root/"configs/deepspeed_zero2.json",
        "launcher ZeRO-2 config differs")
    for flag in ("--step_preference_enabled","--step_preference_packing","--gradient_checkpointing","--supervise_latent_boundaries"):
        require(command.count(flag) == 1,f"launcher lacks {flag}")
    require(not any(token.startswith("--step_packed_capacity") or token.startswith("--step_packed_gpu")
                    or token in ("--resume_from_checkpoint_path","--max_steps") for token in command),"normal arm has capacity/resume/step overrides")


def freeze_launches(args,root):
    summary,manifest=capacity_and_manifest(args)
    require_ignored_roots(root)
    directory=comparison_directory(args.comparison_dir,root,fresh=True)
    identifiers=[getattr(args,arm+"_run_id") for arm in ARMS]
    configs=[]
    for arm,triplet in ARMS.items():
        identifier=getattr(args,arm+"_run_id")
        paths=arm_paths(root,identifier,arm)
        require(all(not path.exists() for path in paths.values()),"arm target already exists")
        raw=getattr(args,arm+"_command_json")
        normalized=normalize_command(raw,identifier,triplet,identifiers)
        validate_launch_tokens(admission._parse_json(raw.encode(),"launcher"),paths,triplet,summary,manifest.identity)
        configs.append(LaunchConfig(1,arm,identifier,triplet,normalized,**shared_launch_fields(args,summary,manifest)))
    require(all(config.normalized_command == configs[0].normalized_command for config in configs),"shared launcher tokens drift")
    directory.mkdir()
    for config in configs:
        publish_json(directory/f"launch-config-{config.arm}.json",config,LaunchConfig)
    return configs


def read_configs(args,root,summary,manifest):
    directory=comparison_directory(args.comparison_dir,root)
    require({path.name for path in directory.glob("launch-config-*.json")} == {f"launch-config-{arm}.json" for arm in ARMS},"missing or extra launch arm")
    configs={arm:read_json(directory/f"launch-config-{arm}.json",LaunchConfig) for arm in ARMS}
    shared=shared_launch_fields(args,summary,manifest)
    for arm,config in configs.items():
        require(config.arm == arm and all(getattr(config,key) == value for key,value in shared.items()),"launch config shared identity differs")
        paths=arm_paths(root,config.run_id,arm)
        replacements={"<CHECKPOINT_DIR>":str(paths["checkpoints"]),"<LOG_FILE>":str(paths["logs"]/"train.log"),
            "<LENGTH_MANIFEST>":str(paths["results"]/"stage2-length-manifest.json"),
            **{key:str(value) for key,value in zip(("<CE_WEIGHT>","<DPO_WEIGHT>","<ENTROPY_WEIGHT>"),ARMS[arm])}}
        command=[replacements.get(token,token) for token in config.normalized_command]
        validate_launch_tokens(command,paths,ARMS[arm],summary,manifest.identity)
    require(all(c.normalized_command == configs["ce"].normalized_command for c in configs.values()),"normalized launcher tokens drift")
    return directory,configs


def validate_arm(args,root,summary,manifest,configs):
    require(args.arm in ARMS,"unknown or missing arm")
    config=configs[args.arm]
    paths=arm_paths(root,config.run_id,args.arm)
    files=adapter_files(paths["checkpoints"])
    runtime_path=paths["checkpoints"]/"stage2_training_runtime_identity.json"
    runtime_sha=digest(read_bytes(runtime_path))
    runtime=validate_runtime(runtime_path,runtime_sha,manifest,run_id=config.run_id,triplet=ARMS[args.arm])
    require(runtime.verification_sha256 == config.verification_sha256,"runtime verifier hash differs")
    require(runtime.training_length_manifest_path == str(paths["results"]/"stage2-length-manifest.json"),"runtime sampler path differs")
    if args.phase == "training":
        return runtime
    require(args.test_data and args.test_data_sha256 == TEST_SHA,"evaluation requires frozen TEST SHA-256")
    predictions=paths["results"]/"evaluation/predictions.jsonl"
    evaluation_path=paths["results"]/"evaluation/evaluation-runtime-identity.json"
    evaluation,evaluation_sha=read_json_with_hash(evaluation_path,EvaluationRuntimeIdentity)
    require(evaluation.git_commit == runtime.git_commit and evaluation.model_dir == str(paths["checkpoints"]),"evaluation runtime model/commit differs")
    metrics,predictions_sha=validate_evaluation(predictions,args.test_data,with_identity=True,test_data_sha256=args.test_data_sha256)
    reopened_config,config_sha=read_json_with_hash(Path(args.comparison_dir)/f"launch-config-{args.arm}.json",LaunchConfig)
    require(reopened_config == config,"launch config changed during validation")
    return dict(rank=1,arm=args.arm,run_id=config.run_id,launch_config_sha256=config_sha,
        checkpoint_sha256={f["relative_path"]:f["sha256"] for f in files},training_runtime_identity_sha256=runtime_sha,
        predictions_sha256=predictions_sha,evaluation_runtime_identity_sha256=evaluation_sha,
        coefficient_triplet=ARMS[args.arm],correct=metrics.correct,source_count=400,coverage_accuracy=metrics.coverage_accuracy,
        valid_gold_accuracy=metrics.valid_gold_accuracy,invalid_gold_count=metrics.invalid_gold_count,
        average_inference_time_sec=metrics.average_inference_time_sec)


def build_comparison(args,root):
    from types import SimpleNamespace
    summary,manifest=capacity_and_manifest(args)
    directory,configs=read_configs(args,root,summary,manifest)
    runs=[]; length_hashes=set()
    for arm in ARMS:
        arm_args=SimpleNamespace(**{**vars(args),"arm":arm,"phase":"evaluation"})
        runs.append(validate_arm(arm_args,root,summary,manifest,configs))
        paths=arm_paths(root,configs[arm].run_id,arm)
        runtime=read_json(paths["checkpoints"]/"stage2_training_runtime_identity.json",TrainingRuntimeIdentity)
        length_hashes.add(runtime.training_length_manifest_sha256)
    require(len(length_hashes) == 1,"cross-arm training length-manifest hashes differ")
    ranked=rank_runs(runs)
    return directory,Comparison(1,"stage2b-dora-ablation-comparison","coverage_accuracy","capacity_admitted_subset",
        [decode(ComparisonRun,r) for r in ranked]),comparison_markdown(ranked)


def preflight(args,root):
    require((args.phase == "freeze" and args.arm is None) or (args.phase in ("training","evaluation") and args.arm in ARMS),"phase/arm preflight differs")
    summary,manifest=capacity_and_manifest(args)
    require(git_identity() == manifest.identity.git_commit,"ablation implementation commit differs")
    require_ignored_roots(root)
    require_idle((0,) if args.phase == "evaluation" else (0,1))
    require_matching_snapshot(read_json(summary.gpu_snapshot_path,PackedGpuSnapshot,summary.gpu_snapshot_sha256),capture_gpu_snapshot())


def _parser():
    parser=argparse.ArgumentParser(description=__doc__); subs=parser.add_subparsers(dest="command",required=True)
    for name in ("freeze-launches","publish-comparison","validate-arm","verify-comparison","preflight"):
        sub=subs.add_parser(name)
        sub.add_argument("--capacity-summary",required=True); sub.add_argument("--capacity-summary-sha256",required=True)
        if name != "preflight": sub.add_argument("--comparison-dir",required=True)
        if name == "freeze-launches":
            for arm in ARMS:
                sub.add_argument(f"--{arm}-run-id",required=True); sub.add_argument(f"--{arm}-command-json",required=True)
        if name in ("publish-comparison","verify-comparison","validate-arm"):
            sub.add_argument("--test-data",required=name != "validate-arm")
            sub.add_argument("--test-data-sha256",required=name != "validate-arm")
        if name in ("preflight","validate-arm"):
            sub.add_argument("--arm",choices=ARMS,required=name == "validate-arm")
            sub.add_argument("--phase",choices=("freeze","training","evaluation") if name == "preflight" else ("training","evaluation"),required=True)
    return parser


def main(argv=None):
    args=_parser().parse_args(argv); root=Path(__file__).resolve().parents[1]
    try:
        if args.command == "preflight": preflight(args,root)
        elif args.command == "freeze-launches": freeze_launches(args,root)
        elif args.command == "validate-arm":
            summary,manifest=capacity_and_manifest(args)
            _,configs=read_configs(args,root,summary,manifest)
            validate_arm(args,root,summary,manifest,configs)
        else:
            directory,comparison,markdown=build_comparison(args,root)
            if args.command == "publish-comparison":
                require(all(not (directory/name).exists() and not (directory/name).is_symlink() for name in ("comparison.json","comparison.md")),"comparison output already exists")
                publish_json(directory/"comparison.json",comparison,Comparison)
                admission._write_exclusively(directory/"comparison.md",markdown.encode(),"comparison Markdown")
            else:
                require(read_json(directory/"comparison.json",Comparison) == comparison
                        and read_bytes(directory/"comparison.md") == markdown.encode(),"published comparison differs from reopened artifacts")
                print("VERIFIED_DORA_ABLATION arms=3 test_rows=400")
        return 0
    except (ValueError,OSError,TypeError) as error:
        print(str(error),file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
