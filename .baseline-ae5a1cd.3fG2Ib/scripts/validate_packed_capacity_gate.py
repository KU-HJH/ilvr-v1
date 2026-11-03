#!/usr/bin/env python3
"""Publish or transitively verify the one bounded v13 capacity decision."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from step_preference import packed_admission as admission
from step_preference.capacity_evidence import (
    StrictRecord, PackedCapacityReport, PackedDistributedCapacityEvidence,
    PackedCommandLog, PackedGpuSnapshot, GpuIdentity, TrainingRuntimeIdentity,
    read_json, read_json_with_hash, read_bytes, canonical, digest, decode, publish_json, require,
    adapter_files, training_length_manifest, git_identity, capture_gpu_snapshot,
    require_matching_snapshot, require_gpu_match, canonical_uuid,
)


@dataclass(frozen=True)
class ArtifactFile(StrictRecord):
    relative_path: str
    size: int
    sha256: str

    def validate(self):
        require(self.relative_path in ("adapter_config.json","adapter_model.safetensors","stage2_peft_metadata.json")
                and self.size > 0,"invalid adapter file identity")
        admission._require_sha256("adapter file",self.sha256)


@dataclass(frozen=True)
class WitnessReport(StrictRecord):
    source_index: int
    sample_id: str
    path: str
    sha256: str
    packed_padded_length: int
    peak_reserved_fraction: float
    status: str

    def validate(self):
        from step_preference.capacity_evidence import absolute
        absolute(self.path); admission._require_sha256("witness",self.sha256)
        require(self.source_index >= 0 and self.sample_id and self.packed_padded_length > 0
                and self.packed_padded_length <= 4608 and self.packed_padded_length % 128 == 0
                and 0 <= self.peak_reserved_fraction <= .85 and self.status == "pass","invalid single-device witness")


@dataclass(frozen=True)
class DistributedWitness(StrictRecord):
    source_index: int
    sample_id: str
    evidence_path: str
    evidence_sha256: str
    checkpoint_path: str
    checkpoint_files: list[ArtifactFile]
    training_runtime_identity_path: str
    training_runtime_identity_sha256: str
    status: str
    rank_peak_reserved_fractions: list[float]

    def validate(self):
        require(self.source_index >= 0 and self.sample_id and self.status == "pass"
                and len(self.rank_peak_reserved_fractions) == 2 and all(0 <= f <= .85 for f in self.rank_peak_reserved_fractions),
                "invalid distributed witness")
        require([f.relative_path for f in self.checkpoint_files] == ["adapter_config.json","adapter_model.safetensors","stage2_peft_metadata.json"],
                "distributed witness requires exactly three adapter files")


@dataclass(frozen=True)
class CapacityCounts(StrictRecord):
    original: int
    clean_admissible: int
    mapped: int
    retained: int
    filtered: int
    boundary_witnesses: int

    def validate(self):
        require(min(asdict(self).values()) >= 0,"capacity counts must be nonnegative")


@dataclass(frozen=True)
class Exclusion(StrictRecord):
    stage: str
    count: int
    reason: str
    source_indices: list[int]
    sample_ids: list[str]

    def validate(self):
        require(self.stage in ("clean_prefilter","packed_admission") and self.count >= 0 and self.reason
                and len(self.source_indices) == len(self.sample_ids) == self.count
                and self.source_indices == sorted(set(self.source_indices)) and all(self.sample_ids), "invalid exclusion identity")


@dataclass(frozen=True)
class PackedCapacitySummary(StrictRecord):
    schema_version: int
    kind: str
    status: str
    pilot_summary_path: str
    pilot_summary_sha256: str
    gpu_snapshot_path: str
    gpu_snapshot_sha256: str
    admission_manifest_path: str
    admission_manifest_sha256: str
    verification_path: str
    verification_sha256: str
    command_log_path: str
    command_log_sha256: str
    witness_reports: list[WitnessReport]
    distributed_witnesses: list[DistributedWitness]
    commands: list[list[str]]
    counts: CapacityCounts
    exclusions: list[Exclusion]
    gpu_identities: list[GpuIdentity]
    training_population: str
    errors: list[str]

    def validate(self):
        require(self.schema_version == 1 and self.kind == "stage2b-packed-capacity" and self.status in ("pass","fail")
                and self.training_population == "capacity_admitted_subset", "invalid capacity summary schema")
        require((self.status == "pass") == (self.errors == []) and len(self.errors) <= 1 and all(self.errors),"invalid capacity summary errors")
        if self.status == "fail":
            return
        require(self.counts.original == 3450 and self.counts.clean_admissible == self.counts.mapped == 3354
                and self.counts.retained + self.counts.filtered == 3354 and self.counts.filtered < 34
                and 1 <= self.counts.boundary_witnesses <= 8, "capacity population gate failed")
        single=[r.source_index for r in self.witness_reports]; distributed=[r.source_index for r in self.distributed_witnesses]
        require(single == distributed == sorted(set(single)) and len(single) == self.counts.boundary_witnesses,
                "capacity summary witness sets differ")
        require([e.stage for e in self.exclusions] == ["clean_prefilter","packed_admission"], "capacity exclusions incomplete")
        require(self.commands and all(c and all(c) for c in self.commands),"capacity command list empty")
        require(len(self.gpu_identities) == 2 and [g.uuid for g in self.gpu_identities] == sorted({g.uuid for g in self.gpu_identities}),
                "capacity GPU identities incomplete or unordered")


def match_witnesses(manifest, pairs):
    records={record.source_index:record.sample_id for record in manifest.records}
    expected=[(index,records[index]) for index in manifest.boundary_witnesses.final_probe_source_indices]
    require(len(pairs) == len(set(pairs)) and sorted(pairs) == expected,
            "witness set must equal complete manifest Pareto-plus-maximum set")
    return expected


def authenticate_pilot(path, expected):
    from scripts import publish_packed_v13_premise as premise
    raw=read_bytes(path)
    pilot=premise.decode_pilot_summary(raw)
    admission._authenticate(raw,expected,"pilot summary")
    require(pilot["gpu"]["name"] == "NVIDIA H100 NVL","pilot GPU must be H100 NVL")
    _, v11_sha, inventory, _ = premise._read_v11(pilot["rejected_v11_summary"]["path"],
        original=Path(pilot["original_data_path"]),original_sha=pilot["original_data_sha256"],checkpoint=Path(pilot["checkpoint_path"]))
    require(v11_sha == pilot["rejected_v11_summary"]["sha256"] and inventory == pilot["checkpoint_files"],"pilot rejected-v11 chain differs")
    for index,row in enumerate((1241,1314)):
        audit,capacity=premise._tail(Path(pilot["audits"][index]["path"]).parent,row,
            checkpoint=Path(pilot["checkpoint_path"]),checkpoint_files=inventory,original_sha=pilot["original_data_sha256"])
        require(audit == pilot["audits"][index] and capacity == pilot["capacity_reports"][index],"pilot boundary chain differs")
    return pilot


def authenticated_admission(manifest_path,manifest_sha,verification_path,verification_sha):
    from main import validate_packed_admission_inputs
    from step_preference.config import StepPreferenceConfig
    manifest,_=admission._strict_json(manifest_path,"admission manifest",admission._manifest_from_payload,manifest_sha)
    identity=manifest.identity
    config=StepPreferenceConfig(enabled=True,mode="stage2b",objective="legacy_dpo",packing_enabled=True,max_packed_sequence_length=4608,
        negative_path=identity.negative_path,negative_manifest=identity.negative_manifest_path,
        stage2a_checkpoint_path=identity.checkpoint_path,packed_admission_manifest=str(Path(manifest_path).resolve(strict=True)),
        packed_admission_manifest_sha256=manifest_sha,packed_admission_verification=str(Path(verification_path).resolve(strict=True)),
        packed_admission_verification_sha256=verification_sha)
    args=SimpleNamespace(data_path=identity.retained_data_path,image_root=identity.image_root_path,latent_size=8,
        max_seq_length_train=8192,supervise_latent_boundaries=True,max_train_samples=None)
    return validate_packed_admission_inputs(args,config)


def validate_runtime(path,expected,manifest,*,run_id,triplet):
    runtime=read_json(path,TrainingRuntimeIdentity,expected)
    require(runtime.git_commit == manifest.identity.git_commit and runtime.run_id == run_id
            and runtime.coefficient_triplet == triplet,"runtime implementation/run/objective identity differs")
    require(runtime.admission_manifest_sha256 == digest(canonical(asdict(manifest))),"runtime admission hash differs")
    training_length_manifest(runtime.training_length_manifest_path,runtime.training_length_manifest_sha256,manifest.identity)
    return runtime


def build_summary(args):
    manifest=authenticated_admission(args.admission_manifest,args.admission_manifest_sha256,args.verification,args.verification_sha256)
    pilot=authenticate_pilot(args.pilot_summary,args.pilot_summary_sha256)
    identity=manifest.identity
    require(pilot["original_data_path"] == identity.original_data_path and pilot["original_data_sha256"] == identity.original_data_sha256
            and pilot["checkpoint_path"] == identity.checkpoint_path
            and pilot["checkpoint_files"] == [asdict(item) for item in identity.checkpoint_files],"pilot/admission source or checkpoint identity differs")
    require(git_identity() == identity.git_commit,"capacity implementation commit differs")
    snapshot=read_json(args.gpu_snapshot,PackedGpuSnapshot,args.gpu_snapshot_sha256)
    require_matching_snapshot(snapshot,capture_gpu_snapshot())
    require_gpu_match(snapshot,decode(GpuIdentity,pilot["gpu"]),0)
    commands=read_json(args.commands_path,PackedCommandLog,args.commands_sha256)
    for paths in (args.capacity_report,args.distributed_evidence,args.distributed_checkpoint):
        resolved=[str(Path(path).resolve(strict=True)) for path in paths]
        require(len(resolved) == len(set(resolved)),"duplicate evidence/checkpoint path")
    require(len(args.distributed_evidence) == len(args.distributed_checkpoint),"distributed evidence/checkpoint arrays differ")
    records={row.source_index:row for row in manifest.records}
    reports=[]
    for path in args.capacity_report:
        report,report_sha=read_json_with_hash(path,PackedCapacityReport)
        require(report.status == "pass","single-device capacity report did not pass")
        record=records.get(report.sample.source_row_index)
        require(record is not None and (report.sample.sample_id,report.sample.clean_length,report.sample.step_count,
                report.sample.branch_count,report.sample.packed_unpadded_length,report.sample.packed_padded_length)
                == (record.sample_id,record.clean_length,record.reasoning_block_count,record.branch_count,
                    record.packed_unpadded_length,record.packed_padded_length),"capacity sample/length differs from manifest")
        require(report.identity.checkpoint_path == identity.checkpoint_path and report.identity.git_commit == identity.git_commit
                and report.identity.gpu_snapshot_path == str(Path(args.gpu_snapshot).resolve(strict=True))
                and report.identity.gpu_snapshot_sha256 == args.gpu_snapshot_sha256,"capacity report implementation/snapshot differs")
        require_gpu_match(snapshot,report.gpu,0)
        reports.append(WitnessReport(record.source_index,record.sample_id,str(Path(path).resolve(strict=True)),report_sha,
            record.packed_padded_length,report.memory.peak_reserved_fraction,"pass"))
    reports.sort(key=lambda r:r.source_index)
    match_witnesses(manifest,[(r.source_index,r.sample_id) for r in reports])
    checkpoints={str(admission._directory(path,"distributed checkpoint")):path for path in args.distributed_checkpoint}
    distributed=[]; consumed=set()
    by_sample={row.sample_id:row for row in manifest.records}
    for path in args.distributed_evidence:
        evidence,evidence_sha=read_json_with_hash(path,PackedDistributedCapacityEvidence)
        require(evidence.checkpoint_path in checkpoints and evidence.checkpoint_path not in consumed,"distributed checkpoint does not match source evidence")
        consumed.add(evidence.checkpoint_path)
        record=by_sample.get(evidence.sample_id)
        require(record is not None,"distributed sample is unknown")
        require(evidence.run_id.endswith(f"-zero2-{record.source_index}"),"distributed checkpoint source identity differs")
        require(evidence.gpu_snapshot_path == str(Path(args.gpu_snapshot).resolve(strict=True))
                and evidence.gpu_snapshot_sha256 == args.gpu_snapshot_sha256,"distributed snapshot differs")
        for gpu in evidence.gpu_identities:
            require_gpu_match(snapshot,gpu,gpu.rank)
        files=adapter_files(evidence.checkpoint_path)
        runtime=validate_runtime(evidence.training_runtime_identity_path,evidence.training_runtime_identity_sha256,
            manifest,run_id=evidence.run_id,triplet=[1,1,.1])
        require(runtime.verification_sha256 == args.verification_sha256 and runtime.global_step == runtime.optimizer_step == 2
                and runtime.positive_pre_step_gradient_count == 2
                and all(r.pre_step_gradient_norms == runtime.first_positive_pre_step_gradient_norms for r in evidence.rank_observations),
                "distributed runtime update/verification identity differs")
        distributed.append(DistributedWitness(record.source_index,record.sample_id,str(Path(path).resolve(strict=True)),evidence_sha,
            evidence.checkpoint_path,[decode(ArtifactFile,item) for item in files],evidence.training_runtime_identity_path,
            evidence.training_runtime_identity_sha256,"pass",[r.peak_reserved_fraction for r in evidence.rank_observations]))
    distributed.sort(key=lambda r:r.source_index)
    match_witnesses(manifest,[(r.source_index,r.sample_id) for r in distributed])
    require(consumed == set(checkpoints),"unused distributed checkpoint")
    from stage2.length_manifest import read_stage2_length_manifest
    from stage2.data import prepare_stage2_example
    clean=read_stage2_length_manifest(identity.clean_length_manifest_path)
    original=admission._source_rows(read_bytes(identity.original_data_path,identity.original_data_sha256))
    clean_indices=set(clean.retained_indices)
    excluded=[index for index in range(len(original)) if index not in clean_indices]
    excluded_ids=[prepare_stage2_example(original[i],image_root=Path(identity.image_root_path),latent_size=8,
        require_reasoning_units=True,assistant_plan_rendering="adjacent-text-space-lf-v1").sample_id for i in excluded]
    exclusions=[Exclusion("clean_prefilter",len(excluded),"clean_length_exceeds_limit",excluded,excluded_ids),
        Exclusion("packed_admission",manifest.counts.filtered,"packed_length_exceeds_limit",list(manifest.filtered_source_indices),list(manifest.filtered_sample_ids))]
    gpu_ids=[GpuIdentity(g.host_physical_index,g.name,g.uuid,g.total_memory_bytes) for g in snapshot.gpus]
    return PackedCapacitySummary(1,"stage2b-packed-capacity","pass",str(Path(args.pilot_summary).resolve(strict=True)),args.pilot_summary_sha256,
        str(Path(args.gpu_snapshot).resolve(strict=True)),args.gpu_snapshot_sha256,str(Path(args.admission_manifest).resolve(strict=True)),args.admission_manifest_sha256,
        str(Path(args.verification).resolve(strict=True)),args.verification_sha256,str(Path(args.commands_path).resolve(strict=True)),args.commands_sha256,
        reports,distributed,commands.commands,CapacityCounts(**asdict(manifest.counts),boundary_witnesses=len(reports)),exclusions,
        sorted(gpu_ids,key=lambda g:g.uuid),"capacity_admitted_subset",[])


def summary_arguments(summary):
    return SimpleNamespace(pilot_summary=summary.pilot_summary_path,pilot_summary_sha256=summary.pilot_summary_sha256,
        gpu_snapshot=summary.gpu_snapshot_path,gpu_snapshot_sha256=summary.gpu_snapshot_sha256,
        admission_manifest=summary.admission_manifest_path,admission_manifest_sha256=summary.admission_manifest_sha256,
        verification=summary.verification_path,verification_sha256=summary.verification_sha256,
        commands_path=summary.command_log_path,commands_sha256=summary.command_log_sha256,
        capacity_report=[r.path for r in summary.witness_reports],distributed_evidence=[r.evidence_path for r in summary.distributed_witnesses],
        distributed_checkpoint=[r.checkpoint_path for r in summary.distributed_witnesses])


def verify_summary(path,expected,gpu_snapshot=None,gpu_snapshot_sha256=None):
    summary=read_json(path,PackedCapacitySummary,expected)
    require(summary.status == "pass","capacity summary did not pass")
    if gpu_snapshot is not None:
        require(str(Path(gpu_snapshot).resolve(strict=True)) == summary.gpu_snapshot_path
                and gpu_snapshot_sha256 == summary.gpu_snapshot_sha256,"requested live snapshot differs")
    rebuilt=build_summary(summary_arguments(summary))
    require(summary == rebuilt,"capacity summary differs from reopened witnesses")
    return summary


def _parser():
    parser=argparse.ArgumentParser(description=__doc__); subs=parser.add_subparsers(dest="command",required=True)
    publish=subs.add_parser("publish")
    for name in ("admission-manifest","pilot-summary","gpu-snapshot","commands-path","verification"):
        publish.add_argument("--"+name,required=True)
        hash_name="commands-sha256" if name == "commands-path" else name+"-sha256"
        publish.add_argument("--"+hash_name,required=True)
    for name in ("capacity-report","distributed-evidence","distributed-checkpoint"):
        publish.add_argument("--"+name,action="append",required=True)
    publish.add_argument("--output-path",required=True)
    verify=subs.add_parser("verify")
    for name in ("summary","summary-sha256","gpu-snapshot","gpu-snapshot-sha256"):
        verify.add_argument("--"+name,required=True)
    return parser


def main(argv=None):
    args=_parser().parse_args(argv)
    try:
        if args.command == "verify":
            verify_summary(args.summary,args.summary_sha256,args.gpu_snapshot,args.gpu_snapshot_sha256)
            print("VERIFIED_CAPACITY_V13 status=pass cap=4608")
        else:
            require(not Path(args.output_path).exists() and not Path(args.output_path).is_symlink(),"capacity summary output already exists")
            try:
                result=build_summary(args)
            except (ValueError,OSError,TypeError) as error:
                # Only independently supplied references survive a failed gate.
                result=PackedCapacitySummary(1,"stage2b-packed-capacity","fail",
                    str(Path(args.pilot_summary).resolve()),args.pilot_summary_sha256,
                    str(Path(args.gpu_snapshot).resolve()),args.gpu_snapshot_sha256,
                    str(Path(args.admission_manifest).resolve()),args.admission_manifest_sha256,
                    str(Path(args.verification).resolve()),args.verification_sha256,
                    str(Path(args.commands_path).resolve()),args.commands_sha256,
                    [],[],[],CapacityCounts(0,0,0,0,0,0),[],[],"capacity_admitted_subset",[str(error) or type(error).__name__])
                publish_json(args.output_path,result,PackedCapacitySummary)
                raise
            publish_json(args.output_path,result,PackedCapacitySummary)
        return 0
    except (ValueError,OSError,TypeError) as error:
        print(str(error),file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
