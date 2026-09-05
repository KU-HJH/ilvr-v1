"""Exact v13 capacity/runtime wire records and their bounded publishers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import types
from typing import get_args, get_origin, get_type_hints
import uuid

from step_preference import packed_admission as admission


RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,127}\Z", re.ASCII)
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "stage2_peft_metadata.json")
LANGUAGE_CLASS = "Qwen2_5_VLSdpaAttention"
VISION_CLASS = "Qwen2_5_VLVisionFlashAttention2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run_id(value, prefix=""):
    require(type(value) is str and RUN_ID.fullmatch(value) and value.startswith(prefix), "invalid run ID")
    return value


def absolute(value):
    require(type(value) is str and value and Path(value).is_absolute()
            and str(Path(value).resolve(strict=False)) == value, "path must be resolved absolute")
    return value


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    if is_dataclass(value):
        value = asdict(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def decode(cls, value):
    """Decode only declared dataclass fields, with no scalar coercion."""
    if get_origin(cls) is types.UnionType:
        for option in get_args(cls):
            try:
                return decode(option, value)
            except (ValueError, TypeError):
                pass
        raise ValueError("invalid nullable field")
    if get_origin(cls) is list:
        require(type(value) is list, "expected JSON array")
        return [decode(get_args(cls)[0], item) for item in value]
    if get_origin(cls) is dict:
        require(type(value) is dict, "expected JSON object")
        key_type,value_type=get_args(cls)
        return {decode(key_type,key):decode(value_type,item) for key,item in value.items()}
    if is_dataclass(cls):
        require(type(value) is dict and set(value) == {f.name for f in fields(cls)},
                f"{cls.__name__} fields do not match exact schema")
        hints = get_type_hints(cls)
        return cls(**{name: decode(hints[name], item) for name, item in value.items()})
    if cls is float:
        require(type(value) in (float, int) and math.isfinite(value), "expected finite number")
    else:
        require(type(value) is cls, f"expected {getattr(cls, '__name__', cls)}")
    return value


def read_bytes(path, expected=None):
    candidate = Path(path)
    require(".." not in candidate.parts, "artifact path contains a dot segment")
    raw = admission._read_strict_bytes(candidate, "capacity artifact")
    if expected is not None:
        admission._authenticate(raw, expected, "capacity artifact")
    return raw


def read_json(path, cls, expected=None):
    return read_json_with_hash(path, cls, expected)[0]


def read_json_with_hash(path, cls, expected=None):
    """Return the decoded record and the digest of those same descriptor bytes."""
    raw = read_bytes(path)
    payload = admission._parse_json(raw, cls.__name__)
    result = decode(cls, payload)
    require(raw == canonical(payload), "artifact must use canonical JSON")
    if expected is not None:
        admission._authenticate(raw, expected, cls.__name__)
    return result, digest(raw)


def publish_json(path, value, cls):
    payload = asdict(value) if is_dataclass(value) else value
    decode(cls, payload)
    raw = canonical(payload)
    admission._write_exclusively(path, raw, cls.__name__)
    return digest(raw)


def git_identity():
    return admission._git_identity()


def require_ignored_roots(root):
    for name in ("results", "logs", "checkpoints"):
        result = subprocess.run(["git", "check-ignore", "-q", "--no-index",
            str(Path(name) / ".codex-ignore-probe")], cwd=root)
        require(result.returncode == 0, f"{name} must ignore generated artifacts")


class StrictRecord:
    def __post_init__(self):
        hints = get_type_hints(type(self))
        for field in fields(self):
            value = getattr(self, field.name)
            if not is_dataclass(value) and not (isinstance(value, list) and any(is_dataclass(x) for x in value)):
                decode(hints[field.name], value)
            if field.name.endswith("sha256") and value is not None and not isinstance(value,dict):
                admission._require_sha256(field.name, value)
            if field.name.endswith("_path") and field.name != "relative_path" and value is not None:
                absolute(value)
        self.validate()

    def validate(self):
        pass


def canonical_uuid(value):
    require(type(value) is str and value, "GPU UUID is missing")
    try:
        return uuid.UUID(value.removeprefix("GPU-").lower()).hex
    except ValueError as error:
        raise ValueError("GPU UUID is invalid") from error


@dataclass(frozen=True)
class GpuIdentity(StrictRecord):
    physical_index: int
    name: str
    uuid: str
    total_memory_bytes: int

    def validate(self):
        require(self.physical_index in (0, 1) and self.name == "NVIDIA H100 NVL"
                and self.total_memory_bytes > 0, "GPU must be physical H100 NVL 0 or 1")
        canonical_uuid(self.uuid)


@dataclass(frozen=True)
class SnapshotGpu(StrictRecord):
    host_physical_index: int
    visible_cuda_index: int
    name: str
    uuid: str
    total_memory_bytes: int

    def validate(self):
        GpuIdentity(self.host_physical_index, self.name, self.uuid, self.total_memory_bytes)
        require(self.visible_cuda_index == self.host_physical_index, "GPU visible/physical indices disagree")


@dataclass(frozen=True)
class PackedGpuSnapshot(StrictRecord):
    schema_version: int
    kind: str
    gpus: list[SnapshotGpu]

    def validate(self):
        require(self.schema_version == 1 and self.kind == "stage2b-packed-gpu-snapshot", "invalid GPU snapshot schema")
        require([x.host_physical_index for x in self.gpus] == [0, 1]
                and len({canonical_uuid(x.uuid) for x in self.gpus}) == 2, "GPU snapshot requires exactly two distinct ranks")


def physical_inventory():
    lines = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.splitlines()
    rows = []
    for line in lines:
        index, name, device_uuid = (part.strip() for part in line.split(","))
        rows.append((int(index), name, device_uuid))
    require(len(rows) == len({x[0] for x in rows}), "duplicate GPU physical index")
    return rows


def observe_gpu(device=0, inventory=None):
    import torch
    index = torch.device(device).index if isinstance(device, (str, torch.device)) else device
    index = 0 if index is None else index
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    require(index < len(visible) and visible[index] in ("0", "1"), "unsupported CUDA physical mapping")
    physical = int(visible[index])
    matches = [row for row in (physical_inventory() if inventory is None else inventory) if row[0] == physical]
    require(len(matches) == 1, "GPU physical index is missing")
    properties = torch.cuda.get_device_properties(index)
    _, name, device_uuid = matches[0]
    require(name == properties.name and canonical_uuid(device_uuid) == canonical_uuid(str(properties.uuid)),
            "nvidia-smi/CUDA GPU identity differs")
    return GpuIdentity(physical, name, canonical_uuid(device_uuid), int(properties.total_memory))


def capture_gpu_snapshot():
    inventory = physical_inventory()
    gpus = [observe_gpu(rank, inventory) for rank in (0, 1)]
    return PackedGpuSnapshot(1, "stage2b-packed-gpu-snapshot", [
        SnapshotGpu(g.physical_index, rank, g.name, g.uuid, g.total_memory_bytes)
        for rank, g in enumerate(gpus)])


def require_matching_snapshot(expected, actual):
    expected = decode(PackedGpuSnapshot, expected) if isinstance(expected, dict) else expected
    actual = decode(PackedGpuSnapshot, actual) if isinstance(actual, dict) else actual
    for left, right in zip(expected.gpus, actual.gpus):
        require((left.host_physical_index, left.visible_cuda_index, left.name, canonical_uuid(left.uuid), left.total_memory_bytes)
                == (right.host_physical_index, right.visible_cuda_index, right.name, canonical_uuid(right.uuid), right.total_memory_bytes),
                "live GPU snapshot changed")


def require_gpu_match(snapshot, gpu, rank):
    entry = snapshot.gpus[rank]
    require((gpu.physical_index, gpu.name, canonical_uuid(gpu.uuid), gpu.total_memory_bytes)
            == (entry.host_physical_index, entry.name, canonical_uuid(entry.uuid), entry.total_memory_bytes),
            "GPU identity does not match snapshot rank")


def require_idle(indices=(0, 1)):
    inventory = {canonical_uuid(row[2]): row[0] for row in physical_inventory()}
    output = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    for line in output.splitlines():
        device_uuid, _pid = (part.strip() for part in line.split(","))
        require(inventory.get(canonical_uuid(device_uuid)) not in indices, "capacity GPU has a compute application")


@dataclass(frozen=True)
class ProbeIdentity(StrictRecord):
    checkpoint_path: str | None
    device: str | None
    dtype: str | None
    attention_implementation: str | None
    vision_attention_implementation: str | None
    tuning_method: str | None
    gradient_checkpointing: bool | None
    git_commit: str | None
    gpu_snapshot_path: str | None
    gpu_snapshot_sha256: str | None


@dataclass(frozen=True)
class ProbeSample(StrictRecord):
    source_row_index: int | None
    sample_id: str | None
    clean_length: int | None
    step_count: int | None
    branch_count: int | None
    packed_unpadded_length: int | None
    packed_padded_length: int | None


@dataclass(frozen=True)
class ProbeLosses(StrictRecord):
    ce: float | None
    dpo: float | None
    entropy: float | None
    total: float | None


@dataclass(frozen=True)
class ProbeMemory(StrictRecord):
    device_total_bytes: int
    baseline_allocated_bytes: int
    peak_allocated_bytes: int
    baseline_reserved_bytes: int
    peak_reserved_bytes: int
    peak_reserved_fraction: float

    def validate(self):
        require(self.device_total_bytes > 0 and min(self.baseline_allocated_bytes, self.baseline_reserved_bytes) >= 0
                and self.peak_allocated_bytes >= self.baseline_allocated_bytes
                and self.peak_reserved_bytes >= self.baseline_reserved_bytes
                and self.peak_reserved_fraction == self.peak_reserved_bytes / self.device_total_bytes,
                "capacity memory counters/fraction disagree")


@dataclass(frozen=True)
class PackedCapacityReport(StrictRecord):
    schema_version: int
    status: str
    identity: ProbeIdentity
    sample: ProbeSample
    losses: ProbeLosses
    gradient_norm: float | None
    memory: ProbeMemory | None
    gpu: GpuIdentity | None
    elapsed_seconds: float | None
    errors: list[str]

    def validate(self):
        require(self.schema_version == 1 and self.status in ("pass", "oom", "error"), "invalid capacity report schema")
        require((self.status == "pass") == (self.errors == []) and len(self.errors) <= 1
                and all(self.errors), "invalid capacity report errors")
        if self.status != "pass":
            return
        require(all(x is not None for x in asdict(self.identity).values()) and
                all(x is not None for x in asdict(self.sample).values()) and
                all(x is not None for x in asdict(self.losses).values()), "capacity pass has unobserved values")
        require(self.identity.device == "cuda:0" and self.identity.dtype == "bfloat16"
                and self.identity.attention_implementation == "sdpa"
                and self.identity.vision_attention_implementation == "flash_attention_2"
                and self.identity.tuning_method == "dora" and self.identity.gradient_checkpointing,
                "capacity method/backend identity differs")
        require(re.fullmatch("[0-9a-f]{40}", self.identity.git_commit), "invalid implementation commit")
        sample = self.sample
        require(sample.source_row_index >= 0 and sample.sample_id and sample.clean_length > 0
                and sample.step_count > 0 and sample.branch_count > 0
                and sample.packed_padded_length >= sample.packed_unpadded_length >= sample.clean_length
                and sample.packed_padded_length % 128 == 0, "capacity sample geometry is invalid")
        require(self.gradient_norm is not None and self.gradient_norm > 0 and self.elapsed_seconds is not None
                and self.elapsed_seconds >= 0 and self.memory is not None and self.gpu is not None,
                "capacity pass has missing or nonpositive observations")
        require(self.memory.device_total_bytes == self.gpu.total_memory_bytes
                and self.memory.peak_reserved_fraction <= .85, "capacity memory exceeds accepted fraction")


@dataclass(frozen=True)
class PackedCommandLog(StrictRecord):
    schema_version: int
    kind: str
    commands: list[list[str]]

    def validate(self):
        require(self.schema_version == 1 and self.kind == "stage2b-packed-command-log"
                and self.commands and all(command and all(command) for command in self.commands), "invalid command log")


@dataclass(frozen=True)
class RankObservation(StrictRecord):
    rank: int
    sample_ids: list[str]
    pre_step_gradient_norms: list[float]
    training_loss: float
    peak_reserved_bytes: int
    total_memory_bytes: int
    peak_reserved_fraction: float

    def validate(self):
        require(self.rank in (0, 1) and len(self.sample_ids) == 4 and len(set(self.sample_ids)) == 1
                and all(self.sample_ids), "capacity rank must observe one sample four times")
        require(len(self.pre_step_gradient_norms) == 2 and min(self.pre_step_gradient_norms) > 0,
                "capacity rank requires two positive pre-step gradients")
        require(self.total_memory_bytes > 0 and self.peak_reserved_bytes >= 0
                and self.peak_reserved_fraction == self.peak_reserved_bytes / self.total_memory_bytes
                and self.peak_reserved_fraction <= .85, "rank capacity memory exceeds accepted fraction")


@dataclass(frozen=True)
class RankGpu(GpuIdentity):
    rank: int

    def validate(self):
        super().validate()
        require(self.rank == self.physical_index, "GPU rank/physical index mismatch")


@dataclass(frozen=True)
class PackedDistributedCapacityEvidence(StrictRecord):
    schema_version: int
    kind: str
    run_id: str
    world_size: int
    process_ranks: list[int]
    global_step: int
    optimizer_step: int
    gradient_accumulation_steps: int
    microbatch_count: int
    sample_id: str
    rank_observations: list[RankObservation]
    gpu_snapshot_path: str
    gpu_snapshot_sha256: str
    gpu_identities: list[RankGpu]
    checkpoint_path: str
    training_runtime_identity_path: str
    training_runtime_identity_sha256: str
    method: str
    zero_stage: int
    finite_loss: bool
    finite_gradient: bool
    nonzero_gradient: bool
    memory_within_limit: bool
    worker_errors: list[str]

    def validate(self):
        run_id(self.run_id)
        require((self.schema_version, self.kind, self.world_size, self.process_ranks, self.global_step,
                 self.optimizer_step, self.gradient_accumulation_steps, self.microbatch_count, self.method, self.zero_stage)
                == (1, "stage2b-packed-distributed-capacity", 2, [0,1], 2, 2, 2, 4, "dora", 2),
                "distributed capacity protocol must be exact two-rank two-step ZeRO-2")
        require([r.rank for r in self.rank_observations] == [0,1] and [r.rank for r in self.gpu_identities] == [0,1],
                "distributed rank records are incomplete or unordered")
        require(len({canonical_uuid(g.uuid) for g in self.gpu_identities}) == 2, "distributed GPU UUIDs must be unique")
        for observation, gpu in zip(self.rank_observations, self.gpu_identities):
            require(observation.sample_ids == [self.sample_id]*4 and observation.total_memory_bytes == gpu.total_memory_bytes,
                    "distributed sample/memory identity differs")
        require(all((self.finite_loss, self.finite_gradient, self.nonzero_gradient, self.memory_within_limit))
                and self.worker_errors == [], "distributed health claims disagree with observations")
        require(Path(self.checkpoint_path).name == self.run_id and self.training_runtime_identity_path
                == str(Path(self.checkpoint_path) / "stage2_training_runtime_identity.json"), "final output-root identity differs")


@dataclass(frozen=True)
class AttentionObservation(StrictRecord):
    rank: int
    language_attention_class: str
    language_attention_layer_count: int
    vision_attention_class: str
    vision_attention_block_count: int
    packed_attention_backend: str
    sdpa_forward_counts: list[int]
    vision_flash_attention_forward_counts: list[int]
    flex_attention_forward_counts: list[int]

    def validate(self):
        require(self.rank in (0,1) and self.language_attention_class == LANGUAGE_CLASS
                and self.vision_attention_class == VISION_CLASS and self.packed_attention_backend == "flex_attention"
                and self.language_attention_layer_count > 0 and self.vision_attention_block_count > 0,
                "concrete attention backend identity differs")
        for values, count in ((self.sdpa_forward_counts,self.language_attention_layer_count),
                (self.flex_attention_forward_counts,self.language_attention_layer_count),
                (self.vision_flash_attention_forward_counts,self.vision_attention_block_count)):
            require(len(values) == count and min(values) > 0, "every concrete module must execute its kernel")


def objective_metric(triplet):
    for weights, metric in (([1,0,0],"loss/ce"),([0,1,0],"loss/dpo"),([0,0,1],"loss/entropy"),([1,1,.1],"loss/total")):
        if list(triplet) == weights:
            return metric
    raise ValueError("unsupported coefficient triplet")


@dataclass(frozen=True)
class TrainingRuntimeIdentity(StrictRecord):
    schema_version: int
    kind: str
    run_id: str
    git_commit: str
    tracked_worktree_clean: bool
    coefficient_triplet: list[float]
    tuning_method: str
    seed: int
    bf16: bool
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    use_reentrant: bool
    global_step: int
    optimizer_step: int
    objective_metric_name: str
    objective_metric_observation_count: int
    finite_objective_metrics: bool
    positive_pre_step_gradient_count: int
    nonfinite_pre_step_gradient_count: int
    first_positive_pre_step_gradient_norms: list[float]
    training_length_manifest_path: str
    training_length_manifest_sha256: str
    admission_manifest_sha256: str
    verification_sha256: str
    world_size: int
    process_ranks: list[int]
    rank_observations: list[AttentionObservation]

    def validate(self):
        run_id(self.run_id)
        require(self.schema_version == 1 and self.kind == "stage2-training-runtime-identity"
                and re.fullmatch("[0-9a-f]{40}",self.git_commit) and self.tracked_worktree_clean,
                "invalid training implementation identity")
        require((self.tuning_method,self.seed,self.bf16,self.per_device_train_batch_size,
                 self.gradient_accumulation_steps,self.gradient_checkpointing,self.use_reentrant)
                == ("dora",42,True,1,2,True,False), "training runtime settings differ")
        require(self.world_size == 2 and self.process_ranks == [0,1]
                and [r.rank for r in self.rank_observations] == [0,1], "training runtime ranks differ")
        require(self.global_step == self.optimizer_step and self.global_step > 0,
                "training requires completed optimizer steps")
        require(self.objective_metric_name == objective_metric(self.coefficient_triplet)
                and self.objective_metric_observation_count > 0 and self.finite_objective_metrics,
                "active objective metric was not observed")
        require(self.positive_pre_step_gradient_count > 0 and self.nonfinite_pre_step_gradient_count == 0
                and len(self.first_positive_pre_step_gradient_norms) == min(2,self.positive_pre_step_gradient_count)
                and min(self.first_positive_pre_step_gradient_norms) > 0,
                "training requires observed positive pre-step gradients")
        require(self.training_length_manifest_sha256 != "0"*64, "dry-run length hash is not runtime evidence")


@dataclass(frozen=True)
class EvaluationRuntimeIdentity(StrictRecord):
    schema_version: int
    kind: str
    git_commit: str
    tracked_worktree_clean: bool
    model_dir: str
    language_attention_class: str
    language_attention_layer_count: int
    vision_attention_class: str
    vision_attention_block_count: int

    def validate(self):
        absolute(self.model_dir)
        require(self.schema_version == 1 and self.kind == "stage2-evaluation-runtime-identity"
                and re.fullmatch("[0-9a-f]{40}",self.git_commit) and self.tracked_worktree_clean
                and self.language_attention_class == LANGUAGE_CLASS and self.vision_attention_class == VISION_CLASS
                and self.language_attention_layer_count > 0 and self.vision_attention_block_count > 0,
                "invalid evaluation runtime identity")


def select_capacity_sample(prepared, authenticated_admission, sample_id):
    require(authenticated_admission is not None and type(sample_id) is str and sample_id
            and sample_id in authenticated_admission.retained_sample_ids, "capacity sample must be authenticated and retained")
    selected = tuple(example for example in prepared if example.sample_id == sample_id)
    require(len(selected) == 1, "capacity sample must match exactly once")
    return selected


class CapacitySampleRecorder:
    def __init__(self, collator, sample_id):
        self.collator = collator
        self.sample_id = sample_id
        self.sample_ids = []

    def __call__(self, examples):
        batch = self.collator(examples)
        require(list(batch["sample_ids"]) == [self.sample_id], "capacity batch sample identity differs")
        self.sample_ids.append(batch["sample_ids"][0])
        return batch


def validate_capacity_options(args, *, world_size=None, zero_stage=None):
    names = ("step_packed_capacity_evidence_path", "step_packed_capacity_sample_id",
             "step_packed_gpu_snapshot", "step_packed_gpu_snapshot_sha256")
    values = [getattr(args,name,None) for name in names]
    if not any(value is not None for value in values):
        return False
    require(all(type(value) is str and value for value in values), "capacity-only options must be supplied together")
    config = args.step_preference_config
    require(config.is_stage2b and config.packing_enabled, "capacity-only options require packed Stage 2B")
    require(args.max_steps == 2 and args.per_device_train_batch_size == 1 and args.gradient_accumulation_steps == 2
            and args.dataloader_num_workers == 0 and args.tuning_method == "dora"
            and args.gradient_checkpointing and getattr(args,"save_final_model",True)
            and not getattr(args,"resume_from_checkpoint_path",None), "invalid capacity-only training settings")
    require(not getattr(args,"benchmark_evidence_path",None), "capacity evidence cannot use four-rank benchmark evidence")
    if world_size is not None:
        require(world_size == 2 and zero_stage == 2, "capacity launch requires world size two and ZeRO-2")
    target = Path(values[0])
    require(target.is_absolute() and ".." not in target.parts, "capacity output must be an absolute canonical result path")
    from artifact_paths import require_canonical_artifact_path
    require_canonical_artifact_path(target,kind="result")
    require(not target.exists() and not target.is_symlink(), "capacity evidence output already exists")
    snapshot = read_json(values[2],PackedGpuSnapshot,values[3])
    import torch
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    require_gpu_match(snapshot,observe_gpu(rank),rank)
    return True


def attention_observation(model, rank):
    from peft_common import resolve_qwen2_5_vl_model
    from stage2.attention_backend import validate_qwen2_5_vl_attention_backends
    qwen = resolve_qwen2_5_vl_model(model)
    validate_qwen2_5_vl_attention_backends(qwen,attention_implementation="sdpa",vision_attention_implementation="flash_attention_2")
    language=[layer.self_attn for layer in qwen.model.layers]
    vision=[block.attn for block in qwen.visual.blocks]
    return AttentionObservation(rank,LANGUAGE_CLASS,len(language),VISION_CLASS,len(vision),"flex_attention",
        [getattr(m,"_sdpa_forward_count",0) for m in language],
        [getattr(m,"_vision_flash_attention_forward_count",0) for m in vision],
        [getattr(m,"_flex_attention_forward_count",0) for m in language])


def gather(local):
    import torch.distributed as dist
    require(dist.is_initialized() and dist.get_world_size() == 2, "capacity runtime requires initialized two-rank collective")
    values=[None,None]
    dist.all_gather_object(values,local)
    require(all(value is not None for value in values), "capacity collective has a missing rank")
    return values


def gather_checked(producer):
    """Every rank participates once, including a rank whose observation failed."""
    try:
        local = {"value": producer(), "error": None}
    except Exception as error:
        local = {"value": None, "error": f"{type(error).__name__}: {error}"}
    records = gather(local)
    errors = [record["error"] for record in records if record["error"] is not None]
    require(not errors, "collective observation failed: " + "; ".join(errors))
    return [record["value"] for record in records]


def publish_collective(path, value, cls):
    import torch.distributed as dist
    error = None
    if dist.get_rank() == 0:
        try:
            publish_json(path, value, cls)
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
    errors = [item["error"] for item in gather({"error": error}) if item["error"] is not None]
    require(not errors, "collective publication failed: " + "; ".join(errors))
    dist.barrier()


def adapter_files(directory):
    from stage2.checkpointing import inspect_stage2_artifact, Stage2ArtifactKind, decode_stage2_peft_metadata_snapshot
    directory=admission._directory(directory,"final Stage 2 output root")
    # Read the three leaf artifacts through descriptors before delegating semantics.
    payloads={name:read_bytes(directory/name) for name in ADAPTER_FILES}
    inspection=inspect_stage2_artifact(directory)
    require(inspection.artifact_kind is Stage2ArtifactKind.PEFT, "final artifact is not Stage 2 PEFT")
    metadata=decode_stage2_peft_metadata_snapshot(inspection.peft_metadata_snapshot)
    require(metadata.stage == "stage2" and metadata.tuning_method == "dora"
            and (metadata.lora_r,metadata.lora_alpha,metadata.lora_dropout)==(32,64,.05),
            "final artifact is not required Stage 2 DoRA")
    config=admission._parse_json(payloads["adapter_config.json"],"adapter config")
    require(config.get("use_dora") is True and (config.get("r"),config.get("lora_alpha"),config.get("lora_dropout"))==(32,64,.05),
            "adapter config differs from DoRA contract")
    for entry in directory.iterdir():
        if entry.name in ADAPTER_FILES or entry.name == "training_args.bin" or entry.is_dir() and entry.name.startswith("checkpoint-"):
            continue
        require(not (entry.suffix in (".safetensors",".bin",".pt",".pth")
                or any(word in entry.name.lower() for word in ("optimizer","scheduler","rng","zero_pp","mp_rank"))),
                "unexpected model/optimizer state in final artifact")
    require(payloads["adapter_model.safetensors"] == inspection.adapter_safetensors
            and payloads["adapter_config.json"] == inspection.adapter_config_json
            and payloads["stage2_peft_metadata.json"] == inspection.peft_metadata_snapshot.metadata_json,
            "adapter changed during inspection")
    return [dict(relative_path=name,size=len(payloads[name]),sha256=digest(payloads[name])) for name in ADAPTER_FILES]


def training_length_manifest(path, expected, identity):
    from stage2.length_manifest import read_stage2_length_manifest
    raw=read_bytes(path,expected)
    manifest=read_stage2_length_manifest(path)
    require(raw == read_bytes(path),"training length manifest changed while validating")
    require(manifest.data_sha256 == identity.retained_data_sha256 and manifest.latent_size == 8
            and manifest.max_sequence_length == 8192 and manifest.supervise_latent_boundaries
            and manifest.assistant_plan_rendering == "adjacent-text-space-lf-v1"
            and manifest.retained_indices == tuple(range(len(manifest.lengths))), "training length manifest identity differs")
    clean=read_stage2_length_manifest(identity.clean_length_manifest_path)
    require(manifest.processor_sha256 == clean.processor_sha256,"training processor identity differs")
    return manifest


def publish_training_runtime(args, trainer):
    import torch.distributed as dist
    from dataclasses import replace
    def observe():
        config=trainer.step_preference_config
        manifest,_=admission._strict_json(config.packed_admission_manifest,"admission",admission._manifest_from_payload,config.packed_admission_manifest_sha256)
        commit=git_identity()
        require(commit == manifest.identity.git_commit,"training implementation commit differs from admission")
        training_length_manifest(args.stage2_length_manifest,args.stage2_length_manifest_sha256,manifest.identity)
        local=attention_observation(trainer.accelerator.unwrap_model(trainer.model),dist.get_rank())
        metric=objective_metric([config.lambda_ce,config.lambda_dpo,config.lambda_entropy])
        count=getattr(trainer,"_runtime_objective_observation_counts",{}).get(metric,0)
        training=trainer.args
        # Validate all local scalar health/settings before entering the collective.
        value=TrainingRuntimeIdentity(1,"stage2-training-runtime-identity",Path(args.save_model_path).name,
            commit,True,[config.lambda_ce,config.lambda_dpo,config.lambda_entropy],args.tuning_method,
            training.seed,training.bf16,training.per_device_train_batch_size,training.gradient_accumulation_steps,
            training.gradient_checkpointing,training.gradient_checkpointing_kwargs["use_reentrant"],
            trainer.state.global_step,trainer._observed_optimizer_step,metric,count,True,
            getattr(trainer,"_positive_pre_step_gradient_count",0),getattr(trainer,"_nonfinite_pre_step_gradient_count",0),
            getattr(trainer,"_observed_pre_step_gradient_norms",[]),
            str(Path(args.stage2_length_manifest).resolve(strict=True)),args.stage2_length_manifest_sha256,
            config.packed_admission_manifest_sha256,config.packed_admission_verification_sha256,
            2,[0,1],[replace(local,rank=rank) for rank in (0,1)])
        payload=asdict(value)
        payload["rank_observations"]=[asdict(local)]
        return payload
    records=gather_checked(observe)
    observations=[record.pop("rank_observations")[0] for record in records]
    require(records[0] == records[1],"runtime update observations disagree across ranks")
    value=decode(TrainingRuntimeIdentity,{**records[0],"rank_observations":observations})
    path=Path(args.save_model_path).resolve(strict=True)/"stage2_training_runtime_identity.json"
    publish_collective(path,value,TrainingRuntimeIdentity)
    expected=digest(canonical(value))
    read_bytes(path,expected)
    return path,expected


def begin_capacity_memory(device):
    import torch
    torch.cuda.synchronize(device)
    baseline=(int(torch.cuda.memory_allocated(device)),int(torch.cuda.memory_reserved(device)))
    torch.cuda.reset_peak_memory_stats(device)
    return baseline


def finish_capacity_memory(device,baseline):
    import torch
    torch.cuda.synchronize(device)
    return max(baseline[1],int(torch.cuda.max_memory_reserved(device)))


def publish_distributed_capacity(args,trainer,train_result,recorder,peak,runtime_path,runtime_sha):
    import torch
    import torch.distributed as dist
    rank=dist.get_rank(); device=trainer.accelerator.device
    def observe():
        torch.cuda.synchronize(device)
        gpu=observe_gpu(device)
        snapshot=read_json(args.step_packed_gpu_snapshot,PackedGpuSnapshot,args.step_packed_gpu_snapshot_sha256)
        require_gpu_match(snapshot,gpu,rank)
        local=RankObservation(rank,recorder.sample_ids,trainer._observed_pre_step_gradient_norms,
            float(train_result.training_loss),peak,gpu.total_memory_bytes,peak/gpu.total_memory_bytes)
        require(local.sample_ids == [args.step_packed_capacity_sample_id]*4 and
            trainer.state.global_step == trainer._observed_optimizer_step == trainer.args.gradient_accumulation_steps == 2
            and trainer.deepspeed_runtime_profile.zero_stage == 2 and args.tuning_method == "dora",
            "capacity optimizer/sample/settings identity differs")
        adapter_files(args.save_model_path)
        return dict(observation=asdict(local),gpu=dict(rank=rank,**asdict(gpu)))
    gathered=gather_checked(observe)
    checkpoint=Path(args.save_model_path).resolve(strict=True)
    observations=[decode(RankObservation,item["observation"]) for item in gathered]
    finite_loss=all(math.isfinite(item.training_loss) for item in observations)
    finite_gradient=all(math.isfinite(norm) for item in observations for norm in item.pre_step_gradient_norms)
    nonzero_gradient=all(norm > 0 for item in observations for norm in item.pre_step_gradient_norms)
    memory_within_limit=all(item.peak_reserved_fraction <= .85 for item in observations)
    value=PackedDistributedCapacityEvidence(1,"stage2b-packed-distributed-capacity",checkpoint.name,2,[0,1],
        trainer.state.global_step,trainer._observed_optimizer_step,trainer.args.gradient_accumulation_steps,
        len(recorder.sample_ids),args.step_packed_capacity_sample_id,observations,
        str(Path(args.step_packed_gpu_snapshot).resolve(strict=True)),args.step_packed_gpu_snapshot_sha256,
        [decode(RankGpu,item["gpu"]) for item in gathered],str(checkpoint),str(runtime_path),runtime_sha,
        args.tuning_method,trainer.deepspeed_runtime_profile.zero_stage,finite_loss,finite_gradient,nonzero_gradient,memory_within_limit,[])
    publish_collective(args.step_packed_capacity_evidence_path,value,PackedDistributedCapacityEvidence)
    return value
