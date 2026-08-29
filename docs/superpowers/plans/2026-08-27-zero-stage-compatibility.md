# ILVR Stage 1/2 ZeRO-1/2/3 Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the validated ZeRO-3 training path so Stage 1 Full/DoRA and Stage 2 Full/LoRA/DoRA support ZeRO-1, ZeRO-2, and ZeRO-3, while recording identity-bound batch, memory, throughput, checkpoint, and resume evidence.

**Architecture:** A small immutable DeepSpeed runtime profile is the single source of truth for ZeRO stage, config digest, and world size across launchers, trainers, checkpoints, and evidence. Stage 1 dispatches optimizer norm and sparse-state behavior by that profile; Stage 2 uses a latent-block-balanced global sampler so all ranks execute the same decoder-forward count, and a focused matrix coordinator reuses exact evidence while running only the missing three-step candidates.

**Tech Stack:** Python 3.11, PyTorch 2.7.1 + CUDA 12.8, DeepSpeed 0.18.0, Transformers/TRL Trainer, Accelerate `BatchSamplerShard`, PEFT 0.17.1, FlashAttention 2.8.3, safetensors, pytest, Bash, NVIDIA B200.

**Spec:** `docs/superpowers/specs/2026-08-27-zero-stage-compatibility-design.md`

## Global Constraints

- Use only `./venv/ilvr-venv` for Python, pip, tests, training, and evaluation; never use system Python or `./ilvr-venv`.
- Execute in the existing isolated worktree `.worktrees/vcsd-ilvr-stage2` on `feat/vcsd-ilvr-stage2`; preserve unrelated edits and generated artifacts.
- Keep ZeRO-3 as the launcher default. ZeRO-1 and ZeRO-2 are explicit selections through canonical config files.
- The official validation environment is exactly two NVIDIA B200 GPUs, but ordinary fresh training must not reject other positive world sizes.
- Stage 1 officially fixes effective global batch at 16 and tests candidates in reverse-size order: `(micro_batch, gradient_accumulation) = (8,1), (4,2), (2,4), (1,8)`.
- Stage 2 keeps `per_device_train_batch_size=1`; its official two-GPU setting is `gradient_accumulation_steps=8`, with no batch sweep.
- A resume must match the checkpoint's ZeRO stage, exact DeepSpeed config digest, and world size; cross-stage or cross-world-size optimizer conversion is unsupported.
- Keep existing `configs/config_stage*.json` files for compatibility, but all new launchers, documentation, and evidence use `configs/deepspeed_zero{1,2,3}.json`.
- Keep the pinned two-GPU ZeRO-2 configs free of `data_parallel_size` and `sequence_parallel_size`; do not reinterpret the existing eight-rank mesh config.
- Stage 1 DoRA uses `r=32`, `alpha=64`, `dropout=0.05`, and Qwen decoder suffixes `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- Use the official model `Qwen/Qwen2.5-VL-3B-Instruct`, seed `42`, bf16, and FlashAttention 2 for hardware evidence.
- A successful Stage 1 candidate is one three-optimizer-step producer that records capacity, throughput, and peak memory together; do not add separate smoke, capacity, or timed passes for the same candidate.
- Run save-to-resume only for the selected successful Stage 1 configuration for each `(ZeRO stage, tuning method)` pair.
- Reuse the already successful Stage 2 ZeRO-1/2 DoRA save/resume feasibility witness unless shared checkpoint code changes invalidate that exact contract.
- Run the Stage 2 rank-skew distributed check exactly once per ZeRO stage after sampler changes, and one lightweight ZeRO-3 regression after common-path changes.
- Before any validation expected to read more than 10 GB or take more than five minutes, inspect exact existing evidence first; never repeat a successful producer, validator, checkpoint-tree hash, or unchanged dependency hash.
- A config file is hashed once when its runtime profile is created; every downstream manifest, checkpoint sidecar, and report passes that digest forward.
- Follow `superpowers:test-driven-development` for each production behavior change and `superpowers:systematic-debugging` for any unexpected test, numerical, OOM, DeepSpeed, NCCL, or timeout failure.
- Do not push, merge, publish, or create a PR without explicit user permission.
- Store every generated weight, final model, and numbered checkpoint under `/root/workspace/ILVR/checkpoints/zero-stage-compatibility/`; never use a linked-worktree-local checkpoint directory.
- Store every generated evidence, witness, manifest, audit, report, table, or plot under `/root/workspace/ILVR/results/zero-stage-compatibility/`.
- Store every training log, execution log, stdout, and stderr under `/root/workspace/ILVR/logs/zero-stage-compatibility/`.
- Give every independent training attempt and standalone probe a distinct experiment ID and reuse that exact ID under checkpoint, result, and log roots. A two-phase checkpoint→same-identity-resume lifecycle keeps one ID because it is one logical experiment; independently reusable selected-save and selected-resume producers use different IDs.
- Unit tests that do not perform real training may use pytest `tmp_path`; every distributed integration harness uses the canonical roots even when its checkpoint is tiny.
- Treat pre-policy `outputs/` artifacts as read-only legacy provenance: never move, rename, overwrite, revalidate, or add new files to them. Reuse only exact identity-matching witnesses at their original absolute paths.

---

## File Structure

### New production files

- `src/artifact_paths.py` — exact canonical checkpoint/result/log roots, safe experiment-ID validation, three-way run path resolution, and CLI path preflight shared by launchers and harnesses.
- `src/deepspeed_runtime.py` — immutable DeepSpeed identity, canonical config validation, one-time digesting, process-safe checkpoint sidecars, and strict resume matching shared by both stages.
- `src/stage2/sampler.py` — deterministic latent-block-balanced global sampler, padding statistics, and checkpoint identity serialization.
- `src/stage1/zero_matrix.py` — fixed-EGB candidate definitions, exact evidence reuse decisions, failure classification, winner selection, and machine-readable matrix reporting.
- `run_stage1_zero_matrix.sh` — two-GPU reverse-order Stage 1 matrix launcher that starts only missing producers and selected resumes.
- `docs/zero-stage-training.md` — canonical config selection, support table, fresh/resume commands, and measured-result tables.

### New configuration files

- `configs/deepspeed_zero1.json` — bf16 ZeRO-1 with auto batch fields and no offload or fixed topology.
- `configs/deepspeed_zero2.json` — bf16 ZeRO-2 with auto batch fields and no offload or fixed topology.
- `configs/deepspeed_zero3.json` — canonical copy of the validated ZeRO-3 policy without fixed data/sequence mesh dimensions.

### Modified production files

- `src/utils_deepseed.py` — expose the canonical ZeRO-3 default without restricting explicit ZeRO-1/2 paths.
- `src/main.py` — bind one runtime profile before Stage 1/2 composition and pass it into Stage 2 trainer construction.
- `src/stage1/composition.py` — consume the profile, pass the actual ZeRO stage into PEFT validation and the trainer, and remove the hard-coded world-size-two preflight from general runtime.
- `src/stage1/peft.py` — register DoRA external parameters only for actual ZeRO-3.
- `src/stage1/trainer.py` — stage-specific pre-step norm strategy, applied-step EMA behavior, sparse save routing, and runtime sidecar/resume enforcement.
- `src/stage1/checkpointing.py` — require the runtime identity sidecar when classifying resumable numbered checkpoints created by the new schema.
- `src/stage1/benchmark.py` — add ZeRO identity to evidence and generalize producer config validation without changing legacy ZeRO-3 report semantics.
- `src/stage2/trainer.py` — install the balanced sampler in multi-GPU mode, save sampler/runtime identities, and validate them before resume.
- `src/stage1/token_rows.py` and `src/stage2/trainer.py` — perform sparse collective gather only for parameters actually partitioned by ZeRO-3.
- `run_stage1_training.sh`, `run_stage1_batch_probe.sh`, `run_stage2_training.sh` — canonical config selection plus checkpoint/result/log ownership and runtime-profile preflight.
- `README.md` — link the detailed ZeRO training guide.

### New and modified tests

- `tests/test_artifact_paths.py` — canonical root resolution, shared experiment identifiers, traversal rejection, worktree-local output rejection, and CLI preflight.
- `tests/test_deepspeed_runtime.py` — profile parsing, topology conflict rejection, stable digest, sidecar round trip, and resume mismatch tests.
- `tests/stage1/test_trainer.py`, `tests/stage1/test_ema.py`, `tests/stage1/test_peft.py`, `tests/stage1/test_composition.py` — ZeRO-1/2 optimizer strategy, exact EMA update, ZeRO-3-only registration, and profile propagation.
- `tests/stage1/test_checkpointing.py` — numbered-checkpoint runtime identity and legacy/fresh artifact classification.
- `tests/stage1/test_benchmark.py`, `tests/stage1/test_zero_matrix.py` — evidence migration, cross-stage reuse rejection, reverse candidates, OOM classification, and winner selection.
- `tests/stage1/zero_compat_diagnostic.py`, `tests/stage1/test_distributed_smoke.py` — opt-in two-rank Full/DoRA step and selected save/resume lifecycle.
- `tests/stage2/test_sampler.py`, `tests/stage2/test_routing.py`, `tests/stage2/test_peft_checkpointing.py` — cross-rank block alignment, padding, epoch determinism, profile propagation, and sidecars.
- `tests/stage2/integration/stage2_rank_skew_probe.py`, `tests/stage2/integration/test_distributed_contract.py` — opt-in one-step production-model rank-skew and peak-memory witness for each canonical ZeRO config.

## Execution Dependency Map

Task 1 defines canonical artifact ownership and the shared runtime identity consumed everywhere else. Task 2 adds Stage 1 optimizer and PEFT behavior; Task 3 adds checkpoint identity to both stages. Task 4 introduces the Stage 2 sampler and its checkpoint contract. Task 5 generalizes producer evidence, and Task 6 builds the bounded Stage 1 matrix workflow on that schema. Task 7 adds the minimal distributed harnesses. Task 8 closes CPU regressions and documentation before any costly run. Tasks 9 and 10 are the only GPU-heavy gates: lifecycle/rank-skew compatibility first, then missing Stage 1 matrix producers and selected resumes. Task 11 records results and performs one proportionate final verification without repeating authenticated runs.

### Task 1: Canonical Artifact Layout and DeepSpeed Runtime Identity

**Files:**
- Create: `src/artifact_paths.py`
- Create: `src/deepspeed_runtime.py`
- Create: `configs/deepspeed_zero1.json`
- Create: `configs/deepspeed_zero2.json`
- Create: `configs/deepspeed_zero3.json`
- Create: `tests/test_deepspeed_runtime.py`
- Create: `tests/test_artifact_paths.py`
- Modify: `src/utils_deepseed.py:95-105`
- Modify: `src/main.py:499-516`
- Modify: `run_stage1_training.sh:50-100`
- Modify: `run_stage1_batch_probe.sh:20-70`
- Modify: `run_stage2_training.sh:35-60`
- Test: `tests/stage1/test_launcher.py`
- Test: `tests/stage1/test_cli.py`
- Test: `tests/stage2/test_routing.py`
- Test: `tests/stage2/integration/test_distributed_contract.py`

**Interfaces:**
- Consumes: one safe experiment ID, a DeepSpeed JSON path or `None`, and the positive process `world_size` discovered before CUDA initialization.
- Produces: `RunArtifactPaths`, `resolve_run_artifact_paths(experiment_id: str) -> RunArtifactPaths`, `require_canonical_artifact_path(path: str | Path, *, kind: Literal["checkpoint", "result", "log"]) -> Path`, `validate_generated_artifact_paths(args: argparse.Namespace) -> RunArtifactPaths`, `DeepSpeedRuntimeProfile`, `validate_supported_deepspeed_policy(payload: Mapping[str, object]) -> None`, `load_deepspeed_runtime_profile(config_path: str | Path | None, *, world_size: int) -> DeepSpeedRuntimeProfile`, `bind_deepspeed_runtime_profile(args: argparse.Namespace, *, world_size: int | None = None) -> DeepSpeedRuntimeProfile`, and `canonical_deepspeed_config_path(zero_stage: int) -> Path`.

- [ ] **Step 1: Write failing canonical artifact-layout tests**

```python
def test_run_paths_share_one_experiment_identifier():
    paths = resolve_run_artifact_paths("stage1-zero2-dora-mb4-ga2-candidate")
    assert paths.checkpoint_dir == Path(
        "/root/workspace/ILVR/"
        "checkpoints/zero-stage-compatibility/stage1-zero2-dora-mb4-ga2-candidate"
    )
    assert paths.result_dir.name == paths.checkpoint_dir.name
    assert paths.log_dir.name == paths.checkpoint_dir.name


@pytest.mark.parametrize("experiment_id", ["../escape", "has/slash", "UPPER", ""])
def test_experiment_identifier_rejects_unsafe_values(experiment_id):
    with pytest.raises(ValueError, match="experiment ID"):
        resolve_run_artifact_paths(experiment_id)


def test_worktree_local_outputs_are_not_canonical():
    with pytest.raises(ValueError, match="canonical result root"):
        require_canonical_artifact_path(
            Path.cwd() / "outputs" / "evidence.json", kind="result"
        )


def test_generated_checkpoint_result_and_log_share_experiment_id(stage1_args):
    paths = resolve_run_artifact_paths("stage1-zero2-dora-mb4-ga2-candidate")
    stage1_args.save_model_path = str(paths.checkpoint_dir)
    stage1_args.log_file = str(paths.log_dir / "train.log")
    stage1_args.benchmark_evidence_path = str(
        paths.result_dir / "producer-evidence.json"
    )
    assert validate_generated_artifact_paths(stage1_args) == paths
```

- [ ] **Step 2: Run artifact-layout tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/test_artifact_paths.py -q
```

Expected: collection fails because `src.artifact_paths` does not exist.

- [ ] **Step 3: Implement exact canonical roots and path containment checks**

```python
CANONICAL_REPOSITORY_ROOT = Path(
    "/root/workspace/ILVR"
)
CANONICAL_CHECKPOINT_ROOT = (
    CANONICAL_REPOSITORY_ROOT / "checkpoints" / "zero-stage-compatibility"
)
CANONICAL_RESULT_ROOT = (
    CANONICAL_REPOSITORY_ROOT / "results" / "zero-stage-compatibility"
)
CANONICAL_LOG_ROOT = (
    CANONICAL_REPOSITORY_ROOT / "logs" / "zero-stage-compatibility"
)
EXPERIMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,127}$")


@dataclass(frozen=True)
class RunArtifactPaths:
    experiment_id: str
    checkpoint_dir: Path
    result_dir: Path
    log_dir: Path


def resolve_run_artifact_paths(experiment_id: str) -> RunArtifactPaths:
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(
            "experiment ID must be 3-128 lowercase letters, digits, or hyphens"
        )
    return RunArtifactPaths(
        experiment_id=experiment_id,
        checkpoint_dir=CANONICAL_CHECKPOINT_ROOT / experiment_id,
        result_dir=CANONICAL_RESULT_ROOT / experiment_id,
        log_dir=CANONICAL_LOG_ROOT / experiment_id,
    )


def require_canonical_artifact_path(
    path: str | Path,
    *,
    kind: Literal["checkpoint", "result", "log"],
) -> Path:
    roots = {
        "checkpoint": CANONICAL_CHECKPOINT_ROOT,
        "result": CANONICAL_RESULT_ROOT,
        "log": CANONICAL_LOG_ROOT,
    }
    candidate = Path(path).expanduser().resolve(strict=False)
    root = roots[kind].resolve(strict=False)
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(f"path must be below the canonical {kind} root: {root}")
    return candidate


def validate_generated_artifact_paths(
    args: argparse.Namespace,
) -> RunArtifactPaths:
    checkpoint_dir = require_canonical_artifact_path(
        args.save_model_path, kind="checkpoint"
    )
    log_path = require_canonical_artifact_path(args.log_file, kind="log")
    experiment_id = checkpoint_dir.name
    paths = resolve_run_artifact_paths(experiment_id)
    if checkpoint_dir != paths.checkpoint_dir or log_path.parent != paths.log_dir:
        raise ValueError("checkpoint and log paths must share one experiment ID")
    for attribute in (
        "benchmark_run_config",
        "benchmark_evidence_path",
        "benchmark_smoke_resume_witness",
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            result = require_canonical_artifact_path(value, kind="result")
            if result.parent != paths.result_dir:
                raise ValueError(
                    f"{attribute} must use result experiment ID {experiment_id}"
                )
    return paths
```

Expose `python -m src.artifact_paths resolve --experiment-id ID` as compact JSON and `validate --kind checkpoint|result|log --path PATH` as an exit-2 preflight. The resolver creates no directory. A fresh producer creates only the resolved artifact directories it will write and fails if any target experiment directory already exists; an in-place same-identity resume may reuse its validated checkpoint/result directory and writes a distinct log file below the same log directory. Call `validate_generated_artifact_paths(args)` in `main_train` immediately after argument parsing and before seeding, model import, or CUDA initialization so direct `src/main.py` execution cannot bypass canonical ownership.

- [ ] **Step 4: Run artifact-layout tests and confirm GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/test_artifact_paths.py -q
```

Expected: all artifact-layout tests pass without creating anything below the canonical roots.

- [ ] **Step 5: Write failing profile and canonical-config tests**

```python
@pytest.mark.parametrize("zero_stage", [1, 2, 3])
def test_canonical_profile_reports_exact_stage_and_one_digest(zero_stage):
    path = canonical_deepspeed_config_path(zero_stage)
    profile = load_deepspeed_runtime_profile(path, world_size=2)
    assert profile.enabled is True
    assert profile.zero_stage == zero_stage
    assert profile.config_path == path.resolve()
    assert len(profile.config_digest) == 64
    assert profile.world_size == 2


def test_two_gpu_profile_rejects_fixed_eight_rank_mesh(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "train_batch_size": "auto",
        "zero_optimization": {"stage": 2},
        "data_parallel_size": 4,
        "sequence_parallel_size": 2,
    }))
    with pytest.raises(ValueError, match="requires world size 8"):
        load_deepspeed_runtime_profile(path, world_size=2)


def test_disabled_profile_has_no_config_identity():
    profile = load_deepspeed_runtime_profile(None, world_size=1)
    assert profile == DeepSpeedRuntimeProfile(
        enabled=False, zero_stage=None, config_path=None,
        config_digest=None, world_size=1,
    )
```

- [ ] **Step 6: Run the focused tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/test_deepspeed_runtime.py -q
```

Expected: collection fails because `src.deepspeed_runtime` and the canonical configs do not exist.

- [ ] **Step 7: Implement the immutable parser and one-time digest**

```python
@dataclass(frozen=True)
class DeepSpeedRuntimeProfile:
    enabled: bool
    zero_stage: int | None
    config_path: Path | None
    config_digest: str | None
    world_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "zero_stage": self.zero_stage,
            "config_path": None if self.config_path is None else str(self.config_path),
            "config_digest": self.config_digest,
            "world_size": self.world_size,
        }


def load_deepspeed_runtime_profile(
    config_path: str | Path | None, *, world_size: int
) -> DeepSpeedRuntimeProfile:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if config_path is None:
        return DeepSpeedRuntimeProfile(False, None, None, None, world_size)
    path = Path(config_path).expanduser().resolve(strict=True)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("DeepSpeed config must be a JSON object")
    zero = payload.get("zero_optimization")
    if not isinstance(zero, Mapping):
        raise ValueError("zero_optimization must be a JSON object")
    zero_stage = zero.get("stage")
    if zero_stage not in (1, 2, 3):
        raise ValueError("zero_optimization.stage must be 1, 2, or 3")
    mesh = int(payload.get("data_parallel_size", 1)) * int(
        payload.get("sequence_parallel_size", 1)
    )
    if mesh > 1 and mesh != world_size:
        raise ValueError(f"DeepSpeed mesh requires world size {mesh}, got {world_size}")
    validate_supported_deepspeed_policy(payload)
    return DeepSpeedRuntimeProfile(
        True, zero_stage, path, hashlib.sha256(raw).hexdigest(), world_size
    )


def canonical_deepspeed_config_path(zero_stage: int) -> Path:
    if zero_stage not in (1, 2, 3):
        raise ValueError("zero_stage must be 1, 2, or 3")
    repository = Path(__file__).resolve().parents[1]
    return repository / "configs" / f"deepspeed_zero{zero_stage}.json"


def bind_deepspeed_runtime_profile(
    args: argparse.Namespace, *, world_size: int | None = None
) -> DeepSpeedRuntimeProfile:
    existing = getattr(args, "deepspeed_runtime_profile", None)
    if existing is not None:
        if not isinstance(existing, DeepSpeedRuntimeProfile):
            raise TypeError("args.deepspeed_runtime_profile has an invalid type")
        return existing
    resolved_world_size = (
        int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else world_size
    )
    profile = load_deepspeed_runtime_profile(
        getattr(args, "deepspeed_config", None), world_size=resolved_world_size
    )
    args.deepspeed_runtime_profile = profile
    return profile
```

Implement the policy check as a pure helper:

```python
def validate_supported_deepspeed_policy(payload: Mapping[str, object]) -> None:
    if payload.get("bf16") != {"enabled": True}:
        raise ValueError("canonical DeepSpeed configs require bf16 enabled")
    for field in (
        "train_micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "train_batch_size",
    ):
        if payload.get(field) != "auto":
            raise ValueError(f"{field} must be 'auto'")
    zero = payload["zero_optimization"]
    for field in ("offload_optimizer", "offload_param"):
        value = zero.get(field)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"{field} must be a JSON object")
        if value is not None and value.get("device", "none") != "none":
            raise ValueError(f"{field} is unsupported for canonical GPU runs")
```

`bind_deepspeed_runtime_profile` must cache the immutable object on `args.deepspeed_runtime_profile`; a second call returns the same object and never re-reads or re-hashes the file.

- [ ] **Step 8: Add the three concrete canonical JSON policies**

Use the validated stage-specific policies rather than inventing a common bucket policy. The canonical ZeRO-2 file is exactly the current `configs/config_stage2.json` policy with `sequence_parallel_size`, `data_parallel_size`, and `sequence_parallel_mode` removed:

```json
{
  "bf16": {"enabled": true},
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "train_batch_size": "auto",
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "allgather_partitions": true,
    "allgather_bucket_size": 200000000,
    "reduce_bucket_size": 200000000
  },
  "gradient_checkpointing": true,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

Use this exact ZeRO-1 file:

```json
{
  "train_micro_batch_size_per_gpu": "auto",
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 1,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 500000000
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

Use this exact ZeRO-3 file:

```json
{
  "train_micro_batch_size_per_gpu": "auto",
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 50000000,
    "stage3_param_persistence_threshold": 1000000,
    "sub_group_size": 1000000000,
    "stage3_max_live_parameters": 1000000000,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

None of the three canonical files contains data/sequence topology or enabled offload.

- [ ] **Step 9: Switch launcher defaults and add config/artifact preflight commands**

Set Stage 1 defaults to `configs/deepspeed_zero3.json`, leave Stage 2 DeepSpeed optional, and validate an explicit config before `accelerate launch`:

```bash
"$repo_root/venv/ilvr-venv/bin/python" -m src.deepspeed_runtime validate \
  --config "$deepspeed_config" \
  --world-size "$num_processes"
```

Expected CLI output is one compact JSON object containing `enabled`, `zero_stage`, `config_path`, `config_digest`, and `world_size`. The CLI exits 2 before GPU launch for malformed policy or topology mismatch.

Every real launcher accepts or derives one experiment ID, resolves its three canonical directories once, and rejects user-supplied checkpoint/result/log paths outside their corresponding roots before creating directories or initializing CUDA. `run_stage1_training.sh` validates `--output-dir` as `checkpoint`, `--log-file` as `log`, and `--producer-evidence`, `--benchmark-run-config`, and `--smoke-resume-witness` as `result`. `run_stage2_training.sh` validates `OUTPUT_DIR` as `checkpoint` and `LOG_FILE` as `log`.

Add `STAGE1_NUM_PROCESSES` and `STAGE2_NUM_PROCESSES`, both defaulting to 2 for DeepSpeed launches and both validated as positive integers. Stage 1 passes the former to its existing Accelerate command instead of hard-coding 2. Stage 2 keeps the current direct one-process Python command when `DEEPSPEED_CONFIG` is absent, but wraps the same `src/main.py` arguments with the repository Accelerate executable when it is present:

```bash
TRAINING_ARGS=(
  --stage stage2
  --load_model_path "$STAGE1_CHECKPOINT"
  --data_path "$DATA_PATH"
  --image_root "$IMAGE_ROOT"
  --save_model_path "$OUTPUT_DIR"
  --log_file "$LOG_FILE"
  --latent_size "$LATENT_SIZE"
  "${BOUNDARY_ARGS[@]}"
  --learning_rate "$LEARNING_RATE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --attention_implementation "$ATTENTION_IMPLEMENTATION"
  "${CHECKPOINTING_ARGS[@]}"
  --max_seq_length_train "$MAX_SEQUENCE_LENGTH"
  "${TUNING_ARGS[@]}"
  "${RESUME_ARGS[@]}"
  "${DEEPSPEED_ARGS[@]}"
  "$@"
)

if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
  STAGE2_NUM_PROCESSES=${STAGE2_NUM_PROCESSES:-2}
  COMMAND=(
    "$REPO_ROOT/venv/ilvr-venv/bin/accelerate" launch
    --num_processes "$STAGE2_NUM_PROCESSES"
    --main_process_port "${STAGE2_MAIN_PROCESS_PORT:-29502}"
    "$REPO_ROOT/src/main.py"
    "${TRAINING_ARGS[@]}"
  )
else
  COMMAND=(
    "$REPO_ROOT/venv/ilvr-venv/bin/python"
    "$REPO_ROOT/src/main.py"
    "${TRAINING_ARGS[@]}"
  )
fi
```

Official benchmark scripts set the process count to 2; fresh non-benchmark users may select another positive count.

- [ ] **Step 10: Run runtime and launcher tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/test_deepspeed_runtime.py \
  tests/test_artifact_paths.py \
  tests/stage1/test_launcher.py \
  tests/stage1/test_cli.py \
  tests/stage2/test_routing.py \
  tests/stage2/integration/test_distributed_contract.py -q
```

Expected: all selected tests pass; dry-run launch commands use canonical ZeRO-3 by default and preserve explicit ZeRO-1/2 paths.

- [ ] **Step 11: Commit the artifact and runtime identity boundaries**

```bash
git add src/artifact_paths.py src/deepspeed_runtime.py src/main.py \
  configs/deepspeed_zero1.json \
  configs/deepspeed_zero2.json configs/deepspeed_zero3.json \
  src/utils_deepseed.py run_stage1_training.sh run_stage1_batch_probe.sh \
  run_stage2_training.sh tests/test_artifact_paths.py \
  tests/test_deepspeed_runtime.py \
  tests/stage1/test_launcher.py tests/stage1/test_cli.py \
  tests/stage2/test_routing.py tests/stage2/integration/test_distributed_contract.py
git commit -m "feat: add canonical ZeRO run ownership"
```

### Task 2: Stage 1 ZeRO-Aware Optimizer and PEFT Lifecycle

**Files:**
- Modify: `src/stage1/trainer.py:39-75,521-641`
- Modify: `src/stage1/ema.py`
- Modify: `src/stage1/peft.py:266-286,510-575`
- Modify: `src/stage1/composition.py:339-559`
- Modify: `src/main.py:459-465,499-516`
- Test: `tests/stage1/test_trainer.py`
- Test: `tests/stage1/test_ema.py`
- Test: `tests/stage1/test_peft.py`
- Test: `tests/stage1/test_composition.py`

**Interfaces:**
- Consumes: `DeepSpeedRuntimeProfile` and `bind_deepspeed_runtime_profile(args, world_size=...)` from Task 1.
- Produces: `collect_deepspeed_pre_step_norm(optimizer: object, *, zero_stage: int) -> torch.Tensor`, `validate_stage1_trainable_boundary(..., zero_stage: int | None)`, and `CustomTrainerStage1(..., deepspeed_runtime_profile: DeepSpeedRuntimeProfile, ...)`.

- [ ] **Step 1: Write failing ZeRO-1/2 norm and exact-EMA tests**

```python
class FakeZero12Optimizer:
    def __init__(self, *, partition_gradients: bool):
        self.partition_gradients = partition_gradients
        self.loss_scale = 2.0
        self.calls = 0

    def scaled_global_norm(self):
        self.calls += 1
        return torch.tensor(12.0)


@pytest.mark.parametrize("stage,partitioned", [(1, False), (2, True)])
def test_zero12_norm_uses_partition_aware_scaled_norm(stage, partitioned):
    optimizer = FakeZero12Optimizer(partition_gradients=partitioned)
    norm = collect_deepspeed_pre_step_norm(optimizer, zero_stage=stage)
    assert norm.item() == pytest.approx(6.0)
    assert optimizer.calls == 1


def test_ema_updates_once_only_when_deepspeed_applies_step(trainer, monkeypatch):
    applied = iter((False, True))
    engine = SimpleNamespace(
        is_gradient_accumulation_boundary=lambda: True,
        was_step_applied=lambda: next(applied),
    )
    finishes = []
    monkeypatch.setattr(trainer, "_validate_deepspeed_gradient_norm", lambda _: None)
    monkeypatch.setattr(
        trainer,
        "_finish_optimizer_step",
        lambda model, *, successful: finishes.append(successful),
    )
    adapter = _DeepSpeedStepAdapter(trainer, engine, lambda: None)
    adapter()
    adapter()
    assert finishes == [False, True]
```

- [ ] **Step 2: Write failing DoRA registration and composition-profile tests**

```python
@pytest.mark.parametrize("zero_stage,expected_calls", [(1, 0), (2, 0), (3, 1)])
def test_dora_external_parameters_are_registered_only_for_zero3(
    dora_model, monkeypatch, zero_stage, expected_calls
):
    calls = []
    monkeypatch.setattr(stage1_peft, "_register_dora_zero3_external_parameters",
                        lambda model: calls.append(model))
    validate_stage1_trainable_boundary(
        dora_model, dora_tuning(), LATENT_IDS, zero_stage=zero_stage
    )
    assert len(calls) == expected_calls


def test_stage1_components_share_bound_runtime_profile(stage1_args, dependencies):
    stage1_args.deepspeed_config = str(canonical_deepspeed_config_path(2))
    profile = bind_deepspeed_runtime_profile(stage1_args, world_size=2)
    components = build_stage1_training_components(stage1_args, dependencies=dependencies)
    assert components.trainer.deepspeed_runtime_profile is profile
```

- [ ] **Step 3: Run the new tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_trainer.py \
  tests/stage1/test_ema.py \
  tests/stage1/test_peft.py \
  tests/stage1/test_composition.py -q
```

Expected: the new profile parameter and ZeRO-1/2 norm strategy are absent, so the added tests fail while existing assertions remain green.

- [ ] **Step 4: Implement the optimizer strategy without changing the step adapter contract**

```python
def collect_deepspeed_pre_step_norm(optimizer: object, *, zero_stage: int) -> torch.Tensor:
    if zero_stage == 3:
        raw_groups = optimizer._get_norm_groups()
        if not isinstance(raw_groups, (tuple, list)) or not raw_groups:
            raise FloatingPointError("ZeRO-3 gradient norm groups are missing")
        groups = torch.stack([
            torch.as_tensor(value).detach().float().reshape(())
            for value in raw_groups
        ])
        norm = torch.linalg.vector_norm(groups) / float(optimizer.loss_scale)
    elif zero_stage in (1, 2):
        expected_partitioning = zero_stage == 2
        if bool(optimizer.partition_gradients) != expected_partitioning:
            raise RuntimeError(
                f"ZeRO-{zero_stage} optimizer partition contract mismatch"
            )
        norm = optimizer.scaled_global_norm() / float(optimizer.loss_scale)
    else:
        raise RuntimeError(f"Unsupported DeepSpeed ZeRO stage: {zero_stage}")
    norm = torch.as_tensor(norm, dtype=torch.float32)
    if norm.numel() != 1 or not torch.isfinite(norm).item():
        raise FloatingPointError("DeepSpeed pre-step gradient norm must be finite")
    return norm.reshape(())
```

Keep `_DeepSpeedStepAdapter.__call__()` as the only actual optimizer-step boundary: collect the norm immediately before the engine step, call the wrapped step once, query public `was_step_applied()`, and update EMA only when true. `_install_deepspeed_step_adapter` must accept `DeepSpeedZeroOptimizer_Stage3` for stage 3 and `DeepSpeedZeroOptimizer` for stages 1/2, rejecting mismatches with the requested profile before training.

```python
def _validate_deepspeed_gradient_norm(self, engine) -> torch.Tensor:
    zero_stage = self.deepspeed_runtime_profile.zero_stage
    if zero_stage is None:
        raise RuntimeError("DeepSpeed norm collection requires an enabled profile")
    norm = collect_deepspeed_pre_step_norm(
        engine.optimizer, zero_stage=zero_stage
    )
    self._record_gradient_diagnostic(norm)
    return norm
```

- [ ] **Step 5: Gate partition-specific EMA and DoRA behavior by the actual stage**

```python
def validate_stage1_trainable_boundary(
    model: nn.Module,
    tuning: Stage1TuningConfig,
    latent_token_ids: Sequence[int],
    *,
    zero_stage: int | None = None,
) -> Stage1ParameterReport:
    report = _validate_adapter_boundary(
        model, tuning, latent_token_ids, zero_stage=zero_stage
    )
    return report


if tuning.method == "dora" and zero_stage == 3:
    _register_dora_zero3_external_parameters(model)
```

Pass `zero_stage` into `build_stage1_tuned_model` and boundary validation. In EMA and sparse-row save helpers, use the existing `ds_id`/partitioned-parameter predicate as the final authority: ZeRO-1/2 reads replicated tensors locally; only a genuinely partitioned tensor enters `deepspeed.zero.GatheredParameters`.

- [ ] **Step 6: Bind the profile once and remove the general-runtime two-rank gate**

At the start of `run_stage1`, bind the profile from `WORLD_SIZE` and pass the same object through components and trainer:

```python
world_size = int(os.environ.get("WORLD_SIZE", "1"))
profile = bind_deepspeed_runtime_profile(args, world_size=world_size)
components = build_stage1_training_components(
    args, dependencies=dependencies, deepspeed_runtime_profile=profile
)
```

The benchmark producer validator may still require official `world_size == 2`; remove only the composition/runtime preflight that blocks ordinary fresh training at other world sizes.

- [ ] **Step 7: Run the Stage 1 focused suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_trainer.py \
  tests/stage1/test_ema.py \
  tests/stage1/test_peft.py \
  tests/stage1/test_composition.py -q
```

Expected: all selected tests pass; no GPU or distributed process is launched.

- [ ] **Step 8: Commit the Stage 1 lifecycle**

```bash
git add src/stage1/trainer.py src/stage1/ema.py src/stage1/peft.py \
  src/stage1/composition.py src/main.py tests/stage1/test_trainer.py \
  tests/stage1/test_ema.py tests/stage1/test_peft.py \
  tests/stage1/test_composition.py
git commit -m "feat: support Stage 1 ZeRO optimizer lifecycles"
```

### Task 3: Identity-Bound Numbered Checkpoints and Resume

**Files:**
- Modify: `src/deepspeed_runtime.py`
- Modify: `src/main.py:467-483`
- Modify: `src/stage1/trainer.py:799-880`
- Modify: `src/stage1/checkpointing.py:1502-1599`
- Modify: `src/stage1/token_rows.py:226-260`
- Modify: `src/stage2/trainer.py`
- Modify: `tests/stage2/integration/tiny_stage2_train.py`
- Test: `tests/test_deepspeed_runtime.py`
- Test: `tests/stage1/test_checkpointing.py`
- Test: `tests/stage1/test_trainer.py`
- Test: `tests/stage2/test_peft_checkpointing.py`

**Interfaces:**
- Consumes: `DeepSpeedRuntimeProfile` from Task 1 and both trainers' existing checkpoint hooks.
- Produces: `DEEPSPEED_RUNTIME_FILENAME`, `write_deepspeed_runtime_sidecar(directory: Path, profile: DeepSpeedRuntimeProfile) -> Path`, `read_deepspeed_runtime_sidecar(directory: Path) -> DeepSpeedRuntimeProfile`, `validate_deepspeed_resume_identity(checkpoint: Path, expected: DeepSpeedRuntimeProfile) -> None`, and `sparse_token_rows_are_partitioned(model: nn.Module) -> bool`.

- [ ] **Step 1: Write failing sidecar and mismatch tests**

```python
def test_runtime_sidecar_round_trip(tmp_path):
    profile = load_deepspeed_runtime_profile(
        canonical_deepspeed_config_path(2), world_size=2
    )
    path = write_deepspeed_runtime_sidecar(tmp_path, profile)
    assert path.name == "deepspeed-runtime.json"
    assert read_deepspeed_runtime_sidecar(tmp_path) == profile


@pytest.mark.parametrize(
    "field,replacement",
    [("zero_stage", 1), ("config_digest", "0" * 64), ("world_size", 4)],
)
def test_resume_rejects_runtime_identity_mismatch(tmp_path, field, replacement):
    expected = profile_for_stage(2, world_size=2)
    payload = {"schema_version": 1, **expected.to_dict(), field: replacement}
    (tmp_path / DEEPSPEED_RUNTIME_FILENAME).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=field):
        validate_deepspeed_resume_identity(tmp_path, expected)
```

- [ ] **Step 2: Write failing trainer save/resume tests**

```python
def test_numbered_checkpoint_writes_runtime_identity(stage1_trainer, tmp_path):
    stage1_trainer.args.output_dir = str(tmp_path)
    stage1_trainer.state.global_step = 3
    stage1_trainer._save_checkpoint(stage1_trainer.model, trial=None)
    payload = json.loads(
        (tmp_path / "checkpoint-3" / DEEPSPEED_RUNTIME_FILENAME).read_text()
    )
    assert payload["zero_stage"] == 2
    assert payload["world_size"] == 2


def test_train_publishes_runtime_identity_at_output_root(stage1_trainer, tmp_path):
    stage1_trainer.args.output_dir = str(tmp_path)
    stage1_trainer.train()
    payload = json.loads((tmp_path / DEEPSPEED_RUNTIME_FILENAME).read_text())
    assert payload["zero_stage"] == 2


def test_train_rejects_checkpoint_before_deepspeed_initialization(
    stage2_trainer, mismatched_checkpoint
):
    with pytest.raises(ValueError, match="zero_stage"):
        stage2_trainer.train(resume_from_checkpoint=str(mismatched_checkpoint))
    assert stage2_trainer.inner_train_calls == 0


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("method", ["full", "lora", "dora"])
def test_stage2_trainer_accepts_every_supported_method_profile(
    zero_stage, method, stage2_trainer_factory
):
    trainer = stage2_trainer_factory(
        method=method, profile=profile_for_stage(zero_stage, world_size=2)
    )
    assert trainer.deepspeed_runtime_profile.zero_stage == zero_stage
    if method == "full":
        assert trainer.stage2_peft_metadata is None
    else:
        assert trainer.stage2_peft_metadata.tuning_method == method
```

- [ ] **Step 3: Run the checkpoint tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/test_deepspeed_runtime.py \
  tests/stage1/test_checkpointing.py \
  tests/stage1/test_trainer.py \
  tests/stage2/test_peft_checkpointing.py -q
```

Expected: sidecar helpers and strict pre-initialization resume checks are missing.

- [ ] **Step 4: Implement atomic small-sidecar I/O and exact comparison**

```python
DEEPSPEED_RUNTIME_FILENAME = "deepspeed-runtime.json"
DEEPSPEED_RUNTIME_SCHEMA_VERSION = 1


def write_deepspeed_runtime_sidecar(
    directory: Path, profile: DeepSpeedRuntimeProfile
) -> Path:
    path = directory / DEEPSPEED_RUNTIME_FILENAME
    payload = {"schema_version": DEEPSPEED_RUNTIME_SCHEMA_VERSION, **profile.to_dict()}
    if path.exists():
        if json.loads(path.read_text()) == payload:
            return path
        raise FileExistsError(f"refusing to replace different runtime identity: {path}")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)
    return path


def validate_deepspeed_resume_identity(
    checkpoint: Path, expected: DeepSpeedRuntimeProfile
) -> None:
    actual = read_deepspeed_runtime_sidecar(checkpoint)
    for field in ("enabled", "zero_stage", "config_digest", "world_size"):
        if getattr(actual, field) != getattr(expected, field):
            raise ValueError(
                f"DeepSpeed resume identity mismatch for {field}: "
                f"checkpoint={getattr(actual, field)!r}, current={getattr(expected, field)!r}"
            )
```

The sidecar stores the canonical path for provenance, but resume portability compares the digest rather than requiring the old absolute path to exist.

- [ ] **Step 5: Wire both trainers to the same save and resume contract**

At the beginning of `train`, rank 0 idempotently writes the same sidecar to `args.output_dir` as the run manifest, propagates write failure across ranks, and then proceeds. After `super()._save_checkpoint`, rank 0 writes the small sidecar into the numbered checkpoint, every rank reduces a one-bit local failure flag, and only then do all ranks synchronize. This mirrors the existing Stage 1 EMA failure propagation and prevents a rank-0 write exception from leaving peers waiting forever. Before either trainer calls `super().train(resume_from_checkpoint=...)`, validate the checkpoint sidecar on every rank from the shared filesystem. A missing sidecar in a numbered checkpoint is an explicit identity error; final Stage 1 handoff artifacts that are not Trainer resumes retain their current classifier.

```python
def train(self, *args, resume_from_checkpoint=None, **kwargs):
    if resume_from_checkpoint:
        validate_deepspeed_resume_identity(
            Path(resume_from_checkpoint), self.deepspeed_runtime_profile
        )
    return super().train(
        *args, resume_from_checkpoint=resume_from_checkpoint, **kwargs
    )
```

Stage 1 retains EMA save ordering and checkpoint rotation deferral. Stage 2 sparse rows call `GatheredParameters` only if the target tensor carries partition metadata; replicated ZeRO-1/2 state is written by rank 0 alone.

```python
def sparse_token_rows_are_partitioned(model: nn.Module) -> bool:
    rows = find_sparse_output_token_rows(model)
    return _is_partitioned(rows.output_delta)


partitioned_sparse_rows = sparse_token_rows_are_partitioned(self.model)
if partitioned_sparse_rows:
    # Every rank enters the collective; only rank 0 publishes to the target.
    if self.args.should_save:
        save_sparse_token_rows(self.model, target)
    else:
        with tempfile.TemporaryDirectory(prefix="stage2-token-rows-") as temporary:
            save_sparse_token_rows(self.model, Path(temporary))
elif self.args.should_save:
    save_sparse_token_rows(self.model, target)
```

Bind and pass the same profile in `run_stage2` before trainer construction:

```python
profile = bind_deepspeed_runtime_profile(
    args, world_size=int(os.environ.get("WORLD_SIZE", "1"))
)
trainer = components.trainer_class(
    model=components.model,
    args=build_stage2_training_arguments(args),
    train_dataset=components.train_dataset,
    data_collator=components.data_collator,
    processing_class=components.processor,
    stage2_peft_metadata=components.peft_metadata,
    deepspeed_runtime_profile=profile,
)
```

Update both `Stage2Trainer` constructions in `tiny_stage2_train.py` to receive the same loaded profile. This keeps the existing tiny Full/LoRA/DoRA save/resume harness usable for targeted compatibility checks.

- [ ] **Step 6: Run focused checkpoint tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/test_deepspeed_runtime.py \
  tests/stage1/test_checkpointing.py \
  tests/stage1/test_trainer.py \
  tests/stage2/test_peft_checkpointing.py -q
```

Expected: all selected tests pass, including ZeRO-1/2 replicated and ZeRO-3 partitioned save branches.

- [ ] **Step 7: Commit the resume identity contract**

```bash
git add src/deepspeed_runtime.py src/main.py src/stage1/trainer.py \
  src/stage1/checkpointing.py src/stage1/token_rows.py src/stage2/trainer.py \
  tests/test_deepspeed_runtime.py tests/stage1/test_checkpointing.py \
  tests/stage1/test_trainer.py tests/stage2/test_peft_checkpointing.py \
  tests/stage2/integration/tiny_stage2_train.py
git commit -m "feat: bind checkpoints to ZeRO runtime identity"
```

### Task 4: Stage 2 Latent-Block-Balanced Sampler

**Files:**
- Create: `src/stage2/sampler.py`
- Create: `tests/stage2/test_sampler.py`
- Modify: `src/stage2/trainer.py`
- Modify: `tests/stage2/test_routing.py`
- Modify: `tests/stage2/test_peft_checkpointing.py`

**Interfaces:**
- Consumes: dataset items exposing integer `latent_block_count`, the Stage 2 trainer's seed/world size, and Trainer's normal epoch propagation.
- Produces: `Stage2SamplerIdentity`, `LatentBlockBalancedSampler(dataset: Dataset, *, num_replicas: int, seed: int)`, `Stage2Trainer.stage2_sampler_identity`, and `STAGE2_SAMPLER_FILENAME`.

- [ ] **Step 1: Write failing ordering, padding, and epoch tests**

```python
def test_global_groups_have_one_latent_count_across_ranks():
    dataset = examples_with_counts([1, 2, 1, 3, 2, 3, 3])
    sampler = LatentBlockBalancedSampler(dataset, num_replicas=2, seed=42)
    indices = list(sampler)
    groups = [indices[offset:offset + 2] for offset in range(0, len(indices), 2)]
    assert all(
        len({dataset[index].latent_block_count for index in group}) == 1
        for group in groups
    )


def test_sampler_padding_and_identity_are_explicit():
    dataset = examples_with_counts([1, 1, 1, 2, 2])
    sampler = LatentBlockBalancedSampler(dataset, num_replicas=2, seed=42)
    assert len(sampler) == 6
    assert dict(sampler.identity.bucket_sizes) == {1: 3, 2: 2}
    assert sampler.identity.padded_samples == 1


def test_epoch_is_deterministic_but_changes_shuffle_order():
    sampler = LatentBlockBalancedSampler(
        examples_with_counts([1, 1, 1, 1, 2, 2, 2, 2]),
        num_replicas=2,
        seed=42,
    )
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == epoch0
    assert epoch1 != epoch0


def test_multigpu_trainer_publishes_sampler_identity(stage2_trainer_factory, tmp_path):
    trainer = stage2_trainer_factory(world_size=2, output_dir=tmp_path)
    trainer.get_train_dataloader()
    payload = json.loads((tmp_path / STAGE2_SAMPLER_FILENAME).read_text())
    assert payload["world_size"] == 2
    assert payload["padded_samples"] >= 0
```

- [ ] **Step 2: Lock the Accelerate sharding assumption with a focused test**

```python
def test_batch_sampler_shard_assigns_aligned_global_groups_to_two_ranks():
    dataset = examples_with_counts([1, 2, 1, 2, 3, 3, 3])
    sampler = LatentBlockBalancedSampler(dataset, num_replicas=2, seed=42)
    batches = BatchSampler(sampler, batch_size=1, drop_last=False)
    rank_batches = [
        list(BatchSamplerShard(batches, num_processes=2, process_index=rank,
                               split_batches=False))
        for rank in range(2)
    ]
    assert len(rank_batches[0]) == len(rank_batches[1])
    for left, right in zip(*rank_batches):
        assert dataset[left[0]].latent_block_count == dataset[right[0]].latent_block_count
```

- [ ] **Step 3: Run sampler tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest tests/stage2/test_sampler.py -q
```

Expected: collection fails because `src.stage2.sampler` does not exist.

- [ ] **Step 4: Implement the global sampler and immutable identity**

```python
STAGE2_SAMPLER_FILENAME = "stage2-sampler.json"


@dataclass(frozen=True)
class Stage2SamplerIdentity:
    schema_version: int
    seed: int
    world_size: int
    dataset_size: int
    bucket_sizes: tuple[tuple[int, int], ...]
    padded_samples: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "world_size": self.world_size,
            "dataset_size": self.dataset_size,
            "bucket_sizes": {str(key): value for key, value in self.bucket_sizes},
            "padded_samples": self.padded_samples,
        }


class LatentBlockBalancedSampler(Sampler[int]):
    def __init__(self, dataset, *, num_replicas: int, seed: int):
        if num_replicas < 2:
            raise ValueError("balanced sampler requires at least two replicas")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.seed = seed
        self.epoch = 0
        self._buckets = _bucket_indices(dataset)
        padded = sum((-len(indices)) % num_replicas for indices in self._buckets.values())
        self.identity = Stage2SamplerIdentity(
            schema_version=1,
            seed=seed,
            world_size=num_replicas,
            dataset_size=len(dataset),
            bucket_sizes=tuple(
                (key, len(value)) for key, value in sorted(self._buckets.items())
            ),
            padded_samples=padded,
        )

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        groups = []
        for block_count, original in sorted(self._buckets.items()):
            order = torch.randperm(len(original), generator=generator).tolist()
            indices = [original[position] for position in order]
            needed = (-len(indices)) % self.num_replicas
            base = tuple(indices)
            indices.extend(
                base[position % len(base)] for position in range(needed)
            )
            groups.extend(
                indices[offset:offset + self.num_replicas]
                for offset in range(0, len(indices), self.num_replicas)
            )
        group_order = torch.randperm(len(groups), generator=generator).tolist()
        return iter(index for position in group_order for index in groups[position])
```

Reject latent counts outside 1 through 6 and empty datasets with actionable `ValueError`s. `__len__` returns original plus deterministic padding; `set_epoch(epoch)` rejects negative epochs.

`Stage2SamplerIdentity.to_dict()` encodes `bucket_sizes` as a JSON object with decimal string keys; its reader converts keys back to integers, sorts them, and reconstructs the tuple before equality comparison.

- [ ] **Step 5: Install only for multi-GPU Stage 2 and preserve world-size-one behavior**

```python
def _get_train_sampler(self):
    if self.args.world_size <= 1:
        self.stage2_sampler_identity = None
        return super()._get_train_sampler()
    sampler = LatentBlockBalancedSampler(
        self.train_dataset,
        num_replicas=self.args.world_size,
        seed=self.args.seed,
    )
    self.stage2_sampler_identity = sampler.identity
    return sampler


def get_train_dataloader(self):
    dataloader = super().get_train_dataloader()
    if self.stage2_sampler_identity is not None:
        self._publish_stage2_sampler_identity(
            Path(self.args.output_dir), global_step=0
        )
    return dataloader
```

When multi-GPU training materializes this sampler, rank 0 idempotently writes `stage2-sampler.json` at the output root so runs without a numbered checkpoint still expose bucket statistics. Write the same identity beside the runtime sidecar in numbered checkpoints with the current `global_step`; resume compares seed, world size, dataset size, bucket sizes, and padding before Trainer restores its own RNG/global-step state. Use the same cross-rank failure reduction as the runtime sidecar. World-size-one runs leave the sampler identity absent and preserve the existing order.

- [ ] **Step 6: Run Stage 2 sampler and routing tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_sampler.py \
  tests/stage2/test_routing.py \
  tests/stage2/test_peft_checkpointing.py -q
```

Expected: all selected tests pass; the single-GPU test observes the prior sampler, and two-rank emulation observes equal latent block counts at every micro-step.

- [ ] **Step 7: Commit the balanced sampler**

```bash
git add src/stage2/sampler.py src/stage2/trainer.py \
  tests/stage2/test_sampler.py tests/stage2/test_routing.py \
  tests/stage2/test_peft_checkpointing.py
git commit -m "feat: balance Stage 2 latent blocks across ranks"
```

### Task 5: Generalize Stage 1 Producer Evidence Across ZeRO Stages

**Files:**
- Modify: `src/stage1/benchmark.py:152-250,1310-1605,2440-2830`
- Modify: `src/stage1/composition.py:562-760`
- Modify: `tests/stage1/test_benchmark.py`
- Modify: `tests/stage1/test_composition.py`

**Interfaces:**
- Consumes: the bound `DeepSpeedRuntimeProfile`, existing audited sample-stream digest, rank memory/timing rows, and producer evidence JSON.
- Produces: evidence fields `zero_stage: int`, `deepspeed_config_digest: str`, `optimizer_steps: int`, `samples_seen: int`, exact reuse key `Stage1ProducerIdentity`, `migrate_stage1_producer_evidence(payload: Mapping[str, object]) -> dict[str, object]`, and `reusable_stage1_producer(producer: Stage1BenchmarkEvidence, requested: Mapping[str, object]) -> bool`.

- [ ] **Step 1: Write failing schema and reuse-isolation tests**

```python
def test_stage1_evidence_serializes_zero_identity(valid_evidence):
    payload = valid_evidence.to_dict()
    assert payload["zero_stage"] == 2
    assert payload["deepspeed_config_digest"] == "d" * 64
    assert payload["optimizer_steps"] == 3
    assert payload["samples_seen"] == 48


@pytest.mark.parametrize(
    "field,replacement",
    [("zero_stage", 1), ("deepspeed_config_digest", "e" * 64),
     ("sample_stream_digest", "f" * 64)],
)
def test_producer_reuse_rejects_identity_change(valid_producer, field, replacement):
    requested = valid_producer.identity.to_dict() | {field: replacement}
    assert reusable_stage1_producer(valid_producer, requested) is False


def test_legacy_zero3_evidence_migrates_without_rehashing(legacy_payload):
    migrated = migrate_stage1_producer_evidence(legacy_payload)
    assert migrated["zero_stage"] == 3
    assert migrated["deepspeed_config_digest"] == legacy_payload["run_config"][
        "deepspeed_config_digest"
    ]
```

- [ ] **Step 2: Run benchmark tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_benchmark.py \
  tests/stage1/test_composition.py -q
```

Expected: evidence lacks the ZeRO-stage fields and the current producer validator rejects accumulation values other than 2.

- [ ] **Step 3: Extend the evidence type and define its exact reuse key**

```python
@dataclass(frozen=True)
class Stage1ProducerIdentity:
    method: str
    zero_stage: int
    deepspeed_config_digest: str
    world_size: int
    micro_batch: int
    gradient_accumulation_steps: int
    effective_global_batch: int
    sample_stream_digest: str
    model_id: str
    seed: int

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Stage1BenchmarkEvidence:
    status: Literal["success", "capacity_oom", "integration_failure"]
    method: str
    zero_stage: int
    deepspeed_config_digest: str
    micro_batch: int
    world_size: int
    gradient_accumulation_steps: int
    effective_global_batch: int
    optimizer_steps: int
    samples_seen: int
    sample_stream_digest: str
    model_id: str
    seed: int
    samples_per_second: float
    steps_per_second: float
    rank_peak_allocated_bytes: tuple[int, ...]
    rank_peak_reserved_bytes: tuple[int, ...]
    model_load_seconds: float
    trainer_seconds: float
    total_seconds: float
    final_save_seconds: float
    trainable_parameters: int
    total_parameters: int
    checkpoint_bytes: int
    final_loss: float
    representative_gradient_norm: float
    git_commit: str
    dependency_versions: dict[str, str]
    gpu_names: tuple[str, ...]
    config_digest: str
    probe_row_indices: tuple[int, ...]

    @property
    def identity(self) -> Stage1ProducerIdentity:
        return Stage1ProducerIdentity(
            method=self.method,
            zero_stage=self.zero_stage,
            deepspeed_config_digest=self.deepspeed_config_digest,
            world_size=self.world_size,
            micro_batch=self.micro_batch,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            effective_global_batch=self.effective_global_batch,
            sample_stream_digest=self.sample_stream_digest,
            model_id=self.model_id,
            seed=self.seed,
        )
```

Retain all existing loss, gradient, parameter, version, audit, and config fields after these required fields. Schema parsing must validate positive counts, finite metrics, two rank-memory entries for official producers, and `effective_global_batch == micro_batch * world_size * gradient_accumulation_steps`.

Implement exact reuse as dataclass equality after parsing the request with the same strict validators:

```python
LEGACY_STAGE1_PRODUCER_SCHEMA_VERSION = 1
STAGE1_PRODUCER_SCHEMA_VERSION = 2


def reusable_stage1_producer(
    producer: Stage1BenchmarkEvidence,
    requested: Mapping[str, object],
) -> bool:
    return producer.identity == Stage1ProducerIdentity(**dict(requested))


def migrate_stage1_producer_evidence(
    payload: Mapping[str, object],
) -> dict[str, object]:
    migrated = copy.deepcopy(dict(payload))
    if migrated.get("schema_version") != LEGACY_STAGE1_PRODUCER_SCHEMA_VERSION:
        raise ValueError("unsupported legacy Stage 1 evidence schema")
    run_config = migrated.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("legacy evidence is missing its sealed run_config")
    digest = _require_digest(
        "deepspeed_config_digest", run_config.get("deepspeed_config_digest")
    )
    migrated["zero_stage"] = 3
    migrated["deepspeed_config_digest"] = digest
    migrated["sample_stream_digest"] = _require_digest(
        "audit_digest", run_config.get("audit_digest")
    )
    migrated["model_id"] = _require_string("model", run_config.get("model"))
    migrated["seed"] = _require_int("seed", run_config.get("seed"))
    migrated["status"] = {
        "success": "success",
        "oom": "capacity_oom",
        "failure": "integration_failure",
    }[migrated["status"]]
    return migrated
```

- [ ] **Step 4: Generalize producer configuration while preserving legacy ZeRO-3 readers**

Replace the hard-coded `gradient_accumulation_steps == 2` rule with:

```python
if payload["producer_kind"] == "zero_matrix":
    if payload["world_size"] != 2:
        raise ValueError("official zero_matrix producers require world_size 2")
    if payload["effective_global_batch"] != 16:
        raise ValueError("official zero_matrix producers require effective global batch 16")
    if payload["optimizer_steps"] != 3:
        raise ValueError("zero_matrix producers require exactly 3 optimizer steps")
elif payload["producer_kind"] == "legacy_zero3":
    _validate_legacy_zero3_controls(payload)
```

`migrate_stage1_producer_evidence` accepts only the previous schema that already contains a sealed DeepSpeed digest, maps legacy status `oom` to `capacity_oom` and `failure` to `integration_failure`, infers stage 3, and reuses that digest. It must not open or hash a config, model, dataset, or checkpoint.

- [ ] **Step 5: Populate evidence from the already-bound profile**

```python
profile = args.deepspeed_runtime_profile
if profile.zero_stage is None or profile.config_digest is None:
    raise ValueError("Stage 1 matrix producers require DeepSpeed")
evidence = Stage1BenchmarkEvidence(
    status="success",
    method=config["method"],
    zero_stage=profile.zero_stage,
    deepspeed_config_digest=profile.config_digest,
    micro_batch=args.per_device_train_batch_size,
    world_size=profile.world_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    effective_global_batch=(
        args.per_device_train_batch_size
        * profile.world_size
        * args.gradient_accumulation_steps
    ),
    optimizer_steps=optimizer_steps,
    samples_seen=samples_seen,
    sample_stream_digest=config["audit_digest"],
    model_id=config["model"],
    seed=config["seed"],
    model_load_seconds=float(model_load_seconds),
    trainer_seconds=float(trainer_seconds),
    total_seconds=float(total_seconds),
    final_save_seconds=float(save_seconds),
    samples_per_second=float(metrics.get("train_samples_per_second", 0.0)),
    steps_per_second=float(metrics.get("train_steps_per_second", 0.0)),
    rank_peak_allocated_bytes=rank_allocated,
    rank_peak_reserved_bytes=rank_reserved,
    trainable_parameters=components.parameter_report.trainable_parameters,
    total_parameters=components.parameter_report.total_parameters,
    checkpoint_bytes=artifact["checkpoint_bytes"],
    final_loss=float(metrics.get("train_loss", 0.0)),
    representative_gradient_norm=float(
        getattr(components.trainer, "_representative_gradient_norm", 0.0) or 0.0
    ),
    git_commit=observed["git_commit"],
    dependency_versions=dict(observed["dependency_versions"]),
    gpu_names=gpu_names,
    config_digest=config["config_digest"],
    probe_row_indices=tuple(config["probe_row_indices"]),
)
```

Do not call `_small_config_digest` once a profile exists. The benchmark run manifest, evidence, and publication copy `profile.config_digest` forward.

- [ ] **Step 6: Run focused benchmark tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_benchmark.py \
  tests/stage1/test_composition.py -q
```

Expected: all selected tests pass; old ZeRO-3 producer fixtures still load, while cross-stage reuse is rejected.

- [ ] **Step 7: Commit the generalized evidence schema**

```bash
git add src/stage1/benchmark.py src/stage1/composition.py \
  tests/stage1/test_benchmark.py tests/stage1/test_composition.py
git commit -m "feat: record ZeRO identity in Stage 1 evidence"
```

### Task 6: Bounded Reverse-Order Stage 1 Matrix Coordinator

**Files:**
- Create: `src/stage1/zero_matrix.py`
- Create: `tests/stage1/test_zero_matrix.py`
- Create: `run_stage1_zero_matrix.sh`
- Modify: `run_stage1_batch_probe.sh`
- Modify: `tests/stage1/test_launcher.py`

**Interfaces:**
- Consumes: `Stage1ProducerIdentity`, producer evidence from Task 5, canonical profiles from Task 1, and existing benchmark sample-audit witnesses.
- Produces: `ZERO_MATRIX_CANDIDATES`, `ZeroMatrixAttempt`, `classify_zero_matrix_attempt(*, returncode: int, stderr: str, producer: Stage1BenchmarkEvidence | None = None) -> Literal["success", "capacity_oom", "integration_failure"]`, `select_fastest_success(rows: Sequence[ZeroMatrixAttempt]) -> ZeroMatrixAttempt`, CLI commands `audit-status`, `prepare`, `reuse-status`, `classify`, and `select`, plus `zero-matrix-report.json`.

- [ ] **Step 1: Write failing candidate, classification, and selection tests**

```python
def test_candidates_are_fixed_egb16_in_reverse_microbatch_order():
    assert [
        (candidate.micro_batch, candidate.gradient_accumulation_steps)
        for candidate in ZERO_MATRIX_CANDIDATES
    ] == [(8, 1), (4, 2), (2, 4), (1, 8)]
    assert all(candidate.effective_global_batch == 16
               for candidate in ZERO_MATRIX_CANDIDATES)


def test_only_cuda_allocator_oom_is_capacity_failure():
    assert classify_zero_matrix_attempt(
        returncode=1, stderr="torch.OutOfMemoryError: CUDA out of memory"
    ) == "capacity_oom"
    assert classify_zero_matrix_attempt(
        returncode=1, stderr="NCCL watchdog collective timeout"
    ) == "integration_failure"


def test_fastest_success_wins_and_failure_is_never_selected():
    rows = [success(micro=4, speed=3.2), success(micro=2, speed=3.8),
            capacity_oom(micro=8)]
    assert select_fastest_success(rows).micro_batch == 2
```

- [ ] **Step 2: Run matrix tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_zero_matrix.py \
  tests/stage1/test_launcher.py -q
```

Expected: the coordinator module and matrix launcher do not exist.

- [ ] **Step 3: Implement fixed candidates and strict evidence selection**

```python
@dataclass(frozen=True)
class ZeroBatchCandidate:
    micro_batch: int
    gradient_accumulation_steps: int
    effective_global_batch: int = 16


ZERO_MATRIX_CANDIDATES = (
    ZeroBatchCandidate(8, 1),
    ZeroBatchCandidate(4, 2),
    ZeroBatchCandidate(2, 4),
    ZeroBatchCandidate(1, 8),
)


@dataclass(frozen=True)
class ZeroMatrixAttempt:
    identity: Stage1ProducerIdentity
    status: Literal["success", "capacity_oom", "integration_failure"]
    samples_per_second: float | None
    evidence_path: Path

    @property
    def micro_batch(self) -> int:
        return self.identity.micro_batch


def classify_zero_matrix_attempt(
    *,
    returncode: int,
    stderr: str,
    producer: Stage1BenchmarkEvidence | None = None,
) -> Literal["success", "capacity_oom", "integration_failure"]:
    if returncode == 0:
        return (
            "success"
            if producer is not None and producer.status == "success"
            else "integration_failure"
        )
    normalized = stderr.casefold()
    oom_markers = (
        "torch.outofmemoryerror",
        "cuda out of memory",
        "cudamalloc failed",
    )
    return (
        "capacity_oom"
        if any(marker in normalized for marker in oom_markers)
        else "integration_failure"
    )


def select_fastest_success(rows: Sequence[ZeroMatrixAttempt]) -> ZeroMatrixAttempt:
    successes = [
        row
        for row in rows
        if row.status == "success" and row.samples_per_second is not None
    ]
    if not successes:
        raise RuntimeError("No successful Stage 1 batch candidate")
    return max(successes, key=lambda row: (row.samples_per_second, row.micro_batch))
```

`classify_zero_matrix_attempt` accepts only an exit result plus the producer's bounded stderr. Match `torch.OutOfMemoryError`, `CUDA out of memory`, or DeepSpeed CUDA allocator OOM as `capacity_oom`; any NCCL, timeout, nonfinite, contract, missing-evidence, or unknown nonzero exit is `integration_failure`. A zero exit without a complete identity-matching producer JSON is also `integration_failure`.

- [ ] **Step 4: Implement the coordinator CLI without model/checkpoint hashing**

```bash
zero_result_root=/root/workspace/ILVR/results/zero-stage-compatibility
./venv/ilvr-venv/bin/python -m src.stage1.zero_matrix reuse-status \
  --matrix-result-dir "$zero_result_root/stage1-zero-matrix" \
  --method dora \
  --deepspeed-config configs/deepspeed_zero2.json \
  --world-size 2 \
  --sample-audit "$zero_result_root/stage1-zero-matrix/sample-audit.json"
```

The command returns JSON rows in fixed candidate order with `reuse=true` only for a complete exact identity. `audit-status --candidate-audit PATH` accepts the canonical result path followed by optional read-only legacy paths, validates only stored path/stat/count identity, prints the first exact match, and exits nonzero if none match; it never writes candidates or reads/hashes dataset contents. `prepare` writes small run configs that reference the selected sample-audit digest. `select` writes `zero-matrix-report.json` atomically under the canonical matrix result directory and copies producer witness paths/digests; it must not re-run validators or hash model/checkpoint trees.

For a full-matrix CLI invocation, load each of the three canonical runtime profiles exactly once into `{zero_stage: profile}` and pass those objects through all method/candidate rows. Do not call the config loader from the inner candidate loop.

- [ ] **Step 5: Implement the reverse-order shell controller and selected resume**

The launcher loops stages `3 2 1`, methods `full dora`, and the four fixed candidates. Before each producer it runs `reuse-status`; reusable success/failure rows are not relaunched. For each actual launch it creates a collision-resistant run suffix and resolves three distinct directories from one experiment ID:

```bash
zero_checkpoint_root=/root/workspace/ILVR/checkpoints/zero-stage-compatibility
zero_result_root=/root/workspace/ILVR/results/zero-stage-compatibility
zero_log_root=/root/workspace/ILVR/logs/zero-stage-compatibility
run_suffix="$(date -u +%Y%m%d-%H%M%S)-$$-$counter"
experiment_id="stage1-zero${zero_stage}-${method}-mb${micro_batch}-ga${gradient_accumulation}-candidate-$run_suffix"
checkpoint_dir="$zero_checkpoint_root/$experiment_id"
result_dir="$zero_result_root/$experiment_id"
log_dir="$zero_log_root/$experiment_id"

mkdir -p -- "$zero_checkpoint_root" "$zero_result_root" "$zero_log_root"
mkdir -- "$checkpoint_dir" "$result_dir" "$log_dir"
"$launcher" \
--method "$method" \
--model "$model" \
--source-data-path "$source_data_path" \
--data-path "$probe_data" \
--image-root "$image_root" \
--deepspeed-config "configs/deepspeed_zero${zero_stage}.json" \
--micro-batch "$micro_batch" \
--gradient-accumulation "$gradient_accumulation" \
--max-steps 3 \
--save-strategy no \
--no-final-save \
--output-dir "$checkpoint_dir" \
--log-file "$log_dir/train.log" \
--benchmark-run-config "$result_dir/run-config.json" \
--producer-evidence "$result_dir/producer-evidence.json" \
>"$log_dir/stdout.log" 2>"$log_dir/stderr.log"
```

After all four rows are present, `select` chooses the highest observed samples/sec among successes. Launch exactly one selected checkpoint-producing run using semantic segment `selected-save` and one one-step same-identity resume using `selected-resume` per `(zero_stage, method)`; append a distinct run suffix to each and use the resulting ID under all three canonical roots. If the selected resume witness already matches, skip it. Refactor `run_stage1_batch_probe.sh` to use the same checkpoint/result/log split and never create an attempt-local `output/`, `evidence/`, or `logs/` bundle under a caller-provided mixed root.

- [ ] **Step 6: Run coordinator and launcher dry-run tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_zero_matrix.py \
  tests/stage1/test_launcher.py -q
bash -n run_stage1_zero_matrix.sh run_stage1_batch_probe.sh
```

Expected: tests pass, shell syntax exits 0, and dry-run output is ordered ZeRO-3 → ZeRO-2 → ZeRO-1 with batch 8 → 4 → 2 → 1.

- [ ] **Step 7: Commit the matrix coordinator**

```bash
git add src/stage1/zero_matrix.py tests/stage1/test_zero_matrix.py \
  run_stage1_zero_matrix.sh run_stage1_batch_probe.sh \
  tests/stage1/test_launcher.py
git commit -m "feat: add bounded Stage 1 ZeRO batch matrix"
```

### Task 7: Minimal Distributed Compatibility Harnesses

**Files:**
- Create: `tests/stage1/zero_compat_diagnostic.py`
- Modify: `tests/stage1/test_distributed_smoke.py`
- Create: `tests/stage2/integration/stage2_rank_skew_probe.py`
- Modify: `tests/stage2/integration/test_distributed_contract.py`

**Interfaces:**
- Consumes: `resolve_run_artifact_paths` from Task 1, canonical configs, runtime sidecars, Stage 1 trainer lifecycle, and Stage 2 balanced sampler.
- Produces: opt-in canonical JSON witnesses for Stage 1 `(zero_stage, method, save/resume)` and Stage 2 `(zero_stage, per-rank latent-block sequence, one applied step)` without accepting arbitrary output paths.

- [ ] **Step 1: Write failing harness command-contract tests**

```python
def test_stage1_diagnostic_accepts_all_supported_pairs():
    parser = build_zero_compat_parser()
    for stage in (1, 2, 3):
        for method in ("full", "dora"):
            args = parser.parse_args([
                "--zero-stage", str(stage), "--method", method,
                "--deepspeed", f"configs/deepspeed_zero{stage}.json",
                "--experiment-id", f"stage1-zero{stage}-{method}-tiny-lifecycle",
            ])
            assert (args.zero_stage, args.method) == (stage, method)


def test_rank_skew_harness_requires_one_optimizer_step():
    args = build_rank_skew_parser().parse_args([
        "--deepspeed", "configs/deepspeed_zero3.json",
        "--model", "Qwen/Qwen2.5-VL-3B-Instruct",
        "--stage1-checkpoint", "/tmp/stage1",
        "--data-path", "/tmp/train.jsonl",
        "--image-root", "/tmp/images",
        "--experiment-id", "stage2-zero3-dora-mb1-ga8-rank-skew",
    ])
    assert args.max_steps == 1
    assert args.micro_batch == 1
    assert args.gradient_accumulation_steps == 8
```

- [ ] **Step 2: Run harness contract tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_distributed_smoke.py \
  tests/stage2/integration/test_distributed_contract.py -q
```

Expected: imports fail for the new harness parser functions.

- [ ] **Step 3: Implement the Stage 1 tiny lifecycle harness**

Adapt the existing tiny model/data fixtures, but parameterize config and method instead of copying the production benchmark. The process must perform one applied optimizer step, save `checkpoint-1`, resume the same identity for one more step, and write only small scalar/path evidence:

```python
paths = resolve_run_artifact_paths(args.experiment_id)
checkpoint = paths.checkpoint_dir / "checkpoint-1"
witness = {
    "schema_version": 1,
    "stage": "stage1",
    "zero_stage": profile.zero_stage,
    "config_digest": profile.config_digest,
    "world_size": profile.world_size,
    "method": args.method,
    "initial_global_step": 1,
    "resumed_global_step": 2,
    "finite_gradient_norm": math.isfinite(observed_norm),
    "checkpoint": str(checkpoint),
}
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    paths.result_dir.mkdir(parents=True, exist_ok=False)
    (paths.result_dir / "witness.json").write_text(
        json.dumps(witness, sort_keys=True, indent=2) + "\n"
    )
```

Both Trainer phases belong to one bounded lifecycle execution: they write below `paths.checkpoint_dir`, the second phase reads its `checkpoint-1`, and the invoking shell redirects combined stdout/stderr below `paths.log_dir`. Update `tiny_stage2_train.py` to use the same single `--experiment-id` contract and place its combined resume witness under `paths.result_dir`.

Use this harness only for ZeRO-1/2 Full/DoRA during Task 9; the existing validated ZeRO-3 lifecycle supplies the regression witness unless common-path tests demand the one lightweight rerun.

- [ ] **Step 4: Implement the Stage 2 rank-skew one-step harness**

Load the official model and Stage 1 handoff once per process, then select a small audited CoMT subset containing at least two latent-count buckets and enough examples for eight micro-steps per rank. Keep DoRA as the single most demanding Stage 2 tuning path for this distributed probe. The collator records the latent count seen on each rank and the recurrence records decoder-forward counts. Reset CUDA peak statistics immediately before training; after one optimizer step, gather the per-rank sequences and memory metrics and assert alignment before writing:

```python
def select_probe_examples(examples: Sequence[Stage2PreparedExample]):
    buckets: dict[int, list[Stage2PreparedExample]] = defaultdict(list)
    for example in examples:
        buckets[example.latent_block_count].append(example)
    available = [count for count, rows in sorted(buckets.items()) if rows]
    if len(available) < 2:
        raise ValueError("rank-skew probe requires at least two latent-count buckets")
    selected = []
    for count in available[:2]:
        rows = buckets[count]
        selected.extend(rows[index % len(rows)] for index in range(8))
    return selected
```

This yields 16 retained examples split evenly across two buckets. With world size 2, micro-batch 1, and accumulation 8, the balanced sampler supplies exactly eight aligned micro-steps per rank without a second data pass.

```python
paths = resolve_run_artifact_paths(args.experiment_id)
witness = {
    "schema_version": 1,
    "stage": "stage2",
    "zero_stage": profile.zero_stage,
    "config_digest": profile.config_digest,
    "world_size": 2,
    "optimizer_steps": trainer.state.global_step,
    "rank_block_counts": gathered_rank_counts,
    "aligned_at_every_micro_step": all(
        len(set(counts)) == 1 for counts in zip(*gathered_rank_counts)
    ),
    "padded_samples": trainer.stage2_sampler_identity.padded_samples,
    "rank_peak_allocated_bytes": gathered_peak_allocated,
    "rank_peak_reserved_bytes": gathered_peak_reserved,
}
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    paths.result_dir.mkdir(parents=True, exist_ok=False)
    (paths.result_dir / "rank-skew.json").write_text(
        json.dumps(witness, sort_keys=True, indent=2) + "\n"
    )
```

Exit nonzero if the step count is not 1, either rank executes other than eight micro-steps, any rank sequence differs in length, any micro-step has differing block counts/forward counts, a memory metric is invalid, or the output profile does not match the selected config. This single probe is both the rank-skew compatibility check and the Stage 2 peak-memory producer; do not add another Stage 2 smoke or memory run.

- [ ] **Step 5: Run only CPU parser/fixture tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_distributed_smoke.py \
  tests/stage2/integration/test_distributed_contract.py -q
```

Expected: all selected tests pass and no distributed process is launched because hardware tests remain opt-in.

- [ ] **Step 6: Commit the bounded harnesses**

```bash
git add tests/stage1/zero_compat_diagnostic.py \
  tests/stage1/test_distributed_smoke.py \
  tests/stage2/integration/stage2_rank_skew_probe.py \
  tests/stage2/integration/test_distributed_contract.py
git commit -m "test: add bounded ZeRO compatibility harnesses"
```

### Task 8: Focused CPU Regression and Operator Documentation

**Files:**
- Create: `docs/zero-stage-training.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the canonical configs, launchers, checkpoint contract, and evidence CLI from Tasks 1–7.
- Produces: runnable fresh/resume examples and a green focused CPU regression witness before GPU work.

- [ ] **Step 1: Write the support and command guide with exact invocations**

Document this support table:

| Stage | Full | LoRA | DoRA | Official 2-GPU batch |
|---|---|---|---|---|
| Stage 1 | ZeRO-1/2/3 | existing legacy path only | ZeRO-1/2/3 | matrix-selected, EGB 16 |
| Stage 2 | ZeRO-1/2/3 | ZeRO-1/2/3 | ZeRO-1/2/3 | micro 1 / accumulation 8 |

Include fresh Stage 1 and Stage 2 commands selecting `configs/deepspeed_zero1.json`, `deepspeed_zero2.json`, or `deepspeed_zero3.json`, plus same-identity resume examples. State that changing stage, digest, or world size is rejected before DeepSpeed initialization.

Add an artifact-ownership section containing the three exact canonical roots, the shared experiment-ID rule, and the read-only legacy `outputs/` rule. State explicitly that checkpoint inputs may come from other validated locations, while every newly generated checkpoint/result/log must be written below its canonical root even when the command runs from a linked worktree.

Use these exact ZeRO-2 examples in the guide; changing only the canonical config filename selects another stage:

```bash
zero_checkpoint_root=/root/workspace/ILVR/checkpoints/zero-stage-compatibility
zero_log_root=/root/workspace/ILVR/logs/zero-stage-compatibility
run_suffix="$(date -u +%Y%m%d-%H%M%S)-$$-${RANDOM}"
experiment_id="stage1-zero2-dora-fresh-example-$run_suffix"

./run_stage1_training.sh \
  --method dora \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path "$PWD/data/TRAIN.jsonl" \
  --image-root "$PWD/data" \
  --output-dir "$zero_checkpoint_root/$experiment_id" \
  --log-file "$zero_log_root/$experiment_id/train.log" \
  --micro-batch 1 \
  --gradient-accumulation 8 \
  --epochs 3 \
  --learning-rate 1e-5 \
  --seed 42 \
  --deepspeed-config "$PWD/configs/deepspeed_zero2.json" \
  --save-strategy steps \
  --save-steps 100

./run_stage1_training.sh \
  --method dora \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path "$PWD/data/TRAIN.jsonl" \
  --image-root "$PWD/data" \
  --output-dir "$zero_checkpoint_root/$experiment_id" \
  --log-file "$zero_log_root/$experiment_id/resume.log" \
  --micro-batch 1 \
  --gradient-accumulation 8 \
  --epochs 3 \
  --learning-rate 1e-5 \
  --seed 42 \
  --deepspeed-config "$PWD/configs/deepspeed_zero2.json" \
  --resume "$zero_checkpoint_root/$experiment_id/checkpoint-100"
```

```bash
zero_checkpoint_root=/root/workspace/ILVR/checkpoints/zero-stage-compatibility
zero_log_root=/root/workspace/ILVR/logs/zero-stage-compatibility
run_suffix="$(date -u +%Y%m%d-%H%M%S)-$$-${RANDOM}"
experiment_id="stage2-zero2-dora-fresh-example-$run_suffix"

STAGE1_CHECKPOINT="$zero_checkpoint_root/stage1-dora-final" \
DATA_PATH="$PWD/data/TRAIN.jsonl" \
IMAGE_ROOT="$PWD/data" \
OUTPUT_DIR="$zero_checkpoint_root/$experiment_id" \
LOG_FILE="$zero_log_root/$experiment_id/train.log" \
TUNING_METHOD=dora \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" \
GRADIENT_ACCUMULATION_STEPS=8 \
ATTENTION_IMPLEMENTATION=flash_attention_2 \
./run_stage2_training.sh

STAGE1_CHECKPOINT="$zero_checkpoint_root/stage1-dora-final" \
DATA_PATH="$PWD/data/TRAIN.jsonl" \
IMAGE_ROOT="$PWD/data" \
OUTPUT_DIR="$zero_checkpoint_root/$experiment_id" \
LOG_FILE="$zero_log_root/$experiment_id/resume.log" \
RESUME_CHECKPOINT="$zero_checkpoint_root/$experiment_id/checkpoint-100" \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" \
GRADIENT_ACCUMULATION_STEPS=8 \
ATTENTION_IMPLEMENTATION=flash_attention_2 \
./run_stage2_training.sh
```

The Stage 1 example uses the universally valid EGB-16 fallback `(1,8)` before measured winners exist. After Task 10, the result table supplies the preferred pair for each ZeRO stage/method.

- [ ] **Step 2: Add empty result-table schemas that cannot be mistaken for measurements**

Use explicit status rows instead of numeric filler:

```markdown
| ZeRO | Method | Micro/GA | Peak reserved GiB | samples/s | Resume |
|---|---|---|---:|---:|---|
| 3 | Full | recorded by Task 10 | not recorded in this document yet | not recorded in this document yet | pending Task 10 |
```

Explain that the matrix report lives at `/root/workspace/ILVR/results/zero-stage-compatibility/stage1-zero-matrix/zero-matrix-report.json` and each Stage 2 rank-skew witness lives below the same canonical result root in its experiment-ID directory; do not claim hardware success before Tasks 9–10.

- [ ] **Step 3: Run one focused CPU suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/test_deepspeed_runtime.py \
  tests/test_artifact_paths.py \
  tests/stage1/test_trainer.py \
  tests/stage1/test_ema.py \
  tests/stage1/test_peft.py \
  tests/stage1/test_checkpointing.py \
  tests/stage1/test_composition.py \
  tests/stage1/test_benchmark.py \
  tests/stage1/test_zero_matrix.py \
  tests/stage1/test_launcher.py \
  tests/stage2/test_sampler.py \
  tests/stage2/test_routing.py \
  tests/stage2/test_peft_checkpointing.py \
  tests/stage2/integration/test_distributed_contract.py -q
```

Expected: all selected tests pass once. Do not immediately rerun the same suite.

- [ ] **Step 4: Run shell syntax checks once**

Run:

```bash
bash -n run_stage1_training.sh run_stage1_batch_probe.sh \
  run_stage1_zero_matrix.sh run_stage2_training.sh
```

Expected: exit code 0 with no output.

- [ ] **Step 5: Commit the guide and any fixture path updates**

```bash
git add docs/zero-stage-training.md README.md
git commit -m "docs: describe ZeRO stage training and resume"
```

### Task 9: Run Only Missing Distributed Compatibility Gates

**Files:**
- Record checkpoints: `/root/workspace/ILVR/checkpoints/zero-stage-compatibility/<experiment-id>/`
- Record witnesses: `/root/workspace/ILVR/results/zero-stage-compatibility/<experiment-id>/*.json`
- Record logs: `/root/workspace/ILVR/logs/zero-stage-compatibility/<experiment-id>/{stdout,stderr}.log`
- Reuse: existing Stage 2 ZeRO-1/2 DoRA save/resume feasibility evidence

**Interfaces:**
- Consumes: Task 7 harnesses and exact existing witnesses.
- Produces: bounded two-rank optimizer/save/resume witnesses and one rank-skew witness per ZeRO stage; no repository code change.

- [ ] **Step 1: Inventory evidence before allocating GPUs**

Run:

```bash
zero_result_root=/root/workspace/ILVR/results/zero-stage-compatibility
for evidence_root in "$zero_result_root" "$PWD/outputs"; do
  if [[ -d "$evidence_root" ]]; then
    rg --files "$evidence_root" | rg '(evidence|witness|rank-skew|resume).*\.json$'
  fi
done | sort -u
```

The second root is legacy read-only provenance. Open only candidate JSON files smaller than 1 MiB and compare `zero_stage`, `config_digest`, `world_size`, method, and harness schema. Do not move legacy files, write into `outputs/`, or read/hash checkpoint weights. The task log records exact reusable paths; no extra manifest is needed.

- [ ] **Step 2: Run missing Stage 1 ZeRO-1 Full/DoRA tiny lifecycles**

Run each missing method once:

```bash
zero_checkpoint_root=/root/workspace/ILVR/checkpoints/zero-stage-compatibility
zero_result_root=/root/workspace/ILVR/results/zero-stage-compatibility
zero_log_root=/root/workspace/ILVR/logs/zero-stage-compatibility

run_suffix="$(date -u +%Y%m%d-%H%M%S)-$$-${RANDOM}"
experiment_id="stage1-zero1-full-tiny-lifecycle-$run_suffix"
mkdir -p -- "$zero_log_root"
mkdir -- "$zero_log_root/$experiment_id"
CUDA_VISIBLE_DEVICES=0,1 ./venv/ilvr-venv/bin/torchrun \
  --standalone --nproc_per_node=2 tests/stage1/zero_compat_diagnostic.py \
  --zero-stage 1 --method full \
  --deepspeed configs/deepspeed_zero1.json \
  --experiment-id "$experiment_id" \
  >"$zero_log_root/$experiment_id/stdout.log" \
  2>"$zero_log_root/$experiment_id/stderr.log"

run_suffix="$(date -u +%Y%m%d-%H%M%S)-$$-${RANDOM}"
experiment_id="stage1-zero1-dora-tiny-lifecycle-$run_suffix"
mkdir -- "$zero_log_root/$experiment_id"
CUDA_VISIBLE_DEVICES=0,1 ./venv/ilvr-venv/bin/torchrun \
  --standalone --nproc_per_node=2 tests/stage1/zero_compat_diagnostic.py \
  --zero-stage 1 --method dora \
  --deepspeed configs/deepspeed_zero1.json \
  --experiment-id "$experiment_id" \
  >"$zero_log_root/$experiment_id/stdout.log" \
  2>"$zero_log_root/$experiment_id/stderr.log"
```

Expected: each command exits 0; its checkpoint, witness, and logs share the experiment ID under their respective canonical roots, and the witness records steps 1→2, finite norm, matching runtime sidecar, and method.

- [ ] **Step 3: Run missing Stage 1 ZeRO-2 Full/DoRA tiny lifecycles**

```bash
zero_log_root=/root/workspace/ILVR/logs/zero-stage-compatibility
run_suffix="$(date -u +%Y%m%d-%H%M%S)-$$-${RANDOM}"
experiment_id="stage1-zero2-full-tiny-lifecycle-$run_suffix"
mkdir -p -- "$zero_log_root"
mkdir -- "$zero_log_root/$experiment_id"
CUDA_VISIBLE_DEVICES=0,1 ./venv/ilvr-venv/bin/torchrun \
  --standalone --nproc_per_node=2 tests/stage1/zero_compat_diagnostic.py \
  --zero-stage 2 --method full \
  --deepspeed configs/deepspeed_zero2.json \
  --experiment-id "$experiment_id" \
  >"$zero_log_root/$experiment_id/stdout.log" \
  2>"$zero_log_root/$experiment_id/stderr.log"

run_suffix="$(date -u +%Y%m%d-%H%M%S)-$$-${RANDOM}"
experiment_id="stage1-zero2-dora-tiny-lifecycle-$run_suffix"
mkdir -- "$zero_log_root/$experiment_id"
CUDA_VISIBLE_DEVICES=0,1 ./venv/ilvr-venv/bin/torchrun \
  --standalone --nproc_per_node=2 tests/stage1/zero_compat_diagnostic.py \
  --zero-stage 2 --method dora \
  --deepspeed configs/deepspeed_zero2.json \
  --experiment-id "$experiment_id" \
  >"$zero_log_root/$experiment_id/stdout.log" \
  2>"$zero_log_root/$experiment_id/stderr.log"
```

Expected: each command exits 0 with the same bounded witness contract. If any run fails unexpectedly, stop the compatibility gate and invoke systematic debugging; do not proceed to the performance matrix.

- [ ] **Step 4: Reuse the prior Stage 2 ZeRO-1/2 DoRA save/resume witnesses**

Verify only their small recorded identity/result documents. If they lack the new sidecar schema because Task 3 changed the checkpoint contract, run the tiny Stage 2 harness exactly once for the affected ZeRO stage; otherwise do not repeat those save/resume jobs.

```bash
zero_log_root=/root/workspace/ILVR/logs/zero-stage-compatibility
run_suffix="$(date -u +%Y%m%d-%H%M%S)-$$-${RANDOM}"
experiment_id="stage2-zero1-dora-tiny-lifecycle-$run_suffix"
mkdir -p -- "$zero_log_root"
mkdir -- "$zero_log_root/$experiment_id"
CUDA_VISIBLE_DEVICES=0,1 ./venv/ilvr-venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  tests/stage2/integration/tiny_stage2_train.py \
  --experiment-id "$experiment_id" \
  --deepspeed configs/deepspeed_zero1.json \
  --tuning-method dora \
  >"$zero_log_root/$experiment_id/stdout.log" \
  2>"$zero_log_root/$experiment_id/stderr.log"

run_suffix="$(date -u +%Y%m%d-%H%M%S)-$$-${RANDOM}"
experiment_id="stage2-zero2-dora-tiny-lifecycle-$run_suffix"
mkdir -- "$zero_log_root/$experiment_id"
CUDA_VISIBLE_DEVICES=0,1 ./venv/ilvr-venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  tests/stage2/integration/tiny_stage2_train.py \
  --experiment-id "$experiment_id" \
  --deepspeed configs/deepspeed_zero2.json \
  --tuning-method dora \
  >"$zero_log_root/$experiment_id/stdout.log" \
  2>"$zero_log_root/$experiment_id/stderr.log"
```

Run only the command(s) whose exact identity check failed in Step 1; do not execute both when either stage already has a matching sidecar-era witness.

- [ ] **Step 5: Run one Stage 2 rank-skew step per ZeRO stage**

Run exactly these three commands, skipping an exact successful witness found in Step 1:

```bash
: "${STAGE1_CHECKPOINT:?set STAGE1_CHECKPOINT to the validated Stage 1 DoRA handoff}"
zero_log_root=/root/workspace/ILVR/logs/zero-stage-compatibility
for zero_stage in 1 2 3; do
  run_suffix="$(date -u +%Y%m%d-%H%M%S)-$$-${RANDOM}"
  experiment_id="stage2-zero${zero_stage}-dora-mb1-ga8-rank-skew-$run_suffix"
  mkdir -p -- "$zero_log_root"
  mkdir -- "$zero_log_root/$experiment_id"
  CUDA_VISIBLE_DEVICES=0,1 ./venv/ilvr-venv/bin/torchrun \
    --standalone --nproc_per_node=2 \
    tests/stage2/integration/stage2_rank_skew_probe.py \
    --deepspeed "configs/deepspeed_zero${zero_stage}.json" \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --stage1-checkpoint "$STAGE1_CHECKPOINT" \
    --data-path "$PWD/data/TRAIN.jsonl" \
    --image-root "$PWD/data" \
    --tuning-method dora \
    --micro-batch 1 \
    --gradient-accumulation-steps 8 \
    --experiment-id "$experiment_id" \
    >"$zero_log_root/$experiment_id/stdout.log" \
    2>"$zero_log_root/$experiment_id/stderr.log"
done
```

Expected: every executed command exits 0 after exactly one optimizer step/eight micro-steps per rank and records equal per-rank latent-block/forward counts plus rank peak allocated/reserved bytes. A timeout/NCCL error is `integration_failure`, never a capacity result.

- [ ] **Step 6: Stop after inspecting the small witnesses**

Run:

```bash
./venv/ilvr-venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path(
    "/root/workspace/ILVR/"
    "results/zero-stage-compatibility"
)
for path in sorted(root.glob("stage*-zero*/**/*.json")):
    payload = json.loads(path.read_text())
    print(path, payload.get("zero_stage"), payload.get("optimizer_steps"),
          payload.get("aligned_at_every_micro_step", payload.get("resumed_global_step")))
PY
```

Expected: each required identity has one successful witness. Do not run a second distributed validator or hash the produced checkpoints.

### Task 10: Run the Missing Stage 1 Batch Matrix in Reverse Order

**Files:**
- Record candidate/selected checkpoints: `/root/workspace/ILVR/checkpoints/zero-stage-compatibility/<experiment-id>/`
- Record per-run evidence: `/root/workspace/ILVR/results/zero-stage-compatibility/<experiment-id>/producer-evidence.json`
- Record aggregate report: `/root/workspace/ILVR/results/zero-stage-compatibility/stage1-zero-matrix/zero-matrix-report.json`
- Record per-run logs: `/root/workspace/ILVR/logs/zero-stage-compatibility/<experiment-id>/`

**Interfaces:**
- Consumes: Task 6 coordinator, Task 9 compatibility success, existing exact ZeRO-3 evidence, and one authenticated sample-audit digest.
- Produces: optimal `(micro_batch, gradient_accumulation)` for ZeRO-3/2/1 × Full/DoRA, peak VRAM, throughput, failure classification, and selected resume identity.

- [ ] **Step 1: Produce or reuse one sample audit**

If no valid sample-audit witness exists, run the benchmark's existing bounded audit command once against `data/TRAIN.jsonl` and `data/`, then pass its digest to every matrix producer. If it exists and matches model, dataset path/stat identity, seed 42, and row selection, do not rerun or rehash the dataset.

```bash
zero_result_root=/root/workspace/ILVR/results/zero-stage-compatibility
canonical_audit="$zero_result_root/stage1-zero-matrix/sample-audit.json"
legacy_audit="$PWD/outputs/stage1-zero-matrix/sample-audit.json"
mkdir -p -- "$zero_result_root/stage1-zero-matrix"
if ! sample_audit=$(PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python -m src.stage1.zero_matrix audit-status \
    --candidate-audit "$canonical_audit" \
    --candidate-audit "$legacy_audit" \
    --data-path "$PWD/data/TRAIN.jsonl" \
    --image-root "$PWD/data" \
    --count 8); then
  PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python -m src.stage1.benchmark audit \
    --data-path "$PWD/data/TRAIN.jsonl" \
    --image-root "$PWD/data" \
    --output "$canonical_audit" \
    --count 8
  sample_audit="$canonical_audit"
fi
printf 'sample_audit=%s\n' "$sample_audit"
```

Expected: an exact canonical or legacy audit path is printed and left untouched, or one new small audit JSON is produced under the canonical result root. Nothing is written into legacy `outputs/`, and the dataset is not read/hashed again in this workflow.

- [ ] **Step 2: Ask the coordinator which producers are reusable**

Run:

```bash
zero_result_root=/root/workspace/ILVR/results/zero-stage-compatibility
canonical_audit="$zero_result_root/stage1-zero-matrix/sample-audit.json"
legacy_audit="$PWD/outputs/stage1-zero-matrix/sample-audit.json"
sample_audit=$(PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python -m src.stage1.zero_matrix audit-status \
  --candidate-audit "$canonical_audit" \
  --candidate-audit "$legacy_audit" \
  --data-path "$PWD/data/TRAIN.jsonl" \
  --image-root "$PWD/data" \
  --count 8)
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/python -m src.stage1.zero_matrix reuse-status \
  --matrix-result-dir "$zero_result_root/stage1-zero-matrix" \
  --world-size 2 \
  --effective-global-batch 16 \
  --sample-audit "$sample_audit"
```

Expected: JSON lists stages 3→2→1, methods Full/DoRA, candidates 8→4→2→1, and exact reuse reasons. If the existing authenticated ZeRO-3 producer identity matches, reuse it. Do not hash its checkpoint tree. Omitting `--method` and `--deepspeed-config` deliberately requests the full canonical matrix; the single-profile form from Task 6 remains available for debugging one row.

- [ ] **Step 3: Launch the reverse matrix once**

Run:

```bash
zero_result_root=/root/workspace/ILVR/results/zero-stage-compatibility
zero_log_root=/root/workspace/ILVR/logs/zero-stage-compatibility
matrix_result_id=stage1-zero-matrix
run_suffix="$(date -u +%Y%m%d-%H%M%S)-$$-${RANDOM}"
controller_log_id="stage1-zero-matrix-controller-$run_suffix"
mkdir -p -- "$zero_log_root"
mkdir -- "$zero_log_root/$controller_log_id"
CUDA_VISIBLE_DEVICES=0,1 \
STAGE1_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
STAGE1_DATA_PATH="$PWD/data/TRAIN.jsonl" \
STAGE1_IMAGE_ROOT="$PWD/data" \
./run_stage1_zero_matrix.sh \
  --matrix-result-dir "$zero_result_root/$matrix_result_id" \
  --world-size 2 \
  --seed 42 \
  >"$zero_log_root/$controller_log_id/stdout.log" \
  2>"$zero_log_root/$controller_log_id/stderr.log"
```

Expected: the controller runs only missing candidates, each for exactly three optimizer steps, in stage order 3→2→1 and candidate order 8→4→2→1. Successful rows contain samples/sec, seconds/step, rank peak allocated/reserved bytes, finite diagnostics, and identity. CUDA allocator failures are recorded as `capacity_oom`; any other failure stops that `(stage, method)` workflow as `integration_failure`.

- [ ] **Step 4: Confirm selected-only resume witnesses**

Inspect `/root/workspace/ILVR/results/zero-stage-compatibility/stage1-zero-matrix/zero-matrix-report.json`. For each successful `(zero_stage, method)`, confirm exactly one selected candidate and exactly one same-identity checkpoint→one-step-resume witness. Confirm every recorded checkpoint/result/log path belongs to its canonical root and uses the same experiment ID. If the controller stopped before selected resume, run its printed exact canonical-path resume command once; do not save/resume losing candidates.

- [ ] **Step 5: Stop compute after the report is complete**

Run the coordinator's small report validator once:

```bash
zero_result_root=/root/workspace/ILVR/results/zero-stage-compatibility
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/python -m src.stage1.zero_matrix select \
  --matrix-result-dir "$zero_result_root/stage1-zero-matrix" \
  --require-selected-resume
```

Expected: exit 0 and the existing report path/digest. This validator reads producer JSON only. Do not rerun any producer, distributed smoke, model validator, or checkpoint hash after success.

### Task 11: Publish Measured Results and Perform One Final Review

**Files:**
- Modify: `docs/zero-stage-training.md`
- Modify: `README.md` only if the final command syntax changed during execution
- Test: targeted changed-file test list from Task 8, only if code changed after its successful run

**Interfaces:**
- Consumes: Task 9 compatibility witnesses and Task 10 machine-readable report.
- Produces: final human-readable support/results table and a clean implementation review; no new expensive evidence.

- [ ] **Step 1: Copy measured scalar results into the guide**

For each Stage 1 stage/method row, copy `micro_batch`, `gradient_accumulation_steps`, `rank_peak_reserved_bytes`, `samples_per_second`, status, and resume result from the canonical `results/zero-stage-compatibility/stage1-zero-matrix/zero-matrix-report.json`. For Stage 2, record fixed micro 1/accumulation 8 plus rank-skew compatibility and any already-observed peak memory witness. Cite the absolute canonical result path and digest; do not derive a new checkpoint digest.

- [ ] **Step 2: Inspect the implementation diff for contract gaps**

Run:

```bash
git diff --check
git status --short
git log --oneline --decorate -12
```

Expected: `git diff --check` exits 0. The status may still show pre-existing `src/stage2/peft.py`, `tests/stage2/test_peft.py`, caches, or outputs; do not stage unrelated changes.

- [ ] **Step 3: Re-run tests only if code changed after Task 8**

If Tasks 9–10 required a production-code fix, first use systematic debugging and add one regression test for that cause, then run only that test file plus its closest integration-contract file. If no production code changed after Task 8, reuse Task 8's green CPU witness and skip this step.

- [ ] **Step 4: Commit only the measured documentation and any focused regression fix**

```bash
git add docs/zero-stage-training.md README.md
git diff --cached --check
git commit -m "docs: record ZeRO batch and memory results"
```

If a focused regression fix exists, commit its production and test files separately before this documentation commit. Never stage generated `checkpoints/`, `results/`, `logs/`, legacy `outputs/`, `cache/`, `__pycache__/`, or unrelated pre-existing changes.

- [ ] **Step 5: Apply completion criteria without another hardware run**

Confirm from existing witnesses that:

1. Stage 1 Full/DoRA lifecycle passed for ZeRO-1/2 and the common ZeRO-3 regression remains valid.
2. Stage 2 exposes Full/LoRA/DoRA for all canonical configs.
3. Each ZeRO stage has one successful rank-skew aligned-step witness.
4. Numbered checkpoints and selected resumes contain matching runtime identity; Stage 1 also contains EMA and Stage 2 sampler identity.
5. Every Stage 1 stage/method has four classified candidates or an explicitly reported integration failure, one selected fastest success, and one selected resume witness.
6. Documentation contains measured values and evidence paths without claiming unsupported world sizes.
7. Every newly generated checkpoint, result, and log path is below its exact canonical root; each independent producer uses one unique experiment ID across its artifact types, bounded same-identity lifecycle phases keep their logical experiment ID, and no legacy `outputs/` path has been modified.

Stop and report immediately when these checks are satisfied; do not add a broad smoke suite, rerun a producer, or recursively validate artifacts.
